"""Independent queued GNN ablations; no edits to the active seven-method run."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import threading
import time

PROJECT=Path(__file__).resolve().parents[1]
for path in (PROJECT,PROJECT/'src'):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from rvloc.config import ExperimentConfig,external_path
from rvloc.runtime import file_sha256,write_json
from gnn_research.train import training_sources,VARIANTS,require


def _now():return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def device_groups(variants,devices):
    require(bool(devices),'At least one device required')
    out={}
    for i,variant in enumerate(variants):out.setdefault(devices[i%len(devices)],[]).append(variant)
    return out


class GraphRunner:
    def __init__(self,config_path,index_path,run_dir,*,variants=VARIANTS,devices=('cpu',),
                 progress_log=None,baseline_evaluation=None,resume=False,process_runner=None,poll_seconds=30,parallel_arms=False):
        self.config=ExperimentConfig.load(config_path)
        self.index_path=Path(index_path).resolve();index=json.loads(self.index_path.read_text())
        require(index.get('schema')=='rvloc.fixed_roi_gt_index.v1' and index.get('supervision_only') is True,'Fixed labels required')
        self.run_dir=external_path(run_dir,field='GNN experiment output')
        require(self.run_dir.is_relative_to(self.config.output_root) and self.run_dir!=self.config.output_root,'Independent configured result run required')
        require(set(variants)<=set(VARIANTS) and len(set(variants))==len(variants) and bool(variants),'Unique known variants required')
        self.variants=tuple(variants);self.devices=tuple(devices);self.groups=device_groups(self.variants,self.devices)
        self.parallel_arms=bool(parallel_arms)
        self.variant_devices={variant:device for device,arms in self.groups.items() for variant in arms}
        self.execution_mode='parallel_arms' if self.parallel_arms else 'per_device_queue'
        self.progress_log=external_path(progress_log,field='GNN progress sink') if progress_log else self.run_dir/'progress.log'
        self.baseline=Path(baseline_evaluation).resolve() if baseline_evaluation else None
        self.resume=resume;self.process_runner=process_runner;self.poll_seconds=poll_seconds
        self.identity=dict(schema='rvloc.mce_gnn_execution.v1',config_path=str(self.config.path),config_sha256=self.config.sha256,
            manifest_sha256=file_sha256(self.config.manifest),index_path=str(self.index_path),index_sha256=file_sha256(self.index_path),
            variants=list(self.variants),device_groups=self.groups,variant_devices=self.variant_devices,
            execution_mode=self.execution_mode,parallel_arms=self.parallel_arms,
            planned_max_concurrent_arms_by_device={device:len(arms) if self.parallel_arms else 1 for device,arms in self.groups.items()},
            concurrency_cost_note='Training throughput is measured under concurrent runs and is not exclusive serial inference latency',
            training_sources=training_sources(),
            checkpoint_rule='fixed final TRAIN epoch; no DEV checkpoint selection',
            inference_matching='frozen ROI-GT identity; no predicted-position rematching',
            protocol='same seed/loss/frame-equal AdamW/update4/episode-tail budgets; candidate CE disabled',
            baseline_evaluation=str(self.baseline) if self.baseline else None,historical_TEST_opened=False)
        self.identity['sha256']=_digest(self.identity)
        self.states={v:dict(status='queued') for v in self.variants};self.active={};self.lock=threading.RLock()
        self.started=time.monotonic();self.status='starting';self.log_stream=None;self.errors=[]

    def _assert(self):
        require(file_sha256(self.config.path)==self.identity['config_sha256'],'Config changed')
        require(file_sha256(self.config.manifest)==self.identity['manifest_sha256'],'Manifest changed')
        require(file_sha256(self.index_path)==self.identity['index_sha256'],'Fixed labels changed')
        require(training_sources()==self.identity['training_sources'],'Graph training sources changed')

    def emit(self,variant,line):
        with self.lock:
            value=f'[{_now()}] [MCE {variant}] {line.rstrip()}'
            print(value,flush=True)
            if self.log_stream is not None:self.log_stream.write(value+'\n');self.log_stream.flush()
            if '[PROGRESS]' in line or 'MCE_TRAIN' in line:
                self.progress_log.parent.mkdir(parents=True,exist_ok=True)
                with self.progress_log.open('a') as out:out.write(value+'\n')

    def snapshot(self):
        with self.lock:
            write_json(self.run_dir/'status.json',dict(schema='rvloc.mce_gnn_status.v1',status=self.status,pid=os.getpid(),
                updated_at_utc=_now(),elapsed_s=time.monotonic()-self.started,identity_sha256=self.identity['sha256'],
                variants=self.states,active_pids=self.active,device_groups=self.groups,variant_devices=self.variant_devices,
                execution_mode=self.execution_mode,errors=self.errors))

    def execute(self,variant,command,logfile):
        env=dict(os.environ,PYTHONPATH=str(PROJECT/'src')+os.pathsep+str(PROJECT),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        if self.process_runner:return self.process_runner(command,env=env,emit=lambda line:self.emit(variant,line))
        with logfile.open('a',buffering=1) as log:
            child=subprocess.Popen(command,cwd=PROJECT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
            with self.lock:self.active[variant]=child.pid;self.snapshot()
            try:
                for line in child.stdout:
                    log.write(line);self.emit(variant,line)
                return child.wait()
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            finally:
                child.stdout.close()
                with self.lock:self.active.pop(variant,None);self.snapshot()

    def stage(self,variant,kind,device):
        self._assert();root=self.run_dir/variant;record_path=root/'stage_receipts'/f'{kind}.json'
        inp=_digest(dict(identity=self.identity['sha256'],variant=variant,kind=kind,device=device))
        if record_path.exists():
            record=json.loads(record_path.read_text());require(record.get('status')=='completed' and record['input_sha256']==inp,'Resume stage identity differs')
            for path,sha in record['artifacts'].items():require(file_sha256(path)==sha,'Completed graph artifact changed')
            for path,sha in record.get('canonical_artifacts',{}).items():require(file_sha256(path)==sha,'Canonical graph artifact changed')
            self.emit(variant,f'RESUME {kind}: certified completed outputs');return record
        attempts=root/kind;attempts.mkdir(parents=True,exist_ok=True);number=1
        while (attempts/f'attempt_{number:04d}').exists():number+=1
        attempt=attempts/f'attempt_{number:04d}';attempt.mkdir()
        python=[sys.executable,'-u'];out=attempt/'results'
        if kind=='train':
            command=python+[str(PROJECT/'gnn_research/train.py'),'--config',str(self.config.path),'--variant',variant,
                '--index',str(self.index_path),'--output-dir',str(out),'--device',device,'--batch-size','4']
            if number>1:
                previous=sorted((self.config.weights_root/attempts.relative_to(self.config.output_root)).glob('attempt_*/results/epoch_*.pt'))
                if previous:command+=['--resume-checkpoint',str(previous[-1])]
            expected=[out/'receipt.json',out/'history.json',out/'identity.json',out/'progress.json'];stage_tool=PROJECT/'gnn_research/train.py'
        elif kind=='inference':
            train=json.loads((root/'train/receipt.json').read_text())
            command=python+[str(PROJECT/'gnn_research/evaluate.py'),'infer','--config',str(self.config.path),
                '--checkpoint',train['checkpoint'],'--output-dir',str(out),'--device',device,'--split','dev','--batch-size','4']
            expected=[out/'predictions.json',out/'reproduction_receipt.json'];stage_tool=PROJECT/'gnn_research/evaluate.py'
        elif kind=='evaluation':
            command=python+[str(PROJECT/'gnn_research/evaluate.py'),'evaluate','--index',str(self.index_path),
                '--predictions',str(root/'inference/predictions.json'),'--output-dir',str(out)]
            expected=[out/name for name in ('evaluation.json','localization_metrics.json','graph_metrics.json','episode_level_results.json','failure_cases.json')];stage_tool=PROJECT/'gnn_research/evaluate.py'
        else:raise ValueError(kind)
        began=time.monotonic();record=dict(schema='rvloc.mce_gnn_stage.v1',status='running',variant=variant,kind=kind,
            input_sha256=inp,command=command,device=device,started_at_utc=_now(),attempt=number,
            stage_tool=str(stage_tool),stage_tool_sha256=file_sha256(stage_tool),runner_sha256=file_sha256(__file__))
        write_json(attempt/'stage.json',record)
        with self.lock:self.states[variant]=dict(status=kind,device=device,attempt=number);self.snapshot()
        if kind=='train':
            live=root/'train/progress.json'
            if live.is_symlink():live.unlink()
            elif live.exists():raise FileExistsError(live)
            live.symlink_to((out/'progress.json').relative_to(live.parent))
        code=self.execute(variant,command,attempt/'stage.log')
        require(code==0,f'{variant}/{kind} exited {code}; see {attempt/"stage.log"}')
        require(all(p.is_file() for p in expected),'Graph stage did not commit its required outputs')
        if kind=='train':
            training=json.loads((out/'receipt.json').read_text());require(training['status']=='trained','Training incomplete')
            require(training['variant']==variant and training['epochs']==self.config.raw['training']['epochs'],'Training protocol differs')
            expected.append(Path(training['checkpoint']))
            write_json(root/'train/receipt.json',training)
        # Fixed canonical paths point to the certified attempt; failed attempts
        # remain intact and report readers do not need to guess the latest one.
        canonical=[]
        if kind=='train':canonical.append(root/'train/receipt.json')
        for source in expected:
            if source.suffix!='.json' or source.name=='receipt.json' and kind=='train':continue
            target=root/kind/source.name
            if target.is_symlink():target.unlink()
            elif target.exists():raise FileExistsError(target)
            target.symlink_to(source.relative_to(target.parent));canonical.append(target)
        record.update(status='completed',finished_at_utc=_now(),elapsed_s=time.monotonic()-began,
            artifacts={str(path):file_sha256(path) for path in expected},
            canonical_artifacts={str(path):file_sha256(path) for path in canonical})
        write_json(attempt/'stage.json',record);write_json(record_path,record)
        self.emit(variant,f'COMPLETE {kind} elapsed_s={record["elapsed_s"]:.3f}')
        return record

    def arm(self,variant,device):
        try:
            for kind in ('train','inference','evaluation'):self.stage(variant,kind,device)
            with self.lock:self.states[variant]=dict(status='evaluated',device=device);self.snapshot()
        except BaseException as error:
            with self.lock:
                self.states[variant]=dict(status='failed',device=device,error=str(error));self.errors.append(dict(variant=variant,error_type=type(error).__name__,error=str(error)));self.snapshot()
            raise

    def comparisons(self):
        from gnn_research.evaluate import compare
        evaluations={v:self.run_dir/v/'evaluation/evaluation.json' for v in self.variants}
        if self.baseline:
            while not self.baseline.is_file():
                self.status='waiting_for_baseline';self.snapshot();self.emit('comparison',f'WAIT frozen G3 fixed evaluation {self.baseline}')
                time.sleep(self.poll_seconds)
            evaluations={'baseline_G3':self.baseline,**evaluations}
        result={};names=list(evaluations)
        for i,name in enumerate(names):
            for other in names[i+1:]:
                path=self.run_dir/'comparisons'/f'{name}_vs_{other}.json'
                if not path.exists():compare(evaluations[name],evaluations[other],path)
                result[name+'_vs_'+other]=str(path)
        return result

    def run(self):
        if not self.resume:
            if self.run_dir.exists():
                require(all(p.name in ('bootstrap.log','preparation.json','run.log') for p in self.run_dir.iterdir()),'Run exists; use --resume or a fresh directory')
            else:self.run_dir.mkdir(parents=True)
        else:require(self.run_dir.is_dir(),'Cannot resume nonexistent graph run')
        with (self.run_dir/'.run.lock').open('a+') as lock:
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise ValueError('Another graph runner owns this directory') from error
            ip=self.run_dir/'execution_identity.json'
            if ip.exists():require(json.loads(ip.read_text())==self.identity,'Graph resume identity differs')
            else:write_json(ip,self.identity)
            self.log_stream=(self.run_dir/'run.log').open('a',buffering=1);self.status='running';self.snapshot()
            try:
                def queue(device,variants):
                    for variant in variants:self.arm(variant,device)
                schedule=[(self.variant_devices[v],[v]) for v in self.variants] if self.parallel_arms else list(self.groups.items())
                with ThreadPoolExecutor(max_workers=len(schedule)) as executor:
                    jobs=[executor.submit(queue,device,variants) for device,variants in schedule]
                    for job in as_completed(jobs):job.result()
                pairs=self.comparisons();self.status='completed';self.snapshot()
                result=dict(schema='rvloc.mce_gnn_completion.v1',status='completed',run_dir=str(self.run_dir),
                    variants=list(self.variants),comparisons=pairs,elapsed_s=time.monotonic()-self.started)
                write_json(self.run_dir/'completion.json',result);return result
            except BaseException as error:
                self.status='failed';self.errors.append(dict(error_type=type(error).__name__,error=str(error)));self.snapshot();raise
            finally:self.log_stream.close();self.log_stream=None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--index',required=True);parser.add_argument('--run-dir',required=True)
    parser.add_argument('--variants',default=','.join(VARIANTS));parser.add_argument('--devices',default='cpu')
    parser.add_argument('--parallel-arms',action='store_true',help='Run one independent worker per arm, including arms mapped to the same device; default queues arms sharing a device')
    parser.add_argument('--progress-log');parser.add_argument('--baseline-evaluation');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();result=GraphRunner(args.config,args.index,args.run_dir,variants=args.variants.split(','),
        devices=args.devices.split(','),parallel_arms=args.parallel_arms,progress_log=args.progress_log,baseline_evaluation=args.baseline_evaluation,resume=args.resume).run()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
