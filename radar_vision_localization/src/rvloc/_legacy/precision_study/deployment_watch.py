"""Export and profile finished formal controls without changing training jobs.

Raw training checkpoints are opened only by explicitly offline export and
equivalence operations. Ordinary inference/profiling uses sanitized copies.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
RUNS=(('B_wide_residual_none','B_wide_residual_none','none'),
      ('ra_re_reference','R_ra_re_reference','ra_re'),
      ('joint_selected_formal','C_joint_volume','joint_volume'),
      ('D_wide_residual_ra_re','D_wide_residual_ra_re','ra_re'))


def stamp():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def completed_run(root,key,mode):
    """Return authenticated completed identity or a small pending status."""
    root=Path(root);directory=root/'training'/key
    if not (directory/'run_manifest.json').is_file() or not (directory/'progress.json').is_file():
        return None,dict(stage='waiting_for_training',step=None)
    manifest=read(directory/'run_manifest.json');progress=read(directory/'progress.json')
    if manifest.get('status')!='complete' or progress.get('stage')!='complete':
        return None,dict(stage='waiting_for_training',training_stage=progress.get('stage'),step=progress.get('step'))
    wanted=dict(optimizer_steps=20000,rank_parameter_max_difference=0,world_size=6,
        global_batch_size=48,prepared_training_groups=11649,prepared_validation_groups=2569,
        all_usable_training_frames_used=True,precision_radar_mode=mode,seed=20261118)
    for name,value in wanted.items():
        if manifest.get(name)!=value:raise ValueError('Completed training mismatch '+name+' for '+key)
    if progress.get('step')!=20000 or manifest.get('hyperparameters',{}).get('steps')!=20000:
        raise ValueError('Formal training did not finish all 20000 updates')
    scope=manifest.get('training_scope',{})
    if scope.get('diagnostic_subset') is not False or scope.get('training_frames_subset') is not False or scope.get('validation_frames_subset') is not False:
        raise ValueError('Formal deployment cannot use a frame subset')
    if manifest.get('learning_rate_schedule',{}).get('total_steps')!=20000:
        raise ValueError('Formal learning-rate horizon changed')
    initial=manifest.get('model_initialization',{})
    if initial.get('loaded_fusion_checkpoint') is not None or initial.get('resume') is not False:
        raise ValueError('Expected fresh formal training')
    plan=read(root/'controller_plan.json')
    if manifest.get('source_sha256')!=plan['source_sha256']:raise ValueError('Formal frozen source differs')
    config=root/'configs'/(key+'.json');source=root/'code'/key
    if sha(config)!=manifest['training_configuration']['sha256']:raise ValueError('Formal config bytes changed')
    from .versions import frozen_study_package
    package,_=frozen_study_package(source)
    if importlib.import_module(package+'.checkpoint').source_hashes()!=manifest['source_sha256']:
        raise ValueError('Actual frozen source package differs')
    checkpoint=directory/'depth.rmse.best.pt'
    if not checkpoint.is_file():raise FileNotFoundError('Finished formal run lacks RMSE-best weights')
    return dict(run_key=key,config=str(config.resolve()),source_root=str(source.resolve()),
        checkpoint=str(checkpoint.resolve()),source_checkpoint_sha256=sha(checkpoint),
        run_manifest_sha256=sha(directory/'run_manifest.json'),
        completed_optimizer_steps=20000,rank_parameter_max_difference=0,
        source_sha256=manifest['source_sha256']),None


def hardware():
    result=dict(at=stamp(),physical_gpu=5,load_average=list(os.getloadavg()),
        allowed_cpu_count=len(os.sched_getaffinity(0)),
        scope='Read-only shared hardware snapshot; no task stopped or rescheduled')
    queries={
        'all_gpus':['nvidia-smi','--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],
        'compute_processes':['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader']}
    for name,command in queries.items():
        try:
            value=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=15,check=False)
            result[name]=dict(returncode=value.returncode,output=value.stdout)
        except Exception as exc:result[name]=dict(error=repr(exc))
    return result


def public_keys(config):
    import numpy as np
    from .visual_data import public_split_manifest
    _,manifest,_=public_split_manifest(config);ids=set(manifest['splits']['val']['episode_ids'])
    keys=[(row['episode'],index) for row in manifest['episodes'] if row['episode'] in ids for index in row['selected_frame_indices']]
    return [list(keys[index]) for index in np.linspace(0,len(keys)-1,24,dtype=int)]


def verify_profile(path,checkpoint,keys,device):
    from .export_deployment import verify_deployment_receipt
    report=read(path);receipt=verify_deployment_receipt(checkpoint,'study')
    expected=dict(checkpoint=str(Path(checkpoint).resolve()),device=device,input_frames=24,
        measured_predictions=120,warmup_predictions=5,repeats_per_frame=5,cpu_torch_threads=1,
        split='val',frames=keys,labels_read=False,test_inputs_permitted=False)
    for name,value in expected.items():
        if report.get(name)!=value:raise ValueError('Protected profile identity mismatch: '+name)
    if report.get('deployment_receipt_sha256')!=sha(Path(checkpoint).parent/'deployment_receipt.json'):
        raise ValueError('Profile does not bind its sanitized deployment receipt')
    if report.get('deployment_state_tensor_sha256')!=receipt['state_tensor_sha256']:
        raise ValueError('Profile tensor identity differs')
    return dict(path=str(Path(path).resolve()),sha256=sha(path),latency_ms=report['latency_ms'],
        gpu_peak_allocated_bytes=report['gpu_peak_allocated_bytes'],device=device,
        input_frames=24,measured_predictions=120)


class DeploymentWatcher:
    def __init__(self,root,interval=15.,once=False):
        self.root=Path(root).resolve();self.interval=interval;self.once=once
        self.progress=self.root/'deployment_progress.json'
        self.state=dict(schema='formal_precision_deployment_progress_v1',pid=os.getpid(),
            started_at=stamp(),status='waiting_for_training',watcher_sha256=sha(__file__),
            runs={},test_inputs_read=False,default_changed=False,existing_tasks_interrupted=False,
            raw_checkpoint_deserialization_scope='Explicit offline export and source-equivalence only; may decode historical validation-report metadata')
        if self.progress.exists():
            old=read(self.progress)
            if old.get('schema')!=self.state['schema'] or old.get('watcher_sha256')!=self.state['watcher_sha256']:
                raise ValueError('Existing deployment progress belongs to another watcher identity')
            self.state['runs']=old.get('runs',{})
    def publish(self):
        self.state['updated_at']=stamp();write(self.progress,self.state)
    def phase(self,name,phase,**values):
        self.state['runs'].setdefault(name,{}).update(stage=phase,updated_at=stamp(),**values);self.publish()
        print(json.dumps(dict(at=stamp(),model=name,stage=phase,**values)),flush=True)
    def execute(self,command,log,*,gpu=False):
        env=os.environ.copy();env.update(PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='5' if gpu else '',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        with Path(log).open('a') as stream:
            stream.write(json.dumps(dict(at=stamp(),command=command,physical_gpu=5 if gpu else None))+'\n');stream.flush()
            subprocess.run(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,
                stderr=subprocess.STDOUT,check=True,timeout=900)
    def process(self,key,name,mode,identity):
        from .export_deployment import export_checkpoint,verify_deployment_receipt
        directory=self.root/'deployment'/name;checkpoint=directory/'model.pt'
        config=read(identity['config'])
        self.phase(name,'offline_export',identity=identity)
        if directory.exists():
            receipt=verify_deployment_receipt(checkpoint,'study')
            if receipt.get('source_checkpoint_sha256')!=identity['source_checkpoint_sha256']:
                raise ValueError('Protected canonical deployment came from another original checkpoint')
        else:
            receipt=export_checkpoint(identity['checkpoint'],directory,kind='study',config=config,source_root=identity['source_root'])
        self.phase(name,'ordinary_inference_equivalence',deployment_checkpoint=str(checkpoint))
        proof=directory/'runtime_boundary_verification.json'
        if not proof.is_file():
            helper=self.root/'deployment/audit_helpers/verify_runtime_boundary.py'
            self.execute([sys.executable,str(helper),'--kind','study','--config',identity['config'],
                '--source',identity['checkpoint'],'--deployment',str(checkpoint),
                '--source-root',identity['source_root'],'--episode','episode_000009','--frame','0'],directory/'ordinary_inference_verification.log')
        evidence=read(proof)
        if (evidence.get('status')!='passed' or evidence.get('source_checkpoint_sha256')!=identity['source_checkpoint_sha256']
                or evidence.get('deployment_checkpoint_sha256')!=sha(checkpoint)
                or evidence.get('exact_depths_positions_valid_mask') is not True
                or evidence.get('test_observations_used') is not False):
            raise ValueError('Ordinary prediction proof differs from canonical weights')
        descriptor=dict(kind='study',model_name=name,config=identity['config'],checkpoint=str(checkpoint),
            source_root=identity['source_root'],configuration_sha256=sha(identity['config']),
            checkpoint_sha256=sha(checkpoint),deployment_receipt_sha256=sha(directory/'deployment_receipt.json'))
        descriptor_path=directory/'deployment_descriptor.json'
        if descriptor_path.exists():
            if read(descriptor_path)!=descriptor:raise ValueError('Protected canonical descriptor identity differs')
        else:write(descriptor_path,descriptor)
        keys=public_keys(config);profiles={}
        for label,device in (('cpu','cpu'),('gpu','cuda:0')):
            result=self.root/'benchmarks'/(name+'_'+label+'.json');result.parent.mkdir(parents=True,exist_ok=True)
            context=result.with_name(name+'_'+label+'_hardware_context.json')
            if not result.exists():
                self.phase(name,'benchmark_'+label,profile=str(result))
                snapshot=dict(schema='ordinary_deployment_hardware_context_v1',model=name,device=device,
                    physical_gpu=5 if label=='gpu' else None,CUDA_VISIBLE_DEVICES='5' if label=='gpu' else '',
                    before=hardware(),shared_GPU_latency_caveat='GPU5 is shared with existing jobs; latency is interference-affected, not isolated throughput')
                write(context,snapshot)
                try:
                    self.execute([sys.executable,'-m','precision_study.benchmark','--kind','study',
                        '--checkpoint',str(checkpoint),'--config',identity['config'],'--source-root',identity['source_root'],
                        '--device',device,'--threads','1','--warmup','5','--repeats','5','--frames','24',
                        '--split','val','--output',str(result)],directory/('benchmark_'+label+'.log'),gpu=label=='gpu')
                finally:
                    snapshot['after']=hardware();write(context,snapshot)
            profiles[label]=verify_profile(result,checkpoint,keys,device)
        if sha(identity['checkpoint'])!=identity['source_checkpoint_sha256']:
            raise ValueError('Original training checkpoint changed during offline verification/profile')
        self.phase(name,'complete',identity=identity,profiles=profiles,
            deployment_receipt=str(directory/'deployment_receipt.json'),ordinary_inference_proof=str(proof),
            descriptor=str(descriptor_path),model_trained=True,completed_optimizer_steps=20000,
            ordinary_inference_labels_read=False,ordinary_profile_labels_read=False)
    def run(self):
        with (self.root/'deployment_watch.lock').open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.publish()
            while True:
                for key,name,mode in RUNS:
                    if self.state['runs'].get(name,{}).get('stage') in ('complete','failed'):continue
                    try:
                        identity,pending=completed_run(self.root,key,mode)
                        if identity is None:self.state['runs'][name]=dict(run_key=key,**pending)
                        else:self.process(key,name,mode,identity)
                    except Exception as exc:self.phase(name,'failed',error=repr(exc))
                stages=[self.state['runs'].get(name,{}).get('stage') for _,name,_ in RUNS]
                self.state['status']='complete' if all(v=='complete' for v in stages) else 'failed' if all(v in ('complete','failed') for v in stages) else 'waiting_for_training'
                self.publish()
                if self.once or self.state['status'] in ('complete','failed'):return int(self.state['status']=='failed')
                time.sleep(self.interval)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'results/precision_study_20261122')
    parser.add_argument('--interval',type=float,default=15.);parser.add_argument('--once',action='store_true')
    args=parser.parse_args(argv)
    if not 1<=args.interval<=60:parser.error('Require a polling interval between 1 and 60 seconds')
    return DeploymentWatcher(args.root,args.interval,args.once).run()


if __name__=='__main__':raise SystemExit(main())
