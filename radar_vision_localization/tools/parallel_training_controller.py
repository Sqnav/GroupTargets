"""Independent accelerated seven-method scheduler; frozen rvloc source is reused.

The initial layer runs resumed G2, fresh G3, and CPU FFT TRAIN inference in
parallel. Their configurations share the same initial snapshot. Only committed
artifacts enter the remaining, unmodified TRAIN/DEV stages. No TEST is read.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import signal
import sys
import threading

from rvloc.config import code_root, external_path
from rvloc.experiment import _Runner, _digest, _utc, stage_plan
from rvloc.runtime import file_sha256, write_json

PARALLEL_NAMES = ('train_g2', 'train_g3', 'train_infer_g1')
TOOLS = ('accelerated_batch_model.py', 'accelerated_training.py', 'accelerated_world_training.py', 'parallel_fft_inference.py', 'parallel_training_controller.py')


def _tools_identity():
    root = Path(__file__).resolve().parent
    return {name:file_sha256(root/name) for name in TOOLS}


class ParallelRunner(_Runner):
    def __init__(self, config_path, run_dir, *, index_path, devices=('cpu',),
                 resume_checkpoint=None, batch_size=4, fft_workers=16, world_engine='accelerated', progress_log=None,
                 resume=False, process_runner=None):
        super().__init__(config_path, run_dir, devices=devices, resume=resume,
                         process_runner=process_runner)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        self.index_path = external_path(index_path, field='fixed ROI labels')
        index = json.loads(self.index_path.read_text())
        if (index.get('schema') != 'rvloc.fixed_roi_gt_index.v1'
                or index.get('supervision_only') is not True
                or index.get('normal_inference_input') is not False
                or index.get('manifest_sha256') != self.identity['manifest_sha256']):
            raise ValueError('A matching frozen supervision-only ROI label index is required')
        self.resume_checkpoint = external_path(resume_checkpoint, field='G2 continuation checkpoint',
                                               allow_none=True)
        self.batch_size = batch_size
        if type(fft_workers) is not int or fft_workers < 1:
            raise ValueError('fft_workers must be a positive integer')
        self.fft_workers = fft_workers
        if world_engine not in ('accelerated','original'):
            raise ValueError('world_engine must be accelerated or original')
        self.world_engine = world_engine
        self.progress_log = external_path(progress_log, field='unified progress log') if progress_log else self.run_dir/'progress.log'
        self.parent = None
        self.parallel_children = {}
        self.parallel_state = 'waiting'
        self._cancel_parallel = False
        self._io_lock = threading.RLock()
        self._protocol_patched = False
        self.identity.update(execution_mode='parallel_initial_layer',
            initial_parallel_stages=list(PARALLEL_NAMES),
            accelerated_methods=[2,3,7] if world_engine=='accelerated' else [2,3],
            causal_unbatched_methods=[] if world_engine=='accelerated' else [7],world_engine=world_engine,
            fixed_index=dict(path=str(self.index_path), sha256=file_sha256(self.index_path)),
            tools_sha256=_tools_identity(), batch_size=batch_size, fft_workers=fft_workers,
            resume_checkpoints={'2':dict(path=str(self.resume_checkpoint),sha256=file_sha256(self.resume_checkpoint))}
                if self.resume_checkpoint else {},
            optimizer_update_policy='episode-bounded four-frame updates' if batch_size==4
                else 'explicit changed batch protocol; consult training receipts',
            progress_log=str(self.progress_log))
        self.identity.pop('sha256',None)
        self.identity['sha256'] = _digest(self.identity)

    def _assert_identity(self):
        super()._assert_identity()
        if file_sha256(self.index_path) != self.identity['fixed_index']['sha256']:
            raise ValueError('Frozen ROI labels changed during this run')
        if _tools_identity() != self.identity['tools_sha256']:
            raise ValueError('Accelerated tool source changed during this run')
        if self.resume_checkpoint and file_sha256(self.resume_checkpoint) != self.identity['resume_checkpoints']['2']['sha256']:
            raise ValueError('G2 source continuation checkpoint changed')

    def _line(self, stage, line, stage_stream=None):
        with self._io_lock:
            message = f'[{_utc()}] [{stage or "experiment"}] {line.rstrip()}'
            self.last_lines.append(message)
            print(message,flush=True)
            if self.log_stream is not None:
                self.log_stream.write(message+'\n');self.log_stream.flush()
            if stage_stream is not None:
                stage_stream.write(message+'\n');stage_stream.flush()
            # Trainers emit one complete row containing their real online
            # fixed-ROI 3D statistic, loss and model elapsed/ETA.
            if '[PROGRESS]' in line or 'TRAIN_ACCEL' in line or ('TRAIN G7 ' in line and 'epoch=' in line):
                self.progress_log.parent.mkdir(parents=True,exist_ok=True)
                with self.progress_log.open('a') as stream:stream.write(message+'\n')

    def _emit(self,line):
        if self.parent is not None:
            self.parent._line(self.current,line,self.stage_stream)
        else:
            self._line(self.current,line,self.stage_stream)

    def _heartbeat(self):
        if self.parent is not None:
            if self.parent._cancel_parallel and self.child_pid is not None:
                try:os.killpg(self.child_pid,signal.SIGTERM)
                except ProcessLookupError:pass
            return self.parent._heartbeat()
        with self._io_lock:
            # The reused source runner writes its generic protocol before its
            # first heartbeat; publish our declared schedule before any stage.
            protocol_path=self.run_dir/'protocol.json'
            if not self._protocol_patched and protocol_path.exists():
                value=json.loads(protocol_path.read_text())
                value.update(parallel_training=True,initial_parallel_stages=list(PARALLEL_NAMES),
                    accelerated_methods=[2,3,7] if self.world_engine=='accelerated' else [2,3],
                    causal_unbatched_methods=[] if self.world_engine=='accelerated' else [7],world_engine=self.world_engine,
                    optimizer_resumption=bool(self.resume_checkpoint),
                    fixed_index=str(self.index_path),batch_size=self.batch_size,fft_workers=self.fft_workers,
                    progress_log=str(self.progress_log),tools_sha256=self.identity['tools_sha256'])
                write_json(protocol_path,value);self._protocol_patched=True
            super()._heartbeat()
            status_path=self.run_dir/'status.json'
            value=json.loads(status_path.read_text())
            certified=list(self.completed)+[name for name in PARALLEL_NAMES
                if name not in self.completed and name in self.parallel_children
                and self.parallel_children[name].parallel_state=='completed']
            value.update(completed_stages=certified,completed_count=len(certified),active_stages={name:dict(state=ctx.parallel_state,pid=ctx.child_pid,
                    device=ctx._device(next(s for s in stage_plan() if s['name']==name)))
                    for name,ctx in self.parallel_children.items() if ctx.parallel_state not in ('completed','failed')},
                child_pids={name:ctx.child_pid for name,ctx in self.parallel_children.items()
                            if ctx.child_pid is not None},
                parallel_stage_states={name:ctx.parallel_state for name,ctx in self.parallel_children.items()},
                parallel_training=True,progress_log=str(self.progress_log))
            write_json(status_path,value)

    def _prepare(self,stage,attempt):
        command,outputs=super()._prepare(stage,attempt)
        if stage['kind']=='train' and stage['method'] in (2,3):
            command=[sys.executable,'-u',str(Path(__file__).with_name('accelerated_training.py')),
                '--config',str(attempt/'config.json'),'--method',str(stage['method']),
                '--index',str(self.index_path),'--output-dir',str(attempt/'results'),
                '--device',self._device(stage),'--batch-size',str(self.batch_size)]
            if stage['method']==2 and self.resume_checkpoint is not None:
                command += ['--resume-checkpoint',str(self.resume_checkpoint)]
        elif stage['kind']=='train' and stage['method']==7 and self.world_engine=='accelerated':
            command=[sys.executable,'-u',str(Path(__file__).with_name('accelerated_world_training.py')),
                '--config',str(attempt/'config.json'),'--method','7',
                '--index',str(self.index_path),'--output-dir',str(attempt/'results'),
                '--device',self._device(stage),'--batch-size',str(self.batch_size)]
        elif stage['kind']=='infer' and stage['method']==1:

            command=[sys.executable,'-u',str(Path(__file__).with_name('parallel_fft_inference.py')),
                '--config',str(attempt/'config.json'),'--split',stage['split'],
                '--output-dir',str(attempt/'results'),'--workers',str(self.fft_workers)]
        return command,outputs

    def _parallel_initial(self):
        self.current='parallel_train_g2_g3_fft_train'
        self._assert_identity()
        snapshots={}
        stages=[next(s for s in stage_plan() if s['name']==name) for name in PARALLEL_NAMES]
        for stage in stages:
            ctx=copy.copy(self)
            ctx.parent=self;ctx.current=stage['name'];ctx.child_pid=None
            ctx.stage_stream=None;ctx.completed=[];ctx.receipts={}
            ctx.predictions={};ctx.evaluations={};ctx.comparisons={}
            ctx.raw=copy.deepcopy(self.raw);ctx.parallel_state='waiting'
            self.parallel_children[stage['name']]=ctx;snapshots[stage['name']]=ctx
        self._emit('PARALLEL BEGIN: G2 continuation, G3 fresh training, CPU G1 TRAIN FFT')
        self._heartbeat()

        def work(stage):
            ctx=snapshots[stage['name']];ctx.parallel_state='running';ctx._heartbeat()
            try:
                _Runner._run_stage(ctx,stage)
                record=ctx.receipts[stage['name']]
                record['resumption']='G2 restores the declared source optimizer; failed attempts restart their declared stage source' if stage['name']=='train_g2' else 'fresh stage or SHA-verified committed artifacts'
                record['execution_engine']='accelerated_batch' if stage['name'] in ('train_g2','train_g3') else 'parallel_fft'
                receipt_path=self.run_dir/'stage_receipts'/(stage['name']+'.json')
                write_json(receipt_path,record)
                attempts=sorted((self.run_dir/'stages'/stage['name']).glob('attempt_*'))
                write_json(attempts[-1]/'attempt_receipt.json',record)
                ctx.parallel_state='completed'
            except BaseException:
                ctx.parallel_state='failed';raise
            finally:ctx._heartbeat()

        failures=[]
        with ThreadPoolExecutor(max_workers=len(stages),thread_name_prefix='initial-layer') as pool:
            futures={pool.submit(work,stage):stage for stage in stages}
            try:
                for future in as_completed(futures):
                    try:future.result()
                    except BaseException as error:
                        failures.append((futures[future]['name'],error));self._cancel_parallel=True
                        for ctx in snapshots.values():
                            if ctx.child_pid is not None:
                                try:os.killpg(ctx.child_pid,signal.SIGTERM)
                                except ProcessLookupError:pass
            except BaseException:
                # Ctrl-C must stop this controller's children before the pool
                # context joins its threads; unrelated GPU jobs are untouched.
                self._cancel_parallel=True
                for ctx in snapshots.values():
                    if ctx.child_pid is not None:
                        try:os.killpg(ctx.child_pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                raise
        if failures:
            raise RuntimeError('Parallel initial layer failed: '+'; '.join(f'{name}: {error}' for name,error in failures))
        # Committed artifacts are collected in protocol order, independent of
        # thread completion order. Every initial stage had empty dependencies.
        for stage in stages:
            receipt=snapshots[stage['name']].receipts[stage['name']]
            self._collect(stage,receipt)
            self.receipts[stage['name']]=receipt;self.completed.append(stage['name'])
        self._emit('PARALLEL COMPLETE: committed initial artifacts collected in fixed protocol order')
        self._heartbeat()

    def _run_stage(self,stage):
        if stage['name']=='train_g2':
            return self._parallel_initial()
        if stage['name'] in PARALLEL_NAMES:
            return  # Already certified by the initial parallel layer.
        return super()._run_stage(stage)

    def _report(self,attempt):
        super()._report(attempt)
        path=attempt/'experiment_summary.json';value=json.loads(path.read_text())
        value['protocol'].update(timing='parallel initial G2/G3/FFT TRAIN layer, then dependency-ordered stages',
            resumed_g2=bool(self.resume_checkpoint),batch_size=self.batch_size,
            initial_parallel_stages=list(PARALLEL_NAMES),
            continuation_cost_note='New-run elapsed time excludes earlier G2 epochs; source checkpoint identity is retained')
        value['protocol']['world_engine']=self.world_engine
        value['acceleration_identity']=self.identity
        write_json(path,value)
        with (attempt/'SEVEN_METHODS_TRAINING_REPORT_CN.md').open('a') as stream:
            stream.write('\n## 并行调度与续训\n\nG2续训、G3独立训练与G1的TRAIN FFT推理同时运行；随后按依赖执行TRAIN标定、因果G7训练和冻结DEV评价。')
            stream.write(f'批量设置为 {self.batch_size} 帧；源检查点和三个加速工具的SHA保存在执行身份中。')
            stream.write('新运行耗时不包含G2源检查点之前已完成的训练时间；总实验成本应合并原运行收据。\n')


def run_parallel(config_path,run_dir,*,index_path,devices=('cpu',),resume_checkpoint=None,
                 batch_size=4,fft_workers=16,world_engine='accelerated',progress_log=None,resume=False,process_runner=None):
    return ParallelRunner(config_path,run_dir,index_path=index_path,devices=devices,
        resume_checkpoint=resume_checkpoint,batch_size=batch_size,fft_workers=fft_workers,world_engine=world_engine,progress_log=progress_log,
        resume=resume,process_runner=process_runner).run()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--index',required=True);p.add_argument('--devices',default='cpu')
    p.add_argument('--resume-checkpoint');p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--fft-workers',type=int,default=16)
    p.add_argument('--world-engine',choices=('accelerated','original'),default='accelerated')
    p.add_argument('--progress-log');p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    result=run_parallel(a.config,a.run_dir,index_path=a.index,devices=a.devices,
        resume_checkpoint=a.resume_checkpoint,batch_size=a.batch_size,fft_workers=a.fft_workers,world_engine=a.world_engine,
        progress_log=a.progress_log,resume=a.resume)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
