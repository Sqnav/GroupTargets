"""Synthetic subprocess-boundary checks, without data, GPU or real weights."""
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
from unittest.mock import patch

import pytest

script=Path(__file__).resolve().parents[1]/'parallel_training_controller.py'
spec=importlib.util.spec_from_file_location('parallel_controller_test',script)
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value));return path


def fixture(tmp):
    root=tmp/'results'
    manifest=write(tmp/'manifest.json',{'synthetic':True})
    index=write(tmp/'index.json',dict(schema='rvloc.fixed_roi_gt_index.v1',
        supervision_only=True,normal_inference_input=False,
        manifest_sha256=tool.file_sha256(manifest)))
    config=json.loads((tool.code_root()/'configs/seven_methods.json').read_text())
    config.update(manifest=str(manifest))
    config['storage']=dict(output_root=str(root),weights_root=str(tmp/'weights'),cache_root=str(tmp/'cache'))
    cfg=write(tmp/'config.json',config)
    old=tmp/'epoch_0057.pt';old.write_bytes(b'synthetic checkpoint source')
    return cfg,index,old,root/'parallel_run'


class FakeCommands:
    def __init__(self,parallel=False):
        self.commands=[];self.barrier=threading.Barrier(3) if parallel else None
        self.lock=threading.Lock()

    def __call__(self,command,*,env,emit,heartbeat):
        def arg(name):return command[command.index(name)+1]
        with self.lock:self.commands.append(command)
        script=Path(command[2]).name
        kind='train' if script in ('accelerated_training.py','accelerated_world_training.py') else 'infer' if script=='parallel_fft_inference.py' else command[4]
        method=int(arg('--method')) if '--method' in command else 1 if script=='parallel_fft_inference.py' else None
        split=arg('--split') if '--split' in command else None
        if self.barrier and ((kind=='train' and method in (2,3)) or (kind=='infer' and method==1 and split=='train')):
            self.barrier.wait(timeout=5)
        config=json.loads(Path(arg('--config')).read_text()) if '--config' in command else None
        out=Path(arg('--output-dir')) if '--output-dir' in command else None
        emit('[PROGRESS] synthetic step=1 loss=.25 RMSE_online=.5 elapsed=1 ETA=0')
        heartbeat()
        if kind=='train':
            rel=out.relative_to(Path(config['storage']['output_root']))
            weights=Path(config['storage']['weights_root'])/rel
            weights.mkdir(parents=True)
            checkpoint=weights/'model.pt';checkpoint.write_bytes(b'synthetic trained checkpoint')
            write(out/'receipt.json',dict(status='trained',method_id=method,checkpoint=str(checkpoint)))
            write(out/'history.json',[dict(epoch=80,optimizer_updates=1)])
        elif kind=='infer':
            write(out/'predictions.json',dict(method_id=method,split=split))
            write(out/'reproduction_receipt.json',dict(status='inference_completed'))
        elif kind=='calibrate':
            inputs=command[command.index('--predictions')+1:command.index('--output')]
            assert len(inputs)==3
            assert all(json.loads(Path(x).read_text())['split']=='train' for x in inputs)
            write(arg('--output'),dict(status='frozen',split='train'))
        elif kind=='evaluate':
            method=json.loads(Path(arg('--predictions')).read_text())['method_id']
            write(out/'evaluation.json',dict(schema='rvloc.evaluation.v1',method_id=method,
                summary=dict(position_rmse_3d_m=1.5,mae_m=1.,p95_m=2.,p99_m=3.),cost=dict(wall_s=1)))
            for name in ('localization_metrics.json','visual_prior_metrics.json','episode_level_results.json','failure_cases.json'):
                write(out/name,{})
        elif kind=='compare':
            write(arg('--output'),dict(status='synthetic_pair'))
        else:raise AssertionError(kind)
        return 0


def execute(cfg,index,old,run,fake,**options):
    with patch.object(tool,'_tools_identity',return_value={'synthetic_tool':'a'*64}),patch('builtins.print'):
        return tool.run_parallel(cfg,run,index_path=index,devices='0,1',resume_checkpoint=old,
                                 process_runner=fake,**options)


def test_initial_train_and_fft_really_overlap_then_complete_seven_methods(tmp_path):
    cfg,index,old,run=fixture(tmp_path);fake=FakeCommands(parallel=True)
    result=execute(cfg,index,old,run,fake)
    assert result['status']=='completed'
    assert len(fake.commands)==42
    summary=json.loads(Path(result['summary']).read_text())
    assert len(summary['methods'])==7
    assert summary['protocol']['resumed_g2'] is True
    assert summary['protocol']['initial_parallel_stages']==list(tool.PARALLEL_NAMES)
    first=fake.commands[:3]
    assert {Path(c[2]).name for c in first}=={'accelerated_training.py','parallel_fft_inference.py'}
    by_method={int(c[c.index('--method')+1]):c for c in first if '--method' in c}
    assert str(old) in by_method[2] and '--resume-checkpoint' not in by_method[3]
    assert all(json.loads(Path(c[c.index('--config')+1]).read_text())['checkpoints']['rgb'] is None for c in first)
    status=json.loads((run/'status.json').read_text())
    assert status['completed_count']==43 and status['status']=='completed'
    assert status['parallel_stage_states']==dict.fromkeys(tool.PARALLEL_NAMES,'completed')
    assert status['child_pids']=={}
    assert '[PROGRESS]' in (run/'progress.log').read_text()
    assert not list(run.rglob('*.pt'))
    assert len(list((tmp_path/'weights').rglob('model.pt')))==3


def test_resume_verifies_parallel_completed_artifacts_without_retraining(tmp_path):
    cfg,index,old,run=fixture(tmp_path)
    execute(cfg,index,old,run,FakeCommands())
    fake=FakeCommands()
    result=execute(cfg,index,old,run,fake,resume=True)
    assert result['status']=='completed' and not fake.commands


def test_fixed_index_or_source_checkpoint_change_rejects_resume(tmp_path):
    cfg,index,old,run=fixture(tmp_path)
    execute(cfg,index,old,run,FakeCommands())
    old.write_bytes(b'changed')
    with pytest.raises(ValueError,match='identical config'):
        execute(cfg,index,old,run,FakeCommands(),resume=True)


def test_sequential_tail_uses_trained_weights_and_calibration(tmp_path):
    cfg,index,old,run=fixture(tmp_path);fake=FakeCommands()
    execute(cfg,index,old,run,fake)
    world=next(c for c in fake.commands if '--method' in c and c[c.index('--method')+1]=='7' and Path(c[2]).name=='accelerated_world_training.py')
    raw=json.loads(Path(world[world.index('--config')+1]).read_text())
    assert raw['checkpoints']['rgb'] and raw['checkpoints']['single_frame']
    assert raw['calibration_artifact']
    protocol=json.loads((run/'protocol.json').read_text())
    assert protocol['parallel_training'] is True and protocol['optimizer_resumption'] is True
    assert protocol['fft_workers']==16
    assert protocol['world_engine']=='accelerated'
    assert '--batch-size' in world


@pytest.mark.parametrize('field,value',[('batch_size',0),('fft_workers',0)])
def test_invalid_execution_size_rejected(tmp_path,field,value):
    cfg,index,old,run=fixture(tmp_path)
    with pytest.raises(ValueError,match='positive integer'):
        execute(cfg,index,old,run,FakeCommands(),**{field:value})


def test_original_causal_world_engine_remains_an_explicit_fallback(tmp_path):
    cfg,index,old,run=fixture(tmp_path);fake=FakeCommands()
    execute(cfg,index,old,run,fake,world_engine='original')
    command=next(c for c in fake.commands if len(c)>4 and c[4]=='train' and '--method' in c and c[c.index('--method')+1]=='7')
    assert command[2:5]==['-m','rvloc','train']
    assert json.loads((run/'protocol.json').read_text())['causal_unbatched_methods']==[7]
