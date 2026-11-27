"""Generated progress/checkpoints only; no training, sensors or GPU touched."""
import importlib.util
import json
from pathlib import Path

import pytest

script = Path(__file__).resolve().parents[1]/'live_training_progress.py'
spec = importlib.util.spec_from_file_location('progress_tool',script)
tool = importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value));return path


def fixture(tmp_path):
    main=tmp_path/'main';main.mkdir()
    write(main/'execution_identity.json',dict(sha256='a'*64,manifest_sha256='b'*64))
    write(main/'status.json',dict(identity_sha256='a'*64,status='running',current_stage='train_g2',elapsed_s=500))
    index=write(tmp_path/'index.json',dict(schema='rvloc.fixed_roi_gt_index.v1',supervision_only=True,
        manifest_sha256='b'*64,summaries={'train':{'frames':100}}))
    attempt=main/'stages/train_g2/attempt_0001'
    write(attempt/'config.json',dict(storage=dict(output_root=str(tmp_path),weights_root=str(tmp_path/'weights'))))
    write(attempt/'results/progress.json',dict(status='training',epoch=4,epochs=80,frames=50,
        optimizer_updates=125,last_loss=.42,elapsed_s=400,eta_s=900))
    return tool.ProgressMonitor(main,index,tmp_path/'monitor',device='cpu'),main


def test_one_line_has_true_loss_scoped_rmse_and_both_times():
    main=dict(current_stage='train_g2',status='running',elapsed_s=3661)
    progress=dict(epoch=9,epochs=80,optimizer_updates=100,last_loss=.42,elapsed_s=3610)
    line=tool.format_line(main,progress,2,dict(completed_epochs=7,position_rmse_3d_m=1.23456),eta=120)
    assert '\n' not in line
    assert 'loss=0.420000' in line and 'TRAIN_RMSE@E7=1.2346m' in line
    assert '本模型已用=01:00:10' in line and '本模型预计剩余=00:02:00' in line
    assert '总已用=01:01:01' in line


def test_pending_never_converts_loss_to_rmse():
    line=tool.format_line({},dict(last_loss=4),2)
    assert 'TRAIN_RMSE=pending' in line and 'TRAIN_RMSE@' not in line
    assert 'TRAIN_RMSE=2' not in line
    assert tool.hms(None)=='待估计'
    assert tool.hms(float('nan'))=='待估计'


def test_eta_uses_reported_value_and_falls_back_to_actual_progress():
    assert tool.stage_eta(dict(status='training',eta_s=7),100)==7
    assert tool.stage_eta(dict(status='completed',eta_s=7),100)==0
    assert tool.stage_eta(dict(epoch=2,epochs=3,frames=50,elapsed_s=30),100)==30
    assert tool.stage_eta(dict(epoch=1,epochs=3,frames=0,elapsed_s=30),100) is None


def test_once_reads_real_progress_without_starting_worker(tmp_path,monkeypatch):
    monitor,main=fixture(tmp_path)
    monkeypatch.setattr(tool.subprocess,'Popen',lambda *a,**k:pytest.fail('once must never start an evaluator'))
    state=monitor.run(once=True)
    assert state['worker_status']=='pending'
    line=(monitor.output_dir/'progress.log').read_text()
    assert 'loss=0.420000' in line and 'TRAIN_RMSE=pending' in line
    assert line==(main/'run.log').read_text()
    assert read(monitor.output_dir/'status.json')['current_stage']=='train_g2'


def read(path): return json.loads(path.read_text())


def test_changed_frozen_index_is_rejected(tmp_path):
    monitor,_=fixture(tmp_path)
    monitor.index_path.write_text(monitor.index_path.read_text()+'\n')
    with pytest.raises(ValueError,match='changed'):monitor.tick(start_worker=False)


def test_measured_checkpoint_epoch_is_not_confused_with_current_epoch(tmp_path):
    monitor,_=fixture(tmp_path)
    config=monitor.context(read(monitor.run_dir/'status.json'))['config']
    monitor.metrics[2]=dict(metric=dict(completed_epochs=2,position_rmse_3d_m=.7),
        training_config_path=str(config),training_config_sha256=tool.sha(config))
    state=monitor.tick(start_worker=False)
    assert 'epoch=4/80' in state['line'] and 'TRAIN_RMSE@E2=0.7000m' in state['line']


def test_failed_worker_is_explicit_not_zero_rmse():
    line=tool.format_line({},dict(last_loss=0),2,worker_status='failed')
    assert 'TRAIN_RMSE=failed' in line and 'TRAIN_RMSE=0' not in line


def test_another_training_attempt_cannot_borrow_previous_rmse(tmp_path):
    monitor,_=fixture(tmp_path)
    monitor.metrics[2]=dict(metric=dict(completed_epochs=2,position_rmse_3d_m=.7),
        training_config_path='different_attempt/config.json',training_config_sha256='wrong')
    state=monitor.tick(start_worker=False)
    assert 'TRAIN_RMSE=pending' in state['line'] and '0.7000m' not in state['line']
