"""Fake committed DEV artifacts, no model, real data or subprocess involved."""
import importlib.util
import json
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[1]/'final_result_lines.py'
spec=importlib.util.spec_from_file_location('final_lines',path);tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def fixture(tmp):
    run=tmp/'run';fixed=tmp/'fixed';out=tmp/'progress'
    write(run/'execution_identity.json',dict(sha256='identity',manifest_sha256='manifest'))
    write(run/'status.json',dict(identity_sha256='identity',status='running',elapsed_s=3661))
    return tool.ResultLines(run,fixed,out)


def artifact(monitor,method,*,split='dev'):
    fixed=method in (2,3,5,6,7)
    base=monitor.fixed/f'g{method}' if fixed else monitor.run/'stages'/f'dev_evaluate_g{method}'
    value=dict(schema='rvloc.fixed_roi_evaluation.v1' if fixed else 'rvloc.evaluation.v1',method_id=method,
        split=split,manifest_sha256='manifest',summary={'position_rmse_3d_m':1.25},frozen_index_sha256='index',predictions_sha256='pred')
    evaluation=write(base/'evaluation.json',value)
    r=dict(status='completed',elapsed_s=2,artifacts={str(evaluation):tool.sha(evaluation)})
    if fixed:r.update(method_id=method,split='dev',evaluation=str(evaluation),frozen_index_sha256='index',predictions_sha256='pred');target=base/'receipt.json'
    else:r.update(stage=dict(name=f'dev_evaluate_g{method}',kind='evaluate',method=method,split='dev'),identity_sha256='identity');target=monitor.run/'stage_receipts'/f'dev_evaluate_g{method}.json'
    write(target,r)
    return evaluation


def test_pending_does_not_fabricate_results(tmp_path):
    m=fixture(tmp_path);s=m.run_forever(once=True)
    assert s['status']=='waiting' and s['emitted_methods']==[]
    assert not (m.output/'progress.log').exists()


def test_fixed_and_fft_have_distinct_scope_and_one_line(tmp_path):
    m=fixture(tmp_path);artifact(m,1);artifact(m,3);m.tick()
    lines=(m.output/'progress.log').read_text().splitlines()
    assert len(lines)==2 and 'G1 DEV_SET_RMSE=1.2500m' in lines[0] and 'G3 DEV_FIXED_RMSE=1.2500m' in lines[1]
    assert '本次评估耗时=00:00:02 总已用=01:01:01' in lines[0]
    m.tick();assert len((m.output/'progress.log').read_text().splitlines())==2
    assert (m.output/'progress.log').read_text()==(m.run/'run.log').read_text()


def test_train_cannot_be_forwarded_as_dev(tmp_path):
    m=fixture(tmp_path);artifact(m,3,split='train')
    with pytest.raises(ValueError,match='DEV'):m.tick()
    assert not (m.output/'progress.log').exists()


def test_changed_evaluation_sha_rejected(tmp_path):
    m=fixture(tmp_path);p=artifact(m,1);p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError,match='SHA'):m.tick()


def test_terminal_failure_preserves_pending(tmp_path):
    m=fixture(tmp_path)
    write(m.run/'status.json',dict(identity_sha256='identity',status='failed',elapsed_s=2))
    assert m.run_forever(once=True)['status']=='incomplete'


def test_main_completion_waits_for_independent_fixed_evaluation(tmp_path):
    m=fixture(tmp_path);artifact(m,1);artifact(m,4)
    write(m.run/'status.json',dict(identity_sha256='identity',status='completed',elapsed_s=2))
    write(m.fixed/'watch_status.json',dict(status='evaluating_g7'))
    assert m.tick()['status']=='waiting'
    for method in (2,3,5,6,7):artifact(m,method)
    write(m.fixed/'watch_status.json',dict(status='completed'))
    assert m.tick()['status']=='completed'


def test_fixed_watcher_failure_does_not_wait_forever(tmp_path):
    m=fixture(tmp_path);artifact(m,1);artifact(m,4)
    write(m.run/'status.json',dict(identity_sha256='identity',status='completed',elapsed_s=2))
    write(m.fixed/'watch_status.json',dict(status='failed'))
    assert m.tick()['status']=='incomplete'
