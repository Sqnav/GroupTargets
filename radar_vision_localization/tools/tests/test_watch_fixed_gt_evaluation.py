"""Synthetic committed-receipt checks; never open real data or start training."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from watch_fixed_gt_evaluation import METHODS, _sha, watch


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def setup(tmp_path):
    run = tmp_path/'main_run'
    index = dump(tmp_path/'labels/index.json', dict(schema='rvloc.fixed_roi_gt_index.v1',
        supervision_only=True, manifest_sha256='a'*64))
    dump(run/'execution_identity.json', dict(sha256='b'*64, manifest_sha256='a'*64))
    dump(run/'status.json', dict(status='running', identity_sha256='b'*64))
    output = tmp_path/'supplement'
    return run, index, output


def commit(run, method):
    prediction = dump(run/f'stages/dev_infer_g{method}/attempt_0001/results/predictions.json',
        dict(schema='rvloc.predictions.v1', method_id=method, split='dev',
             manifest_sha256='a'*64, normal_inference=True, labels_opened=False, frames=[]))
    receipt = dump(run/f'stage_receipts/dev_infer_g{method}.json', dict(status='completed',
        stage=dict(name=f'dev_infer_g{method}', kind='infer', method=method, split='dev'),
        identity_sha256='b'*64, artifacts={str(prediction):_sha(prediction)}))
    return prediction, receipt


class FakeEvaluation:
    def __init__(self):
        self.calls = []

    def __call__(self, index, predictions, output):
        self.calls.append((index, predictions, output))
        assert not output.exists()
        method = json.loads(predictions.read_text())['method_id']
        summary = dict(position_rmse_3d_m=.5, gt_members=3, localized_members=3)
        evaluation = dump(output/'evaluation.json', dict(schema='rvloc.fixed_roi_evaluation.v1',
            method_id=method, split='dev', manifest_sha256='a'*64,
            frozen_index_sha256=_sha(index), predictions_sha256=_sha(predictions), summary=summary))
        dump(output/'localization_metrics.json', summary)
        return dict(status='evaluated', evaluation=str(evaluation), summary=summary)


def test_once_without_predictions_is_waiting_with_no_fabricated_metrics(setup):
    run, index, output = setup
    fake = FakeEvaluation()
    result = watch(index, run, output, once=True, evaluator=fake)
    assert result['status'] == 'waiting'
    assert result['pending_methods'] == list(METHODS)
    assert fake.calls == []
    summary = json.loads((output/'fixed_metrics_summary.json').read_text())
    for method in METHODS:
        assert summary['methods'][str(method)]['summary'] is None
    assert summary['methods']['1']['status'] == 'not_applicable'
    assert summary['methods']['4']['status'] == 'not_applicable'
    assert (output/'watch_status.json').is_file()
    assert (output/'watch_receipt.json').is_file()
    assert (output/'watch.log').is_file()


def test_completed_receipt_evaluates_once_and_resumes_by_verified_artifacts(setup):
    run, index, output = setup
    commit(run, 2)
    fake = FakeEvaluation()
    result = watch(index, run, output, once=True, evaluator=fake)
    assert result['completed_methods'] == [2]
    assert len(fake.calls) == 1
    assert (output/'g2/receipt.json').is_file()
    again = watch(index, run, output, once=True, evaluator=fake)
    assert again['completed_methods'] == [2]
    assert len(fake.calls) == 1
    assert not (output/'g2/attempt_0002').exists()


def test_all_visual_groups_complete_and_fft_stays_not_applicable(setup):
    run, index, output = setup
    for method in METHODS:
        commit(run, method)
    dump(run/'status.json', dict(status='completed', identity_sha256='b'*64))
    fake = FakeEvaluation()
    result = watch(index, run, output, once=True, evaluator=fake)
    assert result['status'] == 'completed'
    assert result['pending_methods'] == []
    assert len(fake.calls) == 5
    assert not (output/'g1').exists() and not (output/'g4').exists()
    assert json.loads((output/'fixed_metrics_summary.json').read_text())['methods']['7']['summary']['position_rmse_3d_m'] == .5


def test_tampered_predictions_fail_closed_before_evaluation(setup):
    run, index, output = setup
    prediction, _ = commit(run, 3)
    prediction.write_text(prediction.read_text()+'\n')
    fake = FakeEvaluation()
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        watch(index, run, output, once=True, evaluator=fake)
    assert fake.calls == []
    assert json.loads((output/'watch_failure.json').read_text())['pending_methods'] == list(METHODS)
    assert json.loads((run/'status.json').read_text())['status'] == 'running'


def test_main_failure_preserves_finished_results_and_reports_unfinished(setup):
    run, index, output = setup
    commit(run, 2)
    dump(run/'status.json', dict(status='failed', identity_sha256='b'*64))
    fake = FakeEvaluation()
    with pytest.raises(RuntimeError, match='Main run status is failed'):
        watch(index, run, output, once=True, evaluator=fake)
    status = json.loads((output/'watch_status.json').read_text())
    assert status['status'] == 'failed'
    assert status['completed_methods'] == [2]
    assert status['pending_methods'] == [3, 5, 6, 7]
    assert (output/'g2/attempt_0001/evaluation.json').is_file()
    assert json.loads((run/'status.json').read_text())['status'] == 'failed'


def test_completed_main_run_missing_expected_receipts_is_not_waiting_forever(setup):
    run, index, output = setup
    dump(run/'status.json', dict(status='completed', identity_sha256='b'*64))
    with pytest.raises(RuntimeError, match='without every expected'):
        watch(index, run, output, once=True, evaluator=FakeEvaluation())


def test_wrong_source_identity_and_changed_completed_outputs_are_rejected(setup):
    run, index, output = setup
    _, stage = commit(run, 5)
    receipt = json.loads(stage.read_text());receipt['identity_sha256'] = 'different'
    dump(stage, receipt)
    with pytest.raises(ValueError, match='different main run'):
        watch(index, run, output, once=True, evaluator=FakeEvaluation())
