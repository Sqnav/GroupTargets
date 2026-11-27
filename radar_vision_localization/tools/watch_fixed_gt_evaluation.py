"""Evaluate committed visual predictions using immutable ROI-to-GT labels.

This companion never changes the running seven-method pipeline, its weights,
normal association, or original set-matching metrics. Only DEV inference
receipts committed by that run can trigger an offline fixed-label evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import time

METHODS = (2, 3, 5, 6, 7)
NOT_APPLICABLE = (1, 4)
SCHEMA = 'rvloc.fixed_evaluation_watch.v1'


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.writing-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _default_evaluator(index, predictions, output):
    path = Path(__file__).with_name('fixed_gt_correspondence.py')
    spec = importlib.util.spec_from_file_location('fixed_gt_correspondence_watch_backend', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate(index, predictions, output)


class FixedEvaluationWatcher:
    def __init__(self, index_path, run_dir, output_dir, *, evaluator=None):
        self.index_path = Path(index_path).expanduser().resolve()
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        project = Path(__file__).resolve().parents[1]
        _require(not self.output_dir.is_relative_to(project), 'Watcher results must stay outside the code directory')
        _require(not self.output_dir.is_relative_to(self.run_dir)
                 and not self.run_dir.is_relative_to(self.output_dir),
                 'Fixed-label supplement must be separate from the main run directory')
        index = _read(self.index_path)
        _require(index.get('schema') == 'rvloc.fixed_roi_gt_index.v1'
                 and index.get('supervision_only') is True, 'Expected a frozen supervision-only ROI-to-GT index')
        self.main_identity_path = self.run_dir/'execution_identity.json'
        main = _read(self.main_identity_path)
        _require(main.get('manifest_sha256') == index.get('manifest_sha256'),
                 'Frozen index and main run use different manifests')
        self.main_identity_sha256 = main['sha256']
        self.evaluator = evaluator or _default_evaluator
        self.identity = dict(schema=SCHEMA, run_dir=str(self.run_dir),
            index_path=str(self.index_path), frozen_index_sha256=_sha(self.index_path),
            manifest_sha256=index['manifest_sha256'],
            execution_identity_sha256=_sha(self.main_identity_path),
            main_run_identity=main['sha256'], watcher_sha256=_sha(__file__),
            evaluator_sha256=None if evaluator is not None else _sha(Path(__file__).with_name('fixed_gt_correspondence.py')),
            methods=list(METHODS), split='dev', training_changed=False,
            normal_inference_changed=False, original_set_metrics_changed=False,
            gt_usage='offline labels and scoring only; never online tracking or model memory')
        self.completed = {}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.main_status = 'unknown'
        self.status = 'starting'

    def log(self, message):
        line = f'[{_now()}] [fixed_roi_gt] {message}'
        print(line, flush=True)
        with (self.output_dir/'watch.log').open('a') as stream:
            stream.write(line+'\n')

    def _assert_identity(self):
        _require(_sha(self.index_path) == self.identity['frozen_index_sha256'], 'Frozen correspondence index changed')
        _require(_sha(self.main_identity_path) == self.identity['execution_identity_sha256'], 'Main execution identity changed')
        _require(_sha(__file__) == self.identity['watcher_sha256'], 'Watcher source changed; choose a new supplement run')
        expected = self.identity['evaluator_sha256']
        if expected is not None:
            _require(_sha(Path(__file__).with_name('fixed_gt_correspondence.py')) == expected,
                     'Fixed-label evaluator source changed; choose a new supplement run')

    def _stage(self, method):
        path = self.run_dir/'stage_receipts'/f'dev_infer_g{method}.json'
        if not path.exists():
            return None
        receipt = _read(path)
        expected = dict(name=f'dev_infer_g{method}', kind='infer', method=method, split='dev')
        _require(receipt.get('status') == 'completed' and receipt.get('stage') == expected,
                 f'G{method} inference receipt is not the completed DEV stage')
        _require(receipt.get('identity_sha256') == self.main_identity_sha256,
                 f'G{method} stage belongs to a different main run')
        candidates = [(Path(p).resolve(), digest) for p, digest in receipt.get('artifacts', {}).items()
                      if Path(p).name == 'predictions.json']
        _require(len(candidates) == 1, f'G{method} receipt needs exactly one predictions.json artifact')
        prediction, digest = candidates[0]
        _require(prediction.is_relative_to(self.run_dir), f'G{method} predictions escaped the main run directory')
        return path, receipt, prediction, digest

    def _restore(self):
        for method in METHODS:
            receipt_path = self.output_dir/f'g{method}'/'receipt.json'
            if not receipt_path.exists():
                continue
            receipt = _read(receipt_path)
            _require(receipt.get('status') == 'completed' and receipt.get('method_id') == method,
                     f'Invalid fixed-evaluation receipt for G{method}')
            _require(receipt.get('frozen_index_sha256') == self.identity['frozen_index_sha256'],
                     f'G{method} fixed evaluation used different labels')
            stage = self._stage(method)
            _require(stage is not None and _sha(stage[0]) == receipt['stage_receipt_sha256'],
                     f'G{method} completed inference receipt changed')
            _require(_sha(stage[2]) == stage[3] == receipt['predictions_sha256'],
                     f'G{method} completed predictions changed')
            for path, digest in receipt['artifacts'].items():
                target = Path(path).resolve()
                _require(target.is_relative_to(self.output_dir/f'g{method}')
                         and target.is_file() and _sha(target) == digest,
                         f'G{method} fixed-evaluation artifact missing or changed: {path}')
            self.completed[method] = receipt
            self.log(f'RESUME G{method}: verified completed evaluation; no recomputation')

    def _evaluate(self, method, stage):
        path, receipt, predictions, expected_sha = stage
        _require(predictions.is_file() and _sha(predictions) == expected_sha,
                 f'G{method} predictions artifact SHA256 mismatch')
        header = _read(predictions)
        _require(header.get('schema') == 'rvloc.predictions.v1' and header.get('method_id') == method
                 and header.get('split') == 'dev' and header.get('normal_inference') is True
                 and header.get('labels_opened') is False
                 and header.get('manifest_sha256') == self.identity['manifest_sha256'],
                 f'G{method} predictions are not persisted sensor-only DEV outputs for this manifest')
        del header
        group = self.output_dir/f'g{method}'
        group.mkdir(exist_ok=True)
        number = 1
        while (group/f'attempt_{number:04d}').exists():
            number += 1
        output = group/f'attempt_{number:04d}'
        self.log(f'START G{method}: certified predictions={predictions} attempt={number}')
        began = time.monotonic()
        result = self.evaluator(self.index_path, predictions, output)
        evaluation = Path(result['evaluation']).resolve()
        _require(evaluation.is_relative_to(output), f'G{method} evaluator returned an external output')
        value = _read(evaluation)
        _require(value.get('schema') == 'rvloc.fixed_roi_evaluation.v1' and value.get('method_id') == method
                 and value.get('split') == 'dev' and value.get('manifest_sha256') == self.identity['manifest_sha256']
                 and value.get('frozen_index_sha256') == self.identity['frozen_index_sha256']
                 and value.get('predictions_sha256') == expected_sha and isinstance(value.get('summary'), dict),
                 f'G{method} evaluator did not certify the expected immutable inputs')
        _require(_sha(predictions) == expected_sha, f'G{method} predictions changed during fixed evaluation')
        self._assert_identity()
        completed = dict(schema='rvloc.fixed_evaluation_stage.v1', status='completed', method_id=method,
            split='dev', completed_at_utc=_now(), elapsed_s=time.monotonic()-began,
            stage_receipt=str(path), stage_receipt_sha256=_sha(path), predictions=str(predictions),
            predictions_sha256=expected_sha, frozen_index_sha256=self.identity['frozen_index_sha256'],
            evaluation=str(evaluation), summary=value['summary'],
            artifacts={str(p):_sha(p) for p in output.rglob('*.json')},
            normal_inference_changed=False, main_pipeline_changed=False)
        _write(group/'receipt.json', completed)
        self.completed[method] = completed
        self.log(f'COMPLETE G{method}: RMSE={value["summary"].get("position_rmse_3d_m")} '
                 f'elapsed_s={completed["elapsed_s"]:.3f} evaluation={evaluation}')

    def snapshot(self, *, error=None):
        pending = [i for i in METHODS if i not in self.completed]
        value = dict(schema=SCHEMA, status=self.status, updated_at_utc=_now(), pid=os.getpid(),
            run_dir=str(self.run_dir), output_dir=str(self.output_dir), index_path=str(self.index_path),
            main_run_status=self.main_status, completed_methods=sorted(self.completed),
            pending_methods=pending, elapsed_s=time.monotonic()-self.started,
            normal_inference_changed=False, original_set_metrics_changed=False, error=error)
        _write(self.output_dir/'watch_status.json', value)
        _write(self.output_dir/'watch_receipt.json', dict(value, identity=self.identity,
            methods={str(i):self.completed[i]['evaluation'] for i in sorted(self.completed)}))
        methods = {str(i):dict(status='not_applicable', reason='FFT outputs do not have ROI-aligned rows; original set evaluation is retained')
                   for i in NOT_APPLICABLE}
        methods.update({str(i):dict(status='completed', summary=self.completed[i]['summary'],
                                   evaluation=self.completed[i]['evaluation']) if i in self.completed
                        else dict(status='waiting_for_committed_DEV_predictions', summary=None)
                        for i in METHODS})
        _write(self.output_dir/'fixed_metrics_summary.json', dict(value, methods=methods,
            metric='fixed ROI-to-GT training correspondence, per-member 3D errors',
            note='Offline annotation agreement is not evidence of perfect normal cross-frame identity association'))
        return value

    def run(self, *, once=False, poll_seconds=30., sleeper=time.sleep):
        _require(isinstance(poll_seconds, (int, float)) and not isinstance(poll_seconds, bool)
                 and math.isfinite(poll_seconds) and poll_seconds > 0, 'poll_seconds must be positive and finite')
        with (self.output_dir/'.watch.lock').open('a+') as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('Another fixed-label watcher owns this supplement directory') from error
            identity_path = self.output_dir/'watch_identity.json'
            if identity_path.exists():
                _require(_read(identity_path) == self.identity, 'Watcher resume identity differs; use a new output directory')
            else:
                _require(not any(self.output_dir.glob('g*/receipt.json')), 'Existing fixed results have no trusted watcher identity')
                _write(identity_path, self.identity)
            self.log(f'BEGIN pid={os.getpid()} methods={METHODS} once={once}')
            try:
                self._restore()
                while True:
                    self._assert_identity()
                    status_path = self.run_dir/'status.json'
                    status = _read(status_path) if status_path.exists() else {}
                    if status:
                        _require(status.get('identity_sha256') == self.main_identity_sha256,
                                 'Main status belongs to a different execution identity')
                    self.main_status = status.get('status', 'unknown')
                    for method in METHODS:
                        stage = self._stage(method)
                        if method in self.completed:
                            _require(stage is not None and _sha(stage[0]) == self.completed[method]['stage_receipt_sha256'],
                                     f'G{method} committed stage changed after evaluation')
                        elif stage is not None:
                            self.status = f'evaluating_g{method}'; self.snapshot()
                            self._evaluate(method, stage)
                    if self.main_status in ('failed', 'blocked', 'cancelled', 'terminated'):
                        raise RuntimeError(f'Main run status is {self.main_status}; fixed evaluation stops with pending methods '
                                           f'{[i for i in METHODS if i not in self.completed]}; completed results are preserved')
                    if len(self.completed) == len(METHODS):
                        self.status = 'completed'; value = self.snapshot()
                        self.log('ALL FIVE VISUAL FIXED-LABEL EVALUATIONS COMPLETE')
                        return value
                    if self.main_status == 'completed':
                        raise RuntimeError('Main run completed without every expected committed visual DEV inference receipt')
                    self.status = 'waiting'; value = self.snapshot()
                    self.log(f'WAIT main_status={self.main_status} completed={sorted(self.completed)} pending={value["pending_methods"]}')
                    if once:
                        return value
                    sleeper(poll_seconds)
            except BaseException as error:
                self.status = 'failed'
                message = type(error).__name__+': '+str(error)
                self.snapshot(error=message)
                _write(self.output_dir/'watch_failure.json', dict(status='failed', error=message, at_utc=_now(),
                    pending_methods=[i for i in METHODS if i not in self.completed],
                    completed_methods=sorted(self.completed), main_run_changed=False))
                self.log('FAILED '+message)
                raise


def watch(index_path, run_dir, output_dir, *, once=False, poll_seconds=30., evaluator=None, sleeper=time.sleep):
    return FixedEvaluationWatcher(index_path, run_dir, output_dir, evaluator=evaluator).run(
        once=once, poll_seconds=poll_seconds, sleeper=sleeper)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--poll-seconds', type=float, default=30.)
    parser.add_argument('--once', action='store_true', help='Evaluate only already committed predictions, then report waiting/completed')
    args = parser.parse_args(argv)
    try:
        result = watch(args.index, args.run_dir, args.output_dir, once=args.once, poll_seconds=args.poll_seconds)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f'fixed_gt_watch: {type(error).__name__}: {error}', flush=True)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
