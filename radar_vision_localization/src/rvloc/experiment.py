"""Auditable, stage-resumable execution of the seven controlled methods.

Only methods 2, 3 and 7 train neural networks. The other methods use FFT or
reuse those weights, with TRAIN-only temporal calibration. Every real stage
runs in a fresh interpreter; this module never opens sensor or TEST payloads.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import copy
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time

from .config import ExperimentConfig, METHODS, code_root, external_path
from .runtime import file_sha256, source_identity, write_json


def _utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def normalize_devices(devices):
    """Accept CPU or visible CUDA indices, without changing CUDA visibility."""
    values = devices.split(',') if isinstance(devices, str) else list(devices)
    result = []
    for value in values:
        value = str(value).strip()
        if value.isdigit(): value = 'cuda:'+value
        if value != 'cpu' and not re.fullmatch(r'cuda:\d+', value):
            raise ValueError('devices must contain cpu, CUDA indices, or cuda:N')
        if value not in result: result.append(value)
    if not result: raise ValueError('At least one execution device is required')
    return tuple(result)


def stage_plan():
    """Fixed protocol; checkpoints are final epochs, never DEV-selected."""
    stages = [dict(name=f'train_g{i}', kind='train', method=i, split='train') for i in (2, 3)]
    stages += [dict(name=f'train_infer_g{i}', kind='infer', method=i, split='train') for i in (1, 2, 3)]
    stages += [dict(name='calibrate_train', kind='calibrate', split='train'),
               dict(name='train_g7', kind='train', method=7, split='train')]
    stages += [dict(name=f'dev_infer_g{i}', kind='infer', method=i, split='dev') for i in range(1, 8)]
    stages += [dict(name=f'dev_evaluate_g{i}', kind='evaluate', method=i, split='dev') for i in range(1, 8)]
    stages += [dict(name=f'compare_g{a}_g{b}', kind='compare', first=a, second=b, split='dev')
               for a, b in itertools.combinations(range(1, 8), 2)]
    stages += [dict(name='report', kind='report', split='dev')]
    return stages


class _Runner:
    def __init__(self, config_path, run_dir, *, devices, resume, process_runner=None):
        self.config = ExperimentConfig.load(config_path)
        if self.config.manifest is None:
            raise ValueError('A verified explicit data manifest is required before starting the experiment')
        self.run_dir = external_path(run_dir, field='experiment run directory')
        if self.run_dir == self.config.output_root or not self.run_dir.is_relative_to(self.config.output_root):
            raise ValueError('Experiment run directory must be inside configured output_root')
        self.devices = normalize_devices(devices)
        self.device_map = {2:self.devices[0], 3:self.devices[min(1, len(self.devices)-1)], 7:self.devices[0]}
        self.resume = resume
        self.process_runner = process_runner
        self.original_source = source_identity()
        self.identity = dict(config_path=str(self.config.path), config_sha256=self.config.sha256,
                             manifest_path=str(self.config.manifest),
                             manifest_sha256=file_sha256(self.config.manifest),
                             source=self.original_source, devices=list(self.devices),
                             schema='rvloc.seven_method_execution.v1',
                             architecture='single_query_depth_v3',
                             runtime_environment=dict(python_executable=sys.executable,
                                 cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                 omp_num_threads=os.environ.get('OMP_NUM_THREADS', '4'),
                                 mkl_num_threads=os.environ.get('MKL_NUM_THREADS', '4')),
                             protocol_sha256=_digest(stage_plan()))
        self.identity['sha256'] = _digest(self.identity)
        self.raw = copy.deepcopy(self.config.raw)
        self.raw['manifest'] = str(self.config.manifest)
        self.raw['storage'] = dict(self.raw.get('storage', {}),
                                   output_root=str(self.config.output_root),
                                   weights_root=str(self.config.weights_root),
                                   cache_root=str(self.config.cache_root))
        # This command intentionally trains fresh v3 networks; historical weights
        # and temporal calibrations are never silently inherited.
        self.raw['checkpoints'] = dict(rgb=None, single_frame=None, world_fusion=None)
        self.raw['calibration_artifact'] = None
        self.completed = []
        self.receipts = {}
        self.predictions = {}
        self.evaluations = {}
        self.comparisons = {}
        self.current = None
        self.child_pid = None
        self.last_lines = deque(maxlen=12)
        self.started = time.monotonic()
        self.status = 'starting'
        self.log_stream = None
        self.stage_stream = None

    def _emit(self, line):
        message = f'[{_utc()}] [{self.current or "experiment"}] {line.rstrip()}'
        self.last_lines.append(message)
        print(message, flush=True)
        if self.log_stream is not None:
            self.log_stream.write(message+'\n'); self.log_stream.flush()
        if self.stage_stream is not None:
            self.stage_stream.write(message+'\n'); self.stage_stream.flush()

    def _heartbeat(self):
        write_json(self.run_dir/'status.json', dict(schema='rvloc.experiment_status.v1',
            status=self.status, updated_at_utc=_utc(), pid=os.getpid(), child_pid=self.child_pid,
            run_dir=str(self.run_dir), run_log=str(self.run_dir/'run.log'),
            identity_sha256=self.identity['sha256'], current_stage=self.current,
            completed_stages=list(self.completed), completed_count=len(self.completed),
            total_stages=len(stage_plan()), elapsed_s=time.monotonic()-self.started,
            device_map=self.device_map, recent_log=list(self.last_lines),
            checkpoint_rule='fixed final TRAIN epoch; no DEV checkpoint selection'))

    def _assert_identity(self):
        if file_sha256(self.config.path) != self.identity['config_sha256']:
            raise ValueError('Configuration changed during this run; start a new run')
        if file_sha256(self.config.manifest) != self.identity['manifest_sha256']:
            raise ValueError('Manifest changed during this run; start a new run')
        if source_identity() != self.original_source:
            raise ValueError('Source identity changed during this run; start a new run')

    def _spawn(self, command, *, env, emit, heartbeat):
        """Drain stdout in a reader thread so silent kernels still get heartbeat."""
        child = subprocess.Popen(command, cwd=str(code_root()), env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1, start_new_session=True)
        self.child_pid = child.pid
        heartbeat()
        messages = queue.Queue()

        def read():
            try:
                for line in child.stdout: messages.put(line)
            finally:
                child.stdout.close(); messages.put(None)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()
        last_beat = time.monotonic()
        try:
            while True:
                try: line = messages.get(timeout=1.)
                except queue.Empty: line = ''
                if line is None: break
                if line: emit(line)
                if time.monotonic()-last_beat >= 5:
                    heartbeat(); last_beat = time.monotonic()
            reader.join()
            return child.wait()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
            raise
        finally:
            self.child_pid = None
            heartbeat()

    def _device(self, stage):
        method = stage.get('method')
        model_method = {5:2, 6:3}.get(method, method)
        return self.device_map.get(model_method, 'cpu')

    def _prepare(self, stage, attempt):
        raw = copy.deepcopy(self.raw)
        raw.setdefault('training', {})['device'] = self._device(stage)
        cfg = write_json(attempt/'config.json', raw)
        command = [sys.executable, '-u', '-m', 'rvloc']
        kind = stage['kind']
        out = attempt/'results'
        outputs = []
        if kind == 'train':
            command += ['train', '--config', str(cfg), '--method', str(stage['method']), '--output-dir', str(out)]
            outputs = [out/'receipt.json', out/'history.json']
        elif kind == 'infer':
            command += ['infer', '--config', str(cfg), '--method', str(stage['method']),
                        '--split', stage['split'], '--device', self._device(stage), '--output-dir', str(out)]
            outputs = [out/'predictions.json', out/'reproduction_receipt.json']
        elif kind == 'calibrate':
            command += ['calibrate', '--config', str(cfg), '--predictions']
            command += [self.predictions[('train', i)] for i in (1, 2, 3)]
            command += ['--output', str(attempt/'calibration.json')]
            outputs = [attempt/'calibration.json']
        elif kind == 'evaluate':
            command += ['evaluate', '--config', str(cfg), '--predictions', self.predictions[('dev', stage['method'])],
                        '--output-dir', str(out)]
            outputs = [out/name for name in ('evaluation.json', 'localization_metrics.json',
                       'visual_prior_metrics.json', 'episode_level_results.json', 'failure_cases.json')]
        elif kind == 'compare':
            settings = raw.get('evaluation', {})
            command += ['compare', '--first', self.evaluations[stage['first']], '--second', self.evaluations[stage['second']],
                        '--output', str(attempt/'paired.json'), '--draws', str(settings.get('bootstrap_draws', 2000)),
                        '--seed', str(settings.get('bootstrap_seed', 20261126))]
            outputs = [attempt/'paired.json']
        elif kind == 'report':
            command = []
            outputs = [attempt/'experiment_summary.json', attempt/'SEVEN_METHODS_TRAINING_REPORT_CN.md']
        else: raise ValueError('Unknown experiment stage: '+kind)
        return command, outputs

    def _collect(self, stage, receipt):
        artifacts = receipt['artifacts']
        kind = stage['kind']
        if kind == 'train':
            path = next(Path(p) for p in artifacts if p.endswith('/receipt.json'))
            record = json.loads(path.read_text())
            if record.get('status') != 'trained' or record.get('method_id') != stage['method']:
                raise ValueError('Training receipt does not certify the requested method')
            self.raw['checkpoints'][METHODS[stage['method']]['model']] = record['checkpoint']
        elif kind == 'infer':
            path = next(p for p in artifacts if p.endswith('/predictions.json'))
            self.predictions[(stage['split'], stage['method'])] = path
        elif kind == 'calibrate':
            self.raw['calibration_artifact'] = next(p for p in artifacts if p.endswith('/calibration.json'))
        elif kind == 'evaluate':
            self.evaluations[stage['method']] = next(p for p in artifacts if p.endswith('/evaluation.json'))
        elif kind == 'compare':
            self.comparisons[f"{stage['first']}_{stage['second']}"] = next(p for p in artifacts if p.endswith('/paired.json'))
        elif kind == 'report':
            for path in artifacts:
                if path.endswith('.md'): self.report_path = path
                if path.endswith('experiment_summary.json'): self.summary_path = path

    def _dependencies(self):
        return {name:receipt['artifacts'] for name, receipt in self.receipts.items()}

    def _run_stage(self, stage):
        self.current = stage['name']
        self._assert_identity()
        input_identity = _digest(dict(stage=stage, identity=self.identity['sha256'],
                                      resolved_config=self.raw, dependencies=self._dependencies()))
        receipt_path = self.run_dir/'stage_receipts'/(stage['name']+'.json')
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('input_sha256') != input_identity:
                raise ValueError('Completed stage input identity mismatch: '+stage['name'])
            if receipt.get('status') != 'completed':
                raise ValueError('Invalid completed stage receipt: '+stage['name'])
            for path, digest in receipt['artifacts'].items():
                if not Path(path).is_file() or file_sha256(path) != digest:
                    raise ValueError('Completed artifact missing or changed: '+path)
            self._collect(stage, receipt)
            self.receipts[stage['name']] = receipt
            self.completed.append(stage['name'])
            self._emit('RESUME verified completed artifacts; skipping computation')
            self._heartbeat()
            return
        stage_root = self.run_dir/'stages'/stage['name']
        stage_root.mkdir(parents=True, exist_ok=True)
        attempt_number = 1
        while (stage_root/f'attempt_{attempt_number:04d}').exists(): attempt_number += 1
        attempt = stage_root/f'attempt_{attempt_number:04d}'
        attempt.mkdir()
        self.stage_stream = (attempt/'stage.log').open('a', buffering=1)
        command, outputs = self._prepare(stage, attempt)
        record = dict(schema='rvloc.stage_attempt.v1', stage=stage, input_sha256=input_identity,
                      identity_sha256=self.identity['sha256'], attempt=attempt_number,
                      command=command, cwd=str(code_root()), device=self._device(stage),
                      started_at_utc=_utc(), status='running',
                      resumption='stage-level only; failed training restarts and does not resume optimizer')
        write_json(attempt/'attempt_receipt.json', record)
        began = time.monotonic()
        self._emit(f'START kind={stage["kind"]} device={self._device(stage)} attempt={attempt_number}')
        self._heartbeat()
        try:
            if stage['kind'] == 'report':
                self._report(attempt)
            else:
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
                env.setdefault('OMP_NUM_THREADS', '4')
                env.setdefault('MKL_NUM_THREADS', '4')
                env['PYTHONPATH'] = str(code_root()/'src')+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
                execute = self.process_runner or self._spawn
                code = execute(command, env=env, emit=self._emit, heartbeat=self._heartbeat)
                if code != 0: raise RuntimeError(f'Stage {stage["name"]} exited with status {code}; see {attempt/"stage.log"}')
            if stage['kind'] == 'train':
                record_path = next(p for p in outputs if p.name == 'receipt.json')
                training = json.loads(record_path.read_text())
                checkpoint = Path(training['checkpoint']).resolve()
                expected_weights = self.config.weights_root/attempt.relative_to(self.config.output_root)/'results'
                if checkpoint.parent != expected_weights or not checkpoint.is_file():
                    raise ValueError('Training checkpoint is outside the dedicated current-attempt weight directory')
                outputs.append(checkpoint)
            if not all(path.is_file() for path in outputs):
                raise ValueError('Stage returned without its required durable outputs')
            self._assert_identity()
            record.update(status='completed', finished_at_utc=_utc(), elapsed_s=time.monotonic()-began,
                          artifacts={str(path):file_sha256(path) for path in outputs})
            self._collect(stage, record)
            write_json(attempt/'attempt_receipt.json', record)
            write_json(receipt_path, record)
            self.receipts[stage['name']] = record
            self.completed.append(stage['name'])
            self._emit(f'COMPLETE elapsed_s={record["elapsed_s"]:.3f} stages={len(self.completed)}/{len(stage_plan())}')
        except BaseException as error:
            record.update(status='failed', finished_at_utc=_utc(), elapsed_s=time.monotonic()-began,
                          error_type=type(error).__name__, error=str(error))
            write_json(attempt/'attempt_receipt.json', record)
            self._emit('FAILED '+type(error).__name__+': '+str(error))
            raise
        finally:
            self.stage_stream.close(); self.stage_stream = None
            self._heartbeat()

    def _report(self, attempt):
        evaluations = {str(i):json.loads(Path(path).read_text()) for i, path in self.evaluations.items()}
        summary = dict(schema='rvloc.seven_method_summary.v1', status='completed',
            architecture='single_query_depth_v3', identity=self.identity,
            methods={str(i):dict(METHODS[i], method_id=i,
                independently_trained=i in (2, 3, 7),
                evaluation_path=self.evaluations[i],
                metrics=evaluations[str(i)]['summary'], cost=evaluations[str(i)].get('cost')) for i in range(1, 8)},
            checkpoints=self.raw['checkpoints'], calibration_artifact=self.raw['calibration_artifact'],
            pairwise_comparisons=self.comparisons,
            protocol=dict(historical_test_opened=False, dev_used_for_checkpoint_selection=False,
                neural_checkpoint_rule='last predeclared TRAIN epoch',
                independent_generalization_claim=False, all_methods_same_manifest=True,
                timing='serial stage execution; reported inference costs are measured by each method',
                no_iq_correction=True, future_navigation=False),
            execution_elapsed_s=time.monotonic()-self.started)
        write_json(attempt/'experiment_summary.json', summary)
        lines = ['# 七组 32 MHz 共点定位实验训练与评价报告', '',
                 '三组神经网络（G2、G3、G7）完成 TRAIN 训练；G1 使用 FFT，G4–G6 复用基础定位器并使用 TRAIN 冻结的关联与 Kalman 参数。', '',
                 '所有神经网络使用预先声明的最后一个训练 epoch；历史 DEV 只用于冻结协议后的离线评价，不能作为独立泛化认证。未读取历史 TEST。', '',
                 '| 组 | 方法 | RMSE (m) | MAE (m) | P95 (m) | P99 (m) |',
                 '|---|---|---:|---:|---:|---:|']
        for i in range(1, 8):
            metrics = evaluations[str(i)]['summary']
            def value(*keys):
                for key in keys:
                    if key in metrics:
                        item = metrics[key]
                        return f'{item:.6f}' if isinstance(item, (int, float)) else str(item)
                return '详见 JSON'
            lines.append(f'| {i} | {METHODS[i]["name"]} | '+ ' | '.join((
                value('position_rmse_3d_m', 'rmse_m'), value('position_mae_3d_m', 'mae_m'),
                value('p95_m', 'position_p95_3d_m'), value('p99_m', 'position_p99_3d_m'))) +' |')
        lines += ['', '完整覆盖率、全 GT 成功率、严重错误、逐成员结果与 episode bootstrap 区间见各 evaluation.json。', '',
                  '21 组配对比较以共同有效定位成员为条件，保留覆盖率差异；95% 与 97.5% 区间及 P99 不确定性见各 paired.json。', '',
                  '模型、数据、源码身份、固定协议与实际耗时保存在 experiment_summary.json；各阶段配置、命令、stdout 和失败记录完整保留。', '',
                  '## 结果文件', '']
        lines += [f'- G{i}: `{path}`' for i, path in sorted(self.evaluations.items())]
        lines += ['', '## 配对比较', '']
        lines += [f'- G{pair.replace("_", " 对 G")}: `{path}`' for pair, path in self.comparisons.items()]
        (attempt/'SEVEN_METHODS_TRAINING_REPORT_CN.md').write_text('\n'.join(lines)+'\n')

    def run(self):
        if self.resume:
            if not self.run_dir.is_dir(): raise ValueError('Cannot resume a nonexistent experiment directory')
        else:
            if self.run_dir.exists():
                allowed = {'bootstrap.log', 'preparation.json', 'run.log'}
                if not self.run_dir.is_dir() or any(p.name not in allowed or not p.is_file()
                                                    for p in self.run_dir.iterdir()):
                    raise FileExistsError('Experiment directory already contains run artifacts; use --resume or a new directory')
            else:
                self.run_dir.mkdir(parents=True, exist_ok=False)
        with (self.run_dir/'.run.lock').open('a+') as lock:
            try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('Another process owns this experiment directory') from error
            identity_path = self.run_dir/'execution_identity.json'
            if self.resume:
                if not identity_path.is_file() or json.loads(identity_path.read_text()) != self.identity:
                    raise ValueError('Resume requires identical config, manifest, source, devices and protocol')
            else:
                write_json(identity_path, self.identity)
                write_json(self.run_dir/'base_config.json', self.raw)
                write_json(self.run_dir/'protocol.json', dict(stages=stage_plan(), device_map=self.device_map,
                    neural_training_methods=[2, 3, 7], train_calibrated_reuse_methods=[4, 5, 6],
                    historical_test_opened=False, checkpoint_rule='final epoch only',
                    stage_resumption=True, optimizer_resumption=False, parallel_training=False))
            self.log_stream = (self.run_dir/'run.log').open('a', buffering=1)
            self.status = 'running'
            self._emit(f'{"RESUME" if self.resume else "BEGIN"} pid={os.getpid()} identity={self.identity["sha256"]}')
            self._heartbeat()
            try:
                for stage in stage_plan(): self._run_stage(stage)
                self.status = 'completed'; self.current = None
                result = dict(status='completed', run_dir=str(self.run_dir),
                    run_log=str(self.run_dir/'run.log'), report=self.report_path,
                    summary=self.summary_path, completed_stages=len(self.completed))
                write_json(self.run_dir/'completion.json', result)
                self._emit('ALL SEVEN METHODS COMPLETE')
                return result
            except BaseException as error:
                self.status = 'failed'
                write_json(self.run_dir/'failure.json', dict(status='failed', stage=self.current,
                    error_type=type(error).__name__, error=str(error), updated_at_utc=_utc(),
                    completed_stages=list(self.completed),
                    resume='Use the same config, source, devices and run directory with --resume. Failed stage outputs remain intact.'))
                raise
            finally:
                self._heartbeat()
                self.log_stream.close(); self.log_stream = None


def run_experiment(config_path, run_dir, *, devices=('cpu',), resume=False, process_runner=None):
    """Run/resume the frozen seven-method protocol with realtime ``run.log``.

    ``process_runner`` is an injectable subprocess boundary for synthetic tests;
    real commands always use a fresh unbuffered Python interpreter. Resume is
    stage-level: an interrupted neural stage restarts without reusing optimizer
    state, in a new attempt directory, preserving all previous attempts.
    """
    return _Runner(config_path, run_dir, devices=devices, resume=resume,
                   process_runner=process_runner).run()


__all__ = ['run_experiment', 'stage_plan', 'normalize_devices']
