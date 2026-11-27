"""Single-line progress for a frozen training run, without modifying its code.

RMSE is measured by a separate full-TRAIN checkpoint evaluator. It is never
inferred from loss and its checkpoint epoch is always printed explicitly.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.writing-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def hms(seconds):
    if seconds is None or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
        return '待估计'
    value = max(0, round(seconds))
    return f'{value//3600:02d}:{value//60%60:02d}:{value%60:02d}'


def stage_eta(progress, frames_per_epoch):
    if progress.get('status') == 'completed': return 0.
    eta = progress.get('eta_s')
    if isinstance(eta, (int, float)) and math.isfinite(eta) and eta >= 0: return float(eta)
    epochs, epoch, frames = (progress.get(k) for k in ('epochs', 'epoch', 'frames'))
    elapsed = progress.get('elapsed_s')
    if not all(isinstance(x, (int, float)) for x in (epochs, epoch, frames, elapsed, frames_per_epoch)):
        return None
    done = max(0, epoch-1)*frames_per_epoch+frames
    return max(0., epochs*frames_per_epoch-done)*elapsed/done if done > 0 else None


def format_line(main, progress, method, metric=None, *, eta=None, worker_status='pending'):
    now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    stage = main.get('current_stage') or main.get('status', 'unknown')
    label = f'G{method}' if method is not None else stage
    epoch = f"{progress.get('epoch','-')}/{progress.get('epochs','-')}"
    loss = progress.get('last_loss', progress.get('mean_loss'))
    loss_text = f'{loss:.6f}' if isinstance(loss, (int, float)) and math.isfinite(loss) else 'pending'
    if metric is not None:
        value = metric.get('position_rmse_3d_m')
        number = f'{value:.4f}m' if isinstance(value, (int, float)) and math.isfinite(value) else 'unavailable'
        rmse = f"TRAIN_RMSE@E{metric['completed_epochs']}={number}"
    else:
        rmse = f'TRAIN_RMSE={"failed" if worker_status == "failed" else "pending"}'
    return (f'[{now}] [PROGRESS] {label} epoch={epoch} update={progress.get("optimizer_updates","-")} '
            f'loss={loss_text} {rmse} 本模型已用={hms(progress.get("elapsed_s"))} '
            f'本模型预计剩余={hms(eta)} 总已用={hms(main.get("elapsed_s"))} RMSE评估={worker_status}')


class ProgressMonitor:
    def __init__(self, run_dir, index_path, output_dir, *, device='cpu', append_main_log=True):
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.index_path = Path(index_path).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        project = Path(__file__).resolve().parents[1]
        if self.output_dir.is_relative_to(project) or self.output_dir.is_relative_to(self.run_dir):
            raise ValueError('Progress artifacts must be outside code and separate from the main run')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.main_identity = read_json(self.run_dir/'execution_identity.json')
        index = read_json(self.index_path)
        if (index.get('schema') != 'rvloc.fixed_roi_gt_index.v1' or index.get('supervision_only') is not True
                or index.get('manifest_sha256') != self.main_identity.get('manifest_sha256')):
            raise ValueError('Frozen labels must match the current run manifest')
        self.frames_per_epoch = index['summaries']['train']['frames']
        self.device, self.append_main_log = device, append_main_log
        self.worker_script = Path(__file__).with_name('checkpoint_train_rmse.py')
        self.identity = dict(schema='rvloc.live_progress_identity.v1', run_dir=str(self.run_dir),
            main_identity_sha256=sha(self.run_dir/'execution_identity.json'),
            frozen_index_sha256=sha(self.index_path), manifest_sha256=index['manifest_sha256'],
            monitor_sha256=sha(__file__), device=device, append_main_log=append_main_log,
            rmse_scope='fixed_checkpoint_train_full', training_changed=False,
            test_or_dev_used_for_train_progress=False)
        self.metrics = {}
        self.attempted = set()
        self.worker = None
        self.worker_stream = None
        self.worker_request = None
        self.worker_status = 'pending'
        self.worker_error = None
        identity_path = self.output_dir/'identity.json'
        if identity_path.exists() and read_json(identity_path) != self.identity:
            raise ValueError('Progress monitor identity changed; use a new output directory')
        write_json(identity_path, self.identity)
        status_path = self.output_dir/'status.json'
        if status_path.exists():
            previous = read_json(status_path)
            for method, record in previous.get('metrics', {}).items():
                path = Path(record['path'])
                if sha(path) != record['sha256']: raise ValueError('Completed RMSE artifact changed')
                self.metrics[int(method)] = record
            self.attempted = set(previous.get('attempted', []))
        self.started = time.monotonic()

    def _assert_identity(self):
        if (sha(self.run_dir/'execution_identity.json') != self.identity['main_identity_sha256']
                or sha(self.index_path) != self.identity['frozen_index_sha256']
                or sha(__file__) != self.identity['monitor_sha256']):
            raise ValueError('Main run, frozen labels, or monitor source changed')

    def context(self, main):
        name = main.get('current_stage')
        if name not in ('train_g2', 'train_g3', 'train_g7'): return None
        method = int(name[-1])
        folders = sorted((self.run_dir/'stages'/name).glob('attempt_*'), reverse=True)
        for attempt in folders:
            progress_path = attempt/'results'/'progress.json'
            config_path = attempt/'config.json'
            if not progress_path.is_file() or not config_path.is_file(): continue
            config = read_json(config_path)
            output_root = Path(config['storage']['output_root']).resolve()
            relative = (attempt/'results').resolve().relative_to(output_root)
            weights = Path(config['storage']['weights_root']).resolve()/relative
            checkpoints = sorted(weights.glob('epoch_[0-9][0-9][0-9][0-9].pt'))
            return dict(method=method, progress=read_json(progress_path), config=config_path,
                        checkpoint=checkpoints[-1] if checkpoints else None)
        return dict(method=method, progress={}, config=None, checkpoint=None)

    def _poll_worker(self):
        if self.worker is None or self.worker.poll() is None: return
        code = self.worker.returncode
        request = self.worker_request
        self.worker_stream.close(); self.worker_stream = None
        self.worker = None
        if code:
            self.worker_status = 'failed'
            self.worker_error = f'Checkpoint evaluator exited {code}; see {request["log"]}'
            write_json(request['output']/'failure.json', dict(error=self.worker_error, request={k:str(v) for k,v in request.items()}))
            return
        path = request['output']/'metrics.json'
        value = read_json(path)
        if (value.get('scope') != 'fixed_checkpoint_train_full' or value.get('method_id') != request['method']
                or value.get('completed_epochs') != request['epoch']
                or value.get('checkpoint_sha256') != request['checkpoint_sha256']
                or value.get('identity', {}).get('config_sha256') != request['config_sha256']):
            raise ValueError('Checkpoint RMSE result does not certify its requested inputs')
        metric = value.get('position_rmse_3d_m')
        if metric is not None and (not isinstance(metric, (int, float)) or not math.isfinite(metric) or metric < 0):
            raise ValueError('Invalid measured RMSE')
        self.metrics[request['method']] = dict(path=str(path), sha256=sha(path), metric=value,
            training_config_path=str(request['config']), training_config_sha256=request['config_sha256'])
        self.worker_status, self.worker_error = 'completed', None

    def _start_worker(self, context):
        if self.worker is not None or context is None or context['checkpoint'] is None: return
        checkpoint = context['checkpoint']
        key = str(checkpoint)
        if key in self.attempted: return
        if not self.worker_script.is_file():
            self.worker_status, self.worker_error = 'pending', 'Checkpoint evaluator is not installed yet'
            return
        epoch = int(checkpoint.stem.split('_')[-1])
        base = self.output_dir/'rmse'/f'g{context["method"]}'/f'epoch_{epoch:04d}'
        number = 1
        while (base/f'attempt_{number:04d}').exists(): number += 1
        output = base/f'attempt_{number:04d}'
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = base/f'attempt_{number:04d}.log'
        self.worker_stream = log_path.open('a', buffering=1)
        command = [sys.executable, '-u', str(self.worker_script), '--config', str(context['config']),
                   '--checkpoint', str(checkpoint), '--index', str(self.index_path),
                   '--device', self.device, '--output-dir', str(output)]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
        self.worker_request = dict(method=context['method'], epoch=epoch, checkpoint=checkpoint,
            checkpoint_sha256=sha(checkpoint), config=context['config'], config_sha256=sha(context['config']),
            output=output, log=log_path, command=command)
        self.worker = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=self.worker_stream,
            stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.attempted.add(key)
        self.worker_status, self.worker_error = 'running', None
        write_json(self.output_dir/'worker_request.json', dict(pid=self.worker.pid,
            **{k:str(v) if isinstance(v, Path) else v for k,v in self.worker_request.items()}))

    def tick(self, *, start_worker=True):
        self._assert_identity()
        self._poll_worker()
        main = read_json(self.run_dir/'status.json')
        if main.get('identity_sha256') != self.main_identity['sha256']:
            raise ValueError('Main status belongs to another run')
        context = self.context(main)
        if start_worker and main.get('status') == 'running': self._start_worker(context)
        method = context['method'] if context else None
        progress = context['progress'] if context else {}
        record = self.metrics.get(method)
        if record is not None and (context is None or context['config'] is None
                or record.get('training_config_path') != str(context['config'])
                or record.get('training_config_sha256') != sha(context['config'])):
            record = None  # Another attempt/model must never borrow an old metric.
        line = format_line(main, progress, method, record['metric'] if record else None,
                           eta=stage_eta(progress, self.frames_per_epoch), worker_status=self.worker_status)
        print(line, flush=True)
        destinations = [self.output_dir/'progress.log']
        if self.append_main_log: destinations.append(self.run_dir/'run.log')
        for path in destinations:
            with path.open('a') as stream: stream.write(line+'\n')
        state = dict(schema='rvloc.live_progress_status.v1', status=main.get('status'),
            updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'), pid=os.getpid(),
            current_stage=main.get('current_stage'), worker_pid=self.worker.pid if self.worker else None,
            worker_status=self.worker_status, worker_error=self.worker_error,
            metrics={str(k):v for k,v in self.metrics.items()}, attempted=sorted(self.attempted), line=line,
            elapsed_s=time.monotonic()-self.started, identity=self.identity)
        write_json(self.output_dir/'status.json', state)
        return state

    def run(self, *, once=False, interval_seconds=10.):
        if not isinstance(interval_seconds, (int, float)) or interval_seconds <= 0 or not math.isfinite(interval_seconds):
            raise ValueError('interval_seconds must be positive finite seconds')
        with (self.output_dir/'.monitor.lock').open('a+') as lock:
            try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error: raise ValueError('Another progress monitor owns this output directory') from error
            try:
                while True:
                    state = self.tick(start_worker=not once)
                    if once or state['status'] in ('completed','failed','cancelled','terminated','blocked'):
                        return state
                    time.sleep(interval_seconds)
            finally:
                if self.worker is not None and self.worker.poll() is None:
                    self.worker.terminate()
                    try: self.worker.wait(timeout=5)
                    except subprocess.TimeoutExpired: self.worker.kill(); self.worker.wait()
                if self.worker_stream is not None: self.worker_stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--interval-seconds', type=float, default=10.)
    parser.add_argument('--no-append-main-log', action='store_true')
    parser.add_argument('--once', action='store_true', help='Print once without starting a checkpoint evaluator')
    args = parser.parse_args(argv)
    state = ProgressMonitor(args.run_dir, args.index, args.output_dir, device=args.device,
        append_main_log=not args.no_append_main_log).run(once=args.once, interval_seconds=args.interval_seconds)
    print(json.dumps({k:state[k] for k in ('status','current_stage','worker_status','worker_error')}, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__': raise SystemExit(main())
