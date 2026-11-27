"""Independent receipt-checked queue for three single-frame D evidence ablations."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .monitor import DEFAULT_OUTPUT, ARMS, snapshot, format_progress

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('wait_predecessors', 'prepare_sequence', 'prepare_aligned', 'roi_export', 'prepare_roi',
          'train_sequence', 'train_aligned', 'train_roi', 'evaluation')
DEPENDENCY_IDENTITIES = (
    dict(name='bandwidth', package='bandwidth_study', output=str(ROOT / 'results/bandwidth_double_20261124'),
         pid=1033366, process_start='700282312', plan_sha256='e1eb71d603f126355527677825607dfae125730a18b38743a42290c1dbbfcee8'),
    dict(name='temporal', package='temporal_depth_study', output=str(ROOT / 'results/temporal_depth_20261124'),
         pid=1095794, process_start='700598722', plan_sha256='0ff8dc9e7ffe4f04bc08fed0424bf6ccce585746c4ae945e4a79ccc58bec7a85'))


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def read_json(path): return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_start(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] in ('Z', 'X') else fields[19]
    except (OSError, ValueError, IndexError): return None


def alive(record):
    return bool(record.get('pid') and record.get('process_start') and process_start(record['pid']) == record['process_start'])


def registered_dependencies():
    result = []
    for original in DEPENDENCY_IDENTITIES:
        value = dict(original); path = Path(value['output']) / 'run_plan.json'
        if sha256(path) != value['plan_sha256']: raise ValueError('Registered predecessor plan changed: ' + value['name'])
        value['source_sha256'] = read_json(path)['source_sha256']; result.append(value)
    return result


def validate_dependency(value):
    root = Path(value['output']).resolve(); state = read_json(root / 'run_state.json')
    if sha256(root / 'run_plan.json') != value['plan_sha256'] or state.get('plan_sha256') != value['plan_sha256']:
        raise ValueError('Predecessor plan identity changed: ' + value['name'])
    package = (ROOT / value['package']).resolve()
    for name, expected in value['source_sha256'].items():
        path = (package / name).resolve()
        if not path.is_relative_to(package) or sha256(path) != expected:
            raise ValueError('Locked predecessor source changed: ' + value['name'] + '/' + name)
    if state.get('controller_pid') != value['pid'] or state.get('controller_process_start') != value['process_start']:
        raise ValueError('Predecessor PID/start-time identity changed: ' + value['name'])
    if state.get('status') in ('failed', 'interrupted'):
        raise RuntimeError('Predecessor stopped; no GPU allocation: ' + value['name'] + ': ' + str(state.get('error')))
    if state.get('status') == 'complete': return True
    if state.get('status') != 'running' or process_start(value['pid']) != value['process_start']:
        raise RuntimeError('Predecessor vanished without complete state: ' + value['name'])
    command = Path(f'/proc/{value["pid"]}/cmdline').read_bytes().decode().split('\0')
    if value['package'] + '.run' not in command or '--output' not in command:
        raise ValueError('Registered predecessor PID belongs to another command')
    given = Path(command[command.index('--output') + 1]).expanduser()
    if not given.is_absolute(): given = Path(f'/proc/{value["pid"]}/cwd').resolve() / given
    if given.resolve() != root: raise ValueError('Predecessor PID owns a different output')
    return False


def dependencies_ready(values): return all([validate_dependency(value) for value in values])


def source_identity():
    names = ('model.py', 'checkpoint.py', 'data.py', 'roi_export.py', 'train.py', 'run.py', 'monitor.py', 'evaluate.py')
    return {name: sha256(Path(__file__).parent / name) for name in names}


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    result.add_argument('--gpus', default='1,4,5'); result.add_argument('--steps', type=int, default=20000)
    result.add_argument('--batch-size', type=int, default=16); result.add_argument('--seed', type=int, default=20261118)
    result.add_argument('--learning-rate', type=float, default=3e-4); result.add_argument('--detection-dropout', type=float, default=.2)
    result.add_argument('--validate-every', type=int, default=1000); result.add_argument('--threads', type=int, default=1)
    result.add_argument('--poll-seconds', type=float, default=15.)
    result.add_argument('--plan-only', action='store_true'); result.add_argument('--detach', action='store_true')
    result.add_argument('--_stage-worker', choices=STAGES, help=argparse.SUPPRESS)
    result.add_argument('--_wait', action='store_true', help=argparse.SUPPRESS)
    result.add_argument('--_plan', type=Path, help=argparse.SUPPRESS)
    return result


def make_plan(args):
    expected = dict(steps=20000, batch_size=16, seed=20261118, learning_rate=3e-4,
                    detection_dropout=.2, validate_every=1000, threads=1)
    if any(getattr(args, name) != value for name, value in expected.items()) or args.gpus != '1,4,5':
        raise ValueError('Arguments differ from the registered three single-intervention budgets')
    if args.poll_seconds < .1: raise ValueError('Polling interval must be >=0.1 seconds')
    output = args.output.expanduser().resolve(); configs = {arm: output / 'configs' / (arm + '.json') for arm in ARMS}
    env = dict(PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_THREADING_LAYER='GNU', PYTHONUNBUFFERED='1')
    stages = [dict(name='wait_predecessors', command=[sys.executable, '-m', 'd_evidence_study.run', '--_wait',
        '--_plan', str(output / 'run_plan.json')], environment=env)]
    for arm in ('sequence', 'aligned'):
        prepared = Path(read_json(configs[arm])['d_evidence_preparation_root']).expanduser().resolve()
        stages.append(dict(name='prepare_' + arm, command=[sys.executable, '-m', 'd_evidence_study.data',
            '--config', str(configs[arm]), '--output', str(prepared), '--splits', 'train', 'val'], environment=env))
    native = ROOT.parent / 'uav_detection/.venv_deimv2/bin/python'
    stages.append(dict(name='roi_export', command=[str(native), '-m', 'd_evidence_study.roi_export',
        '--config', str(configs['roi']), '--output', str(output / 'roi'), '--splits', 'train', 'val', '--device', 'cuda:0'],
        environment=dict(env, CUDA_VISIBLE_DEVICES='1'), gpu_ids=['1']))
    stages.append(dict(name='prepare_roi', command=[sys.executable, '-m', 'd_evidence_study.data',
        '--config', str(configs['roi']), '--output', str(Path(read_json(configs['roi'])['d_evidence_preparation_root']).expanduser().resolve()),
        '--splits', 'train', 'val'], environment=env))
    for arm in ARMS:
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1', '--nproc_per_node=3',
            '-m', 'd_evidence_study.train', '--arm', arm, '--config', str(configs[arm]),
            '--output', str(output / 'training' / arm), '--device', 'cuda']
        for name in expected: command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
        stages.append(dict(name='train_' + arm, command=command,
            environment=dict(env, CUDA_VISIBLE_DEVICES='1,4,5'), gpu_ids=['1', '4', '5']))
    stages.append(dict(name='evaluation', command=[sys.executable, '-m', 'd_evidence_study.evaluate',
        '--experiment', str(output), '--phase', 'finalize', '--device', 'cuda:0'],
        environment=dict(env, CUDA_VISIBLE_DEVICES='1'), gpu_ids=['1']))
    return dict(schema='d_evidence_queued_plan_v1', output=str(output), cwd=str(ROOT),
        settings=dict(expected, world_size=3, global_batch_size=48, gpus=['1', '4', '5'], poll_seconds=args.poll_seconds),
        configurations={arm: str(path) for arm, path in configs.items()},
        configuration_sha256={arm: sha256(path) for arm, path in configs.items()},
        source_sha256=source_identity(), protocol_sha256=sha256(output / 'PROTOCOL.json'),
        split_roster_sha256=sha256(output / 'split_roster.json'), dependencies=registered_dependencies(),
        stages=stages, formal_training_arms=list(ARMS), input_frames=1, initial_fusion_checkpoint=None,
        production_default_changed=False, final_test_inputs_permitted=False,
        evaluation_scope='complete development validation only; no new final test')


def check_plan(plan):
    if source_identity() != plan['source_sha256']: raise ValueError('Locked evidence source changed')
    for arm, path in plan['configurations'].items():
        if sha256(path) != plan['configuration_sha256'][arm]: raise ValueError('Registered evidence configuration changed: ' + arm)
    for name, key in (('PROTOCOL.json', 'protocol_sha256'), ('split_roster.json', 'split_roster_sha256')):
        if sha256(Path(plan['output']) / name) != plan[key]: raise ValueError('Registered evidence protocol or split changed')


def wait_dependencies(plan):
    check_plan(plan); output = Path(plan['output']); started = time.time()
    while True:
        try:
            ready = dependencies_ready(plan['dependencies']); rows = []
            for value in plan['dependencies']:
                state = read_json(Path(value['output']) / 'run_state.json')
                rows.append(dict(name=value['name'], pid=value['pid'], status=state['status'], stage=state.get('current_stage')))
            write_json(output / 'queue_status.json', dict(status='ready' if ready else 'waiting', dependencies=rows,
                elapsed_s=time.time() - started, updated_unix=time.time(), predecessor_test_metrics_read=False, gpu_allocated=False))
            if ready: return 0
        except BaseException as exc:
            write_json(output / 'queue_status.json', dict(status='failed', error=repr(exc), updated_unix=time.time(), gpu_allocated=False))
            raise
        time.sleep(plan['settings']['poll_seconds'])


def gpu_usage(ids):
    response = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True, timeout=30)
    selected = {}
    for row in csv.reader(response.stdout.splitlines()):
        index, uuid, memory = [item.strip() for item in row]
        if index in ids: selected[uuid] = dict(index=index, memory_mib=float(memory), processes=[])
    if len(selected) != len(ids): raise RuntimeError('Requested GPUs unavailable; no CPU fallback')
    response = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=30)
    for row in csv.reader(response.stdout.splitlines()):
        if len(row) >= 4 and row[0].strip() in selected:
            selected[row[0].strip()]['processes'].append(dict(pid=row[1].strip(), name=row[2].strip(), memory_mib=row[3].strip()))
    return [item for item in selected.values() if item['processes'] or item['memory_mib'] > 256.]


def wait_resources(plan, ids):
    while True:
        if not dependencies_ready(plan['dependencies']): raise RuntimeError('Predecessors are not complete; GPU allocation forbidden')
        busy = gpu_usage(ids)
        write_json(Path(plan['output']) / 'resource_status.json', dict(status='waiting' if busy else 'ready',
            gpu_ids=ids, busy=busy, updated_unix=time.time(), cpu_fallback=False))
        if not busy: return
        time.sleep(plan['settings']['poll_seconds'])


def verify_stage(name, plan):
    check_plan(plan); output = Path(plan['output'])
    if name == 'wait_predecessors':
        if not dependencies_ready(plan['dependencies']): raise ValueError('Predecessor work is incomplete')
        return dict(dependencies=plan['dependencies'])
    if name.startswith('prepare_'):
        from .data import validate_preparation
        arm = name.removeprefix('prepare_'); config = read_json(plan['configurations'][arm])
        prepared = Path(config['d_evidence_preparation_root']).expanduser().resolve()
        receipt = validate_preparation(config, prepared, verify_cache=False)
        for split, expected in (('train', 700), ('val', 150)):
            count = receipt['inventories'][split]['episodes']; count = len(count) if isinstance(count, list) else count
            if count != expected: raise ValueError('Preparation changed the registered episode budget')
        return dict(receipt_sha256=sha256(prepared / 'preparation_receipt.json'))
    if name == 'roi_export':
        from .roi_export import validate_export
        receipt = validate_export(read_json(plan['configurations']['roi']), output / 'roi')
        return dict(receipt=receipt)
    if name.startswith('train_'):
        from .train import validate_training
        arm = name.removeprefix('train_'); location = output / 'training' / arm
        validate_training(location); value = read_json(location / 'run_manifest.json')
        if value.get('optimizer_steps') != 20000 or value.get('world_size') != 3 or value.get('global_batch_size') != 48:
            raise ValueError('Training changed registered steps/world/batch')
        for key in ('steps', 'batch_size', 'seed', 'learning_rate', 'detection_dropout', 'validate_every', 'threads'):
            if value.get('hyperparameters', {}).get(key) != plan['settings'][key]: raise ValueError('Training budget mismatch: ' + key)
        if len(value.get('train_episodes', [])) != 700 or len(value.get('validation_episodes', [])) != 150:
            raise ValueError('Training omitted episodes')
        return {name: sha256(location / name) for name in ('training_route_receipt.json', 'run_manifest.json', 'depth.rmse.best.pt')}
    if name == 'evaluation':
        from .evaluate import validate_completion
        receipt = validate_completion(output)
        if receipt.get('status') != 'complete': raise ValueError('Evaluation lacks its verified complete receipt')
        return dict(receipt=receipt, report_sha256=sha256(output / 'REPORT_CN.md'))
    raise ValueError('Unknown stage')


def stage_worker(name, plan_path):
    plan = read_json(plan_path); check_plan(plan); output = Path(plan['output'])
    stage = next(item for item in plan['stages'] if item['name'] == name); path = output / 'stages' / (name + '.json')
    record = dict(schema='d_evidence_stage_v1', stage=name, status='running', pid=os.getpid(),
        process_start=process_start(os.getpid()), plan_sha256=sha256(plan_path), started_unix=time.time(), command=stage['command'])
    write_json(path, record); environment = os.environ.copy()
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'TORCHELASTIC_RUN_ID'): environment.pop(key, None)
    environment.update(stage['environment']); code = 1
    try:
        if stage.get('gpu_ids'): wait_resources(plan, stage['gpu_ids'])
        child = subprocess.Popen(stage['command'], cwd=plan['cwd'], env=environment)
        record['child_pid'] = child.pid; write_json(path, record); code = child.wait()
        record.update(exitcode=code, status='exited' if code == 0 else 'failed')
    except BaseException as exc:
        record.update(exitcode=1, status='failed', error=repr(exc)); raise
    finally:
        record.update(finished_unix=time.time(), elapsed_s=time.time() - record['started_unix']); write_json(path, record)
    return code


def append_progress(output):
    message = format_progress(snapshot(output)); print(message, flush=True)
    with (Path(output) / 'progress.log').open('a') as stream: stream.write(message + '\n')


def run(args, plan):
    output = Path(plan['output']); output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'controller.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = output / 'run_plan.json'
    if plan_path.exists():
        if read_json(plan_path) != plan: raise ValueError('Existing evidence plan differs; preserve it and use a new experiment')
    else: write_json(plan_path, plan)
    state_path = output / 'run_state.json'
    state = read_json(state_path) if state_path.exists() else dict(schema='d_evidence_run_state_v1', started_unix=time.time(), stages={})
    state.update(status='running', controller_pid=os.getpid(), controller_process_start=process_start(os.getpid()),
        settings=plan['settings'], plan_sha256=sha256(plan_path), resumed_unix=time.time(), dependencies=plan['dependencies'])
    state.pop('error', None); write_json(state_path, state)
    try:
        dependencies_ready(plan['dependencies'])
        for stage in plan['stages']:
            name = stage['name']; check_plan(plan); state['current_stage'] = name; write_json(state_path, state)
            path = output / 'stages' / (name + '.json'); record = read_json(path) if path.exists() else {}
            ownership = state['stages'].get(name, {})
            if record.get('status') == 'launching' and not record.get('pid') and alive(ownership):
                record.update(pid=ownership['pid'], process_start=ownership['process_start'])
            if record and record.get('plan_sha256') != sha256(plan_path): raise ValueError('Stage owner differs from this immutable plan')
            if record.get('status') == 'complete':
                evidence = verify_stage(name, plan)
                if evidence != record.get('completion_evidence'): raise ValueError('Completed evidence output changed: ' + name)
                state['stages'][name] = dict(status='complete', reused=True, **evidence); write_json(state_path, state); append_progress(output); continue
            process = None
            if record.get('status') in ('running', 'launching') and alive(record): pass
            elif record.get('status') == 'exited': pass
            elif record.get('status') == 'failed': raise RuntimeError('Stage previously failed; inspect preserved artifacts: ' + name)
            elif record: raise RuntimeError('Stage vanished without exit receipt; do not duplicate: ' + name)
            else:
                log = output / 'logs' / (name + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
                record = dict(schema='d_evidence_stage_v1', stage=name, status='launching', plan_sha256=sha256(plan_path),
                              command=stage['command'], started_unix=time.time()); write_json(path, record)
                with log.open('a', buffering=1) as stream:
                    process = subprocess.Popen([sys.executable, '-m', 'd_evidence_study.run', '--_stage-worker', name,
                        '--_plan', str(plan_path)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                record.update(pid=process.pid, process_start=process_start(process.pid)); state['stages'][name] = dict(record); write_json(state_path, state)
            while True:
                if path.exists():
                    current = read_json(path)
                    if not current.get('pid') and record.get('pid'): current.update(pid=record['pid'], process_start=record['process_start'])
                    record = current
                if record.get('status') in ('exited', 'failed'): break
                if process is not None and process.poll() is not None: raise RuntimeError('Supervisor exited without receipt: ' + name)
                if process is None and not alive(record): raise RuntimeError('Owned stage vanished without exit receipt: ' + name)
                append_progress(output); time.sleep(args.poll_seconds)
            if record.get('exitcode') != 0: raise RuntimeError(f'Stage failed: {name}; exitcode={record.get("exitcode")}; see logs/{name}.log')
            if process is not None: process.wait(timeout=30)
            evidence = verify_stage(name, plan); record.update(status='complete', completion_evidence=evidence, verified_unix=time.time())
            write_json(path, record); state['stages'][name] = dict(status='complete', **evidence); write_json(state_path, state); append_progress(output)
        state.update(status='complete', current_stage=None, finished_unix=time.time(), production_default_changed=False)
        write_json(state_path, state); append_progress(output); return 0
    except BaseException as exc:
        state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc), updated_unix=time.time())
        write_json(state_path, state); append_progress(output); raise
    finally: lock.close()


def main(argv=None):
    args = parser().parse_args(argv)
    if args._stage_worker or args._wait:
        if args._plan is None: raise ValueError('Private stage requires an immutable plan')
        return stage_worker(args._stage_worker, args._plan) if args._stage_worker else wait_dependencies(read_json(args._plan))
    plan = make_plan(args)
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False)); return 0
    if args.detach:
        output = Path(plan['output']); output.mkdir(parents=True, exist_ok=True)
        raw = list(sys.argv[1:] if argv is None else argv)
        command = [sys.executable, '-m', 'd_evidence_study.run'] + [item for item in raw if item != '--detach']
        with (output / 'controller.log').open('a', buffering=1) as stream:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(controller_pid=child.pid, output=str(output), progress_command='tail -f ' + str(output / 'progress.log')), ensure_ascii=False)); return 0
    return run(args, plan)


if __name__ == '__main__': raise SystemExit(main())
