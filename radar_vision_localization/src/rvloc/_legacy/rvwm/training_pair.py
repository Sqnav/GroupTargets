"""Plan or launch one configured RVWM training job on six GPUs.

The historical module name is retained for command compatibility. Planning
uses the standard library only and does not start training or create outputs.
Completed runs, including historical experiments, remain protected.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'observer_referenced_ra_re_continuous_v1'
VARIANTS = ('single',)
LAUNCH_VARIANTS = VARIANTS
PLAN_SCHEMA = 'ra_re_single_frame_training_v1'
PROTECTED = ('progress.json', 'run_manifest.json', 'metrics.jsonl', 'train.log', 'ranks',
             'world.pt', 'world.best.pt', 'world.latest.pt', 'configuration.json')


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        temporary = Path(stream.name)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_configuration(path, variant='single'):
    path = Path(path).resolve()
    raw = path.read_bytes()
    config = json.loads(raw)
    if config.get('architecture_schema') != SCHEMA:
        raise ValueError('Expected the single-frame RA/RE architecture: ' + str(path))
    from .phase_selection import configured_phases
    configured_phases(config)
    if variant not in LAUNCH_VARIANTS:
        raise ValueError('Only the single-frame model is supported')
    window=config.get('window',1)
    if isinstance(window,bool) or not isinstance(window,int) or window!=1:
        raise ValueError('Single-frame input requires integer window=1')
    if config.get('group_temporal_enabled',False) is not False:
        raise ValueError('Only single-frame inputs are supported')
    if any(key in config for key in ('group_temporal_layers','group_temporal_heads')):
        raise ValueError('Temporal architecture settings have been removed')
    required = {'dataset_root': 'dir', 'detector_weights': 'file', 'track_detection_dir': 'dir',
                'visual_split_manifest': 'file'}
    if config.get('split_manifest'):
        required['split_manifest'] = 'file'
    if config.get('visual_point_mode','bbox_center')!='bbox_center' or config.get('keypoint_checkpoint') is not None:
        raise ValueError('Current localization uses detector bbox centers without an ROI keypoint head')
    for key, kind in required.items():
        candidate = Path(config.get(key) or '')
        if (kind == 'file' and not candidate.is_file()) or (kind == 'dir' and not candidate.is_dir()):
            raise FileNotFoundError('Required frozen input missing: ' + key + ' = ' + str(candidate))
    return config, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def _check_outputs(root, variants):
    if (root / 'queue.json').exists() or (root / 'pair_plan.json').exists():
        raise FileExistsError('Existing training run is protected: ' + str(root))
    for name in PROTECTED:
        if (root / name).exists():
            raise FileExistsError('Existing root training output is protected: ' + str(root / name))
    for variant in variants:
        for name in PROTECTED:
            if (root / variant / name).exists():
                raise FileExistsError('Existing variant output is protected: ' + str(root / variant / name))


def build_plan(output, variant='single', single_config=None,
               training_args=(), python=None, environ=None):
    """Validate the input and destination without writing or launching."""
    from .phase_selection import configured_phases
    env = os.environ if environ is None else environ
    if variant not in LAUNCH_VARIANTS:
        raise ValueError('Only the single-frame model is supported')
    variants = VARIANTS
    output = Path(output).resolve()
    path = single_config or env.get('RVWM_TRAIN_CONFIG', ROOT / 'config.json')
    configuration, source = _load_configuration(path,variant)
    configurations, sources = {variants[0]: configuration}, {variants[0]: source}
    _check_outputs(output, variants)
    extras = list(training_args)
    reserved = ('--config', '--output', '--device', '--local-rank', '--local_rank', '--temporal-config')
    if any(arg == flag or arg.startswith(flag + '=') for arg in extras for flag in reserved):
        raise ValueError('The launcher owns configuration, output and device; pass architecture options through its configuration')
    defaults = ['--device', 'cuda:0', '--max-episodes', '0', '--frames-per-episode', '0',
        '--steps', str(env.get('RVWM_TRAIN_STEPS', '20000')),
        '--batch-size', str(env.get('RVWM_BATCH_PER_GPU', '8')),
        '--validation-episodes', '0', '--validation-frames', '0',
        '--validate-every', str(env.get('RVWM_VALIDATE_EVERY', '1000')),
        '--physics-every', '0', '--physics-weight', '0']
    executable = str(python or sys.executable)
    jobs = []
    for name in variants:
        directory = output / name
        configuration = dict(configurations[name], device='cuda:0',
                             uncertainty_calibrated=False, world_checkpoint=str(directory / 'world.best.pt'))
        command = [executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                   '--nproc_per_node=6', '-m', 'rvwm.group_training', '--config',
                   str(directory / 'configuration.json'), '--output', str(directory), *defaults, *extras]
        jobs.append(dict(variant=name, output=str(directory), configuration=configuration,
                         source_configuration=sources[name], command=command,
                         command_display=shlex.join(command), log=str(directory / 'train.log')))
    return dict(schema=PLAN_SCHEMA, output=str(output), jobs=jobs,
                order=list(variants), world_size_per_job=6, scheduling='single_frame_all_six_gpus',
                training_phase=configuration.get('training_phase','track'), training_phases=list(configured_phases(configuration)), train_validation_test_split='same whole-episode manifest',
                observation_contract='one current recorded-phase frame per sample; no history or future input',
                training_started=False)


def check_six_cuda():
    # Only explicit execution imports torch. --plan-only works without a GPU.
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 6:
        raise RuntimeError('Exactly six visible CUDA GPUs are required for execution')
    return [torch.cuda.get_device_name(index) for index in range(6)]


def run_command(job):
    """Stream one torchrun job to terminal and its log; stop it on Ctrl+C."""
    with Path(job['log']).open('x') as log:
        process = subprocess.Popen(job['command'], cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                sys.stdout.write(line); sys.stdout.flush()
                log.write(line); log.flush()
            return process.wait()
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
            raise


def run_plan(plan, runner=None):
    """Execute a prevalidated plan. Tests inject a runner and never train."""
    runner = run_command if runner is None else runner
    root = Path(plan['output'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'launcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This output root already has a running launcher') from error
        _check_outputs(root, plan['order'])
        _write_json(root / 'pair_plan.json', plan)
        began = time.time()
        queue = dict(schema=plan['schema'], status='preparing', order=plan['order'],
                     total_jobs=len(plan['jobs']), completed_jobs=0, active_variant=None,
                     active_run_dir=None, started_unix=began, updated_unix=began,
                     jobs=[dict(variant=j['variant'], output=j['output'], status='pending') for j in plan['jobs']])
        _write_json(root / 'queue.json', queue)
        try:
            for job in plan['jobs']:
                _write_json(Path(job['output']) / 'configuration.json', job['configuration'])
            for index, job in enumerate(plan['jobs']):
                queue.update(status='running', active_variant=job['variant'], active_run_dir=job['output'],
                             active_job=index + 1, updated_unix=time.time())
                queue['jobs'][index]['status'] = 'running'
                _write_json(root / 'queue.json', queue)
                print(f"\n[{index + 1}/{len(plan['jobs'])}] {job['variant']} — six GPUs; log: {job['log']}", flush=True)
                code = int(runner(job))
                queue['jobs'][index].update(status='complete' if code == 0 else 'failed', returncode=code)
                if code:
                    queue.update(status='failed', returncode=code, error='Variant failed: ' + job['variant'])
                    return code
                queue['completed_jobs'] += 1
                _write_json(root / 'queue.json', dict(queue, updated_unix=time.time()))
            queue.update(status='complete', returncode=0)
            return 0
        except KeyboardInterrupt:
            queue.update(status='interrupted', returncode=130, error='Launcher interrupted; active torchrun stopped')
            raise
        except BaseException as error:
            queue.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            queue.update(updated_unix=time.time(), elapsed_s=time.time() - began)
            _write_json(root / 'queue.json', queue)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    output = argv.pop(0) if argv and not argv[0].startswith('-') else 'results/training'
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--output-root', default=output)
    parser.add_argument('--variant', choices=LAUNCH_VARIANTS, default='single')
    parser.add_argument('--single-config', type=Path)
    parser.add_argument('--plan-only', action='store_true', help='validate and print commands; no files, CUDA or training')
    args, extra = parser.parse_known_args(argv)
    plan = build_plan(args.output_root, args.variant, args.single_config, extra)
    if args.plan_only:
        print(json.dumps(plan, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    names = check_six_cuda()
    print('Scheduling: one RA/RE training job, six GPUs; variants: '+', '.join(plan['order']), flush=True)
    print('Only configured recorded phases; whole-episode partitions remain separate.', flush=True)
    for index, name in enumerate(names):
        print('GPU', index, name, flush=True)
    return run_plan(plan)


if __name__ == '__main__':
    def termination_requested(signum,frame):
        # Unwind run_command so its owned torchrun session is also stopped.
        raise KeyboardInterrupt('Single-frame training termination requested')
    signal.signal(signal.SIGTERM,termination_requested)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
