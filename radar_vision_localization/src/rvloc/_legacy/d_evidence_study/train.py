"""Fresh single-frame D ablations with private immutable-trainer callbacks.

No fusion checkpoint initializes a model. The original D parameter tensors,
after train-only prior initialization, must match the registered fresh BW32
control's initial fingerprint before the first formal optimizer update.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

from precision_study.deployed_infer import DEFAULT_DESCRIPTOR, load_descriptor
from precision_study.deployed_train import source_identity
from precision_study.versions import frozen_study_package
from rvwm.data import sha256


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / 'results/d_evidence_20261124'
BASELINE = ROOT / 'results/bandwidth_double_20261124/training/baseline'
ARMS = ('sequence', 'aligned', 'roi')
SCHEMA = 'd_evidence_depth_v1'


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def normalized_config(value, arm):
    config = deepcopy(value)
    if arm not in ARMS or config.get('d_evidence_arm', arm) != arm:
        raise ValueError('CLI arm and registered d_evidence_arm disagree')
    config['d_evidence_arm'] = arm
    if config.get('d_evidence_schema') != SCHEMA or config.get('window') != 1:
        raise ValueError('Independent single-frame evidence schema and window=1 required')
    names = dict(sequence='d_evidence_candidate_sequence', aligned='d_evidence_aligned_first_query',
                 roi='d_evidence_frozen_roi')
    for name, flag in names.items():
        if config.get(flag, name == arm) is not (name == arm):
            raise ValueError('Only the registered single intervention may be active: ' + flag)
        config[flag] = name == arm
    return config


def validate_base_config(config, original):
    mutable = {'depth_checkpoint', 'depth_model_trained', 'completed_optimizer_steps'}
    fixed = {key for key in original if key.startswith(('depth_', 'ra_re_')) and key not in mutable}
    fixed |= {'architecture_schema', 'window', 'group_hypotheses', 'precision_radar_mode',
        'visual_point_mode', 'observer_handling', 'detector_backend', 'detector_weights',
        'detector_weights_sha256', 'detector_confidence', 'dataset_root', 'echo_subdir',
        'split_manifest', 'split_manifest_sha256', 'visual_split_manifest', 'track_detection_dir',
        'training_phases', 'inference_phases', 'training_sampling_policy'}
    for key in fixed:
        if config.get(key) != original.get(key):
            raise ValueError('Evidence ablation changed the common D/data contract: ' + key)


def parameter_digest(model, names):
    """Use the same name/order/float bytes as the unchanged D trainer."""
    named = dict(model.named_parameters())
    digest = hashlib.sha256()
    for name in names:
        if name not in named:
            raise ValueError('A shared original D parameter was removed or renamed: ' + name)
        digest.update(name.encode())
        digest.update(named[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def reference_initialization(model, config, original_model_module, *, formal):
    import torch
    # Only names/order are needed; preserve both CPU and all GPU RNG streams.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        reference = original_model_module.DepthLocalizer(config)
        names = [name for name, _ in reference.named_parameters()]
    del reference
    actual = parameter_digest(model, names)
    run = json.loads((BASELINE / 'run_manifest.json').read_text())
    expected = run['model_initialization']['initial_parameter_sha256']
    if formal and run.get('status') != 'complete':
        raise ValueError('The registered fresh BW32 control has not completed')
    if formal and actual != expected:
        raise ValueError('Shared original D initialization differs from the fresh BW32 control')
    return dict(reference_run_manifest=str(BASELINE / 'run_manifest.json'),
        reference_run_manifest_sha256=sha256(BASELINE / 'run_manifest.json'),
        reference_seed=run['hyperparameters']['seed'], shared_parameter_count=len(names),
        shared_parameter_sha256=actual, reference_initial_parameter_sha256=expected,
        matches_reference=actual == expected, required_before_optimizer=formal,
        fusion_weights_loaded=False)


def frozen_trainer(arm, *, formal=False):
    from . import model as custom_model
    from . import checkpoint as custom_checkpoint
    from .data import load_samples, drop_detection_features
    description, deployment = load_descriptor(DEFAULT_DESCRIPTOR)
    if description['model_name'] != 'D_wide_residual_ra_re':
        raise ValueError('The original D must remain the authenticated production baseline')
    source_identity(description)
    original_alias, package = frozen_study_package(description['source_root'])
    original_model = importlib.import_module(original_alias + '.model')
    original_evaluation = importlib.import_module(original_alias + '.evaluation')
    alias = '_d_evidence_private_training_' + sha256(package / 'training.py')[:16]
    private = types.ModuleType(alias); private.__path__ = [str(package)]; private.__package__ = alias
    sys.modules[alias] = private
    injected_data = types.ModuleType(alias + '.data')
    injected_data.load_samples = load_samples
    injected_data.drop_detection_features = drop_detection_features
    for name, module in (('model', custom_model), ('checkpoint', custom_checkpoint),
                         ('data', injected_data), ('evaluation', original_evaluation)):
        setattr(private, name, module); sys.modules[alias + '.' + name] = module
    spec = importlib.util.spec_from_file_location(alias + '.training', package / 'training.py')
    trainer = importlib.util.module_from_spec(spec); sys.modules[spec.name] = trainer
    spec.loader.exec_module(trainer)
    original_write, original_prepare = trainer.write_json, trainer._prepare_training
    initialization_evidence = {}

    def write_evidence(path, value):
        if Path(path).name in ('run_manifest.json', 'progress.json', 'training_plan.json') and isinstance(value, dict):
            value = dict(value, d_evidence_schema=SCHEMA, d_evidence_arm=arm,
                         input_frames=1, detector_frozen=True, training_from_scratch=True)
            if 'architecture_schema' in value:
                value['base_architecture_schema'] = value['architecture_schema']; value['architecture_schema'] = SCHEMA
            if value.get('model_initialization') and initialization_evidence:
                value['model_initialization'] = dict(value['model_initialization'],
                    base_initial_parameter_sha256=initialization_evidence['shared_parameter_sha256'],
                    base_reference_initial_parameter_sha256=initialization_evidence['reference_initial_parameter_sha256'],
                    base_initialization_matches_reference=initialization_evidence['matches_reference'])
        return original_write(path, value)

    def prepare_with_fairness(model, samples, world, config):
        result = original_prepare(model, samples, world, config)
        evidence, failure = None, None
        try:
            evidence = reference_initialization(model, config, original_model, formal=formal)
            if formal:
                reference = json.loads((BASELINE / 'run_manifest.json').read_text())
                if abs(float(result['depth_m']) - float(reference['depth_prior_initialization']['depth_m'])) > 1e-7:
                    raise ValueError('Train-only projection-depth prior differs from the common BW32 control')
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'
        trainer._collective_error(world, failure, 'shared D initialization fairness')
        initialization_evidence.update(evidence)
        return dict(result, shared_D_initialization=evidence)

    trainer.write_json, trainer._prepare_training = write_evidence, prepare_with_fairness
    binding = dict(training_path=str(package / 'training.py'), training_sha256=sha256(package / 'training.py'),
        evaluation_path=str(package / 'evaluation.py'), evaluation_sha256=sha256(package / 'evaluation.py'),
        model_factory='d_evidence_study.model.DepthLocalizer', checkpoint_callback='d_evidence_study.checkpoint',
        data_callback='d_evidence_study.data.load_samples', dropout_callback='d_evidence_study.data.drop_detection_features',
        production_module_registry_unchanged=True)
    return trainer, description, deployment, binding


def validate_training(output):
    from .checkpoint import source_hashes
    output = Path(output).resolve()
    route = json.loads((output / 'training_route_receipt.json').read_text())
    run = json.loads((output / 'run_manifest.json').read_text())
    if route.get('status') != 'complete' or run.get('status') != 'complete':
        raise ValueError('Evidence training is incomplete')
    if route['evidence_source_sha256'] != source_hashes():
        raise ValueError('Evidence model/data/training sources changed')
    identity = route['effective_configuration']
    if sha256(identity['path']) != identity['sha256'] or run.get('training_configuration') != identity:
        raise ValueError('Effective evidence training configuration changed')
    if run.get('optimizer_steps') != route['optimizer_steps'] or run.get('architecture_schema') != SCHEMA:
        raise ValueError('Training route/schema differs from the real run')
    preparation = route.get('preparation')
    if not preparation or sha256(preparation['path']) != preparation['sha256']:
        raise ValueError('Registered prepared sensor receipt changed or is missing')
    init = run['depth_prior_initialization']['shared_D_initialization']
    if not init.get('matches_reference') or not init.get('required_before_optimizer'):
        raise ValueError('Shared D initialization fairness was not established')
    if sha256(init['reference_run_manifest']) != init['reference_run_manifest_sha256']:
        raise ValueError('The common fresh BW32 reference changed')
    for item in route.get('checkpoints', {}).values():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Evidence checkpoint bytes changed')
    return route


def main(argv=None):
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--arm', choices=ARMS, default='sequence')
    early.add_argument('--dry-run', action='store_true'); early.add_argument('--plan-only', action='store_true')
    early.add_argument('--preflight', action='store_true')
    selected, _ = early.parse_known_args(argv)
    formal = not (selected.dry_run or selected.plan_only or selected.preflight)
    trainer, description, deployment, binding = frozen_trainer(selected.arm, formal=formal)
    parser = trainer.build_parser(); parser.description = __doc__
    parser.set_defaults(config=None, output=None, batch_size=16, seed=20261118, device='cuda')
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    if args.config is None or args.output is None:
        raise ValueError('Explicit independent configuration and output are required')
    config_path = args.config.expanduser().resolve()
    config = normalized_config(json.loads(config_path.read_text()), args.arm)
    validate_base_config(config, json.loads(Path(description['config']).read_text()))
    trainer.validate_options(args, config); plan = trainer.make_plan(config, args)
    if args.preflight or args.plan_only:
        print(json.dumps(dict(status='preflight_passed' if args.preflight else 'planned',
            plan=plan, input_frames=1, model_constructed=False, optimizer_started=False,
            sensor_payloads_read=False, deployment=deployment, callbacks=binding), ensure_ascii=False, indent=2), flush=True)
        return 0
    if formal and not args.device.startswith('cuda'):
        raise ValueError('Formal evidence training requires GPU; CPU is only for explicit dry-run')
    if args.device.startswith('cuda'):
        from .run import registered_dependencies, dependencies_ready
        if not dependencies_ready(registered_dependencies()):
            raise RuntimeError('Existing BW/temporal tasks still own the GPUs; use the queued controller')
    from .data import load_samples
    from .checkpoint import source_hashes
    preparation = None
    if formal:
        from .data import validate_preparation
        prepared = Path(config['d_evidence_preparation_root']).expanduser().resolve()
        validate_preparation(config, prepared, verify_cache=False)
        preparation = dict(path=str(prepared / 'preparation_receipt.json'), sha256=sha256(prepared / 'preparation_receipt.json'))
    rank = int(os.environ.get('RANK', '0')); output = args.output.expanduser().resolve()
    config.update(d_evidence_training_source_sha256=source_hashes(),
        d_evidence_frozen_source_root=str(description['source_root']),
        d_evidence_original_configuration_sha256=deployment['configuration_sha256'])
    effective, ready = output / 'inputs/configuration.json', output / 'inputs/launch_ready.json'
    token = os.environ.get('TORCHELASTIC_RUN_ID', 'single_process'); lock = None
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True); lock = (output / 'route.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((output / name).exists() for name in trainer.PROTECTED):
            raise FileExistsError('Existing evidence training output is protected: ' + str(output))
        write_json(effective, config); write_json(ready, dict(token=token, configuration_sha256=sha256(effective)))
    else:
        deadline = time.monotonic() + 120
        while True:
            try:
                identity = json.loads(ready.read_text())
                if identity['token'] == token and identity['configuration_sha256'] == sha256(effective): break
            except (OSError, ValueError, KeyError): pass
            if time.monotonic() > deadline: raise TimeoutError('Rank zero did not publish this launch configuration')
            time.sleep(.1)
    if json.loads(effective.read_text()) != config: raise ValueError('Rank configurations disagree')
    route_path = output / 'training_route_receipt.json'
    route = dict(schema='d_evidence_training_route_v1', status='starting', arm=args.arm,
        original_configuration=dict(path=str(config_path), sha256=sha256(config_path)),
        effective_configuration=dict(path=str(effective), sha256=sha256(effective)),
        evidence_source_sha256=source_hashes(), deployment=deployment, callbacks=binding,
        preparation=preparation,
        input_frames=1, detector_frozen=True, fusion_training_from_scratch=True,
        initial_fusion_checkpoint=None, baseline_reference=str(BASELINE),
        world_size=int(os.environ.get('WORLD_SIZE', '1')), started_unix=time.time(),
        command=[sys.executable, '-m', 'd_evidence_study.train', *(argv if argv is not None else sys.argv[1:])])
    if rank == 0: write_json(route_path, route)
    args.config = effective
    try:
        result = trainer.train(args, prepared_loader=load_samples)
    except BaseException as exc:
        if rank == 0: write_json(route_path, dict(route, status='failed', error=repr(exc), finished_unix=time.time()))
        raise
    else:
        if rank == 0:
            checkpoints = {}
            for name in ('depth.pt', 'depth.best.pt', 'depth.rmse.best.pt', 'depth.1m.best.pt'):
                path = output / name
                if path.exists(): checkpoints[name] = dict(path=str(path), sha256=sha256(path))
            route.update(status=result['status'], optimizer_steps=result.get('optimizer_steps', 0),
                         finished_unix=time.time(), checkpoints=checkpoints)
            write_json(route_path, route)
            print(json.dumps(dict(status=result['status'], optimizer_steps=result.get('optimizer_steps', 0))), flush=True)
        return 0
    finally:
        if lock is not None: lock.close()


if __name__ == '__main__': raise SystemExit(main())
