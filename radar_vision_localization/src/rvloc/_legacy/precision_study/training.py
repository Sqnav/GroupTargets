"""Independent training for deterministic, metre-supervised single-frame depths.

Only features reach the model's observation input.  Labels remain separate for
training, evaluation, matching and a documented train-only depth initialization.
The existing fusion package and its artifacts are never rewritten.
This isolated trainer retains the same budget, episode-balanced sampler and
optimizer schedule, with independently bound checkpoint and sensor contracts.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from rvwm.data import Dataset
from rvwm.group_model import match_ray_labels
from rvwm.group_training import (EpisodeUniformSampler, TrainingWorld,
    _parameter_difference, sampling_coverage_plan, synchronize_group_gradients)
from rvwm.partitions import EPISODE_SAMPLING, active_splits, partition_receipt
from rvwm.phase_selection import configured_phases, selected_frame_indices
from depth_localization.schedule import make_schedule, learning_rate_for_step


PROJECT = Path(__file__).resolve().parents[1]
SCHEMA = 'single_frame_geometry_guided_depth_v2'
PROTECTED = ('depth.pt', 'depth.best.pt', 'depth.rmse.best.pt', 'depth.1m.best.pt',
             'prior_validation_best.json', 'prior_validation_rmse_best.json',
             'prior_validation_1m_best.json', 'world.pt', 'world.best.pt',
             'run_manifest.json', 'metrics.jsonl', 'progress.json', 'training_plan.json')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def validate_options(args, config):
    if (config.get('architecture_schema') != SCHEMA or config.get('window') != 1
            or config.get('group_hypotheses') != 1):
        raise ValueError('Depth training requires its independent schema, window=1 and group_hypotheses=1')
    if min(args.steps, args.validate_every, args.batch_size, args.threads) < 1:
        raise ValueError('Training steps, validation interval, batch and threads must be positive')
    if min(args.max_episodes, args.frames_per_episode, args.validation_episodes, args.validation_frames) < 0:
        raise ValueError('Data limits cannot be negative; zero means every selected frame/episode')
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('A positive finite learning rate is required')
    if not 0 <= args.detection_dropout <= 1:
        raise ValueError('Detection dropout must be a probability')
    if getattr(args, 'resume', None):
        raise ValueError('Resume is not implemented for this independent depth trainer')
    schedule_steps = int(config.get('precision_schedule_total_steps', args.steps))
    if schedule_steps < args.steps:
        raise ValueError('Schedule horizon cannot be shorter than optimizer budget')
    make_schedule(config, args.learning_rate, schedule_steps)
    if config.get('precision_radar_mode', 'ra_re') not in ('ra_re', 'joint_volume', 'none'):
        raise ValueError('Unknown precision evidence mode')
    if config.get('training_sampling_policy', EPISODE_SAMPLING) != EPISODE_SAMPLING:
        raise ValueError('This trainer fixes equal-episode, per-episode frame-cycle sampling')


def _dataset_context(config):
    if config.get('precision_radar_mode') == 'none':
        from rvwm.partitions import canonical_manifest
        from .visual_data import VisualSensorReader
        _, manifest, _ = canonical_manifest(config)
        splits = {name:list(manifest['splits'][name]['episode_ids']) for name in ('train','val','test')}
        readers = {name:VisualSensorReader(config,name,load_detector=False) for name in splits}
        roles = {episode:name for name,ids in splits.items() for episode in ids}
        class VisualPlan:
            def frames_in_phase(self, episode, phase):
                record=readers[roles[episode]].episode(episode)
                return [row['frame_idx'] for row in record['capture'] if row['phase']==phase]
        return VisualPlan(), splits, partition_receipt(config)
    dataset = Dataset(config['dataset_root'], echo_subdir=config.get('echo_subdir', 'echoes'), verify_sources=True)
    return dataset, active_splits(dataset, config), partition_receipt(config)


def make_plan(config, args, world_size=None):
    """Metadata-only plan: no model, feature reader, IQ/RGB decode or GPU setup."""
    validate_options(args, config)
    size = int(world_size or os.environ.get('WORLD_SIZE', '1'))
    if size < 1:
        raise ValueError('World size must be positive')
    dataset, splits, partition = _dataset_context(config)
    train_ids = list(splits['train'][:args.max_episodes] if args.max_episodes else splits['train'])
    val_ids = list(splits['val'][:args.validation_episodes] if args.validation_episodes else splits['val'])
    dry_run = bool(getattr(args, 'dry_run', False))
    if dry_run:
        train_ids, val_ids = train_ids[:size], val_ids[:1]
    if not train_ids or not val_ids or len(train_ids) < size or set(train_ids) & set(val_ids):
        raise ValueError('Need separate nonempty train/validation episodes and one training episode per rank')
    tc, vc = (1, 1) if dry_run else (args.frames_per_episode, args.validation_frames)
    phases = configured_phases(config)
    counts = {episode: len(selected_frame_indices(dataset, episode, phases, tc)) for episode in train_ids}
    validation_counts = {episode: len(selected_frame_indices(dataset, episode, phases, vc)) for episode in val_ids}
    train_episode_subset = len(train_ids) != len(splits['train'])
    val_episode_subset = len(val_ids) != len(splits['val'])
    train_frame_subset = tc > 0 and any(counts[episode] < len(selected_frame_indices(dataset, episode, phases))
                                        for episode in train_ids)
    val_frame_subset = vc > 0 and any(validation_counts[episode] < len(selected_frame_indices(dataset, episode, phases))
                                    for episode in val_ids)
    if any(count < 1 for count in [*counts.values(), *validation_counts.values()]):
        raise ValueError('Selected episode has no eligible frames')
    shards = []
    for rank in range(size):
        ids = train_ids[rank::size]
        bound = (len(ids) * max(counts[episode] for episode in ids) + args.batch_size - 1) // args.batch_size
        shards.append(dict(rank=rank, episodes=ids, expected_frames=sum(counts[e] for e in ids),
                           full_frame_coverage_steps_upper_bound=bound))
    return dict(status='planned', architecture_schema=SCHEMA, optimizer_started=False,
        model_constructed=False, sensor_payloads_read=False, world_size=size,
        train_episodes=train_ids, validation_episodes=val_ids, partition=partition,
        train_frame_limit=tc, validation_frame_limit=vc, training_phases=list(phases),
        expected_training_frames=sum(counts.values()), expected_validation_frames=sum(validation_counts.values()),
        rank_partitions=shards, full_frame_coverage_steps_upper_bound=max(s['full_frame_coverage_steps_upper_bound'] for s in shards),
        requested_optimizer_steps=0 if dry_run else args.steps, global_batch_size=size * args.batch_size,
        all_frames_requested=not dry_run and args.frames_per_episode == 0,
        validation_all_frames_requested=not dry_run and args.validation_frames == 0,
        all_training_episodes_selected=not train_episode_subset,
        all_validation_episodes_selected=not val_episode_subset,
        training_frames_subset=bool(train_frame_subset), validation_frames_subset=bool(val_frame_subset),
        diagnostic_subset=bool(dry_run or train_episode_subset or val_episode_subset or train_frame_subset or val_frame_subset),
        checkpoint_names=['depth.best.pt', 'depth.rmse.best.pt', 'depth.1m.best.pt', 'depth.pt'], dry_run=dry_run,
        learning_rate_schedule=make_schedule(config, args.learning_rate, int(config.get('precision_schedule_total_steps', args.steps))),
        model_initialization=dict(method='random fusion parameters plus training-only median depth prior',
                                  loaded_fusion_checkpoint=None, resume=False),
        hyperparameters={key: getattr(args, key) for key in ('steps', 'validate_every', 'batch_size', 'learning_rate',
            'detection_dropout', 'seed', 'threads', 'max_episodes', 'frames_per_episode', 'validation_episodes', 'validation_frames')})


def _collective_error(world, error, phase):
    failures = world.gather(str(error) if error else None)
    message = next((f'{phase}: rank {rank}: {value}' for rank, value in enumerate(failures) if value), None) if world.rank == 0 else None
    message = world.broadcast(message)
    if message:
        raise RuntimeError(message)


def _load_stage(world, callback, phase):
    value, failure = None, None
    try:
        value = callback()
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    _collective_error(world, failure, phase)
    return value


def dropout_features(features, positions, rng, probability, config):
    """Remove all per-detection fields together, retain the measured mixed maps."""
    from .data import drop_detection_features
    n = len(features['ray_directions'])
    valid = np.asarray(features['ray_valid'], bool)
    if n < 2 or not valid.any() or rng.random() >= probability:
        return features, False, False
    keep = rng.random(n) > .2
    if not (keep & valid).any():
        keep[int(rng.choice(np.flatnonzero(valid)))] = True
    if keep.all():
        return features, False, False
    reduced = drop_detection_features(features, keep)
    if not match_ray_labels(reduced, positions, config)['diagnostics']['supervision_usable']:
        return features, False, True
    return reduced, True, False


def _prepare_training(model, training, world, config):
    projected = []
    failure = None
    try:
        for sample in training:
            features, positions = sample['features'], np.asarray(sample['positions'])
            assignment = match_ray_labels(features, positions, config)
            if not assignment['diagnostics']['supervision_usable']:
                raise ValueError('Prepared training sample has no accepted supervised ray')
            rows, cols = assignment['rows'], assignment['cols']
            projected.extend(np.sum((positions[cols] - np.asarray(features['ray_origins'])[rows]) *
                                    np.asarray(features['ray_directions'])[rows], axis=1).tolist())
    except Exception as exc:
        failure = f'{type(exc).__name__}: {exc}'
    _collective_error(world, failure, 'training-only initialization')
    values = world.gather(projected)
    median = float(np.median([depth for values_rank in values for depth in values_rank])) if world.rank == 0 else None
    median = world.broadcast(median)
    initialized = callable(getattr(model, 'initialize_depth_prior', None))
    if initialized:
        model.initialize_depth_prior(median)
    return dict(source='training matched projection depths only; no validation labels',
                statistic='median over accepted training member instances', depth_m=median, applied=initialized)


def _scalar(value):
    return float(value.detach()) if isinstance(value, torch.Tensor) else float(value)


def _inventory_digest(rows):
    """Bind prepared sensor/source receipts and selection, without copying GT."""
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _shared_parameter_digest(model):
    prefixes = ('visual_encoder.', 'visual_interaction.', 'coarse_head.',
                'observer_encoder.', 'global_attention.', 'global_norm.',
                'global_head.', 'refine_encoder.', 'refine_interaction.', 'refine_head.')
    digest = hashlib.sha256()
    for name,value in model.named_parameters():
        if name.startswith(prefixes):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def precision_selection_keys(report):
    """Independent validation-only selections; neither consults test labels."""
    summary = report['summary']
    rmse = summary['primary_member_rmse_m']
    negative_rmse = -rmse if rmse is not None else -float('inf')
    one = summary['primary_member_success_ratio_1m']
    five = summary['primary_member_success_ratio_5m']
    return dict(rmse=(negative_rmse, one, five), one_m=(one, negative_rmse, five))


def train(args, prepared_loader=None):
    """Train from authenticated samples; optional loader injection is for fixtures."""
    configuration_bytes = Path(args.config).read_bytes()
    original = json.loads(configuration_bytes)
    validate_options(args, original)
    if getattr(args, 'plan_only', False):
        plan = make_plan(original, args)
        if int(os.environ.get('RANK', '0')) == 0:
            print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
        return plan
    from . import checkpoint, evaluation
    from depth_localization.evaluation import selection_key
    from .model import DepthLocalizer
    if prepared_loader is None:
        from .data import load_samples
        prepared_loader = load_samples
    torch.set_num_threads(args.threads)
    world = TrainingWorld(args.device)
    output = Path(args.output).resolve()
    started = time.time(); begin = time.perf_counter(); lock = None; run = None; claimed = False
    try:
        failure = None
        if world.rank == 0:
            try:
                output.mkdir(parents=True, exist_ok=True)
                lock = (output / 'training.lock').open('a+')
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if any((output / name).exists() for name in PROTECTED):
                    raise FileExistsError('Existing training artifacts are protected: ' + str(output))
            except Exception as exc:
                failure = f'{type(exc).__name__}: {exc}'
        failure = world.broadcast(failure)
        if failure:
            raise FileExistsError(failure)
        claimed = True
        plan = _load_stage(world, lambda: make_plan(original, args, world.size), 'dataset planning')
        config = deepcopy(original)
        config.update(device=world.device, depth_checkpoint=str(output / 'depth.rmse.best.pt'),
            depth_model_trained=False, completed_optimizer_steps=0, training_sampling_policy=EPISODE_SAMPLING)
        np.random.seed(args.seed + world.rank * 100003)
        torch.manual_seed(args.seed + world.rank * 100003)
        rng = np.random.default_rng(args.seed + world.rank * 100003)
        source = checkpoint.source_hashes()
        progress = dict(stage='preparing', step=0, requested_steps=args.steps,
            started_unix=started, world_size=world.size, global_batch_size=world.size * args.batch_size,
            architecture_schema=SCHEMA, model_trained=False, optimizer_steps=0)
        if world.rank == 0:
            write_json(output / 'progress.json', progress)
            write_json(output / 'configuration.json', config)
            write_json(output / 'training_plan.json', plan)
        rank_path = output / 'ranks' / f'rank_{world.rank}'
        local_ids = plan['rank_partitions'][world.rank]['episodes']
        training, manifest, frontend = _load_stage(world, lambda: prepared_loader(
            config, local_ids, 'train', count=plan['train_frame_limit'], include_labels=True), 'training features')
        validation, vmanifest, vfrontend = _load_stage(world, lambda: prepared_loader(
            config, plan['validation_episodes'], 'val', count=plan['validation_frame_limit'], include_labels=True)
            if world.rank == 0 else ([], [], None), 'validation features')
        _collective_error(world, 'Nonempty usable training and validation features required'
            if not training or (world.rank == 0 and not validation) else None, 'usable features')
        coverage = sampling_coverage_plan(training, EPISODE_SAMPLING, args.batch_size)
        insufficient = not args.dry_run and args.frames_per_episode == 0 and args.steps < coverage['full_frame_coverage_steps_upper_bound']
        if world.sum([int(insufficient)])[0]:
            raise ValueError('Requested steps cannot visit every selected usable frame; increase budget or use --dry-run')
        usable_episodes = sorted({sample['episode_id'] for sample in training})
        weight = world.size * len(usable_episodes) / world.sum([len(usable_episodes)])[0]
        partition = dict(rank=world.rank, episodes=local_ids, usable_episodes=usable_episodes,
            groups=len(training), frames=manifest, expected_frames=plan['rank_partitions'][world.rank]['expected_frames'],
            episode_objective_rank_weight=float(weight), coverage=coverage)
        partitions = world.gather(partition)
        write_json(rank_path / 'train_features.json', dict(frames=manifest, frontend=frontend, source_sha256=source))
        error = None
        if world.rank == 0:
            try:
                inventory = [row for part in partitions for row in part['frames']]
                if any(len(part['frames']) != part['expected_frames'] for part in partitions):
                    raise ValueError('Training manifest does not cover selected frames')
                keys = [(row['episode_id'], row['frame_idx']) for row in inventory]
                if len(keys) != len(set(keys)) or len(inventory) != plan['expected_training_frames']:
                    raise ValueError('Duplicate or missing cross-rank training frames')
                if len(vmanifest) != plan['expected_validation_frames']:
                    raise ValueError('Validation manifest does not cover selected frames')
                write_json(output / 'train_inventory.json', dict(frames=inventory, source_sha256=source))
                write_json(output / 'val_features.json', dict(frames=vmanifest, frontend=vfrontend, source_sha256=source))
                run = dict(status='preparing', architecture_schema=SCHEMA, configuration=config,
                    train_episodes=plan['train_episodes'], validation_episodes=plan['validation_episodes'],
                    partition=plan['partition'], model_trained=False, optimizer_steps=0, world_size=world.size,
                    per_rank_batch_size=args.batch_size, global_batch_size=args.batch_size * world.size,
                    prepared_training_groups=sum(part['groups'] for part in partitions), prepared_validation_groups=len(validation),
                    total_training_selected_frames=len(inventory), total_validation_selected_frames=len(vmanifest),
                    inventory_sha256=_inventory_digest(inventory), validation_inventory_sha256=_inventory_digest(vmanifest),
                    training_configuration=dict(path=str(Path(args.config).resolve()), sha256=hashlib.sha256(configuration_bytes).hexdigest()),
                    all_selected_frames_prepared=True, rank_partitions=[{k:v for k,v in part.items() if k!='frames'} for part in partitions],
                    hyperparameters=plan['hyperparameters'], frontend=frontend, source_sha256=source,
                    learning_rate_schedule=plan['learning_rate_schedule'],
                    model_initialization=deepcopy(plan['model_initialization']),
                    training_sampling_policy=EPISODE_SAMPLING, seed=args.seed,
                    model_selection='minimum_matched_rmse_then_all_truth_1m_coverage',
                    coverage_model_selection='primary_5m_then_negative_rmse_then_primary_whole_scene',
                    raw_iq_likelihood=False, posterior_latent=False, hypotheses=1, uncertainty_calibrated=False,
                    precision_radar_mode=config.get('precision_radar_mode','ra_re'),
                    model_parameter_count=None)
            except Exception as exc:
                error = str(exc)
        _collective_error(world, error, 'prepared inventory')
        model = _load_stage(world, lambda: DepthLocalizer(config).to(world.device), 'model initialization')
        world.sync_model(model)
        initialization = _prepare_training(model, training, world, config)
        world.sync_model(model)
        if world.rank == 0:
            run['depth_prior_initialization'] = initialization
            run['model_parameter_count'] = sum(p.numel() for p in model.parameters())
            run['model_initialization']['initial_parameter_sha256'] = _parameter_digest(model)
            run['model_initialization']['shared_parameter_sha256'] = _shared_parameter_digest(model)
            scope = dict(all_frames_requested=plan['all_frames_requested'], frames_per_episode=args.frames_per_episode,
                max_episodes=args.max_episodes, validation_frames=args.validation_frames,
                validation_episodes=args.validation_episodes,
                all_training_episodes_selected=plan['all_training_episodes_selected'],
                all_validation_episodes_selected=plan['all_validation_episodes_selected'],
                training_frames_subset=plan['training_frames_subset'], validation_frames_subset=plan['validation_frames_subset'],
                selected_training_frames=run['prepared_training_groups'], selected_validation_frames=run['prepared_validation_groups'],
                requested_steps=args.steps, diagnostic_subset=plan['diagnostic_subset'])
            run['training_scope'] = scope
            lineage = dict(train_episodes=plan['train_episodes'], validation_episodes=plan['validation_episodes'],
                frontend=frontend, partition=plan['partition'], training_scope=scope, hyperparameters=plan['hyperparameters'],
                inventory_sha256=run['inventory_sha256'], validation_inventory_sha256=run['validation_inventory_sha256'],
                training_configuration=run['training_configuration'], depth_prior_initialization=initialization,
                learning_rate_schedule=plan['learning_rate_schedule'], model_initialization=run['model_initialization'])
            write_json(output / 'run_manifest.json', run)
        if args.dry_run:
            before = {key:value.detach().clone() for key,value in model.state_dict().items()}
            loss, terms = model.training_loss(training[0]['features'], training[0]['positions'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite dry-run objective')
            loss.backward(); synchronize_group_gradients(model, world)
            norm = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
            if not np.isfinite(norm) or any(not torch.equal(before[k], v) for k,v in model.state_dict().items()):
                raise ValueError('Dry run must have finite gradients and unchanged weights/buffers')
            write_json(rank_path / 'graph_validation.json', dict(loss=_scalar(loss), gradient_norm=norm,
                model_trained=bool(model.trained), optimizer_steps=0, parameters_unchanged=True))
            if world.rank == 0:
                report = evaluation.evaluate(model, validation, vmanifest, config, allow_untrained=True)
                write_json(output / 'dry_run_validation.json', report)
                run.update(status='dry_run_complete', optimizer_steps=0, model_trained=False, elapsed_s=time.perf_counter()-begin)
                write_json(output / 'run_manifest.json', run)
                write_json(output / 'progress.json', dict(progress, stage='dry_run_complete', step=0, eta_s=None))
            world.barrier()
            return run if world.rank == 0 else dict(status='dry_run_complete', optimizer_steps=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        sampler = EpisodeUniformSampler(training, args.seed + world.rank * 100003)
        best = None; best_step = None; report = None; losses_history = []; dropped_count = 0; rejected_count = 0
        precision_best = {}; best_by_metric = {}
        loop = time.perf_counter()
        stream = (output / 'metrics.jsonl').open('x') if world.rank == 0 else None
        try:
            for step in range(1, args.steps + 1):
                learning_rate = learning_rate_for_step(step, plan['learning_rate_schedule'])
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate
                model.train(); optimizer.zero_grad(set_to_none=True); batch = []; terms_batch = []
                for index in sampler.next(args.batch_size):
                    sample = training[int(index)]
                    features, dropped, rejected = dropout_features(sample['features'], sample['positions'], rng, args.detection_dropout, config)
                    dropped_count += int(dropped); rejected_count += int(rejected)
                    loss, terms = model.training_loss(features, sample['positions'], formation_index=sample.get('formation_index'))
                    if not terms.get('supervision_usable', False):
                        raise ValueError('Training update has no accepted supervised member')
                    batch.append(loss); terms_batch.append(terms)
                loss = torch.stack(batch).mean() * weight
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite metre-depth objective')
                loss.backward(); synchronize_group_gradients(model, world)
                gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.))
                optimizer.step()
                model.trained = True
                config.update(depth_model_trained=True, completed_optimizer_steps=step)
                model.config.update(depth_model_trained=True, completed_optimizer_steps=step)
                names = ('initial_depth_huber', 'final_depth_huber', 'final_depth_mse', 'matched_rmse_m')
                values = world.sum([_scalar(loss), *[np.mean([_scalar(t[name]) for t in terms_batch]) for name in names]]) / world.size
                counters = world.sum([sum(int(t.get(name, 0)) for t in terms_batch) for name in
                    ('matched_training_count', 'unmatched_detection_count', 'unmatched_label_count')])
                elapsed = time.perf_counter() - loop; rate = step / max(elapsed, 1e-9)
                row = dict(step=step, loss=float(values[0]), **{name:float(value) for name,value in zip(names,values[1:])},
                    learning_rate=learning_rate,
                    gradient_norm_before_clip=gradient, elapsed_s=elapsed, steps_per_second=rate,
                    matched_supervised_members=int(counters[0]), unmatched_detections=int(counters[1]), unmatched_labels=int(counters[2]),
                    model_trained=True, optimizer_steps=step, global_group_updates=step * args.batch_size * world.size)
                if world.rank == 0:
                    losses_history.append(row['loss']); stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                    progress.update(row, stage='training', eta_s=(args.steps-step)/rate)
                    run.update(status='training', model_trained=True, optimizer_steps=step)
                    write_json(output / 'progress.json', progress); write_json(output / 'run_manifest.json', run)
                write_json(rank_path / 'status.json', dict(stage='training', step=step, local_unique_groups=int(sampler.seen.sum())))
                if step == 1 or step % args.validate_every == 0 or step == args.steps:
                    world.barrier()
                    error = None
                    if world.rank == 0:
                        try:
                            if checkpoint.source_hashes() != source:
                                raise ValueError('Depth training source changed during the run')
                            write_json(output / 'progress.json', dict(progress, stage='validating', eta_s=None))
                            report = evaluation.evaluate(model, validation, vmanifest, config)
                            report.update(step=step, selection_key=list(selection_key(report)))
                            with (output / 'prior_validation.jsonl').open('a') as validation_stream:
                                validation_stream.write(json.dumps(report, allow_nan=False)+'\n')
                            if best is None or selection_key(report) > selection_key(best):
                                best, best_step = report, step
                                write_json(output / 'prior_validation_best.json', report)
                                checkpoint.save_checkpoint(checkpoint.make_checkpoint(model, step, validation_report=report,
                                    selection='primary_5m_then_negative_rmse_then_primary_whole_scene', **lineage), output / 'depth.best.pt')
                            for metric, key in precision_selection_keys(report).items():
                                if metric not in precision_best or key > precision_best[metric]:
                                    precision_best[metric] = key
                                    stem = 'rmse' if metric == 'rmse' else '1m'
                                    selection = ('minimum_matched_rmse_then_all_truth_1m_coverage' if metric == 'rmse'
                                                 else 'maximum_all_truth_1m_coverage_then_matched_rmse')
                                    filename = f'depth.{stem}.best.pt'
                                    write_json(output / f'prior_validation_{stem}_best.json', report)
                                    checkpoint.save_checkpoint(checkpoint.make_checkpoint(model, step, validation_report=report,
                                        selection=selection, **lineage), output / filename)
                                    best_by_metric[metric] = dict(step=step, checkpoint=filename, selection=selection,
                                        primary_member_rmse_m=report['summary']['primary_member_rmse_m'],
                                        primary_member_success_ratio_1m=report['summary']['primary_member_success_ratio_1m'],
                                        primary_member_success_ratio_5m=report['summary']['primary_member_success_ratio_5m'])
                            run.update(best_step=best_by_metric['rmse']['step'], coverage_best_step=best_step,
                                       best_by_metric=deepcopy(best_by_metric))
                            progress.update(best_step=best_by_metric['rmse']['step'], coverage_best_step=best_step,
                                            best_by_metric=deepcopy(best_by_metric))
                            write_json(output / 'run_manifest.json', run)
                            checkpoint.save_checkpoint(checkpoint.make_checkpoint(model, step, validation_report=report, **lineage), output / 'depth.pt')
                        except Exception as exc:
                            error = f'{type(exc).__name__}: {exc}'
                    _collective_error(world, error, 'validation/checkpoint')
        finally:
            if stream is not None:
                stream.close()
        difference = _parameter_difference(model, world)
        visited = [dict(episode_id=sample['episode_id'], frame_idx=sample['frame_idx'])
                   for sample,seen in zip(training,sampler.seen) if seen]
        sampling = dict(rank=world.rank, prepared_groups=len(training), actual_unique_groups=len(visited),
            local_group_updates=sampler.draws, visited_frames=visited, detection_dropout_updates=dropped_count,
            rejected_dropout_updates=rejected_count, episode_objective_rank_weight=float(weight))
        write_json(rank_path / 'sampling.json', sampling)
        ranks = world.gather(sampling)
        if args.frames_per_episode == 0 and world.sum([int(not sampler.seen.all())])[0]:
            raise ValueError('Full-frame training did not visit every prepared usable frame')
        if difference != 0:
            raise ValueError('Distributed depth parameters diverged')
        if world.rank == 0:
            unique = sum(row['actual_unique_groups'] for row in ranks)
            # Last step was already evaluated; avoid a second full validation.
            write_json(output / 'prior_validation_final.json', report)
            run.update(status='complete', steps=args.steps, optimizer_steps=args.steps,
                best_step=best_by_metric['rmse']['step'], coverage_best_step=best_step,
                best_by_metric=best_by_metric, final_learning_rate=learning_rate,
                model_trained=True, elapsed_s=time.perf_counter()-begin, training_loop_seconds=time.perf_counter()-loop,
                actual_unique_training_groups=unique, all_usable_training_frames_used=unique==run['prepared_training_groups'],
                actual_global_group_updates=sum(row['local_group_updates'] for row in ranks), rank_parameter_max_difference=difference,
                detection_dropout_updates=sum(row['detection_dropout_updates'] for row in ranks),
                rejected_dropout_updates=sum(row['rejected_dropout_updates'] for row in ranks),
                loss_first_25_mean=float(np.mean(losses_history[:25])), loss_last_25_mean=float(np.mean(losses_history[-25:])))
            write_json(output / 'sampling.json', dict(ranks=ranks, actual_unique_groups=unique, all_usable_frames_used=run['all_usable_training_frames_used']))
            write_json(output / 'configuration.json', config)
            write_json(output / 'run_manifest.json', run)
            write_json(output / 'progress.json', dict(progress, stage='complete', eta_s=0.,
                best_step=best_by_metric['rmse']['step'], coverage_best_step=best_step,
                actual_unique_training_groups=unique, rank_parameter_max_difference=difference))
        world.barrier()
        return run if world.rank == 0 else dict(status='complete', steps=args.steps)
    except BaseException as exc:
        if claimed and world.rank == 0:
            failure = dict(stage='failed', error=f'{type(exc).__name__}: {exc}', updated_unix=time.time())
            write_json(output / 'progress.json', failure)
            if run is not None:
                run.update(status='failed', error=failure['error'])
                write_json(output / 'run_manifest.json', run)
        raise
    finally:
        if lock is not None:
            lock.close()
        world.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT / 'results/precision_study_20261122/configs/joint_volume.json')
    parser.add_argument('--output', type=Path, default=PROJECT / 'results/precision_study_20261122/training/joint_volume')
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--validate-every', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--detection-dropout', type=float, default=.2)
    parser.add_argument('--seed', type=int, default=20261118)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    for name in ('max-episodes', 'frames-per-episode', 'validation-episodes', 'validation-frames'):
        parser.add_argument('--' + name, type=int, default=0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = train(args)
    if not args.plan_only and int(os.environ.get('RANK', '0')) == 0:
        print(json.dumps(dict(status=result['status'], optimizer_steps=result.get('optimizer_steps', 0)), ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
