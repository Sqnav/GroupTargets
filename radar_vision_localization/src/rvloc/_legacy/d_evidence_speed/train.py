"""True minibatch execution of the unchanged D-evidence training contract.

This is an execution adapter, not a new model or a continuation from inference
weights.  Formal runs require an equivalence receipt, CUDA, the registered
three-rank budget, and the completed predecessor experiments.  The original
configuration and inference checkpoint format remain unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

from d_evidence_study import train as original_train
from d_evidence_study import checkpoint as original_checkpoint
from d_evidence_study.data import load_samples, validate_preparation
from depth_localization.evaluation import selection_key
from depth_localization.schedule import learning_rate_for_step
from rvwm.data import sha256
from rvwm.group_training import (EpisodeUniformSampler, TrainingWorld,
    _parameter_difference, sampling_coverage_plan, synchronize_group_gradients)
from rvwm.partitions import EPISODE_SAMPLING

from . import checkpoint

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = original_train.EXPERIMENT
ARMS = original_train.ARMS
SCHEMA = original_train.SCHEMA
write_json = original_train.write_json
METRICS = ('initial_depth_huber', 'final_depth_huber', 'final_depth_mse', 'matched_rmse_m')
COUNTERS = ('matched_training_count', 'unmatched_detection_count', 'unmatched_label_count')


def verify_equivalence(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if value.get('status') not in ('passed', 'complete'):
        raise ValueError('Batched execution equivalence checks have not passed')
    registered = value.get('source_sha256', value.get('execution_adapter_source_sha256', {}))
    if not isinstance(registered, dict) or any(registered.get(key) != digest
            for key, digest in checkpoint.source_hashes().items()):
        raise ValueError('Equivalence receipt does not bind the actual execution sources')
    if value.get('model_state_dict_compatible') is not True or value.get('same_training_contract') is not True:
        raise ValueError('Need verified state-dict compatibility and unchanged training contract')
    return dict(path=str(path), sha256=sha256(path))


def _manifest_value(value, arm):
    value = dict(value, d_evidence_schema=SCHEMA, d_evidence_arm=arm, input_frames=1,
        detector_frozen=True, training_from_scratch=True,
        execution_route='d_evidence_speed.train', execution_adapter_source_sha256=checkpoint.source_hashes())
    if 'architecture_schema' in value:
        value['base_architecture_schema'] = 'single_frame_geometry_guided_depth_v2'
        value['architecture_schema'] = SCHEMA
    return value


def write_manifest(path, value, arm):
    write_json(path, _manifest_value(value, arm))


def reduce_metrics(world, loss, terms, gradient_norm):
    """One aggregate collective and one device-to-host metrics transfer per step.

    Gradient averaging is unchanged.  Metric scalars are never transferred for
    individual samples; frame-mean loss terms are supplied by the batched model.
    """
    device = loss.device
    scalar = lambda x: torch.as_tensor(x, device=device, dtype=torch.float64).detach().reshape(())
    packet = torch.stack([scalar(loss), *[scalar(terms[name]) for name in METRICS],
                          *[scalar(terms[name]) for name in COUNTERS], scalar(gradient_norm)])
    if world.size > 1:
        dist.all_reduce(packet, op=dist.ReduceOp.SUM)
    values = packet.cpu().numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError('Nonfinite batched objective or training statistics')
    return values[:5] / world.size, values[5:8], float(values[8] / world.size)


class PredictionReplay(torch.nn.Module):
    """Replay sensor-only batch outputs into the unchanged offline scoring kernel."""
    def __init__(self, scenes, *, trained, config):
        super().__init__()
        self.scenes, self.trained, self.config = scenes, bool(trained), config

    def predict(self, features, allow_untrained_validation=False):
        if not self.trained and not allow_untrained_validation:
            raise ValueError('An untrained validation model needs the explicit diagnostic flag')
        if id(features) not in self.scenes:
            raise ValueError('The frozen evaluator queried an observation outside the batch inventory')
        return self.scenes[id(features)]


def evaluate_batched(model, samples, manifest, config, evaluation, *, batch_size=16,
                     store=None, allow_untrained=False):
    """Batch neural inference, then use the original matcher and full-GT scorer.

    A replay timing is explicitly excluded from the reported inference timing.
    No label reaches ``predict_batch``; targets are not consulted by inference.
    """
    from .data import GPUFeatureStore
    from rvwm.ray_scoring import observation_rejection_reason
    eligible = [sample for sample in samples
                if observation_rejection_reason(sample['features'], config) is None]
    previous = model.training
    model.eval()
    scenes = {}
    if store is None and eligible:
        sensor_only = [{k: v for k, v in sample.items() if k != 'positions'} for sample in eligible]
        store = GPUFeatureStore(sensor_only, config, device=next(model.parameters()).device)
    cuda = next(model.parameters()).is_cuda
    if cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    tick = time.perf_counter()
    try:
        with torch.no_grad():
            for start in range(0, len(eligible), batch_size):
                indices = np.arange(start, min(start + batch_size, len(eligible)))
                packed, _ = store.batch(indices, dropout_probability=0.)
                batch_scenes = model.predict_batch(packed, allow_untrained_validation=allow_untrained)
                if len(batch_scenes) != len(indices):
                    raise ValueError('Batch prediction cardinality differs from the observation inventory')
                scenes.update((id(eligible[int(index)]['features']), scene)
                              for index, scene in zip(indices, batch_scenes))
        if cuda:
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - tick
        replay = PredictionReplay(scenes, trained=model.trained, config=config)
        report = evaluation.evaluate(replay, samples, manifest, config, allow_untrained=allow_untrained)
        summary = report['summary']
        summary['cpu_scene_replay_seconds_per_frame'] = summary.pop('inference_seconds_per_frame', None)
        summary['inference_seconds_per_frame'] = dict(count=len(scenes),
            mean=inference_seconds / max(len(scenes), 1), mae=None, rmse=None,
            p50_abs=None, p95_abs=None, max_abs=None,
            estimand='amortized actual batched inference, not a per-frame latency distribution')
        summary['latency_scope'] = 'resident sensor-only batch inference; excludes feature loading and offline label scoring'
        summary['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated() if cuda else 0
        summary['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved() if cuda else 0
        report['execution_timing'] = dict(
            route='true batched sensor-only inference followed by unchanged CPU scoring',
            predicted_frames=len(scenes), inference_seconds=inference_seconds,
            inference_seconds_per_frame=inference_seconds / max(len(scenes), 1),
            includes_feature_preload=False, excludes_cpu_offline_label_scoring=True,
            scoring_scene_replay_latency_is_not_inference=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else 0)
        return report, store
    finally:
        model.train(previous)


def _inference_artifacts(output):
    result = {}
    for name in ('depth.pt', 'depth.best.pt', 'depth.rmse.best.pt', 'depth.1m.best.pt'):
        path = output / name
        if path.exists():
            result[name] = torch.load(path, map_location='cpu', weights_only=False)
    return result


def _restore_output(snapshot, output):
    """Archive a crashed tail, then restore the committed model-selection boundary."""
    archive = output / 'resume_archive' / f"step_{snapshot['steps']:08d}_{time.time_ns()}"
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('metrics.jsonl', 'prior_validation.jsonl'):
        path = output / name
        if not path.exists():
            continue
        raw = path.read_bytes(); (archive / name).write_bytes(raw)
        keep = []
        for line in raw.decode().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row.get('step'), int) and row['step'] <= snapshot['steps']:
                keep.append(json.dumps(row, allow_nan=False) + '\n')
        path.write_text(''.join(keep))
    for name in ('run_manifest.json', 'progress.json', 'training_route_receipt.json'):
        path = output / name
        if path.exists():
            (archive / name).write_bytes(path.read_bytes())
    for name, value in snapshot['inference_checkpoints'].items():
        checkpoint.save_checkpoint(value, output / name)
    state = snapshot['progress_state']
    for name, value in state.get('validation_best_reports', {}).items():
        write_json(output / name, value)
    write_json(archive / 'resume_receipt.json', dict(step=snapshot['steps'],
        snapshot_sha256=sha256(output / 'fast_resume.pt'), tail_policy='archived; exact committed boundary restored'))
    return str(archive)


def train(args, *, trainer, config, preparation, equivalence, deployment, binding):
    """The real batched loop; all inputs/targets remain separate and resident."""
    from .data import GPUFeatureStore
    from .model import FastDepthLocalizer
    torch.set_num_threads(args.threads)
    world = TrainingWorld(args.device)
    output = Path(args.output).resolve()
    started, begin = time.time(), time.perf_counter()
    run = None; lock = None; route = None; claimed = False
    source = checkpoint.source_hashes()
    original_source = original_checkpoint.source_hashes()
    evaluation = importlib.import_module(trainer.__package__ + '.evaluation')
    try:
        failure = None
        if world.rank == 0:
            try:
                output.mkdir(parents=True, exist_ok=True)
                lock = (output / 'training.lock').open('a+')
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if args.resume:
                    if Path(args.resume).resolve() != output / 'fast_resume.pt':
                        raise ValueError('Resume only the committed snapshot in this exact output directory')
                elif any((output / name).exists() for name in (*trainer.PROTECTED, 'fast_resume.pt')):
                    raise FileExistsError('Existing training artifacts are protected: ' + str(output))
            except Exception as exc:
                failure = f'{type(exc).__name__}: {exc}'
        failure = world.broadcast(failure)
        if failure:
            raise FileExistsError(failure)
        claimed = True
        # The original plan and data contract see the original options, not fast execution controls.
        original_args = deepcopy(args); original_args.resume = None
        plan = trainer._load_stage(world, lambda: trainer.make_plan(config, original_args, world.size), 'dataset planning')
        config = deepcopy(config)
        config.update(device=world.device, depth_checkpoint=str(output / 'depth.rmse.best.pt'),
            depth_model_trained=False, completed_optimizer_steps=0, training_sampling_policy=EPISODE_SAMPLING)
        np.random.seed(args.seed + world.rank * 100003)
        torch.manual_seed(args.seed + world.rank * 100003)
        rng = np.random.default_rng(args.seed + world.rank * 100003)
        progress = dict(stage='preparing', step=0, requested_steps=args.steps, started_unix=started,
            world_size=world.size, global_batch_size=world.size * args.batch_size, architecture_schema=SCHEMA,
            model_trained=False, optimizer_steps=0)
        if world.rank == 0 and not args.resume:
            write_manifest(output / 'progress.json', progress, args.arm)
            write_json(output / 'configuration.json', config)
            write_manifest(output / 'training_plan.json', plan, args.arm)
        rank_path = output / 'ranks' / f'rank_{world.rank}'
        local_ids = plan['rank_partitions'][world.rank]['episodes']
        training, manifest, frontend = trainer._load_stage(world, lambda: load_samples(
            config, local_ids, 'train', count=plan['train_frame_limit'], include_labels=True), 'training features')
        validation, vmanifest, vfrontend = trainer._load_stage(world, lambda: load_samples(
            config, plan['validation_episodes'], 'val', count=plan['validation_frame_limit'], include_labels=True)
            if world.rank == 0 else ([], [], None), 'validation features')
        trainer._collective_error(world, 'Nonempty usable training and validation features required'
            if not training or (world.rank == 0 and not validation) else None, 'usable features')
        coverage = sampling_coverage_plan(training, EPISODE_SAMPLING, args.batch_size)
        insufficient = not args.dry_run and args.frames_per_episode == 0 and args.steps < coverage['full_frame_coverage_steps_upper_bound']
        if world.sum([int(insufficient)])[0]:
            raise ValueError('Requested training budget cannot visit every selected frame')
        usable_episodes = sorted({row['episode_id'] for row in training})
        weight = world.size * len(usable_episodes) / world.sum([len(usable_episodes)])[0]
        partition = dict(rank=world.rank, episodes=local_ids, usable_episodes=usable_episodes,
            groups=len(training), frames=manifest, expected_frames=plan['rank_partitions'][world.rank]['expected_frames'],
            episode_objective_rank_weight=float(weight), coverage=coverage)
        partitions = world.gather(partition)
        write_json(rank_path / 'train_features.json', dict(frames=manifest, frontend=frontend, source_sha256=original_source))
        failure = None
        if world.rank == 0:
            try:
                inventory = [row for part in partitions for row in part['frames']]
                keys = [(row['episode_id'], row['frame_idx']) for row in inventory]
                if (any(len(part['frames']) != part['expected_frames'] for part in partitions)
                        or len(set(keys)) != len(keys) or len(inventory) != plan['expected_training_frames']
                        or len(vmanifest) != plan['expected_validation_frames']):
                    raise ValueError('Missing/duplicated full training or validation inventory')
                write_json(output / 'train_inventory.json', dict(frames=inventory, source_sha256=original_source))
                write_json(output / 'val_features.json', dict(frames=vmanifest, frontend=vfrontend, source_sha256=original_source))
                effective = output / 'inputs/configuration.json'
                run = dict(status='preparing', architecture_schema=SCHEMA, configuration=config,
                    train_episodes=plan['train_episodes'], validation_episodes=plan['validation_episodes'],
                    partition=plan['partition'], model_trained=False, optimizer_steps=0, world_size=world.size,
                    per_rank_batch_size=args.batch_size, global_batch_size=args.batch_size * world.size,
                    prepared_training_groups=len(inventory), prepared_validation_groups=len(validation),
                    total_training_selected_frames=len(inventory), total_validation_selected_frames=len(vmanifest),
                    inventory_sha256=trainer._inventory_digest(inventory), validation_inventory_sha256=trainer._inventory_digest(vmanifest),
                    training_configuration=dict(path=str(effective), sha256=sha256(effective)),
                    all_selected_frames_prepared=True, rank_partitions=[{k: v for k, v in part.items() if k != 'frames'} for part in partitions],
                    hyperparameters=plan['hyperparameters'], frontend=frontend, source_sha256=original_source,
                    learning_rate_schedule=plan['learning_rate_schedule'], model_initialization=deepcopy(plan['model_initialization']),
                    training_sampling_policy=EPISODE_SAMPLING, seed=args.seed,
                    model_selection='minimum_matched_rmse_then_all_truth_1m_coverage',
                    coverage_model_selection='primary_5m_then_negative_rmse_then_primary_whole_scene',
                    raw_iq_likelihood=False, posterior_latent=False, hypotheses=1, uncertainty_calibrated=False,
                    precision_radar_mode=config.get('precision_radar_mode', 'ra_re'),
                    execution_adapter_source_sha256=source, equivalence_receipt=equivalence)
            except Exception as exc:
                failure = f'{type(exc).__name__}: {exc}'
        trainer._collective_error(world, failure, 'prepared inventory')
        model = trainer._load_stage(world, lambda: FastDepthLocalizer(config).to(world.device), 'model initialization')
        world.sync_model(model)
        initialization = trainer._prepare_training(model, training, world, config)
        world.sync_model(model)
        if world.rank == 0:
            fairness = initialization['shared_D_initialization']
            run['depth_prior_initialization'] = initialization
            run['model_parameter_count'] = sum(p.numel() for p in model.parameters())
            run['model_initialization'].update(initial_parameter_sha256=trainer._parameter_digest(model),
                shared_parameter_sha256=trainer._shared_parameter_digest(model),
                base_initial_parameter_sha256=fairness['shared_parameter_sha256'],
                base_reference_initial_parameter_sha256=fairness['reference_initial_parameter_sha256'],
                base_initialization_matches_reference=fairness['matches_reference'])
            scope = dict(all_frames_requested=plan['all_frames_requested'], frames_per_episode=args.frames_per_episode,
                max_episodes=args.max_episodes, validation_frames=args.validation_frames, validation_episodes=args.validation_episodes,
                all_training_episodes_selected=plan['all_training_episodes_selected'],
                all_validation_episodes_selected=plan['all_validation_episodes_selected'],
                training_frames_subset=plan['training_frames_subset'], validation_frames_subset=plan['validation_frames_subset'],
                selected_training_frames=run['prepared_training_groups'], selected_validation_frames=run['prepared_validation_groups'],
                requested_steps=args.steps, diagnostic_subset=plan['diagnostic_subset'])
            run['training_scope'] = scope
            lineage = {name: run[name] for name in ('train_episodes', 'validation_episodes', 'frontend', 'partition',
                'training_scope', 'hyperparameters', 'inventory_sha256', 'validation_inventory_sha256',
                'training_configuration', 'depth_prior_initialization', 'learning_rate_schedule', 'model_initialization')}
            lineage['equivalence_receipt'] = equivalence
            if not args.resume:
                write_manifest(output / 'run_manifest.json', run, args.arm)
        else:
            lineage = None
        # Validation has its own sensor-only resident store and never supplies targets to prediction.
        store = trainer._load_stage(world, lambda: GPUFeatureStore(training, config, device=world.device), 'resident training store')
        validation_store = None
        if args.dry_run:
            before = {name: value.detach().clone() for name, value in model.state_dict().items()}
            packed, augmentation = store.batch([0], rng=rng, dropout_probability=0.)
            loss, terms = model.training_loss_batch(packed)
            loss.backward(); synchronize_group_gradients(model, world)
            norm = float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
            if not np.isfinite(norm) or any(not torch.equal(before[name], value) for name, value in model.state_dict().items()):
                raise ValueError('Dry run requires finite gradients and unchanged weights/buffers')
            write_json(rank_path / 'graph_validation.json', dict(loss=float(loss.detach()), gradient_norm=norm,
                model_trained=False, optimizer_steps=0, parameters_unchanged=True, true_minibatch=True))
            if world.rank == 0:
                report, _ = evaluate_batched(model, validation, vmanifest, config, evaluation,
                    batch_size=args.batch_size, allow_untrained=True)
                write_json(output / 'dry_run_validation.json', report)
                run.update(status='dry_run_complete', optimizer_steps=0, elapsed_s=time.perf_counter() - begin)
                write_manifest(output / 'run_manifest.json', run, args.arm)
                write_manifest(output / 'progress.json', dict(progress, stage='dry_run_complete', eta_s=None), args.arm)
            world.barrier()
            return run if world.rank == 0 else dict(status='dry_run_complete', optimizer_steps=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        sampler = EpisodeUniformSampler(training, args.seed + world.rank * 100003)
        training_identity = world.broadcast(dict(configuration_sha256=sha256(output / 'inputs/configuration.json'),
            inventory_sha256=run['inventory_sha256'], validation_inventory_sha256=run['validation_inventory_sha256'],
            hyperparameters=plan['hyperparameters'], rank_partitions=plan['rank_partitions'],
            execution_adapter_source_sha256=source, equivalence_receipt=equivalence) if world.rank == 0 else None)
        start_step = 0; best = None; best_step = None; report = None
        precision_best = {}; best_by_metric = {}; losses_history = []; validation_best_reports = {}
        dropped_count = rejected_count = 0; prior_loop_seconds = 0.
        if args.resume:
            snapshot = trainer._load_stage(world, lambda: checkpoint.load_resume(args.resume, config=config,
                schedule=plan['learning_rate_schedule'], training_identity=training_identity, world_size=world.size), 'resume identity')
            state = snapshot['progress_state']
            if world.rank == 0:
                archive = _restore_output(snapshot, output)
                run = deepcopy(state['run']); progress = deepcopy(state['progress'])
                lineage = deepcopy(state['lineage']); best = state['best']; best_step = state['best_step']
                precision_best = state['precision_best']; best_by_metric = state['best_by_metric']
                losses_history = state['losses_history']; report = state['report']
                validation_best_reports = state['validation_best_reports']
                run['resume_archive'] = archive
            start_step, counters = checkpoint.restore_training(snapshot, model, optimizer, sampler, rng,
                rank=world.rank, device=world.device)
            dropped_count = counters['dropped_count']; rejected_count = counters['rejected_count']
            prior_loop_seconds = float(state['loop_seconds'])
            config.update(depth_model_trained=True, completed_optimizer_steps=start_step)
        loop = time.perf_counter()
        stream = (output / 'metrics.jsonl').open('a' if args.resume else 'x') if world.rank == 0 else None
        try:
            for step in range(start_step + 1, args.steps + 1):
                learning_rate = learning_rate_for_step(step, plan['learning_rate_schedule'])
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate
                model.train(); optimizer.zero_grad(set_to_none=True)
                packed, augmentation = store.batch(sampler.next(args.batch_size), rng=rng,
                    dropout_probability=args.detection_dropout)
                dropped_count += int(augmentation['dropped_count']); rejected_count += int(augmentation['rejected_count'])
                loss, terms = model.training_loss_batch(packed)
                loss = loss * weight
                loss.backward(); synchronize_group_gradients(model, world)
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step(); model.trained = True
                config.update(depth_model_trained=True, completed_optimizer_steps=step)
                model.config.update(depth_model_trained=True, completed_optimizer_steps=step)
                values, counters, gradient = reduce_metrics(world, loss, terms, gradient)
                elapsed = prior_loop_seconds + time.perf_counter() - loop
                rate = step / max(elapsed, 1e-9)
                row = dict(step=step, loss=float(values[0]), **{name: float(value) for name, value in zip(METRICS, values[1:])},
                    learning_rate=learning_rate, gradient_norm_before_clip=gradient, elapsed_s=elapsed,
                    steps_per_second=rate, matched_supervised_members=int(counters[0]),
                    unmatched_detections=int(counters[1]), unmatched_labels=int(counters[2]), model_trained=True,
                    optimizer_steps=step, global_group_updates=step * args.batch_size * world.size,
                    execution_route='true_minibatch_gpu_resident')
                validate = step == 1 or step % args.validate_every == 0 or step == args.steps
                if world.rank == 0:
                    losses_history.append(row['loss']); stream.write(json.dumps(row, allow_nan=False) + '\n')
                    progress.update(row, stage='training', eta_s=(args.steps - step) / rate)
                    run.update(status='training', model_trained=True, optimizer_steps=step)
                    if step % args.progress_every == 0 or validate:
                        stream.flush(); write_manifest(output / 'progress.json', progress, args.arm)
                if step % args.progress_every == 0 or validate:
                    write_json(rank_path / 'status.json', dict(stage='training', step=step, local_unique_groups=int(sampler.seen.sum())))
                if validate:
                    world.barrier(); failure = None
                    if world.rank == 0:
                        try:
                            if checkpoint.source_hashes() != source or original_checkpoint.source_hashes() != original_source:
                                raise ValueError('Model contract or execution sources changed during training')
                            write_manifest(output / 'progress.json', dict(progress, stage='validating', eta_s=None), args.arm)
                            report, validation_store = evaluate_batched(model, validation, vmanifest, config, evaluation,
                                batch_size=args.validation_batch_size, store=validation_store)
                            report.update(step=step, selection_key=list(selection_key(report)))
                            with (output / 'prior_validation.jsonl').open('a') as handle:
                                handle.write(json.dumps(report, allow_nan=False) + '\n')
                            if best is None or selection_key(report) > selection_key(best):
                                best, best_step = report, step
                                validation_best_reports['prior_validation_best.json'] = report
                                write_json(output / 'prior_validation_best.json', report)
                                checkpoint.save_checkpoint(checkpoint.make_checkpoint(model, step, validation_report=report,
                                    selection='primary_5m_then_negative_rmse_then_primary_whole_scene', **lineage), output / 'depth.best.pt')
                            for metric, key in trainer.precision_selection_keys(report).items():
                                if metric not in precision_best or key > precision_best[metric]:
                                    precision_best[metric] = key
                                    stem = 'rmse' if metric == 'rmse' else '1m'
                                    selection = ('minimum_matched_rmse_then_all_truth_1m_coverage' if metric == 'rmse'
                                        else 'maximum_all_truth_1m_coverage_then_matched_rmse')
                                    filename = f'depth.{stem}.best.pt'
                                    validation_best_reports[f'prior_validation_{stem}_best.json'] = report
                                    write_json(output / f'prior_validation_{stem}_best.json', report)
                                    checkpoint.save_checkpoint(checkpoint.make_checkpoint(model, step,
                                        validation_report=report, selection=selection, **lineage), output / filename)
                                    best_by_metric[metric] = dict(step=step, checkpoint=filename, selection=selection,
                                        primary_member_rmse_m=report['summary']['primary_member_rmse_m'],
                                        primary_member_success_ratio_1m=report['summary']['primary_member_success_ratio_1m'],
                                        primary_member_success_ratio_5m=report['summary']['primary_member_success_ratio_5m'])
                            run.update(best_step=best_by_metric['rmse']['step'], coverage_best_step=best_step,
                                best_by_metric=deepcopy(best_by_metric))
                            progress.update(best_step=best_by_metric['rmse']['step'], coverage_best_step=best_step,
                                best_by_metric=deepcopy(best_by_metric))
                            write_manifest(output / 'run_manifest.json', run, args.arm)
                            checkpoint.save_checkpoint(checkpoint.make_checkpoint(model, step,
                                validation_report=report, **lineage), output / 'depth.pt')
                        except Exception as exc:
                            failure = f'{type(exc).__name__}: {exc}'
                    trainer._collective_error(world, failure, 'batched validation/checkpoint')
                if validate or step % args.save_every == 0:
                    rank_states = world.gather(checkpoint.capture_rank_state(sampler, rng, rank=world.rank, device=world.device,
                        counters=dict(dropped_count=dropped_count, rejected_count=rejected_count)))
                    failure = None
                    if world.rank == 0:
                        try:
                            state = dict(run=run, progress=progress, lineage=lineage, best=best, best_step=best_step,
                                precision_best=precision_best, best_by_metric=best_by_metric, losses_history=losses_history,
                                report=report, validation_best_reports=validation_best_reports,
                                loop_seconds=prior_loop_seconds + time.perf_counter() - loop)
                            snapshot = checkpoint.make_resume(model, optimizer, step, schedule=plan['learning_rate_schedule'],
                                ranks=rank_states, config=config, training_identity=training_identity, progress_state=state,
                                inference_checkpoints=_inference_artifacts(output), world_size=world.size)
                            checkpoint.save_resume(snapshot, output / 'fast_resume.pt')
                            write_json(output / 'fast_resume_receipt.json', dict(status='committed', step=step,
                                path=str(output / 'fast_resume.pt'), sha256=sha256(output / 'fast_resume.pt'),
                                world_size=world.size, execution_adapter_source_sha256=source,
                                includes_optimizer_rng_sampler=True))
                        except Exception as exc:
                            failure = f'{type(exc).__name__}: {exc}'
                    trainer._collective_error(world, failure, 'complete resumable boundary')
        finally:
            if stream is not None:
                stream.close()
        difference = _parameter_difference(model, world)
        visited = [dict(episode_id=sample['episode_id'], frame_idx=sample['frame_idx'])
                   for sample, seen in zip(training, sampler.seen) if seen]
        sampling = dict(rank=world.rank, prepared_groups=len(training), actual_unique_groups=len(visited),
            local_group_updates=sampler.draws, visited_frames=visited, detection_dropout_updates=dropped_count,
            rejected_dropout_updates=rejected_count, episode_objective_rank_weight=float(weight))
        write_json(rank_path / 'sampling.json', sampling)
        ranks = world.gather(sampling)
        if args.frames_per_episode == 0 and world.sum([int(not sampler.seen.all())])[0]:
            raise ValueError('Full-frame training failed to visit every prepared usable frame')
        if difference != 0:
            raise ValueError('Distributed model parameters diverged')
        if world.rank == 0:
            unique = sum(item['actual_unique_groups'] for item in ranks)
            write_json(output / 'prior_validation_final.json', report)
            run.update(status='complete', steps=args.steps, optimizer_steps=args.steps,
                best_step=best_by_metric['rmse']['step'], coverage_best_step=best_step,
                best_by_metric=best_by_metric, final_learning_rate=learning_rate_for_step(args.steps, plan['learning_rate_schedule']),
                model_trained=True, elapsed_s=time.perf_counter() - begin,
                training_loop_seconds=prior_loop_seconds + time.perf_counter() - loop,
                actual_unique_training_groups=unique, all_usable_training_frames_used=unique == run['prepared_training_groups'],
                actual_global_group_updates=sum(item['local_group_updates'] for item in ranks),
                rank_parameter_max_difference=difference,
                detection_dropout_updates=sum(item['detection_dropout_updates'] for item in ranks),
                rejected_dropout_updates=sum(item['rejected_dropout_updates'] for item in ranks),
                loss_first_25_mean=float(np.mean(losses_history[:25])), loss_last_25_mean=float(np.mean(losses_history[-25:])))
            write_json(output / 'sampling.json', dict(ranks=ranks, actual_unique_groups=unique,
                all_usable_frames_used=run['all_usable_training_frames_used']))
            write_json(output / 'configuration.json', config)
            write_manifest(output / 'run_manifest.json', run, args.arm)
            write_manifest(output / 'progress.json', dict(progress, stage='complete', eta_s=0.,
                best_step=best_by_metric['rmse']['step'], coverage_best_step=best_step,
                actual_unique_training_groups=unique, rank_parameter_max_difference=difference), args.arm)
        world.barrier()
        return run if world.rank == 0 else dict(status='complete', steps=args.steps, optimizer_steps=args.steps)
    except BaseException as exc:
        if claimed and world.rank == 0:
            failure = dict(stage='failed', error=f'{type(exc).__name__}: {exc}', updated_unix=time.time())
            write_manifest(output / 'progress.json', failure, args.arm)
            if run is not None:
                run.update(status='failed', error=failure['error'])
                write_manifest(output / 'run_manifest.json', run, args.arm)
        raise
    finally:
        if lock is not None:
            lock.close()
        world.close()


def validate_training(output):
    """Strengthen, never bypass, the original final-evaluator training receipt."""
    route = original_train.validate_training(output)
    if route.get('execution_adapter_source_sha256') != checkpoint.source_hashes():
        raise ValueError('Batched execution source identity changed')
    receipt = route['equivalence_receipt']
    if verify_equivalence(receipt['path']) != receipt:
        raise ValueError('The registered mathematical equivalence receipt changed')
    run = json.loads((Path(output) / 'run_manifest.json').read_text())
    resume = checkpoint.load_resume(Path(output) / 'fast_resume.pt', world_size=run['world_size'])
    if resume['steps'] != run['optimizer_steps']:
        raise ValueError('Final full resume snapshot is not at the completed optimizer step')
    for name, value in resume['inference_checkpoints'].items():
        current = torch.load(Path(output) / name, map_location='cpu', weights_only=False)
        if current['steps'] != value['steps'] or any(not torch.equal(value['state_dict'][key], tensor)
                for key, tensor in current['state_dict'].items()):
            raise ValueError('Inference export differs from the committed full snapshot: ' + name)
    return route


def main(argv=None):
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--arm', choices=ARMS, default='roi')
    early.add_argument('--dry-run', action='store_true'); early.add_argument('--plan-only', action='store_true')
    early.add_argument('--preflight', action='store_true')
    selected, _ = early.parse_known_args(argv)
    formal = not (selected.dry_run or selected.plan_only or selected.preflight)
    trainer, description, deployment, binding = original_train.frozen_trainer(selected.arm, formal=formal)
    parser = trainer.build_parser(); parser.description = __doc__
    parser.set_defaults(config=EXPERIMENT / 'configs/roi.json', output=EXPERIMENT / 'training/roi',
        batch_size=16, seed=20261118, device='cuda')
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--equivalence-receipt', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--progress-every', type=int, default=10)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--validation-batch-size', type=int, default=16)
    args = parser.parse_args(argv)
    original_args = deepcopy(args); original_args.resume = None
    config_path = args.config.expanduser().resolve()
    config = original_train.normalized_config(json.loads(config_path.read_text()), args.arm)
    original_train.validate_base_config(config, json.loads(Path(description['config']).read_text()))
    trainer.validate_options(original_args, config)
    if min(args.progress_every, args.save_every, args.validation_batch_size) < 1:
        raise ValueError('Progress, snapshot and validation batch intervals must be positive')
    binding = dict(binding, model_factory='d_evidence_speed.model.FastDepthLocalizer',
        checkpoint_callback='d_evidence_speed.checkpoint', trainer_callback='d_evidence_speed.train.train',
        batch_data_callback='d_evidence_speed.data.GPUFeatureStore',
        gradient_reduction='unchanged mean flattened gradients; one combined metrics collective',
        validation_callback='sensor-only predict_batch then unchanged frozen evaluation.evaluate',
        original_training_loop_used=False, numerical_precision='FP32; no AMP or compile')
    plan = trainer.make_plan(config, original_args)
    if args.preflight or args.plan_only:
        print(json.dumps(dict(status='preflight_passed' if args.preflight else 'planned', plan=plan,
            optimizer_started=False, sensor_payloads_read=False, original_configuration_unchanged=True,
            execution_adapter_source_sha256=checkpoint.source_hashes(), callbacks=binding), ensure_ascii=False, indent=2), flush=True)
        return 0
    if formal:
        expected = dict(steps=20000, batch_size=16, seed=20261118, learning_rate=3e-4,
            detection_dropout=.2, validate_every=1000, max_episodes=0, frames_per_episode=0,
            validation_episodes=0, validation_frames=0)
        if (not args.device.startswith('cuda') or int(os.environ.get('WORLD_SIZE', '1')) != 3
                or any(getattr(args, key) != value for key, value in expected.items())):
            raise ValueError('Formal execution retains CUDA, three ranks, global48 and the exact registered 20k contract')
        if args.arm != 'roi':
            raise ValueError('This registered execution amendment only starts the previously unstarted ROI arm')
        from d_evidence_study.run import registered_dependencies, dependencies_ready
        if not dependencies_ready(registered_dependencies()):
            raise RuntimeError('Registered BW/temporal predecessors still own the GPUs')
        aligned = original_train.validate_training(EXPERIMENT / 'training/aligned')
        if aligned.get('optimizer_steps') != 20000:
            raise RuntimeError('The original aligned stage must finish its registered 20k updates first')
        if args.equivalence_receipt is None:
            raise ValueError('Formal minibatch execution requires a passed bound equivalence receipt')
        equivalence = verify_equivalence(args.equivalence_receipt)
        prepared = Path(config['d_evidence_preparation_root']).resolve()
        validate_preparation(config, prepared, verify_cache=False)
        preparation = dict(path=str(prepared / 'preparation_receipt.json'), sha256=sha256(prepared / 'preparation_receipt.json'))
    else:
        preparation = equivalence = None
    config.update(d_evidence_training_source_sha256=original_checkpoint.source_hashes(),
        d_evidence_frozen_source_root=str(description['source_root']),
        d_evidence_original_configuration_sha256=deployment['configuration_sha256'])
    rank = int(os.environ.get('RANK', '0')); output = args.output.expanduser().resolve()
    effective = output / 'inputs/configuration.json'
    ready = output / 'inputs/fast_launch_ready.json'
    token = os.environ.get('TORCHELASTIC_RUN_ID', 'single_process'); lock = None
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True); lock = (output / 'route.lock').open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if json.loads(effective.read_text()) != config:
                raise ValueError('Resume cannot change the original effective configuration')
        elif any((output / name).exists() for name in (*trainer.PROTECTED, 'fast_resume.pt')):
            raise FileExistsError('Existing training artifacts are protected: ' + str(output))
        else:
            write_json(effective, config)
        write_json(ready, dict(token=token, configuration_sha256=sha256(effective)))
    else:
        deadline = time.monotonic() + 120
        while True:
            try:
                value = json.loads(ready.read_text())
                if value['token'] == token and value['configuration_sha256'] == sha256(effective):
                    break
            except (OSError, ValueError, KeyError):
                pass
            if time.monotonic() > deadline:
                raise TimeoutError('Rank zero did not publish this execution configuration')
            time.sleep(.1)
    if json.loads(effective.read_text()) != config:
        raise ValueError('Distributed rank configurations disagree')
    route_path = output / 'training_route_receipt.json'
    route = dict(schema='d_evidence_training_route_v1', status='starting', arm=args.arm,
        original_configuration=dict(path=str(config_path), sha256=sha256(config_path)),
        effective_configuration=dict(path=str(effective), sha256=sha256(effective)),
        evidence_source_sha256=original_checkpoint.source_hashes(), deployment=deployment,
        execution_adapter_source_sha256=checkpoint.source_hashes(), equivalence_receipt=equivalence,
        callbacks=binding, preparation=preparation, input_frames=1, detector_frozen=True,
        fusion_training_from_scratch=True, initial_fusion_checkpoint=None,
        baseline_reference=str(original_train.BASELINE), world_size=int(os.environ.get('WORLD_SIZE', '1')),
        started_unix=time.time(), resumed_from=str(args.resume.resolve()) if args.resume else None,
        command=[sys.executable, '-m', 'd_evidence_speed.train', *(argv if argv is not None else sys.argv[1:])])
    if rank == 0:
        if args.resume and route_path.exists():
            old_route = json.loads(route_path.read_text())
            route['original_started_unix'] = old_route.get('original_started_unix', old_route['started_unix'])
            write_json(output / 'resume_launch_history' / f'{time.time_ns()}.json', old_route)
        write_json(route_path, route)
    args.config = effective
    try:
        result = train(args, trainer=trainer, config=config, preparation=preparation,
            equivalence=equivalence, deployment=deployment, binding=binding)
    except BaseException as exc:
        if rank == 0:
            write_json(route_path, dict(route, status='failed', error=repr(exc), finished_unix=time.time()))
        raise
    else:
        if rank == 0:
            artifacts = {name: dict(path=str(output / name), sha256=sha256(output / name))
                for name in ('depth.pt', 'depth.best.pt', 'depth.rmse.best.pt', 'depth.1m.best.pt')
                if (output / name).exists()}
            route.update(status=result['status'], optimizer_steps=result.get('optimizer_steps', 0),
                finished_unix=time.time(), checkpoints=artifacts)
            write_json(route_path, route)
            print(json.dumps(dict(status=result['status'], optimizer_steps=result.get('optimizer_steps', 0))), flush=True)
        return 0
    finally:
        if lock is not None:
            lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
