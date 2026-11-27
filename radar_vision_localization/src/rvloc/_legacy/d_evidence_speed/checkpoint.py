"""Auditable execution snapshots and unchanged D-evidence inference exports.

The inference format is intentionally the existing format.  Its original source
identity describes that model/data contract; the additional execution identity
describes the batched implementation that actually optimized the weights.
Only ``fast_resume.pt`` is a resumable training checkpoint.
"""
from __future__ import annotations

from copy import deepcopy
import os
import platform
from pathlib import Path
import random

import numpy as np
import torch

from d_evidence_study import checkpoint as original
from rvwm.data import sha256
from rvwm.group_training import atomic_torch_save

ROOT = Path(__file__).resolve().parents[1]
RESUME_KIND = 'd_evidence_batched_training_resume_v1'
RUNTIME_MUTABLE = {'device', 'depth_checkpoint', 'depth_model_trained', 'completed_optimizer_steps'}


def source_hashes():
    """Keep execution and model-contract identities separate, never relabel them."""
    return {f'd_evidence_speed/{name}': sha256(ROOT / 'd_evidence_speed' / name)
            for name in ('model.py', 'data.py', 'train.py', 'checkpoint.py')}


def runtime_identity():
    return dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,
                cuda_rng_capture_policy='explicit per-rank owned device only; unused visible devices untouched')


def stable_config(config):
    return {key: deepcopy(value) for key, value in config.items() if key not in RUNTIME_MUTABLE}


def _owned_cuda_identity(device):
    """Inspect only an explicitly owned CUDA device; CPU never probes CUDA."""
    requested = torch.device(device)
    if requested.type != 'cuda':
        if requested.type != 'cpu':
            raise ValueError('Training snapshots only support CPU or one owned CUDA device')
        return None
    current = torch.cuda.current_device()
    owned = current if requested.index is None else requested.index
    if current != owned:
        raise ValueError('The current CUDA device is not the explicit rank-owned device')
    return dict(owned_device_index=owned, visible_device_count=torch.cuda.device_count(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                policy='one CUDA device per rank; no other device generator is used')


def capture_rank_state(sampler, augmentation_rng, *, rank=0, counters=None, device='cpu'):
    """Capture AFTER an optimizer/validation boundary, before the next draw.

    Sampling and detection dropout use independent NumPy generators.  Global
    NumPy/Python states and every generator used on this rank's explicitly
    owned device are retained. Unused visible CUDA devices are never touched.
    CPU diagnostics do not query CUDA, even on hosts where it is available.
    """
    owned = _owned_cuda_identity(device)
    cuda = None if owned is None else dict(runtime_identity=owned,
        generator_state=torch.cuda.get_rng_state(owned['owned_device_index']).cpu().clone())
    return dict(rank=int(rank), sampler=deepcopy(sampler.state_dict()),
                augmentation_generator=deepcopy(augmentation_rng.bit_generator.state),
                numpy_global=np.random.get_state(), python_random=random.getstate(),
                torch_cpu=torch.get_rng_state().clone(),
                cuda_owned_rng=cuda,
                counters=deepcopy(counters or {}))


def restore_rank_state(state, sampler, augmentation_rng, *, rank=0, device='cpu'):
    if state.get('rank') != int(rank):
        raise ValueError('Resume state belongs to a different distributed rank')
    owned = _owned_cuda_identity(device)
    cuda = state.get('cuda_owned_rng')
    if owned is None:
        if cuda is not None:
            raise ValueError('A CUDA rank snapshot cannot resume CPU training exactly')
    elif not isinstance(cuda, dict) or cuda.get('runtime_identity') != owned:
        raise ValueError('Resume rank-owned CUDA device/visible mapping differs')
    sampler.load_state_dict(state['sampler'])
    augmentation_rng.bit_generator.state = deepcopy(state['augmentation_generator'])
    np.random.set_state(state['numpy_global'])
    random.setstate(state['python_random'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if owned is not None:
        torch.cuda.set_rng_state(cuda['generator_state'], owned['owned_device_index'])
    return deepcopy(state.get('counters', {}))


def make_checkpoint(model, steps, *, execution_adapter_source_sha256=None, **metadata):
    """An old-loader-compatible inference export with truthful execution metadata."""
    return original.make_checkpoint(model, steps,
        execution_adapter_source_sha256=execution_adapter_source_sha256 or source_hashes(),
        execution_route='d_evidence_speed.train: resident FP32 padded minibatches',
        **metadata)


save_checkpoint = original.save_checkpoint


def make_resume(model, optimizer, step, *, schedule, ranks, config, training_identity,
                progress_state, inference_checkpoints, world_size):
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError('A complete optimizer boundary is required for a resume snapshot')
    if len(ranks) != int(world_size) or [row.get('rank') for row in ranks] != list(range(int(world_size))):
        raise ValueError('A complete ordered snapshot from every rank is required')
    if not model.trained:
        raise ValueError('An untrained model is not a completed optimizer snapshot')
    return dict(kind=RESUME_KIND, steps=step, optimizer_steps=step,
        model_config=deepcopy(config), model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        optimizer_state_dict=deepcopy(optimizer.state_dict()), schedule=deepcopy(schedule),
        scheduler_state=dict(completed_steps=step, next_step=step + 1,
            last_learning_rates=[group['lr'] for group in optimizer.param_groups],
            implementation='stateless original learning_rate_for_step; apply before numbered update'),
        world_size=int(world_size), rank_states=deepcopy(ranks),
        training_identity=deepcopy(training_identity), progress_state=deepcopy(progress_state),
        inference_checkpoints=deepcopy(inference_checkpoints),
        execution_adapter_source_sha256=source_hashes(), model_contract_source_sha256=original.source_hashes(),
        runtime_identity=runtime_identity(), model_trained=True)


def save_resume(value, path):
    if value.get('kind') != RESUME_KIND:
        raise ValueError('Not a complete batched training snapshot')
    atomic_torch_save(value, path)


def load_resume(path, *, config=None, schedule=None, training_identity=None, world_size=None,
                check_sources=True):
    value = torch.load(Path(path), map_location='cpu', weights_only=False)
    required = {'model_state_dict', 'optimizer_state_dict', 'rank_states', 'schedule', 'scheduler_state',
                'progress_state', 'training_identity', 'inference_checkpoints', 'runtime_identity'}
    if (value.get('kind') != RESUME_KIND or not required.issubset(value)
            or value.get('model_trained') is not True
            or not isinstance(value.get('steps'), int) or isinstance(value.get('steps'), bool)
            or value['steps'] < 1 or value.get('optimizer_steps') != value['steps']):
        raise ValueError('Require fast_resume.pt with optimizer, RNG and sampler state; inference weights cannot resume')
    if check_sources and (value.get('execution_adapter_source_sha256') != source_hashes()
            or value.get('model_contract_source_sha256') != original.source_hashes()):
        raise ValueError('Resume model contract or execution source identity changed')
    if value['runtime_identity'] != runtime_identity():
        raise ValueError('Resume requires the same numerical runtime/backend settings')
    if config is not None and stable_config(config) != stable_config(value['model_config']):
        raise ValueError('Resume configuration changed')
    if schedule is not None and schedule != value['schedule']:
        raise ValueError('Resume learning-rate schedule/budget changed')
    scheduler = value['scheduler_state']
    if (scheduler.get('completed_steps') != value['steps'] or scheduler.get('next_step') != value['steps'] + 1
            or scheduler.get('last_learning_rates') != [group['lr']
                for group in value['optimizer_state_dict']['param_groups']]):
        raise ValueError('Resume scheduler and AdamW learning-rate boundary differ')
    if training_identity is not None and training_identity != value['training_identity']:
        raise ValueError('Resume data inventory, batch, rank partition or objective identity changed')
    if world_size is not None and int(world_size) != value['world_size']:
        raise ValueError('Resume distributed world size changed')
    if ([row.get('rank') for row in value['rank_states']] != list(range(value['world_size']))):
        raise ValueError('Resume rank state is incomplete')
    if any('cuda_owned_rng' not in row for row in value['rank_states']):
        raise ValueError('Resume rank-owned device RNG identity is missing')
    return value


def restore_training(value, model, optimizer, sampler, augmentation_rng, *, rank=0, device='cpu'):
    model.load_state_dict(value['model_state_dict'], strict=True)
    optimizer.load_state_dict(value['optimizer_state_dict'])
    model.trained = True
    model.config.update(depth_model_trained=True, completed_optimizer_steps=value['steps'])
    counters = restore_rank_state(value['rank_states'][int(rank)], sampler, augmentation_rng, rank=rank, device=device)
    return value['steps'], counters
