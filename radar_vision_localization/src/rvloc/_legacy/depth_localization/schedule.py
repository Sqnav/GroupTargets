"""Deterministic optimizer-step learning rates, without scheduler state or resume.

The returned rate is applied before its numbered optimizer update.  The final
update reaches the configured minimum exactly; validation does not advance it.
"""
from __future__ import annotations

import math


def make_schedule(config, peak_learning_rate, total_steps):
    mode = config.get('depth_lr_schedule', 'constant')
    if mode not in ('constant', 'warmup_cosine'):
        raise ValueError('depth_lr_schedule must be constant or warmup_cosine')
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < 1:
        raise ValueError('Learning-rate total steps must be a positive integer')
    if (isinstance(peak_learning_rate, bool) or not math.isfinite(peak_learning_rate)
            or peak_learning_rate <= 0):
        raise ValueError('Peak learning rate must be positive and finite')
    warmup = config.get('depth_warmup_steps', 0)
    ratio = config.get('depth_min_lr_ratio', 1. if mode == 'constant' else .1)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError('depth_warmup_steps must be a nonnegative integer')
    if isinstance(ratio, bool) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('depth_min_lr_ratio must be finite and in (0, 1]')
    if mode == 'constant' and (warmup != 0 or ratio != 1.):
        raise ValueError('Constant learning rate requires zero warmup and minimum ratio one')
    if mode == 'warmup_cosine' and warmup >= total_steps:
        raise ValueError('Warmup steps must be shorter than the requested optimizer budget')
    return dict(mode=mode, peak_learning_rate=float(peak_learning_rate),
                min_learning_rate=float(peak_learning_rate * ratio), min_lr_ratio=float(ratio),
                warmup_steps=warmup, total_steps=total_steps,
                step_definition='one based optimizer update; applied before optimizer.step',
                checkpoint_resume=False)


def learning_rate_for_step(step, schedule):
    total = schedule['total_steps']
    if isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= total:
        raise ValueError('Learning-rate step must be in the optimizer update budget')
    peak, minimum = schedule['peak_learning_rate'], schedule['min_learning_rate']
    if schedule['mode'] == 'constant':
        return peak
    warmup = schedule['warmup_steps']
    if warmup and step <= warmup:
        return peak * step / warmup
    if total == 1:
        return minimum
    # With zero warmup the first update uses the peak. With warmup, the last
    # warmup update uses the peak and the next update begins the decay.
    decay_start = warmup if warmup else 1
    if step == total:
        return minimum
    if step == decay_start:
        return peak
    fraction = (step - decay_start) / (total - decay_start)
    return minimum + .5 * (peak - minimum) * (1. + math.cos(math.pi * fraction))
