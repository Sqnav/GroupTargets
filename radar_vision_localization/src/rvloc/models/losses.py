"""Training interfaces only; labels never enter model forward."""
from __future__ import annotations

import torch
from torch.nn import functional as F


def _masked_huber(predicted, target, valid, delta):
    if predicted.shape != target.shape or valid.shape != predicted.shape[:valid.ndim]:
        raise ValueError("Prediction, target and validity must be aligned")
    if not bool(valid.any()):
        return predicted.sum()*0.
    selected = target[valid]
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("Valid supervision must be finite")
    return F.huber_loss(predicted[valid], selected, delta=delta)


def localization_loss(output, targets, *, huber_delta=1., visual_prior_weight=.1,
                      mse_weight=.05, prior_weight=.1, rollout_weight=.1, latent_weight=.01):
    """Final depth and a proper visual-prior likelihood; no stage losses.

    ``targets`` is detection-aligned training supervision. Optional
    ``rollout_positions`` has shape [members, horizons, 3] in fixed-station
    metres and ``rollout_valid`` masks missing future labels/associations.
    The caller constructs future targets only for loss computation.
    """
    if output.get("aux", {}).get("architecture") != "single_query_depth_v3":
        raise ValueError("This loss requires the single_query_depth_v3 single-pass architecture")
    if "initial_depths" in output or "initial_positions" in output:
        raise ValueError("Direct depth must not contain an initial/refined prediction pair")
    depth = output["depths"]
    target = torch.as_tensor(targets.get("depths", targets.get("depths_m")), device=depth.device, dtype=depth.dtype)
    if target.shape != depth.shape:
        raise ValueError("Depth targets must match the directly regressed members")
    valid = output["valid_mask"].clone()
    if "valid_mask" in targets:
        valid &= torch.as_tensor(targets["valid_mask"], device=depth.device, dtype=torch.bool)
    mean = output['visual_depth_mean_m']
    std = output['visual_depth_std_m']
    if mean.shape != depth.shape or std.shape != depth.shape:
        raise ValueError('Visual depth mean/std must align with the current members')
    if not torch.isfinite(torch.as_tensor(visual_prior_weight)) or visual_prior_weight <= 0:
        raise ValueError('A positive visual-prior likelihood weight is required to learn uncertainty')
    if bool(valid.any()) and (not bool(torch.isfinite(mean[valid]).all()) or
                             not bool(torch.isfinite(std[valid]).all()) or bool((std[valid] <= 0).any())):
        raise ValueError('Visual prior must have finite mean and positive finite metre std')
    terms = {"depth_huber": _masked_huber(depth, target, valid, huber_delta)}
    terms["depth_mse"] = F.mse_loss(depth[valid], target[valid]) if bool(valid.any()) else depth.sum()*0.
    terms['visual_prior_nll'] = F.gaussian_nll_loss(mean[valid],target[valid],std[valid].square(),full=True) if bool(valid.any()) else depth.sum()*0.
    total = terms["depth_huber"]+mse_weight*terms["depth_mse"]+visual_prior_weight*terms['visual_prior_nll']
    aux = output.get("aux", {})
    if "prior_latent" in aux:
        active = aux["prior_available"] & valid
        terms["latent_dynamics"] = _masked_huber(aux["prior_latent"], aux["posterior_latent"].detach(), active, 1.)
        total = total+latent_weight*terms["latent_dynamics"]
        if "member_positions" in targets:
            position = torch.as_tensor(targets["member_positions"], device=depth.device, dtype=depth.dtype)
            terms["prior_position_huber"] = _masked_huber(aux["prior_positions"], position, active, huber_delta)
            total = total+prior_weight*terms["prior_position_huber"]
        if "rollout_positions" in targets:
            rollout = aux["rollout_positions"]
            future = torch.as_tensor(targets["rollout_positions"], device=depth.device, dtype=depth.dtype)
            mask = torch.as_tensor(targets.get("rollout_valid", valid[:, None].expand(rollout.shape[:2])),
                                   device=depth.device, dtype=torch.bool)
            mask = mask & valid[:, None]
            terms["rollout_position_huber"] = _masked_huber(rollout, future, mask, huber_delta)
            total = total+rollout_weight*terms["rollout_position_huber"]
    terms["total"] = total
    return total, terms
