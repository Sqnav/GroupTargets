"""One uncertainty-conditioned physical RA/RE query for all current members.

Candidate positions use only the visual depth mean/std and public rays. This
module reuses the audited geometry and sampler, performs one paired query,
then interacts the candidate evidence before masked pooling. No GT, global
peak assignment, depth clipping or iterative depth update enters this module.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from rvloc.legacy import activate

activate()
from depth_localization.geometry import candidate_geometry, sample_paired_spectra

DEFAULT_MULTIPLIERS = (-3., -2., -1., -.5, 0., .5, 1., 2., 3.)


class SingleRadarQuery(nn.Module):
    """One batched [member,candidate] query followed by one interaction layer.

    ``visual_std`` has metre units and is a learned candidate-spacing signal;
    it is not a calibrated confidence interval. ``weights`` in diagnostics
    are candidate score/pooling weights, not per-head Transformer matrices.
    """
    query_rounds = 1

    def __init__(self, hidden=64, heads=4, candidate_multipliers=None, *, layers=1, z_offsets=None):
        super().__init__()
        if type(hidden) is not int or hidden < 8 or type(heads) is not int or heads < 1 or hidden % heads:
            raise ValueError("Candidate hidden size >=8 must be divisible by positive attention heads")
        if type(layers) is not int or layers != 1:
            raise ValueError("The registered single-query candidate interaction has exactly one layer")
        if candidate_multipliers is not None and z_offsets is not None:
            raise ValueError("Specify candidate_multipliers or z_offsets, not both")
        multipliers = candidate_multipliers if candidate_multipliers is not None else z_offsets
        multipliers = DEFAULT_MULTIPLIERS if multipliers is None else multipliers
        values = torch.as_tensor(multipliers, dtype=torch.float32)
        if (values.ndim != 1 or not len(values) or not bool(torch.isfinite(values).all())
                or len(torch.unique(values)) != len(values)):
            raise ValueError("Candidate multipliers must be a nonempty finite unique vector")
        self.hidden = hidden
        self.hidden_size = hidden
        self.heads = heads
        self.register_buffer("candidate_multipliers", values.clone())
        self.candidate_encoder = nn.Sequential(nn.Linear(2*hidden+11, hidden), nn.SiLU(),
                                               nn.Linear(hidden, hidden), nn.SiLU())
        layer = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=2*hidden,
            dropout=0., activation="gelu", batch_first=True, norm_first=True)
        self.candidate_interaction = nn.TransformerEncoder(layer, num_layers=1, enable_nested_tensor=False)
        self.candidate_score = nn.Linear(hidden, 1)

    def _real(self, value, name):
        parameter = self.candidate_encoder[0].weight
        value = torch.as_tensor(value, device=parameter.device)
        if value.is_complex() or value.dtype == torch.bool:
            raise ValueError("Real-valued physical/query input required: " + name)
        value = value.to(dtype=parameter.dtype)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Finite physical/query input required: " + name)
        return value

    def _mask(self, value, shape, name):
        mask = torch.as_tensor(value, device=self.candidate_encoder[0].weight.device)
        if mask.dtype != torch.bool or mask.shape != shape:
            raise ValueError("Boolean physical mask with the declared shape required: " + name)
        return mask

    def forward(self, origin, direction, bounds, ray_valid, visual_mean, visual_std,
                ra_map, re_map, range_axis, azimuth_axis, elevation_axis,
                ra_valid, re_valid, range_scale_m):
        origin, direction = self._real(origin, "origin"), self._real(direction, "direction")
        bounds = self._real(bounds, "bounds")
        mean, std = self._real(visual_mean, "visual_mean"), self._real(visual_std, "visual_std")
        if mean.ndim != 1 or std.shape != mean.shape:
            raise ValueError("One visual depth mean and standard deviation per current member required")
        n = len(mean)
        if origin.shape != (n, 3) or direction.shape != (n, 3) or bounds.shape != (n, 2):
            raise ValueError("Public rays must have shapes origin/direction[N,3] and bounds[N,2]")
        valid = self._mask(ray_valid, (n,), "ray_valid")
        if (not bool((std > 0).all()) or not torch.allclose(torch.linalg.vector_norm(direction[valid], dim=-1),
                torch.ones_like(mean[valid]), atol=1e-6, rtol=1e-6)
                or bool((bounds[valid, 0] < 0).any())
                or bool((bounds[valid, 1] <= bounds[valid, 0]).any())):
            raise ValueError("Positive visual standard deviation and unit rays with nonempty support required")
        if isinstance(range_scale_m, bool):
            raise ValueError("Positive public normalization scale required")
        scale = float(range_scale_m)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Positive public normalization scale required")
        ra_map, re_map = self._real(ra_map, "ra_map"), self._real(re_map, "re_map")
        ranges = self._real(range_axis, "range_axis")
        azimuths = self._real(azimuth_axis, "azimuth_axis")
        elevations = self._real(elevation_axis, "elevation_axis")
        for name, axis in (("range", ranges), ("azimuth", azimuths), ("elevation", elevations)):
            if axis.ndim != 1 or len(axis) < 2 or not bool((axis[1:] > axis[:-1]).all()):
                raise ValueError("Finite increasing measured physical axis required: " + name)
        if bool((ranges < 0).any()) or bool((azimuths.abs() >= math.pi/2).any()) or bool((elevations.abs() >= math.pi/2).any()):
            raise ValueError("Measured ranges/angles must use nonnegative metres and front-hemisphere radians")
        if ra_map.shape != (self.hidden, len(ranges), len(azimuths)) or re_map.shape != (self.hidden, len(ranges), len(elevations)):
            raise ValueError("Encoded spectrum maps must be [hidden, native range, measured angle]")
        av = self._mask(ra_valid, ra_map.shape[-2:], "ra_valid")
        ev = self._mask(re_valid, re_map.shape[-2:], "re_valid")
        z = self.candidate_multipliers.to(dtype=mean.dtype)
        offsets = std[:, None]*z[None]
        depths = mean[:, None]+offsets
        if not bool(torch.isfinite(depths).all()):
            raise ValueError("Visual prior generated nonfinite candidate depths")
        # One geometry calculation and one paired RA/RE query for every member.
        # Out-of-support coordinates remain as proposed; masks exclude them.
        geometry = candidate_geometry(origin, direction, depths, bounds, valid)
        if geometry["positions"].requires_grad:
            # An invalid candidate can land exactly on the station, where
            # atan2(0,0) has an undefined derivative. Forward masks alone do
            # not stop 0*NaN in that backward branch. Remove gradients only
            # for physically invalid candidate positions; all forward
            # coordinates/masks and every valid sampling gradient stay intact.
            physical_valid = geometry["valid"]
            geometry["positions"].register_hook(lambda gradient, mask=physical_valid:
                torch.where(mask[..., None], gradient, torch.zeros_like(gradient)))
        if any(not bool(torch.isfinite(geometry[key]).all()) for key in ("ranges", "azimuth", "elevation")):
            raise ValueError("Visual prior generated nonfinite physical query coordinates")
        sampled = sample_paired_spectra(ra_map, re_map, ranges=geometry["ranges"],
            azimuth=geometry["azimuth"], elevation=geometry["elevation"], range_axis=ranges,
            azimuth_axis=azimuths, elevation_axis=elevations, ra_valid=av, re_valid=ev,
            candidate_valid=geometry["valid"])
        usable = sampled["valid"] & geometry["valid"]
        angular = torch.stack((geometry["azimuth"].sin(), geometry["azimuth"].cos(),
                               geometry["elevation"].sin(), geometry["elevation"].cos()), dim=-1)
        scalars = torch.cat((offsets[..., None]/scale, (offsets/std[:, None])[..., None],
            depths[..., None]/scale, geometry["ranges"][..., None]/scale, angular,
            sampled["ra_valid"][..., None].to(mean.dtype), sampled["re_valid"][..., None].to(mean.dtype),
            geometry["valid"][..., None].to(mean.dtype)), dim=-1)
        inputs = torch.cat((sampled["ra"], sampled["re"], scalars), dim=-1)
        # Invalid geometry alone is not radar evidence. Exclude it before any
        # bias-bearing encoding and before interaction/score normalization.
        inputs = torch.where(usable[..., None], inputs, torch.zeros_like(inputs))
        tokens = self.candidate_encoder(inputs)
        tokens = torch.where(usable[..., None], tokens, torch.zeros_like(tokens))
        evidence_available = usable.any(-1)
        interacted = torch.zeros_like(tokens)
        indices = torch.nonzero(evidence_available, as_tuple=False).flatten()
        if len(indices):
            values = self.candidate_interaction(tokens[indices], src_key_padding_mask=~usable[indices])
            values = torch.where(usable[indices, :, None], values, torch.zeros_like(values))
            interacted = interacted.index_copy(0, indices, values)
        scores = self.candidate_score(interacted)[..., 0]
        weights = torch.softmax(scores.masked_fill(~usable, torch.finfo(scores.dtype).min), dim=-1)*usable
        weights = weights/weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).eps)
        pooled = (interacted*weights[..., None]).sum(1)
        pooled = torch.where(evidence_available[:, None], pooled, torch.zeros_like(pooled))
        diagnostics = dict(query_rounds=1, candidate_interaction_layers=1,
            candidate_depths_m=depths, candidate_offsets_m=offsets, candidate_multipliers=z,
            normalized_offsets=offsets/std[:, None], ranges_m=geometry["ranges"],
            azimuth_rad=geometry["azimuth"], elevation_rad=geometry["elevation"],
            geometry_valid=geometry["valid"], ra_valid=sampled["ra_valid"], re_valid=sampled["re_valid"],
            usable=usable, weights=weights, attention_weights=weights,
            evidence_available=evidence_available, sampled_ra=sampled["ra"], sampled_re=sampled["re"],
            pooling_weights_definition="candidate score-softmax weights, not per-head Transformer matrices")
        return pooled, diagnostics


SingleQueryEvidence = SingleRadarQuery
