"""Validated sensor tensors resident once per rank, with exact CPU assignment.

Only tensor gathers and small index uploads occur in ``GPUFeatureStore.batch``.
Labels are kept separately from ``PackedBatch.inputs`` and are never required
for inference. Detection dropout retains the reference RNG and rejection policy;
its Hungarian assignment is solved again on each new retained detection set.
No sensor/ROI cache, locked implementation or trained feature is written here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from d_evidence_study.data import SENSOR_FIELDS
from depth_localization.features import DETECTION_FIELDS, validate_depth_features
from rvwm.group_model import match_ray_labels

MEMBER_FIELDS = tuple(DETECTION_FIELDS) + ('detector_roi_features',)
AXIS_FIELDS = ('range_m', 'azimuth_rad', 'elevation_rad')
FRAME_FIELDS = ('observer_position', 'range_scale_m', 'ra_power', 're_power',
                'ra_valid', 're_valid', 'camera_K', 'camera_image_size_wh')
BOOL_FIELDS = frozenset(('ray_valid', 'ra_valid', 're_valid'))


@dataclass
class PackedBatch:
    inputs: dict
    lengths: tuple
    targets: dict | None = None
    matches: tuple | None = None
    positions: tuple | None = None
    metadata: dict = field(default_factory=dict)


def _numpy(value):
    if isinstance(value, torch.Tensor):
        if value.device.type != 'cpu':
            raise ValueError('Validate/pack CPU observations once before making a resident device store')
        value = value.detach().numpy()
    return np.asarray(value)


def _tensor(value, *, key):
    dtype = torch.bool if key in BOOL_FIELDS else torch.int64 if key == 'camera_image_size_wh' else torch.float32
    # NumPy cache arrays can be immutable. Copy them rather than creating a
    # writable tensor alias; tensor inputs retain their autograd graph.
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    return torch.as_tensor(np.array(value, copy=True), dtype=dtype)


def _checked(features, config):
    extra = set(features) - SENSOR_FIELDS - {'detector_roi_features'}
    if extra:
        raise ValueError('Non-sensor or unknown model input: ' + ', '.join(sorted(extra)))
    detached = {key: _numpy(value) if isinstance(value, torch.Tensor) else value
                for key, value in features.items() if key != 'detector_roi_features'}
    validate_depth_features(detached)
    if 'detector_roi_features' in features:
        value = _numpy(features['detector_roi_features'])
        dim = int(config.get('d_evidence_roi_dim', 1024))
        if value.shape != (len(features['ray_directions']), dim) or not np.isfinite(value).all():
            raise ValueError('Finite frozen ROI vectors must align with every actual detection')
    return detached


def _truth(value):
    value = _numpy(value).astype(float, copy=False)
    if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
        raise ValueError('Separate finite training positions [N,3] required')
    return value


def _target_arrays(features, positions, matches, width, config):
    b = len(features)
    target = dict(depths_m=np.zeros((b, width), np.float32), valid=np.zeros((b, width), bool),
        positions_base_m=np.zeros((b, width, 3), np.float32),
        truth_count=np.array([len(t) for t in positions], np.int64),
        matched_count=np.zeros(b, np.int64))
    actual = []
    for i, (f, truth) in enumerate(zip(features, positions)):
        match = match_ray_labels(f, truth, config) if matches is None else matches[i]
        rows = np.asarray(match['rows'], np.int64); cols = np.asarray(match['cols'], np.int64)
        depths = np.asarray(match['depths_m'], float)
        n = len(f['ray_directions'])
        if (rows.ndim != 1 or cols.shape != rows.shape or depths.shape != rows.shape
                or np.any(rows < 0) or np.any(rows >= n) or np.any(cols < 0) or np.any(cols >= len(truth))
                or len(np.unique(rows)) != len(rows) or len(np.unique(cols)) != len(cols)
                or not np.isfinite(depths).all() or not np.asarray(f['ray_valid'])[rows].all()):
            raise ValueError('Training assignment does not align with current observed detections/labels')
        expected = np.sum((truth[cols] - np.asarray(f['ray_origins'])[rows]) *
                          np.asarray(f['ray_directions'])[rows], axis=-1)
        if not np.allclose(depths, expected, atol=1e-6, rtol=1e-7):
            raise ValueError('Training depths must be GT projections onto the current fixed rays')
        target['depths_m'][i, rows] = depths
        target['valid'][i, rows] = True
        target['positions_base_m'][i, rows] = truth[cols]
        target['matched_count'][i] = len(rows)
        actual.append(match)
    return {key: torch.from_numpy(value) for key, value in target.items()}, tuple(actual)


def pack_batch(features_list, positions=None, matches=None, *, device='cpu', config=None):
    """One-time CPU validation and FP32 packing of an equal-waveform batch.

    ``positions`` and ``matches`` are supervision arguments, never fields in
    the sensor dictionaries. Mixed grids and ROI availability are rejected.
    """
    config = {} if config is None else config
    features = tuple(features_list)
    if not features:
        raise ValueError('A nonempty list of frames is required')
    validated = tuple(_checked(f, config) for f in features)
    lengths = tuple(len(f['ray_directions']) for f in features)
    width = max(1, max(lengths)); first = features[0]
    roi = 'detector_roi_features' in first
    for f in features[1:]:
        if ('detector_roi_features' in f) != roi:
            raise ValueError('Mixed ROI availability in a packed batch')
        for key in AXIS_FIELDS:
            if not np.array_equal(_numpy(f[key]), _numpy(first[key])):
                raise ValueError('Resident batches require identical physical axes: ' + key)
        if float(f['range_scale_m']) != float(first['range_scale_m']):
            raise ValueError('Resident batches require one RF support/range scale')
    inputs = {key: _tensor(first[key], key=key) for key in AXIS_FIELDS}
    inputs['member_mask'] = torch.arange(width)[None] < torch.tensor(lengths)[:, None]
    for key in MEMBER_FIELDS:
        if key == 'detector_roi_features' and not roi:
            continue
        values = []
        for f, n in zip(features, lengths):
            value = _tensor(f[key], key=key)
            padding = value.new_zeros((width - n,) + tuple(value.shape[1:]))
            values.append(torch.cat((value, padding), dim=0))
        inputs[key] = torch.stack(values)
    for key in FRAME_FIELDS:
        try:
            inputs[key] = torch.stack([_tensor(f[key], key=key) for f in features])
        except RuntimeError as exc:
            raise ValueError('Resident batches require equal per-frame map/calibration shapes: ' + key) from exc
    targets = actual = truths = None
    if positions is None:
        if matches is not None:
            raise ValueError('Assignment metadata requires separate supervision positions')
    else:
        truths = tuple(_truth(t) for t in positions)
        if len(truths) != len(features) or (matches is not None and len(matches) != len(features)):
            raise ValueError('One independent training target/assignment is required per frame')
        targets, actual = _target_arrays(validated, truths, matches, width, config)
    device = torch.device(device)
    inputs = {key: value.to(device=device) for key, value in inputs.items()}
    if targets is not None:
        targets = {key: value.to(device=device) for key, value in targets.items()}
    return PackedBatch(inputs, lengths, targets, actual, truths,
                       dict(validated_on_cpu=True, model_inputs_contain_gt=False, dtype='float32'))


class MatchPlan:
    """Reference-equivalent subset Hungarian using immutable pair geometry.

    Deleting detections can change the optimal GT assignment. Cache geometry,
    not a sliced assignment. Each new keep mask gets its own complete solve.
    """
    def __init__(self, features, positions, config=None):
        config = {} if config is None else config
        self.truth = _truth(positions)
        f = {key: _numpy(features[key]) for key in
             ('ray_origins', 'ray_directions', 'ray_valid', 'ray_depth_bounds')}
        f['range_scale_m'] = float(features['range_scale_m'])
        self.full = match_ray_labels(f, self.truth, config)
        self.gate = float(config.get('group_supervision_max_angle_deg', 3.))
        self.n = len(f['ray_origins']); self.active = np.flatnonzero(f['ray_valid'])
        self.valid = np.asarray(f['ray_valid'], bool)
        tolerance = max(1e-6, f['range_scale_m'] * 1e-7)
        self.label_domain = (self.truth[:, 0] > 0.) & (
            np.linalg.norm(self.truth, axis=-1) <= f['range_scale_m'] + tolerance)
        delta = self.truth[None] - np.asarray(f['ray_origins'], float)[:, None]
        self.all_depth = np.sum(delta * np.asarray(f['ray_directions'], float)[:, None], axis=-1)
        distance = np.linalg.norm(delta[self.active], axis=-1)
        unit = delta[self.active] / np.maximum(distance[..., None], 1e-12)
        cosine = np.clip(np.sum(unit * np.asarray(f['ray_directions'], float)[self.active, None], axis=-1), -1., 1.)
        self.angular = np.degrees(np.arccos(cosine))
        self.depth = self.all_depth[self.active]
        self.angle_ok = (distance > 1e-12) & (self.angular <= self.gate + 1e-10)
        bounds = np.asarray(f['ray_depth_bounds'], float)
        self.depth_ok = ((self.depth >= bounds[self.active, 0, None] - tolerance) &
                         (self.depth <= bounds[self.active, 1, None] + tolerance))
        self.support_ok = self.label_domain[None] & self.depth_ok
        self.feasible = self.angle_ok & self.support_ok
        self.cache = {np.ones(self.n, bool).tobytes(): self.full}
        self.new_subset_solves = 0

    def match(self, keep):
        keep = np.asarray(keep)
        if keep.dtype != np.bool_ or keep.shape != (self.n,):
            raise ValueError('Keep mask must be boolean and aligned to the original detections')
        key = keep.tobytes()
        if key in self.cache:
            return self.cache[key]
        selected = np.flatnonzero(keep[self.active]); active = self.active[selected]
        angular, depth = self.angular[selected], self.depth[selected]
        feasible = self.feasible[selected]
        row = col = np.empty(0, np.int64)
        if len(active) and len(self.truth):
            unmatched = float(len(active) + 1)
            cost = np.full((len(active), len(self.truth) + len(active)), unmatched)
            cost[:, :len(self.truth)] = np.where(feasible, angular / self.gate, unmatched * (len(active) + 2))
            ar, ac = linear_sum_assignment(cost)
            ok = ac < len(self.truth); row, col = ar[ok], ac[ok]
            if len(row):
                ok = feasible[row, col]; row, col = row[ok], col[ok]
        # Translate original detection rows into the reduced-frame indexing.
        inverse = np.cumsum(keep) - 1
        rows = inverse[active[row]].astype(np.int64)
        diagnostics = dict(self.full['diagnostics'],
            supervision_usable=bool(len(rows)), matched_training_count=len(rows),
            training_detected_count=int(keep.sum()), valid_training_rays=len(active),
            unmatched_detection_count=len(active) - len(rows), invalid_detection_count=int(keep.sum()) - len(active),
            unmatched_label_count=len(self.truth) - len(rows),
            rejected_angle_pairs=int((~self.angle_ok[selected]).sum()),
            rejected_support_pairs=int((~self.support_ok[selected]).sum()),
            depth_outside_support_pair_count=int((~self.depth_ok[selected]).sum()),
            matched_angle_errors_deg=angular[row, col].tolist(),
            rejection_reason=None if len(rows) else 'no_matches_within_angle_and_physical_support')
        result = dict(rows=rows, cols=col.astype(np.int64), depths_m=depth[row, col], diagnostics=diagnostics)
        self.cache[key] = result; self.new_subset_solves += 1
        return result


class GPUFeatureStore:
    """One read-only sensor/label tensor bank per rank, on CPU or CUDA.

    Construction may validate and upload. ``batch`` never reads a CUDA tensor
    back to the CPU, calls feature validation, or converts an entire spectrum.
    """
    def __init__(self, samples, config, device='cpu'):
        start = perf_counter()
        self.samples = tuple(samples); self.config = dict(config)
        if not self.samples:
            raise ValueError('A resident store requires at least one frame')
        self.features = tuple(s['features'] for s in self.samples)
        labeled = tuple('positions' in s and s['positions'] is not None for s in self.samples)
        if any(labeled) != all(labeled):
            raise ValueError('Mixed labeled and inference-only frames in a resident store')
        self.plans = tuple(MatchPlan(f, s['positions'], config) for f, s in zip(self.features, self.samples)) if all(labeled) else None
        self.positions = tuple(plan.truth for plan in self.plans) if self.plans is not None else None
        full = tuple(p.full for p in self.plans) if self.plans is not None else None
        self.bank = pack_batch(self.features, self.positions, full, device=device, config=config)
        self.device = torch.device(device)
        self.valid = tuple(_numpy(f['ray_valid']).astype(bool, copy=True) for f in self.features)
        self.width = self.bank.inputs['member_mask'].shape[1]
        if self.plans is not None:
            truth_width = max(1, max(map(len, self.positions)))
            truth = np.zeros((len(self.samples), truth_width, 3), np.float32)
            projections = np.zeros((len(self.samples), self.width, truth_width), np.float32)
            for i, p in enumerate(self.plans):
                truth[i, :len(p.truth)] = p.truth
                projections[i, :p.n, :len(p.truth)] = p.all_depth
            self.truth_bank = torch.from_numpy(truth).to(self.device)
            self.projection_bank = torch.from_numpy(projections).to(self.device)
        self.initialization_seconds = perf_counter() - start

    def __len__(self):
        return len(self.samples)

    def batch(self, indices, rng=None, dropout_probability=0., keeps=None):
        indices = tuple(int(i) for i in indices)
        if not indices or min(indices) < 0 or max(indices) >= len(self):
            raise ValueError('Nonempty in-range frame indices are required')
        probability = float(dropout_probability)
        if not np.isfinite(probability) or not 0. <= probability <= 1.:
            raise ValueError('Dropout probability must be in [0,1]')
        if probability and rng is None and keeps is None:
            raise ValueError('Reference-compatible detection dropout requires its NumPy RNG')
        if (probability or keeps is not None) and self.plans is None:
            raise ValueError('Supervision-based training dropout is unavailable for inference-only observations')
        if keeps is not None and len(keeps) != len(indices):
            raise ValueError('One explicit keep mask is required per selected frame')
        selected = []; actual = []; dropped = rejected = 0
        for bi, index in enumerate(indices):
            n = self.bank.lengths[index]; valid = self.valid[index]
            keep = np.ones(n, bool)
            if keeps is not None:
                proposal = np.asarray(keeps[bi])
                if proposal.dtype != np.bool_ or proposal.shape != (n,):
                    raise ValueError('Explicit dropout masks must be aligned booleans')
                keep = proposal.copy()
            elif n >= 2 and valid.any() and rng is not None and rng.random() < probability:
                keep = rng.random(n) > .2
                if not (keep & valid).any():
                    keep[int(rng.choice(np.flatnonzero(valid)))] = True
            match = None
            if self.plans is not None:
                match = self.plans[index].match(keep)
                if not keep.all():
                    if not match['diagnostics']['supervision_usable']:
                        keep = np.ones(n, bool); match = self.plans[index].full; rejected += 1
                    else:
                        dropped += 1
                actual.append(match)
            selected.append(np.flatnonzero(keep))
        lengths = tuple(len(rows) for rows in selected); width = max(1, max(lengths))
        gather = np.zeros((len(indices), width), np.int64)
        for i, rows in enumerate(selected):
            gather[i, :len(rows)] = rows
        ix = torch.tensor(indices, dtype=torch.long, device=self.device)
        member_ix = torch.from_numpy(gather).to(self.device)
        mask = torch.arange(width, device=self.device)[None] < torch.tensor(lengths, device=self.device)[:, None]
        inputs = {}
        for key, value in self.bank.inputs.items():
            if key in AXIS_FIELDS:
                inputs[key] = value
            elif key == 'member_mask':
                inputs[key] = mask
            elif key in MEMBER_FIELDS:
                chosen = value.index_select(0, ix)
                shape = (len(indices), width) + (1,) * (chosen.ndim - 2)
                lookup = member_ix.reshape(shape).expand((len(indices), width) + tuple(chosen.shape[2:]))
                chosen = chosen.gather(1, lookup)
                inputs[key] = chosen * mask.reshape(shape)
            else:
                inputs[key] = value.index_select(0, ix)
        targets = None
        if self.plans is not None:
            col = np.zeros((len(indices), width), np.int64); target_valid = np.zeros((len(indices), width), bool)
            for i, match in enumerate(actual):
                col[i, match['rows']] = match['cols']; target_valid[i, match['rows']] = True
            cols = torch.from_numpy(col).to(self.device)
            tv = torch.from_numpy(target_valid).to(self.device)
            projections = self.projection_bank.index_select(0, ix).gather(1,
                member_ix[..., None].expand(-1, -1, self.projection_bank.shape[2]))
            depths = projections.gather(2, cols[..., None])[..., 0]
            gt = self.truth_bank.index_select(0, ix).gather(1, cols[..., None].expand(-1, -1, 3))
            targets = dict(depths_m=depths * tv, valid=tv, positions_base_m=gt * tv[..., None],
                truth_count=self.bank.targets['truth_count'].index_select(0, ix), matched_count=tv.sum(-1))
        diagnostics = tuple(m['diagnostics'] for m in actual) if self.plans is not None else ()
        stats = dict(dropped_count=dropped, rejected_count=rejected, kept_indices=tuple(selected),
            supervision_usable=tuple(d['supervision_usable'] for d in diagnostics),
            matched_training_count=sum(d['matched_training_count'] for d in diagnostics), diagnostics=diagnostics)
        batch = PackedBatch(inputs, lengths, targets, tuple(actual) if actual else None,
            tuple(self.positions[i] for i in indices) if self.positions is not None else None,
            dict(validated_on_cpu=True, model_inputs_contain_gt=False, resident=True, frame_indices=indices))
        return batch, stats
