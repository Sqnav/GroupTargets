"""Label-free paired RA/RE input probes, separate from normal deployment.

These are sensitivity interventions, not physically re-simulated observations.
In particular, a global range shift moves both group and observer spectra while
the independent observer navigation stays fixed. A response gain of one is not
a correctness requirement. Genuine vision-only performance requires retraining.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from numbers import Integral

import numpy as np

from rvwm.group_observation import feature_digest, validate_features


VERSION = 'paired_ra_re_input_sensitivity_v1'
RANGE_SHIFTS = {
    'range_shift_minus2': -2,
    'range_shift_minus1': -1,
    'range_shift_plus1': 1,
    'range_shift_plus2': 2,
}
INTERVENTIONS = ('baseline', 'zero_radar', 'zero_ra', 'zero_re', 'joint_shuffle', *RANGE_SHIFTS)
_AXES = ('range_m', 'azimuth_rad', 'elevation_rad')
_MASKS = ('ra_valid', 're_valid')


def normalization_audit():
    """The current model removes a common positive RA/RE power multiplier."""
    return dict(
        model_operation='log1p(power / shared_masked_mean_power)',
        scale_floor=1e-20,
        common_positive_gain_is_invariant_above_floor=True,
        ra_and_re_share_one_scale=True,
        interpretation='Little response to a common multiplicative gain is expected by construction; it does not show that the model ignores radar.',
        caveat='Float rounding and the 1e-20 floor can break exact numeric invariance.',
    )


def normalized_model_power(features):
    """Expose only the shared-scale power transform for diagnostic comparisons.

    Geometry channels still enter the actual encoders; these arrays are not a
    replacement for executing the trained model.
    """
    validate_features(features)
    total = 0.; count = 0
    for kind in ('ra', 're'):
        power = np.asarray(features[kind+'_power'], np.float64)
        valid = np.asarray(features[kind+'_valid'], bool)
        total += float(power[valid].sum()); count += int(valid.sum())
    scale = max(total/max(count, 1), 1e-20)
    return {
        kind: np.log1p(np.where(features[kind+'_valid'], features[kind+'_power'], 0.).astype(np.float64)/scale)
        for kind in ('ra', 're')
    }


def zero_fill_range_shift(power, bins):
    """Translate measured power on its first axis; never wrap a boundary peak."""
    if isinstance(bins, (bool, np.bool_)) or not isinstance(bins, Integral):
        raise ValueError('Range displacement must be an integer number of native bins')
    value = np.asarray(power)
    if value.ndim != 2:
        raise ValueError('RA/RE power must be a two-dimensional range-angle map')
    result = np.zeros_like(value); bins = int(bins)
    if bins == 0:
        result[...] = value
    elif 0 < bins < len(value):
        result[bins:] = value[:-bins]
    elif -len(value) < bins < 0:
        result[:bins] = value[-bins:]
    return result


def _native_range_step(features):
    axis = np.asarray(features['range_m'], np.float64)
    difference = np.diff(axis)
    if not np.allclose(difference, difference[0], rtol=1e-7, atol=1e-9):
        raise ValueError('A fixed physical displacement requires a uniform native range axis')
    return float(difference[0])


def _episode(features):
    value = features.get('metadata', {}).get('episode_id')
    if not isinstance(value, str) or not value:
        raise ValueError('Cross-episode radar shuffling requires an explicit source episode_id')
    return value


def _sensor_identity(features):
    meta = features['metadata']
    return dict(episode_id=meta.get('episode_id'), frame_idx=meta.get('frame_idx'),
                feature_sha256=feature_digest(features), raw_hashes=deepcopy(meta.get('raw_hashes', {})))


def _same_measurement_grid(features, donor):
    if float(features['range_scale_m']) != float(donor['range_scale_m']):
        return False
    return all(np.array_equal(features[name], donor[name]) for name in (*_AXES, *_MASKS))


def apply_intervention(features, name, *, donor=None, donor_identity=None):
    """Return a detached feature copy; leave original arrays/caches unchanged.

    Zero power is kept exactly zero. The normal publication gate will reject
    it, so an evaluator may report raw proposer outputs only under an explicit
    diagnostic label. No noise, peak or altered mask is invented to pass it.
    """
    validate_features(features)
    if name not in INTERVENTIONS:
        raise ValueError('Unknown radar input intervention: '+str(name))
    result = deepcopy(features)
    receipt = dict(schema=VERSION, name=name, sensor_source=_sensor_identity(features),
        visual_geometry_unchanged=True, observer_navigation_unchanged=True,
        physical_axes_unchanged=True, validity_masks_unchanged=True,
        labels_used=False, cache_modified=False,
        same_proposal_rng_seed_required=True,
        physically_resimulated=False, input_distribution_status='original' if name=='baseline' else 'intervened_ood',
        interpretation='Input sensitivity only; cannot replace a trained visual-only baseline or identify causal localization gain.')
    if name in ('zero_radar', 'zero_ra', 'zero_re'):
        affected = ('ra', 're') if name=='zero_radar' else (name[5:],)
        for kind in affected:
            result[kind+'_power'] = np.zeros_like(features[kind+'_power'])
        receipt.update(zeroed_spectra=list(affected), normal_quality_gate_expected_rejection='no_radar_signal',
            raw_only_if_gate_rejects=True, shared_model_power_scale_changes=True)
    elif name == 'joint_shuffle':
        if donor is None:
            raise ValueError('Joint RA/RE shuffling requires one source-bound donor feature set')
        validate_features(donor)
        if _episode(features) == _episode(donor):
            raise ValueError('Radar donor must come from another episode')
        if not _same_measurement_grid(features, donor):
            raise ValueError('Radar donor physical axes, range support and validity masks must agree')
        for kind in ('ra', 're'):
            result[kind+'_power'] = np.array(donor[kind+'_power'], copy=True)
        receipt.update(donor_source=_sensor_identity(donor), spectra_are_paired_from_one_donor=True,
            donor_selected_without_labels=True,
            observer_spectrum_navigation_consistent=False,
            physical_consistency_note='Donor spectra include its own observer echo; target-frame observer navigation is deliberately unchanged.')
        if donor_identity is not None:
            # Keep receipts JSON serializable rather than storing donor objects.
            receipt['donor_record'] = json.loads(json.dumps(donor_identity, allow_nan=False))
    elif name in RANGE_SHIFTS:
        bins = RANGE_SHIFTS[name]
        for kind in ('ra', 're'):
            result[kind+'_power'] = zero_fill_range_shift(features[kind+'_power'], bins)
        receipt.update(range_shift_bins=bins, commanded_range_shift_m=bins*_native_range_step(features),
            shift_boundary='zero fill; power shifted out of native support is discarded; no wrap',
            observer_spectrum_navigation_consistent=False,
            physical_consistency_note='Both mixed group and observer spectra move; independent observer navigation and visual rays remain fixed.',
            unit_response_gain_is_not_a_correctness_requirement=True)
    result['metadata']['diagnostic_intervention'] = receipt
    return validate_features(result)


def _grid_signature(features):
    h = hashlib.sha256()
    for name in _AXES:
        value = np.asarray(features[name], dtype='<f8')
        h.update(name.encode()); h.update(np.asarray(value.shape, dtype='<i8').tobytes()); h.update(value.tobytes())
    for name in _MASKS:
        value = np.asarray(features[name], bool)
        h.update(name.encode()); h.update(np.asarray(value.shape, dtype='<i8').tobytes()); h.update(value.tobytes())
    h.update(np.asarray([features['range_scale_m']], dtype='<f8').tobytes())
    return h.hexdigest()


def choose_joint_donor_indices(records, seed=20261121, *, require_all=True):
    """Choose paired donors deterministically, without positions or labels.

    Records contain ``features`` and optionally ``split``. A donor must have a
    different episode and the same physical grid/masks/partition. With
    require_all=False an unavailable donor is explicitly represented by None.
    """
    groups = {}; rows = []
    for i, record in enumerate(records):
        features = record['features']; validate_features(features)
        episode = _episode(features)
        declared = record.get('episode_id', episode)
        if declared != episode:
            raise ValueError('Donor record episode differs from feature provenance')
        key = (record.get('split'), _grid_signature(features))
        groups.setdefault(key, []).append(i); rows.append((key, episode))
    rng = np.random.default_rng(seed); result = []
    for key, episode in rows:
        eligible = [i for i in groups[key] if rows[i][1] != episode]
        if not eligible:
            if require_all:
                raise ValueError('No compatible cross-episode paired radar donor is available')
            result.append(None)
        else:
            result.append(int(eligible[int(rng.integers(len(eligible)))]))
    return result


def paired_response(reference_positions, perturbed_positions, valid_mask=None, *,
                    commanded_range_shift_m=None, ray_directions=None):
    """Compare the same detection slots; do not rematch using changed outputs."""
    if reference_positions is None or perturbed_positions is None:
        return dict(status='unavailable', paired_members=0, reason='one_or_both_outputs_missing')
    reference = np.asarray(reference_positions, float); perturbed = np.asarray(perturbed_positions, float)
    if reference.ndim != 2 or reference.shape[1] != 3 or perturbed.shape != reference.shape:
        raise ValueError('Paired positions must have equal [N,3] shape')
    n = len(reference)
    valid = np.ones(n, bool) if valid_mask is None else np.asarray(valid_mask)
    if valid.shape != (n,) or valid.dtype != np.bool_:
        raise ValueError('Paired validity mask must be boolean [N]')
    if not np.isfinite(reference[valid]).all() or not np.isfinite(perturbed[valid]).all():
        raise ValueError('Valid paired member positions must be finite')
    delta = perturbed[valid]-reference[valid]
    displacement = np.linalg.norm(delta, axis=1)
    radial = np.linalg.norm(perturbed[valid], axis=1)-np.linalg.norm(reference[valid], axis=1)
    result = dict(status='available' if valid.any() else 'no_valid_pairs', paired_members=int(valid.sum()),
        detection_indices=np.flatnonzero(valid).tolist(), member_displacement_m=displacement.tolist(),
        member_base_radial_shift_m=radial.tolist(),
        mean_displacement_m=float(displacement.mean()) if len(delta) else None,
        max_displacement_m=float(displacement.max()) if len(delta) else None,
        rms_displacement_m=float(np.sqrt(np.mean(displacement**2))) if len(delta) else None,
        mean_signed_base_radial_shift_m=float(radial.mean()) if len(delta) else None,
        mean_absolute_base_radial_shift_m=float(np.abs(radial).mean()) if len(delta) else None,
        matching_policy='same observed detection slots, fixed before intervention',
        physically_resimulated=False)
    if commanded_range_shift_m is not None:
        shift = float(commanded_range_shift_m)
        if not np.isfinite(shift) or shift == 0.:
            raise ValueError('Response gain requires a finite nonzero commanded spectrum shift')
        result.update(commanded_range_shift_m=shift,
            mean_base_radial_response_gain=float(radial.mean()/shift) if len(delta) else None,
            response_gain_interpretation='Sensitivity of predicted base range to shifted mixed spectra; unit gain is not required because observer navigation is held fixed.')
    if ray_directions is not None:
        directions = np.asarray(ray_directions, float)
        if directions.shape != (n,3) or not np.isfinite(directions[valid]).all() or not np.allclose(np.linalg.norm(directions[valid],axis=1),1.,rtol=1e-6,atol=1e-6):
            raise ValueError('Paired directions must be finite unit rays [N,3]')
        along = np.sum(delta*directions[valid], axis=1)
        perpendicular = np.linalg.norm(delta-along[:,None]*directions[valid],axis=1)
        result.update(member_camera_ray_depth_shift_m=along.tolist(),
            max_cross_ray_prediction_shift_m=float(perpendicular.max()) if len(delta) else None)
    return result
