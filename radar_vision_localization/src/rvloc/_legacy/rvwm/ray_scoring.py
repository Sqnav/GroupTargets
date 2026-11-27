"""Validate and retain continuous joint modes; no extra radar likelihood.

RA and RE are correlated projections of one observation. This module never
assigns independent precise ranges to detections or reads original IQ. A
model's primary prediction stays primary; sample diversity is not probability.
"""
from __future__ import annotations
import numpy as np

VERSION = 'ra_re_joint_mode_validation_v1'


def observation_rejection_reason(features, config):
    """Shared deployment/validation input gate; no labels or learned score.

    This only establishes basic input usability. Non-flat noise can pass, so
    it must not be described as an SNR test or calibrated detection confidence.
    """
    from .group_observation import validate_features
    validate_features(features)
    count=len(features['ray_directions'])
    if not count:return 'no_detections'
    if count>int(config.get('group_max_members',24)):return 'unsupported_detected_count'
    if not np.asarray(features['ray_valid'],bool).any():return 'no_valid_rays'
    for kind in ('ra','re'):
        values=np.asarray(features[kind+'_power'])[np.asarray(features[kind+'_valid'],bool)]
        if not len(values) or float(values.max())<=np.finfo(np.float32).tiny:
            return 'no_radar_signal'
        if float(np.ptp(values))<=float(values.max())*1e-7:
            return 'uninformative_spectrum'
    return None


def select_scene_modes(features, hypotheses, config):
    from .single_frame import require_single_frame
    require_single_frame(config)
    origins=np.asarray(features['ray_origins'],float)
    directions=np.asarray(features['ray_directions'],float)
    allowed=np.asarray(features['ray_valid'],bool)
    bounds=np.asarray(features['ray_depth_bounds'],float)
    n=len(origins);accepted=[];records=[]
    tolerance=float(config.get('ray_geometry_tolerance_m',1e-4))
    separation=float(config.get('group_mode_separation_m',.5))
    maximum=int(config.get('group_max_output_modes',8))
    if tolerance<=0 or separation<=0 or maximum<1:raise ValueError('Invalid mode validation configuration')
    for index,scene in enumerate(hypotheses):
        position=np.asarray(scene.get('member_positions',[]),float)
        depth=np.asarray(scene.get('depths',[]),float)
        valid=np.asarray(scene.get('valid_mask',allowed),bool)
        reason=None
        if position.shape!=(n,3) or depth.shape!=(n,) or valid.shape!=(n,):reason='invalid_shape'
        elif np.any(valid & ~allowed):reason='invalid_ray_enabled'
        elif not valid.any():reason='no_valid_members'
        elif not np.isfinite(position[valid]).all() or not np.isfinite(depth[valid]).all():reason='nonfinite_state'
        elif np.any(depth[valid]<bounds[valid,0]-tolerance) or np.any(depth[valid]>bounds[valid,1]+tolerance):reason='outside_depth_support'
        elif np.any(np.linalg.norm(position[valid]-(origins+depth[:,None]*directions)[valid],axis=1)>tolerance):reason='violates_visual_ray'
        elif np.any(position[valid,0]<=0) or np.any(np.linalg.norm(position[valid],axis=1)>float(features['range_scale_m'])+tolerance):reason='outside_radar_domain'
        if reason:
            records.append(dict(hypothesis=index,retained=False,reason=reason));continue
        duplicate=None
        for kept in accepted:
            if np.array_equal(valid,kept['valid_mask']):
                distance=float(np.sqrt(np.mean(np.sum((position[valid]-kept['member_positions'][valid])**2,axis=-1))))
                if distance<separation:duplicate=kept['hypothesis_index'];break
        if duplicate is not None:
            records.append(dict(hypothesis=index,retained=False,reason='insufficient_joint_position_difference',duplicate_of=duplicate));continue
        if len(accepted)>=maximum:
            records.append(dict(hypothesis=index,retained=False,reason='output_mode_budget'));continue
        row=dict(scene,member_positions=position,depths=depth,valid_mask=valid,
                 center=position[valid].mean(0),hypothesis_index=index)
        accepted.append(row);records.append(dict(hypothesis=index,retained=True,reason='geometrically_valid_generated_mode'))
    return dict(modes=accepted,diagnostics=dict(schema=VERSION,candidate_decisions=records,
        primary_selection='first geometrically valid model prediction; not maximum likelihood',
        raw_iq_scoring=False,independent_ra_re_likelihood=False,
        physically_verified=False,continuous_position_correction=False,
        uncertainty_calibrated=False))
