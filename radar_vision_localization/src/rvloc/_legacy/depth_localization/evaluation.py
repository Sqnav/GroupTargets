"""Score one deterministic depth vector, with misses in all-truth denominators."""
from __future__ import annotations
import numpy as np
from diagnostics.geometry import freeze_correspondence,decompose_frame,aggregate_frames
from rvwm.ray_scoring import observation_rejection_reason


def prediction_rejection(features,scene,tolerance_m=1e-3):
    n=len(features['ray_directions'])
    p=np.asarray(scene.get('member_positions',[]),float);depth=np.asarray(scene.get('depths',[]),float)
    valid=np.asarray(scene.get('valid_mask',[]))
    if p.shape!=(n,3) or depth.shape!=(n,) or valid.shape!=(n,) or valid.dtype!=np.bool_:return 'invalid_prediction_shape'
    allowed=np.asarray(features['ray_valid'],bool)
    if np.any(valid&~allowed):return 'invalid_ray_enabled'
    if not valid.any():return 'no_valid_members'
    if not np.isfinite(p[valid]).all() or not np.isfinite(depth[valid]).all():return 'nonfinite_prediction'
    bounds=np.asarray(features['ray_depth_bounds'])
    if np.any(depth[valid]<bounds[valid,0]-tolerance_m) or np.any(depth[valid]>bounds[valid,1]+tolerance_m):return 'outside_depth_support'
    on_ray=np.asarray(features['ray_origins'])+depth[:,None]*np.asarray(features['ray_directions'])
    if np.any(np.linalg.norm((p-on_ray)[valid],axis=-1)>tolerance_m):return 'violates_visual_ray'
    if np.any(p[valid,0]<=0) or np.any(np.linalg.norm(p[valid],axis=-1)>float(features['range_scale_m'])+tolerance_m):return 'outside_radar_domain'
    return None


def evaluate(model,samples,manifest,config,allow_untrained=False):
    if not samples and not manifest:raise ValueError('Evaluation requires a declared population')
    by_key={(s['episode_id'],int(s['frame_idx'])):s for s in samples}
    if len(by_key)!=len(samples):raise ValueError('Duplicate depth evaluation sample')
    keys=[(r['episode_id'],int(r['frame_idx'])) for r in manifest]
    if len(set(keys))!=len(keys) or set(by_key)-set(keys):raise ValueError('Depth evaluation inventory differs')
    previous=model.training;model.eval();rows=[];decompositions=[];center_errors=[]
    try:
        for item,key in zip(manifest,keys):
            sample=by_key.get(key);scene=None;reason=item.get('feature_status','missing_observation')
            count=int(item.get('label_count',len(sample['positions']) if sample is not None else 0))
            if sample is None:
                if 'label_count' not in item:raise ValueError('Missing frame needs explicit truth count')
                row=dict(episode_id=key[0],frame_idx=key[1],truth_member_count=count,
                    detection_count=int(item.get('detected_count',0)),primary_member_errors_m=[],
                    primary_center_error_m=None,primary_whole_scene_success_5m=False,
                    observation_status=reason,predicted=False,unmatched_truth_count=count)
            else:
                f=sample['features'];truth=sample['positions'];count=len(truth)
                # Freeze original observation correspondence before model output.
                correspondence=freeze_correspondence(f,truth,config)
                reason=observation_rejection_reason(f,config)
                if reason is None:
                    scene=model.predict(f,allow_untrained_validation=allow_untrained)
                    reason=prediction_rejection(f,scene)
                    if reason is not None:scene=None
                geometry=decompose_frame(f,truth,scene,correspondence=correspondence,config=config)
                geometry.update(episode_id=key[0],frame_idx=key[1]);decompositions.append(geometry)
                members=[m for m in geometry['members'] if m['prediction_usable']]
                errors=[m['error_3d_m'] for m in members];center=None
                if geometry['complete_cardinality_and_localization']:
                    center=float(np.linalg.norm(np.asarray(scene['member_positions']).mean(0)-np.asarray(truth).mean(0)))
                    center_errors.append(center)
                row=dict(episode_id=key[0],frame_idx=key[1],truth_member_count=count,
                    detection_count=geometry['detection_count'],primary_member_errors_m=errors,
                    primary_center_error_m=center,primary_whole_scene_success_5m=geometry['whole_scene_success_5m'],
                    observation_status=reason or 'valid_uncalibrated',predicted=scene is not None,
                    unmatched_truth_count=geometry['unmatched_truth_count'],
                    unmatched_detection_count=geometry['extra_or_unmatched_detection_count'],
                    correspondence_sha256=geometry['correspondence_sha256'])
            rows.append(row)
    finally:model.train(previous)
    errors=np.asarray([e for r in rows for e in r['primary_member_errors_m']],float)
    total=sum(r['truth_member_count'] for r in rows);frames=len(rows)
    summary=dict(frames=frames,truth_member_instances=total,primary_matched_member_instances=len(errors),
        primary_member_rmse_m=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        primary_member_p95_m=float(np.quantile(errors,.95)) if len(errors) else None,
        primary_center_rmse_m=float(np.sqrt(np.mean(np.asarray(center_errors)**2))) if center_errors else None,
        primary_whole_scene_success_ratio_5m=sum(r['primary_whole_scene_success_5m'] for r in rows)/max(frames,1),
        predicted_frames=sum(r['predicted'] for r in rows),complete_center_frames=len(center_errors),
        unmatched_truth_instances=sum(r['unmatched_truth_count'] for r in rows),
        configured_window=1,candidate_count_per_frame=1,model_trained=bool(model.trained),
        allow_untrained_validation=bool(allow_untrained),uncertainty_calibrated=False,
        selection_policy='primary_5m_then_rmse_then_primary_whole_scene_v2',
        primary_success_denominator='all truth members; missing, rejected and unmatched cases count as failures',
        error_scope='matched localized members; full-scene center only for complete detections',
        label_use='offline scoring only; predictions from sensor-only features',
        depth_definition='camera optical-centre slant distance along the unit bbox-centre ray, in metres')
    for threshold in (1,2,5):summary[f'primary_member_success_ratio_{threshold}m']=int(np.sum(errors<=threshold))/max(total,1)
    if decompositions:
        summary['ray_error_decomposition']=aggregate_frames(decompositions)
        summary['ray_error_decomposition_scope']='prepared sensor-feature frames only; top-level coverage includes every declared truth instance'
    return dict(summary=summary,frames=rows)


def selection_key(report):
    s=report['summary'];rmse=s['primary_member_rmse_m']
    return (s['primary_member_success_ratio_5m'],-(rmse if rmse is not None else float('inf')),
            s['primary_whole_scene_success_ratio_5m'])
