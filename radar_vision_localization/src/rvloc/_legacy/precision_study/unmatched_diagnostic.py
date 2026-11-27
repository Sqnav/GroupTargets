"""Explicit offline attribution using only saved validation evaluation JSONL.

No model, source sensor/GT reader, feature cache, IQ, image or test output is
used. Public radar limits and the engineering angle gate are explicit inputs;
saved detector rays and labels are used only to audit the fixed evaluator.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from rvwm.group_model import match_ray_labels
from rvwm.group_observation import _ray_intervals


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _pairwise_bbox_iou(boxes):
    boxes=np.asarray(boxes,float);result=[]
    for a in range(len(boxes)):
        for b in range(a+1,len(boxes)):
            lo=np.maximum(boxes[a,:2],boxes[b,:2]);hi=np.minimum(boxes[a,2:],boxes[b,2:])
            overlap=float(np.maximum(hi-lo,0).prod())
            area=[float(np.maximum(boxes[i,2:]-boxes[i,:2],0).prod()) for i in (a,b)]
            union=sum(area)-overlap
            result.append(dict(detection_indices=[a,b],iou=overlap/union if union>0 else 0.,
                center_distance_px=float(np.linalg.norm((boxes[a,:2]+boxes[a,2:]-boxes[b,:2]-boxes[b,2:])/2))))
    return sorted(result,key=lambda x:x['iou'],reverse=True)[:3]


def diagnose(directory,angle_gate=3.,native_range_max=594.90065978125,unambiguous_range=599.584916):
    directory=Path(directory).resolve()
    # This specialized audit intentionally accepts only the preserved baseline
    # location; it never becomes a generic way to bypass frozen-test gates.
    if directory.name!='v3' or directory.parent.name!='baseline':
        raise ValueError('This diagnostic accepts the saved baseline/v3 validation output only')
    files={key:directory/(key+'.jsonl') for key in ('frames','members','detections')}
    frames={(r['episode_id'],r['frame_idx']):r for r in _rows(files['frames'])}
    members=_rows(files['members']);detections=_rows(files['detections'])
    by_frame_members=defaultdict(list);by_frame_detections=defaultdict(list)
    for row in members:by_frame_members[row['episode_id'],row['frame_idx']].append(row)
    for row in detections:by_frame_detections[row['episode_id'],row['frame_idx']].append(row)
    failures=[r for r in members if r['status']=='unmatched_truth']
    frame_checks={};cases=[]
    for member in failures:
        key=member['episode_id'],member['frame_idx'];f=frames[key]
        origins=np.asarray(f['ray_origin_base_m'],float).reshape(-1,3)
        directions=np.asarray(f['ray_direction_base'],float).reshape(-1,3)
        valid=np.asarray(f['ray_valid'],bool);bounds=np.asarray(f['ray_depth_bounds_m'],float).reshape(-1,2)
        truths=np.asarray(f['truth_positions_base_m'],float).reshape(-1,3)
        features=dict(ray_origins=origins,ray_directions=directions,ray_valid=valid,
            ray_depth_bounds=bounds,range_scale_m=unambiguous_range)
        saved={(r['detection_index'],r['truth_index']) for r in by_frame_members[key] if r['matched']}
        if key not in frame_checks:
            matched=match_ray_labels(features,truths,{'group_supervision_max_angle_deg':angle_gate})
            alternate=match_ray_labels(dict(features,range_scale_m=native_range_max),truths,
                {'group_supervision_max_angle_deg':angle_gate})
            actual=set(zip(matched['rows'].tolist(),matched['cols'].tolist()))
            if actual!=saved:raise AssertionError('Saved assignment differs from current exact matcher: '+str(key))
            saved_detections={(r['detection_index'],r['matched_truth_index']) for r in by_frame_detections[key] if r['matched_truth_index'] is not None}
            if actual!=saved_detections:raise AssertionError('Saved detection/member assignment differs')
            rebuilt,rebuilt_valid=_ray_intervals(origins,directions,np.ones(len(valid),bool),unambiguous_range)
            endpoints=origins+bounds[:,1,None]*directions
            sphere_ends=np.linalg.norm(endpoints[np.abs(endpoints[:,0])>1e-6],axis=-1)
            frame_checks[key]=dict(episode_id=key[0],frame_idx=key[1],detection_count=len(origins),
                truth_count=len(truths),valid_ray_count=int(valid.sum()),matched_count=len(actual),
                exact_current_matcher_reproduces_saved_pairs=True,
                native_last_bin_vs_unambiguous_domain_assignment_equal=actual==set(zip(alternate['rows'].tolist(),alternate['cols'].tolist())),
                analytic_unambiguous_radius_reproduces_saved_valid_mask=bool(np.array_equal(valid,rebuilt_valid)),
                analytic_unambiguous_radius_bounds_max_abs_difference_m=float(np.max(np.abs(rebuilt-bounds),initial=0.)),
                saved_non_front_clipped_upper_endpoint_range_m=sphere_ends.tolist(),
                unmatched_detection_indices=[r['detection_index'] for r in by_frame_detections[key] if r['matched_truth_index'] is None],
                highest_bbox_overlap_pairs=_pairwise_bbox_iou(f.get('bbox_xyxy',[])))
        truth=truths[member['truth_index']]
        if not np.array_equal(truth,np.asarray(member['truth_position_base_m'])):
            raise AssertionError('Saved frame/member truth positions differ')
        delta=truth-origins;rho=np.sum(delta*directions,axis=-1)
        distance=np.linalg.norm(delta,axis=-1)
        angles=np.degrees(np.arccos(np.clip(rho/np.maximum(distance,1e-12),-1,1)))
        tolerance=max(1e-6,unambiguous_range*1e-7)
        angle_ok=(distance>1e-12)&(angles<=angle_gate+1e-10)
        depth_ok=(rho>=bounds[:,0]-tolerance)&(rho<=bounds[:,1]+tolerance)
        domain=bool(truth[0]>0 and np.linalg.norm(truth)<=unambiguous_range+tolerance)
        feasible=valid&angle_ok&depth_ok&domain
        if not len(origins):reason='no_detection_instances_in_frame'
        elif not valid.any():reason='no_valid_observed_rays'
        elif not domain:reason='outside_base_front_or_unambiguous_range_domain'
        elif not (valid&angle_ok).any():reason='no_valid_detection_ray_within_3_degree_bearing_gate'
        elif not feasible.any():reason='bearing_feasible_but_projection_depth_outside_saved_ray_support'
        else:reason='feasible_edge_exists_but_lost_in_one_to_one_assignment'
        candidates=[]
        assignments={r['detection_index']:r['matched_truth_index'] for r in by_frame_detections[key]}
        for i in np.argsort(angles):
            candidates.append(dict(detection_index=int(i),observed_ray_valid=bool(valid[i]),
                bearing_error_deg=float(angles[i]),bearing_gate_pass=bool(angle_ok[i]),
                projection_depth_m=float(rho[i]),saved_ray_support_m=bounds[i].tolist(),
                projection_depth_inside_support=bool(depth_ok[i]),
                ray_perpendicular_error_m=float(np.linalg.norm(delta[i]-rho[i]*directions[i])),
                feasible_for_this_truth=bool(feasible[i]),assigned_truth_index=assignments.get(int(i)),
                bbox_xyxy=f['bbox_xyxy'][i]))
        cases.append(dict(episode_id=key[0],frame_idx=key[1],truth_index=member['truth_index'],
            saved_status=member['status'],confirmed_evaluator_rejection=reason,
            evidence_level='confirmed_from_saved_geometry_and_exact_assignment_reproduction',
            gt_position_base_m=truth.tolist(),base_range_m=float(np.linalg.norm(truth)),
            base_x_positive=bool(truth[0]>0),inside_native_last_range_bin=bool(np.linalg.norm(truth)<=native_range_max),
            inside_unambiguous_range=domain,frame_detection_count=len(origins),frame_truth_count=len(truths),
            frame_detection_count_deficit=max(0,len(truths)-len(origins)),
            feasible_candidate_detection_indices=np.flatnonzero(feasible).tolist(),
            nearest_observed_ray=candidates[0] if candidates else None,all_observed_rays_sorted_by_bearing=candidates,
            root_cause_not_proven=['RGB detector miss versus occlusion/visibility, suppressed boxes, duplicate detections, or bbox-centre/GT-centre disagreement cannot be separated without saved GT boxes/visibility/confidence; these are not stored in these three JSONL files.']))
    checks=list(frame_checks.values());reasons=Counter(c['confirmed_evaluator_rejection'] for c in cases)
    return dict(status='complete_saved_validation_only_offline_diagnostic',input_directory=str(directory),
        inputs={key:dict(path=str(path),sha256=_sha(path)) for key,path in files.items()},
        source_helper_sha256=_sha(__file__),matcher_source_sha256=_sha(Path(__file__).parents[1]/'rvwm/group_model.py'),
        scope=dict(labels_use='Only already saved offline evaluation labels; never model input',
            source_sensor_or_label_store_read=False,feature_cache_read=False,model_or_checkpoint_loaded=False,test_set_read=False,
            evaluation_counts_or_denominators_modified=False),
        thresholds=dict(bearing_gate_deg=angle_gate,native_last_range_bin_m=native_range_max,
            public_unambiguous_range_m=unambiguous_range,
            source='Explicit public radar limits: c=299792458 m/s, K=128, B=32 MHz; R_last=c*(K-1)/(2B), R_unambiguous=c*K/(2B). Range scale is not saved directly in frame JSONL; its analytical support is checked against all saved bounds.'),
        unmatched_truth_count=len(cases),affected_frame_count=len(checks),
        confirmed_evaluator_rejection_counts=dict(reasons),
        empty_detection_frame_count=sum(f['detection_count']==0 for f in checks),
        no_valid_ray_frame_count=sum(f['valid_ray_count']==0 for f in checks),
        unmatched_members_in_detection_count_deficit_frames=sum(c['frame_detection_count']<c['frame_truth_count'] for c in cases),
        unmatched_members_in_equal_detection_count_frames=sum(c['frame_detection_count']==c['frame_truth_count'] for c in cases),
        exact_current_matcher_reproduces_every_affected_saved_frame=all(f['exact_current_matcher_reproduces_saved_pairs'] for f in checks),
        domain_radius_choice_affects_any_affected_assignment=any(not f['native_last_bin_vs_unambiguous_domain_assignment_equal'] for f in checks),
        maximum_analytic_support_difference_m=max((f['analytic_unambiguous_radius_bounds_max_abs_difference_m'] for f in checks),default=0.),
        nearest_ray_bearing_error_range_deg=[min((c['nearest_observed_ray']['bearing_error_deg'] for c in cases),default=None),
            max((c['nearest_observed_ray']['bearing_error_deg'] for c in cases),default=None)],
        interpretation='unmatched_truth is a fixed observed-bearing/support association failure, not an IoU-based detector miss label. No failure denominator is changed. These cases have no feasible bearing edge, so a Hungarian collision is not their cause.',
        affected_frames=checks,cases=cases)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError('Existing diagnostic result is protected')
    result=diagnose(a.baseline);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','unmatched_truth_count','affected_frame_count','confirmed_evaluator_rejection_counts','nearest_ray_bearing_error_range_deg','maximum_analytic_support_difference_m')}))
    return 0


if __name__=='__main__':raise SystemExit(main())
