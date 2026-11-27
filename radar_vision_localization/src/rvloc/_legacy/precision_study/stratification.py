"""Offline, fixed-member geometry strata for saved localization evaluations.

GT radial directions and proximity-cell labels are diagnosis only. No value
computed here enters a model, modifies evaluation populations or selects a
checkpoint. Every GT key appears in a bucket, including missing observed rays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .comparison import _load, _check_population
from .evidence_curves import _bucket_metric


def _key(row):return row['episode_id'],int(row['frame_idx']),int(row['truth_index'])


def compare_geometry_strata(evaluations,reference=None,assignment_path=None,output=None,allow_test=False):
    if isinstance(evaluations,dict):evaluations=list(evaluations.items())
    else:evaluations=list(evaluations)
    if not evaluations or len({name for name,_ in evaluations})!=len(evaluations):
        raise ValueError('Provide uniquely named saved evaluations')
    loaded={name:_load(name,path,allow_test) for name,path in evaluations}
    reference=reference or evaluations[0][0]
    if reference not in loaded:raise ValueError('Unknown geometry assignment reference')
    anchor=loaded[reference]
    errors=[]
    for name,model in loaded.items():
        if name!=reference:errors.extend(_check_population(anchor,model))
    if errors:
        result=dict(status='invalid',errors=errors,groups={},members=[],
            population_intersection_performed=False)
    else:
        old_records={};old_group_metadata={}
        if assignment_path is not None:
            saved=json.loads(Path(assignment_path).read_text())
            old_records={_key(row):row for row in saved['members']}
            if len(old_records)!=len(saved['members']) or set(old_records)!=set(anchor['members']):
                raise ValueError('Existing proxy assignment does not cover the same complete GT population')
            old_group_metadata=saved.get('groups',{})
        records=[]
        fixed_names=('0.00_to_0.25','0.25_to_0.50','0.50_to_0.75','0.75_to_1.00')
        for key,row in anchor['members'].items():
            record=dict(old_records.get(key,{}))
            record.update(episode_id=key[0],frame_idx=key[1],truth_index=key[2],
                radial_depth_sensitivity_abs=None,radial_depth_derivative_signed=None,
                ray_to_base_radial_angle_deg=None,radial_depth_sensitivity_abs_bucket='missing_observed_ray')
            if row['matched']:
                frame=anchor['frames'][key[:2]]
                point=row.get('truth_position_base_m')
                rays=frame.get('ray_direction_base')
                slot=row['detection_index']
                if point is not None and rays is not None:
                    point=np.asarray(point,float);direction=np.asarray(rays[slot],float)
                    norm=float(np.linalg.norm(direction));radius=float(np.linalg.norm(point))
                    if not np.isclose(norm,1.,rtol=1e-6,atol=1e-6):
                        raise ValueError('Geometry stratification requires the actual unit observed ray')
                    if radius>1e-9 and np.isfinite(point).all():
                        signed=float(np.dot(point/radius,direction/norm))
                        signed=float(np.clip(signed,-1.,1.));value=abs(signed)
                        group=fixed_names[min(3,int(np.searchsorted([.25,.5,.75],value,side='right')))]
                        record.update(radial_depth_sensitivity_abs=value,
                            radial_depth_derivative_signed=signed,
                            ray_to_base_radial_angle_deg=float(np.degrees(np.arccos(signed))),
                            radial_depth_sensitivity_abs_bucket=group)
            records.append(record)
        record_map={_key(r):r for r in records}
        families=sorted({field[:-7] for r in records for field in r if field.endswith('_bucket')})
        groups={}
        for family in families:
            assignments={key:r.get(family+'_bucket','missing') for key,r in record_map.items()}
            names=list(dict.fromkeys(assignments.values()))
            if family=='radial_depth_sensitivity_abs':
                names=[*fixed_names,'missing_observed_ray']
            family_groups={}
            for group in names:
                keys=[key for key,value in assignments.items() if value==group]
                if not keys:
                    continue
                metrics={}
                reference_localized={k for k in keys if anchor['members'][k]['status']=='localized'}
                for name,model in loaded.items():
                    members=[model['members'][key] for key in keys]
                    values=_bucket_metric(members)
                    values.update(episode_count=len({k[0] for k in keys}),
                        missing_or_rejected_members=len(members)-values['localized_members'],
                        matched_detection_members=sum(r['matched'] for r in members),
                        unmatched_truth_members=sum(not r['matched'] for r in members),
                        localized_population_matches_reference=
                            {k for k in keys if model['members'][k]['status']=='localized'}==reference_localized)
                    metrics[name]=values
                family_groups[group]=dict(truth_members=len(keys),episode_count=len({k[0] for k in keys}),models=metrics)
            groups[family]=dict(assignment_from_reference_only=True,
                metadata=old_group_metadata.get(family,{}).get('edges'),groups=family_groups)
            if sum(g['truth_members'] for g in family_groups.values())!=len(anchor['members']):
                raise ValueError('A geometry stratum silently dropped a GT member')
        roles={}
        for name,path in evaluations:
            p=Path(path)/'run_info.json';info=json.loads(p.read_text()) if p.exists() else {}
            scope=info.get('checkpoint_training_scope') or {}
            budget=scope.get('requested_steps')
            roles[name]=dict(checkpoint_step=info.get('checkpoint_step'),requested_training_steps=budget,
                scope='screen_only' if budget is not None and budget<20000 else 'full-budget training context or unspecified')
        result=dict(status='valid',reference_geometry_assignment=reference,
            full_gt_members=len(anchor['members']),frames=len(anchor['frames']),
            episode_count=len({k[0] for k in anchor['frames']}),models=list(loaded),model_roles=roles,
            population_intersection_performed=False,groups=groups,members=records,
            radial_geometry_definition='n=p_GT/||p_GT||; signed dR/drho=dot(n,d_observed); sensitivity=abs(dR/drho)',
            radial_angle_unit='degrees; acos of signed dot, from 0 to 180',
            fixed_sensitivity_thresholds=[0.,.25,.5,.75,1.],
            threshold_convention='left-inclusive/right-exclusive except last bin includes 1; unmatched GT remains in missing_observed_ray',
            scope='GT-labelled offline geometry diagnostic only; no value used by inference, training, matching or checkpoint selection',
            inference_input_changed=False,missing_members_dropped=False,
            episode_count_caveat='An episode may occur in several strata; member counts form a disjoint complete partition per family',
            comparison_caveat='Predictions share identical fixed members/observations. Mixed training budgets do not establish a matched-budget architecture effect.',
            frozen_proxy_assignment_path=str(Path(assignment_path).resolve()) if assignment_path else None,
            frozen_proxy_assignment_sha256=hashlib.sha256(Path(assignment_path).read_bytes()).hexdigest() if assignment_path else None)
    if output is not None:
        output=Path(output)
        if output.exists():raise FileExistsError('Existing stratification result is protected')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evaluations',nargs='+',help='NAME=/path/to/saved/evaluation')
    p.add_argument('--reference');p.add_argument('--assignment',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-frozen-test-evaluations',action='store_true')
    a=p.parse_args(argv);pairs=[]
    for value in a.evaluations:
        if '=' not in value:p.error('Each evaluation must be NAME=PATH')
        pairs.append(tuple(value.split('=',1)))
    result=compare_geometry_strata(pairs,a.reference,a.assignment,a.output,a.allow_frozen_test_evaluations)
    print(json.dumps(dict(status=result['status'],output=str(a.output),
        truth_members=result.get('full_gt_members'),families=list(result['groups']))))
    return int(result['status']!='valid')


if __name__=='__main__':raise SystemExit(main())
