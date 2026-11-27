"""Compare saved evaluations, with paired whole-episode bootstrap uncertainty.

No model, dataset, IQ, image or label-store reader is imported. The population
is never silently intersected. Conditional RMSE gains are not accepted when a
model changes which members receive finite locations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

THRESHOLDS=(1,3,5)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _metric(values):
    a=np.asarray(values,float)
    if not len(a):return dict(count=0,rmse=None,mean=None,p95=None)
    return dict(count=len(a),rmse=float(np.sqrt(np.mean(a*a))),mean=float(a.mean()),
                p95=float(np.quantile(np.abs(a),.95)))


def _load(name,path,allow_test=False):
    path=Path(path).resolve()
    info=json.loads((path/'run_info.json').read_text()) if (path/'run_info.json').is_file() else {}
    if info.get('split')=='test' and not allow_test:
        raise ValueError('Test evaluation outputs require an explicitly frozen-final comparison')
    summary=json.loads((path/'summary.json').read_text())
    frames=_rows(path/'frames.jsonl');all_members=_rows(path/'members.jsonl')
    fkeys=[(r['episode_id'],int(r['frame_idx'])) for r in frames]
    mkeys=[(r['episode_id'],int(r['frame_idx']),int(r['truth_index'])) for r in all_members]
    if len(set(fkeys))!=len(fkeys) or len(set(mkeys))!=len(mkeys):
        raise ValueError(name+': duplicate frame/member keys')
    frame_map=dict(zip(fkeys,frames));members={};by_frame=defaultdict(list)
    for key,row in zip(mkeys,all_members):
        if key[:2] not in frame_map:
            raise ValueError(name+': undeclared member frame')
        if not isinstance(row.get('matched'),bool):
            raise ValueError(name+': explicit observed-ray matching flag required')
        status=row['status'];error=row.get('error_3d_m')
        if status=='localized' and (error is None or not np.isfinite(error) or error<0):
            raise ValueError(name+': a localized member has no finite nonnegative error')
        if status=='localized' and not row['matched']:
            raise ValueError(name+': a supposedly localized member is not fixed-matched')
        keep=('episode_id','frame_idx','truth_index','matched','status','detection_index',
              'truth_position_base_m','error_3d_m','base_range_error_m',
              'base_azimuth_error_deg','base_elevation_error_deg')
        members[key]={k:row.get(k) for k in keep}
        by_frame[key[:2]].append((key,members[key]))
    for key,frame in frame_map.items():
        expected=int(frame['truth_count'])
        if min(expected,int(frame['detection_count']))<0:
            raise ValueError(name+': negative declared population')
        actual={k[2] for k,_ in by_frame[key]}
        if actual!=set(range(expected)):
            raise ValueError(name+': GT members were removed or duplicated within a frame')
        matches=[row for _,row in by_frame[key] if row['matched']]
        matched=len(matches)
        if 'matched_count' in frame and matched!=frame['matched_count']:
            raise ValueError(name+': frame/member association count differs')
        if matched>int(frame['detection_count']):
            raise ValueError(name+': more associations than observed detections')
        assigned=[r['detection_index'] for r in matches]
        if (any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<int(frame['detection_count']) for i in assigned)
                or len(set(assigned))!=len(assigned)):
            raise ValueError(name+': invalid or duplicate fixed detection assignments')
    localized={k for k,r in members.items() if r['status']=='localized'}
    valid=[members[k] for k in localized]
    errors=[r['error_3d_m'] for r in valid]
    position=_metric(errors)
    count=len(members);detected=sum(int(f['detection_count']) for f in frames)
    matched=sum(r['matched'] for r in members.values())
    if summary.get('truth_members',count)!=count:
        raise ValueError(name+': saved summary truth population differs from member rows')
    if summary.get('localized_matched_members',len(valid))!=len(valid):
        raise ValueError(name+': saved summary localized population differs from member rows')
    expected=summary.get('member_position_error_m',{}).get('rmse')
    if expected is not None and (position['rmse'] is None or not np.isclose(expected,position['rmse'],atol=1e-8,rtol=1e-8)):
        raise ValueError(name+': saved RMSE does not equal actual member outputs')
    table=dict(name=name,evaluation_dir=str(path),frames=len(frames),truth_members=count,
        detection_instances=detected,matched_detection_instances=matched,
        localized_matched_members=len(valid),unlocalized_or_unmatched_members=count-len(valid),
        matched_detection_rate=matched/count if count else None,
        localization_effective_rate=len(valid)/count if count else None,
        conditional_member_rmse_m=position['rmse'],conditional_member_mean_error_m=position['mean'],
        conditional_member_p95_m=position['p95'],
        failure_counts=dict(Counter(r['status'] for r in members.values() if r['status']!='localized')),
        source_summary_sha256=_sha(path/'summary.json'),source_members_sha256=_sha(path/'members.jsonl'),
        source_frames_sha256=_sha(path/'frames.jsonl'),
        checkpoint=info.get('checkpoint'),checkpoint_sha256=info.get('checkpoint_sha256'),
        checkpoint_step=info.get('checkpoint_step'),split=info.get('split'),
        inference_seconds_per_frame=summary.get('inference_seconds_per_frame'),
        gpu_peak_allocated_bytes=summary.get('gpu_peak_allocated_bytes'),
        gpu_peak_reserved_bytes=summary.get('gpu_peak_reserved_bytes'))
    for t in THRESHOLDS:
        table[f'success_count_{t}m']=sum(e<=t for e in errors)
        table[f'success_ratio_all_gt_{t}m']=sum(e<=t for e in errors)/count if count else None
    for field,label in (('base_range_error_m','range_error_m'),
                        ('base_azimuth_error_deg','azimuth_error_deg'),
                        ('base_elevation_error_deg','elevation_error_deg')):
        values=[r[field] for r in valid if r[field] is not None]
        table[label]=_metric(values)
    return dict(table=table,frames=frame_map,members=members,localized=localized)


def _check_population(reference,model):
    errors=[];name=model['table']['name']
    if set(reference['frames'])!=set(model['frames']):
        errors.append(name+': frame population differs; no primary intersection comparison performed')
    if set(reference['members'])!=set(model['members']):
        errors.append(name+': full GT member population differs; no primary intersection comparison performed')
    if errors:return errors
    for key,frame in reference['frames'].items():
        other=model['frames'][key]
        for field in ('truth_count','detection_count'):
            if frame[field]!=other[field]:errors.append(f'{name}: {field} differs at {key}')
        # Numerical coordinate roundoff is allowed; observations themselves
        # cannot change. Bearing matching maps are checked independently below.
        for field,tolerance in (('bbox_xyxy',1e-6),('ray_origin_base_m',1e-6),('ray_direction_base',1e-7),
                                ('ray_depth_bounds_m',1e-6),('observer_position_base_m',1e-6)):
            if frame.get(field) is not None and other.get(field) is not None:
                a,b=np.asarray(frame[field]),np.asarray(other[field])
                if a.shape!=b.shape or not np.allclose(a,b,rtol=0,atol=tolerance):
                    errors.append(f'{name}: observed {field} differs at {key}')
    for key,row in reference['members'].items():
        other=model['members'][key]
        if row['matched']!=other['matched'] or row['detection_index']!=other['detection_index']:
            errors.append(f'{name}: fixed observed association differs at {key}')
        if row['truth_position_base_m'] is not None and other['truth_position_base_m'] is not None:
            if not np.allclose(row['truth_position_base_m'],other['truth_position_base_m'],rtol=0,atol=1e-6):
                errors.append(f'{name}: scoring truth positions differ at {key}')
    return errors


def _interval(values):
    values=np.asarray(values,float)
    valid=values[np.isfinite(values)]
    return dict(lower=float(np.quantile(valid,.025)) if len(valid) else None,
                upper=float(np.quantile(valid,.975)) if len(valid) else None,
                valid_draws=len(valid),invalid_draws=len(values)-len(valid))


def _episode_totals(model,episodes):
    counts=np.zeros(len(episodes),int);localized=np.zeros(len(episodes),int)
    ssq=np.zeros(len(episodes),float);success={t:np.zeros(len(episodes),int) for t in THRESHOLDS}
    index={e:i for i,e in enumerate(episodes)}
    for key,row in model['members'].items():
        i=index[key[0]];counts[i]+=1
        if row['status']=='localized':
            error=row['error_3d_m'];localized[i]+=1;ssq[i]+=error*error
            for t in THRESHOLDS:success[t][i]+=error<=t
    return dict(truth_count=counts,localized_count=localized,squared_error_sum=ssq,success_count=success)


def compare_evaluations(evaluations,references=None,draws=2000,seed=20261122,allow_test=False):
    """Return strict-population tables and paired member-weighted episode CIs."""
    if isinstance(draws,bool) or not isinstance(draws,int) or draws<1:
        raise ValueError('A positive integer bootstrap draw budget is required')
    if isinstance(evaluations,dict):evaluations=list(evaluations.items())
    else:evaluations=list(evaluations)
    if not evaluations or len({name for name,_ in evaluations})!=len(evaluations):
        raise ValueError('At least one uniquely named evaluation required')
    loaded={}
    for name,path in evaluations:
        try:loaded[name]=_load(name,path,allow_test)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            return dict(status='invalid',population_checks=dict(status='invalid',errors=[f'{name}: {type(exc).__name__}: {exc}']),
                input_evaluations=[dict(name=n,path=str(p)) for n,p in evaluations],
                models=[r['table'] for r in loaded.values()],references=list(references or [evaluations[0][0]]),comparisons=[],
                bootstrap=dict(unit='whole episode',draws=draws,seed=seed),
                scope='only saved evaluation outputs; no intersection or inference performed after invalid input',
                promotion_caveat='No precision gain can be inferred from an incomplete or invalid evaluation population.')
    names=list(loaded);references=list(references or names[:1])
    if any(name not in loaded for name in references):raise ValueError('Unknown reference model')
    errors=[]
    for model in list(loaded.values())[1:]:errors.extend(_check_population(loaded[names[0]],model))
    result=dict(status='invalid' if errors else 'valid',population_checks=dict(status='invalid' if errors else 'passed',errors=errors),
        models=[loaded[n]['table'] for n in names],references=references,comparisons=[],
        bootstrap=dict(unit='whole episode, sampled with replacement; every member/frame in a drawn episode retained',
            draws=draws,seed=seed,confidence=.95,interval='paired percentile',
            metric_weighting='pooled member errors/success counts after each episode-block resample, not an average of episode RMSEs'),
        scope='only already generated evaluation outputs; no dataset, raw label, image, IQ or model access',
        conditional_rmse_caveat='RMSE covers finite fixed-matched localized members; failed members remain in all-GT success and effective-rate denominators',
        promotion_caveat='The minimum statistical gate is not final approval: training-seed stability, mode-specific failure strata and untouched frozen-final test evaluation remain necessary.')
    if errors:return result
    episodes=sorted({key[0] for key in loaded[names[0]]['frames']})
    if not episodes or not loaded[names[0]]['table']['truth_members']:
        result['status']='invalid';result['population_checks']['status']='invalid'
        result['population_checks']['errors'].append('No episodes or no GT members in declared evaluation');return result
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(episodes),size=(draws,len(episodes)))
    totals={name:_episode_totals(model,episodes) for name,model in loaded.items()}
    result['bootstrap'].update(episode_count=len(episodes),episode_ids=episodes,
        draw_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
    bootstrap={}
    for name,total in totals.items():
        n=total['truth_count'][indices].sum(1)
        localized=total['localized_count'][indices].sum(1)
        with np.errstate(divide='ignore',invalid='ignore'):
            bootstrap[name]=dict(rmse=np.sqrt(total['squared_error_sum'][indices].sum(1)/localized),
                effective_rate=localized/n,
                coverage={t:total['success_count'][t][indices].sum(1)/n for t in THRESHOLDS})
    for reference_name in references:
        reference=loaded[reference_name];a=reference['table']
        for name,model in loaded.items():
            if name==reference_name:continue
            b=model['table'];same=reference['localized']==model['localized']
            row=dict(reference=reference_name,model=name,delta_definition='model minus reference; negative RMSE is better, positive coverage/effective rate is better',
                identical_localized_member_set=same,
                dropped_reference_localized_members=len(reference['localized']-model['localized']),
                added_localized_members=len(model['localized']-reference['localized']),
                localized_member_sets_intersected_for_primary_metrics=False,
                conditional_rmse_native_difference_m=(b['conditional_member_rmse_m']-a['conditional_member_rmse_m'])
                    if a['conditional_member_rmse_m'] is not None and b['conditional_member_rmse_m'] is not None else None)
            if same:
                row['paired_rmse_delta_m']=row['conditional_rmse_native_difference_m']
                row['paired_rmse_delta_95ci_m']=_interval(bootstrap[name]['rmse']-bootstrap[reference_name]['rmse'])
                row['rmse_comparison_status']='strictly_paired_identical_localized_population'
            else:
                row['paired_rmse_delta_m']=None;row['paired_rmse_delta_95ci_m']=None
                row['rmse_comparison_status']='not_comparable_localized_sets_differ; native conditional RMSE decrease cannot establish improvement'
            for t in THRESHOLDS:
                row[f'coverage_delta_{t}m']=b[f'success_ratio_all_gt_{t}m']-a[f'success_ratio_all_gt_{t}m']
                row[f'coverage_delta_{t}m_95ci']=_interval(bootstrap[name]['coverage'][t]-bootstrap[reference_name]['coverage'][t])
            row['effective_rate_delta']=b['localization_effective_rate']-a['localization_effective_rate']
            row['effective_rate_delta_95ci']=_interval(bootstrap[name]['effective_rate']-bootstrap[reference_name]['effective_rate'])
            upper=(row['paired_rmse_delta_95ci_m'] or {}).get('upper')
            row['passes_minimum_statistical_gate']=bool(same and upper is not None and upper<0 and row['effective_rate_delta']>=0)
            result['comparisons'].append(row)
    return result


def format_comparison(result):
    lines=['Saved single-frame localization comparison: '+result['status'],
        'All-GT coverage includes every miss/rejection; conditional errors cover finite fixed-matched localized members.',
        'model | frames | GT | localized | RMSE(m) | mean(m) | P95(m) | <=1m | <=3m | <=5m | effective rate']
    def number(v):return 'NA' if v is None else f'{v:.4f}'
    for m in result['models']:
        lines.append(f'{m["name"]} | {m["frames"]} | {m["truth_members"]} | {m["localized_matched_members"]} | '
            f'{number(m["conditional_member_rmse_m"])} | {number(m["conditional_member_mean_error_m"])} | '
            f'{number(m["conditional_member_p95_m"])} | {number(m["success_ratio_all_gt_1m"])} | '
            f'{number(m["success_ratio_all_gt_3m"])} | {number(m["success_ratio_all_gt_5m"])} | '
            f'{number(m["localization_effective_rate"])}')
    lines.extend('INVALID: '+error for error in result['population_checks']['errors'])
    for row in result['comparisons']:
        ci=row['paired_rmse_delta_95ci_m']
        text=(f'{row["model"]} - {row["reference"]}: RMSE delta={number(row["paired_rmse_delta_m"])} m; '
              f'95% episode CI=[{number(ci["lower"])}, {number(ci["upper"])}]' if ci else
              f'{row["model"]} - {row["reference"]}: RMSE NOT COMPARABLE because localized sets differ')
        lines.append(text+f'; minimum statistical gate={row["passes_minimum_statistical_gate"]}')
        for t in THRESHOLDS:
            interval=row[f'coverage_delta_{t}m_95ci']
            lines.append(f'  all-GT <= {t}m coverage delta={number(row[f"coverage_delta_{t}m"])}; '
                f'95% episode CI=[{number(interval["lower"])}, {number(interval["upper"])}]')
    lines.append(result['promotion_caveat'])
    return '\n'.join(lines)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evaluations',nargs='+',help='NAME=/path/to/saved/evaluation')
    p.add_argument('--reference',action='append')
    p.add_argument('--draws',type=int,default=2000);p.add_argument('--seed',type=int,default=20261122)
    p.add_argument('--allow-frozen-test-evaluations',action='store_true')
    p.add_argument('--output',type=Path,required=True,help='JSON output; sibling .txt is written too')
    a=p.parse_args(argv);pairs=[]
    if a.output.suffix.lower()!='.json':p.error('--output must name a .json file')
    for item in a.evaluations:
        if '=' not in item:p.error('Each evaluation must be NAME=PATH')
        name,path=item.split('=',1);pairs.append((name,path))
    if a.output.exists() or a.output.with_suffix('.txt').exists():raise FileExistsError('Existing comparison output is protected')
    result=compare_evaluations(pairs,a.reference,a.draws,a.seed,a.allow_frozen_test_evaluations)
    result['created_at']=datetime.now(timezone(timedelta(hours=8))).isoformat()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    text=format_comparison(result);a.output.with_suffix('.txt').write_text(text+'\n');print(text)
    return int(result['status']=='invalid')


if __name__=='__main__':raise SystemExit(main())
