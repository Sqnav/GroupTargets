"""Offline-only one-to-one evaluation and episode-paired uncertainty.

GT is opened only after a normal prediction file has been persisted. Matching
is a declared evaluation convention, never an association input. Missing
localizations count against all-GT success rates and are not assigned zero
error in the conditional localization RMSE.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import load_manifest, load_labels
from .runtime import file_sha256, write_json, jsonable


def match_positions(predictions, targets, valid=None, max_distance=None):
    predictions = np.asarray(predictions, float).reshape(-1,3)
    targets = np.asarray(targets, float).reshape(-1,3)
    active = np.isfinite(predictions).all(axis=1)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != (len(predictions),): raise ValueError('Invalid prediction valid mask')
        active &= mask
    indices = np.flatnonzero(active)
    if not len(indices) or not len(targets): return []
    distances = np.linalg.norm(predictions[indices,None,:]-targets[None,:,:], axis=-1)
    if max_distance is not None:
        if not np.isfinite(max_distance) or max_distance <= 0: raise ValueError('Invalid frozen matching distance')
        # Explicit dummy alternatives maximize admissible match count first.
        penalty = (float(max_distance)+1)*(len(indices)+len(targets)+1)
        cost = np.where(distances <= max_distance, distances, 2*penalty)
        cost = np.column_stack((cost, np.full((len(indices),len(indices)), penalty)))
    else: cost = distances
    rows, cols = linear_sum_assignment(cost)
    return [(int(indices[r]),int(c),float(distances[r,c])) for r,c in zip(rows,cols)
            if c < len(targets) and (max_distance is None or distances[r,c] <= max_distance)]


def metrics(rows):
    errors = np.asarray([r['error_m'] for r in rows if r['error_m'] is not None],float)
    total = len(rows)
    n = len(errors)
    return dict(gt_members=total, localized_members=n, no_localization_members=total-n,
        coverage=n/total if total else None,
        position_rmse_3d_m=float(np.sqrt(np.mean(errors**2))) if n else None,
        mae_m=float(np.mean(errors)) if n else None,
        p95_m=float(np.quantile(errors,.95)) if n else None,
        p99_m=float(np.quantile(errors,.99)) if n else None,
        over_5m=int(np.count_nonzero(errors>5)), over_10m=int(np.count_nonzero(errors>10)),
        all_gt_success={str(t)+'m':float(np.count_nonzero(errors<=t)/total) if total else None for t in (1,3,5)},
        sse_m2=float(np.sum(errors**2)))


def visual_prior_rows(frame, labels, model_config=None):
    """Offline angular assignment; final 3D errors do not select prior targets."""
    if 'visual_depth_mean_m' not in frame and 'visual_depth_std_m' not in frame:return []
    from .supervision import match_direct_ray_labels
    required=('visual_depth_mean_m','visual_depth_std_m','ray_origins','ray_directions',
              'ray_depth_bounds','range_scale_m','valid')
    if any(key not in frame for key in required):raise ValueError('Incomplete exported visual-prior geometry')
    features={key:frame[key] for key in ('ray_origins','ray_directions','ray_depth_bounds','range_scale_m')}
    features['ray_origins']=np.asarray(features['ray_origins'],float).reshape(-1,3)
    features['ray_directions']=np.asarray(features['ray_directions'],float).reshape(-1,3)
    features['ray_depth_bounds']=np.asarray(features['ray_depth_bounds'],float).reshape(-1,2)
    features['ray_valid']=np.asarray(frame['valid'],bool)
    mean=np.asarray(frame['visual_depth_mean_m'],float);std=np.asarray(frame['visual_depth_std_m'],float)
    if mean.shape!=(len(features['ray_valid']),) or std.shape!=mean.shape:
        raise ValueError('Visual-prior outputs must align with exported rays')
    matched=match_direct_ray_labels(features,labels.positions,model_config)
    rows=[]
    for row,target in zip(matched['rows'],matched['depths_m']):
        if not np.isfinite(mean[row]) or not np.isfinite(std[row]) or std[row]<=0:
            raise ValueError('Visual-prior uncertainty must be positive finite metres')
        error=float(mean[row]-target);sigma=float(std[row])
        rows.append(dict(error_m=error,std_m=sigma,
            gaussian_nll=.5*(error/sigma)**2+float(np.log(sigma))+.5*float(np.log(2*np.pi))))
    return rows


def visual_prior_metrics(rows):
    n=len(rows)
    if not n:return dict(scored_members=0,depth_rmse_m=None,mean_std_m=None,gaussian_nll=None,
                         coverage_1sigma=None,coverage_2sigma=None)
    errors=np.asarray([row['error_m'] for row in rows]);std=np.asarray([row['std_m'] for row in rows])
    return dict(scored_members=n,depth_rmse_m=float(np.sqrt(np.mean(errors**2))),
        mean_std_m=float(np.mean(std)),gaussian_nll=float(np.mean([row['gaussian_nll'] for row in rows])),
        coverage_1sigma=float(np.mean(np.abs(errors)<=std)),coverage_2sigma=float(np.mean(np.abs(errors)<=2*std)),
        matching='Offline public-ray angular/support assignment, independent of final 3D matching',
        interpretation='Empirical uncertainty diagnostics; nominal Gaussian confidence is not guaranteed')


def episode_bootstrap(rows, *, draws=2000, seed=20261126):
    episodes = sorted({r['episode_id'] for r in rows})
    if len(episodes)<2:
        return dict(episode_count=len(episodes), draws=0, reason='At least two independent episodes required', intervals=None)
    if draws < 20: raise ValueError('At least 20 bootstrap draws required')
    groups = [[r for r in rows if r['episode_id']==episode] for episode in episodes]
    samples = {key:[] for key in ('position_rmse_3d_m','mae_m','p95_m','p99_m')}
    rng = np.random.default_rng(seed)
    for chosen in rng.integers(len(groups), size=(draws,len(groups))):
        result = metrics([row for index in chosen for row in groups[index]])
        for key in samples:
            if result[key] is not None: samples[key].append(result[key])
    return dict(episode_count=len(episodes), draws=draws, seed=seed, resampling_unit='episode',
        intervals={key:{'95%':np.quantile(values,[.025,.975]).tolist(),
                        '97.5%':np.quantile(values,[.0125,.9875]).tolist()}
                   for key,values in samples.items() if values},
        p99_note='P99 intervals reflect finite independent episodes; a point decrease alone does not establish tail safety')


def evaluate(config, predictions_path, *, manifest=None, output_dir=None):
    path = Path(predictions_path).resolve()
    # This read precedes every label access.
    prediction_hash = file_sha256(path)
    predictions = json.loads(path.read_text())
    if predictions.get('schema') != 'rvloc.predictions.v1' or not predictions.get('normal_inference') or predictions.get('labels_opened') is not False:
        raise ValueError('Evaluation requires persisted sensor-only predictions')
    if predictions.get('split') not in ('train','dev'): raise ValueError('Historical TEST is reserved')
    if manifest is None:
        if config.manifest is None: raise ValueError('Dataset manifest not configured')
        manifest = load_manifest(config.manifest)
    if predictions.get('manifest_sha256') != manifest.sha256: raise ValueError('Prediction/data identity mismatch')
    frames = {}
    for frame in predictions['frames']:
        key=(frame['episode_id'],frame['frame_idx'])
        if key in frames: raise ValueError('Duplicate predicted frame')
        frames[key]=frame
    records=list(manifest.frames(predictions['split']))
    expected={(r.episode_id,r.frame_idx) for r in records}
    if set(frames)!=expected: raise ValueError('Predictions must cover exactly the declared evaluation frames')
    rows=[]; prior_rows=[]; frame_results=[]; extra_predictions=0
    max_distance=config.raw.get('evaluation',{}).get('match_max_distance_m')
    declared=predictions.get('evaluation_contract')
    if declared != dict(matching='same_per_frame_one_to_one_euclidean',match_max_distance_m=max_distance):
        raise ValueError('Evaluation matching must equal the contract frozen before inference')
    for record in records:
        frame=frames[(record.episode_id,record.frame_idx)]
        if not np.isclose(record.physical_time_s,frame['physical_time_s'],atol=1e-8): raise ValueError('Prediction time mismatch')
        labels=load_labels(manifest,record)
        prior_rows.extend(visual_prior_rows(frame,labels,config.raw.get('model',{})))
        points=np.asarray(frame['positions'],float).reshape(-1,3)
        pairs=match_positions(points,labels.positions,frame['valid'],max_distance)
        by_target={j:(i,error) for i,j,error in pairs}
        extra_predictions+=len(points)-len(pairs)
        for j, identity in enumerate(labels.member_ids):
            match=by_target.get(j)
            index,error=match if match is not None else (None,None)
            rows.append(dict(episode_id=record.episode_id,frame_idx=record.frame_idx,
                physical_time_s=record.physical_time_s,gt_member_id=identity,
                prediction_index=index,estimated_member_id=frame['member_ids'][index] if index is not None else None,
                error_m=error,gt_position_m=labels.positions[j],
                prediction_m=points[index] if index is not None else None))
        frame_results.append(dict(episode_id=record.episode_id,frame_idx=record.frame_idx,
            gt_members=len(labels.positions), outputs=len(points), matched=len(pairs),
            unmatched_outputs=len(points)-len(pairs), runtime_s=frame['runtime_s']))
    settings=config.raw.get('evaluation',{})
    summary=metrics(rows)
    summary['unmatched_outputs']=extra_predictions
    summary['matching']='Offline per-frame Hungarian Euclidean 3D, identical convention for all seven groups; not GT association during inference'
    summary['matching_max_distance_m']=max_distance
    summary['conditional_rmse_warning']='RMSE uses located GT members only; compare coverage, false outputs, and paired common-member RMSE as well'
    summary['uncertainty']=episode_bootstrap(rows,draws=settings.get('bootstrap_draws',2000),seed=settings.get('bootstrap_seed',20261126))
    episodes={e:metrics([r for r in rows if r['episode_id']==e]) for e in manifest.splits[predictions['split']]}
    output=path.parent if output_dir is None else Path(output_dir)
    result=dict(schema='rvloc.evaluation.v1',method_id=predictions['method_id'],split=predictions['split'],
        neural_architecture=predictions.get('neural_architecture'),
        visual_prior_metrics=visual_prior_metrics(prior_rows),
        manifest_sha256=manifest.sha256,predictions_sha256=prediction_hash,predictions_path=str(path),
        config_sha256=config.sha256,summary=summary,episodes=episodes,rows=jsonable(rows),
        frame_results=frame_results,cost=predictions['cost'])
    write_json(output/'evaluation.json',result)
    write_json(output/'localization_metrics.json',summary)
    write_json(output/'visual_prior_metrics.json',result['visual_prior_metrics'])
    write_json(output/'episode_level_results.json',episodes)
    write_json(output/'failure_cases.json',[r for r in rows if r['error_m'] is None or r['error_m']>5])
    return dict(status='offline_evaluation_completed',evaluation=str(output/'evaluation.json'),**summary)


def paired_compare(first, second, *, draws=2000, seed=20261126):
    """Second minus first, retaining common located GT identities explicitly."""
    def load(value): return json.loads(Path(value).read_text()) if isinstance(value,(str,Path)) else value
    a,b=load(first),load(second)
    if a.get('schema')!='rvloc.evaluation.v1' or b.get('schema')!='rvloc.evaluation.v1':
        raise ValueError('Expected rvloc.evaluation.v1 files')
    for key in ('schema','manifest_sha256','split'):
        if a.get(key)!=b.get(key): raise ValueError('Paired evaluations differ: '+key)
    if a.get('summary',{}).get('matching_max_distance_m') != b.get('summary',{}).get('matching_max_distance_m'):
        raise ValueError('Paired evaluations use different matching distances')
    def keyed(value): return {(r['episode_id'],r['frame_idx'],r['gt_member_id']):r for r in value['rows']}
    aa,bb=keyed(a),keyed(b)
    if aa.keys()!=bb.keys(): raise ValueError('Paired evaluations use different GT populations')
    common=[key for key in aa if aa[key]['error_m'] is not None and bb[key]['error_m'] is not None]
    episodes=sorted({key[0] for key in aa})
    grouped={episode:(np.asarray([aa[key]['error_m'] for key in common if key[0]==episode]),
                      np.asarray([bb[key]['error_m'] for key in common if key[0]==episode])) for episode in episodes}
    def statistic(chosen):
        ea=np.concatenate([grouped[episode][0] for episode in chosen])
        eb=np.concatenate([grouped[episode][1] for episode in chosen])
        if not len(ea):return None
        return [float(np.sqrt(np.mean(eb**2))-np.sqrt(np.mean(ea**2))),
                float(np.quantile(eb,.95)-np.quantile(ea,.95)),
                float(np.quantile(eb,.99)-np.quantile(ea,.99))]
    point=statistic(episodes)
    samples=[]
    if len(episodes)>1:
        rng=np.random.default_rng(seed)
        for chosen in rng.integers(len(episodes),size=(draws,len(episodes))):
            value=statistic([episodes[i] for i in chosen])
            if value is not None:samples.append(value)
    intervals={}
    if samples:
        for i,key in enumerate(('rmse_delta_m','p95_delta_m','p99_delta_m')):
            values=np.asarray(samples)[:,i]
            intervals[key]={'95%':np.quantile(values,[.025,.975]).tolist(),'97.5%':np.quantile(values,[.0125,.9875]).tolist()}
    return dict(schema='rvloc.paired_comparison.v1',direction='second minus first',
        manifest_sha256=a['manifest_sha256'],split=a['split'],methods=[a['method_id'],b['method_id']],
        all_gt=len(aa),common_located=len(common),
        first_only_located=sum(aa[k]['error_m'] is not None and bb[k]['error_m'] is None for k in aa),
        second_only_located=sum(bb[k]['error_m'] is not None and aa[k]['error_m'] is None for k in aa),
        common_member_deltas=dict(zip(('rmse_delta_m','p95_delta_m','p99_delta_m'),point)) if point is not None else None,
        episode_intervals=intervals,episode_count=len(episodes),draws=len(samples),seed=seed,
        note='Common-member deltas and whole-population coverage must be read together; no independent TEST evidence is implied')
