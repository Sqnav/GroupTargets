"""Label-free validation diagnostic of native joint-power query availability.

Only measured/current-frame public inputs and authenticated caches are used.
Checkpoint export is an explicit offline artifact operation; the actual probe
loads only the separately verified, label-free deployment copies. No model,
configuration, selection rule, default, or held-out test is changed.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
import torch

from .cache import load_volume
from .data import _strict_sample
from .export_deployment import export_checkpoint, verify_deployment_receipt
from .versions import load_frozen_study_model
from .visual_data import VisualSensorReader
from .volume import positions_to_ruv, sample_power_volume
from depth_localization.geometry import candidate_geometry
from rvwm.data import Dataset

QUANTILES=(0.,.01,.05,.5,.9,.95,.99,.999,1.)
MODEL_KEYS=('ra_re_screen','joint_center_screen','joint_cross7_screen')


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def _plain(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().numpy()
    return np.asarray(value)


def quantiles(values):
    values=np.asarray(values).ravel();finite=values[np.isfinite(values)]
    return dict(count=int(values.size),finite_count=int(finite.size),
        zero_count=int((finite==0).sum()),negative_count=int((finite<0).sum()),
        quantiles={str(q):float(v) for q,v in zip(QUANTILES,np.quantile(finite,QUANTILES))} if len(finite) else {})


def shared_power_scale(features):
    """Exact float32 normalization used by the unchanged global RA/RE branch."""
    ra=torch.as_tensor(features['ra_power'],dtype=torch.float32)
    re=torch.as_tensor(features['re_power'],dtype=torch.float32)
    av=torch.as_tensor(features['ra_valid'],dtype=torch.bool)
    ev=torch.as_tensor(features['re_valid'],dtype=torch.bool)
    return float(((ra.masked_fill(~av,0.).sum()+re.masked_fill(~ev,0.).sum())/
                  (av.sum()+ev.sum()).clamp_min(1)).clamp_min(1e-20))


def query_cross7(features,volume,depths,offsets):
    """Use precisely the production cross7 native-bin offsets and masks."""
    tf=lambda x:torch.as_tensor(np.array(x,copy=True),dtype=torch.float32)
    hypotheses=tf(depths)[:,None]+tf(offsets)[None]
    origin=tf(features['ray_origins']);direction=tf(features['ray_directions'])
    geometry=candidate_geometry(origin,direction,hypotheses,tf(features['ray_depth_bounds']),
                                torch.as_tensor(features['ray_valid'],dtype=torch.bool))
    ruv=positions_to_ruv(geometry['positions']);valid=geometry['valid']&ruv['valid']
    axes=[tf(volume[k]) for k in ('range_m','u_axis','v_axis')]
    dr,du,dv=[float(a[1]-a[0]) for a in axes]
    delta=tf([[0,0,0],[-dr,0,0],[dr,0,0],[0,-du,0],[0,du,0],[0,0,-dv],[0,0,dv]])
    sampled=sample_power_volume(tf(volume['volume_power']),
        ranges=ruv['ranges'][...,None]+delta[:,0],u=ruv['u'][...,None]+delta[:,1],
        v=ruv['v'][...,None]+delta[:,2],range_axis=axes[0],u_axis=axes[1],v_axis=axes[2],
        angular_valid=volume['angular_valid'],candidate_valid=valid[...,None].expand(*valid.shape,7))
    return dict(power=_plain(sampled['values'][...,0]),valid=_plain(sampled['valid']),
        geometry_valid=_plain(geometry['valid']),depths=_plain(hypotheses),
        ruv=np.stack([_plain(ruv[k]) for k in ('ranges','u','v')],-1),
        offsets_ruv=_plain(delta))


def query_statistics(power,valid,power_scale,background_quantiles):
    """Descriptive statistics; empirical volume percentiles are NOT noise labels."""
    p=np.asarray(power,dtype=np.float64)/float(power_scale);mask=np.asarray(valid,bool)
    if p.shape!=mask.shape or p.ndim!=3 or p.shape[-1]!=7:
        raise ValueError('Expected power and validity [detections,candidates,7]')
    center=p[...,0];cm=mask[...,0];any_valid=mask.any(-1)
    maximum=np.where(mask,p,-np.inf).max(-1);maximum=np.where(any_valid,maximum,0.)
    # Adapter multiplies by center-valid, so neighbouring power cannot repair
    # an invalid center. Do not report those candidates as model-usable.
    usable=cm
    floor=np.finfo(np.float32).tiny
    ratio=maximum[usable]/np.maximum(center[usable],floor)
    result=dict(candidate_count=int(cm.size),center_valid_count=int(cm.sum()),
        center_valid_fraction=float(cm.mean()) if cm.size else None,
        all_cross7_valid_fraction=float(mask.all(-1).mean()) if cm.size else None,
        all_query_valid_fraction=float(mask.mean()) if cm.size else None,
        any_cross7_valid_fraction=float(any_valid.mean()) if cm.size else None,
        invalid_center_with_valid_neighbour_count=int((~cm&any_valid).sum()),
        center_power_over_shared_scale=quantiles(center[usable]),
        cross7_max_over_shared_scale=quantiles(maximum[usable]),
        cross7_all_valid_power_over_shared_scale=quantiles(p[mask]),
        cross7_max_to_center_ratio=quantiles(ratio),
        cross7_max_exceeds_center_2x_fraction=float((ratio>2).mean()) if len(ratio) else None,
        cross7_max_exceeds_center_10x_fraction=float((ratio>10).mean()) if len(ratio) else None,
        center_log1p=quantiles(np.log1p(center[usable])),
        cross7_max_log1p=quantiles(np.log1p(maximum[usable])))
    result['below_empirical_physical_volume_percentile']={}
    for q,threshold in background_quantiles.items():
        result['below_empirical_physical_volume_percentile'][str(q)]=dict(
            center_fraction=float((center[usable]<=threshold).mean()) if usable.any() else None,
            cross7_max_fraction=float((maximum[usable]<=threshold).mean()) if usable.any() else None)
    contrasts={}
    for name,values in (('center',center),('cross7_max',maximum)):
        rows=[];relative=[]
        for row,m in zip(values,usable):
            x=row[m]
            if len(x)<2:continue
            rows.append(float(np.log1p(x.max())-np.log1p(x.min())))
            relative.append(float(x.max()/max(float(np.median(x)),floor)))
        contrasts[name]=dict(log1p_peak_minus_min=quantiles(rows),
                            peak_to_candidate_median_ratio=quantiles(relative))
    result['within_ray_nine_candidate_contrast']=contrasts
    return result


def _models(study,output):
    models={};receipts={}
    for key in MODEL_KEYS:
        source=study/'code'/key
        config=json.loads((study/'configs'/(key+'.json')).read_text())
        target=output/'deployment'/key
        original=study/'training'/key/'depth.rmse.best.pt'
        receipt=export_checkpoint(original,target,kind='study',config=config,source_root=source)
        path=target/'model.pt';verify_deployment_receipt(path,'study')
        models[key]=load_frozen_study_model(source,path,config,'cpu')
        receipts[key]=dict(path=str(path),source_checkpoint_path=str(original),
            source_root=str(source),checkpoint_sha256=receipt['checkpoint_sha256'],
            exact_state_dict_preserved=receipt['exact_state_dict_preserved'],
            inference_copy_has_per_sample_validation_metadata=False,
            steps=models[key].checkpoint_metadata['steps'])
    return models,receipts


def _guard_sensor_only(stack,dataset_root):
    def deny(*args,**kwargs):raise AssertionError('Supervision/native-IQ operation forbidden by power diagnostic')
    for name in ('targets','_load','observations'):
        stack.enter_context(patch.object(Dataset,name,side_effect=deny))
    original_load=np.load
    def guarded_load(path,*args,**kwargs):
        if isinstance(path,(str,Path)):
            p=Path(path).resolve()
            if p.is_relative_to(dataset_root) and p.suffix=='.npy':deny()
        return original_load(path,*args,**kwargs)
    stack.enter_context(patch.object(np,'load',side_effect=guarded_load))
    # Recomputing spectra or the cube is unnecessary and would bypass cache.
    stack.enter_context(patch('precision_study.data.spectra_from_iq',side_effect=deny))
    stack.enter_context(patch('precision_study.volume.build_power_volume',side_effect=deny))


def diagnose(study,output,episodes=150,subsample=8192):
    study=Path(study).resolve();output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise FileExistsError('Protected diagnostic output: '+str(output))
    output.mkdir(parents=True,exist_ok=True);(output/'queries').mkdir()
    started=time.monotonic();torch.set_num_threads(1)
    models,receipts=_models(study,output)
    config=models['joint_cross7_screen'].config
    frames=[];pooled=[];queries={k:{s:[] for s in ('initial_coarse9','first_refined_fine9')} for k in MODEL_KEYS}
    reader=None
    with ExitStack() as stack:
        _guard_sensor_only(stack,Path(config['dataset_root']).resolve())
        reader=VisualSensorReader(config,'val');stack.callback(reader.close)
        ids=sorted(reader.ids)
        if episodes<1 or episodes>len(ids):raise ValueError('Requested episode count exceeds frozen validation partition')
        selected=ids[:episodes]
        for n,episode in enumerate(selected):
            available=reader.frame_indices(episode,0)
            if not available:raise ValueError('No approach frame: '+episode)
            index=available[0]
            observed,row,_=_strict_sample(config,reader,episode,index)
            features=observed['features'];volume=features['joint_volume'];raw=volume['volume_power']
            scale=shared_power_scale(features)
            if volume['metadata'].get('runtime_native_iq_read') is not False:
                raise AssertionError('Expected authenticated precomputed power cache')
            u,v=np.meshgrid(volume['u_axis'],volume['v_axis'],indexing='ij')
            physical=(u*u+v*v<=1.)&np.asarray(volume['angular_valid'],bool)
            physical_values=np.asarray(raw[:,physical],np.float64).ravel()/scale
            empirical={str(q):float(np.quantile(physical_values,q)) for q in (.5,.9,.99)}
            rng=np.random.default_rng(20261122+n)
            sampled=physical_values[rng.choice(len(physical_values),min(subsample,len(physical_values)),replace=False)]
            pooled.append(sampled)
            frame=dict(episode_id=episode,frame_idx=index,phase=row['phase'],detection_count=len(features['ray_valid']),
                ray_valid_count=int(np.asarray(features['ray_valid']).sum()),
                ray_direction_norm_max_abs_error=float(np.abs(np.linalg.norm(features['ray_directions'],axis=-1)-1).max(initial=0)),
                volume_shape=list(raw.shape),volume_voxels=int(raw.size),
                volume_finite_count=int(np.isfinite(raw).sum()),volume_zero_count=int((raw==0).sum()),
                volume_negative_count=int((raw<0).sum()),
                angular_mask_valid_fraction=float(np.asarray(volume['angular_valid']).mean()),
                physical_angular_fraction=float(physical.mean()),shared_ra_re_power_scale=scale,
                physical_volume_power_over_shared_scale=quantiles(physical_values),
                empirical_background_percentiles=empirical,
                axis_limits={key:[float(np.min(volume[key])),float(np.max(volume[key]))] for key in ('range_m','u_axis','v_axis')},
                cache_receipt=volume['metadata']['cache_receipt'],cache_power_sha256=volume['metadata']['cache_power_sha256'],
                current_sensor_receipt=row,models={})
            arrays={}
            for key,model in models.items():
                inp=features if model.config['precision_radar_mode']=='joint_volume' else {k:v for k,v in features.items() if k!='joint_volume'}
                with torch.inference_mode():prediction=model(inp)
                stages=prediction['sampling_diagnostics']['stage_depths_m']
                stage_inputs=(('initial_coarse9',prediction['initial_depths'],model.depth_offsets_m),
                              ('first_refined_fine9',stages[0],model.depth_fine_offsets_m))
                frame['models'][key]={}
                for stage,depth,offset in stage_inputs:
                    with torch.inference_mode():q=query_cross7(features,volume,_plain(depth),_plain(offset))
                    stats=query_statistics(q['power'],q['valid'],scale,empirical)
                    frame['models'][key][stage]=stats
                    queries[key][stage].append(dict(power=q['power']/scale,valid=q['valid'],
                        threshold=empirical))
                    for name,value in q.items():arrays[key+'__'+stage+'__'+name]=value
                if key.startswith('joint_'):
                    # Confirm independently sampled center values reproduce the
                    # frozen model's actual forward queries for both rounds.
                    d=prediction['sampling_diagnostics'];checks=[]
                    for (stage,_,_),old_p,old_m in zip(stage_inputs,d['joint_power_per_stage'],d['joint_valid_per_stage']):
                        p=arrays[key+'__'+stage+'__power'][...,0];m=arrays[key+'__'+stage+'__valid'][...,0]
                        checks.append(dict(stage=stage,mask_equal=bool(np.array_equal(m,_plain(old_m))),
                            max_abs_difference=float(np.max(np.abs(p-_plain(old_p)),initial=0.))))
                    if any(not c['mask_equal'] or c['max_abs_difference']!=0 for c in checks):
                        raise AssertionError('Independent diagnostic differs from frozen query')
                    frame['models'][key]['independent_query_forward_equivalence']=checks
            np.savez_compressed(output/'queries'/(episode+f'_frame_{index:05d}.npz'),**arrays)
            frames.append(frame)
            _json(output/'progress.json',dict(status='running',episodes_done=n+1,episodes_total=episodes,
                elapsed_seconds=time.monotonic()-started,labels_read=False,test_set_read=False))
            if (n+1)%15==0:print(json.dumps(dict(episodes_done=n+1,total=episodes,elapsed_s=round(time.monotonic()-started,1))),flush=True)
    summary=dict(status='complete_sensor_only_diagnostic',episode_count=episodes,frame_count=len(frames),
        selected_frames='First approach frame in every selected episode; sorted frozen val episodes, no score-based selection',
        split='val',test_set_read=False,labels_read=False,gt_fields_decoded=False,
        future_observation_values_decoded=False,history_used=False,native_iq_read=False,
        model_or_configuration_changed=False,hyperparameters_selected=False,default_changed=False,
        runtime_supervision_and_iq_guards_passed=True,
        source_sha256={str(Path(__file__).resolve()):_sha(__file__)},checkpoint_receipts=receipts,
        elapsed_seconds=time.monotonic()-started,frames=frames,
        volume_pooled_physical_subsample=dict(samples_per_frame=subsample,sample_count=sum(len(x) for x in pooled),
            sampling='Uniform without replacement per frame, independent deterministic seeds 20261122+sorted episode index',
            normalized_power=quantiles(np.concatenate(pooled))),
        volume_finite_fraction=sum(f['volume_finite_count'] for f in frames)/sum(f['volume_voxels'] for f in frames),
        volume_zero_fraction=sum(f['volume_zero_count'] for f in frames)/sum(f['volume_voxels'] for f in frames),
        shared_power_scale=quantiles([f['shared_ra_re_power_scale'] for f in frames]),query_summary={},
        cautions=['Physical volume empirical percentiles include targets, observer and sidelobes; they are background proxies, not measured thermal-noise floor.',
            'Raw center and cross7 are evaluated at exactly the same nine depths within each model/stage. Different trained models may propose different depths.',
            'The cross7 adapter is multiplied by center-valid. A valid neighbour alone does not make an invalid center usable.',
            'Power is not a calibrated likelihood or target identity. No accuracy, member assignment or hyperparameter selection follows from these label-free statistics.',
            'The retained full tensor includes nonphysical angular bins; query validity additionally enforces the physical direction disk and positive forward X.'])
    for key in MODEL_KEYS:
        summary['query_summary'][key]={}
        for stage,items in queries[key].items():
            p=np.concatenate([x['power'] for x in items]);m=np.concatenate([x['valid'] for x in items])
            s=query_statistics(p,m,1.,{})
            for q in ('.5','.9','.99'):
                actual=str(float(q));parts=[];mparts=[]
                for item in items:
                    cm=item['valid'][...,0];center=item['power'][...,0]
                    maximum=np.where(item['valid'],item['power'],-np.inf).max(-1)
                    parts.extend((center[cm]<=item['threshold'][actual]).tolist())
                    mparts.extend((maximum[cm]<=item['threshold'][actual]).tolist())
                s['below_empirical_physical_volume_percentile'][actual]=dict(
                    center_fraction=float(np.mean(parts)) if parts else None,
                    cross7_max_fraction=float(np.mean(mparts)) if mparts else None)
            summary['query_summary'][key][stage]=s
    _json(output/'summary.json',summary)
    _render(summary,pooled,queries,output)
    _json(output/'progress.json',dict(status='complete',episodes_done=episodes,episodes_total=episodes,
        elapsed_seconds=time.monotonic()-started,labels_read=False,test_set_read=False))
    return summary


def _render(summary,pooled,queries,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=('tab:blue','tab:orange','tab:green');tiny=np.finfo(np.float32).tiny
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    pooled=np.concatenate(pooled)
    axes[0,0].hist(np.log10(np.maximum(pooled,tiny)),bins=100,density=True,color='grey',alpha=.6,label='physical-volume uniform samples')
    for key,color in zip(MODEL_KEYS,colors):
        items=queries[key]['initial_coarse9'];p=np.concatenate([i['power'] for i in items]);m=np.concatenate([i['valid'] for i in items]);cm=m[...,0]
        c=p[...,0][cm];maximum=np.where(m,p,-np.inf).max(-1)[cm]
        axes[0,0].hist(np.log10(np.maximum(c,tiny)),bins=100,density=True,histtype='step',label=key,color=color)
        axes[0,1].plot(np.sort(np.log1p(c)),np.linspace(0,1,len(c)),color=color,label=key+' center')
        axes[0,1].plot(np.sort(np.log1p(maximum)),np.linspace(0,1,len(maximum)),color=color,linestyle='--',label=key+' cross7 max')
    axes[0,0].set(xlabel='log10(P / unchanged RA/RE mean)',ylabel='Density',title='Initial coarse9: normalized native power')
    axes[0,1].set(xlabel='log1p(P / unchanged RA/RE mean)',ylabel='Empirical CDF',title='Same candidates: center vs native cross7 maximum')
    x=np.arange(len(MODEL_KEYS));w=.2
    for j,stage in enumerate(('initial_coarse9','first_refined_fine9')):
        data=[summary['query_summary'][k][stage] for k in MODEL_KEYS]
        axes[1,0].bar(x+(j-.5)*w,[d['center_valid_fraction'] for d in data],w,label=stage+' center')
        axes[1,1].bar(x+(j-.5)*w,[d['cross7_max_exceeds_center_10x_fraction'] for d in data],w,label=stage)
    axes[1,0].set(ylabel='Fraction',ylim=(0,1.05),title='Center validity (required by actual adapter)')
    axes[1,1].set(ylabel='Fraction of center-valid queries',ylim=(0,1.05),title='Neighbourhood max exceeds center by 10x (descriptive)')
    for ax in axes.ravel():ax.legend(fontsize=6);ax.grid(alpha=.2)
    for ax in axes[1]:ax.set_xticks(x,['RARE','joint center','joint cross7'])
    fig.savefig(output/'power_availability.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,stage in zip(axes,('initial_coarse9','first_refined_fine9')):
        for key,color in zip(MODEL_KEYS,colors):
            for method,style in (('center','-'),('cross7_max','--')):
                rows=[]
                for item in queries[key][stage]:
                    values=item['power'][...,0] if method=='center' else np.where(item['valid'],item['power'],-np.inf).max(-1)
                    for row,mask in zip(values,item['valid'][...,0]):
                        if mask.sum()>1:rows.append(np.log1p(row[mask].max())-np.log1p(row[mask].min()))
                ax.plot(np.sort(rows),np.linspace(0,1,len(rows)),style,color=color,label=key+' '+method)
        ax.set(title=stage,xlabel='Across 9 candidates: max - min of log1p(P / scale)',ylabel='Empirical CDF');ax.legend(fontsize=6);ax.grid(alpha=.2)
    fig.savefig(output/'local_contrast.png',dpi=180);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=Path('results/precision_study_20261122'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--episodes',type=int,default=150)
    p.add_argument('--volume-subsample',type=int,default=8192)
    a=p.parse_args(argv)
    s=diagnose(a.study,a.output,a.episodes,a.volume_subsample)
    print(json.dumps(dict(status=s['status'],episodes=s['episode_count'],output=str(a.output),elapsed_s=s['elapsed_seconds'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
