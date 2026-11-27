"""Explicitly label-assisted failure plots; never used for prediction/selection.

The plotted curves are physical, raw-power queries, not calibrated likelihoods
or the network's CNN activations. Alternative maxima do not represent separate
members. No fitted calibration, observer subtraction or IQ changes are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks

from depth_localization.data import load_observation_features
from rvwm.data import Dataset
from .evaluation import distribution
from .volume import build_power_volume, positions_to_ruv, sample_power_volume_numpy

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def raw_rare_queries(features, points):
    """Match the baseline's physical range/az/el bilinear pixel convention."""
    p=np.asarray(points,float)
    radius=np.linalg.norm(p,axis=-1)
    az=np.arctan2(p[...,1],p[...,0])
    el=np.arctan2(p[...,2],np.hypot(p[...,0],p[...,1]))
    result={}
    for kind,angle,axis in (('ra',az,features['azimuth_rad']),('re',el,features['elevation_rad'])):
        axis=np.asarray(axis,float);r=np.asarray(features['range_m'],float)
        valid=(np.isfinite(p).all(-1)&(p[...,0]>0)&(radius>=r[0])&(radius<=r[-1])&
               (angle>=axis[0])&(angle<=axis[-1]))
        coords=np.array([np.interp(radius,r,np.arange(len(r))),np.interp(angle,axis,np.arange(len(axis)))])
        power=map_coordinates(np.asarray(features[kind+'_power']),coords,order=1,mode='constant',cval=0.,prefilter=False)
        weight=map_coordinates(np.asarray(features[kind+'_valid'],float),coords,order=1,mode='constant',cval=0.,prefilter=False)
        valid&=weight>=1.-1e-6
        result[kind+'_power']=np.where(valid,power,np.nan)
        result[kind+'_valid']=valid
    return result


def _peaks(depth, curve):
    value=np.nan_to_num(curve,nan=-1e10)
    indices,_=find_peaks(value,prominence=.08,distance=max(1,len(depth)//150))
    indices=sorted(indices,key=lambda i:float(value[i]),reverse=True)[:5]
    return [dict(depth_m=float(depth[i]),log_power=float(value[i]),grid_index=int(i)) for i in indices]


def _bucket_metric(rows):
    usable=[r for r in rows if r['status']=='localized']
    n=len(rows)
    result=dict(truth_members=n,localized_members=len(usable),
        position_error_m=distribution(r['error_3d_m'] for r in usable),
        failure_counts=dict(Counter(r['status'] for r in rows if r['status']!='localized')))
    for t in (1,3,5):
        result[f'success_ratio_{t}m']=sum(r['error_3d_m']<=t for r in usable)/n if n else None
    return result


def make_buckets(evaluation_dir, radar, output=None, calibration_lookup=None):
    """GT-labelled error stratification, with the same membership across models."""
    evaluation_dir=Path(evaluation_dir)
    frames=read_rows(evaluation_dir/'frames.jsonl');members=read_rows(evaluation_dir/'members.jsonl')
    frame_map={(r['episode_id'],r['frame_idx']):r for r in frames}
    dr=299792458./(2*radar.bandwidth_hz)
    du=radar.wavelength/(radar.shape[2]*radar.spacing_y_m)
    dv=radar.wavelength/(radar.shape[3]*radar.spacing_z_m)
    records=[]
    for row in members:
        frame=frame_map[(row['episode_id'],row['frame_idx'])]
        record=dict(episode_id=row['episode_id'],frame_idx=row['frame_idx'],truth_index=row['truth_index'])
        if 'truth_position_base_m' not in row:
            record.update(base_range_m=None,camera_slant_m=None,bbox_sqrt_area_px=None,
                normalized_bbox_scale=None,member_count=frame['truth_count'],joint_cell_neighbours=None,
                observer_same_joint_cell=None)
            records.append(record);continue
        points=np.asarray(frame['truth_positions_base_m'],float)
        radius=np.linalg.norm(points,axis=-1)
        unit=points/np.maximum(radius[:,None],1e-12)
        j=row['truth_index']
        close=(np.abs(radius-radius[j])<dr)&(np.abs(unit[:,1]-unit[j,1])<du)&(np.abs(unit[:,2]-unit[j,2])<dv)
        close[j]=False
        observer=np.asarray(frame['observer_position_base_m'],float)
        oradius=float(np.linalg.norm(observer));ounit=observer/max(oradius,1e-12)
        observer_cell=bool(abs(oradius-radius[j])<dr and abs(ounit[1]-unit[j,1])<du and abs(ounit[2]-unit[j,2])<dv)
        bbox=None;scale=None
        if row.get('matched'):
            slot=row['detection_index'];box=np.asarray(frame['bbox_xyxy'][slot],float)
            w,h=box[2:]-box[:2];bbox=float(np.sqrt(w*h))
            calibration=(calibration_lookup or {}).get((row['episode_id'],row['frame_idx']))
            if calibration is not None:
                k=np.asarray(calibration,float);scale=float(np.sqrt(w*h/(k[0,0]*k[1,1])))
        record.update(base_range_m=float(radius[j]),camera_slant_m=row.get('oracle',{}).get('true_camera_slant_m'),
            bbox_sqrt_area_px=bbox,normalized_bbox_scale=scale,member_count=frame['truth_count'],
            joint_cell_neighbours=int(close.sum()),observer_same_joint_cell=observer_cell)
        records.append(record)
    by_family={}
    for name in ('base_range_m','camera_slant_m','bbox_sqrt_area_px','normalized_bbox_scale'):
        values=np.asarray([r[name] for r in records if r[name] is not None],float)
        if not len(values):continue
        edges=np.quantile(values,[0,.25,.5,.75,1.])
        groups=defaultdict(list)
        for row,record in zip(members,records):
            value=record[name]
            key='missing' if value is None else f'quartile_{min(4,int(np.searchsorted(edges[1:-1],value,side="right"))+1)}'
            groups[key].append(row);record[name+'_bucket']=key
        by_family[name]=dict(edges=edges.tolist(),groups={k:_bucket_metric(v) for k,v in groups.items()})
    for name in ('member_count','observer_same_joint_cell','joint_cell_neighbours'):
        groups=defaultdict(list)
        for row,record in zip(members,records):
            value=record[name]
            key='missing' if value is None else ('3_or_more' if name=='joint_cell_neighbours' and value>=3 else str(value))
            groups[key].append(row);record[name+'_bucket']=key
        by_family[name]=dict(groups={k:_bucket_metric(v) for k,v in groups.items()})
    result=dict(scope='offline GT-labelled strata only, never model input or checkpoint selection',
        geometry_cell_definition='pairwise |delta R|<c/(2B), |delta u|<lambda/(Ny*dy), |delta v|<lambda/(Nz*dz)',
        geometry_cell_caveat='native Fourier-cell proximity heuristic, not proof of resolvability or a calibrated FFT detection',
        range_resolution_m=dr,native_direction_cell_u=du,native_direction_cell_v=dv,
        groups=by_family,members=records)
    if output is not None:
        Path(output).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'depth_config.json')
    p.add_argument('--evaluation',type=Path,default=ROOT/'results/precision_study_20261122/baseline/v3')
    p.add_argument('--output',type=Path,default=ROOT/'results/precision_study_20261122/evidence_v3')
    p.add_argument('--failures',type=int,default=6);p.add_argument('--successes',type=int,default=2)
    a=p.parse_args(argv)
    if not 0<=a.failures<=6 or not 0<=a.successes<=2:p.error('At most 6 failures and 2 successful diagnostic members')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Existing evidence output is protected')
    config=json.loads(a.config.read_text())
    members=read_rows(a.evaluation/'members.jsonl');frames=read_rows(a.evaluation/'frames.jsonl')
    frame_map={(r['episode_id'],r['frame_idx']):r for r in frames}
    valid=[r for r in members if r['status']=='localized']
    failed=sorted([r for r in valid if r['error_3d_m']>5],key=lambda r:r['error_3d_m'],reverse=True)[:a.failures]
    good=sorted([r for r in valid if r['error_3d_m']<=1],key=lambda r:r['error_3d_m'])
    success=[]
    for fraction in (.4,.6)[:a.successes]:
        if good:success.append(good[min(len(good)-1,int((len(good)-1)*fraction))])
    chosen=[('failure',r) for r in failed]+[('success',r) for r in success]
    dataset=Dataset(config['dataset_root'],echo_subdir=config['echo_subdir'],verify_sources=True)
    sensor_cache={};results=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for index,(kind,member) in enumerate(chosen):
        key=(member['episode_id'],member['frame_idx']);frame=frame_map[key]
        if key not in sensor_cache:
            observation=dataset.observations(*key)
            f=load_observation_features(config,*key,'val')
            volume=build_power_volume(observation.iq,observation.radar,config)
            sensor_cache[key]=(observation,f,volume)
        observation,f,volume=sensor_cache[key]
        slot=member['detection_index'];origin=np.asarray(f['ray_origins'][slot]);direction=np.asarray(f['ray_directions'][slot])
        lower,upper=np.asarray(f['ray_depth_bounds'][slot],float)
        stages={k:v['depth_m'] for k,v in member['stages'].items()}
        gt=member['oracle']['projection_depth_m']
        actual=[gt]+[v for k,v in stages.items() if k!='global_proposal']
        broad_high=min(upper,max(actual+[stages.get('global_proposal',gt)])+20.)
        broad_low=max(lower,min(actual)-20.)
        depth=np.linspace(broad_low,broad_high,4097)
        points=origin[None]+depth[:,None]*direction[None]
        rare=raw_rare_queries(f,points)
        ruv=positions_to_ruv(points)
        joint=sample_power_volume_numpy(volume['volume_power'],ranges=ruv['ranges'],u=ruv['u'],v=ruv['v'],
            range_axis=volume['range_m'],u_axis=volume['u_axis'],v_axis=volume['v_axis'],
            angular_valid=volume['angular_valid'],candidate_valid=ruv['valid'])
        power_scale=max((np.asarray(f['ra_power'])[f['ra_valid']].sum(dtype=np.float64)+
                         np.asarray(f['re_power'])[f['re_valid']].sum(dtype=np.float64))/
                        (np.asarray(f['ra_valid']).sum()+np.asarray(f['re_valid']).sum()),1e-20)
        curves={k:np.log1p(rare[k+'_power']/power_scale) for k in ('ra','re')}
        curves['joint']=np.where(joint['valid'],np.log1p(joint['values'][...,0]/power_scale),np.nan)
        peaks={k:_peaks(depth,v) for k,v in curves.items()}
        stem=f'{index+1:02d}_{kind}_{key[0]}_frame_{key[1]:05d}_detection_{slot}'
        np.savez_compressed(out/(stem+'.npz'),depth_m=depth,**{k+'_log_power':v for k,v in curves.items()},
                            joint_valid=joint['valid'],ra_valid=rare['ra_valid'],re_valid=rare['re_valid'])
        fig,axes=plt.subplots(2,1,figsize=(13,8),constrained_layout=True)
        colors={'ra':'#377eb8','re':'#ff7f00','joint':'#4daf4a'}
        marks=[('GT projection',gt,'black','--'),('initial',stages.get('initial'),'#984ea3',':'),
               ('global proposal',stages.get('global_proposal'),'#999999',':'),
               ('coarse',stages.get('first_refined'),'#e41a1c','-.'),('final',stages.get('final'),'#d000a0','--')]
        for ax in axes:
            for name,value in curves.items():ax.plot(depth,value,label=name.upper()+' raw evidence',color=colors[name],lw=1.2)
            for label,value,color,style in marks:
                if value is not None:ax.axvline(value,label=f'{label}: {value:.2f} m',color=color,linestyle=style,lw=1.1)
            for peak in peaks['joint']:
                ax.plot(peak['depth_m'],peak['log_power'],'v',color=colors['joint'],ms=5)
            ax.set_ylabel('log(1 + power / shared mean)');ax.grid(alpha=.2)
        axes[0].set_xlim(broad_low,broad_high);axes[0].legend(ncol=3,fontsize=8)
        axes[1].set_xlim(max(lower,min(actual)-7.),min(upper,max(actual)+7.))
        axes[1].set_xlabel('Slant depth along the actual camera ray (m)')
        fig.suptitle(f'{kind.upper()}: {key[0]} frame {key[1]}, detection {slot}; error={member["error_3d_m"]:.2f} m\n'
            'Offline GT-labelled diagnostic only; maxima are not independently matched members',fontsize=11)
        fig.savefig(out/(stem+'.png'),dpi=150);plt.close(fig)
        results.append(dict(kind=kind,episode_id=key[0],frame_idx=key[1],detection_index=slot,
            truth_index=member['truth_index'],error_3d_m=member['error_3d_m'],projection_gt_depth_m=gt,
            stages_m=stages,alternative_raw_power_maxima=peaks,power_scale=power_scale,
            raw_curve_npz=stem+'.npz',plot_png=stem+'.png',source_hashes=observation.source_hashes,
            labels_used=True,scope='offline failure explanation only; not a deployable depth estimator'))
        print(json.dumps(dict(case=index+1,total=len(chosen),stem=stem)),flush=True)
    calibration={}
    inventory=a.evaluation.parent/'input_inventory.json'
    if inventory.exists():
        for row in json.loads(inventory.read_text())['manifest']:
            k=row.get('depth_input_receipt',{}).get('camera_K')
            if k is not None:calibration[(row['episode_id'],row['frame_idx'])]=k
    first_observation=next(iter(sensor_cache.values()))[0] if sensor_cache else dataset.observations(frames[0]['episode_id'],frames[0]['frame_idx'])
    for name in ('v3','v2'):
        directory=a.evaluation.parent/name
        if (directory/'members.jsonl').exists():
            make_buckets(directory,first_observation.radar,out/(name+'_diagnostic_buckets.json'),calibration)
    report=dict(created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),cases=results,
        selection='six largest localized errors >5m, then two representative <=1m successful members at 40/60% success-error quantiles; diagnostics only',
        same_input_power_policy='raw joint P and RA/RE share native linear units and original RA/RE mean scale; no peak normalization',
        caveat='All GT markings, case selection and overlap strata are offline diagnostics. Raw curves are not the learned CNN features or calibrated likelihoods. Peaks can contain observer/mixture/side-lobes; no independent-member peak matching.')
    (out/'index.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
