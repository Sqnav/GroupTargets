"""Offline scientific plots from saved, strictly paired evaluations.

Validation is the default. Saved test metrics require an explicit matching
frozen-final receipt, checked before any metric/member file is decoded. No
inference, raw dataset or sensor-cache reader is used. All primary metrics
and comparison gates come from the existing comparison kernel.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from .comparison import compare_evaluations, _load, _metric

ROOT=Path(__file__).resolve().parents[1]
STAGES=('initial','first_global_blend','first_refined','second_global_blend','second_refined','final')
STAGE_LABELS=('initial','blend 1','refine 1','blend 2','refine 2','final')


def _read(path):return json.loads(Path(path).read_text())
def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _number(value):return np.nan if value is None else value


def _records(path):
    """Use complete JSONL records, explicitly count an unfinished final append."""
    raw=Path(path).read_text();lines=raw.splitlines();rows=[];partial=0
    for index,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except ValueError:
            if index==len(lines)-1 and not raw.endswith('\n'):partial+=1
            else:raise
    return rows,partial


def episode_metrics(loaded):
    grouped=defaultdict(list)
    for (episode,_,_),row in loaded['members'].items():grouped[episode].append(row)
    result={}
    for episode,rows in grouped.items():
        errors=[r['error_3d_m'] for r in rows if r['status']=='localized']
        result[episode]=dict(truth_members=len(rows),localized_members=len(errors),
            position_error_m=_metric(errors),
            **{f'success_ratio_{t}m':sum(e<=t for e in errors)/len(rows) for t in (1,3,5)})
    return result


def training_curve(directory):
    directory=Path(directory);metrics,partial=_records(directory/'metrics.jsonl')
    validation,vpartial=_records(directory/'prior_validation.jsonl')
    manifest=_read(directory/'run_manifest.json');schedule=manifest.get('learning_rate_schedule',{})
    constant=manifest.get('hyperparameters',{}).get('learning_rate')
    points=[]
    for row in metrics:
        lr=row.get('learning_rate')
        if lr is None and schedule.get('mode','constant')=='constant':lr=constant
        points.append({k:row.get(k) for k in ('step','loss','initial_depth_huber','final_depth_huber','final_depth_mse','matched_rmse_m')}|{'learning_rate':lr})
    valid=[dict(step=r['step'],rmse_m=r['summary']['primary_member_rmse_m'],
        success_ratio_1m=r['summary'].get('primary_member_success_ratio_1m')) for r in validation]
    best=min(valid,key=lambda r:r['rmse_m']) if valid else None
    common=[r['final_depth_huber'] for r in points[-1000:] if r['final_depth_huber'] is not None]
    return dict(directory=str(directory.resolve()),status=manifest.get('status'),
        requested_steps=manifest.get('hyperparameters',{}).get('steps'),
        last_optimizer_step=points[-1]['step'] if points else None,
        device=manifest.get('configuration',{}).get('device'),seed=manifest.get('seed'),
        metrics=points,validation=valid,best_recorded_validation=best,
        final_recorded_validation=valid[-1] if valid else None,
        final_1000_step_mean_depth_huber=float(np.mean(common)) if common else None,
        ignored_unfinished_records=partial+vpartial,
        learning_rate_source='recorded per optimizer update' if any('learning_rate' in r for r in metrics) else 'configured constant, no per-update LR in historical log',
        source_sha256={p.name:_sha(p) for p in (directory/'metrics.jsonl',directory/'prior_validation.jsonl',directory/'run_manifest.json')},
        interpretation='Online training batches/frame means are not a fixed full-training-set score; total losses may use different objectives. An early best checkpoint alone does not establish overfitting.')


def _test_plot_gate(infos,freeze):
    """Authenticate saved test provenance before comparison loads any metrics.

    Hash only the explicit receipt and current evaluator source bytes. No
    test labels, metric files, model pickle or sensor record is opened here.
    The source gate is deliberately stricter than the old evaluator's optional
    entrypoint field: final plotting always requires that identity as well.
    """
    if freeze is None:
        raise ValueError('Plotting accepts validation outputs only without an explicit --freeze receipt; test remains frozen')
    freeze=Path(freeze).resolve();digest=_sha(freeze);receipt=_read(freeze)
    if not isinstance(receipt,dict) or receipt.get('status')!='frozen_for_test' or not isinstance(receipt.get('models'),dict):
        raise ValueError('Test plotting receipt status must be frozen_for_test with named model identities')
    source_fields=dict(evaluation_source_sha256=_sha(Path(__file__).with_name('evaluation.py')),
        evaluation_entrypoint_source_sha256=_sha(Path(__file__).with_name('evaluate_run.py')))
    for field,actual in source_fields.items():
        if receipt.get(field)!=actual:raise ValueError('Frozen test plotting identity mismatch: '+field)
    split_sha=receipt.get('split_manifest_sha256')
    if not isinstance(split_sha,str) or len(split_sha)!=64 or any(c not in '0123456789abcdef' for c in split_sha):
        raise ValueError('Frozen test plotting identity mismatch: split_manifest_sha256')
    for display_name,info in infos.items():
        verified=info.get('test_freeze');model_name=info.get('model_name')
        if (not isinstance(verified,dict) or verified.get('status')!='verified_before_sensor_or_label_read'
                or verified.get('freeze_sha256')!=digest or verified.get('model_name')!=model_name):
            raise ValueError(display_name+': unverified or different test_freeze identity')
        identity=receipt['models'].get(model_name)
        if not isinstance(identity,dict):raise ValueError(display_name+': model_name absent from frozen-final receipt')
        expected={field:identity.get(field) for field in ('checkpoint_sha256','configuration_sha256')}
        expected.update(source_fields,split_manifest_sha256=split_sha)
        for field,value in expected.items():
            info_field='entrypoint_source_sha256' if field=='evaluation_entrypoint_source_sha256' else field
            if (not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value)
                    or verified.get(field)!=value or info.get(info_field)!=value):
                raise ValueError(display_name+': frozen test plotting identity mismatch: '+field)
    return dict(status='verified_before_metric_or_member_read',freeze_path=str(freeze),
        freeze_sha256=digest,model_names={name:info['model_name'] for name,info in infos.items()},
        **source_fields,split_manifest_sha256=split_sha)


def collect(evaluations,reference=None,training=None,buckets=None,evidence=None,draws=2000,freeze=None):
    pairs=list(evaluations.items()) if isinstance(evaluations,dict) else list(evaluations)
    if not pairs:raise ValueError('At least one saved validation evaluation is required')
    if len({name for name,_ in pairs})!=len(pairs):raise ValueError('Unique evaluation names required before any plotting read')
    infos={name:_read(Path(path)/'run_info.json') for name,path in pairs}
    splits={info.get('split') for info in infos.values()}
    if len(splits)!=1 or not splits<={'val','test'}:
        raise ValueError('All plot inputs must use the same declared val/test partition')
    is_test=splits=={'test'}
    verified_freeze=_test_plot_gate(infos,freeze) if is_test else None
    reference=reference or pairs[0][0]
    comparison=compare_evaluations(pairs,references=[reference],draws=draws,allow_test=is_test)
    if comparison['status']!='valid':raise ValueError('Invalid evaluation population: '+str(comparison['population_checks']['errors']))
    models={};episodes={}
    for name,path in pairs:
        directory=Path(path);summary=_read(directory/'summary.json');loaded=_load(name,directory,is_test)
        models[name]=dict(evaluation_dir=str(directory.resolve()),summary=summary,
            run_info=infos[name])
        episodes[name]=episode_metrics(loaded)
    result=dict(status='frozen_test_plots_only' if is_test else 'validation_plots_only',created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        reference=reference,comparison=comparison,models=models,episodes=episodes,
        training={name:training_curve(path) for name,path in (training or {}).items()},
        buckets={},evidence=None,test_set_read=is_test,test_sensor_inputs_read=False,test_freeze=verified_freeze,default_changed=False,
        scope='Offline rendering of saved '+('explicitly frozen test' if is_test else 'validation')+' outputs; no models, raw label stores, images, IQ or sensor caches are loaded.')
    for name,path in (buckets or {}).items():
        if name not in models:raise ValueError('Bucket model is not evaluated: '+name)
        value=_read(path)
        result['buckets'][name]=dict(path=str(Path(path).resolve()),sha256=_sha(path),
            definition=value.get('geometry_cell_definition'),caveat=value.get('geometry_cell_caveat'),groups=value.get('groups',{}))
    if evidence is not None:
        directory=Path(evidence);index=_read(directory/'index.json')
        result['evidence']=dict(directory=str(directory.resolve()),index=index)
    result['findings']=findings(result)
    return result


def findings(report):
    rows=[]
    for name,value in report['models'].items():
        s=value['summary'];stages=s.get('stage_diagnostics',{})
        rows.append(dict(model=name,matched_rmse_m=s['member_position_error_m']['rmse'],
            parallel_mse_fraction=s.get('parallel_mse_fraction'),perpendicular_mse_fraction=s.get('perpendicular_mse_fraction'),
            fixed_ray_oracle_rmse_m=s.get('fixed_ray_oracle_diagnostic',{}).get('support',{}).get('position_error_m',{}).get('rmse'),
            stage_position_rmse_m={k:v.get('position_error_m',{}).get('rmse') for k,v in stages.items()},
            stage_update_diagnostics=s.get('stage_update_diagnostics',[])))
    return dict(model_diagnostics=rows,paired_comparisons=report['comparison']['comparisons'],
        cautions=['GT ray-oracle points are diagnostics, not deployable predictions or an information-theoretic sensor bound.',
            'Intermediate shared-network states need not monotonically improve; a poor intermediate alone is not evidence the final model fails.',
            'Training/validation curves are shown together; an early best checkpoint does not alone establish overfitting.',
            'Episode bootstrap measures validation-episode uncertainty for these weights; it does not replace training-seed replication or untouched final test evaluation.'])


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _save(plt,fig,output,name):
    fig.savefig(output/name,dpi=180,bbox_inches='tight');plt.close(fig)


def plot_metrics(report,output,plt):
    rows=report['comparison']['models'];names=[r['name'] for r in rows];x=np.arange(len(names));width=.24
    fig,ax=plt.subplots(1,2,figsize=(11.5,4),constrained_layout=True)
    for j,(field,label) in enumerate((('conditional_member_rmse_m','RMSE'),('conditional_member_mean_error_m','Mean'),('conditional_member_p95_m','P95'))):
        values=[_number(r[field]) for r in rows];ax[0].bar(x+(j-1)*width,values,width,label=label)
    for j,t in enumerate((1,3,5)):
        values=[100*r[f'success_ratio_all_gt_{t}m'] for r in rows]
        ax[1].bar(x+(j-1)*width,values,width,label=f'<= {t} m')
    for a in ax:a.set_xticks(x,names);a.legend();a.grid(axis='y',alpha=.2)
    ax[0].set(ylabel='Metres',title='Finite fixed-matched members')
    ax[1].set(ylabel='Percent of ALL GT members',ylim=(0,100),title='Misses / rejections remain failures')
    _save(plt,fig,output,'metrics.png')


def plot_episodes(report,output,plt):
    reference=report['reference'];models=report['episodes'];names=list(models);episodes=sorted(models[reference])
    if any(set(value)!=set(episodes) for value in models.values()):raise ValueError('Episode set differs; never silently intersect')
    if len(names)>1:
        others=[n for n in names if n!=reference];fig,axes=plt.subplots(1,len(others),figsize=(5*len(others),4),squeeze=False,constrained_layout=True)
        paired={r['model']:r['identical_localized_member_set'] for r in report['comparison']['comparisons']}
        for ax,name in zip(axes[0],others):
            identical=paired[name]
            x=[models[reference][e]['position_error_m']['rmse'] if identical else models[reference][e]['success_ratio_3m']*100 for e in episodes]
            y=[models[name][e]['position_error_m']['rmse'] if identical else models[name][e]['success_ratio_3m']*100 for e in episodes]
            finite=[(a,b) for a,b in zip(x,y) if a is not None and b is not None]
            if finite:
                a,b=np.asarray(finite).T;ax.scatter(a,b,s=16,alpha=.65)
                limit=max(float(a.max()),float(b.max()))*1.04;ax.plot([0,limit],[0,limit],'k--',lw=.9)
            ax.text(.02,.98,f'{len(finite)}/{len(episodes)} finite episode points\nAll episodes remain in coverage metrics',
                transform=ax.transAxes,ha='left',va='top',fontsize=7)
            label='episode RMSE (m)' if identical else 'all-GT <=3m coverage (%)'
            ax.set(xlabel=reference+' '+label,ylabel=name+' '+label,
                title='Paired episodes; below diagonal is better' if identical else 'Localized sets differ: no paired RMSE claim')
            ax.grid(alpha=.2)
        _save(plt,fig,output,'episode_paired_scatter.png')
    fig,ax=plt.subplots(2,1,figsize=(12,6),sharex=True,constrained_layout=True)
    for name in names:
        ax[0].plot([models[name][e]['position_error_m']['rmse'] for e in episodes],label=name,lw=1.1)
        ax[1].plot([100*models[name][e]['success_ratio_3m'] for e in episodes],label=name,lw=1.1)
    ax[0].set(ylabel='Episode RMSE (m)',title='Same episode order; gaps mean unlocalized, all episodes remain in coverage')
    ax[1].set(ylabel='All-GT <=3m (%)',xlabel='Validation episode index',ylim=(0,100))
    for a in ax:a.legend();a.grid(alpha=.2)
    _save(plt,fig,output,'episode_curves.png')


def plot_stages(report,output,plt):
    names=list(report['models']);fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for name in names:
        s=report['models'][name]['summary'];stages=s.get('stage_diagnostics',{})
        for a,field in zip(ax[0],('position_error_m','depth_error_m')):
            y=[stages.get(k,{}).get(field,{}).get('rmse') for k in STAGES]
            a.plot(range(len(STAGES)),[np.nan if v is None else v for v in y],marker='o',label=name)
        updates=s.get('stage_update_diagnostics',[])
        for before,after in (('initial','first_refined'),('first_refined','final')):
            row=next((r for r in updates if r['before']==before and r['after']==after),None)
            if row:
                label=name+' '+('round1' if before=='initial' else 'round2');x=len(ax[1,0].patches)//3
                left=0.
                for field,color in (('improved_ratio','#4c9f70'),('worsened_ratio','#cb5c5c'),('unchanged_ratio','#b5b5b5')):
                    value=100*row[field];ax[1,0].barh(x,value,left=left,color=color);left+=value
                ax[1,0].text(-1,x,label,ha='right',va='center',fontsize=8)
    for a,title in zip(ax[0],('Stage 3D RMSE','Stage projection-depth RMSE')):
        a.set_xticks(range(len(STAGES)),STAGE_LABELS,rotation=25);a.set(ylabel='Metres',title=title);a.grid(alpha=.2);a.legend()
    ax[1,0].set(xlim=(0,100),xlabel='Percent of matched members',title='Green: improved / red: worsened / grey: unchanged',yticks=[])
    x=np.arange(len(names));width=.24
    for j,key in enumerate(('initial','global_proposal','final')):
        y=[_number(report['models'][n]['summary'].get('stage_diagnostics',{}).get(key,{}).get('position_error_m',{}).get('rmse')) for n in names]
        ax[1,1].bar(x+(j-1)*width,y,width,label='global alternative' if key=='global_proposal' else key)
    ax[1,1].set_xticks(x,names);ax[1,1].set(ylabel='3D RMSE (m)',title='Global proposal is an alternative, not the final point');ax[1,1].legend(fontsize=8);ax[1,1].grid(axis='y',alpha=.2)
    _save(plt,fig,output,'stage_diagnostics.png')


def plot_error_components(report,output,plt):
    names=list(report['models']);fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True);x=np.arange(len(names));w=.24
    for j,(field,label) in enumerate((('parallel_error_m','Along-ray'),('perpendicular_error_m','Transverse'))):
        y=[_number(report['models'][n]['summary'].get(field,{}).get('rmse')) for n in names]
        ax[0].bar(x+(j-.5)*w,y,w,label=label)
    for j,(key,label) in enumerate((('support','GT-assisted ray oracle'),('model','Actual model'))):
        y=[_number(report['models'][n]['summary']['member_position_error_m']['rmse']) if key=='model' else
            _number(report['models'][n]['summary'].get('fixed_ray_oracle_diagnostic',{}).get(key,{}).get('position_error_m',{}).get('rmse')) for n in names]
        ax[1].bar(x+(j-.5)*w,y,w,label=label)
    for a in ax:a.set_xticks(x,names);a.set_ylabel('Metres');a.legend(fontsize=9);a.grid(axis='y',alpha=.2)
    ax[0].set_title('Fixed-ray orthogonal error decomposition')
    ax[1].set_title('Oracle is a diagnostic, not deployed accuracy')
    _save(plt,fig,output,'ray_error_components.png')


def _smooth(points,field,window=100):
    valid=[(r['step'],r[field]) for r in points if r.get(field) is not None]
    if not valid:return [],[]
    x,y=np.asarray(valid,float).T;window=min(window,len(y))
    return x[window-1:],np.convolve(y,np.ones(window)/window,mode='valid')


def plot_training(report,output,plt):
    if not report['training']:return
    fig,ax=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for name,curve in report['training'].items():
        for a,field in ((ax[0,0],'final_depth_huber'),(ax[0,1],'loss')):
            x,y=_smooth(curve['metrics'],field);a.plot(x,y,label=name)
        rows=curve['validation'];ax[1,0].plot([r['step'] for r in rows],[r['rmse_m'] for r in rows],marker='.',label=name)
        x,y=_smooth(curve['metrics'],'learning_rate',1);ax[1,1].plot(x,y,label=name)
    titles=('Online final-depth Huber, 100-update mean','Total objectives differ: not an accuracy comparison',
            'Independent validation RMSE: all recorded checkpoints','Learning rate per optimizer update')
    for a,title in zip(ax.ravel(),titles):a.set(title=title,xlabel='Optimizer updates');a.legend();a.grid(alpha=.2)
    ax[0,0].set_ylabel('Huber');ax[0,1].set_ylabel('Respective objective');ax[1,0].set_ylabel('Metres');ax[1,1].set_ylabel('Learning rate')
    _save(plt,fig,output,'training_validation_curves.png')


def plot_buckets(report,output,plt):
    if not report['buckets']:return
    fields=('normalized_bbox_scale','joint_cell_neighbours','observer_same_joint_cell')
    fig,ax=plt.subplots(2,len(fields),figsize=(14,7),constrained_layout=True)
    for column,field in enumerate(fields):
        available={n:v['groups'][field] for n,v in report['buckets'].items() if field in v['groups']}
        if not available:continue
        names=list(available);keys=sorted(next(iter(available.values()))['groups']);width=.8/len(names)
        for j,name in enumerate(names):
            groups=available[name]['groups']
            if sorted(groups)!=keys:raise ValueError('Bucket groups differ; never silently intersect')
            if not np.allclose(available[name].get('edges',[]),available[names[0]].get('edges',[]),rtol=0,atol=1e-12):
                raise ValueError('Bucket physical edges differ')
            for key in keys:
                if groups[key]['truth_members']!=available[names[0]]['groups'][key]['truth_members']:
                    raise ValueError('Bucket populations differ')
            x=np.arange(len(keys))-.4+width*(j+.5)
            ax[0,column].bar(x,[_number(groups[k]['position_error_m']['rmse']) for k in keys],width,label=name)
            ax[1,column].bar(x,[100*groups[k]['success_ratio_5m'] for k in keys],width,label=name)
        for a in ax[:,column]:a.set_xticks(range(len(keys)),keys,rotation=25);a.legend(fontsize=8);a.grid(axis='y',alpha=.2)
        ax[0,column].set(title=field,ylabel='Conditional RMSE (m)');ax[1,column].set(ylabel='All-GT <=5m (%)',ylim=(0,100))
    _save(plt,fig,output,'diagnostic_strata.png')


def plot_evidence(report,output,plt):
    if not report['evidence']:return
    evidence=report['evidence'];cases=evidence['index']['cases']
    if not cases:return
    fig,axes=plt.subplots(math.ceil(len(cases)/2),2,figsize=(13,3*math.ceil(len(cases)/2)),squeeze=False,constrained_layout=True)
    for ax,case in zip(axes.ravel(),cases):
        with np.load(Path(evidence['directory'])/case['raw_curve_npz'],allow_pickle=False) as data:
            for name,label in (('ra_log_power','RA'),('re_log_power','RE'),('joint_log_power','Joint')):
                ax.plot(data['depth_m'],data[name],label=label,lw=.9)
        ax.axvline(case['projection_gt_depth_m'],color='red',label='GT projection (diagnostic)',lw=1.2)
        for key,color in (('initial','grey'),('first_refined','orange'),('final','black')):
            if key in case['stages_m']:ax.axvline(case['stages_m'][key],color=color,ls='--',label=key,lw=.8)
        ax.set(title=f'{case["kind"]}: {case["episode_id"]}/{case["frame_idx"]}, error {case["error_3d_m"]:.2f} m',
            xlabel='Camera-ray projection depth (m)',ylabel='log1p(raw power / shared scale)')
        ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
    for ax in axes.ravel()[len(cases):]:ax.set_visible(False)
    _save(plt,fig,output,'ray_evidence_overview.png')


def render(report,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('Existing plot/report output is protected')
    plt=_pyplot()
    for function in (plot_metrics,plot_episodes,plot_stages,plot_error_components,plot_training,plot_buckets,plot_evidence):function(report,output,plt)
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'findings.json').write_text(json.dumps(report['findings'],ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'comparison.json').write_text(json.dumps(report['comparison'],ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (output/'metrics_table.json').write_text(json.dumps(report['comparison']['models'],ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return sorted(p.name for p in output.glob('*.png'))


def _pairs(values,parser):
    result={}
    for value in values or []:
        if '=' not in value:parser.error('Expected NAME=PATH')
        name,path=value.split('=',1)
        if not name or name in result:parser.error('Unique nonempty names required')
        result[name]=Path(path)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('evaluations',nargs='+',help='NAME=/saved/evaluation (val by default)')
    p.add_argument('--reference');p.add_argument('--training',action='append',help='NAME=/training/run')
    p.add_argument('--bucket',action='append',help='NAME=/saved/diagnostic_buckets.json')
    p.add_argument('--evidence',type=Path);p.add_argument('--draws',type=int,default=2000)
    p.add_argument('--freeze',type=Path,help='Explicit frozen_for_test receipt; identity checked before any saved test metric/member read')
    p.add_argument('--output',type=Path,default=ROOT/'results/precision_study_20261122/diagnostic_plots')
    a=p.parse_args(argv);report=collect(_pairs(a.evaluations,p),a.reference,_pairs(a.training,p),_pairs(a.bucket,p),a.evidence,a.draws,freeze=a.freeze)
    files=render(report,a.output);print(json.dumps(dict(status=report['status'],plots=files,output=str(a.output),test_set_read=report['test_set_read'])))
    return 0


if __name__=='__main__':raise SystemExit(main())
