"""Fixed validation-only evidence ablations; production and test are untouched."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from bandwidth_study.evaluate import read, sha, write
from precision_study.deployed_infer import DEFAULT_DESCRIPTOR, load_descriptor
from precision_study.evaluation import evaluate_samples, serializable
from precision_study.evaluate_run import sanitize_diagnostics
from precision_study.versions import load_frozen_study_model

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('sequence', 'aligned', 'roi')
METHODS = ('D32_parent', 'D_deployed', *ARMS)


def identities(exp):
    from .train import validate_training
    from bandwidth_study.train import validate_training as validate_parent_training
    protocol = read(exp/'PROTOCOL.json')
    parent = Path(protocol['resource_queue']['bandwidth_output'])
    temporal = Path(protocol['resource_queue']['parent_output'])
    for folder, key in ((parent,'bandwidth_plan_sha256'),(temporal,'parent_plan_sha256')):
        if sha(folder/'run_plan.json') != protocol['resource_queue'][key] or read(folder/'run_state.json')['status'] != 'complete':
            raise ValueError('Registered predecessor must be complete before new evaluation')
    validate_parent_training(parent/'training/baseline')
    control_run = read(parent/'training/baseline/run_manifest.json')
    expected_initial = control_run['model_initialization']['initial_parameter_sha256']
    description, deployment = load_descriptor(DEFAULT_DESCRIPTOR)
    if deployment['checkpoint_sha256'] != protocol['immutable_baseline']['checkpoint_sha256']:
        raise ValueError('Original production D changed')
    models = {}
    for arm in ARMS:
        folder=exp/'training'/arm
        validate_training(folder)
        run=read(folder/'run_manifest.json')
        if (run.get('optimizer_steps') != 20000 or run.get('global_batch_size') != 48
                or run['model_initialization'].get('base_initial_parameter_sha256') != expected_initial):
            raise ValueError('Training budget or base parameter initialization differs: '+arm)
        config=folder/'inputs/configuration.json'; cp=folder/'depth.rmse.best.pt'
        models[arm]=dict(kind='evidence', configuration_path=str(config),configuration_sha256=sha(config),
            checkpoint_path=str(cp),checkpoint_sha256=sha(cp),run_sha256=sha(folder/'run_manifest.json'),
            model_parameter_count=run['model_parameter_count'])
    cp=parent/'training/baseline/depth.rmse.best.pt';cfg=parent/'training/baseline/inputs/configuration.json'
    models['D32_parent']=dict(kind='study',configuration_path=str(cfg),configuration_sha256=sha(cfg),
        checkpoint_path=str(cp),checkpoint_sha256=sha(cp),source_root=str(description['source_root']),
        model_parameter_count=control_run['model_parameter_count'])
    models['D_deployed']=dict(kind='study',configuration_path=str(description['config']),
        configuration_sha256=sha(description['config']),checkpoint_path=str(description['checkpoint']),
        checkpoint_sha256=sha(description['checkpoint']),source_root=str(description['source_root']),
        model_parameter_count=275909)
    sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    sources.update(protocol=sha(exp/'PROTOCOL.json'),roster=sha(exp/'split_roster.json'))
    return dict(status='frozen_validation_only',models=models,source_sha256=sources,
        base_initial_parameter_sha256=expected_initial,test_access=False)


def freeze(exp):
    value=identities(exp); path=exp/'frozen_validation_identity.json'
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen validation identity changed')
    else:
        write(path,value)
    return value


def evaluate_method(exp,method,device):
    from .checkpoint import load_checkpoint
    from .data import load_samples
    frozen=read(exp/'frozen_validation_identity.json')
    if frozen != identities(exp):
        raise ValueError('Evaluation identity mismatch before observation/label access')
    item=frozen['models'][method]
    config=read(item['configuration_path'])
    sensor_config=config if method in ARMS else read(exp/'configs/sequence.json')
    expected=dict(method=method,split='val',model_identity=item,source_sha256=frozen['source_sha256'])
    out=exp/'evaluation/val'/method
    if (out/'completion_receipt.json').exists():
        if read(out/'completion_receipt.json') != expected:
            raise ValueError('Existing evaluation identity changed')
        return
    if out.exists():
        raise FileExistsError('Partial outputs must be preserved and reviewed before retry: '+str(out))
    if item['kind']=='evidence':
        model=load_checkpoint(item['checkpoint_path'],config,device).eval()
    else:
        model=load_frozen_study_model(item['source_root'],item['checkpoint_path'],config,device).eval()
    roster=read(exp/'split_roster.json')['splits']['val']
    tick=time.perf_counter()
    samples,manifest,frontend=load_samples(sensor_config,roster['episode_ids'],'val',count=0,include_labels=True)
    loading_s=time.perf_counter()-tick
    if len(manifest) != roster['frame_count']:
        raise ValueError('Missing declared validation frames')
    info=dict(**expected,device=device,input_frontend=frontend,data_loading_s=loading_s,
        labels_used_for_prediction=False,labels_used_only_for_scoring=True,
        observation_scope='single current frame; original32MHz IQ-derived RA/RE, unchanged bbox/optical geometry',
        test_opened=False,validation_has_checkpoint_selection_and_historical_exposure=True)
    def pulse(row):
        print(json.dumps(dict(stage='evaluation',method=method,**row)),flush=True)
    result=evaluate_samples(model,samples,manifest,config,capture_stages=True,progress_callback=pulse,run_info=info)
    out.mkdir(parents=True)
    nonfinite=[]
    for group in ('members','frames','detections'):
        with (out/(group+'.jsonl')).open('x') as stream:
            for index,row in enumerate(result[group]):
                stream.write(json.dumps(sanitize_diagnostics(row,f'{group}[{index}]',nonfinite),ensure_ascii=False,allow_nan=False)+'\n')
    for group in ('summary','episodes'):
        write(out/(group+'.json'),result[group])
    if method in ARMS:
        diagnose_candidates(model,samples,result['members'],out)
    write(out/'input_manifest.json',dict(frames=manifest,frontend=frontend))
    write(out/'run_info.json',info)
    write(out/'serialization.json',dict(primary_metrics_changed=False,diagnostic_nonfinite=nonfinite))
    write(out/'completion_receipt.json',expected)


def diagnose_candidates(model,samples,members,out):
    """Separate deterministic first/last-frame pass; outside latency measure.

    GT projection is attached only after prediction. Nearest sampled depth is
    a coverage diagnostic, not a lower bound for continuous regression.
    """
    by_episode=defaultdict(list)
    for sample in samples:
        by_episode[sample['episode_id']].append(sample)
    truths=defaultdict(list)
    for member in members:
        if member.get('matched') and 'oracle' in member:
            truths[member['episode_id'],member['frame_idx']].append(member)
    chosen=[]
    for episode in sorted(by_episode):
        ordered=sorted(by_episode[episode],key=lambda s:s['frame_idx'])
        chosen.append(ordered[0])
        if len(ordered)>1:chosen.append(ordered[-1])
    old=model.capture_candidates
    model.capture_candidates=True
    rows=[];aggregates=[[],[]]
    try:
        for sample in chosen:
            scene=model.predict(sample['features'])
            diagnostic=serializable(scene['sampling_diagnostics'])
            rows.append(dict(episode_id=sample['episode_id'],frame_idx=sample['frame_idx'],
                sensor_only_diagnostics=diagnostic,label_use='projection depth attached after prediction only'))
            for member in truths[sample['episode_id'],sample['frame_idx']]:
                i=member['detection_index'];rho=member['oracle']['projection_depth_m']
                for stage,geometry in enumerate(diagnostic.get('stage_candidate_geometry',[])):
                    candidates=np.asarray(geometry['depths_m'][i],float)
                    usable=np.asarray(geometry['usable'][i],bool)
                    finite=candidates[usable]
                    final=float(scene['depths'][i])
                    aggregates[stage].append(dict(episode_id=sample['episode_id'],frame_idx=sample['frame_idx'],
                        detection_index=i,GT_projection_depth_m=rho,
                        usable_candidate_count=int(usable.sum()),
                        interval_covers_projection=bool(len(finite) and finite.min()<=rho<=finite.max()),
                        nearest_valid_candidate_m=float(np.min(np.abs(finite-rho))) if len(finite) else None,
                        final_outside_valid_sample_interval=bool(not len(finite) or not finite.min()<=final<=finite.max())))
    finally:
        model.capture_candidates=old
    summary=[]
    for index,values in enumerate(aggregates):
        distances=[r['nearest_valid_candidate_m'] for r in values if r['nearest_valid_candidate_m'] is not None]
        summary.append(dict(round=index+1,matched_members=len(values),
            interval_projection_coverage=float(np.mean([r['interval_covers_projection'] for r in values])) if values else None,
            no_usable_candidates=sum(r['usable_candidate_count']==0 for r in values),
            nearest_candidate_p50_m=float(np.median(distances)) if distances else None,
            final_outside_sample_interval_fraction=float(np.mean([r['final_outside_valid_sample_interval'] for r in values])) if values else None))
    write(out/'candidate_diagnostics.json',dict(rule='first and last declared validation frame in every episode',
        frames=len(chosen),rounds=summary,member_rows=aggregates,
        warning='GT-assisted coverage diagnostic only; finite samples are not a continuous regression performance bound',
        included_in_primary_inference_timing=False))
    with (out/'candidate_queries.jsonl').open('x') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')


def paired(first,second,families,draws=2000,seed=20261124):
    """Whole paired trajectories, preserving failures in all-GT denominators."""
    key=lambda r:(r['episode_id'],r['frame_idx'],r['truth_index'])
    a,b={key(r):r for r in first},{key(r):r for r in second}
    if set(a)!=set(b) or len(a)!=len(first) or len(b)!=len(second):
        raise ValueError('GT instances must match without duplicates or deleted failures')
    groups=defaultdict(list)
    for name in sorted(a):
        groups[families.get(name[0],name[0])].append(name)
    def err(rows,k):
        return rows[k]['error_3d_m'] if rows[k]['status']=='localized' else np.nan
    chunks=[(np.asarray([err(a,k) for k in keys],float),np.asarray([err(b,k) for k in keys],float)) for keys in groups.values()]
    if not chunks:
        raise ValueError('No paired trajectories')
    def stats(x,y):
        valid=np.isfinite(x)&np.isfinite(y)
        return [np.sqrt(np.nanmean(y*y))-np.sqrt(np.nanmean(x*x)),
            np.nanquantile(y,.95)-np.nanquantile(x,.95),np.mean(y<=1)-np.mean(x<=1),
            np.mean(y<=3)-np.mean(x<=3),np.mean(x[valid]**2-y[valid]**2)]
    rng=np.random.default_rng(seed); differences=[]
    for _ in range(draws):
        ids=rng.integers(len(chunks),size=len(chunks))
        x=np.concatenate([chunks[i][0] for i in ids]);y=np.concatenate([chunks[i][1] for i in ids])
        differences.append(stats(x,y))
    x=np.concatenate([c[0] for c in chunks]);y=np.concatenate([c[1] for c in chunks])
    out=dict(unit='complete paired scene family or episode',groups=len(chunks),draws=draws,seed=seed,
        total_GT_pairs=len(a),difference_direction='new method minus control; net_MSE_gain positive favors new',
        common_localized=int(np.sum(np.isfinite(x)&np.isfinite(y))),
        status_changes=int(np.sum(np.isfinite(x)!=np.isfinite(y))))
    for i,name in enumerate(('rmse_m','p95_m','all_GT_1m_success','all_GT_3m_success','net_MSE_gain_m2')):
        out[name+'_delta']=float(stats(x,y)[i])
        out[name+'_delta_ci95']=np.quantile(differences,[.025,.975],axis=0)[:,i].tolist()
        out[name+'_delta_ci98_333']=np.quantile(differences,[.05/6,1-.05/6],axis=0)[:,i].tolist()
    common=np.isfinite(x)&np.isfinite(y)
    transition={}
    for before in (False,True):
        for after in (False,True):
            transition[f'{"inside" if before else "outside"}_to_{"inside" if after else "outside"}']=int(np.sum(((x<=1)==before)&((y<=1)==after)))
    out['all_GT_1m_transition']=transition
    out['worsened_fraction_common_localized']=float(np.mean(y[common]>x[common]+1e-6)) if common.any() else None
    out['improved_fraction_common_localized']=float(np.mean(y[common]<x[common]-1e-6)) if common.any() else None
    return out


def compare(exp):
    protocol=read(exp/'PROTOCOL.json')
    original=read(protocol['data']['split_manifest'])
    families={r['episode']:r.get('scene_family_id',r['episode']) for r in original['episodes']}
    members={m:[json.loads(s) for s in (exp/'evaluation/val'/m/'members.jsonl').read_text().splitlines()] for m in METHODS}
    metrics={m:read(exp/'evaluation/val'/m/'summary.json') for m in METHODS}
    frames={m:[json.loads(s) for s in (exp/'evaluation/val'/m/'frames.jsonl').read_text().splitlines()] for m in METHODS}
    pairs={arm:paired(members['D32_parent'],members[arm],families) for arm in ARMS}
    decisions={}
    base=metrics['D32_parent']
    for arm in ARMS:
        s=metrics[arm];p=pairs[arm]
        checks=dict(rmse95_upper_negative=p['rmse_m_delta_ci95'][1]<0,
            P95_nonregression=s['member_position_error_m']['p95_abs']<=base['member_position_error_m']['p95_abs'],
            all_GT_1m_nonregression=s['success_ratio_1m']>=base['success_ratio_1m'],
            localized_count_nonregression=s['localized_matched_members']>=base['localized_matched_members'])
        decisions[arm]=dict(checks=checks,development_criterion_passed=all(checks.values()),
            multiplicity_adjusted_rmse_upper_negative=p['rmse_m_delta_ci98_333'][1]<0,
            independent_generalization_established=False,default_install=False)
    strata={}
    for low,high in ((0,150),(150,250),(250,350),(350,float('inf'))):
        name=f'range_{low}_{high}_m'
        strata[name]={}
        for method in METHODS:
            chosen=[r for r in members[method] if low<=np.linalg.norm(r['truth_position_base_m'])<high]
            # Fixed truth geometry defines offline strata only, never model input.
            strata[name][method]=dict(member_count=len(chosen),localized_count=sum(r['status']=='localized' for r in chosen),
                all_GT_1m_success=sum(r['status']=='localized' and r['error_3d_m']<=1 for r in chosen)/len(chosen) if chosen else None,
                rmse_m=float(np.sqrt(np.mean([r['error_3d_m']**2 for r in chosen if r['status']=='localized']))) if any(r['status']=='localized' for r in chosen) else None)
    out=dict(status='complete',metrics=metrics,paired_vs_fresh_D=pairs,decisions=decisions,strata=strata,
        primary='sequence',validation_only=True,production_default_changed=False,more_experiments_started=False)
    write(exp/'comparison.json',out)
    lines=['# D 九候选证据与视觉初值独立消融\n',
        '本轮固定三个单项：sequence 在原两轮回归器内部增加九候选完整序列的零初始化残差；aligned 只把首轮查谱中心改为全局混合基点；roi 只在成员编码中加入冻结检测器1024维原生ROI特征。网格仍9+9点，连续更新界仍16米；没有最终位置后挂小头、额外IQ拟合、多帧或候选扩窗。\n',
        '三个融合网络均从零训练20,000步，原检测器冻结；GPU1/4/5、每卡16、全局48、seed20261118、原sampler与loss。基础参数初始哈希与已注册带宽32MHz原D对照一致；新增模块额外参数及计算另列，不宣称参数量或GPU时间完全相同。D_deployed是历史发布参考，因果比较使用D32_parent。\n',
        '**评价范围是历史开发验证集150个完整episode；它同时参与checkpoint选择，已经有历史选型暴露。本轮不打开最终测试，不建立独立新场景泛化结论，不更新生产默认。**\n',
        '| 方法 | 参数量 | 3D RMSE m | P95 m | 全GT≤1m | ≤3m | 有效定位/GT | 缓存输入推理ms/帧 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    frozen=read(exp/'frozen_validation_identity.json')
    for method,s in metrics.items():
        e=s['member_position_error_m']; latency=s['inference_seconds_per_frame']['mean']
        lines.append(f"| {method} | {frozen['models'][method]['model_parameter_count']} | {e['rmse']:.4f} | {e['p95_abs']:.4f} | {s['success_ratio_1m']:.2%} | {s['success_ratio_3m']:.2%} | {s['localized_matched_members']}/{s['truth_members']} | {latency*1000:.2f} |")
    lines+=['','配对重采样为完整scene-family/episode，2000次；相邻帧不作为独立轨迹。净平方误差收益只在共同有效成员上计算，失败和漏检仍完整计入全部GT成功率。下列区间均为新方案减同预算D的RMSE差。','']
    for arm in ARMS:
        p=pairs[arm];d=decisions[arm]
        lines.append(f"- {arm}：差 {p['rmse_m_delta']:+.4f} m，95%区间 {p['rmse_m_delta_ci95']}；三比较保守98.33%区间 {p['rmse_m_delta_ci98_333']}。开发门槛{'通过' if d['development_criterion_passed'] else '未通过'}；1米内→外 {p['all_GT_1m_transition']['inside_to_outside']}，外→内 {p['all_GT_1m_transition']['outside_to_inside']}。")
    lines+=['','阶段初深、全局提案/混合、第一轮、第二轮的误差和有害更新比例在各summary.json的stage_diagnostics、stage_update_diagnostics；逐成员stages、逐帧光心/射线和整episode结果保存在evaluation/val/。range分层是真值辅助离线分析，不进入推理。','',
        '时延表只含已缓存传感特征的预测和阶段捕获，不含检测/FFT/读盘。ROI的一次性真实导出耗时与显存见roi/export_receipt.json或roi/performance.json；不能把缓存读取当作实时ROI计算成本。训练实际墙钟时间见training/*/run_manifest.json，显存峰值见各summary.json。','',
        '没有结果后追加组合、阈值、更多种子或网络。未实现分轮独立输出头、不确定度门控、角度分区、观察机参照新分支；这些仍需单独实验与独立校准。本轮到此结束，无论正负结果原D都保持默认。','',
        '复核：`../.venv/bin/python -m d_evidence_study.evaluate --experiment results/d_evidence_20261124 --phase validate`。各权重在training/{sequence,aligned,roi}/depth.rmse.best.pt；协议、输入配置、冻结身份、运行来源分别见PROTOCOL.json、configs/、frozen_validation_identity.json、run_plan.json与training_route_receipt.json。']
    (exp/'REPORT_CN.md').write_text('\n'.join(lines)+'\n')


def validate_completion(exp):
    exp=Path(exp).resolve()
    if read(exp/'frozen_validation_identity.json')!=identities(exp):
        raise ValueError('Final identity no longer matches')
    frozen=read(exp/'frozen_validation_identity.json');files={}
    for method in METHODS:
        out=exp/'evaluation/val'/method
        expected=dict(method=method,split='val',model_identity=frozen['models'][method],source_sha256=frozen['source_sha256'])
        if read(out/'completion_receipt.json')!=expected:
            raise ValueError('Invalid completed evaluation receipt: '+method)
        for name in ('summary.json','episodes.json','members.jsonl','frames.jsonl','detections.jsonl','run_info.json','input_manifest.json','completion_receipt.json'):
            files[str((out/name).relative_to(exp))]=sha(out/name)
        if method in ARMS:
            for name in ('candidate_diagnostics.json','candidate_queries.jsonl'):
                files[str((out/name).relative_to(exp))]=sha(out/name)
    for name in ('comparison.json','REPORT_CN.md','frozen_validation_identity.json'):
        files[name]=sha(exp/name)
    return dict(status='complete',evidence_sha256=files,production_default_changed=False,test_opened=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,default=ROOT/'results/d_evidence_20261124')
    p.add_argument('--phase',choices=('freeze','evaluate','compare','validate','finalize'),required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();exp=a.experiment.resolve();torch.set_num_threads(1)
    if a.phase in ('freeze','finalize'):freeze(exp)
    if a.phase in ('evaluate','finalize'):
        for method in METHODS:evaluate_method(exp,method,a.device)
    if a.phase in ('compare','finalize'):compare(exp)
    if a.phase in ('validate','finalize'):print(json.dumps(validate_completion(exp),ensure_ascii=False))


if __name__=='__main__':
    main()
