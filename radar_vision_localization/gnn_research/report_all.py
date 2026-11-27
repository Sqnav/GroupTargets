"""Chinese report of the explicitly named seven-method and MCE-GNN runs.

Only committed TRAIN/DEV experiment artifacts are read. Missing/invalid/failed
results remain visible; a report is final only after all declared evaluations
and both experiment controllers have completed. This module never opens TEST,
raw observations, labels, neural weights, or deployment entries.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path
import tempfile
import os
import time
from zoneinfo import ZoneInfo

import numpy as np

BASE_NAMES = {1:'仅 FFT',2:'仅 RGB 深度',3:'单次查询雷视融合',4:'FFT＋Kalman',
              5:'RGB＋Kalman',6:'单帧融合＋Kalman',7:'世界状态进入融合内部'}
GRAPH_NAMES = {'isolated':'B：候选图，不共享成员间证据',
               'shared':'C：物理共享雷达证据 GNN',
               'shuffled':'D：打乱物理共享边'}
FIXED_SCHEMA = 'rvloc.fixed_roi_evaluation.v1'
SET_SCHEMA = 'rvloc.evaluation.v1'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    fd,tmp=tempfile.mkstemp(prefix='.writing-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as s:s.write(raw)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def require(condition,message):
    if not condition:raise ValueError(message)


def optional(path):
    return read(path) if Path(path).is_file() else None


def artifact(receipt_path,root,filename,expected_identity=None):
    """Authenticate a declared committed artifact; never glob other runs."""
    receipt_path=Path(receipt_path)
    if not receipt_path.is_file():return None
    receipt=read(receipt_path)
    require(receipt.get('status')=='completed','收据尚未完成：'+str(receipt_path))
    if expected_identity is not None:
        require(receipt.get('identity_sha256')==expected_identity,'收据不属于指定主实验')
    candidates=[Path(p).resolve() for p in receipt.get('artifacts',{}) if Path(p).name==filename]
    require(len(candidates)==1,'收据产物不唯一：'+filename)
    path=candidates[0]
    require(path.is_relative_to(Path(root).resolve()),'产物越出指定实验目录')
    require(path.is_file() and sha(path)==receipt['artifacts'][str(path)],'产物缺失或 SHA 不一致：'+str(path))
    return path,read(path),receipt


def population(value):
    rows=value.get('rows')
    require(isinstance(rows,list),'评价没有逐成员 rows')
    keys=[(r['episode_id'],r['frame_idx'],r['gt_member_id']) for r in rows]
    require(len(keys)==len(set(keys)),'评价逐成员键重复')
    return frozenset(keys)


def validate_evaluation(value,scope,manifest_sha,method=None):
    require(value.get('schema')==(FIXED_SCHEMA if scope=='fixed_roi' else SET_SCHEMA),'评价匹配口径/schema 不一致')
    require(value.get('split')=='dev','仅允许冻结后的 DEV 成绩；不能把 TRAIN 当 DEV')
    require(value.get('manifest_sha256')==manifest_sha,'评价数据身份不一致')
    if method is not None:require(value.get('method_id')==method,'评价方法身份不一致')
    keys=population(value)
    summary=value.get('summary',{})
    require(summary.get('gt_members')==len(keys),'GT 分母与逐成员 rows 不一致')
    errors=np.asarray([r['error_m'] for r in value['rows'] if r.get('error_m') is not None],float)
    require(np.isfinite(errors).all() and np.all(errors>=0),'逐成员误差不是有限非负值')
    actual=float(np.sqrt(np.mean(errors**2))) if len(errors) else None
    stated=summary.get('position_rmse_3d_m')
    require((actual is None and stated is None) or actual is not None and isinstance(stated,(int,float))
            and math.isclose(actual,float(stated),rel_tol=1e-7,abs_tol=1e-7),'RMSE 与逐成员误差复算不一致')
    return keys


def paired_fixed(first,second,*,draws=2000,seed=20261126):
    """Paired episode bootstrap, second minus first, fixed ROI labels only."""
    for value in (first,second):
        require(value.get('schema')==FIXED_SCHEMA and value.get('split')=='dev','配对仅接受 DEV 固定 ROI 口径')
    require(first['manifest_sha256']==second['manifest_sha256'],'配对数据身份不同')
    require(first.get('frozen_index_sha256')==second.get('frozen_index_sha256'),'固定 ROI 标注身份不同')
    require(population(first)==population(second),'配对 GT 人群不同；不得静默取交集')
    aa={(r['episode_id'],r['frame_idx'],r['gt_member_id']):r for r in first['rows']}
    bb={(r['episode_id'],r['frame_idx'],r['gt_member_id']):r for r in second['rows']}
    common=[key for key in aa if aa[key]['error_m'] is not None and bb[key]['error_m'] is not None]
    episodes=sorted({key[0] for key in aa})
    grouped={ep:(np.asarray([aa[k]['error_m'] for k in common if k[0]==ep],float),
                 np.asarray([bb[k]['error_m'] for k in common if k[0]==ep],float)) for ep in episodes}
    def statistic(chosen):
        a=np.concatenate([grouped[ep][0] for ep in chosen]);b=np.concatenate([grouped[ep][1] for ep in chosen])
        if not len(a):return None
        return [float(np.sqrt(np.mean(b*b))-np.sqrt(np.mean(a*a))),
                float(np.quantile(b,.95)-np.quantile(a,.95)),
                float(np.quantile(b,.99)-np.quantile(a,.99))]
    point=statistic(episodes) if episodes else None
    samples=[]
    if len(episodes)>1:
        require(type(draws) is int and draws>=20,'bootstrap 至少 20 次')
        rng=np.random.default_rng(seed)
        for ix in rng.integers(len(episodes),size=(draws,len(episodes))):
            value=statistic([episodes[i] for i in ix])
            if value is not None:samples.append(value)
    intervals={}
    for i,name in enumerate(('rmse_delta_m','p95_delta_m','p99_delta_m')):
        if samples:
            values=np.asarray(samples)[:,i]
            intervals[name]={'95%':np.quantile(values,[.025,.975]).tolist(),
                             '97.5%':np.quantile(values,[.0125,.9875]).tolist()}
    return dict(schema='rvloc.fixed_roi_paired_report.v1',direction='second minus first',
        methods=[first['method_id'],second['method_id']],manifest_sha256=first['manifest_sha256'],
        frozen_index_sha256=first.get('frozen_index_sha256'),all_gt=len(aa),common_located=len(common),
        first_only_located=sum(aa[k]['error_m'] is not None and bb[k]['error_m'] is None for k in aa),
        second_only_located=sum(bb[k]['error_m'] is not None and aa[k]['error_m'] is None for k in aa),
        deltas=dict(zip(('rmse_delta_m','p95_delta_m','p99_delta_m'),point)) if point is not None else None,
        intervals=intervals,episode_count=len(episodes),draws=len(samples),seed=seed,
        note='共同有效成员的配对差；必须并读全 GT 成功率与缺失。多项探索性比较未做多重检验校正。')


def fmt(value,digits=4):
    if value is None:return '—'
    if isinstance(value,(int,float)):
        return str(value) if isinstance(value,int) else f'{value:.{digits}f}'
    return str(value).replace('|','／').replace('\n',' ')


def ci(summary,level):
    item=summary.get('uncertainty',{}).get('intervals',{}).get('position_rmse_3d_m',{}).get(level)
    return '['+', '.join(fmt(x) for x in item)+']' if item is not None else '未生成'


def _state(path):
    value=optional(path)
    return value.get('status','unknown') if value else 'not_started_or_unrecorded'


def _train_progress(main,method):
    stage=main/'stages'/f'train_g{method}'
    # Only attempts of this explicitly requested stage; never other experiments.
    matches=sorted(stage.glob('attempt_*/results/progress.json')) if stage.is_dir() else []
    return read(matches[-1]) if matches else None


def gather(main_run,fixed_results,gnn_run,*,train_metrics=(),fixed_index=None):
    main=Path(main_run).resolve();fixed=Path(fixed_results).resolve();gnn=Path(gnn_run).resolve()
    identity=optional(main/'execution_identity.json')
    manifest_sha=identity.get('manifest_sha256') if identity else None
    result=dict(schema='rvloc.all_versions_summary.v1',generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        status='waiting',main_run=str(main),fixed_results=str(fixed),gnn_run=str(gnn),
        manifest_sha256=manifest_sha,main_identity=identity,
        controller_status=dict(main=_state(main/'status.json'),fixed=_state(fixed/'watch_status.json'),gnn=_state(gnn/'status.json')),
        versions={},errors=[],pairwise={},uncompleted=[],independent_test_claim=False,historical_TEST_opened=False)
    fixed_sha=sha(fixed_index) if fixed_index and Path(fixed_index).is_file() else None
    values={};reference_population=None
    for method in range(1,8):
        key=f'G{method}';item=dict(name=BASE_NAMES[method],status='waiting',set_evaluation=None,
            fixed_evaluation=None,training=None,training_progress=_train_progress(main,method) if method in (2,3,7) else None)
        for scope,receipt_path in [('set',main/'stage_receipts'/f'dev_evaluate_g{method}.json')]+(
                [('fixed_roi',fixed/f'g{method}'/'receipt.json')] if method in (2,3,5,6,7) else []):
            name='fixed_evaluation' if scope=='fixed_roi' else 'set_evaluation'
            try:
                found=artifact(receipt_path,fixed if scope=='fixed_roi' else main,'evaluation.json',
                    expected_identity=identity.get('sha256') if identity and scope=='set' else None)
                if found:
                    path,value,receipt=found
                    if scope=='fixed_roi':
                        stage_path=Path(receipt['stage_receipt']).resolve()
                        require(stage_path.is_relative_to(main) and sha(stage_path)==receipt['stage_receipt_sha256'],
                                '固定评价的推理收据不属于指定主实验或SHA变化')
                        stage=read(stage_path)
                        require(stage.get('identity_sha256')==identity.get('sha256') and stage.get('status')=='completed'
                                and stage.get('stage',{}).get('method')==method,'固定评价使用另一主实验的推理')
                    pop=validate_evaluation(value,scope,manifest_sha,method)
                    if reference_population is None:reference_population=pop
                    require(pop==reference_population,'GT 人群与同轮其他方法不同')
                    if scope=='fixed_roi':
                        if fixed_sha is None:fixed_sha=value.get('frozen_index_sha256')
                        require(value.get('frozen_index_sha256')==fixed_sha,'固定 ROI 标注身份与本轮不一致')
                    item[name]=dict(path=str(path),sha256=sha(path),summary=value['summary'],cost=value.get('cost'),receipt=receipt)
                    values[(key,scope)]=value
            except (ValueError,KeyError,TypeError,OSError) as error:
                result['errors'].append(f'{key} {scope}: {error}');item['status']='invalid'
        if method in (2,3,7):
            try:
                found=artifact(main/'stage_receipts'/f'train_g{method}.json',main,'receipt.json',
                    expected_identity=identity.get('sha256') if identity else None)
                if found:
                    path,value,_=found
                    require(value.get('status')=='trained' and value.get('method_id')==method,'训练收据方法/状态错误')
                    require(value.get('training_identity',{}).get('manifest_sha256')==manifest_sha,'训练数据身份不同')
                    item['training']=dict(path=str(path),sha256=sha(path),receipt=value)
            except (ValueError,KeyError,TypeError,OSError) as error:result['errors'].append(f'{key} train: {error}');item['status']='invalid'
        complete=item['set_evaluation'] is not None and (method in (1,4) or item['fixed_evaluation'] is not None)
        if complete and item['status']!='invalid':item['status']='completed'
        result['versions'][key]=item
    for variant,label in GRAPH_NAMES.items():
        key='GNN_'+variant;folder=gnn/variant
        item=dict(name=label,status='waiting',set_evaluation=None,fixed_evaluation=None,training=None,
                  training_progress=optional(folder/'train/progress.json'),graph_metrics=None)
        try:
            trained=artifact(folder/'stage_receipts/train.json',gnn,'receipt.json')
            if trained:
                train_path,train,train_stage=trained
                require(train_stage.get('variant')==variant and train_stage.get('kind')=='train','GNN 训练阶段身份错误')
                require(train.get('status')=='trained','GNN 训练没有完成')
                tid=train.get('training_identity',train.get('identity',{}))
                candidate=train.get('manifest_sha256',tid.get('manifest_sha256'))
                require(candidate==manifest_sha,'GNN 训练数据身份与基线不同')
                item['training']=dict(path=str(train_path),sha256=sha(train_path),receipt=train,stage_receipt=train_stage)
            evaluated=artifact(folder/'stage_receipts/evaluation.json',gnn,'evaluation.json')
            if evaluated:
                path,value,evaluation_stage=evaluated
                require(evaluation_stage.get('variant')==variant and evaluation_stage.get('kind')=='evaluation','GNN 评价阶段身份错误')
                pop=validate_evaluation(value,'fixed_roi',manifest_sha,'mce_'+variant)
                if reference_population is None:reference_population=pop
                require(pop==reference_population,'GNN 与基线 GT 人群不同')
                if fixed_sha is None:fixed_sha=value.get('frozen_index_sha256')
                require(value.get('frozen_index_sha256')==fixed_sha,'GNN 与基线固定 ROI 标签不同')
                require(item['training'] is not None,'GNN 评价缺少已完成训练收据')
                cost=value.get('cost')
                if cost is None:
                    inf=optional(folder/'inference/reproduction_receipt.json')
                    cost=inf.get('cost') if inf else None
                item['fixed_evaluation']=dict(path=str(path),sha256=sha(path),summary=value['summary'],cost=cost,receipt=evaluation_stage)
                values[(key,'fixed_roi')]=value
                item['graph_metrics']=optional(folder/'evaluation/graph_metrics.json')
                item['status']='completed'
        except (ValueError,KeyError,TypeError,OSError) as error:
            result['errors'].append(f'{key}: {error}');item['status']='invalid'
        result['versions'][key]=item
    result['frozen_index_sha256']=fixed_sha
    result['frozen_train_metrics']=[]
    for metric in train_metrics:
        path=Path(metric)
        if not path.is_file():continue
        try:
            value=read(path)
            require(value.get('scope')=='fixed_checkpoint_train_full' and value.get('split')=='train','冻结 TRAIN 诊断口径错误')
            require(value.get('identity',{}).get('manifest_sha256')==manifest_sha,'冻结 TRAIN 诊断数据身份不同')
            result['frozen_train_metrics'].append(dict(path=str(path),sha256=sha(path),metric=value))
        except (ValueError,KeyError,TypeError,OSError) as error:result['errors'].append(f'TRAIN diagnostic: {error}')
    result['uncompleted']=[key for key,value in result['versions'].items() if value['status']!='completed']
    failed=any(value in ('failed','blocked','cancelled','terminated','incomplete') for value in result['controller_status'].values())
    if result['errors'] or failed:result['status']='incomplete'
    elif not result['uncompleted'] and all(result['controller_status'][key]=='completed' for key in ('main','fixed','gnn')):
        result['status']='completed'
    return result,values


def accuracy_table(lines,items,field):
    lines+=['| 版本 | 状态 | RMSE/m | MAE/m | P95/m | P99/m | RMSE 95%区间 | RMSE 97.5%区间 |',
            '|---|---|---:|---:|---:|---:|---|---|']
    for key,item in items:
        value=item.get(field);s=value['summary'] if value else {}
        lines.append('| '+ ' | '.join([key+' '+item['name'],item['status'],fmt(s.get('position_rmse_3d_m')),
            fmt(s.get('mae_m')),fmt(s.get('p95_m')),fmt(s.get('p99_m')),ci(s,'95%') if value else '未完成',ci(s,'97.5%') if value else '未完成'])+' |')
    lines+=['','| 版本 | GT数 | 有效定位 | 无定位 | >5m | >10m | 全GT≤1m | 全GT≤3m | 全GT≤5m | 未匹配输出 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,item in items:
        value=item.get(field);s=value['summary'] if value else {};success=s.get('all_gt_success',{})
        located=s.get('localized_members',s.get('matched_members'))
        extra=s.get('unmatched_outputs',s.get('unmatched_output_count'))
        lines.append('| '+' | '.join([key,fmt(s.get('gt_members')),fmt(located),fmt(s.get('no_localization_members')),
            fmt(s.get('over_5m')),fmt(s.get('over_10m'))]+[fmt(100*success[k],2)+'%' if success.get(k) is not None else '—' for k in ('1m','3m','5m')]+[fmt(extra)])+' |')
    lines+=['']


def parameter_counts(summary):
    """Count architecture parameters only when recorded configuration exists."""
    for key,item in summary['versions'].items():
        train=item.get('training');rec=train['receipt'] if train else {}
        identity=rec.get('training_identity',rec.get('identity',{}))
        number=rec.get('parameter_count',identity.get('parameter_count'))
        source='actual training receipt' if number is not None else None
        if number is None and key in ('G2','G3','G7') and rec.get('group'):
            raw=identity.get('experiment_config')
            if isinstance(raw,dict) and isinstance(raw.get('model'),dict):
                from rvloc.models import build_model
                options=dict(raw['model'])
                if 'world' in raw:options['newworld']=dict(raw['world'])
                model=build_model(rec['group'],options)
                number=sum(p.numel() for p in model.parameters() if p.requires_grad)
                source='CPU architecture reconstruction from recorded config; no checkpoint weights opened'
        item['parameter_count_info']=dict(count=number,source=source,includes_frozen_DINO=False)


def difficult_subsets(summary,values,output):
    """Freeze this run's G3 hardest located members once; never score missing as zero."""
    baseline=values.get(('G3','fixed_roi'))
    if baseline is None:return
    reference=summary['versions']['G3']['fixed_evaluation']
    path=output/'baseline_G3_difficult_subsets.json'
    frozen=optional(path)
    if frozen:
        require(frozen['baseline_evaluation_sha256']==reference['sha256'],
                '已冻结困难集合的G3基线发生变化；不得重新挑选有利集合')
    else:
        rows=sorted((r for r in baseline['rows'] if r['error_m'] is not None),
                    key=lambda r:(-r['error_m'],r['episode_id'],r['frame_idx'],r['gt_member_id']))
        frozen=dict(schema='rvloc.baseline_difficult_subsets.v1',
            baseline='G3 fixed ROI DEV, declared final checkpoint',
            baseline_evaluation_sha256=reference['sha256'],baseline_evaluation_path=reference['path'],
            baseline_located_members=len(rows),baseline_unlocated_members=len(baseline['rows'])-len(rows),
            selection='ceil(percent * located_baseline_count), descending error with deterministic GT-key ties',
            diagnostic_only=True,used_for_training_or_threshold_selection=False,
            subsets={str(percent):[[r['episode_id'],r['frame_idx'],r['gt_member_id']]
                     for r in rows[:int(math.ceil(len(rows)*percent/100.))]] for percent in (1,5,10)})
        write(path,frozen)
    stats={}
    for (key,scope),evaluation in values.items():
        if scope!='fixed_roi':continue
        keyed={(r['episode_id'],r['frame_idx'],r['gt_member_id']):r for r in evaluation['rows']}
        result={}
        for percent,keys in frozen['subsets'].items():
            require(all(tuple(k) in keyed for k in keys),'困难集合GT键不属于当前评价人群')
            errors=[keyed[tuple(k)]['error_m'] for k in keys]
            observed=[v for v in errors if v is not None]
            missing=len(errors)-len(observed)
            sse=float(np.square(observed).sum()) if observed else None
            result[percent]=dict(total_members=len(keys),located_members=len(observed),missing_members=missing,
                observed_sse_m2=sse,complete_sse_m2=sse if not missing else None,
                missing_not_scored_as_zero=True)
        stats[key]=result
    summary['difficult_subsets']=dict(frozen_path=str(path),frozen_sha256=sha(path),
        definition=frozen,versions=stats,
        note='本轮G3固定基线困难集合，非历史旧D清单；有漏定位时只列已观测SSE，不与完整SSE直接比较。')


def render(summary):
    versions=summary['versions'];done=summary['status']=='completed'
    lines=['# 全部模型与共享雷达证据 GNN 实验中文报告','',
        f'报告状态：**{"全部声明实验已完成" if done else "阶段报告，尚未完成全部实验"}**；生成时间：{summary["generated_at"]}。',
        '本报告只汇总指定的本轮七组实验和三个新增 GNN 消融分支，不扫描历史 TEST。历史原 D 的约1.35m和早期IQ风险策略不混入本轮同口径排名。','',
        '## 数据、模型与评价合同','',
        '- 32MHz，共用冻结 DINOv3 检测/ROI、相机光心作为射线起点与观察机回波参考位置。机体导航安装偏移不作为新的定位网络输入。',
        '- 本轮为视觉均值/标准差先验 → 一次雷达取证 → 连续最终深度；沿用 μ_v＋σ_v×[-3,-2,-1,-0.5,0,0.5,1,2,3]。附件旧两轮结构和固定0、±2、±4、±8、±16m不用于本轮。',
        '- 主回归损失为Huber＋0.05深度MSE＋0.1视觉Gaussian NLL；GNN候选分类辅助权重默认0，避免同时改变图结构和训练目标。G7另有原时序辅助项。',
        '- 正常推理不读取GT位置/ID/人数、未来帧或仿真散射系数。GT仅用于监督和预测提交后的离线评价。没有IQ风险校正。',
        '- 含视觉五组及GNN以相同冻结ROI→GT标注计算成员误差；FFT两组没有ROI行。原七组共同点集评价作为独立口径保留，不把其RMSE与固定成员RMSE混成一张排行榜。',
        '- RMSE/P95/P99仅使用有效定位成员；全GT成功率把无定位计为失败，额外输出另列。帧/成员数不代表独立样本数，置信区间按episode重采样。',
        f'- manifest SHA：`{summary.get("manifest_sha256")}`；固定标注 SHA：`{summary.get("frozen_index_sha256")}`。','',
        '## 七组共同点集 DEV 评价','',
        '逐帧以欧氏距离和做一对一匹配，再计算三维RMSE；该口径不证明成员身份或跟踪关联正确。']
    accuracy_table(lines,[(f'G{i}',versions[f'G{i}']) for i in range(1,8)],'set_evaluation')
    lines+=['## 固定 ROI 成员 DEV 评价','',
        '输出始终与原检测行绑定，按预先固定的GT成员评分，不能在输出偏向另一成员后重新挑选GT。A为当前G3；B/C/D除共享边关系外应匹配参数和训练预算。']
    fixed_items=[(f'G{i}',versions[f'G{i}']) for i in (2,3,5,6,7)]+[(f'GNN_{v}',versions['GNN_'+v]) for v in GRAPH_NAMES]
    accuracy_table(lines,fixed_items,'fixed_evaluation')
    lines+=['## TRAIN监测（不作为DEV成绩）','',
        '| 版本 | 已完成训练epoch | TRAIN在线累计RMSE/m | 训练状态 | 当前运行耗时/s |',
        '|---|---:|---:|---|---:|']
    for key in ['G2','G3','G7']+['GNN_'+v for v in GRAPH_NAMES]:
        item=versions[key];train=item.get('training');rec=train['receipt'] if train else {};progress=item.get('training_progress') or {}
        history=rec.get('history',[]);last=history[-1] if history else progress
        lines.append('| '+' | '.join([key,fmt(rec.get('completed_epochs',rec.get('epochs',progress.get('epoch')))),
            fmt(last.get('online_fixed_train_rmse_m')),rec.get('status',progress.get('status','未开始或未记录')),fmt(rec.get('elapsed_s'))])+' |')
    lines+=['','在线累计RMSE包含训练中持续变化的参数，不能代替固定权重评价；loss也不能直接换算为米制RMSE。续训耗时只记录当前运行段，不能抹去来源检查点之前的训练成本。','']
    if summary['frozen_train_metrics']:
        lines+=['冻结检查点的全量TRAIN诊断：','', '| 方法 | 检查点epoch | RMSE/m | 成员数 | 诊断耗时/s |','|---|---:|---:|---:|---:|']
        for record in summary['frozen_train_metrics']:
            m=record['metric'];lines.append('| '+' | '.join([str(m.get('method_id')),fmt(m.get('completed_epochs')),fmt(m.get('position_rmse_3d_m')),fmt(m.get('matched_members')),fmt(m.get('elapsed_s'))])+' |')
    else:lines+=['未提供已认证的完整冻结检查点TRAIN诊断文件；不从在线损失或图中曲线推算。']
    lines+=['','## 共享雷达证据GNN：机制与消融','',
        '现有G3已经含成员Transformer与全局雷达注意力，不能说成员完全独立。新增假设是：候选通过同一个真实CNN网格证据节点交换共享信息，比仅增加逐成员候选参数更有效。证据节点不是候选局部特征的独立复制，也不强制一峰只归属一个成员。',
        'RA与RE来自同一份IQ，保持候选几何配对；不把二者当独立似然相乘。图只能组织已保留的RA/RE证据，不能创造已丢失的IQ信息。',
        '候选softmax权重仅用于证据汇聚，默认候选CE权重为0时更不能称为校准的深度后验；视觉σ也需通过实际残差覆盖率验证。GT候选覆盖/分类oracle不能当作正常推理定位成绩。',
        '当前PSF边权是max投影RA/RE上的query-conditioned Dirichlet代理，不是经过独立认证的完整IQ前向似然。物理共享假设的价值必须看C对A/B/D的实际固定成员成绩及共享边覆盖。',
        '邻域注意力可参考[GAT原文](https://arxiv.org/abs/1710.10903)，保留求和/度数统计的表达力参考[GIN原文](https://arxiv.org/abs/1810.00826)；这些文献不能保证本任务精度提升。','']
    for variant in GRAPH_NAMES:
        item=versions['GNN_'+variant];graph=item.get('graph_metrics')
        lines+=[f'### {GRAPH_NAMES[variant]}','']
        if graph:
            compact={k:v for k,v in graph.items() if not isinstance(v,list)}
            lines+=['实际图统计（定义以原始graph_metrics.json为准）：','', '```json',json.dumps(compact,ensure_ascii=False,indent=2)[:12000],'```','']
        else:lines+=['图覆盖率、共享节点/边数量及候选召回尚未生成；不按零计，不据此宣布机制有效。','']
    lines+=['## 配对差与不确定性','',
        '以下差值为第二方法减第一方法；共同有效GT成员配对，缺失差异另列。负差表示该统计量下降。95%和97.5%区间为episode bootstrap百分位区间；P99点估计下降不等于尾部安全已证实，三项机制比较为探索性结果。','']
    if summary['pairwise']:
        lines+=['| 对照 → 新方法 | 共同成员 | RMSE差/m | 95%区间 | 97.5%区间 | P95差/m | P99差/m |','|---|---:|---:|---|---|---:|---:|']
        for name,entry in summary['pairwise'].items():
            p=entry['result'];d=p.get('deltas') or {};iv=p.get('intervals',{}).get('rmse_delta_m',{})
            def interval(level):return '['+', '.join(fmt(x) for x in iv[level])+']' if level in iv else '未生成'
            lines.append('| '+' | '.join([name,fmt(p.get('common_located')),fmt(d.get('rmse_delta_m')),interval('95%'),interval('97.5%'),fmt(d.get('p95_delta_m')),fmt(d.get('p99_delta_m'))])+' |')
    else:lines+=['可比的固定成员评价尚未全部提交，配对区间尚未生成。']
    lines+=['','## 实际训练预算与续训来源','',
        '| 版本 | 声明epoch | 实际优化更新 | seed | 学习率 | 每更新帧数 | 续训来源 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for key in ['G2','G3','G7']+['GNN_'+v for v in GRAPH_NAMES]:
        item=versions[key];tr=item.get('training');rec=tr['receipt'] if tr else {}
        tid=rec.get('training_identity',rec.get('identity',{}));raw=tid.get('experiment_config',{})
        setting=tid.get('training_settings',raw.get('training',rec.get('training_settings',{})));engine=tid.get('engine',{})
        resume=rec.get('resume',tid.get('resume'));origin=resume.get('path') if isinstance(resume,dict) else None
        lines.append('| '+' | '.join([key,fmt(rec.get('epochs')),fmt(rec.get('optimizer_updates')),
            fmt(setting.get('seed')),fmt(setting.get('learning_rate')),fmt(engine.get('optimizer_frames',rec.get('batch_size',setting.get('bptt_steps')))),
            fmt(origin) if origin else ('从头训练' if rec else '未完成')])+' |')
    lines+=['','G2包含早期逐帧训练后续接批量内核的来源；其已完成epoch和AdamW状态保留，未重新从零训练。批量浮点归约不保证与原逐帧内核bitwise相同。G3和GNN从头训练的种子与预算以表内实际收据为准，不能用相同epoch数替代对优化步数/参数量的检查。','']
    hard=summary.get('difficult_subsets')
    lines+=['## 本轮G3固定困难成员','']
    if hard:
        lines+=[hard['note'],'', '| 版本 | 固定最差比例 | 固定成员数 | 有效数 | 漏定位 | 已观测SSE/m² | 无漏定位完整SSE/m² |', '|---|---:|---:|---:|---:|---:|---:|']
        for key,subsets in hard['versions'].items():
            for percent,record in subsets.items():
                lines.append('| '+' | '.join([key,percent+'%',fmt(record['total_members']),fmt(record['located_members']),fmt(record['missing_members']),fmt(record['observed_sse_m2']),fmt(record['complete_sse_m2'])])+' |')
    else:lines+=['G3固定DEV评价尚未提交，不能预先挑选本轮困难成员或把未评价成员记为不可辨识。']
    lines+=['','## 计算成本与身份','',
        '| 版本 | 已记录参数量 | 推理wall/s | 吞吐/帧每秒 | 设备/并行度 | 输入与计时范围 |','|---|---:|---:|---:|---|---|']
    for key,item in versions.items():
        train=item.get('training');rec=train['receipt'] if train else {};identity=rec.get('training_identity',rec.get('identity',{}))
        count=item.get('parameter_count_info',{}).get('count',rec.get('parameter_count',identity.get('parameter_count')))
        ev=item.get('fixed_evaluation') or item.get('set_evaluation');cost=ev.get('cost') or {} if ev else {}
        if not cost and item.get('set_evaluation'):cost=item['set_evaluation'].get('cost') or {}
        scope=cost.get('timing_scope','未记录')
        if cost.get('serial_latency_measured') is False:scope+='；未测串行单帧时延'
        lines.append('| '+' | '.join([key,fmt(count),fmt(cost.get('wall_s')),fmt(cost.get('throughput_frames_s')),
            str(cost.get('device','未记录'))+'/'+str(cost.get('parallel_workers','未记录')),fmt(scope)])+' |')
    lines+=['','参数量来自实际训练收据或已记录配置的CPU结构重建（来源保存在JSON）；两者均缺失时显示未记录。冻结DINO前端不计为定位分支新增参数。缓存ROI/RARE条件下的定位器计时不等于RGB/IQ完整端到端时延；FFT的并行worker耗时和不能当作实测串行单帧时延。','',
        '完整的checkpoint、数据、源码、续训来源与成本身份见 `all_model_results_summary.json` 中每个训练/评价收据。未覆盖默认模型、旧权重或部署入口。','',
        '## 未完成与适用范围','',
        '- 独立新航迹泛化、跨训练随机种子验证：本轮未声明完成。历史DEV已经参与过开发，不能作为独立TEST证据。',
        '- 历史旧D困难清单、几何混叠S1/S2/S3和小框分组：只有各评价文件明确生成时才可报告；本报告另列本轮G3固定困难集合，二者不能混称。',
        '- 无GT oracle定位、GT身份驱动关联和真实散射参数诊断不计为实际部署成绩。',
        '- 本轮以机制验证为目的；训练或图覆盖率改善不等同最终定位改善，不强行宣称GNN优于原G3。','']
    if summary['uncompleted']:lines+=['尚未完成/无有效评价：'+', '.join(summary['uncompleted'])+'。','']
    if summary['errors']:lines+=['数据/收据/评价一致性问题：','']+['- '+e for e in summary['errors']]+['']
    lines+=['本报告仅在七组共同评价、视觉固定成员评价、三个GNN分支和两套实验控制器均完成且身份一致时标记“全部完成”。','']
    return '\n'.join(lines)


def generate(main_run,fixed_results,gnn_run,output_dir,*,train_metrics=(),fixed_index=None,draws=2000,seed=20261126):
    output=Path(output_dir).expanduser().resolve()
    project=Path(__file__).resolve().parents[1]
    require(not output.is_relative_to(project),'报告必须存放到代码目录之外')
    output.mkdir(parents=True,exist_ok=True)
    summary,values=gather(main_run,fixed_results,gnn_run,train_metrics=train_metrics,fixed_index=fixed_index)
    parameter_counts(summary)
    try:difficult_subsets(summary,values,output)
    except (ValueError,KeyError,TypeError) as error:
        summary['errors'].append('fixed difficult subsets: '+str(error));summary['status']='incomplete'
    for first,second in [('G3','GNN_shared'),('GNN_isolated','GNN_shared'),('GNN_shuffled','GNN_shared')]:
        a=values.get((first,'fixed_roi'));b=values.get((second,'fixed_roi'))
        if a is None or b is None:continue
        try:
            identity=dict(first_sha256=summary['versions'][first]['fixed_evaluation']['sha256'],
                second_sha256=summary['versions'][second]['fixed_evaluation']['sha256'],draws=draws,seed=seed)
            path=output/'pairwise'/f'{first}_to_{second}.json'
            previous=optional(path)
            value=previous if previous and previous.get('identity')==identity else dict(identity=identity,result=paired_fixed(a,b,draws=draws,seed=seed))
            if previous!=value:write(path,value)
            summary['pairwise'][first+' → '+second]=dict(path=str(path),**value)
        except (ValueError,KeyError,TypeError) as error:
            summary['errors'].append(f'paired {first}/{second}: {error}');summary['status']='incomplete'
    status=dict(schema='rvloc.all_versions_report_status.v1',status=summary['status'],
        generated_at=summary['generated_at'],uncompleted=summary['uncompleted'],errors=summary['errors'],
        report=str(output/'ALL_MODEL_RESULTS_REPORT_CN.md'),summary=str(output/'all_model_results_summary.json'))
    write(output/'all_model_results_summary.json',summary)
    write(output/'ALL_MODEL_RESULTS_REPORT_CN.md',render(summary))
    write(output/'report_status.json',status)
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-run',required=True);parser.add_argument('--fixed-results',required=True)
    parser.add_argument('--gnn-run',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--fixed-index');parser.add_argument('--train-metric',action='append',default=[])
    parser.add_argument('--watch',action='store_true');parser.add_argument('--poll-seconds',type=float,default=60.)
    parser.add_argument('--bootstrap-draws',type=int,default=2000);parser.add_argument('--bootstrap-seed',type=int,default=20261126)
    args=parser.parse_args();require(args.poll_seconds>0 and math.isfinite(args.poll_seconds),'poll秒数必须有限且为正')
    source=sha(__file__)
    while True:
        require(sha(__file__)==source,'报告源码运行中发生变化；请新建运行记录')
        value=generate(args.main_run,args.fixed_results,args.gnn_run,args.output_dir,
            fixed_index=args.fixed_index,train_metrics=args.train_metric,draws=args.bootstrap_draws,seed=args.bootstrap_seed)
        message=f'[{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}] REPORT status={value["status"]} pending={len(value["uncompleted"])} path={value["report"]}'
        print(message,flush=True)
        with (Path(args.output_dir)/'report.log').open('a') as log:log.write(message+'\n')
        if not args.watch or value['status']=='completed':return 0
        if value['status']=='incomplete':return 2
        time.sleep(args.poll_seconds)


if __name__=='__main__':raise SystemExit(main())
