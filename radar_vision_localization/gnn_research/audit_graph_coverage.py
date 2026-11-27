"""Recover shared-edge coverage from authenticated, existing graph arrays.

This postprocessing audit opens no GT labels or model weights, performs no
inference/training, and never rewrites original evaluation JSON artifacts.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

VARIANTS=('isolated','shared','shuffled')


class AuditError(ValueError):pass


def require(condition,message):
    if not condition:raise AuditError(message)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix='.coverage-audit-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    return path


def integers(value,name):
    values=np.asarray(value)
    require(values.dtype.kind in 'fiu' and np.isfinite(values).all(),name+' must contain finite numeric counts')
    require((values>=0).all() and np.equal(values,np.rint(values)).all(),name+' must contain nonnegative integer counts')
    require(not values.size or float(values.max())<2**53,name+' exceeds exact float64 integer range')
    return values.astype(np.int64)


def recover_counts(arrays):
    for key in ('evidence_candidate_degree','evidence_distinct_member_degree','edge_count'):
        require(key in arrays,'Missing raw count array: '+key)
    degree=integers(arrays['evidence_candidate_degree'],'evidence_candidate_degree')
    owners=integers(arrays['evidence_distinct_member_degree'],'evidence_distinct_member_degree')
    require(degree.ndim==owners.ndim==1 and degree.shape==owners.shape,'Evidence count arrays must align')
    require((owners<=degree).all() and ((degree==0)|(owners>=1)).all(),'Distinct member count incompatible with evidence degree')
    total=int(degree.sum());declared=int(integers(arrays['edge_count'],'edge_count').sum())
    require(total==declared,'Sum of evidence in-degrees differs from recorded edge_count')
    if 'candidate_degree' in arrays:
        require(int(integers(arrays['candidate_degree'],'candidate_degree').sum())==total,'Candidate/evidence edge totals disagree')
    if 'unique_evidence_nodes' in arrays:
        require(int(integers(arrays['unique_evidence_nodes'],'unique_evidence_nodes').sum())==len(degree),'Evidence-node total differs from raw array length')
    shared=int(degree[owners>=2].sum())
    return dict(edge_count=total,shared_edge_count=shared,evidence_nodes=len(degree),
        shared_evidence_nodes=int(np.count_nonzero(owners>=2)),
        shared_edge_coverage=shared/total if total else None)


def certified(path,stage):
    path=Path(path).resolve();expected=stage.get('artifacts',{}).get(str(path))
    require(expected is not None,'Artifact is not present in completed stage receipt: '+str(path))
    actual=sha256(path);require(actual==expected,'Completed artifact SHA differs: '+str(path))
    return actual


def audit_variant(run,variant):
    root=Path(run).resolve()/variant
    inf_stage_path=root/'stage_receipts/inference.json';inf_stage=read(inf_stage_path)
    require(inf_stage.get('status')=='completed' and inf_stage.get('variant')==variant and inf_stage.get('kind')=='inference','Inference stage is not certified complete')
    pred_path=(root/'inference/predictions.json').resolve();pred_sha=certified(pred_path,inf_stage)
    predictions=read(pred_path)
    require(predictions.get('normal_inference') is True and predictions.get('labels_opened') is False,'Sensor-only predictions required')
    require(predictions.get('split')=='dev' and predictions.get('variant')==variant,'Expected declared DEV variant only')
    ev_stage_path=root/'stage_receipts/evaluation.json';ev_stage=read(ev_stage_path)
    require(ev_stage.get('status')=='completed' and ev_stage.get('variant')==variant and ev_stage.get('kind')=='evaluation','Evaluation stage is not certified complete')
    metrics_path=(root/'evaluation/graph_metrics.json').resolve();metrics_sha=certified(metrics_path,ev_stage);metrics=read(metrics_path)
    evaluation_path=(root/'evaluation/evaluation.json').resolve();evaluation_sha=certified(evaluation_path,ev_stage)
    # evaluation.json is hashed only; no GT rows are opened by this audit.
    frames={(row['episode_id'],row['frame_idx']):row for row in predictions['frames']}
    require(len(frames)==len(predictions['frames']),'Duplicate prediction frames')
    artifacts=predictions.get('graph_batch_artifacts',[]);require(bool(artifacts),'No raw graph-batch artifacts recorded')
    checked_frames=set();checked_paths=set();batches=[]
    for artifact in artifacts:
        relative=Path(artifact['path']);path=(pred_path.parent/relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(pred_path.parent),'Raw graph batch escaped its prediction directory')
        require(path not in checked_paths,'Duplicate raw batch path');checked_paths.add(path)
        require(sha256(path)==artifact['sha256'],'Raw graph batch SHA differs: '+str(path))
        with np.load(path,allow_pickle=False) as arrays:
            counts=recover_counts(arrays)
            require('source_batch_indices' in arrays,'Missing active-frame mapping')
            source=integers(arrays['source_batch_indices'],'source_batch_indices').reshape(-1)
            require(len(source)==len(np.unique(source)),'Duplicate active-frame rows')
            require(source.tolist()==artifact['source_batch_indices'],'Active-frame mapping differs from committed predictions')
            require(np.asarray(arrays['edge_count']).shape==(len(source),),'Per-active-frame edge shape differs')
        keys=[(row['episode_id'],row['frame_idx']) for row in artifact['frames']]
        require(not checked_frames.intersection(keys) and len(keys)==len(set(keys)),'Duplicate raw-batch frame membership')
        require(all(key in frames for key in keys),'Raw batch names an unknown prediction frame')
        require(all(int(i)<len(keys) for i in source),'Active-frame row exceeds batch frame population')
        per_frame_edges=0
        for i,key in enumerate(keys):
            frame=frames[key]
            require(frame.get('graph_batch_artifact')==str(relative) and frame.get('graph_batch_source_row')==i,'Per-frame raw batch link differs')
            per_frame_edges+=int(frame.get('graph_diagnostics',{}).get('edge_count',0))
        require(per_frame_edges==counts['edge_count'],'Per-frame and raw-batch edge totals differ')
        checked_frames.update(keys);batches.append(dict(path=str(path),sha256=artifact['sha256'],frames=len(keys),**counts))
    require(checked_frames==frames.keys(),'Not every predicted frame has authenticated raw arrays')
    total=sum(batch['edge_count'] for batch in batches);shared=sum(batch['shared_edge_count'] for batch in batches)
    recorded=metrics.get('raw_numeric_sums',{}).get('edge_count')
    require(recorded is not None and float(recorded)==total,'Reconstructed total differs from original graph metrics')
    require(metrics.get('frames')==len(frames),'Original metrics frame population differs')
    has_numerator='shared_edge_count' in metrics.get('raw_numeric_sums',{})
    return dict(status='reconstructed',variant=variant,frames=len(frames),authenticated_batches=len(batches),
        edge_count=total,shared_edge_count=shared,shared_edge_coverage=shared/total if total else None,
        original_exported_shared_edge_coverage=metrics.get('shared_edge_coverage'),
        original_shared_edge_numerator_was_recorded=has_numerator,
        original_shared_member_coverage=metrics.get('shared_member_coverage'),
        original_shared_candidate_coverage=metrics.get('shared_candidate_coverage'),
        evidence=dict(predictions_path=str(pred_path),predictions_sha256=pred_sha,
            inference_stage_path=str(inf_stage_path),inference_stage_sha256=sha256(inf_stage_path),
            graph_metrics_path=str(metrics_path),graph_metrics_sha256=metrics_sha,
            evaluation_path=str(evaluation_path),evaluation_sha256=evaluation_sha,
            evaluation_stage_path=str(ev_stage_path),evaluation_stage_sha256=sha256(ev_stage_path)),batches=batches)


def chinese(audit):
    lines=['# GNN 共享边覆盖率独立审计与勘误','',
        '本审计仅复核现存正常推理输出及其 SHA 认证的图数组，不读取 TEST 或新增 GT，不训练、不运行模型、不改原评价 JSON、模型权重或默认入口。','',
        '原始 `shared_edge_coverage: 0.0` 中，导出逻辑在缺少 `shared_edge_count` 时使用默认零分子。这不是共享边为零的观测结果。成员共享率与边共享率分母不同。','',
        '令证据节点 E 的入边数为 d_E、不同成员数为 m_E，按原图实现定义重建：`shared_edge_coverage = sum(d_E * [m_E >= 2]) / sum(d_E)`。同一成员的多个候选不构成成员间共享；每条候选→证据边按次数计数，未按证据节点数量平均。','',
        '| 分支 | 状态 | 验证批次 / 帧 | 总边数 | 共享边数 | 真实共享边比例 | 原导出值 |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for variant,value in audit['variants'].items():
        ratio=value.get('shared_edge_coverage');percent=f'{ratio*100:.6f}%' if ratio is not None else '未测（null）'
        lines.append(f"| {variant} | {value['status']} | {value.get('authenticated_batches','—')} / {value.get('frames','—')} | {value.get('edge_count','—')} | {value.get('shared_edge_count','—')} | {percent} | {value.get('original_exported_shared_edge_coverage','—')} |")
        if value.get('error'):lines.extend(['',f"{variant} 未能完整重建：{value['error']}。未验证部分不计为零或不可共享。"])
    lines.extend(['','逐批核对了：已完成推理收据中的 predictions SHA、每个 NPZ 的 SHA、活动帧映射、候选与证据两侧度数总和、原始 edge_count、证据节点数量、逐帧边数和全体帧覆盖。','',
        '本次只修正非定位诊断指标，不改变任何逐成员定位误差、RMSE、P95、P99、严重错误数或配对区间。共享比例较高只说明构造的图存在较多共享连接，不能证明这些连接具有正确目标归属，也不能证明共享消息有益。','',
        '原评价 JSON 保留不变；以本审计为共享边比例的更正来源。原主报告在追加勘误前保存逐字节 SHA 快照，补丁收据记录前后 SHA。',''])
    return '\n'.join(lines)


def patch_report(report_path,output,audit,audit_path):
    report_path=Path(report_path).resolve();output=Path(output)
    original=report_path.read_bytes();original_sha=hashlib.sha256(original).hexdigest()
    snapshot=output/'ALL_MODEL_RESULTS_REPORT_CN.before_graph_coverage_erratum.md'
    require(not snapshot.exists(),'Report snapshot already exists; choose a fresh audit directory')
    snapshot.write_bytes(original)
    require(sha256(snapshot)==original_sha,'Report snapshot does not match original bytes')
    rows=[]
    for variant,value in audit['variants'].items():
        ratio=value.get('shared_edge_coverage')
        rows.append(f"| {variant} | {value.get('shared_edge_count','未测')} / {value.get('edge_count','未测')} | {ratio*100:.6f}% |" if ratio is not None else f'| {variant} | 未能完整重建 | 未测（null） |')
    addition='\n\n## 勘误：共享边覆盖率（2026-11-27）\n\n前文原始 JSON 摘录中的 `shared_edge_coverage: 0.0` 为缺失分子 `shared_edge_count` 时的导出默认值，不能解释为实测零共享。原始评价 JSON 保留原样。独立审计根据所有 SHA 认证图批次中的 `evidence_candidate_degree` 与 `evidence_distinct_member_degree` 重建：\n\n`共享边比例 = sum(degree_E[distinct_members_E >= 2]) / sum(degree_E)`\n\n| 分支 | 共享边数 / 总边数 | 经审计的共享边比例 |\n| --- | ---: | ---: |\n'+'\n'.join(rows)+'\n\n本审计同时核对了逐批与逐帧边总数、图节点数和完整帧覆盖。该更正不改变定位结果及模型优劣结论；高共享覆盖不代表证据归属或消息传播正确。详细证据见 ['+audit_path.name+']('+str(audit_path.relative_to(report_path.parent))+') 与 [中文审计说明]('+str((output/'GRAPH_COVERAGE_AUDIT_CN.md').relative_to(report_path.parent))+')。\n'
    require(report_path.read_bytes()==original,'Main report changed concurrently; saved snapshot retained and patch aborted')
    replacement=original+addition.encode('utf-8')
    descriptor,temp=tempfile.mkstemp(prefix='.report-coverage-erratum-',dir=report_path.parent)
    try:
        with os.fdopen(descriptor,'wb') as stream:stream.write(replacement)
        os.replace(temp,report_path)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    return dict(report_path=str(report_path),report_sha256_before=original_sha,report_sha256_after=sha256(report_path),
        original_snapshot_path=str(snapshot),original_snapshot_sha256=sha256(snapshot),change='append-only Chinese graph coverage erratum; original body retained byte-for-byte')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gnn-run',required=True);parser.add_argument('--output-dir',required=True);parser.add_argument('--report')
    args=parser.parse_args(argv);run=Path(args.gnn_run).resolve();output=Path(args.output_dir).resolve()
    require(not output.exists(),'Independent audit output must be new');output.mkdir(parents=True)
    with (output/'.audit.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        audit=dict(schema='rvloc.mce_graph_coverage_audit.v1',created_at_utc=datetime.now(timezone.utc).isoformat(),
            gnn_run=str(run),script_sha256=sha256(__file__),normal_predictions_only=True,
            GT_opened=False,TEST_opened=False,new_inference=False,new_training=False,
            definition='sum(evidence_candidate_degree[evidence_distinct_member_degree >= 2]) / sum(evidence_candidate_degree)',variants={})
        for variant in VARIANTS:
            try:value=audit_variant(run,variant)
            except (AuditError,KeyError,TypeError,OSError,ValueError) as error:
                value=dict(status='not_reconstructed',variant=variant,shared_edge_coverage=None,error=f'{type(error).__name__}: {error}')
            audit['variants'][variant]=value
        audit['status']='complete' if all(v['status']=='reconstructed' for v in audit['variants'].values()) else 'incomplete'
        path=write_json(output/'graph_coverage_audit.json',audit)
        (output/'GRAPH_COVERAGE_AUDIT_CN.md').write_text(chinese(audit))
        patch=patch_report(args.report,output,audit,path) if args.report else None
        # Prove original evaluation and graph-metric files still have their
        # certified SHA after audit/report mutation, without parsing GT rows.
        for value in audit['variants'].values():
            if value['status']=='reconstructed':
                evidence=value['evidence']
                for kind in ('predictions','graph_metrics','evaluation'):
                    require(sha256(evidence[kind+'_path'])==evidence[kind+'_sha256'],'Original artifact changed during audit: '+kind)
        receipt=dict(schema='rvloc.mce_graph_coverage_patch.v1',status=audit['status'],audit_path=str(path),
            audit_sha256=sha256(path),chinese_report_path=str(output/'GRAPH_COVERAGE_AUDIT_CN.md'),
            chinese_report_sha256=sha256(output/'GRAPH_COVERAGE_AUDIT_CN.md'),script_path=str(Path(__file__).resolve()),script_sha256=sha256(__file__),
            command=['python',str(Path(__file__).resolve()),'--gnn-run',str(run),'--output-dir',str(output)]+(['--report',str(Path(args.report).resolve())] if args.report else []),
            original_evaluation_json_unchanged=True,localization_metrics_unchanged=True,report_patch=patch)
        write_json(output/'patch_receipt.json',receipt)
        print(json.dumps(dict(status=audit['status'],audit=str(path),report_patch=patch,variants={k:{field:v.get(field) for field in ('status','edge_count','shared_edge_count','shared_edge_coverage')} for k,v in audit['variants'].items()}),ensure_ascii=False,indent=2))
    return 0 if audit['status']=='complete' else 2

if __name__=='__main__':raise SystemExit(main())
