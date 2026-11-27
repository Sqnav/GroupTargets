"""Generated experiment artifacts only; no real data, weights, or TEST read."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'report_all.py'
spec=importlib.util.spec_from_file_location('report_all_test',path);tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def write(p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p


def fixture(tmp):
    main=tmp/'main';fixed=tmp/'fixed';gnn=tmp/'gnn';out=tmp/'report'
    write(main/'execution_identity.json',dict(sha256='identity',manifest_sha256='manifest'))
    for p in (main/'status.json',fixed/'watch_status.json',gnn/'status.json'):write(p,dict(status='running'))
    return main,fixed,gnn,out


def evaluation(method,scope='fixed_roi',errors=(1.,2.),manifest='manifest'):
    rows=[dict(episode_id=f'ep{i}',frame_idx=0,gt_member_id='member',error_m=e) for i,e in enumerate(errors)]
    valid=[e for e in errors if e is not None]
    summary=dict(gt_members=len(rows),localized_members=len(valid),no_localization_members=len(rows)-len(valid),
        position_rmse_3d_m=float(np.sqrt(np.mean(np.square(valid)))) if valid else None,
        mae_m=float(np.mean(valid)) if valid else None,p95_m=float(np.quantile(valid,.95)) if valid else None,
        p99_m=float(np.quantile(valid,.99)) if valid else None,over_5m=0,over_10m=0,
        all_gt_success={'1m':sum(e<=1 for e in valid)/len(rows),'3m':sum(e<=3 for e in valid)/len(rows),'5m':sum(e<=5 for e in valid)/len(rows)})
    return dict(schema=tool.FIXED_SCHEMA if scope=='fixed_roi' else tool.SET_SCHEMA,
        split='dev',method_id=method,manifest_sha256=manifest,frozen_index_sha256='index',summary=summary,rows=rows)


def main_stage(main,name,filename,value):
    p=write(main/'stages'/name/'results'/filename,value)
    receipt=write(main/'stage_receipts'/(name+'.json'),dict(status='completed',identity_sha256='identity',artifacts={str(p):tool.sha(p)},stage={'method':value.get('method_id')}))
    return receipt,p


def base_results(main,fixed):
    for i in range(1,8):
        main_stage(main,f'dev_evaluate_g{i}','evaluation.json',evaluation(i,'set'))
        if i in (2,3,5,6,7):
            stage,_=main_stage(main,f'dev_infer_g{i}','predictions.json',{'method_id':i})
            p=write(fixed/f'g{i}'/'attempt_0001/evaluation.json',evaluation(i))
            write(fixed/f'g{i}'/'receipt.json',dict(status='completed',artifacts={str(p):tool.sha(p)},
                stage_receipt=str(stage),stage_receipt_sha256=tool.sha(stage)))
        if i in (2,3,7):
            main_stage(main,f'train_g{i}','receipt.json',dict(status='trained',method_id=i,epochs=80,
                elapsed_s=10,history=[dict(online_fixed_train_rmse_m=.5)],
                training_identity=dict(manifest_sha256='manifest')))


def graph_results(gnn):
    for variant in tool.GRAPH_NAMES:
        folder=gnn/variant
        write(folder/'train/receipt.json',dict(status='trained',epochs=80,parameter_count=42,
            training_identity=dict(manifest_sha256='manifest'),history=[dict(online_fixed_train_rmse_m=.4)]))
        write(folder/'evaluation/evaluation.json',evaluation('mce_'+variant,errors=(.8,1.5)))
        write(folder/'evaluation/graph_metrics.json',dict(shared_edge_coverage=.4,definition='synthetic only'))
        graph_receipts(gnn,variant)


def graph_receipts(gnn,variant):
    folder=gnn/variant
    for kind,filename in (('train','receipt.json'),('evaluation','evaluation.json')):
        p=folder/kind/filename
        write(folder/'stage_receipts'/(kind+'.json'),dict(status='completed',variant=variant,kind=kind,artifacts={str(p):tool.sha(p)}))


def test_missing_results_are_waiting_not_zero_or_completed(tmp_path):
    main,fixed,gnn,out=fixture(tmp_path)
    result=tool.generate(main,fixed,gnn,out,draws=20)
    assert result['status']=='waiting' and len(result['uncompleted'])==10
    text=(out/'ALL_MODEL_RESULTS_REPORT_CN.md').read_text()
    assert '阶段报告' in text and '尚未完成' in text
    summary=tool.read(out/'all_model_results_summary.json')
    assert summary['versions']['G3']['fixed_evaluation'] is None


def test_failed_controller_preserves_incomplete_report(tmp_path):
    main,fixed,gnn,out=fixture(tmp_path);write(gnn/'status.json',dict(status='failed'))
    result=tool.generate(main,fixed,gnn,out,draws=20)
    assert result['status']=='incomplete'
    assert '阶段报告' in (out/'ALL_MODEL_RESULTS_REPORT_CN.md').read_text()


def test_complete_report_requires_all_scopes_and_controllers(tmp_path):
    main,fixed,gnn,out=fixture(tmp_path);base_results(main,fixed);graph_results(gnn)
    for p in (main/'status.json',fixed/'watch_status.json',gnn/'status.json'):write(p,dict(status='completed'))
    result=tool.generate(main,fixed,gnn,out,draws=20)
    assert result['status']=='completed' and not result['uncompleted']
    text=(out/'ALL_MODEL_RESULTS_REPORT_CN.md').read_text()
    assert '全部声明实验已完成' in text and '七组共同点集 DEV' in text and '固定 ROI 成员 DEV' in text
    summary=tool.read(out/'all_model_results_summary.json')
    assert len(summary['pairwise'])==3 and summary['versions']['GNN_shared']['training']['receipt']['parameter_count']==42
    # Unchanged inputs reuse the same paired results without changing claims.
    assert tool.generate(main,fixed,gnn,out,draws=20)['status']=='completed'


def test_data_identity_mismatch_invalidates_graph_result(tmp_path):
    main,fixed,gnn,out=fixture(tmp_path);base_results(main,fixed);graph_results(gnn)
    p=gnn/'shared/evaluation/evaluation.json';v=tool.read(p);v['manifest_sha256']='other';write(p,v);graph_receipts(gnn,'shared')
    result=tool.generate(main,fixed,gnn,out,draws=20)
    assert result['status']=='incomplete'
    summary=tool.read(out/'all_model_results_summary.json')
    assert summary['versions']['GNN_shared']['status']=='invalid'
    assert any('数据身份' in e for e in summary['errors'])


def test_population_mismatch_cannot_be_silently_paired(tmp_path):
    a=evaluation(3);b=evaluation('mce_shared');b['rows'][0]['gt_member_id']='different'
    with pytest.raises(ValueError,match='GT 人群'):tool.paired_fixed(a,b,draws=20)


def test_set_matching_and_train_cannot_be_called_fixed_dev():
    with pytest.raises(ValueError,match='口径'):tool.validate_evaluation(evaluation(1,'set'),'fixed_roi','manifest')
    value=evaluation(3);value['split']='train'
    with pytest.raises(ValueError,match='TRAIN'):tool.validate_evaluation(value,'fixed_roi','manifest')


def test_wrong_rmse_or_cross_run_receipt_rejected(tmp_path):
    value=evaluation(3);value['summary']['position_rmse_3d_m']=0.
    with pytest.raises(ValueError,match='复算'):tool.validate_evaluation(value,'fixed_roi','manifest')
    main,fixed,gnn,out=fixture(tmp_path);base_results(main,fixed)
    p=main/'stage_receipts/dev_evaluate_g2.json';v=tool.read(p);v['identity_sha256']='wrong';write(p,v)
    result=tool.generate(main,fixed,gnn,out,draws=20)
    assert result['status']=='incomplete'
    assert any('指定主实验' in e for e in tool.read(out/'all_model_results_summary.json')['errors'])


def test_hard_subsets_are_frozen_once_and_missing_sse_is_not_zero(tmp_path):
    main,fixed,gnn,out=fixture(tmp_path);base_results(main,fixed);graph_results(gnn)
    p=gnn/'shared/evaluation/evaluation.json'
    write(p,evaluation('mce_shared',errors=(.5,None)));graph_receipts(gnn,'shared')
    tool.generate(main,fixed,gnn,out,draws=20)
    summary=tool.read(out/'all_model_results_summary.json')
    hard=summary['difficult_subsets']
    assert hard['definition']['subsets']['1']==[['ep1',0,'member']]
    assert hard['versions']['GNN_shared']['1']['missing_members']==1
    assert hard['versions']['GNN_shared']['1']['observed_sse_m2'] is None
    assert hard['versions']['GNN_shared']['1']['complete_sse_m2'] is None
    # The same chosen GT keys cannot be silently re-selected after baseline changes.
    v=summary['versions']['G3']['fixed_evaluation'];v['sha256']='changed'
    with pytest.raises(ValueError,match='不得重新挑选'):
        tool.difficult_subsets(summary,{('G3','fixed_roi'):evaluation(3)},out)


def test_graph_evaluation_requires_committed_receipt_and_unchanged_sha(tmp_path):
    main,fixed,gnn,out=fixture(tmp_path);base_results(main,fixed);graph_results(gnn)
    receipt=gnn/'shared/stage_receipts/evaluation.json'
    receipt.unlink()
    result=tool.generate(main,fixed,gnn,out,draws=20)
    assert result['status']=='waiting' and 'GNN_shared' in result['uncompleted']
    graph_receipts(gnn,'shared')
    p=gnn/'shared/evaluation/evaluation.json';value=tool.read(p)
    value['rows'][0]['error_m']=99.;write(p,value)
    result=tool.generate(main,fixed,gnn,out,draws=20)
    assert result['status']=='incomplete'
    assert any('SHA' in e for e in tool.read(out/'all_model_results_summary.json')['errors'])
