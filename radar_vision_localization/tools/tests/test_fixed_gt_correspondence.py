"""Synthetic labels/predictions only; normal training code is never changed."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

script = Path(__file__).resolve().parents[1]/'fixed_gt_correspondence.py'
spec = importlib.util.spec_from_file_location('fixed_gt_tool',script)
tool = importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def fixture():
    frame = dict(schema=tool.FRAME_SCHEMA,episode_id='ep',frame_idx=0,physical_time_s=0.,
        normal_sha256='a'*64,visual_sha256='b'*64,labels_sha256='c'*64,detection_count=2,
        bbox_xyxy=[[10,10,20,20],[30,10,40,20]],
        gt_member_ids=['one','two'],gt_positions_m=[[0,0,0],[10,0,0]],
        training_equivalent_assignment=[dict(detection_index=0,gt_member_id='one'),
                                        dict(detection_index=1,gt_member_id='two')])
    prediction = dict(episode_id='ep',frame_idx=0,physical_time_s=0.,
        source_hashes=dict(normal='a'*64,visual='b'*64),
        bbox_xyxy=copy.deepcopy(frame['bbox_xyxy']),positions=[[0,0,0],[10,0,0]],valid=[True,True])
    return frame,prediction


def test_fixed_identity_swap_is_penalized_instead_of_rematched():
    frame,pred=fixture()
    rows,_=tool.score_fixed_frame(pred,frame)
    assert tool.metrics(rows)['position_rmse_3d_m']==0
    pred['positions'].reverse()
    rows,_=tool.score_fixed_frame(pred,frame)
    assert tool.metrics(rows)['position_rmse_3d_m']==10


def test_invalid_prediction_retains_missing_gt_in_denominator():
    frame,pred=fixture();pred['valid'][1]=False;pred['positions'][1]=[None,None,None]
    rows,diag=tool.score_fixed_frame(pred,frame)
    result=tool.metrics(rows)
    assert len(rows)==2 and rows[1]['error_m'] is None
    assert result['no_localization_members']==1
    assert result['all_gt_success']['1m']==.5
    assert diag['assigned_invalid_outputs']==1


@pytest.mark.parametrize('mutation',['normal','visual','bbox','count'])
def test_changed_sensor_identity_or_order_is_rejected(mutation):
    frame,pred=fixture()
    if mutation in ('normal','visual'):pred['source_hashes'][mutation]='d'*64
    elif mutation=='bbox':pred['bbox_xyxy'].reverse()
    else:pred['positions'].append([1,0,0])
    with pytest.raises(ValueError):tool.score_fixed_frame(pred,frame)


def test_gt_box_audit_keeps_duplicates_and_no_overlap_unmatched():
    detected=np.array([[0,0,10,10],[1,1,9,9],[20,0,30,10],[100,100,110,110]],float)
    audited=tool.bbox_assignment_audit(detected,['one','two'],[[0,0,10,10],[20,0,10,10]])
    assert [(r['detection_index'],r['gt_member_id']) for r in audited['assignments']]==[(0,'one'),(2,'two')]
    assert audited['multiple_detection_overlap_gt']==['one']
    assert audited['unmatched_detection_indices']==[1,3]
    assert audited['not_an_identity_oracle'] is True


def test_prepare_and_evaluate_roundtrip_preserves_source_and_missing_members(tmp_path,monkeypatch):
    features=dict(ray_origins=np.array([[1.,0,0],[1,0,0]]),
        ray_directions=np.array([[1.,0,0],[1,1,0]])/np.array([[1.],[np.sqrt(2)]]),
        ray_valid=np.array([True,True]),ray_depth_bounds=np.array([[.1,100],[.1,100]]),
        range_scale_m=100.,bbox_xyxy=np.array([[10.,10,20,20],[30,10,40,20]]))
    obs=SimpleNamespace(episode_id='ep',frame_idx=0,physical_time_s=0.,features=features,
                        source_hashes=dict(normal='a'*64,visual='b'*64))
    labels=SimpleNamespace(member_ids=('one','two'),positions=np.array([[11.,0,0],[11,10,0]]))
    record=SimpleNamespace(frame_idx=0,normal_sha256='a'*64,labels_sha256='c'*64)
    manifest=SimpleNamespace(path=tmp_path/'unused_manifest.json',sha256='e'*64,
        episodes={'ep':(record,)},observations=lambda split,modality:iter([obs] if split=='train' else []))
    config=SimpleNamespace(path=tmp_path/'unused_config.json',manifest=manifest.path,
                            sha256='f'*64,raw={'model':{}})
    monkeypatch.setattr(tool.ExperimentConfig,'load',lambda p:config)
    monkeypatch.setattr(tool,'load_manifest',lambda p:manifest)
    monkeypatch.setattr(tool,'load_labels',lambda m,o:labels)
    capture=dict(group=dict(ids=['one','two'],bboxes_xywh=[[10,10,10,10],[30,10,10,10]]))
    proofs={0:dict(normal_sha256='a'*64,labels_sha256='c'*64,visual_sha256='b'*64)}
    monkeypatch.setattr(tool,'_capture_episode',lambda m,e:({0:capture},proofs,{'synthetic':True}))
    result=tool.prepare(config.path,tmp_path/'annotation')
    index=json.loads(Path(result['index_path']).read_text())
    assert result['summaries']['train']['frozen_pairs']==2
    assert index['supervision_only'] is True and index['normal_inference_input'] is False
    prediction=dict(schema='rvloc.predictions.v1',method_id=3,normal_inference=True,labels_opened=False,
        split='train',manifest_sha256=manifest.sha256,frames=[dict(episode_id='ep',frame_idx=0,physical_time_s=0.,
        source_hashes=dict(normal='a'*64,visual='b'*64),bbox_xyxy=features['bbox_xyxy'].tolist(),
        positions=labels.positions.tolist(),valid=[True,False])])
    path=tmp_path/'predictions.json';path.write_text(json.dumps(prediction))
    scored=tool.evaluate(result['index_path'],path,tmp_path/'evaluation')
    assert scored['summary']['position_rmse_3d_m']==0
    assert scored['summary']['no_localization_members']==1
    assert scored['summary']['all_gt_success']['1m']==.5
    assert scored['summary']['uncertainty']['intervals'] is None
    with pytest.raises(FileExistsError):tool.evaluate(result['index_path'],path,tmp_path/'evaluation')
