"""Synthetic contract tests: fixed identities, missing outputs and graph rows."""
from pathlib import Path
import json
import numpy as np
import pytest

from gnn_research import evaluate as tool
from gnn_research import run as runner
from rvloc.runtime import file_sha256,write_json


def test_mixed_empty_frame_diagnostics_use_active_mapping_and_never_node_length():
    # Node count accidentally equals batch size: it still must not be indexed
    # as a frame quantity. Active-frame tensors have only two entries.
    diag=dict(source_batch_indices=np.array([0,2]),edge_count=np.array([12,45]),
        candidate_depths_m=np.arange(12).reshape(2,2,3),graph_variant='shared',
        evidence_candidate_degree=np.array([2,3,1]),physical_psf_proxy={'range_m':4.7})
    first=tool._row_diagnostic(diag,0,3);empty=tool._row_diagnostic(diag,1,3);last=tool._row_diagnostic(diag,2,3)
    assert first['edge_count']==12 and last['edge_count']==45
    assert last['candidate_depths_m'].tolist()==[[6,7,8],[9,10,11]]
    assert empty['edge_count']==0 and empty['candidate_depths_m']==[]
    assert not empty['frame_had_radar_inputs'] and last['frame_had_radar_inputs']
    assert 'evidence_candidate_degree' not in first
    assert first['physical_psf_proxy']=={'range_m':4.7}
    assert not tool._row_diagnostic({'query_rounds':1},0,3)['frame_had_radar_inputs']


def artifacts(tmp_path):
    boxes=[[0,0,1,1],[2,0,3,1]]
    frozen=dict(schema='rvloc.fixed_roi_gt_frame.v1',episode_id='e0',frame_idx=0,
        physical_time_s=0.,normal_sha256='normal',visual_sha256='visual',detection_count=2,
        bbox_xyxy=boxes,gt_member_ids=['a','b','c'],gt_positions_m=[[0,0,0],[10,0,0],[20,0,0]],
        training_equivalent_assignment=[dict(detection_index=0,gt_member_id='a',target_depth_m=0.),
        dict(detection_index=1,gt_member_id='b',target_depth_m=10.)])
    frame=write_json(tmp_path/'frames/f0.json',frozen)
    index=write_json(tmp_path/'index.json',dict(schema='rvloc.fixed_roi_gt_index.v1',supervision_only=True,
        manifest_sha256='manifest',frames=[dict(split='dev',episode_id='e0',frame_idx=0,path='frames/f0.json',sha256=file_sha256(frame))]))
    pred=write_json(tmp_path/'predictions.json',dict(schema='rvloc.predictions.v1',method_id='mce_shared',variant='shared',
        neural_architecture=tool.ARCHITECTURE,manifest_sha256='manifest',normal_inference=True,labels_opened=False,split='dev',
        cost={'device':'cpu'},frames=[dict(episode_id='e0',frame_idx=0,physical_time_s=0.,
        positions=[[10,0,0],[0,0,0]],valid=[True,True],bbox_xyxy=boxes,source_hashes={'normal':'normal','visual':'visual'},
        radar_query={'candidate_depths_m':[[0,1,2],[8,10,12]]},graph_diagnostics={})]))
    return index,pred,frame


def test_fixed_evaluation_penalizes_identity_swap_and_retains_unlocalized_gt(tmp_path):
    index,pred,_=artifacts(tmp_path)
    tool.evaluate(index,pred,tmp_path/'score',draws=8)
    value=json.loads((tmp_path/'score/evaluation.json').read_text())
    assert [r['error_m'] for r in value['rows']]==[10.,10.,None]
    assert value['summary']['position_rmse_3d_m']==10.
    assert value['summary']['gt_members']==3
    assert value['graph_metrics']['offline_candidate_support']['depth_span_coverage']==1.
    assert len(json.loads((tmp_path/'score/failure_cases.json').read_text()))==3


def test_evaluation_refuses_frame_identity_drift_before_scoring(tmp_path):
    index,pred,frame=artifacts(tmp_path)
    value=json.loads(frame.read_text());value['gt_positions_m'][0][0]=99
    write_json(frame,value)
    with pytest.raises(ValueError,match='SHA differs'):tool.evaluate(index,pred,tmp_path/'score',draws=8)


def test_reserved_test_cannot_enter_normal_inference(tmp_path):
    with pytest.raises(ValueError,match='TEST'):tool.infer('unused','unused',tmp_path/'out',split='test')


def test_device_queue_preserves_shared_card_sequential_order():
    assert runner.device_groups(('shared','isolated','shuffled'),('cuda:0','cuda:1','cuda:1'))=={
        'cuda:0':['shared'],'cuda:1':['isolated','shuffled']}
    with pytest.raises(ValueError):runner.device_groups(('shared',),())


def test_sensor_inference_commits_mixed_empty_rows_and_raw_batch_artifact(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import torch
    observations=[]
    for i,n in enumerate((2,0,1)):
        features=dict(ray_directions=np.ones((n,3)),detection_scores=np.ones(n),bbox_xyxy=np.zeros((n,4)),
            ray_origins=np.zeros((n,3)),ray_depth_bounds=np.tile([0.,20.],(n,1)),range_scale_m=20.)
        observations.append(SimpleNamespace(episode_id='e0',frame_idx=i,physical_time_s=float(i),
            source_hashes={'normal':str(i),'visual':str(i)},features=features))
    def public_observations(split,**kw):
        assert split=='dev' and kw==dict(modality='rgb_radar',prepared_only=True)
        return iter(observations)
    manifest=SimpleNamespace(sha256='manifest',observations=public_observations)
    monkeypatch.setattr(tool.ExperimentConfig,'load',lambda path:SimpleNamespace(manifest='manifest.json',sha256='config'))
    monkeypatch.setattr(tool,'load_manifest',lambda path:manifest)
    metadata=dict(manifest_sha256='manifest',training_sources=tool.training_sources(),completed_epochs=80,
        optimizer_updates=8,center_contract='observer_scatter_at_camera_optical_center',
        observer_position_reference='camera_optical_center',body_navigation_is_model_input=False)
    monkeypatch.setattr(tool,'load_graph_checkpoint',lambda *a,**kw:(None,dict(metadata=metadata,variant='shared'),'ckpt'))
    monkeypatch.setattr(tool,'pack_features',lambda features,**kw:features)
    def forward(model,features):
        b=len(features);n=max(len(f['ray_directions']) for f in features);active=[i for i,f in enumerate(features) if len(f['ray_directions'])]
        valid=torch.zeros((b,n),dtype=torch.bool)
        for i,f in enumerate(features):valid[i,:len(f['ray_directions'])]=True
        diag=dict(source_batch_indices=torch.tensor(active),edge_count=torch.arange(len(active))+12,
            candidate_depths_m=torch.zeros((len(active),n,3)),evidence_candidate_degree=torch.tensor([2.,3.,1.]))
        return dict(member_positions=torch.zeros((b,n,3)),depths=torch.ones((b,n)),valid_mask=valid,
            visual_depth_mean_m=torch.ones((b,n)),visual_depth_std_m=torch.ones((b,n)),
            graph_diagnostics=diag,sampling_diagnostics=diag)
    monkeypatch.setattr(tool,'forward',forward)
    tool.infer('config.json','model.pt',tmp_path/'inference',batch_size=3)
    result=json.loads((tmp_path/'inference/predictions.json').read_text())
    assert not result['labels_opened'] and result['normal_inference']
    assert result['frames'][1]['graph_diagnostics']['edge_count']==0
    assert not result['frames'][1]['graph_diagnostics']['frame_had_radar_inputs']
    assert result['frames'][2]['graph_diagnostics']['edge_count']==13
    assert 'evidence_candidate_degree' not in result['frames'][0]['graph_diagnostics']
    assert len({f['runtime_s'] for f in result['frames']})==1
    artifact=result['graph_batch_artifacts'][0]
    assert file_sha256(tmp_path/'inference'/artifact['path'])==artifact['sha256']
    with np.load(tmp_path/'inference'/artifact['path'],allow_pickle=False) as raw:
        assert raw['evidence_candidate_degree'].tolist()==[2.,3.,1.]
    assert result['cost']['serial_profile_frames']==3


def test_runner_resume_rejects_changed_completed_stage(tmp_path):
    config=write_json(tmp_path/'config.json',dict(schema='rvloc.experiment.v1',bandwidth_hz=32e6,
        center_contract='observer_scatter_at_camera_optical_center',model={},training={'epochs':2},
        evaluation={'primary':'position_rmse_3d_m'},manifest=str(write_json(tmp_path/'manifest.json',{})),
        storage=dict(output_root=str(tmp_path/'results'),cache_root=str(tmp_path/'cache'),weights_root=str(tmp_path/'weights'))))
    index=write_json(tmp_path/'index.json',dict(schema='rvloc.fixed_roi_gt_index.v1',supervision_only=True))
    def process(command,**kw):
        out=Path(command[command.index('--output-dir')+1]);out.mkdir(parents=True)
        checkpoint=tmp_path/'weights/final.pt';checkpoint.parent.mkdir(exist_ok=True);checkpoint.write_text('synthetic checkpoint')
        write_json(out/'receipt.json',dict(status='trained',variant='shared',epochs=2,checkpoint=str(checkpoint)))
        write_json(out/'history.json',[]);write_json(out/'identity.json',{'synthetic':True});write_json(out/'progress.json',{'status':'completed'})
        return 0
    job=runner.GraphRunner(config,index,tmp_path/'results/run',variants=('shared',),process_runner=process)
    record=job.stage('shared','train','cpu')
    assert record['status']=='completed'
    job.stage('shared','train','cpu')
    canonical=tmp_path/'results/run/shared/train/receipt.json'
    receipt=json.loads(canonical.read_text());receipt['epochs']=99;write_json(canonical,receipt)
    with pytest.raises(ValueError,match='Canonical'):job.stage('shared','train','cpu')


def test_parallel_arms_start_independent_workers_on_shared_device(tmp_path,monkeypatch):
    import threading
    config=write_json(tmp_path/'parallel_config.json',dict(schema='rvloc.experiment.v1',bandwidth_hz=32e6,
        center_contract='observer_scatter_at_camera_optical_center',model={},training={'epochs':2},
        evaluation={'primary':'position_rmse_3d_m'},manifest=str(write_json(tmp_path/'manifest.json',{})),
        storage=dict(output_root=str(tmp_path/'results'),cache_root=str(tmp_path/'cache'),weights_root=str(tmp_path/'weights'))))
    index=write_json(tmp_path/'parallel_index.json',dict(schema='rvloc.fixed_roi_gt_index.v1',supervision_only=True))
    job=runner.GraphRunner(config,index,tmp_path/'results/run_parallel',devices=('cuda:0','cuda:1','cuda:1'),parallel_arms=True)
    barrier=threading.Barrier(3);observed={};threads=set()
    def arm(variant,device):
        with job.lock:observed[variant]=device;threads.add(threading.get_ident())
        barrier.wait(timeout=5)
        with job.lock:job.states[variant]={'status':'evaluated','device':device}
    monkeypatch.setattr(job,'arm',arm);monkeypatch.setattr(job,'comparisons',lambda:{})
    assert job.run()['status']=='completed'
    assert observed=={'shared':'cuda:0','isolated':'cuda:1','shuffled':'cuda:1'}
    assert len(threads)==3
    identity=json.loads((tmp_path/'results/run_parallel/execution_identity.json').read_text())
    assert identity['execution_mode']=='parallel_arms'
    assert identity['variant_devices']==observed
    assert identity['planned_max_concurrent_arms_by_device']=={'cuda:0':1,'cuda:1':2}
