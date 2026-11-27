"""Original world recurrence is the numerical/gradient oracle; synthetic only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

path=Path(__file__).resolve().parents[1]/'accelerated_world_training.py'
spec=importlib.util.spec_from_file_location('accelerated_world_training_test',path)
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
sys.path.insert(0,str(path.parents[1]/'tests'))
from test_models import synthetic_features
from rvloc.models import build_model,localization_loss,load_checkpoint


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


def sequence():
    ids=[[101,None,303],[303,101],[101,303,None],[101]]
    frames=[synthetic_features(n=len(value),frame=i,episode='synthetic',ids=value) for i,value in enumerate(ids)]
    return frames,[dict(episode_id='synthetic',frame_index=i,timestamp_s=float(i),member_ids=value,
                       ray_valid=tuple(bool(v) for v in frames[i]['ray_valid'])) for i,value in enumerate(ids)]


def loss(output):
    depth=torch.full_like(output['depths'],20.)
    n=len(depth);truth=torch.tensor([25.,2.,3.],dtype=depth.dtype).expand(n,3)
    return localization_loss(output,dict(depths=depth,member_positions=truth,valid_mask=output['valid_mask']))[0]


def test_four_frame_bptt_outputs_state_and_parameter_gradients_match_original():
    torch.manual_seed(847);original=build_model(7).double().train()
    torch.nn.init.normal_(original.memory_bridge.weight,std=.03)
    fast=copy.deepcopy(original)
    frames,ids=sequence();packed=tool.engine.pack_features(frames,dtype=torch.float64)
    outputs,memory=tool.forward_sequence(fast,packed,ids)
    reference=[];state=None
    for f in frames:
        value=original(f,state);state=value['state'];reference.append(value)
    for actual,expected in zip(outputs,reference):
        for key in ('depths','visual_depth_mean_m','visual_depth_std_m','member_positions','valid_mask','center'):
            torch.testing.assert_close(actual[key],expected[key],rtol=1e-9,atol=1e-9)
        for key in ('prior_latent','posterior_latent','prior_positions','rollout_positions','memory_injection','prior_available'):
            torch.testing.assert_close(actual['aux'][key],expected['aux'][key],rtol=1e-9,atol=1e-9)
        assert actual['state'].episode_id==expected['state'].episode_id
        assert actual['state'].members.keys()==expected['state'].members.keys()
        for member in actual['state'].members:
            torch.testing.assert_close(actual['state'].members[member].latent,expected['state'].members[member].latent,rtol=1e-9,atol=1e-9)
    torch.stack([loss(o) for o in outputs]).mean().backward()
    torch.stack([loss(o) for o in reference]).mean().backward()
    for (name,a),(other,b) in zip(original.named_parameters(),fast.named_parameters()):
        assert name==other and (a.grad is None)==(b.grad is None),name
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=2e-8,atol=2e-8,msg=name)
    assert None not in memory.members


def test_episode_reset_and_detach_are_kept():
    torch.manual_seed(848);model=build_model(7).double().train();frames,ids=sequence()
    first,memory=tool.forward_sequence(model,tool.engine.pack_features(frames[:2],dtype=torch.float64),ids[:2])
    memory=memory.detached()
    assert all(value.latent.grad_fn is None for value in memory.members.values())
    fresh=synthetic_features(n=2,frame=0,episode='another',ids=[101,None])
    meta=dict(episode_id='another',frame_index=0,timestamp_s=0.,member_ids=[101,None],ray_valid=(True,True))
    actual,new=tool.forward_sequence(model,tool.engine.pack_features([fresh],dtype=torch.float64),[meta],memory)
    expected=model(fresh,None)
    torch.testing.assert_close(actual[0]['depths'],expected['depths'],rtol=1e-9,atol=1e-9)
    assert new.episode_id=='another' and list(new.members)==[101]
    assert not actual[0]['aux']['prior_available'].any()


def test_no_future_frame_can_change_an_earlier_prediction():
    torch.manual_seed(849);model=build_model(7).double().train();frames,ids=sequence()
    original,_=tool.forward_sequence(model,tool.engine.pack_features(frames,dtype=torch.float64),ids)
    frames[-1]['detector_roi_features']*=1000
    frames[-1]['ra_power']*=500
    modified,_=tool.forward_sequence(model,tool.engine.pack_features(frames,dtype=torch.float64),ids)
    for i in range(3):torch.testing.assert_close(original[i]['depths'],modified[i]['depths'],rtol=0,atol=0)


def test_public_ids_are_precomputed_from_observations_without_labels():
    frames,ids=sequence()
    values=[(SimpleNamespace(episode_id='synthetic',frame_idx=i,physical_time_s=float(i),features=f),
             {'gt_ids':['never read']}) for i,f in enumerate(frames)]
    class Assoc:
        def step(self,index,time,boxes,roi,scores):return ids[index]['member_ids']
    got=tool.public_identities(values,Assoc)
    assert [v['member_ids'] for v in got]==[v['member_ids'] for v in ids]
    assert all('gt_ids' not in v for v in got)


def dataset():
    values=[]
    for episode,total in [('a',5),('b',2)]:
        for i in range(total):
            f=synthetic_features(n=3,frame=i,episode=episode)
            for key in ('episode_id','member_ids','frame_index','timestamp_s'):f.pop(key)
            rho=np.full(3,20.+.1*i);truth=f['ray_origins']+rho[:,None]*f['ray_directions']
            obs=SimpleNamespace(episode_id=episode,frame_idx=i,physical_time_s=float(i),features=f)
            fixed=dict(rows=np.arange(3),columns=np.arange(3),truth=truth,depths=rho,detection_count=3,gt_count=3)
            values.append((obs,fixed))
    return values,dict(full_train=True,synthetic_only=True,manifest_sha256='a'*64,
                      observation_identity_sha256='b'*64,fixed_index_sha256='c'*64)


def test_tiny_training_preserves_optimizer_boundaries_and_standard_checkpoint(tmp_path):
    root=tmp_path/'storage'
    cfg=dict(schema='rvloc.experiment.v1',bandwidth_hz=32e6,center_contract='observer_scatter_at_camera_optical_center',
        manifest=None,storage=dict(output_root=str(root/'results'),weights_root=str(root/'weights'),cache_root=str(root/'cache')),
        model={},world=dict(hidden_size=64,rollout_horizons=[1,3]),
        training=dict(epochs=1,truncation_frames=4,log_every_updates=1,seed=20261118),
        evaluation=dict(primary='position_rmse_3d_m'))
    config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
    class PublicAssoc:
        calibration_sha256='d'*64
        def step(self,index,time,boxes,roi,scores):return [101,None,303]
    receipt=tool.train(config,'unused',root/'results/world',prepared_data=dataset(),association_factory=PublicAssoc)
    assert receipt['optimizer_updates']==3 and receipt['history'][0]['frames']==7
    assert receipt['history'][0]['unassigned_visual_members']==7
    model,payload=load_checkpoint(receipt['checkpoint'],expected_group=7)
    assert payload['metadata']['calibration_sha256']=='d'*64
    assert payload['metadata']['synthetic_only'] is True
    assert payload['metadata']['engine']['original_update_schedule_preserved']
    assert payload['metadata']['inference_uses_gt_ids'] is False
    assert not list((root/'results').rglob('*.pt'))
