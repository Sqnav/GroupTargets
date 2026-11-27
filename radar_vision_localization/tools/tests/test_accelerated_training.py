"""Synthetic optimizer schedule, loss and resume checks, no real data."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

path=Path(__file__).resolve().parents[1]/'accelerated_training.py'
spec=importlib.util.spec_from_file_location('accelerated_training_test',path)
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
sys.path.insert(0,str(path.parents[1]/'tests'))
from test_models import synthetic_features
from rvloc.models import localization_loss


def test_exact_update_boundaries_keep_episode_tail_flush():
    episodes=['a']*5+['b']*2+['c']*8
    assert tool.group_indices(episodes,4)==[[0,1,2,3],[4],[5,6],[7,8,9,10],[11,12,13,14]]
    assert tool.group_indices(episodes,64)==[list(range(15))]
    with pytest.raises(ValueError):tool.group_indices(episodes,0)


def test_batched_loss_equals_mean_of_per_frame_losses_with_unequal_member_counts():
    torch.manual_seed(4)
    depths=torch.randn((3,4),requires_grad=True)
    means=torch.randn((3,4),requires_grad=True)
    std=(torch.rand((3,4))+.5).requires_grad_()
    valid=torch.tensor([[True,False,False,False],[True,True,True,True],[False,False,False,False]])
    target=torch.randn((3,4))
    output=dict(depths=depths,visual_depth_mean_m=means,visual_depth_std_m=std,valid_mask=valid)
    targets=dict(depths=target,valid_mask=valid,eligible=torch.tensor([True,True,False]),matched_frames=2)
    loss,terms=tool.fixed_frame_loss(output,targets)
    reference=[]
    for i in (0,1):
        out={k:v[i] for k,v in output.items()};out['aux']={'architecture':'single_query_depth_v3'}
        reference.append(localization_loss(out,{'depths':target[i],'valid_mask':valid[i]})[0])
    expected=torch.stack(reference).mean()
    torch.testing.assert_close(loss,expected,rtol=1e-6,atol=1e-6)
    actual_grad=torch.autograd.grad(loss,(depths,means,std),retain_graph=True)
    expect_grad=torch.autograd.grad(expected,(depths,means,std))
    for a,b in zip(actual_grad,expect_grad):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-6)
    assert all(torch.equal(g[2],torch.zeros_like(g[2])) for g in actual_grad)


def dataset():
    values=[]
    for ep,count in [('a',5),('b',2)]:
        for frame in range(count):
            f=synthetic_features(frame=frame,episode=ep)
            for key in ('episode_id','member_ids','frame_index','timestamp_s'):f.pop(key)
            n=len(f['ray_directions']);rho=np.full(n,20.+frame*.1)
            truth=f['ray_origins']+rho[:,None]*f['ray_directions']
            obs=SimpleNamespace(episode_id=ep,frame_idx=frame,features=f)
            fixed=dict(rows=np.arange(n),columns=np.arange(n),truth=truth,depths=rho,detection_count=n,gt_count=n)
            values.append((obs,fixed))
    identity=dict(full_train=True,synthetic_only=True,manifest_sha256='a'*64,
                  observation_identity_sha256='b'*64,fixed_index_sha256='c'*64,frames=7,matched_members=21)
    return values,identity


def config(tmp,epochs):
    root=tmp/'storage'
    data=dict(schema='rvloc.experiment.v1',bandwidth_hz=32e6,
        center_contract='observer_scatter_at_camera_optical_center',manifest=None,
        storage=dict(output_root=str(root/'results'),weights_root=str(root/'weights'),cache_root=str(root/'cache')),
        model={},training=dict(epochs=epochs,device='cpu',truncation_frames=4,log_every_updates=1,
            learning_rate=.0003,weight_decay=.0001,warmup_epochs=2,seed=20261118,gradient_clip=5.),
        evaluation=dict(primary='position_rmse_3d_m'))
    path=tmp/f'config_{epochs}.json';path.write_text(json.dumps(data));return path,root


def test_epoch_boundary_resume_restores_adamw_and_does_not_repeat_completed_epochs(tmp_path):
    torch.set_num_threads(1)
    c2,root=config(tmp_path,2);c1,_=config(tmp_path,1)
    prepared=dataset()
    reference=tool.train(c2,2,'unused',root/'results/full',prepared_data=prepared)
    first=tool.train(c1,2,'unused',root/'results/first',prepared_data=prepared)
    last=Path(first['weights_dir'])/'epoch_0001.pt'
    resumed=tool.train(c2,2,'unused',root/'results/resumed',resume_checkpoint=last,prepared_data=prepared)
    assert first['optimizer_updates']==3
    assert resumed['optimizer_updates']==reference['optimizer_updates']==6
    assert resumed['new_epochs']==1 and resumed['resume']['optimizer_restored']
    assert resumed['resume']['rng_restored']
    assert not (Path(resumed['weights_dir'])/'epoch_0001.pt').exists()
    one=torch.load(reference['checkpoint'],map_location='cpu',weights_only=True)
    two=torch.load(resumed['checkpoint'],map_location='cpu',weights_only=True)
    for key in one['state_dict']:torch.testing.assert_close(one['state_dict'][key],two['state_dict'][key],atol=0,rtol=0)
    assert two['metadata']['synthetic_only'] is True
    assert two['metadata']['engine']['original_update_schedule_preserved']
    assert two['metadata']['observer_position_reference']=='camera_optical_center'
    assert resumed['history'][0]['epoch']==1 and resumed['history'][1]['epoch']==2
    assert resumed['history'][-1]['online_fixed_train_rmse_m']>=0
    assert list((root/'results').rglob('*.pt'))==[]
    assert (Path(resumed['output_dir'])/'receipt.json').is_file()


def test_refuses_overwrite_or_cross_method_resume(tmp_path):
    torch.set_num_threads(1)
    c1,root=config(tmp_path,1)
    first=tool.train(c1,2,'unused',root/'results/first',prepared_data=dataset())
    with pytest.raises(FileExistsError):tool.train(c1,2,'unused',root/'results/first',prepared_data=dataset())
    config_obj=tool.ExperimentConfig.load(c1);settings=tool._settings(config_obj)
    with pytest.raises(ValueError,match='method/data'):
        tool.resume_model(Path(first['weights_dir'])/'epoch_0001.pt',config_obj,'a'*64,3,'cpu',settings)
