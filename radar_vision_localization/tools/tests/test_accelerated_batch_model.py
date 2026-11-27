"""Numerical/gradient equivalence to frozen single-frame weights on synthetic data."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from rvloc.models import build_model
from rvloc.models.single_query import SingleRadarQuery
from rvloc.legacy import activate
activate()
from depth_localization.geometry import sample_paired_spectra

path=Path(__file__).resolve().parents[1]/'accelerated_batch_model.py'
spec=importlib.util.spec_from_file_location('accelerated_batch_model_test',path)
tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)
fixture_path=path.parents[0].parent/'tests/test_models.py'
spec2=importlib.util.spec_from_file_location('single_frame_synthetic_fixtures',fixture_path)
fixtures=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(fixtures)


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


def features(n=3,seed=10):
    f=fixtures.synthetic_features(n=n)
    rng=np.random.default_rng(seed)
    f['detector_roi_features']=rng.normal(size=(n,1024)).astype(np.float32)
    f['ra_power']=rng.uniform(.01,2.,f['ra_power'].shape).astype(np.float32)
    f['re_power']=rng.uniform(.01,2.,f['re_power'].shape).astype(np.float32)
    return f


@pytest.mark.parametrize('group',[2,3])
def test_batch_outputs_match_independent_frames_and_padding(group):
    torch.manual_seed(42);model=build_model(group).double().train()
    fs=[features(3,1),features(1,2),features(5,3),features(0,4)]
    fs[0]['ray_valid'][1]=False
    batch=tool.pack_features(fs,visual=group==2,dtype=torch.float64)
    actual=tool.forward_batch(model,batch)
    for i,f in enumerate(fs):
        expected=model(f)
        for key in ('depths','visual_depth_mean_m','visual_depth_std_m','member_positions','valid_mask'):
            torch.testing.assert_close(actual[key][i,:len(f['ray_directions'])],expected[key],rtol=1e-9,atol=1e-9)
        torch.testing.assert_close(actual['center'][i],expected['center'],rtol=1e-9,atol=1e-9)
    assert not actual['valid_mask'][-1].any()
    assert torch.count_nonzero(actual['member_positions'][-1])==0


@pytest.mark.parametrize('group',[2,3])
def test_parameter_gradients_match_mean_of_independent_frame_losses(group):
    torch.manual_seed(41);original=build_model(group).double().train();batched=copy.deepcopy(original)
    fs=[features(3,4),features(1,5),features(4,6)]
    batch=tool.pack_features(fs,visual=group==2,dtype=torch.float64)
    output=tool.forward_batch(batched,batch)
    losses=[]
    for i,f in enumerate(fs):
        count=len(f['ray_directions'])
        losses.append(output['depths'][i,:count].square().mean()+.2*output['visual_depth_std_m'][i,:count].mean())
    torch.stack(losses).mean().backward()
    reference=[]
    for f in fs:
        value=original(f);reference.append(value['depths'].square().mean()+.2*value['visual_depth_std_m'].mean())
    torch.stack(reference).mean().backward()
    for (name,a),(name2,b) in zip(original.named_parameters(),batched.named_parameters()):
        assert name==name2
        assert (a.grad is None)==(b.grad is None),name
        if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=2e-8,atol=2e-8,msg=name)


def test_frames_cannot_attend_to_other_frames():
    torch.manual_seed(44);model=build_model(3).double().eval()
    one,two=features(3,1),features(3,2)
    first=tool.forward_batch(model,tool.pack_features([one,two],dtype=torch.float64))
    two['detector_roi_features']*=1000;two['ra_power']*=500
    changed=tool.forward_batch(model,tool.pack_features([one,two],dtype=torch.float64))
    torch.testing.assert_close(first['depths'][0],changed['depths'][0],rtol=0,atol=0)


def test_batched_sampler_matches_trusted_nonuniform_axes_and_independent_masks():
    torch.manual_seed(22)
    ranges=torch.tensor([0.,4.,11.,30.,80.],dtype=torch.float64)
    angles=torch.tensor([-.8,-.2,.1,.7],dtype=torch.float64)
    image=torch.randn(2,3,5,4,dtype=torch.float64,requires_grad=True)
    masks=torch.ones(2,5,4,dtype=torch.bool);masks[1,2,2]=False
    query_r=torch.tensor([[[5.,12.,90.]],[[1.,11.,20.]]],dtype=torch.float64,requires_grad=True)
    query_a=torch.tensor([[[.2,-.1,.3]],[[.1,.1,.6]]],dtype=torch.float64,requires_grad=True)
    valid=torch.ones(2,1,3,dtype=torch.bool)
    sampled,ok=tool._sample(image,masks,query_r,query_a,ranges,angles,valid)
    for i in range(2):
        reference=sample_paired_spectra(image[i],image[i],ranges=query_r[i],azimuth=query_a[i],elevation=query_a[i],
            range_axis=ranges,azimuth_axis=angles,elevation_axis=angles,ra_valid=masks[i],re_valid=masks[i],candidate_valid=valid[i])
        torch.testing.assert_close(sampled[i],reference['ra'],rtol=0,atol=0)
        assert torch.equal(ok[i],reference['ra_valid'])
    assert not ok[0,0,-1]
    assert not ok[1,0,1]
    gradient=torch.autograd.grad(sampled.sum(),(image,query_r,query_a))
    assert all(torch.isfinite(x).all() for x in gradient)


def test_all_missing_evidence_preserves_unused_candidate_transformer_grad_none():
    torch.manual_seed(12);a=build_model(3).double().train();b=copy.deepcopy(a)
    fs=[features(3,7),features(2,9)]
    for f in fs:f['ra_valid'][:]=False;f['re_valid'][:]=False
    out=tool.forward_batch(b,tool.pack_features(fs,dtype=torch.float64))
    sum(out['depths'][i,:len(f['ray_directions'])].sum() for i,f in enumerate(fs)).backward()
    sum(a(f)['depths'].sum() for f in fs).backward()
    for (name,p),(name2,q) in zip(a.named_parameters(),b.named_parameters()):
        assert name==name2
        assert (p.grad is None)==(q.grad is None),name
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-8,atol=1e-8,msg=name)


def test_rgb_never_inspects_radar_and_world_core_requires_explicit_prior():
    model=build_model(2)
    f=features()
    class Poison:
        def __array__(self,*args,**kwargs):raise AssertionError('RGB opened radar')
    for key in ('ra_power','re_power','ra_valid','re_valid','range_m','azimuth_rad','elevation_rad'):f[key]=Poison()
    tool.forward_batch(model,tool.pack_features([f],visual=True))
    world=build_model(7)
    with pytest.raises(ValueError,match='explicit causal'):
        tool.forward_batch(world,tool.pack_features([features()]))


def test_g7_evidence_can_be_preencoded_then_causal_prior_injected_per_frame():
    torch.manual_seed(66);reference=build_model(7).double().train();batched=copy.deepcopy(reference)
    fs=[features(3,1),features(1,2),features(4,3),features(2,4)]
    packed=tool.pack_features(fs,dtype=torch.float64)
    evidence=tool.prepare_evidence(batched,packed)
    expected_loss=[];actual_loss=[]
    for i,f in enumerate(fs):
        n=len(f['ray_directions'])
        prior=torch.randn(n,64,dtype=torch.float64)
        original=reference.core({k:v for k,v in f.items() if k not in
            ('episode_id','frame_index','timestamp_s','member_ids')},fusion_prior=prior)
        padded_prior=torch.zeros(1,packed['ray_valid'].shape[1],64,dtype=torch.float64)
        padded_prior[0,:n]=prior
        result=tool.finish_depth(batched,tool.slice_prepared(evidence,i),fusion_prior=padded_prior)
        torch.testing.assert_close(result['depths'][0,:n],original['depths'],rtol=1e-9,atol=1e-9)
        expected_loss.append(original['depths'].square().mean())
        actual_loss.append(result['depths'][0,:n].square().mean())
    torch.stack(expected_loss).mean().backward();torch.stack(actual_loss).mean().backward()
    for (name,p),(name2,q) in zip(reference.named_parameters(),batched.named_parameters()):
        assert name==name2 and (p.grad is None)==(q.grad is None),name
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-8,atol=2e-8,msg=name)


@pytest.mark.parametrize('group',[2,3])
def test_float32_outputs_are_close_to_original(group):
    torch.manual_seed(67);model=build_model(group).train()
    fs=[features(3,1),features(1,2),features(4,3),features(2,4)]
    result=tool.forward_batch(model,tool.pack_features(fs,visual=group==2))
    for i,f in enumerate(fs):
        original=model(f)
        torch.testing.assert_close(result['depths'][i,:len(f['ray_directions'])],original['depths'],rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(result['member_positions'][i,:len(f['ray_directions'])],original['member_positions'],rtol=1e-5,atol=1e-5)
