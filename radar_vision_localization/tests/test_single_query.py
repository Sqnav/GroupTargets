"""Single-query tests use smooth synthetic feature maps, never radar data.

The fixture exercises the audited physical sampler rather than manufacturing
an independent scattering/forward model. No GT is passed to normal queries.
"""
from unittest.mock import patch

import pytest
import torch

from rvloc.models import single_query
from rvloc.models.single_query import DEFAULT_MULTIPLIERS, SingleRadarQuery


def fixture(n=2, *, gradients=False):
    torch.manual_seed(841)
    hidden=8
    ranges=torch.tensor([0.,8.,15.,25.,40.,80.])
    az=torch.tensor([-1.2,-.4,-.1,.15,.5,1.2])
    el=torch.tensor([-1.1,-.3,.05,.4,1.1])
    channel=torch.arange(1,hidden+1,dtype=torch.float32)[:,None,None]
    ra=(channel*(ranges[None,:,None]/8+az[None,None,:]*.7)).clone().requires_grad_(gradients)
    re=(channel*(ranges[None,:,None]/11+el[None,None,:]*1.3)).clone().requires_grad_(gradients)
    mean=torch.linspace(15.,20.,n).requires_grad_(gradients)
    std=torch.linspace(1.1,1.7,n).requires_grad_(gradients)
    args=dict(origin=torch.tensor([[5.,2.,1.]]).expand(n,3).clone(),
              direction=torch.tensor([[1.,0.,0.]]).expand(n,3).clone(),
              bounds=torch.tensor([[1.,60.]]).expand(n,2).clone(),ray_valid=torch.ones(n,dtype=torch.bool),
              visual_mean=mean,visual_std=std,ra_map=ra,re_map=re,range_axis=ranges,
              azimuth_axis=az,elevation_axis=el,ra_valid=torch.ones(ra.shape[-2:],dtype=torch.bool),
              re_valid=torch.ones(re.shape[-2:],dtype=torch.bool),range_scale_m=200.)
    return SingleRadarQuery(hidden,2).eval(),args


def test_one_paired_query_for_all_members_and_matches_trusted_coordinates():
    module,args=fixture(4)
    original=single_query.sample_paired_spectra
    with patch.object(single_query,'sample_paired_spectra',wraps=original) as sampled, \
         patch.object(single_query,'candidate_geometry',wraps=single_query.candidate_geometry) as geometry:
        pooled,diag=module(**args)
    assert sampled.call_count==1 and geometry.call_count==1
    assert sampled.call_args.kwargs['ranges'].shape==(4,9)
    assert pooled.shape==(4,8) and torch.isfinite(pooled).all()
    expected=args['visual_mean'][:,None]+args['visual_std'][:,None]*torch.tensor(DEFAULT_MULTIPLIERS)[None]
    assert torch.equal(diag['candidate_depths_m'],expected)
    torch.testing.assert_close(diag['candidate_offsets_m'],expected-args['visual_mean'][:,None])
    torch.testing.assert_close(diag['weights'].sum(-1),torch.ones(4))
    assert diag['query_rounds']==1 and diag['candidate_interaction_layers']==1
    assert len(module.candidate_interaction.layers)==1
    assert all(name not in str(dict(module.named_modules())) for name in ('refine_head','refine_encoder','fine_query'))
    reference=single_query.candidate_geometry(args['origin'],args['direction'],expected,args['bounds'],args['ray_valid'])
    raw=original(args['ra_map'],args['re_map'],ranges=reference['ranges'],azimuth=reference['azimuth'],
        elevation=reference['elevation'],range_axis=args['range_axis'],azimuth_axis=args['azimuth_axis'],
        elevation_axis=args['elevation_axis'],ra_valid=args['ra_valid'],re_valid=args['re_valid'],candidate_valid=reference['valid'])
    torch.testing.assert_close(diag['sampled_ra'],raw['ra'])
    torch.testing.assert_close(diag['sampled_re'],raw['re'])
    assert torch.equal(diag['ra_valid'],raw['ra_valid'])
    assert torch.equal(diag['re_valid'],raw['re_valid'])


def test_candidate_outside_optical_support_is_masked_without_clipping_or_duplication():
    module,args=fixture(1)
    args.update(visual_mean=torch.tensor([2.]),visual_std=torch.tensor([2.]),bounds=torch.tensor([[1.5,3.5]]))
    _,diag=module(**args)
    expected=torch.tensor([[-4.,-2.,0.,1.,2.,3.,4.,6.,8.]])
    assert torch.equal(diag['candidate_depths_m'],expected)
    assert len(torch.unique(diag['candidate_depths_m']))==9
    expected_valid=torch.tensor([[False,False,False,False,True,True,False,False,False]])
    assert torch.equal(diag['geometry_valid'],expected_valid)
    assert torch.equal(diag['usable'],expected_valid)
    assert torch.equal(diag['weights'][~expected_valid],torch.zeros(7))
    torch.testing.assert_close(diag['weights'].sum(-1),torch.ones(1))


@pytest.mark.parametrize('missing',['ra','re'])
def test_ra_and_re_masks_remain_independent(missing):
    module,args=fixture()
    args[missing+'_valid'][:]=False
    pooled,diag=module(**args)
    present='re' if missing=='ra' else 'ra'
    assert not diag[missing+'_valid'].any()
    assert diag[present+'_valid'].all()
    assert diag['usable'].all() and diag['evidence_available'].all()
    assert torch.equal(diag['sampled_'+missing],torch.zeros_like(diag['sampled_'+missing]))
    torch.testing.assert_close(diag['weights'].sum(-1),torch.ones(2))
    assert torch.isfinite(pooled).all()


def test_all_invalid_returns_zero_without_bias_or_all_masked_transformer_nan():
    module,args=fixture()
    args['ra_valid'][:]=False;args['re_valid'][:]=False
    with torch.no_grad():
        module.candidate_encoder[0].bias.fill_(12.)
        module.candidate_score.bias.fill_(100.)
    with patch.object(module.candidate_interaction,'forward',side_effect=AssertionError('all-invalid attention ran')):
        pooled,diag=module(**args)
    assert torch.equal(pooled,torch.zeros_like(pooled))
    assert torch.equal(diag['weights'],torch.zeros_like(diag['weights']))
    assert not diag['evidence_available'].any() and not diag['usable'].any()
    assert torch.isfinite(pooled).all() and torch.isfinite(diag['weights']).all()
    # There was a query, but no usable measurement evidence was invented.
    assert diag['query_rounds']==1


def test_mixed_invalid_members_are_isolated_inside_one_candidate_batch():
    module,args=fixture(3)
    args['origin'][1]=torch.tensor([-100.,0.,1.]);args['direction'][1]=torch.tensor([-1.,0.,0.])
    args['ray_valid'][2]=False
    pooled,diag=module(**args)
    assert diag['evidence_available'].tolist()==[True,False,False]
    assert torch.equal(pooled[1:],torch.zeros_like(pooled[1:]))
    assert torch.equal(diag['weights'][1:],torch.zeros_like(diag['weights'][1:]))
    assert torch.isfinite(pooled).all()


def test_physical_axis_miss_does_not_invalidate_camera_ray_or_create_evidence():
    module,args=fixture(1)
    args.update(visual_mean=torch.tensor([50.]),visual_std=torch.tensor([1.]),bounds=torch.tensor([[1.,80.]]))
    args['range_axis']=torch.tensor([0.,1.,2.,3.,4.,5.])
    pooled,diag=module(**args)
    assert diag['geometry_valid'].all()
    assert not diag['usable'].any() and not diag['ra_valid'].any() and not diag['re_valid'].any()
    assert torch.equal(pooled,torch.zeros_like(pooled))
    assert args['ray_valid'].tolist()==[True]


def test_raw_sampler_and_interaction_gradients_reach_visual_mean_std_and_both_maps():
    module,args=fixture(gradients=True)
    pooled,diag=module(**args)
    coefficient=torch.arange(1.,10.)[None,:,None]
    # This loss uses raw sampled maps, so geometry scalar encoding cannot
    # masquerade as a differentiable physical sampling path.
    raw_loss=((diag['sampled_ra']+diag['sampled_re'])*coefficient).sum()
    raw_loss.backward(retain_graph=True)
    for field in ('visual_mean','visual_std','ra_map','re_map'):
        grad=args[field].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>1e-6,field
        grad.zero_()
    pooled.square().sum().backward()
    for field in ('visual_mean','visual_std','ra_map','re_map'):
        grad=args[field].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>1e-8,field
    assert module.candidate_interaction.layers[0].self_attn.in_proj_weight.grad.abs().sum()>0
    assert module.candidate_score.weight.grad.abs().sum()>0


def test_empty_member_batch_keeps_one_paired_call_and_finite_empty_outputs():
    module,args=fixture(0)
    with patch.object(single_query,'sample_paired_spectra',wraps=single_query.sample_paired_spectra) as call:
        pooled,diag=module(**args)
    assert call.call_count==1
    assert pooled.shape==(0,8) and diag['weights'].shape==(0,9)
    assert torch.isfinite(pooled).all()


@pytest.mark.parametrize('kwargs',[dict(candidate_multipliers=[0.,0.]),dict(candidate_multipliers=[float('nan')]),dict(layers=2),dict(hidden=9,heads=2)])
def test_invalid_query_contract_is_rejected(kwargs):
    with pytest.raises(ValueError):SingleRadarQuery(**kwargs)


def test_inference_query_does_not_accept_ground_truth_or_nonpositive_std():
    module,args=fixture()
    with pytest.raises(TypeError):module(**args,target_positions=torch.zeros(2,3))
    args['visual_std'][0]=0
    with pytest.raises(ValueError,match='standard deviation'):module(**args)


@pytest.mark.parametrize('loss_kind',['raw','pooled'])
def test_station_origin_invalid_candidate_has_finite_backward_without_forward_changes(loss_kind):
    torch.manual_seed(654)
    module=SingleRadarQuery(8,2,[-1.,0.,1.]).eval()
    mean=torch.tensor([1.],requires_grad=True)
    std=torch.tensor([.25],requires_grad=True)
    ra=torch.randn(8,3,3,requires_grad=True)
    re=torch.randn(8,3,3,requires_grad=True)
    args=dict(origin=torch.tensor([[1.,0.,0.]]),direction=torch.tensor([[-1.,0.,0.]]),
        bounds=torch.tensor([[.1,2.]]),ray_valid=torch.tensor([True]),visual_mean=mean,visual_std=std,
        ra_map=ra,re_map=re,range_axis=torch.tensor([0.,1.,2.]),azimuth_axis=torch.tensor([-.5,0.,.5]),
        elevation_axis=torch.tensor([-.5,0.,.5]),ra_valid=torch.ones(3,3,dtype=torch.bool),
        re_valid=torch.ones(3,3,dtype=torch.bool),range_scale_m=3.)
    pooled,diag=module(**args)
    assert torch.equal(diag['candidate_depths_m'],torch.tensor([[.75,1.,1.25]]))
    assert diag['geometry_valid'].tolist()==[[True,False,False]]
    assert diag['usable'].tolist()==[[True,False,False]]
    assert torch.equal(diag['sampled_ra'][:,1:],torch.zeros_like(diag['sampled_ra'][:,1:]))
    assert torch.isfinite(pooled).all()
    loss=(diag['sampled_ra']+diag['sampled_re']).square().sum() if loss_kind=='raw' else pooled.square().sum()
    loss.backward()
    for name,value in [('mean',mean),('std',std),('ra',ra),('re',re)]:
        assert value.grad is not None and torch.isfinite(value.grad).all(),name
        assert value.grad.abs().sum()>1e-9,name
