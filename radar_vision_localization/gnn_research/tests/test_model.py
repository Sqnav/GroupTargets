"""One-query graph branch boundaries, controls and differentiability."""
import copy
import importlib.util
from pathlib import Path
import pytest
import torch
from rvloc.models import build_model
from gnn_research.model import build_graph_model,forward_batch,pack_features

p=Path(__file__).resolve().parents[2]/'tests/test_models.py'
spec=importlib.util.spec_from_file_location('graph_fixture',p);fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


@pytest.fixture(autouse=True)
def threads():torch.set_num_threads(1)


def feature(n=3,seed=2):
    f=fixture.synthetic_features(n=n)
    generator=torch.Generator().manual_seed(seed)
    f['detector_roi_features']=torch.randn(n,1024,generator=generator).numpy()
    return f


def test_backbone_initialisation_and_control_parameter_initialisation_match():
    torch.manual_seed(20);base=build_model(3)
    torch.manual_seed(20);graph=build_graph_model()
    for name,value in base.state_dict().items():
        if not name.startswith('core.local_query.'):
            torch.testing.assert_close(value,graph.state_dict()[name],rtol=0,atol=0)
    torch.manual_seed(20);control=build_graph_model(variant='isolated')
    for name,value in graph.state_dict().items():torch.testing.assert_close(value,control.state_dict()[name],rtol=0,atol=0)


@pytest.mark.parametrize('variant',['shared','isolated','shuffled'])
def test_batch_padding_isolation_finite_gradients_and_one_query(variant):
    torch.manual_seed(5);model=build_graph_model(variant=variant).double()
    fs=[feature(3),feature(1,3),feature(0)]
    packed=pack_features(fs,dtype=torch.float64)
    out=forward_batch(model,packed)
    assert out['aux']['query_rounds']==1
    assert torch.equal(out['valid_mask'].sum(1),torch.tensor([3,1,0]))
    diag=out['graph_diagnostics']
    for name in ('valid_candidate_count','shared_candidate_count','valid_member_count','shared_member_count',
                 'unique_evidence_nodes','edge_count','shuffled_changed_edge_count',
                 'geometry_valid_candidate_count','connected_member_count'):
        assert diag[name].shape==(2,),name
    expected=packed['ray_origins']+out['depths'][...,None]*packed['ray_directions']
    torch.testing.assert_close(out['member_positions'],expected*out['valid_mask'][...,None])
    loss=out['depths'].sum()+.1*out['visual_depth_std_m'].sum();loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    changed=copy.deepcopy(fs);changed[1]['detector_roi_features']*=20
    before=forward_batch(model,packed)['depths'][0]
    after=forward_batch(model,pack_features(changed,dtype=torch.float64))['depths'][0]
    torch.testing.assert_close(before,after,rtol=0,atol=0)


def test_member_permutation_equivariance():
    torch.manual_seed(9);model=build_graph_model().double().eval();f=feature(3)
    perm=[2,0,1];changed=copy.deepcopy(f)
    for key in ('ray_origins','ray_directions','ray_depth_bounds','ray_valid','visual_scale','detector_roi_features','bbox_xyxy','detection_scores'):
        changed[key]=changed[key][perm]
    a=model(f)['depths'];b=model(changed)['depths']
    torch.testing.assert_close(a[perm],b,rtol=1e-8,atol=1e-8)


def test_all_invalid_radar_has_zero_local_evidence_and_finite_outputs():
    torch.manual_seed(14);model=build_graph_model().double();f=feature()
    f['ra_valid'][:]=False;f['re_valid'][:]=False
    out=model(f)
    assert not out['sampling_diagnostics']['evidence_available'].any()
    assert out['sampling_diagnostics']['edge_count'].sum()==0
    assert torch.isfinite(out['depths']).all()
    out['depths'].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_graph_forbids_gt_and_future_normal_inputs():
    model=build_graph_model();f=feature();f['true_depths']=[10,20,30]
    with pytest.raises(ValueError,match='forbid'):model(f)
