"""Physical topology and real sharing tests; generated tensors only."""
import torch
from gnn_research.graph import build_graph,rf_scales,MessageLayer


def fixture(variant='shared'):
    dtype=torch.float64
    radius=torch.tensor([[[20.,20.5],[20.,20.5]]],dtype=dtype)
    az=torch.tensor([[[.05,.055],[.052,.058]]],dtype=dtype)
    el=torch.full_like(az,.12)
    g=dict(ranges=radius,azimuth=az,elevation=el,valid=torch.ones_like(radius,dtype=torch.bool))
    r=torch.arange(10,dtype=dtype)*299792458/(2*32e6)
    a=torch.linspace(-.6,.6,25,dtype=dtype);e=torch.linspace(-.5,.5,21,dtype=dtype)
    torch.manual_seed(7)
    ra=torch.randn(1,8,10,25,dtype=dtype);re=torch.randn(1,8,10,21,dtype=dtype)
    av=torch.ones(1,10,25,dtype=torch.bool);ev=torch.ones(1,10,21,dtype=torch.bool)
    return build_graph(g,ra,re,r,a,e,av,ev,2,rf_scales(),variant)


def test_real_grid_nodes_are_deduplicated_and_members_share():
    shared=fixture();isolated=fixture('isolated')
    assert len(shared['node_grid_ids'])==len(torch.unique(shared['node_grid_ids']))
    assert len(isolated['node_grid_ids'])>len(shared['node_grid_ids'])
    assert shared['evidence_member_degree'].max()==2
    assert isolated['evidence_member_degree'].max()==1
    assert shared['diagnostics']['shared_member_count'].tolist()==[2]
    assert isolated['diagnostics']['shared_member_count'].tolist()==[0]
    # Multiple hypotheses of one member are explicitly distinct from members.
    assert (isolated['evidence_candidate_degree']>1).any()


def test_shuffle_rewires_features_without_relabelling_and_preserves_degrees():
    shared=fixture();wrong=fixture('shuffled')
    torch.testing.assert_close(shared['node_features'],wrong['node_features'],rtol=0,atol=0)
    assert torch.equal(shared['candidate_index'],wrong['candidate_index'])
    torch.testing.assert_close(shared['attributes'],wrong['attributes'],rtol=0,atol=0)
    assert torch.equal(shared['evidence_candidate_degree'],wrong['evidence_candidate_degree'])
    assert not torch.equal(shared['evidence_index'],wrong['evidence_index'])
    assert wrong['diagnostics']['shuffled_changed_edge_count'].sum()>0


def test_cross_member_information_path_exists_only_with_shared_nodes():
    torch.manual_seed(12);layer=MessageLayer(8).double()
    def run(kind,change):
        graph=fixture(kind);candidate=torch.randn(4,8,dtype=torch.float64,generator=torch.Generator().manual_seed(2))
        if change:candidate[:2]+=torch.arange(8,dtype=torch.float64)*.5
        evidence=graph['node_features'];member=torch.randn(2,8,dtype=torch.float64,generator=torch.Generator().manual_seed(3))
        mass=torch.full((4,),.5,dtype=torch.float64)
        return layer(candidate,evidence,member,graph,mass)[0][2:]
    torch.testing.assert_close(run('isolated',False),run('isolated',True),rtol=0,atol=0)
    assert not torch.allclose(run('shared',False),run('shared',True),atol=1e-8,rtol=1e-8)


def test_rf_scales_do_not_use_fft_angle_sampling_as_resolution():
    scales=rf_scales()
    assert abs(scales['range_first_null_m']-4.68425715625)<1e-12
    assert abs(scales['uy_first_null']-.125)<1e-12
    assert abs(scales['uz_first_null']-.125)<1e-12
