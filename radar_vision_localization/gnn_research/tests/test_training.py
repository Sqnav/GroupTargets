"""Synthetic checks only: no real dataset, checkpoints or GPU are used."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import copy
import pytest
import torch

path=Path(__file__).resolve().parents[1]/'train.py'
spec=importlib.util.spec_from_file_location('mce_train_test',path);tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.value=torch.nn.Parameter(torch.tensor(1.))
        self.config={'architecture':'single_query_depth_v3','rf':{},'graph_variant':'shared'}
        self.core=SimpleNamespace(trained=False)


def test_fixed_sse_uses_original_roi_identity_and_three_coordinate_sum(monkeypatch):
    model=Tiny();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    def prediction(m,packed):
        assert set(packed)=={'public'}
        p=m.value*torch.tensor([[[1.,2.,2.],[10.,0.,0.]]])
        return dict(depths=m.value*torch.tensor([[1.,10.]]),member_positions=p,
                    valid_mask=torch.tensor([[True,True]]),visual_depth_mean_m=m.value*torch.tensor([[1.,10.]]),
                    visual_depth_std_m=torch.ones((1,2)))
    monkeypatch.setattr(tool,'forward',prediction)
    targets=dict(depths=torch.zeros((1,2)),positions=torch.zeros((1,2,3),dtype=torch.float64),valid_mask=torch.tensor([[True,True]]),eligible=torch.tensor([True]),matched_frames=1)
    batch=dict(inputs={'public':1},targets=targets,matched_frames=1)
    _,_,sse,count,_,_=tool.step(model,optimizer,batch,{'grad_clip':5.})
    assert sse.item()==109 and count.item()==2
    assert model.value.item()!=1.


def test_private_labels_are_not_forward_inputs(monkeypatch):
    model=Tiny();optimizer=torch.optim.AdamW(model.parameters(),lr=.001);seen=[]
    def prediction(m,packed):
        seen.append(copy.deepcopy(packed));z=m.value.reshape(1,1)
        return dict(depths=z,member_positions=z[...,None].expand(1,1,3),valid_mask=torch.ones((1,1),dtype=torch.bool),visual_depth_mean_m=z,visual_depth_std_m=torch.ones_like(z))
    monkeypatch.setattr(tool,'forward',prediction)
    private={'depths':torch.tensor([[2.]]),'positions':torch.tensor([[[2.,3.,4.]]],dtype=torch.float64),'valid_mask':torch.ones((1,1),dtype=torch.bool),'eligible':torch.tensor([True]),'matched_frames':1}
    tool.step(model,optimizer,dict(inputs={'public_sensor':True},targets=private,matched_frames=1),{'grad_clip':5.})
    assert seen==[{'public_sensor':True}]


def test_graph_checkpoint_schema_and_variant_are_not_interchangeable(tmp_path,monkeypatch):
    model=Tiny();path=tmp_path/'model.pt'
    tool.save_graph_checkpoint(model,path,variant='shared',metadata={'optimizer_updates':1})
    monkeypatch.setattr(tool,'build',lambda options,variant,rf:Tiny())
    loaded,payload,sha=tool.load_graph_checkpoint(path,expected_variant='shared')
    assert loaded.core.trained and payload['schema']==tool.SCHEMA and len(sha)==64
    with pytest.raises(ValueError,match='variant'):tool.load_graph_checkpoint(path,expected_variant='isolated')
    old=tmp_path/'old.pt';torch.save({'schema':'rvloc_seven_groups_single_query_depth_v3'},old)
    with pytest.raises(ValueError,match='MCE-GNN'):tool.load_graph_checkpoint(old)


def test_checkpoint_refuses_overwriting_incomplete_serialization(tmp_path):
    model=Tiny();path=tmp_path/'epoch.pt'
    tool.save_graph_checkpoint(model,path,variant='shared')
    value=torch.load(path,weights_only=True)
    assert value['architecture']==tool.ARCHITECTURE
    assert not list(tmp_path.glob('.checkpoint-*'))


def test_numpy_rf_and_metadata_roundtrip_remain_weights_only_safe(tmp_path,monkeypatch):
    import numpy as np
    import random
    model=Tiny();model.config['graph_rf']={'frequencies_hz':[np.float64(2.4e10)],'c_m_s':np.float64(299792458.)}
    state=tool.base._rng_state('cpu');state['diagnostic_numpy']=np.array([1.,2.],dtype=np.float64)
    metadata={'optimizer_updates':np.int64(1),'rng_state':state,'metric':np.float64(1.2)}
    path=tmp_path/'safe.pt';tool.save_graph_checkpoint(model,path,variant='shared',metadata=metadata)
    value=torch.load(path,weights_only=True)
    assert type(value['config']['graph_rf']['frequencies_hz'][0]) is float
    assert type(value['metadata']['metric']) is float
    assert torch.equal(value['metadata']['rng_state']['torch_cpu'],state['torch_cpu'])
    assert isinstance(value['metadata']['rng_state']['python'],tuple)
    original=random.getstate();tool.base._restore_rng(value['metadata']['rng_state'],'cpu');assert random.getstate()==original
    received={}
    def build(options,variant,rf):received.update(rf);return Tiny()
    monkeypatch.setattr(tool,'build',build)
    tool.load_graph_checkpoint(path)
    assert received=={'frequencies_hz':[2.4e10],'c_m_s':299792458.}
