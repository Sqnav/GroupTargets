"""Synthetic fixed-label/checkpoint monitoring tests; no real data or GPUs."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

path = Path(__file__).resolve().parents[1]/'checkpoint_train_rmse.py'
spec = importlib.util.spec_from_file_location('checkpoint_train_rmse_test_backend', path)
tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)


def test_three_coordinate_squares_are_summed_and_swaps_are_not_rematched():
    fixed = dict(detection_count=2,gt_count=2,rows=np.array([0,1]),columns=np.array([0,1]),
                 truth=np.array([[0.,0,0],[10.,0,0]]))
    errors, counts = tool._errors(np.array([[1.,2,2],[10.,0,0]]),np.array([True,True]),fixed)
    assert errors.tolist()==[3.,0.]
    assert float(np.sqrt(np.mean(errors**2)))==pytest.approx(np.sqrt(4.5))
    errors,_=tool._errors(np.array([[10.,0,0],[0.,0,0]]),np.array([True,True]),fixed)
    assert errors.tolist()==[10.,10.]
    assert counts['matched_members']==2


def test_invalid_outputs_keep_missing_gt_but_valid_nonfinite_outputs_fail():
    fixed=dict(detection_count=2,gt_count=3,rows=np.array([0,1]),columns=np.array([0,1]),
               truth=np.array([[0.,0,0],[10.,0,0],[20.,0,0]]))
    errors,counts=tool._errors([[0.,0,0],[np.nan,np.nan,np.nan]],np.array([True,False]),fixed)
    assert errors.tolist()==[0.]
    assert counts['gt_members']-counts['matched_members']==2
    with pytest.raises(ValueError,match='nonfinite'):
        tool._errors([[0.,0,0],[np.nan,0,0]],np.array([True,True]),fixed)


@pytest.mark.parametrize('name',['model.pt','epoch_0000.pt','incomplete.pt'])
def test_only_positive_committed_epoch_checkpoint_is_allowed(tmp_path,name):
    p=tmp_path/name
    if name=='epoch_0000.pt':
        torch.save(dict(schema=tool.CHECKPOINT_SCHEMA,architecture=tool.ARCHITECTURE,
            config={'architecture':tool.ARCHITECTURE},group='rgb',metadata={'completed_epochs':0}),p)
    with pytest.raises(ValueError):tool._checkpoint(p,'f'*64,'cpu')


class FakeModel(torch.nn.Module):
    def __init__(self,method,seen):
        super().__init__();self.method=method;self.seen=seen
    def forward(self,features,memory):
        assert not any(k in features for k in ('gt_member_ids','gt_positions_m','training_equivalent_assignment'))
        self.seen.append((features.get('episode_id'),features.get('member_ids'),memory is None))
        return dict(member_positions=torch.tensor([[1.,2,2],[10.,0,0]]),
                    valid_mask=torch.tensor([True,True]),state='normal-state')


def environment(tmp_path,monkeypatch,method=2):
    side=tmp_path/'annotation';side.mkdir()
    records=[];observations=[];refs=[]
    for ep in ('episode_a','episode_b'):
        features=dict(bbox_xyxy=np.array([[1.,2,3,4],[5,6,7,8]]),
            ray_origins=np.zeros((2,3)),ray_directions=np.array([[1.,0,0],[1,0,0]]),
            ray_depth_bounds=np.array([[.1,100],[.1,100]]),detection_scores=np.ones(2),
            detector_roi_features=np.ones((2,1024)))
        obs=SimpleNamespace(episode_id=ep,frame_idx=0,physical_time_s=0.,features=features,
            source_hashes=dict(normal='a'*64,visual='b'*64),iq=None)
        record=SimpleNamespace(episode_id=ep,frame_idx=0,normal_sha256='a'*64,labels_sha256='c'*64)
        fixed=dict(schema='rvloc.fixed_roi_gt_frame.v1',episode_id=ep,frame_idx=0,physical_time_s=0.,
            normal_sha256='a'*64,labels_sha256='c'*64,visual_sha256='b'*64,detection_count=2,
            bbox_xyxy=features['bbox_xyxy'].tolist(),ray_origins=features['ray_origins'].tolist(),
            ray_directions=features['ray_directions'].tolist(),ray_depth_bounds=features['ray_depth_bounds'].tolist(),
            gt_member_ids=['secret_gt_A','secret_gt_B','secret_gt_C'],gt_positions_m=[[0,0,0],[10,0,0],[20,0,0]],
            training_equivalent_assignment=[dict(detection_index=0,gt_index=0,gt_member_id='secret_gt_A',gt_position_m=[0,0,0]),
                                           dict(detection_index=1,gt_index=1,gt_member_id='secret_gt_B',gt_position_m=[10,0,0])])
        p=side/(ep+'.json');p.write_text(json.dumps(fixed));refs.append(dict(episode_id=ep,frame_idx=0,split='train',path=p.name,sha256=tool.file_sha256(p)))
        observations.append(obs);records.append(record)
    refs.append(dict(episode_id='DEV_NOT_OPENED',frame_idx=0,split='dev',path='DO_NOT_OPEN',sha256='0'*64))
    index=side/'fixed_roi_gt_index.json';index.write_text(json.dumps(dict(schema='rvloc.fixed_roi_gt_index.v1',
        manifest_sha256='e'*64,supervision_only=True,normal_inference_input=False,frames=refs)))
    requests=[]
    def obs_iter(split,**options):
        assert split=='train';requests.append((split,options));return iter(observations)
    manifest=SimpleNamespace(path=tmp_path/'manifest.json',sha256='e'*64,
        frames=lambda split:iter(records) if split=='train' else pytest.fail('Only TRAIN may be requested'),
        observations=obs_iter,splits={'train':('episode_a','episode_b'),'dev':('DEV_NOT_OPENED',)})
    config=SimpleNamespace(path=tmp_path/'config.json',sha256='f'*64,manifest=manifest.path)
    monkeypatch.setattr(tool.ExperimentConfig,'load',lambda _:config)
    monkeypatch.setattr(tool,'load_manifest',lambda _:manifest)
    checkpoint=tmp_path/'epoch_0001.pt';checkpoint.write_bytes(b'synthetic only')
    seen=[]
    def load(path,manifest_sha,device):
        assert manifest_sha==manifest.sha256
        return FakeModel(method,seen),{'calibration_sha256':'calibration'},dict(path=str(checkpoint),sha256=tool.file_sha256(checkpoint),method_id=method,completed_epochs=1,bytes=14)
    monkeypatch.setattr(tool,'_checkpoint',load)
    return config,index,checkpoint,observations,requests,seen


def test_full_train_member_rmse_missing_and_cpu_cache_without_dev_reads(tmp_path,monkeypatch):
    config,index,checkpoint,obs,requests,seen=environment(tmp_path,monkeypatch)
    cache={}
    result=tool.compute(config.path,checkpoint,index,cache=cache,output_dir=tmp_path/'results')
    assert result['position_rmse_3d_m']==pytest.approx(np.sqrt(4.5))
    assert result['frames']==2 and result['matched_members']==4
    assert result['gt_members']==6 and result['no_localization_members']==2
    assert result['all_gt_success']['1m']==pytest.approx(2/6)
    assert result['scope']=='fixed_checkpoint_train_full' and result['online_training_statistic'] is False
    assert not result['read_dev'] and not result['read_test']
    assert requests==[('train',{'modality':'rgb'})]
    result=tool.compute(config.path,checkpoint,index,cache=cache)
    assert len(requests)==1 and result['cost']['public_cpu_input_cache_reused'] is True
    assert Path(tmp_path/'results/metrics.json').is_file()


def test_fusion_uses_prepared_only_and_world_ids_are_observed_not_gt(tmp_path,monkeypatch):
    config,index,checkpoint,obs,requests,seen=environment(tmp_path,monkeypatch,7)
    class Association:
        def step(self,frame,timestamp,boxes,roi,scores):return ['observed_11','observed_29']
    def factory():return Association()
    factory.calibration_sha256='calibration'
    monkeypatch.setattr(tool,'_association_factory',lambda c,m:factory)
    result=tool.compute(config.path,checkpoint,index)
    assert requests==[('train',{'modality':'rgb_radar','prepared_only':True})]
    assert seen==[('episode_a',['observed_11','observed_29'],True),('episode_b',['observed_11','observed_29'],True)]
    assert result['inference_uses_gt_ids'] is False


def test_fixed_bbox_identity_mismatch_fails_closed(tmp_path,monkeypatch):
    config,index,checkpoint,obs,requests,seen=environment(tmp_path,monkeypatch)
    obs[0].features['bbox_xyxy']=obs[0].features['bbox_xyxy'][::-1]
    with pytest.raises(ValueError,match='bbox'):
        tool.compute(config.path,checkpoint,index)
