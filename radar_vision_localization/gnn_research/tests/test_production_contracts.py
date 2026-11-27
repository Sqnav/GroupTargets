"""Independent production-boundary audit with generated data only."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

from gnn_research.model import build_graph_model,forward_batch,pack_features
from gnn_research.train import save_graph_checkpoint,load_graph_checkpoint,base
from rvloc.contracts import RFMetadata
from gnn_research.evaluate import _row_diagnostic

fixture_path=Path(__file__).resolve().parents[2]/'tests/test_models.py'
spec=importlib.util.spec_from_file_location('production_graph_fixture',fixture_path)
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


@pytest.mark.parametrize('variant',['shared','isolated','shuffled'])
def test_real_checkpoint_preserves_explicit_rf_and_predictions(tmp_path,variant):
    torch.set_num_threads(1);torch.manual_seed(117)
    rf=RFMetadata.from_mapping(dict(carrier_hz=4.9e9,bandwidth_hz=32e6,
        pulse_interval_s=.0025,frame_interval_s=1.,frequency_grid='centered_baseband_uniform',shape=[40,128,16,16],spacing_y_m=.0273,spacing_z_m=.0381,
        frequencies_hz=(np.arange(128)-64)*32e6/128)).as_dict()
    assert isinstance(rf['frequencies_hz'][0],np.float64)
    model=build_graph_model(variant=variant,rf=rf).eval()
    features=[fixture.synthetic_features(n=3),fixture.synthetic_features(n=0)]
    packed=pack_features(features)
    with torch.inference_mode():before=forward_batch(model,packed)
    path=tmp_path/(variant+'.pt')
    rng=base._rng_state('cpu')
    save_graph_checkpoint(model,path,variant=variant,metadata={'optimizer_updates':2,'rng_state':rng})
    loaded,payload,_=load_graph_checkpoint(path,expected_variant=variant)
    assert payload['config']['graph_rf']==rf
    assert isinstance(payload['config']['graph_rf']['frequencies_hz'][0],float)
    assert isinstance(payload['metadata']['rng_state']['python'],tuple)
    assert payload['metadata']['rng_state']['python']==rng['python']
    torch.testing.assert_close(payload['metadata']['rng_state']['torch_cpu'],rng['torch_cpu'],rtol=0,atol=0)
    base._restore_rng(payload['metadata']['rng_state'],'cpu')
    assert loaded.core.local_query.scales==model.core.local_query.scales
    assert loaded.core.local_query.scales['spacing_y_m']==.0273
    assert loaded.core.local_query.scales['spacing_z_m']==.0381
    with torch.inference_mode():after=forward_batch(loaded,packed)
    for field in ('depths','member_positions','valid_mask','visual_depth_mean_m','visual_depth_std_m'):
        torch.testing.assert_close(before[field],after[field],rtol=0,atol=0)
    assert not after['valid_mask'][1].any()
    assert after['graph_diagnostics']['source_batch_indices'].tolist()==[0]


def test_padding_frames_preserve_fixed_detection_row_cardinality():
    torch.set_num_threads(1);model=build_graph_model().eval()
    data=[fixture.synthetic_features(n=0),fixture.synthetic_features(n=1),fixture.synthetic_features(n=3)]
    with torch.inference_mode():out=forward_batch(model,pack_features(data))
    assert out['counts']==(0,1,3)
    for row,n in enumerate(out['counts']):
        assert out['member_positions'][row,:n].shape==(n,3)
        assert out['valid_mask'][row,:n].shape==(n,)
        assert np.isfinite(out['member_positions'][row,:n].numpy()).all()
    # E-node diagnostic arrays are never mistaken for packed frame arrays.
    diagnostic={key:(value.detach().numpy() if hasattr(value,'detach') else value)
                for key,value in out['graph_diagnostics'].items()}
    assert _row_diagnostic(diagnostic,0,3)['frame_had_radar_inputs'] is False
    assert _row_diagnostic(diagnostic,1,3)['frame_had_radar_inputs'] is True
    assert _row_diagnostic(diagnostic,2,3)['frame_had_radar_inputs'] is True
