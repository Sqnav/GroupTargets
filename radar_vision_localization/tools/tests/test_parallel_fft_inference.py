import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np

p=Path(__file__).resolve().parents[1]/'parallel_fft_inference.py'
spec=importlib.util.spec_from_file_location('parallel_fft_test',p);tool=importlib.util.module_from_spec(spec);spec.loader.exec_module(tool)


def test_threaded_inference_preserves_frame_order_and_predictions(tmp_path,monkeypatch):
    config=SimpleNamespace(manifest='unused',raw={'radar':{},'evaluation':{}},sha256='config')
    monkeypatch.setattr(tool.ExperimentConfig,'load',lambda p:config)
    monkeypatch.setattr(tool,'_verify_center',lambda m:None)
    records=[SimpleNamespace(frame_idx=i) for i in range(9)]
    manifest=SimpleNamespace(sha256='manifest',frames=lambda split:iter(records))
    def fake(m,frame,detector):
        return {'episode_id':'ep','frame_idx':frame.frame_idx,'runtime_s':.01,'positions':[[frame.frame_idx,0,0]]}
    monkeypatch.setattr(tool,'detect_frame',fake)
    result=tool.infer_fft('config','train',tmp_path/'result',workers=3,manifest=manifest,detector=object())
    import json
    value=json.loads(Path(result['predictions']).read_text())
    assert [r['frame_idx'] for r in value['frames']]==list(range(9))
    assert value['labels_opened'] is False and value['normal_inference'] is True
    assert value['cost']['parallel_workers']==3 and value['cost']['serial_latency_measured'] is False


def test_exact_detector_is_called_with_current_scatter_reference(monkeypatch):
    obs=SimpleNamespace(episode_id='ep',frame_idx=2,physical_time_s=2.,iq=np.array([1+1j]),
        calibration=object(),observer_scatter_reference_position=np.array([1.,2.,3.]),source_hashes={'iq':'sha'})
    monkeypatch.setattr(tool,'load_observation',lambda m,f,**kw:obs)
    class Detector:
        def detect(self,iq,station,observer):
            assert iq is obs.iq and station is obs.calibration
            np.testing.assert_array_equal(observer,[1,2,3])
            return SimpleDetections()
    class SimpleDetections:
        positions=np.array([[4.,5.,6.]]);valid=np.array([True]);scores=np.array([.9]);quality=np.array([20.]);observer_near=np.array([False]);metadata={}
        def __len__(self):return 1
    result=tool.detect_frame(None,None,Detector())
    assert result['positions']==[[4,5,6]] and result['raw_positions']==result['positions']
    assert result['member_ids']==[None]
