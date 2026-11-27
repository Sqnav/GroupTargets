"""Seven-route generated-asset smoke; no experiment, fitting or real data.

Temporary random model checkpoints test wiring only and are labelled as such.
All calibration parameters below belong solely to this synthetic fixture.
"""
import copy
from contextlib import contextmanager, ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy import fft

from rvloc.config import METHODS
from rvloc.data import load_manifest, prepare_radar, sha256
from rvloc.models import build_model, load_checkpoint, save_checkpoint
from rvloc.pipeline import MethodRunner
from test_data import synthetic_manifest, write_json


@contextmanager
def single_query_contract(expected_calls=None):
    """Count the one vectorized paired query and forbid old two-round D."""
    import rvloc.models.single_query as queries
    names = (
        "roi_radar_study.model.StudyDepthLocalizer.__init__",
        "roi_radar_study.model.StudyDepthLocalizer.forward",
        "roi_radar_study.model.StudyDepthLocalizer.forward_batch",
    )
    with ExitStack() as stack:
        for name in names:
            stack.enter_context(patch(name, side_effect=AssertionError("Legacy two-round refinement path executed: " + name)))
        sampler=stack.enter_context(patch("rvloc.models.single_query.sample_paired_spectra", wraps=queries.sample_paired_spectra))
        yield sampler
        if expected_calls is not None:
            assert sampler.call_count==expected_calls


def assert_single_query_model(model):
    assert model.config["architecture"] == "single_query_depth_v3"
    assert model.config["depth_refinement_steps"] == 0
    forbidden = ("refine_", "coarse_head", "global_head", "depth_fine_offsets")
    assert not [name for name in model.state_dict() if any(part in name for part in forbidden)]
    assert (getattr(model.core,"local_query",None) is not None) is (not model.visual)


@pytest.fixture
def sequence(tmp_path):
    path, manifest, normal = synthetic_manifest(tmp_path)
    data = Path(manifest["data_root"])
    # Invert an exact native range/spatial FFT cell; this is not a simulator.
    profile = np.zeros((128, 16, 16), np.complex64)
    profile[20, :, :] = np.sqrt(128)
    frequency = fft.fftshift(fft.fft(profile, axis=0, norm="ortho"), axes=0)
    np.save(data / "iq.npy", np.broadcast_to(frequency, (40, 128, 16, 16)).copy())
    normal["iq"]["sha256"] = sha256(data / "iq.npy")
    parsed = load_manifest(path)
    prepare_radar(data / "iq.npy", data / "radar.npz", rf=parsed.rf, angle_bins=9, spatial_oversampling=1)
    normal["radar"] = dict(path="radar.npz", sha256=sha256(data / "radar.npz"))
    frames = []
    for index in range(4):
        row = copy.deepcopy(normal)
        row.update(frame_idx=index, physical_time_s=10. + index)
        row["navigation"] = dict(physical_time_s=[10. + index], position_m=[[5, 0, 1]], velocity_mps=[[0, 0, 0]])
        normal_name = f"normal_{index}.json"
        frames.append(dict(frame_idx=index, physical_time_s=10. + index,
            normal_path=normal_name, normal_sha256=write_json(data / normal_name, row),
            labels_path=f"labels_{index}_deliberately_absent.json", labels_sha256="0" * 64))
    manifest["episodes"][0]["frames"] = frames
    write_json(path, manifest)
    return load_manifest(path), tmp_path


def synthetic_calibration():
    settings = dict(acceleration_variance=.1, measurement_variance=1., initial_velocity_variance=10.,
                    measurement_covariance=np.eye(3).tolist())
    return dict(visual_params=dict(appearance_weight=.05),
                radar_association=dict(position_gate_m=30.),
                kalman={key: copy.deepcopy(settings) for key in ("fft", "rgb", "single_frame")})


@pytest.mark.parametrize("method_id", range(1, 8))
def test_all_seven_routes_use_sensor_only_causal_pipeline(sequence, method_id):
    manifest, temporary = sequence
    torch.set_num_threads(1)
    torch.manual_seed(14)
    description = METHODS[method_id]
    config = SimpleNamespace(raw=dict(radar=dict(spatial_oversampling=1)))
    model = None
    if description["model"] is not None:
        with single_query_contract(0):
            model = build_model(description["model"])
        assert_single_query_model(model)
        checkpoint = temporary / f"synthetic_only_{description['model']}.pt"
        save_checkpoint(model, checkpoint, metadata=dict(synthetic_only=True, manifest_sha256=manifest.sha256,
                        center_contract="observer_scatter_at_camera_optical_center"))
        with single_query_contract(0):
            model, metadata = load_checkpoint(checkpoint, expected_group=description["model"])
        assert_single_query_model(model)
        assert metadata["architecture"] == "single_query_depth_v3"
        assert metadata["metadata"]["synthetic_only"] is True
    runner = MethodRunner(method_id, config, manifest.rf, model=model, calibration=synthetic_calibration())
    original = type(manifest)._read
    opened = []
    def guarded(self, relative, expected):
        assert not relative.startswith("labels"), "Normal inference opened supervision"
        if description["input"] == "rgb":
            assert relative not in ("iq.npy", "radar.npz"), "RGB opened a radar asset"
        if description["input"] == "radar":
            assert relative != "visual.npz", "FFT-only opened visual features"
        opened.append(relative)
        return original(self, relative, expected)
    model_outputs = []
    hook = None if model is None else model.register_forward_hook(lambda module, inputs, output: model_outputs.append(output))
    try:
        with patch.object(type(manifest), "_read", guarded), single_query_contract() as sampler:
            predictions=[]
            for obs in manifest.observations("train", modality=description["input"]):
                sampler.reset_mock()
                predictions.append(runner.step(obs))
                assert sampler.call_count==(1 if method_id in (3,6,7) else 0)
    finally:
        if hook is not None:
            hook.remove()
    assert len(predictions) == 4
    assert [row["physical_time_s"] for row in predictions] == [10., 11., 12., 13.]
    for output,persistable in zip(model_outputs,predictions):
        assert output["aux"]["architecture"] == "single_query_depth_v3"
        assert "initial_depths" not in output and "initial_positions" not in output
        assert output["sampling_diagnostics"]["query_rounds"] == (0 if model.visual else 1)
        assert not output["sampling_diagnostics"].get("stage_depths_m",[])
        assert output["visual_depth_mean_m"].shape==output["depths"].shape
        assert output["visual_depth_std_m"].shape==output["depths"].shape
        assert bool(torch.isfinite(output["visual_depth_std_m"]).all())
        assert bool((output["visual_depth_std_m"][output["valid_mask"]]>0).all())
        for field in ("visual_depth_mean_m","visual_depth_std_m"):
            assert field in persistable
            np.testing.assert_allclose(persistable[field],output[field].detach().cpu().numpy(),atol=0.,rtol=0.)
        if model.visual:
            torch.testing.assert_close(output["depths"],output["visual_depth_mean_m"],atol=0.,rtol=0.)
    assert all(len(row["positions"]) == 1 and row["valid"] == [True] for row in predictions)
    assert all(np.isfinite(row["positions"]).all() for row in predictions)
    assert all("offline-only-id" not in str(row) for row in predictions)
    if method_id in (4, 5, 6):
        assert predictions[-1]["velocities"] is not None
    if method_id == 7:
        assert runner.memory.episode_id == "generated_episode"
    if description["input"] != "radar":
        for row in predictions:
            # Base model outputs retain the exact calibrated optical ray.
            np.testing.assert_allclose(np.asarray(row["raw_positions"])[0],
                np.array([5., 0., 2.]) + row["depths"][0] * np.array([1., 0., 0.]), atol=1e-5)
    assert "center.json" in opened


def test_runner_rejects_gaps_and_resets_at_episode_boundaries(sequence):
    from dataclasses import replace
    manifest, _ = sequence
    observations = list(manifest.observations("train", modality="radar"))
    runner = MethodRunner(4, SimpleNamespace(raw=dict(radar=dict(spatial_oversampling=1))),
                          manifest.rf, calibration=synthetic_calibration())
    first = runner.step(observations[0])
    with pytest.raises(ValueError, match="continuous"):
        runner.step(observations[2])
    runner.step(observations[1])
    restarted = runner.step(replace(observations[0], episode_id="new_generated_episode", physical_time_s=0.))
    assert restarted["member_ids"] == first["member_ids"]
    assert runner.last_time == 0.


def test_world_history_changes_final_depth_but_not_visual_prior_or_query(sequence):
    manifest, _ = sequence
    torch.set_num_threads(1)
    torch.manual_seed(37)
    model = build_model(7)
    with torch.no_grad():
        model.memory_bridge.weight.normal_(std=.2)
        model.core.depth_head[-1].weight.copy_(torch.linspace(-.1, .1, model.core.hidden_dim)[None])
        model.core.depth_head[-1].bias.zero_()
    config = SimpleNamespace(raw=dict(radar=dict(spatial_oversampling=1)))
    with_history = MethodRunner(7, config, manifest.rf, model=copy.deepcopy(model), calibration=synthetic_calibration())
    without_history = MethodRunner(7, config, manifest.rf, model=copy.deepcopy(model), calibration=synthetic_calibration())
    history_outputs=[];birth_outputs=[]
    history_hook=with_history.model.register_forward_hook(lambda module,inputs,output:history_outputs.append(output))
    birth_hook=without_history.model.register_forward_hook(lambda module,inputs,output:birth_outputs.append(output))
    try:
        with single_query_contract(3):
            observations = list(manifest.observations("train", modality="rgb_radar"))
            with_history.step(observations[0])
            current = with_history.step(observations[1])
            birth = without_history.step(observations[1])
    finally:
        history_hook.remove();birth_hook.remove()
    assert current["member_ids"] == birth["member_ids"]
    previous,current_birth=history_outputs[-1],birth_outputs[-1]
    for field in ("visual_depth_mean_m","visual_depth_std_m"):
        torch.testing.assert_close(previous[field],current_birth[field],atol=0.,rtol=0.)
    torch.testing.assert_close(previous["sampling_diagnostics"]["candidate_depths_m"],
        current_birth["sampling_diagnostics"]["candidate_depths_m"],atol=0.,rtol=0.)
    assert np.max(np.abs(np.asarray(current["depths"])-np.asarray(birth["depths"]))) > 1e-6
    for result in (current, birth):
        np.testing.assert_allclose(np.asarray(result["raw_positions"])[0],
            np.array([5., 0., 2.]) + result["depths"][0] * np.array([1., 0., 0.]), atol=1e-5)
