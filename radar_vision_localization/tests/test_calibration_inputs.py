"""Offline calibration collection checks using generated assets only."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import torch

import pytest

from rvloc.calibration import CalibrationError
from rvloc.config import ExperimentConfig
from rvloc.models import build_model
from rvloc.pipeline import infer
from rvloc.calibration_inputs import collect_calibration_inputs
from rvloc.data import load_manifest
from test_data import write_json
from test_pipeline import sequence  # noqa: F401 - generated causal multi-frame fixture


def offline_fixture(sequence):
    manifest, root = sequence
    raw = json.loads(manifest.path.read_text())
    data = Path(raw["data_root"])
    for frame in raw["episodes"][0]["frames"]:
        labels = dict(schema="rvloc.labels.v1", episode_id="generated_episode", frame_idx=frame["frame_idx"],
                      physical_time_s=frame["physical_time_s"], positions_m=[[25, 0, 2]], member_ids=["offline-only-id"])
        frame["labels_sha256"] = write_json(data / frame["labels_path"], labels)
    write_json(manifest.path, raw)
    manifest = load_manifest(manifest.path)
    paths = {}
    for method, source in ((1, "fft"), (2, "rgb"), (3, "single_frame")):
        # Deliberately inaccurate RGB depth must not affect visual GT pairing.
        point = [9999, 0, 2] if method == 2 else [26, 0, 2]
        value = dict(schema="rvloc.predictions.v1", method_id=method, split="train", manifest_sha256=manifest.sha256,
            normal_inference=True, labels_opened=False, coordinate_frame="base_local",
            neural_architecture=None if method == 1 else "single_query_depth_v3",
            evaluation_contract=dict(matching="same_per_frame_one_to_one_euclidean", match_max_distance_m=None),
            frames=[dict(episode_id=row.episode_id, frame_idx=row.frame_idx, physical_time_s=row.physical_time_s,
                         positions=[point], valid=[True],
                         architecture="fft_power_peaks_v1" if method == 1 else "single_query_depth_v3",
                         visual_depth_mean_m=None if method==1 else ([9994.] if method==2 else [21.]),
                         visual_depth_std_m=None if method==1 else [.5])
                    for row in manifest.frames("train")])
        if method==1:
            for row in value["frames"]:
                row.pop("visual_depth_mean_m");row.pop("visual_depth_std_m")
        path = root / f"persisted_{source}.json"
        write_json(path, value)
        paths[source] = path
    return manifest, paths


def test_collects_separate_train_supervision_and_pure_pixel_pairs(sequence):
    manifest, paths = offline_fixture(sequence)
    results = collect_calibration_inputs(manifest, paths)
    frames = len(list(manifest.frames("train")))
    assert len(results["residual_records"]) == 3 * frames
    assert len(results["trajectory_records"]) == frames
    assert len(results["visual_pair_records"]) == frames - 2
    assert {row["source"] for row in results["residual_records"]} == {"fft", "rgb", "single_frame"}
    assert results["visual_pair_records"][0]["center_residual"] == 0.
    assert results["visual_pair_records"][0]["size_residual"] == 0.
    assert results["receipt"]["models_executed"] is False
    assert results["receipt"]["residual_counts"] == dict(fft=frames, rgb=frames, single_frame=frames)
    assert all(row["split"] == "train" for name in ("residual_records", "trajectory_records", "visual_pair_records") for row in results[name])
    assert all("member_ids" not in obs.features for obs in manifest.observations("train"))


@pytest.mark.parametrize("mutation", ["dev", "labels_opened", "missing_frame", "hash"])
def test_rejects_invalid_predictions_before_opening_labels(sequence, mutation):
    manifest, paths = offline_fixture(sequence)
    value = json.loads(paths["rgb"].read_text())
    if mutation == "dev":
        value["split"] = "dev"
    elif mutation == "labels_opened":
        value["labels_opened"] = True
    elif mutation == "missing_frame":
        value["frames"].pop()
    else:
        value["manifest_sha256"] = "0" * 64
    write_json(paths["rgb"], value)
    with patch("rvloc.calibration_inputs.load_labels", side_effect=AssertionError("Label read before preflight")):
        with pytest.raises(CalibrationError):
            collect_calibration_inputs(manifest, paths)


def test_insufficient_pixel_pairs_are_explicit_and_do_not_use_rgb_depth(sequence):
    manifest, paths = offline_fixture(sequence)
    raw = json.loads(manifest.path.read_text())
    data = Path(raw["data_root"])
    # All points remain available to residual matching; none project into box.
    for frame in raw["episodes"][0]["frames"]:
        labels = json.loads((data / frame["labels_path"]).read_text())
        labels["positions_m"] = [[25, 20, 2]]
        frame["labels_sha256"] = write_json(data / frame["labels_path"], labels)
    write_json(manifest.path, raw)
    manifest = load_manifest(manifest.path)
    for path in paths.values():
        value = json.loads(path.read_text())
        value["manifest_sha256"] = manifest.sha256
        write_json(path, value)
    with pytest.raises(CalibrationError, match="three consecutive"):
        collect_calibration_inputs(manifest, paths)


@pytest.mark.parametrize("mutation", ["legacy_top_architecture", "direct_v2_top_architecture", "missing_top_architecture", "legacy_frame_architecture", "changed_matching_contract"])
def test_current_architecture_and_matching_are_checked_before_labels(sequence, mutation):
    manifest, paths = offline_fixture(sequence)
    value = json.loads(paths["rgb"].read_text())
    if mutation == "legacy_top_architecture":
        value["neural_architecture"] = "coarse_fine_v1"
    elif mutation == "direct_v2_top_architecture":
        value["neural_architecture"] = "direct_depth_v2"
    elif mutation == "missing_top_architecture":
        value.pop("neural_architecture")
    elif mutation == "legacy_frame_architecture":
        value["frames"][0]["architecture"] = "coarse_fine_v1"
    else:
        value["evaluation_contract"]["match_max_distance_m"] = 3.
    write_json(paths["rgb"], value)
    with patch("rvloc.calibration_inputs.load_labels", side_effect=AssertionError("Version/matching preflight opened supervision")):
        with pytest.raises(ValueError):
            collect_calibration_inputs(manifest, paths)


def test_real_inference_exports_are_accepted_by_offline_collector(sequence):
    """Exercise the actual JSON handoff with generated assets/random weights."""
    manifest, placeholders = offline_fixture(sequence)
    root = manifest.path.parent
    raw = dict(schema="rvloc.experiment.v1", bandwidth_hz=32e6,
        center_contract="observer_scatter_at_camera_optical_center", manifest=str(manifest.path),
        calibration_artifact=None, checkpoints={"rgb":None,"single_frame":None,"world_fusion":None},
        storage=dict(output_root=str(root/"exports"),weights_root=str(root/"weights"),cache_root=str(root/"cache")),
        radar=dict(spatial_oversampling=1),model={},
        evaluation=dict(primary="position_rmse_3d_m", match_max_distance_m=None))
    config_path = root/"generated_experiment.json"
    write_json(config_path,raw)
    config=ExperimentConfig.load(config_path)
    torch.set_num_threads(1)
    torch.manual_seed(83)
    exported={}
    for method,source in ((1,"fft"),(2,"rgb"),(3,"single_frame")):
        model=None if method==1 else build_model(method)
        # Injected random models exist only in this synthetic test. The CLI's
        # trained-checkpoint guards are separately tested in provenance.
        with patch("rvloc.data.load_labels",side_effect=AssertionError("Inference opened GT")), \
             patch("rvloc.evaluation.load_labels",side_effect=AssertionError("Inference opened GT")):
            run=infer(config,method,split="train",manifest=manifest,model=model,output_dir=root/"exports"/source)
        exported[source]=Path(run["predictions"])
        value=json.loads(exported[source].read_text())
        assert value["neural_architecture"]==(None if method==1 else "single_query_depth_v3")
        assert value["labels_opened"] is False
        if method!=1:
            assert all("visual_depth_mean_m" in row and "visual_depth_std_m" in row for row in value["frames"])
    result=collect_calibration_inputs(manifest,exported)
    assert len(result["residual_records"])==3*len(list(manifest.frames("train")))
    assert len(result["visual_pair_records"])==len(list(manifest.frames("train")))-2
    for source,path in exported.items():
        assert result["receipt"]["prediction_files"][source]["path"]==str(path)
