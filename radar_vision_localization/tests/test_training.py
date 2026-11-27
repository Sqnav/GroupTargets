"""Tiny synthetic optimizer checks; no actual dataset or pretrained weights."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rvloc.config import ExperimentConfig
from rvloc.models import WorldFusionLocalizer, load_checkpoint, build_model
from rvloc.training import train
from rvloc.models import localization_loss
from test_models import synthetic_features


class SyntheticManifest:
    sha256 = "synthetic-manifest-sha256"

    def __init__(self, episodes=(5, 2)):
        self.episodes = episodes
        self.requests = []

    def observations(self, split, *, modality):
        self.requests.append((split, modality))
        if split != "train":
            raise AssertionError("Trainer may only open TRAIN observations")
        for episode, frames in enumerate(self.episodes):
            for index in range(frames):
                features = synthetic_features(frame=index, episode=f"synthetic-{episode}")
                for key in ("episode_id", "member_ids", "frame_index", "timestamp_s"):
                    features.pop(key)
                yield SimpleNamespace(episode_id=f"synthetic-{episode}", frame_idx=index,
                    physical_time_s=float(index), features=features,
                    source_hashes={"normal": "synthetic", "manifest": self.sha256})


class _PositionsOnlyLabels:
    def __init__(self, observation):
        self.positions = observation.features["ray_origins"] + \
            (20.+observation.frame_idx*.1)*observation.features["ray_directions"]

    @property
    def member_ids(self):
        raise AssertionError("GT member IDs must never be read by neural training")


class SyntheticAssociation:
    def step(self, index, timestamp, boxes, roi, scores):
        return np.asarray([101, None, 303], dtype=object)


class RecordingWorld(WorldFusionLocalizer):
    def __init__(self):
        super().__init__()
        self.boundaries = []

    def forward(self, features, memory=None):
        self.boundaries.append((features["episode_id"], features["frame_index"], memory is None,
            None if memory is None else all(value.latent.grad_fn is None for value in memory.members.values())))
        if any(key in features for key in ("targets", "labels", "positions", "true_depths")):
            raise AssertionError("Training supervision leaked into normal model input")
        output = super().forward(features, memory)
        if "initial_depths" in output or "initial_positions" in output:
            raise AssertionError("A single-query architecture cannot expose an initial/refined prediction pair")
        if output["aux"].get("architecture") != "single_query_depth_v3":
            raise AssertionError("World training must use the single-query architecture")
        return output


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, temporary, **training):
        root = Path(temporary)
        payload = dict(schema="rvloc.experiment.v1", bandwidth_hz=32e6,
            center_contract="observer_scatter_at_camera_optical_center", manifest=None,
            storage={"output_root": str(root/"outputs"), "weights_root": str(root/"weights"),
                     "cache_root": str(root/"cache")},
            model={}, training={"epochs": 1, "device": "cpu", "truncation_frames": 4,
                                "gradient_clip": 5., **training},
            evaluation={"primary": "position_rmse_3d_m"})
        path = root/"experiment.json"
        path.write_text(json.dumps(payload))
        return ExperimentConfig.load(path)

    def test_world_train_uses_visual_ids_truncates_resets_and_saves_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(temporary)
            manifest, model = SyntheticManifest(), RecordingWorld()
            before = model.transition.weight_hh.detach().clone()
            with patch("rvloc.training.load_labels", side_effect=lambda m, o: _PositionsOnlyLabels(o)):
                receipt = train(config, 7, manifest=manifest, model=model,
                                association_factory=SyntheticAssociation)
            self.assertEqual(receipt["optimizer_updates"], 3)
            self.assertEqual(receipt["history"][0]["frames"], 7)
            self.assertEqual(receipt["history"][0]["unassigned_visual_members"], 7)
            self.assertGreater(float((before-model.transition.weight_hh.detach()).abs().sum()), 0.)
            self.assertEqual(model.boundaries[0][2], True)
            self.assertEqual(model.boundaries[4][3], True)
            self.assertEqual(model.boundaries[5][2], True)
            self.assertEqual(manifest.requests, [("train", "rgb_radar")])
            loaded, checkpoint = load_checkpoint(receipt["checkpoint"], expected_group=7)
            metadata = checkpoint["metadata"]
            self.assertEqual(metadata["manifest_sha256"], manifest.sha256)
            self.assertEqual(metadata["config_sha256"], config.sha256)
            self.assertEqual(metadata["center_contract"], config.raw["center_contract"])
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertFalse(receipt["future_rollout_supervision"])
            self.assertEqual(receipt["architecture"], "single_query_depth_v3")
            self.assertEqual(metadata["architecture"], "single_query_depth_v3")
            self.assertEqual(checkpoint["architecture"], "single_query_depth_v3")
            self.assertEqual(loaded.config["depth_refinement_steps"], 0)
            self.assertNotIn("initial_huber", receipt["history"][0]["loss"])
            self.assertIn("visual_prior_nll", receipt["history"][0]["loss"])
            self.assertTrue(Path(receipt["output_dir"], "receipt.json").is_file())
            self.assertTrue(Path(receipt["output_dir"], "history.json").is_file())
            self.assertEqual(list(Path(receipt["output_dir"]).rglob("*.pt")), [])
            self.assertEqual(Path(receipt["weights_dir"]).parent, config.weights_root)
            self.assertEqual(Path(receipt["weights_dir"]).name, Path(receipt["output_dir"]).name)
            self.assertEqual(Path(receipt["checkpoint"]), Path(receipt["weights_dir"])/"model.pt")
            self.assertTrue((Path(receipt["weights_dir"])/"epoch_0001.pt").is_file())
            self.assertFalse((Path(receipt["weights_dir"])/"receipt.json").exists())

    def test_rgb_and_single_frame_have_independent_explicit_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(temporary)
            receipts = []
            for group, modality in ((2, "rgb"), (3, "rgb_radar")):
                manifest = SyntheticManifest((2,))
                model = build_model(group)
                std_head_before = model.core.visual_prior_head[-1].weight[1].detach().clone()
                loss_calls = []
                def single_query_loss(output, targets, **kwargs):
                    self.assertNotIn("initial_depths", output)
                    self.assertNotIn("initial_positions", output)
                    self.assertNotIn("initial_weight", kwargs)
                    value, terms = localization_loss(output, targets, **kwargs)
                    self.assertNotIn("initial_huber", terms)
                    loss_calls.append(tuple(terms))
                    return value, terms
                with patch("rvloc.training.load_labels", side_effect=lambda m, o: _PositionsOnlyLabels(o)), \
                     patch("rvloc.training.localization_loss", side_effect=single_query_loss):
                    receipt = train(config, group, manifest=manifest, model=model)
                self.assertEqual(manifest.requests, [("train", modality)])
                self.assertEqual(receipt["optimizer_updates"], 1)
                self.assertGreater(float((model.core.visual_prior_head[-1].weight[1]-std_head_before).abs().sum()),0.)
                self.assertEqual(len(loss_calls), 2)
                self.assertEqual(set(loss_calls[0]), {"depth_huber", "depth_mse", "visual_prior_nll", "total"})
                self.assertEqual(receipt["architecture"], "single_query_depth_v3")
                self.assertNotIn("initial_huber", receipt["history"][0]["loss"])
                loaded, checkpoint = load_checkpoint(receipt["checkpoint"], expected_group=group)
                self.assertEqual(checkpoint["architecture"], "single_query_depth_v3")
                self.assertEqual(loaded.config["depth_refinement_steps"], 0)
                self.assertFalse(any("refine_" in key or "coarse_head" in key for key in loaded.state_dict()))
                receipts.append(receipt)
            self.assertNotEqual(receipts[0]["output_dir"], receipts[1]["output_dir"])
            self.assertNotEqual(receipts[0]["weights_dir"], receipts[1]["weights_dir"])

    def test_sensor_replay_reuses_verified_observations_without_reopening_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            config=self.config(temporary,epochs=2,cache_observations=True,log_every_updates=1)
            manifest=SyntheticManifest((2,))
            with patch("rvloc.training.load_labels",side_effect=lambda m,o:_PositionsOnlyLabels(o)):
                receipt=train(config,2,manifest=manifest)
            self.assertEqual(manifest.requests,[("train","rgb")])
            self.assertEqual(receipt['optimizer_updates'],2)
            progress=json.loads(Path(receipt['output_dir'],'progress.json').read_text())
            self.assertEqual(progress['status'],'completed')
            self.assertEqual(progress['epoch'],2)
            self.assertEqual(progress['optimizer_updates'],2)
            self.assertEqual(receipt['history'][0]['frames'],receipt['history'][1]['frames'])

    def test_refuses_caching_raw_iq_for_fusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            config=self.config(temporary,cache_observations=True)
            with self.assertRaisesRegex(ValueError,'authenticated prepared-only'):
                train(config,3,manifest=SyntheticManifest((1,)))

    def test_empty_and_unmatched_train_cannot_create_successful_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(temporary)
            with self.assertRaisesRegex(ValueError, "no observations"):
                train(config, 2, manifest=SyntheticManifest(()))
            self.assertFalse(config.output_root.exists())
            with patch("rvloc.training.load_labels", return_value=SimpleNamespace(positions=np.array([[900., 0., 0.]]))):
                with self.assertRaisesRegex(ValueError, "no matched supervision"):
                    train(config, 3, manifest=SyntheticManifest((2,)))
            self.assertEqual(list(config.output_root.rglob("*.pt")), [])
            self.assertEqual(list(config.weights_root.rglob("*.pt")), [])

    def test_custom_run_has_matching_weight_namespace_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            config=self.config(temporary)
            destination=config.output_root/"nested"/"custom-run"
            with patch("rvloc.training.load_labels",side_effect=lambda m,o:_PositionsOnlyLabels(o)):
                receipt=train(config,2,manifest=SyntheticManifest((2,)),output_dir=destination)
            self.assertEqual(receipt['run_id'],'nested/custom-run')
            self.assertEqual(Path(receipt['weights_dir']),config.weights_root/'nested/custom-run')
            # Even if the result path is free, an existing weight run prevents
            # checkpoint replacement and no new result directory is created.
            reserved=config.weights_root/'reserved'
            reserved.mkdir()
            marker=reserved/'model.pt';marker.write_bytes(b'preserve-original')
            with self.assertRaises(FileExistsError):
                train(config,2,manifest=SyntheticManifest((2,)),output_dir=config.output_root/'reserved')
            self.assertEqual(marker.read_bytes(),b'preserve-original')
            self.assertFalse((config.output_root/'reserved').exists())
            with self.assertRaisesRegex(ValueError,'inside configured output_root'):
                train(config,2,manifest=SyntheticManifest((2,)),output_dir=Path(temporary)/'outside')

    def test_configuration_and_method_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(temporary)
            with self.assertRaisesRegex(ValueError, "neural arms"):
                train(config, 6, manifest=SyntheticManifest((2,)))
            bad = self.config(temporary, bptt_steps=0)
            with self.assertRaisesRegex(ValueError, "bptt_steps"):
                train(bad, 2, manifest=SyntheticManifest((2,)))


if __name__ == "__main__":
    unittest.main()
