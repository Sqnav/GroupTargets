"""Production guards exercised with mocks and temporary generated JSON only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from rvloc.config import ExperimentConfig, code_root
from rvloc.evaluation import evaluate, paired_compare
from rvloc.models import load_checkpoint
from rvloc.models.checkpoint import SCHEMA as MODEL_CHECKPOINT_SCHEMA
from rvloc.pipeline import infer
from rvloc.runtime import write_json
from rvloc.training import train


class ProvenanceTests(unittest.TestCase):
    def config(self, directory, **overrides):
        root = Path(directory)
        raw = dict(schema="rvloc.experiment.v1", bandwidth_hz=32e6,
            center_contract="observer_scatter_at_camera_optical_center", manifest=None,
            storage={"output_root": str(root/"outputs"), "cache_root": str(root/"cache")},
            checkpoints={"rgb": str(root/"rgb.pt"), "world_fusion": str(root/"world.pt")},
            evaluation={"primary": "position_rmse_3d_m", "match_max_distance_m": 3.},
            **overrides)
        path = root/"experiment.json"
        path.write_text(json.dumps(raw))
        return ExperimentConfig.load(path)

    def metadata(self):
        return dict(optimizer_updates=4, completed_epochs=1, architecture="single_query_depth_v3",
            manifest_sha256="generated-manifest", center_contract="observer_scatter_at_camera_optical_center",
            observer_position_reference="camera_optical_center", body_navigation_is_model_input=False)

    def manifest(self):
        return SimpleNamespace(sha256="generated-manifest", rf=object(),
            observations=Mock(side_effect=AssertionError("Rejected provenance must not open observations")))

    def test_disk_inference_rejects_synthetic_and_untrained_checkpoints_before_sensor_io(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(temporary)
            for changes in ({"synthetic_only": True}, {"optimizer_updates": 0},
                            {"completed_epochs": 0}, {"optimizer_updates": 0, "completed_epochs": 0}):
                with self.subTest(changes=changes):
                    manifest = self.manifest()
                    metadata = dict(self.metadata(), **changes)
                    with patch("rvloc.models.load_checkpoint", return_value=(object(), {"metadata": metadata})), \
                         patch("rvloc.pipeline.file_sha256", return_value="generated-checkpoint"), \
                         patch("rvloc.pipeline.load_calibration", return_value=None), \
                         patch("rvloc.pipeline.MethodRunner") as runner:
                        with self.assertRaisesRegex(ValueError, "actually trained checkpoint"):
                            infer(config, 2, manifest=manifest)
                    manifest.observations.assert_not_called()
                    runner.assert_not_called()

    def test_world_inference_rejects_different_visual_calibration_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(temporary)
            manifest = self.manifest()
            metadata = dict(self.metadata(), calibration_sha256="TRAIN-frozen-A")
            with patch("rvloc.models.load_checkpoint", return_value=(object(), {"metadata": metadata})), \
                 patch("rvloc.pipeline.file_sha256", return_value="generated-checkpoint"), \
                 patch("rvloc.pipeline.load_calibration", return_value={"_sha256": "TRAIN-frozen-B"}), \
                 patch("rvloc.pipeline.MethodRunner") as runner:
                with self.assertRaisesRegex(ValueError, "different visual association calibration"):
                    infer(config, 7, manifest=manifest)
            manifest.observations.assert_not_called()
            runner.assert_not_called()

    def test_old_two_round_and_direct_versions_cannot_be_loaded_as_single_query_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"synthetic_legacy_checkpoint.pt"
            base = dict(group="single_frame", config={"architecture":"single_query_depth_v3"},
                        state_dict={}, metadata=self.metadata())
            cases = [dict(schema="rvloc_seven_groups_models_v1", architecture="single_query_depth_v3"),
                     dict(schema="rvloc_seven_groups_direct_depth_v2", architecture="direct_depth_v2"),
                     dict(schema=MODEL_CHECKPOINT_SCHEMA, architecture="coarse_fine_v1"),
                     dict(schema=MODEL_CHECKPOINT_SCHEMA)]
            for identity in cases:
                with self.subTest(identity=identity):
                    torch.save(dict(base, **identity), path)
                    with patch("rvloc.models.build_model", side_effect=AssertionError("Rejected version must not construct a model")) as factory:
                        with self.assertRaises(ValueError):
                            load_checkpoint(path, expected_group=3)
                    factory.assert_not_called()

    def test_evaluation_matching_contract_is_frozen_before_labels_are_opened(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(temporary)
            manifest = SimpleNamespace(sha256="generated-manifest", frames=Mock(return_value=[]))
            base = dict(schema="rvloc.predictions.v1", normal_inference=True, labels_opened=False,
                split="dev", manifest_sha256=manifest.sha256, frames=[])
            path = Path(temporary)/"generated_predictions.json"
            contracts = [None,
                dict(matching="same_per_frame_one_to_one_euclidean", match_max_distance_m=5.),
                dict(matching="changed_matching", match_max_distance_m=3.)]
            for contract in contracts:
                with self.subTest(contract=contract):
                    path.write_text(json.dumps(dict(base, evaluation_contract=contract)))
                    with patch("rvloc.evaluation.load_labels") as labels, \
                         patch("rvloc.evaluation.write_json") as writer:
                        with self.assertRaisesRegex(ValueError, "contract frozen before inference"):
                            evaluate(config, path, manifest=manifest)
                    labels.assert_not_called()
                    writer.assert_not_called()

    def test_paired_comparison_rejects_different_matching_distances(self):
        first = dict(schema="rvloc.evaluation.v1", manifest_sha256="generated-manifest",
                     split="dev", summary={"matching_max_distance_m": 3.}, rows=[])
        second = deepcopy(first)
        second["summary"]["matching_max_distance_m"] = 5.
        with self.assertRaisesRegex(ValueError, "different matching distances"):
            paired_compare(first, second)

    def test_pending_default_config_fails_before_loading_data_or_building_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(temporary)
            with patch("rvloc.training.load_manifest") as training_manifest, \
                 patch("rvloc.training.build_model") as factory:
                with self.assertRaisesRegex(ValueError, "ready, explicit data manifest"):
                    train(config, 2)
            training_manifest.assert_not_called()
            factory.assert_not_called()
            with patch("rvloc.pipeline.load_manifest") as inference_manifest:
                with self.assertRaisesRegex(ValueError, "Dataset pending"):
                    infer(config, 2)
            inference_manifest.assert_not_called()
            self.assertFalse(config.output_root.exists())

    def test_artifacts_and_inference_outputs_cannot_be_written_into_code_tree(self):
        # Resolve the live root so this remains valid after staging is renamed.
        forbidden = code_root()/"must-not-create-provenance-test"/"artifact.json"
        with self.assertRaisesRegex(ValueError, "code-only project"):
            write_json(forbidden, {"generated": True})
        self.assertFalse(forbidden.parent.exists())
        with tempfile.TemporaryDirectory() as temporary:
            config = self.config(temporary)
            with patch("rvloc.pipeline.load_manifest") as loader:
                with self.assertRaisesRegex(ValueError, "code-only project"):
                    infer(config, 2, output_dir=forbidden.parent)
            loader.assert_not_called()
        self.assertFalse(forbidden.parent.exists())


if __name__ == "__main__":
    unittest.main()
