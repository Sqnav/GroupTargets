"""Strict replay of frozen DINOv3 detections at their bbox centres."""
from __future__ import annotations

from pathlib import Path

from .data import sha256
from .phase_selection import configured_phases
from .track_detection_store import TrackDetectionStore
from .vision import RGBFrontend, frontend_metadata, validate_visual_configuration


def _store(config, split, detector_identity=None):
    from .detector_contract import detector_contract
    validate_visual_configuration(config)
    identity = detector_identity or detector_contract(config)
    return TrackDetectionStore(config['track_detection_dir'], dataset_root=config['dataset_root'],
        split_manifest=config['visual_split_manifest'], weights_sha256=identity['weights_sha256'],
        confidence=identity['confidence'], imgsz=None, split=split, phases=configured_phases(config),
        backend='dinov3_fpn', detector_contract=identity)


class ExportedRGBFrontend(RGBFrontend):
    """Hash-verified bbox export; missing frames fail without online fallback."""
    def __init__(self, config, *, split=None, keypoint_checkpoint=None, device='cpu'):
        from .detector_contract import detector_contract
        validate_visual_configuration(config, keypoint_checkpoint)
        self.config = dict(config)
        self.device = str(device)
        self.backend = 'dinov3_fpn'
        identity = detector_contract(config)
        path = Path(identity['weights'])
        if sha256(path) != identity['weights_sha256']:
            raise ValueError('Export frontend DINO weights changed')
        self.detector_contract = identity
        self.track_store = _store(config, split or config.get('track_detection_split', 'train'), identity)
        self.metadata = frontend_metadata(config, identity)
        self.metadata.update(track_detection_dir=str(self.track_store.directory),
            track_detection_split=self.track_store.split,
            track_detection_manifest_sha256=self.track_store.manifest_sha256,
            detector_execution='strict SHA-verified frozen RGB export; missing frames fail, no online fallback',
            replay_adapter_sha256=sha256(__file__))
        self.keypoint = None
        self.cache_hits = 0
        self.cache_misses = 0

    def _raw_detect(self, observations):
        raise RuntimeError('Export-only frontend: missing verified selected-phase frame or wrong detector split')

    def close(self):
        pass
