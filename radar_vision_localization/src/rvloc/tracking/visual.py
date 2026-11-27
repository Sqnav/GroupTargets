"""Thin visual-only adapter around the frozen audited visual-gated tracker."""
from __future__ import annotations

import numpy as np


class VisualAssociator:
    """Return estimated member IDs aligned to all current image observations.

    The five-argument input boundary cannot accept 3D, IQ, camera pose, radar
    evidence, or target annotations. G5/G6/G7 construct this same class with
    identical TRAIN-frozen parameters. None means the visual gate did not
    assign a persistent member; the observation is still retained.
    """

    def __init__(self, params=None, roster_config=None):
        from .._legacy.visual_association.tracker import VisualAssociationTracker
        self._tracker = VisualAssociationTracker(params=params, variant="visual_gated",
                                                 roster_config=roster_config)
        self.last_result = None

    def reset(self):
        self._tracker.reset()
        self.last_result = None

    @property
    def tracker(self):
        return self._tracker

    def step(self, frame_idx, time, boxes, roi=None, scores=None):
        if roi is None:
            raise ValueError("Visual-gated association requires the visual ROI descriptors")
        if scores is None:
            scores = np.ones(len(boxes), dtype=float)
        frame = dict(frame_idx=frame_idx, time=time, boxes=boxes, roi=roi, scores=scores)
        self.last_result = self._tracker.step(frame)
        identities = np.full(len(boxes), None, dtype=object)
        for row in self.last_result["outputs"]:
            identities[row["detection_index"]] = row["track_id"]
        return identities
