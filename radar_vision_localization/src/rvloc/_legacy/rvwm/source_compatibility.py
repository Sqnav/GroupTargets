"""Exact current-source binding for newly trained single-frame checkpoints."""
from pathlib import Path

from .data import sha256


def verify_inference_sources(checkpoint, metadata):
    from .group_model import GROUP_SCHEMA
    from .group_training import _source_hashes
    if metadata.get('group_schema') != GROUP_SCHEMA:
        raise ValueError('Only current single-frame checkpoints are supported')
    recorded = metadata.get('feature_source_sha256')
    current = _source_hashes()
    if not isinstance(recorded, dict) or set(recorded) != set(current):
        raise ValueError('Checkpoint feature/model source changed or source provenance is incomplete')
    for name, expected in current.items():
        if recorded[name] != expected:
            raise ValueError('Feature/model source changed after training: ' + name)
    return dict(status='original_training_sources',
                checkpoint_sha256=sha256(Path(checkpoint)),
                compatible_changed_sources=[], receipt_sha256=None)
