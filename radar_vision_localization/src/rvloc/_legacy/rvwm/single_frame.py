"""Strict current-frame contract for the active RA/RE localization pipeline."""
from __future__ import annotations

import numpy as np


def require_single_frame(config):
    """Accept only the current single-frame localization architecture."""
    if config.get('architecture_schema', 'observer_referenced_ra_re_continuous_v1') != 'observer_referenced_ra_re_continuous_v1':
        raise ValueError('Only the single-frame RA/RE localization architecture is supported')
    enabled = config.get('group_temporal_enabled', False)
    if not isinstance(enabled, (bool, np.bool_)) or enabled:
        raise ValueError('Only single-frame inputs are supported; temporal mode has been removed')
    window = config.get('window', 1)
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window != 1:
        raise ValueError('Single-frame inputs require window=1')
    if any(key in config for key in ('group_temporal_layers', 'group_temporal_heads')):
        raise ValueError('Temporal architecture options have been removed')
    return 1


def validate_single_frame_features(features):
    """Reject historical observations rather than silently discard their data.

    Empty legacy history fields are accepted for old single-frame archives.
    """
    past = features.get('history', ())
    if not isinstance(past, (tuple, list)) or len(past):
        raise ValueError('Only single-frame features are accepted; history input has been removed')
    if any(key in features for key in ('temporal_schema', 'history_matches', 'history_dt', 'sensor_context')):
        raise ValueError('Only single-frame features are accepted; temporal context has been removed')
    return features


def validate_model_features(features, config):
    """Validate the active single-frame model boundary."""
    require_single_frame(config)
    return validate_single_frame_features(features)
