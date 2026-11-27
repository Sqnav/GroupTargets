"""Recorded capture phases bound to the detector's frozen source inventory.

The current dataset contains approach frames only. Phase filtering never
changes its frozen family split or silently reuses a different detector cache.
"""
SUPPORTED_PHASES = ('approach', 'track')


def validate_phases(values):
    if (not isinstance(values, (list, tuple)) or not values
            or any(not isinstance(value, str) or value not in SUPPORTED_PHASES for value in values)
            or len(values) != len(set(values))):
        raise ValueError('Explicit nonempty unique phases must be approach and/or track')
    return tuple(value for value in SUPPORTED_PHASES if value in values)


def configured_phases(config, purpose='training'):
    if purpose not in ('training', 'inference'):
        raise ValueError('Unknown frame-selection purpose')
    plural = config.get(purpose + '_phases')
    if plural is None and purpose == 'inference':
        plural = config.get('training_phases')
    if plural is not None:
        return validate_phases(plural)
    if config.get(purpose + '_phase', 'track') != 'track':
        raise ValueError('Legacy ' + purpose + '_phase=track is required; use explicit ' + purpose + '_phases for approach support')
    return ('track',)


def phase_label(phases):
    return '+'.join(validate_phases(phases))


def required_phase(phases):
    values = validate_phases(phases)
    return values[0] if len(values) == 1 else values


def selected_frame_indices(dataset, episode, phases, count=0):
    if count < 0:
        raise ValueError('Frame count must be nonnegative; 0 selects every eligible frame')
    phases = validate_phases(phases)
    indices = sorted({index for phase in phases for index in dataset.frames_in_phase(episode, phase)})
    if count and indices:
        import numpy as np
        offsets = np.linspace(0, len(indices)-1, min(count, len(indices)), dtype=int)
        indices = [indices[i] for i in offsets]
    return tuple(indices)


def inventory_phases(manifest):
    request = manifest.get('request', {})
    return validate_phases(request.get('selected_phases', [request.get('phase', 'track')]))
