"""Atomic JSON output used by the single-frame localization pipeline."""
from pathlib import Path
import json
import tempfile


def write_json(path, value):
    from .__main__ import sanitize
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        json.dump(sanitize(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        temporary = Path(stream.name)
    temporary.replace(path)
