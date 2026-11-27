"""External experiment artifacts and reproducible inference boundaries."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import uuid

import numpy as np

from .config import code_root, external_path


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [jsonable(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return jsonable(value.tolist())
    if isinstance(value, np.generic): return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, value):
    path = external_path(path, field='artifact')
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.writing-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()


def source_identity():
    digest = hashlib.sha256()
    files = sorted(code_root().joinpath('src').rglob('*.py'))
    for path in files:
        digest.update(str(path.relative_to(code_root())).encode())
        digest.update(bytes.fromhex(file_sha256(path)))
    return dict(sha256=digest.hexdigest(), python_files=len(files),
                python=sys.version.split()[0], platform=platform.platform())


def new_run(config, name):
    # No date- or method-only name that could overwrite a previous experiment.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = config.output_root / (stamp+'_'+name+'_'+uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    return output


def load_calibration(config, manifest, *, required=False):
    value = config.raw.get('calibration_artifact')
    if value is None:
        if required:
            raise ValueError('TRAIN calibration is required for temporal association/filtering; configure calibration_artifact')
        return None
    path = external_path(value, field='calibration_artifact', base=config.path.parent)
    result = json.loads(path.read_text())
    if result.get('schema') != 'rvloc.calibration.v1' or result.get('split') != 'train':
        raise ValueError('Expected TRAIN-only rvloc.calibration.v1')
    if result.get('manifest_sha256') != manifest.sha256:
        raise ValueError('Calibration belongs to a different data manifest')
    if result.get('status') != 'frozen':
        raise ValueError('Calibration has not been frozen')
    from .calibration import validate_calibration_artifact
    validate_calibration_artifact(result, manifest_sha256=manifest.sha256,
                                  train_episode_ids=manifest.splits.get('train', ()))
    result['_path'] = str(path)
    result['_sha256'] = file_sha256(path)
    return result
