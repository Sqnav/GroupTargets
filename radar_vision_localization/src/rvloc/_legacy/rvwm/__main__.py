"""Single-frame radar/vision fusion localization command-line entrypoint."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sanitize(value):
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [sanitize(item) for item in value]
    return value


def output_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Write outputs inside ' + str(ROOT))
    return path


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(value), ensure_ascii=False,
                              indent=2, allow_nan=False) + '\n')


def main(argv=None):
    from .group_cli import main as group_main
    return group_main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
