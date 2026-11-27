"""Activate only the vendored, code-only numerical/model implementation."""
from pathlib import Path
import sys


def activate():
    path=Path(__file__).resolve().parent/'_legacy'
    for package in ('roi_radar_study','d_evidence_speed','d_evidence_study','depth_localization','precision_study','rvwm','visual_association'):
        loaded=sys.modules.get(package)
        source=getattr(loaded,'__file__',None)
        if source and not Path(source).resolve().is_relative_to(path):
            raise RuntimeError(f'Conflicting external legacy package already imported: {package}')
    if str(path) not in sys.path:sys.path.insert(0,str(path))
    return path
