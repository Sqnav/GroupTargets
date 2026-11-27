"""Strictly load a precision-study checkpoint with its immutable code snapshot.

The ordinary ``depth_localization`` and ``rvwm`` imports remain the unchanged
baseline dependencies. No source receipt, checkpoint hash or state is patched.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys


def _package_directory(source_root):
    root=Path(source_root).expanduser().resolve()
    if (root/'checkpoint.py').is_file() and (root/'model.py').is_file():
        package=root
    elif (root/'precision_study/checkpoint.py').is_file():
        package=root/'precision_study'
    else:
        raise FileNotFoundError('No frozen precision_study package under '+str(root))
    if not (package/'__init__.py').is_file():
        raise FileNotFoundError('Frozen package needs its original __init__.py')
    return package


def frozen_study_package(source_root):
    """Return a path/content-specific package alias without changing sys.path."""
    package=_package_directory(source_root)
    digest=hashlib.sha256(str(package).encode())
    files=sorted(package.glob('*.py'))
    for path in files:
        digest.update(path.name.encode());digest.update(path.read_bytes())
    name='_precision_study_frozen_'+digest.hexdigest()[:24]
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,package/'__init__.py',
                                                   submodule_search_locations=[str(package)])
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name,None)
            raise
    return name,package


def load_frozen_study_model(source_root, checkpoint, config=None, device='cpu'):
    """Load through the snapshot's own strict kind/source/config/state checks.

    ``source_root`` can name a code root containing ``precision_study`` or the
    package directory itself. A mismatched snapshot is rejected by its normal
    loader; this helper never substitutes current code or changes saved hashes.
    """
    name,package=frozen_study_package(source_root)
    loader=importlib.import_module(name+'.checkpoint')
    model=loader.load_checkpoint(checkpoint,config=config,device=device)
    model._precision_frozen_source_package=name
    model._precision_frozen_source_root=str(package.parent)
    return model


def frozen_study_sensor_loader(model):
    """Return the same snapshot's ordinary sensor reader, if one was loaded."""
    name=getattr(model,'_precision_frozen_source_package',None)
    if name is None:
        raise ValueError('Model was not loaded from an explicit frozen study snapshot')
    return importlib.import_module(name+'.data').load_observation_features
