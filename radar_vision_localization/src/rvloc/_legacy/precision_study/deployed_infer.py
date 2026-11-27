"""Ordinary inference using one fixed, authenticated deployment description.

The command line selects only the current observation, device and output.
Model identity comes from DEFAULT_DESCRIPTOR, installed separately after the
final model decision. There is no fallback to training checkpoints.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_DESCRIPTOR=ROOT/'results/deployment/default/deployment_descriptor.json'
FIELDS=frozenset(('kind','model_name','config','checkpoint','source_root',
                  'configuration_sha256','checkpoint_sha256','deployment_receipt_sha256'))


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_descriptor(path):
    """Authenticate descriptor references before importing a model/data entry."""
    path=Path(path).resolve()
    raw=path.read_bytes();description=json.loads(raw)
    if not isinstance(description,dict) or set(description)!=FIELDS:
        raise ValueError('Deployment description must contain exactly the fixed model identity fields')
    if description['kind'] not in ('v3','study'):
        raise ValueError('Deployment kind must explicitly be v3 or study')
    if not isinstance(description['model_name'],str) or not re.fullmatch(r'[A-Za-z0-9_-]+',description['model_name']):
        raise ValueError('Deployment model_name must be a nonempty identifier')
    for key in ('configuration_sha256','checkpoint_sha256','deployment_receipt_sha256'):
        if not isinstance(description[key],str) or not re.fullmatch(r'[0-9a-f]{64}',description[key]):
            raise ValueError('Deployment description needs a complete SHA256: '+key)
    resolved=dict(description)
    for key in ('config','checkpoint'):
        if not isinstance(description[key],str) or not description[key]:
            raise ValueError('Deployment description needs a fixed path: '+key)
        value=Path(description[key]).expanduser()
        resolved[key]=(path.parent/value).resolve() if not value.is_absolute() else value.resolve()
        if not resolved[key].is_file():raise FileNotFoundError('Deployment '+key+' missing: '+str(resolved[key]))
    source=description['source_root']
    if description['kind']=='v3':
        if source is not None:raise ValueError('Original v3 requires source_root=null and its explicit baseline loader')
    else:
        if not isinstance(source,str) or not source:raise ValueError('Study deployment requires a fixed frozen source_root')
        source=Path(source).expanduser()
        source=(path.parent/source).resolve() if not source.is_absolute() else source.resolve()
        package=source if (source/'checkpoint.py').is_file() else source/'precision_study'
        if not all((package/name).is_file() for name in ('__init__.py','checkpoint.py','model.py','data.py')):
            raise FileNotFoundError('Study deployment frozen source package missing')
    resolved['source_root']=source
    receipt=resolved['checkpoint'].parent/'deployment_receipt.json'
    paths={'configuration_sha256':resolved['config'],'checkpoint_sha256':resolved['checkpoint'],
           'deployment_receipt_sha256':receipt}
    for key,target in paths.items():
        if _sha(target)!=description[key]:raise ValueError('Fixed deployment identity changed: '+key)
    # This helper verifies the sanitized receipt before any torch.load. Its
    # import does not construct a model or access observation/label readers.
    from .export_deployment import verify_deployment_receipt
    verify_deployment_receipt(resolved['checkpoint'],description['kind'])
    if path.read_bytes()!=raw:raise ValueError('Deployment description changed during verification')
    return resolved,dict(descriptor_path=str(path),descriptor_sha256=hashlib.sha256(raw).hexdigest(),
        model_name=description['model_name'],kind=description['kind'],
        configuration_sha256=description['configuration_sha256'],checkpoint_sha256=description['checkpoint_sha256'],
        deployment_receipt_sha256=description['deployment_receipt_sha256'])


def main(argv=None,*,descriptor_path=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode');parser.add_argument('--frame',type=int,default=0)
    parser.add_argument('--split',choices=('train','val'),default='val')
    parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path)
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args(argv)
    if args.frame<0 or args.threads<1:parser.error('Require a nonnegative frame and positive thread count')
    # descriptor_path is an offline Python injection point for audit/tests.
    # It has deliberately no CLI option or environment-variable override.
    fixed=Path(descriptor_path) if descriptor_path is not None else DEFAULT_DESCRIPTOR
    description,identity=load_descriptor(fixed)
    output=args.output or ROOT/'results/deployed_examples'/description['model_name']/datetime.now(
        timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    command=['--config',str(description['config']),'--checkpoint',str(description['checkpoint']),
        '--split',args.split,'--frame',str(args.frame),'--device',args.device,
        '--threads',str(args.threads),'--output',str(output)]
    if args.episode:command+=['--episode',args.episode]
    if description['kind']=='v3':command+=['--baseline-v3']
    else:command+=['--source-root',str(description['source_root'])]
    # Recheck the entire fixed identity immediately before handing control to
    # strict infer. It independently checks checkpoint/config/source lineage.
    if load_descriptor(fixed)[1]!=identity:raise ValueError('Deployment identity changed before inference')
    from .infer import main as strict_infer
    result=strict_infer(command)
    identity.update(entry_sha256=_sha(__file__),input_frames=1,split=args.split,
        test_inputs_permitted=False,ordinary_labels_read=False)
    with (output.resolve()/'deployment_route_receipt.json').open('x') as stream:
        json.dump(identity,stream,indent=2,allow_nan=False);stream.write('\n')
    return result


if __name__=='__main__':raise SystemExit(main())
