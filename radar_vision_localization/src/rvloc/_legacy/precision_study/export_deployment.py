"""Offline export of label-free deployment copies; training artifacts stay intact.

Loading the source checkpoint here is an explicit offline artifact operation.
Ordinary inference verifies the generated receipt before unpickling model.pt.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import torch

from rvwm.data import sha256
from depth_localization.checkpoint import save_checkpoint

SCHEMA='strict_label_free_deployment_v1'
MODEL_FIELDS=frozenset(('kind','schema','feature_schema','steps','model_trained','model_config',
    'state_dict','source_sha256','radar_mode','input_frames','output','uncertainty_calibrated','sampled_layouts'))
LINEAGE_FIELDS=frozenset(('train_episodes','validation_episodes','training_scope','hyperparameters',
    'inventory_sha256','validation_inventory_sha256','training_configuration','learning_rate_schedule','model_initialization'))
FORBIDDEN_FIELDS=frozenset(('validation_report','frames','members','detections','truth_member_count',
    'primary_member_errors_m','truth_positions','truth_positions_base_m','positions','targets','labels',
    'target_positions','target_ids','true_count','member_count','true_depths','depth_labels',
    'depth_map','projected_uv','future_observations'))


def tensor_state_sha256(state):
    digest=hashlib.sha256()
    for name in sorted(state):
        value=state[name]
        if not isinstance(value,torch.Tensor):raise ValueError('State dictionary must contain tensors only')
        value=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_no_sample_labels(value):
    if isinstance(value,dict):
        bad=FORBIDDEN_FIELDS & value.keys()
        if bad:raise ValueError('Per-sample label/evaluation metadata in deployment copy: '+str(sorted(bad)))
        for key,item in value.items():
            if key!='state_dict':assert_no_sample_labels(item)
    elif isinstance(value,(list,tuple)):
        for item in value:
            if isinstance(item,(dict,list,tuple)):assert_no_sample_labels(item)


def _strict_load(kind,path,config=None,source_root=None):
    if kind=='study':
        if source_root:
            from .versions import load_frozen_study_model
            return load_frozen_study_model(source_root,path,config,'cpu')
        from .checkpoint import load_checkpoint
    elif kind=='v3':
        from depth_localization.checkpoint import load_checkpoint
    elif kind=='v2':
        if source_root is None:raise ValueError('v2 export requires its immutable --source-root')
        from .reproduce import frozen_v2_loader
        load_checkpoint=frozen_v2_loader(source_root).load_checkpoint
    else:raise ValueError('Unknown checkpoint kind')
    return load_checkpoint(path,config,'cpu')


def verify_deployment_receipt(checkpoint,expected_kind=None):
    """Verify a public receipt and checkpoint bytes before any torch.load."""
    checkpoint=Path(checkpoint).resolve();path=checkpoint.parent/'deployment_receipt.json'
    if not path.is_file():
        raise ValueError('Ordinary inference requires a label-free deployment export and deployment_receipt.json')
    receipt=json.loads(path.read_text())
    if (receipt.get('schema')!=SCHEMA or receipt.get('status')!='verified_label_free'
            or receipt.get('sanitized') is not True or receipt.get('checkpoint_file')!=checkpoint.name):
        raise ValueError('Invalid label-free deployment receipt')
    if expected_kind is not None and receipt.get('model_kind')!=expected_kind:
        raise ValueError('Deployment checkpoint kind differs from the explicit inference mode')
    if receipt.get('checkpoint_sha256')!=sha256(checkpoint):
        raise ValueError('Deployment checkpoint SHA differs from its verified export')
    return receipt


def export_checkpoint(checkpoint,output,*,kind,config=None,source_root=None):
    """Export, strict-reload and compare all parameter/buffer values offline."""
    checkpoint=Path(checkpoint).resolve();output=Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Deployment output is protected: '+str(output))
    original_sha=sha256(checkpoint)
    raw=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=_strict_load(kind,checkpoint,config,source_root)
    if sha256(checkpoint)!=original_sha:raise ValueError('Source checkpoint changed during deployment export')
    expected_state=tensor_state_sha256(raw['state_dict'])
    if tensor_state_sha256(model.state_dict())!=expected_state:
        raise ValueError('Strict source loader changed state tensor content')
    payload={name:deepcopy(value) for name,value in raw.items() if name in MODEL_FIELDS|LINEAGE_FIELDS}
    if not payload.get('train_episodes') or not isinstance(payload.get('training_scope'),dict):
        raise ValueError('Deployment export requires original training episode lineage')
    payload['deployment_export']=dict(schema=SCHEMA,sanitized=True,source_checkpoint_sha256=original_sha,
        state_tensor_sha256=expected_state,exporter_sha256=sha256(__file__))
    assert_no_sample_labels(payload)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.export_claim').open('x') as stream:stream.write(original_sha+'\n')
    destination=output/'model.pt';save_checkpoint(payload,destination)
    restored=_strict_load(kind,destination,config,source_root)
    if tensor_state_sha256(restored.state_dict())!=expected_state:raise ValueError('Exported model tensors differ')
    assert_no_sample_labels(restored.checkpoint_metadata)
    if restored.config!=model.config:raise ValueError('Export changed model configuration')
    if raw['source_sha256']!=restored.checkpoint_metadata['source_sha256']:
        raise ValueError('Export changed model source identity')
    receipt=dict(schema=SCHEMA,status='verified_label_free',sanitized=True,
        exported_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),model_kind=kind,
        checkpoint_file=destination.name,checkpoint_sha256=sha256(destination),
        source_checkpoint_path=str(checkpoint),source_checkpoint_sha256=original_sha,
        source_root=str(Path(source_root).resolve()) if source_root else None,
        state_tensor_sha256=expected_state,state_tensor_count=len(raw['state_dict']),
        exact_state_dict_preserved=True,model_configuration_preserved=True,source_identity_preserved=True,
        source_sha256=raw['source_sha256'],model_configuration_sha256=hashlib.sha256(json.dumps(
            raw['model_config'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        retained_payload_fields=sorted(payload),removed_payload_fields=sorted(set(raw)-set(payload)),
        source_may_contain_validation_labels=True,per_sample_validation_metadata_retained=False,
        operation_scope='Explicit offline training-artifact export; generated deployment copy is label-free',
        exporter_sha256=sha256(__file__))
    (output/'deployment_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    verify_deployment_receipt(destination,kind)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('study','v3','v2'),required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path);parser.add_argument('--source-root',type=Path)
    args=parser.parse_args(argv)
    config=json.loads(args.config.read_text()) if args.config else None
    report=export_checkpoint(args.checkpoint,args.output,kind=args.kind,config=config,source_root=args.source_root)
    print(json.dumps(dict(status=report['status'],checkpoint=str(args.output.resolve()/'model.pt'),
        receipt=str(args.output.resolve()/'deployment_receipt.json'),exact_state_dict_preserved=True),ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
