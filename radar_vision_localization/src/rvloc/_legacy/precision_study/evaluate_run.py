"""Explicit, complete-episode offline evaluation with a pre-read test freeze.

Test source pixels, IQ, calibration records and labels are inaccessible through
this entry until the named model/config/evaluator/split identities match the
frozen-final receipt. This command never trains or selects a checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import torch

from .evaluation import evaluate_samples, serializable, _geometric_gate
from rvwm.ray_scoring import observation_rejection_reason

ROOT=Path(__file__).resolve().parents[1]
EVALUATION_SOURCE=Path(__file__).with_name('evaluation.py')


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def validate_test_freeze(model_name,checkpoint,configuration,split_manifest,freeze,
                         evaluation_source=EVALUATION_SOURCE):
    """Validate only source/weight/config/partition bytes, before data readers."""
    if freeze is None:
        raise ValueError('A frozen-final receipt is required before reading test inputs or labels')
    receipt=json.loads(Path(freeze).read_text())
    if not isinstance(receipt,dict) or not isinstance(receipt.get('models'),dict):
        raise ValueError('Frozen-final receipt must contain a model identity dictionary')
    if receipt.get('status')!='frozen_for_test':
        raise ValueError('Test receipt status must be frozen_for_test')
    item=receipt.get('models',{}).get(model_name)
    if not isinstance(item,dict):
        raise ValueError('Model name is absent from the frozen-final test receipt')
    actual=dict(checkpoint_sha256=sha256(checkpoint),configuration_sha256=sha256(configuration),
        evaluation_source_sha256=sha256(evaluation_source),split_manifest_sha256=sha256(split_manifest),
        evaluation_entrypoint_source_sha256=sha256(__file__))
    for name in ('checkpoint_sha256','configuration_sha256'):
        if item.get(name)!=actual[name]:
            raise ValueError('Frozen test identity mismatch: '+name)
    for name in ('evaluation_source_sha256','split_manifest_sha256'):
        if receipt.get(name)!=actual[name]:
            raise ValueError('Frozen test identity mismatch: '+name)
    if ('evaluation_entrypoint_source_sha256' in receipt and
            receipt['evaluation_entrypoint_source_sha256']!=actual['evaluation_entrypoint_source_sha256']):
        raise ValueError('Frozen test identity mismatch: evaluation_entrypoint_source_sha256')
    return dict(status='verified_before_sensor_or_label_read',freeze_path=str(Path(freeze).resolve()),
        freeze_sha256=sha256(freeze),model_name=model_name,**actual)


def _load_model_and_reader(kind,checkpoint,code_root,config,device):
    if kind=='study':
        from .versions import load_frozen_study_model
        model=load_frozen_study_model(code_root,checkpoint,config,device)
        package=model._precision_frozen_source_package
        loader=importlib.import_module(package+'.data').load_samples
    else:
        from .data import load_samples as loader
        if config.get('precision_radar_mode','ra_re')!='ra_re':
            raise ValueError('Original v3/v2 evaluation needs its actual RA/RE input configuration')
        if kind=='v3':
            from depth_localization.checkpoint import load_checkpoint
        else:
            from .reproduce import frozen_v2_loader
            snapshot=code_root or ROOT/'results/training/depth_v2/source_snapshot_before_restart'
            load_checkpoint=frozen_v2_loader(snapshot).load_checkpoint
        model=load_checkpoint(checkpoint,config,device)
    return model,loader


def _write_json(path,value):
    path=Path(path)
    descriptor,temporary=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(descriptor,'w') as stream:
            json.dump(serializable(value),stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)



def sanitize_diagnostics(value,path,records):
    """Convert only exported diagnostic non-finites to null, with exact paths.

    Primary summaries are deliberately never sent through this helper. Their
    existing strict JSON writer rejects nonfinite metrics rather than making
    them zero or silently accepting them.
    """
    value=serializable(value)
    if isinstance(value,dict):
        return {str(key):sanitize_diagnostics(item,path+'.'+str(key),records) for key,item in value.items()}
    if isinstance(value,list):
        return [sanitize_diagnostics(item,f'{path}[{index}]',records) for index,item in enumerate(value)]
    if isinstance(value,float) and not math.isfinite(value):
        records.append(dict(path=path,original_nonfinite_repr=repr(value),replacement=None))
        return None
    return value


def _split_ids(config,split_path,role):
    manifest=json.loads(Path(split_path).read_text())
    sets={name:set(manifest['splits'][name]['episode_ids']) for name in ('train','val','test')}
    if any(len(sets[name])!=len(manifest['splits'][name]['episode_ids']) for name in sets):
        raise ValueError('Duplicate episode IDs in frozen partition')
    if any(sets[a]&sets[b] for a,b in (('train','val'),('train','test'),('val','test'))):
        raise ValueError('Train, validation and test must contain disjoint whole episodes')
    ids=list(manifest['splits'][role]['episode_ids'])
    if not ids:raise ValueError('Requested frozen split is empty')
    return ids,manifest


def main(argv=None):
    requested=list(sys.argv[1:] if argv is None else argv)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=('study','v3','v2'),required=True)
    p.add_argument('--model-name',required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--code-root',type=Path)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=1);p.add_argument('--device',default='cpu')
    p.add_argument('--freeze',type=Path)
    a=p.parse_args(requested)
    if a.threads<1:p.error('A positive thread count is required')
    if a.kind=='study' and a.code_root is None:p.error('Study checkpoints require their immutable --code-root')
    if a.split=='test' and a.freeze is None:p.error('Freeze final choices before reading any test inputs or labels')
    config_bytes=a.config.read_bytes();config=json.loads(config_bytes)
    split_path=Path(config.get('split_manifest') or config['visual_split_manifest'])
    # Hashing immutable partition bytes authenticates source identity. Its
    # episode contents are not decoded until the complete test gate succeeds.
    split_sha=sha256(split_path)
    if config.get('split_manifest_sha256') not in (None,split_sha):
        raise ValueError('Evaluation configuration partition SHA mismatch')
    freeze_receipt=None
    if a.split=='test':
        freeze_receipt=validate_test_freeze(a.model_name,a.checkpoint,a.config,split_path,a.freeze)
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Existing evaluation output is protected: '+str(out))
    # Exclusive claim prevents two simultaneous invocations using an empty dir.
    claim=dict(pid=os.getpid(),started_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        command=[sys.executable,'-m','precision_study.evaluate_run',*requested])
    with (out/'evaluation_claim.json').open('x') as stream:json.dump(claim,stream,indent=2)
    started=time.perf_counter()
    _write_json(out/'progress.json',dict(stage='checking_model_and_partition',model=a.model_name,split=a.split,
        test_freeze=freeze_receipt,labels_used_for_prediction=False))
    try:
        ids,partition=_split_ids(config,split_path,a.split)
        torch.set_num_threads(a.threads)
        model,loader=_load_model_and_reader(a.kind,a.checkpoint,a.code_root,config,a.device)
        metadata=getattr(model,'checkpoint_metadata',{})
        trained=metadata.get('train_episodes')
        if not isinstance(trained,list) or not trained or not isinstance(metadata.get('training_scope'),dict):
            raise ValueError('Trained model is missing explicit training episode/scope lineage')
        if set(trained)&set(ids):raise ValueError('Evaluation episodes overlap this model training split')
        if a.split=='test':
            # Reject a checkpoint/config/source replacement between the first
            # hash gate and strict loading; no test reader has been called yet.
            if validate_test_freeze(a.model_name,a.checkpoint,a.config,split_path,a.freeze)!=freeze_receipt:
                raise ValueError('Frozen final identities changed during model loading')
        _write_json(out/'progress.json',dict(stage='loading_explicit_offline_evaluation',model=a.model_name,
            split=a.split,episodes=len(ids),test_freeze=freeze_receipt,
            labels_read_for_scoring_only=True,labels_used_for_prediction=False))
        samples,manifest,frontend=loader(config,ids,a.split,count=0,include_labels=True)
        if set(r['episode_id'] for r in manifest)!=set(ids):
            raise ValueError('Evaluation reader omitted or added a declared whole episode')
        keys=[(r['episode_id'],int(r['frame_idx'])) for r in manifest]
        if len(keys)!=len(set(keys)):raise ValueError('Evaluation reader returned duplicate frames')
        declared={r['episode']:r.get('selected_frame_indices') for r in partition.get('episodes',[])}
        # Current frozen inventory explicitly lists all approach frames. If a
        # different phase contract is used, leave source-reader validation in
        # charge rather than forcing an inappropriate inventory intersection.
        if config.get('training_phases',['approach'])==['approach'] and all(declared.get(e) is not None for e in ids):
            expected={(e,int(i)) for e in ids for i in declared[e]}
            if set(keys)!=expected:raise ValueError('Reader did not retain every frozen selected frame')
        positions={key:i for i,key in enumerate(keys)}
        by_feature={id(s['features']):positions[(s['episode_id'],int(s['frame_idx']))] for s in samples}
        progress_frames=0;next_mark=100
        def pulse(row):
            nonlocal progress_frames
            done=int(row['frames'])
            if done<progress_frames:return
            progress_frames=done
            value=dict(stage='evaluating',model=a.model_name,split=a.split,frames=done,total_frames=len(manifest),
                elapsed_s=time.perf_counter()-started,labels_used_for_prediction=False,
                labels_used_for_offline_scoring=True,test_freeze_verified=a.split=='test' and freeze_receipt is not None)
            _write_json(out/'progress.json',value)
            print(json.dumps(value,ensure_ascii=False),flush=True)
        def observed_gate(features,cfg):
            nonlocal next_mark
            # This exact observation-gate hook runs before the next prepared
            # frame. It lets the frozen evaluator retain its original loop
            # while emitting completed-frame progress every 100 samples.
            completed=by_feature[id(features)]
            while next_mark<=completed:
                pulse(dict(frames=next_mark));next_mark+=100
            return observation_rejection_reason(features,cfg) if 'ra_power' in features and 're_power' in features else _geometric_gate(features,cfg)
        run_info=dict(model_name=a.model_name,kind=a.kind,split=a.split,validation_or_test_episodes=ids,
            configuration=config,configuration_path=str(a.config.resolve()),
            configuration_sha256=hashlib.sha256(config_bytes).hexdigest(),checkpoint=str(a.checkpoint.resolve()),
            checkpoint_sha256=sha256(a.checkpoint),checkpoint_step=metadata.get('steps'),
            checkpoint_source_sha256=metadata.get('source_sha256'),checkpoint_training_scope=metadata.get('training_scope'),
            checkpoint_train_episodes=trained,code_root=str(a.code_root.resolve()) if a.code_root else
                str(ROOT/'results/training/depth_v2/source_snapshot_before_restart') if a.kind=='v2' else str(ROOT),
            split_manifest_sha256=split_sha,evaluation_source_sha256=sha256(EVALUATION_SOURCE),
            entrypoint_source_sha256=sha256(__file__),command=claim['command'],started_at=claim['started_at'],
            device=a.device,threads=a.threads,torch_version=str(torch.__version__),python=platform.python_version(),
            platform=platform.platform(),machine=platform.machine(),cpu_logical_cores=os.cpu_count(),
            frontend=frontend,declared_frames=len(manifest),loaded_frames=len(samples),
            labels_used_for_prediction=False,labels_used_for_offline_scoring=True,
            normal_inference_entrypoint=False,complete_episode_evaluation=True,diagnostic_subset=False,
            diagnostic_serialization_policy='nonfinite diagnostic values become null with complete paths/counts in serialization_sanitization.json; primary metrics are not altered',
            test_freeze=freeze_receipt)
        result=evaluate_samples(model,samples,manifest,config,capture_stages=True,
            progress_callback=pulse,run_info=run_info,observation_gate=observed_gate)
        pulse(dict(frames=len(manifest)))
        nonfinite_records=[]
        for name in ('frames','members','detections'):
            with (out/(name+'.jsonl')).open('x') as stream:
                for index,row in enumerate(result[name]):
                    safe=sanitize_diagnostics(row,f'{name}[{index}]',nonfinite_records)
                    stream.write(json.dumps(safe,ensure_ascii=False,allow_nan=False)+'\n')
        _write_json(out/'serialization_sanitization.json',dict(
            status='diagnostic_nonfinite_replaced_with_null' if nonfinite_records else 'no_nonfinite_diagnostics',
            nonfinite_value_count=len(nonfinite_records),records=nonfinite_records,
            primary_metrics_changed=False,primary_failure_denominators_changed=False,
            policy='Rejected prediction diagnostics only; no primary metric is passed through sanitization or changed to zero'))
        for name,value in (('summary',result['summary']),('episodes',result['episodes']),('run_info',run_info)):
            _write_json(out/(name+'.json'),value)
        _write_json(out/'input_manifest.json',dict(frames=manifest,frontend=frontend))
        _write_json(out/'progress.json',dict(stage='complete',model=a.model_name,split=a.split,
            frames=len(manifest),truth_members=result['summary']['truth_members'],
            elapsed_s=time.perf_counter()-started,labels_used_for_prediction=False,
            labels_used_for_offline_scoring=True,test_freeze=freeze_receipt))
        print(json.dumps(dict(status='complete',model=a.model_name,output=str(out),
            rmse_m=result['summary']['member_position_error_m']['rmse']),ensure_ascii=False),flush=True)
        return 0
    except BaseException as exc:
        _write_json(out/'progress.json',dict(stage='failed',model=a.model_name,split=a.split,
            error=f'{type(exc).__name__}: {exc}',elapsed_s=time.perf_counter()-started,
            labels_used_for_prediction=False,test_freeze=freeze_receipt))
        raise


if __name__=='__main__':raise SystemExit(main())
