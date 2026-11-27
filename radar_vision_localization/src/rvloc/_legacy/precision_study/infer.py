"""Strict current-frame inference. Offline scoring is a separate command.

This entry point never constructs Dataset or requests target labels. Both the
study and explicit baseline-v3 path use the new sensor-only input reader.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from rvwm.data import sha256
from rvwm.ray_scoring import observation_rejection_reason
from rvwm.validation import write_json
from depth_localization import checkpoint as baseline_checkpoint
from depth_localization.evaluation import prediction_rejection
from . import checkpoint as study_checkpoint
from .evaluation import _geometric_gate, serializable
from .visual import SCHEMA as VISUAL_SCHEMA, validate_visual_features
from .visual_data import public_split_manifest
from .export_deployment import verify_deployment_receipt, assert_no_sample_labels

ROOT=Path(__file__).resolve().parents[1]


def load_observation_features(config, episode, frame_idx, split):
    # Late import allows this entry point to remain independently testable.
    from .data import load_observation_features as sensor_loader
    return sensor_loader(config, episode, frame_idx, split)


def _input_digest(features):
    """Bind numeric observation content; exclude uninterpreted metadata."""
    digest=hashlib.sha256()
    def add(prefix,value):
        if isinstance(value,torch.Tensor):value=value.detach().cpu().numpy()
        if isinstance(value,np.ndarray):
            value=np.ascontiguousarray(value)
            digest.update(prefix.encode());digest.update(str(value.dtype).encode())
            digest.update(repr(value.shape).encode());digest.update(value.tobytes())
        elif isinstance(value,dict):
            for name in sorted(value):
                if name not in ('metadata','depth_metadata','volume_receipt'):add(prefix+'/'+name,value[name])
        elif isinstance(value,(str,int,float,bool,type(None))):
            digest.update(prefix.encode());digest.update(repr(value).encode())
    add('',features)
    return digest.hexdigest()


def _source_receipt(features):
    metadata=features.get('metadata',{})
    if not isinstance(metadata,dict):metadata={}
    depth=features.get('depth_metadata',{})
    depth=depth if isinstance(depth,dict) else {}
    provenance=depth.get('provenance',{})
    provenance=provenance if isinstance(provenance,dict) else {}
    hashes={}
    allowed={'rgb','iq','source_trajectory','source_metadata','source_capture_frames_sha256',
             'source_capture_metadata_sha256','rgb_sha256','iq_sha256','capture_hash','metadata_hash'}
    for source in (metadata.get('hashes',{}),metadata.get('source_hashes',{}),provenance.get('source_hashes',{})):
        if isinstance(source,dict):
            hashes.update({k:v for k,v in source.items() if k in allowed and isinstance(v,(str,type(None)))})
    result=dict(sensor_hashes=hashes,numeric_input_sha256=_input_digest(features))
    volume=features.get('joint_volume',{})
    if isinstance(volume,dict) and isinstance(volume.get('metadata'),dict):
        vm=volume['metadata']
        result['joint_volume_metadata_sha256']=hashlib.sha256(json.dumps(serializable(vm),sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest()
        result['joint_volume_version']=vm.get('version')
    receipt=metadata.get('volume_receipt') or features.get('volume_receipt')
    if isinstance(receipt,dict):
        result['volume_receipt_sha256']=hashlib.sha256(json.dumps(serializable(receipt),sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest()
    return result


def _validate_lineage(model, manifest, episode, split):
    if model.trained is not True:raise ValueError('Normal inference requires a genuinely trained model')
    meta=getattr(model,'checkpoint_metadata',None)
    if not isinstance(meta,dict):raise ValueError('Checkpoint lacks training-data lineage')
    assert_no_sample_labels(meta)
    train=meta.get('train_episodes');scope=meta.get('training_scope');steps=meta.get('steps')
    if (not isinstance(train,list) or not train or any(not isinstance(x,str) for x in train)
            or len(set(train))!=len(train) or not isinstance(scope,dict)
            or isinstance(steps,bool) or not isinstance(steps,int) or steps<1):
        raise ValueError('Checkpoint lacks valid training-data lineage and optimizer history')
    declared=manifest['splits']['train']['episode_ids']
    if not set(train)<=set(declared):raise ValueError('Checkpoint training episodes violate the frozen episode split')
    if split!='train' and episode in train:raise ValueError('Held-out episode overlaps checkpoint training')
    return meta


def _reject_supervision_fields(value):
    forbidden={'positions','targets','labels','target_positions','true_positions','true_count',
        'target_count','target_ids','true_depths','depth_labels','depth_map','projected_uv',
        'true_projection','future','future_frames','future_observations','formation_index'}
    if isinstance(value,dict):
        bad=forbidden & value.keys()
        if bad:raise ValueError('Supervision/future fields are forbidden in ordinary inference: '+str(sorted(bad)))
        for item in value.values():
            if isinstance(item,dict):_reject_supervision_fields(item)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--split',choices=('train','val'),default='val',
        help='Ordinary examples use train/val; test inputs require evaluate_run and its frozen-final receipt')
    parser.add_argument('--episode');parser.add_argument('--frame',type=int,default=0)
    parser.add_argument('--output',type=Path);parser.add_argument('--device',default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--source-root',type=Path,help='Immutable precision-study code snapshot')
    parser.add_argument('--baseline-v3',action='store_true',help='Explicitly load the original v3 kind; no automatic fallback')
    args=parser.parse_args(argv)
    if args.frame<0 or args.threads<1:parser.error('Require a nonnegative frame and positive thread count')
    config_bytes=args.config.read_bytes();config=json.loads(config_bytes)
    if type(config.get('window',1)) is not int or config.get('window',1)!=1:
        parser.error('Only one current frame is supported')
    mode=config.get('precision_radar_mode','ra_re')
    if mode not in ('none','ra_re','joint_volume'):parser.error('Unknown observed-input mode')
    if args.baseline_v3 and mode!='ra_re':parser.error('baseline-v3 explicitly requires original RA/RE inputs')
    if args.baseline_v3 and args.source_root:parser.error('--source-root is for study checkpoints; baseline-v3 uses its original strict loader')
    _,manifest,split_digest=public_split_manifest(config)
    members=manifest['splits'][args.split]['episode_ids']
    if not members:parser.error('Requested frozen episode split is empty')
    episode=args.episode or members[0]
    if episode not in members:parser.error('Episode is outside the frozen requested split')
    path=args.checkpoint or Path(config.get('precision_checkpoint') or config.get('depth_checkpoint') or '')
    if not path.is_file():raise FileNotFoundError('A matching trained checkpoint is required: '+str(path))
    output=args.output or ROOT/'results/precision_examples'/episode/datetime.now(
        timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output=output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Existing inference output is protected: '+str(output))
    deployment=verify_deployment_receipt(path,'v3' if args.baseline_v3 else 'study')
    torch.set_num_threads(args.threads)
    if args.source_root:
        from .versions import load_frozen_study_model, frozen_study_sensor_loader
        model=load_frozen_study_model(args.source_root,path,config,args.device)
        sensor_loader=frozen_study_sensor_loader(model)
    else:
        loader=baseline_checkpoint.load_checkpoint if args.baseline_v3 else study_checkpoint.load_checkpoint
        model=loader(path,config,args.device)
        sensor_loader=load_observation_features
    metadata=_validate_lineage(model,manifest,episode,args.split)
    preprocessing_started=time.perf_counter()
    loaded=sensor_loader(config,episode,args.frame,args.split)
    features=loaded['features'] if isinstance(loaded,dict) and 'features' in loaded else loaded
    _reject_supervision_fields(features)
    preprocessing_seconds=time.perf_counter()-preprocessing_started
    if mode=='none':
        if features.get('schema')!=VISUAL_SCHEMA:raise ValueError('No-radar mode requires the strict visual-only feature contract')
        validate_visual_features(features);reason=_geometric_gate(features,config)
    else:
        if features.get('schema')==VISUAL_SCHEMA:raise ValueError('Radar mode cannot accept a visual-only observation')
        reason=observation_rejection_reason(features,config)
    cuda=str(args.device).startswith('cuda')
    if cuda:
        torch.cuda.synchronize(args.device);torch.cuda.reset_peak_memory_stats(args.device)
    started=time.perf_counter();prediction=None
    if reason is None:
        prediction=model.predict(features,allow_untrained_validation=False)
        reason=prediction_rejection(features,prediction)
        if reason is not None:prediction=None
    if cuda:torch.cuda.synchronize(args.device)
    elapsed=time.perf_counter()-started
    checkpoint_receipt=dict(path=str(path.resolve()),sha256=sha256(path),steps=metadata['steps'],
        kind=metadata.get('kind'),training_scope=metadata['training_scope'],baseline_v3=args.baseline_v3)
    payload=dict(episode_id=episode,frame_idx=args.frame,split=args.split,status=reason or 'valid_uncalibrated',
        prediction=prediction,model_trained=True,input_frames=1,radar_mode=mode,
        coordinate_frame='base_local',depth_definition='camera optical-centre slant metres along the unit observed bbox-centre ray',
        uncertainty_calibrated=False,valid_mask_semantics='public geometric support mask; not localization confidence',
        preprocessing_seconds=preprocessing_seconds,inference_seconds=elapsed,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if cuda else None,
        checkpoint=checkpoint_receipt)
    output.mkdir(parents=True,exist_ok=True)
    # Check again at publication, so a competing writer cannot be overwritten.
    if any(output.iterdir()):raise FileExistsError('Output became nonempty during inference')
    # Exclusive publication marker also protects simultaneous writers that
    # both observed an empty directory before prediction.
    with (output/'.publication_claim').open('x') as stream:stream.write(sha256(__file__)+'\n')
    write_json(output/'prediction.json',serializable(payload))
    np.savez_compressed(output/'prediction_arrays.npz',**({name:np.asarray(prediction[name]) for name in
        ('depths','initial_depths','member_positions','initial_positions','valid_mask') if name in prediction} if prediction else {}))
    write_json(output/'receipt.json',serializable(dict(schema='strict_single_frame_precision_inference_v1',
        episode_id=episode,frame_idx=args.frame,split=args.split,labels_used_for_prediction=False,
        target_fields_read=False,future_observations_read=False,radar_mode=mode,
        configuration_sha256=hashlib.sha256(config_bytes).hexdigest(),split_manifest_sha256=split_digest,
        inference_entry_sha256=sha256(__file__),model_source_sha256=metadata.get('source_sha256',{}),
        checkpoint=checkpoint_receipt,deployment_receipt_sha256=sha256(path.parent/'deployment_receipt.json'),
        deployment_state_tensor_sha256=deployment['state_tensor_sha256'],
        inputs=_source_receipt(features))))
    print(json.dumps(dict(output=str(output),status=payload['status'],model_trained=True,
        detections=len(features['ray_directions']),radar_mode=mode,labels_read=False),ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
