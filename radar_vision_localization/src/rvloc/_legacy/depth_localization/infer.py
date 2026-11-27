"""Run one sensor-only deterministic depth prediction, with optional later scoring."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import time
import numpy as np
import torch

from rvwm.data import Dataset,sha256
from rvwm.partitions import active_splits
from rvwm.validation import write_json
from rvwm.ray_scoring import observation_rejection_reason
from .checkpoint import ROOT,SCHEMA,load_checkpoint,source_hashes
from .evaluation import prediction_rejection


def serializable(value):
    if isinstance(value,dict):return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serializable(v) for v in value]
    if isinstance(value,np.ndarray):return serializable(value.tolist())
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,torch.Tensor):return serializable(value.detach().cpu().numpy())
    return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'depth_config.json')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--episode');p.add_argument('--frame',type=int,default=0)
    p.add_argument('--output',type=Path);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--evaluate',action='store_true',help='Read GT only after the sensor-only prediction has been computed')
    p.add_argument('--allow-untrained-validation',action='store_true',help='Interface check only; never writes trained weights or reports accuracy')
    a=p.parse_args(argv);cfg=json.loads(a.config.read_text())
    if cfg.get('architecture_schema')!=SCHEMA or cfg.get('window')!=1:p.error('A matching single-frame depth-v2 configuration is required')
    if a.threads<1 or a.frame<0:p.error('Invalid thread count/frame')
    if a.evaluate and a.allow_untrained_validation:p.error('Untrained interface validation cannot report localization accuracy')
    torch.set_num_threads(a.threads)
    dataset=Dataset(cfg['dataset_root'],echo_subdir=cfg['echo_subdir'],verify_sources=True)
    splits=active_splits(dataset,cfg);episode=a.episode or splits[a.split][0]
    if episode not in splits[a.split]:p.error('Episode is outside the frozen requested split')
    if a.allow_untrained_validation:
        if a.checkpoint:p.error('Untrained validation cannot load weights')
        from .model import DepthLocalizer
        torch.manual_seed(int(cfg.get('seed',20261118)));model=DepthLocalizer(cfg).to(a.device).eval()
        checkpoint=dict(status='untrained_interface_validation',steps=0,path=None)
    else:
        path=a.checkpoint or Path(cfg['depth_checkpoint'])
        if not path.is_file():raise FileNotFoundError('Train the new depth-v2 model first; legacy world.best.pt cannot be used: '+str(path))
        model=load_checkpoint(path,cfg,a.device)
        metadata=model.checkpoint_metadata
        trained_ids=metadata.get('train_episodes')
        if not isinstance(trained_ids,list) or not trained_ids or not isinstance(metadata.get('training_scope'),dict):
            raise ValueError('Depth checkpoint is missing training-data lineage; use the versioned training entrypoint')
        if a.split!='train' and episode in trained_ids:raise ValueError('Held-out inference episode overlaps depth training')
        checkpoint=dict(path=str(path.resolve()),sha256=sha256(path),steps=metadata['steps'],
                        training_scope=metadata['training_scope'])
    from .data import load_observation_features
    # This API must not load target positions; test coverage enforces it.
    loaded=load_observation_features(cfg,episode,a.frame,a.split)
    features=loaded['features'] if isinstance(loaded,dict) and 'features' in loaded else loaded
    reason=observation_rejection_reason(features,cfg);prediction=None;started=time.perf_counter()
    if reason is None:
        prediction=model.predict(features,allow_untrained_validation=a.allow_untrained_validation)
        reason=prediction_rejection(features,prediction)
        if reason is not None:prediction=None
    elapsed=time.perf_counter()-started
    out=a.output or ROOT/'results/depth_examples'/episode/datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Existing depth inference output is protected')
    payload=dict(episode_id=episode,frame_idx=a.frame,split=a.split,status=reason or 'valid_uncalibrated',
        prediction=prediction,depth_definition='camera optical-centre slant distance along unit bbox-centre ray, metres',
        coordinate_frame='base_local',single_depth_vector=True,sampled_layouts=False,elapsed_s=elapsed,
        validation_only=a.allow_untrained_validation,model_trained=bool(model.trained),checkpoint=checkpoint)
    write_json(out/'prediction.json',serializable(payload))
    from .features import feature_digest
    write_json(out/'manifest.json',serializable(dict(schema=SCHEMA,status='complete',configuration=cfg,
        input_frames=1,labels_used_for_prediction=False,source_sha256=source_hashes(),checkpoint=checkpoint,
        feature_provenance=features.get('metadata',{}),depth_input_receipt=features.get('depth_metadata',{}),
        depth_feature_digest=feature_digest(features),evaluation_requested=a.evaluate)))
    np.savez_compressed(out/'prediction_arrays.npz',**({key:np.asarray(prediction[key]) for key in
        ('depths','initial_depths','member_positions','initial_positions','valid_mask') if key in prediction} if prediction else {}))
    if a.evaluate:
        # A separate label-reading path, invoked only after prediction.json exists.
        from .evaluation import evaluate
        from diagnostics.data import _world_frame
        targets=dataset.targets(episode,a.frame)
        station_frame=_world_frame(episode,dataset._load(episode),a.frame)
        positions=station_frame.to_local_positions(targets.positions).astype(np.float32)
        sample=dict(features=features,positions=positions,episode_id=episode,frame_idx=a.frame)
        row=dict(episode_id=episode,frame_idx=a.frame,label_count=len(positions),
                 detected_count=len(features['ray_directions']),feature_status='prepared')
        write_json(out/'evaluation.json',evaluate(model,[sample],[row],cfg))
    print(json.dumps(dict(output=str(out),status=payload['status'],members=0 if prediction is None else len(prediction['depths']),
                         model_trained=bool(model.trained),validation_only=a.allow_untrained_validation),ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
