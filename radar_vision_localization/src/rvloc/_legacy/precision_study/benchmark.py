"""Measure sensor-only prediction after feature preparation, without labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN={'positions','targets','labels','target_positions','true_positions','true_count',
    'target_ids','true_ids','depth_labels','true_depths','depth_map','ground_truth',
    'true_projection_points','future','future_frames'}


def benchmark_model(model, features_list, device='cpu', warmup=3, repeats=3):
    """Time ordinary ``predict(features)``; no hooks, matching or label reads.

    Each provided feature dictionary is timed ``repeats`` times after ``warmup``
    untimed calls. Device transfer of features inside predict is included; RGB
    detection, IQ/FFT preprocessing, cache/data I/O and metric scoring are not.
    The model's original placement and training state are restored afterwards.
    """
    features_list=list(features_list)
    if not features_list:
        raise ValueError('At least one prepared sensor feature dictionary required')
    if (isinstance(warmup,bool) or not isinstance(warmup,int) or warmup<0 or
            isinstance(repeats,bool) or not isinstance(repeats,int) or repeats<1):
        raise ValueError('Warmup must be nonnegative and repeats positive integers')
    for features in features_list:
        if not isinstance(features,dict) or set(features)&FORBIDDEN:
            raise ValueError('Benchmark accepts sensor-only features, never labels or future data')
    device=torch.device(device)
    parameters=list(model.parameters())
    original_device=parameters[0].device if parameters else torch.device('cpu')
    previous=model.training
    cuda=device.type=='cuda'
    if cuda and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA benchmark but no CUDA device is available')
    model.to(device);model.eval()
    durations=[]
    peak_allocated=peak_reserved=0
    try:
        with torch.no_grad():
            for i in range(warmup):
                model.predict(features_list[i%len(features_list)])
            if cuda:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(repeats):
                for features in features_list:
                    if cuda:torch.cuda.synchronize(device)
                    tick=time.perf_counter_ns()
                    model.predict(features)
                    if cuda:torch.cuda.synchronize(device)
                    durations.append((time.perf_counter_ns()-tick)/1e6)
            if cuda:
                peak_allocated=int(torch.cuda.max_memory_allocated(device))
                peak_reserved=int(torch.cuda.max_memory_reserved(device))
    finally:
        model.to(original_device);model.train(previous)
    a=np.asarray(durations,float)
    metadata=getattr(model,'checkpoint_metadata',{})
    identity={k:metadata[k] for k in ('kind','schema','steps','checkpoint_sha256','source_sha256','radar_mode') if k in metadata}
    return dict(status='complete',device=str(device),input_frames=len(features_list),
        measured_predictions=len(durations),repeats_per_frame=repeats,warmup_predictions=warmup,
        latency_ms=dict(mean=float(a.mean()),p50=float(np.quantile(a,.5)),
            p95=float(np.quantile(a,.95)),minimum=float(a.min()),maximum=float(a.max())),
        prediction_seconds=sum(durations)/1000.,
        gpu_peak_allocated_bytes=peak_allocated,gpu_peak_reserved_bytes=peak_reserved,
        gpu_memory_scope='PyTorch allocator peak in this process after warmup; excludes CUDA driver and other processes',
        parameter_count=sum(p.numel() for p in parameters),
        trainable_parameter_count=sum(p.numel() for p in parameters if p.requires_grad),
        scope='ordinary sensor-only predict; excludes frozen detector, IQ/FFT feature preparation, data/cache I/O, labels, matching and stage-capture hooks',
        feature_host_to_device_copy_in_predict_included=True,labels_read=False,
        cuda_synchronized_before_and_after_each_prediction=cuda,
        torch_version=str(torch.__version__),cpu_torch_threads=torch.get_num_threads(),
        model_identity=identity,model_configuration=getattr(model,'config',None),
        frozen_source_root=getattr(model,'_precision_frozen_source_root',None),
        benchmark_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--source-root',type=Path)
    p.add_argument('--kind',choices=('study','v3','v2'),default='study')
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--frames',type=int,default=12)
    p.add_argument('--split',choices=('train','val'),default='val',help='Timing only uses train/val; test evaluation has its own freeze gate')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if min(a.frames,a.threads)<1:p.error('Positive frame and thread counts required')
    if a.output.exists():raise FileExistsError('Existing benchmark result is protected')
    from .export_deployment import verify_deployment_receipt, assert_no_sample_labels
    deployment=verify_deployment_receipt(a.checkpoint,a.kind)
    torch.set_num_threads(a.threads)
    config=json.loads(a.config.read_text()) if a.config else None
    if a.kind=='study':
        if a.source_root:
            from .versions import load_frozen_study_model,frozen_study_sensor_loader
            model=load_frozen_study_model(a.source_root,a.checkpoint,config,a.device)
            loader=frozen_study_sensor_loader(model)
        else:
            from .checkpoint import load_checkpoint
            from .data import load_observation_features as loader
            model=load_checkpoint(a.checkpoint,config,a.device)
    else:
        from .data import load_observation_features as loader
        if a.kind=='v3':
            from depth_localization.checkpoint import load_checkpoint
        else:
            from .reproduce import frozen_v2_loader
            snapshot=a.source_root or ROOT/'results/training/depth_v2/source_snapshot_before_restart'
            load_checkpoint=frozen_v2_loader(snapshot).load_checkpoint
        model=load_checkpoint(a.checkpoint,config,a.device)
    config=model.config
    assert_no_sample_labels(model.checkpoint_metadata)
    from .visual_data import public_split_manifest
    _,split,split_sha=public_split_manifest(config)
    ids=split['splits'][a.split]['episode_ids']
    candidates=[(r['episode'],index) for r in split['episodes'] if r['episode'] in ids
                for index in r['selected_frame_indices']]
    selected=np.linspace(0,len(candidates)-1,min(a.frames,len(candidates)),dtype=int)
    keys=[candidates[i] for i in selected]
    features=[loader(config,episode,index,a.split) for episode,index in keys]
    report=benchmark_model(model,features,a.device,a.warmup,a.repeats)
    report.update(measured_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                  checkpoint=str(a.checkpoint.resolve()),split=a.split,frames=keys,
                  source_root=str(a.source_root.resolve()) if a.source_root else None,
                  checkpoint_kind=a.kind,command=[sys.executable,*sys.argv])
    report.update(split_manifest_sha256=split_sha,
        deployment_receipt_sha256=hashlib.sha256((a.checkpoint.parent/'deployment_receipt.json').read_bytes()).hexdigest(),
        deployment_state_tensor_sha256=deployment['state_tensor_sha256'],test_inputs_permitted=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
