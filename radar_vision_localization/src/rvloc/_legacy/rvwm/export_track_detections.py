"""Export the configured frozen RGB detector for explicitly bound phases.

Run this module with the detector's Python environment. Each device processes a
separate set of episodes. An episode file appears only after all its RGBs have
been processed. Reruns verify the configuration and completed episode files.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from .phase_selection import configured_phases,inventory_phases,phase_label

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_line(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n'


def publish(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        f.write(data)
    Path(f.name).replace(path)


def frozen_inputs(config_path=None):
    """Read the explicitly selected sensor/export configuration.

    The default remains the project's current config.json. Every worker must
    receive the same path; a coordinator's alternate data root must not be
    silently replaced with the default in its child processes.
    """
    config_path = Path(config_path or PROJECT / 'config.json').expanduser().resolve()
    config = json.loads(config_path.read_text())
    split_path = Path(config['visual_split_manifest'])
    split_bytes = split_path.read_bytes()
    expected_split_sha = config.get('split_manifest_sha256')
    if expected_split_sha and sha(split_bytes) != expected_split_sha:
        raise ValueError('Frozen detector split manifest SHA256 mismatch')
    split = json.loads(split_bytes)
    partition = {ep: name for name, part in split['splits'].items() for ep in part['episode_ids']}
    episodes = {ep['episode']: ep for ep in split['episodes']}
    root = Path(config['dataset_root']).resolve()
    if set(partition) != set(episodes) or set(episodes) != {p.name for p in (root/'data').glob('episode_*') if p.is_dir()}:
        raise ValueError('Archive episodes and frozen detector split disagree')
    phases=configured_phases(config)
    if phases!=inventory_phases(split):
        raise ValueError('Detector selected phases differ from the frozen source inventory; rebuild binding/export explicitly')
    from .partitions import FAMILY_POLICY,EPISODE_SAMPLING,active_splits
    family_policy=split.get('request',{}).get('split_policy')==FAMILY_POLICY
    for episode in episodes:
        path=root/'data'/episode/'metadata.json'
        if path.is_file() and json.loads(path.read_text()).get('dataset_split_policy')==FAMILY_POLICY and not family_policy:
            raise ValueError('Scene-family source cannot be exported with a legacy episode split')
    if family_policy:
        source=SimpleNamespace(root=root,episodes=sorted(episodes),verify_sources=True,
            group_key=lambda e:episodes[e].get('frames_sha256'))
        active_splits(source,dict(config,split_manifest=str(split_path),visual_split_manifest=str(split_path)))
    contract = dict(schema='rvwm_track_rgb_detections_v1' if phases==('track',) else 'rvwm_selected_phase_rgb_detections_v2', dataset_root=str(root), phase=phase_label(phases),
        split_manifest=str(split_path), split_manifest_sha256=sha(split_bytes),
        box_format='xyxy in original RGB pixels', center_definition='bbox midpoint, without ROI refinement',
        exporter_sha256=sha(Path(__file__).read_bytes()))
    if config.get('detector_backend','dinov3_fpn')!='dinov3_fpn':
        raise ValueError('Frozen RGB export requires the trained DINOv3 detector')
    from .detector_contract import detector_contract
    contract.update(detector_contract(config))
    if phases!=('track',):contract['selected_phases']=list(phases)
    if family_policy:
        contract.update(dataset_split_policy=FAMILY_POLICY,training_sampling_policy=EPISODE_SAMPLING,
            family_partition=split['family_partition'],
            scene_family_ids={e:row['scene_family_id'] for e,row in episodes.items()})
    return config, episodes, partition, contract


def read_rgb(path):
    import numpy as np
    from PIL import Image
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        rgb = np.array(image.convert('RGB'))
    return rgb, sha(data)


def export_episode(root, ep, split, expected, detector, out, readers):
    directory = root / 'data' / ep
    raw = (directory/'frames.jsonl').read_bytes()
    source_sha = sha(raw)
    if source_sha != expected['frames_sha256'] or sha((directory/'metadata.json').read_bytes()) != expected['metadata_sha256']:
        raise ValueError(f'Episode does not match the frozen source: {ep}')
    # Only phase, frame index and relative RGB path are used to drive inference.
    phases=tuple(expected.get('selected_phases',['track']))
    frames = [(r['frame_idx'], r['rgb_path'],r['phase']) for r in map(json.loads,raw.splitlines()) if r['phase'] in phases]
    if len(frames) != expected['frame_count'] or len({i for i,_,_ in frames}) != len(frames):
        raise ValueError(f'Unexpected selected-phase frame inventory for {ep}')
    if 'selected_frame_indices' in expected and [i for i,_,_ in frames]!=expected['selected_frame_indices']:
        raise ValueError(f'Selected-phase frame indices differ from frozen inventory for {ep}')
    for _, name,_ in frames:
        if not (directory/name).resolve().is_relative_to(directory.resolve()):
            raise ValueError('RGB path outside episode')
    dest = out/split/(ep+'.jsonl')
    receipt = out/'receipts'/(ep+'.json')
    if dest.exists() and receipt.exists():
        stamp = json.loads(receipt.read_text())
        if sha(dest.read_bytes()) != stamp['prediction_sha256'] or stamp['source_frames_sha256'] != source_sha:
            raise ValueError(f'Changed completed output: {ep}')
        return stamp
    futures = [readers.submit(read_rgb,directory/name) for _,name,_ in frames]
    rows = []; count = 0
    for (index,name,phase), future in zip(frames,futures):
        rgb, rgb_sha = future.result()
        dets = []
        for i, detection in enumerate(detector.detect(rgb)):
            x0,y0,x1,y1 = detection.bbox_xyxy
            dets.append(dict(detection_id=i,class_id=int(detection.class_id),bbox_xyxy=[x0,y0,x1,y1],
                             center_px=[(x0+x1)/2,(y0+y1)/2],score=float(detection.score)))
        rows.append(json_line(dict(episode_id=ep,frame_idx=index,split=split,phase=phase,rgb_path=name,
                        rgb_sha256=rgb_sha,rgb_content_sha256=sha(memoryview(rgb).cast('B')),
                        source_frames_sha256=source_sha,image_size_wh=[rgb.shape[1],rgb.shape[0]],detections=dets)))
        count += len(dets)
    data = ''.join(rows)
    publish(dest,data)
    stamp = dict(episode_id=ep,split=split,frames=len(rows),detections=count,source_frames_sha256=source_sha,
                 prediction_sha256=sha(data.encode()))
    publish(receipt,json_line(stamp))
    return stamp


def worker(out, slot, workers, config_path=None):
    import torch
    sys.path.insert(0,str(WORKSPACE))
    config, episodes, partition, contract = frozen_inputs(config_path)
    torch.set_num_threads(1)
    manifest_path=out/'manifest.json'
    if not manifest_path.is_file():raise ValueError('DINO export worker requires its coordinator manifest')
    declared=json.loads(manifest_path.read_text())
    if any(declared.get(key)!=value for key,value in contract.items()):
        raise ValueError('DINO export worker/output detector identity mismatch')
    from uav_detection.dinov3_detector import DINOv3Detector
    detector=DINOv3Detector(contract['weights'],device='cuda:0',confidence=contract['confidence'])
    detector.model.requires_grad_(False).eval()
    actual_digest=detector.metadata['weights_sha256']
    for key in ('architecture','confidence'):
        if detector.metadata.get(key)!=contract[key]:raise ValueError('Unexpected detector '+key)
    if detector.metadata.get('split_manifest_sha256')!=contract['detector_split_manifest_sha256']:
        raise ValueError('Detector weights belong to another training partition')
    if actual_digest != contract['weights_sha256']:
        raise ValueError('Unexpected detector weights')
    publish(out/'workers'/f'{slot}.json',json_line(dict(detector.metadata,export_detector_contract=contract)))
    started = time.monotonic(); total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as readers:
        for i, ep in enumerate(sorted(episodes)[slot::workers],1):
            stamp = export_episode(Path(config['dataset_root']),ep,partition[ep],episodes[ep],detector,out,readers)
            total += stamp['frames']
            print(f"{i} episodes, {total} frames, {time.monotonic()-started:.1f}s", flush=True)


def coordinator(args):
    config_path = Path(getattr(args, 'config', None) or PROJECT / 'config.json').expanduser().resolve()
    _, episodes, partition, contract = frozen_inputs(config_path)
    out = args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = out/'manifest.json'
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            if any(previous.get(k)!=v for k,v in contract.items()):
                raise ValueError('Output directory belongs to a different run')
        elif any(p.name!='run.lock' for p in out.iterdir()):
            raise ValueError('Output directory has unrelated files')
        devices=args.devices.split(',')
        if len(devices)!=len(set(devices)) or any(not d for d in devices):
            raise ValueError('Specify unique GPU indices or UUIDs')
        manifest_values=dict(**contract,complete=False,devices=devices,expected_episodes=len(episodes),
                             expected_frames=sum(e['frame_count'] for e in episodes.values()))
        publish(manifest,json_line(manifest_values)); (out/'workers').mkdir(exist_ok=True)
        started=time.monotonic();processes=[]
        try:
            for slot, device in enumerate(devices):
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                cmd=[sys.executable,'-m','rvwm.export_track_detections','--config',str(config_path),
                     '--output',str(out),'--slot',str(slot),'--workers',str(len(devices))]
                with (out/'workers'/f'{slot}.log').open('a') as log:
                    processes.append(subprocess.Popen(cmd,env=env,cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT))
            while any(p.poll() is None for p in processes):
                for slot,p in enumerate(processes):
                    if p.returncode not in (None,0):
                        raise RuntimeError(f'Worker {slot} failed; see {out}/workers/{slot}.log')
                time.sleep(2)
            if any(p.returncode!=0 for p in processes):
                raise RuntimeError('A worker failed; inspect worker logs')
        finally:
            for p in processes:
                if p.poll() is None:p.terminate()
            for p in processes:p.wait()
        counts={name:dict(episodes=0,frames=0,detections=0) for name in set(partition.values())}
        for ep in sorted(episodes):
            stamp=json.loads((out/'receipts'/(ep+'.json')).read_text()); part=partition[ep]
            target=out/part/(ep+'.jsonl')
            if (stamp['episode_id']!=ep or stamp['split']!=part or stamp['frames']!=episodes[ep]['frame_count']
                or sha(target.read_bytes())!=stamp['prediction_sha256']):raise ValueError(f'Incomplete export for {ep}')
            counts[part]['frames']+=stamp['frames'];counts[part]['detections']+=stamp['detections'];counts[part]['episodes']+=1
        manifest_values.update(complete=True,splits=counts,elapsed_seconds=time.monotonic()-started)
        publish(manifest,json.dumps(manifest_values,indent=2)+'\n');print(json.dumps(counts),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'config.json')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--devices',default='0')
    parser.add_argument('--slot',type=int);parser.add_argument('--workers',type=int)
    args=parser.parse_args(argv)
    if args.slot is None:coordinator(args)
    else:worker(args.output,args.slot,args.workers,args.config)
    return 0


if __name__=='__main__':raise SystemExit(main())
