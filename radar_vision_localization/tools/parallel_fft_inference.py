"""Exact native FFT detection across independent TRAIN/DEV frames on CPU."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from rvloc.config import ExperimentConfig, resolve_method, external_path
from rvloc.data import load_manifest, load_observation, _verify_center
from rvloc.radar import FFTDetector
from rvloc.runtime import file_sha256, source_identity, write_json, jsonable


def detect_frame(manifest, frame, detector):
    observation=load_observation(manifest,frame,modality='radar',_center_verified=True)
    started=time.perf_counter()
    detected=detector.detect(observation.iq,station=observation.calibration,
                             observer=observation.observer_scatter_reference_position)
    return jsonable(dict(episode_id=observation.episode_id,frame_idx=observation.frame_idx,
        physical_time_s=observation.physical_time_s,positions=detected.positions,
        raw_positions=detected.positions,valid=detected.valid,member_ids=[None]*len(detected),
        depths=None,scores=detected.scores,radar_quality=detected.quality,velocities=None,
        runtime_s=time.perf_counter()-started,source_hashes=observation.source_hashes,
        architecture='fft_power_peaks_v1',observer_near=detected.observer_near,radar=detected.metadata))


def infer_fft(config_path,split,output_dir,*,workers=16,manifest=None,detector=None):
    if split not in ('train','dev'):raise ValueError('Historical TEST is reserved')
    if type(workers) is not int or not 1<=workers<=64:raise ValueError('workers must be 1..64')
    config=ExperimentConfig.load(config_path)
    output=external_path(output_dir,field='parallel FFT output')
    if output.exists():raise FileExistsError('Choose a new FFT output directory')
    if manifest is None:manifest=load_manifest(config.manifest)
    _verify_center(manifest)
    if detector is None:detector=FFTDetector(manifest.rf,config.raw.get('radar'))
    records=list(manifest.frames(split))
    if not records:raise ValueError('No frames available')
    started=time.perf_counter();frames=[]
    # Executor.map preserves declared episode/frame order. Each task uses only
    # its current IQ; detector contains immutable axes/settings, no history.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for value in executor.map(lambda frame:detect_frame(manifest,frame,detector),records):
            frames.append(value)
            if len(frames)%250==0:
                elapsed=time.perf_counter()-started
                print(f'FFT_PARALLEL split={split} frames={len(frames)}/{len(records)} workers={workers} elapsed_s={elapsed:.1f} fps={len(frames)/elapsed:.2f}',flush=True)
    elapsed=time.perf_counter()-started
    _,method=resolve_method(1)
    cost=dict(wall_s=elapsed,frames=len(frames),serial_model_total_s=sum(x['runtime_s'] for x in frames),
        serial_model_total_s_is_concurrent_worker_sum=True,serial_latency_measured=False,
        throughput_frames_s=len(frames)/elapsed,device='cpu',parallel_workers=workers,
        includes_asset_io_and_validation=True,raw_iq_fft_included=True,rgb_detector_included=False,
        input_mode='raw_iq',timing_scope='measured parallel raw-IQ throughput; not sequential single-frame latency')
    predictions=dict(schema='rvloc.predictions.v1',method_id=1,method=method,split=split,
        manifest_sha256=manifest.sha256,config_sha256=config.sha256,source=source_identity(),
        checkpoint_sha256=None,neural_architecture=None,calibration_sha256=None,
        coordinate_frame='base_local',normal_inference=True,labels_opened=False,
        evaluation_contract=dict(matching='same_per_frame_one_to_one_euclidean',
            match_max_distance_m=config.raw.get('evaluation',{}).get('match_max_distance_m')),
        frames=frames,cost=cost)
    path=write_json(output/'predictions.json',predictions)
    write_json(output/'reproduction_receipt.json',dict(status='inference_completed',predictions=str(path),
        predictions_sha256=file_sha256(path),manifest_sha256=manifest.sha256,config_sha256=config.sha256,
        checkpoint_sha256=None,source=predictions['source'],tool_sha256=file_sha256(__file__),cost=cost))
    return dict(status='inference_completed',predictions=str(path),frames=len(frames),cost=cost)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--split',choices=('train','dev'),required=True)
    parser.add_argument('--output-dir',required=True);parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args();print(json.dumps(infer_fft(args.config,args.split,args.output_dir,workers=args.workers),indent=2))

if __name__=='__main__':main()
