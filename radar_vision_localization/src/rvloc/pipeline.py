"""Seven sensor-isolated methods, with causal per-episode state.

No labels are imported here. Normal predictions are committed before a
separate evaluation command is permitted to open supervision.
"""
from __future__ import annotations

import time

import numpy as np

from .config import resolve_method, external_path
from .data import load_manifest
from .runtime import (file_sha256, load_calibration, new_run, source_identity,
                      write_json, jsonable)


class MethodRunner:
    def __init__(self, method, config, rf, *, model=None, calibration=None, device='cpu'):
        from .tracking import KalmanBank, RadarAssociator, VisualAssociator
        self.method_id, self.method = resolve_method(method)
        self.config, self.device = config, device
        self.model = model
        if model is not None and getattr(model,'group',None) != self.method['model']:
            raise ValueError('Model architecture does not match requested method')
        self.calibration = calibration
        self.episode = None
        self.memory = None
        self.last_time = None
        self.kalman, self.visual_association, self.radar_association = None, None, None
        self.measurement_covariance = None
        self.detector = None
        if self.method['input'] == 'radar':
            from .radar import FFTDetector
            self.detector = FFTDetector(rf, config.raw.get('radar'))
        if self.method_id in (5, 6, 7):
            if calibration is None or not calibration.get('visual_params'):
                raise ValueError('Shared TRAIN-frozen visual association is required for groups 5/6/7')
            self.visual_association = VisualAssociator(params=calibration['visual_params'])
        if self.method_id == 4:
            if calibration is None or not calibration.get('radar_association'):
                raise ValueError('TRAIN-frozen radar association is required for group 4')
            self.radar_association = RadarAssociator(**calibration['radar_association'])
        if self.method['temporal'] == 'kalman':
            source = {4:'fft',5:'rgb',6:'single_frame'}[self.method_id]
            settings = (calibration or {}).get('kalman', {}).get(source)
            if not settings:
                raise ValueError('Missing TRAIN Kalman calibration for '+source)
            settings = dict(settings)
            self.measurement_covariance = np.asarray(settings.pop('measurement_covariance'), dtype=float)
            self.kalman = KalmanBank(**settings)
        if self.model is not None:
            self.model.to(device).eval()

    def reset(self, episode):
        self.episode, self.memory, self.last_time = episode, None, None
        for state in (self.kalman, self.visual_association, self.radar_association):
            if state is not None: state.reset()

    def step(self, observation):
        if observation.episode_id != self.episode: self.reset(observation.episode_id)
        if self.last_time is not None and not np.isclose(observation.physical_time_s-self.last_time, 1., atol=1e-8):
            raise ValueError('Frames must be continuous and use physical_time_s at 1 s intervals')
        started = time.perf_counter()
        features = observation.features
        ids, depths, quality, scores = None, None, None, None
        if self.detector is not None:
            detections = self.detector.detect(observation.iq, station=observation.calibration,
                                             observer=observation.observer_scatter_reference_position)
            positions = detections.positions.copy()
            valid = detections.valid.copy()
            scores, quality = detections.scores, detections.quality
            extra = dict(observer_near=detections.observer_near, radar=detections.metadata)
            if self.radar_association is not None:
                ids = self.radar_association.step(observation.physical_time_s, positions)
        else:
            import torch
            if self.model is None: raise ValueError('A trained checkpoint is required; random weights are never used for inference')
            if self.visual_association is not None:
                ids = self.visual_association.step(observation.frame_idx, observation.physical_time_s,
                    features['bbox_xyxy'], features['detector_roi_features'], features['detection_scores'])
            model_features = dict(features)
            if self.method_id == 7:
                model_features.update(episode_id=observation.episode_id, frame_index=observation.frame_idx,
                    timestamp_s=observation.physical_time_s, member_ids=ids.tolist())
            with torch.inference_mode():
                result = self.model(model_features, memory=self.memory)
                if self.method_id == 7: self.memory = result['state']
            positions = result['member_positions'].detach().cpu().numpy()
            valid = result['valid_mask'].detach().cpu().numpy().astype(bool)
            depths = result['depths'].detach().cpu().numpy()
            scores = np.asarray(features['detection_scores'])
            extra = dict(bbox_xyxy=features['bbox_xyxy'],
                visual_depth_mean_m=result['visual_depth_mean_m'].detach().cpu().numpy(),
                visual_depth_std_m=result['visual_depth_std_m'].detach().cpu().numpy(),
                ray_origins=features['ray_origins'],ray_directions=features['ray_directions'],
                ray_depth_bounds=features['ray_depth_bounds'],range_scale_m=features['range_scale_m'])
            diagnostic=result.get('sampling_diagnostics',{})
            keep=('query_rounds','candidate_depths_m','candidate_offsets_m','ra_valid','re_valid',
                  'geometry_valid','usable','weights','candidate_multipliers','normalized_offsets',
                  'ranges_m','azimuth_rad','elevation_rad','evidence_available')
            extra['radar_query']={key:(value.detach().cpu().numpy() if hasattr(value,'detach') else value)
                                  for key,value in diagnostic.items() if key in keep}
            if self.visual_association is not None:
                extra['association'] = self.visual_association.last_result
        raw_positions = positions.copy()
        velocities = None
        if self.kalman is not None:
            positions = self.kalman.update(observation.physical_time_s, ids, positions,
                                         covariance=self.measurement_covariance, valid=valid)
            velocities = self.kalman.last_velocities
        if self.device.startswith('cuda'):
            import torch
            torch.cuda.synchronize()
        self.last_time = observation.physical_time_s
        return jsonable(dict(episode_id=observation.episode_id, frame_idx=observation.frame_idx,
            physical_time_s=observation.physical_time_s, positions=positions,
            raw_positions=raw_positions, valid=valid,
            member_ids=ids if ids is not None else [None]*len(positions), depths=depths,
            scores=scores, radar_quality=quality, velocities=velocities,
            runtime_s=time.perf_counter()-started, source_hashes=observation.source_hashes,
            architecture=getattr(self.model, 'config', {}).get('architecture') if self.model is not None else 'fft_power_peaks_v1',
            **extra))


def infer(config, method, *, split='dev', device='cpu', manifest=None, model=None, output_dir=None):
    if split not in ('train', 'dev'): raise ValueError('Only TRAIN/DEV are permitted; historical TEST is reserved')
    if output_dir is not None:
        output_dir=external_path(output_dir,field='inference output_dir')
        if output_dir.exists():raise ValueError('Inference output directory already exists')
    if manifest is None:
        if config.manifest is None: raise ValueError('Dataset pending: set a verified external manifest after data repair')
        manifest = load_manifest(config.manifest)
    method_id, description = resolve_method(method)
    calibration = load_calibration(config, manifest, required=method_id in (4,5,6,7))
    checkpoint_hash = None
    if description['model'] is not None and model is None:
        from .models import load_checkpoint
        checkpoint = config.checkpoint(description['model'])
        if checkpoint is None: raise ValueError('No checkpoint configured for '+description['model'])
        model, payload = load_checkpoint(checkpoint, expected_group=description['model'], device=device)
        checkpoint_hash = file_sha256(checkpoint)
        metadata = payload.get('metadata', {})
        if metadata.get('synthetic_only') or metadata.get('optimizer_updates',0) <= 0 or metadata.get('completed_epochs',0) <= 0:
            raise ValueError('Inference requires an actually trained checkpoint with training provenance')
        if metadata.get('manifest_sha256') != manifest.sha256:
            raise ValueError('Checkpoint data identity differs from the corrected manifest')
        if metadata.get('center_contract') != 'observer_scatter_at_camera_optical_center':
            raise ValueError('Checkpoint lacks co-centered training provenance')
        if (metadata.get('observer_position_reference') != 'camera_optical_center'
                or metadata.get('body_navigation_is_model_input') is not False):
            raise ValueError('Checkpoint must certify camera optical-centre inputs and exclude body navigation from neural features')
        if method_id == 7 and metadata.get('calibration_sha256') != calibration['_sha256']:
            raise ValueError('World-model checkpoint was trained with different visual association calibration')
    runner = MethodRunner(method_id, config, manifest.rf, model=model, calibration=calibration, device=device)
    frames, seen = [], set()
    start = time.perf_counter()
    observation_options = {'modality':description['input']}
    if config.raw.get('training',{}).get('prepared_only',False) and description['input']=='rgb_radar':
        observation_options['prepared_only']=True
    print(f'INFER G{method_id} START split={split} device={device} prepared_only={observation_options.get("prepared_only",False)}',flush=True)
    for observation in manifest.observations(split, **observation_options):
        key = (observation.episode_id, observation.frame_idx)
        if key in seen: raise ValueError('Duplicate input frame')
        seen.add(key)
        frames.append(runner.step(observation))
        if len(frames)%100 == 0:
            elapsed=time.perf_counter()-start
            print(f'INFER G{method_id} split={split} frames={len(frames)} elapsed_s={elapsed:.1f} frames_s={len(frames)/elapsed:.2f}',flush=True)
    if not frames: raise ValueError('No frames available for inference')
    elapsed = time.perf_counter()-start
    output = new_run(config, description['name']) if output_dir is None else output_dir
    predictions = dict(schema='rvloc.predictions.v1', method_id=method_id, method=description,
        split=split, manifest_sha256=manifest.sha256, config_sha256=config.sha256,
        source=source_identity(), checkpoint_sha256=checkpoint_hash,
        neural_architecture='single_query_depth_v3' if description['model'] is not None else None,
        calibration_sha256=calibration.get('_sha256') if calibration else None,
        coordinate_frame='base_local', normal_inference=True, labels_opened=False,
        evaluation_contract=dict(matching='same_per_frame_one_to_one_euclidean',
            match_max_distance_m=config.raw.get('evaluation',{}).get('match_max_distance_m')),
        frames=frames, cost=dict(wall_s=elapsed, frames=len(frames),
            serial_model_total_s=sum(x['runtime_s'] for x in frames),
            throughput_frames_s=len(frames)/elapsed, device=device, parallel_workers=1,
            includes_asset_io_and_validation=True,
            raw_iq_fft_included=description['input']=='radar',
            rgb_detector_included=False,
            input_mode='raw_iq' if description['input']=='radar' else
                'authenticated_prepared_roi_ra_re' if observation_options.get('prepared_only') else
                'prepared_roi_raw_iq' if description['input']=='rgb_radar' else 'prepared_roi',
            timing_scope='localizer and declared input loading; frozen RGB detector not timed'))
    path = write_json(output/'predictions.json', predictions)
    write_json(output/'reproduction_receipt.json', dict(status='inference_completed',
        predictions=str(path), predictions_sha256=file_sha256(path),
        manifest_sha256=manifest.sha256, config_sha256=config.sha256,
        checkpoint_sha256=checkpoint_hash, source=predictions['source'], cost=predictions['cost']))
    print(f'INFER G{method_id} COMPLETE split={split} frames={len(frames)} elapsed_s={elapsed:.1f} predictions={path}',flush=True)
    return dict(predictions=str(path), frames=len(frames), status='inference_completed')
