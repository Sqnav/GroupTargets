"""Single-frame training of continuous joint ray depths from RA/RE spectra.

Sensor feature preparation finishes before labels are opened. Checkpoint
selection calls the deployed observation-only proposer, never the supervised
latent posterior. All preprocessing caches are immutable provenance bindings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import tempfile
import shutil
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist

from .coordinates import FixedStationFrame
from .data import Dataset, sha256
from .group_model import GroupSceneModel, GROUP_SCHEMA, FORMATIONS, checkpoint, match_ray_labels
from .group_observation import describe_observation, feature_digest, observation_cache_identity, validate_features
from .ray_scoring import select_scene_modes,observation_rejection_reason
from .prior_validation import frame_seed
from .validation import write_json
from .visual_refinement import ExportedRGBFrontend
from .partitions import active_splits,partition_receipt,EPISODE_SAMPLING
from .single_frame import require_single_frame, validate_model_features
from .phase_selection import configured_phases,phase_label,required_phase,selected_frame_indices

PROJECT = Path(__file__).resolve().parents[1]


def strict_splits(dataset, config):
    """Use the same complete source partition as the frozen detector."""
    return active_splits(dataset,config)


def _frontend(config, split, device):
    frontend = ExportedRGBFrontend(config,split=split,device=device)
    frontend.metadata['appearance_used_by_world_model'] = False
    return frontend


def _source_hashes():
    names = ('group_observation.py','group_model.py','group_training.py','partitions.py','data.py','vision.py',
             'visual_refinement.py','coordinates.py','geometry.py','types.py','ray_scoring.py',
             'single_frame.py','phase_selection.py',
             'detector_contract.py','dinov3_worker.py')
    return {name:sha256(Path(__file__).with_name(name)) for name in names}


def cached_features(observation, detections, config, frontend_metadata, cache_root):
    # Validate even on a hit, and bind actual arrays rather than trusting a
    # caller's possibly stale source-file receipt.
    identity = observation_cache_identity(observation)
    metadata = {k:v for k,v in frontend_metadata.items() if k != 'worker_log'}
    # Model/optimizer settings do not change this per-frame preprocessing.
    common_bins=config.get('ra_re_angle_bins',65)
    feature_config={key:config.get(key,default) for key,default in (
        ('ra_re_window','rectangular'),('ra_re_angle_bins',65),
        ('architecture_schema',GROUP_SCHEMA),('reference_range_m',500.),
        ('ra_re_azimuth_bins',common_bins),('ra_re_elevation_bins',common_bins),
        ('ra_re_spatial_oversampling',4),('ra_re_range_chunk',4),('ra_re_angle_limit_deg',85.))}
    receipt = dict(schema='base-ray-ra-re-feature-cache-v3',raw_hashes=observation.source_hashes,
        episode_id=observation.episode_id,frame_idx=observation.frame_idx,
        configuration=feature_config,frontend=metadata,source_sha256=_source_hashes(),
        sensor_identity=identity,
        detections=[dict(uv=np.asarray(d.uv).tolist()) for d in detections])
    key = hashlib.sha256(json.dumps(receipt,sort_keys=True).encode()).hexdigest()
    path = Path(cache_root)/(key+'.pt');path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        stored = torch.load(path,map_location='cpu',weights_only=False)
        if stored['receipt'] != receipt or feature_digest(stored['features']) != stored['feature_digest']:
            raise ValueError('Group feature cache provenance mismatch')
        validate_features(stored['features'])
        return stored['features'],dict(path=str(path.resolve()),sha256=sha256(path),hit=True)
    features = describe_observation(observation,detections,config)
    # Concurrent readers must never observe a partially written tensor file.
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as temporary:
        temporary_path=Path(temporary.name)
    try:
        torch.save(dict(receipt=receipt,features=features,feature_digest=feature_digest(features)),temporary_path)
        os.replace(temporary_path,path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return features,dict(path=str(path.resolve()),sha256=sha256(path),hit=False)


def _supervision(dataset, episode, index, frame):
    targets = dataset.targets(episode,index)
    metadata = dataset._load(episode)[2]  # training/evaluation labels only
    name = metadata.get('group_structure',metadata.get('scenario_spec',{}).get('formation_type'))
    # Position supervision does not require a formation-category label.
    return dict(positions=frame.to_local_positions(targets.positions).astype(np.float32),
        formation_index=FORMATIONS.index(name) if name in FORMATIONS else None,
        formation_type=name or 'unknown')


def frame_indices(dataset, episode, count, config=None):
    phases=configured_phases(config or {})
    if phases==('track',) and count>0:
        return dataset.sample_track_frames(episode,count)
    return selected_frame_indices(dataset,episode,phases,count)


def validate_training_phase(config):
    return configured_phases(config)


def prepare(dataset, config, ids, split, count, device, output, rank_status=None):
    """Prepare every selected current frame before opening its separate labels."""
    phases=validate_training_phase(config)
    window=require_single_frame(config)
    frontend = _frontend(config,split,device);samples=[];manifest=[]
    try:
        for number,episode in enumerate(ids):
            station_frame=None
            for index in frame_indices(dataset,episode,count,config):
                raw=dataset.observations(episode,index,required_phase=required_phase(phases))
                if station_frame is None:station_frame=FixedStationFrame.from_observation(raw)
                adapted=station_frame.observations(raw)  # validates one fixed station
                detections=frontend.detect(raw)
                features,cache=cached_features(adapted,detections,config,frontend.metadata,
                    config.get('group_feature_cache',PROJECT/'cache/features'))
                validate_model_features(features,config)
                recorded_phase=next(phase for phase in phases if index in dataset.frames_in_phase(episode,phase))
                row = dict(episode_id=episode,frame_idx=index,split=split,phase=recorded_phase,
                    hashes=dict(raw.source_hashes),detected_count=len(detections),
                    frame_interval_s=1.,feature_status='no_detections',configured_window=window)
                if detections:
                    labels = _supervision(dataset,episode,index,station_frame)
                    complete = len(detections)==len(labels['positions'])
                    matching=match_ray_labels(features,labels['positions'],config)
                    rejection=observation_rejection_reason(features,config)
                    usable=matching['diagnostics']['supervision_usable'] and rejection is None
                    if usable:
                        sample=dict(features=features,**labels,episode_id=episode,frame_idx=index,
                            detections=detections,complete_detection_cardinality=complete)
                        samples.append(sample)
                    row.update(feature_status='prepared' if usable else rejection or 'no_valid_supervision',cache=cache,feature_digest=feature_digest(features),
                        label_count=len(labels['positions']),formation_type=labels['formation_type'],
                        complete_detection_cardinality=complete,supervision=matching['diagnostics'])
                else:
                    row['label_count']=len(dataset.targets(episode,index).positions)
                manifest.append(row)
                if rank_status is not None and (len(manifest)%10==0):
                    path,initial=rank_status
                    write_json(path,dict(initial,stage='preparing',prepared_frames=len(manifest),
                        prepared_groups=len(samples),updated_unix=time.time()))
            if (number+1)%8==0 or number+1==len(ids):
                print(json.dumps(dict(stage='prepare_group_features',split=split,episodes=number+1,
                    total_episodes=len(ids),frames=len(manifest))),flush=True)
        write_json(Path(output)/f'{split}_features.json',dict(schema=config.get('architecture_schema',GROUP_SCHEMA),split=split,episodes=ids,phase=phase_label(phases),selected_phases=list(phases),
            configured_window=window,input_policy='one current observation from configured recorded phases',
            frontend=frontend.metadata,frames=manifest,source_sha256=_source_hashes(),
            no_detection_frames=sum(r['feature_status']=='no_detections' for r in manifest)))
        return samples,manifest,dict(frontend.metadata)
    finally:
        frontend.close()


def detection_dropout(features, rng, probability=.15):
    """Drop visual rays only; retain the same mixed RA/RE observation."""
    n = len(features['ray_directions'])
    valid = np.asarray(features['ray_valid'],bool)
    if n<2 or not valid.any() or rng.random()>=probability:return features,False
    retained = rng.random(n) > .2
    if not (retained & valid).any():retained[int(rng.choice(np.flatnonzero(valid)))]=True
    augmented = dict(features)
    for name in ('ray_origins','ray_directions','ray_valid','ray_depth_bounds'):
        augmented[name]=np.asarray(features[name])[retained].copy()
    return augmented,int(retained.sum())!=n


def supervision_checked_dropout(features, positions, rng, probability, config):
    """Cancel augmentation if it removes every accepted supervised detection."""
    reduced,dropped=detection_dropout(features,rng,probability)
    if dropped and not match_ray_labels(reduced,positions,config)['diagnostics']['supervision_usable']:
        return features,False,True
    return reduced,dropped,False


def _angular_label_pairs(features, truth, config=None):
    """Evaluation-only gated correspondence, independent of predicted depths."""
    assignment=match_ray_labels(features,truth,config)
    return assignment['rows'],assignment['cols']


def evaluate_prior(model, samples, manifest, seed, hypotheses=32, *, allow_untrained_validation=False):
    """Evaluate raw proposals, published modes and the actual primary separately.

    Run exactly the deployment mode filter before labels score any candidate.
    Oracle candidate coverage is a diagnostic, never primary output accuracy.
    """
    previous_training=model.training
    config=getattr(model,'config',{})
    require_single_frame(config)
    formation_trained=bool(model.trained and config.get('formation_head_trained',False))
    formation_status='trained_auxiliary_head' if formation_trained else 'not_trained_or_disabled'
    model.eval();rows=[]
    try:
        by_key={(s['episode_id'],s['frame_idx']):s for s in samples}
        for item in manifest:
            episode,index=item['episode_id'],item['frame_idx'];sample=by_key.get((episode,index))
            if sample is None and 'label_count' not in item:
                raise ValueError('Validation manifest needs label_count for rejected or missing-detection frames')
            row=dict(episode_id=episode,frame_idx=index,center_covered_5m=False,
                whole_scene_covered_5m=False,best_scene_rmse_m=None,best_center_error_m=None,
                primary_center_error_m=None,primary_formation_correct=None,formation_head_status=formation_status,
                primary_detected_subset_center_error_m=None,primary_max_member_error_m=None,
                primary_member_errors_m=[],primary_successful_members_5m=0,
                primary_whole_scene_success_5m=False,primary_hypothesis_index=None,
                published_whole_scene_covered_5m=False,published_center_covered_5m=False,
                published_mode_count=0,published_hypothesis_indices=[],candidate_decisions=[],
                truth_member_count=int(item.get('label_count',0)),
                detection_count=item['detected_count'],candidate_count=0,uncertainty_calibrated=False)
            if sample is not None:
                row['truth_member_count']=len(sample['positions'])
                validate_model_features(sample['features'],config)
                rejection=observation_rejection_reason(sample['features'],config)
                row['observation_status']=rejection or 'valid_uncalibrated'
                if rejection is not None:
                    rows.append(row)
                    continue
                scenes=model.propose(sample['features'],hypotheses,
                    np.random.default_rng(frame_seed(seed,episode,index)),
                    allow_untrained_validation=allow_untrained_validation)
                selected=select_scene_modes(sample['features'],scenes,config)
                published={s['hypothesis_index'] for s in selected['modes']}
                primary_index=selected['modes'][0]['hypothesis_index'] if selected['modes'] else None
                row.update(published_mode_count=len(published),published_hypothesis_indices=sorted(published),
                    primary_hypothesis_index=primary_index,
                    candidate_decisions=selected['diagnostics']['candidate_decisions'])
                truth=np.asarray(sample['positions'],float)
                row['truth_member_count']=len(truth)
                assignment=match_ray_labels(sample['features'],truth,config)
                pair_rows,pair_cols=assignment['rows'],assignment['cols']
                row['correspondence']=assignment['diagnostics']
                errors=[]
                with torch.no_grad():
                    for k,scene in enumerate(scenes):
                        positions=np.asarray(scene['member_positions'],float)
                        valid=np.asarray(scene.get('valid_mask',sample['features']['ray_valid']),bool)
                        if positions.shape!=(len(sample['features']['ray_origins']),3) or valid.shape!=(len(positions),):
                            continue
                        keep=valid[pair_rows]&np.isfinite(positions[pair_rows]).all(axis=-1)
                        a,b=pair_rows[keep],pair_cols[keep]
                        if not len(a):continue
                        distance=np.linalg.norm(positions[a]-truth[b],axis=-1)
                        rmse=float(np.sqrt(np.mean(distance**2)))
                        max_error=float(distance.max())
                        complete=len(a)==len(positions)==len(truth) and bool(valid.all())
                        subset_center_error=float(np.linalg.norm(positions[a].mean(0)-truth[b].mean(0)))
                        center_error=subset_center_error if complete else None
                        errors.append(dict(hypothesis=k,center_error_m=center_error,
                            detected_subset_center_error_m=subset_center_error,
                            matched_scene_rmse_m=rmse,max_member_error_m=max_error,
                            complete_cardinality=complete,matched_count=len(a),published=k in published))
                        if k==primary_index:
                            row.update(primary_center_error_m=center_error,
                                primary_detected_subset_center_error_m=subset_center_error,
                                primary_max_member_error_m=max_error,
                                primary_member_errors_m=distance.tolist(),
                                primary_successful_members_5m=int(np.sum(distance<=5.)),
                                primary_whole_scene_success_5m=bool(complete and max_error<=5.))
                            label=sample.get('formation_index')
                            if formation_trained and label is not None and hasattr(model,'formation_logits'):
                                row['primary_formation_correct']=int(model.formation_logits(model._tensor(scene['latent'])).argmax())==label
                row.update(candidate_count=len(scenes),candidates=errors,
                    center_covered_5m=any(e['complete_cardinality'] and e['center_error_m']<=5. for e in errors),
                    whole_scene_covered_5m=any(e['complete_cardinality'] and e['max_member_error_m']<=5. for e in errors),
                    published_center_covered_5m=any(e['published'] and e['complete_cardinality'] and e['center_error_m']<=5. for e in errors),
                    published_whole_scene_covered_5m=any(e['published'] and e['complete_cardinality'] and e['max_member_error_m']<=5. for e in errors),
                    best_center_error_m=min((e['center_error_m'] for e in errors if e['complete_cardinality']),default=None),
                    best_scene_rmse_m=min((e['matched_scene_rmse_m'] for e in errors if e['complete_cardinality']),default=None))
            rows.append(row)
    finally:
        model.train(previous_training)
    best=[r['best_scene_rmse_m'] for r in rows if r['best_scene_rmse_m'] is not None]
    center=[r['primary_center_error_m'] for r in rows if r['primary_center_error_m'] is not None]
    classified=[r['primary_formation_correct'] for r in rows if r['primary_formation_correct'] is not None]
    n=max(len(rows),1)
    member_errors=[e for r in rows for e in r['primary_member_errors_m']]
    truth_count=sum(r['truth_member_count'] for r in rows)
    summary=dict(frames=len(rows),center_covered_5m=sum(r['center_covered_5m'] for r in rows),
        configured_window=require_single_frame(config),input_policy='one current observation; labels separate',
        deployment_sampling_seed=int(seed),
        whole_scene_covered_5m=sum(r['whole_scene_covered_5m'] for r in rows),
        whole_scene_coverage_5m=sum(r['whole_scene_covered_5m'] for r in rows)/n,
        center_coverage_5m=sum(r['center_covered_5m'] for r in rows)/n,
        mean_best_scene_rmse_m=float(np.mean(best)) if best else None,
        primary_center_rmse_m=float(np.sqrt(np.mean(np.square(center)))) if center else None,
        primary_member_success_ratio_5m=sum(r['primary_successful_members_5m'] for r in rows)/max(truth_count,1),
        primary_member_rmse_m=float(np.sqrt(np.mean(np.square(member_errors)))) if member_errors else None,
        primary_member_p95_m=float(np.quantile(member_errors,.95)) if member_errors else None,
        primary_whole_scene_success_ratio_5m=sum(r['primary_whole_scene_success_5m'] for r in rows)/n,
        published_whole_scene_coverage_5m=sum(r['published_whole_scene_covered_5m'] for r in rows)/n,
        published_center_coverage_5m=sum(r['published_center_covered_5m'] for r in rows)/n,
        truth_member_instances=truth_count,
        primary_success_denominator='all labelled members, including no-detection, rejected and unmatched cases',
        primary_matched_member_instances=len(member_errors),
        primary_formation_accuracy=float(np.mean(classified)) if classified else None,
        formation_head_trained=formation_trained,formation_head_status=formation_status,
        classification_used_for_selection=False,
        selection_policy='primary_member_success_then_primary_rmse_then_published_scene_coverage_v1',
        candidate_coverage_scope='whole_scene_coverage and center_coverage are oracle diagnostics over all generated candidates',
        published_coverage_scope='oracle diagnostics only over modes kept by the deployment selector',
        center_scope='full label set only for complete valid detections; partial detected subsets reported separately',
        label_use='after observation-only generation; never latent posterior',
        allow_untrained_validation=bool(allow_untrained_validation),
        model_trained=bool(model.trained),uncertainty_calibrated=False)
    return dict(summary=summary,frames=rows)


def selection_key(report):
    s=report['summary']
    required=('primary_member_success_ratio_5m','primary_member_rmse_m','published_whole_scene_coverage_5m')
    if any(key not in s for key in required):
        raise ValueError('Checkpoint selection requires deployed-primary and published-mode metrics')
    rmse=s['primary_member_rmse_m']
    return (s['primary_member_success_ratio_5m'],
            -(rmse if rmse is not None else float('inf')),s['published_whole_scene_coverage_5m'])


def count_baseline(train, validation):
    """Optional diagnostic only; formation class never selects a checkpoint."""
    labelled=[row for row in train if row.get('formation_index') is not None]
    val=[row for row in validation if row.get('formation_index') is not None]
    if not labelled:return dict(train_count_mapping={},validation_accuracy=None,used_for_selection=False)
    counts={}
    for row in labelled:
        n=len(row['features']['ray_directions']);counts.setdefault(n,np.zeros(len(FORMATIONS),int))[row['formation_index']]+=1
    majority=np.bincount([r['formation_index'] for r in labelled],minlength=len(FORMATIONS)).argmax()
    predicted=[int(counts[n].argmax()) if n in counts else int(majority)
               for n in [len(s['features']['ray_directions']) for s in val]]
    return dict(train_count_mapping={str(n):FORMATIONS[int(v.argmax())] for n,v in counts.items()},
        validation_accuracy=float(np.mean([p==s['formation_index'] for p,s in zip(predicted,val)])) if val else None,
        used_for_selection=False,
        caveat='Formation cardinalities are confounded in current data; count-only accuracy is not spatial reasoning.')


class EpochSampler:
    """Shuffled epochs without replacement; every usable local frame is visited."""
    def __init__(self, size, seed):
        if int(size)<1:raise ValueError('Need at least one local usable group')
        self.size=int(size);self.rng=np.random.default_rng(seed)
        self.order=self.rng.permutation(self.size);self.cursor=0;self.completed_epochs=0
        self.seen=np.zeros(self.size,bool);self.draws=0

    def next(self, batch_size):
        selected=[]
        while len(selected)<int(batch_size):
            if self.cursor==self.size:
                self.completed_epochs+=1;self.order=self.rng.permutation(self.size);self.cursor=0
            take=min(int(batch_size)-len(selected),self.size-self.cursor)
            selected.extend(self.order[self.cursor:self.cursor+take].tolist());self.cursor+=take
        selected=np.asarray(selected,int);self.seen[selected]=True;self.draws+=len(selected)
        return selected

    def state_dict(self):
        return dict(size=self.size,order=self.order.copy(),cursor=self.cursor,
            completed_epochs=self.completed_epochs,seen=self.seen.copy(),draws=self.draws,
            generator_state=self.rng.bit_generator.state)

    def load_state_dict(self,state):
        if state.get('policy')==EPISODE_SAMPLING:raise ValueError('Cannot restore episode-balanced state into a frame sampler')
        order=np.asarray(state['order']);seen=np.asarray(state['seen'])
        cursor=int(state['cursor']);draws=int(state['draws']);epochs=int(state['completed_epochs'])
        if (int(state['size'])!=self.size or order.shape!=(self.size,) or order.dtype.kind not in 'iu'
                or not np.array_equal(np.sort(order),np.arange(self.size)) or seen.shape!=(self.size,)
                or seen.dtype!=np.bool_ or not 0<=cursor<=self.size or min(draws,epochs)<0
                or draws!=epochs*self.size+cursor):
            raise ValueError('Resume sampler inventory/order/cursor is incompatible')
        self.rng.bit_generator.state=state['generator_state']
        self.order=order.copy();self.seen=seen.copy();self.cursor=cursor;self.draws=draws;self.completed_epochs=epochs


class EpisodeUniformSampler:
    """One frame per episode each shuffled round; frame cycles are per episode."""
    def __init__(self,samples,seed):
        self.inventory=[(row['episode_id'],int(row['frame_idx'])) for row in samples]
        if not self.inventory or len(set(self.inventory))!=len(self.inventory):
            raise ValueError('Episode sampler needs a nonempty unique frame inventory')
        self.episodes=sorted({row[0] for row in self.inventory})
        self.members=[np.array([i for i,row in enumerate(self.inventory) if row[0]==episode],int)
            for episode in self.episodes]
        self.size=len(samples);self.rng=np.random.default_rng(seed)
        self.order=self.rng.permutation(len(self.episodes));self.cursor=0;self.completed_epochs=0
        self.frame_orders=[self.rng.permutation(ids) for ids in self.members]
        self.frame_cursors=np.zeros(len(self.episodes),int);self.frame_cycles=np.zeros(len(self.episodes),int)
        self.seen=np.zeros(self.size,bool);self.draws=0

    def next(self,batch_size):
        if int(batch_size)<1:raise ValueError('Sampler batch size must be positive')
        selected=[]
        for _ in range(int(batch_size)):
            if self.cursor==len(self.episodes):
                self.completed_epochs+=1;self.order=self.rng.permutation(len(self.episodes));self.cursor=0
            episode=int(self.order[self.cursor]);self.cursor+=1
            if self.frame_cursors[episode]==len(self.members[episode]):
                self.frame_cycles[episode]+=1;self.frame_orders[episode]=self.rng.permutation(self.members[episode])
                self.frame_cursors[episode]=0
            selected.append(self.frame_orders[episode][self.frame_cursors[episode]])
            self.frame_cursors[episode]+=1
        selected=np.asarray(selected,int);self.seen[selected]=True;self.draws+=len(selected)
        return selected

    def state_dict(self):
        return dict(policy=EPISODE_SAMPLING,size=self.size,inventory=list(self.inventory),episodes=list(self.episodes),
            order=self.order.copy(),cursor=self.cursor,completed_epochs=self.completed_epochs,
            frame_orders=[row.copy() for row in self.frame_orders],frame_cursors=self.frame_cursors.copy(),
            frame_cycles=self.frame_cycles.copy(),seen=self.seen.copy(),draws=self.draws,
            generator_state=self.rng.bit_generator.state)

    def load_state_dict(self,state):
        count=len(self.episodes);order=np.asarray(state.get('order',[]));seen=np.asarray(state.get('seen',[]))
        cursor=int(state.get('cursor',-1));epochs=int(state.get('completed_epochs',-1));draws=int(state.get('draws',-1))
        cursors=np.asarray(state.get('frame_cursors',[]));cycles=np.asarray(state.get('frame_cycles',[]))
        orders=[np.asarray(row) for row in state.get('frame_orders',[])]
        valid=(state.get('policy')==EPISODE_SAMPLING and state.get('size')==self.size
            and [tuple(row) for row in state.get('inventory',[])]==self.inventory and state.get('episodes')==self.episodes
            and order.shape==(count,) and order.dtype.kind in 'iu'
            and np.array_equal(np.sort(order),np.arange(count)) and 0<=cursor<=count
            and epochs>=0 and draws==epochs*count+cursor and seen.shape==(self.size,) and seen.dtype==np.bool_
            and cursors.shape==(count,) and cycles.shape==(count,) and cursors.dtype.kind in 'iu'
            and cycles.dtype.kind in 'iu' and np.all(cycles>=0) and len(orders)==count)
        if valid:
            expected=np.full(count,epochs,int);expected[order[:cursor]]+=1
            for i,ids in enumerate(self.members):
                valid=valid and (orders[i].shape==ids.shape and orders[i].dtype.kind in 'iu'
                    and np.array_equal(np.sort(orders[i]),ids) and 0<=cursors[i]<=len(ids)
                    and cycles[i]*len(ids)+cursors[i]==expected[i])
        if not valid:raise ValueError('Resume episode sampler inventory/order/cursor is incompatible')
        self.rng.bit_generator.state=state['generator_state'];self.order=order.copy();self.cursor=cursor
        self.completed_epochs=epochs;self.draws=draws;self.seen=seen.copy()
        self.frame_orders=[row.copy() for row in orders];self.frame_cursors=cursors.copy();self.frame_cycles=cycles.copy()


def sampling_policy(config,partition):
    frozen=partition.get('training_sampling_policy')
    selected=config.get('training_sampling_policy',frozen or 'uniform_frame_v1')
    if selected not in ('uniform_frame_v1',EPISODE_SAMPLING) or (frozen and selected!=frozen):
        raise ValueError('Training sampling differs from the frozen dataset policy')
    return selected


def sampling_coverage_plan(samples, policy, batch_size):
    """Bound steps needed to visit every usable frame, without sensor labels.

    Episode balancing deliberately gives a short and long episode equal weight.
    Each episode therefore needs as many rounds as the longest local trajectory
    before all its shuffled frame cycles are guaranteed to have been visited.
    """
    if int(batch_size)<1 or not samples:
        raise ValueError('A positive batch and nonempty usable frame inventory are required')
    counts={}
    inventory=set()
    for sample in samples:
        key=(sample['episode_id'],int(sample['frame_idx']))
        if key in inventory:raise ValueError('Duplicate usable training frame in coverage plan')
        inventory.add(key);counts[key[0]]=counts.get(key[0],0)+1
    if policy==EPISODE_SAMPLING:
        draws=len(counts)*max(counts.values())
    elif policy=='uniform_frame_v1':
        draws=len(samples)
    else:
        raise ValueError('Unknown training sampler policy')
    return dict(policy=policy,usable_frames=len(samples),usable_episodes=len(counts),
        max_usable_frames_per_episode=max(counts.values()),
        full_frame_coverage_draws_upper_bound=draws,
        full_frame_coverage_steps_upper_bound=(draws+int(batch_size)-1)//int(batch_size))


def atomic_torch_save(value,path):
    """Publish a whole checkpoint only after serialization and fsync succeed."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.tmp',delete=False) as stream:
        temporary=Path(stream.name)
    try:
        torch.save(value,temporary)
        with temporary.open('rb') as stream:os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def _training_options(args):
    names=('steps','max_episodes','frames_per_episode','validation_episodes','validation_frames',
        'validate_every','batch_size','learning_rate','physics_every','physics_weight','detection_dropout','seed','threads')
    return {name:getattr(args,name) for name in names}


def _stable_model_config(config):
    mutable={'device','world_model_trained','completed_optimizer_steps','formation_head_trained'}
    return {key:value for key,value in config.items() if key not in mutable}


def inventory_digest(inventory):
    """Cache hits are runtime events; they cannot change training identity."""
    stable=[]
    for row in inventory:
        item=dict(row)
        if 'cache' in item:item['cache']={k:v for k,v in item['cache'].items() if k!='hit'}
        stable.append(item)
    return hashlib.sha256(json.dumps(stable,sort_keys=True).encode()).hexdigest()


def read_resume_checkpoint(path,output,args):
    path=Path(path).resolve();output=Path(output).resolve()
    if path!=output/'world.latest.pt':
        raise ValueError('Resume requires this output directory\'s world.latest.pt training state')
    value=torch.load(path,map_location='cpu',weights_only=False)
    required={'steps','model_config','state_dict','optimizer_state_dict','rank_states','world_size',
        'global_batch_size','train_episodes','validation_episodes','train_inventory_sha256','source_sha256',
        'best_step','best_validation','training_options','metric_history','inference_checkpoint','best_checkpoint'}
    if (not isinstance(value,dict) or value.get('kind')!='group_scene_training_resume'
            or value.get('group_schema') != GROUP_SCHEMA or not required<=value.keys()):
        raise ValueError('Require a complete current training-resume checkpoint, not inference weights')
    step=value['steps']
    if isinstance(step,bool) or not isinstance(step,int) or not 1<=step<=args.steps:
        raise ValueError('Resume step is outside the requested original training budget')
    if value['training_options']!=_training_options(args):
        raise ValueError('Resume training options differ from the saved run')
    if value['source_sha256']!=_source_hashes():
        raise ValueError('Resume source code differs from the saved run')
    if value.get('torch_version')!=str(torch.__version__):
        raise ValueError('Resume PyTorch version differs from the saved optimizer/runtime')
    history=value['metric_history']
    if not isinstance(history,list) or [r.get('step') for r in history]!=list(range(1,step+1)):
        raise ValueError('Resume metric history does not cover all completed steps')
    if (value['inference_checkpoint'].get('steps')!=step
            or value['best_checkpoint'].get('steps')!=value['best_step']
            or not isinstance(value['best_step'],int) or not 1<=value['best_step']<=step
            or not isinstance(value['best_validation'],dict)):
        raise ValueError('Resume primary/best checkpoint lineage is inconsistent')
    return value


def validate_resume_state(value,config,world,train_ids,val_ids,inventory_sha256):
    if (value['world_size']!=world.size or value['global_batch_size']!=world.size*value['training_options']['batch_size']
            or value['train_episodes']!=train_ids or value['validation_episodes']!=val_ids
            or value['train_inventory_sha256']!=inventory_sha256
            or _stable_model_config(value['model_config'])!=_stable_model_config(config)):
        raise ValueError('Resume model/data/inventory/world-size/batch identity differs')
    ranks=value['rank_states']
    if not isinstance(ranks,list) or sorted(r.get('rank') for r in ranks)!=list(range(world.size)):
        raise ValueError('Resume rank state inventory is incomplete or duplicated')
    state=next(r for r in ranks if r['rank']==world.rank)
    required={'sampler','augmentation_generator_state','numpy_global_state','torch_cpu_rng_state',
        'torch_device_rng_state','physical_updates','physical_skips','dropout_updates','rejected_dropout_updates'}
    if not required<=state.keys():raise ValueError('Resume rank RNG/counters are incomplete')
    if int(state['sampler']['draws'])!=value['steps']*value['training_options']['batch_size']:
        raise ValueError('Resume sampler draw count differs from optimizer/batch history')
    if any(not isinstance(state[name],int) or state[name]<0 for name in
           ('physical_updates','physical_skips','dropout_updates','rejected_dropout_updates')):
        raise ValueError('Resume rank update counters must be nonnegative integers')
    return state


def restore_training_state(value,rank_state,model,optimizer,sampler,rng,device):
    """Restore RNG last: preparation and module construction consume randomness."""
    model.load_state_dict(value['state_dict'],strict=True)
    expected=optimizer.state_dict()['param_groups'];saved=value['optimizer_state_dict']['param_groups']
    if len(expected)!=len(saved) or any(
            len(left['params'])!=len(right['params']) or
            {k:v for k,v in left.items() if k!='params'}!={k:v for k,v in right.items() if k!='params'}
            for left,right in zip(expected,saved)):
        raise ValueError('Resume optimizer parameter groups or hyperparameters differ')
    optimizer.load_state_dict(value['optimizer_state_dict'])
    sampler.load_state_dict(rank_state['sampler'])
    rng.bit_generator.state=rank_state['augmentation_generator_state']
    model.config.update(world_model_trained=True,completed_optimizer_steps=value['steps'],
        formation_head_trained=value['model_config'].get('formation_head_trained',False))
    model.trained=True
    np.random.set_state(rank_state['numpy_global_state'])
    torch.set_rng_state(rank_state['torch_cpu_rng_state'])
    if str(device).startswith('cuda'):
        if rank_state['torch_device_rng_state'] is None:raise ValueError('Resume CUDA RNG state missing')
        torch.cuda.set_rng_state(rank_state['torch_device_rng_state'],device)


def restore_resume_artifacts(output,value):
    """Keep old attempt evidence, then roll public metrics/weights to the snapshot."""
    output=Path(output)
    archive=output/'resume_history'/str(time.time_ns());archive.mkdir(parents=True,exist_ok=False)
    for name in ('metrics.jsonl','prior_validation.jsonl','progress.json','run_manifest.json',
                 'world.pt','world.best.pt','configuration.json'):
        path=output/name
        if path.is_file():shutil.copy2(path,archive/name)
    metrics=value['metric_history']
    (output/'metrics.jsonl').write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in metrics))
    prior=output/'prior_validation.jsonl'
    if prior.exists():
        retained=[]
        for line in prior.read_text().splitlines():
            try:row=json.loads(line)
            except json.JSONDecodeError:continue
            if isinstance(row.get('step'),int) and row['step']<=value['steps']:retained.append(row)
        prior.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in retained))
    atomic_torch_save(value['inference_checkpoint'],output/'world.pt')
    atomic_torch_save(value['best_checkpoint'],output/'world.best.pt')
    write_json(output/'prior_validation_best.json',value['best_validation'])
    write_json(archive/'resume_receipt.json',dict(resumed_from_step=value['steps'],
        source_checkpoint_sha256=sha256(output/'world.latest.pt'),tail_policy='archived before snapshot rollback'))
    return str(archive)


class TrainingWorld:
    """One averaged-gradient model, one process per visible device."""
    def __init__(self, device):
        self.rank=int(os.environ.get('RANK','0'));self.size=int(os.environ.get('WORLD_SIZE','1'))
        self.local_rank=int(os.environ.get('LOCAL_RANK','0'));self.owner=False
        if str(device).startswith('cuda'):
            if not torch.cuda.is_available():raise RuntimeError('Requested CUDA is unavailable')
            self.device=f'cuda:{self.local_rank}' if self.size>1 else str(device)
            torch.cuda.set_device(self.device)
        else:self.device='cpu'
        if self.size>1 and not dist.is_initialized():
            dist.init_process_group('nccl' if self.device.startswith('cuda') else 'gloo',timeout=timedelta(minutes=30))
            self.owner=True

    def barrier(self):
        if self.size>1:dist.barrier()

    def broadcast(self, value):
        box=[value]
        if self.size>1:dist.broadcast_object_list(box,src=0)
        return box[0]

    def gather(self, value):
        if self.size==1:return [value]
        values=[None]*self.size if self.rank==0 else None
        dist.gather_object(value,values,dst=0)
        return values

    def sum(self, values):
        tensor=torch.as_tensor(values,dtype=torch.float64,device=self.device)
        if self.size>1:dist.all_reduce(tensor,op=dist.ReduceOp.SUM)
        return tensor.cpu().numpy()

    def sync_model(self, model):
        if self.size>1:
            for parameter in model.parameters():dist.broadcast(parameter.data,src=0)
            for buffer in model.buffers():dist.broadcast(buffer.data,src=0)

    def close(self):
        if self.owner and dist.is_initialized():dist.destroy_process_group()


def synchronize_group_gradients(model, world):
    """Average dense gradients, including zero for locally unused parameters."""
    parameters=[p for p in model.parameters() if p.requires_grad]
    flattened=torch.cat([p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros_like(p).reshape(-1)
                         for p in parameters])
    finite=torch.tensor(int(bool(torch.isfinite(flattened).all())),device=world.device,dtype=torch.int32)
    if world.size>1:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
    if not int(finite):raise FloatingPointError('At least one rank has non-finite group gradients')
    if world.size>1:
        dist.all_reduce(flattened,op=dist.ReduceOp.SUM);flattened.div_(world.size)
    offset=0
    for parameter in parameters:
        length=parameter.numel();gradient=flattened[offset:offset+length].view_as(parameter)
        if parameter.grad is None:parameter.grad=gradient.clone()
        else:parameter.grad.copy_(gradient)
        offset+=length


def _parameter_difference(model, world):
    actual=torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    expected=actual.clone()
    if world.size>1:dist.broadcast(expected,src=0)
    difference=(actual-expected).abs().max()
    if world.size>1:dist.all_reduce(difference,op=dist.ReduceOp.MAX)
    return float(difference)


def validate_training_graph(model, sample, world=None):
    """Check finite forward/backward without an optimizer or any parameter update."""
    before={name:value.detach().clone() for name,value in model.state_dict().items()}
    model.train();model.zero_grad(set_to_none=True)
    loss,terms=model.training_loss(sample['features'],sample['positions'],
        formation_index=sample.get('formation_index'))
    if not terms.get('supervision_usable',False):
        raise ValueError('Gradient validation requires accepted supervision')
    if not torch.isfinite(loss):raise FloatingPointError('Non-finite continuous-depth training objective')
    loss.backward()
    if world is not None:synchronize_group_gradients(model,world)
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError('Missing or non-finite model gradients')
    norm=float(torch.sqrt(sum(g.detach().square().sum() for g in gradients)))
    unchanged=all(torch.equal(before[name],value) for name,value in model.state_dict().items())
    if not unchanged:raise AssertionError('Forward/backward validation modified model parameters or buffers')
    report=dict(loss=float(loss.detach()),gradient_norm=norm,finite_gradients=True,
        parameters_unchanged=True,optimizer_steps=0,checkpoint_written=False,
        model_trained=bool(model.trained),uncertainty_calibrated=False,
        supervision={name:value for name,value in terms.items()
                     if isinstance(value,(bool,int,float,str)) or value is None},
        losses={name:float(value.detach()) for name,value in terms.items()
                if isinstance(value,torch.Tensor) and value.ndim==0})
    model.zero_grad(set_to_none=True)
    return report


def validate_training_options(args):
    if int(getattr(args,'physics_every',0))!=0 or float(getattr(args,'physics_weight',0.))!=0.:
        raise ValueError('RA/RE-only training does not use the legacy raw-IQ physics flags; '
                         'set --physics-every 0 --physics-weight 0')
    if min(args.steps,args.validate_every,args.batch_size)<1:
        raise ValueError('Positive training budgets required')
    if not 0.<=float(args.detection_dropout)<=1.:raise ValueError('Detection dropout must be in [0,1]')


def record_completed_optimizer_step(model, config, steps, *, formation_supervised=False):
    """Bookkeeping invoked only after the optimizer actually completed a step."""
    if isinstance(steps,bool) or int(steps)!=steps or int(steps)<1:
        raise ValueError('A completed positive optimizer-step count is required')
    formation_trained=bool(config.get('formation_head_trained',False) or formation_supervised)
    config.update(world_model_trained=True,completed_optimizer_steps=int(steps),
        formation_head_trained=formation_trained)
    model.config.update(world_model_trained=True,completed_optimizer_steps=int(steps),
        formation_head_trained=formation_trained)
    model.trained=True


def train(args):
    validate_training_options(args)
    training_config=json.loads(Path(args.config).read_text())
    validate_training_phase(training_config)
    require_single_frame(training_config)
    dry_run=bool(getattr(args,'dry_run',False))
    resume_path=getattr(args,'resume',None)
    if resume_path and dry_run:raise ValueError('Resume cannot be combined with a zero-update dry run')
    resume=read_resume_checkpoint(resume_path,args.output,args) if resume_path else None
    torch.set_num_threads(int(args.threads));world=TrainingWorld(args.device)
    output=Path(args.output).resolve();begin=time.perf_counter();started=time.time()
    status_path=output/'ranks'/f'rank_{world.rank}.json';progress={};run=None
    try:
        failure=None
        if world.rank==0:
            output.mkdir(parents=True,exist_ok=True)
            if not resume and any((output/name).exists() for name in ('world.pt','world.best.pt','progress.json','metrics.jsonl','run_manifest.json')):
                failure='Existing group training artifacts are protected: '+str(output)
        failure=world.broadcast(failure)
        if failure:raise FileExistsError(failure)
        world.barrier()
        seed=int(args.seed)+world.rank*100003
        np.random.seed(seed);torch.manual_seed(seed)
        config=dict(training_config);config.update(device=world.device,training_phase=phase_label(configured_phases(config)),
            visual_point_mode=config.get('visual_point_mode','bbox_center'),
            architecture_schema=training_config.get('architecture_schema',GROUP_SCHEMA),group_refinement_steps=0,
            world_model_trained=False,completed_optimizer_steps=0,formation_head_trained=False,
            group_hypotheses=int(config.get('group_hypotheses',32)),uncertainty_calibrated=False)
        partition=partition_receipt(config)
        config['active_partition']=partition
        sampler_policy=sampling_policy(config,partition)
        if sampler_policy==EPISODE_SAMPLING:config['training_sampling_policy']=sampler_policy
        config.setdefault('track_detection_dir',str(PROJECT/'results/detections'))
        dataset=Dataset(config['dataset_root'],echo_subdir=config['echo_subdir']);splits=strict_splits(dataset,config)
        train_ids=splits['train'][:args.max_episodes] if args.max_episodes else splits['train']
        val_ids=splits['val'][:args.validation_episodes] if args.validation_episodes else splits['val']
        if dry_run:
            # A dry run is intentionally bounded even if the full-data defaults
            # were supplied. It performs no optimizer step and saves no weights.
            train_ids=train_ids[:world.size];val_ids=val_ids[:1]
        diagnostic_frames=1
        train_frame_count=diagnostic_frames if dry_run else args.frames_per_episode
        val_frame_count=diagnostic_frames if dry_run else args.validation_frames
        if not train_ids or not val_ids:raise ValueError('Need separate nonempty training and validation episodes')
        local_ids=train_ids[world.rank::world.size]
        if not local_ids:raise ValueError('Every training rank needs at least one episode')
        expected=sum(len(frame_indices(dataset,e,train_frame_count,config)) for e in local_ids)
        initial=dict(rank=world.rank,world_size=world.size,device=world.device,stage='preparing',
            expected_frames=expected,prepared_frames=0,prepared_groups=0,train_episodes=local_ids,
            updated_unix=time.time())
        write_json(status_path,initial)
        progress=dict(stage='preparing',model_kind='group_scene',step=resume['steps'] if resume else 0,requested_steps=args.steps,started_unix=started,
            world_size=world.size,global_batch_size=args.batch_size*world.size,uncertainty_calibrated=False)
        if world.rank==0:
            write_json(output/'progress.json',progress);write_json(output/'configuration.json',config)
        local_out=output/'ranks'/f'rank_{world.rank}' if world.size>1 else output
        training,tmanifest,frontend=prepare(dataset,config,local_ids,'train',train_frame_count,world.device,local_out,
            rank_status=(status_path,initial))
        write_json(status_path,dict(initial,stage='prepared',prepared_frames=len(tmanifest),prepared_groups=len(training),updated_unix=time.time()))
        validation=[];vmanifest=[]
        if world.rank==0:
            validation,vmanifest,_=prepare(dataset,config,val_ids,'val',val_frame_count,world.device,output)
        if not training or (world.rank==0 and not validation):raise ValueError('No detected train/validation group observations')
        coverage=sampling_coverage_plan(training,sampler_policy,args.batch_size)
        insufficient=not dry_run and args.frames_per_episode==0 and args.steps<coverage['full_frame_coverage_steps_upper_bound']
        if world.sum([int(insufficient)])[0]:
            raise ValueError('Full-frame training budget cannot cover every usable frame on every rank; '
                'increase --steps or use --dry-run for a bounded implementation check')
        local_counts={}
        for sample in training:
            if sample.get('formation_index') is not None:
                n=len(sample['features']['ray_directions']);local_counts.setdefault(n,np.zeros(len(FORMATIONS),int))[sample['formation_index']]+=1
        usable_episodes=sorted({sample['episode_id'] for sample in training})
        usable_global=int(world.sum([len(usable_episodes)])[0])
        # DDP averages ranks equally. Correct uneven episode shards so each
        # usable episode has the same expected objective weight globally.
        episode_rank_weight=world.size*len(usable_episodes)/usable_global if sampler_policy==EPISODE_SAMPLING else 1.
        partitions=world.gather(dict(rank=world.rank,episodes=local_ids,usable_episodes=usable_episodes,
            episode_objective_rank_weight=episode_rank_weight,frames=tmanifest,groups=len(training),
            expected_frames=expected,sampling_coverage=coverage,
            count_class_matrix={str(k):v.tolist() for k,v in local_counts.items()}))
        if world.rank==0:
            inventory=[row for part in partitions for row in part['frames']]
            if len({(r['episode_id'],r['frame_idx']) for r in inventory})!=len(inventory):raise ValueError('Duplicate cross-rank training frame')
            if sorted(e for part in partitions for e in part['episodes'])!=sorted(train_ids):raise ValueError('Cross-rank episode partition incomplete')
            if any(part['expected_frames']!=len(part['frames']) for part in partitions):raise ValueError('Training frame preparation incomplete')
            expected_validation=sum(len(frame_indices(dataset,e,val_frame_count,config)) for e in val_ids)
            if expected_validation!=len(vmanifest):raise ValueError('Validation frame preparation incomplete')
            count_matrix={}
            for part in partitions:
                for n,counts in part['count_class_matrix'].items():count_matrix.setdefault(n,np.zeros(len(FORMATIONS),int));count_matrix[n]+=counts
            majority=sum(count_matrix.values()).argmax() if count_matrix else None
            labelled_validation=[s for s in validation if s.get('formation_index') is not None]
            predicted=[int(count_matrix[str(len(s['features']['ray_directions']))].argmax())
                       if str(len(s['features']['ray_directions'])) in count_matrix else int(majority)
                       for s in labelled_validation] if majority is not None else []
            baseline=dict(train_count_mapping={n:FORMATIONS[int(v.argmax())] for n,v in count_matrix.items()},
                validation_accuracy=float(np.mean([p==s['formation_index'] for p,s in zip(predicted,labelled_validation)])) if predicted else None,
                evaluated_nonempty_groups=len(validation),full_validation_frames=len(vmanifest),
                used_for_selection=False,
                caveat='Current formation labels are confounded with cardinality; this baseline is not spatial reasoning.')
            run=dict(schema=config.get('architecture_schema',GROUP_SCHEMA),status='validating_implementation' if dry_run else 'training',configuration=config,partition=partition,train_episodes=train_ids,validation_episodes=val_ids,
                model_trained=False,optimizer_steps=0,
                strict_split_counts={k:len(v) for k,v in splits.items()},training_phase=phase_label(configured_phases(config)),training_phases=list(configured_phases(config)),world_size=world.size,
                per_rank_batch_size=args.batch_size,global_batch_size=args.batch_size*world.size,
                prepared_training_groups=sum(part['groups'] for part in partitions),prepared_validation_groups=len(validation),
                total_training_selected_frames=len(inventory),all_selected_phase_frames_used=not dry_run and args.frames_per_episode==0,
                total_validation_selected_frames=len(vmanifest),
                all_selected_validation_phase_frames_used=not dry_run and args.validation_frames==0,
                full_usable_frame_coverage_steps_upper_bound=max(
                    part['sampling_coverage']['full_frame_coverage_steps_upper_bound'] for part in partitions),
                excluded_training_frame_counts={status:sum(row['feature_status']==status for row in inventory)
                    for status in sorted({row['feature_status'] for row in inventory}-{'prepared'})},
                total_training_track_frames=sum(r.get('phase','track')=='track' for r in inventory),
                selected_phase_counts={phase:sum(r.get('phase','track')==phase for r in inventory) for phase in configured_phases(config)},
                no_detection_training_frames=sum(r['feature_status']=='no_detections' for r in inventory),
                inventory_sha256=inventory_digest(inventory),
                rank_partitions=[{k:v for k,v in part.items() if k!='frames'} for part in partitions],
                frontend=frontend,source_sha256=_source_hashes(),seed=args.seed,
                deployment_sampling_seed=int(config['seed']),
                hyperparameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                depth_normalization_source='public observation ray intervals; no training formation spacing or fixed distance',
                decoder='RA/RE same-range features; object interaction and shared latent generate continuous joint depths',
                model_variant='single',configured_window=require_single_frame(config),
                input_policy='one current observation with a fixed station; no history or future input',
                position_refinement=False,
                world_model_inputs=['base ray origins','base ray directions','base observer position','RA power and axes','RE power and axes'],
                input_depth_grid=False,appearance_input=False,raw_iq_likelihood=False,
                missing_scene_policy='supervise detected rays only; mixed RA/RE retained; full scene completeness remains unknown',
                model_selection='deployed primary member 5m success, primary RMSE, retained-mode coverage; raw coverage diagnostic only',
                supervision_policy='gated observation-ray correspondence; reject unsupported labels; record unmatched members',
                uncertainty_calibrated=False,oracle_mode=False,count_only_baseline=baseline,
                training_sampling_policy=sampler_policy,
                sampling=('uniform episode rounds with per-episode shuffled frame cycles; unequal DDP shard sizes corrected'
                    if sampler_policy==EPISODE_SAMPLING else 'rank-local shuffled epochs without replacement; actual unique visited frames reported separately'),
                gradient_synchronization='all-reduce average before gradient clipping and every optimizer step')
            write_json(output/'train_inventory.json',dict(frames=inventory,episodes=train_ids,sha256=run['inventory_sha256']))
            if not resume:write_json(output/'run_manifest.json',run)
            write_json(output/'configuration.json',config)
        current_inventory_sha=world.broadcast(run['inventory_sha256'] if world.rank==0 else None)
        resume_rank=None
        if resume:
            resume_rank=validate_resume_state(resume,config,world,train_ids,val_ids,current_inventory_sha)
        world.barrier()
        model=GroupSceneModel(config).to(world.device);world.sync_model(model)
        if dry_run:
            graph=validate_training_graph(model,training[-1],world)
            graph_rows=world.gather(dict(rank=world.rank,**graph))
            if world.rank==0:
                report=evaluate_prior(model,validation,vmanifest,int(config['seed']),min(config['group_hypotheses'],4),
                    allow_untrained_validation=True)
                write_json(output/'prior_validation_untrained.json',report)
                run.update(status='dry_run_complete',optimizer_steps=0,steps=0,model_trained=False,
                    checkpoint_written=False,gradient_checks=graph_rows,elapsed_s=time.perf_counter()-begin,
                    validation_scope='implementation only; untrained outputs are not localization performance')
                write_json(output/'run_manifest.json',run)
                write_json(output/'progress.json',dict(progress,stage='complete',step=0,requested_steps=0,
                    validation_only=True,optimizer_steps=0,elapsed_s=run['elapsed_s'],eta_s=0.))
                print(json.dumps(dict(status='dry_run_complete',optimizer_steps=0,checkpoint_written=False)),flush=True)
            write_json(status_path,dict(initial,stage='complete',validation_only=True,optimizer_steps=0))
            world.barrier();return run
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
        rng=np.random.default_rng(seed)
        sampler=EpisodeUniformSampler(training,seed) if sampler_policy==EPISODE_SAMPLING else EpochSampler(len(training),seed)
        best=None;best_step=None;metrics=[];physical_updates=0;physical_skips=0;dropout_updates=0
        rejected_dropout_updates=0
        start_step=1;previous_loop_seconds=0.
        if resume:
            restore_training_state(resume,resume_rank,model,optimizer,sampler,rng,world.device)
            config.update(world_model_trained=True,completed_optimizer_steps=resume['steps'],
                formation_head_trained=model.config.get('formation_head_trained',False))
            start_step=resume['steps']+1;best=resume['best_validation'];best_step=resume['best_step']
            metrics=list(resume['metric_history']);previous_loop_seconds=float(metrics[-1]['elapsed_s'])
            physical_updates=int(resume_rank['physical_updates']);physical_skips=int(resume_rank['physical_skips'])
            dropout_updates=int(resume_rank['dropout_updates']);rejected_dropout_updates=int(resume_rank['rejected_dropout_updates'])
            if world.rank==0:
                archive=restore_resume_artifacts(output,resume)
                run.update(model_trained=True,optimizer_steps=resume['steps'],resume_from=str(Path(resume_path).resolve()),
                    resumed_from_step=resume['steps'],resume_archive=archive)
                write_json(output/'run_manifest.json',run);write_json(output/'configuration.json',config)
        world.barrier()
        loop=time.perf_counter();stream=(output/'metrics.jsonl').open('a' if resume else 'x') if world.rank==0 else None
        try:
            for step in range(start_step,args.steps+1):
                model.train();optimizer.zero_grad(set_to_none=True);losses=[];reports=[];batch_features=[]
                selected=sampler.next(args.batch_size)
                for sample_index in selected:
                    sample=training[int(sample_index)]
                    features,dropped,rejected=supervision_checked_dropout(sample['features'],sample['positions'],
                        rng,args.detection_dropout,config)
                    rejected_dropout_updates+=int(rejected)
                    batch_features.append(features);dropout_updates+=int(dropped)
                    loss,terms=model.training_loss(features,sample['positions'],formation_index=sample.get('formation_index'))
                    if not terms.get('supervision_usable',False):
                        raise ValueError('Prepared training sample has no accepted supervision; no optimizer update allowed')
                    losses.append(loss);reports.append(terms)
                loss=torch.stack(losses).mean()*episode_rank_weight;physics_value=0.
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite group training objective')
                loss.backward();synchronize_group_gradients(model,world)
                gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.));optimizer.step()
                formation_supervised=False
                if float(config.get('group_formation_loss_weight',0.))>0:
                    formation_supervised=bool(world.sum([int(any(training[int(i)].get('formation_index') is not None
                        for i in selected))])[0])
                record_completed_optimizer_step(model,config,step,formation_supervised=formation_supervised)
                elapsed=previous_loop_seconds+time.perf_counter()-loop;rate=step/max(elapsed,1e-9)
                stats=world.sum([float(loss.detach()),float(torch.stack(losses).mean().detach()),physics_value,
                    np.mean([float(t['shape_reconstruction'].detach()) for t in reports]),
                    np.mean([float(t['latent_kl'].detach()) for t in reports]),
                    np.mean([float(t['center_nll'].detach()) for t in reports]),physical_updates,physical_skips])/world.size
                row=dict(step=step,loss=float(stats[0]),proposal_loss=float(stats[1]),physical_updates=int(round(stats[6]*world.size)),
                    physical_skips=int(round(stats[7]*world.size)),
                    gradient_norm_before_clip=gradient,elapsed_s=elapsed,steps_per_second=rate,
                    shape_loss=float(stats[3]),latent_kl=float(stats[4]),center_nll=float(stats[5]),
                    ray_depth_loss=float(stats[3]),depth_nll=float(stats[5]),joint_scene_loss=float(stats[1]),
                    input_schema='base_ray_ra_re_v1',raw_iq_likelihood=False,
                    model_trained=True,optimizer_steps=step,
                    global_batch_size=args.batch_size*world.size,global_group_updates=step*args.batch_size*world.size)
                supervision_counts=world.sum([sum(t['matched_training_count'] for t in reports),
                    sum(t['unmatched_detection_count'] for t in reports),
                    sum(t['unmatched_label_count'] for t in reports),rejected_dropout_updates])
                row.update(matched_supervised_members=int(supervision_counts[0]),
                    unmatched_detections=int(supervision_counts[1]),unmatched_labels=int(supervision_counts[2]),
                    rejected_dropout_updates=int(supervision_counts[3]))
                if step==1 or step%25==0 or step==args.steps:
                    row['global_unique_training_groups']=int(world.sum([int(sampler.seen.sum())])[0])
                if world.rank==0:
                    metrics.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    write_json(output/'progress.json',dict(progress,**row,stage='training',eta_s=(args.steps-step)/rate))
                    if step==1 or step%25==0 or step==args.steps:
                        run.update(model_trained=True,optimizer_steps=step)
                        write_json(output/'run_manifest.json',run);write_json(output/'configuration.json',config)
                        print(json.dumps(dict(stage='training',**row,requested_steps=args.steps,eta_s=(args.steps-step)/rate)),flush=True)
                if step==1 or step%25==0 or step==args.steps:
                    write_json(status_path,dict(initial,stage='training',prepared_frames=len(tmanifest),prepared_groups=len(training),
                        step=step,requested_steps=args.steps,loss=float(loss.detach()),local_unique_groups=int(sampler.seen.sum()),
                        local_group_updates=sampler.draws,updated_unix=time.time()))
                if step==1 or step%args.validate_every==0 or step==args.steps:
                    world.barrier()
                    rank_resume=world.gather(dict(rank=world.rank,sampler=sampler.state_dict(),
                        augmentation_generator_state=rng.bit_generator.state,
                        numpy_global_state=np.random.get_state(),torch_cpu_rng_state=torch.get_rng_state(),
                        torch_device_rng_state=torch.cuda.get_rng_state(world.device) if world.device.startswith('cuda') else None,
                        physical_updates=physical_updates,physical_skips=physical_skips,dropout_updates=dropout_updates,
                        rejected_dropout_updates=rejected_dropout_updates))
                    if world.rank==0:
                        write_json(output/'progress.json',dict(progress,**row,stage='validating',eta_s=None))
                        report=evaluate_prior(model,validation,vmanifest,int(config['seed']),config['group_hypotheses'])
                        report.update(step=step,selection_key=[v if np.isfinite(v) else None for v in selection_key(report)])
                        with (output/'prior_validation.jsonl').open('a') as record:record.write(json.dumps(report)+'\n')
                        if best is None or selection_key(report)>selection_key(best):
                            best,best_step=report,step
                            atomic_torch_save(checkpoint(model,step,train_episodes=train_ids,validation_episodes=val_ids,
                                model_trained=True,optimizer_steps=step,
                                training_phase=phase_label(configured_phases(config)),training_phases=list(configured_phases(config)),frontend=frontend,feature_source_sha256=_source_hashes(),
                                world_size=world.size,selection='deployed primary accuracy with retained-mode coverage',validation_report=report),output/'world.best.pt')
                            write_json(output/'prior_validation_best.json',report)
                        inference_checkpoint=checkpoint(model,step,train_episodes=train_ids,validation_episodes=val_ids,
                            model_trained=True,optimizer_steps=step,
                            training_phase=phase_label(configured_phases(config)),training_phases=list(configured_phases(config)),frontend=frontend,feature_source_sha256=_source_hashes(),world_size=world.size)
                        atomic_torch_save(inference_checkpoint,output/'world.pt')
                        # A distinct training-state kind cannot accidentally
                        # be loaded as an inference checkpoint. All rank-local
                        # streams/cursors and Adam moments are preserved.
                        atomic_torch_save(dict(kind='group_scene_training_resume',group_schema=config.get('architecture_schema',GROUP_SCHEMA),
                            steps=step,model_config=dict(model.config),state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),rank_states=rank_resume,
                            world_size=world.size,global_batch_size=args.batch_size*world.size,
                            train_episodes=train_ids,validation_episodes=val_ids,
                            train_inventory_sha256=run['inventory_sha256'],source_sha256=_source_hashes(),
                            best_step=best_step,best_validation=best,
                            training_options=_training_options(args),torch_version=str(torch.__version__),
                            metric_history=list(metrics),inference_checkpoint=inference_checkpoint,
                            best_checkpoint=torch.load(output/'world.best.pt',map_location='cpu',weights_only=False),
                            resume_note='Strict same-run --resume restores optimizer, all rank RNG/sampler states, metrics and best selection.'),
                            output/'world.latest.pt')
                    world.barrier()
        finally:
            if stream is not None:stream.close()
        difference=_parameter_difference(model,world)
        visited=[dict(episode_id=s['episode_id'],frame_idx=s['frame_idx']) for s,seen in zip(training,sampler.seen) if seen]
        sampling=dict(rank=world.rank,prepared_groups=len(training),actual_unique_groups=len(visited),
            local_group_updates=sampler.draws,
            complete_local_passes=(int(np.min(sampler.frame_cycles+(sampler.frame_cursors==np.array([len(ids) for ids in sampler.members]))))
                if sampler_policy==EPISODE_SAMPLING else sampler.draws//len(training)),
            visited_frames=visited,physical_updates=physical_updates,physical_skips=physical_skips,detection_dropout_updates=dropout_updates)
        if sampler_policy==EPISODE_SAMPLING:
            sampling.update(policy=sampler_policy,episode_objective_rank_weight=episode_rank_weight,
                complete_episode_rounds=sampler.draws//len(sampler.episodes),
                episode_draws={episode:int(cycles*len(ids)+cursor) for episode,ids,cycles,cursor in
                    zip(sampler.episodes,sampler.members,sampler.frame_cycles,sampler.frame_cursors)})
        write_json(local_out/'sampling.json',sampling)
        sampling_rows=world.gather(sampling)
        if args.frames_per_episode==0 and world.sum([int(not sampler.seen.all())])[0]:
            raise ValueError('Full-frame training did not visit every usable training frame')
        if world.rank==0:
            model.trained=True;final=evaluate_prior(model,validation,vmanifest,int(config['seed']),config['group_hypotheses'])
            write_json(output/'prior_validation_final.json',final)
            actual_unique=sum(r['actual_unique_groups'] for r in sampling_rows)
            run.update(status='complete',steps=args.steps,best_step=best_step,
                model_trained=True,optimizer_steps=args.steps,
                physical_updates=sum(r['physical_updates'] for r in sampling_rows),
                physical_skips=sum(r['physical_skips'] for r in sampling_rows),
                detection_dropout_updates=sum(r['detection_dropout_updates'] for r in sampling_rows),
                elapsed_s=time.perf_counter()-begin,training_loop_seconds=previous_loop_seconds+time.perf_counter()-loop,
                actual_unique_training_groups=actual_unique,
                all_usable_training_frames_used=actual_unique==run['prepared_training_groups'],
                actual_global_group_updates=sum(r['local_group_updates'] for r in sampling_rows),
                rank_parameter_max_difference=difference,
                selected_checkpoint_sha256=sha256(output/'world.best.pt'),final_checkpoint_sha256=sha256(output/'world.pt'),
                loss_first_25_mean=float(np.mean([r['loss'] for r in metrics[:25]])),
                loss_last_25_mean=float(np.mean([r['loss'] for r in metrics[-25:]])),
                decoder_parameter_gradients_trained=True,
                limitation='Current-scene group inference; completed training alone does not establish localization precision or calibrated uncertainty.')
            if difference!=0.:raise ValueError('Distributed group ranks diverged')
            write_json(output/'sampling.json',dict(ranks=sampling_rows,actual_unique_groups=actual_unique,
                all_usable_frames_used=run['all_usable_training_frames_used']))
            write_json(output/'run_manifest.json',run)
            write_json(output/'progress.json',dict(progress,**metrics[-1],stage='complete',eta_s=0.,best_step=best_step,
                actual_unique_training_groups=actual_unique,rank_parameter_max_difference=difference))
            print(json.dumps(dict(status='complete',steps=args.steps,best_step=best_step,world_size=world.size,
                actual_unique_training_groups=actual_unique,all_usable_frames_used=run['all_usable_training_frames_used'],
                physical_updates=run['physical_updates'],elapsed_s=run['elapsed_s'],best_prior=best['summary'])),flush=True)
        write_json(status_path,dict(initial,stage='complete',prepared_frames=len(tmanifest),prepared_groups=len(training),
            step=args.steps,requested_steps=args.steps,local_unique_groups=int(sampler.seen.sum()),updated_unix=time.time()))
        world.barrier();return run
    except BaseException as error:
        failure=dict(progress,stage='failed',rank=world.rank,error=f'{type(error).__name__}: {error}',elapsed_s=time.perf_counter()-begin)
        write_json(status_path,failure)
        if world.rank==0:write_json(output/'progress.json',failure)
        raise
    finally:
        world.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=PROJECT/'config.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--max-episodes',type=int,default=0,help='0 uses all strict training episodes')
    parser.add_argument('--frames-per-episode',type=int,default=0,help='0 selects every frame in configured training_phases')
    parser.add_argument('--validation-episodes',type=int,default=0,help='0 uses all assigned validation episodes')
    parser.add_argument('--validation-frames',type=int,default=0,help='0 selects all validation frames in configured training_phases')
    parser.add_argument('--validate-every',type=int,default=500)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--physics-every',type=int,default=0,help='Must be 0: the RA/RE model has no raw-IQ physics loss')
    parser.add_argument('--physics-weight',type=float,default=0.,help='Must be 0 for the RA/RE-only interface')
    parser.add_argument('--dry-run',action='store_true',help='One train/validation frame per rank; forward/backward only, no optimizer or checkpoint')
    parser.add_argument('--resume',type=Path,help='Resume this output directory from its verified world.latest.pt training state')
    parser.add_argument('--detection-dropout',type=float,default=.2)
    parser.add_argument('--seed',type=int,default=20261118)
    parser.add_argument('--device',default='cpu');parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    if min(args.steps,args.validate_every,args.batch_size)<1 or min(args.frames_per_episode,args.validation_frames,args.max_episodes,args.validation_episodes)<0:
        parser.error('Positive training budgets required')
    train(args)


if __name__=='__main__':main()
