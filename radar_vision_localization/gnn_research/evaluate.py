"""Persist sensor-only graph predictions, then score fixed ROI identities.

Prediction never opens annotation frames. GT is accessed only after the full
prediction artifact has been committed, using the predeclared fixed index.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
for path in (PROJECT,PROJECT/'src',PROJECT/'tools'):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
from gnn_research.train import load_graph_checkpoint,forward,ARCHITECTURE,require,training_sources
from rvloc.config import ExperimentConfig,external_path
from rvloc.data import load_manifest
from rvloc.evaluation import metrics,episode_bootstrap,paired_compare
from rvloc.runtime import file_sha256,write_json,jsonable,source_identity
from accelerated_batch_model import pack_features
from fixed_gt_correspondence import score_fixed_frame


def _detach(value):
    if hasattr(value,'detach'):return value.detach().cpu().numpy()
    if isinstance(value,dict):return {k:_detach(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_detach(v) for v in value]
    return value


FRAME_DIAGNOSTICS={
    'valid_candidate_count','shared_candidate_count','valid_member_count','shared_member_count',
    'unique_evidence_nodes','edge_count','shared_edge_count','shuffled_changed_edge_count',
    'geometry_valid_candidate_count','connected_member_count','candidate_degree','candidate_shared_mask',
    'candidate_depths_m','candidate_offsets_m','geometry_valid','ra_valid','re_valid','usable',
    'weights','evidence_available'}
GLOBAL_DIAGNOSTICS={'query_rounds','graph_variant','graph_architecture','physical_psf_proxy',
    'candidate_sharing_denominator','member_sharing_denominator','evidence_node_key',
    'shuffled_attributes','degree_is_not_true_target_count'}


def _row_diagnostic(value,index,batch_size):
    """Extract active-frame fields; never interpret evidence-node arrays as frames."""
    if not isinstance(value,dict):return {}
    source=np.asarray(value.get('source_batch_indices',[]),int).reshape(-1)
    matches=np.flatnonzero(source==index);active=int(matches[0]) if len(matches) else None
    result={k:v for k,v in value.items() if k in GLOBAL_DIAGNOSTICS}
    for key in FRAME_DIAGNOSTICS:
        if key not in value:continue
        arr=np.asarray(value[key])
        if arr.ndim and len(arr)==len(source):
            result[key]=arr[active] if active is not None else (0 if arr.ndim==1 else [])
    result['frame_had_radar_inputs']=active is not None
    return result


def infer(config_path,checkpoint_path,output_dir,*,device='cpu',split='dev',batch_size=4):
    require(split in ('train','dev'),'Historical TEST is reserved')
    require(type(batch_size) is int and batch_size>0,'Positive inference batch required')
    config=ExperimentConfig.load(config_path);manifest=load_manifest(config.manifest)
    destination=external_path(output_dir,field='MCE sensor-only predictions')
    require(not destination.exists(),'Inference output exists; use an independent directory')
    model,payload,checkpoint_sha=load_graph_checkpoint(checkpoint_path,device=device)
    metadata=payload.get('metadata',{})
    require(metadata.get('training_sources')==training_sources(),'Checkpoint and running graph core sources differ')
    require(metadata.get('manifest_sha256')==manifest.sha256,'Checkpoint/data manifest differ')
    require(metadata.get('completed_epochs',0)>0 and metadata.get('optimizer_updates',0)>0 and not metadata.get('synthetic_only',False),'Actually trained checkpoint required')
    require(metadata.get('center_contract')=='observer_scatter_at_camera_optical_center'
        and metadata.get('observer_position_reference')=='camera_optical_center'
        and metadata.get('body_navigation_is_model_input') is False,'Camera-centre provenance differs')
    started=time.perf_counter();pending=[];frames=[];compute_s=0.;profile_observations=[];batch_artifacts=[]
    def flush():
        nonlocal compute_s
        if not pending:return
        packed=pack_features([o.features for o in pending],device=device,visual=False)
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        tick=time.perf_counter()
        with torch.inference_mode():output=forward(model,packed)
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        batch_elapsed=time.perf_counter()-tick;compute_s+=batch_elapsed
        batch=len(pending)
        raw_graph=output.get('graph_diagnostics',{})
        graph_batch=_detach(raw_graph)
        query_batch=graph_batch if output.get('sampling_diagnostics') is raw_graph else _detach(output.get('sampling_diagnostics',{}))
        batch_path=destination/'graph_batches'/f'batch_{len(batch_artifacts):05d}.npz'
        batch_path.parent.mkdir(parents=True,exist_ok=True)
        arrays={key:value for key,value in graph_batch.items() if isinstance(value,np.ndarray)}
        np.savez_compressed(batch_path,**arrays)
        batch_artifacts.append(dict(path=batch_path.relative_to(destination).as_posix(),
            sha256=file_sha256(batch_path),frames=[dict(episode_id=o.episode_id,frame_idx=o.frame_idx) for o in pending],
            source_batch_indices=graph_batch.get('source_batch_indices',np.array([],int)).tolist(),
            evidence_arrays_are_batch_level=True))
        for i,observation in enumerate(pending):
            n=len(observation.features['ray_directions'])
            tensor=lambda key:output[key][i,:n].detach().cpu().numpy()
            diagnostic=_row_diagnostic(query_batch,i,batch)
            graph=_row_diagnostic(graph_batch,i,batch)
            features=observation.features
            frames.append(jsonable(dict(episode_id=observation.episode_id,frame_idx=observation.frame_idx,
                physical_time_s=observation.physical_time_s,positions=tensor('member_positions'),
                valid=tensor('valid_mask'),depths=tensor('depths'),member_ids=[None]*n,
                scores=features['detection_scores'],bbox_xyxy=features['bbox_xyxy'],
                visual_depth_mean_m=tensor('visual_depth_mean_m'),visual_depth_std_m=tensor('visual_depth_std_m'),
                ray_origins=features['ray_origins'],ray_directions=features['ray_directions'],
                ray_depth_bounds=features['ray_depth_bounds'],range_scale_m=features['range_scale_m'],
                source_hashes=observation.source_hashes,architecture=ARCHITECTURE,
                radar_query=diagnostic,graph_diagnostics=graph,
                graph_batch_artifact=batch_artifacts[-1]['path'],graph_batch_source_row=i,
                runtime_s=batch_elapsed/batch)))
        pending.clear()
    for observation in manifest.observations(split,modality='rgb_radar',prepared_only=True):
        pending.append(observation)
        if len(profile_observations)<32:profile_observations.append(observation)
        if len(pending)==batch_size:flush()
        if len(frames) and len(frames)%500==0:
            print(f'MCE_INFER {payload["variant"]} split={split} frames={len(frames)} elapsed_s={time.perf_counter()-started:.1f}',flush=True)
    flush();require(bool(frames),'No inference frames')
    elapsed=time.perf_counter()-started
    serial=[]
    with torch.inference_mode():
        for observation in profile_observations:
            packed=pack_features([observation.features],device=device,visual=False)
            if str(device).startswith('cuda'):torch.cuda.synchronize(device)
            tick=time.perf_counter();forward(model,packed)
            if str(device).startswith('cuda'):torch.cuda.synchronize(device)
            serial.append(time.perf_counter()-tick)
    cost=dict(wall_s=elapsed,wall_scope='batched prediction pass including sensor IO/validation/graph NPZ, excluding final JSON serialization and separate serial profile',
        frames=len(frames),throughput_frames_s=len(frames)/elapsed,
        batched_model_total_s=compute_s,batch_size=batch_size,device=str(device),parallel_workers=1,
        serial_latency_measured=True,serial_profile_frames=len(serial),
        serial_model_mean_s=float(np.mean(serial)),serial_model_p50_s=float(np.percentile(serial,50)),
        serial_model_p95_s=float(np.percentile(serial,95)),serial_profile_total_s=sum(serial),
        serial_profile_scope='locator forward only, warmed by preceding DEV batch pass; first 32 declared DEV frames',
        input_mode='authenticated_prepared_roi_ra_re',
        includes_rgb_detector=False,includes_raw_iq_fft=False,includes_asset_io_and_validation=True,
        timing_scope='batched locator plus declared cached sensor input; not complete RGB/IQ frontend')
    result=dict(schema='rvloc.predictions.v1',method_id='mce_'+payload['variant'],
        variant=payload['variant'],neural_architecture=ARCHITECTURE,split=split,
        manifest_sha256=manifest.sha256,config_sha256=config.sha256,checkpoint_sha256=checkpoint_sha,
        source=source_identity(),graph_sources=metadata['training_sources'],tool_sha256=file_sha256(__file__),coordinate_frame='base_local',
        normal_inference=True,labels_opened=False,inference_uses_gt_ids=False,
        evaluation_contract=dict(matching='frozen_roi_to_gt_correspondence',index_determined_before_predictions=True),
        frames=frames,cost=cost,graph_batch_artifacts=batch_artifacts)
    path=write_json(destination/'predictions.json',result)
    write_json(destination/'reproduction_receipt.json',dict(status='inference_completed',
        predictions=str(path),predictions_sha256=file_sha256(path),config_sha256=config.sha256,
        manifest_sha256=manifest.sha256,checkpoint_sha256=checkpoint_sha,cost=cost,
        source=result['source'],graph_sources=metadata['training_sources'],
        full_stage_elapsed_before_receipt_s=time.perf_counter()-started,
        normal_inference=True,labels_opened=False))
    return dict(status='inference_completed',predictions=str(path),cost=cost)


def graph_metrics(frames):
    """Aggregate documented observable graph diagnostics without GT claims."""
    fields=Counter();samples=Counter()
    for frame in frames:
        diagnostic=frame.get('graph_diagnostics',{})
        if not isinstance(diagnostic,dict):continue
        for key,value in diagnostic.items():
            if isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value):
                fields[key]+=float(value);samples[key]+=1
    result=dict(schema='rvloc.mce_graph_metrics.v1',frames=len(frames),
        observable_only=True,GT_used=False,raw_numeric_sums=dict(fields),
        numeric_frame_means={k:fields[k]/samples[k] for k in fields},
        coverage_definition='Shared means evidence used by candidates of different observed members, not merely different candidates of one member; per-frame diagnostics remain in predictions.json and complete batched arrays remain in SHA-linked graph_batches NPZ files')
    # Derive ratios only when the model actually exports suitable counts.
    for out,numerator,denominator in (
        ('shared_candidate_coverage','shared_candidate_count','valid_candidate_count'),
        ('shared_member_coverage','shared_member_count','valid_member_count'),
        ('shared_edge_coverage','shared_edge_count','edge_count')):
        if fields.get(denominator,0)>0:result[out]=fields.get(numerator,0)/fields[denominator]
    return result


def evaluate(index_path,predictions_path,output_dir,*,draws=2000,seed=20261126):
    index_path=Path(index_path).resolve();index=json.loads(index_path.read_text())
    require(index.get('schema')=='rvloc.fixed_roi_gt_index.v1' and index.get('supervision_only') is True,'Fixed identity index required')
    pred_path=Path(predictions_path).resolve();prediction=json.loads(pred_path.read_text())
    require(prediction.get('schema')=='rvloc.predictions.v1' and prediction.get('normal_inference') is True and prediction.get('labels_opened') is False,'Committed normal predictions required')
    require(prediction.get('neural_architecture')==ARCHITECTURE,'Not an MCE graph prediction')
    require(prediction.get('manifest_sha256')==index.get('manifest_sha256'),'Data identities differ')
    for artifact in prediction.get('graph_batch_artifacts',[]):
        path=(pred_path.parent/artifact['path']).resolve()
        require(path.is_relative_to(pred_path.parent),'Graph batch artifact escaped inference root')
        require(file_sha256(path)==artifact['sha256'],'Graph batch artifact SHA differs')
    split=prediction['split'];require(split in ('train','dev'),'Historical TEST is reserved')
    expected={(r['episode_id'],r['frame_idx']):r for r in index['frames'] if r['split']==split}
    frames={(r['episode_id'],r['frame_idx']):r for r in prediction['frames']}
    require(len(frames)==len(prediction['frames']) and frames.keys()==expected.keys(),'Prediction frame population differs')
    destination=external_path(output_dir,field='MCE fixed evaluation')
    require(not destination.exists(),'Evaluation exists; choose another independent directory')
    rows=[];counts=Counter();coverage=[]
    for key,ref in expected.items():
        path=(index_path.parent/ref['path']).resolve();require(path.is_relative_to(index_path.parent),'Fixed annotation escaped index root')
        require(file_sha256(path)==ref['sha256'],'Frozen annotation SHA differs')
        frozen=json.loads(path.read_text());current,diagnostic=score_fixed_frame(frames[key],frozen)
        rows.extend(current);counts.update(diagnostic)
        query=frames[key].get('radar_query',{})
        depths=np.asarray(query.get('candidate_depths_m',[]),float)
        if depths.ndim==2:
            for pair in frozen['training_equivalent_assignment']:
                index_row=pair['detection_index']
                if index_row<len(depths) and depths.shape[1]:
                    target=pair['target_depth_m'];proposed=depths[index_row]
                    finite=proposed[np.isfinite(proposed)]
                    if len(finite):coverage.append(dict(span_covered=bool(finite.min()<=target<=finite.max()),nearest_depth_distance_m=float(np.min(np.abs(finite-target)))))
    summary=metrics(rows);summary.update(counts)
    summary.update(matching='fixed ROI-to-GT training correspondence; no predicted-position reassignment',
        matching_max_distance_m=None,not_tracking_identity_accuracy=True,
        uncertainty=episode_bootstrap(rows,draws=draws,seed=seed))
    graph=graph_metrics(prediction['frames'])
    graph['offline_candidate_support']=dict(scored_members=len(coverage),
        depth_span_coverage=float(np.mean([x['span_covered'] for x in coverage])) if coverage else None,
        mean_nearest_depth_distance_m=float(np.mean([x['nearest_depth_distance_m'] for x in coverage])) if coverage else None,
        GT_used_only_after_committed_predictions=True,
        interpretation='Visual-depth candidate bracket diagnostic; not radar detectability or an oracle deployment score')
    result=dict(schema='rvloc.fixed_roi_evaluation.v1',method_id=prediction['method_id'],
        variant=prediction['variant'],neural_architecture=ARCHITECTURE,split=split,
        manifest_sha256=prediction['manifest_sha256'],frozen_index_sha256=file_sha256(index_path),
        predictions_sha256=file_sha256(pred_path),predictions_path=str(pred_path),
        summary=summary,rows=rows,graph_metrics=graph,cost=prediction['cost'],
        episodes={ep:metrics([r for r in rows if r['episode_id']==ep]) for ep in sorted({r['episode_id'] for r in rows})})
    path=write_json(destination/'evaluation.json',result)
    write_json(destination/'localization_metrics.json',summary);write_json(destination/'graph_metrics.json',graph)
    write_json(destination/'episode_level_results.json',result['episodes'])
    write_json(destination/'failure_cases.json',[r for r in rows if r['error_m'] is None or r['error_m']>5])
    return dict(status='evaluation_completed',evaluation=str(path),summary=summary)


def compare(first_path,second_path,output_path,*,draws=2000,seed=20261126):
    first=json.loads(Path(first_path).read_text());second=json.loads(Path(second_path).read_text())
    for value in (first,second):require(value.get('schema')=='rvloc.fixed_roi_evaluation.v1','Compare only fixed-member evaluations')
    require(first.get('frozen_index_sha256')==second.get('frozen_index_sha256'),'Fixed label identities differ')
    # The existing paired bootstrap helper is schema-gated but mathematically
    # generic over GT-keyed rows. Adapt in memory; original artifacts stay exact.
    a=dict(first,schema='rvloc.evaluation.v1');b=dict(second,schema='rvloc.evaluation.v1')
    result=paired_compare(a,b,draws=draws,seed=seed)
    result.update(schema='rvloc.mce_graph_paired.v1',source_schemas=[first['schema'],second['schema']],
        first_path=str(Path(first_path).resolve()),second_path=str(Path(second_path).resolve()),
        first_sha256=file_sha256(first_path),second_sha256=file_sha256(second_path),
        fixed_members_only=True)
    write_json(output_path,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    inf=commands.add_parser('infer');inf.add_argument('--config',required=True);inf.add_argument('--checkpoint',required=True)
    inf.add_argument('--output-dir',required=True);inf.add_argument('--device',default='cpu');inf.add_argument('--split',choices=('train','dev'),default='dev');inf.add_argument('--batch-size',type=int,default=4)
    ev=commands.add_parser('evaluate');ev.add_argument('--index',required=True);ev.add_argument('--predictions',required=True);ev.add_argument('--output-dir',required=True)
    pair=commands.add_parser('compare');pair.add_argument('--first',required=True);pair.add_argument('--second',required=True);pair.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.command=='infer':result=infer(args.config,args.checkpoint,args.output_dir,device=args.device,split=args.split,batch_size=args.batch_size)
    elif args.command=='evaluate':result=evaluate(args.index,args.predictions,args.output_dir)
    else:result=compare(args.first,args.second,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
