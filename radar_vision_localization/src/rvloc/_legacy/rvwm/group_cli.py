"""RA/RE-conditioned continuous ray localization; no hidden IQ scoring."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from .data import Dataset,sha256
from .group_evaluation import GroupEvaluator,MemberEvaluator
from .group_localization import GroupSceneLocalizer
from .group_model import GroupSceneModel,load_group_model,GROUP_SCHEMA
from .types import ObservationWindow
from .validation import write_json
from .partitions import active_splits,partition_receipt,checkpoint_partition_receipt
from .single_frame import require_single_frame,validate_model_features
from .phase_selection import configured_phases,selected_frame_indices,required_phase

ROOT=Path(__file__).resolve().parents[1]


def build_plan(dataset,config,args):
    require_single_frame(config)
    if args.plan:
        value=json.loads(Path(args.plan).read_text())
        rows=value if isinstance(value,list) else value.get('selection',value.get('all_selection'))
        if not rows:raise ValueError('A nonempty causal frame plan is required')
    else:
        if not args.episodes:raise ValueError('Supply --episodes or --plan')
        rows=[]
        for episode in args.episodes:
            available=selected_frame_indices(dataset,episode,configured_phases(config,'inference'))
            for index in range(args.start_frame,min(dataset.frames(episode),args.start_frame+args.frames)):
                if index not in available:raise ValueError(f'{episode}/{index} is outside configured inference phases')
                rows.append(dict(episode_id=episode,frame_idx=index,history=[index]))
    seen=set();result=[]
    for row in rows:
        key=(row['episode_id'],int(row['frame_idx']))
        declared=list(row.get('history',[key[1]]))
        available=set(selected_frame_indices(dataset,key[0],configured_phases(config,'inference')))
        valid=(key not in seen and len(declared)==1 and type(declared[0]) is int
            and declared==[key[1]] and key[1] in available)
        if not valid:
            raise ValueError('Single-frame plan requires one current frame in configured phases; history, future and duplicate entries are forbidden')
        seen.add(key);result.append(dict(episode_id=key[0],frame_idx=key[1],history=declared))
    if not result:raise ValueError('No requested frames in configured phases')
    if args.rank<0 or args.rank>=args.world_size:raise ValueError('Invalid inference shard')
    return result,result[args.rank::args.world_size]


def _frontend(config,args):
    from .visual_refinement import ExportedRGBFrontend
    # Existing exports are a strict execution of the frozen RGB detector.
    # They contain boxes/scores, never labels. Missing receipts fail explicitly.
    frontend=ExportedRGBFrontend(config,split=args.split,device=args.device)
    frontend.metadata.update(appearance='not_used_by_localizer',localizer_input='bbox_center_ray_geometry_only')
    return frontend


def save_spectra(path,features):
    """Save one current frame as numeric arrays and JSON, never pickle."""
    from .__main__ import sanitize
    from .group_observation import validate_features
    validate_model_features(features,dict(window=1))
    validate_features(features)
    values={}
    for key,value in features.items():
        if key in ('metadata','sensor_context','history'):continue
        array=np.asarray(value)
        if array.dtype.hasobject:raise ValueError('Spectrum archives cannot contain pickled/object arrays')
        values[key]=array
    for key in ('metadata','sensor_context'):
        if key in features:
            values[key+'_json']=np.array(json.dumps(sanitize(features[key]),sort_keys=True,allow_nan=False))
    np.savez_compressed(path,**values)


def load_spectra(path,config):
    """Load one frame of numeric rays and spectra without sensor/label access."""
    from .group_observation import build_spectral_features,FEATURE_SCHEMA
    require_single_frame(config)
    with np.load(path,allow_pickle=False) as data:
        if 'history_count' in data:
            value=data['history_count']
            if value.shape!=() or value.dtype.kind not in 'iu' or int(value)!=0:
                raise ValueError('Single-frame spectrum archives cannot contain history')
        if any(key.startswith('history') and key!='history_count' for key in data.files):
            raise ValueError('Single-frame spectrum archives cannot contain history')
        if 'schema' in data and str(data['schema'].item())!=FEATURE_SCHEMA:
            raise ValueError('Spectral input schema mismatch')
        required=('ray_origins','ray_directions','observer_position','ra_power','re_power',
                  'range_m','azimuth_rad','elevation_rad')
        values={key:data[key].copy() for key in required}
        for key in ('range_scale_m','ray_valid','ra_valid','re_valid'):
            if key in data:values[key]=data[key].copy()
        if 'metadata_json' in data:values['metadata']=json.loads(str(data['metadata_json'].item()))
        features=build_spectral_features(**values)
        if 'sensor_context_json' in data:features['sensor_context']=json.loads(str(data['sensor_context_json'].item()))
    validate_model_features(features,config)
    return features


def infer_spectra(args,config):
    from .__main__ import output_path
    features=load_spectra(args.input,config)
    validation=bool(args.allow_untrained_validation)
    if validation:
        if args.world_checkpoint:raise ValueError('Untrained validation does not load weights')
        torch.manual_seed(int(config['seed']))
        model=GroupSceneModel(config).to(args.device).eval();weight=dict(status='untrained_validation',steps=0,path=None,sha256=None)
    else:
        path=args.world_checkpoint or Path(config['world_checkpoint'])
        model=load_group_model(path,config,args.device);weight=dict(path=str(Path(path).resolve()),sha256=sha256(path),steps=model.checkpoint_metadata['steps'])
        from .source_compatibility import verify_inference_sources
        source_compatibility=verify_inference_sources(path,model.checkpoint_metadata)
    config['world_model_trained']=bool(model.trained)
    localizer=GroupSceneLocalizer(config,None,model,allow_untrained_validation=validation)
    started=time.perf_counter()
    result=localizer.estimate_spectra(features,episode_id=args.episode_id,frame_idx=args.frame_idx,timestamp_s=args.timestamp_s)
    result['solver_diagnostics']['elapsed_s']=time.perf_counter()-started
    output=output_path(args.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'predictions.jsonl').exists():raise FileExistsError('Existing predictions protected')
    from .__main__ import sanitize
    (output/'predictions.jsonl').write_text(json.dumps(sanitize(result),ensure_ascii=False,allow_nan=False)+'\n')
    write_json(output/'configuration.json',config)
    write_json(output/'manifest.json',dict(schema=config.get('architecture_schema',GROUP_SCHEMA),status='complete',configuration=config,world_checkpoint=weight,
        input_path=str(args.input.resolve()),input_sha256=sha256(args.input),validation_only=validation,
        ideal_observation_mode=False,seed=config['seed'],uncertainty_calibrated=False,
        feature_metadata=features['metadata'],feature_sha256=result['solver_diagnostics']['feature_sha256'],
        source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        source_compatibility=locals().get('source_compatibility'),
        input_frames=1,
        input_contract='one current frame of base rays, observer navigation, numeric RA/RE and physical axes'))
    print(json.dumps(dict(output=str(output),status=result['group']['localization_status'],detected_count=result['detected_count'])))


def infer(args,config):
    from .__main__ import output_path,sanitize
    if getattr(args,'mode','rgb')!='rgb':raise ValueError('Only RGB detection input is supported')
    for argument,key in (('window','window'),('hypotheses','group_hypotheses'),('refinement_steps','group_refinement_steps')):
        value=getattr(args,argument)
        if value is not None:config[key]=value
    window=require_single_frame(config)
    if int(config.get('group_refinement_steps',0))!=0:
        raise ValueError('RA/RE-only model has no IQ position refinement; use --refinement-steps 0')
    if getattr(args,'observer_handling',None):config['observer_handling']=args.observer_handling
    if args.detector_weights:config['detector_weights']=str(Path(args.detector_weights).resolve())
    config.update(device=args.device,track_detection_split=args.split)
    validation_only=bool(getattr(args,'allow_untrained_validation',False))
    if validation_only and args.evaluate:
        raise ValueError('Untrained engineering validation cannot be reported as localization performance; omit --evaluate')
    checkpoint=Path(args.world_checkpoint or config.get('world_checkpoint',''))
    if validation_only:
        if args.world_checkpoint:raise ValueError('Untrained validation does not load checkpoints')
        torch.manual_seed(int(config['seed']))
        model=GroupSceneModel(config).to(args.device).eval();trained={}
        weight_receipt=dict(path=None,sha256=None,steps=0,train_episodes=[],validation_episodes=[],status='untrained_validation')
    else:
        if not checkpoint.is_file():
            raise FileNotFoundError('Trained single-frame RA/RE localization weights required: '+str(checkpoint))
        model=load_group_model(checkpoint,config,args.device);trained=model.checkpoint_metadata
        weight_receipt=dict(path=str(checkpoint.resolve()),sha256=sha256(checkpoint),steps=trained['steps'],
            train_episodes=trained.get('train_episodes',[]),validation_episodes=trained.get('validation_episodes',[]))
        if args.mode=='rgb':
            expected=trained.get('frontend',{})
            for field,path in (('weights_sha256',config['detector_weights']),):
                if path and expected.get(field)!=sha256(path):raise ValueError('Weights belong to another RGB frontend: '+field)
            from .detector_contract import verify_trained_detector_contract
            verify_trained_detector_contract(config,expected)
            from .source_compatibility import verify_inference_sources
            source_compatibility=verify_inference_sources(checkpoint,trained)
            for key in ('ra_re_azimuth_bins','ra_re_elevation_bins','ra_re_spatial_oversampling','visual_point_mode','pixel_std'):
                if key in model.config and config.get(key)!=model.config[key]:
                    raise ValueError('Deployed feature configuration differs from training: '+key)
    config['world_model_trained']=bool(model.trained)
    dataset=Dataset(config['dataset_root'],echo_subdir=config['echo_subdir']);args.dataset=dataset
    plan,local=build_plan(dataset,config,args)
    strict=active_splits(dataset,config)
    allowed=set(strict[args.split])
    lineage=dict(status='untrained_validation') if validation_only else checkpoint_partition_receipt(trained,dataset,config)
    if not {r['episode_id'] for r in plan}<=allowed:
        raise ValueError('Inference plan is outside the requested episode partition')
    overlap=sorted({r['episode_id'] for r in plan}&set(trained.get('train_episodes',[])))
    if args.split!='train' and overlap:raise ValueError('Evaluation episodes overlap group training')
    output=output_path(args.output);output.mkdir(parents=True,exist_ok=True)
    prefix=output/f'part_{args.rank}' if args.world_size>1 else output/'run'
    predictions=output/f'part_{args.rank}.predictions.jsonl' if args.world_size>1 else output/'predictions.jsonl'
    if predictions.exists():raise FileExistsError('Existing predictions protected: '+str(predictions))
    manifest_path=output/f'part_{args.rank}.manifest.json' if args.world_size>1 else output/'manifest.json'
    evaluation_path=output/f'part_{args.rank}.evaluation.json' if args.world_size>1 else output/'evaluation.json'
    front=None;began=time.perf_counter();inputs=[];evaluator=GroupEvaluator();member_evaluator=MemberEvaluator()
    manifest=dict(schema=config.get('architecture_schema',GROUP_SCHEMA),status='preparing',configuration=config,rank=args.rank,world_size=args.world_size,
        input_frames=window,
        selection=local,all_selection=plan,split=args.split,evaluation_scope=args.evaluation_scope,
        ideal_observation_mode=False,uncertainty_calibrated=False,
        world_checkpoint=weight_receipt,validation_only=validation_only,
        model_training_overlap=overlap,partition=partition_receipt(config),checkpoint_partition_lineage=lineage,
        source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        source_compatibility=locals().get('source_compatibility'),
        input_provenance=inputs,scope='current detected members in base_local, detected-set centre and continuous structure',
        timestamp_policy='physical frame timestamp at 1s intervals; pulse PRI independent; current frame in configured phases only',
        radar_preprocessing='rectangular RA/RE with shared physical range axis; correlated power projections; no raw IQ scoring',
        position_correction=False,visual_constraint='exact calibrated bbox-centre ray',
        expected_frames=len(local),completed_frames=0)
    try:
        front=_frontend(config,args);manifest.update(frontend=front.metadata,status='running')
        if args.mode=='rgb' and trained.get('frontend',{}).get('keypoint_mode') not in (None,front.metadata.get('keypoint_mode')):
            raise ValueError('Training/inference visual point modes differ')
        localizer=GroupSceneLocalizer(config,front,model,allow_untrained_validation=validation_only)
        write_json(output/'configuration.json',config);write_json(manifest_path,manifest)
        with predictions.open('x') as stream:
            for row in local:
                t0=time.perf_counter()
                observations=[dataset.observations(row['episode_id'],i,required_phase=required_phase(configured_phases(config,'inference'))) for i in row['history']]
                result=localizer.estimate(ObservationWindow(tuple(observations)))
                if getattr(args,'save_spectra',False) and localizer.last_features is not None:
                    save_spectra(output/f"{row['episode_id']}_{row['frame_idx']:05d}.spectra.npz",localizer.last_features)
                if str(args.device).startswith('cuda'):torch.cuda.synchronize()
                elapsed=time.perf_counter()-t0
                result['solver_diagnostics']['elapsed_s']=elapsed
                stream.write(json.dumps(sanitize(result),ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                inputs.append(dict(**row,hashes=[dict(frame_idx=o.frame_idx,hashes=o.source_hashes) for o in observations],
                    feature_sha256=result['solver_diagnostics'].get('feature_sha256')))
                if args.evaluate:
                    # Truth is opened only AFTER the normal result is committed.
                    truth=dataset.targets(row['episode_id'],row['frame_idx'])
                    evaluator.update(result,truth,observations[-1],None)
                    member_evaluator.update(result,truth,observations[-1])
                    write_json(evaluation_path,dict(summary=evaluator.summary(),frames=evaluator.frames,
                        members=dict(summary=member_evaluator.summary(),frames=member_evaluator.frames)))
                manifest.update(completed_frames=manifest['completed_frames']+1,seconds=time.perf_counter()-began)
                write_json(manifest_path,manifest)
                print(json.dumps(dict(episode_id=row['episode_id'],frame_idx=row['frame_idx'],completed_frames=manifest['completed_frames'],
                    expected_frames=len(local),detected_count=result['detected_count'],
                    localization_status=result['group']['localization_status'],seconds=elapsed),ensure_ascii=False),flush=True)
        manifest.update(status='complete',predictions_sha256=sha256(predictions),seconds=time.perf_counter()-began)
    except BaseException as error:
        manifest.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        write_json(manifest_path,manifest)
        if front is not None:front.close()
    print(json.dumps(dict(output=str(output),summary=evaluator.summary() if args.evaluate else None),ensure_ascii=False),flush=True)


def evaluate(args,config):
    from .__main__ import output_path
    dataset=Dataset(config['dataset_root'],echo_subdir=config['echo_subdir']);evaluator=GroupEvaluator();members=MemberEvaluator()
    for line in Path(args.predictions).read_text().splitlines():
        if not line.strip():continue
        result=json.loads(line);ep,index=result['episode_id'],result['frame_idx']
        observation=dataset.observations(ep,index,required_phase=required_phase(configured_phases(config,'inference')))
        if result.get('solver_diagnostics',{}).get('validation_only'):raise ValueError('Engineering validation is not localization performance')
        truth=dataset.targets(ep,index)
        if result.get('schema')!='ra_re_member_localization_v1':raise ValueError('Only current member-localization predictions can be evaluated')
        evaluator.update(result,truth,observation,None);members.update(result,truth,observation)
    report=dict(summary=evaluator.summary(),frames=evaluator.frames,members=dict(summary=members.summary(),frames=members.frames),predictions_sha256=sha256(args.predictions),
        uncertainty_calibrated=False,ideal_observation_mode='read originating run manifest')
    write_json(output_path(args.output),report);print(json.dumps(report['summary'],ensure_ascii=False))


def merge(args,config):
    """Verify every shard and independently reevaluate the saved group outputs."""
    from .__main__ import output_path,sanitize
    output=output_path(args.output);manifests=[];records={};saved_frames=[];saved_member_frames=[]
    key=lambda row:(row['episode_id'],int(row['frame_idx']))
    for rank in range(args.world_size):
        path=output/f'part_{rank}.manifest.json';m=json.loads(path.read_text())
        if m.get('schema')!=GROUP_SCHEMA:raise ValueError('Only current single-frame localization shards can be merged')
        if m['status']!='complete' or m['rank']!=rank or m['world_size']!=args.world_size:
            raise ValueError('Incomplete inference shard: '+str(rank))
        planned=m['all_selection']
        if len({key(r) for r in planned})!=len(planned) or m['selection']!=planned[rank::args.world_size]:
            raise ValueError('Inference shard selection differs from the declared unique frame plan')
        artifact=Path(m['world_checkpoint']['path'])
        if not artifact.is_file() or sha256(artifact)!=m['world_checkpoint']['sha256']:
            raise ValueError('Inference checkpoint artifact changed after prediction')
        if manifests:
            for field in ('configuration','world_checkpoint','source_sha256','all_selection','split','evaluation_scope','ideal_observation_mode'):
                if manifests[0][field]!=m[field]:raise ValueError('Inference shards differ: '+field)
        pred=output/f'part_{rank}.predictions.jsonl'
        if sha256(pred)!=m['predictions_sha256']:raise ValueError('Changed prediction shard')
        rows=[json.loads(line) for line in pred.read_text().splitlines() if line]
        if len(rows)!=len(m['selection']) or {key(r) for r in rows}!={key(r) for r in m['selection']}:
            raise ValueError('Inference shard does not cover its declared frames')
        for row in rows:
            if key(row) in records:raise ValueError('Duplicate prediction in shards')
            records[key(row)]=row
        evaluation=output/f'part_{rank}.evaluation.json'
        if not evaluation.is_file():raise ValueError('Run every shard with --evaluate before merge')
        saved=json.loads(evaluation.read_text());check=GroupEvaluator();check.frames=saved['frames']
        if len(check.frames)!=len(m['selection']) or {key(r) for r in check.frames}!={key(r) for r in m['selection']}:
            raise ValueError('Shard evaluation omits or duplicates a declared frame')
        if check.summary()!=saved['summary']:raise ValueError('Shard evaluation summary is inconsistent')
        if 'members' in saved:
            member_check=MemberEvaluator();member_check.frames=saved['members']['frames']
            if len(member_check.frames)!=len(m['selection']) or {key(r) for r in member_check.frames}!={key(r) for r in m['selection']}:
                raise ValueError('Shard member evaluation omits or duplicates a frame')
            if member_check.summary()!=saved['members']['summary']:raise ValueError('Shard member summary is inconsistent')
            saved_member_frames.extend(member_check.frames)
        else:raise ValueError('RA/RE shard member evaluation missing')
        saved_frames.extend(saved['frames']);manifests.append(m)
    m=manifests[0];expected={key(r) for r in m['all_selection']}
    if set(records)!=expected:raise ValueError('Merged predictions do not cover the complete declared plan')
    cfg=m['configuration'];dataset=Dataset(cfg['dataset_root'],echo_subdir=cfg['echo_subdir']);evaluator=GroupEvaluator();member_evaluator=MemberEvaluator()
    for k in sorted(records):
        observation=dataset.observations(k[0],k[1],required_phase=required_phase(configured_phases(config,'inference')))
        truth=dataset.targets(k[0],k[1])
        if records[k].get('schema')!='ra_re_member_localization_v1':raise ValueError('Only current member-localization predictions can be merged')
        evaluator.update(records[k],truth,observation,None)
        member_evaluator.update(records[k],truth,observation)
    check=GroupEvaluator();check.frames=sorted(saved_frames,key=key)
    if check.frames!=evaluator.frames:raise ValueError('Independent per-frame evaluation differs from worker records')
    if evaluator.summary()!=check.summary():raise ValueError('Independent evaluation differs from worker evaluation')
    if saved_member_frames:
        member_check=MemberEvaluator();member_check.frames=sorted(saved_member_frames,key=key)
        if member_check.frames!=member_evaluator.frames or member_check.summary()!=member_evaluator.summary():
            raise ValueError('Independent member evaluation differs from worker records')
    predictions=output/'predictions.jsonl'
    payload=''.join(json.dumps(sanitize(records[k]),ensure_ascii=False,allow_nan=False)+'\n' for k in sorted(records))
    if predictions.exists() and predictions.read_text()!=payload:raise FileExistsError('Different merged predictions already exist')
    predictions.write_text(payload)
    merged=dict(m,rank=None,selection=m['all_selection'],completed_frames=len(records),expected_frames=len(records),
        input_provenance=[r for part in manifests for r in part['input_provenance']],
        predictions_sha256=sha256(predictions),workers=[dict(rank=p['rank'],seconds=p['seconds'],
            manifest_sha256=sha256(output/f"part_{p['rank']}.manifest.json")) for p in manifests])
    write_json(output/'manifest.json',merged)
    write_json(output/'evaluation.json',dict(summary=evaluator.summary(),frames=evaluator.frames,
        per_episode={ep:_episode_summary(evaluator.frames,ep) for ep in sorted({k[0] for k in records})},
        independent_reevaluation_verified=True,predictions_sha256=sha256(predictions),uncertainty_calibrated=False,
        members=dict(summary=member_evaluator.summary(),frames=member_evaluator.frames)))
    print(json.dumps(evaluator.summary(),ensure_ascii=False))


def _episode_summary(frames,episode):
    evaluator=GroupEvaluator();evaluator.frames=[r for r in frames if r['episode_id']==episode]
    return evaluator.summary()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=ROOT/'config.json')
    p.add_argument('--dataset-root');p.add_argument('--threads',type=int,default=1)
    sub=p.add_subparsers(dest='command',required=True)
    auditing=sub.add_parser('audit');auditing.add_argument('--episodes',nargs='+')
    auditing.add_argument('--output',default='results/group_scene/data_audit.json')
    train=sub.add_parser('train');train.add_argument('--kind',choices=['world'],default='world')
    train.add_argument('--output',type=Path,required=True);train.add_argument('--steps',type=int,default=20000)
    train.add_argument('--max-episodes',type=int,default=0);train.add_argument('--frames-per-episode',type=int,default=0)
    train.add_argument('--validation-episodes',type=int,default=0);train.add_argument('--validation-frames',type=int,default=0)
    train.add_argument('--validate-every',type=int,default=1000);train.add_argument('--batch-size',type=int,default=8)
    train.add_argument('--learning-rate',type=float,default=3e-4);train.add_argument('--physics-every',type=int,default=0)
    train.add_argument('--physics-weight',type=float,default=0.);train.add_argument('--detection-dropout',type=float,default=.2)
    train.add_argument('--seed',type=int,default=20261118);train.add_argument('--device',default='cpu')
    train.add_argument('--dry-run',action='store_true',help='Prepare inputs and check gradients without optimization')
    infer_parser=sub.add_parser('infer');infer_parser.add_argument('--episodes',nargs='+');infer_parser.add_argument('--plan',type=Path)
    infer_parser.add_argument('--split',choices=['train','val','test'],default='val')
    infer_parser.add_argument('--evaluation-scope',choices=['development','heldout'],default='development')
    infer_parser.add_argument('--start-frame',type=int,default=0);infer_parser.add_argument('--frames',type=int,default=1)
    infer_parser.add_argument('--window',type=int,choices=[1],help='Exactly one current frame per sample')
    infer_parser.add_argument('--hypotheses',type=int)
    infer_parser.add_argument('--refinement-steps',type=int);infer_parser.add_argument('--world-checkpoint',type=Path)
    infer_parser.add_argument('--observer-handling',choices=['condition'])
    infer_parser.add_argument('--detector-weights')
    infer_parser.add_argument('--mode',choices=['rgb'],default='rgb')
    infer_parser.add_argument('--rank',type=int,default=0);infer_parser.add_argument('--world-size',type=int,default=1)
    infer_parser.add_argument('--device',default='cpu');infer_parser.add_argument('--output',required=True)
    infer_parser.add_argument('--evaluate',action='store_true')
    infer_parser.add_argument('--allow-untrained-validation',action='store_true')
    infer_parser.add_argument('--save-spectra',action='store_true')
    direct=sub.add_parser('infer-spectra',help='Only numeric rays, observer position and RA/RE; no dataset or IQ access')
    direct.add_argument('--input',type=Path,required=True);direct.add_argument('--output',required=True)
    direct.add_argument('--world-checkpoint',type=Path);direct.add_argument('--device',default='cpu')
    direct.add_argument('--episode-id',required=True);direct.add_argument('--frame-idx',type=int,required=True)
    direct.add_argument('--timestamp-s',type=float,required=True)
    direct.add_argument('--allow-untrained-validation',action='store_true')
    evaluation=sub.add_parser('evaluate');evaluation.add_argument('--predictions',required=True);evaluation.add_argument('--output',required=True)
    merging=sub.add_parser('merge');merging.add_argument('--output',required=True);merging.add_argument('--world-size',type=int,required=True)
    args=p.parse_args(argv);config=json.loads(args.config.read_text())
    require_single_frame(config)
    if config.get('architecture_schema',GROUP_SCHEMA)!=GROUP_SCHEMA:
        raise ValueError('Only the current single-frame localization architecture is supported')
    if config.get('detector_backend','dinov3_fpn')!='dinov3_fpn':
        raise ValueError('Only the trained DINOv3 detector is supported')
    if config.get('keypoint_checkpoint') is not None or config.get('visual_point_mode','bbox_center')!='bbox_center':
        raise ValueError('The visual input is the detection-box center; ROI keypoint heads are unsupported')
    if args.dataset_root:config['dataset_root']=args.dataset_root
    torch.set_num_threads(max(1,args.threads));np.random.seed(int(config['seed']));torch.manual_seed(int(config['seed']))
    if args.command=='train':
        from .group_training import train as train_group
        if args.dataset_root:raise ValueError('Bind a changed dataset root in a dedicated config before training')
        train_group(args);return 0
    if args.command=='infer':return infer(args,config)
    if args.command=='infer-spectra':return infer_spectra(args,config)
    if args.command=='merge':return merge(args,config)
    if args.command=='audit':
        from .data import audit as audit_data
        from .__main__ import output_path
        dataset=Dataset(config['dataset_root'],echo_subdir=config['echo_subdir'])
        report=audit_data(dataset,args.episodes or dataset.episodes[:3])
        write_json(output_path(args.output),report);print(json.dumps(report,ensure_ascii=False));return 0
    return evaluate(args,config)


if __name__=='__main__':main()
