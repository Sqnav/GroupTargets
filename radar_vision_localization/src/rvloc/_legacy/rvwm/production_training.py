"""Wait for the current certified dataset, then train the configured RVWM.

This controller never edits capture/echo sources or runs a second FFT producer.
Its durable status, owned child processes and logs survive a chat disconnection.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .phase_selection import configured_phases,phase_label,required_phase,selected_frame_indices

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parent


def read_json(path):
    return json.loads(Path(path).read_text())


def _configured_jobs():
    config=read_json(ROOT/'config.json')
    from .single_frame import require_single_frame
    require_single_frame(config)
    return [('single',ROOT/'config.json')]


def _plan_schema():
    return 'ra_re_single_frame_training_v1'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class Controller:
    def __init__(self,args):
        self.args=args;self.output=Path(args.output).resolve()
        self.output.mkdir(parents=True,exist_ok=True)
        self.path=self.output/'controller.json'
        self.state=dict(schema='rvwm_production_training_controller_v1',status='running',stage='initializing',
            pid=os.getpid(),started_unix=time.time(),training_started=False,training_completed=False,
            build_status=str(args.build_status),dataset_root=str(args.dataset_root),
            training_output=str(args.training_output),requested_steps=args.steps,world_size=6,
            uncertainty_calibrated=False,commands=[],failures=[])
        if self.path.exists():
            previous=read_json(self.path)
            if previous.get('status')=='complete':raise FileExistsError('This production controller already completed')
            self.state.update(started_unix=previous.get('started_unix',time.time()),
                commands=previous.get('commands',[]),failures=previous.get('failures',[]),
                previous_controller_pid=previous.get('pid'))
        self.update()

    def update(self,**values):
        from .training_pair import _write_json
        self.state.update(values,updated_unix=time.time())
        _write_json(self.path,self.state)

    def command(self,name,command,*,retries=0,env=None,log_path=None):
        """Stream owned children to stage logs; retry only explicitly safe stages."""
        attempts=0
        while True:
            attempts+=1
            log_path=Path(log_path) if log_path is not None else self.output/(name+'.log')
            log_path.parent.mkdir(parents=True,exist_ok=True)
            entry=dict(stage=name,command=[str(v) for v in command],attempt=attempts,
                started_unix=time.time(),log=str(log_path),
                log_offset_bytes=log_path.stat().st_size if log_path.exists() else 0)
            self.state['commands'].append(entry)
            self.update(stage=name)
            print(json.dumps(dict(event='stage_start',**entry)),flush=True)
            with log_path.open('a') as log:
                child=subprocess.Popen([str(v) for v in command],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                self.update(child_pid=child.pid)
                try:
                    while child.poll() is None:
                        self.update(child_pid=child.pid,child_elapsed_s=time.time()-entry['started_unix'])
                        time.sleep(min(self.args.poll_seconds,10.))
                except BaseException:
                    if child.poll() is None:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=20)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            entry.update(returncode=child.returncode,finished_unix=time.time(),
                         log_end_bytes=log_path.stat().st_size)
            self.update(child_pid=None)
            if child.returncode==0:return
            self.state['failures'].append(dict(entry))
            self.update()
            if attempts>retries:
                raise RuntimeError(f'{name} exited {child.returncode}; see {log_path}')
            time.sleep(min(30.,self.args.poll_seconds))

    def wait_for_build(self):
        if self.args.build_status is None:
            self.update(stage='binding_published_dataset')
            return
        from .production_readiness import inspect_build
        while True:
            check=inspect_build(self.args.build_status,self.args.expected_episodes)
            self.update(stage='waiting_for_dataset_and_fft',build=check)
            if check['ready']:return check
            if check['terminal_failure']:
                raise RuntimeError('Source dataset/FFT did not pass certification: '+check['reason'])
            print(json.dumps(dict(event='waiting_for_dataset',**check)),flush=True)
            time.sleep(self.args.poll_seconds)

    def bind_data(self):
        from .training_pair import _write_json
        self.update(stage='binding_current_dataset')
        active_config=read_json(ROOT/'config.json')
        if active_config.get('detector_backend')!='dinov3_fpn':
            raise ValueError('Production localization requires the frozen DINOv3 detector')
        receipt=self._bind_dino_manifest(active_config)
        for config_path in (ROOT/'config.json',):
            config=read_json(config_path)
            config.update(dataset_root=str(self.args.dataset_root.resolve()),echo_subdir='echoes',
                visual_split_manifest=receipt['path'],split_manifest=receipt['path'],
                split_manifest_sha256=receipt['sha256'],track_detection_dir=str(self.output/'detections'),
                training_phase=phase_label(receipt.get('selected_phases',['track'])),
                inference_phase=phase_label(receipt.get('selected_phases',['track'])))
            if receipt.get('selected_phases',['track'])!=['track']:
                config.update(training_phases=receipt['selected_phases'],inference_phases=receipt['selected_phases'])
            else:
                config.pop('training_phases',None);config.pop('inference_phases',None)
            if receipt.get('training_sampling_policy'):
                config.update(training_sampling_policy=receipt['training_sampling_policy'],
                    dataset_split_policy=receipt['dataset_split_policy'])
            else:
                config.pop('training_sampling_policy',None);config.pop('dataset_split_policy',None)
            _write_json(config_path,config)
        self.update(dataset_binding=receipt)

    def _bind_dino_manifest(self,config):
        """Reuse the detector's exact frozen partition, never generate a new one.

        A checkpoint trained on the published approach batch already defines
        the detector/fusion family split. Binding validates the manifest and mixed
        capture provenance without copying or rewriting that manifest.
        """
        from .data import Dataset,_mixed_echo
        from .detector_contract import detector_contract
        from .partitions import POLICY,ROLES,active_splits
        from .phase_selection import inventory_phases

        root=self.args.dataset_root.expanduser().resolve()
        contract=detector_contract(dict(config,dataset_root=str(root)))
        path=Path(contract['detector_split_manifest']).resolve()
        manifest=read_json(path)
        expected_sha=contract['detector_split_manifest_sha256']
        if digest(path)!=expected_sha:
            raise ValueError('Frozen DINO detector split changed during binding')
        if Path(manifest.get('dataset_root','')).resolve()!=root:
            raise ValueError('Frozen DINO detector split points to another dataset')
        bound=dict(config,dataset_root=str(root),echo_subdir='echoes',
            visual_split_manifest=str(path),split_manifest=str(path),
            split_manifest_sha256=expected_sha,split_policy=POLICY)
        dataset=Dataset(root,verify_sources=True,echo_subdir='echoes')
        membership=active_splits(dataset,bound)
        if len(dataset.episodes)!=self.args.expected_episodes:
            raise ValueError('Frozen DINO detector split has the wrong expected episode count')
        # Dataset._load verifies metadata and trajectory/echo-manifest hashes;
        # enforce mixed observations too, rather than accepting group-only IQ.
        for episode in dataset.episodes:
            if not _mixed_echo(dataset._load(episode)[3]):
                raise ValueError('DINO fusion binding requires mixed group-plus-observer IQ: '+episode)
            dataset._loaded.pop(episode,None)
        counts={role:len(membership[role]) for role in ROLES}
        frame_counts={role:manifest['splits'][role]['frame_count'] for role in ROLES}
        if (any(type(value) is not int or value<1 for value in frame_counts.values())
                or sum(frame_counts.values())!=manifest.get('frame_count')):
            raise ValueError('Frozen DINO detector split frame counts disagree')
        if digest(path)!=expected_sha:
            raise ValueError('Frozen DINO detector split changed during source audit')
        receipt=dict(path=str(path),sha256=expected_sha,counts=counts,frame_counts=frame_counts,
            parent_manifest_sha256=manifest.get('parent_manifest',{}).get('sha256'),
            recaptured_episodes=manifest.get('recaptured_episodes',[]),
            source_changed_episodes=manifest.get('source_changed_episodes',[]),
            membership_preserved=True,mixed_echo_audit_completed=True,
            selected_phases=list(inventory_phases(manifest)),
            binding_mode='reuse_frozen_detector_manifest',detector_backend=contract['detector_backend'],
            detector_weights_sha256=contract['weights_sha256'],manifest_rewritten=False)
        request=manifest.get('request',{})
        if request.get('training_sampling_policy'):
            receipt.update(training_sampling_policy=request['training_sampling_policy'],
                dataset_split_policy=request['split_policy'],family_partition=manifest['family_partition'])
        return receipt

    def audit_interfaces(self):
        from .data import Dataset
        from .partitions import active_splits
        from .group_observation import describe_observation,validate_features
        from .visual_refinement import ExportedRGBFrontend
        from .group_model import GroupSceneModel
        from .group_localization import GroupSceneLocalizer
        from .types import ObservationWindow
        from .training_pair import _write_json
        import numpy as np
        import torch
        torch.set_num_threads(1)
        config=read_json(ROOT/'config.json')
        dataset=Dataset(config['dataset_root'],echo_subdir='echoes')
        splits=active_splits(dataset,config)
        audited=[]
        for role in ('train','val','test'):
            frontend=ExportedRGBFrontend(config,split=role,device='cpu')
            episode=splits[role][0]
            indices=selected_frame_indices(dataset,episode,configured_phases(config))
            observations=[dataset.observations(episode,i,required_phase=required_phase(configured_phases(config))) for i in indices[:1]]
            if not observations:raise ValueError('No selected-phase observations in '+episode)
            detections=frontend.detect(observations[-1])
            features=describe_observation(observations[-1],detections,config)
            validate_features(features)
            for variant,path in _configured_jobs():
                cfg=read_json(path);torch.manual_seed(cfg['seed'])
                model=GroupSceneModel(cfg)
                localizer=GroupSceneLocalizer(cfg,frontend,model,allow_untrained_validation=True)
                result=localizer.estimate(ObservationWindow(tuple(observations)))
                if result['localized_count']!=0:raise AssertionError('Untrained positions published')
                audited.append(dict(split=role,episode_id=episode,variant=variant,frame_idx=observations[-1].frame_idx,
                    iq_shape=list(observations[-1].iq.shape),iq_dtype=str(observations[-1].iq.dtype),
                    ra_shape=list(features['ra_power'].shape),re_shape=list(features['re_power'].shape),
                    detected_count=result['detected_count'],input_frames=len(observations)))
            frontend.close()
        _write_json(self.output/'interface_audit.json',dict(status='passed',episodes=len(dataset.episodes),
            splits={k:len(v) for k,v in splits.items()},samples=audited,
            normal_inputs_do_not_read_labels=True,scope='sensor/adapter engineering; not localization accuracy'))
        self.update(interface_audit=str(self.output/'interface_audit.json'))

    def train_pair(self):
        """Launch one configured job; retain the historical method name."""
        output=self.args.training_output.resolve()
        queue=output/'queue.json'
        if queue.exists():
            status=read_json(queue)
            if status.get('status')=='complete':
                jobs=status.get('jobs',[])
                if (status.get('schema')!=_plan_schema()
                        or status.get('order')!=[name for name,_ in _configured_jobs()]
                        or status.get('total_jobs')!=1 or status.get('completed_jobs')!=1
                        or len(jobs)!=1 or any(job.get('variant')!=name
                            or job.get('status')!='complete'
                            or Path(job.get('output','')).resolve()!=output/name
                            for name,job in zip((name for name,_ in _configured_jobs()),jobs))):
                    raise ValueError('Existing queue does not identify the current single-frame training job; historical runs remain protected')
                # A completed queue alone says nothing about which dataset or
                # source version produced its weights. Verify the complete
                # lineage before treating a previous run as this task's work.
                self.verify_training()
                self.update(training_started=True,training_reused_verified=True)
                return
            if status.get('status')=='failed' and any(c.get('stage')=='training_pair' for c in self.state['commands']):
                self.recover_training()
                return
            raise RuntimeError('Incomplete existing training output requires an inspected recovery: '+str(output))
        self.update(training_started=True)
        try:
            self.command('training_pair',['bash',ROOT/'run_training.sh',output,
                '--variant','single','--steps',str(self.args.steps),'--validate-every',str(self.args.validate_every)])
        except RuntimeError:
            # A completed variant is retained. Only named transient execution
            # failures permit automatic recovery; data/provenance/numeric
            # errors are never disguised by reinitializing or changing data.
            self.recover_training()

    def recover_training(self):
        from .training_pair import _write_json
        output=self.args.training_output.resolve()
        plan=read_json(output/'pair_plan.json');queue=read_json(output/'queue.json')
        if (plan.get('schema')!=_plan_schema()
                or Path(plan.get('output','')).resolve()!=output or plan.get('order')!=[name for name,_ in _configured_jobs()]):
            raise ValueError('Cannot recover a historical or unrelated training plan as a single-frame job')
        known=('out of memory','nccl','connection reset by peer','connection closed by peer',
               'socket timeout','timed out','cuda error: unknown error','resource temporarily unavailable')
        forbidden=('valueerror:','assertionerror:','keyerror:','filenotfounderror:',
                   'floatingpointerror:','non-finite','provenance','source version','source sha','mismatch')
        for index,job in enumerate(plan['jobs']):
            item=queue['jobs'][index]
            if item['status']=='complete':continue
            directory=Path(job['output']).resolve()
            if directory!=output/job['variant']:
                raise ValueError('Recovery output differs from the owned plan')
            if job['variant'] not in [name for name,_ in _configured_jobs()]:raise ValueError('Removed training variant in recovery plan')
            live=read_json(ROOT/'config.json')
            for key in ('dataset_root','echo_subdir','split_manifest','split_manifest_sha256','window'):
                if job['configuration'].get(key)!=live.get(key):
                    raise ValueError('Recovery plan does not match the current data/variant configuration: '+key)
            original_log=Path(job['log'])
            for attempt in range(1,4):
                has_state=any((directory/name).exists() for name in ('progress.json','run_manifest.json','metrics.jsonl'))
                if has_state:
                    latest=next((c for c in reversed(self.state['commands']) if c.get('log')==str(original_log)
                        and c.get('returncode') not in (None,0)),{})
                    offset=latest.get('log_offset_bytes',0)
                    with original_log.open('rb') as stream:
                        stream.seek(offset);text=stream.read().decode(errors='replace')[-100000:].lower()
                    if any(marker in text for marker in forbidden) or not any(marker in text for marker in known):
                        raise RuntimeError('Training failure requires a code/data diagnosis; automatic restart refused: '+str(original_log))
                command=list(job['command'])
                resume=directory/'world.latest.pt'
                recovery='new_pending_variant'
                if has_state and resume.exists():
                    command+=['--resume',str(resume)]
                    recovery='restore_optimizer_sampler_and_rng'
                elif has_state:
                    archive=self.output/'failed_attempts'/f"{job['variant']}_{time.time_ns()}"
                    archive.parent.mkdir(parents=True,exist_ok=True)
                    directory.rename(archive)
                    directory.mkdir(parents=True)
                    _write_json(directory/'configuration.json',job['configuration'])
                    recovery='restart_from_seed_no_complete_snapshot; partial_updates_discarded_and_archived'
                    self.state['failures'].append(dict(variant=job['variant'],archive=str(archive),recovery=recovery))
                queue.update(status='running',active_variant=job['variant'],active_run_dir=str(directory),active_job=index+1)
                item.update(status='running',recovery_attempt=attempt,recovery=recovery)
                _write_json(output/'queue.json',queue)
                self.update(stage='recovering_training',recovery_variant=job['variant'],recovery_attempt=attempt,recovery=recovery)
                try:
                    self.command('training_recovery_'+job['variant'],command,log_path=original_log)
                except RuntimeError:
                    item.update(status='failed');queue.update(status='failed')
                    _write_json(output/'queue.json',queue)
                    if attempt==3:raise
                    time.sleep(min(30.,self.args.poll_seconds))
                else:
                    item.update(status='complete',returncode=0)
                    queue.update(completed_jobs=sum(j['status']=='complete' for j in queue['jobs']))
                    _write_json(output/'queue.json',queue)
                    break
        queue.update(status='complete',returncode=0,completed_jobs=1,updated_unix=time.time())
        _write_json(output/'queue.json',queue)

    def preflight(self,variant,path,environment):
        """Reuse only a certified preflight with current data and source hashes."""
        from .group_training import _source_hashes
        destination=self.output/'preflight'/variant
        receipt=destination/'run_manifest.json'
        current=read_json(path)
        if receipt.exists():
            old=read_json(receipt)
            configuration=old.get('configuration',{})
            checks=old.get('gradient_checks',[])
            dynamic={'device','world_model_trained','completed_optimizer_steps','formation_head_trained','active_partition'}
            normalized=lambda value:{k:v for k,v in value.items() if k not in dynamic}
            same=normalized(configuration)==normalized(current)
            if (old.get('status')=='dry_run_complete' and old.get('optimizer_steps')==0
                    and old.get('world_size')==6 and old.get('source_sha256')==_source_hashes() and same
                    and sorted(c.get('rank',-1) for c in checks)==list(range(6))
                    and all(c.get('parameters_unchanged') and c.get('finite_gradients')
                        and c.get('optimizer_steps')==0 for c in checks)):
                return
            archive=self.output/'failed_attempts'/f'preflight_{variant}_{time.time_ns()}'
            archive.parent.mkdir(parents=True,exist_ok=True);destination.rename(archive)
        elif destination.exists() and any(destination.iterdir()):
            archive=self.output/'failed_attempts'/f'preflight_{variant}_{time.time_ns()}'
            archive.parent.mkdir(parents=True,exist_ok=True);destination.rename(archive)
        self.command('preflight_'+variant,[sys.executable,'-m','torch.distributed.run','--standalone',
            '--nnodes=1','--nproc_per_node=6','-m','rvwm.group_training','--config',path,
            '--output',destination,'--device','cuda:0','--dry-run'],env=environment)

    def verify_training(self):
        from .group_model import load_group_model,GROUP_SCHEMA
        from .group_training import _source_hashes
        from .partitions import canonical_manifest
        from .detector_contract import detector_contract,verify_trained_detector_contract

        def check_configuration(recorded,current,split_sha,where):
            if not isinstance(recorded,dict):raise ValueError('Missing recorded configuration: '+where)
            path_fields=('dataset_root','split_manifest','visual_split_manifest','track_detection_dir')
            for key in path_fields:
                if (not recorded.get(key) or not current.get(key)
                        or Path(recorded[key]).resolve()!=Path(current[key]).resolve()):
                    raise ValueError('Training dataset/configuration mismatch '+key+': '+where)
            for key,default in (('echo_subdir',None),('training_phase',None),('window',1)):
                if recorded.get(key,default)!=current.get(key,default):
                    raise ValueError('Training dataset/configuration mismatch '+key+': '+where)
            if configured_phases(recorded)!=configured_phases(current):
                raise ValueError('Training selected phases differ: '+where)
            if recorded.get('split_manifest_sha256')!=split_sha:
                raise ValueError('Training split content differs from the current capture binding: '+where)
            if current.get('detector_backend')=='dinov3_fpn':
                if recorded.get('detector_backend')!='dinov3_fpn':
                    raise ValueError('Training detector backend differs: '+where)
                if detector_contract(recorded)!=detector_contract(current):
                    raise ValueError('Training detector contract differs: '+where)

        def check_frontend(frontend,current,where):
            if current.get('detector_backend')=='dinov3_fpn':
                if not isinstance(frontend,dict):
                    raise ValueError('Missing trained DINO frontend metadata: '+where)
                verify_trained_detector_contract(current,frontend)
                if frontend.get('weights_sha256')!=detector_contract(current)['weights_sha256']:
                    raise ValueError('Training frontend weights SHA256 differs: '+where)

        def check_membership(recorded,expected,key,where):
            values=recorded.get(key)
            if (not isinstance(values,list) or len(values)!=len(set(values))
                    or sorted(values)!=sorted(expected)):
                raise ValueError('Training episode membership mismatch '+key+': '+where)

        source=_source_hashes()
        result={}
        for variant,path in _configured_jobs():
            directory=self.args.training_output/variant
            current=read_json(path)
            _,split,split_sha=canonical_manifest(current)
            binding=self.state.get('dataset_binding',{})
            if binding.get('sha256')!=split_sha or Path(binding.get('path','')).resolve()!=Path(current['split_manifest']).resolve():
                raise ValueError('Current split does not match this controller dataset binding')
            if Path(current['dataset_root']).resolve()!=self.args.dataset_root.resolve():
                raise ValueError('Current training configuration points outside the requested dataset')
            expected_train=split['splits']['train']['episode_ids']
            expected_val=split['splits']['val']['episode_ids']
            manifest=read_json(directory/'run_manifest.json')
            if (manifest.get('schema')!=current.get('architecture_schema',GROUP_SCHEMA) or manifest.get('status')!='complete'
                    or manifest.get('steps')!=self.args.steps or manifest.get('optimizer_steps')!=self.args.steps
                    or manifest.get('rank_parameter_max_difference')!=0. or manifest.get('world_size')!=6
                    or manifest.get('training_phase')!=phase_label(configured_phases(current)) or manifest.get('model_trained') is not True):
                raise ValueError('Training completion is not verified: '+variant)
            check_configuration(manifest.get('configuration'),current,split_sha,variant+' manifest')
            check_frontend(manifest.get('frontend',{}),current,variant+' manifest')
            check_membership(manifest,expected_train,'train_episodes',variant+' manifest')
            check_membership(manifest,expected_val,'validation_episodes',variant+' manifest')
            if manifest.get('source_sha256')!=source:
                raise ValueError('Training source version differs from the current source: '+variant)
            partition=manifest.get('partition',{})
            if partition.get('sha256')!=split_sha:
                raise ValueError('Training partition receipt differs from current binding: '+variant)
            best_step=manifest.get('best_step')
            if isinstance(best_step,bool) or not isinstance(best_step,int) or not 1<=best_step<=self.args.steps:
                raise ValueError('Invalid selected checkpoint step: '+variant)
            checked={}
            for filename,step,hash_key in [('world.best.pt',best_step,'selected_checkpoint_sha256'),
                                         ('world.pt',self.args.steps,'final_checkpoint_sha256')]:
                checkpoint=directory/filename
                checkpoint_sha=digest(checkpoint)
                if manifest.get(hash_key)!=checkpoint_sha:
                    raise ValueError('Checkpoint bytes do not match completed run receipt: '+str(checkpoint))
                model=load_group_model(checkpoint,current,'cpu')
                metadata=getattr(model,'checkpoint_metadata',{})
                if (metadata.get('kind')!='group_scene_world_model' or metadata.get('group_schema')!=current.get('architecture_schema',GROUP_SCHEMA)
                        or metadata.get('steps')!=step or metadata.get('optimizer_steps')!=step
                        or metadata.get('model_trained') is not True or not model.trained
                        or metadata.get('world_size')!=6 or metadata.get('training_phase')!=phase_label(configured_phases(current))):
                    raise ValueError('Checkpoint training metadata is not verified: '+str(checkpoint))
                check_configuration(metadata.get('model_config'),current,split_sha,str(checkpoint))
                check_frontend(metadata.get('frontend',{}),current,str(checkpoint))
                check_membership(metadata,expected_train,'train_episodes',str(checkpoint))
                check_membership(metadata,expected_val,'validation_episodes',str(checkpoint))
                if metadata.get('feature_source_sha256')!=source:
                    raise ValueError('Checkpoint source version differs from completed training: '+str(checkpoint))
                checked[filename]=dict(path=str(checkpoint),sha256=checkpoint_sha,step=step)
            result[variant]=dict(status='complete',optimizer_steps=manifest['optimizer_steps'],
                best_step=best_step,checkpoint=checked['world.best.pt']['path'],
                checkpoint_sha256=checked['world.best.pt']['sha256'],
                final_checkpoint=checked['world.pt'],partition_sha256=split_sha,
                final_validation=read_json(directory/'prior_validation_final.json')['summary'],
                trained=True,lineage_verified=True,uncertainty_calibrated=False)
        self.update(training_completed=True,results=result)
        return result

    def run(self):
        self.wait_for_build()
        self.bind_data()
        environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
            PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT)+os.pathsep+str(WORKSPACE),
            CUDA_VISIBLE_DEVICES='0,1,2,3,4,5',CUDA_DEVICE_ORDER='PCI_BUS_ID',
            TORCH_NCCL_ASYNC_ERROR_HANDLING='1',NCCL_DEBUG='WARN')
        # Export is resumable per complete, hash-verified episode. A retry never
        # rewrites source images or silently accepts another detector contract.
        self.command('rgb_detection_export',[self.args.detector_python,'-m','rvwm.export_track_detections',
            '--config',ROOT/'config.json','--output',self.output/'detections','--devices','0,1,2,3,4,5'],
            retries=1,env=environment)
        self.update(stage='auditing_interfaces')
        self.audit_interfaces()
        for variant,path in _configured_jobs():
            self.preflight(variant,path,environment)
        os.environ.update(environment)
        self.train_pair()
        self.verify_training()
        self.update(status='complete',stage='complete',finished_unix=time.time())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-status',type=Path,help='Optional build completion receipt; otherwise validate the published source directly')
    parser.add_argument('--dataset-root',type=Path,default=WORKSPACE/'dataset_construction')
    parser.add_argument('--output',type=Path,default=ROOT/'results/production')
    parser.add_argument('--training-output',type=Path,default=ROOT/'results/training')
    parser.add_argument('--detector-python',type=Path,default=WORKSPACE/'uav_detection/.venv_deimv2/bin/python')
    parser.add_argument('--expected-episodes',type=int,default=1000)
    parser.add_argument('--poll-seconds',type=float,default=30.)
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--validate-every',type=int,default=1000)
    parser.add_argument('--status',action='store_true',help='Read durable controller progress without starting any task')
    args=parser.parse_args(argv)
    if args.status:
        print(json.dumps(read_json(args.output/'controller.json'),ensure_ascii=False,indent=2))
        return 0
    if min(args.poll_seconds,args.expected_episodes,args.steps,args.validate_every)<=0:parser.error('Positive budgets required')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        controller=Controller(args)
        try:controller.run()
        except BaseException as error:
            controller.update(status='failed',error=f'{type(error).__name__}: {error}',finished_unix=time.time())
            raise


def termination_requested(signum,frame):
    raise KeyboardInterrupt('Production controller termination requested')


if __name__=='__main__':
    signal.signal(signal.SIGTERM,termination_requested)
    main()
