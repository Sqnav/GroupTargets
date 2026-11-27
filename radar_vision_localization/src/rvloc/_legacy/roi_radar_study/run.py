"""Register and execute the visual / 32 MHz / 64 MHz ROI comparison."""
import argparse
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from precision_study.training import write_json
from precision_study.deployed_infer import DEFAULT_DESCRIPTOR,load_descriptor
from precision_study.deployed_train import source_identity
from rvwm.data import sha256

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/roi_radar_20261125'
ARMS=('visual','32mhz','64mhz')
GPUS=dict(zip(ARMS,('0,1','2,3','4,5')))
BW=ROOT/'results/bandwidth_double_20261124'
ROI=ROOT/'results/d_evidence_20261124'
REFERENCE=ROOT/'results/roi_followup_20261125_run2'


def read(path):return json.loads(Path(path).read_text())


def authorized_gpu_groups():
    # All six GPUs were explicitly authorized. Existing applications remain;
    # only adequate free memory is required for a new disjoint study group.
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                 '--format=csv,noheader,nounits'],text=True)
    free={row.split(',')[0].strip():int(row.split(',')[1]) for row in raw.splitlines()}
    return {group for group in GPUS.values() if all(free.get(gpu,0)>=16384 for gpu in group.split(','))}


def controller():
    name='roi_radar_study._preserved_controller'
    if name not in sys.modules:
        path=ROOT/'roi_followup/run.py'
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module)
        module.ARMS=ARMS;module.GPUS=GPUS;module.DEFAULT=DEFAULT
        module.available_gpus=authorized_gpu_groups
    return sys.modules[name]


def prepare(output):
    from .checkpoint import source_hashes
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Existing experiments are protected: '+str(output))
    description,deployment=load_descriptor(DEFAULT_DESCRIPTOR);source_identity(description)
    for parent in (BW,ROI,REFERENCE):
        if read(parent/'run_state.json')['status']!='complete':
            raise ValueError('Complete registered predecessor first: '+str(parent))
    output.mkdir(parents=True)
    roster=read(ROI/'split_roster.json');write_json(output/'split_roster.json',roster)
    protocol=dict(schema='roi_visual_radar_bandwidth_study_v1',
        registered_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        objective='Measure radar gain after frozen ROI and the extra effect of bandwidth plus frequency sampling',
        arms=dict(visual='Frozen ROI, observed bbox/rays, measured navigation, member interaction and geometry-only two-round depth updates; no IQ/spectrum reading',
                  **{'32mhz':'Original ROI architecture on original 32 MHz / 128-frequency observations',
                     '64mhz':'Same ROI architecture on existing 64 MHz / 256-frequency observations'}),
        common=dict(input_frames=1,detector_frozen=True,ROI_payload_unchanged=True,
            visual_ray_and_navigation_fair=True,shared_nonradar_blocks_unchanged=True,
            candidate_offsets_coarse_m=[0,-2,2,-4,4,-8,8,-16,16],
            candidate_offsets_fine_m=[0,-.25,.25,-.5,.5,-1,1,-2,2],
            correction_limits_m=[16,16],initial_Huber_weight=.3,depth_MSE_weight=.05,
            fusion_initialization='fresh; no old D or ROI weights',sampling='complete no-replacement frame epochs with equal-episode loss weights',
            capacity_notice='Visual removes radar encoders and replaces spectrum candidate encoding with common 8D geometry; retained nonradar blocks unchanged; parameter counts reported'),
        budget=dict(epochs=80,world_size_per_arm=2,global_batch_size=96,per_rank_batch_size=48,
            seed=20261118,learning_rate=3e-4,warmup_epochs=2,minimum_lr_ratio=.1,
            validation_every_epochs=5,save_every_epochs=5,validation_batch_size=32,
            detection_dropout=.2,threads=1),
        data=dict(train_episodes=700,train_frames=11649,validation_episodes=150,validation_frames=2569,
            roster_sha256=sha256(output/'split_roster.json'),
            new_images_or_IQ_generated=False,final_test_observations_or_labels_read=False,
            independent_new_trajectories_available_in_canonical_pool=False,
            exposure='Historical development validation used for checkpoint selection and past architecture decisions; historical test also previously used, never relabelled independent'),
        physics=dict(carrier_hz=4.9e9,pulses=40,array=[16,16],frequency_interval_hz=250000,
            bandwidths_hz=[32e6,64e6],frequency_counts=[128,256],
            range_limit_m=599.584916,range_resolution_m=[4.68425715625,2.342128578125],
            interpretation='64 MHz changes bandwidth, frequency sample count, native FFT grid and processing gain together; not resolution alone'),
        evaluation=dict(primary_comparisons=['32mhz minus visual','64mhz minus 32mhz'],
            auxiliary='64mhz minus visual; never substituted for a primary hypothesis',
            bootstrap=dict(draws=2000,unit='whole paired episode',seed=20261125,levels=[.95,.975]),
            metrics=['matched 3D RMSE','P95','all-GT1/3/5m success','miss/invalid counts','stage errors',
                     '1m transitions','per-episode differences','timing/memory','group-bias versus relative-error decomposition'],
            development_gate='primary 97.5% paired RMSE difference upper<0; P95 no increase; all-GT1m no decrease; no extra failures',
            decomposition='Offline GT only: per-frame mean residual squared plus centered residual MSE; pooled by common matched member count, all missing GT remain in normal success metrics'),
        stop='Exactly three registered 80-epoch runs and evaluation. No model expansion, extra seeds, post-result thresholds or automatic deployment.',
        default_D=deployment,production_default_changed=False,resources=dict(GPUs=GPUS,
            authorization='User explicitly permitted all six GPUs',
            policy='three disjoint two-rank groups; at least16GiB free per GPU; existing processes never stopped'))
    write_json(output/'PROTOCOL.json',protocol)
    configs=output/'configs';configs.mkdir()
    base=read(REFERENCE/'configs/control.json')
    for key in list(base):
        if key.startswith('roi_followup_') or key in ('precision_schedule_total_steps','depth_warmup_steps'):
            base.pop(key)
    for arm in ARMS:
        cfg=dict(base,study_arm=arm,study_protocol_path=str(output/'PROTOCOL.json'),
            study_protocol_sha256=sha256(output/'PROTOCOL.json'),
            study_roi_export_receipt_sha256=sha256(ROI/'roi/export_receipt.json'),
            precision_mode='visual' if arm=='visual' else 'radar',
            precision_radar_mode='none' if arm=='visual' else 'ra_re',
            model_input='current frozen ROI, bbox/ray and public navigation'+(' only' if arm=='visual' else ', measured RA/RE'),
            study_observation_scope='no IQ or spectrum payloads' if arm=='visual' else 'existing calibrated mixed group+observer radar observations')
        if arm!='visual':
            variant='baseline' if arm=='32mhz' else 'bandwidth2'
            path=BW/'configs'/(variant+'.json');sensor=read(path)
            cfg.update({key:value for key,value in sensor.items() if key.startswith('bandwidth_')})
            cfg.update(study_sensor_config_path=str(path),study_sensor_preparation=str(BW/'prepare'/variant))
        write_json(configs/(arm+'.json'),cfg)
    commands={arm:[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
        '--nproc_per_node=2','-m','roi_radar_study.driver','--config',str(configs/(arm+'.json')),
        '--descriptor',str(DEFAULT_DESCRIPTOR),'--output',str(output/'training'/arm),
        '--epochs','80','--warmup-epochs','2','--batch-size','48','--validate-every-epochs','5',
        '--save-every-epochs','5','--validation-batch-size','32','--learning-rate','0.0003',
        '--minimum-lr-ratio','0.1','--seed','20261118','--detection-dropout','0.2','--threads','1',
        '--device','cuda:0'] for arm in ARMS}
    plan=dict(schema='roi_radar_execution_v1',output=str(output),
        protocol_sha256=sha256(output/'PROTOCOL.json'),roster_sha256=sha256(output/'split_roster.json'),
        source_sha256=source_hashes(),config_sha256={arm:sha256(configs/(arm+'.json')) for arm in ARMS},
        train_commands=commands,GPU_ids=GPUS,
        evaluation_command=[sys.executable,'-m','roi_radar_study.evaluate','--experiment',str(output),
                            '--phase','finalize','--device','cuda:0'])
    write_json(output/'run_plan.json',plan)
    write_json(output/'input_sources.json',dict(original_ROI_export=str(ROI/'roi/export_receipt.json'),
        ROI_export_sha256=sha256(ROI/'roi/export_receipt.json'),
        original32_configuration=str(BW/'configs/baseline.json'),
        existing64_configuration=str(BW/'configs/bandwidth2.json'),
        existing64_generation_receipt=str(BW/'generation_receipt.json'),
        existing64_generation_receipt_sha256=sha256(BW/'generation_receipt.json'),
        dataset_root=base['dataset_root'],RGB_and_detection_cache_unchanged=True,
        visual_loader_does_not_read_radar_payload_or_parent_radar_cache=True))
    return plan


def execute(output):
    implementation=controller();implementation.execute(output)
    path=Path(output)/'run_state.json';state=read(path)
    if state['status']=='complete':
        for arm in ARMS:
            progress=read(Path(output)/'training'/arm/'progress.json')
            if progress['stage']!='complete' or progress['completed_epochs']!=80:
                raise ValueError('Cannot mark an incomplete training complete')
            state['arms'][arm].update(status='complete',returncode=0)
        state['training_status_display_reconciled_from_completed_manifests']=True
        write_json(path,state)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--phase',choices=('prepare','execute','verify'),required=True)
    a=p.parse_args(argv)
    if a.phase=='prepare':
        plan=prepare(a.output);print(json.dumps(dict(status='registered',output=plan['output'])))
    elif a.phase=='execute':execute(a.output)
    else:controller().verify(read(a.output/'run_plan.json'));print('Registered sources and protocol verified')


if __name__=='__main__':main()
