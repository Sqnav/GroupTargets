"""Audited controller handoff: preserve aligned training, accelerate only ROI."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from d_evidence_study import run as legacy
from d_evidence_study.monitor import snapshot,format_progress

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/d_evidence_speed_20261125'
ORIGINAL=ROOT/'results/d_evidence_20261124'
EXPECTED_PID=1158190
EXPECTED_START='700880591'
read=legacy.read_json
write=legacy.write_json
sha=legacy.sha256
process_start=legacy.process_start


def source_identity():
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def verify_equivalence(path):
    value=read(path)
    if (value.get('status')!='passed' or value.get('model_state_dict_compatible') is not True
            or value.get('same_training_contract') is not True or value.get('measured_speedup',0)<=1.2):
        raise ValueError('Actual numerical equivalence and GPU speedup must pass before migration')
    if value.get('source_sha256')!=source_identity():raise ValueError('Executor changed after GPU equivalence check')
    cpu=value.get('cpu_tests_receipt')
    if not cpu or sha(cpu['path'])!=cpu['sha256']:raise ValueError('CPU regression receipt changed')
    checked=read(cpu['path'])
    if checked.get('status')!='passed' or checked.get('source_sha256')!=value['source_sha256']:
        raise ValueError('CPU and GPU checks were not performed on the same executor')
    return value


def make_plan(output):
    output=Path(output).resolve();protocol=read(output/'PROTOCOL.json');original=Path(protocol['original_experiment'])
    if sha(original/'run_plan.json')!=protocol['original_plan_sha256']:raise ValueError('Original plan changed')
    old=read(original/'run_plan.json');legacy.check_plan(old)
    equivalence=output/'equivalence_receipt.json';verify_equivalence(equivalence)
    eq=read(equivalence)
    if eq.get('original_run_plan_sha256')!=protocol['original_plan_sha256']:
        raise ValueError('Equivalence measured another original experiment')
    if eq.get('original_configuration_sha256')!=old['configuration_sha256']['roi']:
        raise ValueError('Equivalence measured another ROI configuration')
    config=original/'configs/roi.json'
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=3',
        '-m','d_evidence_speed.train','--arm','roi','--config',str(config),
        '--output',str(original/'training/roi'),'--device','cuda','--equivalence-receipt',str(equivalence)]
    for key in ('steps','batch_size','seed','learning_rate','detection_dropout','validate_every','threads'):
        command += ['--'+key.replace('_','-'),str(old['settings'][key])]
    env=dict(PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_THREADING_LAYER='GNU',PYTHONUNBUFFERED='1')
    return dict(schema='D_evidence_execution_handoff_v1',output=str(output),original=str(original),
        original_plan_sha256=sha(original/'run_plan.json'),protocol_sha256=sha(output/'PROTOCOL.json'),
        equivalence=dict(path=str(equivalence),sha256=sha(equivalence)),source_sha256=source_identity(),
        expected_original_controller=dict(pid=EXPECTED_PID,process_start=EXPECTED_START),
        fast_training=dict(command=command,environment=dict(env,CUDA_VISIBLE_DEVICES='1,4,5')),
        evaluation=dict(command=[sys.executable,'-m','d_evidence_study.evaluate','--experiment',str(original),
            '--phase','finalize','--device','cuda:0'],environment=dict(env,CUDA_VISIBLE_DEVICES='1')),
        GPU_ids=['1','4','5'],original_evaluator_retained=True,
        completed_sequence_preserved=True,ongoing_aligned_optimizer_not_interrupted=True,
        old_sources_modified=False,production_default_changed=False)


def check_plan(plan):
    output=Path(plan['output']);original=Path(plan['original'])
    if source_identity()!=plan['source_sha256']:raise ValueError('Locked speed executor source changed')
    if sha(output/'PROTOCOL.json')!=plan['protocol_sha256']:raise ValueError('Optimization protocol changed')
    if sha(plan['equivalence']['path'])!=plan['equivalence']['sha256']:raise ValueError('Equivalence receipt changed')
    verify_equivalence(plan['equivalence']['path'])
    if sha(original/'run_plan.json')!=plan['original_plan_sha256']:raise ValueError('Original plan changed')
    legacy.check_plan(read(original/'run_plan.json'))


def owned_controller(pid,start,original):
    if process_start(pid)!=start:raise ValueError('Original controller PID/start changed')
    args=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
    if ('d_evidence_study.run' not in args or '--_stage-worker' in args or '--_wait' in args
            or '--output' not in args):raise ValueError('Only the original top controller may receive a handoff signal')
    path=Path(args[args.index('--output')+1])
    if not path.is_absolute():path=Path(f'/proc/{pid}/cwd').resolve()/path
    if path.resolve()!=Path(original).resolve():raise ValueError('Controller belongs to another output')


def request_handoff(plan):
    """SIGINT one authenticated top controller, never its training process tree."""
    check_plan(plan);original=Path(plan['original']);state=read(original/'run_state.json')
    if state.get('status')!='running' or state.get('current_stage')!='train_aligned':
        raise ValueError('Migrate only while aligned remains in its original supervised stage')
    if (original/'stages/train_roi.json').exists() or (original/'training/roi/run_manifest.json').exists():
        raise ValueError('ROI has already started; do not discard its optimizer state')
    owner=plan['expected_original_controller']
    if state['controller_pid']!=owner['pid'] or state['controller_process_start']!=owner['process_start']:
        raise ValueError('Unexpected legacy controller owner')
    owned_controller(owner['pid'],owner['process_start'],original)
    stage=read(original/'stages/train_aligned.json')
    if stage.get('status') not in ('running','launching','exited','complete'):
        raise ValueError('The aligned stage is not healthy')
    if stage.get('status') in ('running','launching') and not legacy.alive(stage):
        raise RuntimeError('The exact aligned supervisor is not alive')
    backup=Path(plan['output'])/'before_handoff';backup.mkdir(exist_ok=True)
    write(backup/'run_state.json',state);write(backup/'train_aligned_stage.json',stage)
    os.kill(owner['pid'],signal.SIGINT)  # NOT killpg; aligned supervisor owns a separate session.
    deadline=time.monotonic()+30
    while process_start(owner['pid'])==owner['process_start']:
        if time.monotonic()>deadline:raise TimeoutError('Original top controller has not released ownership')
        time.sleep(.1)
    after=read(original/'stages/train_aligned.json')
    if stage.get('status') in ('running','launching') and after.get('status') in ('running','launching'):
        if (after.get('pid'),after.get('process_start'))!=(stage.get('pid'),stage.get('process_start')) or not legacy.alive(after):
            raise RuntimeError('Aligned ownership changed during top-controller handoff')
    write(Path(plan['output'])/'handoff_receipt.json',dict(status='controller_only_handoff_complete',
        original_controller=owner,new_controller_pid=os.getpid(),new_controller_start=process_start(os.getpid()),
        aligned_supervisor=dict(pid=after.get('pid'),process_start=after.get('process_start'),child_pid=after.get('child_pid')),
        training_processes_signalled=False,optimizer_restarted=False,original_plan_sha256=plan['original_plan_sha256'],
        execution_plan_sha256=sha(Path(plan['output'])/'run_plan.json'),at_unix=time.time()))


def pulse(plan,state):
    output=Path(plan['output']);original=Path(plan['original'])
    state['updated_unix']=time.time();write(output/'run_state.json',state)
    eq=read(plan['equivalence']['path'])
    message=f"执行器升级：ROI单卡前后向基准×{eq['measured_speedup']:.2f}；当前aligned继续原训练，后续ROI等价真batch。\n"+format_progress(snapshot(original))
    print(message,flush=True)
    for path in (output/'progress.log',original/'progress.log'):
        with path.open('a') as stream:stream.write(message+'\n')
    roi_path=original/'training/roi/progress.json'
    roi=read(roi_path) if roi_path.exists() else {}
    text=['# D 训练执行器提速\n',
        '仅改变训练执行方式：真batch前向、每rank输入常驻GPU、一次性输入验证、合并batch指标通信。模型参数/状态字典、FP32策略、投影深度损失、采样、检测dropout、20k步/global48预算保持。\n',
        f"GPU实测batch16前后向：原逐样本{eq['scalar_step_median_s']:.6f}s，真batch{eq['batch_step_median_s']:.6f}s，配对微基准×{eq['measured_speedup']:.2f}。包含实际输入gather/dropout；排除优化器、NCCL、完整验证和加载，不直接等同整段训练加速倍数。\n",
        f"最大深度差{eq['depth_max_abs_m']:.8g}m，loss相对差{eq['loss_relative_error']:.8g}，梯度相对L2差{eq['gradient_relative_L2']:.8g}。原参数名/shape完全兼容；CPU输入/掩码/dropout/恢复回归与GPU数值检查通过后才迁移。\n",
        '已完成sequence及正在运行aligned原样保留。旧权重缺优化器/RNG/采样状态，未伪装精确续训；只交接顶层控制器，aligned的独立stage/torchrun/rank进程不收信号。未开始ROI从零训练使用新执行器。\n',
        '新fast_resume.pt保存优化器、调度和每rank采样/随机状态；depth*.pt仍供原推理/评价器读取，额外记录实际执行源码。旧源码/计划/配置不改，执行变更以独立PROTOCOL/run_plan/handoff_receipt和原run_state.active_execution_amendment登记。\n',
        f"当前阶段：{state.get('current_stage')}；ROI步数：{roi.get('step',0)}/20000。ROI实际已用{roi.get('elapsed_s')}秒，当前ETA {roi.get('eta_s')}秒；尚未开始时没有整段实测速度。\n",
        'FP32批量运算可能改变浮点累加顺序，因此不宣称训练轨迹逐位相同；完整恢复仅保证相同加速引擎与环境内的优化器边界恢复，已做连续/恢复微训练回归。\n',
        '复核与进度：equivalence_receipt.json、cpu_tests_receipt.json、run_plan.json、handoff_receipt.json；原结果目录继续保存权重与统一评价。生产默认D不变。\n']
    (output/'REPORT_CN.md').write_text('\n'.join(text)+'\n')


def adopt_aligned(plan,state):
    original=Path(plan['original']);old=read(original/'run_plan.json');path=original/'stages/train_aligned.json'
    while True:
        check_plan(plan);r=read(path)
        if r.get('status')=='complete':
            evidence=legacy.verify_stage('train_aligned',old)
            if evidence!=r.get('completion_evidence'):raise ValueError('Completed aligned receipt changed')
            break
        if r.get('status')=='exited':
            if r.get('exitcode')!=0:raise RuntimeError('Aligned stage failed')
            evidence=legacy.verify_stage('train_aligned',old)
            r.update(status='complete',completion_evidence=evidence,verified_unix=time.time(),verified_by_execution_handoff=True)
            write(path,r);break
        if r.get('status')=='failed' or not legacy.alive(r):raise RuntimeError('Aligned stage stopped; no new training launched')
        state['current_stage']='adopt_train_aligned';pulse(plan,state);time.sleep(15)
    old_state=read(original/'run_state.json')
    old_state['stages']['train_aligned']=dict(status='complete',**evidence);write(original/'run_state.json',old_state)
    state['aligned_complete']=True;write(Path(plan['output'])/'run_state.json',state)


def wait_GPUs(plan,ids):
    while True:
        busy=legacy.gpu_usage(ids)
        write(Path(plan['output'])/'resources.json',dict(gpu_ids=ids,busy=busy,status='waiting' if busy else 'ready',at_unix=time.time()))
        if not busy:return
        time.sleep(10)


def worker(plan,name):
    """Persist actual child exit even if the top controller later disconnects."""
    check_plan(plan);original=Path(plan['original']);path=original/'stages'/(name+'.json')
    definition=plan['fast_training'] if name=='train_roi' else plan['evaluation']
    record=dict(schema='d_evidence_stage_v1',stage=name,status='running',pid=os.getpid(),process_start=process_start(os.getpid()),
        plan_sha256=plan['original_plan_sha256'],execution_amendment_sha256=sha(Path(plan['output'])/'run_plan.json'),
        command=definition['command'],started_unix=time.time(),execution_adapter='d_evidence_speed')
    write(path,record);env=os.environ.copy()
    for k in ('RANK','LOCAL_RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','TORCHELASTIC_RUN_ID'):env.pop(k,None)
    env.update(definition['environment']);code=1
    try:
        child=subprocess.Popen(definition['command'],cwd=ROOT,env=env)
        record['child_pid']=child.pid;write(path,record);code=child.wait()
        record.update(status='exited' if code==0 else 'failed',exitcode=code)
    except BaseException as exc:record.update(status='failed',error=repr(exc),exitcode=1);raise
    finally:record.update(finished_unix=time.time(),elapsed_s=time.time()-record['started_unix']);write(path,record)
    return code


def owned_stage(plan,name,state):
    original=Path(plan['original']);path=original/'stages'/(name+'.json')
    if not path.exists():
        wait_GPUs(plan,['1','4','5'] if name=='train_roi' else ['1'])
        log=Path(plan['output'])/'logs'/(name+'.log');log.parent.mkdir(exist_ok=True)
        with log.open('a') as stream:
            child=subprocess.Popen([sys.executable,'-m','d_evidence_speed.run','--output',plan['output'],'--_worker',name],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        state[name+'_supervisor']=dict(pid=child.pid,process_start=process_start(child.pid))
        write(Path(plan['output'])/'run_state.json',state)
        deadline=time.monotonic()+30
        while not path.exists():
            if child.poll() is not None or time.monotonic()>deadline:raise RuntimeError('New stage did not publish its receipt')
            time.sleep(.1)
    while True:
        r=read(path)
        if r.get('execution_amendment_sha256')!=sha(Path(plan['output'])/'run_plan.json'):
            raise ValueError('Do not adopt a stage from another execution plan')
        if r.get('status')=='complete':
            evidence=legacy.verify_stage(name,read(original/'run_plan.json'))
            if r.get('completion_evidence')!=evidence:raise ValueError('Completed stage receipt changed: '+name)
            break
        if r.get('status')=='exited':
            if r.get('exitcode')!=0:raise RuntimeError('Stage returned failure: '+name)
            if name=='train_roi':
                from .train import validate_training
                validate_training(original/'training/roi')
            evidence=legacy.verify_stage(name,read(original/'run_plan.json'))
            r.update(status='complete',completion_evidence=evidence,verified_unix=time.time());write(path,r);break
        if r.get('status')=='failed' or not legacy.alive(r):raise RuntimeError('Owned stage failed/disappeared: '+name)
        state['current_stage']=name
        oldstate=read(original/'run_state.json');oldstate['current_stage']=name;write(original/'run_state.json',oldstate)
        pulse(plan,state);time.sleep(15)
    evidence=legacy.verify_stage(name,read(original/'run_plan.json'))
    oldstate=read(original/'run_state.json');oldstate['stages'][name]=dict(status='complete',**evidence);write(original/'run_state.json',oldstate)


def run(plan):
    output=Path(plan['output']);original=Path(plan['original']);check_plan(plan)
    lock=(output/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    oldlock=None;adopted=False;restore_old=False
    state=dict(status='running',controller_pid=os.getpid(),controller_process_start=process_start(os.getpid()),
        current_stage='preparing_handoff',started_unix=time.time(),source_sha256=plan['source_sha256'])
    try:
        if not (output/'handoff_receipt.json').exists():request_handoff(plan)
        oldlock=(original/'controller.lock').open('a+');fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB);adopted=True
        oldstate=read(original/'run_state.json');oldstate.update(status='running',controller_pid=os.getpid(),
            controller_process_start=process_start(os.getpid()),active_execution_amendment=dict(path=str(output/'run_plan.json'),sha256=sha(output/'run_plan.json')))
        oldstate.pop('error',None);write(original/'run_state.json',oldstate)
        adopt_aligned(plan,state)
        owned_stage(plan,'train_roi',state);owned_stage(plan,'evaluation',state)
        oldstate=read(original/'run_state.json');oldstate.update(status='complete',current_stage=None,finished_unix=time.time(),production_default_changed=False)
        write(original/'run_state.json',oldstate);state.update(status='complete',current_stage=None,finished_unix=time.time());pulse(plan,state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));write(output/'run_state.json',state)
        if adopted:
            oldstate=read(original/'run_state.json');oldstate.update(status='failed',error='speed controller: '+repr(exc));write(original/'run_state.json',oldstate)
        # Before any new ROI stage exists, the unchanged legacy controller can
        # still adopt its unchanged aligned supervisor. Avoid an orphan queue
        # if a handoff-only operation fails after releasing the old owner.
        restore_old=((output/'handoff_receipt.json').exists() and not (original/'stages/train_roi.json').exists()
            and not (original/'training/roi/run_manifest.json').exists())
        raise
    finally:
        if oldlock is not None:oldlock.close()
        lock.close()
        if restore_old:
            oldstate=read(original/'run_state.json');oldstate.pop('active_execution_amendment',None)
            oldstate.update(status='interrupted',execution_handoff_rolled_back=True);write(original/'run_state.json',oldstate)
            with (original/'controller.log').open('a') as stream:
                child=subprocess.Popen([sys.executable,'-m','d_evidence_study.run','--output',str(original)],cwd=ROOT,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            write(output/'rollback_receipt.json',dict(status='legacy_controller_restarted',pid=child.pid,
                training_processes_signalled=False,reason=state.get('error'),at_unix=time.time()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--detach',action='store_true');p.add_argument('--plan-only',action='store_true')
    p.add_argument('--_worker',choices=('train_roi','evaluation'));a=p.parse_args();output=a.output.resolve()
    if a._worker:return worker(read(output/'run_plan.json'),a._worker)
    plan=make_plan(output)
    if a.plan_only:print(json.dumps(plan,ensure_ascii=False,indent=2));return
    path=output/'run_plan.json'
    if path.exists() and read(path)!=plan:raise ValueError('Existing execution plan differs')
    write(path,plan)
    if a.detach:
        with (output/'controller.log').open('a') as stream:
            child=subprocess.Popen([sys.executable,'-m','d_evidence_speed.run','--output',str(output)],cwd=ROOT,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(controller_pid=child.pid,output=str(output),progress=str(output/'progress.log'))));return
    run(plan)


if __name__=='__main__':raise SystemExit(main())
