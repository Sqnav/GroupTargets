"""Short, paired GPU equivalence/throughput check; no experiment optimizer steps."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from bandwidth_study.data import atomic_json
from d_evidence_study.data import load_samples
from d_evidence_study.model import EvidenceDepthLocalizer
from d_evidence_study.train import frozen_trainer
from precision_study.deployed_infer import DEFAULT_DESCRIPTOR,load_descriptor
from rvwm.partitions import canonical_manifest
from .model import FastDepthLocalizer
from .data import GPUFeatureStore

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'results/d_evidence_speed_20261125'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten_grad(model):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten() for p in model.parameters()])


def run(exp,device,repeats):
    if not device.startswith('cuda') or not torch.cuda.is_available():raise ValueError('Actual GPU benchmark required')
    torch.set_num_threads(1)
    cpu_path=exp/'cpu_tests_receipt.json'
    cpu=json.loads(cpu_path.read_text())
    current_sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    if cpu.get('status')!='passed' or cpu.get('source_sha256')!=current_sources:
        raise ValueError('Current CPU loss/gradient/dropout/resume regressions must pass before GPU benchmark')
    legacy_root=ROOT/'results/d_evidence_20261124'
    config=json.loads((legacy_root/'configs/roi.json').read_text())
    ids=canonical_manifest(config)[1]['splits']['train']['episode_ids']
    chosen=sorted(np.random.default_rng(20261125).choice(sorted(ids),16,replace=False).tolist())
    samples,manifest,_=load_samples(config,chosen,'train',count=1,include_labels=True)
    features=[s['features'] for s in samples];positions=[s['positions'] for s in samples]
    if len(samples)!=16:raise ValueError('Exactly16 authenticated current training frames required')
    torch.manual_seed(20261118);old=EvidenceDepthLocalizer(config)
    torch.manual_seed(20261118);fast=FastDepthLocalizer(config)
    # Trained core + deliberately nonzero ROI tests actual branches. These
    # independent benchmark copies are never formal training initialization.
    description,_=load_descriptor(DEFAULT_DESCRIPTOR)
    state=torch.load(description['checkpoint'],map_location='cpu',weights_only=False)['state_dict']
    old.load_state_dict({**old.state_dict(),**state},strict=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(20261125)
        with torch.no_grad():old.roi_projection.weight.normal_(0,.001)
    fast.load_state_dict(old.state_dict(),strict=True)
    old.to(device).train();fast.to(device).train()
    store=GPUFeatureStore(samples,config,device=device)
    indices=np.arange(16,dtype=int)
    packed,_=store.batch(indices,dropout_probability=0.)
    old.zero_grad(set_to_none=True)
    singles=[old.training_loss(f,p) for f,p in zip(features,positions)]
    old_loss=torch.stack([r[0] for r in singles]).mean()
    old_depths=torch.cat([r[1]['prediction']['depths'] for r in singles])
    old_loss.backward();g0=flatten_grad(old).clone()
    fast.zero_grad(set_to_none=True)
    fast_loss,terms=fast.training_loss_batch(packed)
    fast_loss.backward();g1=flatten_grad(fast)
    predicted=terms['prediction']
    lengths=packed.lengths
    fdepths=torch.cat([predicted['depths'][i,:int(n)] for i,n in enumerate(lengths)])
    error=float((old_depths.detach()-fdepths.detach()).abs().max())
    loss_error=float((old_loss.detach()-fast_loss.detach()).abs()/old_loss.detach().abs().clamp_min(1e-12))
    grad_error=float(torch.linalg.vector_norm(g0-g1)/torch.linalg.vector_norm(g0).clamp_min(1e-12))
    del singles,g0,g1,old_depths,fdepths,predicted,terms,old_loss,fast_loss
    trainer,_,_,_=frozen_trainer('roi',formal=False)
    def scalar_step(seed):
        old.zero_grad(set_to_none=True);rng=np.random.default_rng(seed);loss=[]
        for sample in samples:
            f,_,_=trainer.dropout_features(sample['features'],sample['positions'],rng,.2,config)
            loss.append(old.training_loss(f,sample['positions'])[0])
        value=torch.stack(loss).mean();value.backward();return value.detach()
    def batch_step(seed):
        fast.zero_grad(set_to_none=True)
        p,_=store.batch(indices,rng=np.random.default_rng(seed),dropout_probability=.2)
        value=fast.training_loss_batch(p)[0];value.backward();return value.detach()
    aug0=scalar_step(20261125);g0=flatten_grad(old).clone()
    aug1=batch_step(20261125);g1=flatten_grad(fast)
    augmentation_loss_error=float((aug0-aug1).abs()/aug0.abs().clamp_min(1e-12))
    augmentation_grad_error=float(torch.linalg.vector_norm(g0-g1)/torch.linalg.vector_norm(g0).clamp_min(1e-12))
    del g0,g1
    for j in range(2):scalar_step(99+j);batch_step(99+j)
    timing={'scalar':[],'batched':[]}
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    for j in range(repeats):
        order=('scalar','batched') if j%2==0 else ('batched','scalar')
        for name in order:
            torch.cuda.synchronize();start=time.perf_counter()
            (scalar_step if name=='scalar' else batch_step)(20261125+j)
            torch.cuda.synchronize();timing[name].append(time.perf_counter()-start)
    speed=float(np.median(timing['scalar'])/np.median(timing['batched']))
    final_sources={str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    checks=dict(model_state_dict_compatible=set(old.state_dict())==set(fast.state_dict()),
        depth_max_abs_below_1mm=error<.001,loss_relative_below_1e_4=loss_error<1e-4,
        gradient_relative_L2_below_5e_4=grad_error<5e-4,
        augmented_loss_relative_below_1e_4=augmentation_loss_error<1e-4,
        augmented_gradient_relative_L2_below_5e_4=augmentation_grad_error<5e-4,
        source_identity_unchanged=final_sources==current_sources,
        actual_GPU_speedup_above_1_2=speed>1.2)
    result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
        model_state_dict_compatible=checks['model_state_dict_compatible'],same_training_contract=cpu.get('status')=='passed',
        cpu_tests_receipt=dict(path=str(cpu_path),sha256=sha(cpu_path)),
        batch_size=16,arm='roi',device=device,torch_version=str(torch.__version__),
        parameter_count=sum(p.numel() for p in old.parameters()),
        dtype='float32',matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        depth_max_abs_m=error,loss_relative_error=loss_error,gradient_relative_L2=grad_error,
        augmented_loss_relative_error=augmentation_loss_error,augmented_gradient_relative_L2=augmentation_grad_error,
        scalar_step_median_s=float(np.median(timing['scalar'])),batch_step_median_s=float(np.median(timing['batched'])),
        measured_speedup=speed,timing=timing,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        scope='singleGPU paired forward/backward with actual cached16frames and same augmentation seeds; includes gather/dropout, excludes optimizer/NCCL/fullvalidation/load. Own existing aligned training shares GPU, no pause.',
        formal_optimizer_steps=0,formal_fusion_initialization_changed=False,
        stress_weights='publishedD core plus nonzeroROI in independent throwaway copies; never used to initialize formalROI',
        inputs=[dict(episode_id=s['episode_id'],frame_idx=s['frame_idx']) for s in samples],
        original_configuration_sha256=sha(legacy_root/'configs/roi.json'),
        original_run_plan_sha256=sha(legacy_root/'run_plan.json'),
        source_sha256=final_sources,
        protocol_sha256=sha(exp/'PROTOCOL.json'),same_frozen_detector=True)
    atomic_json(exp/'equivalence_receipt.json',result)
    print(json.dumps({k:result[k] for k in ('status','depth_max_abs_m','loss_relative_error','gradient_relative_L2','scalar_step_median_s','batch_step_median_s','measured_speedup')},indent=2),flush=True)
    if result['status']!='passed':raise RuntimeError('Do not install: equivalence or GPU speed criterion failed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--device',default='cuda:0');p.add_argument('--repeats',type=int,default=7)
    a=p.parse_args()
    if a.repeats<3:raise ValueError('At least3 paired timing trials required')
    run(a.output.resolve(),a.device,a.repeats)


if __name__=='__main__':main()
