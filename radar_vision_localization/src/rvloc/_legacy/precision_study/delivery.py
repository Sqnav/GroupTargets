"""Consolidate saved evaluations into one metrics CSV and artifact index.

Pure offline reporting: no model, checkpoint pickle, dataset, sensor reader,
training or default-changing operation is imported or performed. Test output
requires the same explicit freeze gate as report_plots before metric reads.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import numpy as np
from . import report_plots
from .comparison import _load,_metric,compare_evaluations

ROOT=Path(__file__).resolve().parents[1]


def _read(path):return json.loads(Path(path).read_text())


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def artifact(path):
    """Only index existing files; logs may be growing, explicitly retain that."""
    if not path:return None
    path=Path(path).resolve()
    if not path.is_file():return None
    before=path.stat();digest=_sha(path);after=path.stat()
    return dict(path=str(path),sha256=digest,size_bytes=after.st_size,
                stable_during_hash=(before.st_mtime_ns,before.st_size)==(after.st_mtime_ns,after.st_size))


def _benchmark_identity(report,info):
    if report.get('status')!='complete' or report.get('split') not in ('train','val') or report.get('labels_read') is not False:
        return None,'not a completed label-free train/val benchmark'
    identity=report.get('model_identity',{})
    if identity.get('steps')!=info.get('checkpoint_step'):return None,'checkpoint optimizer step differs'
    observed=identity.get('checkpoint_sha256');expected=info.get('checkpoint_sha256')
    if not expected:return None,'evaluation checkpoint identity missing'
    if observed==expected:return dict(method='exact_checkpoint_sha256'),None
    measured=report.get('checkpoint')
    if not measured:return None,'benchmark checkpoint path missing'
    receipt=Path(measured).parent/'deployment_receipt.json'
    if not receipt.is_file():return None,'checkpoint SHA differs and deployment lineage missing'
    value=_read(receipt)
    if (value.get('status')!='verified_label_free' or value.get('sanitized') is not True
            or _sha(receipt)!=report.get('deployment_receipt_sha256')
            or value.get('checkpoint_sha256')!=observed
            or value.get('source_checkpoint_sha256')!=expected
            or value.get('state_tensor_sha256')!=report.get('deployment_state_tensor_sha256')
            or value.get('exact_state_dict_preserved') is not True):
        return None,'deployment receipt does not bind this evaluated checkpoint'
    return dict(method='exact_state_dict_preserved_deployment_lineage',receipt=artifact(receipt)),None


def benchmarks_for(name,info,directory,root):
    """Match identities, never assign another seed/prefix timing by architecture."""
    root=Path(root).resolve();result={};aliases={name,info.get('model_name'),Path(directory).name}
    for device in ('cpu','gpu'):
        candidates=[];rejected=[]
        for path in sorted(root.glob('*_'+device+'.json')):
            value=_read(path);binding,reason=_benchmark_identity(value,info)
            if reason:
                if path.stem[:-len('_'+device)] in aliases:rejected.append(dict(path=str(path),reason=reason))
                continue
            correct_device=value.get('device','').startswith('cpu' if device=='cpu' else 'cuda')
            if not correct_device:continue
            source=value.get('model_identity',{}).get('source_sha256')
            if source is not None and info.get('checkpoint_source_sha256') is not None and source!=info['checkpoint_source_sha256']:
                rejected.append(dict(path=str(path),reason='checkpoint source hashes differ'));continue
            candidates.append((path,value,binding))
        named=[x for x in candidates if x[0].stem[:-len('_'+device)] in aliases]
        selected=named if named else candidates
        if len(selected)!=1:
            result[device]=dict(status='missing' if not selected else 'ambiguous',measurement=None,
                rejected_candidates=rejected,matching_candidate_paths=[str(x[0]) for x in selected]);continue
        path,value,binding=selected[0]
        keep=('status','device','input_frames','measured_predictions','repeats_per_frame','warmup_predictions',
              'latency_ms','gpu_peak_allocated_bytes','gpu_peak_reserved_bytes','gpu_memory_scope','scope',
              'feature_host_to_device_copy_in_predict_included','labels_read','cuda_synchronized_before_and_after_each_prediction',
              'torch_version','cpu_torch_threads','measured_at','checkpoint','split','source_root','frozen_source_root',
              'checkpoint_kind','command','split_manifest_sha256','deployment_state_tensor_sha256','parameter_count')
        result[device]=dict(status='identity_verified',measurement={k:value.get(k) for k in keep},
            file=artifact(path),identity_binding=binding,
            hardware_context=artifact(path.with_name(path.stem+'_hardware_context.json')),
            rejected_candidates=rejected)
    return result


def _axis_metric(members,field):
    values=[r[field] for r in members if r['status']=='localized' and r.get(field) is not None and np.isfinite(r[field])]
    metric=_metric(values);metric['mae']=float(np.mean(np.abs(values))) if values else None
    return metric


def _training_index(info):
    checkpoint=Path(info['checkpoint']).resolve() if info.get('checkpoint') else None
    if checkpoint is None:return {}
    receipt=checkpoint.parent/'deployment_receipt.json'
    if receipt.is_file():
        value=_read(receipt)
        if value.get('checkpoint_sha256')==info.get('checkpoint_sha256') and value.get('source_checkpoint_path'):
            checkpoint=Path(value['source_checkpoint_path']).resolve()
    directory=checkpoint.parent
    names=('run_manifest.json','launch_receipt.json','metrics.jsonl','prior_validation.jsonl','progress.json')
    indexed={name:artifact(directory/name) for name in names if (directory/name).is_file()}
    logs=[artifact(path) for path in sorted(directory.glob('*.log')) if path.is_file()]
    if logs:indexed['logs']=logs
    return dict(directory=str(directory),files=indexed) if indexed else {}


def aggregate(evaluations,benchmarks,references=None,freeze=None,draws=2000):
    # This performs the complete all-input provenance gate first. No benchmark,
    # member/summary second pass or artifact index is read before that gate.
    pairs=dict(evaluations) if isinstance(evaluations,dict) else list(evaluations)
    names=list(pairs) if isinstance(pairs,dict) else [n for n,_ in pairs]
    refs=list(references or names[:1])
    prepared=report_plots.collect(pairs,reference=refs[0],draws=draws,freeze=freeze)
    comparison=prepared['comparison'] if len(refs)==1 else compare_evaluations(pairs,references=refs,draws=draws,allow_test=prepared['test_set_read'])
    if comparison['status']!='valid':raise ValueError('Strict evaluation population failed')
    rows=[];models={}
    for table in comparison['models']:
        name=table['name'];saved=prepared['models'][name];info=saved['run_info'];summary=saved['summary']
        directory=Path(saved['evaluation_dir']);loaded=_load(name,directory,prepared['test_set_read'])
        members=list(loaded['members'].values());timing=benchmarks_for(name,info,directory,benchmarks)
        source_files={key:artifact(directory/(key+suffix)) for key,suffix in
            (('frames','.jsonl'),('members','.jsonl'),('detections','.jsonl'),('episodes','.json'),
             ('summary','.json'),('run_info','.json'),('input_manifest','.json'),('input_inventory','.json'))}
        training=_training_index(info);trainfiles=training.get('files',{})
        row=dict(model=name,split=info.get('split'),frames_count=table['frames'],gt_members_count=table['truth_members'],
            detection_instances_count=table['detection_instances'],matched_detection_count=table['matched_detection_instances'],
            localized_matched_members_count=table['localized_matched_members'],
            unmatched_gt_count=table['truth_members']-table['matched_detection_instances'],
            unmatched_detection_count=table['detection_instances']-table['matched_detection_instances'],
            unlocalized_matched_members_count=table['matched_detection_instances']-table['localized_matched_members'],
            failure_total_count=table['unlocalized_or_unmatched_members'],
            position_rmse_m=table['conditional_member_rmse_m'],position_mean_error_m=table['conditional_member_mean_error_m'],
            position_p95_abs_m=table['conditional_member_p95_m'],matched_gt_ratio=table['matched_detection_rate'],
            localization_effective_gt_ratio=table['localization_effective_rate'],
            checkpoint_step=info.get('checkpoint_step'),checkpoint_path=info.get('checkpoint'),
            checkpoint_sha256=info.get('checkpoint_sha256'),configuration_path=info.get('configuration_path'),
            configuration_sha256=info.get('configuration_sha256'),code_root=info.get('code_root'),
            evaluation_directory=str(directory),evaluation_device=info.get('device'),
            evaluation_command_json=json.dumps(info.get('command'),ensure_ascii=False),
            evaluation_command_shell=shlex.join(info['command']) if isinstance(info.get('command'),list) else None,
            evaluation_source_sha256=info.get('evaluation_source_sha256'),
            entrypoint_source_sha256=info.get('entrypoint_source_sha256'),
            training_run_manifest_path=trainfiles.get('run_manifest.json',{}).get('path'),
            training_launch_receipt_path=trainfiles.get('launch_receipt.json',{}).get('path'),
            training_log_paths_json=json.dumps([x['path'] for x in trainfiles.get('logs',[])]))
        axis={}
        for label,field,unit in (('range','base_range_error_m','m'),('azimuth','base_azimuth_error_deg','deg'),('elevation','base_elevation_error_deg','deg')):
            metric=_axis_metric(members,field);axis[label]=metric
            row[label+'_error_count']=metric['count']
            for key in ('mae','rmse','p95'):row[label+'_'+key+'_'+unit]=metric[key]
        for t in (1,3,5):
            row[f'success_all_gt_{t}m_ratio']=table[f'success_ratio_all_gt_{t}m']
            row[f'success_all_gt_{t}m_count']=table[f'success_count_{t}m']
        for status,count in table['failure_counts'].items():row['failure_status_'+status+'_count']=count
        latency=summary.get('inference_seconds_per_frame',{})
        row.update(evaluation_capture_latency_mean_ms=latency.get('mean')*1000 if latency.get('mean') is not None else None,
            evaluation_capture_latency_p95_ms=latency.get('p95_abs')*1000 if latency.get('p95_abs') is not None else None,
            evaluation_capture_latency_scope=summary.get('latency_scope'))
        for device,benchmark in timing.items():
            value=benchmark['measurement'] or {};latency=value.get('latency_ms') or {}
            row.update({device+'_benchmark_status':benchmark['status'],device+'_latency_mean_ms':latency.get('mean'),
                device+'_latency_p95_ms':latency.get('p95'),device+'_latency_scope':value.get('scope'),
                device+'_benchmark_path':benchmark.get('file',{}).get('path')})
        gpu=timing['gpu']['measurement'] or {}
        row.update(gpu_peak_allocated_bytes=gpu.get('gpu_peak_allocated_bytes'),
            gpu_peak_reserved_bytes=gpu.get('gpu_peak_reserved_bytes'),gpu_memory_scope=gpu.get('gpu_memory_scope'))
        for key in ('frames','members','detections','episodes'):
            row[key+'_path']=(source_files[key] or {}).get('path');row[key+'_sha256']=(source_files[key] or {}).get('sha256')
        row['failure_counts_json']=json.dumps(table['failure_counts'],sort_keys=True)
        rows.append(row)
        model=dict(metrics=row,axis_errors=axis,failure_counts=table['failure_counts'],
            fixed_ray_oracle_diagnostic=summary.get('fixed_ray_oracle_diagnostic'),
            stage_diagnostics=summary.get('stage_diagnostics'),stage_update_diagnostics=summary.get('stage_update_diagnostics'),
            paths=source_files,training=training,benchmarks=timing,
            identity=dict(checkpoint=artifact(info.get('checkpoint')),configuration=artifact(info.get('configuration_path')),
                checkpoint_recorded_sha256=info.get('checkpoint_sha256'),configuration_recorded_sha256=info.get('configuration_sha256'),
                checkpoint_source_sha256=info.get('checkpoint_source_sha256'),code_root=info.get('code_root'),
                evaluation_source_sha256=info.get('evaluation_source_sha256'),entrypoint_source_sha256=info.get('entrypoint_source_sha256'),
                evaluation_command=info.get('command'),test_freeze=info.get('test_freeze')))
        for field in ('checkpoint','configuration'):
            actual=model['identity'][field];expected=model['identity'][field+'_recorded_sha256']
            model['identity'][field+'_current_file_matches_saved_identity']=actual['sha256']==expected if actual and expected else None
        models[name]=model
    failure_fields=sorted({key for row in rows for key in row if key.startswith('failure_status_')})
    for row in rows:
        for field in failure_fields:row.setdefault(field,0)
    return dict(status='saved_evaluation_delivery',created_at=datetime.now(timezone.utc).isoformat(),
        test_set_read=prepared['test_set_read'],test_sensor_inputs_read=False,test_freeze=prepared['test_freeze'],
        default_changed=False,effect_judgement_or_model_selection_performed=False,
        benchmark_root=str(Path(benchmarks).resolve()),references=refs,models=models,metrics=rows,comparison=comparison,
        metric_units=dict(position='m',range='m',azimuth='deg',elevation='deg',latency='ms',gpu_allocator_memory='bytes',
            success_and_matching_ratios='dimensionless fractions in [0,1], all-GT denominators; never percent',
            error_statistics='finite localized fixed-matched members only; reported error_count retains metric-specific missingness'),
        scope='Only saved offline evaluation outputs, public benchmark/deployment JSON receipts, and artifact byte hashes. All declared GT members remain in success/coverage denominators; benchmark gaps are null, not inferred from another architecture/seed.',
        source_sha256={str(Path(__file__).resolve()):_sha(__file__),str(Path(report_plots.__file__).resolve()):_sha(report_plots.__file__)})


def write_delivery(report,output):
    output=Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):raise FileExistsError('Protected delivery output: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(key for row in report['metrics'] for key in row))
    with (output/'metrics.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(report['metrics'])
    report=dict(report,outputs=dict(metrics_csv=artifact(output/'metrics.csv')))
    with (output/'report.json').open('x') as stream:json.dump(report,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evaluations',nargs='+',help='NAME=/saved/evaluation')
    p.add_argument('--benchmarks',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',action='append');p.add_argument('--freeze',type=Path);p.add_argument('--draws',type=int,default=2000)
    a=p.parse_args(argv)
    if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Protected delivery output')
    report=aggregate(report_plots._pairs(a.evaluations,p),a.benchmarks,a.reference,a.freeze,a.draws)
    write_delivery(report,a.output)
    print(json.dumps(dict(status=report['status'],models=list(report['models']),output=str(a.output),test_set_read=report['test_set_read'])))
    return 0


if __name__=='__main__':raise SystemExit(main())
