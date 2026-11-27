"""Read-only gate for the current dataset builder's final acceptance record.

This gate observes the exact requested batch status, rather than an unrelated
FFT summary that may still describe old IQ. It does not launch, resume or alter
the producer. Process liveness and full input provenance are separate checks.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def inspect_build(status_path, expected_episodes=1000):
    """Return a JSON-safe readiness decision from one immutable read snapshot.

    A running producer, missing status or transient incomplete JSON is waiting.
    An explicit failure or a contradictory declared-complete record is blocked.
    Historical attempts are deliberately ignored once final certificates pass.
    """
    path=Path(status_path).resolve()
    result=dict(ready=False,terminal_failure=False,reason='waiting_for_status',
                source_path=str(path),status_sha256=None,build_pid=None,progress={})

    def decision(reason, failed=False):
        result.update(reason=reason,terminal_failure=bool(failed))
        return result

    def exact_int(value):
        return isinstance(value,int) and not isinstance(value,bool)

    if not exact_int(expected_episodes) or expected_episodes<1:
        raise ValueError('Expected episode count must be a positive integer')
    try:
        raw=path.read_bytes()
    except FileNotFoundError:
        return result
    result['status_sha256']=hashlib.sha256(raw).hexdigest()
    try:
        status=json.loads(raw)
    except (json.JSONDecodeError,UnicodeDecodeError):
        return decision('waiting_for_atomic_status_snapshot')
    if not isinstance(status,dict):
        return decision('invalid_status_document',True)
    pid=status.get('pid')
    result['build_pid']=pid if exact_int(pid) and pid>0 else None
    processing=status.get('processing',{})
    processing=processing if isinstance(processing,dict) else {}
    quality=processing.get('quality_control',{})
    quality=quality if isinstance(quality,dict) else {}
    phase_counts=status.get('phase_counts',{})
    phase_counts=phase_counts if isinstance(phase_counts,dict) else {}
    stages=processing.get('stages',{})
    stages=stages if isinstance(stages,dict) else {}
    result['progress']=dict(status=status.get('status'),stage=status.get('stage'),
        completed=status.get('completed'),expected_episodes=expected_episodes,
        failed=status.get('failed'),retry_pending=status.get('retry_pending'),
        completed_frames=status.get('completed_frames'),updated_unix=status.get('updated_unix'),
        phase_counts=phase_counts,processing_status=processing.get('status'),
        processing_active_stage=processing.get('active_stage'),
        processing_stage_status={name:stage.get('status') for name,stage in stages.items() if isinstance(stage,dict)},
        quality_control_passed=quality.get('passed'))
    if status.get('count')!=expected_episodes or not exact_int(status.get('count')):
        return decision('unexpected_batch_episode_count',True)
    if status.get('status') in ('failed','interrupted','stopped','cancelled'):
        result['error']=str(status.get('error',''))
        return decision('builder_terminal_failure',True)
    if processing.get('status') in ('failed','interrupted','stopped'):
        return decision('processing_terminal_failure',True)
    for name in ('fft','validate_fft','observer_analysis'):
        stage=stages.get(name,{})
        if (isinstance(stage,dict) and stage.get('status') in ('failed','interrupted','stopped')
                and not (processing.get('status')=='running' and processing.get('active_stage')!=name)):
            return decision('processing_stage_failure:'+name,True)
    declared_complete=status.get('status')=='complete'
    if not declared_complete:
        return decision('waiting_for_complete_generation_and_preprocessing')

    # A declared-complete batch must be internally consistent. Missing evidence
    # cannot silently pass just because an older external summary looks valid.
    if status.get('completed')!=expected_episodes or status.get('failed')!=0 or status.get('retry_pending',0)!=0:
        return decision('inconsistent_final_episode_totals',True)
    for name in ('plan','capture','echo'):
        count=phase_counts.get(name,{})
        if (not isinstance(count,dict) or count.get('passed')!=expected_episodes
                or any(count.get(key,0)!=0 for key in ('failed','blocked','pending'))):
            return decision('incomplete_final_phase:'+name,True)
    if status.get('core_complete') is not True:
        return decision('missing_core_completion_certificate',True)
    diagnostic_only=(status.get('radar_dataset_policy')=='unconstrained_detection_v1')
    if diagnostic_only:
        if (status.get('radar_diagnostics_complete') is not True
                or quality.get('observer_detection_required') is not False
                or quality.get('diagnostic_only') is not True
                or quality.get('radar_dataset_policy')!='unconstrained_detection_v1'
                or quality.get('integrity_passed') is not True):
            return decision('missing_unconstrained_radar_diagnostic_certificate',True)
    elif status.get('observer_qualified_complete') is not True:
        return decision('missing_observer_completion_certificate',True)
    if processing.get('status')!='complete':
        return decision('missing_final_processing_certificate',True)
    for name in ('fft','validate_fft','observer_analysis'):
        stage=stages.get(name,{})
        if (not isinstance(stage,dict) or stage.get('status')!='complete'
                or stage.get('returncode')!=0):
            return decision('missing_successful_processing_stage:'+name,True)
    if quality.get('passed') is not True:
        return decision('quality_control_failed_or_missing',True)
    formal=quality.get('formal_validation',{})
    if diagnostic_only:
        if (not isinstance(formal,dict) or formal.get('acceptance_required') is not False
                or formal.get('required_frames')!=0
                or formal.get('acceptance_passed') is not True or formal.get('checked_all_frames') is not True):
            return decision('missing_formal_all_frame_diagnostic_certificate',True)
    elif not isinstance(formal,dict) or formal.get('passed') is not True or formal.get('all_frames_certified') is not True:
        return decision('missing_formal_all_frame_certificate',True)
    keys=('total_frames','included_frames','expected_capture_frames') if diagnostic_only else (
        'total_frames','included_frames','physical_matched_frames','expected_capture_frames')
    totals=[quality.get(key) for key in keys]
    totals.append(formal.get('frames'))
    if any(not exact_int(value) or value<=0 for value in totals) or len(set(totals))!=1:
        return decision('inconsistent_all_frame_acceptance_totals',True)
    if status.get('completed_frames')!=totals[0]:
        return decision('completed_frames_disagree_with_acceptance',True)
    if diagnostic_only:
        matched=quality.get('physical_matched_frames')
        if not exact_int(matched) or not 0<=matched<=totals[0]:
            return decision('invalid_observer_diagnostic_match_count',True)
    if quality.get('failed_episodes',[]) and not diagnostic_only:
        return decision('quality_control_contains_failed_episodes',True)
    result.update(ready=True,reason='complete_generation_and_all_frame_diagnostics' if diagnostic_only else
                  'complete_generation_preprocessing_and_all_frame_acceptance',accepted_frames=totals[0])
    if diagnostic_only:
        result.update(observer_detection_required=False,observer_detection_success_not_guaranteed=True)
    return result
