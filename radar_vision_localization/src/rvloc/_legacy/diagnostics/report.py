"""Publish an honest live comparison without loading models or changing runs.

The frozen-fusion interventions and the retrained visual/geometry control have
different meanings.  Only completed, protocol-matched training runs contribute
to the final fusion-versus-control comparison.  This module uses only stdlib so
its watcher does not allocate GPU memory or import the training runtime.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT / 'results/diagnostics_radar_20261121'
VARIANT = 'retrained_visual_geometry_only'
REQUIRED_ARMS = ('baseline', 'zero_radar', 'zero_ra', 'zero_re', 'joint_shuffle',
                 'range_shift_minus2', 'range_shift_minus1', 'range_shift_plus1',
                 'range_shift_plus2', 'oracle_true_bearing')
METRICS = (
    ('member_rmse_m', '成员三维 RMSE（米）', False),
    ('member_p95_m', '成员三维 P95（米）', False),
    ('coverage_1m', '成员 ≤1米覆盖率', True),
    ('coverage_2m', '成员 ≤2米覆盖率', True),
    ('coverage_5m', '成员 ≤5米覆盖率', True),
    ('whole_scene_coverage_5m', '整群全部 ≤5米比例', True),
    ('center_rmse_m', '完整群中心 RMSE（米）', False),
)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _read(path, errors):
    path = Path(path)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('JSON root must be an object')
        return value
    except (OSError, ValueError) as exc:
        errors.append(dict(path=str(path), error=str(exc)))
        return None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _receipt(path):
    path = Path(path)
    return dict(path=str(path), sha256=_sha(path)) if path.is_file() else None


def _atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            stream.write(text)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def primary_metrics(report):
    """Preserve deployed-primary metrics; never substitute the candidate oracle."""
    summary = report['summary']
    truth = summary.get('truth_member_instances')
    rows = report.get('frames', [])
    errors = [float(value) for row in rows for value in row.get('primary_member_errors_m', [])
              if _number(value)]
    result = dict(frames=summary.get('frames'), truth_members=truth,
        matched_members=summary.get('primary_matched_member_instances'),
        member_rmse_m=summary.get('primary_member_rmse_m'),
        member_p95_m=summary.get('primary_member_p95_m'),
        coverage_5m=summary.get('primary_member_success_ratio_5m'),
        whole_scene_coverage_5m=summary.get('primary_whole_scene_success_ratio_5m'),
        center_rmse_m=summary.get('primary_center_rmse_m'),
        selected_step=report.get('step'), selection_policy=summary.get('selection_policy'),
        deployment_sampling_seed=summary.get('deployment_sampling_seed'),
        coverage_denominator='all truth members, including unmatched and rejected',
        error_scope='matched members only; center RMSE uses complete matching frames')
    for threshold in (1, 2):
        result[f'coverage_{threshold}m'] = (sum(value <= threshold for value in errors) / truth
            if rows and _number(truth) and truth > 0 else None)
    if isinstance(truth, int) and isinstance(result['matched_members'], int):
        result['unmatched_truth_members'] = truth - result['matched_members']
    if rows:
        result['extra_detection_instances'] = sum(max(0, row.get('detection_count', 0)
            - len(row.get('primary_member_errors_m', []))) for row in rows)
        result['complete_center_frames'] = sum(_number(row.get('primary_center_error_m')) for row in rows)
    return result


def _frame_keys(report):
    rows = report.get('frames', [])
    return [(row.get('episode_id'), row.get('frame_idx')) for row in rows]


def protocol_checks(fusion, baseline, fusion_report, baseline_report):
    """Explicitly establish the matched control before reporting radar gain."""
    checks = []

    def check(name, left, right):
        checks.append(dict(name=name, passed=left is not None and left == right))

    for name in ('train_episodes', 'validation_episodes', 'source_sha256', 'prepared_training_groups',
                 'prepared_validation_groups', 'world_size', 'global_batch_size', 'configured_window',
                 'model_selection', 'seed', 'deployment_sampling_seed', 'training_sampling_policy'):
        check(name, fusion.get(name), baseline.get(name))
    for name in ('steps', 'validate_every', 'batch_size', 'learning_rate', 'detection_dropout', 'seed',
                 'physics_every', 'physics_weight', 'max_episodes', 'frames_per_episode',
                 'validation_episodes', 'validation_frames'):
        check('hyperparameters.' + name, fusion.get('hyperparameters', {}).get(name),
              baseline.get('hyperparameters', {}).get(name))
    fc, bc = fusion.get('configuration', {}), baseline.get('configuration', {})
    for name in ('architecture_schema', 'group_hidden_dim', 'group_latent_dim', 'group_max_members',
                 'group_attention_heads', 'group_interaction_layers', 'group_hypotheses',
                 'detector_backend', 'detector_weights_sha256', 'detector_confidence',
                 'split_manifest_sha256', 'visual_point_mode', 'coordinate_frame', 'window'):
        check('configuration.' + name, fc.get(name), bc.get(name))
    for name in ('weights_sha256', 'visual_split_manifest_sha256', 'track_detection_manifest_sha256'):
        check('frontend.' + name, fusion.get('frontend', {}).get(name), baseline.get('frontend', {}).get(name))
    check('validation.frames', fusion_report['summary'].get('frames'), baseline_report['summary'].get('frames'))
    check('validation.truth_member_instances', fusion_report['summary'].get('truth_member_instances'),
          baseline_report['summary'].get('truth_member_instances'))
    keys = _frame_keys(fusion_report)
    check('validation.exact_episode_frame_membership', keys or None, _frame_keys(baseline_report))
    for name in ('selection_policy', 'deployment_sampling_seed'):
        check('validation.' + name, fusion_report['summary'].get(name), baseline_report['summary'].get(name))
    checks.extend([
        dict(name='fusion.original_run_complete', passed=fusion.get('status') == 'complete'
             and fusion.get('model_trained') is True),
        dict(name='baseline.is_retrained_visual_geometry_control', passed=baseline.get('model_variant') == VARIANT
             and baseline.get('raw_measured_radar_used') is False),
        dict(name='baseline.explicit_zero_measurement_contract', passed=baseline.get('diagnostic_baseline_contract', {}).get('measured_radar_used') is False),
        dict(name='baseline.full_training_frame_coverage', passed=baseline.get('all_usable_training_frames_used') is True),
        dict(name='baseline.optimizer_complete', passed=baseline.get('optimizer_steps') == fusion.get('steps')
             and baseline.get('steps') == fusion.get('steps')),
        dict(name='fusion.best_validation_matches_selected_checkpoint', passed=isinstance(fusion_report.get('step'), int)
             and fusion_report['step'] > 0 and fusion_report['step'] == fusion.get('best_step')),
        dict(name='baseline.best_validation_matches_selected_checkpoint', passed=isinstance(baseline_report.get('step'), int)
             and baseline_report['step'] > 0 and baseline_report['step'] == baseline.get('best_step')),
    ])
    return checks


def _frozen_status(root, fusion, fusion_metrics, errors):
    summary_path = root / 'run.summary.json'
    if not summary_path.exists():
        summary_path = root / 'summary.json'
    summary = _read(summary_path, errors)
    partial = [_read(path, errors) for path in sorted(root.glob('part_*/manifest.json'))]
    partial = [row for row in partial if row]
    result = dict(status='pending', root=str(root), completed_frames=sum(row.get('completed_frames', 0) for row in partial),
        expected_frames=max((row.get('expected_total_frames', 0) for row in partial), default=0),
        shard_statuses={str(row.get('rank')): row.get('status') for row in partial})
    if any(row.get('status') == 'failed' for row in partial):
        result['status'] = 'failed'
    if summary is None:
        return result
    result.update(summary_receipt=_receipt(summary_path), summary=summary)
    if summary.get('status') != 'complete':
        return result
    checks = [dict(name='frozen.complete_validation_population', passed=summary.get('split') == 'val'
                   and summary.get('frames') == fusion_metrics.get('frames')),
              dict(name='frozen.correct_selected_fusion_checkpoint', passed=bool(fusion.get('selected_checkpoint_sha256'))
                   and summary.get('checkpoint', {}).get('sha256') == fusion.get('selected_checkpoint_sha256')),
              dict(name='frozen.all_requested_interventions', passed=all(arm in summary.get('arms', {}) for arm in REQUIRED_ARMS))]
    result.update(checks=checks, status='complete' if all(row['passed'] for row in checks) else 'invalid')
    result['completed_frames'] = summary.get('frames')
    return result


def build_comparison(output_root=DEFAULT_OUTPUT, fusion_root=None, frozen_root=None, baseline_root=None):
    output_root = Path(output_root).resolve()
    fusion_root = Path(fusion_root or PROJECT / 'results/training/single').resolve()
    frozen_root = Path(frozen_root or output_root / 'frozen_model').resolve()
    baseline_root = Path(baseline_root or output_root / 'visual_baseline').resolve()
    errors = []
    fusion = _read(fusion_root / 'run_manifest.json', errors)
    fusion_report = _read(fusion_root / 'prior_validation_best.json', errors)
    baseline = _read(baseline_root / 'run_manifest.json', errors)
    progress = _read(baseline_root / 'progress.json', errors) or {}
    baseline_report = _read(baseline_root / 'prior_validation_best.json', errors)
    result = dict(schema='radar_contribution_comparison_v1', status='pending',
        generated_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        output_root=str(output_root), read_errors=errors, final_comparison=None,
        limitation='single seeded validation comparison; not an independent test or statistical proof')
    if fusion is None or fusion_report is None or not isinstance(fusion_report.get('summary'), dict):
        result['reason'] = 'original_fusion_receipts_not_ready'
        return result
    fm = primary_metrics(fusion_report)
    result['fusion'] = dict(root=str(fusion_root), status=fusion.get('status'), metrics=fm,
        manifest_receipt=_receipt(fusion_root / 'run_manifest.json'),
        validation_receipt=_receipt(fusion_root / 'prior_validation_best.json'))
    frozen = _frozen_status(frozen_root, fusion, fm, errors)
    result['frozen_model'] = frozen
    requested = (baseline or {}).get('hyperparameters', {}).get('steps', progress.get('requested_steps', 20000))
    bs = dict(root=str(baseline_root), status='pending', stage=progress.get('stage', (baseline or {}).get('status', 'not_started')),
        completed_steps=progress.get('step', (baseline or {}).get('optimizer_steps', 0)), requested_steps=requested,
        eta_s=progress.get('eta_s'), steps_per_second=progress.get('steps_per_second'),
        measured_radar_used=False, interpretation='DINO bbox-centre rays/navigation/public geometry; no RGB appearance or measured spectra')
    result['visual_baseline'] = bs
    if baseline_report and isinstance(baseline_report.get('summary'), dict):
        report_step = baseline_report.get('step')
        if isinstance(report_step, int) and report_step > 0:
            bs.update(preliminary_best_metrics=primary_metrics(baseline_report), preliminary_only=True)
        else:
            bs['initial_untrained_validation'] = primary_metrics(baseline_report)
    checks = []
    if progress.get('stage') == 'failed' or (baseline or {}).get('status') == 'failed':
        bs.update(status='failed', error=progress.get('error'))
    elif baseline and baseline.get('status') == 'complete' and progress.get('stage') == 'complete':
        if baseline_report and isinstance(baseline_report.get('summary'), dict):
            checks = protocol_checks(fusion, baseline, fusion_report, baseline_report)
            bs.update(status='complete' if all(row['passed'] for row in checks) else 'invalid', checks=checks,
                metrics=primary_metrics(baseline_report), preliminary_only=False,
                manifest_receipt=_receipt(baseline_root / 'run_manifest.json'),
                validation_receipt=_receipt(baseline_root / 'prior_validation_best.json'))
        else:
            bs['status'] = 'invalid'
            bs['error'] = 'Completed baseline has no selected-best validation report'
    if 'failed' in (bs['status'], frozen['status']):
        result['status'] = 'failed'
    elif 'invalid' in (bs['status'], frozen['status']):
        result['status'] = 'invalid'
    elif bs['status'] == 'complete' and frozen['status'] == 'complete' and not errors:
        result['status'] = 'complete'
    if result['status'] == 'complete':
        bm = bs['metrics']
        values = {}
        for name, _, ratio in METRICS:
            f, b = fm.get(name), bm.get(name)
            if _number(f) and _number(b):
                # Positive always means the measured-radar fusion is better.
                values[name] = dict(fusion=f, visual_geometry=b, fusion_minus_visual=f - b,
                    fusion_improvement=(f - b) * 100 if ratio else b - f,
                    improvement_unit='percentage_points' if ratio else 'metres')
        result['final_comparison'] = dict(status='complete', metrics=values,
            selection='each run own validation-selected best; same primary selection protocol',
            interpretation='positive improvement favours fusion; no multi-seed confidence interval; no independent test')
    return result


def _format(value, ratio=False):
    if not _number(value):
        return '—'
    return f'{100 * value:.2f}%' if ratio else f'{value:.3f}'


def render_markdown(result):
    labels = dict(pending='等待完成', complete='已完成', failed='执行失败', invalid='核验未通过')
    lines = [f'更新：{result["generated_at"]}。状态：{labels.get(result["status"], result["status"])}。', '']
    if 'fusion' not in result:
        lines += ['原融合训练记录尚未就绪，不能形成比较结论。', '']
    else:
        fusion = result['fusion']['metrics']
        lines += [f'固定融合结果采用验证集选出的第 {fusion.get("selected_step")} 步权重，'
            f'{fusion.get("frames")} 帧、{fusion.get("truth_members")} 个真实成员。'
            f'主方案成员 RMSE {_format(fusion.get("member_rmse_m"))} 米，'
            f'≤5米覆盖率 {_format(fusion.get("coverage_5m"), True)}。', '']
        baseline = result['visual_baseline']
        stage = dict(training='训练中', validating='验证中', preparing='准备输入', complete='完成',
                     failed='失败', not_started='尚未启动', dry_run_complete='仅完成预演').get(baseline['stage'], baseline['stage'])
        eta = f'，当前预计剩余 {baseline["eta_s"] / 60:.1f} 分钟' if _number(baseline.get('eta_s')) else ''
        lines += [f'独立视觉几何控制组：{stage}，{baseline["completed_steps"]}/{baseline["requested_steps"]} 步{eta}。'
            '它使用同一 DINO 检测框中心射线、观察机导航和公开传感器几何，训练和验证的实测 RA/RE 功率均严格置零。'
            '这是重新训练的同架构控制组，不是完整 RGB 单目深度网络。', '']
        comparison = result.get('final_comparison')
        if comparison:
            lines += ['独立控制组已完成，以下比较各自按相同验证规则选出的最佳权重。正的增益表示融合更好。', '',
                      '| 指标 | 固定融合 | 重训视觉几何 | 融合增益 |', '|---|---:|---:|---:|']
            for name, label, ratio in METRICS:
                item = comparison['metrics'].get(name, {})
                improvement = item.get('fusion_improvement')
                gain = (f'{improvement:+.2f} 个百分点' if ratio else f'{improvement:+.3f} 米') if _number(improvement) else '—'
                lines.append(f'| {label} | {_format(item.get("fusion"), ratio)} | {_format(item.get("visual_geometry"), ratio)} | {gain} |')
            lines += ['', '这是同一次种子设置下的验证集对照；尚不能据此声称独立测试精度或统计显著性。', '']
        else:
            lines += ['最终融合增益比较：pending。控制组及诊断完成并通过同数据、同架构和同训练协议核验前，不能判断雷达是否带来定位增益。', '']
            if baseline.get('preliminary_best_metrics'):
                preliminary = baseline['preliminary_best_metrics']
                lines += [f'训练中的临时最佳验证结果：第 {preliminary.get("selected_step")} 步，RMSE '
                    f'{_format(preliminary.get("member_rmse_m"))} 米，≤5米覆盖率 '
                    f'{_format(preliminary.get("coverage_5m"), True)}；此结果仍是临时值。', '']
            if baseline.get('initial_untrained_validation'):
                lines += ['当前最佳报告仍来自初始化，尚未形成已训练步骤的验证结果；优化器步数不能替代验证结果。', '']
        failed_checks = [check['name'] for check in baseline.get('checks', []) if not check['passed']]
        if failed_checks:
            lines += ['控制组比较核验未通过：' + '、'.join(failed_checks) + '。', '']
        frozen = result['frozen_model']
        lines += [f'固定权重诊断：{labels.get(frozen["status"], frozen["status"])}，'
            f'{frozen.get("completed_frames", 0)}/{frozen.get("expected_frames") or fusion.get("frames")} 帧。', '']
        if frozen['status'] == 'complete':
            arms = frozen['summary']['arms']
            normal = arms['baseline']['raw_model']
            cpu_rmse, gpu_rmse = normal.get('member_error', {}).get('rmse'), fusion.get('member_rmse_m')
            cpu_text = f'{cpu_rmse:.6f}' if _number(cpu_rmse) else '—'
            gpu_text = f'{gpu_rmse:.6f}' if _number(gpu_rmse) else '—'
            lines += [f'本表来自固定权重的 CPU 诊断，原训练验证采用 GPU。CPU 主方案 RMSE '
                f'{cpu_text} 米，GPU 原记录 {gpu_text} 米；小幅浮点差异可能改变临界样本。'
                '重训控制组的公平比较始终采用上方原 GPU 验证记录。', '',
                f'按同一观测射线匹配分解，沿射线误差贡献 {_format(normal.get("parallel_mse_fraction"), True)} 的三维 MSE，'
                f'垂直射线贡献 {_format(normal.get("perpendicular_mse_fraction"), True)}。'
                f'固定射线和解析支撑内的 GT 辅助位置下界 RMSE 为 '
                f'{_format(normal.get("supported_ray_floor", {}).get("rmse"))} 米；该下界不是模型可部署的精度。', '',
                '| 干预 | 原始模型 RMSE（米） | ≤5米覆盖率 | 同槽位移均值（米） | 正常流程可发布帧 |',
                '|---|---:|---:|---:|---:|']
            names = dict(baseline='原始实测 RA/RE', zero_radar='RA/RE 全零（分布外）', zero_ra='RA 全零（分布外）',
                         zero_re='RE 全零（分布外）', joint_shuffle='跨场景联合打乱 RA/RE',
                         range_shift_minus2='距离轴移 −2 栅格', range_shift_minus1='距离轴移 −1 栅格',
                         range_shift_plus1='距离轴移 +1 栅格', range_shift_plus2='距离轴移 +2 栅格',
                         oracle_true_bearing='GT 真方位（仅诊断）')
            for name in REQUIRED_ARMS:
                arm = arms[name]; raw = arm['raw_model']
                displacement = arm.get('paired_response', {}).get('displacement_m', {}).get('mean')
                lines.append(f'| {names[name]} | {_format(raw.get("member_error", {}).get("rmse"))} | '
                    f'{_format(raw.get("success_ratio_5m"), True)} | {_format(displacement)} | '
                    f'{arm.get("normally_publishable_primary_count", 0)}/{raw.get("frames", 0)} |')
            lines += ['', '置零和打乱是冻结模型的敏感性探针；全零谱会被原融合流程的雷达信号门限拒绝。'
                '表中的原始输出仅供诊断，不能冒充正常部署结果，也不能替代重新训练的视觉控制组。'
                'GT 真方位使用了标签，只能用于判断检测方位误差的影响。', '']
            gains = []
            for name in ('range_shift_minus2', 'range_shift_minus1', 'range_shift_plus1', 'range_shift_plus2'):
                gain = arms[name].get('paired_response', {}).get('mean_frame_radial_response_gain')
                if _number(gain):
                    gains.append(f'{names[name]}：{gain:.4f}')
            if gains:
                lines += ['基站径向位移/谱距离移动的逐帧平均响应增益：' + '；'.join(gains) + '。'
                    '全谱平移也移动观察机谱，但观察机导航保持固定，因此属于几何不一致的分布外输入。'
                    '小幅响应不能单独证明模型未使用雷达或未学习距离；编码器保留了各距离行及其物理坐标。', '']
        invalid_frozen = [check['name'] for check in frozen.get('checks', []) if not check['passed']]
        if invalid_frozen:
            lines += ['固定权重诊断核验未通过：' + '、'.join(invalid_frozen) + '。', '']
        lines += ['成员覆盖率以全部真实成员为分母，漏检和未定位计失败；RMSE/P95 只统计已匹配成员，'
            '额外检测不包含在该误差均值中。群中心 RMSE 只统计完整匹配帧。'
            '从 32 个候选中借助真值挑最好的覆盖率属于 oracle 诊断，不能作为主方案实际精度。', '']
    if result['read_errors']:
        lines += ['部分记录暂未读到有效 JSON，本次报告保留 pending；下次更新会重试：', '']
        lines += ['- ' + item['path'] + '：' + item['error'] for item in result['read_errors']]
    lines += ['', '机器统计：comparison.json。原融合训练、权重及传感器缓存均不由本报告修改。', '']
    return '\n'.join(lines)


def write_report(output_root=DEFAULT_OUTPUT, **kwargs):
    result = build_comparison(output_root, **kwargs)
    output_root = Path(output_root)
    _atomic(output_root / 'comparison.json', json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    _atomic(output_root / 'report.md', render_markdown(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--fusion-root', type=Path, default=PROJECT / 'results/training/single')
    parser.add_argument('--frozen-root', type=Path)
    parser.add_argument('--baseline-root', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--watch', action='store_true')
    mode.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=float, default=15.)
    args = parser.parse_args(argv)
    if not 0 < args.interval <= 30:
        parser.error('--interval must be greater than zero and at most 30 seconds')
    while True:
        result = write_report(args.output_root, fusion_root=args.fusion_root, frozen_root=args.frozen_root,
                              baseline_root=args.baseline_root)
        baseline = result.get('visual_baseline', {})
        print(json.dumps(dict(status=result['status'], baseline_step=baseline.get('completed_steps'),
                              report=str(args.output_root / 'report.md')), ensure_ascii=False), flush=True)
        if not args.watch or result['status'] in ('complete', 'failed', 'invalid'):
            return int(result['status'] in ('failed', 'invalid'))
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
