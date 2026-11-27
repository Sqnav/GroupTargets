"""Chinese, read-only progress for the queued single-frame evidence ablations."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics
import time

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'results/d_evidence_20261124'
ARMS = ('sequence', 'aligned', 'roi')


def read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError): return {}


def duration(seconds):
    if seconds is None: return '待估算'
    value = max(0, int(seconds))
    return f'{value // 3600:02d}:{value // 60 % 60:02d}:{value % 60:02d}'


def snapshot(output):
    output = Path(output)
    return dict(state=read_json(output / 'run_state.json'), queue=read_json(output / 'queue_status.json'),
        resources=read_json(output / 'resource_status.json'), roi=read_json(output / 'roi/progress.json'),
        preparations={name: read_json(output / 'prepare' / name / 'progress.json') for name in ARMS},
        training={arm: read_json(output / 'training' / arm / 'progress.json') for arm in ARMS},
        best={arm: read_json(output / 'training' / arm / 'prior_validation_rmse_best.json') for arm in ARMS})


def training_eta(data):
    rows = data.get('training', {})
    rates = [r['steps_per_second'] for r in rows.values() if r.get('steps_per_second', 0) > 0]
    if not rates: return None
    fallback = statistics.median(rates)
    return sum(max(0, r.get('requested_steps', 20000) - r.get('step', r.get('optimizer_steps', 0))) /
        (r.get('steps_per_second') if r.get('steps_per_second', 0) > 0 else fallback)
        for r in (rows.get(arm, {}) for arm in ARMS))


def format_progress(data):
    state = data.get('state', {}); current = state.get('current_stage') or '等待'
    now = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
    lines = [f'[{now} 北京时间] 单帧D证据实验：{state.get("status", "尚未启动")} | 阶段：{current}']
    if state.get('started_unix'): lines.append('含排队累计耗时：' + duration(time.time() - state['started_unix']))
    queue = data.get('queue', {})
    if queue and queue.get('status') != 'ready':
        for parent in queue.get('dependencies', []):
            lines.append('前置 {}：{} | 阶段 {} | PID {}'.format(parent['name'], parent['status'], parent.get('stage'), parent['pid']))
        lines.append('等待已有BW和多帧任务完整结束；新实验尚未分配GPU。')
    resources = data.get('resources', {})
    if resources.get('status') == 'waiting':
        lines.append('等待GPU：' + json.dumps(resources.get('busy', []), ensure_ascii=False))
    if current.startswith('prepare_'):
        prep = data.get('preparations', {}).get(current.removeprefix('prepare_'), {})
        if prep:
            complete_frames = sum(item.get('frames', 0) for item in prep.get('inventories', {}).values())
            lines.append('特征准备：{}/{} | 已用 {}'.format(
                prep.get('completed_frames', prep.get('frames', prep.get('completed', complete_frames or '?'))),
                prep.get('total_frames', prep.get('total', complete_frames or '?')), duration(prep.get('elapsed_s'))))
    if current == 'roi_export':
        roi = data.get('roi', {})
        lines.append('冻结检测器ROI：{}/{} 帧 | 已用 {} | 剩余 {}'.format(
            roi.get('frames_complete', roi.get('completed_frames', roi.get('frames', '?'))),
            roi.get('frames_total', roi.get('total_frames', '?')),
            duration(roi.get('elapsed_s')), duration(roi.get('eta_s'))))
    trained = 0
    for arm in ARMS:
        row = data.get('training', {}).get(arm, {})
        step = row.get('step', row.get('optimizer_steps', 0)); trained += step
        if row:
            loss = row.get('loss'); loss = f'{loss:.6f}' if isinstance(loss, (int, float)) else '待更新'
            lines.append('{}：{}/{} 步 | Loss {} | 已用 {} | 剩余 {}'.format(
                arm, step, row.get('requested_steps', 20000), loss,
                duration(row.get('elapsed_s')), duration(row.get('eta_s'))))
            report = data.get('best', {}).get(arm, {}); summary = report.get('summary', {})
            rmse, one = summary.get('primary_member_rmse_m'), summary.get('primary_member_success_ratio_1m')
            if isinstance(rmse, (int, float)):
                text = f'{arm} 最佳验证：第 {report.get("step", "?")} 步 | RMSE {rmse:.4f} m'
                if isinstance(one, (int, float)): text += f' | 全GT≤1m {one:.2%}'
                lines.append(text)
    lines.append(f'三臂累计训练：{trained}/60000 步 | 全局batch48')
    estimate = training_eta(data)
    if estimate is not None:
        lines.append('剩余训练粗估：' + duration(estimate) + '（未启动臂按已测速率外推；不含排队、导出和评价）')
    if state.get('error'): lines.append('已停止：' + str(state['error']))
    if state.get('status') == 'complete': lines.append('三个独立单项训练及统一开发评价完成；生产默认D保持不变。')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--watch', action='store_true'); parser.add_argument('--interval', type=float, default=15.)
    parser.add_argument('--json', action='store_true'); args = parser.parse_args(argv)
    if args.interval < .1: parser.error('Polling interval must be >= 0.1 seconds')
    while True:
        data = snapshot(args.output)
        print(json.dumps(data, ensure_ascii=False) if args.json else format_progress(data), flush=True)
        if not args.watch or data['state'].get('status') in ('complete', 'failed', 'interrupted'): return 0
        time.sleep(args.interval)


if __name__ == '__main__': raise SystemExit(main())
