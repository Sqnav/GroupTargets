"""Display saved multi-GPU preparation progress and training losses."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
import json
import math
import shutil
from pathlib import Path
import sys
import tempfile
import time


TERMINAL_STAGES = frozenset(('complete', 'failed', 'interrupted'))


def read_json(path):
    """A missing or incomplete atomic status publication is retried next poll."""
    try:
        value = json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def read_status(directory):
    directory = Path(directory)
    queue = read_json(directory / 'queue.json')
    root = directory
    if queue and queue.get('schema') in ('ra_re_single_frame_training_v1', 'ra_re_single_temporal_training_pair_v1'):
        active = queue.get('active_run_dir')
        if active:
            candidate = Path(active).resolve()
            if candidate.parent != root.resolve():
                raise ValueError('Training queue points outside its output root')
            directory = candidate
    progress = read_json(directory / 'progress.json') or {}
    ranks = [value for path in sorted((directory / 'ranks').glob('rank_*.json'))
             if (value := read_json(path)) is not None]
    failed = next((rank for rank in ranks if rank.get('stage') == 'failed'), None)
    if failed is not None and progress.get('stage') not in TERMINAL_STAGES:
        progress = dict(progress, stage='failed', error=failed.get('error', 'worker failed'),
                        failed_rank=failed.get('rank'), updated_unix=failed.get('updated_unix'))
    if queue and queue.get('schema') in ('ra_re_single_frame_training_v1', 'ra_re_single_temporal_training_pair_v1'):
        progress = dict(progress, active_variant=queue.get('active_variant'),
                        pair_active_job=queue.get('active_job'), pair_total_jobs=queue.get('total_jobs'),
                        pair_completed_jobs=queue.get('completed_jobs'), pair_status=queue.get('status'),
                        active_run_dir=str(directory), pair_started_unix=queue.get('started_unix'))
        if queue.get('status') in TERMINAL_STAGES:
            progress.update(stage=queue['status'], updated_unix=queue.get('updated_unix'))
            if queue.get('error'):
                progress['error'] = queue['error']
        elif progress.get('stage') in TERMINAL_STAGES:
            # Retain historical multi-job routing while the single job awaits
            # its launcher's final atomic completion publication.
            stage = ('waiting_next_variant' if queue.get('total_jobs', 1) > 1 else 'finalizing')
            progress.update(stage=stage, eta_s=None)
    return progress, ranks


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _number(value, precision=4):
    number = _finite(value)
    return f'{number:.{precision}f}' if number is not None else '—'


def _duration(value):
    seconds = _finite(value)
    if seconds is None or seconds < 0:
        return '—'
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def _phase(progress, ranks):
    stage = progress.get('stage', 'waiting')
    if progress.get('model_kind')=='group_scene':return stage
    if stage in ('preparing', 'initializing') and ranks:
        frames = sum(_finite(row.get('prepared_frames')) or 0 for row in ranks)
        total = sum(_finite(row.get('expected_frames')) or 0 for row in ranks)
        if total and frames >= total:
            return 'preparing_transitions'
        return 'preparing'
    return stage


def _completion(progress, ranks, stage):
    if stage in ('preparing', 'preparing_transitions'):
        done_key, total_key = ('prepared_frames', 'expected_frames') if stage == 'preparing' else ('prepared_transitions', 'expected_transitions')
        done = sum(_finite(row.get(done_key)) or 0 for row in ranks)
        total = sum(_finite(row.get(total_key)) or 0 for row in ranks)
    else:
        done = _finite(progress.get('step')) or 0
        total = _finite(progress.get('requested_steps')) or 0
    if stage == 'complete':
        return 1., done, total
    # Validation has its own duration and no per-item counter yet.
    if stage == 'validating' or total <= 0:
        return None, done, total
    return max(0., min(1., done / total)), done, total


class ProgressTimer:
    """Estimate the current preparation stage from observed per-worker rates."""
    def __init__(self):
        self.stage = None
        self.samples = deque()

    def eta(self, stage, progress, ranks, now):
        if stage not in ('preparing', 'preparing_transitions'):
            self.samples.clear()
            self.stage = stage
            return _finite(progress.get('eta_s')) if stage == 'training' else 0. if stage == 'complete' else None
        done_key, total_key = ('prepared_frames', 'expected_frames') if stage == 'preparing' else ('prepared_transitions', 'expected_transitions')
        counters = {row['rank']: (_finite(row.get(done_key)) or 0,
                                 _finite(row.get(total_key)) or 0) for row in ranks if 'rank' in row}
        if self.stage != stage:
            self.stage = stage
            self.samples.clear()
        self.samples.append((now, counters))
        while len(self.samples) > 2 and self.samples[1][0] < now - 60:
            self.samples.popleft()
        started, previous = self.samples[0]
        if now - started < 2 or not counters:
            return None
        estimates = []
        for rank, (done, total) in counters.items():
            remaining = max(0., total - done)
            if not remaining:
                continue
            before = previous.get(rank)
            if before is None or done <= before[0]:
                return None
            estimates.append(remaining * (now - started) / (done - before[0]))
        return max(estimates, default=0.)


def _run_start(directory, progress):
    # The launcher creates its lock at the start of the actual background job.
    for name in ('launcher.lock', 'configuration.json'):
        try:
            return (Path(directory) / name).stat().st_mtime
        except FileNotFoundError:
            pass
    return _finite(progress.get('started_unix'))


def _training_progress(progress):
    done = max(0., _finite(progress.get('step')) or 0.)
    total = _finite(progress.get('requested_steps')) or 0.
    ratio = min(1., done / total) if total > 0 else 0.
    return done, total, ratio


def format_status(progress, ranks, *, elapsed_s=None, eta_s=None, width=100):
    """Plain-text view with optimizer progress permanently visible."""
    stage = _phase(progress, ranks)
    done, total, ratio = _training_progress(progress)
    step_text = f'{int(done):,} / {int(total):,}' if total else f'{int(done):,} / —'
    lines = [f'训练步数   {step_text}    {ratio * 100:.1f}%']
    bar_width = max(8, min(56, width - 8))
    filled = int(bar_width * ratio)
    lines.append('━' * filled + '─' * (bar_width - filled))
    loss = 'Loss ' + _number(progress.get('loss'))
    components = (f'提案 {_number(progress.get("proposal_loss"))}   '
                  f'转移 {_number(progress.get("transition_loss"))}   '
                  f'物理 {_number(progress.get("physical_training_loss"))}')
    if 'shape_loss' in progress:
        physical=progress.get('physical_training_loss')
        # Group training evaluates native IQ only at configured physical
        # updates. A zero contribution on other steps is not a zero residual.
        if physical==0.:physical=None
        components = (f'中心 {_number(progress.get("center_nll"))}   '
                      f'结构 {_number(progress.get("shape_loss"))}   '
                      f'类别 {_number(progress.get("formation_loss"))}   '
                      f'IQ {_number(physical)}')
    if 'depth_nll' in progress:
        physical=progress.get('physical_training_loss')
        if physical==0.:physical=None
        components=(f'深度NLL {_number(progress.get("depth_nll"))}   '
                    f'射线深度 {_number(progress.get("ray_depth_loss"))}   '
                    f'类别 {_number(progress.get("formation_loss"))}   IQ {_number(physical)}')
    if progress.get('input_schema')=='base_ray_ra_re_v1':
        components=(f'深度NLL {_number(progress.get("depth_nll"))}   '
                    f'三维 {_number(progress.get("shape_loss"))}   '
                    f'KL {_number(progress.get("latent_kl"))}')
    lines.append(loss + ('    ' + components if width >= 100 and progress.get('loss') is not None else ''))
    elapsed = elapsed_s if elapsed_s is not None else progress.get('elapsed_s')
    if eta_s is None and stage == 'training':
        eta_s = _finite(progress.get('eta_s'))
    if stage == 'complete':
        eta_s = 0.
    remaining = '≈' + _duration(eta_s) if eta_s is not None else '估算中'
    lines.append(f'已用 {_duration(elapsed)}    本阶段剩余 {remaining}')
    if progress.get('pair_total_jobs'):
        lines.append(f'版本 {progress.get("active_variant") or "待启动"} '
                     f'· 已完成 {progress.get("pair_completed_jobs", 0)}/{progress["pair_total_jobs"]} '
                     + ('· 各任务依次使用 6 卡' if progress['pair_total_jobs'] > 1 else '· 单帧模型使用 6 卡'))
    if stage in ('preparing', 'preparing_transitions'):
        fraction, _, _ = _completion(progress, ranks, stage)
        name = '准备输入特征' if stage == 'preparing' else '准备时序样本'
        if progress.get('model_kind')=='group_scene':name='准备观测与验证特征'
        percent = f'{fraction * 100:.1f}%' if fraction is not None else '估算中'
        lines.append(f'{name} {percent} · 尚未开始参数更新')
    elif stage == 'training':
        speed = _finite(progress.get('steps_per_second'))
        lines.append('同步训练中' + (f' · {speed:.2f} step/s' if speed is not None else ''))
    else:
        labels = dict(waiting='等待训练任务', initializing='初始化', validating='验证中',
                      waiting_next_variant='等待下一个版本', finalizing='保存完成状态', complete='训练完成', failed='训练失败', interrupted='训练中断')
        lines.append(labels.get(stage, stage))
    if width < 100 and progress.get('loss') is not None:
        lines.append(components)
    if progress.get('error'):
        lines.append(str(progress['error']).replace('\n', ' ')[:120])
    return '\n'.join(lines)


def _panel(progress, ranks, elapsed, eta, width):
    """Rich handles redraws, terminal resizing and restoring the cursor."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.text import Text
    lines = format_status(progress, ranks, elapsed_s=elapsed, eta_s=eta, width=max(32, width-4)).splitlines()
    _, _, ratio = _training_progress(progress)
    filled = int(len(lines[1]) * ratio)
    bar = Text(lines[1][:filled], style='cyan')
    bar.append(lines[1][filled:], style='bright_black')
    items = [Text(lines[0], style='bold'), bar, Text(lines[2], style='bold green'),
             Text(lines[3], style='dim')]
    items.extend(Text(line, style='dim') for line in lines[4:])
    return Panel(Group(*items), title='世界模型训练', title_align='left',
                 border_style='bright_black', padding=(0, 1))


def read_metrics(path):
    """Read complete loss entries while a writer may be appending a line."""
    records = []
    try:
        stream = Path(path).open()
    except FileNotFoundError:
        return records
    with stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (isinstance(row, dict) and _finite(row.get('step')) is not None
                    and _finite(row.get('loss')) is not None):
                records.append(row)
    return records


def plot_losses(directory):
    """Write a standalone figure using complete entries from metrics.jsonl."""
    directory = Path(directory)
    records = read_metrics(directory / 'metrics.jsonl')
    if not records:
        return False
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    ray=any('depth_nll' in row for row in records)
    group=any('shape_loss' in row for row in records)
    curves=(('loss','Total'),('center_nll','Centre'),('shape_loss','Shape'),
            ('formation_loss','Formation'),('physical_training_loss','IQ')) if group else (
            ('loss','Total'),('proposal_loss','Proposal'),('transition_loss','Transition'),('physical_training_loss','Physics'))
    if ray:curves=(('loss','Total'),('depth_nll','Depth NLL'),('ray_depth_loss','Ray depth'),('formation_loss','Formation'),('physical_training_loss','IQ'))
    if any(row.get('input_schema')=='base_ray_ra_re_v1' for row in records):
        curves=(('loss','Total'),('depth_nll','Depth NLL'),('shape_loss','3D position'),('latent_kl','Latent KL'))
    for key, label in curves:
        valid = [row for row in records if _finite(row.get(key)) is not None]
        if group and key=='physical_training_loss':
            valid=[];previous_updates=0
            for row in records:
                updates=row.get('physical_updates',0)
                if updates>previous_updates and _finite(row.get(key)) is not None:valid.append(row)
                previous_updates=updates
        if valid:
            ax.plot([row['step'] for row in valid], [row[key] for row in valid],
                    label=label, linewidth=1.2, alpha=.8)
    ax.set(xlabel='Optimizer step', ylabel='Loss', title='Distributed training loss')
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    with tempfile.NamedTemporaryFile(suffix='.png', dir=directory, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        fig.savefig(temporary, dpi=140)
        temporary.replace(directory / 'loss.png')
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--once', action='store_true', help='print the current snapshot and exit')
    parser.add_argument('--plot', action='store_true', help='also update loss.png when losses are available')
    parser.add_argument('--interval', type=float, default=2., help='seconds between status reads')
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error('--interval must be positive')
    previous = None
    plotted_at = 0.
    timer = ProgressTimer()
    previous_variant = None
    try:
        with ExitStack() as stack:
            live = None
            if sys.stdout.isatty():
                try:
                    from rich.live import Live
                    live = stack.enter_context(Live('', auto_refresh=False, transient=False))
                except ImportError:
                    pass
            while True:
                progress, ranks = read_status(args.run_dir)
                if progress.get('active_variant') != previous_variant:
                    timer = ProgressTimer()
                    plotted_at = 0.
                    previous_variant = progress.get('active_variant')
                now = time.time()
                stage = _phase(progress, ranks)
                active_dir = Path(progress.get('active_run_dir', args.run_dir))
                started = _run_start(active_dir, progress)
                ended = _finite(progress.get('updated_unix')) if progress.get('stage') in TERMINAL_STAGES else None
                elapsed = max(0., (ended or now) - started) if started is not None else progress.get('elapsed_s')
                eta = timer.eta(stage, progress, ranks, time.monotonic())
                width = shutil.get_terminal_size((100, 24)).columns
                signature = (progress.get('active_variant'), stage, progress.get('step'), progress.get('loss'),
                             tuple((r.get('prepared_frames'), r.get('prepared_transitions')) for r in ranks))
                if live is not None:
                    live.update(_panel(progress, ranks, elapsed, eta, width), refresh=True)
                elif signature != previous:
                    print(format_status(progress, ranks, elapsed_s=elapsed, eta_s=eta, width=width), flush=True)
                previous = signature
                terminal = progress.get('stage') in TERMINAL_STAGES
                if args.plot and (time.monotonic() - plotted_at >= 10 or terminal or args.once):
                    try:
                        plot_losses(active_dir)
                    except ImportError as error:
                        print(f'Loss figure unavailable: {error}', file=sys.stderr)
                        args.plot = False
                    plotted_at = time.monotonic()
                if args.once or terminal:
                    return int(progress.get('stage') in ('failed', 'interrupted'))
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print('已退出监视，训练继续运行。', flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
