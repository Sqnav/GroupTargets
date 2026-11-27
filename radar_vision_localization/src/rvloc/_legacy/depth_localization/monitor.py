"""Read depth training progress without touching a running trainer."""
import argparse
import json
from pathlib import Path
import time
from .checkpoint import ROOT


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=ROOT/'results/training/depth_v2')
    p.add_argument('--once',action='store_true');p.add_argument('--interval',type=float,default=5.)
    a=p.parse_args(argv)
    if not 0<a.interval<=30:p.error('interval must be in (0,30] seconds')
    while True:
        path=a.run_dir/'progress.json'
        if path.exists():
            try:d=json.loads(path.read_text())
            except (OSError,ValueError):d={'stage':'publishing_progress'}
            keys=('stage','step','requested_steps','loss','initial_depth_huber','final_depth_huber','eta_s','error')
            print(json.dumps({k:d[k] for k in keys if k in d},ensure_ascii=False),flush=True)
            if a.once or d.get('stage') in ('complete','failed','dry_run_complete'):return int(d.get('stage')=='failed')
        elif a.once:print(json.dumps({'stage':'not_started','run_dir':str(a.run_dir)}));return 0
        else:print('Waiting for depth training progress: '+str(path),flush=True)
        time.sleep(a.interval)


if __name__=='__main__':raise SystemExit(main())
