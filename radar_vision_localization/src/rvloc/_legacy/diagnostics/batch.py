"""Durable CPU shard launcher; merge only a complete, authenticated population."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from rvwm.validation import write_json
from .data import ROOT
from .run import merge,source_receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    args=parser.parse_args(argv)
    if args.workers<1:parser.error('workers must be positive')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    status_path=output/'batch_status.json'
    if status_path.exists():raise FileExistsError('Existing diagnostic batch is protected')
    started=time.monotonic();processes=[];streams=[]
    receipt=dict(status='starting',workers=args.workers,source_sha256=source_receipt(),pids=[])
    write_json(status_path,receipt)
    env=os.environ.copy()
    env.update(PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    try:
        for rank in range(args.workers):
            stream=(output/f'worker_{rank}.log').open('x');streams.append(stream)
            command=[sys.executable,'-m','diagnostics.run','--config',str(args.config.resolve()),
                     '--output',str(output),'--rank',str(rank),'--world-size',str(args.workers),'--device','cpu']
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            processes.append(process);receipt['pids'].append(process.pid)
        receipt['status']='running';write_json(status_path,receipt)
        while any(process.poll() is None for process in processes):
            receipt.update(elapsed_s=time.monotonic()-started,exit_codes=[p.poll() for p in processes])
            write_json(status_path,receipt);time.sleep(3)
        codes=[p.returncode for p in processes]
        if any(codes):raise RuntimeError('Diagnostic shard failures: '+str(codes))
        result=merge(output)
        receipt.update(status='complete',frames=result['frames'],exit_codes=codes)
    except BaseException as error:
        receipt.update(status='failed',error=f'{type(error).__name__}: {error}')
        for process in processes:
            if process.poll() is None:process.terminate()
        raise
    finally:
        receipt['elapsed_s']=time.monotonic()-started;write_json(status_path,receipt)
        for stream in streams:stream.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
