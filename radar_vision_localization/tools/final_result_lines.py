"""Append authenticated final DEV results to the existing one-line tail."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from zoneinfo import ZoneInfo


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as s:
        for block in iter(lambda:s.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.writing-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as s:json.dump(value,s,ensure_ascii=False,indent=2,allow_nan=False);s.write('\n')
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def hms(value):
    if not isinstance(value,(int,float)) or not math.isfinite(value):return '未知'
    n=max(0,round(value));return f'{n//3600:02d}:{n//60%60:02d}:{n%60:02d}'


def require(ok,message):
    if not ok:raise ValueError(message)


class ResultLines:
    def __init__(self,run_dir,fixed_dir,output_dir):
        self.run=Path(run_dir).resolve();self.fixed=Path(fixed_dir).resolve();self.output=Path(output_dir).resolve()
        require(not self.output.is_relative_to(Path(__file__).resolve().parents[1]),'Results must stay outside code')
        self.output.mkdir(parents=True,exist_ok=True)
        self.main_identity=read(self.run/'execution_identity.json')
        self.identity=dict(schema='rvloc.final_result_lines_identity.v1',run_dir=str(self.run),
            fixed_dir=str(self.fixed),main_identity_sha256=sha(self.run/'execution_identity.json'),
            source_sha256=sha(__file__),training_changed=False)
        path=self.output/'result_lines_identity.json'
        if path.exists():require(read(path)==self.identity,'Final result forwarder identity changed')
        else:write(path,self.identity)
        path=self.output/'result_lines_receipt.json'
        self.emitted=read(path).get('emitted',{}) if path.exists() else {}
        for record in self.emitted.values():
            require(sha(record['receipt'])==record['receipt_sha256'] and sha(record['evaluation'])==record['evaluation_sha256'],
                    'Previously emitted DEV result changed')

    def result(self,method):
        is_fixed=method in (2,3,5,6,7)
        path=self.fixed/f'g{method}'/'receipt.json' if is_fixed else self.run/'stage_receipts'/f'dev_evaluate_g{method}.json'
        if not path.exists():return None
        receipt=read(path)
        require(receipt.get('status')=='completed','DEV receipt is not complete')
        if is_fixed:
            require(receipt.get('method_id')==method and receipt.get('split')=='dev','Fixed result method/split differs')
            evaluation=Path(receipt['evaluation']).resolve()
            require(evaluation.is_relative_to(self.fixed/f'g{method}'),'Fixed evaluation escaped result directory')
        else:
            require(receipt.get('stage')==dict(name=f'dev_evaluate_g{method}',kind='evaluate',method=method,split='dev')
                    and receipt.get('identity_sha256')==self.main_identity['sha256'],'Main DEV receipt belongs to another stage/run')
            candidates=[Path(p).resolve() for p in receipt['artifacts'] if Path(p).name=='evaluation.json']
            require(len(candidates)==1,'DEV receipt needs exactly one evaluation artifact')
            evaluation=candidates[0];require(evaluation.is_relative_to(self.run),'Main evaluation escaped run')
        require(str(evaluation) in receipt['artifacts'] and sha(evaluation)==receipt['artifacts'][str(evaluation)],'DEV evaluation artifact SHA differs')
        value=read(evaluation)
        require(value.get('schema')==('rvloc.fixed_roi_evaluation.v1' if is_fixed else 'rvloc.evaluation.v1')
                and value.get('method_id')==method and value.get('split')=='dev'
                and value.get('manifest_sha256')==self.main_identity['manifest_sha256'],'Result is not this run\'s DEV evaluation')
        if is_fixed:
            require(value.get('frozen_index_sha256')==receipt.get('frozen_index_sha256')
                    and value.get('predictions_sha256')==receipt.get('predictions_sha256'),'Fixed evaluation input identity differs')
        rmse=value['summary'].get('position_rmse_3d_m')
        require(rmse is None or isinstance(rmse,(int,float)) and math.isfinite(rmse) and rmse>=0,'Invalid DEV RMSE')
        return dict(receipt=str(path),receipt_sha256=sha(path),evaluation=str(evaluation),evaluation_sha256=sha(evaluation),
                    metric='DEV_FIXED_RMSE' if is_fixed else 'DEV_SET_RMSE',rmse_m=rmse,elapsed_s=receipt.get('elapsed_s'))

    def tick(self):
        require(sha(self.run/'execution_identity.json')==self.identity['main_identity_sha256'] and sha(__file__)==self.identity['source_sha256'],'Forwarder identity changed')
        main=read(self.run/'status.json')
        require(main.get('identity_sha256')==self.main_identity['sha256'],'Main status identity differs')
        for method in range(1,8):
            key=str(method);value=self.result(method)
            if value is None:continue
            if key in self.emitted:
                require(value['receipt_sha256']==self.emitted[key]['receipt_sha256'],'Previously emitted receipt changed');continue
            number='unavailable' if value['rmse_m'] is None else f'{value["rmse_m"]:.4f}m'
            stamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
            line=f'[{stamp}] [RESULT] G{method} {value["metric"]}={number} 已完成 本次评估耗时={hms(value["elapsed_s"])} 总已用={hms(main.get("elapsed_s"))}'
            for dest in (self.output/'progress.log',self.run/'run.log'):
                with dest.open('a') as s:s.write(line+'\n')
            print(line,flush=True);self.emitted[key]=dict(value,line=line)
            write(self.output/'result_lines_receipt.json',dict(identity=self.identity,emitted=self.emitted))
        terminal=main.get('status') in ('completed','failed','blocked','cancelled','terminated')
        pending=[i for i in range(1,8) if str(i) not in self.emitted]
        if main.get('status')=='completed' and pending and all(i in (2,3,5,6,7) for i in pending):
            watcher=self.fixed/'watch_status.json'
            fixed_status=read(watcher).get('status','') if watcher.exists() else ''
            # The fixed evaluator is independent and may finish just after
            # the main report. Keep waiting for its committed receipts.
            if fixed_status in ('starting','running','waiting') or fixed_status.startswith('evaluating_g'):
                terminal=False
        state=dict(schema='rvloc.final_result_lines_status.v1',status='completed' if len(self.emitted)==7 else 'incomplete' if terminal else 'waiting',
            main_status=main.get('status'),pid=os.getpid(),emitted_methods=sorted(map(int,self.emitted)),pending_methods=[i for i in range(1,8) if str(i) not in self.emitted],
            updated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'))
        write(self.output/'result_lines_status.json',state);return state

    def run_forever(self,once=False,poll_seconds=30):
        require(isinstance(poll_seconds,(int,float)) and math.isfinite(poll_seconds) and poll_seconds>0,'Invalid poll seconds')
        with (self.output/'.result_lines.lock').open('a+') as lock:
            try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as e:raise ValueError('Another result forwarder owns this directory') from e
            while True:
                state=self.tick()
                if once or state['status']!='waiting':return state
                time.sleep(poll_seconds)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True);p.add_argument('--fixed-evaluation-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--once',action='store_true');p.add_argument('--poll-seconds',type=float,default=30)
    a=p.parse_args();value=ResultLines(a.run_dir,a.fixed_evaluation_dir,a.output_dir).run_forever(a.once,a.poll_seconds)
    print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)
    return 0 if value['status']!='incomplete' else 2


if __name__=='__main__':raise SystemExit(main())
