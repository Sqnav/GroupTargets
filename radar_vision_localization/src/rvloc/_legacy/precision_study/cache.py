"""New authenticated joint-power caches; original IQ and feature caches are read-only."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from rvwm.group_observation import spectra_from_iq, SPECTRUM_VERSION
from rvwm import group_observation as baseline_projection
from .volume import build_power_volume, C, VERSION
from .visual_data import VisualSensorReader

SCHEMA='joint_ruv_volume_cache_v1'
_READERS={};_HASHES={}


def _stamp(path):
    p=Path(path);s=p.stat()
    return dict(path=str(p.resolve()),size=s.st_size,mtime_ns=s.st_mtime_ns)


def _sha(path):
    identity=_stamp(path);key=tuple(identity.values())
    if key not in _HASHES:
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
        _HASHES[key]=h.hexdigest()
    return _HASHES[key]


def _reader(config,split):
    names=('dataset_root','echo_subdir','split_manifest','split_manifest_sha256',
           'visual_split_manifest','training_phases','inference_phases')
    key=(split,json.dumps({k:config.get(k) for k in names},sort_keys=True))
    if key not in _READERS:_READERS[key]=VisualSensorReader(config,split,load_detector=False)
    return _READERS[key]


def _paths(config,episode,index):
    root=Path(config['precision_volume_cache']).resolve()
    for forbidden in ('dataset_root','group_feature_cache'):
        if config.get(forbidden) and root.is_relative_to(Path(config[forbidden]).resolve()):
            raise ValueError('Joint cache must be separate from original sources and feature cache')
    folder=root/episode;stem=f'frame_{index:05d}'
    return folder/(stem+'.power.npy'),folder/(stem+'.spectra.npz'),folder/(stem+'.receipt.json')


def _context(config,episode,index,split):
    reader=_reader(config,split);sensor=reader.sensor_frame(episode,index);radar=sensor['radar']
    if sensor['phase'] not in config.get('training_phases',['approach']):raise ValueError('Frame outside requested single-frame phase')
    d=Path(config['dataset_root']).resolve()/'data'/episode;e=d/config.get('echo_subdir','echoes')
    stats={name:_stamp(path) for name,path in dict(iq=sensor['iq_path'],capture=d/'frames.jsonl',
        metadata=d/'metadata.json',echo_frames=e/'frames.jsonl',echo_metadata=e/'metadata.json').items()}
    preprocessing=dict(window=config.get('ra_re_window','rectangular'),
        spatial_oversampling=config.get('ra_re_spatial_oversampling',4),range_chunk=config.get('ra_re_range_chunk',4),
        azimuth_bins=config.get('ra_re_azimuth_bins',config.get('ra_re_angle_bins',65)),
        elevation_bins=config.get('ra_re_elevation_bins',config.get('ra_re_angle_bins',65)),
        angle_limit_deg=config.get('ra_re_angle_limit_deg',85.))
    calibration={k:getattr(radar,k) for k in ('carrier_hz','bandwidth_hz','pulse_interval_s','spacing_y_m','spacing_z_m')}
    calibration['shape']=list(radar.shape)
    identity=dict(schema=SCHEMA,episode_id=episode,frame_idx=index,split=split,
        split_manifest_sha256=reader.split_digest,volume_source_sha256=_sha(Path(__file__).with_name('volume.py')),
        projection_source_sha256=_sha(baseline_projection.__file__),
        preprocessing=preprocessing,calibration=calibration,source_stats=stats,
        source_iq_sha256=sensor['iq_sha256'],capture_sha256=sensor['capture_hash'],metadata_sha256=sensor['metadata_hash'])
    over=int(preprocessing['spatial_oversampling']);k,ny,nz=radar.shape[1:]
    axes=dict(range_m=(np.arange(k)*C/(2*radar.bandwidth_hz)).tolist(),
        u_axis=(np.fft.fftshift(np.fft.fftfreq(ny*over))*radar.wavelength/radar.spacing_y_m).tolist(),
        v_axis=(np.fft.fftshift(np.fft.fftfreq(nz*over))*radar.wavelength/radar.spacing_z_m).tolist())
    return sensor,identity,axes


def _verified_record(config,episode,index,split,verify):
    _,identity,axes=_context(config,episode,index,split);power,spectra,receipt=_paths(config,episode,index)
    record=json.loads(receipt.read_text())
    if any(record.get(k)!=v for k,v in identity.items()):raise ValueError('Joint cache source, config or calibration differs')
    if record.get('axes')!=axes:raise ValueError('Joint cache axes differ from public calibration')
    metadata=record.get('volume_metadata',{})
    if metadata.get('version')!=VERSION or any(metadata.get(k) is not False for k in ('uses_labels','uses_history','uses_future')):
        raise ValueError('Joint cache has no valid sensor-only metadata contract')
    for kind,path in (('power',power),('spectra',spectra)):
        if record.get(kind+'_stat')!=_stamp(path):raise ValueError('Joint cache '+kind+' identity changed')
        if verify and _sha(path)!=record[kind+'_sha256']:raise ValueError('Joint cache '+kind+' SHA256 mismatch')
    return record,axes,power,spectra,receipt


def load_volume(config,episode,index,split,verify=True):
    """Load verified mmap power without opening native IQ or target labels.

    Native IQ SHA is checked at construction. Runtime compares its frozen
    manifest SHA and source size/mtime, while hashing the volume once per file
    identity. Sensor manifests are authenticated by VisualSensorReader.
    """
    record,axes,power,spectra,receipt=_verified_record(config,episode,index,split,verify)
    value=np.load(power,mmap_mode='r',allow_pickle=False)
    shape=(len(axes['range_m']),len(axes['u_axis']),len(axes['v_axis']))
    if value.dtype!=np.float32 or value.shape!=shape:raise ValueError('Joint cache tensor shape/dtype differs')
    metadata=dict(record['volume_metadata'],source_iq_file_sha256=record['source_iq_sha256'],
        cache_power_sha256=record['power_sha256'],cache_receipt=str(receipt),runtime_native_iq_read=False,
        spectra_path=str(spectra),spectra_sha256=record['spectra_sha256'],
        source_integrity_policy='native SHA verified at build; frozen source manifest SHA and immutable size/mtime checked at load')
    return dict(volume_power=value,range_m=np.asarray(axes['range_m']),u_axis=np.asarray(axes['u_axis']),
        v_axis=np.asarray(axes['v_axis']),angular_valid=np.ones(shape[1:],bool),metadata=metadata)


def load_spectra(config,episode,index,split,verify=True):
    """Load original RA/RE, without opening IQ or reprojecting rounded joint power."""
    record,_,_,path,_=_verified_record(config,episode,index,split,verify)
    with np.load(path,allow_pickle=False) as stored:
        result={k:stored[k].copy() for k in ('ra_power','re_power','range_m','azimuth_rad','elevation_rad','ra_valid','re_valid')}
        result['metadata']=json.loads(str(stored['metadata_json'].item()))
    if result['metadata'].get('version')!=SPECTRUM_VERSION:raise ValueError('Cached RA/RE preprocessing version differs')
    if result['ra_power'].shape!=(len(result['range_m']),len(result['azimuth_rad'])) or result['re_power'].shape!=(len(result['range_m']),len(result['elevation_rad'])):
        raise ValueError('Cached RA/RE shape differs from axes')
    return result


def _write_json(path,value):
    descriptor,name=tempfile.mkstemp(prefix='.'+Path(path).name+'.',dir=Path(path).parent)
    try:
        with os.fdopen(descriptor,'w') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)


def build_cached_volume(config,episode,index,split):
    """Explicit builder; commit power atomically, then its provenance receipt."""
    sensor,identity,axes=_context(config,episode,index,split);power,spectra,receipt=_paths(config,episode,index)
    if receipt.exists():
        load_volume(config,episode,index,split,verify=True)
        return dict(reused=True,episode_id=episode,frame_idx=index,power_path=str(power))
    before=identity['source_stats']['iq']
    if _sha(sensor['iq_path'])!=sensor['iq_sha256']:raise ValueError('Native IQ differs from frozen frame SHA256')
    iq=np.load(sensor['iq_path'],allow_pickle=False)
    volume=build_power_volume(iq,sensor['radar'],config)
    projections=spectra_from_iq(iq,sensor['radar'],config)
    if _stamp(sensor['iq_path'])!=before:raise ValueError('Native IQ changed while constructing cache')
    if _context(config,episode,index,split)[1]!=identity:raise ValueError('Sensor sources changed while constructing cache')
    if any(not np.array_equal(volume[k],axes[k]) for k in axes):raise ValueError('Power builder calibration axes differ')
    power.parent.mkdir(parents=True,exist_ok=True)
    descriptor,name=tempfile.mkstemp(prefix='.'+power.name+'.',dir=power.parent)
    try:
        with os.fdopen(descriptor,'wb') as stream:np.save(stream,volume['volume_power'],allow_pickle=False)
        os.replace(name,power)
    finally:
        if os.path.exists(name):os.unlink(name)
    descriptor,name=tempfile.mkstemp(prefix='.'+spectra.name+'.',dir=power.parent)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            np.savez_compressed(stream,**{k:v for k,v in projections.items() if k!='metadata'},
                metadata_json=np.asarray(json.dumps(projections['metadata'],sort_keys=True)))
        os.replace(name,spectra)
    finally:
        if os.path.exists(name):os.unlink(name)
    record=dict(identity,axes=axes,power_stat=_stamp(power),power_sha256=_sha(power),
        spectra_stat=_stamp(spectra),spectra_sha256=_sha(spectra),
        volume_metadata=volume['metadata'],created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        uses_labels=False,uses_history=False,uses_future=False)
    _write_json(receipt,record)
    return dict(reused=False,episode_id=episode,frame_idx=index,power_path=str(power))


build_volume=build_cached_volume


def _worker_init():
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
    import torch
    torch.set_num_threads(1)


def _episode_worker(config,episode,split,count):
    indices=_reader(config,split).frame_indices(episode,count);reused=0;errors=[]
    for index in indices:
        try:reused+=int(build_cached_volume(config,episode,index,split)['reused'])
        except Exception as exc:errors.append(dict(frame_idx=index,error=f'{type(exc).__name__}: {exc}'))
    return dict(episode_id=episode,split=split,frames=len(indices),reused=reused,
                created=len(indices)-reused-len(errors),failed=len(errors),errors=errors)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val'])
    p.add_argument('--workers',type=int,default=12);p.add_argument('--max-episodes',type=int,default=0)
    p.add_argument('--frames-per-episode',type=int,default=0)
    a=p.parse_args(argv)
    if a.workers<1 or min(a.max_episodes,a.frames_per_episode)<0:p.error('Positive workers/nonnegative limits required')
    config=json.loads(a.config.read_text());root=Path(config['precision_volume_cache']).resolve();root.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for split in dict.fromkeys(a.splits):
        ids=sorted(_reader(config,split).ids);ids=ids[:a.max_episodes] if a.max_episodes else ids
        jobs.extend((episode,split) for episode in ids)
    started=time.perf_counter();rows=[]
    summary=dict(status='building',config=str(a.config.resolve()),splits=a.splits,workers=a.workers,
        selected_episodes=len(jobs),completed_episodes=0,frames=0,created=0,reused=0,failed=0,
        uses_labels=False,test_set_read='test' in a.splits,diagnostic_subset=bool(a.max_episodes or a.frames_per_episode),errors=[])
    _write_json(root/'build_progress.json',summary)
    _worker_init()
    with ProcessPoolExecutor(max_workers=a.workers,initializer=_worker_init) as pool:
        tasks={pool.submit(_episode_worker,config,e,s,a.frames_per_episode):(e,s) for e,s in jobs}
        for future in as_completed(tasks):
            e,s=tasks[future]
            try:row=future.result()
            except Exception as exc:row=dict(episode_id=e,split=s,frames=0,created=0,reused=0,failed=1,errors=[dict(error=f'{type(exc).__name__}: {exc}')])
            rows.append(row);summary['completed_episodes']=len(rows)
            for k in ('frames','created','reused','failed'):summary[k]+=row[k]
            if row['errors']:summary['errors'].append(row)
            summary['elapsed_s']=time.perf_counter()-started
            _write_json(root/'build_progress.json',summary)
            print(json.dumps({k:summary[k] for k in ('completed_episodes','selected_episodes','frames','created','reused','failed','elapsed_s')}),flush=True)
    summary.update(status='failed' if summary['failed'] else 'complete',episodes=rows)
    _write_json(root/'build_summary.json',summary);_write_json(root/'build_progress.json',summary)
    return int(bool(summary['failed']))


if __name__=='__main__':raise SystemExit(main())
