"""Plot real unwindowed IQ spectra and explain the fixed-station frame adapter.

No target labels, detector peaks, smoothing or edits to trained sources/caches.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import fft
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rvwm.data import Dataset,sha256
from rvwm.coordinates import FixedStationFrame
from rvwm.group_observation import spectra_from_iq,feature_digest
from .data import ROOT,verify_original_sources


def edges(values):
    values=np.asarray(values,float)
    return np.r_[values[0]-(values[1]-values[0])/2,(values[:-1]+values[1:])/2,values[-1]+(values[-1]-values[-2])/2]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',default='episode_000009');parser.add_argument('--frame',type=int,default=0)
    parser.add_argument('--output',type=Path);args=parser.parse_args(argv)
    cfg=json.loads((ROOT/'config.json').read_text());verify_original_sources(cfg)
    dataset=Dataset(cfg['dataset_root'],echo_subdir=cfg['echo_subdir'],verify_sources=True)
    raw=dataset.observations(args.episode,args.frame,required_phase='approach')
    frame=FixedStationFrame.from_observation(raw);local=frame.observations(raw)
    before=spectra_from_iq(raw.iq,raw.radar,cfg);after=spectra_from_iq(local.iq,local.radar,cfg)
    inventory=json.loads((ROOT/'results/training/single/val_features.json').read_text())
    row=next(r for r in inventory['frames'] if r['episode_id']==args.episode and r['frame_idx']==args.frame)
    cache=Path(row['cache']['path'])
    if sha256(cache)!=row['cache']['sha256']:raise ValueError('Prepared feature cache changed')
    cached=torch.load(cache,map_location='cpu',weights_only=False)
    if feature_digest(cached['features'])!=row['feature_digest']:raise ValueError('Cached feature digest differs')
    features=cached['features']
    differences={name:dict(adapter_max_abs=float(np.max(np.abs(before[name]-after[name]))),
                           cache_max_abs=float(np.max(np.abs(after[name]-features[name])))) for name in ('ra_power','re_power')}
    if any(v['adapter_max_abs']!=0 or v['cache_max_abs']!=0 for v in differences.values()):
        raise ValueError('Recomputed spectra differ from coordinate adapter or actual training inputs')
    output=args.output or ROOT/'results/spectrum_example_20261121'/args.episode/f'frame_{args.frame:05d}'
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError('Use a fresh spectrum output directory')
    # Array-FFT projections are kept separate from physical-angle RA/RE. Both
    # reference the same station, but projection and angular interpolation do
    # not commute and are not merely alternative axis labels.
    q,k,ny,nz=raw.iq.shape;oversampling=int(cfg.get('ra_re_spatial_oversampling',4));my,mz=ny*oversampling,nz*oversampling
    profiles=fft.ifft(fft.ifftshift(raw.iq,axes=1),axis=1,norm='ortho',workers=1)
    native_power=np.empty((k,my,mz),np.float64)
    for start in range(0,k,4):
        beam=fft.fft2(profiles[:,start:start+4],s=(my,mz),axes=(-2,-1),workers=1)
        native_power[start:start+4]=fft.fftshift(np.mean(np.abs(beam)**2,axis=0,dtype=np.float64)/(ny*nz),axes=(-2,-1))
    uy=fft.fftshift(fft.fftfreq(my))*raw.radar.wavelength/raw.radar.spacing_y_m
    uz=fft.fftshift(fft.fftfreq(mz))*raw.radar.wavelength/raw.radar.spacing_z_m
    physical=uy[:,None]**2+uz[None,:]**2<1.
    native_y=np.where(physical[None],native_power,-np.inf).max(axis=2)
    native_z=np.where(physical[None],native_power,-np.inf).max(axis=1)
    native_y[~np.isfinite(native_y)]=np.nan;native_z[~np.isfinite(native_z)]=np.nan
    reference=max(float(np.nanmax(native_y)),float(np.nanmax(native_z)),float(after['ra_power'].max()),float(after['re_power'].max()))
    floor=-50.
    def db(power):return np.maximum(10*np.log10(np.maximum(np.asarray(power,float),reference*1e-14)/reference),floor)
    ranges=after['range_m'];az=np.rad2deg(after['azimuth_rad']);el=np.rad2deg(after['elevation_rad'])
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def panel(ax,power,angles,title,xlabel,limits=None):
        mesh=ax.pcolormesh(edges(angles),edges(ranges),db(power),cmap='magma',vmin=floor,vmax=0,shading='flat',rasterized=True)
        ax.set_title(title);ax.set_xlabel(xlabel);ax.set_ylabel('Range from station (m)')
        ax.set_xlim(angles[0],angles[-1]);ax.set_ylim(*(limits or (0,float(edges(ranges)[-1]))))
        return mesh
    def finish(fig,axes,mesh,name,title,note):
        fig.get_layout_engine().set(rect=(0,.055,1,.945))
        fig.suptitle(title,fontsize=14);fig.colorbar(mesh,ax=np.asarray(axes).ravel().tolist(),label='Power (dB, one shared display reference)',pad=.025,shrink=.86)
        fig.text(.5,.015,note,ha='center',fontsize=9);fig.savefig(output/name,dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    panel(axes[0],after['ra_power'],az,'RA: station range vs azimuth','Station azimuth (deg; +Y/right)')
    mesh=panel(axes[1],after['re_power'],el,'RE: station range vs elevation','Station elevation (deg; +Z/up)')
    finish(fig,axes,mesh,'ra_re_base_local.png',f'{args.episode} / frame {args.frame:05d} | Actual model RA/RE input',
        'Rectangular window (no Hann/Hamming). Mixed group + observer. Native 128 range bins; 65 physical-angle bins. No detection overlays.')
    # Display-only range zoom chosen from actual measured power, without GT.
    profile=np.maximum(after['ra_power'].max(1),after['re_power'].max(1));active=np.flatnonzero(profile>=profile.max()*1e-2)
    lo=max(0,int(active[0])-4);hi=min(k-1,int(active[-1])+4);limits=(float(edges(ranges)[lo]),float(edges(ranges)[hi+1]))
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    panel(axes[0],after['ra_power'],az,'RA: display-only range zoom','Station azimuth (deg)',limits)
    mesh=panel(axes[1],after['re_power'],el,'RE: display-only range zoom','Station elevation (deg)',limits)
    finish(fig,axes,mesh,'ra_re_base_local_zoom.png',f'{args.episode} / frame {args.frame:05d} | Same data, range-axis zoom',
        'Only plot limits changed; numeric arrays, angle resolution, windows and shared color scale are unchanged.')
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    for i,(label,values) in enumerate([('Before geometry adapter',before),('After NED -> base_local geometry adapter',after)]):
        panel(axes[i,0],values['ra_power'],az,label+' | RA','Station azimuth (deg)')
        mesh=panel(axes[i,1],values['re_power'],el,label+' | RE','Station elevation (deg)')
    finish(fig,axes,mesh,'ra_re_coordinate_adapter_comparison.png','Coordinate-adapter check: identical spectra before / after',
        'IQ already uses array axes. The adapter changes camera/navigation geometry, not IQ or RA/RE. Exact max absolute difference: 0.')
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
    panel(axes[0,0],native_y,uy,'Native array-FFT power projection','Direction cosine u_y')
    panel(axes[0,1],native_z,uz,'Native array-FFT power projection','Direction cosine u_z')
    panel(axes[1,0],after['ra_power'],az,'Actual model RA in station frame','Azimuth (deg)')
    mesh=panel(axes[1,1],after['re_power'],el,'Actual model RE in station frame','Elevation (deg)')
    finish(fig,axes,mesh,'fft_to_physical_angles.png','Array FFT -> physical-angle RA/RE (same station reference frame)',
        '16x16 array -> 64x64 padded FFT; no resolution gain. u_y=cos(el)sin(az), u_z=sin(el). Nonlinear resampling precedes angle projection.')
    np.savez_compressed(output/'spectra.npz',range_m=ranges,azimuth_rad=after['azimuth_rad'],elevation_rad=after['elevation_rad'],
        ra_power_before=before['ra_power'],re_power_before=before['re_power'],ra_power_base=after['ra_power'],re_power_base=after['re_power'],
        direction_cosine_y=uy,direction_cosine_z=uz,native_range_uy_power=native_y,native_range_uz_power=native_z)
    loaded=dataset._load(args.episode);iq_file=loaded[1]/loaded[5][args.frame]['iq_path']
    record=dict(episode_id=args.episode,frame_idx=args.frame,source_iq=str(iq_file),source_iq_sha256=sha256(iq_file),
        source_rgb_sha256=raw.source_hashes['rgb'],cached_features_sha256=sha256(cache),differences=differences,
        station_position_ned_m=raw.station_position.tolist(),array_basis_in_ned=raw.array_basis.tolist(),
        coordinate_transform='p_base=(p_NED-station_position_NED)@array_basis_in_NED',
        coordinate_note='RA/RE already reference the station array. The geometry adapter does not rotate or shift these spectra.',
        signal_metadata=after['metadata'],ground_truth_used=False,display=dict(shared_power_reference=reference,dynamic_range_db=50,
        smoothing=False,display_range_zoom_m=limits,native_direction_cosine_mask='u_y^2+u_z^2<1; forward physical directions'),
        source_sha256={'plot_script':sha256(__file__),'group_observation.py':sha256(ROOT/'rvwm/group_observation.py')})
    record['output_sha256']={p.name:sha256(p) for p in output.iterdir() if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),differences=differences,source_iq_sha256=record['source_iq_sha256']),ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
