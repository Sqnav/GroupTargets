"""Independent noiseless point-source checks for current FFT/query conventions.

This constructs diagnostic tones from their physical phase equation. It does
not call or modify dataset/echo production generators, read target labels, or
fit calibration to expected answers.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates

from rvwm.types import RadarConfig
from rvwm.coordinates import FixedStationFrame
from rvwm.group_observation import spectra_from_iq
from .volume import C, build_power_volume, positions_to_ruv, sample_power_volume_numpy

ROOT = Path(__file__).resolve().parents[1]


def diagnostic_point_iq(radar, points):
    """Independent unit-amplitude far-field points (R,u,v,complex amplitude)."""
    q, k, ny, nz = radar.shape
    # Finite, centered baseband frequencies, NOT an inclusive linspace.
    frequency = (np.arange(k, dtype=float)-k//2) * radar.bandwidth_hz/k
    iy = np.arange(ny, dtype=float)*radar.spacing_y_m
    iz = np.arange(nz, dtype=float)*radar.spacing_z_m
    signal = np.zeros((k, ny, nz), np.complex128)
    for radius, u, v, amplitude in points:
        if radius <= 0 or u*u+v*v >= 1:
            raise ValueError('Diagnostic point must be in the forward physical hemisphere')
        phase = (-4.*np.pi*frequency[:, None, None]*radius/C
                 + 2.*np.pi/radar.wavelength*(iy[None, :, None]*u+iz[None, None, :]*v))
        signal += complex(amplitude)*np.exp(1j*phase)
    return np.broadcast_to(signal, (q, k, ny, nz)).copy()


def _at(volume, radius, u, v):
    return sample_power_volume_numpy(volume['volume_power'], ranges=np.asarray(radius),
        u=np.asarray(u), v=np.asarray(v), range_axis=volume['range_m'],
        u_axis=volume['u_axis'], v_axis=volume['v_axis'])


def _power_at_rare(spectra, radius, azimuth, elevation):
    def query(power, angles, angle):
        r = np.interp(radius, spectra['range_m'], np.arange(len(spectra['range_m'])))
        a = np.interp(angle, angles, np.arange(len(angles)))
        return float(map_coordinates(power, np.array([[r], [a]]), order=1, prefilter=False)[0])
    return [query(spectra['ra_power'], spectra['azimuth_rad'], azimuth),
            query(spectra['re_power'], spectra['elevation_rad'], elevation)]


def projection_equivalence(iq, radar, volume, config):
    """Verify keeping the pre-max cube does not silently change RA/RE power."""
    reference = spectra_from_iq(iq, radar, config)
    az, el = reference['azimuth_rad'], reference['elevation_rad']
    u = np.cos(el)[None, :]*np.sin(az)[:, None]
    v = np.broadcast_to(np.sin(el)[None, :], u.shape)
    ra, re = [], []
    for index, radius in enumerate(reference['range_m']):
        sampled = _at(volume, np.full(u.shape, radius), u, v)['values'][..., 0]
        ra.append(sampled.max(1)); re.append(sampled.max(0))
    actual = [np.asarray(ra), np.asarray(re)]
    errors = {}
    for name, value, expected in zip(('ra_power', 're_power'), actual,
                                    (reference['ra_power'], reference['re_power'])):
        errors[name] = dict(max_abs=float(np.max(np.abs(value-expected))),
            relative_to_peak=float(np.max(np.abs(value-expected))/max(float(expected.max()), 1e-30)))
        if errors[name]['relative_to_peak'] > 2e-6:
            raise AssertionError('Joint FFT changed existing projection power: '+name)
    return errors, reference


def actual_observation_check(config_path, episode='episode_000009', index=0):
    """Compare the same joint transform against an authenticated real cache."""
    from rvwm.data import Dataset
    from depth_localization.data import load_observation_features
    config=json.loads(Path(config_path).read_text())
    dataset=Dataset(config['dataset_root'],echo_subdir=config['echo_subdir'],verify_sources=True)
    observation=dataset.observations(episode,index)
    volume=build_power_volume(observation.iq,observation.radar,config)
    errors,recomputed=projection_equivalence(observation.iq,observation.radar,volume,config)
    loaded=load_observation_features(config,episode,index,'val')
    features=loaded.get('features',loaded)
    cached={}
    for name in ('ra_power','re_power'):
        value=np.asarray(features[name]);expected=recomputed[name]
        difference=float(np.max(np.abs(value-expected)))
        relative=difference/max(float(expected.max()),1e-30)
        assert relative<2e-6,(name,difference,relative)
        cached[name]=dict(max_abs=difference,relative_to_peak=relative)
    return dict(status='passed',episode=episode,frame_idx=index,
        labels_loaded=False,input='Dataset.observations and authenticated sensor-only feature loader',
        source_hashes=observation.source_hashes,iq_shape=list(observation.iq.shape),
        joint_projection_vs_recomputed=errors,cached_ra_re_vs_recomputed=cached,
        range_bin_m=float(np.diff(volume['range_m'])[0]),
        array_basis_in_ned=observation.array_basis.tolist(),station_ned=observation.station_position.tolist())


def run(output, make_plot=True):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    wavelength = C/4.9e9
    radar = RadarConfig(4.9e9, 32e6, .0025, (40, 128, 16, 16), wavelength/2, wavelength/2)
    config = dict(ra_re_window='rectangular', ra_re_spatial_oversampling=4,
                  ra_re_azimuth_bins=65, ra_re_elevation_bins=65)
    dr = C/(2*radar.bandwidth_hz)
    cases = [(9., 0., 0.), (24., .25, .1875), (39., -.3125, .25),
             (57., .375, -.21875), (42.37, -.283, -.173), (63.22, .193, .341)]
    rows = []
    for bins, u, v in cases:
        radius = bins*dr
        iq = diagnostic_point_iq(radar, [(radius, u, v, 1.)])
        volume = build_power_volume(iq, radar, config)
        index = np.unravel_index(int(np.argmax(volume['volume_power'])), volume['volume_power'].shape)
        peak = [float(volume[name][i]) for name, i in zip(('range_m','u_axis','v_axis'), index)]
        query = _at(volume, radius, u, v)
        wrong_sign = _at(volume, radius, -u, -v)
        errors = np.abs(np.asarray(peak)-[radius, u, v])
        resolution = [dr, np.diff(volume['u_axis'])[0], np.diff(volume['v_axis'])[0]]
        assert np.all(errors <= np.asarray(resolution)*.50001), (peak, radius, u, v)
        assert bool(query['valid']) and float(query['values'][0]) > .05*float(volume['volume_power'].max())
        if abs(u)+abs(v) > .05:
            assert float(query['values'][0]) > 20*float(wrong_sign['values'][0])
        rows.append(dict(expected=dict(range_m=radius,u=u,v=v),peak=dict(zip(('range_m','u','v'),peak)),
            peak_absolute_error=dict(zip(('range_m','u','v'),errors.tolist())),
            query_power=float(query['values'][0]),wrong_sign_power=float(wrong_sign['values'][0]),
            peak_power=float(volume['volume_power'].max()), passed=True))
    # Equal-|elevation| points make the false az/el cross combinations exact
    # cross combinations in (u,v), avoiding a misleading independent-angle test.
    radius = 30*dr
    pair = [(-.25,-.25),(.25,.25)]
    iq = diagnostic_point_iq(radar, [(radius,u,v,1.) for u,v in pair])
    volume = build_power_volume(iq, radar, config)
    equivalence, spectra = projection_equivalence(iq, radar, volume, config)
    combinations = []
    for u, v, real in [(-.25,-.25,True),(.25,.25,True),(-.25,.25,False),(.25,-.25,False)]:
        azimuth = float(np.arctan2(u, np.sqrt(1-u*u-v*v)))
        elevation = float(np.arcsin(v))
        ra, re = _power_at_rare(spectra, radius, azimuth, elevation)
        combinations.append(dict(u=u,v=v,azimuth_deg=np.rad2deg(azimuth),
            elevation_deg=np.rad2deg(elevation),is_real_target=real,
            joint_power=float(_at(volume,radius,u,v)['values'][0]),
            ra_power=ra,re_power=re,paired_min_power=min(ra,re)))
    true = min(p['joint_power'] for p in combinations if p['is_real_target'])
    cross = max(p['joint_power'] for p in combinations if not p['is_real_target'])
    assert cross < true*1e-6
    paired_cross=min(p['paired_min_power'] for p in combinations if not p['is_real_target'])
    paired_true=max(p['paired_min_power'] for p in combinations if p['is_real_target'])
    assert paired_cross/paired_true>.9
    # Coordinate frame is an orthogonal handedness-changing basis, not SO(3).
    heading = np.deg2rad(37.);c,s=np.cos(heading),np.sin(heading)
    basis = np.array([[c,-s,0.],[s,c,0.],[0.,0.,-1.]])
    frame = FixedStationFrame(np.array([50.,-70.,-10.]),basis)
    local = np.array([[120.,40.,-30.],[70.,-20.,15.]])
    world = frame.origin_ned+local@basis.T
    recovered = frame.to_local_positions(world)
    assert np.allclose(recovered,local,rtol=0,atol=1e-12)
    coordinate = dict(station_ned=frame.origin_ned.tolist(),array_basis_to_ned=basis.tolist(),
        determinant=float(np.linalg.det(basis)),max_roundtrip_error_m=float(np.max(np.abs(recovered-local))),
        conversion='local=(ned-station_ned) @ array_basis_to_ned',passed=True)
    sources = [ROOT/'rvwm/group_observation.py',ROOT/'depth_localization/geometry.py',
               ROOT/'rvwm/coordinates.py',ROOT.parent/'dataset_construction/scripts/echo_acceleration.py',
               Path(__file__),Path(__file__).with_name('volume.py')]
    report = dict(status='passed',created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        scope='independent noiseless far-field point diagnostic; not dataset labels or localization accuracy',
        radar=dict(carrier_hz=radar.carrier_hz,bandwidth_hz=radar.bandwidth_hz,
            pulse_interval_s=radar.pulse_interval_s,shape=list(radar.shape),spacing_y_m=radar.spacing_y_m,
            spacing_z_m=radar.spacing_z_m,range_bin_m=dr,spatial_fft_shape=[64,64]),
        conventions=dict(generation_range_phase='-4*pi*f*R/c',generation_array_phase='+2*pi*(dy*iy*u+dz*iz*v)/lambda',
            frequency_grid='(arange(K)-K//2)*B/K, endpoint-exclusive',range_inverse='ifft(ifftshift(frequency),norm=ortho)',
            spatial_forward='fft2(y,z), then fftshift',native_spatial_axes='fftshift(fftfreq(M))*lambda/spacing',
            geometry='u=y/R=cos(el)*sin(az), v=z/R=sin(el), array x>0',
            query_grid_order='torch grid_sample: (v,u,R); align_corners=True; explicit periodic angular endpoint'),
        conclusion='No sign/axis/shift error found in current RA/RE calibration. Joint cube retains pairing lost by independent max projections.',
        single_point_cases=rows,projection_equivalence=equivalence,same_range_two_targets=combinations,
        false_cross_to_true_joint_power_ratio=cross/true,
        false_cross_to_true_projected_min_power_ratio=paired_cross/paired_true,
        coordinate_roundtrip=coordinate,
        actual_sensor_cache_check=actual_observation_check(ROOT/'depth_config.json'),
        limitations=['Far-field synthetic points test the established narrowband array convention; they do not prove inverse identifiability for moving extended scatterers or the near-field observer.'],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (output/'axis_diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    if make_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,3,figsize=(13,3.7),constrained_layout=True)
        plane=volume['volume_power'][30]
        ax[0].imshow(10*np.log10(np.maximum(plane/plane.max(),1e-6)).T,origin='lower',
            extent=[volume['u_axis'][0],volume['u_axis'][-1],volume['v_axis'][0],volume['v_axis'][-1]],
            vmin=-50,vmax=0,aspect='equal',cmap='viridis')
        for p in combinations:
            ax[0].scatter(p['u'],p['v'],marker='o' if p['is_real_target'] else 'x',
                          color='red' if p['is_real_target'] else 'white',s=70,linewidths=1.8)
        ax[0].set(xlim=(-.7,.7),ylim=(-.7,.7),xlabel='u = y / R',ylabel='v = z / R',title='Joint power: cross pairs rejected')
        for target,(name,axis,label) in zip(ax[1:],[('ra_power','azimuth_rad','Azimuth (deg)'),('re_power','elevation_rad','Elevation (deg)')]):
            power=spectra[name][30]
            target.plot(np.rad2deg(spectra[axis]),10*np.log10(np.maximum(power/power.max(),1e-6)))
            target.set(xlim=(-45,45),ylim=(-50,2),xlabel=label,ylabel='Relative power (dB)',title='Projection supports both angle choices')
            target.grid(alpha=.2)
        fig.savefig(output/'same_range_pairing.png',dpi=170);plt.close(fig)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/precision_study_20261122/axis_diagnostic')
    parser.add_argument('--no-plot',action='store_true')
    args=parser.parse_args(argv);report=run(args.output,not args.no_plot)
    print(json.dumps(dict(status=report['status'],cases=len(report['single_point_cases']),
                         cross_ratio=report['false_cross_to_true_joint_power_ratio'],output=str(args.output))))


if __name__=='__main__':main()
