"""Read-only synchronized observations and explicitly separate supervision."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from PIL import Image
from .types import Camera, Observations, ObserverPose, RadarConfig, Targets, Calibration, ObserverReference


MIXED_ECHO_SCHEMAS = frozenset((
    'group_plus_observer_reused_noise_channel_v1',
    'group_plus_observer_direct_channel_v2',
))


def _mixed_echo(metadata):
    schema = metadata.get('schema_version', '')
    if isinstance(schema, str) and schema.startswith('group_plus_observer_') and schema not in MIXED_ECHO_SCHEMAS:
        raise ValueError('Unsupported mixed echo schema: ' + schema)
    return schema in MIXED_ECHO_SCHEMAS


def _capture_modalities(metadata):
    """Accept depth-free v8 only when the source explicitly freezes that contract."""
    keys=('capture_modalities','depth_saved','depth_required')
    is_v8=metadata.get('schema_version')=='city10_group_clear_approach_v8_frame_png'
    if not is_v8 and not any(key in metadata for key in keys):return ('rgb','depth','segmentation')
    if (is_v8
            and metadata.get('capture_modalities')==['rgb','segmentation']
            and metadata.get('depth_saved') is False and metadata.get('depth_required') is False):
        if any(key.startswith('depth_') and key not in ('depth_saved','depth_required') for key in metadata):
            raise ValueError('Depth-free capture contains contradictory depth encoding/storage declarations')
        return ('rgb','segmentation')
    raise ValueError('Unsupported or inconsistent explicit capture modalities/depth contract')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):digest.update(block)
    return digest.hexdigest()


def quaternion_rotation(q):
    q = np.asarray(q, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError('Invalid camera quaternion')
    w,x,y,z = q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def _json(path):
    return json.loads(Path(path).read_text())


def _safe_file(root, value):
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Source path escapes episode folder')
    return path


def split_episodes(episodes, seed=20261114, fractions=(.8,.1,.1), groups=None):
    """Hash original trajectory groups, never frames, into deterministic splits."""
    if len(fractions) != 3 or min(fractions) < 0 or not np.isclose(sum(fractions),1):
        raise ValueError('Expected three nonnegative fractions summing to one')
    result = dict(train=[],val=[],test=[])
    for episode in sorted(episodes):
        group = str(groups[episode] if groups is not None else episode)
        number = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16],16)/2**64
        name = 'train' if number < fractions[0] else 'val' if number < sum(fractions[:2]) else 'test'
        result[name].append(episode)
    return result


class Dataset:
    """Only observations() belongs in inference; targets() is supervision only.

    The normal observation allowlist excludes counts, IDs, target geometry,
    noise_rms, scattering amplitudes, planned actions, and metric depth.
    """
    def __init__(self, root, verify_sources=True, echo_subdir='echoes'):
        self.root = Path(root).expanduser().resolve()
        self.verify_sources = bool(verify_sources)
        if echo_subdir not in ('echoes','echoes_with_observer'):raise ValueError('Explicit known echo_subdir required')
        self.echo_subdir=echo_subdir
        data = {p.parent.name for p in (self.root/'data').glob('episode_*/metadata.json')}
        echoes = {p.parent.parent.name for p in (self.root/'data').glob('episode_*/'+echo_subdir+'/metadata.json')}
        legacy = {p.parent.name for p in (self.root/'echoes').glob('episode_*/metadata.json')}
        if legacy & echoes:raise ValueError('Ambiguous duplicate legacy and nested echoes for '+str(sorted(legacy & echoes)))
        if legacy and not echoes:raise ValueError('RVWM requires data/episode_*/echoes; legacy flat echoes are unsupported')
        self.episodes = sorted(data & echoes)
        self._loaded = {}

    def _load(self, episode):
        if re.fullmatch(r'episode_[0-9]{6,}', str(episode)) is None:raise ValueError('Invalid episode ID')
        if episode not in self.episodes:raise FileNotFoundError(f'No aligned data/echoes: {episode}')
        if episode in self._loaded:return self._loaded[episode]
        droot = self.root/'data'/episode
        eroot = droot/self.echo_subdir
        dm,em = _json(droot/'metadata.json'),_json(eroot/'metadata.json')
        if dm.get('status')!='complete' or em.get('status')!='complete':raise ValueError('Incomplete source episode')
        draw = (droot/'frames.jsonl').read_bytes()
        drows = [json.loads(line) for line in draw.splitlines() if line.strip()]
        eraw = (eroot/'frames.jsonl').read_bytes()
        erows = [json.loads(line) for line in eraw.splitlines() if line.strip()]
        if (not drows or len(drows)!=len(erows) or len(drows)!=dm.get('frame_count')
                or len(erows)!=em.get('frame_count')):raise ValueError('Source frame count mismatch')
        expected = list(range(len(drows)))
        if [r['frame_idx'] for r in drows]!=expected or [r['frame_idx'] for r in erows]!=expected:
            raise ValueError('Source frame indices must be contiguous from zero')
        provenance = em.get('provenance',{})
        mixed=_mixed_echo(em)
        capture_hash=provenance.get('source_capture_frames_sha256') if mixed else provenance.get('source_sha256')
        metadata_hash=provenance.get('source_capture_metadata_sha256') if mixed else provenance.get('metadata_sha256')
        if self.echo_subdir=='echoes_with_observer' and not mixed:raise ValueError('Expected explicitly mixed observer IQ')
        if self.verify_sources:
            manifest_hash=em.get('frames_sha256')
            if (em.get('schema_version')=='group_plus_observer_direct_channel_v2' or manifest_hash is not None):
                if manifest_hash!=hashlib.sha256(eraw).hexdigest():
                    raise ValueError('Echo frame manifest SHA256 mismatch')
            if capture_hash!=hashlib.sha256(draw).hexdigest():
                raise ValueError('Echo source trajectory SHA256 mismatch')
            if metadata_hash!=sha256(droot/'metadata.json'):
                raise ValueError('Echo source metadata SHA256 mismatch')
        if em.get('iq_axes')!=['pulse','centered_subcarrier','array_y','array_z']:
            raise ValueError('Unsupported/missing IQ axis convention')
        if 'known transmit symbols removed' not in em.get('signal_kind',''):
            raise ValueError('Expected calibrated frequency-domain channel IQ')
        config = provenance.get('config',{})
        required = ['carrier_hz','bandwidth_hz','pulse_interval_s','frame_interval_s','iq_shape']
        if any(key not in config for key in required):raise ValueError('Missing frozen public radar parameters')
        if any(not math.isfinite(config[key]) or config[key]<=0 for key in required[:-1]):
            raise ValueError('Invalid frozen radar parameters')
        dt = float(config['frame_interval_s'])
        if not np.isclose(dt,1.,atol=1e-10):
            raise ValueError('This adapter requires physical frame interval 1 second; source differs')
        times = np.array([r['physical_time_s'] for r in erows],float)
        if not np.isfinite(times).all() or np.any(np.diff(times)<=0) or not np.allclose(np.diff(times),dt,atol=1e-8):
            raise ValueError('Physical frame timestamps disagree with frozen frame interval')
        modalities=_capture_modalities(dm)
        for data,echo in zip(drows,erows):
            if data.get('sim_timestamp_ns')!=echo.get('source_sim_timestamp_ns'):
                raise ValueError('RGB/IQ provenance timestamps disagree')
            camera=data['camera']
            if 'depth' not in modalities and ('depth_path' in data or 'depth_timestamp_ns' in camera):
                raise ValueError('Depth-free capture contains contradictory depth fields')
            stamps=[camera.get(f'{kind}_timestamp_ns') for kind in modalities]
            if any(s!=data['sim_timestamp_ns'] for s in stamps):raise ValueError('Camera modalities are not synchronized')
        public_config={key:config[key] for key in required}
        half_wave=299792458./(2*config['carrier_hz'])
        public_config.update(spacing_y_m=float(config.get('spacing_y_m',half_wave)),
                             spacing_z_m=float(config.get('spacing_z_m',half_wave)))
        public_config.update(capture_hash=capture_hash, metadata_hash=metadata_hash)
        value=(droot,eroot,dm,em,drows,erows,public_config)
        self._loaded[episode]=value
        return value

    def frames(self,episode):return len(self._load(episode)[4])

    def frames_in_phase(self, episode, phase, *, velocity_labels=False):
        """Return actual capture indices, including disjoint segments if present."""
        rows = self._load(episode)[4]
        if any(not isinstance(row.get('phase'), str) for row in rows):
            raise ValueError(f'Missing recorded phase in {episode}')
        indices = tuple(row['frame_idx'] for row in rows if row['phase'] == phase)
        if velocity_labels:
            # Outgoing-segment velocity uses the next frame; the terminal
            # episode frame uses the preceding segment.
            last = len(rows) - 1
            in_phase = set(indices)
            indices = tuple(i for i in indices if (i + 1 if i < last else i - 1) in in_phase)
        return indices

    def sample_track_frames(self, episode, count, *, velocity_labels=False):
        if count < 1:
            raise ValueError('Sample count must be positive')
        indices = self.frames_in_phase(episode, 'track', velocity_labels=velocity_labels)
        if not indices:
            raise ValueError(f'{episode} has no usable track frames')
        offsets = np.linspace(0, len(indices)-1, min(count, len(indices)), dtype=int)
        return tuple(indices[i] for i in offsets)

    @property
    def frame_interval_s(self):
        # Split/metadata inspection must not eagerly retain every sequence's
        # labels and frame records in memory before small-sample training.
        intervals={float(_json(self.root/'data'/e/self.echo_subdir/'metadata.json')['provenance']['config']['frame_interval_s'])
                   for e in self.episodes}
        if len(intervals)!=1:raise ValueError('Dataset contains missing or inconsistent physical frame intervals')
        return intervals.pop()

    def group_key(self,episode):
        # Noise/site variants of the same actual image trajectory share this
        # source digest. It is independent of the radar version and seed.
        if episode not in self.episodes:raise FileNotFoundError(episode)
        provenance=_json(self.root/'data'/episode/self.echo_subdir/'metadata.json')['provenance']
        return provenance.get('source_capture_frames_sha256',provenance.get('source_sha256'))

    def scene_family_id(self,episode):
        from .partitions import capture_family
        if episode not in self.episodes:raise FileNotFoundError(episode)
        return capture_family(_json(self.root/'data'/episode/'metadata.json'),episode)

    def splits(self,seed=20261114,fractions=(.8,.1,.1)):
        """Historical hash split, retained for archived runs and inner audits.

        Current configured pipelines use partitions.active_splits(dataset,
        config), whose explicit canonical manifest is shared with the detector.
        """
        from .partitions import FAMILY_POLICY
        if any(_json(self.root/'data'/e/'metadata.json').get('dataset_split_policy')==FAMILY_POLICY for e in self.episodes):
            raise ValueError('Scene-family sources require an explicit frozen 70/15/15 split_manifest')
        return split_episodes(self.episodes,seed,fractions,{e:self.group_key(e) for e in self.episodes})

    def observations(self,episode,frame,*,required_phase=None):
        droot,eroot,dm,em,drows,erows,config=self._load(episode)
        dt=float(config['frame_interval_s'])
        if not 0<=frame<len(drows):raise IndexError(frame)
        row,echo=drows[frame],erows[frame]
        allowed_phases = (required_phase,) if isinstance(required_phase,str) else tuple(required_phase or ())
        if required_phase is not None and row.get('phase') not in allowed_phases:
            raise ValueError(f'{episode}/{frame} is outside required phase {required_phase}')
        has_previous = frame > 0 and (required_phase is None or drows[frame-1].get('phase') in allowed_phases)
        image_path=_safe_file(droot,row['rgb_path']);iq_path=_safe_file(eroot,echo['iq_path'])
        if self.verify_sources and sha256(iq_path)!=echo.get('iq_sha256'):raise ValueError('IQ SHA256 mismatch')
        rgb=np.array(Image.open(image_path).convert('RGB'),copy=True)
        iq=np.load(iq_path,allow_pickle=False)
        shape=tuple(config['iq_shape'])
        if iq.shape!=shape or iq.dtype!=np.complex64 or not np.isfinite(iq).all():raise ValueError('Expected finite complex64 IQ tensor')
        width,height=map(int,dm['resolution'])
        if rgb.shape!=(height,width,3):raise ValueError('RGB dimensions disagree with calibration')
        # collect_group.capture_verified calls projection_check with the live
        # reported camera FOV. Configured nominal FOV is only a legacy fallback
        # when no reported calibration was recorded.
        reported=dm.get('reported_camera_fov_deg')
        fov=float(reported if reported is not None else dm['fov_deg'])
        if not 0<fov<180:raise ValueError('Invalid horizontal FOV')
        focal=width/(2*math.tan(math.radians(fov)/2))
        K=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1.]])
        cam=row['camera']
        camera=Camera(K,np.array(cam['camera_position_ned'],float),
            quaternion_rotation(cam['camera_orientation_wxyz']),width,height)
        # The optical centre is offset from the vehicle origin. Only the
        # measured observer state supplies a body pose; never infer it from
        # target labels, expert/reference actions, or the camera position.
        observer=None
        body=row.get('observer')
        if isinstance(body,dict) and 'position_ned' in body and 'orientation_wxyz' in body:
            position=np.asarray(body['position_ned'],float)
            if position.shape!=(3,) or not np.isfinite(position).all():
                raise ValueError('Invalid measured observer body position')
            observer=ObserverPose(position.copy(),quaternion_rotation(body['orientation_wxyz']))
        radar=RadarConfig(float(config['carrier_hz']),float(config['bandwidth_hz']),
            float(config['pulse_interval_s']),shape,config['spacing_y_m'],config['spacing_z_m'],float(config['frame_interval_s']))
        station=np.array(echo['station_position_ned_m'],float)
        heading=math.radians(float(echo['station_heading_deg']))
        c,s=math.cos(heading),math.sin(heading)
        basis=np.array([[c,-s,0],[s,c,0],[0,0,-1.]])
        hashes=dict(iq=echo.get('iq_sha256'),rgb=sha256(image_path),
                    source_trajectory=config['capture_hash'],source_metadata=config['metadata_hash'])
        policy=em.get('observer_policy',{})
        calibration=Calibration(K.copy(),np.zeros(5),np.asarray(dm.get('camera_offset_body_m',[]),float),radar,
            policy.get('motion_model','linear_radial_doppler'),policy.get('front_policy','forward_only'))
        reference=None
        if observer is not None:
            # The production echo velocity uses OUTGOING differences. Never
            # read it: it exposes the next frame. A causal navigation velocity
            # estimate is optional; the solver always refits its CPI velocity.
            velocity=None
            source='recorded simulator body origin; fixed position; unknown CPI velocity'
            if has_previous:
                previous=drows[frame-1].get('observer',{}).get('position_ned')
                if previous is not None:
                    velocity=(observer.position-np.asarray(previous,float))/dt
                    source='recorded simulator body origin; CPI velocity initialized by past-only navigation difference'
            reference=ObserverReference(observer.position.copy(),velocity,None,source,
                str(dm.get('observer_vehicle','observer')),np.zeros(3))
        timing=dict(physical_time_source='echo.physical_time_s',frame_interval_s=dt,
            pulse_interval_s=radar.pulse_interval_s,render_timestamp_ns=row['sim_timestamp_ns'],
            render_time_semantics='modality alignment only; not physical inter-frame elapsed time',
            render_delta_s=(row['sim_timestamp_ns']-drows[frame-1]['sim_timestamp_ns'])/1e9 if has_previous else None)
        timing['render_physical_interval_disagrees']=has_previous and not np.isclose(timing['render_delta_s'],dt,atol=.01)
        timing['observer_echo_in_iq']=_mixed_echo(em)
        return Observations(episode,int(frame),float(echo['physical_time_s']),rgb,iq,camera,radar,station,basis,hashes,observer,
                            calibration,reference,timing)

    def targets(self,episode,frame):
        _,_,metadata,_,drows,erows,_=self._load(episode)
        if not 0<=frame<len(drows):raise IndexError(frame)
        row,echo=drows[frame],erows[frame];group=row['group']
        ids=tuple(group['ids'])
        if tuple(echo['target_ids'])!=ids:raise ValueError('Supervision identity mismatch')
        p=np.asarray(group['positions_ned'],float)
        if not np.allclose(p,echo['target_positions_ned_m'],atol=1e-5):raise ValueError('RGB/radar supervision centre mismatch')
        # Invisible targets can have no segmentation box even when their
        # projected reference point is geometrically inside the image. Keep
        # label rows aligned with IDs; do not delete members from supervision.
        boxes=np.full((len(ids),4),np.nan)
        for i,box in enumerate(group['bboxes_xywh']):
            if box is not None and np.asarray(box).shape==(4,):boxes[i]=box
        boxes[:,2:]+=boxes[:,:2]
        keypoints=np.full((len(ids),2),np.nan)
        for i,uv in enumerate(group['projected_uv']):
            if uv is not None and np.asarray(uv).shape==(2,):keypoints[i]=uv
        minimum_pixels=metadata.get('min_target_visible_pixels',4)
        if type(minimum_pixels) is not int or minimum_pixels<4:
            raise ValueError('Invalid frozen minimum target visibility threshold')
        visible=(np.asarray(group['target_visible_pixels'])>=minimum_pixels)&np.isfinite(boxes).all(axis=1)&np.isfinite(keypoints).all(axis=1)
        return Targets(p.copy(),np.array(echo['target_velocities_ned_m_s'],float),ids,
            keypoints,boxes,visible)


def audit(root,episodes=None):
    dataset=root if isinstance(root,Dataset) else Dataset(root)
    rows=[]
    for episode in episodes or dataset.episodes:
        _,_,dm,em,dr,er,config=dataset._load(episode)
        rows.append(dict(episode_id=episode,frames=len(dr),physical_frame_interval_s=config['frame_interval_s'],
            pulse_interval_s=config['pulse_interval_s'],iq_axes=em['iq_axes'],iq_shape=config['iq_shape'],
            array_spacing_y_m=config['spacing_y_m'],array_spacing_z_m=config['spacing_z_m'],
            spacing_source='frozen explicit spacing, or verified radar_signal.RadarConfig half-wavelength default',
            nominal_hfov_deg=dm['fov_deg'],reported_hfov_deg=dm.get('reported_camera_fov_deg'),
            calibration_source=('recorded reported_camera_fov_deg, matching capture_verified projection_check'
                if dm.get('reported_camera_fov_deg') is not None else 'legacy nominal fov_deg fallback; reported calibration missing'),
            source_sha256=config['capture_hash'],timestamp_provenance_matched=True,
            group_key=dataset.group_key(episode)))
    return dict(episodes=rows,coordinate_convention='NED world; camera FRD; array forward/lateral/up',
                inference_inputs='RGB,IQ,public waveform and sensor poses only')
