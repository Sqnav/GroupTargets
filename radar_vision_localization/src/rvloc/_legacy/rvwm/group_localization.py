"""Base-frame rays + observer position + RA/RE -> continuous member positions.

The numeric-spectrum entrypoint never accepts raw IQ or labels. The RGB/IQ
adapter only constructs the five declared inputs. Untrained validation emits
explicit candidate diagnostics and null localization fields.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import numpy as np
from .coordinates import FixedStationFrame,spherical_from_local
from .group_observation import describe_observation,feature_digest,validate_features,SPECTRUM_VERSION
from .group_evaluation import shape_statistics
from .prior_validation import frame_seed
from .ray_scoring import select_scene_modes,observation_rejection_reason
from .types import ObservationWindow
from .single_frame import require_single_frame, validate_model_features


def describe_structure(positions,latent=None,model=None,config=None):
    return dict(shape_statistics(np.asarray(positions,float)),type='unknown',
        classification_status='continuous_geometry_only',scene_completeness='unknown',
        member_count_used=len(positions),scope='current_localized_detection_set',
        uncertainty_calibrated=False)


class GroupSceneLocalizer:
    def __init__(self,config,frontend,world_model,*,allow_untrained_validation=False):
        self.config=dict(config);self.frontend=frontend;self.model=world_model
        self.validation_only=bool(allow_untrained_validation)
        if not getattr(world_model,'trained',False) and not self.validation_only:
            raise RuntimeError('RA/RE inference requires trained compatible weights; use explicit untrained validation only for engineering checks')
        require_single_frame(config)
        if hasattr(world_model,'config'):require_single_frame(world_model.config)
        if int(config.get('group_refinement_steps',0))!=0:raise ValueError('RA/RE-only model has no IQ position refinement')
        if config.get('observer_handling','condition')!='condition':
            raise ValueError('RA/RE observer handling is condition; ignore/subtract/joint IQ backends are not this model')
        self._detections={};self.last_features=None

    def _detect(self,observation):
        key=(observation.episode_id,observation.frame_idx,observation.source_hashes.get('rgb') or hashlib.sha256(observation.rgb.tobytes()).hexdigest())
        if key not in self._detections:
            factor=float(self.config.get('pixel_covariance_multiplier',1.))
            if not np.isfinite(factor) or factor<=0:raise ValueError('Invalid pixel covariance scale')
            self._detections[key]=[replace(d,covariance=np.asarray(d.covariance)*factor) for d in self.frontend.detect(observation)]
        if len(self._detections)>10:self._detections.pop(next(iter(self._detections)))
        return self._detections[key]

    def _result(self,episode_id,frame_idx,timestamp_s,detections,n):
        targets=[]
        for i in range(n):
            detection=detections[i] if detections is not None else None
            targets.append(dict(detection_id=i,center_px=None if detection is None else np.asarray(detection.uv).tolist(),
                detection_score=None if detection is None else float(detection.score),position_base_m=None,
                range_m=None,azimuth_deg=None,elevation_deg=None,covariance_r_az_el=None,
                localization_status='unlocalized',alternatives=[]))
        return dict(schema='ra_re_member_localization_v1',episode_id=str(episode_id),frame_idx=int(frame_idx),
            timestamp_s=float(timestamp_s),coordinate_frame='base_local',angle_unit='deg',
            covariance_order=['range_m','azimuth_deg','elevation_deg'],detected_count=n,localized_count=0,
            localized_group_count=0,targets=targets,
            group=dict(center_base_m=None,range_m=None,azimuth_deg=None,elevation_deg=None,covariance_r_az_el=None,
                       localization_status='unlocalized',scope='current_visual_detection_set',scene_completeness='unknown'),
            structure=dict(type='unknown',classification_status='unavailable',scene_completeness='unknown'),
            alternatives=[],observer_diagnostics=dict(excluded_from_group=True),
            solver_diagnostics=dict(method='ra_re_conditioned_continuous_joint_depth',history_used=False,
                input_frames=1,
                output_scope='current_detection_set_only',persistent_object_ids=False,
                raw_iq_scoring=False,continuous_position_correction=False,
                world_model_loaded=bool(getattr(self.model,'trained',False)),validation_only=self.validation_only,
                evaluation_angle_gate_deg=float(self.config.get('group_supervision_max_angle_deg',3.)),
                uncertainty_calibrated=False),uncertainty_calibrated=False)

    @staticmethod
    def _reject(result,status):
        result['group']['localization_status']=status
        for target in result['targets']:target['localization_status']=status
        result['solver_diagnostics']['status']=status
        return result

    def estimate(self,observation_window):
        self.last_features=None
        if not isinstance(observation_window,ObservationWindow):raise TypeError('Only ObservationWindow accepted; labels are separate')
        require_single_frame(self.config)
        if len(observation_window.frames)!=1:
            raise ValueError('Only single-frame ObservationWindow inputs are supported')
        current=observation_window.current
        if self.frontend is None:raise ValueError('RGB adapter requires a frontend; use estimate_spectra for numeric inputs')
        detections=self._detect(current);n=len(detections)
        result=self._result(current.episode_id,current.frame_idx,current.timestamp,detections,n)
        result['solver_diagnostics']['supplied_frames']=len(observation_window.frames)
        if current.observer_reference is None:return self._reject(result,'missing_observer_reference')
        if current.observer_reference.from_iq:raise ValueError('Observer position must be an independent navigation reference')
        if not n:return self._reject(result,'no_detections')
        if current.calibration is not None and np.any(np.asarray(current.calibration.distortion)!=0):
            raise ValueError('Nonzero distortion requires explicit undistortion before rays')
        adapted=FixedStationFrame.from_observation(current).observations(current)
        features=describe_observation(adapted,detections,self.config)
        self.last_features=features
        result=self.estimate_spectra(features,episode_id=current.episode_id,frame_idx=current.frame_idx,
                                    timestamp_s=current.timestamp,detections=detections)
        result['solver_diagnostics']['supplied_frames']=len(observation_window.frames)
        result['observer_diagnostics']['reference_source']=current.observer_reference.source
        return result

    def estimate_spectra(self,features,*,episode_id,frame_idx,timestamp_s,detections=None):
        """Direct numeric interface: five inputs plus physical axes/masks; no IQ/RGB/labels."""
        validate_features(features)
        validate_model_features(features,self.config)
        for key,value in (('episode_id',str(episode_id)),('frame_idx',int(frame_idx)),('timestamp_s',float(timestamp_s))):
            if key in features['metadata'] and features['metadata'][key]!=value:
                raise ValueError('Spectrum provenance does not match requested '+key)
        n=len(features['ray_directions'])
        if detections is not None and len(detections)!=n:raise ValueError('Detection/ray cardinality differs')
        if not np.isfinite(timestamp_s):raise ValueError('Finite timestamp required')
        result=self._result(episode_id,frame_idx,timestamp_s,detections,n)
        result['observer_diagnostics'].update(position_base_m=np.asarray(features['observer_position']).tolist(),
            usage='shared geometric condition; observer stays in mixed spectra',complex_echo_known=False)
        diag=result['solver_diagnostics']
        diag.update(history_used=False,input_frames=1,configured_window=1,
            method='ra_re_conditioned_continuous_joint_depth')
        diag.update(feature_sha256=feature_digest(features),feature_metadata=features['metadata'],
            model_inputs=['ray_origins_base','ray_directions_base','observer_position_base','RA','RE'],
            geometry_metadata=['range_m','azimuth_rad','elevation_rad','valid_masks','analytic_ray_bounds'],
            radar_evidence='correlated RA/RE projections; not two independent likelihoods',
            radar_quality_gate='basic nonzero/nonflat validity only; not calibrated noise rejection',
            noise_rejection_calibrated=False,
            uncertainty_status='not_estimated_or_calibrated',member_labels_read=False,future_observations_read=False)
        required_preprocessing=dict(version=SPECTRUM_VERSION,
            projection='RA=max physical elevation; RE=max physical azimuth',
            range_transform='ifft(ifftshift(frequency), norm=ortho); native length',
            pulse_reduction='mean native-pulse beam power; sum unitary Doppler powers / Q by Parseval',
            normalization='linear power; no independent RA/RE peak normalization, no noise whitening')
        declared_matches=[]
        for sample in (features,):
            preprocessing=sample['metadata'].get('preprocessing',{})
            matched=all(preprocessing.get(k)==v for k,v in required_preprocessing.items())
            declared_matches.append(matched)
            if preprocessing and not matched:
                raise ValueError('RA/RE preprocessing differs from the current trained-input contract')
        diag['preprocessing_compatibility_verified']=all(declared_matches)
        diag['preprocessing_compatibility_scope']='declared preprocessing metadata; external arrays are not independently regenerated'
        rejection=observation_rejection_reason(features,self.config)
        if rejection is not None:return self._reject(result,rejection)
        seed=frame_seed(int(self.config.get('seed',20261114)),str(episode_id),int(frame_idx))
        hypotheses=self.model.propose(features,int(self.config.get('group_hypotheses',32)),np.random.default_rng(seed),
            allow_untrained_validation=self.validation_only)
        selected=select_scene_modes(features,hypotheses,self.config)
        diag.update(selected['diagnostics'],proposal_seed=seed,world_proposal_diagnostics=self.model.last_proposal_diagnostics)
        modes=selected['modes']
        if not modes:return self._reject(result,'no_valid_generated_scene')
        if self.validation_only:
            diag['validation_candidates']=[dict(mode_id=s['hypothesis_index'],member_positions_base_m=s['member_positions'].tolist(),
                depths_m=np.asarray(s['depths']).tolist(),valid_mask=s['valid_mask'].tolist(),
                mode_source=s.get('mode_source','unspecified')) for s in modes]
            return self._reject(result,'untrained_validation_only' if not getattr(self.model,'trained',False) else 'validation_only')
        primary=modes[0];valid=primary['valid_mask'];positions=primary['member_positions']
        for i,target in enumerate(result['targets']):
            if not valid[i]:target['localization_status']='invalid_ray_or_state';continue
            target.update(position_base_m=positions[i].tolist(),**spherical_from_local(positions[i:i+1])[0],
                          localization_status='estimated_uncalibrated')
            result['localized_count']+=1
        if valid.all():
            centre=positions.mean(0)
            result['group'].update(center_base_m=centre.tolist(),**spherical_from_local(centre[None])[0],
                                   localization_status='estimated_uncalibrated')
            result['localized_group_count']=1
        else:result['group']['localization_status']='partial_detection_localization'
        result['structure']=describe_structure(positions[valid])
        diag.update(status='estimated_uncalibrated',selected_mode_id=primary['hypothesis_index'],
                    selected_depths_m=np.asarray(primary['depths']).tolist(),
                    primary_prediction_kind=primary.get('mode_source','unspecified'))
        for scene in modes[1:]:
            v=scene['valid_mask'];p=scene['member_positions'];mode_id=scene['hypothesis_index']
            alternate=dict(mode_id=mode_id,member_positions_base_m=[p[i].tolist() if v[i] else None for i in range(n)],
                center_base_m=p.mean(0).tolist() if v.all() else None,
                support_kind='uncalibrated generated alternative; not posterior probability',
                mode_source=scene.get('mode_source','conditional_random_sample'))
            result['alternatives'].append(alternate)
            for i,target in enumerate(result['targets']):
                if not v[i]:continue
                target['alternatives'].append(dict(mode_id=mode_id,position_base_m=p[i].tolist(),
                    **spherical_from_local(p[i:i+1])[0],support_kind=alternate['support_kind']))
                if valid[i] and np.linalg.norm(p[i]-positions[i])>=float(self.config.get('group_mode_separation_m',.5)):
                    target['localization_status']='multiple_generated_modes_uncalibrated'
        if len(modes)>1 and valid.all():result['group']['localization_status']='multiple_generated_modes_uncalibrated'
        return result
