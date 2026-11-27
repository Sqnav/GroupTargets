"""Synthetic one-query checks; no user data, weights or real training."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rvloc.models import (build_model, localization_loss, save_checkpoint, load_checkpoint,
                          WorldModelMemory, MemberState)
from rvloc.data import DIRECT_FUSION_SCHEMA, DIRECT_DEPTH_SCHEMA
from rvloc.models.single_frame import ARCHITECTURE, model_config


def synthetic_features(n=3, frame=0, episode="synthetic", ids=None):
    rng = np.random.default_rng(10)
    directions = np.column_stack((np.ones(n), np.linspace(-.08,.08,n), np.full(n,.02)))
    directions /= np.linalg.norm(directions,axis=1,keepdims=True)
    boxes = np.tile(np.array([288.,198.,312.,222.]),(n,1))
    scores = np.full(n,.8)
    nx,ny = (boxes[:,2]-boxes[:,0])/500.,(boxes[:,3]-boxes[:,1])/500.
    features = dict(schema=DIRECT_FUSION_SCHEMA,depth_schema=DIRECT_DEPTH_SCHEMA,
        ray_origins=np.tile(np.array([5.,2.,3.]),(n,1)),ray_directions=directions,
        observer_position=np.array([4.8,2.,3.]),ray_depth_bounds=np.tile([.1,150.],(n,1)),
        ray_valid=np.ones(n,dtype=bool),bbox_xyxy=boxes,detection_scores=scores,
        visual_scale=np.column_stack((nx,ny,np.sqrt(nx*ny),np.log(nx),np.log(ny),
                                      np.zeros(n),scores)),
        camera_K=np.array([[500.,0.,320.],[0.,500.,240.],[0.,0.,1.]]),
        camera_image_size_wh=np.array([640,480]),range_scale_m=150.,metadata=dict(synthetic_only=True),
        ra_power=rng.uniform(.01,2.,(20,17)).astype(np.float32),
        re_power=rng.uniform(.01,2.,(20,15)).astype(np.float32),
        ra_valid=np.ones((20,17),dtype=bool),re_valid=np.ones((20,15),dtype=bool),
        range_m=np.linspace(0.,150.,20),azimuth_rad=np.linspace(-.9,.9,17),
        elevation_rad=np.linspace(-.8,.8,15),
        detector_roi_features=rng.normal(size=(n,1024)).astype(np.float32))
    features.update(episode_id=episode,frame_index=frame,timestamp_s=float(frame),
                    member_ids=list(range(n)) if ids is None else list(ids))
    return features


class _NeverReadRadar:
    def __array__(self,*args,**kwargs):
        raise AssertionError("RGB must never inspect a radar payload")


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setUp(self):torch.manual_seed(17)

    def assert_no_iterative_modules(self,model):
        for name in model.state_dict():
            self.assertFalse(any(word in name for word in
                ('coarse_head','global_head','refine_','candidate_sequence','depth_offsets','depth_fine_offsets')),name)
        self.assertEqual(model.config['depth_refinement_steps'],0)

    def assert_on_ray(self,features,output):
        expected=torch.as_tensor(features['ray_origins'],dtype=torch.float32)+output['depths'][:,None]*torch.as_tensor(features['ray_directions'],dtype=torch.float32)
        torch.testing.assert_close(output['member_positions'],expected*output['valid_mask'][:,None])

    def test_rgb_is_visual_mean_and_std_without_any_radar_query(self):
        model=build_model(2);features=synthetic_features()
        for key in ('ra_power','re_power','ra_valid','re_valid','range_m','azimuth_rad','elevation_rad'):
            features[key]=_NeverReadRadar()
        with patch('torch.load',side_effect=AssertionError('No checkpoints')), \
             patch('numpy.load',side_effect=AssertionError('No sensor files')), \
             patch('rvloc.models.single_query.sample_paired_spectra',side_effect=AssertionError('RGB has no query')):
            output=model(features)
        self.assertFalse(hasattr(model.core,'ra_encoder'));self.assertFalse(hasattr(model.core,'local_query'))
        self.assertFalse(hasattr(model.core,'depth_head'))
        self.assertEqual(output['aux']['architecture'],ARCHITECTURE)
        self.assertFalse(output['aux']['radar_used']);self.assertEqual(output['aux']['query_rounds'],0)
        torch.testing.assert_close(output['depths'],output['visual_depth_mean_m'],atol=0.,rtol=0.)
        self.assertTrue(bool((output['visual_depth_std_m']>=.25).all()))
        self.assertTrue(bool((output['visual_depth_std_m']<=16.).all()))
        self.assertNotIn('initial_depths',output);self.assertNotIn('initial_positions',output)
        self.assert_no_iterative_modules(model);self.assert_on_ray(features,output)

    def test_fusion_encodes_each_map_once_and_queries_all_candidates_once(self):
        import rvloc.models.single_query as query_module
        for group in (3,7):
            model=build_model(group);features=synthetic_features()
            with patch.object(model.core.ra_encoder,'forward',wraps=model.core.ra_encoder.forward) as ra, \
                 patch.object(model.core.re_encoder,'forward',wraps=model.core.re_encoder.forward) as re, \
                 patch.object(query_module,'sample_paired_spectra',wraps=query_module.sample_paired_spectra) as sampler:
                output=model(features)
            self.assertEqual(ra.call_count,1);self.assertEqual(re.call_count,1);self.assertEqual(sampler.call_count,1)
            self.assertEqual(output['aux']['query_rounds'],1)
            diagnostic=output['sampling_diagnostics']
            self.assertEqual(tuple(diagnostic['candidate_depths_m'].shape),(3,9))
            self.assertEqual(diagnostic['stage_depths_m'],[])
            self.assertEqual(diagnostic['refinement_steps'],0)
            self.assert_no_iterative_modules(model);self.assert_on_ray(features,output)

    def test_candidate_locations_follow_metric_visual_uncertainty_once(self):
        model=build_model(3);features=synthetic_features()
        first=model(features);d=first['sampling_diagnostics']
        z=torch.tensor([-3.,-2.,-1.,-.5,0.,.5,1.,2.,3.])
        torch.testing.assert_close(d['candidate_depths_m'],first['visual_depth_mean_m'][:,None]+first['visual_depth_std_m'][:,None]*z)
        torch.testing.assert_close(d['candidate_offsets_m'],first['visual_depth_std_m'][:,None]*z)
        torch.testing.assert_close(d['normalized_offsets'],z.expand(3,-1))
        with torch.no_grad():model.core.visual_prior_head[-1].bias[1]+=2.
        second=model(features)
        torch.testing.assert_close(first['visual_depth_mean_m'],second['visual_depth_mean_m'],atol=0.,rtol=0.)
        self.assertTrue(bool((second['visual_depth_std_m']>first['visual_depth_std_m']).all()))
        self.assertTrue(bool((second['sampling_diagnostics']['candidate_offsets_m'][:,-1]>d['candidate_offsets_m'][:,-1]).all()))

    def test_radar_changes_fused_depth_but_never_the_visual_prior(self):
        model=build_model(3);features=synthetic_features();first=model(features)
        changed=synthetic_features();changed['ra_power'][:10,:8]*=25.
        second=model(changed)
        torch.testing.assert_close(first['visual_depth_mean_m'],second['visual_depth_mean_m'],atol=0.,rtol=0.)
        torch.testing.assert_close(first['visual_depth_std_m'],second['visual_depth_std_m'],atol=0.,rtol=0.)
        torch.testing.assert_close(first['sampling_diagnostics']['candidate_depths_m'],second['sampling_diagnostics']['candidate_depths_m'],atol=0.,rtol=0.)
        self.assertGreater(float((first['depths']-second['depths']).abs().sum()),1e-5)

    def test_visual_prior_initialization_is_identical_across_sensor_groups(self):
        torch.manual_seed(22);rgb=build_model(2)
        torch.manual_seed(22);fusion=build_model(3)
        features=synthetic_features();a,b=rgb(features),fusion(features)
        for key in ('visual_depth_mean_m','visual_depth_std_m'):
            torch.testing.assert_close(a[key],b[key],atol=0.,rtol=0.)

    def test_final_depth_is_continuous_beyond_candidate_window(self):
        model=build_model(3);features=synthetic_features()
        with torch.no_grad():
            model.core.depth_head[-1].weight.zero_();model.core.depth_head[-1].bias.fill_(12.)
        output=model(features);d=output['sampling_diagnostics']
        self.assertTrue(bool((output['depths']>d['candidate_depths_m'].max(-1).values+1.).all()))
        self.assertTrue(bool((output['depths']<150.).all()))
        self.assertEqual(d['query_rounds'],1);self.assert_on_ray(features,output)

    def test_radar_candidate_and_visual_uncertainty_gradients_are_real(self):
        model=build_model(3);features=synthetic_features()
        for key in ('ra_power','re_power','detector_roi_features'):
            features[key]=torch.tensor(features[key],requires_grad=True)
        output=model(features)
        targets=dict(depths=output['depths'].detach()+1.,member_positions=output['member_positions'].detach())
        loss,terms=localization_loss(output,targets);loss.backward()
        self.assertIn('visual_prior_nll',terms)
        for key in ('ra_power','re_power'):
            self.assertIsNotNone(features[key].grad)
            self.assertGreater(float(features[key].grad.abs().sum()),0.)
            self.assertTrue(bool(torch.isfinite(features[key].grad).all()))
        self.assertIsNone(features['detector_roi_features'].grad)
        for module in (model.core.ra_encoder,model.core.re_encoder,model.core.roi_projection,
                       model.core.local_query.candidate_interaction,model.core.visual_prior_head,model.core.depth_head):
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.)
        self.assertGreater(float(model.core.visual_prior_head[-1].weight.grad[1].abs().sum()),0.)

    def test_world_copies_one_query_group3_and_birth_has_zero_injection(self):
        base=build_model(3);world=build_model(7);world.initialize_from_single_frame(base)
        features=synthetic_features();expected=base(features);actual=world(features)
        torch.testing.assert_close(actual['depths'],expected['depths'],atol=0.,rtol=0.)
        self.assertEqual(float(actual['aux']['memory_injection'].abs().sum()),0.)
        self.assertFalse(bool(actual['aux']['prior_available'].any()))
        self.assertEqual(actual['aux']['architecture'],ARCHITECTURE)
        self.assertEqual(actual['aux']['query_rounds'],1)
        self.assertEqual(set(actual['state'].members),set(features['member_ids']))

    def test_world_history_changes_only_post_query_fusion_not_candidates(self):
        world=build_model(7)
        with torch.no_grad():world.memory_bridge.weight.normal_(std=.12)
        features=synthetic_features(frame=1)
        a=WorldModelMemory('synthetic',{key:MemberState(torch.ones(64),0.,0) for key in range(3)})
        b=WorldModelMemory('synthetic',{key:MemberState(-torch.ones(64),0.,0) for key in range(3)})
        first,second=world(features,a),world(features,b)
        for key in ('visual_depth_mean_m','visual_depth_std_m'):
            torch.testing.assert_close(first[key],second[key],atol=0.,rtol=0.)
        for key in ('candidate_depths_m','candidate_offsets_m','sampled_ra','sampled_re'):
            torch.testing.assert_close(first['sampling_diagnostics'][key],second['sampling_diagnostics'][key],atol=0.,rtol=0.)
        self.assertGreater(float((first['depths']-second['depths']).abs().sum()),1e-5)
        self.assert_on_ray(features,first)

    def test_memory_identity_episode_causality_and_timeout(self):
        world=build_model(7);first=world(synthetic_features())
        second=world(synthetic_features(frame=1,ids=[2,0,9]),first['state'])
        self.assertEqual(second['aux']['prior_available'].tolist(),[True,True,False])
        unassigned=world(synthetic_features(frame=1,ids=[None,None,0]),first['state'])
        self.assertEqual(unassigned['aux']['prior_available'].tolist(),[False,False,True])
        self.assertNotIn(None,unassigned['state'].members)
        other=world(synthetic_features(frame=1,episode='other'),first['state'])
        self.assertFalse(bool(other['aux']['prior_available'].any()))
        self.assertEqual(other['state'].episode_id,'other')
        stale=world(synthetic_features(frame=20),first['state'])
        self.assertFalse(bool(stale['aux']['prior_available'].any()))
        with self.assertRaisesRegex(ValueError,'strictly causal'):world(synthetic_features(),first['state'])
        broken=synthetic_features();broken['member_ids']=[0,0,1]
        with self.assertRaisesRegex(ValueError,'Unique'):world(broken)

    def test_world_rollout_one_query_loss_and_priors_have_finite_gradients(self):
        world=build_model(7,{'newworld':{'hidden_size':32,'rollout_horizons':[1,3]}})
        first=world(synthetic_features())
        output=world(synthetic_features(frame=1),first['state'].detached())
        targets=dict(depths=output['depths'].detach()+1.,member_positions=output['member_positions'].detach(),
            rollout_positions=output['aux']['rollout_positions'].detach()+2.,rollout_valid=torch.ones((3,2),dtype=torch.bool))
        loss,terms=localization_loss(output,targets);loss.backward()
        self.assertNotIn('initial_huber',terms);self.assertIn('visual_prior_nll',terms)
        self.assertEqual(tuple(output['aux']['rollout_positions'].shape),(3,2,3))
        self.assertIn('rollout_position_huber',terms)
        for module in (world.transition,world.posterior,world.position_decoder,world.memory_bridge,
                       world.core.local_query,world.core.visual_prior_head,world.core.depth_head):
            grads=[p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(bool(torch.isfinite(g).all()) for g in grads))
            self.assertGreater(sum(float(g.abs().sum()) for g in grads),0.)

    def test_world_bridge_follows_configured_member_width(self):
        world=build_model(7,dict(depth_hidden_dim=32,newworld=dict(hidden_size=16)))
        self.assertEqual(world.memory_bridge.out_features,32)
        self.assertEqual(tuple(world(synthetic_features())['state'].members[0].latent.shape),(16,))

    def test_public_depth_support_survives_no_local_radar_evidence(self):
        features=synthetic_features();features['range_scale_m']=250.;features['ray_depth_bounds'][:]=[.25,300.]
        model=build_model(3)
        with torch.no_grad():
            model.core.visual_prior_head[-1].weight.zero_();model.core.visual_prior_head[-1].bias[0]=6.
        output=model(features)
        self.assertTrue(bool((output['visual_depth_mean_m']>200.).all()))
        self.assertFalse(bool(output['sampling_diagnostics']['evidence_available'].any()))
        self.assertTrue(bool((output['depths']>.25).all()));self.assertTrue(bool((output['depths']<300.).all()))
        self.assertEqual(output['valid_mask'].tolist(),[True]*3)
        self.assertEqual(output['aux']['query_rounds'],1)

    def test_empty_invalid_and_forbidden_inputs(self):
        for group in (2,3,7):
            model=build_model(group)
            for count in (0,2):
                features=synthetic_features(n=count);features['ray_valid'][:]=False;features['ray_depth_bounds'][:]=0.
                output=model(features)
                self.assertEqual(tuple(output['member_positions'].shape),(count,3))
                self.assertTrue(bool(torch.isfinite(output['depths']).all()))
                self.assertTrue(bool(torch.isfinite(output['visual_depth_std_m']).all()))
                self.assertFalse(bool(output['valid_mask'].any()))
            for key in ('iq','risk_score','search_features','targets','future','depth_metadata'):
                features=synthetic_features();features[key]=object()
                with self.assertRaises(ValueError):model(features)

    def test_iterative_configs_and_invalid_visual_uncertainty_are_rejected(self):
        for config in ({'depth_refinement_steps':2},{'depth_offsets_m':[0.]},{'depth_fine_offsets_m':[0.]},
                       {'depth_max_correction_m':16.},{'depth_initial_loss_weight':.3},{'query_rounds':2},
                       {'local_query_enabled':False},{'d_evidence_capture_candidates':True},
                       {'visual_std_min_m':0.},{'visual_std_initial_m':32.},{'visual_std_max_m':.1}):
            with self.assertRaises(ValueError):build_model(3,config)
        with self.assertRaisesRegex(ValueError,'group'):build_model(2,{'query_rounds':1})
        self.assertEqual(model_config()['architecture'],ARCHITECTURE)

    def test_checkpoint_roundtrip_is_architecture_bound(self):
        model=build_model(7);features=synthetic_features()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'synthetic.pt';save_checkpoint(model,path,metadata={'synthetic_only':True})
            loaded,payload=load_checkpoint(path,expected_group=7)
            self.assertTrue(payload['metadata']['synthetic_only']);self.assertEqual(payload['architecture'],ARCHITECTURE)
            torch.testing.assert_close(model(features)['depths'],loaded(features)['depths'],atol=0.,rtol=0.)
            with self.assertRaisesRegex(ValueError,'group'):load_checkpoint(path,expected_group=2)


if __name__=='__main__':unittest.main()
