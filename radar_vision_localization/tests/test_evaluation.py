"""Synthetic offline-metric checks; no user dataset or learned weights."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rvloc.evaluation import (evaluate, match_positions, metrics, paired_compare, episode_bootstrap,
                              visual_prior_rows, visual_prior_metrics)


class EvaluationTests(unittest.TestCase):
    def test_visual_uncertainty_uses_public_ray_supervision_not_final_position(self):
        frame=dict(visual_depth_mean_m=[21.],visual_depth_std_m=[2.],
            ray_origins=[[0.,0.,0.]],ray_directions=[[1.,0.,0.]],ray_depth_bounds=[[.1,100.]],
            range_scale_m=10.,valid=[True],positions=[[9999.,0.,0.]])
        labels=SimpleNamespace(positions=np.array([[20.,0.,0.]]))
        rows=visual_prior_rows(frame,labels)
        result=visual_prior_metrics(rows)
        self.assertEqual(result['scored_members'],1)
        self.assertEqual(result['depth_rmse_m'],1.)
        self.assertEqual(result['coverage_1sigma'],1.)
        self.assertAlmostEqual(result['gaussian_nll'],.125+np.log(2)+.5*np.log(2*np.pi))
        frame['visual_depth_std_m']=[0.]
        with self.assertRaises(ValueError):visual_prior_rows(frame,labels)

    def test_empty_exported_visual_prior_is_unscored(self):
        frame=dict(visual_depth_mean_m=[],visual_depth_std_m=[],ray_origins=[],ray_directions=[],
                   ray_depth_bounds=[],range_scale_m=10.,valid=[])
        rows=visual_prior_rows(frame,SimpleNamespace(positions=np.empty((0,3))))
        result=visual_prior_metrics(rows)
        self.assertEqual(result['scored_members'],0)
        self.assertIsNone(result['coverage_1sigma'])

    def test_one_to_one_and_unmatched_are_explicit(self):
        pairs=match_positions([[0,0,0],[0.1,0,0],[100,0,0]],[[0,0,0],[2,0,0]],max_distance=1)
        self.assertEqual(pairs,[(0,0,0.)])
        result=metrics([dict(error_m=0),dict(error_m=None),dict(error_m=4)])
        self.assertAlmostEqual(result['position_rmse_3d_m'],np.sqrt(8))
        self.assertAlmostEqual(result['all_gt_success']['1m'],1/3)
        self.assertEqual(result['no_localization_members'],1)

    def test_invalid_prediction_retained_as_missing(self):
        self.assertEqual(match_positions([[0,0,0]],[[0,0,0]],valid=[False]),[])
        self.assertEqual(match_positions([[None,None,None]],[[0,0,0]]),[])
        self.assertIsNone(metrics([])['position_rmse_3d_m'])

    def test_bootstrap_resamples_episodes_and_handles_sparse_tail(self):
        rows=[dict(episode_id='a',error_m=1.),dict(episode_id='b',error_m=3.)]
        result=episode_bootstrap(rows,draws=50,seed=3)
        self.assertEqual(result['resampling_unit'],'episode')
        self.assertEqual(result['intervals']['p99_m']['95%'],[1.,3.])
        self.assertIsNone(episode_bootstrap(rows[:1],draws=50)['intervals'])

    def test_paired_identity_matching_not_list_order(self):
        a=dict(schema='rvloc.evaluation.v1',manifest_sha256='a'*64,split='dev',method_id=3,
            rows=[dict(episode_id=e,frame_idx=0,gt_member_id='g',error_m=x) for e,x in [('a',3.),('b',4.)]])
        b=copy.deepcopy(a);b['method_id']=7;b['rows'].reverse()
        for row in b['rows']:row['error_m']-=1
        result=paired_compare(a,b,draws=50)
        self.assertEqual(result['common_located'],2)
        self.assertLess(result['common_member_deltas']['rmse_delta_m'],0)
        b['rows'][0]['error_m']=None
        self.assertEqual(paired_compare(a,b,draws=50)['first_only_located'],1)

    def test_labels_open_only_after_prediction_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'predictions.json'
            frame=SimpleNamespace(episode_id='ep',frame_idx=0,physical_time_s=0.)
            manifest=SimpleNamespace(sha256='a'*64,splits={'train':('ep',)},frames=lambda split:[frame])
            prediction=dict(schema='rvloc.predictions.v1',normal_inference=True,labels_opened=False,
                evaluation_contract=dict(matching='same_per_frame_one_to_one_euclidean',match_max_distance_m=None),
                split='train',manifest_sha256=manifest.sha256,method_id=2,cost={},frames=[dict(
                episode_id='ep',frame_idx=0,physical_time_s=0.,positions=[[2.,0,0]],valid=[True],
                member_ids=[None],runtime_s=.001)])
            path.write_text(json.dumps(prediction))
            config=SimpleNamespace(raw={'evaluation':{'bootstrap_draws':20}},sha256='b'*64)
            def labels(*args):
                self.assertTrue(path.exists())
                self.assertEqual(json.loads(path.read_text())['labels_opened'],False)
                return SimpleNamespace(positions=np.array([[1.,0,0]]),member_ids=('offline-id',))
            with patch('rvloc.evaluation.load_labels',side_effect=labels):
                result=evaluate(config,path,manifest=manifest)
            self.assertEqual(result['position_rmse_3d_m'],1.)
            self.assertTrue((root/'failure_cases.json').exists())
            prediction['labels_opened']=True;path.write_text(json.dumps(prediction))
            with patch('rvloc.evaluation.load_labels',side_effect=AssertionError('GT opened')):
                with self.assertRaises(ValueError):evaluate(config,path,manifest=manifest)


if __name__=='__main__':unittest.main()
