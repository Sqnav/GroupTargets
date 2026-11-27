"""Planning and storage guards with zero dataset reads."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rvloc.cli import execute, parser
from rvloc.config import ExperimentConfig, code_root


class ConfigTests(unittest.TestCase):
    def config(self):
        return json.loads((code_root()/'configs/seven_methods.json').read_text())

    def test_plan_is_metadata_only(self):
        args=parser().parse_args(['plan','--config',str(code_root()/'configs/seven_methods.json')])
        with patch('torch.load',side_effect=AssertionError('checkpoint opened')):
            result=execute(args)
        self.assertEqual(result['status'],'code_ready_data_pending')
        self.assertEqual(len(result['methods']),7)
        self.assertFalse(result['training_started'])
        self.assertEqual(result['weights'],'/data7/ysq/GroupTargets/weights/radar_vision_localization')

    def test_reject_risk_and_inside_source_artifact(self):
        base=self.config()
        for mutation in ('risk','path','weights_path','overlapping_weights'):
            value=copy.deepcopy(base)
            if mutation=='risk':value['iq_risk_correction']=False
            elif mutation=='path':value['storage']['output_root']=str(code_root()/'results')
            elif mutation=='weights_path':value['storage']['weights_root']=str(code_root()/'weights')
            else:value['storage']['weights_root']=value['storage']['output_root']+'/weights'
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'config.json';path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):ExperimentConfig.load(path)

    def test_legacy_config_and_positional_constructor_get_separate_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);value=self.config()
            value['storage']={'output_root':str(root/'outputs'),'cache_root':str(root/'cache')}
            path=root/'config.json';path.write_text(json.dumps(value))
            config=ExperimentConfig.load(path)
            self.assertEqual(config.weights_root,root/'weights')
            compatible=ExperimentConfig(config.path,config.raw,config.manifest,config.output_root,config.cache_root)
            self.assertEqual(compatible.weights_root,root/'weights')
            self.assertFalse(config.weights_root.exists())
            self.assertFalse(config.output_root.exists())

    def test_no_historical_test_command(self):
        with self.assertRaises(SystemExit):parser().parse_args(['infer','--method','3','--config','x','--split','test'])


if __name__=='__main__':unittest.main()
