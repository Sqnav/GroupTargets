"""Synthetic subprocess-boundary tests; no sensors, real weights or GT opened."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rvloc.config import code_root
from rvloc.experiment import _Runner, normalize_devices, run_experiment, stage_plan


class FakeCommands:
    def __init__(self, *, fail_stage=None):
        self.commands = []
        self.fail_stage = fail_stage

    def __call__(self, command, *, env, emit, heartbeat):
        self.commands.append(command)
        self.env = env
        self.assert_unbuffered(env)
        kind = command[4]

        def arg(name): return command[command.index(name)+1]
        def write(path, value):
            path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value)); return path

        config = json.loads(Path(arg('--config')).read_text()) if '--config' in command else None
        if '--output-dir' in command:
            output = Path(arg('--output-dir'))
            if output.exists(): raise AssertionError('Runner precreated the stage output directory')
        emit('synthetic child progress epoch=1/1 step=1 loss=0.125')
        heartbeat()
        if self.fail_stage == (kind, arg('--method') if '--method' in command else None):
            self.fail_stage = None
            return 9
        if kind == 'train':
            method = int(arg('--method'))
            relative = output.relative_to(Path(config['storage']['output_root']))
            weights = Path(config['storage']['weights_root'])/relative
            weights.mkdir(parents=True)
            checkpoint = weights/'model.pt'; checkpoint.write_bytes(b'synthetic, not a neural checkpoint')
            write(output/'receipt.json', dict(status='trained', method_id=method, checkpoint=str(checkpoint)))
            write(output/'history.json', [dict(epoch=1, optimizer_updates=1)])
        elif kind == 'infer':
            method = int(arg('--method'))
            if method in (4, 5, 6, 7) and config['calibration_artifact'] is None:
                raise AssertionError('Temporal inference started before TRAIN calibration')
            write(output/'predictions.json', dict(method_id=method, split=arg('--split')))
            write(output/'reproduction_receipt.json', dict(status='inference_completed'))
        elif kind == 'calibrate':
            inputs = command[command.index('--predictions')+1:command.index('--output')]
            for path in inputs:
                if json.loads(Path(path).read_text())['split'] != 'train':
                    raise AssertionError('Non-TRAIN input reached calibration')
            write(arg('--output'), dict(status='frozen', split='train'))
        elif kind == 'evaluate':
            prediction = json.loads(Path(arg('--predictions')).read_text())
            write(output/'evaluation.json', dict(schema='rvloc.evaluation.v1', method_id=prediction['method_id'],
                summary=dict(position_rmse_3d_m=1.25, mae_m=.9, p95_m=2.3, p99_m=4.1), cost=dict(wall_s=.01)))
            for name in ('localization_metrics.json', 'visual_prior_metrics.json',
                         'episode_level_results.json', 'failure_cases.json'):
                write(output/name, {})
        elif kind == 'compare':
            write(arg('--output'), dict(status='synthetic_paired', difference_rmse_m=0.))
        else: raise AssertionError('Unexpected child command: '+kind)
        return 0

    @staticmethod
    def assert_unbuffered(env):
        if env.get('PYTHONUNBUFFERED') != '1' or env.get('PYTHONDONTWRITEBYTECODE') != '1':
            raise AssertionError('Subprocess output must be immediate and source tree must stay code-only')


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = self.root/'config.json'
        self.manifest = self.root/'manifest.json'
        self.manifest.write_text(json.dumps(dict(status='synthetic_only', train=['episode_a'], dev=['episode_b'])))
        self.raw = json.loads((code_root()/'configs/seven_methods.json').read_text())
        self.raw['manifest'] = str(self.manifest)
        self.raw['storage'] = dict(output_root=str(self.root/'results'),
                                  weights_root=str(self.root/'weights'), cache_root=str(self.root/'cache'))
        self.config.write_text(json.dumps(self.raw))
        self.run = self.root/'results'/'run_a'

    def tearDown(self): self.temporary.cleanup()

    def execute(self, fake=None, **kwargs):
        with patch('builtins.print'):
            return run_experiment(self.config, self.run, process_runner=fake or FakeCommands(), **kwargs)

    def test_plan_has_only_three_neural_training_stages_and_no_test(self):
        stages = stage_plan()
        self.assertEqual(len(stages), 43)
        self.assertEqual([s['method'] for s in stages if s['kind'] == 'train'], [2, 3, 7])
        self.assertEqual(sum(s['kind'] == 'compare' for s in stages), 21)
        self.assertNotIn('test', {s['split'] for s in stages})

    def test_complete_protocol_and_honest_report(self):
        fake = FakeCommands()
        result = self.execute(fake, devices='1,4')
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(fake.commands), 42)
        status = json.loads((self.run/'status.json').read_text())
        self.assertEqual(status['completed_count'], 43)
        self.assertEqual(status['status'], 'completed')
        self.assertEqual(status['device_map'], {'2':'cuda:1', '3':'cuda:4', '7':'cuda:1'})
        self.assertIn('epoch=1/1', (self.run/'run.log').read_text())
        summary = json.loads(Path(result['summary']).read_text())
        self.assertEqual([int(i) for i, item in summary['methods'].items() if item['independently_trained']], [2, 3, 7])
        self.assertEqual(len(summary['pairwise_comparisons']), 21)
        self.assertFalse(summary['protocol']['historical_test_opened'])
        self.assertTrue(Path(result['report']).is_file())
        self.assertFalse(list(self.run.rglob('*.pt')))
        self.assertEqual(len(list((self.root/'weights').rglob('model.pt'))), 3)

    def test_completed_run_resume_only_verifies_outputs(self):
        self.execute(devices='cpu')
        fake = FakeCommands()
        result = self.execute(fake, devices='cpu', resume=True)
        self.assertEqual(result['completed_stages'], 43)
        self.assertEqual(fake.commands, [])

    def test_failure_preserved_and_next_attempt_restarts_only_failed_stage(self):
        fake = FakeCommands(fail_stage=('train', '3'))
        with self.assertRaisesRegex(RuntimeError, 'status 9'):
            self.execute(fake, devices='cpu')
        status = json.loads((self.run/'status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['completed_stages'], ['train_g2'])
        failure = self.run/'stages'/'train_g3'/'attempt_0001'/'attempt_receipt.json'
        self.assertEqual(json.loads(failure.read_text())['status'], 'failed')
        second = FakeCommands()
        self.execute(second, devices='cpu', resume=True)
        self.assertEqual(second.commands[0][second.commands[0].index('--method')+1], '3')
        self.assertTrue(failure.is_file())
        self.assertTrue((failure.parent.parent/'attempt_0002'/'attempt_receipt.json').is_file())
        self.assertFalse(json.loads((self.run/'protocol.json').read_text())['optimizer_resumption'])

    def test_resumption_rejects_changed_artifacts_config_manifest_or_source(self):
        self.execute(devices='cpu')
        checkpoint = next((self.root/'weights').rglob('model.pt'))
        checkpoint.write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'artifact missing or changed'):
            self.execute(devices='cpu', resume=True)
        for mutation in ('config', 'manifest', 'source', 'device'):
            with self.subTest(mutation=mutation):
                if mutation == 'config':
                    original = self.config.read_text(); self.config.write_text(original+'\n')
                    try:
                        with self.assertRaisesRegex(ValueError, 'identical config'):
                            self.execute(devices='cpu', resume=True)
                    finally: self.config.write_text(original)
                elif mutation == 'manifest':
                    original = self.manifest.read_text(); self.manifest.write_text(original+'\n')
                    try:
                        with self.assertRaisesRegex(ValueError, 'identical config'):
                            self.execute(devices='cpu', resume=True)
                    finally: self.manifest.write_text(original)
                elif mutation == 'source':
                    with patch('rvloc.experiment.source_identity', return_value={'sha256':'different'}):
                        with self.assertRaisesRegex(ValueError, 'identical config'):
                            self.execute(devices='cpu', resume=True)
                else:
                    with self.assertRaisesRegex(ValueError, 'identical config'):
                        self.execute(devices='cuda:0', resume=True)

    def test_no_overwrite_pending_data_and_device_validation(self):
        self.execute(devices='cpu')
        with self.assertRaises(FileExistsError): self.execute(devices='cpu')
        self.raw['manifest'] = None; self.config.write_text(json.dumps(self.raw))
        with self.assertRaisesRegex(ValueError, 'verified explicit data manifest'):
            self.execute(devices='cpu', resume=True)
        self.assertEqual(normalize_devices('0,cuda:1,0'), ('cuda:0', 'cuda:1'))
        for value in ('', 'all', 'cuda', 'cuda:-1', 'cpu,anything'):
            with self.subTest(value=value), self.assertRaises(ValueError): normalize_devices(value)

    def test_live_child_stdout_and_heartbeat_without_sensor_access(self):
        runner = _Runner(self.config, self.run, devices='cpu', resume=False)
        messages = []; beats = []
        code = runner._spawn([os.sys.executable, '-u', '-c', 'print("live synthetic child", flush=True)'],
                             env=dict(os.environ), emit=messages.append,
                             heartbeat=lambda: beats.append(runner.child_pid))
        self.assertEqual(code, 0)
        self.assertIn('live synthetic child', ''.join(messages))
        self.assertTrue(any(isinstance(pid, int) for pid in beats))
        self.assertIsNone(beats[-1])

    def test_preparation_only_directory_and_log_are_preserved(self):
        self.run.mkdir(parents=True)
        (self.run/'preparation.json').write_text('{"synthetic":true}')
        (self.run/'bootstrap.log').write_text('preparing\n')
        (self.run/'run.log').write_text('original bootstrap line\n')
        self.execute(devices='cpu')
        self.assertTrue((self.run/'run.log').read_text().startswith('original bootstrap line\n'))
        self.assertEqual((self.run/'bootstrap.log').read_text(), 'preparing\n')


if __name__ == '__main__': unittest.main()
