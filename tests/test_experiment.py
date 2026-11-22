from pathlib import Path
import tempfile
import unittest

from unirank.experiment import argument_parser, base_metric_names, prepare_run, write_json_exclusive
from tests.source_loader import load_symbols


class ExperimentTests(unittest.TestCase):
    def test_false_bf16_is_false_and_omission_preserves_config(self):
        self.assertIs(argument_parser().parse_args(['--enable_bf16', 'False']).enable_bf16, False)
        self.assertIs(argument_parser().parse_args(['--enable_bf16', 'true']).enable_bf16, True)
        self.assertIsNone(argument_parser().parse_args([]).enable_bf16)

    def test_task_metric_keys_are_resolved(self):
        self.assertEqual(base_metric_names(['is_click_AUC', 'cart_logloss', 'AUC'],
                                          ['is_click', 'cart']), ['AUC', 'logloss'])

    def test_monitor_never_silently_uses_zero(self):
        monitor_class = load_symbols('unirank/utils.py', ['Monitor'], {})['Monitor']
        self.assertEqual(monitor_class({'is_click_AUC': 1}).get_value({'is_click_AUC': .7}), .7)
        with self.assertRaises(KeyError):
            monitor_class('is_click_AUC').get_value({'AUC': .7})

    def test_unique_run_snapshots_and_collision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'config'
            config.mkdir()
            (config / 'model_config.yaml').write_text('test: true\n', encoding='utf-8')
            params = {'model_id': 'Smoke', 'seed': 1, 'run_root': str(root / 'runs')}
            prepared = prepare_run(params, config, 'fixed')
            directory = Path(prepared['run_dir'])
            self.assertTrue((directory / 'run.json').is_file())
            self.assertEqual((directory / 'config/model_config.yaml').read_text(), 'test: true\n')
            with self.assertRaises(FileExistsError):
                prepare_run(params, config, 'fixed')
            with self.assertRaises(ValueError):
                prepare_run(params, config, '../escape')
            write_json_exclusive(directory / 'metrics.json', {'test': .5})
            with self.assertRaises(FileExistsError):
                write_json_exclusive(directory / 'metrics.json', {'test': .9})


if __name__ == '__main__':
    unittest.main()
