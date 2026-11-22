from pathlib import Path
import glob
import os
import unittest
import yaml

from tests.source_loader import ROOT, load_symbols


class ProjectConfigTests(unittest.TestCase):
    def test_effective_project_configuration(self):
        namespace = dict(glob=glob, os=os, yaml=yaml)
        load_symbols('unirank/utils.py', ['load_config', 'load_model_config', 'load_dataset_config'], namespace)
        directory = str(ROOT / 'config/project')
        smoke = namespace['load_config'](directory, 'RankMixer_Taobao_Action_Smoke')
        self.assertEqual(smoke['block_cache_size'], 1)
        self.assertEqual(smoke['num_workers'], 0)
        self.assertEqual(smoke['num_tasks'], len(smoke['label_col']))
        self.assertEqual(smoke['history_feedback_policy'], 'exposure_only')
        self.assertFalse(smoke['eval_drop_last'])
        self.assertTrue(smoke['verify_checkpoint_reload'])
        self.assertEqual(smoke['monitor'], {'is_click_AUC': 1})
        reference = namespace['load_config'](directory, 'RankMixer_Taobao_Action_Reference')
        self.assertEqual(reference['history_feedback_policy'], 'original')
        self.assertEqual(reference['monitor'], {'AUC': 1, 'logloss': 0})
        self.assertTrue(reference['keep_checkpoint'])
        self.assertTrue(reference['eval_drop_last'])
        self.assertEqual(reference['batch_size'], 8192)

    def test_lab_paths_preserve_model_protocol(self):
        namespace = dict(glob=glob, os=os, yaml=yaml)
        load_symbols('unirank/utils.py', ['load_config', 'load_model_config', 'load_dataset_config'], namespace)
        for experiment in ('Reference', 'Exposure', 'Smoke'):
            name = 'RankMixer_Taobao_Action_' + experiment
            project = namespace['load_config'](str(ROOT / 'config/project'), name)
            lab = namespace['load_config'](str(ROOT / 'config/lab'), name)
            for key in ('model', 'num_tasks', 'batch_size', 'max_len', 'monitor', 'history_feedback_policy'):
                self.assertEqual(project[key], lab[key])
            self.assertEqual(lab['data_root'], '/mnt/nas/volume2/hsc/datasets')
            self.assertEqual(lab['run_root'], '/mnt/nas/volume2/hsc/experiments')
            self.assertTrue(lab['train_data'].startswith(lab['data_root'] + '/'))
            if lab.get('require_data_audit'):
                self.assertTrue(lab['data_audit_report'].startswith('/mnt/nas/volume2/hsc/reports/'))


if __name__ == '__main__':
    unittest.main()
