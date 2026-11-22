"""Audit validation logic with in-memory I/O; real Parquet tests are separate."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from unirank.experiment import write_json_exclusive
from unirank.taobao import LABELS, audit_dataset, validate_audit_report


class AuditLogicTests(unittest.TestCase):
    def fixture(self, root):
        frames = {}
        meta = {'label': LABELS, 'vocab_size': {'user_id': 2, 'item_id': 6, 'action': 17},
                'action_vocab': {'exposure': 6, 'is_click|cart|fav|buy': 14},
                'sample_size': dict(total=15, train=5, valid=5, test=5)}
        (root / 'meta_data.json').write_text(json.dumps(meta), encoding='utf-8')
        for split in ('train', 'valid', 'test'):
            data = pd.DataFrame({'user_index': [0] * 5, 'item_index': range(1, 6),
                                 'seq_len': range(5), 'user_id': [1] * 5})
            for label in LABELS:
                data[label] = [0, 1, 0, 1, 0]
            tables = {'data': data,
                      'user_info': pd.DataFrame({'user_index': [0], 'full_item_seq': [[1, 2, 3, 4, 5]],
                                                'full_timestamp_seq': [[100, 100, 200, 300, 400]],
                                                'full_action_seq': [[6, 14, 6, 14, 6]]}),
                      'item_info': pd.DataFrame({'item_index': range(1, 6), 'item_id': range(1, 6)})}
            for kind, frame in tables.items():
                path = root / split / kind / 'part-00000.parquet'
                path.parent.mkdir(parents=True)
                path.write_bytes(b'in-memory-I/O-fixture')
                frames[str(path)] = frame
        return frames

    def io(self, frames):
        fake_arrow = ModuleType('pyarrow')
        fake_parquet = ModuleType('pyarrow.parquet')
        fake_parquet.ParquetFile = lambda path: SimpleNamespace(iter_batches=lambda **kwargs: [
            SimpleNamespace(to_pandas=lambda: frames[str(path)].copy())])
        fake_arrow.parquet = fake_parquet
        return patch.dict(sys.modules, {'pyarrow': fake_arrow, 'pyarrow.parquet': fake_parquet})

    def test_statistics_quartiles_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = self.fixture(root)
            with self.io(frames), patch.object(pd, 'read_parquet', side_effect=lambda path: frames[str(path)].copy()):
                report = audit_dataset(root, 2)
            self.assertEqual(report['train_L_prefix_quartiles'], np.quantile([0, 0, 2, 3, 4], [.25, .5, .75]).tolist())
            self.assertEqual(report['splits']['train']['history_ties_removed'], 1)
            self.assertAlmostEqual(report['splits']['train']['mean_L_input'], 1.2)
            self.assertFalse(report['data_revision_verified'])
            write_json_exclusive(root / 'audit.json', report)
            self.assertEqual(len(validate_audit_report(root / 'audit.json', root, 2)), 64)
            (root / 'test/data/part-00000.parquet').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                validate_audit_report(root / 'audit.json', root, 2)

    def test_duplicate_target_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frames = self.fixture(root)
            frame = frames[str(root / 'train/data/part-00000.parquet')]
            frame.loc[1, ['seq_len', 'item_index']] = [0, 1]
            with self.io(frames), patch.object(pd, 'read_parquet', side_effect=lambda path: frames[str(path)].copy()):
                with self.assertRaisesRegex(ValueError, 'Duplicate target identity'):
                    audit_dataset(root, 2)


if __name__ == '__main__':
    unittest.main()
