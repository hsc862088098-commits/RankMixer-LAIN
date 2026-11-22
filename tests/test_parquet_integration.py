"""Real Parquet and model integration; skipped explicitly without dependencies."""
from collections import OrderedDict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from unirank.taobao import LABELS, audit_dataset, prepare_smoke_subset, validate_audit_report
from unirank.experiment import write_json_exclusive

HAS_PARQUET = importlib.util.find_spec('pyarrow') is not None
MODEL_DEPENDENCIES = ['pyarrow', 'polars', 'h5py', 'keras_preprocessing']
HAS_MODEL = all(importlib.util.find_spec(name) is not None for name in MODEL_DEPENDENCIES)


def fixture(root):
    """Five exposures, including a same-timestamp pair; no public dataset needed."""
    user_fields = ['cms_segid', 'cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level',
                   'shopping_level', 'occupation', 'new_user_class_level', 'pid', 'is_weekend', 'hour']
    item_fields = ['item_id', 'cate_id', 'campaign_id', 'customer_id', 'brand', 'price_bucket']
    vocab = {name: 8 for name in user_fields + item_fields + ['user_id']}
    vocab['action'] = 17
    meta = {'label': LABELS, 'vocab_size': vocab,
            'action_vocab': {'exposure': 6, 'is_click|cart|fav|buy': 14},
            'sample_size': dict(total=15, train=5, valid=5, test=5),
            'blocked_layout': {split + '_blocks': 1 for split in ('train', 'valid', 'test')}}
    root.mkdir()
    (root / 'meta_data.json').write_text(json.dumps(meta), encoding='utf-8')
    for split in ('train', 'valid', 'test'):
        for kind in ('data', 'user_info', 'item_info'):
            (root / split / kind).mkdir(parents=True)
        data = pd.DataFrame({'user_index': [0] * 5, 'item_index': range(1, 6),
                             'seq_len': range(5), 'user_id': [1] * 5})
        for field in user_fields:
            data[field] = 1
        for label in LABELS:
            data[label] = np.array([0, 1, 0, 1, 0], dtype=np.float32)
        data.to_parquet(root / split / 'data/part-00000.parquet', index=False)
        pd.DataFrame({'user_index': [0], 'full_item_seq': [[1, 2, 3, 4, 5]],
                      'full_action_seq': [[6, 14, 6, 14, 6]],
                      'full_timestamp_seq': [[100, 100, 200, 300, 400]]}).to_parquet(
                          root / split / 'user_info/part-00000.parquet', index=False)
        items = pd.DataFrame({'item_index': range(1, 6)})
        for field in item_fields:
            items[field] = np.arange(1, 6)
        items.to_parquet(root / split / 'item_info/part-00000.parquet', index=False)
    features = OrderedDict((name, {'type': 'meta'}) for name in ('user_index', 'item_index', 'seq_len'))
    for name, size in vocab.items():
        features[name] = {'type': 'categorical', 'vocab_size': size, 'padding_idx': 0}
        if name in item_fields:
            features[name]['source'] = 'item'
        elif name == 'action':
            features[name]['source'] = 'action'
    return features


@unittest.skipUnless(HAS_PARQUET, 'pyarrow is not installed; real Parquet integration not executed')
class ParquetTests(unittest.TestCase):
    def test_audit_and_changed_file_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'data'
            fixture(root)
            report = audit_dataset(root, 2)
            self.assertEqual(report['splits']['train']['history_ties_removed'], 1)
            self.assertEqual(report['splits']['test']['rows'], 5)
            self.assertAlmostEqual(report['splits']['train']['mean_L_prefix'], 1.8)
            path = Path(temporary) / 'audit.json'
            write_json_exclusive(path, report)
            self.assertEqual(len(validate_audit_report(path, root, 2)), 64)
            with (root / 'meta_data.json').open('a') as stream:
                stream.write(' ')
            with self.assertRaises(ValueError):
                validate_audit_report(path, root, 2)

    def test_subset_determinism_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / 'source'
            fixture(root)
            original_meta = (root / 'meta_data.json').read_bytes()
            for name in ('one', 'two'):
                prepare_smoke_subset(root, temporary / name, (2, 2, 2))
                audit_dataset(temporary / name, 2)
            for split in ('train', 'valid', 'test'):
                pd.testing.assert_frame_equal(
                    pd.read_parquet(temporary / 'one' / split / 'data/part-00000.parquet'),
                    pd.read_parquet(temporary / 'two' / split / 'data/part-00000.parquet'))
            self.assertEqual(original_meta, (root / 'meta_data.json').read_bytes())
            with self.assertRaises(FileExistsError):
                prepare_smoke_subset(root, temporary / 'one', (2, 2, 2))

    def test_real_loader_preserves_tail_and_sanitizes_actions(self):
        from types import SimpleNamespace
        from unirank.pytorch.dataloaders import UniRankDataloader
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'data'
            features = fixture(root)
            loader = UniRankDataloader(
                SimpleNamespace(features=features, labels=LABELS), root / 'test/data',
                user_info=root / 'test/user_info', item_info=root / 'test/item_info',
                batch_size=2, num_workers=0, max_len=2, split='test', eval_drop_last=False,
                history_feedback_policy='exposure_only', verify_eval_sample_count=True)
            batches = list(loader)
            self.assertEqual([len(batch[0]['seq_len']) for batch in batches], [2, 2, 1])
            self.assertEqual(torch.cat([b[0]['seq_len'] for b in batches]).tolist(), list(range(5)))
            for _, items, mask, multi_masks in batches:
                self.assertTrue(torch.equal(items['action'][:, :-1], mask.long() * 6))
                self.assertEqual(items['action'][:, -1].sum().item(), 0)
                self.assertTrue(all(m.count_nonzero() == 0 for m in multi_masks))


@unittest.skipUnless(HAS_MODEL, 'full model dependencies are missing; end-to-end model test not executed')
class FullModelTests(unittest.TestCase):
    def test_real_rankmixer_update_and_checkpoint_reload(self):
        from unirank.features import FeatureMap
        from unirank.pytorch.dataloaders import UniRankDataloader
        from model_zoo.RankMixer import RankMixer
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            root = temporary / 'data'
            features = fixture(root)
            feature_map = FeatureMap('SyntheticTaobao', str(root))
            feature_map.features = features
            feature_map.labels = LABELS
            feature_map.default_emb_dim = 16
            feature_map.num_fields = feature_map.get_num_fields()
            feature_map.group_id = 'user_id'
            loader = UniRankDataloader(feature_map, root / 'train/data',
                                      user_info=root / 'train/user_info', item_info=root / 'train/item_info',
                                      batch_size=5, num_workers=0, max_len=100,
                                      history_feedback_policy='exposure_only', drop_last=False)
            model = RankMixer(feature_map, task='binary_classification', num_tasks=4,
                              embedding_dim=16, token_dim=256, num_ns_token=4, num_layers=3,
                              expansion_factor=2, verbose=0, model_root=str(temporary / 'models'),
                              dense_optimizer='AdamW', sparse_optimizer='Adagrad', dense_learning_rate=1e-4,
                              loss=['binary_crossentropy'] * 4, metrics=['AUC', 'logloss'],
                              enable_torch_compile=False, enable_bf16=False)
            batch = next(iter(loader))
            model.samples_seen = model._batch_index = 0
            model._max_gradient_norm = 10
            parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
            loss = model.train_step(batch)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(not torch.equal(parameters[name], p) for name, p in model.named_parameters()))
            model.eval()
            with torch.no_grad():
                before = {name: value.clone() for name, value in model(batch).items()}
                model.save_weights(model.checkpoint)
                model.load_weights(model.checkpoint)
                after = model(batch)
            self.assertEqual(len(before), 4)
            for name in before:
                self.assertEqual(tuple(before[name].shape), (5, 1))
                torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
