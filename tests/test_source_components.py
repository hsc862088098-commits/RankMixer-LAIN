from collections import OrderedDict, defaultdict
from contextlib import nullcontext
import json
import logging
import os
import random
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from unirank.experiment import base_metric_names
from unirank.history import (exposure_history_batch, paired_part_ids,
                             validate_history_policy, validate_user_sequence)
from unirank.metrics import evaluate_metrics
from unirank.pytorch.torch_utils import build_batch_dict_from_tensor
from tests.source_loader import load_symbols


class CollatorTests(unittest.TestCase):
    def collator(self, policy):
        namespace = dict(np=np, pd=pd, torch=torch, json=json, OrderedDict=OrderedDict,
                         exposure_history_batch=exposure_history_batch,
                         validate_history_policy=validate_history_policy,
                         validate_user_sequence=validate_user_sequence,
                         build_batch_dict_from_tensor=build_batch_dict_from_tensor)
        cls = load_symbols('unirank/pytorch/dataloaders/unirank_dataloader.py',
                           ['BlockedBatchCollator'], namespace)['BlockedBatchCollator']
        collator = cls.__new__(cls)
        collator.max_len = 2
        collator.padding = 'pre'
        collator.history_feedback_policy = policy
        collator.exposure_id = 6
        collator.action_task_table = torch.zeros(17, 4)
        collator.action_task_table[1, 3] = 1
        columns = ['user_index', 'item_index', 'seq_len', 'user_id', 'is_click', 'cart', 'fav', 'buy']
        collator.batch_cols = list(zip(columns, range(len(columns))))
        collator.all_cols = set(columns + ['item_id', 'action'])
        def array(value):
            result = np.empty(1, dtype=object)
            result[0] = np.array(value)
            return result
        side = dict(user_index_min=0, user_index_max=0, user_row_lookup=torch.tensor([0]),
                    user_item_seqs=array([1, 2, 3, 4]), user_action_seqs=array([1, 1, 1, 1]),
                    user_timestamp_seqs=array([100, 100, 200, 300]),
                    item_index_min=0, item_index_max=4, item_row_lookup=torch.arange(5),
                    item_tensors={'item_id': torch.tensor([0, 11, 22, 33, 44])})
        collator._load_block_side_info = lambda *args: side
        return collator, side

    def payload(self, position=2):
        return {'rows': np.array([[0, position + 1, position, 1, 0, 0, 0, 0]]),
                'user_info_file': 'unused', 'item_info_file': 'unused'}

    def test_actual_collator_ignores_future_feedback_and_target_labels(self):
        collator, side = self.collator('exposure_only')
        first = collator(self.payload())
        side['user_action_seqs'][0][:] = 14
        changed = self.payload()
        changed['rows'][0, 4:] = 1
        second = collator(changed)
        for key in first[1]:
            torch.testing.assert_close(first[1][key], second[1][key])
        torch.testing.assert_close(first[2], second[2])
        self.assertEqual(first[1]['action'].tolist(), [[6, 6, 0]])
        self.assertTrue(all(mask.count_nonzero() == 0 for mask in first[3]))

    def test_original_retains_feedback_and_target_action_zero(self):
        collator, side = self.collator('original')
        output = collator(self.payload())
        self.assertEqual(output[1]['action'].tolist(), [[1, 1, 0]])
        self.assertEqual(output[3][3].tolist(), [[1, 1]])

    def test_same_timestamp_and_empty_history_are_masked(self):
        collator, _ = self.collator('exposure_only')
        for position in [0, 1]:
            result = collator(self.payload(position))
            self.assertEqual(result[2].sum().item(), 0)
            self.assertEqual(result[0]['history_length_prefix'].item(), 0)


class DatasetIterationTests(unittest.TestCase):
    def test_tail_batches_and_worker_sharding(self):
        frames = {'d0': pd.DataFrame({'identity': np.arange(5)}),
                  'd1': pd.DataFrame({'identity': np.arange(5, 10)})}
        namespace = dict(np=np, random=random, IterableDataset=torch.utils.data.IterableDataset,
                         pd=SimpleNamespace(read_parquet=lambda path, **kwargs: frames[path]),
                         paired_part_ids=paired_part_ids,
                         get_worker_info=lambda: None,
                         dataframe_to_darray=lambda frame: (frame.to_numpy(), {'identity': 0}))
        cls = load_symbols('unirank/pytorch/dataloaders/unirank_dataloader.py',
                           ['BlockedParquetBatchDataset'], namespace)['BlockedParquetBatchDataset']
        dataset = cls.__new__(cls)
        dataset.rank_blocks = [{'part_id': n, 'data_file': f'd{n}', 'user_info_file': 'u',
                                'item_info_file': 'i'} for n in range(2)]
        dataset.rank = dataset.epoch = 0
        dataset.shuffle = False
        dataset.columns = None
        dataset.drop_last = False
        dataset.batch_size = 3
        dataset.block_row_counts = {0: 5, 1: 5}
        self.assertEqual(dataset._count_batches(), 4)
        self.assertEqual(np.concatenate([p['rows'][:, 0] for p in dataset]).tolist(), list(range(10)))
        identities = []
        for worker in range(2):
            namespace['get_worker_info'] = lambda worker=worker: SimpleNamespace(id=worker, num_workers=2)
            identities.extend(np.concatenate([p['rows'][:, 0] for p in dataset]).tolist())
        self.assertEqual(sorted(identities), list(range(10)))
        dataset.drop_last = True
        self.assertEqual(dataset._count_batches(), 2)


class EvaluationTests(unittest.TestCase):
    def model(self):
        namespace = dict(torch=torch, nn=nn, np=np, BaseModel=nn.Module,
                         defaultdict=defaultdict, OrderedDict=OrderedDict, logging=logging,
                         base_metric_names=base_metric_names)
        cls = load_symbols('unirank/pytorch/models/multitask_model.py',
                           ['MultiTaskModel'], namespace)['MultiTaskModel']
        class Dummy(cls):
            def __init__(self):
                nn.Module.__init__(self)
                self.feature_map = SimpleNamespace(labels=['is_click', 'cart', 'fav', 'buy'], group_id=None)
                self.device = torch.device('cpu')
                self._verbose = 0
                self.validation_metrics = ['AUC', 'logloss']
            def _get_amp_context(self):
                return nullcontext()
            def _is_distributed(self):
                return False
            def _is_main_process(self):
                return True
            def forward(self, inputs):
                return {label + '_pred': inputs[1] for label in self.feature_map.labels}
            def evaluate_metrics(self, *args):
                return evaluate_metrics(*args)
        return Dummy()

    def batches(self, expected=4):
        class Batches(list):
            verify_eval_sample_count = True
            num_samples = expected
        labels = {label: torch.tensor([0., 1., 0., 1.]) for label in ['is_click', 'cart', 'fav', 'buy']}
        return Batches([(labels, torch.tensor([[.1], [.9], [.2], [.8]]))])

    def test_click_monitor_is_calculated_as_auc(self):
        result = self.model().evaluate(self.batches(), metrics=['is_click_AUC'])
        self.assertEqual(result['is_click_AUC'], 1.)
        self.assertEqual(result['AUC'], 1.)

    def test_dropped_eval_rows_raise(self):
        with self.assertRaisesRegex(ValueError, 'sample count mismatch'):
            self.model().evaluate(self.batches(expected=5))


class TensorTests(unittest.TestCase):
    def test_base_model_excludes_history_metadata_and_reloads_weights(self):
        namespace = dict(nn=nn, torch=torch, os=os)
        base = load_symbols('unirank/pytorch/models/rank_model.py', ['BaseModel'], namespace)['BaseModel']
        class Tiny(base):
            def __init__(self):
                nn.Module.__init__(self)
                self.device = torch.device('cpu')
                self.feature_map = SimpleNamespace(labels=['is_click'], features={'user_id': {'type': 'categorical'}})
                self.linear = nn.Linear(2, 1)
        model = Tiny()
        batch = ({'user_id': torch.tensor([1]), 'is_click': torch.tensor([0.]),
                  'history_length_input': torch.tensor([0])}, {'item_id': torch.tensor([[0, 1]])}, torch.zeros(1, 1))
        user, _, _ = model.get_inputs(batch)
        self.assertEqual(set(user), {'user_id'})
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'missing' / 'nested' / 'weights.model'
            before = model.linear.weight.detach().clone()
            model.save_weights(path)
            with torch.no_grad():
                model.linear.weight.add_(10)
            model.load_weights(path)
            torch.testing.assert_close(model.linear.weight, before, rtol=0, atol=0)

    def test_rankmixer_and_attention_shapes_masks_and_gradients(self):
        namespace = dict(torch=torch, nn=nn, F=F)
        for path, name in [
            ('unirank/pytorch/layers/interactions/token_mix.py', 'MultiHeadTokenMixing'),
            ('unirank/pytorch/layers/interactions/per_token_ffn.py', 'PerTokenFeedForward'),
            ('model_zoo/RankMixer.py', 'RankMixerBlock'),
            ('unirank/pytorch/layers/attentions/dot_product_attention.py', 'ScaledDotProductAttention'),
            ('unirank/pytorch/layers/attentions/target_attention.py', 'MultiHeadTargetAttention')]:
            load_symbols(path, [name], namespace)
        torch.manual_seed(20262027)
        block = namespace['RankMixerBlock'](256, 4, 3, expand=2)
        x = torch.randn(2, 4, 256, requires_grad=True)
        output = block(x)
        output.square().mean().backward()
        self.assertEqual(tuple(output.shape), (2, 4, 256))
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters()))
        attention = namespace['MultiHeadTargetAttention'](112, 256).eval()
        query = torch.randn(3, 112, requires_grad=True)
        history = torch.randn(3, 100, 112, requires_grad=True)
        mask = torch.zeros(3, 100)
        mask[1, -1] = 1
        mask[2] = 1
        output = attention(query, history, mask)
        changed = history.detach().clone()
        changed[mask == 0] = 1000
        torch.testing.assert_close(output, attention(query.detach(), changed, mask))
        self.assertEqual(output[0].count_nonzero().item(), 0)
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(query.grad).all() and torch.isfinite(history.grad).all())


if __name__ == '__main__':
    unittest.main()
