import unittest
import numpy as np

from unirank.history import (exposure_history_batch, paired_part_ids,
                             strict_prefix_length, validate_user_sequence)


class HistoryTests(unittest.TestCase):
    def batch(self, position, max_len=2, padding='pre'):
        return exposure_history_batch([[1, 2, 3, 4]], [[100, 100, 200, 300]],
                                      [position], [position + 1], max_len, 6, padding)

    def test_ties_are_removed_before_truncation(self):
        items, actions, lengths = self.batch(1)
        np.testing.assert_array_equal(items, [[0, 0]])
        np.testing.assert_array_equal(actions, [[0, 0]])
        self.assertEqual(lengths['history_ties_removed'].tolist(), [1])
        self.assertEqual(lengths['history_length_before_filter'].tolist(), [1])
        self.assertEqual(lengths['history_length_prefix'].tolist(), [0])

    def test_recent_history_and_lengths(self):
        items, actions, lengths = self.batch(3)
        np.testing.assert_array_equal(items, [[2, 3]])
        np.testing.assert_array_equal(actions, [[6, 6]])
        self.assertEqual(lengths['history_length_prefix'].tolist(), [3])
        self.assertEqual(lengths['history_length_input'].tolist(), [2])

    def test_padding_both_directions(self):
        for padding, expected in [('pre', [[0, 0, 1, 2]]), ('post', [[1, 2, 0, 0]])]:
            with self.subTest(padding=padding):
                items, actions, _ = self.batch(2, 4, padding)
                np.testing.assert_array_equal(items, expected)
                np.testing.assert_array_equal(actions > 0, items > 0)

    def test_empty_history(self):
        items, actions, lengths = self.batch(0)
        self.assertEqual(int(items.sum() + actions.sum()), 0)
        self.assertEqual(lengths['history_length_input'].tolist(), [0])

    def test_invalid_target_and_positions(self):
        for position in [-1, 4, 1.5]:
            with self.subTest(position=position), self.assertRaises(ValueError):
                strict_prefix_length([100, 100, 200, 300], position)
        with self.assertRaises(ValueError):
            exposure_history_batch([[1, 2]], [[100, 200]], [1], [1], 2, 6)

    def test_invalid_timestamp_sequences(self):
        for items, times in [([1, 2], [200, 100]), ([1], [100, 200]),
                             ([0], [100]), ([1], [100.5])]:
            with self.subTest(items=items, times=times), self.assertRaises(ValueError):
                validate_user_sequence(items, times)

    def test_part_mismatch_does_not_silently_drop_data(self):
        self.assertEqual(paired_part_ids({0: 'a'}, {0: 'b'}, {0: 'c'}), [0])
        with self.assertRaises(ValueError):
            paired_part_ids({0: 'a', 1: 'b'}, {0: 'c'}, {0: 'd'})


if __name__ == '__main__':
    unittest.main()
