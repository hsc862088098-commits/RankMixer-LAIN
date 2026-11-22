"""History selection shared by the loader and the Taobao audit.

This module deliberately has no model or Parquet dependencies.
"""

import numpy as np


HISTORY_METADATA_KEYS = frozenset({
    "history_length_before_filter", "history_length_prefix",
    "history_length_input", "history_ties_removed",
})


def validate_history_policy(policy):
    if policy not in {"original", "exposure_only"}:
        raise ValueError(f"Unknown history_feedback_policy: {policy!r}")
    return policy


def paired_part_ids(data_parts, user_parts, item_parts):
    groups = [set(parts) for parts in (data_parts, user_parts, item_parts)]
    if not groups[0] or not (groups[0] == groups[1] == groups[2]):
        raise ValueError(
            "Blocked Parquet part IDs must match exactly: "
            f"data={sorted(groups[0])}, user_info={sorted(groups[1])}, "
            f"item_info={sorted(groups[2])}"
        )
    return sorted(groups[0])


def validate_user_sequence(items, timestamps):
    items, timestamps = np.asarray(items), np.asarray(timestamps)
    if items.ndim != 1 or timestamps.ndim != 1 or len(items) != len(timestamps):
        raise ValueError("Item and exposure timestamp sequences must be aligned 1D arrays")
    if not np.issubdtype(items.dtype, np.integer) or np.any(items <= 0):
        raise ValueError("Full history item indices must be positive integers (0 is padding)")
    if not np.issubdtype(timestamps.dtype, np.integer):
        raise ValueError("Exposure timestamps must be integers in the metadata timestamp unit")
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("Exposure timestamps must be monotonically nondecreasing")


def strict_prefix_length(timestamps, position):
    """Exclude the target, future events, and all ties with the target time."""
    if int(position) != position or not 0 <= position < len(timestamps):
        raise ValueError(f"seq_len={position} does not index a target exposure")
    position = int(position)
    return int(np.searchsorted(timestamps[:position], timestamps[position], side="left"))


def exposure_history_batch(item_sequences, timestamp_sequences, positions,
                           target_items, max_len, exposure_id, padding="pre"):
    """Build feedback-independent inputs; timestamps must be validated at load time."""
    if max_len <= 0 or exposure_id <= 0 or padding not in {"pre", "post"}:
        raise ValueError("Invalid max_len, exposure action ID, or padding mode")
    count = len(positions)
    if not all(len(values) == count for values in
               (item_sequences, timestamp_sequences, target_items)):
        raise ValueError("History batch arrays have inconsistent lengths")
    items = np.zeros((count, max_len), dtype=np.int64)
    before = np.asarray(positions)
    if np.any(before != before.astype(np.int64)):
        raise ValueError("seq_len must contain integers")
    before = before.astype(np.int64)
    prefixes = np.empty(count, dtype=np.int64)
    for row, position in enumerate(before):
        sequence, times = item_sequences[row], timestamp_sequences[row]
        stop = strict_prefix_length(times, position)
        if int(sequence[position]) != int(target_items[row]):
            raise ValueError("Target item does not match user history at seq_len")
        prefixes[row] = stop
        history = np.asarray(sequence[max(0, stop - max_len):stop], dtype=np.int64)
        if len(history):
            start = max_len - len(history) if padding == "pre" else 0
            items[row, start:start + len(history)] = history
    valid = items > 0
    actions = np.where(valid, exposure_id, 0).astype(np.int64)
    metadata = {
        "history_length_before_filter": before,
        "history_length_prefix": prefixes,
        "history_length_input": valid.sum(axis=1).astype(np.int64),
        "history_ties_removed": before - prefixes,
    }
    return items, actions, metadata
