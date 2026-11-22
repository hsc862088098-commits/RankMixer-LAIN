"""Offline audit and deterministic smoke-subset preparation for Taobao.

Parquet dependencies are imported only when accessing data, so --help and
dependency-free history tests also work on a development-only machine.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from unirank.experiment import TAOBAO_REVISION, write_json_exclusive
from unirank.history import paired_part_ids, strict_prefix_length, validate_user_sequence


LABELS = ["is_click", "cart", "fav", "buy"]
SPLITS = ("train", "valid", "test")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def part_maps(root, split):
    result = []
    for kind in ('data', 'user_info', 'item_info'):
        parts = {}
        for path in sorted((Path(root) / split / kind).glob('part-*.parquet')):
            number = path.stem.removeprefix('part-')
            if not number.isdigit() or int(number) in parts:
                raise ValueError(f'Invalid or duplicate part ID: {path}')
            parts[int(number)] = path
        result.append(parts)
    paired_part_ids(*result)
    return result


def _check_categorical(frame, vocabs):
    for column in set(frame.columns).intersection(vocabs):
        if column in {'user_index', 'item_index', 'timestamp'}:
            continue  # block-local indices are checked against their side tables
        values = frame[column].to_numpy()
        if np.any(values != values.astype(np.int64)) or np.any(values < 0) or np.any(values >= vocabs[column]):
            raise ValueError(f'Out-of-vocabulary or noninteger values in {column}')


def audit_dataset(root, max_len=100):
    import pandas as pd
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    if (root / 'PREPARATION_INCOMPLETE').exists():
        raise ValueError('Smoke preparation is incomplete; do not audit or train this directory')
    if max_len <= 0:
        raise ValueError('max_len must be positive')
    meta = json.loads((root / 'meta_data.json').read_text(encoding='utf-8'))
    if meta.get('label') != LABELS:
        raise ValueError(f'Task order must be {LABELS}')
    action_vocab = meta.get('action_vocab', {})
    if int(action_vocab.get('exposure', 0)) <= 0:
        raise ValueError('Missing positive exposure action ID')
    action_ids = {int(value) for value in action_vocab.values()}
    if (len(action_ids) != len(action_vocab) or min(action_ids) <= 0
            or max(action_ids) >= meta['vocab_size']['action']):
        raise ValueError('Action codes must be unique positive integers')
    report = {'status': 'passed', 'dataset_root': str(root), 'max_len': max_len,
              'history_feedback_policy': 'exposure_only',
              'data_revision_expected': TAOBAO_REVISION, 'data_revision_verified': False,
              'feedback_event_timestamps_available': False,
              'limitation': 'Exposure timestamps do not establish feedback availability.',
              'splits': {}, 'files': []}
    report['files'].append({'path': 'meta_data.json', 'size': (root / 'meta_data.json').stat().st_size,
                            'sha256': sha256_file(root / 'meta_data.json')})
    train_lengths = Counter()
    for split in SPLITS:
        data_parts, users_parts, items_parts = part_maps(root, split)
        rows = 0
        positives = np.zeros(len(LABELS), dtype=np.int64)
        ties_removed = 0
        affected = 0
        min_prefix, max_prefix = None, 0
        before_sum = prefix_sum = input_sum = 0
        seen_users = set()
        for part in data_parts:
            users = pd.read_parquet(users_parts[part])
            items = pd.read_parquet(items_parts[part])
            if users['user_index'].duplicated().any() or items['item_index'].duplicated().any():
                raise ValueError(f'Duplicate side-table index: {split}/{part}')
            _check_categorical(items, meta['vocab_size'])
            item_indices = set(items['item_index'].astype(int))
            user_lookup = {}
            for _, user in users.iterrows():
                sequence, timestamps = user['full_item_seq'], user['full_timestamp_seq']
                validate_user_sequence(sequence, timestamps)
                if not set(int(item) for item in sequence).issubset(item_indices):
                    raise ValueError(f'History references missing item: {split}/{part}')
                actions = user['full_action_seq']
                if len(actions) != len(sequence) or not set(int(a) for a in actions).issubset(action_ids):
                    raise ValueError(f'Invalid action sequence: {split}/{part}')
                user_lookup[int(user['user_index'])] = (sequence, timestamps)
            identities = []
            block_users = set()
            local_to_global = {}
            for record_batch in pq.ParquetFile(data_parts[part]).iter_batches(batch_size=65536):
                frame = record_batch.to_pandas()
                _check_categorical(frame, meta['vocab_size'])
                labels = frame[LABELS].to_numpy()
                if not np.isin(labels, (0, 1)).all():
                    raise ValueError(f'Non-binary or missing labels: {split}/{part}')
                positives += labels.sum(axis=0).astype(np.int64)
                identities.append(frame[['user_index', 'seq_len']].to_numpy())
                block_users.update(int(u) for u in frame['user_id'].unique())
                for user, position, target, global_user in frame[
                        ['user_index', 'seq_len', 'item_index', 'user_id']].itertuples(index=False, name=None):
                    if int(user) != user or int(target) != target:
                        raise ValueError('Noninteger block-local index')
                    sequence, timestamps = user_lookup[int(user)]
                    stop = strict_prefix_length(timestamps, position)
                    position = int(position)
                    if int(sequence[position]) != int(target):
                        raise ValueError(f'Target mismatch: {split}/{part}, user={user}, position={position}')
                    if local_to_global.setdefault(int(user), int(global_user)) != int(global_user):
                        raise ValueError('A local user index maps to multiple global users')
                    removed = position - stop
                    ties_removed += removed
                    affected += int(removed > 0)
                    before_sum += position
                    prefix_sum += stop
                    input_sum += min(stop, max_len)
                    min_prefix = stop if min_prefix is None else min(min_prefix, stop)
                    max_prefix = max(max_prefix, stop)
                    if split == 'train':
                        train_lengths[stop] += 1
                rows += len(frame)
            if identities:
                identities = np.concatenate(identities)
                if len(np.unique(identities, axis=0)) != len(identities):
                    raise ValueError(f'Duplicate target identity: {split}/{part}')
            if seen_users.intersection(block_users):
                raise ValueError(f'A user occurs in multiple blocks within {split}')
            seen_users.update(block_users)
            for path in (data_parts[part], users_parts[part], items_parts[part]):
                report['files'].append({'path': str(path.relative_to(root)).replace('\\', '/'),
                                        'size': path.stat().st_size, 'sha256': sha256_file(path)})
        if rows != meta['sample_size'][split]:
            raise ValueError(f'{split} row count {rows} != metadata {meta["sample_size"][split]}')
        if not rows or np.any(positives == 0) or np.any(positives == rows):
            raise ValueError(f'{split} must contain both classes for every task')
        report['splits'][split] = {
            'rows': rows, 'users': len(seen_users), 'parts': len(data_parts),
            'positive_counts': dict(zip(LABELS, positives.tolist())),
            'tie_affected_targets': affected, 'history_ties_removed': ties_removed,
            'mean_length_before_filter': before_sum / rows,
            'mean_L_prefix': prefix_sum / rows, 'mean_L_input': input_sum / rows,
            'min_L_prefix': min_prefix, 'max_L_prefix': max_prefix,
        }
    if sum(value['rows'] for value in report['splits'].values()) != meta['sample_size']['total']:
        raise ValueError('Total row count does not match metadata')
    # Exact sample-weighted linear quantiles from a compact integer histogram.
    values = sorted(train_lengths)
    cumulative = np.cumsum([train_lengths[v] for v in values])
    def at_rank(rank):
        return values[int(np.searchsorted(cumulative, rank + 1))]
    quantiles = []
    for q in (.25, .5, .75):
        rank = (int(cumulative[-1]) - 1) * q
        low, high = int(np.floor(rank)), int(np.ceil(rank))
        quantiles.append(at_rank(low) + (at_rank(high) - at_rank(low)) * (rank - low))
    report['train_L_prefix_quartiles'] = quantiles
    report['length_group_boundaries'] = sorted(set(quantiles))
    return report


def validate_audit_report(report_path, root, max_len):
    root = Path(root).resolve()
    if (root / 'PREPARATION_INCOMPLETE').exists():
        raise ValueError('Smoke preparation is incomplete')
    report = json.loads(Path(report_path).read_text(encoding='utf-8'))
    if (report.get('status') != 'passed' or Path(report['dataset_root']).resolve() != root
            or report.get('history_feedback_policy') != 'exposure_only'
            or report.get('max_len') != max_len or not report.get('files')):
        raise ValueError('Audit report does not match the dataset/history policy/max_len')
    for entry in report['files']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root) or path.stat().st_size != entry['size'] or sha256_file(path) != entry['sha256']:
            raise ValueError(f'Audited file changed or escaped the dataset root: {entry["path"]}')
    expected_files = {'meta_data.json'}
    for split in SPLITS:
        for parts in part_maps(root, split):
            expected_files.update(str(path.relative_to(root)).replace('\\', '/') for path in parts.values())
    if expected_files != {entry['path'] for entry in report['files']}:
        raise ValueError('Audit does not cover the exact current set of dataset files')
    return sha256_file(report_path)


def prepare_smoke_subset(source, destination, sizes=(8192, 4096, 4096), seed=20262027):
    import pandas as pd

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if len(sizes) != 3 or any(size <= 0 for size in sizes):
        raise ValueError('Specify three positive subset sizes')
    if destination == source or destination.is_relative_to(source):
        raise ValueError('The subset must be outside the source dataset')
    meta = json.loads((source / 'meta_data.json').read_text(encoding='utf-8'))
    if meta.get('label') != LABELS:
        raise ValueError('Unexpected Taobao label order')
    destination.mkdir(parents=True, exist_ok=False)
    marker = destination / 'PREPARATION_INCOMPLETE'
    marker.write_text('Do not train until preparation finishes.\n', encoding='utf-8')
    manifest = {'seed': seed, 'source': str(source), 'source_revision_expected': TAOBAO_REVISION,
                'smoke_only': True, 'splits': {}}
    for split_number, (split, requested) in enumerate(zip(SPLITS, sizes)):
        maps = part_maps(source, split)
        count = 0
        positives = np.zeros(4, dtype=np.int64)
        selected_parts = []
        for part, path in maps[0].items():
            frame = pd.read_parquet(path)
            if not len(frame):
                continue
            permutation = np.random.default_rng(seed + split_number * 100000 + part).permutation(len(frame))
            take = min(len(frame), max(1, requested - count))
            while True:
                subset = frame.iloc[np.sort(permutation[:take])]
                totals = positives + subset[LABELS].sum().to_numpy(dtype=np.int64)
                if (count + take >= requested and np.all(totals > 0)
                        and np.all(totals < count + take)) or take == len(frame):
                    break
                take = min(len(frame), max(take + 1, take * 2))
            for kind in ('data', 'user_info', 'item_info'):
                (destination / split / kind).mkdir(parents=True, exist_ok=True)
            subset.to_parquet(destination / split / 'data' / path.name, index=False)
            for kind, parts in zip(('user_info', 'item_info'), maps[1:]):
                shutil.copy2(parts[part], destination / split / kind / parts[part].name)
            count += take
            positives = totals
            selected_parts.append({'part_id': part, 'rows': take})
            if count >= requested and np.all(positives > 0) and np.all(positives < count):
                break
        if count < requested or np.any(positives == 0) or np.any(positives == count):
            raise ValueError(f'Cannot build a two-class smoke subset for {split}; partial output retained')
        meta['sample_size'][split] = count
        meta['blocked_layout'][split + '_blocks'] = len(selected_parts)
        manifest['splits'][split] = {'requested_rows': requested, 'rows': count, 'parts': selected_parts}
    meta['sample_size']['total'] = sum(meta['sample_size'][s] for s in SPLITS)
    meta['smoke_subset'] = manifest
    write_json_exclusive(destination / 'meta_data.json', meta)
    write_json_exclusive(destination / 'smoke_manifest.json', manifest)
    marker.unlink()
    return manifest
