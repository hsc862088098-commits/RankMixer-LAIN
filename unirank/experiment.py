"""Small, dependency-free helpers for reproducible experiment runs."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid


SOURCE_COMMIT = "90bb99a60724e975ec565836266fd86dcad0039d"
TAOBAO_REVISION = "fd01fb8edbfa58bb4b0d6556f1d5838074b0dbe5"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false, yes/no, or 1/0")


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./config/')
    parser.add_argument('--expid', default='RankMixer_KuaiRand_Video_Action')
    parser.add_argument('--gpu', default='0,1,2,3')
    parser.add_argument('--enable_bf16', type=parse_bool, default=None,
                        help='Override config BF16 using true or false')
    parser.add_argument('--run_id', default=None,
                        help='Unique output directory name; existing names are rejected')
    return parser


def base_metric_names(metrics, labels):
    """Convert selection keys such as is_click_AUC into metric names."""
    result = []
    for key in metrics:
        metric = key
        for label in sorted(labels, key=len, reverse=True):
            if key.startswith(label + "_"):
                metric = key[len(label) + 1:]
                break
        if metric not in result:
            result.append(metric)
    return result


def git_state(repo):
    def git(*arguments):
        return subprocess.check_output(
            ['git', '-C', str(repo), *arguments], text=True, stderr=subprocess.DEVNULL).strip()
    try:
        return {"commit": git('rev-parse', 'HEAD'),
                "dirty": bool(git('status', '--porcelain'))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def write_json_exclusive(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, default=str, allow_nan=False)
        stream.write('\n')


def prepare_run(params, config_dir, run_id=None, runtime=None):
    """Reserve a run directory atomically and snapshot resolved configuration."""
    params = dict(params)
    state = git_state(Path(__file__).resolve().parents[1])
    if params.get('require_clean_git', False) and (
            state['dirty'] is not False or state['commit'] is None):
        raise RuntimeError('Formal runs require a clean Git checkout with a readable commit')
    run_id = run_id or '{}_{}_s{}_{}'.format(
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'),
        params['model_id'], params['seed'], uuid.uuid4().hex[:8])
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,179}', run_id) or run_id in {'.', '..'}:
        raise ValueError('run_id must be a single safe directory name (at most 180 characters)')
    directory = Path(params.get('run_root', './experiments')).resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    params.update(run_id=run_id, run_dir=str(directory), model_root=str(directory))
    shutil.copytree(config_dir, directory / 'config')
    if params.get('data_audit_report'):
        shutil.copy2(params['data_audit_report'], directory / 'data_audit.json')
    packages = {}
    for name in ('torch', 'numpy', 'pandas', 'pyarrow', 'polars', 'heavyball', 'scikit-learn'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    write_json_exclusive(directory / 'run.json', {
        'status': 'prepared', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'git': state, 'upstream_commit': SOURCE_COMMIT,
        'data_revision_expected': params.get('data_revision'),
        'data_revision_verified': False,
        'python': sys.version, 'executable': sys.executable,
        'argv': sys.argv, 'working_directory': str(Path.cwd()),
        'packages': packages, 'runtime': runtime or {}, 'params': params,
    })
    return params
