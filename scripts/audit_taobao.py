"""Audit local Taobao data; no downloads or model training."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unirank.experiment import write_json_exclusive
from unirank.taobao import audit_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-len', type=int, default=100)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use a new audit report path')
    report = audit_dataset(args.data, args.max_len)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(args.output, report)
    print(f'Audit passed: {args.output.resolve()}')
    for split, stats in report['splits'].items():
        print(split, stats)


if __name__ == '__main__':
    main()
