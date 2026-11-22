"""Create a deterministic, separate Taobao smoke dataset from local files."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from unirank.taobao import prepare_smoke_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--sizes', type=int, nargs=3, default=(8192, 4096, 4096),
                        metavar=('TRAIN', 'VALID', 'TEST'))
    parser.add_argument('--seed', type=int, default=20262027)
    args = parser.parse_args()
    print(json.dumps(prepare_smoke_subset(args.source, args.destination, args.sizes, args.seed), indent=2))


if __name__ == '__main__':
    main()
