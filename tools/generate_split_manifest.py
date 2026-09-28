"""CLI for generating the project's fixed stratified dataset split."""

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from split_manifest import generate_split_manifest  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the fixed stratified flower train/val/test manifests."
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=REPOSITORY_ROOT / 'flower_photos',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=REPOSITORY_ROOT / 'splits',
    )
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    info = generate_split_manifest(
        data_root=args.data_root,
        output_dir=args.output_dir,
        seed=42,
        force=args.force,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
