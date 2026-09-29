"""Read-only readiness checks for formal E0-E6 experiment runs."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ('daisy', 'dandelion', 'roses', 'sunflowers', 'tulips')
MANIFEST_NAMES = ('train.txt', 'val.txt', 'test.txt', 'split_info.json')
OUTPUT_PATHS = (
    Path('checkpoints'),
    Path('results/evaluation'),
    Path('results/curves'),
    Path('results/confusion_matrix'),
)


class Reporter:
    def __init__(self):
        self.failures = 0
        self.warnings = 0

    def pass_(self, message):
        print(f'[PASS] {message}')

    def fail(self, message):
        self.failures += 1
        print(f'[FAIL] {message}')

    def warn(self, message):
        self.warnings += 1
        print(f'[WARN] {message}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Check E0-E6 experiment readiness without training or data changes.'
    )
    parser.add_argument(
        '--data-root',
        type=Path,
        default=REPO_ROOT / 'flower_photos',
        help='Flower dataset root. Defaults to <repo>/flower_photos.',
    )
    parser.add_argument(
        '--splits-dir',
        type=Path,
        default=REPO_ROOT / 'splits',
        help='Fixed split manifest directory. Defaults to <repo>/splits.',
    )
    parser.add_argument(
        '--pretrained-mode',
        choices=('online', 'offline'),
        default='online',
        help='How E4/E5/E6 will obtain the official MindCV checkpoint.',
    )
    parser.add_argument(
        '--pretrained-checkpoint',
        type=Path,
        help='Official MindCV ResNet18 ImageNet checkpoint for offline mode.',
    )
    return parser.parse_args()


def run_git(arguments):
    return subprocess.run(
        ['git', *arguments],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def check_git(reporter):
    try:
        commit = run_git(['rev-parse', 'HEAD'])
        reporter.pass_(f'Git commit SHA: {commit}')
        status = run_git(['status', '--porcelain'])
        if status:
            reporter.fail('Working tree is dirty; commit or account for every change first.')
            for line in status.splitlines():
                print(f'       {line}')
        else:
            reporter.pass_('Working tree is clean.')
    except (OSError, subprocess.CalledProcessError) as error:
        reporter.fail(f'Cannot inspect Git state: {error}')


def installed_version(distribution_name):
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_runtime(reporter):
    print(f'[INFO] Python: {sys.version.split()[0]} ({sys.executable})')
    dependencies = (
        ('mindspore', 'mindspore'),
        ('mindcv', 'mindcv'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('easydict', 'easydict'),
    )
    for module_name, distribution_name in dependencies:
        version = installed_version(distribution_name)
        available = importlib.util.find_spec(module_name) is not None
        if not available:
            reporter.fail(f'Runtime dependency is unavailable: {module_name}')
            continue
        reporter.pass_(f'{module_name} is installed (version={version or "unknown"}).')
        if module_name == 'mindcv' and version != '0.3.0':
            reporter.fail(f'MindCV must be 0.3.0 for E4/E5/E6; found {version}.')

    nvidia_smi = shutil.which('nvidia-smi')
    if not nvidia_smi:
        reporter.fail(
            'nvidia-smi is unavailable; GPU/driver readiness for device_target="GPU" '
            'cannot be confirmed.'
        )
        return
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                '--query-gpu=name,driver_version',
                '--format=csv,noheader',
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        gpu_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not gpu_lines:
            reporter.fail('nvidia-smi returned no GPU information.')
        else:
            reporter.pass_(f'GPU/driver reported by nvidia-smi: {"; ".join(gpu_lines)}')
            reporter.warn(
                'CUDA and MindSpore GPU compatibility still require a runtime smoke test.'
            )
    except (OSError, subprocess.CalledProcessError) as error:
        reporter.fail(f'nvidia-smi check failed: {error}')


def check_dataset_and_manifests(reporter, data_root, splits_dir):
    data_root = data_root.resolve()
    splits_dir = splits_dir.resolve()
    if not data_root.is_dir():
        reporter.fail(f'Dataset root does not exist: {data_root}')
        return
    reporter.pass_(f'Dataset root exists: {data_root}')

    missing_classes = [name for name in CLASS_NAMES if not (data_root / name).is_dir()]
    if missing_classes:
        reporter.fail(f'Missing class directories: {missing_classes}')
    else:
        reporter.pass_(f'All five class directories exist: {list(CLASS_NAMES)}')

    missing_manifests = [name for name in MANIFEST_NAMES if not (splits_dir / name).is_file()]
    if missing_manifests:
        reporter.fail(f'Missing split files in {splits_dir}: {missing_manifests}')
        return
    reporter.pass_(f'All fixed split files exist in {splits_dir}.')

    try:
        sys.path.insert(0, str(REPO_ROOT))
        from split_manifest import load_and_validate_split_manifests

        entries, info = load_and_validate_split_manifests(data_root, splits_dir)
        reporter.pass_(
            'Manifest SHA-256, labels, disjointness, union, sample files, and '
            'dataset drift checks passed.'
        )
        reporter.pass_(f'Split counts: {info["split_counts"]}')
        reporter.pass_(f'Train class counts: {info["split_class_counts"]["train"]}')
        print(f'[INFO] Manifest SHA-256: {info["manifest_sha256"]}')
        if set(entries) != {'train', 'val', 'test'}:
            reporter.fail(f'Unexpected split names: {sorted(entries)}')
    except Exception as error:  # The validator provides the actionable reason.
        reporter.fail(f'Fixed split validation failed: {error}')


def check_pretrained(reporter, mode, checkpoint):
    if mode == 'online':
        if checkpoint is not None:
            reporter.warn('--pretrained-checkpoint is ignored in online preflight mode.')
        reporter.warn(
            'Online E4/E5/E6 mode selected. Network access or a populated MindCV cache '
            'must be confirmed when the model is instantiated.'
        )
        return

    if checkpoint is None:
        reporter.fail('Offline pretrained mode requires --pretrained-checkpoint PATH.')
        return
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        reporter.fail(f'Offline pretrained checkpoint does not exist: {checkpoint}')
        return
    if checkpoint.stat().st_size <= 0:
        reporter.fail(f'Offline pretrained checkpoint is empty: {checkpoint}')
        return
    reporter.pass_(
        f'Offline pretrained checkpoint exists and is non-empty: {checkpoint} '
        f'({checkpoint.stat().st_size} bytes)'
    )
    reporter.warn(
        'Exact MindCV ResNet18 parameter compatibility is enforced by ms3.py at model '
        'load time and still requires a MindSpore runtime smoke test.'
    )


def check_output_paths(reporter):
    invalid_paths = [
        str(REPO_ROOT / path)
        for path in OUTPUT_PATHS
        if (REPO_ROOT / path).exists() and not (REPO_ROOT / path).is_dir()
    ]
    if invalid_paths:
        reporter.fail(f'Output paths exist but are not directories: {invalid_paths}')
        return
    try:
        with tempfile.TemporaryDirectory(prefix='flower-preflight-', dir=REPO_ROOT) as root:
            probe_root = Path(root)
            for relative_path in OUTPUT_PATHS:
                (probe_root / relative_path).mkdir(parents=True, exist_ok=False)
        reporter.pass_(
            'Checkpoint/result directory creation probe passed in a temporary directory.'
        )
    except OSError as error:
        reporter.fail(f'Cannot create required output directory structure: {error}')


def main():
    args = parse_args()
    reporter = Reporter()
    print('Flower E0-E6 experiment preflight (read-only; no training)')
    print(f'[INFO] Repository: {REPO_ROOT}')
    check_git(reporter)
    check_runtime(reporter)
    check_dataset_and_manifests(reporter, args.data_root, args.splits_dir)
    check_pretrained(reporter, args.pretrained_mode, args.pretrained_checkpoint)
    check_output_paths(reporter)
    print(f'[INFO] Warnings: {reporter.warnings}')
    if reporter.failures:
        print(f'RESULT: FAIL ({reporter.failures} failed check(s))')
        return 1
    print('RESULT: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
