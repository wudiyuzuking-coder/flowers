"""Generate, validate, and load fixed stratified dataset split manifests."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path, PurePosixPath

import numpy as np


CLASS_NAMES = ('daisy', 'dandelion', 'roses', 'sunflowers', 'tulips')
CLASS_INDEXING = {name: index for index, name in enumerate(CLASS_NAMES)}
SPLIT_NAMES = ('train', 'val', 'test')
SPLIT_RATIOS = (0.7, 0.1, 0.2)
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def discover_samples(data_root):
    """Return sorted relative POSIX image paths grouped by class."""
    data_root = Path(data_root)
    samples_by_class = {}
    for class_name in CLASS_NAMES:
        class_directory = data_root / class_name
        if not class_directory.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_directory}")
        samples = sorted(
            path.relative_to(data_root).as_posix()
            for path in class_directory.rglob('*')
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not samples:
            raise ValueError(f"No supported images found for class: {class_name}")
        samples_by_class[class_name] = samples
    return samples_by_class


def _allocate_counts(sample_count):
    """Allocate integer 70/10/20 counts using the largest-remainder method."""
    raw_counts = [sample_count * ratio for ratio in SPLIT_RATIOS]
    counts = [math.floor(value) for value in raw_counts]
    remainder = sample_count - sum(counts)
    fractional_order = sorted(
        range(len(raw_counts)),
        key=lambda index: (-(raw_counts[index] - counts[index]), index),
    )
    for index in fractional_order[:remainder]:
        counts[index] += 1
    if sample_count >= len(SPLIT_NAMES) and any(count == 0 for count in counts):
        raise ValueError(f"Class with {sample_count} samples cannot populate every split.")
    return dict(zip(SPLIT_NAMES, counts))


def _validate_entries(split_entries, expected_samples):
    split_sets = {name: set(entries) for name, entries in split_entries.items()}
    for name, entries in split_entries.items():
        if len(entries) != len(split_sets[name]):
            raise ValueError(f"Duplicate path found inside {name} manifest.")
    for left_index, left_name in enumerate(SPLIT_NAMES):
        for right_name in SPLIT_NAMES[left_index + 1:]:
            overlap = split_sets[left_name] & split_sets[right_name]
            if overlap:
                raise ValueError(
                    f"Split overlap between {left_name} and {right_name}: "
                    f"{sorted(overlap)[:3]}"
                )
    manifest_union = set().union(*(split_sets[name] for name in SPLIT_NAMES))
    if manifest_union != set(expected_samples):
        missing = set(expected_samples) - manifest_union
        extra = manifest_union - set(expected_samples)
        raise ValueError(
            f"Manifest union mismatch: missing={len(missing)}, extra={len(extra)}"
        )


def _class_counts(entries):
    counts = {name: 0 for name in CLASS_NAMES}
    for relative_path in entries:
        parts = PurePosixPath(relative_path).parts
        if not parts or parts[0] not in counts:
            raise ValueError(f"Invalid class path in manifest: {relative_path}")
        counts[parts[0]] += 1
    return counts


def generate_split_manifest(data_root='flower_photos', output_dir='splits', seed=42, force=False):
    """Generate one immutable stratified split bundle, refusing overwrite by default."""
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    manifest_paths = {name: output_dir / f'{name}.txt' for name in SPLIT_NAMES}
    info_path = output_dir / 'split_info.json'
    existing = [path for path in (*manifest_paths.values(), info_path) if path.exists()]
    if existing and not force:
        existing_names = ', '.join(path.as_posix() for path in existing)
        raise FileExistsError(
            f"Fixed split already exists ({existing_names}). Use --force to replace it."
        )

    samples_by_class = discover_samples(data_root)
    split_entries = {name: [] for name in SPLIT_NAMES}
    for class_name in CLASS_NAMES:
        class_samples = list(samples_by_class[class_name])
        random.Random(f'{seed}:{class_name}').shuffle(class_samples)
        counts = _allocate_counts(len(class_samples))
        train_end = counts['train']
        val_end = train_end + counts['val']
        split_entries['train'].extend(class_samples[:train_end])
        split_entries['val'].extend(class_samples[train_end:val_end])
        split_entries['test'].extend(class_samples[val_end:])

    # Keep membership stratified while avoiding class-blocked manifest ordering.
    for split_name in SPLIT_NAMES:
        random.Random(f'{seed}:split:{split_name}').shuffle(split_entries[split_name])

    expected_samples = [
        relative_path
        for class_name in CLASS_NAMES
        for relative_path in samples_by_class[class_name]
    ]
    _validate_entries(split_entries, expected_samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, path in manifest_paths.items():
        manifest_text = '\n'.join(split_entries[split_name]) + '\n'
        path.write_bytes(manifest_text.encode('utf-8'))

    split_class_counts = {
        name: _class_counts(entries) for name, entries in split_entries.items()
    }
    info = {
        'seed': seed,
        'split_ratios': dict(zip(SPLIT_NAMES, SPLIT_RATIOS)),
        'data_root': data_root.name,
        'total_samples': len(expected_samples),
        'class_counts': {
            name: len(samples_by_class[name]) for name in CLASS_NAMES
        },
        'split_counts': {
            name: len(split_entries[name]) for name in SPLIT_NAMES
        },
        'split_class_counts': split_class_counts,
        'manifest_sha256': {
            path.name: _sha256(path) for path in manifest_paths.values()
        },
    }
    info_text = json.dumps(info, ensure_ascii=False, indent=2) + '\n'
    info_path.write_bytes(info_text.encode('utf-8'))
    return info


def load_and_validate_split_manifests(data_root='flower_photos', splits_dir='splits'):
    """Load manifests and reject hash, overlap, coverage, label, or dataset drift."""
    data_root = Path(data_root)
    splits_dir = Path(splits_dir)
    info_path = splits_dir / 'split_info.json'
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Missing {info_path}. Run: python tools/generate_split_manifest.py"
        )
    info = json.loads(info_path.read_text(encoding='utf-8'))
    if info.get('seed') != 42:
        raise ValueError(f"Expected split seed 42, found {info.get('seed')}")
    expected_ratios = dict(zip(SPLIT_NAMES, SPLIT_RATIOS))
    if info.get('split_ratios') != expected_ratios:
        raise ValueError(
            f"Expected split ratios {expected_ratios}, found {info.get('split_ratios')}"
        )
    if Path(info.get('data_root', '')).is_absolute():
        raise ValueError("split_info.json must not contain an absolute data_root.")

    entries = {}
    for split_name in SPLIT_NAMES:
        manifest_path = splits_dir / f'{split_name}.txt'
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing split manifest: {manifest_path}")
        expected_hash = info.get('manifest_sha256', {}).get(manifest_path.name)
        actual_hash = _sha256(manifest_path)
        if expected_hash != actual_hash:
            raise ValueError(f"SHA-256 mismatch for {manifest_path}")
        entries[split_name] = [
            line.strip()
            for line in manifest_path.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]

    discovered = discover_samples(data_root)
    expected_samples = [
        relative_path
        for class_name in CLASS_NAMES
        for relative_path in discovered[class_name]
    ]
    _validate_entries(entries, expected_samples)
    if info.get('total_samples') != len(expected_samples):
        raise ValueError("split_info.json total_samples does not match the dataset.")
    if info.get('class_counts') != {
        name: len(discovered[name]) for name in CLASS_NAMES
    }:
        raise ValueError("split_info.json class_counts do not match the dataset.")
    if info.get('split_counts') != {
        name: len(entries[name]) for name in SPLIT_NAMES
    }:
        raise ValueError("split_info.json split_counts do not match the manifests.")
    if info.get('split_class_counts') != {
        name: _class_counts(entries[name]) for name in SPLIT_NAMES
    }:
        raise ValueError("split_info.json split_class_counts do not match the manifests.")

    resolved_root = data_root.resolve()
    for split_name in SPLIT_NAMES:
        for relative_path in entries[split_name]:
            pure_path = PurePosixPath(relative_path)
            if pure_path.is_absolute() or '..' in pure_path.parts:
                raise ValueError(f"Unsafe relative path: {relative_path}")
            image_path = data_root.joinpath(*pure_path.parts).resolve()
            try:
                image_path.relative_to(resolved_root)
            except ValueError as error:
                raise ValueError(f"Path escapes data root: {relative_path}") from error
            if not image_path.is_file():
                raise FileNotFoundError(f"Manifest image is missing: {image_path}")
    return entries, info


class ManifestImageSource:
    """MindSpore GeneratorDataset source yielding encoded bytes and integer labels."""

    def __init__(self, data_root, relative_paths, class_indexing=None):
        self.data_root = Path(data_root)
        self.relative_paths = list(relative_paths)
        self.class_indexing = class_indexing or CLASS_INDEXING

    def __len__(self):
        return len(self.relative_paths)

    def __getitem__(self, index):
        relative_path = self.relative_paths[index]
        parts = PurePosixPath(relative_path).parts
        class_name = parts[0]
        label = self.class_indexing[class_name]
        image_path = self.data_root.joinpath(*parts)
        return np.fromfile(str(image_path), dtype=np.uint8), np.int32(label)
