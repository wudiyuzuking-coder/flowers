"""Train-manifest-only class-weight statistics for the E6 experiment."""

from pathlib import Path, PurePosixPath

import numpy as np


def load_train_class_weights(train_manifest_path, class_names):
    """Return train counts and normalized inverse-frequency weights in label order."""
    train_manifest_path = Path(train_manifest_path)
    if not train_manifest_path.is_file():
        raise FileNotFoundError(f"Missing train manifest: {train_manifest_path}")
    if not class_names or len(class_names) != len(set(class_names)):
        raise ValueError("class_names must be non-empty and unique.")

    entries = [
        line.strip()
        for line in train_manifest_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    if not entries:
        raise ValueError("Train manifest must contain at least one sample.")
    if len(entries) != len(set(entries)):
        raise ValueError("Duplicate path found in train manifest.")

    counts = {class_name: 0 for class_name in class_names}
    for relative_path in entries:
        path = PurePosixPath(relative_path)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError(f"Unsafe train manifest path: {relative_path}")
        if not path.parts or path.parts[0] not in counts:
            raise ValueError(f"Unknown class in train manifest path: {relative_path}")
        counts[path.parts[0]] += 1

    if sum(counts.values()) != len(entries):
        raise ValueError("Class counts do not cover the complete train manifest.")
    count_array = np.asarray(
        [counts[class_name] for class_name in class_names], dtype=np.int64
    )
    if np.any(count_array <= 0):
        missing = [
            class_name
            for class_name, count in zip(class_names, count_array)
            if count <= 0
        ]
        raise ValueError(f"Every class must occur in train.txt; missing={missing}")

    sample_count = int(count_array.sum())
    class_count = len(class_names)
    weights = (sample_count / (class_count * count_array.astype(np.float64))).astype(
        np.float32
    )
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("Class weights must be finite and strictly positive.")
    return counts, weights
