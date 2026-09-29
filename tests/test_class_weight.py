"""Pure-Python tests for E6 class-weight statistics and weighted reduction math."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from class_weight import load_train_class_weights


CLASS_NAMES = ('daisy', 'dandelion', 'roses', 'sunflowers', 'tulips')


class ClassWeightTests(unittest.TestCase):
    def test_repository_train_manifest_counts_and_formula(self):
        manifest = Path(__file__).resolve().parents[1] / 'splits' / 'train.txt'
        entries = [
            line.strip()
            for line in manifest.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        independent_counts = Counter(entry.split('/', 1)[0] for entry in entries)
        counts, weights = load_train_class_weights(manifest, CLASS_NAMES)

        self.assertEqual(sum(counts.values()), len(entries))
        self.assertEqual(
            list(counts.values()),
            [independent_counts[class_name] for class_name in CLASS_NAMES],
        )
        expected = np.asarray(
            [
                len(entries) / (len(CLASS_NAMES) * counts[class_name])
                for class_name in CLASS_NAMES
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(weights, expected, rtol=0, atol=0)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights > 0))

    def test_normalized_inverse_frequency_and_weighted_mean(self):
        expected_counts = np.asarray([10, 20, 40, 20, 10], dtype=np.int64)
        entries = [
            f'{class_name}/{sample_index}.jpg'
            for class_name, count in zip(CLASS_NAMES, expected_counts)
            for sample_index in range(count)
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / 'train.txt'
            manifest.write_text('\n'.join(entries) + '\n', encoding='utf-8')
            counts, weights = load_train_class_weights(manifest, CLASS_NAMES)

        expected_weights = (
            len(entries) / (len(CLASS_NAMES) * expected_counts.astype(np.float64))
        ).astype(np.float32)
        self.assertEqual(list(counts.values()), expected_counts.tolist())
        self.assertEqual(weights.dtype, np.float32)
        np.testing.assert_allclose(weights, expected_weights, rtol=0, atol=0)
        self.assertTrue(np.all(np.isfinite(weights)))
        self.assertTrue(np.all(weights > 0))
        np.testing.assert_allclose(
            expected_counts * weights,
            np.full(len(CLASS_NAMES), len(entries) / len(CLASS_NAMES)),
            rtol=1e-6,
        )
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[1], weights[2])

        per_sample_ce = np.asarray([0.2, 0.4, 0.8], dtype=np.float32)
        labels = np.asarray([0, 2, 4], dtype=np.int32)
        weighted_mean = np.mean(per_sample_ce * weights[labels])
        expected_mean = np.mean(
            [0.2 * expected_weights[0], 0.4 * expected_weights[2], 0.8 * expected_weights[4]]
        )
        self.assertAlmostEqual(float(weighted_mean), float(expected_mean), places=7)

    def test_rejects_a_class_missing_from_train_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            manifest = Path(temporary_directory) / 'train.txt'
            manifest.write_text(
                '\n'.join(f'{name}/a.jpg' for name in CLASS_NAMES[:-1]) + '\n',
                encoding='utf-8',
            )
            with self.assertRaisesRegex(ValueError, 'Every class must occur'):
                load_train_class_weights(manifest, CLASS_NAMES)


if __name__ == '__main__':
    unittest.main()
