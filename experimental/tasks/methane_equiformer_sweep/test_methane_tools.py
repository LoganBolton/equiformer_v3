"""Fast regression tests for methane split and evaluation semantics."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from evaluate_predictions import compute_metrics  # noqa: E402
from hippynn_splits import index_sha256, make_hippynn_splits  # noqa: E402


class SplitTests(unittest.TestCase):
    def test_seed42_100k_known_hashes(self) -> None:
        splits = make_hippynn_splits(100_000, 42)
        self.assertEqual({name: len(value) for name, value in splits.items()}, {
            "train": 80_000,
            "valid": 10_000,
            "internal_test": 10_000,
            "external_test": 80_000,
        })
        self.assertEqual(index_sha256(splits["train"]), "f3698919c96ff93586067e9cbe41484e4dca69fbb0f9b24fc96f5c79c51decef")
        self.assertEqual(index_sha256(splits["valid"]), "8629cfd9ac8354b4f7bf7f274ecbecc46ca7c3c0ce901cda196f4c74cda9cffe")
        self.assertEqual(index_sha256(splits["internal_test"]), "240ef7c3984b00fc1d11e573bff8a7c35148feb64b239368aa14b2c0b02fd28e")
        self.assertEqual(index_sha256(splits["external_test"]), "aea4a4998328eca534782c713b83870fded8e7e501ee3dd931dbd3c303cca1ab")

    def test_splits_are_sorted_disjoint_and_complete(self) -> None:
        splits = make_hippynn_splits(101, 7, external_size=3)
        internal = [splits[name] for name in ("train", "valid", "internal_test")]
        for values in splits.values():
            self.assertTrue(np.all(values[:-1] < values[1:]))
        np.testing.assert_array_equal(np.sort(np.concatenate(internal)), np.arange(101))
        self.assertEqual(sum(len(np.intersect1d(a, b)) for i, a in enumerate(internal) for b in internal[i + 1 :]), 0)
        np.testing.assert_array_equal(splits["external_test"], np.arange(101, 104))


class MetricTests(unittest.TestCase):
    def test_componentwise_metrics(self) -> None:
        target_forces = np.arange(30, dtype=float).reshape(2, 5, 3)
        metrics = compute_metrics(
            np.array([10, 11]),
            np.array([1.0, 3.0]),
            np.array([2.0, 2.0]),
            target_forces,
            target_forces + 1.0,
        )
        self.assertEqual(metrics["num_force_components"], 30)
        self.assertEqual(metrics["energy_mae"], 1.0)
        self.assertEqual(metrics["energy_rmse"], 1.0)
        self.assertEqual(metrics["force_mae"], 1.0)
        self.assertEqual(metrics["force_rmse"], 1.0)

    def test_duplicate_source_indices_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates"):
            compute_metrics(
                np.array([4, 4]),
                np.zeros(2),
                np.zeros(2),
                np.zeros((2, 5, 3)),
                np.zeros((2, 5, 3)),
            )

    def test_vector_norm_is_not_componentwise_mae(self) -> None:
        target = np.zeros((1, 1, 3))
        prediction = np.ones((1, 1, 3))
        metrics = compute_metrics(np.array([0]), np.array([0.0]), np.array([1.0]), target, prediction)
        self.assertEqual(metrics["force_mae"], 1.0)
        self.assertNotEqual(metrics["force_mae"], float(np.linalg.norm(prediction[0, 0])))
        self.assertIsNone(metrics["energy_r2"])


if __name__ == "__main__":
    unittest.main()
