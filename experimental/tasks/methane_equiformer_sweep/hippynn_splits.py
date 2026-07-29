#!/usr/bin/env python
"""Exact source-frame splits used by HIPPYNN's methane training database."""

from __future__ import annotations

import hashlib

import numpy as np
import torch


def make_hippynn_splits(
    data_size: int,
    split_seed: int,
    external_size: int = 80_000,
) -> dict[str, np.ndarray]:
    """Reproduce HIPPYNN's methane split construction exactly.

    HIPPYNN uses ``Database(seed=model_seed, test_size=0.1, valid_size=0.1)``.
    Its split logic:

    1. Build the development pool from the first ``data_size`` sequential frames.
    2. Draw the internal test set first with ``torch.randperm``.
    3. Remove that set from the pool.
    4. Draw validation with fraction ``0.1 / 0.9`` using the same generator.
    5. Sort the selected indices within each split.
    6. Use the sorted remainder as train.
    7. Use the next ``external_size`` sequential frames as the external test set.
    """
    if data_size <= 0 or external_size <= 0:
        raise ValueError("data_size and external_size must be positive")

    generator = torch.Generator().manual_seed(split_seed)
    remaining = torch.arange(data_size, dtype=torch.int64)

    internal_test_count = int(0.1 * len(remaining))
    permutation = torch.randperm(len(remaining), generator=generator)
    internal_test = remaining[permutation[:internal_test_count]].sort().values
    remaining = remaining[~torch.isin(remaining, internal_test)]

    validation_fraction = 0.1 / (1.0 - 0.1)
    validation_count = int(validation_fraction * len(remaining))
    permutation = torch.randperm(len(remaining), generator=generator)
    validation = remaining[permutation[:validation_count]].sort().values
    train = remaining[~torch.isin(remaining, validation)].sort().values

    return {
        "train": train.numpy(),
        "valid": validation.numpy(),
        "internal_test": internal_test.numpy(),
        "external_test": np.arange(data_size, data_size + external_size, dtype=np.int64),
    }


def index_sha256(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
