#!/usr/bin/env python
"""Exact source-frame splits used by HIPPYNN's methane training database."""

from __future__ import annotations

import hashlib

import numpy as np
import torch


def make_hippynn_splits(data_size: int, seed: int, external_size: int = 80_000) -> dict[str, np.ndarray]:
    """Reproduce Database.make_trainvalidtest_split(test_size=.1, valid_size=.1).

    HIPPYNN draws internal test first, removes it, then draws validation using
    0.1 / 0.9 of the remainder. Each selected split is sorted by source index.
    """
    if data_size <= 0 or external_size <= 0:
        raise ValueError("data_size and external_size must be positive")
    generator = torch.Generator().manual_seed(seed)
    remaining = torch.arange(data_size, dtype=torch.int64)

    test_count = int(0.1 * len(remaining))
    permutation = torch.randperm(len(remaining), generator=generator)
    internal_test = remaining[permutation[:test_count]].sort().values
    keep = ~torch.isin(remaining, internal_test)
    remaining = remaining[keep]

    valid_fraction = 0.1 / (1.0 - 0.1)
    valid_count = int(valid_fraction * len(remaining))
    permutation = torch.randperm(len(remaining), generator=generator)
    valid = remaining[permutation[:valid_count]].sort().values
    keep = ~torch.isin(remaining, valid)
    train = remaining[keep].sort().values

    return {
        "train": train.numpy(),
        "valid": valid.numpy(),
        "internal_test": internal_test.numpy(),
        "external_test": np.arange(data_size, data_size + external_size, dtype=np.int64),
    }


def index_sha256(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
