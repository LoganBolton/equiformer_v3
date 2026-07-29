#!/usr/bin/env python
"""Independently reproduce HIPPYNN splits and compare saved Equiformer indices."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from hippynn_splits import index_sha256, make_hippynn_splits


def independent_hippynn_splits(data_size: int, split_seed: int, external_size: int) -> dict[str, np.ndarray]:
    generator = torch.Generator().manual_seed(split_seed)
    pool = torch.arange(data_size, dtype=torch.int64)
    n_internal_test = int(0.1 * data_size)
    internal_test = pool[torch.randperm(pool.numel(), generator=generator)[:n_internal_test]].sort().values
    remaining = pool[~torch.isin(pool, internal_test)]
    n_valid = int((0.1 / 0.9) * remaining.numel())
    valid = remaining[torch.randperm(remaining.numel(), generator=generator)[:n_valid]].sort().values
    train = remaining[~torch.isin(remaining, valid)].sort().values
    return {
        "train": train.numpy(),
        "valid": valid.numpy(),
        "internal_test": internal_test.numpy(),
        "external_test": np.arange(data_size, data_size + external_size, dtype=np.int64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-size", type=int, required=True)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--external-test-size", type=int, default=80_000)
    parser.add_argument("--indices", type=Path)
    args = parser.parse_args()

    expected = independent_hippynn_splits(args.data_size, args.split_seed, args.external_test_size)
    actual = make_hippynn_splits(args.data_size, args.split_seed, args.external_test_size)
    if args.indices:
        with np.load(args.indices) as saved:
            actual = {name: saved[name] for name in expected}

    failed = False
    for name in expected:
        ordered = np.array_equal(actual[name], expected[name])
        membership = np.array_equal(np.sort(actual[name]), np.sort(expected[name]))
        print(
            f"{name}: count={len(actual[name])} ordered_equal={ordered} "
            f"set_equal={membership} sha256={index_sha256(actual[name])} "
            f"first={actual[name][:8].tolist()}"
        )
        failed |= not ordered
    if failed:
        raise SystemExit("ERROR: Equiformer and HIPPYNN split indices differ")
    print("PASS: exact ordered split equality")


if __name__ == "__main__":
    main()
