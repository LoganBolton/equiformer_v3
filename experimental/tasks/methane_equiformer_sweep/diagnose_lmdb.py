#!/usr/bin/env python
"""Validate methane LMDB labels, provenance, nonperiodicity, and geometry."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lmdb", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--radius", type=float, default=10.3)
    parser.add_argument("--allow-periodic", action="store_true", help="diagnostic only; valid methane exports are nonperiodic")
    args = parser.parse_args()
    env = lmdb.open(str(args.lmdb), subdir=False, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        length = pickle.loads(txn.get(b"length"))
        selected = sorted(set(np.linspace(0, length - 1, min(args.samples, length), dtype=int).tolist()))
        for row in selected:
            data = pickle.loads(txn.get(str(row).encode("ascii")))
            pos = torch.as_tensor(data.pos, dtype=torch.float64)
            natoms = int(data.natoms)
            distances = torch.cdist(pos, pos)
            mask = ~torch.eye(natoms, dtype=torch.bool)
            pair_distances = distances[mask]
            directed_edges = int((pair_distances < args.radius).sum())
            pbc = torch.as_tensor(data.pbc, dtype=torch.bool).reshape(-1)
            force_shape = tuple(torch.as_tensor(data.forces).shape)
            sid = int(data.sid)
            print(
                f"row={row} sid={sid} atoms={natoms} directed_edges={directed_edges} "
                f"min_distance={pair_distances.min():.8g} max_distance={pair_distances.max():.8g} "
                f"pbc={pbc.tolist()} energy={torch.as_tensor(data.energy).reshape(-1).tolist()} "
                f"force_shape={force_shape}"
            )
            if natoms != 5:
                raise SystemExit(f"ERROR: row {row} has {natoms} atoms, expected 5")
            if force_shape != (5, 3):
                raise SystemExit(f"ERROR: row {row} force shape is {force_shape}, expected (5, 3)")
            if pbc.any() and not args.allow_periodic:
                raise SystemExit(f"ERROR: row {row} is periodic")
            if not torch.isfinite(pos).all() or not torch.isfinite(torch.as_tensor(data.forces)).all():
                raise SystemExit(f"ERROR: row {row} contains non-finite positions/forces")
    env.close()
    print(f"PASS: checked {len(selected)} of {length} samples")


if __name__ == "__main__":
    main()
