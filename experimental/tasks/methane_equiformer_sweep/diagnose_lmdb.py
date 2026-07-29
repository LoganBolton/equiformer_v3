#!/usr/bin/env python
"""Validate methane LMDB labels, provenance, nonperiodicity, and geometry."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch

EXPECTED_ATOMIC_NUMBERS = [6, 1, 1, 1, 1]
HARTREE_TO_KCAL_MOL = 627.5096080305927
HARTREE_PER_BOHR_TO_KCAL_MOL_PER_ANG = 51.42208619083232 * 23.060541945329334
HIPHOP_ENERGY_MEAN = -25042.327220945674


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lmdb", type=Path)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--radius", type=float, default=10.3)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    manifest = None
    if args.manifest:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        print(
            "manifest:",
            json.dumps(
                {
                    "dataset_name": manifest.get("dataset_name"),
                    "purpose": manifest.get("purpose"),
                    "data_size": manifest.get("data_size"),
                    "external_test_size": manifest.get("external_test_size"),
                    "split_seed": manifest.get("split_seed"),
                },
                sort_keys=True,
            ),
        )

    env = lmdb.open(str(args.lmdb), subdir=False, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        length = int(pickle.loads(txn.get(b"length")))
        if args.expected_count is not None and length != args.expected_count:
            raise SystemExit(f"ERROR: LMDB length {length} != expected {args.expected_count}")
        selected = sorted(set(np.linspace(0, length - 1, min(args.samples, length), dtype=int).tolist()))
        seen_sids = []
        for row in selected:
            data = pickle.loads(txn.get(str(row).encode("ascii")))
            pos = torch.as_tensor(data.pos, dtype=torch.float64)
            natoms = int(data.natoms)
            distances = torch.cdist(pos, pos)
            mask = ~torch.eye(natoms, dtype=torch.bool)
            pair_distances = distances[mask]
            directed_edges = int((pair_distances < args.radius).sum().item())
            pbc = torch.as_tensor(data.pbc, dtype=torch.bool).reshape(-1)
            force_tensor = torch.as_tensor(data.forces, dtype=torch.float64)
            force_shape = tuple(force_tensor.shape)
            sid = int(data.sid)
            seen_sids.append(sid)
            atomic_numbers = torch.as_tensor(data.atomic_numbers, dtype=torch.int64).tolist()
            energy = float(torch.as_tensor(data.energy).reshape(-1)[0])
            print(
                f"row={row} sid={sid} atoms={natoms} atomic_numbers={atomic_numbers} directed_edges={directed_edges} "
                f"min_distance={pair_distances.min():.8g} max_distance={pair_distances.max():.8g} "
                f"pbc={pbc.tolist()} energy={energy:.8f} force_shape={force_shape}"
            )
            if natoms != 5:
                raise SystemExit(f"ERROR: row {row} has {natoms} atoms, expected 5")
            if atomic_numbers != EXPECTED_ATOMIC_NUMBERS:
                raise SystemExit(
                    f"ERROR: row {row} atomic numbers/order {atomic_numbers} != {EXPECTED_ATOMIC_NUMBERS}"
                )
            if force_shape != (5, 3):
                raise SystemExit(f"ERROR: row {row} force shape is {force_shape}, expected (5, 3)")
            if pbc.any():
                raise SystemExit(f"ERROR: row {row} is periodic")
            if directed_edges != 20:
                raise SystemExit(f"ERROR: row {row} has {directed_edges} directed non-self edges, expected 20")
            if not torch.isfinite(pos).all() or not torch.isfinite(force_tensor).all():
                raise SystemExit(f"ERROR: row {row} contains non-finite positions/forces")
            if not np.isfinite(energy):
                raise SystemExit(f"ERROR: row {row} contains non-finite energy")
    env.close()
    print(f"PASS: checked {len(selected)} of {length} samples; sid_preview={seen_sids[:8]}")
    print(
        "unit_check:",
        json.dumps(
            {
                "energy_conversion_factor": HARTREE_TO_KCAL_MOL,
                "force_conversion_factor": HARTREE_PER_BOHR_TO_KCAL_MOL_PER_ANG,
                "energy_offset_kcal_per_mol": HIPHOP_ENERGY_MEAN,
            },
            sort_keys=True,
        ),
    )


if __name__ == "__main__":
    main()
