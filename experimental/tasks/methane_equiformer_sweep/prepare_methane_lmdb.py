#!/usr/bin/env python
"""Prepare the methane extxyz data as FairChem LMDBs for EquiformerV3."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import ase.io
import lmdb
import numpy as np
import torch
from ase.constraints import FixAtoms
from torch_geometric.data import Data
from tqdm import tqdm


HARTREE_TO_KCAL_MOL = 627.5096080305927
HARTREE_PER_BOHR_TO_KCAL_MOL_PER_ANG = 51.42208619083232 * 23.060541945329334
HIPHOP_ENERGY_MEAN = -25042.327220945674


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-src", type=Path, default=Path("datasets/methane.extxyz"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--data-size", type=positive_int, default=1_000_000)
    parser.add_argument("--data-sizes", type=positive_int, nargs="+", default=None)
    parser.add_argument("--test-set-size", type=positive_int, default=80_000)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--heldout-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-neighbors", type=int, default=128)
    parser.add_argument("--radius", type=float, default=10.3)
    parser.add_argument("--molecule-cell-size", type=float, default=32.0)
    parser.add_argument("--map-size-gb", type=int, default=512)
    parser.add_argument("--commit-interval", type=positive_int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_src.exists():
        raise FileNotFoundError(f"Missing methane extxyz: {args.data_src}")
    if not 0.0 < args.valid_fraction < 1.0:
        raise ValueError(f"--valid-fraction must be in (0, 1), got {args.valid_fraction}.")
    if not 0.0 <= args.heldout_fraction < 1.0:
        raise ValueError(f"--heldout-fraction must be in [0, 1), got {args.heldout_fraction}.")
    if args.valid_fraction + args.heldout_fraction >= 1.0:
        raise ValueError("--valid-fraction + --heldout-fraction must be less than 1.")

    for data_size in args.data_sizes or [args.data_size]:
        prepare_one_size(args, data_size)


def prepare_one_size(args: argparse.Namespace, data_size: int) -> None:
    dataset_dir = args.output_root / f"methane_train{data_size}_test{args.test_set_size}"
    if dataset_dir.exists():
        if not args.overwrite:
            print(f"{dataset_dir} already exists; use --overwrite to rebuild.")
            return
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True)

    valid_count = int(round(data_size * args.valid_fraction))
    heldout_count = int(round(data_size * args.heldout_fraction))
    train_count = data_size - valid_count - heldout_count
    split_counts = {
        "train": train_count,
        "val": valid_count,
        "heldout": heldout_count,
        "test": args.test_set_size,
    }

    writers = {
        split: LmdbWriter(dataset_dir / split / "data.lmdb", args.map_size_gb, args.commit_interval)
        for split in split_counts
    }
    natoms = {split: [] for split in split_counts}

    rng = np.random.default_rng(args.seed)
    train_perm = rng.permutation(data_size)
    valid_original_indices = set(int(i) for i in train_perm[train_count:])
    heldout_original_indices = set(int(i) for i in train_perm[train_count + valid_count :])
    valid_original_indices -= heldout_original_indices

    try:
        total = data_size + args.test_set_size
        frames = ase.io.iread(args.data_src)
        for frame_index, atoms in enumerate(tqdm(frames, total=total, desc=f"Converting methane {data_size}")):
            if frame_index >= total:
                break
            split = split_for_index(frame_index, data_size, valid_original_indices, heldout_original_indices)
            data = convert_methane_atoms(atoms, molecule_cell_size=args.molecule_cell_size)
            data.sid = frame_index
            data.fid = 0
            data.energy = torch.tensor(
                [float(data.energy) * HARTREE_TO_KCAL_MOL - HIPHOP_ENERGY_MEAN],
                dtype=torch.float32,
            )
            data.forces = data.forces * HARTREE_PER_BOHR_TO_KCAL_MOL_PER_ANG
            writers[split].write(data)
            natoms[split].append(int(data.natoms))
    finally:
        for writer in writers.values():
            writer.close()

    for split, count in split_counts.items():
        if writers[split].count != count:
            raise RuntimeError(f"Expected {count} {split} samples, wrote {writers[split].count}.")
        np.savez(dataset_dir / split / "metadata.npz", natoms=np.asarray(natoms[split], dtype=np.int64))

    manifest = {
        "data_src": str(args.data_src),
        "data_size": data_size,
        "test_set_size": args.test_set_size,
        "valid_fraction": args.valid_fraction,
        "heldout_fraction": args.heldout_fraction,
        "seed": args.seed,
        "energy_units": "kcal/mol shifted by HIPHOP_ENERGY_MEAN",
        "force_units": "kcal/mol/Angstrom",
        "hiphop_energy_mean": HIPHOP_ENERGY_MEAN,
        "splits": split_counts,
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote methane LMDB dataset to {dataset_dir}")


def convert_methane_atoms(atoms, molecule_cell_size: float) -> Data:
    atoms_copy = atoms.copy()
    if atoms_copy.cell.volume == 0.0:
        atoms_copy.center(vacuum=molecule_cell_size)
    cell = torch.tensor(np.asarray(atoms_copy.get_cell(complete=True)), dtype=torch.float32).view(1, 3, 3)
    positions = torch.tensor(np.asarray(atoms_copy.get_positions()), dtype=torch.float32)
    atomic_numbers = torch.tensor(atoms_copy.get_atomic_numbers(), dtype=torch.uint8)
    tags = torch.tensor(atoms_copy.get_tags(), dtype=torch.int)
    fixed = torch.zeros(len(atoms_copy), dtype=torch.int)
    for constraint in getattr(atoms_copy, "constraints", []):
        if isinstance(constraint, FixAtoms):
            fixed[constraint.index] = 1

    return Data(
        cell=cell,
        pos=positions,
        atomic_numbers=atomic_numbers,
        natoms=positions.shape[0],
        tags=tags,
        fixed=fixed,
        pbc=torch.tensor([True, True, True], dtype=torch.bool),
        energy=float(atoms.get_potential_energy(apply_constraint=False)),
        forces=torch.tensor(atoms.get_forces(apply_constraint=False), dtype=torch.float32),
    )


def split_for_index(
    frame_index: int,
    data_size: int,
    valid_original_indices: set[int],
    heldout_original_indices: set[int],
) -> str:
    if frame_index >= data_size:
        return "test"
    if frame_index in heldout_original_indices:
        return "heldout"
    if frame_index in valid_original_indices:
        return "val"
    return "train"


class LmdbWriter:
    def __init__(self, path: Path, map_size_gb: int, commit_interval: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(
            str(path),
            map_size=map_size_gb * 1024**3,
            subdir=False,
            meminit=False,
            map_async=True,
        )
        self.count = 0
        self.commit_interval = commit_interval
        self.txn = self.env.begin(write=True)

    def write(self, data: object) -> None:
        self.txn.put(str(self.count).encode("ascii"), pickle.dumps(data, protocol=-1))
        self.count += 1
        if self.count % self.commit_interval == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self) -> None:
        self.txn.put("length".encode("ascii"), pickle.dumps(self.count, protocol=-1))
        self.txn.commit()
        self.txn = None
        self.env.sync()
        self.env.close()


if __name__ == "__main__":
    main()
