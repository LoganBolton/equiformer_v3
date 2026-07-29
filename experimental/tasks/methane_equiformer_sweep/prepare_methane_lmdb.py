#!/usr/bin/env python
"""Prepare the methane extxyz data as FairChem LMDBs for EquiformerV3."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from torch_geometric.data import Data
from tqdm import tqdm

from hippynn_splits import index_sha256, make_hippynn_splits

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
    parser.add_argument("--data-src", type=Path, default=Path("experimental/datasets/methane.extxyz"))
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--data-size", type=positive_int, default=1_000_000)
    parser.add_argument("--data-sizes", type=positive_int, nargs="+", default=None)
    parser.add_argument("--test-set-size", type=positive_int, default=80_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-neighbors", type=int, default=4)
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
    for data_size in args.data_sizes or [args.data_size]:
        prepare_one_size(args, data_size)


def prepare_one_size(args: argparse.Namespace, data_size: int) -> None:
    dataset_dir = args.output_root / f"methane_train{data_size}_test{args.test_set_size}_seed{args.seed}"
    if dataset_dir.exists():
        if not args.overwrite:
            manifest = dataset_dir / "manifest.json"
            indices = dataset_dir / "split_indices.npz"
            if manifest.is_file() and indices.is_file():
                print(f"{dataset_dir} is complete; use --overwrite to rebuild.")
                return
            raise RuntimeError(
                f"{dataset_dir} exists without completion metadata. Move it aside or use --overwrite."
            )
        shutil.rmtree(dataset_dir)
    build_dir = dataset_dir.with_name(f".{dataset_dir.name}.building-{uuid.uuid4().hex}")
    build_dir.mkdir(parents=True)

    indices = make_hippynn_splits(data_size, args.seed, args.test_set_size)
    train_count = len(indices["train"])
    valid_count = len(indices["valid"])
    heldout_count = len(indices["internal_test"])
    split_counts = {
        "train": train_count,
        "val": valid_count,
        "heldout": heldout_count,
        "test": args.test_set_size,
    }

    writers = {
        split: LmdbWriter(build_dir / split / "data.lmdb", args.map_size_gb, args.commit_interval)
        for split in split_counts
    }
    natoms = {split: [] for split in split_counts}

    valid_original_indices = set(indices["valid"].tolist())
    heldout_original_indices = set(indices["internal_test"].tolist())

    try:
        total = data_size + args.test_set_size
        frames = iter_methane_extxyz(args.data_src)
        for frame_index, atoms in enumerate(tqdm(frames, total=total, desc=f"Converting methane {data_size}")):
            if frame_index >= total:
                break
            split = split_for_index(frame_index, data_size, valid_original_indices, heldout_original_indices)
            data = convert_methane_atoms(atoms, molecule_cell_size=args.molecule_cell_size)
            if int(data.natoms) != 5:
                raise RuntimeError(f"Source frame {frame_index} has {int(data.natoms)} atoms; expected methane (5).")
            if sorted(data.atomic_numbers.tolist()) != [1, 1, 1, 1, 6]:
                raise RuntimeError(f"Source frame {frame_index} does not have CH4 atomic numbers.")
            if tuple(data.forces.shape) != (5, 3):
                raise RuntimeError(f"Source frame {frame_index} forces have shape {tuple(data.forces.shape)}.")
            if not torch.isfinite(data.pos).all() or not torch.isfinite(data.forces).all():
                raise RuntimeError(f"Source frame {frame_index} has non-finite positions or forces.")
            data.sid = frame_index
            data.fid = 0
            data.energy = torch.tensor(
                [float(data.energy) * HARTREE_TO_KCAL_MOL - HIPHOP_ENERGY_MEAN],
                dtype=torch.float32,
            )
            data.forces = data.forces * HARTREE_PER_BOHR_TO_KCAL_MOL_PER_ANG
            if not torch.isfinite(data.energy).all() or not torch.isfinite(data.forces).all():
                raise RuntimeError(f"Source frame {frame_index} has non-finite converted labels.")
            writers[split].write(data)
            natoms[split].append(int(data.natoms))
    finally:
        for writer in writers.values():
            writer.close()

    for split, count in split_counts.items():
        if writers[split].count != count:
            raise RuntimeError(f"Expected {count} {split} samples, wrote {writers[split].count}.")
        np.savez(build_dir / split / "metadata.npz", natoms=np.asarray(natoms[split], dtype=np.int64))

    np.savez_compressed(build_dir / "split_indices.npz", **indices)
    command = " ".join([sys.executable, *sys.argv])
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = {
        "source_trajectory": str(args.data_src.resolve()),
        "data_size": data_size,
        "test_set_size": args.test_set_size,
        "valid_fraction_of_full_pool": 0.1,
        "internal_test_fraction_of_full_pool": 0.1,
        "seed": args.seed,
        "energy_units": "kcal/mol shifted by HIPHOP_ENERGY_MEAN",
        "force_units": "kcal/mol/Angstrom",
        "hiphop_energy_mean": HIPHOP_ENERGY_MEAN,
        "splits": split_counts,
        "index_sha256": {name: index_sha256(value) for name, value in indices.items()},
        "git_commit": commit,
        "generation_command": command,
        "selection": "first data_size frames; external test is immediately following frames",
        "position_units": "Angstrom",
        "label_dtype": "float32",
        "atomic_numbers": [1, 6],
        "atoms_per_frame": 5,
        "pbc": [False, False, False],
        "max_radius_angstrom": args.radius,
        "max_neighbors": args.max_neighbors,
    }
    (build_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_dir.rename(dataset_dir)
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
        pbc=torch.tensor([False, False, False], dtype=torch.bool),
        energy=float(atoms.get_potential_energy(apply_constraint=False)),
        forces=torch.tensor(atoms.get_forces(apply_constraint=False), dtype=torch.float32),
    )


ENERGY_PATTERN = re.compile(r"(?:^|\s)energy=([^\s]+)")


def iter_methane_extxyz(path: Path):
    """Stream this fixed-size CH4 extxyz without ASE's full-file index scan."""
    with path.open("r", encoding="utf-8") as handle:
        frame_index = 0
        while True:
            count_line = handle.readline()
            if not count_line:
                return
            try:
                count = int(count_line.strip())
            except ValueError as exc:
                raise ValueError(f"Malformed atom count at source frame {frame_index}: {count_line!r}") from exc
            if count != 5:
                raise ValueError(f"Source frame {frame_index} contains {count} atoms, expected 5")
            header = handle.readline()
            match = ENERGY_PATTERN.search(header)
            if match is None:
                raise ValueError(f"Source frame {frame_index} header has no energy")
            symbols: list[str] = []
            positions: list[list[float]] = []
            forces: list[list[float]] = []
            for atom_index in range(count):
                fields = handle.readline().split()
                if len(fields) != 7:
                    raise ValueError(
                        f"Source frame {frame_index}, atom {atom_index} has {len(fields)} fields, expected 7"
                    )
                symbols.append(fields[0])
                values = [float(value) for value in fields[1:]]
                positions.append(values[:3])
                forces.append(values[3:])
            atoms = Atoms(symbols=symbols, positions=positions, pbc=False)
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=float(match.group(1)),
                forces=np.asarray(forces, dtype=np.float64),
            )
            yield atoms
            frame_index += 1


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
