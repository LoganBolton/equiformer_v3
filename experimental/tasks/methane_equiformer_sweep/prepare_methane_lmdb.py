#!/usr/bin/env python
"""Prepare the methane extxyz data as shared FairChem LMDBs for EquiformerV3."""

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
EXPECTED_METHANE_ATOMIC_NUMBERS = [6, 1, 1, 1, 1]
ENERGY_PATTERN = re.compile(r"(?:^|\s)energy=([^\s]+)")


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
    parser.add_argument("--external-test-size", type=positive_int, default=80_000)
    parser.add_argument(
        "--model-seeds",
        type=int,
        nargs="+",
        default=[42, 1776],
        help="Each seed is used for both model initialization and the HIPPYNN data split.",
    )
    parser.add_argument("--max-neighbors", type=int, default=4)
    parser.add_argument("--radius", type=float, default=10.3)
    parser.add_argument("--molecule-cell-size", type=float, default=32.0)
    parser.add_argument("--map-size-gb", type=int, default=512)
    parser.add_argument("--commit-interval", type=positive_int, default=4096)
    parser.add_argument("--purpose", type=str, default="shared_equiformer_dataset")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dataset_name(data_size: int, external_test_size: int, split_seed: int) -> str:
    return f"methane_train{data_size}_test{external_test_size}_seed{split_seed}"


def main() -> None:
    args = parse_args()
    if not args.data_src.exists():
        raise FileNotFoundError(f"Missing methane extxyz: {args.data_src}")
    for data_size in args.data_sizes or [args.data_size]:
        prepare_one_size(args, data_size)


def prepare_one_size(args: argparse.Namespace, data_size: int) -> None:
    seeds = list(dict.fromkeys(args.model_seeds))
    datasets: dict[int, dict] = {}
    for seed in seeds:
        dataset_dir = args.output_root / dataset_name(data_size, args.external_test_size, seed)
        if dataset_dir.exists():
            if not args.overwrite:
                manifest = dataset_dir / "manifest.json"
                indices_path = dataset_dir / "split_indices.npz"
                if manifest.is_file() and indices_path.is_file():
                    print(f"{dataset_dir} is complete; use --overwrite to rebuild.")
                    continue
                raise RuntimeError(
                    f"{dataset_dir} exists without completion metadata. Move it aside or use --overwrite."
                )
            shutil.rmtree(dataset_dir)

        build_dir = dataset_dir.with_name(f".{dataset_dir.name}.building-{uuid.uuid4().hex}")
        build_dir.mkdir(parents=True)
        indices = make_hippynn_splits(data_size, seed, args.external_test_size)
        split_counts = {
            "train": len(indices["train"]),
            "val": len(indices["valid"]),
            "heldout": len(indices["internal_test"]),
            "test": args.external_test_size,
        }
        assignments = np.zeros(data_size, dtype=np.uint8)
        assignments[indices["valid"]] = 1
        assignments[indices["internal_test"]] = 2
        writers = {
            split: LmdbWriter(build_dir / split / "data.lmdb", args.map_size_gb, args.commit_interval)
            for split in split_counts
        }
        datasets[seed] = {
            "dataset_dir": dataset_dir,
            "build_dir": build_dir,
            "indices": indices,
            "split_counts": split_counts,
            "assignments": assignments,
            "writers": writers,
            "natoms": {split: [] for split in split_counts},
            "energy_sum": {split: 0.0 for split in split_counts},
            "energy_sum_sq": {split: 0.0 for split in split_counts},
        }

    if not datasets:
        return

    total_frames_needed = data_size + args.external_test_size
    try:
        frames = iter_methane_extxyz(args.data_src)
        for frame_index, atoms in enumerate(
            tqdm(frames, total=total_frames_needed, desc=f"Converting {dataset_dir.name}")
        ):
            if frame_index >= total_frames_needed:
                break
            data = convert_methane_atoms(atoms, molecule_cell_size=args.molecule_cell_size)
            validate_unconverted_frame(data, frame_index)
            data.sid = frame_index
            data.fid = 0
            data.energy = torch.tensor(
                [float(data.energy) * HARTREE_TO_KCAL_MOL - HIPHOP_ENERGY_MEAN],
                dtype=torch.float32,
            )
            data.forces = data.forces * HARTREE_PER_BOHR_TO_KCAL_MOL_PER_ANG
            validate_converted_frame(data, frame_index, args.radius)
            energy = float(data.energy)
            payload = pickle.dumps(data, protocol=-1)
            for state in datasets.values():
                split = split_for_assignment(frame_index, data_size, state["assignments"])
                state["writers"][split].write_payload(payload)
                state["natoms"][split].append(int(data.natoms))
                state["energy_sum"][split] += energy
                state["energy_sum_sq"][split] += energy * energy
    finally:
        for state in datasets.values():
            for writer in state["writers"].values():
                writer.close()

    command = " ".join([sys.executable, *sys.argv])
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    for seed, state in datasets.items():
        split_counts = state["split_counts"]
        for split, count in split_counts.items():
            writer = state["writers"][split]
            if writer.count != count:
                raise RuntimeError(f"Seed {seed}: expected {count} {split} samples, wrote {writer.count}.")
            np.savez(
                state["build_dir"] / split / "metadata.npz",
                natoms=np.asarray(state["natoms"][split], dtype=np.int64),
            )

        np.savez_compressed(state["build_dir"] / "split_indices.npz", **state["indices"])
        label_statistics = {}
        for split, count in split_counts.items():
            mean = state["energy_sum"][split] / count
            variance = max(0.0, state["energy_sum_sq"][split] / count - mean * mean)
            label_statistics[split] = {
                "energy_mean": mean,
                "energy_std_ddof0": variance**0.5,
            }
        manifest = {
            "dataset_name": state["dataset_dir"].name,
            "purpose": args.purpose,
            "source_trajectory": str(args.data_src.resolve()),
            "data_size": data_size,
            "external_test_size": args.external_test_size,
            "split_seed": seed,
            "model_seed": seed,
            "split_matches_hippynn_model_seed": True,
            "valid_fraction_of_full_pool": 0.1,
            "internal_test_fraction_of_full_pool": 0.1,
            "energy_units": "kcal/mol shifted by HIPHOP_ENERGY_MEAN",
            "force_units": "kcal/mol/Angstrom",
            "hiphop_energy_mean": HIPHOP_ENERGY_MEAN,
            "energy_conversion_factor": HARTREE_TO_KCAL_MOL,
            "force_conversion_factor": HARTREE_PER_BOHR_TO_KCAL_MOL_PER_ANG,
            "splits": split_counts,
            "label_statistics": label_statistics,
            "index_sha256": {name: index_sha256(value) for name, value in state["indices"].items()},
            "git_commit": commit,
            "generation_command": command,
            "selection": "first data_size frames define the development pool; the next external_test_size sequential frames define the external test set",
            "position_units": "Angstrom",
            "label_dtype": "float32",
            "atomic_numbers": EXPECTED_METHANE_ATOMIC_NUMBERS,
            "atoms_per_frame": 5,
            "pbc": [False, False, False],
            "max_radius_angstrom": args.radius,
            "max_neighbors": args.max_neighbors,
        }
        (state["build_dir"] / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        state["build_dir"].rename(state["dataset_dir"])
        print(f"Wrote methane LMDB dataset to {state['dataset_dir']}")


def validate_unconverted_frame(data: Data, frame_index: int) -> None:
    atomic_numbers = data.atomic_numbers.tolist()
    if int(data.natoms) != 5:
        raise RuntimeError(f"Source frame {frame_index} has {int(data.natoms)} atoms; expected methane (5).")
    if atomic_numbers != EXPECTED_METHANE_ATOMIC_NUMBERS:
        raise RuntimeError(
            f"Source frame {frame_index} atomic numbers/order {atomic_numbers} != {EXPECTED_METHANE_ATOMIC_NUMBERS}."
        )
    if tuple(data.forces.shape) != (5, 3):
        raise RuntimeError(f"Source frame {frame_index} forces have shape {tuple(data.forces.shape)}.")
    if torch.as_tensor(data.pbc, dtype=torch.bool).any():
        raise RuntimeError(f"Source frame {frame_index} is periodic; methane export must be nonperiodic.")
    if not torch.isfinite(data.pos).all() or not torch.isfinite(data.forces).all():
        raise RuntimeError(f"Source frame {frame_index} has non-finite positions or forces.")


def validate_converted_frame(data: Data, frame_index: int, radius: float) -> None:
    if not torch.isfinite(data.energy).all() or not torch.isfinite(data.forces).all():
        raise RuntimeError(f"Source frame {frame_index} has non-finite converted labels.")
    pos = torch.as_tensor(data.pos, dtype=torch.float64)
    distances = torch.cdist(pos, pos)
    natoms = int(data.natoms)
    mask = ~torch.eye(natoms, dtype=torch.bool)
    directed_edges = int((distances[mask] < radius).sum().item())
    if directed_edges != 20:
        raise RuntimeError(
            f"Source frame {frame_index} has {directed_edges} directed non-self edges within {radius} A; expected 20."
        )


def convert_methane_atoms(atoms: Atoms, molecule_cell_size: float) -> Data:
    atoms_copy = atoms.copy()
    if atoms_copy.cell.volume == 0.0:
        atoms_copy.center(vacuum=molecule_cell_size)
    cell = torch.tensor(np.asarray(atoms_copy.get_cell(complete=True)), dtype=torch.float32).view(1, 3, 3)
    positions = torch.tensor(np.asarray(atoms_copy.get_positions()), dtype=torch.float32)
    atomic_numbers = torch.tensor(atoms_copy.get_atomic_numbers(), dtype=torch.int64)
    tags = torch.tensor(atoms_copy.get_tags(), dtype=torch.int64)
    fixed = torch.zeros(len(atoms_copy), dtype=torch.int64)
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


def split_for_assignment(frame_index: int, data_size: int, assignments: np.ndarray) -> str:
    if frame_index >= data_size:
        return "test"
    assignment = int(assignments[frame_index])
    if assignment == 2:
        return "heldout"
    if assignment == 1:
        return "val"
    if assignment == 0:
        return "train"
    raise RuntimeError(f"Invalid split assignment {assignment} for source frame {frame_index}")


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
        self.write_payload(pickle.dumps(data, protocol=-1))

    def write_payload(self, payload: bytes) -> None:
        self.txn.put(str(self.count).encode("ascii"), payload)
        self.count += 1
        if self.count % self.commit_interval == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self) -> None:
        self.txn.put(b"length", pickle.dumps(self.count, protocol=-1))
        self.txn.commit()
        self.txn = None
        self.env.sync()
        self.env.close()


if __name__ == "__main__":
    main()
