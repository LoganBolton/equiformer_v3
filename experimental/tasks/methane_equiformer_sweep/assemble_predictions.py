#!/usr/bin/env python
"""Join Fair-Chem prediction output to LMDB targets by methane source index."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import lmdb
import numpy as np


def parse_sid(identifier: object) -> int:
    text = identifier.decode() if isinstance(identifier, bytes) else str(identifier)
    # Exported samples use fid=0, so Fair-Chem writes "<sid>_0".
    return int(text.split("_", 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fairchem_predictions", type=Path)
    parser.add_argument("target_lmdb", type=Path)
    parser.add_argument("--output", type=Path, default=Path("predictions.npz"))
    args = parser.parse_args()

    with np.load(args.fairchem_predictions) as raw:
        source_indices = np.asarray([parse_sid(item) for item in raw["ids"]], dtype=np.int64)
        predicted_energy = np.asarray(raw["energy"], dtype=np.float64).reshape(-1)
        flat_forces = np.asarray(raw["forces"], dtype=np.float64)
        boundaries = np.asarray(raw["chunk_idx"], dtype=np.int64)
    force_chunks = np.split(flat_forces, boundaries.tolist())
    if len(force_chunks) != len(source_indices):
        raise ValueError(f"force chunks ({len(force_chunks)}) != IDs ({len(source_indices)})")
    if len(np.unique(source_indices)) != len(source_indices):
        raise ValueError("Fair-Chem prediction IDs contain duplicate source indices")
    if predicted_energy.size != len(source_indices):
        raise ValueError("Fair-Chem energy count does not match IDs")
    prediction_by_sid = {
        int(sid): (float(energy), force)
        for sid, energy, force in zip(source_indices, predicted_energy, force_chunks)
    }

    target_sid: list[int] = []
    target_energy: list[float] = []
    target_forces: list[np.ndarray] = []
    predicted_energy_aligned: list[float] = []
    predicted_forces_aligned: list[np.ndarray] = []
    env = lmdb.open(str(args.target_lmdb), subdir=False, readonly=True, lock=False, readahead=False)
    with env.begin() as txn:
        length = int(pickle.loads(txn.get(b"length")))
        for row in range(length):
            data = pickle.loads(txn.get(str(row).encode("ascii")))
            sid = int(data.sid)
            if sid not in prediction_by_sid:
                raise ValueError(f"missing prediction for target source frame {sid}")
            energy_prediction, force_prediction = prediction_by_sid.pop(sid)
            force_target = np.asarray(data.forces, dtype=np.float64)
            if force_prediction.shape != force_target.shape:
                raise ValueError(
                    f"source frame {sid}: predicted force shape {force_prediction.shape} "
                    f"!= target {force_target.shape}"
                )
            target_sid.append(sid)
            target_energy.append(float(np.asarray(data.energy).reshape(-1)[0]))
            target_forces.append(force_target)
            predicted_energy_aligned.append(energy_prediction)
            predicted_forces_aligned.append(force_prediction)
    env.close()
    if prediction_by_sid:
        raise ValueError(f"{len(prediction_by_sid)} prediction IDs were absent from the target LMDB")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        source_indices=np.asarray(target_sid, dtype=np.int64),
        target_energy=np.asarray(target_energy, dtype=np.float64),
        predicted_energy=np.asarray(predicted_energy_aligned, dtype=np.float64),
        target_forces=np.stack(target_forces),
        predicted_forces=np.stack(predicted_forces_aligned),
    )
    print(f"PASS: aligned {len(target_sid)} frames by source index; wrote {args.output}")


if __name__ == "__main__":
    main()
