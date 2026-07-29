#!/usr/bin/env python
"""Compute HIPPYNN-compatible metrics from aligned methane predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def regression_metrics(target: np.ndarray, prediction: np.ndarray, prefix: str) -> dict[str, float | None]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError(f"{prefix} target/prediction shapes differ: {target.shape} != {prediction.shape}")
    if target.size == 0:
        raise ValueError(f"{prefix} arrays are empty")
    error = prediction - target
    denominator = np.sum((target - target.mean()) ** 2)
    r2 = None if denominator == 0 else 1.0 - float(np.sum(error**2) / denominator)
    return {
        f"{prefix}_mae": float(np.mean(np.abs(error))),
        f"{prefix}_rmse": float(np.sqrt(np.mean(error**2))),
        f"{prefix}_r2": r2,
    }


def compute_metrics(
    source_indices: np.ndarray,
    energy_target: np.ndarray,
    energy_prediction: np.ndarray,
    forces_target: np.ndarray,
    forces_prediction: np.ndarray,
) -> dict[str, object]:
    source_indices = np.asarray(source_indices, dtype=np.int64).reshape(-1)
    energy_target = np.asarray(energy_target)
    energy_prediction = np.asarray(energy_prediction)
    forces_target = np.asarray(forces_target)
    forces_prediction = np.asarray(forces_prediction)
    n_frames = source_indices.size
    if np.unique(source_indices).size != n_frames:
        raise ValueError("source_indices contains duplicates")
    if energy_target.size != n_frames or energy_prediction.size != n_frames:
        raise ValueError("energy arrays do not contain one scalar per source frame")
    if forces_target.shape != forces_prediction.shape:
        raise ValueError(f"force shapes differ: {forces_target.shape} != {forces_prediction.shape}")
    if forces_target.ndim != 3 or forces_target.shape[0] != n_frames or forces_target.shape[-1] != 3:
        raise ValueError(f"expected forces shaped (frames, atoms, 3), got {forces_target.shape}")
    metrics: dict[str, object] = {
        "num_frames": n_frames,
        "num_force_components": int(forces_target.size),
        "force_metric_definition": "componentwise; all frame/atom/Cartesian components flattened",
    }
    metrics.update(regression_metrics(energy_target, energy_prediction, "energy"))
    metrics.update(regression_metrics(forces_target, forces_prediction, "force"))
    return metrics


def pick(data: np.lib.npyio.NpzFile, *names: str) -> np.ndarray:
    for name in names:
        if name in data:
            return data[name]
    raise KeyError(f"prediction archive lacks all accepted keys: {names}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.predictions.with_name("metrics.json")
    with np.load(args.predictions) as data:
        metrics = compute_metrics(
            pick(data, "source_indices", "sid"),
            pick(data, "target_energy", "energy_target"),
            pick(data, "predicted_energy", "energy_prediction"),
            pick(data, "target_forces", "forces_target"),
            pick(data, "predicted_forces", "forces_prediction"),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
