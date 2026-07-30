#!/usr/bin/env python
"""Fail fast unless a methane sweep exactly matches the HIPPYNN protocol."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import lmdb
import yaml


TASK_DIR = Path(__file__).resolve().parent


def resolve(path: str, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def lmdb_count(directory: Path) -> int:
    paths = sorted(directory.glob("*.lmdb"))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one LMDB in {directory}, found {len(paths)}")
    env = lmdb.open(
        str(paths[0]),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
    )
    try:
        with env.begin() as txn:
            encoded = txn.get(b"length")
            return pickle.loads(encoded) if encoded is not None else txn.stat()["entries"]
    finally:
        env.close()


def check_config(item: dict, manifest_base: Path) -> None:
    required_item_keys = {
        "name",
        "config",
        "dataset_id",
        "data_size",
        "external_test_size",
        "split_seed",
        "model_seed",
        "lmax",
        "mmax",
        "learning_rate",
        "run_dir",
    }
    missing = required_item_keys - item.keys()
    if missing:
        raise ValueError(f"Sweep item is stale; missing keys: {sorted(missing)}")
    if item["split_seed"] != item["model_seed"]:
        raise ValueError(f"{item['name']}: split seed must equal model seed")

    config_path = resolve(item["config"], manifest_base)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = config["model"]
    task = config["task"]
    optim = config["optim"]
    expected_model = {
        "num_layers": 1,
        "lmax": item["lmax"],
        "mmax": item["lmax"],
        "max_radius": 10.3,
        "max_neighbors": 4,
        "use_pbc": False,
        "direct_prediction": False,
        "regress_forces": True,
        "avg_num_nodes": 1,
        "avg_degree": 4,
    }
    for key, expected in expected_model.items():
        if model.get(key) != expected:
            raise ValueError(f"{config_path}: model.{key}={model.get(key)!r}, expected {expected!r}")
    if not task.get("methane_hippynn_loss"):
        raise ValueError(f"{config_path}: exact HIPPYNN methane loss is not enabled")
    if task.get("methane_l2_regularization") != 1.0e-6:
        raise ValueError(f"{config_path}: expected methane L2 coefficient 1e-6")
    if config["evaluation_metrics"].get("primary_metric") != "energy_mae":
        raise ValueError(f"{config_path}: best checkpoint must be selected by energy_mae")
    if optim.get("optimizer") != "Adam" or optim.get("lr_initial") != 2.5e-3:
        raise ValueError(f"{config_path}: expected Adam with lr=2.5e-3")
    if optim.get("batch_size") != 256:
        raise ValueError(f"{config_path}: expected HIPPYNN starting batch size 256")
    if optim.get("scheduler") != "ReduceLROnPlateau":
        raise ValueError(f"{config_path}: expected plateau scheduler")

    train_dir = Path(config["dataset"]["train"]["src"])
    dataset_dir = train_dir.parent
    dataset_manifest_path = dataset_dir / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("split_seed") != item["model_seed"]:
        raise ValueError(f"{dataset_manifest_path}: split/model seed mismatch")
    if not dataset_manifest.get("split_matches_hippynn_model_seed"):
        raise ValueError(f"{dataset_manifest_path}: not marked as a HIPPYNN-matched split")
    for split, expected_count in dataset_manifest["splits"].items():
        actual_count = lmdb_count(dataset_dir / split)
        if actual_count != expected_count:
            raise ValueError(
                f"{dataset_dir / split}: {actual_count} records, expected {expected_count}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest:
        raise ValueError("Sweep manifest is empty")
    for item in manifest:
        check_config(item, TASK_DIR)
    print(f"PASS: {len(manifest)} HIPPYNN-matched Equiformer configurations are launch-ready")


if __name__ == "__main__":
    main()
