#!/usr/bin/env python
"""Generate EquiformerV3 methane sweep configs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=TASK_DIR / "configs")
    parser.add_argument("--data-root", type=Path, default=TASK_DIR / "data")
    parser.add_argument("--run-dir", type=Path, default=TASK_DIR / "runs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1776, 250])
    parser.add_argument("--lmax-values", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--data-sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[5e-5, 1e-4, 2e-4, 4e-4])
    parser.add_argument("--test-set-size", type=int, default=80_000)
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--eval-batch-size", type=int, default=65_536)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    configs = []
    for seed, lmax, data_size, lr in itertools.product(
        args.seeds,
        args.lmax_values,
        args.data_sizes,
        args.learning_rates,
    ):
        name = f"methane_eqv3_l{lmax}_data{data_size}_lr{format_lr(lr)}_seed{seed}"
        config_path = args.output_dir / f"{name}.yml"
        dataset_dir = args.data_root / f"methane_train{data_size}_test{args.test_set_size}"
        config = build_config(args, name, dataset_dir, seed, lmax, lr)
        config_path.write_text(json.dumps(config, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        configs.append(
            {
                "name": name,
                "config": display_path(config_path),
                "seed": seed,
                "lmax": lmax,
                "data_size": data_size,
                "learning_rate": lr,
                "run_dir": display_path(args.run_dir / name),
            }
        )

    manifest_path = args.output_dir / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(configs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(configs)} configs and {manifest_path}")


def build_config(
    args: argparse.Namespace,
    name: str,
    dataset_dir: Path,
    seed: int,
    lmax: int,
    lr: float,
) -> dict[str, Any]:
    grid_resolution = [14, 8] if lmax <= 4 else [20, 8]
    ffn_grid_resolution = [14, 14] if lmax <= 4 else [20, 20]
    return {
        "trainer": "equiformer_v3_dens_trainer",
        "logger": "tensorboard",
        "hide_eval_progressbar": False,
        "task": {"strict_load": True},
        "dataset": {
            "train": dataset_config(dataset_dir / "train"),
            "val": dataset_config(dataset_dir / "val"),
            "test": dataset_config(dataset_dir / "test"),
        },
        "outputs": {
            "energy": {"shape": 1, "level": "system", "property": "energy"},
            "forces": {
                "irrep_dim": 1,
                "level": "atom",
                "property": "forces",
                "train_on_free_atoms": True,
                "eval_on_free_atoms": True,
            },
        },
        "loss_functions": [
            {"energy": {"fn": "mae", "coefficient": 1}},
            {"forces": {"fn": "l2mae", "coefficient": 1}},
        ],
        "evaluation_metrics": {
            "primary_metric": "forces_mae",
            "metrics": {
                "energy": ["mae", "per_atom_mae"],
                "forces": ["mae", "cosine_similarity", "magnitude_error"],
            },
        },
        "model": {
            "name": "equiformer_v3",
            "use_pbc": True,
            "use_pbc_single": False,
            "otf_graph": True,
            "regress_forces": True,
            "regress_stress": False,
            "direct_prediction": False,
            "max_neighbors": 128,
            "max_radius": 10.3,
            "num_radial_basis": 20,
            "max_num_elements": 128,
            "num_layers": 1,
            "num_channels": 32,
            "attn_hidden_channels": 32,
            "num_heads": 8,
            "attn_alpha_channels": 64,
            "attn_value_channels": 16,
            "ffn_hidden_channels": 128,
            "norm_type": "merge_layer_norm",
            "lmax": lmax,
            "mmax": min(2, lmax),
            "attn_grid_resolution_list": grid_resolution,
            "ffn_grid_resolution_list": ffn_grid_resolution,
            "edge_channels": 64,
            "use_atom_edge_embedding": True,
            "use_envelope": True,
            "attn_activation": "sep-merge_gates2_swiglu",
            "use_attn_renorm": True,
            "use_add_merge": False,
            "use_rad_l_parametrization": True,
            "softcap": None,
            "ffn_activation": "sep-merge_gates2_swiglu",
            "use_grid_mlp": True,
            "use_gate_force_head": True,
            "alpha_drop": 0.0,
            "attn_mask_rate": 0.0,
            "attn_weights_drop": 0.0,
            "value_drop": 0.0,
            "drop_path_rate": 0.0,
            "proj_drop": 0.0,
            "ffn_drop": 0.0,
            "gradient_checkpointing_block_list": [0],
            "enforce_max_neighbors_strictly": True,
            "avg_num_nodes": 1,
        },
        "optim": {
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "grad_accumulation_steps": 1,
            "load_balancing": False,
            "load_balancing_on_error": "warn_and_no_balance",
            "num_workers": args.num_workers,
            "lr_initial": lr,
            "optimizer": "AdamW",
            "optimizer_params": {"weight_decay": 0.001, "betas": [0.9, 0.95]},
            "scheduler": "LambdaLR",
            "scheduler_params": {
                "lambda_type": "cosine",
                "warmup_factor": 0.0,
                "warmup_epochs": 0.1,
                "lr_min_factor": 0.01,
            },
            "max_epochs": args.epochs,
            "clip_grad_norm": 100,
            "ema_decay": None,
            "eval_every": 5000,
            "checkpoint_every": 5000,
            "use_compile": False,
            "use_denoising_pos": False,
            "denoising_pos_coefficient": 10,
            "denoising_pos_params": {"prob": 0.0},
        },
        "cmd_note": {
            "name": name,
            "seed": seed,
            "run_dir": str(args.run_dir / name),
        },
    }


def dataset_config(src: Path) -> dict[str, Any]:
    return {
        "format": "lmdb",
        "src": str(src),
        "transforms": {
            "normalizer": {
                "energy": {"mean": 0.0, "stdev": 1.0},
                "forces": {"mean": 0.0, "stdev": 1.0},
            }
        },
    }


def format_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("+", "").replace("-", "m")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(TASK_DIR))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
