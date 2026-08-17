#!/usr/bin/env python
"""Generate EquiformerV3 methane sweep configs."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import yaml


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=TASK_DIR / "configs")
    parser.add_argument("--data-root", type=Path, default=TASK_DIR / "data")
    parser.add_argument("--run-dir", type=Path, default=TASK_DIR / "runs")
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[42, 1776])
    parser.add_argument("--lmax-values", type=int, nargs="+", default=[3, 4])
    parser.add_argument(
        "--mmax",
        type=int,
        default=None,
        help="Maximum order. By default use mmax=lmax, the closest analogue to full HIPPYNN lmax.",
    )
    parser.add_argument("--data-size", type=int, default=1_000_000)
    parser.add_argument("--data-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--external-test-size", type=int, default=80_000)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[2.5e-3])
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Override evaluation batch size; defaults to 512 for lmax=3 and 4096 for lmax=4.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--name-suffix", default="")
    return parser.parse_args()


def dataset_name(data_size: int, external_test_size: int, seed: int) -> str:
    return f"methane_train{data_size}_test{external_test_size}_seed{seed}"


def main() -> None:
    args = parse_args()
    # Resolve to absolute paths so the generated configs stay valid no matter
    # which directory training is later launched from.
    args.output_dir = args.output_dir.resolve()
    args.data_root = args.data_root.resolve()
    args.run_dir = args.run_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    data_sizes = args.data_sizes or [args.data_size]
    configs = []
    for model_seed, lmax, data_size, lr in itertools.product(
        args.model_seeds,
        args.lmax_values,
        data_sizes,
        args.learning_rates,
    ):
        mmax = lmax if args.mmax is None else min(args.mmax, lmax)
        dataset_id = dataset_name(data_size, args.external_test_size, model_seed)
        suffix = args.name_suffix.strip()
        if suffix and not suffix.startswith("_"):
            suffix = f"_{suffix}"
        name = (
            f"methane_eqv3_l{lmax}_m{mmax}_data{data_size}_test{args.external_test_size}"
            f"_lr{format_lr(lr)}_seed{model_seed}{suffix}"
        )
        config_path = args.output_dir / f"{name}.yml"
        dataset_dir = args.data_root / dataset_id
        run_dir = args.run_dir / dataset_id / f"seed{model_seed}" / f"l{lmax}_m{mmax}_lr{format_lr(lr)}"
        config = build_config(args, name, dataset_dir, dataset_id, model_seed, lmax, mmax, lr)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        configs.append(
            {
                "name": name,
                "config": display_path(config_path),
                "dataset_id": dataset_id,
                "data_size": data_size,
                "external_test_size": args.external_test_size,
                "split_seed": model_seed,
                "seed": model_seed,
                "model_seed": model_seed,
                "lmax": lmax,
                "mmax": mmax,
                "learning_rate": lr,
                "run_dir": display_path(run_dir),
            }
        )

    manifest_path = args.output_dir / "sweep_manifest.json"
    manifest_path.write_text(json.dumps(configs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(configs)} configs and {manifest_path}")


def build_config(
    args: argparse.Namespace,
    name: str,
    dataset_dir: Path,
    dataset_id: str,
    model_seed: int,
    lmax: int,
    mmax: int,
    lr: float,
) -> dict[str, Any]:
    grid_resolution = [14, 8] if lmax <= 4 else [20, 8]
    ffn_grid_resolution = [14, 14] if lmax <= 4 else [20, 20]
    manifest = load_dataset_manifest(dataset_dir, model_seed)
    train_count = manifest["splits"]["train"]
    steps_per_epoch = (train_count + args.batch_size - 1) // args.batch_size
    eval_every = args.eval_every or max(1, steps_per_epoch)
    checkpoint_every = args.checkpoint_every or max(1, steps_per_epoch)
    eval_batch_size = args.eval_batch_size or (512 if lmax == 3 else 4096)
    energy_mean = manifest["label_statistics"]["train"]["energy_mean"]
    run_dir = args.run_dir / dataset_id / f"seed{model_seed}" / f"l{lmax}_m{mmax}_lr{format_lr(lr)}"
    return {
        "trainer": "equiformer_v3_dens_trainer",
        "logger": "tensorboard",
        "hide_eval_progressbar": False,
        "seed": model_seed,
        "task": {
            "strict_load": True,
            "dataset": "lmdb",
            "primary_metric": "energy_mae",
            "methane_hippynn_loss": True,
            "methane_l2_regularization": 1.0e-6,
        },
        "dataset": {
            "train": dataset_config(dataset_dir / "train", energy_mean),
            "val": dataset_config(dataset_dir / "val", energy_mean),
            "test": dataset_config(dataset_dir / "test", energy_mean),
            "metadata": {
                "dataset_id": dataset_id,
                "split_seed": model_seed,
                "model_seed": model_seed,
                "data_size": inferred_data_size(dataset_dir),
                "external_test_size": args.external_test_size,
                "shared_across_model_seeds": False,
                "split_matches_hippynn_model_seed": True,
            },
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
            {"forces": {"fn": "mae", "coefficient": 1}},
        ],
        "evaluation_metrics": {
            "primary_metric": "energy_mae",
            "metrics": {
                "energy": ["mae", "per_atom_mae"],
                "forces": ["mae", "cosine_similarity", "magnitude_error"],
            },
        },
        "model": {
            "name": "equiformer_v3",
            "use_pbc": False,
            "use_pbc_single": False,
            "otf_graph": True,
            "regress_forces": True,
            "regress_stress": False,
            "direct_prediction": False,
            "max_neighbors": 4,
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
            "mmax": mmax,
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
            "avg_degree": 4,
        },
        "optim": {
            "batch_size": args.batch_size,
            "eval_batch_size": eval_batch_size,
            "grad_accumulation_steps": 1,
            "load_balancing": False,
            "load_balancing_on_error": "warn_and_no_balance",
            "num_workers": args.num_workers,
            "lr_initial": lr,
            "optimizer": "Adam",
            "optimizer_params": {"weight_decay": 0.0},
            "scheduler": "ReduceLROnPlateau",
            "scheduler_params": {
                "mode": "min",
                "factor": 0.5,
                "patience": 150,
                "threshold": 0.0001,
                "threshold_mode": "rel",
            },
            "max_epochs": args.epochs,
            "clip_grad_norm": 100,
            "ema_decay": None,
            "eval_every": eval_every,
            "checkpoint_every": checkpoint_every,
            "use_compile": False,
            "use_denoising_pos": False,
            "denoising_pos_coefficient": 10,
            "denoising_pos_params": {"prob": 0.0},
        },
        "cmd_note": {
            "name": name,
            "dataset_id": dataset_id,
            "split_seed": model_seed,
            "model_seed": model_seed,
            "run_dir": str(run_dir),
            "methodology": [
                "The model seed also generates the train/validation/internal-test split, exactly as in the HIPPYNN methane program.",
                "The loss is energy RMSE + energy MAE + componentwise force RMSE + componentwise force MAE + 1e-6 weight L2.",
                "Adam starts at 2.5e-3. Fair-Chem uses the HIPPYNN plateau LR rule but a fixed starting batch because it cannot safely rebuild loaders on a batch-size plateau.",
            ],
        },
    }


def dataset_config(src: Path, energy_mean: float = 0.0) -> dict[str, Any]:
    return {
        "format": "lmdb",
        "src": str(src),
        "transforms": {
            "normalizer": {
                "energy": {"mean": energy_mean, "stdev": 1.0},
                "forces": {"mean": 0.0, "stdev": 1.0},
            }
        },
    }


def load_dataset_manifest(dataset_dir: Path, expected_seed: int) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Complete seed-specific dataset manifest not found: {manifest_path}. "
            "Run prepare_methane_lmdb.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("split_seed") != expected_seed:
        raise ValueError(
            f"{manifest_path} has split_seed={manifest.get('split_seed')}, expected {expected_seed}"
        )
    data_size = inferred_data_size(dataset_dir)
    heldout_count = int(0.1 * data_size)
    remaining_count = data_size - heldout_count
    val_count = int((0.1 / 0.9) * remaining_count)
    expected = {
        "train": remaining_count - val_count,
        "val": val_count,
        "heldout": heldout_count,
    }
    expected["test"] = manifest["external_test_size"]
    if manifest.get("splits") != expected:
        raise ValueError(f"{manifest_path} split counts {manifest.get('splits')} != {expected}")
    return manifest


def inferred_data_size(dataset_dir: Path) -> int:
    return int(dataset_dir.name.split("_test", 1)[0].removeprefix("methane_train"))


def format_lr(lr: float) -> str:
    return f"{lr:.8g}".replace(".", "p").replace("+", "").replace("-", "m")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(TASK_DIR))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
