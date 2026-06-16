"""Train the many-body EquiformerV3 test model on the rotating-ring dataset.

This benchmark intentionally uses only dataset-defined edges. The ring
generator already returns a directed, bidirectional ``edge_index`` for each
graph, and the local EquiformerV3BodyOrderTest consumes ``data.edge_index``
directly instead of constructing a radius graph.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from math import pi
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataListLoader, DataLoader
from torch_geometric.nn import DataParallel as GeometricDataParallel


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[2]
BENCHMARKS_ROOT = REPO_ROOT / "external" / "rotation-invariant-neural-networks" / "benchmarks"

if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(TASK_DIR / ".matplotlib-cache"))

# gaunt_self_tensor_product.py loads const_wigner2gaunt.pt from the process cwd.
os.chdir(TASK_DIR)

from equiformer_v3_body_order_test import EquiformerV3BodyOrderTest
from rotating_ring_experiment_utils import (
    add_experiment_io_args,
    make_result_record,
    run_sweep,
    write_result_json,
)
from rotating_ring.generate_data.rotating_ring_dataset import create_rotating_ring_dataset


MODEL_CONFIGS = {
    "gate": ("gate", "gate", False),
    "sep_s2": ("sep_s2", "sep_s2", False),
    "sep_s2_swiglu": ("sep_s2_swiglu", "sep_s2_swiglu", False),
    "sep-merge_s2_swiglu": ("sep-merge_s2_swiglu", "sep-merge_s2_swiglu", False),
    "sep-merge_gates2_swiglu": ("sep-merge_gates2_swiglu", "sep-merge_gates2_swiglu", False),
}


# Edit these lists when you want to run a batch of experiments. Sweep mode runs
# every graph config against every model config and every lmax in lmax_values.
RING_SWEEP_GRAPH_CONFIGS = [
    {"name": "inner1_outer4", "ring_n_inner": 1, "ring_n_outer": 4},
    {"name": "inner2_outer1", "ring_n_inner": 2, "ring_n_outer": 1},
    {"name": "inner2_outer2", "ring_n_inner": 2, "ring_n_outer": 2},
    {"name": "inner2_outer3", "ring_n_inner": 2, "ring_n_outer": 3},
    {"name": "inner2_outer4", "ring_n_inner": 2, "ring_n_outer": 4},
]


RING_SWEEP_MODEL_CONFIGS = [
    {
        "name": "sep_merge_gates2_swiglu_lmax2_3_4_5_6",
        "model_config": "sep-merge_gates2_swiglu",
        "lmax_values": [2, 3, 4, 5, 6],
    },
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_io_args(parser, TASK_DIR)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--success-margin", type=float, default=0.1)
    parser.add_argument("--stop-at-accuracy", type=float, default=1.0)
    parser.add_argument("--print-freq", type=int, default=10)
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help="CUDA device IDs for graph DataParallel. Defaults to all visible GPUs.",
    )
    parser.add_argument(
        "--no-data-parallel",
        action="store_true",
        help="Disable graph DataParallel even when multiple CUDA devices are visible.",
    )

    parser.add_argument("--ring-n-graphs", type=int, default=100)
    parser.add_argument("--ring-seed", type=int, default=0)
    parser.add_argument("--ring-n-inner", type=int, default=3)
    parser.add_argument("--ring-n-outer", type=int, default=3)
    parser.add_argument("--ring-inner-radius-min", type=float, default=1.0)
    parser.add_argument("--ring-inner-radius-max", type=float, default=1.8)
    parser.add_argument("--ring-outer-gap-min", type=float, default=0.8)
    parser.add_argument("--ring-outer-gap-max", type=float, default=1.6)
    parser.add_argument("--ring-outer-rotation-frac-min", type=float, default=0.0)
    parser.add_argument("--ring-outer-rotation-frac-max", type=float, default=0.45)
    parser.add_argument("--ring-outer-3d-rotation-deg", type=float, default=0.0)
    parser.add_argument("--ring-outer-3d-axis-deg", type=float, default=0.0)
    parser.add_argument("--ring-global-rotation-frac-min", type=float, default=0.0)
    parser.add_argument("--ring-global-rotation-frac-max", type=float, default=0.0)
    parser.add_argument("--ring-random-parameters", action="store_true")
    parser.add_argument("--ring-shuffle", action="store_true")
    parser.add_argument("--add-inner-ring-edges", action="store_true")
    parser.add_argument("--add-outer-ring-edges", action="store_true")

    parser.add_argument("--model-config", choices=tuple(MODEL_CONFIGS), default="sep-merge_gates2_swiglu")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--num-ffns", type=int, default=2)
    parser.add_argument("--use-attn", action="store_true")
    parser.add_argument("--use-gaunt-self-tensor-product", action="store_true")
    parser.add_argument("--num-channels", type=int, default=128)
    parser.add_argument("--ffn-hidden-channels", type=int, default=512)
    parser.add_argument("--attn-hidden-channels", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--attn-alpha-channels", type=int, default=64)
    parser.add_argument("--attn-value-channels", type=int, default=16)
    parser.add_argument("--lmax", type=int, default=6)
    parser.add_argument("--mmax", type=int, default=2)
    parser.add_argument("--num-radial-basis", type=int, default=128)
    parser.add_argument("--edge-channels", type=int, default=128)
    parser.add_argument("--max-radius", type=float, default=10.0)
    parser.add_argument("--norm-type", type=str, default="equivariant_layer_norm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def create_ring_pyg_dataset(args: argparse.Namespace) -> list[Data]:
    if args.ring_n_graphs <= 0:
        raise ValueError(f"--ring-n-graphs must be positive, got {args.ring_n_graphs}.")

    envs = create_rotating_ring_dataset(
        n_graphs=args.ring_n_graphs,
        seed=args.ring_seed,
        n_inner=args.ring_n_inner,
        n_outer=args.ring_n_outer,
        inner_radius_range=(args.ring_inner_radius_min, args.ring_inner_radius_max),
        outer_gap_range=(args.ring_outer_gap_min, args.ring_outer_gap_max),
        outer_rotation_fraction_range=(
            args.ring_outer_rotation_frac_min,
            args.ring_outer_rotation_frac_max,
        ),
        outer_3d_rotation_range=(0.0, args.ring_outer_3d_rotation_deg * pi / 180.0),
        outer_3d_axis_angle=args.ring_outer_3d_axis_deg * pi / 180.0,
        global_rotation_fraction_range=(
            args.ring_global_rotation_frac_min,
            args.ring_global_rotation_frac_max,
        ),
        smooth_order=not args.ring_random_parameters,
        shuffle=args.ring_shuffle,
        add_inner_ring_edges=args.add_inner_ring_edges,
        add_outer_ring_edges=args.add_outer_ring_edges,
    )

    dataset = []
    for env in envs:
        dataset.append(
            Data(
                atoms=env.Z.clone(),
                pos=env.R.clone().to(torch.get_default_dtype()),
                edge_index=env.edge_index.clone(),
                y=torch.tensor([env.label], dtype=torch.long),
                node_role=env.node_role.clone(),
                name=env.name,
            )
        )
    return dataset


def make_model(args: argparse.Namespace) -> EquiformerV3BodyOrderTest:
    attn_activation, ffn_activation, use_grid_mlp = MODEL_CONFIGS[args.model_config]
    return EquiformerV3BodyOrderTest(
        max_neighbors=50,
        max_radius=args.max_radius,
        num_radial_basis=args.num_radial_basis,
        max_num_elements=128,
        num_layers=args.num_layers,
        num_channels=args.num_channels,
        attn_hidden_channels=args.attn_hidden_channels,
        num_heads=args.num_heads,
        attn_alpha_channels=args.attn_alpha_channels,
        attn_value_channels=args.attn_value_channels,
        ffn_hidden_channels=args.ffn_hidden_channels,
        norm_type=args.norm_type,
        lmax=args.lmax,
        mmax=args.mmax,
        attn_grid_resolution_list=[20, 8],
        ffn_grid_resolution_list=[20, 20],
        edge_channels=args.edge_channels,
        use_atom_edge_embedding=True,
        use_envelope=True,
        attn_activation=attn_activation,
        use_attn_renorm=True,
        use_add_merge=False,
        use_rad_l_parametrization=True,
        softcap=None,
        ffn_activation=ffn_activation,
        use_grid_mlp=use_grid_mlp,
        alpha_drop=0.0,
        attn_mask_rate=0.0,
        attn_weights_drop=0.0,
        value_drop=0.0,
        drop_path_rate=0.0,
        proj_drop=0.0,
        ffn_drop=0.0,
        avg_num_nodes=1,
        avg_degree=1,
        use_attn=args.use_attn,
        num_ffns=args.num_ffns,
        use_gaunt_self_tensor_product=args.use_gaunt_self_tensor_product,
        num_output_channels=2,
    )


def accuracy_and_margin(logits: torch.Tensor, targets: torch.Tensor, success_margin: float) -> tuple[float, float]:
    predictions = torch.argmax(logits, dim=1)
    accuracy = (predictions == targets).to(torch.float32).mean().item()
    correct_logits = logits.gather(1, targets.view(-1, 1)).squeeze(1)
    other_logits = logits.gather(1, (1 - targets).view(-1, 1)).squeeze(1)
    margin_accuracy = ((correct_logits - other_logits) >= success_margin).to(torch.float32).mean().item()
    return float(accuracy), float(margin_accuracy)


def resolve_device(args: argparse.Namespace) -> tuple[torch.device, list[int]]:
    requested = torch.device(args.device)
    if requested.type != "cuda":
        return requested, []
    if not torch.cuda.is_available():
        raise ValueError("--device requested CUDA, but torch.cuda.is_available() is False.")

    if args.gpu_ids is not None:
        device_ids = list(args.gpu_ids)
    elif requested.index is not None:
        device_ids = [requested.index]
    else:
        device_ids = list(range(torch.cuda.device_count()))

    if not device_ids:
        raise ValueError("No CUDA device IDs were selected.")
    if max(device_ids) >= torch.cuda.device_count() or min(device_ids) < 0:
        raise ValueError(
            f"Selected --gpu-ids {device_ids}, but only {torch.cuda.device_count()} CUDA devices are visible."
        )

    primary_device = torch.device(f"cuda:{device_ids[0]}")
    if args.no_data_parallel or len(device_ids) < 2:
        return primary_device, []
    return primary_device, device_ids


def make_loader(dataset: list[Data], batch_size: int, shuffle: bool, use_data_parallel: bool) -> Any:
    loader_class = DataListLoader if use_data_parallel else DataLoader
    return loader_class(dataset, batch_size=batch_size, shuffle=shuffle)


def batch_targets(batch: Any, device: torch.device) -> torch.Tensor:
    if isinstance(batch, list):
        return torch.cat([data.y for data in batch], dim=0).to(device)
    return batch.y


def batch_num_graphs(batch: Any) -> int:
    if isinstance(batch, list):
        return len(batch)
    return batch.num_graphs


def run_epoch(
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_data_parallel: bool,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        if not use_data_parallel:
            batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        targets = batch_targets(batch, logits.device)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_num_graphs(batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    success_margin: float,
    use_data_parallel: bool,
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    logits_parts = []
    target_parts = []
    for batch in loader:
        if not use_data_parallel:
            batch = batch.to(device)
        logits = model(batch)
        logits_parts.append(logits.detach().cpu())
        target_parts.append(batch_targets(batch, logits.device).detach().cpu())

    logits = torch.cat(logits_parts, dim=0)
    targets = torch.cat(target_parts, dim=0)
    accuracy, margin_accuracy = accuracy_and_margin(logits, targets, success_margin)
    return accuracy, margin_accuracy, logits


def train(args: argparse.Namespace) -> dict[str, object]:
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device, data_parallel_device_ids = resolve_device(args)
    use_data_parallel = len(data_parallel_device_ids) > 1
    dataset = create_ring_pyg_dataset(args)
    train_loader = make_loader(dataset, args.batch_size, shuffle=True, use_data_parallel=use_data_parallel)
    eval_loader = make_loader(dataset, args.batch_size, shuffle=False, use_data_parallel=use_data_parallel)

    base_model = make_model(args)
    if use_data_parallel:
        if args.batch_size < len(data_parallel_device_ids):
            raise ValueError(
                f"--batch-size ({args.batch_size}) must be at least the number of DataParallel GPUs "
                f"({len(data_parallel_device_ids)})."
            )
        model = GeometricDataParallel(
            base_model,
            device_ids=data_parallel_device_ids,
            output_device=data_parallel_device_ids[0],
        ).to(device)
        print(f"Using graph DataParallel on CUDA devices {data_parallel_device_ids}", flush=True)
    else:
        model = base_model.to(device)
        print(f"Using device {device}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_accuracy = 0.0
    best_margin_accuracy = 0.0
    final_loss = 0.0
    final_logits = None
    final_epoch = 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        final_loss = run_epoch(model, train_loader, optimizer, device, use_data_parallel)
        accuracy, margin_accuracy, final_logits = evaluate(
            model,
            eval_loader,
            device,
            args.success_margin,
            use_data_parallel,
        )
        best_accuracy = max(best_accuracy, accuracy)
        best_margin_accuracy = max(best_margin_accuracy, margin_accuracy)
        final_epoch = epoch

        if epoch == 1 or epoch % args.print_freq == 0:
            elapsed_ms = (time.perf_counter() - epoch_start) * 1000.0
            print(
                f"Epoch: [{epoch}]\t loss: {final_loss:.5f}, "
                f"acc: {accuracy:.3f}, margin_acc: {margin_accuracy:.3f}, "
                f"time: {elapsed_ms:.0f}ms",
                flush=True,
            )

        if margin_accuracy >= args.stop_at_accuracy:
            break

    train_time = time.perf_counter() - start_time
    assert final_logits is not None
    return {
        "epoch": final_epoch,
        "loss": float(final_loss),
        "accuracy": float(accuracy),
        "margin_accuracy": float(margin_accuracy),
        "best_accuracy": float(best_accuracy),
        "best_margin_accuracy": float(best_margin_accuracy),
        "train_time": float(train_time),
        "logits": final_logits.tolist(),
        "num_graphs": len(dataset),
        "num_nodes": dataset[0].num_nodes,
        "num_directed_edges": int(dataset[0].edge_index.shape[1]),
        "data_parallel_device_ids": data_parallel_device_ids,
    }


def main() -> None:
    args = parse_args()
    if args.run_sweep:
        run_sweep(
            args=args,
            graph_configs=RING_SWEEP_GRAPH_CONFIGS,
            model_configs=RING_SWEEP_MODEL_CONFIGS,
            train_script=Path(__file__).resolve(),
            repo_root=REPO_ROOT,
            cuda_available=torch.cuda.is_available(),
            cuda_device_count=torch.cuda.device_count(),
        )
        return

    result = train(args)
    result_record = make_result_record(args, result)
    if args.results_json is not None:
        write_result_json(args.results_json, result_record)
        print(f"Wrote JSON result to {args.results_json}", flush=True)

    print(
        f"Done: epoch {result['epoch']} | loss {result['loss']:.6f} | "
        f"acc {result['accuracy']:.3f} | margin_acc {result['margin_accuracy']:.3f} | "
        f"best_acc {result['best_accuracy']:.3f} | "
        f"best_margin_acc {result['best_margin_accuracy']:.3f} | "
        f"graphs {result['num_graphs']} | nodes/graph {result['num_nodes']} | "
        f"directed_edges/graph {result['num_directed_edges']} | "
        f"data_parallel_gpus {result['data_parallel_device_ids']} | "
        f"time {result['train_time']:.2f}s"
    )


if __name__ == "__main__":
    main()
