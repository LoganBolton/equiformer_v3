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

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[2]
BENCHMARKS_ROOT = REPO_ROOT / "external" / "rotation-invariant-neural-networks" / "benchmarks"

if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

# gaunt_self_tensor_product.py loads const_wigner2gaunt.pt from the process cwd.
os.chdir(TASK_DIR)

from equiformer_v3_body_order_test import EquiformerV3BodyOrderTest
from rotating_ring.generate_data.rotating_ring_dataset import create_rotating_ring_dataset


MODEL_CONFIGS = {
    "gate": ("gate", "gate", False),
    "sep_s2": ("sep_s2", "sep_s2", False),
    "sep_s2_swiglu": ("sep_s2_swiglu", "sep_s2_swiglu", False),
    "sep-merge_s2_swiglu": ("sep-merge_s2_swiglu", "sep-merge_s2_swiglu", False),
    "sep-merge_gates2_swiglu": ("sep-merge_gates2_swiglu", "sep-merge_gates2_swiglu", False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--success-margin", type=float, default=0.1)
    parser.add_argument("--stop-at-accuracy", type=float, default=1.0)
    parser.add_argument("--print-freq", type=int, default=10)

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


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    success_margin: float,
) -> tuple[float, float, torch.Tensor]:
    model.eval()
    logits_parts = []
    target_parts = []
    for batch in loader:
        batch = batch.to(device)
        logits_parts.append(model(batch).detach().cpu())
        target_parts.append(batch.y.detach().cpu())

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

    device = torch.device(args.device)
    dataset = create_ring_pyg_dataset(args)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = make_model(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_accuracy = 0.0
    best_margin_accuracy = 0.0
    final_loss = 0.0
    final_logits = None
    final_epoch = 0
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        final_loss = run_epoch(model, train_loader, optimizer, device)
        accuracy, margin_accuracy, final_logits = evaluate(model, eval_loader, device, args.success_margin)
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
    }


def main() -> None:
    args = parse_args()
    result = train(args)
    print(
        f"Done: epoch {result['epoch']} | loss {result['loss']:.6f} | "
        f"acc {result['accuracy']:.3f} | margin_acc {result['margin_accuracy']:.3f} | "
        f"best_acc {result['best_accuracy']:.3f} | "
        f"best_margin_acc {result['best_margin_accuracy']:.3f} | "
        f"graphs {result['num_graphs']} | nodes/graph {result['num_nodes']} | "
        f"directed_edges/graph {result['num_directed_edges']} | "
        f"time {result['train_time']:.2f}s"
    )


if __name__ == "__main__":
    main()
