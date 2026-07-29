"""Sweep launching and result serialization for rotating-ring experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any


SWEEP_ONLY_ARG_NAMES = {
    "run_sweep",
    "sweep_gpu_ids",
    "sweep_results_dir",
    "sweep_python",
    "sweep_dry_run",
    "sweep_resume",
    "results_json",
    "experiment_name",
}


def add_experiment_io_args(parser: argparse.ArgumentParser, task_dir: Path) -> None:
    parser.add_argument(
        "--run-sweep",
        action="store_true",
        help="Run RING_SWEEP_GRAPH_CONFIGS x RING_SWEEP_MODEL_CONFIGS across multiple GPUs.",
    )
    parser.add_argument(
        "--sweep-gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help="CUDA device IDs to use for sweep workers. Defaults to --gpu-ids or all visible GPUs.",
    )
    parser.add_argument(
        "--sweep-results-dir",
        type=Path,
        default=task_dir / "sweep_results",
        help="Directory for one log file per sweep experiment.",
    )
    parser.add_argument(
        "--sweep-python",
        type=str,
        default=sys.executable,
        help="Python executable used for sweep child processes.",
    )
    parser.add_argument(
        "--sweep-dry-run",
        action="store_true",
        help="Print sweep commands without running them.",
    )
    parser.add_argument(
        "--sweep-resume",
        action="store_true",
        help="Skip experiments with valid existing result JSON files and run only missing experiments.",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Optional path for a machine-readable JSON result from one training run.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional stable name stored in machine-readable result files.",
    )


def make_result_record(
    args: argparse.Namespace,
    result: dict[str, object],
    experiment_name: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": experiment_name or args.experiment_name,
        "status": "ok",
        "config": _result_config(args),
        "metrics": {
            "epoch": result["epoch"],
            "loss": result["loss"],
            "accuracy": result["accuracy"],
            "margin_accuracy": result["margin_accuracy"],
            "best_accuracy": result["best_accuracy"],
            "best_margin_accuracy": result["best_margin_accuracy"],
            "train_time": result["train_time"],
        },
        "dataset": {
            "num_graphs": result["num_graphs"],
            "num_nodes": result["num_nodes"],
            "num_directed_edges": result["num_directed_edges"],
        },
        "runtime": {
            "data_parallel_device_ids": result["data_parallel_device_ids"],
        },
        "logits": result["logits"],
    }


def write_result_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_sweep(
    args: argparse.Namespace,
    graph_configs: list[dict[str, Any]],
    model_configs: list[dict[str, Any]],
    train_script: Path,
    repo_root: Path,
    cuda_available: bool,
    cuda_device_count: int,
) -> None:
    gpu_ids = _resolve_sweep_gpu_ids(args, cuda_available, cuda_device_count)
    experiments = _build_sweep_experiments(args, graph_configs, model_configs, train_script)
    args.sweep_results_dir.mkdir(parents=True, exist_ok=True)
    indexed_experiments = list(enumerate(experiments, start=1))
    result_records: list[dict[str, Any]] = []
    if args.sweep_resume:
        indexed_experiments, result_records = _resume_partition(args, indexed_experiments)

    print(
        f"Launching {len(indexed_experiments)} experiments on GPUs {gpu_ids}; "
        f"logs: {args.sweep_results_dir}",
        flush=True,
    )
    if args.sweep_resume:
        print(f"Resume: keeping {len(result_records)} completed experiments.", flush=True)

    if args.sweep_dry_run:
        _print_sweep_dry_run(args, indexed_experiments, gpu_ids)
        return

    queue = deque(indexed_experiments)
    running: dict[int, dict[str, Any]] = {}
    failures = []

    while queue or running:
        _start_available_jobs(args, queue, running, gpu_ids, repo_root)
        time.sleep(2.0)
        _collect_finished_jobs(running, result_records, failures, len(experiments))

    _write_sweep_result_files(args.sweep_results_dir, result_records)

    if failures:
        print("Sweep failures:", flush=True)
        for job in failures:
            print(f"  {job['experiment']['name']} -> {job['log_path']}", flush=True)
        raise SystemExit(1)

    print(
        f"Sweep complete. Results: {args.sweep_results_dir / 'sweep_results.json'} | "
        f"{args.sweep_results_dir / 'sweep_results.jsonl'} | "
        f"{args.sweep_results_dir / 'sweep_results.csv'}",
        flush=True,
    )


def _resume_partition(
    args: argparse.Namespace,
    indexed_experiments: list[tuple[int, dict[str, Any]]],
) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    pending = []
    completed = []
    for index, experiment in indexed_experiments:
        result_path = args.sweep_results_dir / f"{index:03d}_{experiment['name']}.result.json"
        try:
            record = json.loads(result_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pending.append((index, experiment))
            continue
        if record.get("status") != "ok" or record.get("experiment_name") != experiment["name"]:
            pending.append((index, experiment))
            continue
        record["result_path"] = str(result_path)
        completed.append(record)
    return pending, completed


def _result_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "training": {
            "epochs": args.epochs,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "success_margin": args.success_margin,
            "stop_at_accuracy": args.stop_at_accuracy,
        },
        "graph": {
            "ring_n_graphs": args.ring_n_graphs,
            "ring_seed": args.ring_seed,
            "ring_n_inner": args.ring_n_inner,
            "ring_n_outer": args.ring_n_outer,
            "ring_inner_radius_min": args.ring_inner_radius_min,
            "ring_inner_radius_max": args.ring_inner_radius_max,
            "ring_outer_gap_min": args.ring_outer_gap_min,
            "ring_outer_gap_max": args.ring_outer_gap_max,
            "ring_outer_rotation_frac_min": args.ring_outer_rotation_frac_min,
            "ring_outer_rotation_frac_max": args.ring_outer_rotation_frac_max,
            "ring_outer_3d_rotation_deg": args.ring_outer_3d_rotation_deg,
            "ring_outer_3d_axis_deg": args.ring_outer_3d_axis_deg,
            "ring_global_rotation_frac_min": args.ring_global_rotation_frac_min,
            "ring_global_rotation_frac_max": args.ring_global_rotation_frac_max,
            "ring_z_phase_sample": args.ring_z_phase_sample,
            "ring_z_phase_radius": args.ring_z_phase_radius,
            "ring_z_phase_far_inner_rotation_deg": args.ring_z_phase_far_inner_rotation_deg,
            "ring_random_parameters": args.ring_random_parameters,
            "ring_shuffle": args.ring_shuffle,
            "add_inner_ring_edges": args.add_inner_ring_edges,
            "add_outer_ring_edges": args.add_outer_ring_edges,
        },
        "model": {
            "model_config": args.model_config,
            "num_layers": args.num_layers,
            "num_ffns": args.num_ffns,
            "use_attn": args.use_attn,
            "use_gaunt_self_tensor_product": args.use_gaunt_self_tensor_product,
            "num_channels": args.num_channels,
            "ffn_hidden_channels": args.ffn_hidden_channels,
            "attn_hidden_channels": args.attn_hidden_channels,
            "num_heads": args.num_heads,
            "attn_alpha_channels": args.attn_alpha_channels,
            "attn_value_channels": args.attn_value_channels,
            "lmax": args.lmax,
            "mmax": args.mmax,
            "num_radial_basis": args.num_radial_basis,
            "edge_channels": args.edge_channels,
            "max_radius": args.max_radius,
            "norm_type": args.norm_type,
        },
        "runtime": {
            "device": args.device,
            "gpu_ids": args.gpu_ids,
            "no_data_parallel": args.no_data_parallel,
        },
    }


def _resolve_sweep_gpu_ids(
    args: argparse.Namespace,
    cuda_available: bool,
    cuda_device_count: int,
) -> list[int]:
    if args.sweep_gpu_ids is not None:
        gpu_ids = list(args.sweep_gpu_ids)
    elif args.gpu_ids is not None:
        gpu_ids = list(args.gpu_ids)
    elif cuda_available:
        gpu_ids = list(range(cuda_device_count))
    else:
        gpu_ids = []

    if not gpu_ids:
        raise ValueError("Sweep mode needs at least one GPU. Pass --sweep-gpu-ids or use a CUDA environment.")
    if cuda_available and (max(gpu_ids) >= cuda_device_count or min(gpu_ids) < 0):
        raise ValueError(f"Selected sweep GPUs {gpu_ids}, but only {cuda_device_count} CUDA devices are visible.")
    return gpu_ids


def _build_sweep_experiments(
    args: argparse.Namespace,
    graph_configs: list[dict[str, Any]],
    model_configs: list[dict[str, Any]],
    train_script: Path,
) -> list[dict[str, Any]]:
    common_args = _sweep_common_args(args)
    experiments = []
    for graph_config in graph_configs:
        for model_config in model_configs:
            lmax_values = model_config.get("lmax_values", [model_config.get("lmax", args.lmax)])
            num_layers_values = model_config.get("num_layers_values", [model_config.get("num_layers", args.num_layers)])
            seed_values = model_config.get("seed_values", [args.seed])
            for lmax in lmax_values:
                for num_layers in num_layers_values:
                    for seed in seed_values:
                        experiment_name = (
                            f"{graph_config['name']}__{model_config['name']}__"
                            f"lmax{lmax}__layers{num_layers}__seed{seed}"
                        )
                        safe_experiment_name = _sanitize_experiment_name(experiment_name)
                        command = [args.sweep_python, str(train_script)]
                        for arg_name, value in common_args.items():
                            _append_cli_arg(command, arg_name, value)
                        for arg_name, value in _sweep_overrides(
                            graph_config, model_config, lmax, num_layers, seed, safe_experiment_name
                        ).items():
                            _append_cli_arg(command, arg_name, value)

                        experiments.append(
                            {
                                "name": safe_experiment_name,
                                "graph_config": graph_config,
                                "model_config": model_config,
                                "lmax": lmax,
                                "num_layers": num_layers,
                                "seed": seed,
                                "command": command,
                            }
                        )
    return experiments


def _sweep_common_args(args: argparse.Namespace) -> dict[str, Any]:
    skipped = set(SWEEP_ONLY_ARG_NAMES)
    skipped.update(
        {
            "device",
            "gpu_ids",
            "no_data_parallel",
            "ring_n_inner",
            "ring_n_outer",
            "model_config",
            "lmax",
            "num_layers",
            "seed",
            "use_gaunt_self_tensor_product",
        }
    )
    return {key: value for key, value in vars(args).items() if key not in skipped}


def _sweep_overrides(
    graph_config: dict[str, Any],
    model_config: dict[str, Any],
    lmax: int,
    num_layers: int,
    seed: int,
    experiment_name: str,
) -> dict[str, Any]:
    overrides = {
        "ring_n_inner": graph_config["ring_n_inner"],
        "ring_n_outer": graph_config["ring_n_outer"],
        "model_config": model_config["model_config"],
        "lmax": lmax,
        "num_layers": num_layers,
        "seed": seed,
        "experiment_name": experiment_name,
        "device": "cuda:0",
        "gpu_ids": [0],
        "no_data_parallel": True,
    }
    if model_config.get("use_gaunt_self_tensor_product", False):
        overrides["use_gaunt_self_tensor_product"] = True
    return overrides


def _print_sweep_dry_run(
    args: argparse.Namespace,
    indexed_experiments: list[tuple[int, dict[str, Any]]],
    gpu_ids: list[int],
) -> None:
    for queue_index, (index, experiment) in enumerate(indexed_experiments):
        gpu_id = gpu_ids[queue_index % len(gpu_ids)]
        result_path = args.sweep_results_dir / f"{index:03d}_{experiment['name']}.result.json"
        command_parts = [*experiment["command"], "--results-json", str(result_path)]
        command = " ".join(command_parts)
        print(f"[dry-run {index:03d}] CUDA_VISIBLE_DEVICES={gpu_id} {command}")


def _start_available_jobs(
    args: argparse.Namespace,
    queue: deque[tuple[int, dict[str, Any]]],
    running: dict[int, dict[str, Any]],
    gpu_ids: list[int],
    repo_root: Path,
) -> None:
    for gpu_id in gpu_ids:
        if gpu_id in running or not queue:
            continue

        index, experiment = queue.popleft()
        log_path = args.sweep_results_dir / f"{index:03d}_{experiment['name']}.log"
        result_path = args.sweep_results_dir / f"{index:03d}_{experiment['name']}.result.json"
        command = [*experiment["command"], "--results-json", str(result_path)]
        log_file = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        running[gpu_id] = {
            "index": index,
            "experiment": experiment,
            "log_path": log_path,
            "result_path": result_path,
            "log_file": log_file,
            "process": process,
            "start_time": time.perf_counter(),
        }
        print(
            f"[start {index:03d}] gpu {gpu_id} {experiment['name']} -> {log_path}",
            flush=True,
        )


def _collect_finished_jobs(
    running: dict[int, dict[str, Any]],
    result_records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    total_experiments: int,
) -> None:
    for gpu_id, job in list(running.items()):
        return_code = job["process"].poll()
        if return_code is None:
            continue

        job["log_file"].close()
        elapsed = time.perf_counter() - job["start_time"]
        status = "ok" if return_code == 0 else f"failed rc={return_code}"
        print(
            f"[done  {job['index']:03d}/{total_experiments:03d}] gpu {gpu_id} "
            f"{job['experiment']['name']} | {status} | {elapsed:.1f}s",
            flush=True,
        )

        record = _finished_job_record(job, gpu_id, return_code)
        result_records.append(record)
        if record["status"] != "ok":
            failures.append(job)
        del running[gpu_id]


def _finished_job_record(job: dict[str, Any], gpu_id: int, return_code: int) -> dict[str, Any]:
    if return_code != 0:
        return _job_failure_record(job, gpu_id, return_code, "failed")
    if not job["result_path"].exists():
        return _job_failure_record(job, gpu_id, return_code, "missing_result_json")

    result_record = json.loads(job["result_path"].read_text(encoding="utf-8"))
    result_record["experiment_name"] = result_record.get("experiment_name") or job["experiment"]["name"]
    result_record["gpu_id"] = gpu_id
    result_record["return_code"] = return_code
    result_record["log_path"] = str(job["log_path"])
    result_record["result_path"] = str(job["result_path"])
    return result_record


def _job_failure_record(
    job: dict[str, Any],
    gpu_id: int,
    return_code: int,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_name": job["experiment"]["name"],
        "status": status,
        "gpu_id": gpu_id,
        "return_code": return_code,
        "log_path": str(job["log_path"]),
        "result_path": str(job["result_path"]),
        "config": {
            "graph": job["experiment"]["graph_config"],
            "model": {
                **job["experiment"]["model_config"],
                "lmax": job["experiment"]["lmax"],
                "num_layers": job["experiment"]["num_layers"],
            },
        },
    }


def _write_sweep_result_files(results_dir: Path, records: list[dict[str, Any]]) -> None:
    json_path = results_dir / "sweep_results.json"
    jsonl_path = results_dir / "sweep_results.jsonl"
    csv_path = results_dir / "sweep_results.csv"

    json_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for record in records:
            jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")

    rows = [_flatten_summary_record(record) for record in records]
    fieldnames = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_summary_record(record: dict[str, Any]) -> dict[str, Any]:
    config = record.get("config", {})
    graph_config = config.get("graph", {})
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    metrics = record.get("metrics", {})
    dataset = record.get("dataset", {})
    return {
        "experiment_name": record.get("experiment_name"),
        "status": record.get("status"),
        "ring_n_inner": graph_config.get("ring_n_inner"),
        "ring_n_outer": graph_config.get("ring_n_outer"),
        "ring_n_graphs": graph_config.get("ring_n_graphs"),
        "ring_seed": graph_config.get("ring_seed"),
        "model_config": model_config.get("model_config"),
        "lmax": model_config.get("lmax"),
        "mmax": model_config.get("mmax"),
        "num_layers": model_config.get("num_layers"),
        "num_ffns": model_config.get("num_ffns"),
        "num_channels": model_config.get("num_channels"),
        "use_attn": model_config.get("use_attn"),
        "use_gaunt_self_tensor_product": model_config.get("use_gaunt_self_tensor_product"),
        "epochs_requested": training_config.get("epochs"),
        "epoch": metrics.get("epoch"),
        "loss": metrics.get("loss"),
        "accuracy": metrics.get("accuracy"),
        "margin_accuracy": metrics.get("margin_accuracy"),
        "best_accuracy": metrics.get("best_accuracy"),
        "best_margin_accuracy": metrics.get("best_margin_accuracy"),
        "train_time": metrics.get("train_time"),
        "num_graphs": dataset.get("num_graphs"),
        "num_nodes": dataset.get("num_nodes"),
        "num_directed_edges": dataset.get("num_directed_edges"),
        "gpu_id": record.get("gpu_id"),
        "return_code": record.get("return_code"),
        "log_path": record.get("log_path"),
        "result_path": record.get("result_path"),
    }


def _append_cli_arg(command: list[str], arg_name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            command.append(_cli_flag(arg_name))
        return
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        if value:
            command.append(_cli_flag(arg_name))
            command.extend(str(item) for item in value)
        return
    command.extend([_cli_flag(arg_name), str(value)])


def _cli_flag(arg_name: str) -> str:
    return "--" + arg_name.replace("_", "-")


def _sanitize_experiment_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)
