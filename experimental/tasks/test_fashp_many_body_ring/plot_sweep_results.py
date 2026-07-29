#!/usr/bin/env python3
"""Plot sweep results as heatmap comparing margin accuracy across graph configurations and model parameters."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import matplotlib.colors as mcolors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to sweep_results.csv",
    )
    parser.add_argument(
        "--additional-csv",
        type=Path,
        nargs="*",
        default=[],
        help="Additional result CSVs to combine with --csv before averaging each heatmap cell.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots. Defaults to same directory as CSV.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="margin_accuracy",
        choices=["margin_accuracy", "accuracy", "best_margin_accuracy", "best_accuracy", "loss"],
        help="Metric to plot on Y-axis",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="num_layers",
        choices=["num_layers", "lmax", "model_config"],
        help="Parameter to create separate plots for",
    )
    parser.add_argument(
        "--y-axis",
        type=str,
        default="lmax",
        choices=["lmax", "num_layers", "model_config"],
        help="Parameter to use on Y-axis of heatmap",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Show values as text in each cell",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 6],
        help="Figure size (width, height)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figures",
    )
    return parser.parse_args()


def create_graph_label(row):
    """Create a readable label for graph configuration."""
    return f"{int(row['ring_n_inner'])}in/{int(row['ring_n_outer'])}out"


def plot_sweep_results(df, metric, group_by, y_axis, output_dir, figsize, dpi, annotate):
    """Create heatmap plots grouped by a parameter."""

    if metric == "margin_accuracy":
        df = df.copy()
        df[metric] = np.where(np.isclose(df[metric], 1.0), 1.0, 0.5)
    
    # Create graph configuration labels
    df["graph_config"] = df.apply(create_graph_label, axis=1)
    
    # Get unique values for grouping
    group_values = sorted(df[group_by].unique())
    
    print(f"Creating {len(group_values)} heatmap plots grouped by {group_by}...")
    
    for group_value in group_values:
        # Filter data for this group
        df_group = df[df[group_by] == group_value].copy()
        
        # Get unique graph configs and y-axis values
        graph_configs = sorted(
            df_group[["ring_n_inner", "ring_n_outer", "graph_config"]].drop_duplicates().values.tolist(),
            key=lambda x: (x[0], x[1])
        )
        graph_labels = [x[2] for x in graph_configs]
        y_values = sorted(df_group[y_axis].unique())
        
        # Create matrix for heatmap
        matrix = np.full((len(y_values), len(graph_labels)), np.nan)
        
        for i, y_val in enumerate(y_values):
            for j, (n_inner, n_outer, label) in enumerate(graph_configs):
                rows = df_group[
                    (df_group["ring_n_inner"] == n_inner) &
                    (df_group["ring_n_outer"] == n_outer) &
                    (df_group[y_axis] == y_val)
                ]
                if len(rows) > 0:
                    matrix[i, j] = rows[metric].mean()
        
        # Flip matrix so lowest y-value is at bottom
        matrix = np.flipud(matrix)
        y_values_display = list(reversed(y_values))
        
        # Calculate figure size to make cells square
        cell_size = 0.6  # inches per cell
        fig_width = len(graph_labels) * cell_size + 2  # +2 for colorbar and margins
        fig_height = len(y_values) * cell_size + 1.5  # +1.5 for title and labels
        
        # Create figure
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        
        # Create custom colormap (red -> yellow -> green)
        if metric == "margin_accuracy":
            cmap = mcolors.ListedColormap(["red", "green"], name="binary_accuracy")
            norm = mcolors.BoundaryNorm([0.0, 0.75, 1.01], cmap.N)
        else:
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "accuracy",
                ["red", "yellow", "green"]
            )
            norm = mcolors.Normalize(vmin=0.5 if metric == "accuracy" else 0, vmax=1)
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
        
        # Add grid lines between cells
        for i in range(len(y_values) + 1):
            ax.axhline(i - 0.5, color='black', linewidth=1.5)
        for j in range(len(graph_labels) + 1):
            ax.axvline(j - 0.5, color='black', linewidth=1.5)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(graph_labels)))
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_xticklabels(graph_labels)
        ax.set_yticklabels(y_values_display)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations if requested
        if annotate:
            for i in range(len(y_values)):
                for j in range(len(graph_labels)):
                    if not np.isnan(matrix[i, j]):
                        ax.text(
                            j, i - 0.10, f"{matrix[i, j]:.2f}",
                            ha="center", va="center",
                            color="black" if matrix[i, j] > 0.5 else "white",
                            fontsize=8
                        )
                        y_val = y_values_display[i]
                        n_averaged = len(
                            df_group[
                                (df_group["ring_n_inner"] == graph_configs[j][0]) &
                                (df_group["ring_n_outer"] == graph_configs[j][1]) &
                                (df_group[y_axis] == y_val)
                            ]
                        )
                        ax.text(
                            j, i + 0.22, f"n={n_averaged}",
                            ha="center", va="center",
                            color="black" if matrix[i, j] > 0.5 else "white",
                            fontsize=5,
                        )
        
        # Labels and title
        ax.set_xlabel("Graph Configuration", fontsize=12, fontweight="bold")
        ax.set_ylabel(y_axis, fontsize=12, fontweight="bold")
        if group_by == "num_layers":
            title = f"Equiformer num layers = {group_value}"
        else:
            title = f"{group_by} = {group_value}"
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        plt.tight_layout()
        
        # Save figure
        safe_group_value = str(group_value).replace("/", "_").replace(" ", "_")
        output_path = output_dir / f"{metric}_heatmap__{group_by}_{safe_group_value}.png"
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {output_path}")
        plt.close()
    
    print(f"Done! Saved {len(group_values)} plots to {output_dir}")


def main():
    args = parse_args()
    
    # Read CSV
    if not args.csv.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv}")
    
    csv_paths = [args.csv, *args.additional_csv]
    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        print(f"Reading: {csv_path}")
    df = pd.concat([pd.read_csv(csv_path) for csv_path in csv_paths], ignore_index=True)
    
    # Filter to only successful experiments
    df = df[df["status"] == "ok"].copy()
    print(f"Found {len(df)} successful experiments across {len(csv_paths)} result files")
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    plot_sweep_results(
        df=df,
        metric=args.metric,
        group_by=args.group_by,
        y_axis=args.y_axis,
        output_dir=output_dir,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
        annotate=args.annotate,
    )


if __name__ == "__main__":
    main()
