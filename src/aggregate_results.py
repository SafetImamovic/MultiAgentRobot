"""
Aggregate multi-seed experiment results and generate summary table + plots.

Usage:
    python src/aggregate_results.py --results-dir results
"""

import os
import sys
import json
import argparse
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ALGO_DISPLAY_NAMES = {
    "ppo": "PPO (Centralized)",
    "mappo": "MAPPO",
    "qmix": "QMIX",
    "idqn": "IDQN",
    "dqn_joint": "DQN (Joint)",
    "random": "Random",
}

ALGO_ORDER = ["ppo", "mappo", "qmix", "idqn", "dqn_joint", "random"]


def load_all_metrics(results_dir):
    """Load eval_metrics.json from all (algo, seed) directories."""
    data = {}
    for algo in ALGO_ORDER:
        algo_dir = os.path.join(results_dir, algo)
        if not os.path.isdir(algo_dir):
            continue
        seed_dirs = sorted(glob(os.path.join(algo_dir, "seed_*")))
        metrics_list = []
        for sd in seed_dirs:
            path = os.path.join(sd, "eval_metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    metrics_list.append(json.load(f))
        if metrics_list:
            data[algo] = metrics_list
    return data


def compute_summary(data):
    """Compute mean +/- std for each algorithm and metric."""
    rows = []
    for algo in ALGO_ORDER:
        if algo not in data:
            continue
        metrics_list = data[algo]
        n_seeds = len(metrics_list)
        row = {"algorithm": ALGO_DISPLAY_NAMES.get(algo, algo), "n_seeds": n_seeds}

        for key in ["success_rate", "avg_reward", "avg_coverage", "avg_collisions"]:
            vals = [m[key] for m in metrics_list]
            row[f"{key}_mean"] = np.mean(vals)
            row[f"{key}_std"] = np.std(vals)
            row[f"{key}_str"] = f"{np.mean(vals):.1f} $\\pm$ {np.std(vals):.1f}"

        rows.append(row)
    return rows


def save_summary_csv(rows, output_dir):
    """Save summary table as CSV."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "summary_table.csv")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Summary table saved to: {path}")
    return df


def print_latex_table(rows):
    """Print LaTeX-ready table for the paper."""
    print("\n% --- LaTeX Table (copy into paper) ---")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Algorithm comparison on 8$\\times$8 grid with 2 agents (5 seeds, mean $\\pm$ std)}\\label{tab:algorithm_comparison}")
    print("\\begin{tabular}{lcccc}")
    print("\\toprule")
    print("Algorithm & Success Rate & Avg. Reward & Coverage & Collisions \\\\")
    print("\\midrule")
    for row in rows:
        name = row["algorithm"]
        sr = row["success_rate_str"]
        ar = row["avg_reward_str"]
        ac = row["avg_coverage_str"]
        co = row["avg_collisions_str"]
        print(f"{name} & {sr}\\% & {ar} & {ac}\\% & {co} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print("% --- End LaTeX Table ---\n")


def plot_bar_comparison(rows, output_dir):
    """Generate bar chart with error bars for all metrics."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        ("success_rate", "Success Rate (%)", "tab:blue"),
        ("avg_reward", "Average Reward", "tab:green"),
        ("avg_coverage", "Coverage (%)", "tab:orange"),
        ("avg_collisions", "Average Collisions", "tab:red"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Algorithm Comparison (5 Seeds, Mean ± Std)", fontsize=16, fontweight="bold")

    names = [r["algorithm"] for r in rows]

    for ax, (key, title, color) in zip(axes.flat, metrics):
        means = [r[f"{key}_mean"] for r in rows]
        stds = [r[f"{key}_std"] for r in rows]

        bars = ax.bar(names, means, yerr=stds, capsize=5, color=color, alpha=0.7,
                       edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.5,
                    f"{mean:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_bar_chart.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Bar chart saved to: {path}")


def plot_learning_curves(results_dir, output_dir):
    """Plot training curves with std bands from episode_data.csv files."""
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Learning Curves (5 Seeds, Mean ± Std)", fontsize=16, fontweight="bold")

    colors = {
        "ppo": "#2196F3", "mappo": "#9C27B0", "qmix": "#00BCD4",
        "idqn": "#4CAF50", "dqn_joint": "#FF9800", "random": "#9E9E9E",
    }

    metrics_cols = [
        ("total_reward", "Total Reward"),
        ("coverage_pct", "Coverage (%)"),
        ("total_collisions", "Collisions"),
        ("terminated", "Success Rate (Running Avg)"),
    ]

    # Y-axis limits for clean presentation
    y_limits = {
        "total_reward": (-200, 300),
        "coverage_pct": (0, 1.1),
        "total_collisions": (0, 50),
        "terminated": (-0.05, 1.1),
    }

    # Map eval metric keys to episode_data.csv column names for random baseline
    random_metric_map = {
        "total_reward": "avg_reward",
        "coverage_pct": "avg_coverage",
        "total_collisions": "avg_collisions",
        "terminated": "success_rate",
    }

    # Load random baseline eval metrics for horizontal reference lines
    random_means = {}
    random_dir = os.path.join(results_dir, "random")
    if os.path.isdir(random_dir):
        random_vals = {col: [] for col, _ in metrics_cols}
        for sd in sorted(glob(os.path.join(random_dir, "seed_*"))):
            path = os.path.join(sd, "eval_metrics.json")
            if os.path.exists(path):
                with open(path) as f:
                    m = json.load(f)
                    random_vals["total_reward"].append(m["avg_reward"])
                    random_vals["coverage_pct"].append(m["avg_coverage"] / 100.0)
                    random_vals["total_collisions"].append(m["avg_collisions"])
                    random_vals["terminated"].append(m["success_rate"] / 100.0)
        for col in random_vals:
            if random_vals[col]:
                random_means[col] = np.mean(random_vals[col])

    # First pass: load data and find max episode count
    N_POINTS = 300
    algo_data = {}  # algo -> {min_len, seed_dfs}

    for algo in ALGO_ORDER:
        if algo == "random":
            continue

        algo_dir = os.path.join(results_dir, algo)
        if not os.path.isdir(algo_dir):
            continue

        seed_dfs = []
        for sd in sorted(glob(os.path.join(algo_dir, "seed_*"))):
            csv_path = os.path.join(sd, "episode_data.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) > 0:
                        seed_dfs.append(df)
                except Exception:
                    continue

        if not seed_dfs:
            continue

        min_len = min(len(df) for df in seed_dfs)
        if min_len < 10:
            continue

        algo_data[algo] = {"min_len": min_len, "seed_dfs": seed_dfs}

    max_episodes = min(2000, max((d["min_len"] for d in algo_data.values()), default=1))
    x = np.linspace(0, max_episodes, N_POINTS)

    # Second pass: plot each algorithm, extending short ones to max_episodes
    for algo, ad in algo_data.items():
        min_len = ad["min_len"]
        seed_dfs = ad["seed_dfs"]
        smooth_window = max(1, min_len // 50)

        for ax_idx, (col, title) in enumerate(metrics_cols):
            ax = axes.flat[ax_idx]
            seed_series = []
            for df in seed_dfs:
                smoothed = df[col].iloc[:min_len].rolling(
                    window=smooth_window, min_periods=1, center=True
                ).mean().values
                # Resample to N_POINTS over this algo's real range
                real_indices = np.linspace(0, len(smoothed) - 1, N_POINTS).astype(int)
                resampled = smoothed[real_indices]
                # Extend to max_episodes: hold final value flat
                real_x = np.linspace(0, min_len, N_POINTS)
                extended = np.interp(x, real_x, resampled,
                                     right=resampled[-1])
                seed_series.append(extended)

            seed_array = np.array(seed_series)
            mean = np.mean(seed_array, axis=0)
            std = np.std(seed_array, axis=0)

            color = colors.get(algo, "black")
            label = ALGO_DISPLAY_NAMES.get(algo, algo)
            ax.plot(x, mean, color=color, label=label, linewidth=2.0)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

    # Add random baseline as dashed horizontal line
    for ax_idx, (col, title) in enumerate(metrics_cols):
        ax = axes.flat[ax_idx]
        if col in random_means:
            ax.axhline(y=random_means[col], color=colors["random"], linestyle="--",
                        linewidth=2.0, label="Random", alpha=0.8)

    for ax_idx, (col, title) in enumerate(metrics_cols):
        ax = axes.flat[ax_idx]
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.legend(fontsize=9, loc="best", framealpha=0.9)
        ax.set_ylim(y_limits.get(col))
        ax.set_xlim(0, max_episodes)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Learning curves saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experiment results")
    parser.add_argument("--results-dir", default="results", help="Root results directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots/tables")
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.results_dir, "summary")

    data = load_all_metrics(args.results_dir)
    if not data:
        print(f"No results found in {args.results_dir}/")
        sys.exit(1)

    print(f"Found results for: {list(data.keys())}")
    for algo, metrics_list in data.items():
        print(f"  {algo}: {len(metrics_list)} seeds")

    rows = compute_summary(data)
    save_summary_csv(rows, output_dir)
    print_latex_table(rows)
    plot_bar_comparison(rows, output_dir)
    plot_learning_curves(args.results_dir, output_dir)

    print("\nDone! Copy the LaTeX table above into your paper.")


if __name__ == "__main__":
    main()
