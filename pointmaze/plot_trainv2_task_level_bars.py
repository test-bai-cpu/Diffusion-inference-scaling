"""
Plot train-v2 MapCond vs DFS success rates as per-task bar charts.

For each task, the figure shows grouped bars at each difficulty level. Bar
height is the mean success rate over maze variants, error bars show +/-1 std
over variants, and dots show the individual variant success rates.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_success_rate import parse_results, _style_axes


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    "results_dfs_pointmaze-giant-newvar-navigate-v0_"
    "mazev1-cond-trainonv2.txt"
)
DEFAULT_DFS_INPUT = "results_newvar_level.txt"
DEFAULT_OUTPUT_PATTERN = "trainv2_success_rate_bars_task{task}.png"


def _resolve_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return HERE / path


def _unique_ints(values):
    return list(dict.fromkeys(int(value) for value in values))


def load_results(input_path, method_label, tasks=None, levels=None):
    """Load result rows for one method and keep only new-variant rows."""
    path = _resolve_path(input_path)
    if not path.exists():
        raise SystemExit(f"Missing result file: {path}")

    df = parse_results(str(path))
    if df.empty:
        raise SystemExit(f"No parsable rows found in {path}")

    df = df[~df.get("is_original", False)].copy()
    if tasks is not None:
        df = df[df["task"].isin(tasks)].copy()
    if levels is not None:
        df = df[df["level"].isin(levels)].copy()

    if df.empty:
        raise SystemExit("No rows left after filtering.")
    df["method"] = method_label
    df["source"] = path.name
    return df


def _level_stats(task_df, method_order):
    """Mean duplicate rows per variant first, then summarize over variants."""
    variant_means = (
        task_df.groupby(["method", "task", "level", "variant"], observed=True)
               ["success_rate"]
               .mean()
               .reset_index()
    )
    summary = (
        variant_means.groupby(["method", "level"], observed=True)["success_rate"]
                     .agg(mean="mean",
                          std=lambda x: x.std(ddof=0),
                          count="count")
                     .reset_index()
                     .sort_values(["level", "method"])
    )
    summary["std"] = summary["std"].fillna(0.0)
    variant_means["method"] = pd.Categorical(
        variant_means["method"], categories=method_order, ordered=True,
    )
    summary["method"] = pd.Categorical(
        summary["method"], categories=method_order, ordered=True,
    )
    return variant_means, summary


def _set_ylim(ax, summary, raw_values=None, show_error=True):
    if show_error:
        y_lo = (summary["mean"] - summary["std"]).min()
        y_hi = (summary["mean"] + summary["std"]).max()
    else:
        y_lo = summary["mean"].min()
        y_hi = summary["mean"].max()

    if raw_values is not None:
        raw_values = np.asarray(raw_values, dtype=float)
        if len(raw_values):
            y_lo = min(y_lo, np.nanmin(raw_values))
            y_hi = max(y_hi, np.nanmax(raw_values))

    y_pad = max(2.0, 0.06 * (y_hi - y_lo))
    bottom = 0.0
    top = min(105.0, y_hi + y_pad) if y_hi < 100.0 else min(105.0, y_hi + y_pad)
    ax.set_ylim(bottom, top)


def _style_legend(ax):
    leg = ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=9, frameon=True, framealpha=0.92,
        edgecolor="#cccccc", fancybox=False,
        handlelength=1.6, handletextpad=0.55,
        borderpad=0.45, labelspacing=0.32, borderaxespad=0.0,
    )
    leg.get_frame().set_linewidth(0.6)


def plot_task_bars(df, task, output_path, method_order,
                   show_error=True, show_dots=True):
    """Plot one task as bars over difficulty levels."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.25,
                  rc={"font.family": "DejaVu Sans"})

    task_df = df[df["task"] == task].copy()
    if task_df.empty:
        raise SystemExit(f"No rows found for task {task}.")

    variant_means, summary = _level_stats(task_df, method_order)
    present_methods = set(task_df["method"].astype(str))
    methods = [method for method in method_order if method in present_methods]
    levels = sorted(task_df["level"].astype(int).unique())
    level_to_x = {level: i for i, level in enumerate(levels)}
    x = np.arange(len(levels))
    palette = dict(zip(method_order, sns.color_palette("deep", n_colors=len(method_order))))
    bar_width = min(0.34, 0.76 / max(len(methods), 1))
    offsets = {
        method: (i - (len(methods) - 1) / 2.0) * bar_width
        for i, method in enumerate(methods)
    }

    fig_width = max(6.0, 1.0 * len(levels) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, 3.8))

    for method in methods:
        method_summary = summary[summary["method"].astype(str) == method].copy()
        positions = np.array([
            level_to_x[int(level)] + offsets[method]
            for level in method_summary["level"]
        ], dtype=float)
        means = method_summary["mean"].to_numpy(dtype=float)
        stds = method_summary["std"].to_numpy(dtype=float)
        ax.bar(
            positions, means,
            width=bar_width * 0.92, color=palette[method],
            edgecolor="#333333", linewidth=0.6, zorder=2,
            label=method,
        )

        if show_error:
            ax.errorbar(
                positions, means, yerr=stds,
                fmt="none", ecolor="#333333", elinewidth=1.0,
                capsize=3.5, capthick=1.0, zorder=4,
            )

        if show_dots:
            for _, row in method_summary.iterrows():
                level = int(row["level"])
                level_points = (
                    variant_means.loc[
                        (variant_means["method"].astype(str) == method)
                        & (variant_means["level"].astype(int) == level),
                        ["variant", "success_rate"],
                    ]
                    .sort_values("variant")
                )
                points = level_points["success_rate"].to_numpy(dtype=float)
                if len(points) == 1:
                    jitter = np.array([0.0])
                else:
                    jitter = np.linspace(-0.08, 0.08, len(points))
                bar_center = level_to_x[level] + offsets[method]
                ax.scatter(
                    np.full(len(points), bar_center) + jitter, points,
                    s=28, facecolor="white", edgecolor=palette[method],
                    linewidth=0.8, zorder=5,
                )

    _set_ylim(
        ax, summary,
        raw_values=variant_means["success_rate"],
        show_error=show_error,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(level) for level in levels])
    ax.set_xlabel("Difficulty Level", labelpad=6)
    ax.set_ylabel("Success Rate (%)", labelpad=6)
    ax.set_title(f"DFS vs Train v2 MapCond: Task {int(task)}", fontsize=11, pad=8)

    _style_axes(ax)
    _style_legend(ax)
    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def print_summary(df, method_order):
    rows = []
    for task in sorted(df["task"].unique()):
        task_df = df[df["task"] == task]
        _, summary = _level_stats(task_df, method_order)
        for _, row in summary.iterrows():
            rows.append({
                "method": str(row["method"]),
                "task": int(task),
                "level": int(row["level"]),
                "mean": row["mean"],
                "std": row["std"],
                "count": int(row["count"]),
            })

    table = pd.DataFrame(rows)
    print("=" * 96)
    print("DFS vs train-v2 success rate by task and difficulty level")
    print("=" * 96)
    print(
        table.to_string(
            index=False,
            formatters={
                "mean": "{:.2f}".format,
                "std": "{:.2f}".format,
            },
        )
    )
    print("=" * 96)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Train-v2 result file. Default: %(default)s")
    parser.add_argument("--dfs-input", default=DEFAULT_DFS_INPUT,
                        help="DFS baseline result file. Default: %(default)s")
    parser.add_argument("--input-label", default="Train v2 MapCond",
                        help="Legend label for --input.")
    parser.add_argument("--dfs-label", default="DFS",
                        help="Legend label for --dfs-input.")
    parser.add_argument("--task", nargs="+", type=int, default=None,
                        help="Task id(s) to plot. Default: all tasks found.")
    parser.add_argument("--levels", nargs="+", type=int, default=None,
                        help="Levels to include. Default: all levels found.")
    parser.add_argument("--output-dir", default=str(HERE),
                        help="Directory for generated figures. Default: script directory.")
    parser.add_argument("--output", default=None,
                        help="Output path for a single task. If multiple tasks are plotted, "
                             "_taskT is inserted before the suffix.")
    parser.add_argument("--no-error", action="store_true",
                        help="Hide std error bars.")
    parser.add_argument("--no-dots", action="store_true",
                        help="Hide individual variant dots.")
    args = parser.parse_args()

    tasks = _unique_ints(args.task) if args.task is not None else None
    levels = _unique_ints(args.levels) if args.levels is not None else None
    method_order = [args.dfs_label, args.input_label]

    train_df = load_results(
        args.input, args.input_label, tasks=tasks, levels=levels,
    )
    dfs_df = load_results(
        args.dfs_input, args.dfs_label, tasks=tasks, levels=levels,
    )
    df = pd.concat([dfs_df, train_df], ignore_index=True)
    df["method"] = pd.Categorical(
        df["method"], categories=method_order, ordered=True,
    )

    tasks_to_plot = sorted(train_df["task"].unique()) if tasks is None else tasks
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_summary(df[df["task"].isin(tasks_to_plot)].copy(), method_order)

    for task in tasks_to_plot:
        if args.output is None:
            output = output_dir / DEFAULT_OUTPUT_PATTERN.format(task=int(task))
        else:
            output_path = Path(args.output)
            if len(tasks_to_plot) == 1:
                output = output_path
            else:
                output = output_path.with_name(
                    f"{output_path.stem}_task{int(task)}{output_path.suffix}"
                )

        plot_task_bars(
            df, int(task), str(output), method_order,
            show_error=not args.no_error,
            show_dots=not args.no_dots,
        )


if __name__ == "__main__":
    main()
