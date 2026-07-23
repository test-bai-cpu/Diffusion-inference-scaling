"""
Compare success-rate curves for selected pointmaze methods.

Default inputs:
  - pure DFS
  - DFS + distance guidance, no corner check
  - MapCond + corner + distance, MapCond trained on maze v2
  - MapCond + corner + distance, MapCond trained on maze v1

The default task figure plots Task 1 across levels 1..4, averaging the three
maze variants at each level and showing +/-1 std across variants. Pass multiple
tasks as "--task 1 2 3" to write one aggregate figure that averages those tasks
at each level. Use "--all-tasks" to write one separate figure per task found.
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

DEFAULT_METHODS = (
    ("results_newvar_level.txt", "Baseline"),
    # ("results_dfs_pointmaze-giant-newvar-navigate-v0_dfs_trainall.txt", "Baseline"),
    # ("results_newvar_bfs_guidance_level.txt", "DFS + distance (no corner)"),
    (
        "results_dfs_pointmaze-giant-newvar-navigate-v0_"
        "mazev1-cond-trainonv2.txt",
        "Ours",
    ),
    # (
    #     "results_dfs_pointmaze-giant-newvar-navigate-v0_"
    #     "mazev1-cond-trainonv1.txt",
    #     "MapCond + corner + distance (train v1)",
    # ),
)


def _resolve_path(path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return HERE / path


def _input_specs(specs):
    """Parse repeated PATH=LABEL specs, or return the default method list."""
    if not specs:
        return [(_resolve_path(path), label) for path, label in DEFAULT_METHODS]

    parsed = []
    for spec in specs:
        if "=" in spec:
            path, label = spec.split("=", 1)
        else:
            path = spec
            label = Path(path).stem
            label = label.replace("results_dfs_pointmaze-giant-newvar-navigate-v0_", "")
        parsed.append((_resolve_path(path), label))
    return parsed


def _unique_ints(values):
    return list(dict.fromkeys(int(value) for value in values))


def _task_label(tasks):
    tasks = sorted(_unique_ints(tasks))
    if not tasks:
        return "tasks"
    if tasks == list(range(tasks[0], tasks[-1] + 1)):
        return f"tasks {tasks[0]}-{tasks[-1]}"
    return "tasks " + ", ".join(str(task) for task in tasks)


def _task_suffix(tasks):
    tasks = sorted(_unique_ints(tasks))
    if not tasks:
        return "tasks"
    if tasks == list(range(tasks[0], tasks[-1] + 1)):
        return f"tasks{tasks[0]}-{tasks[-1]}"
    return "tasks" + "-".join(str(task) for task in tasks)


def load_method_results(specs, tasks=None, levels=None, strict=False):
    """Load all method logs into one DataFrame."""
    frames = []
    labels = []
    for path, label in specs:
        labels.append(label)
        if not path.exists():
            msg = f"Missing result file: {path}"
            if strict:
                raise SystemExit(msg)
            print(f"Warning: {msg}")
            continue

        df = parse_results(str(path))
        if df.empty:
            msg = f"No parsable rows found in {path}"
            if strict:
                raise SystemExit(msg)
            print(f"Warning: {msg}")
            continue

        # Keep only new-variant rows. Original-maze baselines are useful in
        # plot_success_rate.py, but this comparison is level 1..N only.
        df = df[~df.get("is_original", False)].copy()
        if tasks is not None:
            df = df[df["task"].isin(tasks)].copy()
        if levels is not None:
            df = df[df["level"].isin(levels)].copy()

        if df.empty:
            msg = f"No matching rows after filtering in {path}"
            if strict:
                raise SystemExit(msg)
            print(f"Warning: {msg}")
            continue

        df["method"] = label
        df["source"] = path.name
        frames.append(df)

    if not frames:
        raise SystemExit("No method rows loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined["method"] = pd.Categorical(
        combined["method"], categories=labels, ordered=True,
    )
    return combined


def _method_stats(df):
    variant_means = (df.groupby(["method", "task", "level", "variant"], observed=True)
                       ["success_rate"].mean()
                       .reset_index())
    stats = (variant_means.groupby(["method", "task", "level"], observed=True)
                          ["success_rate"]
                          .agg(mean="mean",
                               std=lambda x: x.std(ddof=0),
                               count="count")
                          .reset_index())
    return stats


def _task_average_stats(df, tasks):
    """Average variants first, then average the selected tasks per level."""
    tasks = _unique_ints(tasks)
    task_df = df[df["task"].isin(tasks)].copy()
    if task_df.empty:
        raise SystemExit(f"No rows found for {_task_label(tasks)}.")

    variant_means = (
        task_df.groupby(["method", "task", "level", "variant"], observed=True)
               ["success_rate"]
               .mean()
               .reset_index()
    )
    task_means = (
        variant_means.groupby(["method", "task", "level"], observed=True)
                     ["success_rate"]
                     .mean()
                     .reset_index()
    )
    stats = (
        task_means.groupby(["method", "level"], observed=True)["success_rate"]
                  .agg(mean="mean",
                       std=lambda x: x.std(ddof=0),
                       count="count")
                  .reset_index()
    )
    return stats


def _style_outside_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles=handles, labels=labels,
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        fontsize=9, frameon=True, framealpha=0.92,
        edgecolor="#cccccc", fancybox=False,
        handlelength=2.0, handletextpad=0.55,
        borderpad=0.45, labelspacing=0.32, borderaxespad=0.0,
    )
    leg.get_frame().set_linewidth(0.6)
    if leg.get_title() is not None:
        leg.get_title().set_visible(False)
    legend_handles = getattr(leg, "legend_handles", getattr(leg, "legendHandles", []))
    for h in legend_handles:
        h.set_markersize(5)
        h.set_markeredgewidth(0.6)
        h.set_linewidth(1.6)


def _set_ylim(ax, stats, show_band=True):
    if show_band:
        y_lo = (stats["mean"] - stats["std"]).min()
        y_hi = (stats["mean"] + stats["std"]).max()
    else:
        y_lo = stats["mean"].min()
        y_hi = stats["mean"].max()
    y_pad = max(2.0, 0.06 * (y_hi - y_lo))
    bottom = max(0.0, y_lo - y_pad) if y_lo > 0.0 else y_lo - y_pad
    top = min(100.0, y_hi + y_pad) if y_hi < 100.0 else y_hi + y_pad
    ax.set_ylim(bottom, top)


def plot_task_comparison(df, task, output_path, show_band=True):
    """Plot method curves for one task across levels."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.25,
                  rc={"font.family": "DejaVu Sans"})

    task_df = df[df["task"] == task].copy()
    if task_df.empty:
        raise SystemExit(f"No rows found for task {task}.")

    stats = _method_stats(task_df)
    methods = [method for method in df["method"].cat.categories
               if method in set(task_df["method"].dropna())]
    palette = dict(zip(methods, sns.color_palette("deep", n_colors=len(methods))))
    markers = dict(zip(methods, ["o", "s", "D", "^", "v", "P", "X", "h"][:len(methods)]))

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for method in methods:
        sub = stats[stats["method"] == method].sort_values("level")
        x = sub["level"].to_numpy(dtype=float)
        mean = sub["mean"].to_numpy(dtype=float)
        std = sub["std"].to_numpy(dtype=float)
        if show_band:
            ax.fill_between(
                x, mean - std, mean + std,
                color=palette[method], alpha=0.15, linewidth=0, zorder=1,
            )
        ax.plot(
            x, mean, color=palette[method], linewidth=2.3,
            marker=markers[method], markersize=7,
            markeredgecolor="white", markeredgewidth=0.9,
            zorder=3, label=method,
        )

    levels = sorted(task_df["level"].unique())
    ax.set_xticks(levels)
    ax.set_xticklabels([str(int(level)) for level in levels])
    if len(levels) == 1:
        ax.set_xlim(levels[0] - 0.3, levels[0] + 0.3)
    else:
        ax.set_xlim(levels[0] - 0.12, levels[-1] + 0.12)

    _set_ylim(ax, stats, show_band=show_band)
    ax.set_xlabel("Difficulty Level", labelpad=6)
    ax.set_ylabel("Average Success Rate (%)", labelpad=6)
    ax.set_title(f"Task {task}: method comparison", fontsize=11, pad=8)

    _style_axes(ax)
    _style_outside_legend(ax)
    plt.tight_layout(pad=0.8)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def plot_task_average_comparison(df, tasks, output_path, show_band=True):
    """Plot method curves after averaging selected tasks at each level."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.25,
                  rc={"font.family": "DejaVu Sans"})

    tasks = _unique_ints(tasks)
    task_df = df[df["task"].isin(tasks)].copy()
    if task_df.empty:
        raise SystemExit(f"No rows found for {_task_label(tasks)}.")

    stats = _task_average_stats(task_df, tasks)
    methods = [method for method in df["method"].cat.categories
               if method in set(task_df["method"].dropna())]
    palette = dict(zip(methods, sns.color_palette("deep", n_colors=len(methods))))
    markers = dict(zip(methods, ["o", "s", "D", "^", "v", "P", "X", "h"][:len(methods)]))

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for method in methods:
        sub = stats[stats["method"] == method].sort_values("level")
        x = sub["level"].to_numpy(dtype=float)
        mean = sub["mean"].to_numpy(dtype=float)
        std = sub["std"].to_numpy(dtype=float)
        if show_band:
            ax.fill_between(
                x, mean - std, mean + std,
                color=palette[method], alpha=0.15, linewidth=0, zorder=1,
            )
        ax.plot(
            x, mean, color=palette[method], linewidth=2.3,
            marker=markers[method], markersize=7,
            markeredgecolor="white", markeredgewidth=0.9,
            zorder=3, label=method,
        )

    levels = sorted(task_df["level"].unique())
    ax.set_xticks(levels)
    ax.set_xticklabels([str(int(level)) for level in levels])
    if len(levels) == 1:
        ax.set_xlim(levels[0] - 0.3, levels[0] + 0.3)
    else:
        ax.set_xlim(levels[0] - 0.12, levels[-1] + 0.12)

    _set_ylim(ax, stats, show_band=show_band)
    ax.set_xlabel("Difficulty Level", labelpad=6)
    ax.set_ylabel("Task-Averaged Success Rate (%)", labelpad=6)
    ax.set_title(f"{_task_label(tasks).title()}: method comparison", fontsize=11, pad=8)

    _style_axes(ax)
    _style_outside_legend(ax)
    plt.tight_layout(pad=0.8)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def print_summary(df):
    stats = _method_stats(df)
    for task in sorted(stats["task"].unique()):
        sub = stats[stats["task"] == task]
        table = sub.pivot(index="level", columns="method", values="mean")
        print("=" * 96)
        print(f"Task {int(task)} mean success rate (%) over variants")
        print("=" * 96)
        print(table.round(2).to_string())
        print("=" * 96)


def print_task_average_summary(df, tasks):
    tasks = _unique_ints(tasks)
    task_df = df[df["task"].isin(tasks)].copy()
    if task_df.empty:
        print(f"Warning: No rows found for {_task_label(tasks)}.")
        return

    methods = [method for method in df["method"].cat.categories
               if method in set(task_df["method"].dropna())]
    expected = set(tasks)
    for method in methods:
        found = set(task_df.loc[task_df["method"] == method, "task"])
        missing = sorted(expected - found)
        if missing:
            print(f"Warning: {method} is missing aggregate tasks: {missing}")

    stats = _task_average_stats(task_df, tasks)
    table = stats.pivot(index="level", columns="method", values="mean")
    counts = stats.pivot(index="level", columns="method", values="count")
    print("=" * 96)
    print(f"{_task_label(tasks).title()} mean success rate (%) over tasks")
    print("=" * 96)
    print(table.round(2).to_string())
    print("-" * 96)
    print("Number of tasks contributing to each mean")
    print(counts.fillna(0).astype(int).to_string())
    print("=" * 96)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[],
                        help="Result file as PATH=LABEL. Can be repeated. "
                             "Defaults to the four requested method logs.")
    parser.add_argument("--task", nargs="+", type=int, default=[1],
                        help="Task id(s) to plot. One id writes one task figure; "
                             "multiple ids write one task-averaged figure. "
                             "Example: --task 1 2 3. Default: 1.")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Generate one figure for every task found after filtering.")
    parser.add_argument("--levels", nargs="+", type=int, default=None,
                        help="Levels to include. Default: all levels found.")
    parser.add_argument("--output", default=None,
                        help="Output path. With --all-tasks, _taskT is inserted. "
                             "With multiple --task values, this is the aggregate output.")
    parser.add_argument("--no-band", action="store_true",
                        help="Disable the +/-1 std shaded band.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any requested file has no matching rows.")
    args = parser.parse_args()

    specs = _input_specs(args.input)
    requested_tasks = _unique_ints(args.task)
    tasks = None if args.all_tasks else requested_tasks
    df = load_method_results(
        specs, tasks=tasks, levels=args.levels, strict=args.strict,
    )

    if args.all_tasks:
        tasks_to_plot = sorted(df["task"].unique())
        print_summary(df)
        for task in tasks_to_plot:
            if args.output is None:
                output = HERE / f"method_success_rate_comparison_task{int(task)}.png"
            else:
                output_path = Path(args.output)
                output = output_path.with_name(
                    f"{output_path.stem}_task{int(task)}{output_path.suffix}"
                )

            plot_task_comparison(
                df, int(task), str(output), show_band=not args.no_band,
            )
    elif len(requested_tasks) > 1:
        print_task_average_summary(df, requested_tasks)
        if args.output is None:
            output = HERE / (
                f"method_success_rate_comparison_average_"
                f"{_task_suffix(requested_tasks)}.png"
            )
        else:
            output = Path(args.output)

        plot_task_average_comparison(
            df, requested_tasks, str(output), show_band=not args.no_band,
        )
    else:
        tasks_to_plot = requested_tasks
        print_summary(df)
        task = tasks_to_plot[0]
        if args.output is None:
            output = HERE / f"method_success_rate_comparison_task{int(task)}.png"
        else:
            output = Path(args.output)

        plot_task_comparison(
            df, int(task), str(output), show_band=not args.no_band,
        )


if __name__ == "__main__":
    main()
