"""
Plot run_origin_v2 ablation results for completed task/level slices.

By default this compares pure DFS from results_newvar_level.txt, the
MapCond + corner + distance runs trained on maze v1/v2, and the four ablation
txt logs launched by run_origin_v2.sh for Task 1, Level 1, Variants 1..3. The
styling intentionally reuses the publication-style helpers in plot_success_rate.py.

Pass multiple tasks as "--task 1 2 3" to write task-averaged ablation plots,
one for each requested level.
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_success_rate import (
    parse_results,
    _group_stats,
    _style_axes,
)


HERE = Path(__file__).resolve().parent

DEFAULT_ABLATIONS = (
    # ("results_newvar_level.txt", "Baseline"),
    ("results_dfs_pointmaze-giant-newvar-navigate-v0_dfs_trainall.txt", "Baseline"),
    (
        "results_dfs_pointmaze-giant-newvar-navigate-v0_"
        "mazev1-cond-trainonv2.txt",
        # "MapCond + corner + distance",
        "Ours",
    ),
    # (
    #     "results_dfs_pointmaze-giant-newvar-navigate-v0_"
    #     "fullrun-mazev1-mapcond-corner-nodist.txt",
    #     "MapCond + corner",
    # ),
    # (
    #     "results_dfs_pointmaze-giant-newvar-navigate-v0_"
    #     "fullrun-mazev1-mapcond-dist-nocorner.txt",
    #     "MapCond + distance",
    # ),
    (
        "results_dfs_pointmaze-giant-newvar-navigate-v0_"
        "fullrun-mazev1-mapcond-nocornerdist.txt",
        "Remove distance field and corner check",
    ),
    (
        "results_dfs_pointmaze-giant-newvar-navigate-v0_"
        "fullrun-mazev1-distcorner-nomapcond.txt",
        "Remove map condition",
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
    """Parse repeated PATH=LABEL specs, or return the default comparisons."""
    if not specs:
        return [(_resolve_path(path), label) for path, label in DEFAULT_ABLATIONS]

    parsed = []
    for spec in specs:
        if "=" in spec:
            path, label = spec.split("=", 1)
        else:
            path = spec
            label = Path(path).stem
            label = label.replace("results_detailed_", "")
            label = label.replace("results_dfs_pointmaze-giant-newvar-navigate-v0_", "")
        parsed.append((_resolve_path(path), label))
    return parsed


def _unique_ints(values):
    if values is None:
        return []
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


def _single_task_label(tasks):
    tasks = _unique_ints(tasks)
    if len(tasks) != 1:
        raise ValueError("Expected exactly one task.")
    return tasks[0]


def load_ablation_results(specs, tasks=None, levels=None, variants=None, strict=False):
    """Load and filter all ablation result files into one long-form DataFrame."""
    tasks = None if tasks is None else _unique_ints(tasks)
    levels = None if levels is None else _unique_ints(levels)
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

        if tasks is not None:
            df = df[df["task"].isin(tasks)].copy()
        if levels is not None:
            df = df[df["level"].isin(levels)].copy()
        if variants is not None:
            df = df[df["variant"].isin(variants)].copy()

        if df.empty:
            msg = f"No matching rows after filtering in {path}"
            if strict:
                raise SystemExit(msg)
            print(f"Warning: {msg}")
            continue

        df["ablation"] = label
        df["source"] = path.name
        frames.append(df)

    if not frames:
        raise SystemExit("No ablation rows loaded.")

    combined = pd.concat(frames, ignore_index=True)
    combined["ablation"] = pd.Categorical(combined["ablation"],
                                          categories=labels,
                                          ordered=True)

    if variants is not None:
        expected = set(variants)
        for label in labels:
            found = set(combined.loc[combined["ablation"] == label, "variant"])
            missing = sorted(expected - found)
            if missing:
                msg = f"{label} is missing variants: {missing}"
                if strict:
                    raise SystemExit(msg)
                print(f"Warning: {msg}")

    return combined


def _task_variant_means(df, tasks):
    """Average repeated rows for each task/level/variant."""
    tasks = _unique_ints(tasks)
    task_df = df[df["task"].isin(tasks)].copy()
    if task_df.empty:
        raise SystemExit(f"No rows found for {_task_label(tasks)}.")

    return (
        task_df.groupby(["ablation", "task", "level", "variant"], observed=True)
               ["success_rate"]
               .mean()
               .reset_index()
    )


def _task_average_stats(df, tasks):
    """Average variants first, then average selected tasks per level."""
    variant_means = _task_variant_means(df, tasks)
    task_means = (
        variant_means.groupby(["ablation", "task", "level"], observed=True)
                     ["success_rate"]
                     .mean()
                     .reset_index()
    )
    stats = (
        task_means.groupby(["ablation", "level"], observed=True)["success_rate"]
                  .agg(mean="mean",
                       std=lambda x: x.std(ddof=0),
                       count="count")
                  .reset_index()
                  .fillna({"std": 0.0})
    )
    return stats, task_means


def _set_common_ylim(ax, values, errors=None, extra_values=None):
    values = np.asarray(values, dtype=float)
    if errors is None:
        lower = values
        upper = values
    else:
        errors = np.asarray(errors, dtype=float)
        lower = values - errors
        upper = values + errors

    if extra_values is not None:
        extra_values = np.asarray(extra_values, dtype=float)
        extra_values = extra_values[np.isfinite(extra_values)]
        if len(extra_values):
            lower = np.concatenate([lower, extra_values])
            upper = np.concatenate([upper, extra_values])

    y_lo = np.nanmin(lower)
    y_hi = np.nanmax(upper)
    y_pad = max(2.0, 0.06 * (y_hi - y_lo))
    bottom = y_lo - y_pad
    top = y_hi + y_pad
    if y_lo > 0.0:
        bottom = max(0.0, bottom)
    if y_hi < 100.0:
        top = min(100.0, top)
    ax.set_ylim(bottom, top)


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


def plot_by_variant(df, output_path, task, level, show_band=True):
    """Plot one line per ablation across maze variants."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.25,
                  rc={"font.family": "DejaVu Sans"})

    labels = [label for label in df["ablation"].cat.categories
              if label in set(df["ablation"].dropna())]
    palette = dict(zip(labels, sns.color_palette("deep", n_colors=len(labels))))
    stats = _group_stats(df, ["ablation", "variant"])
    offsets = dict(zip(labels, np.linspace(-0.055, 0.055, len(labels))))
    markers = dict(zip(labels, ["o", "s", "D", "^", "v", "P", "X", "h"][:len(labels)]))

    fig, ax = plt.subplots(figsize=(6.1, 3.8))
    for label in labels:
        sub = stats[stats["ablation"] == label].sort_values("variant")
        x = sub["variant"].to_numpy(dtype=float) + offsets[label]
        mean = sub["mean"].to_numpy(dtype=float)
        std = sub["std"].to_numpy(dtype=float)
        if show_band:
            ax.fill_between(
                x, mean - std, mean + std,
                color=palette[label], alpha=0.12, linewidth=0, zorder=1,
            )
        ax.plot(
            x, mean,
            color=palette[label], linewidth=2.2,
            marker=markers[label], markersize=7,
            markeredgecolor="white", markeredgewidth=0.9,
            zorder=3, label=label,
        )

    variants = sorted(df["variant"].unique())
    ax.set_xticks(variants)
    ax.set_xticklabels([str(int(v)) for v in variants])
    if len(variants) == 1:
        ax.set_xlim(variants[0] - 0.3, variants[0] + 0.3)
    else:
        ax.set_xlim(variants[0] - 0.2, variants[-1] + 0.2)

    _set_common_ylim(ax, stats["mean"], stats["std"] if show_band else None)
    ax.set_xlabel("Maze Variant", labelpad=6)
    ax.set_ylabel("Success Rate (%)", labelpad=6)
    ax.set_title(f"Task {task}, Level {level}: ablation comparison",
                 fontsize=11, pad=8)

    _style_axes(ax)
    _style_outside_legend(ax)
    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def plot_summary(df, output_path, task, level, show_error=True):
    """Plot mean +/- std across maze variants for each ablation."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.25,
                  rc={"font.family": "DejaVu Sans"})

    variant_means = (df.groupby(["ablation", "variant"], observed=True)["success_rate"]
                       .mean()
                       .reset_index())
    summary = (variant_means.groupby("ablation", observed=True)["success_rate"]
                            .agg(mean="mean",
                                 std=lambda x: x.std(ddof=0),
                                 count="count")
                            .reset_index())
    labels = summary["ablation"].astype(str).tolist()
    colors = sns.color_palette("deep", n_colors=len(labels))
    x = np.arange(len(labels))

    fig_width = max(6.4, 1.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 3.8))
    ax.bar(
        x, summary["mean"],
        width=0.62, color=colors, edgecolor="#333333", linewidth=0.6,
        zorder=2,
    )

    if show_error:
        ax.errorbar(
            x, summary["mean"], yerr=summary["std"],
            fmt="none", ecolor="#333333", elinewidth=1.0,
            capsize=3.5, capthick=1.0, zorder=4,
        )

    for i, label in enumerate(labels):
        points = variant_means.loc[
            variant_means["ablation"].astype(str) == label, "success_rate"
        ].to_numpy(dtype=float)
        if len(points) == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(-0.13, 0.13, len(points))
        ax.scatter(
            np.full(len(points), x[i]) + jitter, points,
            s=28, facecolor="white", edgecolor="#222222",
            linewidth=0.6, zorder=5,
        )

    _set_common_ylim(
        ax, summary["mean"],
        summary["std"] if show_error else None,
        extra_values=variant_means["success_rate"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["\n".join(textwrap.wrap(label, width=13)) for label in labels],
        fontsize=9,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Average Success Rate (%)", labelpad=6)
    ax.set_title(f"Task {task}, Level {level}: mean over maze variants",
                 fontsize=11, pad=8)

    _style_axes(ax)
    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def plot_task_average_summary(df, output_path, tasks, level, show_error=True):
    """Plot mean +/- std across selected tasks for one level."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.25,
                  rc={"font.family": "DejaVu Sans"})

    level_df = df[df["level"] == level].copy()
    if level_df.empty:
        raise SystemExit(f"No rows found for {_task_label(tasks)}, level {level}.")

    stats, _ = _task_average_stats(level_df, tasks)
    variant_means = _task_variant_means(level_df, tasks)
    summary = stats[stats["level"] == level].copy()
    labels = [label for label in level_df["ablation"].cat.categories
              if label in set(summary["ablation"].dropna())]
    summary = summary[summary["ablation"].isin(labels)].sort_values("ablation")
    labels = summary["ablation"].astype(str).tolist()
    colors = sns.color_palette("deep", n_colors=len(labels))
    x = np.arange(len(labels))

    fig_width = max(6.4, 1.55 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 3.8))
    ax.bar(
        x, summary["mean"],
        width=0.62, color=colors, edgecolor="#333333", linewidth=0.6,
        zorder=2,
    )

    if show_error:
        ax.errorbar(
            x, summary["mean"], yerr=summary["std"],
            fmt="none", ecolor="#333333", elinewidth=1.0,
            capsize=3.5, capthick=1.0, zorder=4,
        )

    for i, label in enumerate(labels):
        points = (
            variant_means.loc[
                (variant_means["level"] == level)
                & (variant_means["ablation"].astype(str) == label)
            ]
            .sort_values(["task", "variant"])["success_rate"]
            .to_numpy(dtype=float)
        )
        if len(points) == 1:
            jitter = np.array([0.0])
        else:
            jitter = np.linspace(-0.2, 0.2, len(points))
        ax.scatter(
            np.full(len(points), x[i]) + jitter, points,
            s=22, facecolor="white", edgecolor="#222222",
            linewidth=0.55, alpha=0.9, zorder=5,
        )

    _set_common_ylim(
        ax, summary["mean"],
        summary["std"] if show_error else None,
        extra_values=variant_means["success_rate"],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["\n".join(textwrap.wrap(label, width=13)) for label in labels],
        fontsize=9,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Task-Averaged Success Rate (%)", labelpad=6)
    ax.set_title(f"{_task_label(tasks).title()}, Level {level}: mean over tasks",
                 fontsize=11, pad=8)

    _style_axes(ax)
    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def print_summary(df):
    variant_table = (df.groupby(["variant", "ablation"], observed=True)["success_rate"]
                       .mean()
                       .unstack("ablation"))
    summary = (variant_table.agg(["mean", lambda x: x.std(ddof=0)])
                            .rename(index={"<lambda>": "std"}))

    print("=" * 88)
    print("Success Rate (%) by maze variant")
    print("=" * 88)
    print(variant_table.round(2).to_string())
    print("-" * 88)
    print(summary.round(2).to_string())
    print("=" * 88)


def print_task_average_summary(df, tasks):
    tasks = _unique_ints(tasks)
    task_df = df[df["task"].isin(tasks)].copy()
    if task_df.empty:
        print(f"Warning: No rows found for {_task_label(tasks)}.")
        return

    labels = [label for label in df["ablation"].cat.categories
              if label in set(task_df["ablation"].dropna())]
    expected = set(tasks)
    for level in sorted(task_df["level"].unique()):
        for label in labels:
            found = set(task_df.loc[
                (task_df["level"] == level) & (task_df["ablation"] == label),
                "task",
            ])
            missing = sorted(expected - found)
            if missing:
                print(
                    f"Warning: {label} is missing aggregate tasks {missing} "
                    f"for level {int(level)}"
                )

    stats, _ = _task_average_stats(task_df, tasks)
    table = stats.pivot(index="level", columns="ablation", values="mean")
    counts = stats.pivot(index="level", columns="ablation", values="count")
    print("=" * 96)
    print(f"{_task_label(tasks).title()} mean success rate (%) over tasks")
    print("=" * 96)
    print(table.round(2).to_string())
    print("-" * 96)
    print("Number of tasks contributing to each mean")
    print(counts.fillna(0).astype(int).to_string())
    print("=" * 96)


def _summary_output_path(path_arg, tasks, level, multi_levels):
    tasks = _unique_ints(tasks)
    if path_arg is None:
        if len(tasks) > 1:
            return HERE / (
                f"all_train_ablation_success_rate_average_"
                f"{_task_suffix(tasks)}_level{level}_summary.png"
            )
        task = _single_task_label(tasks)
        return HERE / f"all_train_ablation_success_rate_task{task}_level{level}_summary.png"

    output_path = Path(path_arg)
    if multi_levels:
        return output_path.with_name(
            f"{output_path.stem}_level{level}{output_path.suffix}"
        )
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", default=[],
                        help="Result file as PATH=LABEL. Can be repeated. "
                             "Defaults to pure DFS, MapCond + corner + "
                             "distance trained on maze v1/v2, and four "
                             "run_origin_v2 ablation txt logs.")
    parser.add_argument("--task", nargs="+", type=int, default=[1],
                        help="Task id(s) to plot. One id writes one task/level "
                             "summary; multiple ids write task-averaged summaries. "
                             "Example: --task 1 2 3. Default: 1.")
    parser.add_argument("--level", "--levels", nargs="+", type=int, default=[1],
                        dest="levels",
                        help="Difficulty level(s) to plot. Default: 1.")
    parser.add_argument("--variants", nargs="+", type=int, default=[1, 2, 3],
                        help="Maze variant ids to include.")
    parser.add_argument("--output",
                        default=None,
                        help="Output path for the per-variant comparison figure.")
    parser.add_argument("--summary-output",
                        default=None,
                        help="Output path for the mean/std summary figure.")
    parser.add_argument("--no-band", action="store_true",
                        help="Disable shaded bands / summary error bars.")
    parser.add_argument("--strict", action="store_true",
                        help="Fail if any requested file or variant is missing.")
    args = parser.parse_args()

    specs = _input_specs(args.input)
    requested_tasks = _unique_ints(args.task)
    requested_levels = _unique_ints(args.levels)

    df = load_ablation_results(
        specs, tasks=requested_tasks, levels=requested_levels,
        variants=args.variants, strict=args.strict,
    )

    if len(requested_tasks) > 1:
        print_task_average_summary(df, requested_tasks)
        for level in requested_levels:
            output = _summary_output_path(
                args.summary_output, requested_tasks, level,
                multi_levels=len(requested_levels) > 1,
            )
            plot_task_average_summary(
                df, str(output), tasks=requested_tasks, level=level,
                show_error=not args.no_band,
            )
    else:
        task = _single_task_label(requested_tasks)
        for level in requested_levels:
            level_df = df[df["level"] == level].copy()
            if level_df.empty:
                msg = f"No rows found for task {task}, level {level}."
                if args.strict:
                    raise SystemExit(msg)
                print(f"Warning: {msg}")
                continue

            print_summary(level_df)
            # plot_by_variant(
            #     level_df, args.output, task=task, level=level,
            #     show_band=not args.no_band,
            # )
            output = _summary_output_path(
                args.summary_output, requested_tasks, level,
                multi_levels=len(requested_levels) > 1,
            )
            plot_summary(
                level_df, str(output), task=task, level=level,
                show_error=not args.no_band,
            )


if __name__ == "__main__":
    main()
