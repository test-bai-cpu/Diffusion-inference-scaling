"""
Plot average success rate per task vs difficulty level for DFS pointmaze results,
using seaborn for styling.

Each task has:
  - one baseline run on the original maze (mapped to difficulty 0.0)
  - several variants per level on the new-variation maze
Both kinds of lines are parsed from the same results file.
"""

import re
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_FILE = "results_newvar_level.txt"
OUTPUT_FILE = "success_rate_by_task_level.png"


def level_to_difficulty(level, n_new_levels):
    """Map raw level index to a normalized difficulty.
    Level 0 (original maze) -> 0.0.
    Levels 1..N_new -> 0.2, 0.4, ... (i.e. lv / (N_new + 1))."""
    if level == 0:
        return 0.0
    return level / (n_new_levels + 1)


# Matches the new-variation runs: dfs-taskT-levelL-variantV
VARIANT_RE = re.compile(
    r"dfs-task(?P<task>\d+)-level(?P<level>\d+)-variant(?P<variant>\d+)"
    r".*?Success Rate:\s*(?P<sr>[-+]?\d*\.?\d+)"
)

# Matches the original-maze baseline runs: original_maze_dfs_taskT
ORIGINAL_RE = re.compile(
    r"original_maze_dfs_task(?P<task>\d+)"
    r".*?Success Rate:\s*(?P<sr>[-+]?\d*\.?\d+)"
)

# Matches the map-guidance (BFS) runs: dfs-df-taskT-levelL-variantV
GUIDANCE_RE = re.compile(
    r"dfs-df-task(?P<task>\d+)-level(?P<level>\d+)-variant(?P<variant>\d+)"
    r".*?Success Rate:\s*(?P<sr>[-+]?\d*\.?\d+)"
)


def parse_guidance_results(path):
    """Parse the map-guidance results file into a long-form DataFrame."""
    rows = []
    with open(path, "r") as f:
        for line in f:
            m = GUIDANCE_RE.search(line)
            if m:
                rows.append({
                    "task": int(m.group("task")),
                    "level": int(m.group("level")),
                    "variant": int(m.group("variant")),
                    "success_rate": float(m.group("sr")),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    n_levels = df["level"].nunique()
    df["difficulty"] = df["level"].apply(lambda lv: lv / (n_levels + 1))
    df["task_label"] = "Task " + df["task"].astype(str)
    return df


def plot_method_comparison(df_dfs, df_guidance, output_path, task,
                           show_band=True):
    """Compare DFS vs map-guidance for a single task across difficulty levels.

    Uses only the non-original (level >= 1) rows from df_dfs to match the
    map-guidance file's coverage.
    """
    sns.set_theme(style="ticks", context="paper", font_scale=1.25,
                  rc={"font.family": "DejaVu Sans"})

    dfs_sub = df_dfs[(df_dfs["task"] == task) & (~df_dfs["is_original"])].copy()
    dfs_sub["method"] = "DFS"

    gd_sub = df_guidance[df_guidance["task"] == task].copy()
    gd_sub["method"] = "Map-Guidance"

    # Sanity check: keep only difficulties present in both, so the
    # comparison is apples-to-apples.
    common_diffs = sorted(set(dfs_sub["difficulty"]) & set(gd_sub["difficulty"]))
    if not common_diffs:
        raise SystemExit("No overlapping difficulty levels between DFS and guidance.")
    dfs_sub = dfs_sub[dfs_sub["difficulty"].isin(common_diffs)]
    gd_sub = gd_sub[gd_sub["difficulty"].isin(common_diffs)]

    combined = pd.concat(
        [dfs_sub[["difficulty", "success_rate", "method"]],
         gd_sub[["difficulty", "success_rate", "method"]]],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    deep = sns.color_palette("deep")
    method_order = ["DFS", "Map-Guidance"]
    palette = {"DFS": deep[0], "Map-Guidance": deep[1]}

    sns.lineplot(
        data=combined,
        x="difficulty",
        y="success_rate",
        hue="method",
        hue_order=method_order,
        palette=palette,
        errorbar=("sd", 1) if show_band else None,
        err_style="band",
        linewidth=2.4,
        marker="o",
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=0.9,
        alpha=0.95,
        ax=ax,
        legend="full",
    )

    x_pad = 0.025
    ax.set_xticks(common_diffs)
    ax.set_xticklabels([f"{d:.1f}" for d in common_diffs])
    ax.set_xlim(common_diffs[0] - x_pad, common_diffs[-1] + x_pad)

    # Tight y range.
    if show_band:
        grouped = combined.groupby(["method", "difficulty"])["success_rate"]
        means = grouped.mean()
        sds = grouped.std(ddof=0).fillna(0.0)
        y_lo = (means - sds).min()
        y_hi = (means + sds).max()
    else:
        means = combined.groupby(["method", "difficulty"])["success_rate"].mean()
        y_lo, y_hi = means.min(), means.max()
    y_pad = max(2.0, 0.06 * (y_hi - y_lo))
    ax.set_ylim(max(0.0, y_lo - y_pad), min(100.0, y_hi + y_pad))

    ax.set_xlabel("Difficulty Level", labelpad=6)
    ax.set_ylabel("Average Success Rate (%)", labelpad=6)
    ax.set_title(f"Task {task}: DFS vs Map-Guidance", fontsize=11, pad=8)

    ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#444444")
    ax.tick_params(axis="both", which="major", length=4, width=0.8,
                   color="#444444", labelcolor="#222222", pad=3)

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles=handles, labels=labels,
        loc="upper right",
        fontsize=9, frameon=True, framealpha=0.92,
        edgecolor="#cccccc", fancybox=False,
        handlelength=2.0, handletextpad=0.55,
        borderpad=0.45, labelspacing=0.32, borderaxespad=0.6,
    )
    leg.get_frame().set_linewidth(0.6)
    if leg.get_title() is not None:
        leg.get_title().set_visible(False)
    for h in leg.legend_handles:
        h.set_markersize(5)
        h.set_markeredgewidth(0.6)
        h.set_linewidth(1.6)

    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved figure to: {output_path}")


def parse_results(path):
    """Parse the results file into a long-form DataFrame.

    Adds an `is_original` flag so we can style baseline points differently
    if desired. Original baseline runs are given level 0 and variant 0.
    """
    rows = []
    with open(path, "r") as f:
        for line in f:
            m = VARIANT_RE.search(line)
            if m:
                rows.append({
                    "task": int(m.group("task")),
                    "level": int(m.group("level")),
                    "variant": int(m.group("variant")),
                    "success_rate": float(m.group("sr")),
                    "is_original": False,
                })
                continue
            m = ORIGINAL_RE.search(line)
            if m:
                rows.append({
                    "task": int(m.group("task")),
                    "level": 0,
                    "variant": 0,
                    "success_rate": float(m.group("sr")),
                    "is_original": True,
                })

    df = pd.DataFrame(rows)
    n_new_levels = df.loc[~df["is_original"], "level"].nunique()
    df["difficulty"] = df["level"].apply(lambda lv: level_to_difficulty(lv, n_new_levels))
    df["task_label"] = "Task " + df["task"].astype(str)
    return df


def plot_success_rate(df, output_path, show_band=True):
    # Clean publication style.
    sns.set_theme(style="ticks", context="paper", font_scale=1.25,
                  rc={"font.family": "DejaVu Sans"})

    tasks_sorted = sorted(df["task"].unique())
    hue_order = [f"Task {t}" for t in tasks_sorted]
    palette = sns.color_palette("deep", n_colors=len(hue_order))

    fig, ax = plt.subplots(figsize=(5.4, 3.8))

    sns.lineplot(
        data=df,
        x="difficulty",
        y="success_rate",
        hue="task_label",
        hue_order=hue_order,
        palette=palette,
        errorbar=("sd", 1) if show_band else None,
        err_style="band",
        linewidth=2.2,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=0.8,
        alpha=0.95,
        ax=ax,
        legend="full",
    )

    # X axis: include difficulty 0 (original maze) as the leftmost tick.
    diffs = sorted(df["difficulty"].unique())
    x_pad = 0.025
    ax.set_xlim(diffs[0] - x_pad, diffs[-1] + x_pad)
    ax.set_xticks(diffs)
    ax.set_xticklabels([f"{d:.1f}" for d in diffs])

    # Tight y range hugging the data.
    if show_band:
        grouped = df.groupby(["task", "difficulty"])["success_rate"]
        means = grouped.mean()
        sds = grouped.std(ddof=0).fillna(0.0)
        y_lo = (means - sds).min()
        y_hi = (means + sds).max()
    else:
        means = df.groupby(["task", "difficulty"])["success_rate"].mean()
        y_lo, y_hi = means.min(), means.max()
    y_pad = max(2.0, 0.04 * (y_hi - y_lo))
    ax.set_ylim(max(0.0, y_lo - y_pad), min(100.0, y_hi + y_pad))

    ax.set_xlabel("Difficulty Level", labelpad=6)
    ax.set_ylabel("Average Success Rate (%)", labelpad=6)

    # Subtle horizontal grid only.
    ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    # Light spines and ticks.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#444444")
    ax.tick_params(axis="both", which="major", length=4, width=0.8,
                   color="#444444", labelcolor="#222222", pad=3)

    # Legend: upper right, compact, line + small dot per entry.
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles=handles, labels=labels,
        loc="upper right",
        fontsize=9, frameon=True, framealpha=0.92,
        edgecolor="#cccccc", fancybox=False,
        handlelength=2.0, handletextpad=0.55,
        borderpad=0.45, labelspacing=0.32, borderaxespad=0.6,
    )
    leg.get_frame().set_linewidth(0.6)
    if leg.get_title() is not None:
        leg.get_title().set_visible(False)
    for h in leg.legend_handles:
        h.set_markersize(5)
        h.set_markeredgewidth(0.6)
        h.set_linewidth(1.6)

    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved figure to: {output_path}")


def plot_overall_average(df, output_path, show_band=True):
    """Plot the success rate averaged over all tasks at each difficulty level.

    For each (task, difficulty) we first take the mean across variants,
    then average those per-task means across the 5 tasks. The shaded band
    is +/-1 std across tasks (i.e. how much tasks disagree at this difficulty).
    """
    sns.set_theme(style="ticks", context="paper", font_scale=1.25,
                  rc={"font.family": "DejaVu Sans"})

    # Per-task mean at each difficulty (collapsing variants first).
    task_means = (df.groupby(["task", "difficulty"])["success_rate"]
                    .mean()
                    .reset_index())

    # Overall: mean and std across tasks at each difficulty.
    overall = (task_means.groupby("difficulty")["success_rate"]
                          .agg(mean="mean", std="std", count="count")
                          .reset_index())
    overall["std"] = overall["std"].fillna(0.0)

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    color = sns.color_palette("deep")[0]  # single bold color (blue)

    if show_band:
        ax.fill_between(
            overall["difficulty"],
            overall["mean"] - overall["std"],
            overall["mean"] + overall["std"],
            color=color, alpha=0.18, linewidth=0, zorder=1,
        )

    ax.plot(
        overall["difficulty"], overall["mean"],
        color=color, linewidth=2.4,
        marker="o", markersize=8,
        markeredgecolor="white", markeredgewidth=0.9,
        zorder=3, label="Mean over 5 tasks",
    )

    # X axis with all difficulties present.
    diffs = sorted(df["difficulty"].unique())
    x_pad = 0.025
    ax.set_xlim(diffs[0] - x_pad, diffs[-1] + x_pad)
    ax.set_xticks(diffs)
    ax.set_xticklabels([f"{d:.1f}" for d in diffs])

    # Tight y range.
    if show_band:
        y_lo = (overall["mean"] - overall["std"]).min()
        y_hi = (overall["mean"] + overall["std"]).max()
    else:
        y_lo = overall["mean"].min()
        y_hi = overall["mean"].max()
    y_pad = max(2.0, 0.06 * (y_hi - y_lo))
    ax.set_ylim(max(0.0, y_lo - y_pad), min(100.0, y_hi + y_pad))

    ax.set_xlabel("Difficulty Level", labelpad=6)
    ax.set_ylabel("Average Success Rate (%)", labelpad=6)

    ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#444444")
    ax.tick_params(axis="both", which="major", length=4, width=0.8,
                   color="#444444", labelcolor="#222222", pad=3)

    leg = ax.legend(
        loc="upper right",
        fontsize=9, frameon=True, framealpha=0.92,
        edgecolor="#cccccc", fancybox=False,
        handlelength=2.0, handletextpad=0.55,
        borderpad=0.45, labelspacing=0.32, borderaxespad=0.6,
    )
    leg.get_frame().set_linewidth(0.6)
    for h in leg.legend_handles:
        h.set_markersize(5)
        h.set_markeredgewidth(0.6)
        h.set_linewidth(1.6)

    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved figure to: {output_path}")


def print_summary(df):
    diffs = sorted(df["difficulty"].unique())
    tasks = sorted(df["task"].unique())
    print("=" * 88)
    print("Average Success Rate (%) per Task per Difficulty (mean +/- std over variants)")
    print("Difficulty 0.0 = original maze (single baseline run, std = 0).")
    print("=" * 88)
    header = f"{'Task':<8}" + "".join([f"Diff {d:.1f}{'':<8}" for d in diffs]) + "Task Avg"
    print(header)
    print("-" * 88)
    for t in tasks:
        row = f"Task {t:<4}"
        per_diff_means = []
        for d in diffs:
            sub = df[(df["task"] == t) & (df["difficulty"] == d)]["success_rate"]
            if len(sub):
                m = sub.mean()
                s = sub.std(ddof=0) if len(sub) > 1 else 0.0
                per_diff_means.append(m)
                row += f"{m:>6.2f} +/-{s:<5.2f}"
            else:
                row += f"{'N/A':>14}"
        if per_diff_means:
            row += f"  {np.mean(per_diff_means):>6.2f}"
        print(row)
    print("-" * 88)
    row = f"{'Diff Avg':<8}"
    for d in diffs:
        m = df[df["difficulty"] == d].groupby("task")["success_rate"].mean().mean()
        row += f"{m:>6.2f}         "
    print(row)
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", default=INPUT_FILE)
    parser.add_argument("--output", "-o", default=OUTPUT_FILE,
                        help="Output path for the per-task figure.")
    parser.add_argument("--overall-output", default=None,
                        help="Output path for the overall-average figure. "
                             "Defaults to <output>_overall.<ext>.")
    parser.add_argument("--guidance-input", default=None,
                        help="Path to the map-guidance results file. "
                             "If provided, a comparison figure is generated.")
    parser.add_argument("--comparison-output", default=None,
                        help="Output path for the DFS vs Map-Guidance figure. "
                             "Defaults to <output>_compare_taskT.<ext>.")
    parser.add_argument("--comparison-task", type=int, default=1,
                        help="Which task to compare DFS vs Map-Guidance for "
                             "(default: 1).")
    parser.add_argument("--no-band", action="store_true",
                        help="Disable the +/-1 std shaded band.")
    args = parser.parse_args()

    df = parse_results(args.input)
    if df.empty:
        raise SystemExit(f"No matching lines found in {args.input}.")

    # Default overall output: insert "_overall" before the extension.
    if args.overall_output is None:
        if "." in args.output:
            stem, ext = args.output.rsplit(".", 1)
            args.overall_output = f"{stem}_overall.{ext}"
        else:
            args.overall_output = args.output + "_overall"

    print_summary(df)
    plot_success_rate(df, args.output, show_band=not args.no_band)
    plot_overall_average(df, args.overall_output, show_band=not args.no_band)

    if args.guidance_input is not None:
        df_guidance = parse_guidance_results(args.guidance_input)
        if df_guidance.empty:
            print(f"Warning: no map-guidance entries found in {args.guidance_input}.")
        else:
            if args.comparison_output is None:
                if "." in args.output:
                    stem, ext = args.output.rsplit(".", 1)
                    args.comparison_output = (
                        f"{stem}_compare_task{args.comparison_task}.{ext}")
                else:
                    args.comparison_output = (
                        f"{args.output}_compare_task{args.comparison_task}")
            plot_method_comparison(
                df, df_guidance, args.comparison_output,
                task=args.comparison_task,
                show_band=not args.no_band,
            )


if __name__ == "__main__":
    main()

# # Per-task + overall (as before), plus DFS vs Map-Guidance for task 1
# python plot_success_rate.py \
#     -i results_newvar_level.txt \
#     --guidance-input results_newvar_bfs_guidance_level.txt \
#     --comparison-task 1

# Custom paths
# python plot_success_rate.py -i results_dfs_pointmaze-giant-newvar-navigate-v0-dfs.txt -o pure_dfs_results.png --no-band

# Disable the ±1 std shaded band
# python plot_success_rate.py --no-band