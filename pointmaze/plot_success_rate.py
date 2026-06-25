"""
Plot average success rate per task vs difficulty level for DFS pointmaze results,
using seaborn for styling.
"""

import re
import argparse

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_FILE = "results_dfs_pointmaze-giant-newvar-navigate-v0.txt"
OUTPUT_FILE = "success_rate_by_task_level.png"


def level_to_difficulty(level, n_levels):
    """Map raw level index (1..N) to a normalized difficulty.
    For 4 levels this yields 0.2, 0.4, 0.6, 0.8."""
    return level / (n_levels + 1)


LINE_RE = re.compile(
    r"dfs-task(?P<task>\d+)-level(?P<level>\d+)-variant(?P<variant>\d+)"
    r".*?Success Rate:\s*(?P<sr>[-+]?\d*\.?\d+)"
)


def parse_results(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            rows.append({
                "task": int(m.group("task")),
                "level": int(m.group("level")),
                "variant": int(m.group("variant")),
                "success_rate": float(m.group("sr")),
            })
    df = pd.DataFrame(rows)
    n_levels = df["level"].nunique()
    df["difficulty"] = df["level"].apply(lambda lv: level_to_difficulty(lv, n_levels))
    df["task_label"] = "Task " + df["task"].astype(str)
    return df


def plot_success_rate(df, output_path, show_band=True):
    # Clean publication style. White background, no grid, light spines.
    sns.set_theme(style="ticks", context="paper", font_scale=1.25,
                  rc={"font.family": "DejaVu Sans"})

    tasks_sorted = sorted(df["task"].unique())
    hue_order = [f"Task {t}" for t in tasks_sorted]
    palette = sns.color_palette("deep", n_colors=len(hue_order))

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    sns.lineplot(
        data=df,
        x="difficulty",
        y="success_rate",
        hue="task_label",
        hue_order=hue_order,
        palette=palette,
        errorbar=("sd", 1) if show_band else None,
        err_style="band",
        linewidth=4.0,
        marker="o",
        markersize=10,
        markeredgecolor="white",
        markeredgewidth=0.8,
        alpha=0.95,
        ax=ax,
        legend="full",
    )

    # Tight axis ranges.
    diffs = sorted(df["difficulty"].unique())
    x_pad = 0.025
    ax.set_xlim(diffs[0] - x_pad, diffs[-1] + x_pad)
    ax.set_xticks(diffs)
    ax.set_xticklabels([f"{d:.1f}" for d in diffs])

    if show_band:
        grouped = df.groupby(["task", "difficulty"])["success_rate"]
        means = grouped.mean()
        sds = grouped.std(ddof=0)
        y_lo = (means - sds).min()
        y_hi = (means + sds).max()
    else:
        means = df.groupby(["task", "difficulty"])["success_rate"].mean()
        y_lo, y_hi = means.min(), means.max()
    y_pad = max(2.0, 0.04 * (y_hi - y_lo))
    ax.set_ylim(max(0.0, y_lo - y_pad), min(100.0, y_hi + y_pad))

    ax.set_xlabel("Difficulty Level", labelpad=6)
    ax.set_ylabel("Average Success Rate (%)", labelpad=6)

    # Very subtle horizontal grid only, behind the data.
    ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    # Thin, light spines. Keep all four for a framed look, but understated.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("#444444")

    ax.tick_params(axis="both", which="major", length=4, width=0.8,
                   color="#444444", labelcolor="#222222", pad=3)

    # Legend: upper right, compact, with a line + small dot per entry.
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        fontsize=9,
        frameon=True,
        framealpha=0.92,
        edgecolor="#cccccc",
        fancybox=False,
        handlelength=2.0,
        handletextpad=0.55,
        borderpad=0.45,
        labelspacing=0.32,
        borderaxespad=0.6,
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


def print_summary(df):
    diffs = sorted(df["difficulty"].unique())
    tasks = sorted(df["task"].unique())
    print("=" * 80)
    print("Average Success Rate (%) per Task per Difficulty (mean +/- std over variants)")
    print("=" * 80)
    header = f"{'Task':<8}" + "".join([f"Diff {d:.1f}{'':<8}" for d in diffs]) + "Task Avg"
    print(header)
    print("-" * 80)
    for t in tasks:
        row = f"Task {t:<4}"
        per_diff_means = []
        for d in diffs:
            sub = df[(df["task"] == t) & (df["difficulty"] == d)]["success_rate"]
            if len(sub):
                m, s = sub.mean(), sub.std(ddof=0)
                per_diff_means.append(m)
                row += f"{m:>6.2f} +/-{s:<5.2f}"
            else:
                row += f"{'N/A':>14}"
        if per_diff_means:
            row += f"  {np.mean(per_diff_means):>6.2f}"
        print(row)
    print("-" * 80)
    row = f"{'Diff Avg':<8}"
    for d in diffs:
        m = df[df["difficulty"] == d].groupby("task")["success_rate"].mean().mean()
        row += f"{m:>6.2f}         "
    print(row)
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", default=INPUT_FILE)
    parser.add_argument("--output", "-o", default=OUTPUT_FILE)
    parser.add_argument("--no-band", action="store_true",
                        help="Disable the +/-1 std shaded band.")
    args = parser.parse_args()

    df = parse_results(args.input)
    if df.empty:
        raise SystemExit(f"No matching lines found in {args.input}.")

    print_summary(df)
    plot_success_rate(df, args.output, show_band=not args.no_band)


if __name__ == "__main__":
    main()

# Custom paths
# python plot_success_rate.py -i results_dfs_pointmaze-giant-newvar-navigate-v0-dfs.txt -o pure_dfs_results.png --no-band

# Disable the ±1 std shaded band
# python plot_success_rate.py --no-band