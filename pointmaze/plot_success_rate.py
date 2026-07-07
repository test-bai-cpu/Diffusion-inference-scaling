"""
Plot average success rate per task vs difficulty level for pointmaze results,
using seaborn for styling. With two input files, also plot method comparisons.

Each task has:
  - one baseline run on the original maze (mapped to difficulty 0.0)
  - several variants per level on the new-variation maze
Both txt result logs and simple CSV files are supported.
"""

import re
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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


# Matches any new-variation run containing: taskT-levelL-variantV
VARIANT_RE = re.compile(
    r"task(?P<task>\d+)-level(?P<level>\d+)-variant(?P<variant>\d+)"
    r".*?Success Rate:\s*(?P<sr>[-+]?\d*\.?\d+)"
)

# Matches the original-maze baseline runs: original_maze_dfs_taskT
ORIGINAL_RE = re.compile(
    r"original_maze_dfs_task(?P<task>\d+)"
    r".*?Success Rate:\s*(?P<sr>[-+]?\d*\.?\d+)"
)

RUN_PART_RE = re.compile(
    r"task(?P<task>\d+)-level(?P<level>\d+)-variant(?P<variant>\d+)"
)


def _with_difficulty(df):
    """Add normalized difficulty and task label columns."""
    if df.empty:
        return df
    df = df.copy()
    if "is_original" not in df:
        df["is_original"] = df["level"].eq(0)
    n_new_levels = df.loc[~df["is_original"], "level"].nunique()
    df["difficulty"] = df["level"].apply(lambda lv: level_to_difficulty(lv, n_new_levels))
    df["task_label"] = "Task " + df["task"].astype(int).astype(str)
    return df


def _parse_run_name(run_name):
    """Extract task/level/variant from a run label when present."""
    m = RUN_PART_RE.search(str(run_name))
    if not m:
        return None
    return {
        "task": int(m.group("task")),
        "level": int(m.group("level")),
        "variant": int(m.group("variant")),
        "is_original": False,
    }


def parse_csv_results(path):
    """
    Parse a detailed CSV if it already has task/level/variant columns.

    Accepted success columns, in order: success_rate, total_reward, success.
    If a success column is 0..1, it is converted to percent.
    """
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    success_col = None
    for col in ("success_rate", "total_reward", "success"):
        if col in raw.columns:
            success_col = col
            break
    if success_col is None:
        return pd.DataFrame()

    rows = []
    for _, r in raw.iterrows():
        task = r.get("task")
        if pd.isna(task) or str(task).lower() == "average":
            continue

        parsed = None
        if "level" in raw.columns and "variant" in raw.columns:
            if pd.notna(r.get("level")) and pd.notna(r.get("variant")):
                parsed = {
                    "task": int(task),
                    "level": int(r["level"]),
                    "variant": int(r["variant"]),
                    "is_original": int(r["level"]) == 0,
                }
        if parsed is None and "run" in raw.columns:
            parsed = _parse_run_name(r["run"])
            if parsed is not None:
                parsed["task"] = int(task)

        if parsed is None:
            continue

        sr = float(r[success_col])
        if 0.0 <= sr <= 1.0:
            sr *= 100.0
        parsed["success_rate"] = sr
        rows.append(parsed)

    return _with_difficulty(pd.DataFrame(rows))


def _safe_label(label):
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()


def _insert_suffix(path, suffix):
    if "." in path:
        stem, ext = path.rsplit(".", 1)
        return f"{stem}{suffix}.{ext}"
    return path + suffix


def _style_axes(ax):
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


def _style_legend(ax):
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
    legend_handles = getattr(leg, "legend_handles", getattr(leg, "legendHandles", []))
    for h in legend_handles:
        h.set_markersize(5)
        h.set_markeredgewidth(0.6)
        h.set_linewidth(1.6)


def _common_difficulties(*dfs):
    sets = [set(df["difficulty"]) for df in dfs if not df.empty]
    if not sets:
        return []
    return sorted(set.intersection(*sets))


def _group_stats(df, group_cols):
    return (df.groupby(group_cols)["success_rate"]
              .agg(mean="mean", std=lambda x: x.std(ddof=0))
              .reset_index()
              .fillna({"std": 0.0}))


def _plot_mean_std(ax, stats, x_col, label, color, show_band=True,
                   linewidth=2.4, markersize=8):
    x = stats[x_col].to_numpy(dtype=float)
    mean = stats["mean"].to_numpy(dtype=float)
    std = stats["std"].to_numpy(dtype=float)
    order = np.argsort(x)
    x, mean, std = x[order], mean[order], std[order]
    if show_band:
        ax.fill_between(
            x, mean - std, mean + std,
            color=color, alpha=0.18, linewidth=0, zorder=1,
        )
    ax.plot(
        x, mean,
        color=color, linewidth=linewidth,
        marker="o", markersize=markersize,
        markeredgecolor="white", markeredgewidth=0.9,
        zorder=3, label=label,
    )


def plot_method_comparison(df_a, df_b, output_path, task,
                           method_a="DFS", method_b="DFS+Guidance",
                           show_band=True):
    """Compare two methods for a single task across difficulty levels."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.25,
                  rc={"font.family": "DejaVu Sans"})

    a_sub = df_a[df_a["task"] == task].copy()
    a_sub["method"] = method_a

    b_sub = df_b[df_b["task"] == task].copy()
    b_sub["method"] = method_b

    # Sanity check: keep only difficulties present in both, so the
    # comparison is apples-to-apples.
    common_diffs = _common_difficulties(a_sub, b_sub)
    if not common_diffs:
        raise SystemExit(f"No overlapping difficulty levels for task {task}.")
    a_sub = a_sub[a_sub["difficulty"].isin(common_diffs)]
    b_sub = b_sub[b_sub["difficulty"].isin(common_diffs)]

    combined = pd.concat(
        [a_sub[["difficulty", "success_rate", "method"]],
         b_sub[["difficulty", "success_rate", "method"]]],
        ignore_index=True,
    )

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    deep = sns.color_palette("deep")
    method_order = [method_a, method_b]
    palette = {method_a: deep[0], method_b: deep[1]}

    for method in method_order:
        stats = _group_stats(combined[combined["method"] == method], ["difficulty"])
        _plot_mean_std(
            ax, stats, "difficulty", method, palette[method],
            show_band=show_band, linewidth=2.4, markersize=8,
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
    ax.set_title(f"Task {task}: {method_a} vs {method_b}", fontsize=11, pad=8)

    _style_axes(ax)
    _style_legend(ax)

    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def parse_results(path):
    """Parse a txt or CSV results file into a long-form DataFrame.

    Adds an `is_original` flag so we can style baseline points differently
    if desired. Original baseline runs are given level 0 and variant 0.
    """
    if path.lower().endswith(".csv"):
        df_csv = parse_csv_results(path)
        if not df_csv.empty:
            return df_csv

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

    return _with_difficulty(pd.DataFrame(rows))


def plot_success_rate(df, output_path, show_band=True, method_label=None):
    # Clean publication style.
    sns.set_theme(style="ticks", context="paper", font_scale=1.25,
                  rc={"font.family": "DejaVu Sans"})

    tasks_sorted = sorted(df["task"].unique())
    hue_order = [f"Task {t}" for t in tasks_sorted]
    palette = sns.color_palette("deep", n_colors=len(hue_order))

    fig, ax = plt.subplots(figsize=(5.4, 3.8))

    for task_label, color in zip(hue_order, palette):
        sub = df[df["task_label"] == task_label]
        stats = _group_stats(sub, ["difficulty"])
        _plot_mean_std(
            ax, stats, "difficulty", task_label, color,
            show_band=show_band, linewidth=2.2, markersize=7,
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
    if method_label:
        ax.set_title(f"{method_label}: success by task", fontsize=11, pad=8)

    _style_axes(ax)
    _style_legend(ax)

    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def plot_overall_average(df, output_path, show_band=True, method_label=None):
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
    plot_stats = overall.rename(columns={"difficulty": "x"})
    _plot_mean_std(
        ax, plot_stats, "x",
        f"{method_label}: mean over tasks" if method_label else "Mean over tasks",
        color, show_band=show_band, linewidth=2.4, markersize=8,
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
    if method_label:
        ax.set_title(f"{method_label}: mean over tasks", fontsize=11, pad=8)

    _style_axes(ax)
    _style_legend(ax)

    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def _overall_by_method(df, method_label):
    """
    Return per-difficulty mean/std over tasks for one method.

    For each task/difficulty, variants are averaged first. Then the plotted
    mean/std are computed across tasks.
    """
    task_means = (df.groupby(["task", "difficulty"])["success_rate"]
                    .mean()
                    .reset_index())
    overall = (task_means.groupby("difficulty")["success_rate"]
                          .agg(mean="mean", std="std", count="count")
                          .reset_index())
    overall["std"] = overall["std"].fillna(0.0)
    overall["method"] = method_label
    return overall


def plot_overall_method_comparison(df_a, df_b, output_path,
                                   method_a="DFS", method_b="DFS+Guidance",
                                   show_band=True):
    """Compare two methods after averaging variants, then averaging over tasks."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.25,
                  rc={"font.family": "DejaVu Sans"})

    overall_a = _overall_by_method(df_a, method_a)
    overall_b = _overall_by_method(df_b, method_b)
    common_diffs = _common_difficulties(overall_a, overall_b)
    if not common_diffs:
        raise SystemExit("No overlapping difficulty levels for overall comparison.")
    overall_a = overall_a[overall_a["difficulty"].isin(common_diffs)]
    overall_b = overall_b[overall_b["difficulty"].isin(common_diffs)]

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    deep = sns.color_palette("deep")
    for overall, label, color in (
        (overall_a, method_a, deep[0]),
        (overall_b, method_b, deep[1]),
    ):
        plot_stats = overall.rename(columns={"difficulty": "x"})
        _plot_mean_std(
            ax, plot_stats, "x", label, color,
            show_band=show_band, linewidth=2.4, markersize=8,
        )

    x_pad = 0.025
    ax.set_xticks(common_diffs)
    ax.set_xticklabels([f"{d:.1f}" for d in common_diffs])
    ax.set_xlim(common_diffs[0] - x_pad, common_diffs[-1] + x_pad)

    combined = pd.concat([overall_a, overall_b], ignore_index=True)
    if show_band:
        y_lo = (combined["mean"] - combined["std"]).min()
        y_hi = (combined["mean"] + combined["std"]).max()
    else:
        y_lo, y_hi = combined["mean"].min(), combined["mean"].max()
    y_pad = max(2.0, 0.06 * (y_hi - y_lo))
    ax.set_ylim(max(0.0, y_lo - y_pad), min(100.0, y_hi + y_pad))

    ax.set_xlabel("Difficulty Level", labelpad=6)
    ax.set_ylabel("Average Success Rate (%)", labelpad=6)
    ax.set_title(f"Overall: {method_a} vs {method_b}", fontsize=11, pad=8)

    _style_axes(ax)
    _style_legend(ax)

    plt.tight_layout(pad=0.4)
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved figure to: {output_path}")


def print_summary(df, label=None):
    diffs = sorted(df["difficulty"].unique())
    tasks = sorted(df["task"].unique())
    print("=" * 88)
    if label:
        print(label)
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
                        help="Output path for the first method's per-task figure.")
    parser.add_argument("--input-label", default="DFS",
                        help="Legend/title label for --input.")
    parser.add_argument("--overall-output", default=None,
                        help="Output path for the first method's overall-average figure. "
                             "Defaults to <output>_overall.<ext>.")
    parser.add_argument("--guidance-input", default=None,
                        help="Path to the second method's results file. "
                             "If provided, all second-method and comparison figures are generated.")
    parser.add_argument("--guidance-label", default="DFS+Guidance",
                        help="Legend/title label for --guidance-input.")
    parser.add_argument("--guidance-output", default=None,
                        help="Output path for the second method's per-task figure. "
                             "Defaults to <output>_<guidance-label>.<ext>.")
    parser.add_argument("--guidance-overall-output", default=None,
                        help="Output path for the second method's overall-average figure. "
                             "Defaults to <output>_<guidance-label>_overall.<ext>.")
    parser.add_argument("--comparison-output", default=None,
                        help="Output path for one task-comparison figure. "
                             "If plotting multiple tasks, _taskT is inserted before the extension.")
    parser.add_argument("--comparison-overall-output", default=None,
                        help="Output path for the two-method overall comparison. "
                             "Defaults to <output>_compare_overall.<ext>.")
    parser.add_argument("--comparison-task", type=int, default=None,
                        help="Only plot this one task comparison. "
                             "By default, plot every task present in both inputs.")
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

    print_summary(df, label=args.input_label)
    plot_success_rate(
        df, args.output, show_band=not args.no_band,
        method_label=args.input_label,
    )
    plot_overall_average(
        df, args.overall_output, show_band=not args.no_band,
        method_label=args.input_label,
    )

    if args.guidance_input is not None:
        df_guidance = parse_results(args.guidance_input)
        if df_guidance.empty:
            print(f"Warning: no second-method entries found in {args.guidance_input}.")
        else:
            safe_guidance = _safe_label(args.guidance_label)
            if args.guidance_output is None:
                args.guidance_output = _insert_suffix(args.output, f"_{safe_guidance}")
            if args.guidance_overall_output is None:
                args.guidance_overall_output = _insert_suffix(
                    args.output, f"_{safe_guidance}_overall")

            print_summary(df_guidance, label=args.guidance_label)
            plot_success_rate(
                df_guidance, args.guidance_output,
                show_band=not args.no_band,
                method_label=args.guidance_label,
            )
            plot_overall_average(
                df_guidance, args.guidance_overall_output,
                show_band=not args.no_band,
                method_label=args.guidance_label,
            )

            common_tasks = sorted(set(df["task"]) & set(df_guidance["task"]))
            if args.comparison_task is not None:
                common_tasks = [args.comparison_task]
            if not common_tasks:
                raise SystemExit("No overlapping tasks between the two inputs.")

            many_tasks = len(common_tasks) > 1
            for task in common_tasks:
                if args.comparison_output is None:
                    comparison_output = _insert_suffix(args.output, f"_compare_task{task}")
                elif many_tasks:
                    comparison_output = _insert_suffix(args.comparison_output, f"_task{task}")
                else:
                    comparison_output = args.comparison_output
                plot_method_comparison(
                    df, df_guidance, comparison_output,
                    task=task,
                    method_a=args.input_label,
                    method_b=args.guidance_label,
                    show_band=not args.no_band,
                )

            if args.comparison_overall_output is None:
                args.comparison_overall_output = _insert_suffix(
                    args.output, "_compare_overall")
            plot_overall_method_comparison(
                df, df_guidance, args.comparison_overall_output,
                method_a=args.input_label,
                method_b=args.guidance_label,
                show_band=not args.no_band,
            )


if __name__ == "__main__":
    main()

# Per-method per-task + overall, per-task comparisons for every common task,
# plus one overall comparison averaged over tasks.
# python plot_success_rate.py -i results_newvar_level.txt --guidance-input results_newvar_bfs_guidance_level.txt

# Custom paths
# python plot_success_rate.py -i results_dfs_pointmaze-giant-newvar-navigate-v0-dfs.txt -o pure_dfs_results.png --no-band

# Plot only one task comparison.
# python plot_success_rate.py -i results_newvar_level.txt \
#     --guidance-input results_newvar_bfs_guidance_level.txt \
#     --comparison-task 1

# Disable the ±1 std shaded band
# python plot_success_rate.py --no-band