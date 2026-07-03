#!/usr/bin/env python
"""
Plot how each GOOD_FEATURE (and the composite difficulty score) varies with
difficulty level, across all 5 giant tasks.

For every task JSON in ``maze_variants_v2/`` this reads the per-variant feature
values, groups them by ``level_index`` (0..3 -> L1..L4), and draws a 2x3 panel
figure:

    panel 0 : difficulty_score  (the composite that bins the ladder)
    panel 1 : spectral_dist     (Laplacian-spectrum structural distance)
    panel 2 : sp_ratio          (variant shortest path / base shortest path)
    panel 3 : net_shift         (mean BFS-distance-to-goal increase)
    panel 4 : deadend_delta     (increase in dead-end cell fraction)
    panel 5 : n_blocked         (number of walls added)

Each panel shows one mean line per task (over the 3 variants at each level)
with the raw per-variant values as faint points behind it.

Usage:
    python plot_feature_vs_level.py                      # writes feature_vs_level.png
    python plot_feature_vs_level.py --out mystuff.png
    python plot_feature_vs_level.py --variants-dir maze_variants_v2
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt


# The 5 features that make up the composite difficulty score, plus the score
# itself. Keys must match the field names stored in each variant JSON entry.
GOOD_FEATURES = ["spectral_dist", "sp_ratio", "net_shift", "deadend_delta", "n_blocked"]

PANELS = [
    ("difficulty_score", "difficulty score (composite)"),
    ("spectral_dist", "spectral distance"),
    ("sp_ratio", "shortest-path ratio"),
    ("net_shift", "net distance shift"),
    ("deadend_delta", "dead-end fraction increase"),
    ("n_blocked", "walls blocked"),
]


def load_task(path: str):
    """Return {field: {level_index: [values over variants]}} for one task JSON."""
    d = json.load(open(path))
    by_level: dict[int, list] = {}
    for v in d["variants"]:
        by_level.setdefault(v["level_index"], []).append(v)
    fields = GOOD_FEATURES + ["difficulty_score"]
    raw = {f: {L: [m[f] for m in by_level[L]] for L in sorted(by_level)} for f in fields}
    return raw, sorted(by_level)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants-dir", default="maze_variants_v2",
                    help="directory holding giant_task{1..5}.json (default: maze_variants_v2)")
    ap.add_argument("--out", default="feature_vs_level.png",
                    help="output PNG path (default: feature_vs_level.png)")
    ap.add_argument("--tasks", default="1,2,3,4,5",
                    help="comma-separated task numbers to include (default: 1,2,3,4,5)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    vdir = args.variants_dir
    if not os.path.isabs(vdir):
        vdir = os.path.join(here, vdir)

    task_nums = [s.strip() for s in args.tasks.split(",") if s.strip()]
    tasks = [f"task{n}" for n in task_nums]

    raw = {}
    level_idx = None
    for t in tasks:
        path = os.path.join(vdir, f"giant_{t}.json")
        raw[t], level_idx = load_task(path)

    # mean per level (line), keeping raw points for the scatter
    means = {t: {f: [np.mean(raw[t][f][L]) for L in level_idx] for f in GOOD_FEATURES + ["difficulty_score"]}
             for t in tasks}

    # L1..L4 on the x-axis (level_index 0..3 shifted to 1-based labels)
    xs = [i + 1 for i in range(len(level_idx))]
    cmap = plt.get_cmap("viridis")
    colors = {t: cmap(i / max(1, len(tasks) - 1)) for i, t in enumerate(tasks)}

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6))
    for ax, (f, title) in zip(axes.flat, PANELS):
        for t in tasks:
            for xi, L in enumerate(level_idx):
                ys = raw[t][f][L]
                ax.scatter([xs[xi]] * len(ys), ys, s=10, color=colors[t],
                           alpha=0.25, zorder=1, edgecolors="none")
            ax.plot(xs, means[t][f], "-o", color=colors[t], ms=4, lw=1.6,
                    zorder=3, label=t.replace("task", "task "))
        ax.set_title(title)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"L{k}" for k in xs])
        ax.margins(x=0.08)
        if f == "difficulty_score":
            ax.set_ylabel("score (0-1)")
            ax.set_facecolor("#f6f6fb")  # highlight the headline panel
    for ax in axes[1, :]:
        ax.set_xlabel("difficulty level")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(tasks), frameon=False,
               bbox_to_anchor=(0.5, 1.005), columnspacing=1.6)
    fig.suptitle("Each feature rises with difficulty level "
                 "(mean line + per-variant points)", y=1.045, fontsize=11)
    fig.tight_layout()

    out = args.out
    if not os.path.isabs(out):
        out = os.path.join(here, out)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
