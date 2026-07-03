#!/usr/bin/env python
"""
Re-plot the saved maze variants in ``maze_variants_v2/`` using the SAME contact-
sheet convention as ``generate_variants._plot_ladder`` (the one that produced
variant_ladder.png), but driven entirely from the saved JSON -- no regeneration.

Rendering convention (identical to _plot_ladder):
    * origin='lower'  -> maze row 0 at the BOTTOM, y increasing upward
      (world-coordinate framing, matching reroute_disruption.plot_suite).
    * blue line   : forced optimal path (recomputed from the map with
                    difficulty_metric.optimal_path_cells).
    * red block   : an added wall (broken_cells).
    * teal outline: an opened wall (opened_cells).
    * green dot   : start,  gold star : goal.
    * caption per cell: d{difficulty_score}  +{n_blocked}b / -{n_opened}o.

Usage:
    python plot_variant_maps.py                         # -> variant_ladder_v2.png (all tasks)
    python plot_variant_maps.py --out sheet.png
    python plot_variant_maps.py --tasks 1,3             # subset of tasks
    python plot_variant_maps.py --per-task              # one PNG per task instead of one sheet
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# make sibling modules (difficulty_metric, maze_utils) importable regardless of cwd
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import difficulty_metric as dm


def _load(vdir, task):
    return json.load(open(os.path.join(vdir, f"giant_{task}.json")))


def _draw_cell(ax, v, start, goal, H, W):
    """Render one variant into ax, matching _plot_ladder exactly."""
    ax.set_xticks([]); ax.set_yticks([])
    grid = np.array(v["maze_map"])
    ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, origin="lower")
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(-0.5, H - 0.5)
    ax.set_aspect("equal")
    for (bi, bj) in v["broken_cells"]:                       # added walls
        ax.add_patch(Rectangle((bj - .5, bi - .5), 1, 1,
                               color="#d1495b", alpha=.85, zorder=3))
    for (oi, oj) in v.get("opened_cells", []):               # opened walls
        ax.add_patch(Rectangle((oj - .5, oi - .5), 1, 1,
                               facecolor="none", edgecolor="#2a9d8f",
                               lw=1.8, zorder=3))
    path = dm.optimal_path_cells(grid, tuple(start), tuple(goal))
    if path:
        pi = [p[0] for p in path]; pj = [p[1] for p in path]
        ax.plot(pj, pi, color="#1f6feb", lw=1.6, zorder=4)
    ax.plot(start[1], start[0], "o", color="#2e7d32", ms=5, zorder=5)
    ax.plot(goal[1], goal[0], "*", color="#f0a202", ms=9, zorder=5)
    ax.text(0.5, -0.02,
            f"d{v['difficulty_score']:.2f} +{v['n_blocked']}b/-{v.get('n_opened', 0)}o",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5)


def plot_sheet(vdir, tasks, out):
    """One row per task, one column per (level, variant) -- the full contact sheet."""
    d0 = _load(vdir, tasks[0])
    n_lvl, n_var = d0["n_levels"], d0["n_variants"]
    base = np.array(d0["base_maze"])
    H, W = base.shape
    ncols = n_lvl * n_var
    fig, axes = plt.subplots(len(tasks), ncols,
                             figsize=(1.7 * ncols, 2.0 * len(tasks)), squeeze=False)
    for r, t in enumerate(tasks):
        data = _load(vdir, t)
        start, goal = data["start"], data["goal"]
        cells = {(v["level_index"], v["variant_index"]): v for v in data["variants"]}
        col = 0
        for lv in range(n_lvl):
            for vi in range(n_var):
                ax = axes[r][col]; col += 1
                v = cells.get((lv, vi))
                if v is None:
                    ax.axis("off"); continue
                _draw_cell(ax, v, start, goal, H, W)
                if r == 0:
                    ax.set_title(f"L{lv+1}.{vi+1}", fontsize=9)
                if col - 1 == 0:
                    ax.set_ylabel(t, fontsize=10)
    fig.suptitle("Variant ladder (row 0 at bottom): forced optimal path (blue), added walls (red), "
                 "opened walls (teal outline), start (green dot), goal (gold star)\n"
                 "Columns grouped by level L1..L%d (each with %d path-distinct variants); "
                 "difficulty rises left->right" % (n_lvl, n_var),
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def plot_per_task(vdir, task, out):
    """One PNG for a single task: rows = levels, cols = variants."""
    data = _load(vdir, task)
    n_lvl, n_var = data["n_levels"], data["n_variants"]
    base = np.array(data["base_maze"])
    H, W = base.shape
    start, goal = data["start"], data["goal"]
    cells = {(v["level_index"], v["variant_index"]): v for v in data["variants"]}
    fig, axes = plt.subplots(n_lvl, n_var,
                             figsize=(2.0 * n_var, 2.2 * n_lvl), squeeze=False)
    for lv in range(n_lvl):
        for vi in range(n_var):
            ax = axes[lv][vi]
            v = cells.get((lv, vi))
            if v is None:
                ax.axis("off"); continue
            _draw_cell(ax, v, start, goal, H, W)
            if lv == 0:
                ax.set_title(f"variant {vi+1}", fontsize=9)
            if vi == 0:
                ax.set_ylabel(f"L{lv+1}", fontsize=10)
    fig.suptitle(f"{task}: variant ladder (row 0 at bottom) -- rows = difficulty level, "
                 "cols = path-distinct variants", fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants-dir", default="maze_variants_v2")
    ap.add_argument("--out", default="variant_ladder_v2.png",
                    help="output PNG (contact-sheet mode); ignored with --per-task")
    ap.add_argument("--tasks", default="1,2,3,4,5")
    ap.add_argument("--per-task", action="store_true",
                    help="write one PNG per task (giant_task{N}_ladder.png) instead of one sheet")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    vdir = args.variants_dir
    if not os.path.isabs(vdir):
        vdir = os.path.join(here, vdir)
    tasks = [f"task{n.strip()}" for n in args.tasks.split(",") if n.strip()]

    if args.per_task:
        for t in tasks:
            out = os.path.join(here, f"giant_{t}_ladder.png")
            plot_per_task(vdir, t, out)
    else:
        out = args.out if os.path.isabs(args.out) else os.path.join(here, args.out)
        plot_sheet(vdir, tasks, out)


if __name__ == "__main__":
    main()
