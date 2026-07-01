#!/usr/bin/env python
"""
generate_variants.py
====================
Rebuilt maze-variant generator for graded, *monotone* planner difficulty.

Why a rewrite
-------------
The old RDI generator opened AND blocked walls in the same edit. A variant could
therefore be structurally very different yet net-EASIER (it opened a shortcut),
which is exactly why RDI rank-correlated with measured success at only
rho = -0.30..-0.37 and why nominal "levels" were non-monotone.

This generator removes that confound at the source:

  * EDITS ONLY ADD WALLS. Free cells near the task's optimal corridor are turned
    into walls; walls are never opened. Consequently every calibrated difficulty
    feature (sp_ratio >= 1, net goal-field shift >= 0, corridor novelty, dead-end
    growth, #blocked) is monotone in the amount of blocking -- so binning the
    calibrated metric into quantiles produces a monotone success ladder.
  * NO-OPS ARE REJECTED. A candidate is kept only if it actually moves the
    optimal path (path length increases OR the set of path cells changes).
  * GOAL STAYS REACHABLE. Candidates that disconnect start->goal are discarded.
  * START & GOAL CELLS ARE NEVER BLOCKED.

Pipeline
--------
For each giant task T (start/goal are the env's hardcoded giant tasks):
  1. Build a POOL of candidate variants at a spread of blocking budgets, each
     biased to cut the current optimal corridor (+ its 1-ring) so the reroute is
     forced.
  2. Score every candidate with the CALIBRATED metric
     (difficulty_metric.rank_composite over the pool).
  3. Quantile-bin the pool into `--n-levels` levels; from each level pick
     `--n-variants` structurally-distinct candidates (min pairwise blocked-cell
     Jaccard).
  4. Emit `giant_task{T}.json` in the OGBench schema the pipeline already loads
     (variants[] indexed by maze_variant_idx = level*n_variants + variant), so it
     is drop-in for base_pipeline._make_env.

Output goes to a NEW directory (default maze_update/maze_variants_v2) so the
original maze_variants/ is left untouched; point run.py --maze_json_dir at it.

Run:
    python maze_update/generate_variants.py --seed 0
    python maze_update/generate_variants.py --tasks 1 2 --n-levels 4 --n-variants 3
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import difficulty_metric as dm

# Env's hardcoded giant tasks (locomaze/maze.py set_tasks, 'giant'): (init_ij, goal_ij)
GIANT_TASKS = {
    1: ((1, 1), (10, 14)),
    2: ((1, 14), (10, 1)),
    3: ((8, 14), (1, 1)),
    4: ((8, 3), (5, 12)),
    5: ((5, 9), (3, 8)),
}


# --------------------------------------------------------------------------- #
# base map + training traffic
# --------------------------------------------------------------------------- #
def load_base_maze():
    """Base giant maze grid (0 free / 1 wall) from the existing task1 file."""
    f = os.path.join(HERE, "maze_variants", "giant_task1.json")
    return np.array(json.load(open(f))["base_maze"], dtype=int)


def load_traffic(shape):
    obs = os.path.join(REPO, "pointmaze", "ogbench", "data",
                       "pointmaze-giant-navigate-v0", "observations.npy")
    if not os.path.exists(obs):
        print(f"[warn] {obs} missing; corridor-novelty term will be 0")
        return None
    return dm.traffic_map_from_observations(np.load(obs), shape)


# --------------------------------------------------------------------------- #
# candidate generation: block cells near the optimal corridor
# --------------------------------------------------------------------------- #
def _corridor_cells(grid, start, goal, ring=1):
    """Optimal-path cells plus their `ring`-neighborhood free cells (block pool)."""
    path = dm.optimal_path_cells(grid, start, goal)
    H, W = grid.shape
    cells = set()
    for (i, j) in path:
        for di in range(-ring, ring + 1):
            for dj in range(-ring, ring + 1):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and grid[ni, nj] == 0:
                    cells.add((ni, nj))
    cells.discard(tuple(start)); cells.discard(tuple(goal))
    return path, sorted(cells)


def make_candidate(base, start, goal, n_block, rng):
    """
    Block up to `n_block` corridor cells. Returns (grid, blocked_set) or None if
    it can't produce a reachable, path-moving edit.
    """
    start = tuple(start); goal = tuple(goal)
    grid = base.copy()
    base_path, pool = _corridor_cells(grid, start, goal, ring=1)
    base_len = dm.shortest_path_len(base, start, goal)
    base_pathset = set(base_path)
    if not pool:
        return None
    rng.shuffle(pool)
    blocked = set()
    for (i, j) in pool:
        if len(blocked) >= n_block:
            break
        grid[i, j] = 1
        # keep goal reachable; otherwise revert this block
        if not np.isfinite(dm.shortest_path_len(grid, start, goal)):
            grid[i, j] = 0
            continue
        blocked.add((i, j))
        # re-aim at the CURRENT optimal corridor so later blocks keep biting
        if len(blocked) < n_block:
            _, pool2 = _corridor_cells(grid, start, goal, ring=1)
            for c in pool2:
                if c not in blocked and c not in pool:
                    pool.append(c)
    if not blocked:
        return None
    # reject no-ops: path must have actually moved
    new_path = dm.optimal_path_cells(grid, start, goal)
    new_len = dm.shortest_path_len(grid, start, goal)
    if new_len <= base_len and set(new_path) == base_pathset:
        return None
    return grid, blocked


def build_pool(base, start, goal, traffic, seed, budgets, per_budget):
    """A scored pool of unique candidates across a spread of blocking budgets."""
    rng = np.random.default_rng(seed)
    seen = set()
    cands = []
    for nb in budgets:
        made = 0
        tries = 0
        while made < per_budget and tries < per_budget * 12:
            tries += 1
            out = make_candidate(base, start, goal, nb, rng)
            if out is None:
                continue
            grid, blocked = out
            key = frozenset(blocked)
            if key in seen:
                continue
            seen.add(key)
            res = dm.compute_variant_features(base, grid, start, goal, traffic)
            cands.append(dict(grid=grid, blocked=blocked, features=res.features,
                              structural=res.structural, n_block_target=nb))
            made += 1
    return cands


# --------------------------------------------------------------------------- #
# select structurally-distinct variants within each quantile level
# --------------------------------------------------------------------------- #
def _jaccard(a, b):
    a, b = set(a), set(b)
    u = len(a | b)
    return 1.0 - len(a & b) / u if u else 0.0


def pick_distinct(members, k, rng):
    """Greedy farthest-point pick of k candidates by blocked-cell Jaccard."""
    if len(members) <= k:
        return members
    idx0 = int(rng.integers(len(members)))
    chosen_idx = [idx0]
    while len(chosen_idx) < k:
        best, best_d = None, -1.0
        for mi, m in enumerate(members):
            if mi in chosen_idx:
                continue
            d = min(_jaccard(m["blocked"], members[ci]["blocked"]) for ci in chosen_idx)
            if d > best_d:
                best_d, best = d, mi
        chosen_idx.append(best)
    return [members[i] for i in chosen_idx]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def generate_task(task_id, base, traffic, seed, n_levels, n_variants,
                  pool_per_budget, max_block):
    start, goal = GIANT_TASKS[task_id]
    budgets = list(range(1, max_block + 1))
    pool = build_pool(base, start, goal, traffic, seed + task_id,
                      budgets, pool_per_budget)
    if len(pool) < n_levels * n_variants:
        print(f"[warn] task{task_id}: only {len(pool)} candidates in pool")
    scores = dm.rank_composite([c["features"] for c in pool])
    for c, s in zip(pool, scores):
        c["difficulty"] = float(s)
    levels = dm.quantile_levels(scores, n_levels)
    rng = np.random.default_rng(seed + 100 + task_id)

    variants = []
    level_target = np.linspace(0.2, 0.65, n_levels)  # legacy target_level labels
    for lv in range(n_levels):
        members = [c for c, L in zip(pool, levels) if L == lv]
        members.sort(key=lambda c: c["difficulty"])
        chosen = pick_distinct(members, n_variants, rng)
        chosen.sort(key=lambda c: c["difficulty"])
        for vi, c in enumerate(chosen):
            f = c["features"]
            variants.append(dict(
                maze_type="giant", task=f"task{task_id}",
                start=list(start), goal=list(goal),
                start_xy=list(dm.ij_to_xy(*start)), goal_xy=list(dm.ij_to_xy(*goal)),
                target_level=float(round(level_target[lv], 2)),
                variant_index=vi, level_index=lv,
                difficulty=float(c["difficulty"]),
                n_blocked=int(f.get("n_blocked", 0)),
                n_opened=int(f.get("n_opened", 0)),
                sp_ratio=float(f.get("sp_ratio", 1.0)),
                net_shift=float(f.get("net_shift", 0.0)),
                corridor_novelty=float(f.get("corridor_novelty", 0.0)),
                deadend_delta=float(f.get("deadend_delta", 0.0)),
                spectral_dist=float(c["structural"].get("spectral_dist", float("nan"))),
                edge_jaccard_dist=float(c["structural"].get("edge_jaccard_dist", float("nan"))),
                goal_reachable=True,
                broken_cells=[list(b) for b in sorted(c["blocked"])],
                maze_map=c["grid"].astype(int).tolist(),
            ))
    return dict(
        maze_type="giant", task=f"task{task_id}", seed=seed,
        start=list(start), goal=list(goal),
        metric="calibrated_planner_difficulty_rankcomposite",
        good_features=list(dm.GOOD_FEATURES),
        n_levels=n_levels, n_variants=n_variants,
        base_maze=base.astype(int).tolist(),
        variants=variants,
    ), pool, scores, levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-levels", type=int, default=4)
    ap.add_argument("--n-variants", type=int, default=3)
    ap.add_argument("--pool-per-budget", type=int, default=40)
    ap.add_argument("--max-block", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "maze_variants_v2"))
    ap.add_argument("--ladder-png", default=os.path.join(HERE, "variant_ladder.png"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    base = load_base_maze()
    traffic = load_traffic(base.shape)
    print(f"base maze {base.shape}, {int((base==0).sum())} free cells")

    summary = {}
    ladder_data = {}
    for t in args.tasks:
        data, pool, scores, levels = generate_task(
            t, base, traffic, args.seed, args.n_levels, args.n_variants,
            args.pool_per_budget, args.max_block)
        out_file = os.path.join(args.out, f"giant_task{t}.json")
        json.dump(data, open(out_file, "w"))
        # per-level difficulty of the CHOSEN variants
        by_lvl = {}
        for v in data["variants"]:
            by_lvl.setdefault(v["level_index"], []).append(v["difficulty"])
        summary[t] = {lv: float(np.mean(ds)) for lv, ds in sorted(by_lvl.items())}
        ladder_data[t] = data
        chosen_line = "  ".join(f"L{lv+1}:{summary[t][lv]:.2f}" for lv in sorted(summary[t]))
        print(f"task{t}: pool={len(pool)}  chosen difficulty  {chosen_line}  -> {out_file}")

    # monotonicity check on chosen-variant difficulty
    print("\n== chosen-variant difficulty monotonic in level? ==")
    for t in args.tasks:
        vals = [summary[t][lv] for lv in sorted(summary[t])]
        mono = all(x <= y + 1e-9 for x, y in zip(vals, vals[1:]))
        print(f"  task{t}: {[f'{v:.2f}' for v in vals]}  {'OK' if mono else 'NON-MONOTONE'}")

    _plot_ladder(args.ladder_png, ladder_data, base)
    print(f"\nsaved contact sheet {args.ladder_png}")


def _plot_ladder(out, ladder_data, base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    tasks = sorted(ladder_data)
    n_lvl = ladder_data[tasks[0]]["n_levels"]
    fig, axes = plt.subplots(len(tasks), n_lvl,
                             figsize=(2.3 * n_lvl, 2.3 * len(tasks)), squeeze=False)
    H, W = base.shape
    for r, t in enumerate(tasks):
        data = ladder_data[t]
        start, goal = data["start"], data["goal"]
        by_lvl = {}
        for v in data["variants"]:
            by_lvl.setdefault(v["level_index"], []).append(v)
        for c in range(n_lvl):
            ax = axes[r][c]
            v = min(by_lvl.get(c, [{}]), key=lambda z: z.get("variant_index", 0)) if by_lvl.get(c) else None
            ax.set_xticks([]); ax.set_yticks([])
            if v is None:
                ax.axis("off"); continue
            grid = np.array(v["maze_map"])
            ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, origin="upper")
            for (bi, bj) in v["broken_cells"]:
                ax.add_patch(Rectangle((bj - .5, bi - .5), 1, 1, color="#d1495b", alpha=.85))
            path = dm.optimal_path_cells(grid, tuple(start), tuple(goal))
            if path:
                pi = [p[0] for p in path]; pj = [p[1] for p in path]
                ax.plot(pj, pi, color="#1f6feb", lw=1.6)
            ax.plot(start[1], start[0], "o", color="#2e7d32", ms=5)
            ax.plot(goal[1], goal[0], "*", color="#f0a202", ms=9)
            if r == 0:
                ax.set_title(f"Level {c+1}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"task{t}", fontsize=10)
            ax.text(0.5, W and -0.06 or 0, f"d={v['difficulty']:.2f} | +{v['n_blocked']}w",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7)
    fig.suptitle("Variant ladder: added walls (red), forced optimal path (blue), "
                 "start (green), goal (gold)\nDifficulty rises left->right by calibrated metric",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
