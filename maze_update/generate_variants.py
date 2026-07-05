#!/usr/bin/env python
"""
generate_variants.py
====================
Maze-variant generator

Pipeline
--------
For each giant task T (start/goal are the env's hardcoded giant tasks):
  1. Build a POOL of candidate variants over a spread of (block, open) budgets,
     each biased to cut the current optimal corridor so a reroute is forced.
  2. De-duplicate the pool by optimal-path signature (unique routes only).
  3. Score every unique-path candidate with the difficulty_score metric
     (difficulty_metric.rank_composite over the pool).
  4. Quantile-bin the pool into `--n-levels` levels; from each level pick
     `--n-variants` path-spread candidates (farthest-point on path Jaccard).
     Cross-level path-distinctness is automatic (unique-path pool).
  5. Emit `giant_task{T}.json` in the OGBench schema the pipeline already loads
     (variants[] indexed by maze_variant_idx = level*n_variants + variant), so it
     is drop-in for base_pipeline._make_env / run_eval.py.

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
import route_space as rs

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
# Authoritative maze definition lives in ogbench's locomaze/maze.py. The giant
# maze_map is a LOCAL inside the env __init__ (an `if self._maze_type == 'giant'`
# branch), so it can't be imported; we parse the literal straight out of the
# source with ast. This keeps the generator locked to the upstream source of
# truth instead of a cached copy under maze_variants/ (which is a previous
# generation's output and may be deleted).
MAZE_PY = os.path.join(REPO, "pointmaze", "ogbench", "ogbench",
                       "locomaze", "maze.py")


def load_base_maze(maze_type="giant", maze_py_path=MAZE_PY):
    """Base giant maze grid (0 free / 1 wall), parsed from ogbench maze.py.

    Walks the AST of maze.py for the `if self._maze_type == '<maze_type>':`
    branch and returns the `maze_map = [[...]]` literal assigned inside it.
    """
    import ast
    if not os.path.exists(maze_py_path):
        raise FileNotFoundError(
            f"ogbench maze.py not found at {maze_py_path}; cannot read the "
            f"authoritative {maze_type!r} maze definition.")
    tree = ast.parse(open(maze_py_path).read())
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) \
           and any(isinstance(c, ast.Constant) and c.value == maze_type
                   for c in node.test.comparators):
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(
                        getattr(t, "id", None) == "maze_map" for t in stmt.targets):
                    found = ast.literal_eval(stmt.value)
    if found is None:
        raise RuntimeError(
            f"maze_map literal for {maze_type!r} not found in {maze_py_path}")
    return np.array(found, dtype=int)


def load_traffic(shape):
    obs = os.path.join(REPO, "pointmaze", "ogbench", "data",
                       "pointmaze-giant-navigate-v0", "observations.npy")
    if not os.path.exists(obs):
        print(f"[warn] {obs} missing; corridor-novelty term will be 0")
        return None
    return dm.traffic_map_from_observations(np.load(obs), shape)


# --------------------------------------------------------------------------- #
# candidate generation: open alternate corridors + block the current one
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


def _interior_walls(grid):
    """Interior (non-border) wall cells -- the pool of walls we may open."""
    H, W = grid.shape
    return [(i, j) for i in range(1, H - 1) for j in range(1, W - 1) if grid[i, j] == 1]


def make_candidate(base, start, goal, n_block, n_open, rng):
    """
    Produce one edit: OPEN up to `n_open` interior walls (to create alternate
    corridors), then BLOCK up to `n_block` cells on the current optimal corridor
    (to force the reroute onto one of the new/other corridors).

    Returns (grid, blocked_set, opened_set) or None if it can't produce a
    reachable, path-moving, non-shortcut edit. The start & goal cells and the
    outer border frame are never modified. A candidate is rejected if its optimal
    path is SHORTER than base (a shortcut = easier) or identical to base (no-op).
    """
    start = tuple(start); goal = tuple(goal)
    grid = base.copy()
    base_len = dm.shortest_path_len(base, start, goal)
    base_path = dm.optimal_path_cells(base, start, goal)
    base_pathset = set(base_path)

    # (1) OPEN a few interior walls to seed alternate corridors.
    opened = set()
    if n_open > 0:
        walls = _interior_walls(grid)
        rng.shuffle(walls)
        for (i, j) in walls[:n_open]:
            grid[i, j] = 0
            opened.add((i, j))

    # (2) BLOCK cells on/around the CURRENT optimal corridor so the reroute is
    #     forced onto a different corridor. Re-aim after each block.
    _, pool = _corridor_cells(grid, start, goal, ring=1)
    # never block start/goal or a cell we just opened (that would be a pointless
    # open+reblock); blocking an originally-free cell is what forces the detour
    pool = [c for c in pool if c not in opened]
    rng.shuffle(pool)
    blocked = set()
    seen_pool = set(pool)
    for (i, j) in pool:
        if len(blocked) >= n_block:
            break
        grid[i, j] = 1
        if not np.isfinite(dm.shortest_path_len(grid, start, goal)):
            grid[i, j] = 0
            continue
        blocked.add((i, j))
        if len(blocked) < n_block:
            _, pool2 = _corridor_cells(grid, start, goal, ring=1)
            for c in pool2:
                if c not in blocked and c not in opened and c not in seen_pool:
                    pool.append(c); seen_pool.add(c)

    if not blocked and not opened:
        return None
    if not np.isfinite(dm.shortest_path_len(grid, start, goal)):
        return None

    new_path = dm.optimal_path_cells(grid, start, goal)
    new_len = dm.shortest_path_len(grid, start, goal)
    # reject shortcuts (easier) and exact no-ops (same path)
    if new_len < base_len:
        return None
    if set(new_path) == base_pathset:
        return None
    # trim opened walls that ended up unused by the new optimal path AND are not
    # adjacent to it -- keeps the edit description honest (opens that actually
    # matter). We only drop opens that are NOT on and NOT touching the new path.
    if opened:
        pathset = set(new_path)
        keep = set()
        for (i, j) in opened:
            touches = any((i + di, j + dj) in pathset
                          for di, dj in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)))
            if touches:
                keep.add((i, j))
        for (i, j) in opened - keep:
            grid[i, j] = 1  # re-close the irrelevant opening
        opened = keep
        # re-verify after re-closing
        if not np.isfinite(dm.shortest_path_len(grid, start, goal)):
            return None
        new_path = dm.optimal_path_cells(grid, start, goal)
        if dm.shortest_path_len(grid, start, goal) < base_len or set(new_path) == base_pathset:
            return None

    return grid, blocked, opened


def build_pool(base, start, goal, traffic, seed, budgets, per_budget,
               route_k=3, overlap_tol=0.75):
    """
    A scored pool of candidates, DE-DUPLICATED BY TOP-k ROUTE-SET SIGNATURE.

    Each budget is a (n_block, n_open) pair. A candidate is keyed not by its
    single optimal path but by the SET of its up-to-``route_k`` structurally
    distinct routes (``route_space.route_signature``): two candidates collide
    only when their whole diverse-route set matches. This captures variants
    that share one geodesic yet differ in their alternate corridors, and
    empirically widens the distinct pool ~1.5-2x versus single-path keying.

    For every distinct route-set we keep a single representative -- the one
    produced with the fewest total edits -- so every pool entry traces a unique
    route set. This is what guarantees the ladder is route-distinct across and
    within levels.
    """
    rng = np.random.default_rng(seed)
    by_routes = {}   # route-set signature -> best candidate dict (fewest edits)
    for (nb, no) in budgets:
        tries = 0
        made_here = 0
        while made_here < per_budget and tries < per_budget * 16:
            tries += 1
            out = make_candidate(base, start, goal, nb, no, rng)
            if out is None:
                continue
            grid, blocked, opened = out
            sig = rs.route_signature(grid, start, goal, k=route_k,
                                     overlap_tol=overlap_tol)
            n_edit = len(blocked) + len(opened)
            prev = by_routes.get(sig)
            if prev is not None and prev["n_edit"] <= n_edit:
                continue
            # optimal path retained for path_len and within-level distinctness
            opt = tuple(dm.optimal_path_cells(grid, start, goal))
            res = dm.compute_variant_features(base, grid, start, goal, traffic)
            by_routes[sig] = dict(grid=grid, blocked=set(blocked), opened=set(opened),
                                  path=opt, routes=sig, n_edit=n_edit,
                                  features=res.features,
                                  structural=res.structural,
                                  n_block_target=nb, n_open_target=no)
            made_here += 1
    return list(by_routes.values())


# --------------------------------------------------------------------------- #
# select path-distinct variants within each quantile level
# --------------------------------------------------------------------------- #
def _jaccard(a, b):
    a, b = set(a), set(b)
    u = len(a | b)
    return 1.0 - len(a & b) / u if u else 0.0


def pick_distinct(members, k, rng):
    """
    Greedy farthest-point pick of k candidates spread by OPTIMAL-PATH Jaccard.

    Selecting on the path itself (not just the edited cells) maximizes how
    differently the agent must actually travel within a level. Cross-level
    path-distinctness is already guaranteed by the unique-path pool.
    """
    if len(members) <= k:
        return members
    idx0 = int(rng.integers(len(members)))
    chosen_idx = [idx0]
    while len(chosen_idx) < k:
        best, best_d = None, -1.0
        for mi, m in enumerate(members):
            if mi in chosen_idx:
                continue
            d = min(_jaccard(m["path"], members[ci]["path"]) for ci in chosen_idx)
            if d > best_d:
                best_d, best = d, mi
        chosen_idx.append(best)
    return [members[i] for i in chosen_idx]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def generate_task(task_id, base, traffic, seed, n_levels, n_variants,
                  pool_per_budget, max_block, max_open):
    start, goal = GIANT_TASKS[task_id]
    # (n_block, n_open) budget grid: every block level crossed with 0..max_open
    # opens. n_block starts at 0 so a pure-open reroute is also reachable.
    budgets = [(nb, no) for nb in range(0, max_block + 1)
               for no in range(0, max_open + 1)
               if (nb + no) > 0]
    pool = build_pool(base, start, goal, traffic, seed + task_id,
                      budgets, pool_per_budget)
    if len(pool) < n_levels * n_variants:
        print(f"[warn] task{task_id}: only {len(pool)} unique-path candidates in pool")
    scores = dm.rank_composite([c["features"] for c in pool])
    for c, s in zip(pool, scores):
        c["difficulty_score"] = float(s)
    levels = dm.quantile_levels(scores, n_levels)
    rng = np.random.default_rng(seed + 100 + task_id)

    variants = []
    level_target = np.linspace(0.2, 0.65, n_levels)  # legacy target_level labels
    # Ladder-wide optimal-path distinctness. Pool members are de-duplicated by
    # their top-k ROUTE SET, so two members can still share one optimal path
    # (differing only in alternates). We additionally forbid any two CHOSEN
    # variants -- within OR across levels -- from tracing the same optimal path.
    used_paths = set()
    for lv in range(n_levels):
        members = [c for c, L in zip(pool, levels) if L == lv]
        members.sort(key=lambda c: c["difficulty_score"])
        # collapse to one representative per optimal path (fewest edits), and
        # drop any path already spent by an earlier level
        by_opt = {}
        for c in members:
            opt = c["path"]
            if opt in used_paths:
                continue
            prev = by_opt.get(opt)
            if prev is None or c["n_edit"] < prev["n_edit"]:
                by_opt[opt] = c
        members = sorted(by_opt.values(), key=lambda c: c["difficulty_score"])
        chosen = pick_distinct(members, n_variants, rng)
        for c in chosen:
            used_paths.add(c["path"])
        chosen.sort(key=lambda c: c["difficulty_score"])
        for vi, c in enumerate(chosen):
            f = c["features"]
            variants.append(dict(
                maze_type="giant", task=f"task{task_id}",
                start=list(start), goal=list(goal),
                start_xy=list(dm.ij_to_xy(*start)), goal_xy=list(dm.ij_to_xy(*goal)),
                target_level=float(round(level_target[lv], 2)),
                variant_index=vi, level_index=lv,
                difficulty_score=float(c["difficulty_score"]),
                n_blocked=int(f.get("n_blocked", 0)),
                n_opened=int(f.get("n_opened", 0)),
                path_len=int(len(c["path"])),
                sp_ratio=float(f.get("sp_ratio", 1.0)),
                net_shift=float(f.get("net_shift", 0.0)),
                corridor_novelty=float(f.get("corridor_novelty", 0.0)),
                deadend_delta=float(f.get("deadend_delta", 0.0)),
                spectral_dist=float(c["structural"].get("spectral_dist", float("nan"))),
                edge_jaccard_dist=float(c["structural"].get("edge_jaccard_dist", float("nan"))),
                goal_reachable=True,
                broken_cells=[list(b) for b in sorted(c["blocked"])],
                opened_cells=[list(o) for o in sorted(c["opened"])],
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
    ap.add_argument("--max-block", type=int, default=8)
    ap.add_argument("--max-open", type=int, default=4,
                    help="max interior walls a candidate may open (0 = block-only)")
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
            args.pool_per_budget, args.max_block, args.max_open)
        out_file = os.path.join(args.out, f"giant_task{t}.json")
        json.dump(data, open(out_file, "w"))
        # per-level difficulty of the CHOSEN variants
        by_lvl = {}
        for v in data["variants"]:
            by_lvl.setdefault(v["level_index"], []).append(v["difficulty_score"])
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

    # path-distinctness audit: every chosen variant must trace a UNIQUE optimal
    # path -- no two variants (within OR across levels) may share a route.
    print("\n== chosen-variant optimal paths all distinct? ==")
    for t in args.tasks:
        data = ladder_data[t]
        start, goal = tuple(data["start"]), tuple(data["goal"])
        sigs = [tuple(dm.optimal_path_cells(np.array(v["maze_map"]), start, goal))
                for v in data["variants"]]
        n_var = len(sigs); n_uniq = len(set(sigs))
        print(f"  task{t}: {n_uniq}/{n_var} distinct paths  "
              f"{'OK' if n_uniq == n_var else 'DUPLICATE PATHS'}")

    _plot_ladder(args.ladder_png, ladder_data, base)
    print(f"\nsaved contact sheet {args.ladder_png}")


def _plot_ladder(out, ladder_data, base):
    """
    Contact sheet: one ROW per task, one COLUMN per (level, variant) so every
    chosen variant's distinct optimal path is visible side by side.

    Orientation follows the ORIGINAL convention: origin='lower', i.e. maze row 0
    at the BOTTOM and y increasing upward (the same world-coordinate framing as
    reroute_disruption.plot_suite), NOT the array-display row-0-at-top.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    tasks = sorted(ladder_data)
    n_lvl = ladder_data[tasks[0]]["n_levels"]
    n_var = ladder_data[tasks[0]]["n_variants"]
    ncols = n_lvl * n_var
    fig, axes = plt.subplots(len(tasks), ncols,
                             figsize=(1.7 * ncols, 2.0 * len(tasks)), squeeze=False)
    H, W = base.shape
    for r, t in enumerate(tasks):
        data = ladder_data[t]
        start, goal = data["start"], data["goal"]
        cells = {}
        for v in data["variants"]:
            cells[(v["level_index"], v["variant_index"])] = v
        col = 0
        for lv in range(n_lvl):
            for vi in range(n_var):
                ax = axes[r][col]; col += 1
                ax.set_xticks([]); ax.set_yticks([])
                v = cells.get((lv, vi))
                if v is None:
                    ax.axis("off"); continue
                grid = np.array(v["maze_map"])
                # row 0 at bottom -> original inverted-y print convention
                ax.imshow(grid, cmap="Greys", vmin=0, vmax=1, origin="lower")
                ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(-0.5, H - 0.5)
                ax.set_aspect("equal")
                for (bi, bj) in v["broken_cells"]:               # added walls
                    ax.add_patch(Rectangle((bj - .5, bi - .5), 1, 1,
                                           color="#d1495b", alpha=.85, zorder=3))
                for (oi, oj) in v.get("opened_cells", []):        # opened walls
                    ax.add_patch(Rectangle((oj - .5, oi - .5), 1, 1,
                                           facecolor="none", edgecolor="#2a9d8f",
                                           lw=1.8, zorder=3))
                path = dm.optimal_path_cells(grid, tuple(start), tuple(goal))
                if path:
                    pi = [p[0] for p in path]; pj = [p[1] for p in path]
                    ax.plot(pj, pi, color="#1f6feb", lw=1.6, zorder=4)
                ax.plot(start[1], start[0], "o", color="#2e7d32", ms=5, zorder=5)
                ax.plot(goal[1], goal[0], "*", color="#f0a202", ms=9, zorder=5)
                if r == 0:
                    ax.set_title(f"L{lv+1}.{vi+1}", fontsize=9)
                if col - 1 == 0:
                    ax.set_ylabel(f"task{t}", fontsize=10)
                ax.text(0.5, -0.02, f"d{v['difficulty_score']:.2f} +{v['n_blocked']}b/-{v.get('n_opened',0)}o",
                        transform=ax.transAxes, ha="center", va="top", fontsize=6.5)
    fig.suptitle("Variant ladder (row 0 at bottom): forced optimal path (blue), added walls (red), "
                 "opened walls (teal outline), start (green dot), goal (gold star)\n"
                 "Columns grouped by level L1..L%d (each with %d path-distinct variants); "
                 "difficulty rises left->right" % (n_lvl, n_var),
                 fontsize=10, y=1.01)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")


if __name__ == "__main__":
    main()
