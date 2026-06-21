"""
generate_data_ood.py
=====================

Generate OOD maze variants whose disruption is measured against the *training
data distribution* of an OGBench navigate dataset (e.g. pointmaze-giant), not a
single optimal path.

Disruption definition (matches the user's intent)
-------------------------------------------------
Breaking part of the maze is severe when BOTH:
  (1) the demonstrated data would need a very different route to get around the
      break (the data has no easy nearby alternative), and
  (2) a large fraction of the data actually collides with the break.
A small, easily-detoured, low-traffic edit is mild.

So the score multiplies a per-edit "reroute severity" (from the data's own
connectivity graph) by the "collision fraction" (share of demonstrated
transitions invalidated), then blends with raw coverage loss:

    data_disruption = w_sev * reroute_severity_norm
                    + w_col * collision_fraction
                    + w_cov * coverage_loss

All terms in [0, 1]. reroute_severity_norm is the traffic-weighted, detour-
weighted severity of the broken cells, normalized to [0, 1] across the maze.

Level targeting with randomness
-------------------------------
generate_variants_at_level(target, n_variants, ...) returns several DISTINCT
mazes that all score near `target`, using different random seeds / edit choices.
Same level, different variants.

Requires: the dataset directory (observations.npy, terminals.npy, ...) and the
base maze map + tasks (imported from generate_ood_mazes).
"""

from __future__ import annotations
import heapq
import json
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

import data_disruption as dd


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
from maze_utils import MAZE_MAPS, MAZE_TASKS, ij_to_xy, shortest_path, _neighbors


# ===========================================================================
# Data connectivity model: per-cell reroute severity
# ===========================================================================

@dataclass
class DataModel:
    traffic: dd.TrafficModel
    adj: dict                       # cell -> {neighbor: traffic}
    cell_severity: dict             # cell -> reroute severity (data detour cost)
    severity_norm: dict             # cell -> severity * traffic, normalized [0,1]
    max_cell_traffic: int
    base_grid: np.ndarray
    max_disruption: float = 1.0     # max achievable score while keeping goal reachable


def _build_adj(traffic: dd.TrafficModel):
    adj = defaultdict(dict)
    for (a, b), w in traffic.transitions.items():
        adj[a][b] = adj[a].get(b, 0) + w
        adj[b][a] = adj[b].get(a, 0) + w
    return adj


def _reroute_severity(adj, cell):
    """How much longer (in data-graph hops) the demonstrated routes between this
    cell's neighbors become when the cell is removed. Disconnection -> large."""
    nbs = list(adj[cell].keys())
    if len(nbs) < 2:
        return 0.0
    # dijkstra (hop count) from nbs[0] with `cell` banned
    src = nbs[0]
    dist = {src: 0}
    pq = [(0, src)]
    while pq:
        dc, c = heapq.heappop(pq)
        if dc > dist.get(c, 1e9):
            continue
        for nb in adj[c]:
            if nb == cell:
                continue
            nd = dc + 1
            if nd < dist.get(nb, 1e9):
                dist[nb] = nd
                heapq.heappush(pq, (nd, nb))
    incs = []
    for nb in nbs[1:]:
        d = dist.get(nb, None)
        incs.append(10.0 if d is None else max(d - 2, 0))
    return sum(incs) / len(incs)


def _compute_max_disruption(model: DataModel, start, goal, n_ref=5) -> float:
    """Block the top n_ref highest-severity cells (keeping maze reachable) to
    define the reference maximum. 1.0 = disruption of the n_ref worst cells."""
    ranked = sorted(model.traffic.visited_cells,
                    key=lambda c: -model.severity_norm.get(c, 0.0))
    grid = model.base_grid.copy()
    blocked = 0
    for c in ranked:
        if c in (start, goal):
            continue
        grid[c] = 1
        if shortest_path(grid, start, goal) is None:
            grid[c] = 0
        else:
            blocked += 1
            if blocked >= n_ref:
                break
    return _raw_disruption(model, grid)


def _raw_disruption(model: DataModel, new_grid) -> float:
    """Compute data_disruption without normalizing (used during calibration)."""
    tm = model.traffic
    new_grid = np.asarray(new_grid)
    invalid = sum(w for (a, b), w in tm.transitions.items()
                  if dd._transition_invalid(a, b, new_grid))
    collision = invalid / max(tm.total_transitions, 1)
    broken = [c for c in tm.visited_cells if new_grid[c] == 1]
    coverage = len(broken) / max(len(tm.visited_cells), 1)
    severity = float(min(sum(model.severity_norm.get(c, 0.0) for c in broken), 1.0))
    return float(min(max(
        WEIGHTS["severity"] * severity
        + WEIGHTS["collision"] * collision
        + WEIGHTS["coverage"] * coverage, 0.0), 1.0))


def build_data_model(data_dir, maze_type, subsample=1,
                     task="task3") -> DataModel:
    traffic = dd.build_traffic_model(data_dir, subsample=subsample)
    adj = _build_adj(traffic)
    base = np.array(MAZE_MAPS[maze_type])
    cells = list(traffic.cell_visits.keys())
    sev = {c: _reroute_severity(adj, c) for c in cells}
    max_traffic = max(traffic.cell_visits.values())
    # combine severity with traffic, then normalize to [0,1]
    raw = {c: sev[c] * traffic.cell_visits[c] for c in cells}
    mx = max(raw.values()) or 1.0
    sev_norm = {c: raw[c] / mx for c in cells}
    model = DataModel(traffic, adj, sev, sev_norm, max_traffic, base)
    # calibrate: find the max achievable score while keeping goal reachable
    start, goal = MAZE_TASKS[maze_type][task]
    model.max_disruption = max(_compute_max_disruption(model, start, goal), 1e-6)
    print(f"  [calibration] max reachable disruption = {model.max_disruption:.3f}")
    return model


# ===========================================================================
# Scoring an edited maze against the data distribution
# ===========================================================================

@dataclass
class DataScore:
    reroute_severity: float = 0.0    # traffic+detour weighted, normalized
    collision_fraction: float = 0.0  # share of demonstrated transitions invalid
    coverage_loss: float = 0.0       # share of visited cells now walled
    data_disruption: float = 0.0
    n_invalid: int = 0
    n_total: int = 0
    broken_cells: list = field(default_factory=list)
    label: str = ""


WEIGHTS = {"severity": 0.50, "collision": 0.35, "coverage": 0.15}


def score(model: DataModel, new_grid) -> DataScore:
    new_grid = np.asarray(new_grid)
    tm = model.traffic
    s = DataScore()
    s.n_total = tm.total_transitions

    # collision fraction (traffic-weighted invalid transitions)
    invalid = 0
    for (a, b), w in tm.transitions.items():
        if dd._transition_invalid(a, b, new_grid):
            invalid += w
    s.n_invalid = invalid
    s.collision_fraction = invalid / max(tm.total_transitions, 1)

    # broken cells = visited cells that became walls
    broken = [c for c in tm.visited_cells if new_grid[c] == 1]
    s.broken_cells = [list(map(int, c)) for c in broken]
    s.coverage_loss = len(broken) / max(len(tm.visited_cells), 1)

    # reroute severity = sum of normalized per-cell severities of broken cells,
    # capped at 1 (so breaking the single worst bottleneck ~ saturates).
    s.reroute_severity = float(min(sum(model.severity_norm.get(c, 0.0) for c in broken), 1.0))

    raw = (WEIGHTS["severity"] * s.reroute_severity
           + WEIGHTS["collision"] * s.collision_fraction
           + WEIGHTS["coverage"] * s.coverage_loss)
    s.data_disruption = float(min(max(raw / model.max_disruption, 0.0), 1.0))

    v = s.data_disruption
    s.label = ("minimal" if v < 0.10 else "minor" if v < 0.30 else
               "moderate" if v < 0.55 else "major" if v < 0.80 else "severe")
    return s


# ===========================================================================
# Variant generation: target a level, with randomness -> multiple variants
# ===========================================================================

def _interior_walls(grid):
    H, W = grid.shape
    return [(i, j) for i in range(1, H - 1) for j in range(1, W - 1) if grid[i, j] == 1]


def _goal_reachable(grid, start, goal):
    return shortest_path(grid, start, goal) is not None


def k_shortest_paths(grid, start, goal, k=5):
    """Return up to k shortest paths from start to goal using BFS.
    Each node may be visited up to k times to allow diverging routes."""
    from collections import defaultdict
    heap = [(0, [start])]
    count = defaultdict(int)
    paths = []
    while heap and len(paths) < k:
        cost, path = heapq.heappop(heap)
        node = path[-1]
        count[node] += 1
        if count[node] > k:
            continue
        if node == goal:
            paths.append(path)
            continue
        for nb in _neighbors(node, grid):
            if count[nb] <= k:
                heapq.heappush(heap, (cost + 1, path + [nb]))
    return paths


def _random_edit(model, start, goal, target=0.0, n_open_max=6):
    """One random edit: block cells sampled with severity-weighted probability so
    that higher targets preferentially hit bottleneck cells rather than just
    blocking more cells. Optionally opens some walls."""
    base = model.base_grid
    new = base.copy()
    visited = [c for c in model.traffic.visited_cells if c not in (start, goal)]
    int_walls = _interior_walls(base)

    # Exponent scales with target: low target ≈ uniform, high target ≈ bottleneck-only
    exponent = 1.0 + target * 3.0
    weights = np.array([model.severity_norm.get(c, 0.0) ** exponent for c in visited])
    if weights.sum() == 0:
        weights = np.ones(len(visited))
    weights /= weights.sum()

    # target=0.1 -> max 2, target=0.6 -> max 4; position drives difficulty, not quantity
    n_block_max = max(1, round(1 + target * 4))
    n_block = random.randint(1, n_block_max)
    indices = np.random.choice(len(visited), size=min(n_block, len(visited)), replace=False, p=weights)
    for i in indices:
        c = visited[i]
        new[c] = 1
        if not _goal_reachable(new, start, goal):
            new[c] = 0  # revert — never disconnect the maze

    # Open walls with probability that decreases with target: high target = fewer shortcuts
    if random.random() < (1.0 - target) and int_walls:
        n_open = random.randint(1, n_open_max)
        # Bias toward low-traffic walls so we don't accidentally reduce disruption
        open_weights = np.array([1.0 / (1.0 + model.traffic.cell_visits.get(c, 0)) for c in int_walls])
        open_weights /= open_weights.sum()
        indices = np.random.choice(len(int_walls), size=min(n_open, len(int_walls)), replace=False, p=open_weights)
        for i in indices:
            new[int_walls[i]] = 0
    return new


def generate_variants_at_level(model, task_start, task_goal, target,
                               n_variants=3, tol=0.06,
                               allow_unreachable=True):
    """Keep sampling until exactly `n_variants` DISTINCT mazes scoring near `target` are found."""
    found = []
    seen = set()
    attempts = 0

    while len(found) < n_variants:
        new = _random_edit(model, task_start, task_goal, target)
        attempts += 1

        reachable = _goal_reachable(new, task_start, task_goal)
        if not reachable and not allow_unreachable:
            continue

        sc = score(model, new)
        if attempts % 500 == 0:
            print(f"  target={target:.1f} | attempts={attempts} | found={len(found)}/{n_variants} "
                  f"| last_score={sc.data_disruption:.3f}", flush=True)

        if abs(sc.data_disruption - target) <= tol:
            key = new.tobytes()
            if key in seen:
                continue
            seen.add(key)
            found.append((new, sc, reachable))
            print(f"  target={target:.1f} | found {len(found)}/{n_variants} "
                  f"(score={sc.data_disruption:.3f}, attempts={attempts})", flush=True)

    return found


# ===========================================================================
# Suite + output
# ===========================================================================

@dataclass
class Record:
    maze_type: str
    task: str
    start: tuple
    goal: tuple
    start_xy: tuple
    goal_xy: tuple
    target_level: float
    variant_index: int
    data_disruption: float
    components: dict
    collision_fraction: float
    reroute_severity: float
    coverage_loss: float
    goal_reachable: bool
    broken_cells: list
    maze_map: list


def generate_suite(data_dir, maze_type="giant", task="task3",
                   levels=None, variants_per_level=3, subsample=1):
    if levels is None:
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    model = build_data_model(data_dir, maze_type, subsample=subsample, task=task)
    start, goal = MAZE_TASKS[maze_type][task]
    records = []
    for lvl in levels:
        print(f"\n[level {lvl}]", flush=True)
        hits = generate_variants_at_level(
            model, start, goal, lvl,
            n_variants=variants_per_level)
        for vi, (new, sc, reachable) in enumerate(hits):
            records.append(Record(
                maze_type=maze_type, task=task, start=start, goal=goal,
                start_xy=ij_to_xy(start), goal_xy=ij_to_xy(goal),
                target_level=lvl, variant_index=vi,
                data_disruption=round(sc.data_disruption, 4),
                components={
                    "reroute_severity": round(sc.reroute_severity, 4),
                    "collision_fraction": round(sc.collision_fraction, 4),
                    "coverage_loss": round(sc.coverage_loss, 4),
                },
                collision_fraction=round(sc.collision_fraction, 4),
                reroute_severity=round(sc.reroute_severity, 4),
                coverage_loss=round(sc.coverage_loss, 4),
                goal_reachable=reachable,
                broken_cells=sc.broken_cells,
                maze_map=new.tolist(),
            ))
    return records, model


# ===========================================================================
# Visualization
# ===========================================================================

def plot_suite(records, model, outpath, maze_type, task):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = model.base_grid
    tm = model.traffic
    # group by level
    by_level = {}
    for r in records:
        by_level.setdefault(r.target_level, []).append(r)
    levels = sorted(by_level)
    n_var = max(len(v) for v in by_level.values())

    rows = len(levels)
    cols = 2 + n_var  # col 0: original, col 1: traffic heatmap, rest: variants
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows))
    axes = np.atleast_2d(axes)
    if axes.shape != (rows, cols):
        axes = axes.reshape(rows, cols)

    # traffic heatmap (log scale)
    H, W = base.shape
    heat = np.zeros((H, W))
    for (i, j), c in tm.cell_visits.items():
        heat[i, j] = np.log1p(c)
    # mask walls so they show as grey, not white
    heat_display = np.ma.masked_where(base == 1, heat)

    start, goal = records[0].start, records[0].goal
    orig_paths = k_shortest_paths(base, start, goal, k=5)

    def _mark_start_goal(ax):
        ax.scatter([start[1]], [start[0]], c="limegreen", s=120, marker="*",
                   edgecolors="k", zorder=5)
        ax.scatter([goal[1]], [goal[0]], c="gold", s=120, marker="X",
                   edgecolors="k", zorder=5)

    def _draw_paths(ax, paths, color, lw=1.8, base_alpha=0.8):
        for i, path in enumerate(paths):
            if path and len(path) > 1:
                alpha = base_alpha * (0.9 ** i)  # each subsequent path slightly more transparent
                ax.plot([p[1] for p in path], [p[0] for p in path],
                        color=color, lw=lw, alpha=alpha, zorder=4)

    for ri, lvl in enumerate(levels):
        # --- col 0: original maze + original path ---
        ax0 = axes[ri, 0]
        ax0.set_xticks([]); ax0.set_yticks([])
        ax0.imshow(base, cmap="binary", origin="upper")
        _draw_paths(ax0, orig_paths, color="royalblue")
        _mark_start_goal(ax0)
        if ri == 0:
            ax0.set_title("original", fontsize=8)

        # --- col 1: traffic heatmap ---
        ax1 = axes[ri, 1]
        ax1.set_xticks([]); ax1.set_yticks([])
        ax1.imshow(base, cmap="binary", origin="upper", alpha=0.3)
        ax1.imshow(heat_display, cmap="hot", origin="upper", alpha=0.85)
        _mark_start_goal(ax1)
        if ri == 0:
            ax1.set_title("traffic", fontsize=8)

        # --- cols 2+: generated variants ---
        level_recs = by_level[lvl]
        for ci in range(n_var):
            ax = axes[ri, 2 + ci]
            ax.set_xticks([]); ax.set_yticks([])
            if ci >= len(level_recs):
                ax.axis("off"); continue
            r = level_recs[ci]
            new = np.array(r.maze_map)
            ax.imshow(new, cmap="binary", origin="upper")
            for (i, j) in r.broken_cells:
                t = tm.cell_visits.get((i, j), 0)
                ax.scatter([j], [i], marker="s", s=120,
                           c=[[1, 0.2, 0.2]], alpha=0.4 + 0.6 * t / model.max_cell_traffic,
                           edgecolors="darkred")
            opened = list(zip(*np.where((new == 0) & (base == 1))))
            if opened:
                ax.scatter([j for _, j in opened], [i for i, _ in opened],
                           marker="+", s=60, c="dodgerblue")
            # original paths (blue, faint) vs new paths (orange)
            _draw_paths(ax, orig_paths, color="royalblue", lw=1.2, base_alpha=0.35)
            new_paths = k_shortest_paths(new, start, goal, k=5)
            _draw_paths(ax, new_paths, color="darkorange", lw=1.8)
            _mark_start_goal(ax)
            tag = "reachable" if r.goal_reachable else "UNREACHABLE"
            ax.set_title(f"L{lvl:.1f} v{r.variant_index} | {r.data_disruption:.2f}\n"
                         f"sev {r.reroute_severity:.2f} col {r.collision_fraction:.2f} | {tag}",
                         fontsize=8)

    fig.suptitle(f"Data-distribution OOD variants: pointmaze-{maze_type} {task}\n"
                 f"blue=original top-5 paths  orange=new top-5 paths  red sq=broken cell  * start  X goal",
                 fontsize=12, y=1.002)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return outpath


if __name__ == "__main__":
    maze_type = "giant"
    task = "task5"
    seed = 1

    set_seed(seed)
    DATA = f"../pointmaze/ogbench/data/pointmaze-{maze_type}-navigate-v0"
    recs, model = generate_suite(DATA, maze_type=maze_type, task=task,
                                 levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                 variants_per_level=3)

    print(f"\nData model: {model.traffic.n_episodes} episodes, "
          f"{model.traffic.total_transitions} transitions, "
          f"{len(model.traffic.visited_cells)} cells")
    print(f"\n{'level':>6} {'var':>4} {'score':>7} {'reroute':>8} {'collision':>10} {'cover':>7} reachable")
    for r in recs:
        print(f"{r.target_level:>6.1f} {r.variant_index:>4} {r.data_disruption:>7.3f} "
              f"{r.reroute_severity:>8.3f} {r.collision_fraction:>10.3f} "
              f"{r.coverage_loss:>7.3f} {r.goal_reachable}")

    os.makedirs("maze_variants", exist_ok=True)
    outpath = f"maze_variants/{maze_type}_{task}.png"
    plot_suite(recs, model, outpath, maze_type, task)
    print(f"\nSaved plot to {outpath}")

    jsonpath = f"maze_variants/{maze_type}_{task}.json"
    with open(jsonpath, "w") as f:
        json.dump({
            "maze_type": maze_type,
            "task": task,
            "seed": seed,
            "base_maze": model.base_grid.tolist(),
            "max_disruption": model.max_disruption,
            "variants": [asdict(r) for r in recs],
        }, f, indent=2)
    print(f"Saved data  to {jsonpath}")
