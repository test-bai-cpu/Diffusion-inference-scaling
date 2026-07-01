"""
difficulty_metric.py
====================

Calibrated, data-aware difficulty metric for maze variants.

Motivation
----------
The previous generator graded variants by the Reroute Disruption Index (RDI),
a *symmetric graph-edit* style score (how many diverse routes an edit breaks,
spatial detour, traffic on broken routes). On the 60 logged DFS runs, RDI
rank-correlates with measured success at only Spearman rho = -0.295, with 20%
of level-pairs inverted (a "harder" level solved more often than an "easier"
one). The reason: a diffusion planner trained on ONE map fails from
*distributional shift* and *task-relative geometry*, not from how many training
routes an edit happens to intersect.

This module separates two axes that RDI conflated:

  1. PLANNER DIFFICULTY (asymmetric, task- and data-relative) -- predicts
     whether the diffusion planner will struggle. Built from:
       * sp_ratio        : new_shortest_path(start->goal) / old_shortest_path.
                           Must be >= 1 to be "harder"; < 1 means the edit
                           actually opened a shortcut (easier).
       * goalfield_shift : mean |D_var(cell) - D_base(cell)| over cells free in
                           both, where D is the BFS distance-to-goal field. How
                           much the whole distance-to-goal landscape moved --
                           exactly the field the guidance method descends on.
       * corridor_novelty: fraction of the NEW optimal path's cells that carry
                           near-zero *training* traffic. The out-of-distribution
                           term -- the planner has never seen these corridors.
       * deadend_pressure: growth in number/severity of dead-end pockets the
                           agent can wander into near the optimal corridor.

  2. STRUCTURAL NOVELTY (symmetric, direction-blind) -- "how different does the
     maze look", reported SEPARATELY (not used to bin difficulty). Built from
     spectral (Laplacian eigenvalue) distance and edge-edit (Jaccard) distance.
     Use this to make the stronger paper claim: a variant is both structurally
     distant AND harder.

Coordinate convention (verified against ogbench/locomaze/maze.py and the
variant JSONs): world xy = (j*maze_unit - offset, i*maze_unit - offset) with
maze_unit = 4, offset = 4. So cell (i, j): x = j*4 - 4, y = i*4 - 4.

Grid convention: 0 = free, 1 = wall (ogbench).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
from scipy.stats import rankdata

try:
    import networkx as nx
    _HAVE_NX = True
except Exception:  # pragma: no cover
    _HAVE_NX = False

MAZE_UNIT = 4.0
OFFSET = 4.0


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------
def ij_to_xy(i: int, j: int) -> tuple[float, float]:
    return (j * MAZE_UNIT - OFFSET, i * MAZE_UNIT - OFFSET)


def xy_to_ij(x: float, y: float) -> tuple[int, int]:
    j = int(round((x + OFFSET) / MAZE_UNIT))
    i = int(round((y + OFFSET) / MAZE_UNIT))
    return i, j


# ---------------------------------------------------------------------------
# Graph / geodesic primitives (4-connectivity -- diagonal is NOT traversable)
# ---------------------------------------------------------------------------
_N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def bfs_distance_field(grid: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    """BFS distance in cells from `source` over free cells. inf for wall/unreachable."""
    H, W = grid.shape
    D = np.full((H, W), np.inf, dtype=np.float64)
    si, sj = source
    if grid[si, sj] != 0:
        return D
    D[si, sj] = 0.0
    q = deque([source])
    while q:
        i, j = q.popleft()
        for di, dj in _N4:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni, nj] == 0 and np.isinf(D[ni, nj]):
                D[ni, nj] = D[i, j] + 1.0
                q.append((ni, nj))
    return D


def shortest_path_len(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> float:
    return float(bfs_distance_field(grid, goal)[start[0], start[1]])


def optimal_path_cells(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """One shortest path start->goal as a list of cells (greedy descent on the goal field)."""
    D = bfs_distance_field(grid, goal)
    if np.isinf(D[start[0], start[1]]):
        return []
    path = [start]
    cur = start
    H, W = grid.shape
    while cur != goal:
        ci, cj = cur
        best = None
        best_d = D[ci, cj]
        for di, dj in _N4:
            ni, nj = ci + di, cj + dj
            if 0 <= ni < H and 0 <= nj < W and D[ni, nj] < best_d:
                best_d = D[ni, nj]
                best = (ni, nj)
        if best is None:
            break
        path.append(best)
        cur = best
    return path


def grid_to_graph(grid: np.ndarray) -> "nx.Graph":
    H, W = grid.shape
    G = nx.Graph()
    for i in range(H):
        for j in range(W):
            if grid[i, j] == 0:
                G.add_node((i, j))
                for di, dj in ((1, 0), (0, 1)):
                    ni, nj = i + di, j + dj
                    if ni < H and nj < W and grid[ni, nj] == 0:
                        G.add_edge((i, j), (ni, nj))
    return G


# ---------------------------------------------------------------------------
# Training traffic (per-cell visitation from the OGBench dataset)
# ---------------------------------------------------------------------------
def traffic_map_from_observations(observations: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """
    Per-cell normalized visitation frequency from demonstrated (x, y) observations.
    Returns array of shape grid_shape summing to 1 over free cells.
    """
    H, W = grid_shape
    counts = np.zeros((H, W), dtype=np.float64)
    for x, y in observations[:, :2]:
        i, j = xy_to_ij(float(x), float(y))
        if 0 <= i < H and 0 <= j < W:
            counts[i, j] += 1.0
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


# ---------------------------------------------------------------------------
# Dead-end pockets (structural, connectivity-aware)
# ---------------------------------------------------------------------------
def deadend_score(grid: np.ndarray) -> float:
    """
    Fraction of free cells that are 'dead-end-like': free cells with <= 1 free
    4-neighbor (a pocket the agent can enter and must reverse out of).
    """
    H, W = grid.shape
    free = grid == 0
    n_free = int(free.sum())
    if n_free == 0:
        return 0.0
    deg = np.zeros((H, W), dtype=np.int32)
    for di, dj in _N4:
        shifted = np.zeros_like(free)
        si0, si1 = max(0, di), H + min(0, di)
        sj0, sj1 = max(0, dj), W + min(0, dj)
        ti0, ti1 = max(0, -di), H + min(0, -di)
        tj0, tj1 = max(0, -dj), W + min(0, -dj)
        shifted[ti0:ti1, tj0:tj1] = free[si0:si1, sj0:sj1]
        deg += (free & shifted).astype(np.int32)
    dead = free & (deg <= 1)
    return float(dead.sum()) / float(n_free)


# ---------------------------------------------------------------------------
# Structural novelty (symmetric) -- reported separately, NOT used to bin levels
# ---------------------------------------------------------------------------
def structural_distance(base_grid: np.ndarray, var_grid: np.ndarray) -> dict:
    out: dict = {}
    fb, fv = base_grid == 0, var_grid == 0
    inter = np.logical_and(fb, fv).sum()
    union = np.logical_or(fb, fv).sum()
    out["cell_jaccard_dist"] = float(1.0 - inter / union) if union else 0.0
    if _HAVE_NX:
        Gb, Gv = grid_to_graph(base_grid), grid_to_graph(var_grid)
        Eb, Ev = set(Gb.edges()), set(Gv.edges())
        u = len(Eb | Ev)
        out["edge_jaccard_dist"] = float(1.0 - len(Eb & Ev) / u) if u else 0.0
        try:
            lb = np.sort(nx.laplacian_spectrum(Gb))
            lv = np.sort(nx.laplacian_spectrum(Gv))
            k = min(len(lb), len(lv))
            # normalize by sqrt(k) so it is size-comparable across maps
            out["spectral_dist"] = float(np.linalg.norm(lb[:k] - lv[:k]) / np.sqrt(k))
        except Exception:
            out["spectral_dist"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# The planner-difficulty metric
# ---------------------------------------------------------------------------
# CALIBRATION RESULT (60 logged DFS runs on the giant maze):
#   RANK composite over GOOD_FEATURES ......... Spearman rho = -0.531 vs success
#   RDI (old metric) .......................... Spearman rho = -0.374
#   nominal target_level ...................... Spearman rho = -0.295
# The authoritative difficulty score is the DATA-RELATIVE RANK COMPOSITE
# (`rank_composite`): each feature is turned into a population percentile (0..1),
# then averaged over GOOD_FEATURES. Ranking is what removes the scale/units
# problem and is what was calibrated. `compute_variant_features` produces the raw
# directional features for one variant; `rank_composite` combines a BATCH of them.
#
# All difficulty features are DIRECTIONAL: only changes that make the task harder
# (goal farther, path longer, more OOD corridor, more dead-ends) push the score
# up. The earlier non-directional goalfield_shift / corridor_novelty flipped sign
# because the old generator opened and blocked walls simultaneously (a variant
# could be structurally very different yet net-EASIER). Directional net_shift and
# novelty x elongation fix that.

# Features whose equal-weight rank composite achieved rho = -0.531.
GOOD_FEATURES = ("spectral", "sp_ratio", "net_shift", "deadend_delta",
                 "novelty_x_elong", "n_blocked")


@dataclass
class DifficultyResult:
    difficulty: float          # NaN for a single call w/o population; use rank_composite
    reachable: bool
    features: dict = field(default_factory=dict)
    structural: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def compute_variant_features(
    base_grid: np.ndarray,
    var_grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    traffic: Optional[np.ndarray] = None,
    novelty_traffic_eps: float = 1e-5,
) -> DifficultyResult:
    """
    Raw DIRECTIONAL planner-difficulty features for one variant, relative to base.

    Returns a DifficultyResult whose `features` dict feeds `rank_composite`.
    `difficulty` is left NaN here because the calibrated score is a population
    percentile -- call `rank_composite` over a batch of these to get [0,1] scores.
    A variant with an unreachable goal is flagged reachable=False and given
    saturating (maximally hard) feature values.
    """
    base_grid = np.asarray(base_grid)
    var_grid = np.asarray(var_grid)

    Db = bfs_distance_field(base_grid, goal)
    Dv = bfs_distance_field(var_grid, goal)
    sp_base = float(Db[start[0], start[1]])
    sp_var = float(Dv[start[0], start[1]])
    reachable = np.isfinite(sp_var)

    structural = structural_distance(base_grid, var_grid)
    f: dict = {}
    f["spectral"] = structural.get("spectral_dist", np.nan)
    f["edge_jac"] = structural.get("edge_jaccard_dist", np.nan)
    f["n_blocked"] = int(((base_grid == 0) & (var_grid == 1)).sum())
    f["n_opened"] = int(((base_grid == 1) & (var_grid == 0)).sum())
    f["net_walls"] = f["n_blocked"] - f["n_opened"]

    if not reachable:
        f.update(dict(sp_ratio=3.0, sp_elong=2.0, farther_mean=np.inf, closer_mean=0.0,
                      net_shift=np.inf, corridor_novelty=1.0, novelty_x_elong=2.0,
                      deadend_delta=max(0.0, deadend_score(var_grid) - deadend_score(base_grid)),
                      sp_base=sp_base, sp_var=np.inf))
        return DifficultyResult(difficulty=float("nan"), reachable=False,
                                features=f, structural=structural)

    # (1) shortest-path elongation start->goal (>=1 harder; <1 is a shortcut = easier)
    f["sp_ratio"] = (sp_var / sp_base) if sp_base > 0 else 1.0
    f["sp_elong"] = max(0.0, f["sp_ratio"] - 1.0)

    # (2) DIRECTIONAL goal-distance field shift over cells free in both maps.
    both = (base_grid == 0) & (var_grid == 0) & np.isfinite(Db) & np.isfinite(Dv)
    dshift = np.where(both, Dv - Db, 0.0)
    farther = np.where(both, np.maximum(0.0, dshift), 0.0)   # got farther from goal
    closer = np.where(both, np.maximum(0.0, -dshift), 0.0)   # got closer (easier)
    f["farther_mean"] = float(farther[both].mean()) if both.any() else 0.0
    f["closer_mean"] = float(closer[both].mean()) if both.any() else 0.0
    f["net_shift"] = f["farther_mean"] - f["closer_mean"]

    # (3) corridor novelty on the NEW optimal path, GATED by elongation so that
    #     a novel-but-shorter reroute does not read as harder.
    path = optimal_path_cells(var_grid, start, goal)
    if traffic is not None and len(path) > 0:
        nov = float(np.mean([1.0 if traffic[i, j] <= novelty_traffic_eps else 0.0
                             for (i, j) in path]))
    else:
        nov = 0.0
    f["corridor_novelty"] = nov
    f["novelty_x_elong"] = nov * f["sp_elong"]

    # (4) dead-end pressure: growth in dead-end pocket fraction near the maze.
    f["deadend_delta"] = max(0.0, deadend_score(var_grid) - deadend_score(base_grid))

    f["sp_base"] = sp_base
    f["sp_var"] = sp_var
    return DifficultyResult(difficulty=float("nan"), reachable=True,
                            features=f, structural=structural)


def _percentile_rank(x: np.ndarray) -> np.ndarray:
    """Map values to population percentile in [0,1]; NaNs stay NaN."""
    x = np.asarray(x, dtype=float)
    m = np.isfinite(x)
    r = np.full_like(x, np.nan)
    n = int(m.sum())
    if n > 1:
        r[m] = (rankdata(x[m]) - 1) / (n - 1)
    elif n == 1:
        r[m] = 0.5
    return r


def rank_composite(feature_dicts, keys=GOOD_FEATURES) -> np.ndarray:
    """
    Calibrated data-relative difficulty for a BATCH of variants.

    feature_dicts : sequence of `DifficultyResult.features` dicts (or DifficultyResult).
    Returns an array of difficulty scores in [0,1] (population percentile mean over
    `keys`). Higher = harder. This is the score to bin into quantile levels.
    """
    dicts = [d.features if isinstance(d, DifficultyResult) else d for d in feature_dicts]
    cols = []
    for k in keys:
        vals = np.array([float(d.get(k, np.nan)) for d in dicts], dtype=float)
        cols.append(_percentile_rank(vals))
    mat = np.column_stack(cols) if cols else np.zeros((len(dicts), 0))
    with np.errstate(invalid="ignore"):
        return np.nanmean(mat, axis=1)


def quantile_levels(scores: np.ndarray, k: int = 4) -> np.ndarray:
    """Bin difficulty scores into k equal-population levels 0..k-1 (0 = easiest)."""
    scores = np.asarray(scores, dtype=float)
    q = np.quantile(scores[np.isfinite(scores)], np.linspace(0, 1, k + 1))
    return np.clip(np.digitize(scores, q[1:-1]), 0, k - 1)
