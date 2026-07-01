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
@dataclass
class DifficultyWeights:
    w_sp: float = 0.40          # start->goal shortest-path elongation
    w_goalfield: float = 0.25   # global distance-to-goal landscape shift
    w_novelty: float = 0.25     # corridor OOD (near-zero training traffic)
    w_deadend: float = 0.10     # increase in dead-end pockets
    # saturating scales (map raw quantity -> [0,1] via 1 - exp(-x/scale) or ratio caps)
    sp_ratio_cap: float = 2.0   # sp_ratio of sp_ratio_cap -> full score
    goalfield_scale: float = 4.0
    deadend_scale: float = 0.15
    novelty_traffic_eps: float = 1e-5  # a cell is 'novel' if traffic below this


@dataclass
class DifficultyResult:
    difficulty: float
    reachable: bool
    components: dict = field(default_factory=dict)
    structural: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def _sat(x: float, scale: float) -> float:
    """Saturating map [0,inf) -> [0,1)."""
    if not np.isfinite(x):
        return 1.0
    return float(1.0 - np.exp(-max(0.0, x) / scale))


def compute_difficulty(
    base_grid: np.ndarray,
    var_grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    traffic: Optional[np.ndarray] = None,
    weights: DifficultyWeights = DifficultyWeights(),
) -> DifficultyResult:
    """
    Compute planner-difficulty of `var_grid` for task (start, goal) relative to
    `base_grid`. Higher = harder for a map-blind diffusion planner. A variant
    whose goal is unreachable saturates to 1.0.
    """
    base_grid = np.asarray(base_grid)
    var_grid = np.asarray(var_grid)

    sp_base = shortest_path_len(base_grid, start, goal)
    sp_var = shortest_path_len(var_grid, start, goal)

    comp: dict = {}
    structural = structural_distance(base_grid, var_grid)

    if not np.isfinite(sp_var):
        comp.update(dict(sp_ratio=float("inf"), sp_term=1.0, goalfield_shift=float("inf"),
                         goalfield_term=1.0, corridor_novelty=1.0, novelty_term=1.0,
                         deadend_delta=0.0, deadend_term=0.0, sp_base=sp_base, sp_var=float("inf")))
        return DifficultyResult(difficulty=1.0, reachable=False, components=comp, structural=structural)

    # --- (1) shortest-path elongation ---
    sp_ratio = sp_var / sp_base if sp_base > 0 else 1.0
    # only elongation counts; a shortcut (ratio<1) does NOT make it harder
    sp_term = _sat(max(0.0, sp_ratio - 1.0), scale=(weights.sp_ratio_cap - 1.0))
    comp["sp_ratio"] = sp_ratio
    comp["sp_term"] = sp_term

    # --- (2) goal-distance field shift over cells free in both ---
    Db = bfs_distance_field(base_grid, goal)
    Dv = bfs_distance_field(var_grid, goal)
    both = (base_grid == 0) & (var_grid == 0) & np.isfinite(Db) & np.isfinite(Dv)
    goalfield_shift = float(np.abs(Dv - Db)[both].mean()) if both.any() else 0.0
    goalfield_term = _sat(goalfield_shift, scale=weights.goalfield_scale)
    comp["goalfield_shift"] = goalfield_shift
    comp["goalfield_term"] = goalfield_term

    # --- (3) corridor novelty: new optimal path cells with ~zero training traffic ---
    path = optimal_path_cells(var_grid, start, goal)
    if traffic is not None and len(path) > 0:
        novel = sum(1 for (i, j) in path if traffic[i, j] <= weights.novelty_traffic_eps)
        corridor_novelty = novel / len(path)
    else:
        corridor_novelty = 0.0
    comp["corridor_novelty"] = corridor_novelty

    # --- (4) dead-end pressure: growth in dead-end pocket fraction ---
    de_base = deadend_score(base_grid)
    de_var = deadend_score(var_grid)
    deadend_delta = max(0.0, de_var - de_base)
    deadend_term = _sat(deadend_delta, scale=weights.deadend_scale)
    comp["deadend_delta"] = deadend_delta
    comp["deadend_term"] = deadend_term
    comp["sp_base"] = sp_base
    comp["sp_var"] = sp_var

    difficulty = (
        weights.w_sp * sp_term
        + weights.w_goalfield * goalfield_term
        + weights.w_novelty * corridor_novelty
        + weights.w_deadend * deadend_term
    )
    difficulty = float(min(1.0, difficulty))
    return DifficultyResult(difficulty=difficulty, reachable=True, components=comp, structural=structural)
