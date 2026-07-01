"""
distance_field_v2.py
====================
Corrected distance-to-goal field + dead-end detector.

Bugs in distance_field.smooth_distance_field this replaces
----------------------------------------------------------
  1. CROSS-WALL BLUR. `gaussian_filter` mixes values across thin (1-cell) walls.
     Two free regions on opposite sides of a wall have very different geodesic
     distances-to-goal; blurring them together produces a field whose gradient
     points straight THROUGH the wall toward the goal -- a shortcut the maze does
     not have (the distance-field analogue of corner-cutting).
  2. inf_replace=1000 LEAK. The 1000-fill for walls/unreachable cells bleeds into
     neighbouring free cells under the blur, corrupting nearby gradients.
  3. 8-CONNECTIVITY option. Diagonal moves are NOT traversable here, so BFS must
     be 4-connected; an 8-connected field under-estimates true travel distance.

Fixes
-----
  * `masked_diffusion_smooth`: smoothing by Jacobi diffusion that only averages a
    free cell with its 4-connected FREE neighbours. No value ever crosses a wall,
    so corridor gradients stay smooth while wall barriers stay sharp.
  * `fill_walls_monotone`: wall cells are filled to be >= their free neighbours
    (nearest-free value + penalty, grown inward), so the sampled gradient always
    points AWAY from walls, never into them.
  * BFS pinned to 4-connectivity.
  * Dead-end detection:
      - `deadend_map`: free cells in dead-end pockets (iterative leaf-prune of
        degree-1 free cells that are neither start nor goal), i.e. tendrils the
        agent can wander into and must back out of.
      - `stalled_progress`: a per-trajectory scalar (mean positive goal-distance
        increment) for DFS to trigger backtracking when a rollout stops making
        progress or reverses -- the dead-end trapping failure mode.
"""
from __future__ import annotations
import numpy as np
from collections import deque

try:
    import torch
    import torch.nn.functional as F
    _HAVE_TORCH = True
except Exception:
    _HAVE_TORCH = False

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --------------------------------------------------------------------------- #
# BFS distance-to-goal (4-connected)
# --------------------------------------------------------------------------- #
def compute_distance_field(maze_grid: np.ndarray, goal_cell: tuple) -> np.ndarray:
    """4-connected BFS distance in cells from goal over free cells (0=free,1=wall).
    Walls and unreachable free cells are inf."""
    H, W = maze_grid.shape
    D = np.full((H, W), np.inf, dtype=np.float32)
    gi, gj = goal_cell
    if maze_grid[gi, gj] == 1:
        return D
    D[gi, gj] = 0.0
    q = deque([(gi, gj)])
    while q:
        i, j = q.popleft()
        for di, dj in _N4:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and maze_grid[ni, nj] == 0 and np.isinf(D[ni, nj]):
                D[ni, nj] = D[i, j] + 1.0
                q.append((ni, nj))
    return D


# --------------------------------------------------------------------------- #
# wall-respecting smoothing
# --------------------------------------------------------------------------- #
def masked_diffusion_smooth(D: np.ndarray, maze_grid: np.ndarray,
                            iters: int = 8, alpha: float = 0.5,
                            unreachable_fill: float = None) -> np.ndarray:
    """
    Smooth the distance field WITHOUT crossing walls.

    Each free cell relaxes toward the mean of its 4-connected FREE neighbours:
        D <- (1-alpha)*D + alpha * mean_free_neighbours(D)
    Walls never contribute and are never updated here, so no value crosses a
    wall. Unreachable free cells (inf) are set to `unreachable_fill` (default:
    max finite distance + 1) BEFORE smoothing so they read as far-but-finite.
    """
    H, W = D.shape
    free = (maze_grid == 0)
    out = D.copy().astype(np.float64)
    finite = np.isfinite(out) & free
    max_fin = out[finite].max() if finite.any() else 1.0
    if unreachable_fill is None:
        unreachable_fill = max_fin + 1.0
    unreach = free & ~np.isfinite(out)
    out[unreach] = unreachable_fill
    out[~free] = np.nan  # exclude walls from the neighbour mean

    for _ in range(iters):
        acc = np.zeros((H, W)); cnt = np.zeros((H, W))
        for di, dj in _N4:
            sh = np.full((H, W), np.nan)
            si0, si1 = max(0, di), H + min(0, di)
            sj0, sj1 = max(0, dj), W + min(0, dj)
            ti0, ti1 = max(0, -di), H + min(0, -di)
            tj0, tj1 = max(0, -dj), W + min(0, -dj)
            sh[ti0:ti1, tj0:tj1] = out[si0:si1, sj0:sj1]
            valid = ~np.isnan(sh)
            acc[valid] += sh[valid]; cnt[valid] += 1
        nb_mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), out)
        new = (1 - alpha) * out + alpha * nb_mean
        new[~free] = np.nan
        out = new
    return out


def fill_walls_monotone(D_free: np.ndarray, maze_grid: np.ndarray,
                        penalty: float = 2.0) -> np.ndarray:
    """
    Fill wall cells so that from EVERY adjacent free cell a step into the wall is
    strictly uphill (higher cost). A wall cell adjacent to free space gets
        max(free-neighbour values) + penalty
    which guarantees it exceeds each of those free neighbours; deeper wall layers
    grow inward with +penalty each, so they only get higher. This removes any
    false "shortcut through the wall" gradient (a wall touching the goal must NOT
    inherit the goal's low value -- the earlier min()+penalty version did, letting
    a distant free cell descend straight into the wall).
    """
    H, W = D_free.shape
    out = D_free.copy()
    wall = (maze_grid == 1)
    max_fin = np.nanmax(out[~wall]) if (~wall).any() else 1.0
    out[wall] = np.inf
    frontier = deque()
    # seed: wall cells adjacent to a filled free cell -> max(free nb) + penalty
    for i in range(H):
        for j in range(W):
            if wall[i, j]:
                nb = [out[i+di, j+dj] for di, dj in _N4
                      if 0 <= i+di < H and 0 <= j+dj < W
                      and not wall[i+di, j+dj] and np.isfinite(out[i+di, j+dj])]
                if nb:
                    out[i, j] = max(nb) + penalty
                    frontier.append((i, j))
    # grow inward: interior wall cells sit above the wall layer around them
    while frontier:
        i, j = frontier.popleft()
        for di, dj in _N4:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and wall[ni, nj] and np.isinf(out[ni, nj]):
                out[ni, nj] = out[i, j] + penalty
                frontier.append((ni, nj))
    out[np.isinf(out)] = max_fin + penalty  # fully enclosed walls
    return out.astype(np.float32)


def build_field(maze_grid, goal_cell, iters=8, alpha=0.5, wall_penalty=2.0):
    """BFS -> masked smooth (no cross-wall blur) -> monotone wall fill. Returns
    a finite (H,W) field ready for bilinear sampling."""
    D = compute_distance_field(maze_grid, goal_cell)
    Dsm = masked_diffusion_smooth(D, maze_grid, iters=iters, alpha=alpha)
    return fill_walls_monotone(Dsm, maze_grid, penalty=wall_penalty)


# --------------------------------------------------------------------------- #
# dead-end detection
# --------------------------------------------------------------------------- #
def deadend_map(maze_grid: np.ndarray, start=None, goal=None) -> np.ndarray:
    """
    Boolean (H,W): free cells in dead-end pockets. Iteratively prune degree-1
    free cells (only one free neighbour) that are neither start nor goal; a
    pruned cell is a dead-end tendril. Loops/corridors (degree>=2) survive.
    """
    H, W = maze_grid.shape
    free = (maze_grid == 0)
    deg = np.zeros((H, W), dtype=int)
    for i in range(H):
        for j in range(W):
            if free[i, j]:
                deg[i, j] = sum(1 for di, dj in _N4
                                if 0 <= i+di < H and 0 <= j+dj < W and free[i+di, j+dj])
    dead = np.zeros((H, W), dtype=bool)
    keep = set()
    if start is not None: keep.add(tuple(start))
    if goal is not None: keep.add(tuple(goal))
    alive = free.copy()
    changed = True
    while changed:
        changed = False
        for i in range(H):
            for j in range(W):
                if alive[i, j] and (i, j) not in keep and deg[i, j] <= 1:
                    alive[i, j] = False; dead[i, j] = True; changed = True
                    for di, dj in _N4:
                        ni, nj = i+di, j+dj
                        if 0 <= ni < H and 0 <= nj < W and alive[ni, nj]:
                            deg[ni, nj] -= 1
    return dead


def deadend_severity(maze_grid, start=None, goal=None) -> float:
    """Fraction of free cells that are dead-end pockets (map-level scalar)."""
    free = (maze_grid == 0)
    n = int(free.sum())
    return float(deadend_map(maze_grid, start, goal).sum()) / n if n else 0.0


# --------------------------------------------------------------------------- #
# per-trajectory progress scalars (for DFS backtracking)
# --------------------------------------------------------------------------- #
def stalled_progress(distances):
    """
    Mean positive increment of goal-distance along a rollout: 0 = strictly
    approaching the goal, large = stalling/reversing (dead-end trapping).
    `distances`: (T,) or (B,T) numpy or torch. Returns float or (B,) same type.
    """
    if _HAVE_TORCH and isinstance(distances, torch.Tensor):
        d = distances
        if d.ndim == 1:
            return torch.clamp(d[1:] - d[:-1], min=0.0).mean()
        return torch.clamp(d[:, 1:] - d[:, :-1], min=0.0).mean(dim=1)
    d = np.asarray(distances, dtype=float)
    if d.ndim == 1:
        return float(np.clip(np.diff(d), 0, None).mean()) if d.size > 1 else 0.0
    return np.clip(np.diff(d, axis=1), 0, None).mean(axis=1)


def is_dead_end_rollout(distances, tol=0.0, frac=0.5):
    """
    Heuristic DFS trigger: True if the rollout fails to make net progress on more
    than `frac` of its steps (goal-distance not decreasing beyond `tol`).
    """
    d = np.asarray(distances, dtype=float)
    if d.ndim == 1:
        d = d[None]
    inc = (np.diff(d, axis=1) >= -tol).mean(axis=1)
    out = inc > frac
    return bool(out[0]) if out.size == 1 else out
