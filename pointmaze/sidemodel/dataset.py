"""
dataset.py — build map-conditioned training samples for MCGN.
=============================================================
For each free cell of a maze (or each visited waypoint of a demo trajectory) we
build one training sample:

    occ    (1,K,K)  local occupancy crop centered on the cell (1=wall, 0=free);
                    out-of-bounds padded as wall.
    goal   (3,)     [unit_gx, unit_gy, log1p(D)] where (gx,gy) is the direction
                    from the cell to the goal in GRID units and D is the geodesic
                    distance-to-goal from the corrected field (build_field).
    target (2,)     true geodesic descent direction = unit(-∇D) estimated from
                    the 4-connected neighbor with the smallest field value.

Because the sample is centered and local, samples from many mazes are
interchangeable; train on the giant maze's free cells + demo waypoints and the
model generalizes to unseen layouts.

The distance field comes from distance_field_v2.build_field (wall-respecting,
strictly-uphill wall fill), so targets never point through a wall.
"""
import os, sys, importlib.util
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SEARCH = os.path.join(os.path.dirname(_HERE), 'search')


def _load(fn, nm):
    spec = importlib.util.spec_from_file_location(nm, os.path.join(_SEARCH, fn))
    m = importlib.util.module_from_spec(spec)
    sys.modules[nm] = m
    spec.loader.exec_module(m)
    return m


try:
    from search.distance_field_v2 import build_field
except Exception:
    build_field = _load('distance_field_v2.py', '_df2_ds').build_field


def occupancy_crop(maze_map, i, j, K):
    """(K,K) crop of maze_map centered at (i,j); out-of-bounds = wall (1)."""
    r = K // 2
    H, W = maze_map.shape
    crop = np.ones((K, K), dtype=np.float32)               # default wall
    for a in range(-r, r + 1):
        for b in range(-r, r + 1):
            ii, jj = i + a, j + b
            if 0 <= ii < H and 0 <= jj < W:
                crop[a + r, b + r] = float(maze_map[ii, jj])
    return crop


def geodesic_descent_dir(field, i, j):
    """Unit direction (dx_grid, dy_grid) toward the smallest-field 4-neighbor.
    Returns (0,0) if no strictly-lower finite neighbor (goal / trapped)."""
    H, W = field.shape
    best = field[i, j]; bi, bj = i, j
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ii, jj = i + di, j + dj
        if 0 <= ii < H and 0 <= jj < W and np.isfinite(field[ii, jj]) and field[ii, jj] < best:
            best = field[ii, jj]; bi, bj = ii, jj
    # grid dx along +j (x), dy along +i (y)
    dx, dy = (bj - j), (bi - i)
    n = (dx * dx + dy * dy) ** 0.5
    if n == 0:
        return 0.0, 0.0
    return dx / n, dy / n


def build_samples(maze_map, goal_ij, K=15, iters=8, alpha=0.5, wall_penalty=2.0,
                  cells=None):
    """Return dict of arrays: occ (N,1,K,K), goal (N,3), target (N,2), ij (N,2).

    cells : optional list of (i,j) to sample (e.g. demo waypoints). Default =
    every free cell with a valid descent direction."""
    maze_map = np.asarray(maze_map, dtype=int)
    field = build_field(maze_map, goal_ij, iters=iters, alpha=alpha,
                        wall_penalty=wall_penalty)
    H, W = maze_map.shape
    gi, gj = goal_ij
    if cells is None:
        cells = [(i, j) for i in range(H) for j in range(W)
                 if maze_map[i, j] == 0]
    occ, goal, tgt, ijs = [], [], [], []
    for (i, j) in cells:
        if maze_map[i, j] != 0:
            continue
        d = geodesic_descent_dir(field, i, j)
        if d == (0.0, 0.0):
            continue                                       # goal / trapped: skip
        occ.append(occupancy_crop(maze_map, i, j, K)[None])
        gx, gy = (gj - j), (gi - i)
        gn = (gx * gx + gy * gy) ** 0.5 + 1e-6
        D = field[i, j]
        Dv = float(D) if np.isfinite(D) else float(np.nanmax(field[np.isfinite(field)]))
        goal.append([gx / gn, gy / gn, np.log1p(Dv)])
        tgt.append(list(d))
        ijs.append([i, j])
    return {
        'occ': np.asarray(occ, dtype=np.float32),
        'goal': np.asarray(goal, dtype=np.float32),
        'target': np.asarray(tgt, dtype=np.float32),
        'ij': np.asarray(ijs, dtype=np.int64),
        'field': field,
    }
