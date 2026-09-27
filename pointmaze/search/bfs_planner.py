"""
Classical (non-diffusion) BFS planning utilities for the bfs-greedy and
bfs-cspace baselines (search/methods/bfs_baseline_pipes.py).

These exist to answer a specific reviewer objection to the BFS
distance-field guidance in the main method: "if you already compute a BFS
distance field to the goal, don't you already have the solution?" This
module implements the most literal version of "just use the BFS field
directly" (bfs-greedy) and a properly-engineered classical pipeline built on
top of it (bfs-cspace), so both can be rolled out through the exact same
real-env execution path as the diffusion+search method for an apples-to-
apples comparison.

Does NOT modify anything under search/methods/{dfs,bfs,tfg,adaptive_dfs,
staged_dfs,base_guidance}.py, search/base_pipeline.py, search/configs.py,
search/maze_verifier.py, search/distance_field*.py, run.py, or
search/script_utils.py -- those are the main method's code and are untouched
by this file or by search/methods/bfs_baseline_pipes.py.
"""
from collections import deque

import numpy as np


def get_agent_radius(env) -> float:
    """Read the point-mass's true collision radius directly from the
    compiled MuJoCo model (geom 'pointbody', a sphere), rather than
    guessing it. Confirmed by live introspection on pointmaze-giant:
    model.geom('pointbody').size == [0.7, 0, 0]. Read live (not hardcoded)
    so it stays correct if this is ever pointed at a different embodiment.
    """
    u = env.unwrapped if hasattr(env, 'unwrapped') else env
    return float(u.model.geom('pointbody').size[0])


def bfs_backtrack_cells(maze_map: np.ndarray, start_ij, goal_ij, connectivity: int = 4):
    """Shortest path in grid cells from start_ij to goal_ij via BFS with
    parent pointers. Same connectivity convention as
    search/distance_field.py:compute_distance_field -- 4 means N/S/E/W only,
    no diagonal edges, matching --dist_connectivity 4, the value the main
    method's DistanceFieldVerifier actually runs with (confirmed in
    run_guidance.sh / run_origin_best_config.sh and the saved args.json of
    the main method's own completed runs).

    Returns a list of (i, j) cells from start to goal inclusive, or None if
    unreachable at grid resolution.
    """
    H, W = maze_map.shape
    if maze_map[start_ij] == 1 or maze_map[goal_ij] == 1:
        return None
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    parent = {start_ij: None}
    q = deque([start_ij])
    while q:
        cur = q.popleft()
        if cur == goal_ij:
            break
        i, j = cur
        for di, dj in deltas:
            nxt = (i + di, j + dj)
            ni, nj = nxt
            if 0 <= ni < H and 0 <= nj < W and maze_map[ni, nj] == 0 and nxt not in parent:
                parent[nxt] = cur
                q.append(nxt)

    if goal_ij not in parent:
        return None

    path = [goal_ij]
    while path[-1] != start_ij:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def forbidden_corner_points_np(maze_map: np.ndarray, maze_unit: float,
                                offset_x: float, offset_y: float):
    """Pure-numpy re-derivation of
    search/maze_verifier.py:get_forbidden_corner_points (that version is
    torch/env-coupled, built for gradient guidance). Same definition: the
    world-space center of every 2x2 grid block where the maze wall pattern
    is a diagonal pinch (wall/free stacked diagonally against free/wall) --
    a zero-width gap a point mass can numerically slip through without
    literally entering either wall's box.
    """
    H, W = maze_map.shape
    corners = []
    for i in range(H - 1):
        for j in range(W - 1):
            nw, ne = maze_map[i, j] == 1, maze_map[i, j + 1] == 1
            sw, se = maze_map[i + 1, j] == 1, maze_map[i + 1, j + 1] == 1
            if (nw and se and not ne and not sw) or (ne and sw and not nw and not se):
                x = (j + 0.5) * maze_unit - offset_x
                y = (i + 0.5) * maze_unit - offset_y
                corners.append((x, y))
    return corners


def _segment_hits_inflated_wall(p0, p1, wall_centers, half_extent, n_samples):
    """Dense-sample the segment and test each sample against every
    Minkowski-inflated (by agent radius) axis-aligned wall box.
    O(n_samples * n_walls); both are small for this maze size (grid is
    12x16, segments are at most ~80 world units), so this is cheap enough
    that exact geometric slab-clipping isn't worth the extra bug surface
    for a one-off baseline script.
    """
    if len(wall_centers) == 0:
        return False
    ts = np.linspace(0.0, 1.0, n_samples)
    pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]          # (n_samples, 2)
    centers = np.asarray(wall_centers)                             # (n_walls, 2)
    dx = np.abs(pts[:, None, 0] - centers[None, :, 0])
    dy = np.abs(pts[:, None, 1] - centers[None, :, 1])
    inside = (dx <= half_extent) & (dy <= half_extent)
    return bool(inside.any())


def _segment_hits_corner(p0, p1, corner_points, radius, n_samples):
    if len(corner_points) == 0 or radius <= 0:
        return False
    ts = np.linspace(0.0, 1.0, n_samples)
    pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
    corners = np.asarray(corner_points)
    d2 = ((pts[:, None, :] - corners[None, :, :]) ** 2).sum(axis=-1)
    return bool((d2 <= radius ** 2).any())


def shortcut_path(waypoints, wall_centers, wall_half_extent, corner_points,
                   corner_radius, n_samples_per_unit: float = 10.0):
    """Greedy 'string-pulling': from each kept waypoint, jump to the
    farthest later waypoint reachable by a straight line that clears every
    inflated wall box and every forbidden diagonal corner zone, falling
    back one waypoint at a time when nothing further is clear.

    waypoints: ordered list of (2,) float arrays, world xy, first/last are
    the exact per-episode start/goal. wall_half_extent and corner_radius
    should already include the agent's true radius (Minkowski sum) --
    see get_agent_radius.
    """
    if len(waypoints) <= 2:
        return list(waypoints)

    def is_free(p0, p1):
        length = float(np.linalg.norm(p1 - p0))
        n = max(4, int(np.ceil(length * n_samples_per_unit)))
        if _segment_hits_inflated_wall(p0, p1, wall_centers, wall_half_extent, n):
            return False
        if _segment_hits_corner(p0, p1, corner_points, corner_radius, n):
            return False
        return True

    n = len(waypoints)
    out = [waypoints[0]]
    i = 0
    while i < n - 1:
        j = n - 1
        while j > i + 1 and not is_free(waypoints[i], waypoints[j]):
            j -= 1
        out.append(waypoints[j])
        i = j
    return out


def densify_path(waypoints, horizon: int) -> np.ndarray:
    """Resample a polyline to exactly `horizon` points by arc length, so
    index 0 is waypoints[0] exactly, index horizon-1 is waypoints[-1]
    exactly, and spacing in between is uniform in arc length.

    This matches the main method's plan representation: its diffuser output
    is a fixed horizon-length sequence with apply_conditioning hard-pinning
    index 0 to the real per-episode observation and index horizon-1 to the
    real per-episode goal (see search/base_pipeline.py's `cond` dict). Using
    the same fixed length and the same hard-pinned endpoints means the
    downstream P-controller and rollout loop can be byte-for-byte identical
    to the main method's for a genuine same-execution-path comparison.
    """
    pts = np.asarray(waypoints, dtype=np.float64)
    if len(pts) == 1:
        return np.repeat(pts, horizon, axis=0)
    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 1e-9:
        return np.repeat(pts[:1], horizon, axis=0)

    targets = np.linspace(0.0, total, horizon)
    out = np.empty((horizon, 2), dtype=np.float64)
    seg_idx = 0
    for k, t in enumerate(targets):
        while seg_idx < len(seg_len) - 1 and t > cum[seg_idx + 1]:
            seg_idx += 1
        denom = seg_len[seg_idx] if seg_len[seg_idx] > 1e-9 else 1.0
        frac = (t - cum[seg_idx]) / denom
        frac = min(max(frac, 0.0), 1.0)
        out[k] = pts[seg_idx] + frac * seg[seg_idx]
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def path_length(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
