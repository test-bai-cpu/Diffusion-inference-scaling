"""
episode_metrics.py
==================
Per-episode instrumentation that SEPARATES the outcomes the raw success rate
conflates:

    success        : reached the goal (terminal / positive return)
    collision_rate : fraction of realized-rollout timesteps in contact with (or
                     penetrating) a wall, using an exact box signed-distance with
                     a ball-radius inflation -- the geometry the corner-safe
                     verifier uses. Distinguishes "planned a clean path" from
                     "scraped along / pressed into walls".
    cornercut_rate : fraction of timesteps sitting inside a DIAGONAL-PINCH gap
                     (two walls touching only at a corner). These are the points
                     the old inside-only+max verifier scored 0 and where the sim
                     blocks the agent -- the corner-cutting failure mode.
    deadend_frac   : fraction of timesteps whose grid cell is a dead-end pocket
                     (leaf-pruned tendril) -- time wasted wandering into traps.
    stalled_prog   : mean positive increment of BFS goal-distance along the
                     realized path (0 = monotone approach, high = reversing).
    final_gap      : BFS goal-distance (cells) of the last position (0 = at goal).

All metrics are computed from the realized rollout (an (N,>=2) array of observed
states; columns 0,1 = world x,y) plus the maze geometry -- pure numpy, no torch,
no GPU deps, so it runs in-sim on the user's machine and in the sandbox.

Coordinate convention (confirmed): world xy = (j*U - Ox, i*U - Oy); cell (i,j)
from  i = round((y+Oy)/U),  j = round((x+Ox)/U).
"""
from __future__ import annotations
import numpy as np

try:
    from search.distance_field_v2 import compute_distance_field, deadend_map
except Exception:
    import importlib.util, os
    _spec = importlib.util.spec_from_file_location(
        "_dfm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "distance_field_v2.py"))
    _m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
    compute_distance_field, deadend_map = _m.compute_distance_field, _m.deadend_map

_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def wall_boxes(maze_grid, maze_unit, offset_x, offset_y):
    """Axis-aligned wall boxes as (centers (K,2) xy, half (2,)). Each wall cell
    (i,j) -> center (j*U-Ox, i*U-Oy), half-extent U/2 on each axis."""
    H, W = maze_grid.shape
    ii, jj = np.where(maze_grid == 1)
    cx = jj * maze_unit - offset_x
    cy = ii * maze_unit - offset_y
    centers = np.stack([cx, cy], axis=1).astype(float)
    half = np.array([maze_unit / 2.0, maze_unit / 2.0])
    return centers, half


def box_sdf(points, centers, half):
    """Signed distance from each point (N,2) to the NEAREST box (K boxes share
    `half`). Negative inside. Returns (N,) min over boxes."""
    d = np.abs(points[:, None, :] - centers[None, :, :]) - half[None, None, :]  # (N,K,2)
    outside = np.linalg.norm(np.clip(d, 0, None), axis=2)                        # (N,K)
    inside = np.clip(np.max(d, axis=2), None, 0)                                 # (N,K)
    sdf = outside + inside                                                       # (N,K)
    return sdf.min(axis=1)                                                       # (N,)


def diagonal_pinch_points(maze_grid, maze_unit, offset_x, offset_y):
    """World (x,y) of every diagonal-pinch corner: a 2x2 cell block with exactly
    two walls on one diagonal and two free on the other -- the gap the sim blocks
    but a naive verifier lets through. Returns (P,2)."""
    H, W = maze_grid.shape
    pts = []
    for i in range(H - 1):
        for j in range(W - 1):
            a = maze_grid[i, j];     b = maze_grid[i, j + 1]
            c = maze_grid[i + 1, j]; d = maze_grid[i + 1, j + 1]
            diag1 = (a == 1 and d == 1 and b == 0 and c == 0)
            diag2 = (b == 1 and c == 1 and a == 0 and d == 0)
            if diag1 or diag2:
                # shared corner is between the 4 cells: at half-step offset
                px = (j + 0.5) * maze_unit - offset_x
                py = (i + 0.5) * maze_unit - offset_y
                pts.append((px, py))
    return np.array(pts, dtype=float) if pts else np.zeros((0, 2))


def xy_to_ij(x, y, maze_unit, offset_x, offset_y):
    return (int(round((y + offset_y) / maze_unit)),
            int(round((x + offset_x) / maze_unit)))


# --------------------------------------------------------------------------- #
# main scorer
# --------------------------------------------------------------------------- #
def score_episode(rollout, maze_grid, goal_ij, maze_unit, offset_x, offset_y,
                  success=None, total_reward=None, terminal=None,
                  ball_radius=0.5, contact_margin=0.15, pinch_radius=0.9,
                  start_ij=None):
    """
    Score one realized rollout. Returns a flat dict of scalars (all rate metrics
    in [0,1] so the pipeline's *100 averaging turns them into percentages).

    `rollout`  : (N, >=2) array; cols 0,1 = world x,y.
    `success`  : bool if known; else inferred from total_reward>0 or terminal.
    """
    r = np.asarray(rollout, dtype=float)
    xy = r[:, :2]
    N = len(xy)

    if success is None:
        if total_reward is not None:
            success = bool(total_reward > 0)
        elif terminal is not None:
            success = bool(terminal)
        else:
            success = False

    centers, half = wall_boxes(maze_grid, maze_unit, offset_x, offset_y)
    # contact / penetration: point within (ball_radius+margin) of any wall face
    sdf = box_sdf(xy, centers, half) if len(centers) else np.full(N, np.inf)
    in_contact = sdf < (ball_radius + contact_margin)
    collision_rate = float(in_contact.mean())

    # corner-cut: within pinch_radius of any diagonal-pinch corner
    pinch = diagonal_pinch_points(maze_grid, maze_unit, offset_x, offset_y)
    if len(pinch):
        dmin = np.linalg.norm(xy[:, None, :] - pinch[None, :, :], axis=2).min(axis=1)
        cornercut_rate = float((dmin < pinch_radius).mean())
    else:
        cornercut_rate = 0.0

    # dead-end time fraction
    dmap = deadend_map(maze_grid, start=start_ij, goal=goal_ij)
    H, W = maze_grid.shape
    ij = np.array([xy_to_ij(x, y, maze_unit, offset_x, offset_y) for x, y in xy])
    ij[:, 0] = np.clip(ij[:, 0], 0, H - 1); ij[:, 1] = np.clip(ij[:, 1], 0, W - 1)
    in_dead = dmap[ij[:, 0], ij[:, 1]]
    deadend_frac = float(in_dead.mean())

    # progress along realized path (BFS goal distance)
    D = compute_distance_field(maze_grid, goal_ij)
    dvals = D[ij[:, 0], ij[:, 1]].astype(float)
    finite = np.isfinite(dvals)
    if finite.sum() >= 2:
        seq = dvals[finite]
        stalled = float(np.clip(np.diff(seq), 0, None).mean())
        final_gap = float(seq[-1])
    else:
        stalled = 0.0
        final_gap = float('nan')

    return {
        'success': float(bool(success)),
        'collision_rate': collision_rate,
        'cornercut_rate': cornercut_rate,
        'deadend_frac': deadend_frac,
        'stalled_prog': stalled,
        'final_gap': final_gap,
        'steps': float(N),
    }
