"""
Per-timestep local map features ("feature grid" conditioning).

Motivation: the global-embedding pathway compresses the whole maze into one
vector shared by every timestep, so the network must *recall* where walls are.
On topology-OOD variants (corridors blocked relative to the training family)
the prior confidently draws memorized routes through walls (see the
level3-variant2 postmortem: all rollouts pinned at 4 chokepoints, every final
plan forced_best with wall cost 17-240). Per-timestep sampling fixes the
delivery: each trajectory timestep receives the map content *at its own (x,y)*
as extra input channels, so "this cell is a wall" and "distance-to-goal here"
arrive locally instead of being decoded from a summary.

Channels (local_channels=2):
  0. occupancy   -- 1=wall, 0=free, from the padded training canvas
  1. dist2goal   -- BFS distance to the goal cell, normalized to [0,1]
                    (walls/unreachable = 1). At inference this is computed on
                    the TRUE variant grid, so it encodes the correct detour
                    topology even when the learned prior never saw such a route.

Coordinate chain (all affine, precomputed once, stored as model buffers so
checkpoints carry it):
  normalized xy in [-1,1]  --(shared GlobalNormalizer bounds)-->  world xy
  world xy                 --(maze_unit=4, offset=4)-->            canvas cell
  canvas cell              --(align_corners=False)-->              grid_sample uv
"""
from collections import deque

import numpy as np

from . import maze_grids as MG


# --------------------------------------------------------------------------- #
#  BFS distance field on the canvas grid
# --------------------------------------------------------------------------- #

def bfs_distance_field(grid, goal_ij, normalize=True):
    """
    4-connected BFS over free cells (0=free, 1=wall) from goal_ij=(gi, gj).
    Returns float32 [H, W]. If normalize: distances scaled by 1/(H+W) and
    clipped to 1; wall / unreachable cells = 1.0 (the "worst" value, so the
    channel reads as 'do not be here'). If the goal cell is a wall, the whole
    field is 1.0.
    """
    g = np.asarray(grid)
    H, W = g.shape
    D = np.full((H, W), np.inf, dtype=np.float64)
    gi, gj = int(goal_ij[0]), int(goal_ij[1])
    if 0 <= gi < H and 0 <= gj < W and g[gi, gj] == 0:
        D[gi, gj] = 0.0
        q = deque([(gi, gj)])
        while q:
            i, j = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and g[ni, nj] == 0 \
                        and D[ni, nj] == np.inf:
                    D[ni, nj] = D[i, j] + 1.0
                    q.append((ni, nj))
    if not normalize:
        return D.astype(np.float32)
    out = np.clip(D / float(H + W), 0.0, 1.0)
    out[~np.isfinite(D)] = 1.0
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Coordinate mapping
# --------------------------------------------------------------------------- #

def world_xy_to_cell(x, y):
    """World (x, y) -> nearest canvas cell (i, j). Inverse of MG.cell_to_xy."""
    j = int(round((float(x) + MG.OFFSET_X) / MG.MAZE_UNIT))
    i = int(round((float(y) + MG.OFFSET_Y) / MG.MAZE_UNIT))
    return i, j


def norm_xy_to_uv_affine(obs_xy_min, obs_xy_max, canvas_hw):
    """
    Build the affine (scale, shift) mapping NORMALIZED trajectory xy (in
    [-1,1], per the shared GlobalNormalizer) to F.grid_sample uv coordinates
    over a [Hc, Wc] canvas feature map with align_corners=False.

    Derivation per axis (x -> u along width; y -> v along height):
        world  w  = n * (max-min)/2 + (max+min)/2
        cellf  c  = (w + OFFSET) / MAZE_UNIT          (cell centers at ints)
        pixel     : grid_sample(align_corners=False) puts pixel k's center at
                    u = 2*(k + 0.5)/size - 1
        =>     u  = 2*(c + 0.5)/size - 1
    Returns (scale[2], shift[2]) float32 arrays ordered (x/u, y/v) --
    the same ordering grid_sample expects in its last dimension.
    """
    Hc, Wc = int(canvas_hw[0]), int(canvas_hw[1])
    mn = np.asarray(obs_xy_min, np.float64)[:2]
    mx = np.asarray(obs_xy_max, np.float64)[:2]
    sizes = np.array([Wc, Hc], np.float64)          # x spans width, y height
    offs = np.array([MG.OFFSET_X, MG.OFFSET_Y], np.float64)

    half_range = (mx - mn) / 2.0
    mid = (mx + mn) / 2.0
    # n -> cellf:  c = (n*half + mid + off) / UNIT
    a_cell = half_range / MG.MAZE_UNIT
    b_cell = (mid + offs) / MG.MAZE_UNIT
    # cellf -> uv: u = (2/size)*c + (1/size - 1)
    scale = (2.0 / sizes) * a_cell
    shift = (2.0 / sizes) * (b_cell + 0.5) - 1.0
    return scale.astype(np.float32), shift.astype(np.float32)


def build_local_stack(occ_grid, goal_ij=None):
    """
    Stack the local-conditioning channels for one map: [2, Hc, Wc] float32,
    channel 0 = occupancy, channel 1 = normalized BFS distance to goal_ij
    (all-ones "unknown" field when goal_ij is None).
    """
    occ = np.asarray(occ_grid, np.float32)
    if goal_ij is None:
        dist = np.ones_like(occ)
    else:
        dist = bfs_distance_field(occ, goal_ij)
    return np.stack([occ, dist], axis=0)
