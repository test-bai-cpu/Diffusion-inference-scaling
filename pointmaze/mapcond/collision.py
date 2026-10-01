"""
Differentiable wall-collision penalty for map-conditional training.

Why this exists
---------------
The diffusion objective has no negative examples. Every demonstration is a
valid path on its own map, so the loss only ever asks the model to reproduce
trajectories that happen to be legal -- nothing anywhere penalizes putting a
waypoint inside a wall. The conditioning probe (pointmaze/cond_sensitivity.py)
measured the consequence on the trained DiT: opening a cell moves the plans a
lot (+65pp of samples reroute through a newly opened cell), closing one moves
them almost not at all (-3pp), and on task1/var7 NOT ONE of 64 sampled plans
lands below the DFS acceptance threshold -- the search is always picking the
least-bad of a distribution that contains no clean plan at all.

This term supplies the gradient that was missing: "for THIS map, keep the
predicted trajectory out of the walls".

Mechanism
---------
Per map, a penetration-depth field (0 on free cells; inside walls, BFS distance
in cells to the nearest free cell) is bilinearly sampled at each predicted
waypoint, reusing the same normalized-xy -> grid_sample-uv affine as the
local-feature track (local_features.norm_xy_to_uv_affine). Differentiable
w.r.t. the trajectory and naturally batched over per-sample maps, which matters
because every item in a training batch comes from a different map.

Depth rather than raw occupancy on purpose: bilinear interpolation of a binary
grid is flat at 1.0 deep inside a wall block, so a waypoint stranded in the
middle of a wall would get no gradient direction at all. The BFS depth field
slopes outward from anywhere inside a wall.

Scope: wall PENETRATION only. The corner-cut detectors in
search/maze_verifier.py are binary cell-index tests with no usable gradient, so
they are deliberately not part of this term -- a trajectory can still clip a
diagonal corner without being penalized here.
"""
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

try:
    # Already a dependency of the eval path (search/distance_field.py uses
    # scipy.ndimage). Gives the true Euclidean distance; the BFS fallback below
    # is 4-connected, i.e. Manhattan, which overestimates depth by up to sqrt(2)
    # in diagonal regions (measured on task1/var7: max depth 8.00 vs 5.66, a
    # 2.34 world-unit disagreement) and is ~38x slower.
    from scipy.ndimage import distance_transform_edt as _edt
except ImportError:  # pragma: no cover
    _edt = None


def _bfs_depth(wall_mask):
    """4-connected multi-source BFS fallback for when scipy is unavailable.
    Returns depth in SUBCELLS. Manhattan, so it overestimates diagonal depth --
    prefer the EDT path."""
    H, W = wall_mask.shape
    depth = np.full((H, W), -1, dtype=np.int32)
    q = deque()
    for i in range(H):
        for j in range(W):
            if not wall_mask[i, j]:
                depth[i, j] = 0
                q.append((i, j))
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and depth[ni, nj] < 0:
                depth[ni, nj] = depth[i, j] + 1
                q.append((ni, nj))
    depth[depth < 0] = H + W
    return depth.astype(np.float64)


def wall_depth_field(grid, upsample=8, maze_unit=4.0):
    """
    0 in free space; inside walls, distance (in WORLD units) to the nearest
    free space. `grid` is HxW with nonzero = wall. Returns float32
    [H*upsample, W*upsample].

    The upsampling is not cosmetic. At cell resolution, bilinearly sampling
    this field blends the four nearest cell CENTRES, so any point within one
    cell of a wall picks up a nonzero value -- and in a maze whose corridors
    are one cell wide, that is nearly every waypoint, including ones running
    down the middle of a legal corridor. Measured on real plans, the
    cell-resolution version scored the cleanest plan of a batch (verifier cost
    1.0) WORSE than one that drove through a wall (cost 21.4), and put gradient
    on ~75% of all waypoints. Subcell resolution keeps free space at exactly 0
    except within one subcell of a boundary.

    The normalized-xy -> uv affine is resolution independent (it maps world
    extent to [-1,1] regardless of how many texels cover it), so callers keep
    using local_features.norm_xy_to_uv_affine with the COARSE canvas size.
    """
    g = np.asarray(grid) > 0.5
    if upsample > 1:
        g = np.repeat(np.repeat(g, upsample, axis=0), upsample, axis=1)
    if not g.any():
        return np.zeros(g.shape, dtype=np.float32)
    if g.all():
        # No free space anywhere to measure a distance to; saturate rather than
        # returning something that would reward going there.
        return np.full(g.shape, float(sum(g.shape)), dtype=np.float32)
    # distance_transform_edt measures, for every nonzero entry, the distance to
    # the nearest zero -- i.e. exactly "distance from inside a wall out to free
    # space", and 0 on free cells.
    depth = _edt(g) if _edt is not None else _bfs_depth(g)
    return depth.astype(np.float32) * (maze_unit / float(upsample))


_FIELD_CACHE = {}


def depth_fields(maze, upsample=8, maze_unit=4.0):
    """
    [B, H, W] occupancy (or [B, C, H, W], channel 0 = occupancy) ->
    [B, 1, H*upsample, W*upsample] depth fields on the same device.

    Cached per distinct grid: a training run pools ~60 maps, so the BFS runs
    ~60 times for the whole run rather than once per optimization step.
    """
    if maze.dim() == 4:
        maze = maze[:, 0]
    host = maze.detach().to("cpu").numpy()
    fields = []
    for k in range(host.shape[0]):
        key = (host[k].tobytes(), host[k].shape, upsample)
        f = _FIELD_CACHE.get(key)
        if f is None:
            f = wall_depth_field(host[k], upsample=upsample, maze_unit=maze_unit)
            _FIELD_CACHE[key] = f
        fields.append(f)
    out = torch.from_numpy(np.stack(fields)).unsqueeze(1)
    return out.to(device=maze.device, dtype=torch.float32)


def forbidden_corners(grid, maze_unit=4.0, offset_x=4.0, offset_y=4.0):
    """
    World-space corner points of diagonal wall pinches, i.e. the 2x2 patterns

        wall free       free wall
        free wall   or  wall free

    whose two free cells touch only at a zero-width vertex. Same rule as
    search/maze_verifier.get_forbidden_corner_points, vectorized.
    """
    g = np.asarray(grid) > 0.5
    nw, ne = g[:-1, :-1], g[:-1, 1:]
    sw, se = g[1:, :-1], g[1:, 1:]
    hit = (nw & se & ~ne & ~sw) | (ne & sw & ~nw & ~se)
    ii, jj = np.nonzero(hit)
    x = (jj + 0.5) * maze_unit - offset_x
    y = (ii + 0.5) * maze_unit - offset_y
    return np.stack([x, y], axis=-1).astype(np.float32)


_CORNER_CACHE = {}


def corner_points(maze, maze_unit=4.0):
    """
    [B, H, W] occupancy -> [B, K, 2] world-space forbidden corners, right-padded
    with a far-away sentinel so maps with different corner counts batch together
    (the sentinel is far outside the canvas, so its clearance term is always 0).
    """
    if maze.dim() == 4:
        maze = maze[:, 0]
    host = maze.detach().to("cpu").numpy()
    per_map = []
    for k in range(host.shape[0]):
        key = (host[k].tobytes(), host[k].shape)
        c = _CORNER_CACHE.get(key)
        if c is None:
            c = forbidden_corners(host[k], maze_unit=maze_unit)
            _CORNER_CACHE[key] = c
        per_map.append(c)
    K = max(1, max(len(c) for c in per_map))
    out = np.full((len(per_map), K, 2), 1e6, dtype=np.float32)
    for k, c in enumerate(per_map):
        if len(c):
            out[k, :len(c)] = c
    return torch.from_numpy(out).to(device=maze.device, dtype=torch.float32)


def corner_clearance_penalty(xy_world, corners, radius):
    """
    Smooth counterpart of search/maze_verifier.batched_corner_zone_transition_loss.

    That detector computes the distance from each trajectory SEGMENT to each
    forbidden corner -- differentiable all the way -- and then destroys the
    gradient with `<= radius**2`, because it only ever had to feed a DFS
    accept/reject test. Here the same distance is hinged instead:

        penalty = relu(radius - d)

    0 while the segment keeps `radius` clearance from the pinch, growing as it
    closes in, maximal when it passes exactly through. -grad pushes the segment
    off the corner.

    This is what the wall-depth term cannot see: slipping between two diagonally
    adjacent wall cells grazes both boxes with ~0 penetration, so the depth
    field reads ~0 while this reads ~radius.

    Isotropic, like the binary detector it replaces: it penalizes approaching the
    vertex from a legal side too, not just crossing the pinch. That is the same
    rule DFS already enforces at inference (`point_hit` fires on proximity
    regardless of direction), so training and acceptance stay consistent; the
    practical effect is "round your corners", which is the documented failure
    mode anyway.

    xy_world : [B, T, 2] positions in WORLD units
    corners  : [B, K, 2] from corner_points()
    returns  : [B] mean clearance violation over segments
    """
    p0 = xy_world[:, :-1]                                  # [B,T-1,2]
    seg = xy_world[:, 1:] - p0
    seg_len2 = seg.square().sum(-1, keepdim=True).clamp_min(1e-8)

    rel = corners[:, None] - p0[:, :, None]                # [B,T-1,K,2]
    proj = (rel * seg[:, :, None]).sum(-1, keepdim=True) / seg_len2[:, :, None]
    closest = p0[:, :, None] + proj.clamp(0.0, 1.0) * seg[:, :, None]
    d = (closest - corners[:, None]).square().sum(-1).clamp_min(1e-12).sqrt()
    return F.relu(radius - d).amax(dim=-1).mean(dim=1)


_SAFE_CELL_CACHE = {}


def non_articulation_free_cells(grid):
    """
    Free cells that can be closed without splitting the free space in two.

    Closing an articulation point would make start and goal unreachable from
    each other, and since both are pinned by apply_conditioning the model would
    be asked for a trajectory that cannot exist -- a negative with no achievable
    answer. Restricting to non-articulation cells keeps every manufactured
    negative well posed: a legal re-route always exists.

    Computed once per map (90-ish free cells x one BFS each) and cached.
    """
    g = np.asarray(grid) > 0.5
    H, W = g.shape
    free = [(i, j) for i in range(H) for j in range(W) if not g[i, j]]
    if len(free) <= 2:
        return []
    safe = []
    for c in free:
        start = next((f for f in free if f != c), None)
        seen = {start}
        q = deque([start])
        while q:
            i, j = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (i + di, j + dj)
                if (0 <= n[0] < H and 0 <= n[1] < W and not g[n] and n != c
                        and n not in seen):
                    seen.add(n)
                    q.append(n)
        if len(seen) == len(free) - 1:
            safe.append(c)
    return safe


def block_route_cell(maze, xy_world, generator=None, maze_unit=4.0,
                     offset_x=4.0, offset_y=4.0):
    """
    Manufacture negatives: for each sample, close one free cell that its own
    trajectory passes through.

    This reproduces, as a training case, exactly the failure diagnosed on the
    OOD variants -- "a cell that is open in almost every map is closed in this
    one" -- and it does so with a trajectory that is guaranteed to violate the
    perturbed map, at EVERY noise level. That is the point: with the map and the
    demo matched, the predicted x0 is already legal below t~100 (measured), so
    the penalty has no signal exactly where the real failures appear.

    The perturbed map is only ever used for a penalty-only pass. It must NOT be
    paired with the reconstruction loss: the demo is illegal on the perturbed
    map, so asking the model to both reproduce it and avoid the wall is a
    direct contradiction.

    maze     : [B, H, W] occupancy
    xy_world : [B, T, 2] trajectory positions in WORLD units
    returns  : (perturbed [B,H,W], valid mask [B] bool, closed cells list)
    """
    host = maze.detach().to("cpu").numpy().copy()
    xy = xy_world.detach().to("cpu").numpy()
    B, H, W = host.shape
    rng = np.random if generator is None else generator

    jj = np.floor((xy[..., 0] + offset_x + 0.5 * maze_unit) / maze_unit).astype(int)
    ii = np.floor((xy[..., 1] + offset_y + 0.5 * maze_unit) / maze_unit).astype(int)

    valid = np.zeros(B, dtype=bool)
    closed = []
    for b in range(B):
        key = host[b].tobytes()
        safe = _SAFE_CELL_CACHE.get(key)
        if safe is None:
            safe = set(non_articulation_free_cells(host[b]))
            _SAFE_CELL_CACHE[key] = safe
        # drop the first/last few steps: start and goal are pinned by
        # apply_conditioning, so closing a cell under them is unanswerable.
        seen = {(int(a), int(c)) for a, c in zip(ii[b, 5:-5], jj[b, 5:-5])
                if 0 <= a < H and 0 <= c < W}
        cands = sorted(seen & safe)
        if not cands:
            closed.append(None)
            continue
        cell = cands[rng.randint(len(cands))]
        host[b][cell] = 1.0
        valid[b] = True
        closed.append(cell)

    return (torch.from_numpy(host).to(maze.device, maze.dtype),
            torch.from_numpy(valid).to(maze.device),
            closed)


def collision_penalty(xy_norm, maze, uv_scale, uv_shift, upsample=8,
                      maze_unit=4.0):
    """
    xy_norm : [B, T, 2] predicted positions in the shared normalizer's [-1, 1]
    maze    : [B, H, W] or [B, C, H, W] occupancy for each sample's own map
    returns : [B] mean penetration depth in WORLD units along each trajectory

    Mean rather than sum so the scale does not move when the horizon changes.
    """
    field = depth_fields(maze, upsample=upsample, maze_unit=maze_unit)
    grid = xy_norm * uv_scale + uv_shift                         # [B,T,2]
    grid = grid.unsqueeze(2)                                     # [B,T,1,2]
    sampled = F.grid_sample(field, grid, mode="bilinear",
                            padding_mode="border", align_corners=False)
    return sampled[:, 0, :, 0].mean(dim=1)
