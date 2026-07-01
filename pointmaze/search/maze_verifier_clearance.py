"""
maze_verifier_clearance.py
==========================
Corner-safe, clearance-aware collision verifier for the pointmaze search.

Why replace maze_verifier.MazeVerifier
--------------------------------------
The original `batched_inside_wall_loss_non_adjacent` has three defects that let
the planner cut corners the MuJoCo simulator actually blocks:

  1. IS_INSIDE GATE. The per-wall distance is forced to 0 for any point NOT
     strictly inside the wall box. A waypoint sitting in the diagonal gap between
     two walls that touch at a corner (world (18,18) in the giant maze) is inside
     neither box, so its collision loss is exactly 0 -- the verifier calls it
     free while the simulator blocks it.
  2. NO CLEARANCE MARGIN. The point mass has finite size and MuJoCo blocks it
     when it overlaps an inflated wall. A trajectory grazing a wall face at
     distance 0 reads loss 0.
  3. max(dim=-1) OVER WALLS + point-only. Only the single worst wall contributes,
     and only at the sampled waypoints -- a segment between two "free" waypoints
     can still pass straight through a wall or a corner.

This module fixes all three:

  * EXACT SIGNED DISTANCE to each axis-aligned wall box (negative inside,
    positive outside) -- no is_inside gate, defined everywhere.
  * CLEARANCE MARGIN `margin`: hinge penalty relu(margin - sdf) so grazing and
    corner-pinch points are penalized. margin < half the corridor width (2.0) so
    centerline travel down a legal 4-wide corridor stays free.
  * SUM of squared hinges over walls (a corner pinched by two walls gets both) +
    an explicit diagonal-corner term.
  * SEGMENT SUPERSAMPLING: penalize interpolated points between consecutive
    waypoints, so a straight jump across a wall/corner is caught.

Drop-in: `CornerSafeMazeVerifier` exposes the same update_env / get_guidance
signature as MazeVerifier; register it wherever MazeVerifier is constructed.
"""
import torch


# --------------------------------------------------------------------------- #
# geometry (pure torch, differentiable a.e.)
# --------------------------------------------------------------------------- #
def box_sdf(pos, centers, halfs):
    """
    Exact signed distance from points to axis-aligned boxes.

    pos     : (..., 2)
    centers : (W, 2)   box centers
    halfs   : (W, 2)   box half-extents
    returns : (..., W) signed distance (negative inside, 0 on face, positive out)
    """
    # broadcast points against walls: (..., 1, 2) vs (W, 2)
    d = torch.abs(pos.unsqueeze(-2) - centers) - halfs           # (..., W, 2)
    outside = torch.linalg.norm(torch.clamp(d, min=0.0), dim=-1)  # (..., W)
    inside = torch.clamp(torch.amax(d, dim=-1), max=0.0)          # (..., W)
    return outside + inside


def clearance_loss(pos, centers, halfs, margin):
    """
    Sum of squared clearance hinges over walls: sum_w relu(margin - sdf_w)^2.
    pos: (..., 2) -> loss: (...)  (reduced over walls, not over batch/time).
    """
    sdf = box_sdf(pos, centers, halfs)             # (..., W)
    hinge = torch.clamp(margin - sdf, min=0.0)     # (..., W)
    return (hinge ** 2).sum(dim=-1)                # (...)


def segment_clearance_loss(positions, centers, halfs, margin, n_sub=4):
    """
    Clearance loss on waypoints AND on interpolated segment samples so a straight
    jump between two waypoints cannot tunnel through a wall or corner.

    positions : (B, T, 2)
    returns   : (B, T) per-waypoint loss where entry t also carries the segment
                (t -> t+1) samples (last waypoint keeps its point loss).
    """
    B, T, _ = positions.shape
    point_loss = clearance_loss(positions, centers, halfs, margin)   # (B, T)
    if T < 2 or n_sub <= 1:
        return point_loss
    p0 = positions[:, :-1, :]              # (B, T-1, 2)
    p1 = positions[:, 1:, :]              # (B, T-1, 2)
    ts = torch.linspace(0.0, 1.0, n_sub + 1, device=positions.device)[1:-1]  # interior
    seg = torch.zeros((B, T - 1), device=positions.device)
    for a in ts:
        mid = p0 + (p1 - p0) * a          # (B, T-1, 2)
        seg = seg + clearance_loss(mid, centers, halfs, margin)
    out = point_loss.clone()
    out[:, :-1] = out[:, :-1] + seg
    return out


# --------------------------------------------------------------------------- #
# wall extraction (vectorised; ball inflation applied for every maze type)
# --------------------------------------------------------------------------- #
def get_wall_tensors(env, device, ball_radius=0.0):
    """
    Return (centers, halfs, corner_pairs) as torch tensors.
    centers : (W, 2)   world-xy centers of wall cells
    halfs   : (W, 2)   half-extents (maze_unit/2 + ball_radius)
    corner_pairs : (P, 2) world-xy points where two walls touch only diagonally
                   (the pinch points that let a path slip through).
    """
    mm = env.maze_map
    H, W = mm.shape
    half = env._maze_unit / 2.0 + ball_radius
    centers, halfs = [], []
    wall_ij = []
    for i in range(H):
        for j in range(W):
            if mm[i, j] == 1:
                cx = j * env._maze_unit - env._offset_x
                cy = i * env._maze_unit - env._offset_y
                centers.append((cx, cy))
                halfs.append((half, half))
                wall_ij.append((i, j))
    wall_set = set(wall_ij)

    # diagonal-corner pinch points: (i,j) wall and (i+di,j+dj) wall touch only at
    # a corner when the two orthogonal cells between them are BOTH free.
    corners = set()
    for (i, j) in wall_ij:
        for di, dj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            ni, nj = i + di, j + dj
            if (ni, nj) in wall_set:
                oi, oj = (i, nj), (ni, j)   # the two orthogonal neighbours
                free_o1 = (0 <= oi[0] < H and 0 <= oi[1] < W and mm[oi] == 0)
                free_o2 = (0 <= oj[0] < H and 0 <= oj[1] < W and mm[oj] == 0)
                if free_o1 and free_o2:
                    # shared corner in world xy = between the two wall centers
                    ci = (i + ni) / 2.0
                    cj = (j + nj) / 2.0
                    px = cj * env._maze_unit - env._offset_x
                    py = ci * env._maze_unit - env._offset_y
                    corners.add((round(px, 4), round(py, 4)))

    centers = torch.tensor(centers, dtype=torch.float32, device=device) if centers \
        else torch.zeros((0, 2), device=device)
    halfs = torch.tensor(halfs, dtype=torch.float32, device=device) if halfs \
        else torch.zeros((0, 2), device=device)
    corner_pairs = torch.tensor(sorted(corners), dtype=torch.float32, device=device) if corners \
        else torch.zeros((0, 2), device=device)
    return centers, halfs, corner_pairs


def corner_gap_loss(positions, corner_pts, radius):
    """
    Explicit penalty for waypoints within `radius` of a diagonal pinch point.
    positions: (B, T, 2), corner_pts: (P, 2) -> (B, T).
    Strong, isotropic well exactly at the gap that clearance alone barely closes.
    """
    if corner_pts.shape[0] == 0:
        return torch.zeros(positions.shape[:2], device=positions.device)
    d = torch.linalg.norm(positions.unsqueeze(-2) - corner_pts, dim=-1)  # (B,T,P)
    hinge = torch.clamp(radius - d, min=0.0)
    return (hinge ** 2).sum(dim=-1)


# --------------------------------------------------------------------------- #
# verifier (drop-in for MazeVerifier)
# --------------------------------------------------------------------------- #
try:
    from search.utils import check_grad_fn, rescale_grad
    from search.configs import Arguments
except Exception:  # allow import in a bare sandbox for geometry tests
    Arguments = object
    def check_grad_fn(x):  # noqa
        return None
    def rescale_grad(g, clip_scale=1.0, **kw):  # noqa
        return g


class CornerSafeMazeVerifier:
    """
    Clearance-aware, corner-safe collision guidance.

    margin       : clearance distance in world units (default 0.7, < corridor
                   half-width 2.0). Points/segments closer than this to any wall
                   face are penalized.
    ball_radius  : wall inflation applied to every maze type (MuJoCo blocks the
                   inflated ball). Effective wall half-extent = maze_unit/2 + this.
    n_sub        : segment supersampling density (>=2 enables segment check).
    corner_radius: radius of the explicit diagonal-pinch penalty well.
    """
    def __init__(self, args=None, margin=0.7, ball_radius=0.5, n_sub=4,
                 corner_radius=0.9, corner_weight=1.0):
        dev = getattr(args, "device", "cpu") if args is not None else "cpu"
        self.device = torch.device(dev)
        self.margin = float(margin)
        self.ball_radius = float(ball_radius)
        self.n_sub = int(n_sub)
        self.corner_radius = float(corner_radius)
        self.corner_weight = float(corner_weight)
        self.centers = self.halfs = self.corner_pts = None

    def update_env(self, env):
        self.env = env
        self.centers, self.halfs, self.corner_pts = get_wall_tensors(
            env, self.device, ball_radius=self.ball_radius)

    def collision_loss(self, positions):
        """(B,T,2) world-xy -> (B,T) nonnegative collision loss (no is_inside gate)."""
        seg = segment_clearance_loss(positions, self.centers, self.halfs,
                                     self.margin, n_sub=self.n_sub)
        cor = corner_gap_loss(positions, self.corner_pts, self.corner_radius)
        return seg + self.corner_weight * cor

    def get_guidance(self, x, func=lambda x: x, post_process=lambda x: x,
                     return_logp=False, check_grad=True, **kwargs):
        if check_grad:
            check_grad_fn(x)
        x = post_process(func(x))
        x = x[..., 2:4]                       # x, y coordinates only
        loss = self.collision_loss(x)         # (B, T)
        log_probs = loss * -1
        if return_logp:
            return log_probs.sum(dim=tuple(range(1, log_probs.ndim)))
        grad = torch.autograd.grad(log_probs.mean(), x)[0]
        return rescale_grad(grad, clip_scale=1.0, **kwargs)
