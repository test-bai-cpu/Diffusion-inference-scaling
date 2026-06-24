import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from scipy.ndimage import gaussian_filter


def compute_distance_field(maze_grid: np.ndarray, goal_cell: tuple,
                           connectivity: int = 4) -> np.ndarray:
    """
    BFS from goal_cell over free cells (0=free, 1=wall). Returns distance in cells.
    Walls and unreachable free cells receive np.inf.

    Args:
        maze_grid: (H, W) array, 1=wall, 0=free
        goal_cell: (gi, gj) grid indices of the goal
        connectivity: 4 (up/down/left/right) or 8 (includes diagonals)

    Returns:
        D: (H, W) float array, inf for walls/unreachable
    """
    H, W = maze_grid.shape
    D = np.full((H, W), np.inf, dtype=np.float32)
    gi, gj = goal_cell

    if maze_grid[gi, gj] == 1:
        return D  # goal is inside a wall

    D[gi, gj] = 0.0
    queue = deque([(gi, gj)])

    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        i, j = queue.popleft()
        cur_d = D[i, j]
        for di, dj in deltas:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and maze_grid[ni, nj] == 0 and D[ni, nj] == np.inf:
                D[ni, nj] = cur_d + 1.0
                queue.append((ni, nj))

    return D


def smooth_distance_field(D: np.ndarray, sigma: float = 0.5,
                          inf_replace: float = 1000.0) -> np.ndarray:
    """
    Replace inf with a large finite value, then optionally apply Gaussian blur.
    The blur helps produce smoother gradients at cell boundaries.

    Args:
        D: (H, W) distance array with possible inf values
        sigma: std dev for Gaussian blur (0 = no blur)
        inf_replace: value to substitute for inf before blurring

    Returns:
        D_out: (H, W) finite float array
    """
    D_out = D.copy()
    D_out[np.isinf(D_out)] = inf_replace
    if sigma > 0:
        D_out = gaussian_filter(D_out.astype(np.float64), sigma=sigma).astype(np.float32)
    return D_out


def world_to_normalized_grid(xy_world: torch.Tensor, maze_unit: float,
                              offset_x: float, offset_y: float,
                              grid_shape: tuple) -> torch.Tensor:
    """
    Convert world (x, y) coordinates to normalized grid coords for F.grid_sample.

    OGBench coordinate convention:
        i_float = (y + offset_y + 0.5*maze_unit) / maze_unit   (row, height axis)
        j_float = (x + offset_x + 0.5*maze_unit) / maze_unit   (col, width axis)

    F.grid_sample uses align_corners=True where:
        grid[..., 0] = x_norm  maps to the WIDTH  dimension (columns j)
        grid[..., 1] = y_norm  maps to the HEIGHT dimension (rows i)
        -1 → index 0,  +1 → index W-1 or H-1

    Args:
        xy_world: (..., 2) tensor of (x, y) world coordinates
        maze_unit, offset_x, offset_y: maze parameters from env
        grid_shape: (H, W)

    Returns:
        norm_grid: (..., 2) tensor of (x_norm, y_norm) in [-1, 1]
    """
    H, W = grid_shape
    x = xy_world[..., 0]
    y = xy_world[..., 1]

    # Cell center (i, j) is at world (j*maze_unit - offset_x, i*maze_unit - offset_y),
    # so the continuous inverse is just (coord + offset) / maze_unit.
    # Do NOT add 0.5*maze_unit here — that half-cell shift is only for integer floor-division.
    j_float = (x + offset_x) / maze_unit  # column
    i_float = (y + offset_y) / maze_unit  # row

    x_norm = j_float / (W - 1) * 2.0 - 1.0
    y_norm = i_float / (H - 1) * 2.0 - 1.0

    return torch.stack([x_norm, y_norm], dim=-1)


def bilinear_sample_distance(D_tensor: torch.Tensor, traj_xy_world: torch.Tensor,
                              maze_unit: float, offset_x: float,
                              offset_y: float) -> torch.Tensor:
    """
    Differentiably sample the distance field at trajectory positions.

    Args:
        D_tensor: (1, 1, H, W) precomputed distance field (no inf values)
        traj_xy_world: (B, T, 2) trajectory in world (x, y) coordinates
        maze_unit, offset_x, offset_y: maze coordinate parameters

    Returns:
        distances: (B, T) sampled distance values, differentiable w.r.t. traj_xy_world
    """
    B, T, _ = traj_xy_world.shape
    H, W = D_tensor.shape[2], D_tensor.shape[3]

    # Build normalized grid: (B, T, 2)
    grid = world_to_normalized_grid(traj_xy_world, maze_unit, offset_x, offset_y, (H, W))

    # F.grid_sample expects (N, H_out, W_out, 2); treat T as H_out, 1 as W_out
    grid_4d = grid.unsqueeze(2)  # (B, T, 1, 2)

    # Expand D_tensor to match batch size
    D_exp = D_tensor.expand(B, 1, H, W)

    # Sample: output (B, 1, T, 1)
    sampled = F.grid_sample(D_exp, grid_4d, mode='bilinear',
                            padding_mode='border', align_corners=True)

    return sampled[:, 0, :, 0]  # (B, T)
