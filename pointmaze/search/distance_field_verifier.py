import numpy as np
import torch

from search.distance_field import (
    compute_distance_field,
    smooth_distance_field,
    bilinear_sample_distance,
)


class DistanceFieldVerifier:
    """
    Inference-time verifier using a BFS distance-to-goal field.

    Mirrors MazeVerifier's interface so it can be plugged in anywhere
    MazeVerifier is used (including inside CompositeVerifier).

    The BFS field is computed lazily: the first time get_guidance is called
    for a new goal cell, we run BFS and cache the result.  Subsequent calls
    with the same goal reuse the cached tensor.

    Args are read from the Arguments dataclass:
        dist_omega      – scale applied to -distance cost to produce logprob
        dist_mode       – how the per-timestep distances are aggregated:
                          'endpoint' | 'sum' | 'weighted' | 'monotonic'
        dist_smooth_sigma – Gaussian blur std dev on the raw BFS field
        dist_connectivity – 4 or 8 for BFS
    """

    def __init__(self, args):
        self.args = args
        self.omega       = getattr(args, 'dist_omega', 1.0)
        self.mode        = getattr(args, 'dist_mode', 'sum')
        self.sigma       = getattr(args, 'dist_smooth_sigma', 0.5)
        self.connectivity = getattr(args, 'dist_connectivity', 4)
        self.device      = torch.device(args.device)

        # Set by update_env
        self.env      = None
        self.maze_map = None
        self.maze_unit  = None
        self.offset_x   = None
        self.offset_y   = None

        # Cache: keyed on goal_ij tuple
        self._cached_goal_ij = None
        self._D_tensor       = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_env(self, env):
        """Call whenever the environment (or maze layout) changes."""
        self.env      = env
        self.maze_map = env.maze_map.copy()
        self.maze_unit = env._maze_unit
        self.offset_x  = env._offset_x
        self.offset_y  = env._offset_y
        # Invalidate cache — maze topology may have changed
        self._cached_goal_ij = None
        self._D_tensor       = None

    def get_guidance(self, x, func=lambda x: x, post_process=lambda x: x,
                     return_logp=False, check_grad=True, **kwargs):
        """
        Compute distance-field logprob or gradient, matching MazeVerifier's signature.

        Args:
            x:            (B, T, 4) trajectory in NORMALIZED [-1, 1] space
            func:         optional pre-transform (identity by default)
            post_process: unnormalize callback — maps normalized (B, T, 4) → world (B, T, 4)
            return_logp:  if True return (B,) logprob; else return (B, T, 4) gradient
            check_grad:   if True, assert x.requires_grad
            **kwargs:     forwarded from guide_step (contains 'cond', etc.)

        Returns:
            logprob (B,) or gradient (B, T, 4)
        """
        if check_grad:
            assert x.requires_grad, "x must require grad for gradient computation"

        self._maybe_recompute_bfs()

        if self._D_tensor is None:
            # BFS not yet available (env not set or goal unknown)
            if return_logp:
                return torch.zeros(x.shape[0], device=x.device)
            return torch.zeros_like(x)

        # 1. Unnormalize to world coordinates
        x_world = post_process(func(x))            # (B, T, 4)

        # 2. Extract (x, y) positions from observation dims
        xy_world = x_world[..., 2:4]              # (B, T, 2)

        # 3. Sample distance field at each trajectory position
        distances = bilinear_sample_distance(
            self._D_tensor, xy_world,
            self.maze_unit, self.offset_x, self.offset_y,
        )                                          # (B, T)

        # 4. Aggregate distances into a per-particle cost
        cost = self._aggregate(distances)          # (B,)

        # 5. logprob = -omega * cost  (high distance = bad = low logprob)
        logp = -self.omega * cost                  # (B,)

        if return_logp:
            return logp

        # 6. Gradient w.r.t. the original normalized input x
        grad_out = torch.autograd.grad(logp.sum(), x, create_graph=False)[0]
        return grad_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_recompute_bfs(self):
        """Recompute BFS if the goal cell has changed."""
        if self.env is None:
            return
        task_info = getattr(self.env, 'cur_task_info', None)
        if task_info is None:
            return

        goal_xy = task_info['goal_xy']
        goal_ij = self._xy_to_ij(goal_xy[0], goal_xy[1])

        if goal_ij == self._cached_goal_ij:
            return  # Cache is still valid

        self._cached_goal_ij = goal_ij
        D      = compute_distance_field(self.maze_map, goal_ij, connectivity=self.connectivity)
        D_fin  = smooth_distance_field(D, sigma=self.sigma, inf_replace=1000.0)
        self._D_tensor = (
            torch.tensor(D_fin, dtype=torch.float32, device=self.device)
            .unsqueeze(0).unsqueeze(0)              # (1, 1, H, W)
        )

    def _xy_to_ij(self, x, y):
        """World (x, y) → integer cell (i, j), matching env.xy_to_ij."""
        mu = self.maze_unit
        i = int((y + self.offset_y + 0.5 * mu) / mu)
        j = int((x + self.offset_x + 0.5 * mu) / mu)
        return (i, j)

    def _aggregate(self, distances: torch.Tensor) -> torch.Tensor:
        """Aggregate (B, T) per-timestep distances into (B,) cost.

        Conditioning via apply_conditioning pins t=0 to start and t=T-1 to goal,
        so distances[:, -1] ≈ 0 always — 'endpoint' is only useful when the
        conditioning is NOT enforced (e.g. diagnostic runs).

        'sum' (mean per timestep) is the recommended default: it penalises
        trajectories that spend time far from the goal without exploding the
        logprob magnitude relative to the DFS wall threshold.
        """
        if self.mode == 'endpoint':
            return distances[:, -1]
        if self.mode == 'sum':
            # Mean over timesteps so the scale is comparable to wall logprob
            return distances.mean(dim=1)
        if self.mode == 'weighted':
            # Linearly increasing weight 0→1; normalised to sum to 1
            T = distances.shape[1]
            w = torch.linspace(0.0, 1.0, T, device=distances.device)
            w = w / w.sum()
            return (distances * w).sum(dim=1)
        if self.mode == 'monotonic':
            # Mean of positive increments (going away from goal)
            diffs = torch.clamp(distances[:, 1:] - distances[:, :-1], min=0.0)
            return diffs.mean(dim=1)
        # Fallback
        return distances.mean(dim=1)
