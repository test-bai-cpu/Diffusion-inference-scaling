"""
distance_field_verifier_v2.py
=============================
Drop-in replacement for DistanceFieldVerifier that uses the corrected,
wall-respecting field from distance_field_v2.build_field (no cross-wall blur,
strictly-uphill wall fill, 4-connected BFS) and exposes a dead-end / stalled-
progress signal for DFS backtracking.

Same public interface as MazeVerifier / DistanceFieldVerifier:
    update_env(env)      -> (re)bind maze + invalidate cache
    get_guidance(x, ...) -> (B,) logprob or (B,T,4) gradient

Extra methods used by the search layer (MAFGS / DFS):
    progress_penalty(x, post_process)  -> (B,) stalled-progress cost
    deadend_mask()                     -> (H,W) bool dead-end pockets (cached)

Args (Arguments dataclass), with fallback defaults:
    dist_omega          scale on -distance cost                       (1.0)
    dist_mode           'sum'|'endpoint'|'weighted'|'monotonic'       ('sum')
    dist_iters          masked-smoothing Jacobi iterations            (8)
    dist_alpha          smoothing step                                (0.5)
    dist_wall_penalty   +cost per wall layer (keeps steps uphill)     (2.0)
    dist_progress_omega scale on stalled-progress penalty             (0.0=off)
"""
import numpy as np
import torch

try:
    from search.distance_field_v2 import (
        build_field, deadend_map, stalled_progress,
    )
    from search.distance_field import bilinear_sample_distance
except Exception:  # allow standalone import in the sandbox
    import importlib.util, os
    _HERE = os.path.dirname(os.path.abspath(__file__))
    def _load(fn, nm):
        spec = importlib.util.spec_from_file_location(nm, os.path.join(_HERE, fn))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    _df2 = _load('distance_field_v2.py', '_df2')
    _df1 = _load('distance_field.py', '_df1')
    build_field, deadend_map, stalled_progress = _df2.build_field, _df2.deadend_map, _df2.stalled_progress
    bilinear_sample_distance = _df1.bilinear_sample_distance


class DistanceFieldVerifierV2:
    def __init__(self, args):
        self.args         = args
        self.omega        = getattr(args, 'dist_omega', 1.0)
        self.mode         = getattr(args, 'dist_mode', 'sum')
        self.iters        = getattr(args, 'dist_iters', 8)
        self.alpha        = getattr(args, 'dist_alpha', 0.5)
        self.wall_penalty = getattr(args, 'dist_wall_penalty', 2.0)
        self.prog_omega   = getattr(args, 'dist_progress_omega', 0.0)
        self.device       = torch.device(getattr(args, 'device', 'cpu'))
        self.env = self.maze_map = None
        self.maze_unit = self.offset_x = self.offset_y = None
        self._cached_goal_ij = None
        self._D_tensor = None
        self._deadend = None

    # ---- interface ----------------------------------------------------
    def update_env(self, env):
        self.env      = env
        self.maze_map = env.maze_map.copy()
        self.maze_unit = env._maze_unit
        self.offset_x  = env._offset_x
        self.offset_y  = env._offset_y
        self._cached_goal_ij = None
        self._D_tensor = None
        self._deadend  = None

    def get_guidance(self, x, func=lambda x: x, post_process=lambda x: x,
                     return_logp=False, check_grad=True, **kwargs):
        if check_grad:
            assert x.requires_grad, "x must require grad"
        self._maybe_recompute()
        if self._D_tensor is None:
            return (torch.zeros(x.shape[0], device=x.device) if return_logp
                    else torch.zeros_like(x))
        x_world = post_process(func(x))
        xy_world = x_world[..., 2:4]
        distances = bilinear_sample_distance(
            self._D_tensor, xy_world, self.maze_unit, self.offset_x, self.offset_y)  # (B,T)
        cost = self._aggregate(distances)
        if self.prog_omega > 0:
            cost = cost + self.prog_omega * torch.clamp(
                distances[:, 1:] - distances[:, :-1], min=0.0).mean(dim=1)
        logp = -self.omega * cost
        if return_logp:
            return logp
        return torch.autograd.grad(logp.sum(), x, create_graph=False)[0]

    # ---- search-layer helpers ----------------------------------------
    def progress_penalty(self, x, post_process=lambda x: x):
        """(B,) mean positive goal-distance increment — DFS backtrack signal."""
        self._maybe_recompute()
        if self._D_tensor is None:
            return torch.zeros(x.shape[0], device=x.device)
        with torch.no_grad():
            xy = post_process(x)[..., 2:4]
            d = bilinear_sample_distance(self._D_tensor, xy, self.maze_unit,
                                         self.offset_x, self.offset_y)
            return torch.clamp(d[:, 1:] - d[:, :-1], min=0.0).mean(dim=1)

    def deadend_mask(self):
        if self._deadend is None and self.maze_map is not None:
            gij = self._cached_goal_ij
            self._deadend = deadend_map(self.maze_map, goal=gij)
        return self._deadend

    # ---- internal -----------------------------------------------------
    def _maybe_recompute(self):
        if self.env is None:
            return
        task_info = getattr(self.env, 'cur_task_info', None)
        if task_info is None:
            return
        gx, gy = task_info['goal_xy']
        gij = self._xy_to_ij(gx, gy)
        if gij == self._cached_goal_ij:
            return
        self._cached_goal_ij = gij
        F = build_field(self.maze_map, gij, iters=self.iters,
                        alpha=self.alpha, wall_penalty=self.wall_penalty)
        self._D_tensor = torch.tensor(F, dtype=torch.float32, device=self.device)\
                              .unsqueeze(0).unsqueeze(0)
        self._deadend = None  # recompute lazily for new goal

    def _xy_to_ij(self, x, y):
        mu = self.maze_unit
        return (int((y + self.offset_y + 0.5*mu)/mu),
                int((x + self.offset_x + 0.5*mu)/mu))

    def _aggregate(self, d):
        if self.mode == 'endpoint':
            return d[:, -1]
        if self.mode == 'weighted':
            T = d.shape[1]; w = torch.linspace(0,1,T,device=d.device); w = w/w.sum()
            return (d*w).sum(dim=1)
        if self.mode == 'monotonic':
            return torch.clamp(d[:,1:]-d[:,:-1], min=0.0).mean(dim=1)
        return d.mean(dim=1)  # 'sum'
