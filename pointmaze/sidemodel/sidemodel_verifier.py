"""
sidemodel_verifier.py — MCGN as an inference-time guidance term.
================================================================
Wraps a trained MapConditionedGuidanceNet so it plugs into the MAFGS composite
with the same get_guidance(x, return_logp=...) interface as the analytic
verifiers. For each waypoint it crops the local occupancy of the CURRENT maze,
queries the network for a predicted geodesic descent direction, and rewards
waypoint motion that aligns with it:

    logp  = sum_t  < step_t , predicted_dir_t >          (higher = better)
    grad  = d logp / d positions

Because the network conditions only on local crops + relative goal (never a map
id), the SAME checkpoint guides on unseen mazes. This term is ADDED to MAFGS and
still passes through the analytic feasibility gate, so a wrong prediction cannot
override corner/collision/dead-end rejection.

update_env(env) rebinds the maze occupancy grid + goal + coordinate transform.
"""
import os, sys, importlib.util
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(fn, nm):
    spec = importlib.util.spec_from_file_location(nm, os.path.join(_HERE, fn))
    m = importlib.util.module_from_spec(spec); sys.modules[nm] = m
    spec.loader.exec_module(m); return m


try:
    from sidemodel.model import MapConditionedGuidanceNet
    from sidemodel.dataset import occupancy_crop
except Exception:
    MapConditionedGuidanceNet = _load('model.py', '_mcgn_m').MapConditionedGuidanceNet
    occupancy_crop = _load('dataset.py', '_mcgn_d').occupancy_crop


class MCGNVerifier:
    def __init__(self, args=None, ckpt=None, weight=1.0):
        dev = getattr(args, 'device', 'cpu') if args is not None else 'cpu'
        self.device = torch.device(dev)
        self.weight = float(getattr(args, 'mcgn_weight', weight)) if args is not None else weight
        ckpt = ckpt or (getattr(args, 'mcgn_ckpt', None) if args is not None else None)
        self.patch = 15
        self.net = None
        if ckpt and os.path.exists(ckpt):
            blob = torch.load(ckpt, map_location=self.device)
            self.patch = int(blob.get('patch', 15))
            self.net = MapConditionedGuidanceNet(patch=self.patch).to(self.device)
            self.net.load_state_dict(blob['state_dict'])
            self.net.eval()
        self.maze_map = None

    def update_env(self, env):
        self.maze_map = np.asarray(env.maze_map, dtype=int)
        self.mu = float(env._maze_unit); self.ox = float(env._offset_x); self.oy = float(env._offset_y)
        gx, gy = env.cur_task_info['goal_xy']
        self.goal_ij = (int(round((gy + self.oy) / self.mu)),
                        int(round((gx + self.ox) / self.mu)))
        # precompute a per-free-cell predicted direction table (cheap, map-sized)
        self._dir_table = None
        if self.net is not None:
            self._build_dir_table()

    @torch.no_grad()
    def _build_dir_table(self):
        H, W = self.maze_map.shape
        gi, gj = self.goal_ij
        occ, cells = [], []
        for i in range(H):
            for j in range(W):
                if self.maze_map[i, j] == 0:
                    occ.append(occupancy_crop(self.maze_map, i, j, self.patch)[None])
                    cells.append((i, j))
        if not occ:
            self._dir_table = np.zeros((H, W, 2), np.float32); return
        occ = torch.tensor(np.asarray(occ), device=self.device)
        goal = []
        for (i, j) in cells:
            gx, gy = (gj - j), (gi - i)
            gn = (gx * gx + gy * gy) ** 0.5 + 1e-6
            goal.append([gx / gn, gy / gn, 0.0])           # log-dist unused at inference table
        pred = self.net(occ, torch.tensor(np.asarray(goal, np.float32), device=self.device))
        pred = (pred / (pred.norm(dim=-1, keepdim=True) + 1e-6)).cpu().numpy()
        tab = np.zeros((H, W, 2), np.float32)
        for k, (i, j) in enumerate(cells):
            tab[i, j] = pred[k]
        self._dir_table = tab

    def _dirs_at(self, pos):
        """pos (...,2) world-xy -> (...,2) predicted unit dir in WORLD xy."""
        H, W = self.maze_map.shape
        j = torch.round((pos[..., 0] + self.ox) / self.mu).long().clamp(0, W - 1)
        i = torch.round((pos[..., 1] + self.oy) / self.mu).long().clamp(0, H - 1)
        tab = torch.as_tensor(self._dir_table, device=pos.device)
        d = tab[i, j]                                       # (...,2) grid dir == world dir (axis aligned, +j=+x,+i=+y)
        return d

    def get_guidance(self, x, func=lambda x: x, post_process=lambda x: x,
                     return_logp=False, check_grad=True, **kwargs):
        if self.net is None or self._dir_table is None:
            # no model: contribute nothing (identity)
            if return_logp:
                return torch.zeros(x.shape[0], device=x.device)
            return torch.zeros_like(x)
        x = post_process(func(x))
        pos = x[..., 2:4]                                   # (B,T,2)
        with torch.no_grad():
            dirs = self._dirs_at(pos)                       # (B,T,2)
        steps = pos[:, 1:, :] - pos[:, :-1, :]              # (B,T-1,2)
        align = (steps * dirs[:, :-1, :]).sum(-1)           # (B,T-1)
        logp = self.weight * align.sum(dim=tuple(range(1, align.ndim)))
        if return_logp:
            return logp
        grad = torch.autograd.grad(logp.mean(), x)[0]
        return grad
