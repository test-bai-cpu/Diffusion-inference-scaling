"""
mafgs.py — Map-Aware Feasibility-Guided Search
==============================================
A search method for the pointmaze diffusion planner that is (a) map-aware, so it
transfers to unseen maze layouts, and (b) explicitly targets the two failure
modes of the baseline: diagonal corner-cutting and dead-end trapping.

Three pieces, all reading the CURRENT maze geometry (not a fixed training map):

  GUIDANCE  (drives the diffusion gradient, like TFG/DFS' local_search):
      map-aware composite verifier =
          CornerSafeMazeVerifier   (clearance + diagonal-pinch repulsion)
        + DistanceFieldVerifierV2  (wall-respecting goal-distance descent)
      weighted by (mafgs_clear_weight, mafgs_field_weight). This is the guidance
      logprob whose gradient nudges waypoints toward feasible, goal-descending
      paths on whatever map update_env last received.

  FEASIBILITY GATE  (hard accept/reject, on top of the logprob threshold):
      A candidate is INFEASIBLE — force a backtrack regardless of logprob — if
        * any corner-gap penalty fires   (corner-cut through a diagonal pinch), OR
        * segment clearance loss exceeds mafgs_collision_tol (wall grazing /
          tunneling between waypoints), OR
        * stalled-progress on the goal-distance field exceeds mafgs_stall_tol
          (the rollout stops descending / reverses -> dead-end trap).
      This is what makes MAFGS reject the specific defects the baseline accepts.

  SEARCH  (budgeted backtracking, DFS-style):
      Reuses the DFS recur-depth backtracking loop. At each evaluation step it
      keeps a top-k buffer of candidate x_prev keyed by logprob; on an infeasible
      or low-logprob candidate it spends budget to re-noise back `recur_depth`
      steps and resample, and when budget is exhausted it falls back to the best
      buffered candidate. Beam width = per_sample_batch_size (best-first over the
      batch); depth = recur_depth; all under the shared `budget`.

Registered as --method mafgs (see script_utils.get_pipe / get_args).
"""
import torch
from typing import Tuple
from search.methods.base_guidance import BaseGuidance
from search.methods.tfg import TFGGuidance
from search.configs import Arguments
from diffuser.models.helpers import apply_conditioning

from search.maze_verifier_clearance import CornerSafeMazeVerifier
from search.distance_field_verifier_v2 import DistanceFieldVerifierV2


class _MapAwareComposite:
    """Clearance + goal-distance-field composite with a feasibility read-out.

    get_guidance() matches the verifier interface (summed weighted logprob or
    gradient). feasibility() returns per-sample hard-constraint diagnostics used
    by the gate."""

    def __init__(self, args):
        self.args = args
        self.clear = CornerSafeMazeVerifier(
            args,
            margin=getattr(args, 'mafgs_margin', 0.7),
            ball_radius=getattr(args, 'mafgs_ball_radius', 0.5),
            n_sub=getattr(args, 'mafgs_n_sub', 4),
            corner_radius=getattr(args, 'mafgs_corner_radius', 0.9),
            corner_weight=getattr(args, 'mafgs_corner_weight', 1.0),
        )
        self.field = DistanceFieldVerifierV2(args)
        self.w_clear = getattr(args, 'mafgs_clear_weight', 1.0)
        self.w_field = getattr(args, 'mafgs_field_weight', 1.0)
        self.collision_tol = getattr(args, 'mafgs_collision_tol', 0.05)
        self.stall_tol     = getattr(args, 'mafgs_stall_tol', 0.10)
        self.deadend_tol   = getattr(args, 'mafgs_deadend_tol', 0.15)
        # Optional learned map-conditioned side model (MCGN). Added to the
        # guidance signal ONLY when a checkpoint is configured; it never enters
        # the feasibility gate, so a bad prediction cannot cut a corner or enter
        # a dead-end that the analytic gate would reject. Absent a checkpoint the
        # verifier contributes exactly zero (identity), so plain MAFGS is
        # unchanged.
        self.mcgn = None
        mcgn_ckpt = getattr(args, 'mcgn_ckpt', '') or ''
        if mcgn_ckpt:
            try:
                from sidemodel.sidemodel_verifier import MCGNVerifier
                self.mcgn = MCGNVerifier(args, ckpt=mcgn_ckpt,
                                         weight=getattr(args, 'mcgn_weight', 0.5))
            except Exception as e:
                print(f"[MAFGS] MCGN side model unavailable ({e}); "
                      f"continuing with analytic guidance only.")
                self.mcgn = None

    def update_env(self, env):
        self.clear.update_env(env)
        self.field.update_env(env)
        if self.mcgn is not None:
            self.mcgn.update_env(env)

    def get_guidance(self, x, return_logp=False, **kwargs):
        lc = self.clear.get_guidance(x, return_logp=return_logp, **kwargs)
        lf = self.field.get_guidance(x, return_logp=return_logp, **kwargs)
        g = self.w_clear * lc + self.w_field * lf
        if self.mcgn is not None:
            g = g + self.mcgn.get_guidance(x, return_logp=return_logp, **kwargs)
        return g

    def acceptance_logp(self, x, **kwargs):
        """Clearance-only logprob for the DFS soft threshold. The goal-field's
        ABSOLUTE distance is always large early in a rollout, so folding it into
        a fixed threshold would conflate 'far from goal' with 'bad path'. The
        goal-field drives the gradient (get_guidance) and the hard gate (stall /
        dead-end) instead."""
        return self.w_clear * self.clear.get_guidance(x, return_logp=True, **kwargs)

    @torch.no_grad()
    def feasibility(self, x, post_process=lambda x: x, **kwargs):
        """Return (infeasible_mask (B,), diag dict) for a batch of trajectories."""
        xw = post_process(x)
        pos = xw[..., 2:4]                                   # (B,T,2)
        # corner-cut: any diagonal-pinch penalty firing
        from search.maze_verifier_clearance import corner_gap_loss, segment_clearance_loss
        corner = corner_gap_loss(pos, self.clear.corner_pts, self.clear.corner_radius)  # (B,T)
        corner_hit = corner.sum(dim=tuple(range(1, corner.ndim)))
        # collision / tunneling
        seg = segment_clearance_loss(pos, self.clear.centers, self.clear.halfs,
                                     self.clear.margin, n_sub=self.clear.n_sub)  # (B,T)
        collision = seg.sum(dim=tuple(range(1, seg.ndim)))
        # stalled progress on the goal-distance field
        stall = self.field.progress_penalty(x, post_process=post_process)         # (B,)
        # dead-end occupancy: fraction of waypoints inside leaf-pruned pockets
        # (undiluted trap detector — robust to a short detour in a long rollout)
        deadend = self._deadend_occupancy(pos)                                    # (B,)
        infeasible = ((corner_hit > 0) | (collision > self.collision_tol)
                      | (stall > self.stall_tol) | (deadend > self.deadend_tol))
        return infeasible, {'corner': corner_hit, 'collision': collision,
                            'stall': stall, 'deadend': deadend}

    @torch.no_grad()
    def _deadend_occupancy(self, pos):
        """(B,T,2) world-xy -> (B,) fraction of waypoints in dead-end pockets."""
        dmask = self.field.deadend_mask()
        if dmask is None:
            return torch.zeros(pos.shape[0], device=pos.device)
        mu = self.field.maze_unit; ox = self.field.offset_x; oy = self.field.offset_y
        H, W = dmask.shape
        j = torch.round((pos[..., 0] + ox) / mu).long().clamp(0, W - 1)
        i = torch.round((pos[..., 1] + oy) / mu).long().clamp(0, H - 1)
        dm = torch.as_tensor(dmask, dtype=torch.bool, device=pos.device)
        occ = dm[i, j].float()                                                    # (B,T)
        return occ.mean(dim=tuple(range(1, occ.ndim)))


class MAFGSGuidance(BaseGuidance):
    def __init__(self, args: Arguments, **kwargs):
        super().__init__(args, **kwargs)
        self.local_search = TFGGuidance(args, **kwargs)
        # CRITICAL: TFGGuidance builds its own plain MazeVerifier, so the
        # denoise-time gradient would ignore the map. Share our map-aware
        # composite so the guidance gradient is clearance + goal-field on the
        # CURRENT maze.
        self.local_search.guider = self.guider
        self.reset()

    # replace the default MazeVerifier guider with the map-aware composite
    @staticmethod
    def _build_verifier(args):
        return _MapAwareComposite(args)

    def update_env(self, env):
        super().update_env(env)
        self.local_search.update_env(env)

    def reset(self, **kwargs):
        self.budget = self.args.budget
        self.buffer = [{} for _ in range(self.args.inference_steps)]

    def get_threshold(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.threshold_schedule == 'decrease':
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.threshold_schedule == 'increase':
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        else:
            scheduler = torch.ones_like(alpha_prod_ts)
        return self.args.threshold / (scheduler[t] * len(scheduler) / scheduler.sum())

    def evaluation_steps(self, **kwargs):
        return list(range(self.args.start_step, self.args.inference_steps, self.args.step_size))

    def guide_step(self, x, i, unet, ts, alpha_prod_ts, alpha_prod_t_prevs, eta, **kwargs) -> Tuple[torch.Tensor, dict]:
        # 1) local guided denoise step (map-aware gradient lives in self.guider,
        #    which TFG's local_search shares because both were built from args)
        x_prev, extra = self.local_search.guide_step(
            x, i, unet, ts, alpha_prod_ts, alpha_prod_t_prevs, eta, **kwargs)
        x0 = extra["x0"]

        # soft-threshold score: clearance-only (feasibility-oriented), NOT the
        # goal-field absolute distance (which is large early and would reject
        # perfectly good far-from-goal trajectories).
        logprobs = self.guider.acceptance_logp(x0, check_grad=False, **kwargs)

        accept = True
        if i in self.evaluation_steps():
            # feasibility gate on the predicted clean trajectory
            # (post_process=self.unnormalize arrives inside kwargs from the policy)
            infeasible, diag = self.guider.feasibility(x0, **kwargs)
            any_infeasible = bool(infeasible.any().item())

            self.buffer[i][logprobs.sum().item()] = x_prev
            over_thr = (-logprobs.sum() > self.get_threshold(i, alpha_prod_ts, alpha_prod_t_prevs))
            reject = (over_thr or any_infeasible)

            if reject and self.budget > 0:
                accept = False
                self.budget -= 1
            elif reject and self.budget == 0:
                # budget exhausted: fall back to best buffered candidate
                accept = True
                x_prev = self.buffer[i][max(self.buffer[i].keys())]
            else:
                accept = True

        if accept:
            return x_prev, {"i_next": i + 1}

        # 2) backtrack: re-noise back recur_depth steps and resample
        cond = kwargs['cond']
        next_noise_level = max(0, i - self.args.recur_depth)
        if next_noise_level == 0:
            x = torch.randn_like(x)
            x = apply_conditioning(x, cond, 2)
            return x, {"i_next": next_noise_level}
        alpha_prod_t = alpha_prod_ts[next_noise_level]
        alpha_prod_t_prev = alpha_prod_t_prevs[i]
        x = self._predict_xt(x_prev, alpha_prod_t, alpha_prod_t_prev, **kwargs).detach().requires_grad_(False)
        x = apply_conditioning(x, cond, 2)
        return x, {"i_next": next_noise_level}
