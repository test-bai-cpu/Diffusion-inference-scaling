from search.methods.tfg import TFGGuidance
from search.methods.base_guidance import BaseGuidance
from diffuser.models.helpers import apply_conditioning
from search.configs import Arguments
import torch
from typing import Tuple
from torch.autograd import grad
from search.utils import rescale_grad
import math


class DFSGuidance(BaseGuidance):
    def __init__(self, args:Arguments, **kwargs):
        super().__init__(args, **kwargs)
        self.local_search = TFGGuidance(args, **kwargs)
        self.reset()


    def get_threshold(self, t, alpha_prod_ts, alpha_prod_t_prevs):
        if self.args.threshold_schedule == 'decrease':    # beta_t
            scheduler = 1 - alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.threshold_schedule == 'increase':  # alpha_t
            scheduler = alpha_prod_ts / alpha_prod_t_prevs
        elif self.args.threshold_schedule == 'constant':  # 1
            scheduler = torch.ones_like(alpha_prod_ts)
        return self.args.threshold / (scheduler[t] * len(scheduler) / scheduler.sum())  # to be tested

    def evaluation_steps(self, **kwargs):
        start = self.args.start_step
        step = self.args.step_size
        end = self.args.inference_steps
        return list(range(start, end, step))

    def update_env(self, env):
        super().update_env(env)
        self.local_search.update_env(env)

    def reset(self, **kwargs):
        self.budget = self.args.budget
        self.buffer = [{} for _ in range(self.args.inference_steps)]
        self._monitor_count = 0

    def _verifier_stats(self):
        stats = getattr(self.guider, "last_stats", {})
        if "MazeVerifier" in stats:
            return stats["MazeVerifier"]
        return stats

    def _violation_cost(self, total_cost):
        """
        Cost used for the accept/reject THRESHOLD test.

        With a CompositeVerifier (distance field on), the total cost carries a
        nonzero "route preference" floor from DistanceFieldVerifier even for
        perfectly legal trajectories, so thresholding on it never accepts and
        the budget is always exhausted. Threshold on the MazeVerifier
        (violation) component only; ranking in the buffer / forced_best still
        uses the combined total, so the distance preference keeps deciding
        WHICH candidate wins -- it just no longer decides WHETHER a clean
        candidate may be accepted early.

        Falls back to total_cost when the guider is a bare MazeVerifier
        (distance field off), which is bit-identical to the old behaviour.
        """
        stats = getattr(self.guider, "last_stats", {})
        maze = stats.get("MazeVerifier")
        if isinstance(maze, dict) and "weighted_logp_sum" in maze:
            return -maze["weighted_logp_sum"]
        return float(total_cost)

    def _print_monitor(self, i, t, cost, threshold, accept, forced_best=False,
                       violation=None):
        if not getattr(self.args, "verifier_monitor", False):
            return
        freq = max(1, int(getattr(self.args, "verifier_monitor_freq", 1)))
        self._monitor_count += 1
        if (self._monitor_count - 1) % freq != 0:
            return

        stats = self._verifier_stats()
        wall = stats.get("wall_cost_mean", float("nan"))
        transition = stats.get("corner_transition_mean", float("nan"))
        transition_raw = stats.get("corner_transition_raw_mean", float("nan"))
        cell_raw = stats.get("corner_cell_transition_raw_mean", float("nan"))
        zone_raw = stats.get("corner_zone_transition_raw_mean", float("nan"))
        wall_hit = stats.get("wall_hit_frac", float("nan"))
        transition_hit = stats.get("corner_transition_hit_frac", float("nan"))
        n_corners = stats.get("n_forbidden_corners", -1)
        radius = stats.get("corner_radius", float("nan"))
        reason = "forced_best" if forced_best else ("accept" if accept else "reject")
        viol = float(cost) if violation is None else float(violation)
        print(
            "[verifier] step=%02d t=%d cost=%.3f viol=%.3f threshold=%.3f %s "
            "budget=%d wall=%.3f "
            "trans=%.3f raw_trans=%.3f raw_cell=%.1f raw_zone=%.1f "
            "hit(wall/trans)=%.3f/%.3f "
            "corners=%s radius=%.3f" % (
                i, int(t.item()) if hasattr(t, "item") else int(t),
                float(cost), viol, float(threshold), reason, int(self.budget),
                float(wall), float(transition), float(transition_raw),
                float(cell_raw), float(zone_raw), float(wall_hit), float(transition_hit),
                str(n_corners), float(radius),
            ),
            flush=True,
        )

    def guide_step(
            self,
            x: torch.Tensor,
            i: int,
            unet: torch.nn.Module,
            ts: torch.LongTensor,
            alpha_prod_ts: torch.Tensor,
            alpha_prod_t_prevs: torch.Tensor,
            eta: float,
            **kwargs,
    ) -> Tuple[int, torch.Tensor]:
        
        x_prev, extra_results_dict = self.local_search.guide_step(x, i, unet, ts, alpha_prod_ts, alpha_prod_t_prevs, eta, **kwargs)
        x0 = extra_results_dict["x0"]
        logprobs = self.guider.get_guidance(x0, return_logp=True, check_grad=False, **kwargs)
        
        # if i == len(ts) - 1:
        #     print(logprobs.sum().item())
        
        accept = True
        if i in self.evaluation_steps():
            self.buffer[i][logprobs.sum().item()] = x_prev
            cost = -logprobs.sum()
            violation = self._violation_cost(cost)
            threshold = self.get_threshold(i, alpha_prod_ts, alpha_prod_t_prevs)
            forced_best = False
            if violation > threshold and self.budget > 0:
                accept = False
                self.budget -= 1
            elif violation > threshold and self.budget == 0:
                accept = True
                forced_best = True
                x_prev = self.buffer[i][max(self.buffer[i].keys())] 
            else:
                accept = True
            self._print_monitor(i, ts[i], cost.detach().cpu().item(),
                                threshold.detach().cpu().item(), accept,
                                forced_best=forced_best, violation=violation)
        
        if accept:
            return x_prev, {"i_next": i + 1}

        else:
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
                
