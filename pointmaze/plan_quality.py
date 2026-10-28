"""
Plan quality of the raw generative prior, scored with the deployed verifier.

This is the primary metric for anything that tries to make the model itself
produce legal plans (as opposed to making the search better at picking among
illegal ones). It samples from the diffusion prior with no DFS and no search,
then scores each plan with the SAME MazeVerifier terms run.py thresholds on.

The number that matters is clean%: the fraction of sampled plans whose verifier
cost is below the DFS acceptance threshold. Measured on task1/var7 at
state_1000000.pt it is 0.0% for every guidance setting tried -- the search is
always falling back to forced_best because the distribution it is sampling from
contains no acceptable plan at all. Any change aimed at the generative side
(collision penalty, cross-attention, ...) should move this number first.

Usage (from pointmaze/, env `maze`):
    python plan_quality.py --ckpt logs/.../state_1000000.pt \
        --task 1 --maze_variant_idx 7 --n_samples 64 --guidance 0 2
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from cond_sensitivity import sample_plans
from mapcond.inference import load_mapcond_diffusion
from mapcond import maze_grids as MG
from search.maze_verifier import (
    get_wall_boxes_and_adjacency, get_forbidden_corner_points,
    batched_inside_wall_loss_non_adjacent, batched_diagonal_transition_loss,
    batched_corner_zone_transition_loss,
)


class _GridEnv:
    """Minimal stand-in for the MuJoCo env: MazeVerifier's helpers only read
    these five attributes, so scoring needs no simulator."""

    def __init__(self, grid, maze_type="giant"):
        self.maze_map = np.asarray(grid, dtype=np.int64)
        self._maze_unit = MG.MAZE_UNIT
        self._offset_x = MG.OFFSET_X
        self._offset_y = MG.OFFSET_Y
        self._maze_type = maze_type

    def ij_to_xy(self, ij):
        i, j = ij
        return (j * self._maze_unit - self._offset_x,
                i * self._maze_unit - self._offset_y)


def score_plans(xy, grid, corner_radius_frac, corner_transition_weight, threshold):
    """Per-plan verifier cost, matching search/maze_verifier.py exactly."""
    env = _GridEnv(grid)
    wall_boxes, adjacency = get_wall_boxes_and_adjacency(env, "cpu")
    corners = get_forbidden_corner_points(env, "cpu")
    maze_grid = torch.tensor(env.maze_map, dtype=torch.long)

    p = torch.as_tensor(xy, dtype=torch.float32)
    wall = batched_inside_wall_loss_non_adjacent(p, wall_boxes, adjacency)
    cell_t = batched_diagonal_transition_loss(p, maze_grid, MG.MAZE_UNIT,
                                              MG.OFFSET_X, MG.OFFSET_Y)
    zone_t = batched_corner_zone_transition_loss(
        p, corners, corner_radius_frac * MG.MAZE_UNIT)
    trans = torch.maximum(cell_t, zone_t)

    wall_sum = wall.sum(dim=1).numpy()
    trans_sum = (corner_transition_weight * trans).sum(dim=1).numpy()
    total = wall_sum + trans_sum
    return {
        "wall_sum": float(wall_sum.mean()),
        "hits": float((wall > 0).sum(dim=1).float().mean()),
        "trans": float(trans_sum.mean()),
        "total": float(total.mean()),
        "median": float(np.median(total)),
        "best": float(total.min()),
        "clean_frac": float((total < threshold).mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="+", required=True,
                   help="one or more checkpoints to compare")
    p.add_argument("--maze_json_dir", type=str, default="../maze_update/maze_variants")
    p.add_argument("--maze_type", type=str, default="giant")
    p.add_argument("--task", type=int, default=1)
    p.add_argument("--maze_variant_idx", type=int, default=7)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--guidance", nargs="+", type=float, default=[0.0, 2.0])
    p.add_argument("--seed", type=int, default=0)
    # defaults match run_global_film.sh / the DiT evals
    p.add_argument("--corner_radius_frac", type=float, default=0.25)
    p.add_argument("--corner_transition_weight", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=3.0,
                   help="DFS acceptance threshold (--dfs_threshold_base)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    json_path = os.path.join(args.maze_json_dir,
                             f"{args.maze_type}_task{args.task}.json")
    with open(json_path) as f:
        variant = json.load(f)["variants"][args.maze_variant_idx]
    start_xy = np.asarray(variant["start_xy"], np.float32)
    goal_xy = np.asarray(variant["goal_xy"], np.float32)

    print(f"{'checkpoint':>34} {'w':>4} {'wall_sum':>9} {'hits':>6} {'trans':>8} "
          f"{'total':>9} {'median':>9} {'best':>8} {'clean%':>7}")
    for ckpt in args.ckpt:
        diffusion, normalizer, cfg = load_mapcond_diffusion(
            ckpt, device=args.device, use_ema=True)
        horizon = int(cfg["horizon"])
        grid, _ = MG.pad_to_canvas(
            np.asarray(variant["maze_map"], dtype=np.int64),
            tuple(cfg["canvas_hw"]), pad_value=1,
            anchor=cfg.get("pad_anchor", "corner"))
        grid = grid.astype(np.float32)

        for w in args.guidance:
            diffusion.model.guidance_scale = float(w)
            xy = sample_plans(diffusion, normalizer, grid, start_xy, goal_xy,
                              args.n_samples, horizon, args.seed, args.device)
            m = score_plans(xy, grid, args.corner_radius_frac,
                            args.corner_transition_weight, args.threshold)
            name = os.path.basename(os.path.dirname(os.path.dirname(ckpt))) \
                + "/" + os.path.basename(ckpt)
            print(f"{name[-34:]:>34} {w:>4g} {m['wall_sum']:>9.2f} {m['hits']:>6.1f} "
                  f"{m['trans']:>8.2f} {m['total']:>9.2f} {m['median']:>9.2f} "
                  f"{m['best']:>8.2f} {100*m['clean_frac']:>6.1f}%")


if __name__ == "__main__":
    main()
