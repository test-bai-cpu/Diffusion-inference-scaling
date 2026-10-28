"""
Map-conditioning sensitivity probe: does the model actually READ the bound map?

Motivation (see the t1l3v2 postmortem): plans repeatedly walk straight through a
handful of cells that are OPEN in the base giant map and in most variants of the
same task, but CLOSED in the variant being planned on -- e.g. (5,6) is open in
9/12 task1 variants, (9,6) in 11/12. The hypothesis is that a single global
conditioning vector cannot override the model's learned prior for one specific
cell. This script measures that directly, with no search in the loop.

Method: sample plans twice from the SAME model with the SAME noise, changing only
one cell of the bound occupancy grid (wall -> open). Everything else -- start,
goal, horizon, RNG -- is held fixed, so any difference in the sampled plans is
caused by that one cell of conditioning. Reported per probed cell:

    enter_frac   fraction of sampled plans that put at least one waypoint in the cell
    steps_in     mean number of waypoints inside the cell
    d_local      mean |xy_true - xy_flipped| for waypoints near the cell
    d_global     mean |xy_true - xy_flipped| over the whole trajectory

A model that reads the map should show a LARGE gap in enter_frac between the
"cell is wall" and "cell is open" conditions. A control cell far from the route
gives the noise floor: if flipping the relevant cell moves the plans about as
much as flipping an irrelevant one, the conditioning is not being used spatially.

Deliberately samples the raw diffusion prior (full ancestral loop, no DFS, no
verifier): the search pipeline rejects/retries against the TRUE grid, which would
mask exactly the behaviour being measured here.

Usage (from pointmaze/, env `maze`):
    python cond_sensitivity.py \
        --ckpt logs/mapcond-mazev1-seed2-dit/variants_train_giant/state_1000000.pt \
        --task 1 --maze_variant_idx 7 --cells 5,6 9,6 --n_samples 64
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

from mapcond.inference import load_mapcond_diffusion
from mapcond import maze_grids as MG


def parse_cell(s):
    i, j = s.split(",")
    return int(i), int(j)


def world_to_cell(xy):
    """World (x, y) -> canvas cell (i, j), matching MazeEnv.xy_to_ij."""
    j = np.floor((xy[..., 0] + MG.OFFSET_X + 0.5 * MG.MAZE_UNIT) / MG.MAZE_UNIT)
    i = np.floor((xy[..., 1] + MG.OFFSET_Y + 0.5 * MG.MAZE_UNIT) / MG.MAZE_UNIT)
    return i.astype(int), j.astype(int)


def cell_center(cell):
    i, j = cell
    return np.array([j * MG.MAZE_UNIT - MG.OFFSET_X,
                     i * MG.MAZE_UNIT - MG.OFFSET_Y], dtype=np.float32)


@torch.no_grad()
def sample_plans(diffusion, normalizer, grid, start_xy, goal_xy, n_samples,
                 horizon, seed, device):
    """Sample `n_samples` plans with the given grid bound. Returns world-frame
    xy [n_samples, horizon, 2]. The RNG is reseeded here so that two calls with
    identical shapes draw an identical noise sequence -- that pairing is what
    makes the A/B comparison a controlled one."""
    diffusion.model.bind_maze(torch.as_tensor(grid, dtype=torch.float32))

    start_n = normalizer.normalize(np.asarray(start_xy, np.float32), "observations")
    goal_n = normalizer.normalize(np.asarray(goal_xy, np.float32), "observations")
    cond = {
        0: torch.as_tensor(start_n, dtype=torch.float32, device=device)
             .unsqueeze(0).repeat(n_samples, 1),
        horizon - 1: torch.as_tensor(goal_n, dtype=torch.float32, device=device)
             .unsqueeze(0).repeat(n_samples, 1),
    }

    torch.manual_seed(seed)
    traj = diffusion.conditional_sample(cond, horizon=horizon, verbose=False)
    xy_norm = traj[..., 2:4].cpu().numpy()
    return normalizer.unnormalize(xy_norm, "observations")


def metrics_for_cell(xy_true, xy_flip, cell, local_radius):
    """A/B metrics for one probed cell ('before' = the true variant grid)."""
    out = {}
    for tag, xy in (("before", xy_true), ("after", xy_flip)):
        i, j = world_to_cell(xy)
        inside = (i == cell[0]) & (j == cell[1])          # [N, T]
        out[f"enter_frac_{tag}"] = float(inside.any(axis=1).mean())
        out[f"steps_in_{tag}"] = float(inside.sum(axis=1).mean())

    diff = np.linalg.norm(xy_true - xy_flip, axis=-1)      # [N, T]
    out["d_global"] = float(diff.mean())

    centre = cell_center(cell)
    near = (np.linalg.norm(xy_true - centre, axis=-1) < local_radius) | \
           (np.linalg.norm(xy_flip - centre, axis=-1) < local_radius)
    out["d_local"] = float(diff[near].mean()) if near.any() else float("nan")
    out["n_near"] = int(near.sum())
    return out


def visit_counts(xy, shape):
    """How many sampled plans put at least one waypoint in each cell."""
    H, W = shape
    counts = np.zeros((H, W), dtype=np.int32)
    i, j = world_to_cell(xy)
    for k in range(xy.shape[0]):
        ii, jj = i[k], j[k]
        ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
        for a, b in set(zip(ii[ok].tolist(), jj[ok].tolist())):
            counts[a, b] += 1
    return counts


def route_distance_map(counts, min_visits):
    """Grid distance (Chebyshev) from every cell to the nearest cell the route
    actually uses. Used to pick distance-matched controls."""
    H, W = counts.shape
    route = [(i, j) for i in range(H) for j in range(W) if counts[i, j] >= min_visits]
    dist = np.full((H, W), np.inf)
    for i in range(H):
        for j in range(W):
            if route:
                dist[i, j] = min(max(abs(i - ri), abs(j - rj)) for ri, rj in route)
    return dist


def pick_control_cells(grid, dist, buckets, exclude):
    """One INTERIOR wall cell per distance-from-route bucket.

    Interior only: the perimeter ring is a wall in every map the model has ever
    seen (native borders plus pad_to_canvas padding), so flipping a perimeter
    cell probes an out-of-distribution map rather than an irrelevant one -- it
    cannot serve as a noise floor. Bucketing by distance to the route turns the
    control into a locality curve instead of a single point.
    """
    H, W = grid.shape
    out = []
    for lo, hi in buckets:
        best = None
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if grid[i, j] != 1 or (i, j) in exclude:
                    continue
                d = dist[i, j]
                if lo <= d <= hi and (best is None or d < dist[best]):
                    best = (i, j)
        if best is not None:
            out.append((best, float(dist[best])))
            exclude.add(best)
    return out


def plot(grid, xy_by_cond, probe_cell, new_val, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = grid.shape
    extent = [-MG.OFFSET_X - MG.MAZE_UNIT / 2, (W - 1) * MG.MAZE_UNIT - MG.OFFSET_X + MG.MAZE_UNIT / 2,
              -MG.OFFSET_Y - MG.MAZE_UNIT / 2, (H - 1) * MG.MAZE_UNIT - MG.OFFSET_Y + MG.MAZE_UNIT / 2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (tag, xy) in zip(axes, xy_by_cond.items()):
        g = grid.copy()
        if tag != "true":
            g[probe_cell] = new_val
        ax.imshow(1 - g, cmap="gray", origin="lower", extent=extent,
                  interpolation="nearest", vmin=0, vmax=1)
        for k in range(xy.shape[0]):
            ax.plot(xy[k, :, 0], xy[k, :, 1], lw=0.6, alpha=0.25, color="#1f6f8b")
        c = cell_center(probe_cell)
        ax.add_patch(plt.Rectangle((c[0] - MG.MAZE_UNIT / 2, c[1] - MG.MAZE_UNIT / 2),
                                   MG.MAZE_UNIT, MG.MAZE_UNIT,
                                   fill=False, edgecolor="red", lw=2))
        ax.set_title(f"cell {probe_cell} = {tag}")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[cond_sensitivity] wrote {out_png}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="logs/mapcond-mazev1-seed2-dit/variants_train_giant/state_1000000.pt")
    p.add_argument("--maze_json_dir", type=str, default="../maze_update/maze_variants")
    p.add_argument("--maze_type", type=str, default="giant")
    p.add_argument("--task", type=int, default=1)
    p.add_argument("--maze_variant_idx", type=int, default=7)
    p.add_argument("--cells", nargs="+", type=parse_cell, default=[(5, 6), (9, 6)],
                   help='cells to flip wall->open, as "i,j"')
    p.add_argument("--no_control", action="store_true",
                   help="skip the distance-matched control cells")
    p.add_argument("--control_buckets", nargs="+", type=parse_cell,
                   default=[(1, 1), (2, 3), (4, 6)],
                   help='grid-distance-from-route buckets "lo,hi" to draw one '
                        'interior control cell from each')
    p.add_argument("--no_close_probe", action="store_true",
                   help="skip the open->wall probe on the busiest route cell")
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--guidance", nargs="+", type=float, default=[0.0, 2.0],
                   help="CFG scales to probe; 0 = raw conditional model, "
                        "2 = the value run_global_film.sh / the DiT evals use")
    p.add_argument("--local_radius", type=float, default=2 * MG.MAZE_UNIT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--negative_map", type=str, default="none",
                   choices=["none", "base"],
                   help="what the CFG negative pass conditions on. 'none' = the "
                        "trained null embedding (standard CFG). 'base' = the "
                        "base giant map, so guidance amplifies 'this variant vs "
                        "the map the prior is anchored to' instead of "
                        "'conditional vs nothing'. Inference-only.")
    p.add_argument("--out_dir", type=str, default="figures/cond_sensitivity")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    diffusion, normalizer, cfg = load_mapcond_diffusion(
        args.ckpt, device=args.device, use_ema=True)
    horizon = int(cfg["horizon"])
    canvas_hw = tuple(cfg["canvas_hw"])
    print(f"[cond_sensitivity] {os.path.basename(args.ckpt)} "
          f"backbone={cfg.get('backbone', 'unet')} horizon={horizon} "
          f"canvas={canvas_hw}", flush=True)

    json_path = os.path.join(args.maze_json_dir,
                             f"{args.maze_type}_task{args.task}.json")
    with open(json_path) as f:
        spec = json.load(f)
    variant = spec["variants"][args.maze_variant_idx]
    raw_grid = np.asarray(variant["maze_map"], dtype=np.int64)
    grid, _ = MG.pad_to_canvas(raw_grid, canvas_hw, pad_value=1,
                               anchor=cfg.get("pad_anchor", "corner"))
    grid = grid.astype(np.float32)
    start_xy = np.asarray(variant["start_xy"], np.float32)
    goal_xy = np.asarray(variant["goal_xy"], np.float32)

    if args.negative_map == "base":
        base_raw = MG.parse_grid(args.maze_type)
        base_grid, _ = MG.pad_to_canvas(base_raw, canvas_hw, pad_value=1,
                                        anchor=cfg.get("pad_anchor", "corner"))
        diffusion.model.bind_negative_maze(base_grid.astype(np.float32))
        n_diff = int((base_grid != grid).sum())
        print(f"[cond_sensitivity] CFG negative = base {args.maze_type} map "
              f"({n_diff} cells differ from this variant)", flush=True)

    variant_grids = [np.asarray(v["maze_map"], dtype=np.int64) for v in spec["variants"]]

    def open_freq(cell):
        i, j = cell
        if i >= variant_grids[0].shape[0] or j >= variant_grids[0].shape[1]:
            return None
        return sum(1 for g in variant_grids if g[i, j] == 0), len(variant_grids)

    for w in args.guidance:
        diffusion.model.guidance_scale = float(w)
        print(f"\n===== guidance_scale = {w} =====", flush=True)

        xy_true = sample_plans(diffusion, normalizer, grid, start_xy, goal_xy,
                               args.n_samples, horizon, args.seed, args.device)

        counts = visit_counts(xy_true, grid.shape)
        dist = route_distance_map(counts, max(1, args.n_samples // 4))

        # ---- build the probe set ----
        # 1. the cells under investigation: wall -> open
        probes = [(c, 0.0, "open") for c in args.cells]
        used = set(args.cells)

        # 2. the CLOSE direction, which is what the real failure looks like:
        #    take the open cell the route relies on most and wall it off. A model
        #    that reads the map has to reroute; one that does not will drive
        #    straight through, exactly as it does on the real variants.
        if not args.no_close_probe:
            cand = [(counts[i, j], (i, j))
                    for i in range(1, grid.shape[0] - 1)
                    for j in range(1, grid.shape[1] - 1)
                    if grid[i, j] == 0 and counts[i, j] >= args.n_samples // 2
                    and np.hypot(*(cell_center((i, j)) - start_xy)) > 1.5 * MG.MAZE_UNIT
                    and np.hypot(*(cell_center((i, j)) - goal_xy)) > 1.5 * MG.MAZE_UNIT]
            if cand:
                _, top = max(cand)
                probes.append((top, 1.0, "wall"))
                used.add(top)

        # 3. distance-matched INTERIOR controls (see pick_control_cells docstring)
        if not args.no_control:
            for cell, d in pick_control_cells(grid, dist, args.control_buckets, used):
                probes.append((cell, 0.0, "open"))
                print(f"[cond_sensitivity] control cell {cell} "
                      f"(grid distance {d:.0f} from route)", flush=True)

        print(f"\n{'cell':>8} {'flip':>12} {'dist':>5} {'variants':>9} "
              f"{'enter%(before)':>14} {'enter%(after)':>14} {'d_local':>9} {'d_global':>9}")
        for cell, new_val, tag in probes:
            flipped = grid.copy()
            flipped[cell] = new_val
            xy_flip = sample_plans(diffusion, normalizer, flipped, start_xy, goal_xy,
                                   args.n_samples, horizon, args.seed, args.device)
            m = metrics_for_cell(xy_true, xy_flip, cell, args.local_radius)
            of = open_freq(cell)
            flip_str = ("wall->open" if new_val == 0 else "open->wall")
            d_str = "inf" if not np.isfinite(dist[cell]) else f"{dist[cell]:.0f}"
            of_str = "-" if of is None else f"{of[0]}/{of[1]}"
            print(f"{str(cell):>8} {flip_str:>12} {d_str:>5} {of_str:>9} "
                  f"{100*m['enter_frac_before']:>13.1f}% "
                  f"{100*m['enter_frac_after']:>13.1f}% "
                  f"{m['d_local']:>9.3f} {m['d_global']:>9.3f}")

            fig_tag = f"cell{cell[0]}-{cell[1]}_{flip_str.replace('->','2')}_w{w:g}"
            plot(grid, {"true": xy_true, tag: xy_flip}, cell, new_val,
                 os.path.join(args.out_dir, f"{fig_tag}.png"),
                 f"task{args.task} var{args.maze_variant_idx}  cell {cell}  "
                 f"{flip_str}  guidance={w:g}  n={args.n_samples}")


if __name__ == "__main__":
    main()
