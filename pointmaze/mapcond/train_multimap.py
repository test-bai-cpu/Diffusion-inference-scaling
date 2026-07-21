"""
Multi-map training for the map-conditional diffusion planner. MUJOCO-FREE.

Trains ONE model on several pointmaze maps at once, with the occupancy grid
fed to the network as conditioning (MapConditionalTemporalUnet /
MapConditionalGaussianDiffusion). This is the core of "make the map an input":
the same weights must plan on medium / large / giant / teleport, so at
inference a *new* grid selects the right behaviour instead of being baked in.

No mujoco / no OGBench env is imported here -- training uses only the offline
.npy trajectories and ast-parsed grids. The simulator is needed only for
rollout EVALUATION (run.py on the cluster), not for training.

Checkpoint layout (compatible with the repo's Trainer.save):
    state_{step}.pt = {step, model, ema, normalizer, config}
`normalizer` and `config` are extra fields so inference can rebuild the exact
same dataset geometry (shared normalizer, canvas size, horizon) without
re-fitting. The repo's loader ignores unknown keys, so this stays compatible.

Usage (from pointmaze/, env `mapcond`):
    python -m mapcond.train_multimap \
        --maps medium large giant teleport \
        --horizon 256 --n_train_steps 1000000 \
        --batch_size 32 --savepath logs/mapcond/H256_T256

    # include all requested giant task/variant maps:
    python -m mapcond.train_multimap \
        --maps medium large giant teleport \
        --variant_tasks 1 2 3 4 \
        --savepath logs/mapcond/base_plus_giant_variants

    # quick local CPU sanity run (tiny, mujoco-free):
    python -m mapcond.train_multimap --smoke
"""
import os, sys, json, time, argparse, copy, pickle
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_POINTMAZE = os.path.abspath(os.path.join(_HERE, ".."))
if _POINTMAZE not in sys.path:
    sys.path.insert(0, _POINTMAZE)

from mapcond.dataset import MultiMazeGoalDataset, map_batch_collate
from mapcond import data_utils as DU
from mapcond.models import (
    MapConditionalTemporalUnet, MapConditionalGaussianDiffusion,
)


# ----------------------------- EMA (matches repo) -----------------------------
class EMA:
    def __init__(self, beta): self.beta = beta
    def update(self, ema_model, model):
        for ep, p in zip(ema_model.parameters(), model.parameters()):
            ep.data = ep.data * self.beta + (1 - self.beta) * p.data


def cycle(dl):
    while True:
        for b in dl:
            yield b


def build(args, dataset):
    if args.map_blind:
        assert args.local_channels == 0, \
            "--map_blind is incompatible with --local_channels"
    if args.local_film:
        assert args.local_channels > 0, \
            "--local_film requires --local_channels > 0"
    if args.local_channels > 0:
        from mapcond.models import LocalMapConditionalTemporalUnet
        from mapcond import local_features as LF
        if args.local_film:
            from mapcond.film_models import FiLMLocalMapConditionalTemporalUnet
            model_cls = FiLMLocalMapConditionalTemporalUnet
        else:
            model_cls = LocalMapConditionalTemporalUnet
        model = model_cls(
            horizon=args.horizon,
            transition_dim=dataset.transition_dim,
            cond_dim=dataset.observation_dim,
            dim=args.dim,
            dim_mults=tuple(args.dim_mults),
            pool_size=args.pool_size,
            cfg_dropout=args.cfg_dropout,
            local_channels=args.local_channels,
        )
        lo, hi = dataset.normalizer.bounds["observations"]
        scale, shift = LF.norm_xy_to_uv_affine(lo[:2], hi[:2],
                                               dataset.canvas_hw)
        model.set_coord_map(scale, shift)
        print(f"[train_multimap] local_channels={args.local_channels} "
              f"coord_map scale={scale.tolist()} shift={shift.tolist()}",
              flush=True)
        model = model.to(args.device)
    else:
        model = MapConditionalTemporalUnet(
            horizon=args.horizon,
            transition_dim=dataset.transition_dim,
            cond_dim=dataset.observation_dim,
            dim=args.dim,
            dim_mults=tuple(args.dim_mults),
            pool_size=args.pool_size,
            cfg_dropout=args.cfg_dropout,
        ).to(args.device)
    diffusion = MapConditionalGaussianDiffusion(
        model,
        horizon=args.horizon,
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=True,
        predict_epsilon=False,
    ).to(args.device)
    return model, diffusion


def build_map_specs(args):
    """Build explicit specs when giant variants are requested."""
    specs = DU.base_map_specs(args.maps)
    if args.variant_tasks:
        specs.extend(DU.giant_variant_map_specs(
            variant_dir=args.variant_dir,
            variant_json_dir=args.variant_json_dir,
            tasks=args.variant_tasks,
            vars=args.variant_vars,
            validation=args.variant_val,
        ))
    return specs


def save_ckpt(path, step, model, ema_model, dataset, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        # extra fields so inference can rebuild identical geometry:
        "normalizer": pickle.dumps(dataset.normalizer),
        "config": {
            "maps": dataset.map_ids,
            "base_maps": list(args.maps),
            "map_ids": dataset.map_ids,
            "map_specs": dataset.serializable_map_specs(),
            "variant_dir": args.variant_dir if args.variant_tasks else None,
            "variant_json_dir": args.variant_json_dir if args.variant_tasks else None,
            "variant_tasks": list(args.variant_tasks),
            "variant_vars": list(args.variant_vars),
            "variant_val": bool(args.variant_val),
            "horizon": args.horizon,
            "n_diffusion_steps": args.n_diffusion_steps,
            "dim": args.dim,
            "dim_mults": list(args.dim_mults),
            "pool_size": args.pool_size,
            "cfg_dropout": args.cfg_dropout,
            "local_channels": args.local_channels,
            "local_film": bool(args.local_film),
            "map_blind": bool(args.map_blind),
            "canvas_hw": list(dataset.canvas_hw),
            "pad_anchor": dataset.pad_anchor,
            "transition_dim": dataset.transition_dim,
            "observation_dim": dataset.observation_dim,
            "action_dim": dataset.action_dim,
        },
    }, path)
    print(f"[train_multimap] saved {path}", flush=True)


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.savepath, exist_ok=True)

    # -------- dataset (shared normalizer across maps) --------
    if args.variant_tasks:
        dataset = MultiMazeGoalDataset(
            map_specs=build_map_specs(args),
            horizon=args.horizon,
            canvas_hw=None,                  # giant defines 12x16
            pad_anchor="corner",
            max_episodes_per_map=args.max_episodes_per_map,
            local_dist=(args.local_channels >= 2),
        )
    else:
        dataset = MultiMazeGoalDataset(
            maze_types=args.maps,
            horizon=args.horizon,
            canvas_hw=None,                  # giant defines 12x16
            pad_anchor="corner",
            max_episodes_per_map=args.max_episodes_per_map,
            local_dist=(args.local_channels >= 2),
        )
    preview = dataset.map_ids[:8]
    suffix = "" if len(dataset.map_ids) <= len(preview) else " ..."
    print(f"[train_multimap] map_count={len(dataset.map_ids)} "
          f"maps={preview}{suffix} windows={len(dataset)} "
          f"canvas={dataset.canvas_hw}", flush=True)
    print(f"[train_multimap] normalizer={dataset.normalizer}", flush=True)
    # persist the fitted normalizer next to checkpoints
    with open(os.path.join(args.savepath, "global_normalizer.pkl"), "wb") as f:
        pickle.dump(dataset.normalizer, f)

    loader = cycle(torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device != "cpu"),
        collate_fn=map_batch_collate,
    ))

    # -------- model + ema + optim --------
    model, diffusion = build(args, dataset)
    ema = EMA(args.ema_decay)
    ema_model = copy.deepcopy(diffusion)
    ema_model.load_state_dict(diffusion.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_multimap] model params: {n_params:,}", flush=True)

    loss_log = []
    tb = None
    if not getattr(args, "no_tensorboard", False):
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(log_dir=os.path.join(args.savepath, "tb"))
            print(f"[train_multimap] tensorboard -> {tb.log_dir}", flush=True)
        except Exception as e:
            print(f"[train_multimap] tensorboard disabled "
                  f"({type(e).__name__}: {e})", flush=True)      # (step, loss)
    t0 = time.time()
    for step in range(args.n_train_steps):
        opt.zero_grad()
        for _ in range(args.gradient_accumulate_every):
            batch = next(loader)
            trajs = batch.trajectories.to(args.device)
            cond = {k: v.to(args.device) for k, v in batch.conditions.items()}
            maze = None if args.map_blind else batch.maze.to(args.device)
            loss, info = diffusion.loss(trajs, cond, maze=maze)
            (loss / args.gradient_accumulate_every).backward()
        opt.step()

        # EMA update (repo schedule: warmup then decay every k steps)
        if step % args.update_ema_every == 0:
            if step < args.step_start_ema:
                ema_model.load_state_dict(diffusion.state_dict())
            else:
                ema.update(ema_model, diffusion)

        if step % args.log_freq == 0:
            loss_log.append((step, float(loss)))
            rate = (step + 1) / (time.time() - t0)
            film_norm = _film_norm(diffusion)
            film_str = f" | film {film_norm:9.5f}" if film_norm is not None else ""
            print(f"{step:>7d} | loss {float(loss):8.5f} | {rate:6.1f} it/s"
                  f"{film_str}", flush=True)
            # crash-safe incremental log, watchable while training runs
            _csv = os.path.join(args.savepath, "loss_log.csv")
            _new = not os.path.exists(_csv)
            with open(_csv, "a") as f:
                if _new:
                    f.write("step,loss,it_per_s,film_norm\n")
                f.write(f"{step},{float(loss):.6f},{rate:.3f},"
                        f"{'' if film_norm is None else f'{film_norm:.6f}'}\n")
            if tb is not None:
                tb.add_scalar("train/loss", float(loss), step)
                tb.add_scalar("train/it_per_s", rate, step)
                if film_norm is not None:
                    tb.add_scalar("train/film_norm", film_norm, step)

        if step > 0 and step % args.save_freq == 0:
            save_ckpt(os.path.join(args.savepath, f"state_{step}.pt"),
                      step, diffusion, ema_model, dataset, args)
            with open(os.path.join(args.savepath, "loss_log.json"), "w") as f:
                json.dump(loss_log, f)
            _plot_loss(loss_log, os.path.join(args.savepath, "train_loss.png"))

    # final checkpoint + loss log
    save_ckpt(os.path.join(args.savepath, f"state_{args.n_train_steps}.pt"),
              args.n_train_steps, diffusion, ema_model, dataset, args)
    with open(os.path.join(args.savepath, "loss_log.json"), "w") as f:
        json.dump(loss_log, f)
    _plot_loss(loss_log, os.path.join(args.savepath, "train_loss.png"))
    if tb is not None:
        tb.close()
    print(f"[train_multimap] done in {time.time()-t0:.1f}s", flush=True)
    return loss_log


def _film_norm(diffusion):
    """Total weight norm of all FiLM projections, or None for non-FiLM
    models. Growth from zero is the direct evidence that the network is
    starting to USE the per-block local signal; a norm stuck near zero after
    hundreds of thousands of steps means the pathway is being ignored."""
    try:
        from mapcond.film_models import FiLMBlockWrap
    except Exception:
        return None
    total, found = 0.0, False
    for m in diffusion.modules():
        if isinstance(m, FiLMBlockWrap):
            found = True
            total += float(m.film_proj.weight.norm()) + \
                     float(m.film_proj.bias.norm())
    return total if found else None


def _plot_loss(loss_log, out_png):
    if not loss_log:
        return
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps, losses = zip(*loss_log)
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        ax.plot(steps, losses, color="#1f6f8b", lw=1.3)
        ax.set_yscale("log")
        ax.set_xlabel("training step"); ax.set_ylabel("loss (L2, log)")
        ax.set_title("Multi-map training loss")
        fig.tight_layout(); fig.savefig(out_png, dpi=150)
        print(f"[train_multimap] wrote {out_png}", flush=True)
    except Exception as e:
        print(f"[train_multimap] plot skipped: {type(e).__name__}: {e}", flush=True)


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--maps", nargs="+", default=["medium", "large", "giant", "teleport"])
    p.add_argument("--variant_dir", type=str, default=DU.DEFAULT_VARIANT_DIR,
                   help="directory containing pointmaze-giant task/var .npz files")
    p.add_argument("--variant_json_dir", type=str, default=DU.DEFAULT_VARIANT_JSON_DIR,
                   help="directory containing giant_task{N}.json variant grids")
    p.add_argument("--variant_tasks", nargs="+", type=int, default=[],
                   help="giant variant task ids to include, e.g. 1 2 3 4")
    p.add_argument("--variant_vars", nargs="+", type=int, default=None,
                   help="variant ids per task; defaults to 0..11 when tasks are set")
    p.add_argument("--variant_val", action="store_true",
                   help="load -val.npz twins instead of training .npz files")
    p.add_argument("--horizon", type=int, default=600)
    p.add_argument("--n_diffusion_steps", type=int, default=256)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--dim_mults", nargs="+", type=int, default=[1, 4, 8])
    p.add_argument("--pool_size", type=int, default=4,
                   help="MapEncoder spatial pool grid; 1 = original global "
                        "average pool, 4 = keep 4x4 layout information.")
    p.add_argument("--map_blind", action="store_true",
                   help="Train the ORIGINAL map-blind method on the pooled "
                        "multi-map data: the maze is never shown to the model "
                        "(forward runs the maze=None branch, bit-identical to "
                        "plain TemporalUnet). The checkpoint records this so "
                        "inference disables map binding automatically. "
                        "Ablation arm isolating data diversity from "
                        "conditioning.")
    p.add_argument("--no_tensorboard", action="store_true",
                   help="Disable TensorBoard logging (enabled by default when "
                        "torch.utils.tensorboard is importable).")
    p.add_argument("--local_film", action="store_true",
                   help="With --local_channels > 0: additionally inject the "
                        "per-timestep local features into EVERY residual "
                        "block via zero-initialized FiLM (multiplicative "
                        "gating), instead of input-concat only.")
    p.add_argument("--local_channels", type=int, default=0,
                   help="0 = global embedding only (current default); 2 = "
                        "also sample occupancy + BFS-distance channels at "
                        "each trajectory timestep (feature-grid conditioning).")
    p.add_argument("--cfg_dropout", type=float, default=0.1,
                   help="Prob. of replacing the map embedding with the learned "
                        "null embedding during training (classifier-free "
                        "guidance). 0 disables CFG training.")
    p.add_argument("--n_train_steps", type=int, default=1_000_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--gradient_accumulate_every", type=int, default=2)
    p.add_argument("--ema_decay", type=float, default=0.995)
    p.add_argument("--step_start_ema", type=int, default=2000)
    p.add_argument("--update_ema_every", type=int, default=10)
    p.add_argument("--loss_type", type=str, default="l2")
    p.add_argument("--log_freq", type=int, default=100)
    p.add_argument("--save_freq", type=int, default=20000)
    p.add_argument("--max_episodes_per_map", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--savepath", type=str, default="logs/mapcond/H256_T256")
    p.add_argument("--smoke", action="store_true",
                   help="tiny CPU run: few steps, few episodes, small savepath")
    args = p.parse_args(argv)
    if args.variant_vars is None:
        args.variant_vars = list(range(12)) if args.variant_tasks else []
    if args.smoke:
        args.n_train_steps = 30
        args.save_freq = 20
        args.log_freq = 5
        args.max_episodes_per_map = 3
        args.device = "cpu"
        args.num_workers = 0
        args.savepath = "logs/mapcond/smoke"
    return args


if __name__ == "__main__":
    train(get_args())
