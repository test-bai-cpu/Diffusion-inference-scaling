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
    if args.backbone == "dit":
        assert not args.global_film and args.local_channels == 0 \
            and not args.local_film, \
            "--backbone dit only implements the Global-FiLM-equivalent tier " \
            "so far (adaLN-Zero IS global FiLM, built into MapConditionalDiT1D " \
            "-- see mapcond/dit_models.py); --global_film/--local_channels/" \
            "--local_film are U-Net-only for now"
        from mapcond.dit_models import MapConditionalDiT1D
        model = MapConditionalDiT1D(
            horizon=args.horizon,
            transition_dim=dataset.transition_dim,
            cond_dim=dataset.observation_dim,
            hidden=args.dit_hidden,
            heads=args.dit_heads,
            depth=args.dit_depth,
            patch_size=args.dit_patch_size,
            mlp_ratio=args.dit_mlp_ratio,
            pool_size=args.pool_size,
            cfg_dropout=args.cfg_dropout,
        ).to(args.device)
    elif args.global_film:
        assert args.local_channels == 0, \
            "--global_film is a global-only tier for now; incompatible " \
            "with --local_channels (use --local_film for the local tier)"
        assert not args.map_blind, \
            "--global_film is incompatible with --map_blind (there is no " \
            "other map pathway left to be blind about)"
        from mapcond.global_film_models import GlobalFiLMTemporalUnet
        model = GlobalFiLMTemporalUnet(
            horizon=args.horizon,
            transition_dim=dataset.transition_dim,
            cond_dim=dataset.observation_dim,
            dim=args.dim,
            dim_mults=tuple(args.dim_mults),
            pool_size=args.pool_size,
            cfg_dropout=args.cfg_dropout,
        ).to(args.device)
    elif args.local_channels > 0:
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
    if args.collision_weight > 0 or args.corner_weight > 0:
        assert not args.map_blind, \
            "--collision_weight/--corner_weight need the maze (map_blind passes maze=None)"
        from mapcond import local_features as LF
        lo, hi = dataset.normalizer.bounds["observations"]
        scale, shift = LF.norm_xy_to_uv_affine(lo[:2], hi[:2], dataset.canvas_hw)
        radius = args.corner_radius_frac * LF.MG.MAZE_UNIT
        diffusion.set_collision_penalty(args.collision_weight, scale, shift,
                                        corner_weight=args.corner_weight,
                                        corner_radius=radius,
                                        obs_min=lo[:2], obs_max=hi[:2])
        print(f"[train_multimap] penalties: wall={args.collision_weight} "
              f"corner={args.corner_weight} (radius={radius:.2f} world units)",
              flush=True)
    return model, diffusion


def build_map_specs(args, validation=None):
    """Build explicit specs when giant variants are requested.

    `validation` overrides args.variant_val and additionally swaps the base maps
    to their `-val` twins, so the same call builds either the training set or
    the held-out set over the SAME maps.
    """
    val = args.variant_val if validation is None else validation
    specs = DU.base_map_specs(args.maps, validation=val)
    if args.variant_tasks:
        specs.extend(DU.giant_variant_map_specs(
            variant_dir=args.variant_dir,
            variant_json_dir=args.variant_json_dir,
            tasks=args.variant_tasks,
            vars=args.variant_vars,
            validation=val,
        ))
    return specs


def build_val_loader(args, dataset):
    """
    Held-out loader over the `-val` twins of the very same maps.

    OGBench and the variant generator both ship a disjoint 10% split (50
    episodes vs 500) for every map already, so nothing has to be carved out of
    the training set -- and unlike a window-level split of the training data,
    these episodes share no overlap at all with what the model trains on. The
    training normalizer is reused, never refit: a separately fitted one would
    put the two sets in different coordinate frames.

    A fixed random subset is drawn once so the reported number is a
    deterministic function of the weights, not a different sample each time.
    """
    val_ds = MultiMazeGoalDataset(
        map_specs=build_map_specs(args, validation=True),
        horizon=args.horizon,
        normalizer=dataset.normalizer,
        canvas_hw=dataset.canvas_hw,
        pad_anchor=dataset.pad_anchor,
        max_episodes_per_map=args.max_episodes_per_map,
        local_dist=(args.local_channels >= 2),
    )
    n = min(len(val_ds), args.val_batches * args.batch_size)
    idx = np.random.RandomState(args.val_seed).choice(len(val_ds), n, replace=False)
    subset = torch.utils.data.Subset(val_ds, sorted(idx.tolist()))
    print(f"[train_multimap] val: {len(val_ds)} windows over {len(val_ds.map_ids)} "
          f"maps, evaluating a fixed {n}", flush=True)
    return torch.utils.data.DataLoader(
        subset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=map_batch_collate)


@torch.no_grad()
def validate(diffusion, loader, args):
    """
    Mean loss on the held-out set. Returns (plain diffusion loss, penalties).

    The plain diffusion term is reported separately because it is the only
    number comparable ACROSS runs: the penalty weights differ between arms, so
    a combined total cannot be lined up against a baseline.

    Runs under eval() so CFG dropout does not fire (a validation number should
    not depend on which samples randomly had their map hidden), and restores
    the RNG afterwards so that turning validation on does not alter the
    training trajectory by a single draw.
    """
    rng_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(args.val_seed)
    was_training = diffusion.training
    diffusion.eval()

    tot, pen_tot, n = 0.0, 0.0, 0
    for batch in loader:
        trajs = batch.trajectories.to(args.device)
        cond = {k: v.to(args.device) for k, v in batch.conditions.items()}
        maze = None if args.map_blind else batch.maze.to(args.device)
        loss, info = diffusion.loss(trajs, cond, maze=maze)
        pen = (args.collision_weight * float(info.get("collision", 0.0))
               + args.corner_weight * float(info.get("corner", 0.0)))
        tot += float(loss)
        pen_tot += pen
        n += 1

    if was_training:
        diffusion.train()
    torch.set_rng_state(rng_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return (tot - pen_tot) / max(n, 1), pen_tot / max(n, 1)


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
            "global_film": bool(args.global_film),
            "map_blind": bool(args.map_blind),
            "backbone": args.backbone,
            "dit_hidden": args.dit_hidden,
            "dit_heads": args.dit_heads,
            "dit_depth": args.dit_depth,
            "dit_patch_size": args.dit_patch_size,
            "dit_mlp_ratio": args.dit_mlp_ratio,
            "collision_weight": args.collision_weight,
            "corner_weight": args.corner_weight,
            "negative_weight": args.negative_weight,
            "val_freq": args.val_freq,
            "corner_radius_frac": args.corner_radius_frac,
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

    val_loader = build_val_loader(args, dataset) if args.val_freq > 0 else None

    # -------- model + ema + optim --------
    model, diffusion = build(args, dataset)
    if args.init_from:
        # Warm start: weights only. The optimizer, EMA schedule and step
        # counter all start fresh, so this is "continue from these weights
        # under a new objective", not a crash resume -- which is exactly what
        # an ablation like --collision_weight wants, since it keeps the
        # backbone fixed and changes one term.
        _ck = torch.load(args.init_from, map_location=args.device, weights_only=False)
        _state = _ck["ema"] if ("ema" in _ck and args.init_from_ema) else _ck["model"]
        missing, unexpected = diffusion.load_state_dict(_state, strict=False)
        assert not unexpected, f"unexpected keys in {args.init_from}: {unexpected}"
        if missing:
            print(f"[train_multimap] warm start: {len(missing)} missing keys "
                  f"left at init, e.g. {missing[:4]}", flush=True)
        print(f"[train_multimap] warm-started from {args.init_from} "
              f"(step {_ck.get('step')}, ema={args.init_from_ema})", flush=True)
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
            if args.negative_weight > 0 and maze is not None:
                from mapcond.collision import block_route_cell
                _lo = torch.as_tensor(dataset.normalizer.bounds["observations"][0][:2],
                                      device=args.device)
                _hi = torch.as_tensor(dataset.normalizer.bounds["observations"][1][:2],
                                      device=args.device)
                xy_world = (trajs[..., 2:4] + 1) * 0.5 * (_hi - _lo) + _lo
                maze_neg, nvalid, _ = block_route_cell(maze, xy_world)
                if bool(nvalid.any()):
                    nloss, ninfo = diffusion.negative_loss(
                        trajs, cond, maze_neg, valid=nvalid)
                    loss = loss + args.negative_weight * nloss
                    info = {**info, **ninfo}
            (loss / args.gradient_accumulate_every).backward()
        opt.step()

        # EMA update (repo schedule: warmup then decay every k steps)
        if step % args.update_ema_every == 0:
            if step < args.step_start_ema:
                ema_model.load_state_dict(diffusion.state_dict())
            else:
                ema.update(ema_model, diffusion)

        val_loss = val_pen = None
        if val_loader is not None and step % args.val_freq == 0:
            val_loss, val_pen = validate(diffusion, val_loader, args)

        if step % args.log_freq == 0 or val_loss is not None:
            loss_log.append((step, float(loss)))
            rate = (step + 1) / (time.time() - t0)
            film_norm = _film_norm(diffusion)
            film_str = f" | film {film_norm:9.5f}" if film_norm is not None else ""
            coll = info.get("collision") if isinstance(info, dict) else None
            coll = None if coll is None else float(coll)
            corn = info.get("corner") if isinstance(info, dict) else None
            corn = None if corn is None else float(corn)
            neg = info.get("neg_collision") if isinstance(info, dict) else None
            neg = None if neg is None else float(neg)
            coll_str = f" | coll {coll:8.5f}" if coll is not None else ""
            coll_str += f" | corn {corn:8.5f}" if corn is not None else ""
            coll_str += f" | neg {neg:8.5f}" if neg is not None else ""
            coll_str += f" | VAL {val_loss:8.5f}" if val_loss is not None else ""
            print(f"{step:>7d} | loss {float(loss):8.5f} | {rate:6.1f} it/s"
                  f"{film_str}{coll_str}", flush=True)
            # crash-safe incremental log, watchable while training runs
            _csv = os.path.join(args.savepath, "loss_log.csv")
            _new = not os.path.exists(_csv)
            with open(_csv, "a") as f:
                if _new:
                    f.write("step,loss,it_per_s,film_norm,collision,corner,"
                            "neg_collision,val_loss,val_penalty\n")
                f.write(f"{step},{float(loss):.6f},{rate:.3f},"
                        f"{'' if film_norm is None else f'{film_norm:.6f}'},"
                        f"{'' if coll is None else f'{coll:.6f}'},"
                        f"{'' if corn is None else f'{corn:.6f}'},"
                        f"{'' if neg is None else f'{neg:.6f}'},"
                        f"{'' if val_loss is None else f'{val_loss:.6f}'},"
                        f"{'' if val_pen is None else f'{val_pen:.6f}'}\n")
            if tb is not None:
                tb.add_scalar("train/loss", float(loss), step)
                tb.add_scalar("train/it_per_s", rate, step)
                if film_norm is not None:
                    tb.add_scalar("train/film_norm", film_norm, step)
                if coll is not None:
                    tb.add_scalar("train/collision", coll, step)
                if corn is not None:
                    tb.add_scalar("train/corner", corn, step)
                if neg is not None:
                    tb.add_scalar("train/neg_collision", neg, step)
                if val_loss is not None:
                    tb.add_scalar("val/loss", val_loss, step)
                    tb.add_scalar("val/penalty", val_pen, step)

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
    """Total weight norm of all zero-initialized conditioning-gate
    projections, or None for backbones that have none. Same diagnostic across
    three different mechanisms, all zero-init at construction so gamma=beta=0
    (or shift=scale=gate=0) regardless of what feeds them:
      * local FiLM:  mapcond.film_models.FiLMBlockWrap.film_out
      * global FiLM: mapcond.global_film_models.GlobalFiLMBlockWrap.film_out
      * DiT adaLN-Zero: mapcond.dit1d.DiTBlock1d / FinalLayer1d's
        adaLN_modulation[-1] -- architecturally the same trick (Peebles & Xie
        zero-init the final adaLN linear so a fresh block starts as an exact
        identity), just applied to tokens instead of conv channels.
    Growth from zero is the direct evidence that the network is starting to
    USE the conditioning signal (time embedding AND map embedding are summed
    into the same vector before any of these gates, in every mechanism above,
    so this cannot separate "using time" from "using the map" -- it can only
    tell you the combined pathway isn't being ignored); a norm stuck near zero
    after hundreds of thousands of steps means the pathway is being ignored."""
    # (class, accessor) pairs: accessor(m) -> the zero-init nn.Linear/Conv1d to
    # measure. Kept as callables rather than a fixed attribute name because the
    # U-Net wraps store it directly (m.film_out) while the DiT blocks store it
    # inside a Sequential (m.adaLN_modulation[-1]).
    wrap_classes = []
    try:
        from mapcond.film_models import FiLMBlockWrap
        wrap_classes.append((FiLMBlockWrap, lambda m: m.film_out))
    except Exception:
        pass
    try:
        from mapcond.global_film_models import GlobalFiLMBlockWrap
        wrap_classes.append((GlobalFiLMBlockWrap, lambda m: m.film_out))
    except Exception:
        pass
    try:
        from mapcond.dit1d import DiTBlock1d, FinalLayer1d
        wrap_classes.append((DiTBlock1d, lambda m: m.adaLN_modulation[-1]))
        wrap_classes.append((FinalLayer1d, lambda m: m.adaLN_modulation[-1]))
    except Exception:
        pass
    if not wrap_classes:
        return None
    total, found = 0.0, False
    for m in diffusion.modules():
        for cls, get_layer in wrap_classes:
            if isinstance(m, cls):
                found = True
                layer = get_layer(m)
                total += float(layer.weight.norm()) + float(layer.bias.norm())
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
    p.add_argument("--global_film", action="store_true",
                   help="Inject the GLOBAL map embedding into every residual "
                        "block via zero-initialized FiLM (mapcond.global_"
                        "film_models.GlobalFiLMTemporalUnet), instead of "
                        "MapConditionalTemporalUnet's addition into the time "
                        "embedding. A clean single-variable alternative to "
                        "the default global tier -- incompatible with "
                        "--local_channels > 0 and --map_blind.")
    p.add_argument("--backbone", type=str, default="unet", choices=["unet", "dit"],
                   help="Denoiser backbone. 'unet' (default) is the existing "
                        "Conv1d TemporalUnet family -- --global_film/"
                        "--local_channels/--local_film/--map_blind all apply "
                        "as before, nothing here changes that path. 'dit' is "
                        "mapcond.dit_models.MapConditionalDiT1D: patchify "
                        "tokenization + sinusoidal position embedding + "
                        "adaLN-Zero conditioning (the Global-FiLM-equivalent "
                        "tier only, for now -- incompatible with "
                        "--global_film/--local_channels/--local_film).")
    p.add_argument("--dit_hidden", type=int, default=256,
                   help="DiT token width.")
    p.add_argument("--dit_heads", type=int, default=8,
                   help="DiT attention heads (head_dim = dit_hidden/dit_heads).")
    p.add_argument("--dit_depth", type=int, default=4,
                   help="Number of DiT blocks.")
    p.add_argument("--dit_patch_size", type=int, default=4,
                   help="Waypoints per token; --horizon must be divisible by "
                        "this.")
    p.add_argument("--dit_mlp_ratio", type=float, default=4.0,
                   help="DiT block MLP hidden width, as a multiple of "
                        "dit_hidden.")
    p.add_argument("--init_from", type=str, default="",
                   help="Warm start from a train_multimap checkpoint (weights "
                        "only; fresh optimizer, EMA and step counter). Model "
                        "config must match -- this loads weights, it does not "
                        "rebuild the architecture from the checkpoint.")
    p.add_argument("--init_from_ema", action="store_true", default=True,
                   help="Warm start from the EMA weights (default) rather than "
                        "the raw model weights.")
    p.add_argument("--collision_weight", type=float, default=0.0,
                   help="Weight of the differentiable wall-collision penalty on "
                        "the predicted x0 (mapcond.collision). 0 (default) is "
                        "off and leaves the loss bit-identical to before. The "
                        "diffusion objective has no negative examples -- every "
                        "demo is a legal path, so nothing otherwise penalizes "
                        "putting a waypoint inside a wall.")
    p.add_argument("--corner_weight", type=float, default=0.0,
                   help="Weight of the corner-clearance penalty, a smooth "
                        "counterpart of the DFS corner detectors (which are "
                        "binary cell-index tests with no gradient). The wall "
                        "term is blind to corner cuts: slipping through a "
                        "diagonal pinch grazes both wall boxes with ~0 "
                        "penetration. Separate from --collision_weight so the "
                        "two failure modes stay separately attributable.")
    p.add_argument("--negative_weight", type=float, default=0.0,
                   help="Weight of the penalty-only pass on a perturbed map "
                        "(mapcond.collision.block_route_cell closes a cell the "
                        "trajectory runs through, manufacturing the exact OOD "
                        "failure mode as a training case). 0 = off. Needs "
                        "--collision_weight and/or --corner_weight to be set, "
                        "since those are the terms it applies. Costs one extra "
                        "forward/backward per step.")
    p.add_argument("--corner_radius_frac", type=float, default=0.25,
                   help="Clearance radius for --corner_weight, as a fraction of "
                        "maze_unit. 0.25 matches the value run_global_film.sh "
                        "and the DiT evals pass to the verifier.")
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
    p.add_argument("--val_freq", type=int, default=5000,
                   help="Evaluate the held-out `-val` twins of the training "
                        "maps every N steps (0 disables). OGBench and the "
                        "variant generator both ship a disjoint 50-episode "
                        "split per map, so nothing is carved out of training. "
                        "Side-effect free: runs under eval() and restores the "
                        "RNG, so the training trajectory is unchanged.")
    p.add_argument("--val_batches", type=int, default=4,
                   help="Batches per validation pass. A fixed random subset is "
                        "drawn once, so the number is comparable across steps.")
    p.add_argument("--val_seed", type=int, default=0,
                   help="Seed for choosing the validation subset and its noise.")
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
        args.val_freq = 10
        args.val_batches = 1
        args.savepath = "logs/mapcond/smoke"
    return args


if __name__ == "__main__":
    train(get_args())
