"""
Inference-side helpers: load a map-conditional checkpoint and thread the map
grid through the EXISTING search/verifier pipeline with zero edits to search/.

Why this is opt-in and non-invasive
-----------------------------------
The search pipeline calls the network as `unet(x, cond, batched_t)` inside
guide_step (base/dfs/bfs). The map is CONSTANT for a whole episode, so instead
of threading a `maze=` kwarg through every guide_step signature we:

  1. build the OOD map's occupancy grid on the SAME canvas used in training
     (corner-padded to canvas_hw), from the live env's `maze_map`;
  2. call `unet.bind_maze(grid)` once per task.

`MapConditionalTemporalUnet.forward` falls back to the bound grid whenever
maze=None, so every denoise step is now map-conditioned while the DFS/BFS
search and the MazeVerifier / DistanceFieldVerifier keep running unchanged.
Binding None restores map-blind behaviour.

Two entry points:
  * load_mapcond_diffusion(ckpt_path, ...) -> (diffusion, normalizer, config)
        rebuilds the map-conditional GaussianDiffusion + shared normalizer from
        a train_multimap checkpoint (drop-in for utils.load_diffusion's .ema).
  * bind_env_map(unet, env, canvas_hw, pad_anchor) -> grid
        derive + bind the current env's grid; call once per task in the pipe.
"""
import os, sys, pickle
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_POINTMAZE = os.path.abspath(os.path.join(_HERE, ".."))
if _POINTMAZE not in sys.path:
    sys.path.insert(0, _POINTMAZE)

from mapcond.models import (
    MapConditionalTemporalUnet, MapConditionalGaussianDiffusion,
)
from mapcond import maze_grids as MG


def _infer_encoder_geometry(state):
    """
    Recover (pool_size, has_null_emb) from a state dict, so checkpoints saved
    BEFORE pool_size/cfg_dropout existed in the config still load correctly
    (they were trained with pool_size=1 and no null embedding).
    """
    hidden = None
    in_feat = None
    has_null = False
    for k, v in state.items():
        if k.endswith("map_encoder.conv.0.weight"):
            hidden = v.shape[0]
        elif k.endswith("map_encoder.head.0.weight"):
            in_feat = v.shape[1]
        elif k.endswith("null_map_emb"):
            has_null = True
    if hidden is None or in_feat is None:
        return 1, has_null
    pool_sq = in_feat // (hidden * 2)
    return int(round(pool_sq ** 0.5)), has_null


def load_mapcond_diffusion(ckpt_path, device="cpu", use_ema=True, horizon=None,
                           guidance_scale=0.0):
    """
    Rebuild a map-conditional diffusion model from a train_multimap checkpoint.

    Returns (diffusion, normalizer, config). `diffusion.model` is the
    MapConditionalTemporalUnet; call bind_env_map(diffusion.model, env, ...)
    before rolling out each task. `horizon` overrides the sampling horizon
    (the repo sets diffusion.horizon = sampling_horizon at inference).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    normalizer = pickle.loads(ckpt["normalizer"])

    state = ckpt["ema"] if (use_ema and "ema" in ckpt) else ckpt["model"]
    inferred_pool, has_null = _infer_encoder_geometry(state)
    pool_size = cfg.get("pool_size", inferred_pool)
    cfg_dropout = cfg.get("cfg_dropout", 0.0)

    local_k = int(cfg.get("local_channels", 0))
    if local_k > 0:
        if cfg.get("local_film", False):
            from mapcond.film_models import FiLMLocalMapConditionalTemporalUnet
            model_cls = FiLMLocalMapConditionalTemporalUnet
        else:
            from mapcond.models import LocalMapConditionalTemporalUnet
            model_cls = LocalMapConditionalTemporalUnet
        extra = {"local_channels": local_k}
    else:
        model_cls, extra = MapConditionalTemporalUnet, {}
    model = model_cls(
        horizon=cfg["horizon"],
        transition_dim=cfg["transition_dim"],
        cond_dim=cfg["observation_dim"],
        dim=cfg["dim"],
        dim_mults=tuple(cfg["dim_mults"]),
        **extra,
        pool_size=pool_size,
        cfg_dropout=cfg_dropout,
    )
    diffusion = MapConditionalGaussianDiffusion(
        model,
        horizon=cfg["horizon"],
        observation_dim=cfg["observation_dim"],
        action_dim=cfg["action_dim"],
        n_timesteps=cfg["n_diffusion_steps"],
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=False,
    )
    if has_null:
        diffusion.load_state_dict(state)
    else:
        # pre-CFG checkpoint: no null_map_emb in the state dict. Load the rest
        # and leave the (zero) null embedding untrained -- but then CFG must
        # stay off, because the null pathway was never trained.
        missing, unexpected = diffusion.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        assert all(m.endswith("null_map_emb") for m in missing), \
            f"unexpected missing keys: {missing}"
        if guidance_scale > 0:
            print("[inference] WARNING: checkpoint was trained without "
                  "cfg_dropout; null pathway is untrained. Forcing "
                  "guidance_scale = 0.")
            guidance_scale = 0.0
    diffusion.model.guidance_scale = float(guidance_scale)
    diffusion.model.map_blind = bool(cfg.get("map_blind", False))
    if diffusion.model.map_blind:
        if guidance_scale > 0:
            print("[inference] WARNING: map-blind baseline; forcing "
                  "guidance_scale = 0.")
            diffusion.model.guidance_scale = 0.0
        print("[mapcond] MAP-BLIND baseline checkpoint: maze conditioning "
              "disabled (bind_maze is a no-op).")
    diffusion.to(device).eval()
    if horizon is not None:
        diffusion.horizon = horizon
    # stash geometry for the binder -- on BOTH the wrapper and the unet, since
    # bind_env_map reads canvas_hw from the unet it is handed.
    diffusion._mapcond_config = cfg
    diffusion.model._mapcond_config = cfg
    return diffusion, normalizer, cfg


def env_to_canvas_grid(env, canvas_hw, pad_anchor="corner"):
    """
    Convert the live env's maze_map (0=open, 1=wall) to the training canvas.

    The env grid uses the SAME convention and coordinate constants as training
    (MAZE_UNIT/offsets shared across maps), so we only corner-pad it to the
    common canvas. Returns a float32 [Hc, Wc] array.
    """
    raw = np.asarray(env.maze_map).astype(np.float32)
    padded, _ = MG.pad_to_canvas(raw, canvas_hw, pad_value=1, anchor=pad_anchor)
    return padded.astype(np.float32)


def bind_env_map(unet, env, canvas_hw=None, pad_anchor="corner"):
    """
    Build the current env's occupancy grid on the training canvas and bind it to
    the map-conditional unet for the whole episode. Returns the grid (or None if
    the unet is not map-conditional / has no bind_maze -- a no-op for safety).
    """
    if not hasattr(unet, "bind_maze"):
        return None
    if canvas_hw is None:
        cfg = getattr(unet, "_mapcond_config", None)
        canvas_hw = tuple(cfg["canvas_hw"]) if cfg else None
        if canvas_hw is None:
            # fall back to the training default canvas (giant defines 12x16)
            canvas_hw = MG.canvas_size(MG.all_grids())
    grid = env_to_canvas_grid(env, canvas_hw, pad_anchor=pad_anchor)
    local_k = int(getattr(unet, "local_channels", 0))
    if local_k >= 2:
        # Build the per-timestep channel stack on the TRUE grid: occupancy +
        # BFS distance to this task's goal. The distance field is computed on
        # the actual (possibly OOD) variant topology, so the correct detour is
        # delivered to the model as input even if the learned prior never saw
        # such a route.
        from mapcond import local_features as LF
        goal_ij = None
        goal_xy = None
        ti = getattr(env, "cur_task_info", None)
        if ti is not None and "goal_xy" in ti:
            gx, gy = float(ti["goal_xy"][0]), float(ti["goal_xy"][1])
            goal_xy = (round(gx, 2), round(gy, 2))
            goal_ij = LF.world_xy_to_cell(gx, gy)
        else:
            print("[mapcond] WARNING: env.cur_task_info missing; binding "
                  "all-ones distance channel (goal unknown)")
        stack = LF.build_local_stack(grid, goal_ij)
        unet.bind_maze(stack)
        print(f"[mapcond] bound local stack {stack.shape} "
              f"(goal_xy = {goal_xy} -> goal cell = {goal_ij})")
        return stack
    unet.bind_maze(grid)
    return grid
