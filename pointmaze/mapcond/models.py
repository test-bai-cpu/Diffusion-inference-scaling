"""
Map-conditional subclasses of the repo's TemporalUnet and GaussianDiffusion.

Design goal: touch nothing in diffuser/. These are drop-in subclasses that add
a global map-conditioning signal (a CNN embedding of the occupancy grid added
to the diffusion time embedding). When maze=None they behave EXACTLY like the
parents, so the paper's single-map reproduction is unaffected.

Conditioning path (global / FiLM-style):
    t = time_mlp(time) + map_mlp(map_encoder(maze))
The existing ResidualTemporalBlock already broadcasts `t` across the horizon,
so no block-level change is needed -- the map signal rides the pathway that the
timestep already uses.
"""
import einops
import torch
import torch.nn as nn

from diffuser.models.temporal import TemporalUnet
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models.helpers import apply_conditioning
import diffuser.utils as utils

from .map_encoder import MapEncoder


class MapConditionalTemporalUnet(TemporalUnet):
    """TemporalUnet + a MapEncoder whose embedding is added to the time embedding."""

    def __init__(self, horizon, transition_dim, cond_dim,
                 dim=32, dim_mults=(1, 2, 4, 8),
                 map_encoder_hidden=32, zero_init_map=False,
                 pool_size=4, cfg_dropout=0.1):
        super().__init__(horizon, transition_dim, cond_dim,
                         dim=dim, dim_mults=dim_mults)
        # time embedding is `dim`-wide (time_dim = dim in the parent); match it.
        self.map_encoder = MapEncoder(out_dim=dim, hidden=map_encoder_hidden,
                                      pool_size=pool_size)
        # ---- classifier-free guidance (CFG) ----
        # null_map_emb stands in for "maze unknown". During training, each
        # sample's encoder output is replaced by it with prob cfg_dropout, so
        # the model learns BOTH p(x|maze) and a marginal-ish p(x). At sampling,
        # guidance_scale w > 0 blends the two passes:
        #     out = (1 + w) * out_cond - w * out_null
        # w lives on the module (not the checkpoint) -- a pure inference knob.
        # Applied to the raw model output, so it is parameterization-agnostic
        # (works for predict_epsilon True or False; this repo uses x0).
        self.cfg_dropout = float(cfg_dropout)
        self.guidance_scale = 0.0
        self.null_map_emb = nn.Parameter(torch.zeros(dim))
        self.map_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(dim, dim),
        )
        # Backward-compatibility with the map-blind parent is STRUCTURAL: when
        # forward() is called with maze=None the whole map pathway is skipped,
        # so maze=None reproduces the parent exactly regardless of init.
        #
        # zero_init_map=True zeroes the last layer so that even WITH a maze the
        # output starts identical to the map-blind model (ControlNet-style). We
        # leave it OFF by default: this model is trained from scratch (there is
        # no pretrained map-blind checkpoint to preserve), and a zero last layer
        # also zeroes the gradient into the MapEncoder on the first step. A
        # normal small init lets the map pathway learn from step 0.
        if zero_init_map:
            nn.init.zeros_(self.map_mlp[-1].weight)
            nn.init.zeros_(self.map_mlp[-1].bias)
        else:
            nn.init.normal_(self.map_mlp[-1].weight, std=0.02)
            nn.init.zeros_(self.map_mlp[-1].bias)

        # Optional episode-level bound grid (see bind_maze). Registered as a
        # buffer so .to(device) moves it; None => map-blind unless maze= passed.
        self._bound_maze = None

    def bind_maze(self, maze):
        """
        Bind an occupancy grid for the whole episode so the search pipeline can
        keep calling unet(x, cond, t) unchanged -- forward() falls back to this
        grid when maze=None. Pass a [H,W] / [1,H,W] tensor, or None to clear.

        This is what makes the map-conditioning OPT-IN at inference without
        editing any search/ code: the pipeline binds the OOD map once per task,
        every denoise step then sees it, and clearing restores map-blind.

        On map-blind baseline checkpoints (trained with --map_blind, map
        pathway never received gradients) binding is refused: injecting an
        UNTRAINED encoder's output would corrupt sampling, so the model stays
        on the maze=None branch, bit-identical to plain TemporalUnet.
        """
        if getattr(self, "map_blind", False):
            if maze is not None:
                print("[mapcond] map-blind baseline: ignoring bind_maze()")
            self._bound_maze = None
            return
        if maze is None:
            self._bound_maze = None
            return
        maze = torch.as_tensor(maze, dtype=torch.float32)
        if maze.dim() == 2:            # [H,W] -> [1,H,W]
            maze = maze.unsqueeze(0)
        dev = next(self.parameters()).device
        self._bound_maze = maze.to(dev)

    def forward(self, x, cond, time, maze=None):
        """
        x    : [B, horizon, transition_dim]
        maze : [B, H, W] or [B, 1, H, W] occupancy grid, or None. When None we
               fall back to a grid bound via bind_maze() (if any); otherwise the
               model is map-blind (identical to the parent TemporalUnet).
        """
        x = einops.rearrange(x, "b h t -> b t h")

        if maze is None and self._bound_maze is not None:
            # expand the single bound grid to the current batch size
            maze = self._bound_maze.expand(x.shape[0], *self._bound_maze.shape[1:])

        t_base = self.time_mlp(time)

        if maze is None:
            # map-blind: skip the map pathway entirely. This branch stays
            # bit-identical to the parent TemporalUnet (backward compat).
            out = self._run_trunk(x, t_base)
            return einops.rearrange(out, "b t h -> b h t")

        emb = self.map_encoder(maze)
        if self.training and self.cfg_dropout > 0:
            # per-sample conditioning dropout: replace encoder output with the
            # learned null embedding, so the null pathway is trained too.
            drop = torch.rand(emb.shape[0], device=emb.device) < self.cfg_dropout
            emb = torch.where(drop[:, None], self.null_map_emb.unsqueeze(0), emb)

        out = self._run_trunk(x, t_base + self.map_mlp(emb))

        if (not self.training) and self.guidance_scale > 0:
            null = self.null_map_emb.unsqueeze(0).expand(emb.shape[0], -1)
            out_null = self._run_trunk(x, t_base + self.map_mlp(null))
            out = (1 + self.guidance_scale) * out - self.guidance_scale * out_null

        return einops.rearrange(out, "b t h -> b h t")

    def _run_trunk(self, x, t):
        """U-shaped trunk on channels-first x [B, C, H] with embedding t."""
        h = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        return self.final_conv(x)


class LocalMapConditionalTemporalUnet(MapConditionalTemporalUnet):
    """
    Adds per-timestep local map conditioning ("feature grid") on top of the
    global-embedding + CFG pathway.

    The bound/passed maze is now a CHANNEL STACK [C, Hc, Wc] (see
    local_features.build_local_stack; C = local_channels, channel 0 must be
    occupancy). Channel 0 feeds the global MapEncoder as before; ALL channels
    are bilinearly sampled at each trajectory timestep's normalized (x, y)
    and concatenated to the UNet input, so every denoise step receives the
    map content under its own feet -- walls and distance-to-goal are delivered
    locally instead of decoded from a summary vector.

    Coordinate mapping is a fixed affine (shared normalizer + fixed canvas),
    installed once via set_coord_map() and stored in checkpoint buffers.

    CFG semantics: conditioning = global embedding AND local channels
    together. Training dropout nulls both for the dropped samples; the
    guidance's unconditional pass uses the null embedding and zeroed local
    channels. maze=None runs "map-blind" with zeroed local channels (well
    defined, but NOT bit-identical to TemporalUnet -- the first block has
    extra input weights).
    """

    def __init__(self, horizon, transition_dim, cond_dim,
                 dim=32, dim_mults=(1, 2, 4, 8),
                 map_encoder_hidden=32, zero_init_map=False,
                 pool_size=4, cfg_dropout=0.1, local_channels=2):
        super().__init__(horizon, transition_dim, cond_dim,
                         dim=dim, dim_mults=dim_mults,
                         map_encoder_hidden=map_encoder_hidden,
                         zero_init_map=zero_init_map,
                         pool_size=pool_size, cfg_dropout=cfg_dropout)
        self.local_channels = int(local_channels)
        assert self.local_channels >= 1, \
            "use MapConditionalTemporalUnet for local_channels=0"
        # rebuild ONLY the first down-block resnet to accept the extra input
        # channels; everything downstream (incl. final_conv -> transition_dim)
        # is unchanged.
        from diffuser.models.temporal import ResidualTemporalBlock
        first_out = dim * dim_mults[0]
        self.downs[0][0] = ResidualTemporalBlock(
            transition_dim + self.local_channels, first_out,
            embed_dim=dim, horizon=horizon)
        # normalized-xy -> grid_sample-uv affine; installed by set_coord_map,
        # persisted in checkpoints as buffers.
        self.register_buffer("uv_scale", torch.zeros(2))
        self.register_buffer("uv_shift", torch.zeros(2))
        self.register_buffer("coord_map_ready", torch.zeros(1))

    def set_coord_map(self, scale, shift):
        with torch.no_grad():
            self.uv_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))
            self.uv_shift.copy_(torch.as_tensor(shift, dtype=torch.float32))
            self.coord_map_ready.fill_(1.0)

    def bind_maze(self, maze):
        """Accepts [Hc,Wc] (occupancy only -> dist channel zeros), [C,Hc,Wc],
        or None to clear."""
        if maze is None:
            self._bound_maze = None
            return
        maze = torch.as_tensor(maze, dtype=torch.float32)
        if maze.dim() == 2:
            pad = maze.new_zeros(self.local_channels - 1, *maze.shape)
            maze = torch.cat([maze.unsqueeze(0), pad], dim=0)
        assert maze.dim() == 3 and maze.shape[0] == self.local_channels, \
            f"expected [{self.local_channels},H,W] stack, got {tuple(maze.shape)}"
        dev = next(self.parameters()).device
        self._bound_maze = maze.unsqueeze(0).to(dev)   # [1,C,H,W]

    def sample_local(self, maze, xy_norm):
        """
        maze [B,C,Hc,Wc], xy_norm [B,H,2] in [-1,1] -> [B,C,H] bilinear
        samples at each timestep's position (border padding: outside the
        canvas reads as the edge, which is wall for occupancy).
        """
        assert bool(self.coord_map_ready.item()), \
            "coord map not installed; call set_coord_map() (train_multimap " \
            "does this) or load a checkpoint that contains it"
        grid = xy_norm * self.uv_scale + self.uv_shift          # [B,H,2]
        grid = grid.unsqueeze(2)                                 # [B,H,1,2]
        out = torch.nn.functional.grid_sample(
            maze, grid, mode="bilinear", padding_mode="border",
            align_corners=False)                                 # [B,C,H,1]
        return out[..., 0]

    def forward(self, x, cond, time, maze=None):
        B = x.shape[0]
        if maze is None and self._bound_maze is not None:
            maze = self._bound_maze.expand(B, *self._bound_maze.shape[1:])
        if maze is not None and maze.dim() == 3:                 # [B,H,W] occ
            pad = maze.new_zeros(B, self.local_channels - 1, *maze.shape[1:])
            maze = torch.cat([maze.unsqueeze(1), pad], dim=1)

        xy = x[..., 2:4]                                         # normalized
        x_cf = einops.rearrange(x, "b h t -> b t h")
        t_base = self.time_mlp(time)

        if maze is None:
            local = x_cf.new_zeros(B, self.local_channels, x_cf.shape[-1])
            out = self._run_trunk(torch.cat([x_cf, local], dim=1), t_base)
            return einops.rearrange(out, "b t h -> b h t")

        emb = self.map_encoder(maze[:, :1])
        local = self.sample_local(maze, xy)
        if self.training and self.cfg_dropout > 0:
            drop = torch.rand(B, device=x.device) < self.cfg_dropout
            emb = torch.where(drop[:, None], self.null_map_emb.unsqueeze(0), emb)
            local = local * (~drop)[:, None, None].float()

        out = self._run_trunk(torch.cat([x_cf, local], dim=1),
                              t_base + self.map_mlp(emb))

        if (not self.training) and self.guidance_scale > 0:
            null = self.null_map_emb.unsqueeze(0).expand(B, -1)
            zeros = torch.zeros_like(local)
            out_null = self._run_trunk(torch.cat([x_cf, zeros], dim=1),
                                       t_base + self.map_mlp(null))
            out = (1 + self.guidance_scale) * out - self.guidance_scale * out_null

        return einops.rearrange(out, "b t h -> b h t")


class MapConditionalGaussianDiffusion(GaussianDiffusion):
    """
    GaussianDiffusion that threads a maze grid through training and sampling to
    self.model(x, cond, t, maze). Every method below mirrors the parent body and
    only adds the `maze` passthrough; maze=None reproduces the parent exactly.
    """

    # -------- training --------
    def set_collision_penalty(self, weight, scale, shift,
                              corner_weight=0.0, corner_radius=1.0,
                              obs_min=None, obs_max=None):
        """
        Enable the differentiable wall-collision term (mapcond.collision).

        weight=0 (the default state, i.e. never calling this) leaves p_losses
        bit-identical to the plain diffusion loss, so every existing checkpoint
        and training command is unaffected.

        scale/shift: the normalized-xy -> grid_sample-uv affine from
        local_features.norm_xy_to_uv_affine, i.e. the same coordinate chain the
        local-feature track already uses.

        Every diffusion timestep is weighted equally, deliberately. An
        SNR-style schedule (scale by alphas_cumprod[t], "loose early, strict
        late", mirroring the DFS noise_scaled threshold) was implemented and
        then removed, because measuring where the violations actually live
        shows it would delete the objective. Predicted-x0 collision penalty by
        timestep, trained DiT on its own training data:

            t=5..100   -> 0.00000      t=150 -> 0.00016
            t=200      -> 0.01258      t=255 -> 0.38727

        All of the signal sits at the TOP of the schedule; below t=100 the
        prediction from a noised legal demo is already legal, so there is
        nothing to weight. alphas_cumprod[t] is ~1 there and ~0 at t=255, i.e.
        exactly inverted relative to the signal.

        Note what this measurement also means: no reweighting can fix the real
        gap. The model's confident, clean-looking-but-illegal plans appear at
        LOW t at sampling time, and training never puts it in that state
        (teacher forcing feeds it noised legal demos). Closing that needs
        violating states in the data, not a different weighting -- that is what
        negative_loss() and --negative_weight are for; see MAPCOND_EXP.md
        section 5.

        corner_weight: weight of the SEPARATE corner-clearance term
        (mapcond.collision.corner_clearance_penalty). Kept separate from
        `weight` on purpose -- wall penetration and corner cutting are two
        different failure modes with two different geometries, and the wall
        term is blind to corner cuts (slipping through a diagonal pinch grazes
        both wall boxes with ~0 penetration). 0 disables it. obs_min/obs_max
        are the shared normalizer's observation bounds, needed because the
        corner geometry is defined in world units, not normalized ones.
        """
        self.collision_weight = float(weight)
        self.corner_weight = float(corner_weight)
        self.corner_radius = float(corner_radius)
        dev = self.betas.device
        self.collision_uv_scale = torch.as_tensor(scale, dtype=torch.float32, device=dev)
        self.collision_uv_shift = torch.as_tensor(shift, dtype=torch.float32, device=dev)
        if obs_min is not None:
            self.obs_min = torch.as_tensor(obs_min, dtype=torch.float32, device=dev)
            self.obs_max = torch.as_tensor(obs_max, dtype=torch.float32, device=dev)

    def _cfg_keep_mask(self, batch_size, device):
        """
        Per-sample weight for the collision terms: 0 for samples whose map was
        dropped by classifier-free-guidance dropout in the forward pass just
        run, 1 otherwise. Returns (weights, normalizer) for a masked mean.

        Backbones advertise the drop mask as `_last_cfg_drop` (see
        MapConditionalDiT1D.forward for why the penalty must skip those
        samples). A backbone that does not set it gets the unmasked behaviour,
        which is what the U-Net arms do today -- they would each need the same
        one-line record if the penalty is ever run on them.
        """
        drop = getattr(self.model, "_last_cfg_drop", None)
        if drop is None:
            keep = torch.ones(batch_size, device=device)
        else:
            keep = (~drop).float()
        return keep, keep.sum().clamp_min(1.0)

    def p_losses(self, x_start, cond, t, maze=None):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t, maze)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        wall_w = getattr(self, "collision_weight", 0.0)
        corner_w = getattr(self, "corner_weight", 0.0)
        if (wall_w > 0 or corner_w > 0) and maze is not None:
            from .collision import (collision_penalty, corner_points,
                                    corner_clearance_penalty)
            # predict_epsilon is False in this repo, so x_recon IS the predicted
            # trajectory and channels action_dim: are its positions.
            xy = x_recon[..., self.action_dim:]
            info = dict(info)
            keep, denom = self._cfg_keep_mask(len(x_start), x_start.device)

            if wall_w > 0:
                pen = (collision_penalty(xy, maze, self.collision_uv_scale,
                                         self.collision_uv_shift)
                       * keep).sum() / denom
                loss = loss + wall_w * pen
                info["collision"] = pen.detach()

            if corner_w > 0:
                # corner geometry lives in world units; unnormalize affinely
                # (no clipping -- a clamp here would kill the gradient exactly
                # where the prediction has strayed out of range).
                xy_world = (xy + 1) * 0.5 * (self.obs_max - self.obs_min) + self.obs_min
                cpen = (corner_clearance_penalty(
                    xy_world, corner_points(maze), self.corner_radius)
                    * keep).sum() / denom
                loss = loss + corner_w * cpen
                info["corner"] = cpen.detach()
        return loss, info

    def loss(self, x, cond, maze=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, maze=maze)

    def negative_loss(self, x, cond, maze_blocked, valid=None, t=None):
        """
        Penalty-only pass on a map the trajectory is known to violate
        (mapcond.collision.block_route_cell closes a cell the trajectory runs
        through).

        No reconstruction term, deliberately: `x` is illegal on `maze_blocked`,
        so asking the model to reproduce it AND to avoid the wall would be two
        contradictory objectives and would train it toward something that is
        neither.

        Why this pass exists: with the map and the demo matched, the predicted
        x0 is already collision-free below t~100 (measured; see
        set_collision_penalty), so the ordinary penalty has no signal in the
        low-noise regime -- which is precisely where sampling produces
        confident, clean-looking, wall-crossing plans. Here the noised input IS
        a wall-crossing trajectory at every noise level, so the model gets the
        missing gradient: "from this state, on this map, predict something that
        is not in the wall".

        `valid` masks out samples where no safe cell could be closed.
        """
        if t is None:
            t = torch.randint(0, self.n_timesteps, (len(x),),
                              device=x.device).long()
        noise = torch.randn_like(x)
        x_noisy = apply_conditioning(self.q_sample(x_start=x, t=t, noise=noise),
                                     cond, self.action_dim)
        x_recon = self.model(x_noisy, cond, t, maze_blocked)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        from .collision import (collision_penalty, corner_points,
                                corner_clearance_penalty)
        xy = x_recon[..., self.action_dim:]
        if valid is None:
            valid = torch.ones(len(x), dtype=torch.bool, device=x.device)
        # CFG dropout applies to this pass too, and it matters more here: the
        # whole content of a negative is "on THIS map that cell is a wall", so a
        # sample that was not shown the map carries no usable signal at all.
        keep, _ = self._cfg_keep_mask(len(x), x.device)
        w = valid.float() * keep
        denom = w.sum().clamp_min(1.0)

        loss = x.new_zeros(())
        info = {}
        if getattr(self, "collision_weight", 0.0) > 0:
            pen = (collision_penalty(xy, maze_blocked, self.collision_uv_scale,
                                     self.collision_uv_shift) * w).sum() / denom
            loss = loss + self.collision_weight * pen
            info["neg_collision"] = pen.detach()
        if getattr(self, "corner_weight", 0.0) > 0:
            xy_world = (xy + 1) * 0.5 * (self.obs_max - self.obs_min) + self.obs_min
            cpen = (corner_clearance_penalty(xy_world, corner_points(maze_blocked),
                                             self.corner_radius) * w).sum() / denom
            loss = loss + self.corner_weight * cpen
            info["neg_corner"] = cpen.detach()
        return loss, info

    # -------- sampling --------
    def p_mean_variance(self, x, cond, t, maze=None):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.model(x, cond, t, maze))
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, maze=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, cond=cond, t=t, maze=maze)
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, maze=None, verbose=True, return_diffusion=False):
        device = self.betas.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        if return_diffusion:
            diffusion = [x]
        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, maze=maze)
            x = apply_conditioning(x, cond, self.action_dim)
            progress.update({"t": i})
            if return_diffusion:
                diffusion.append(x)
        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        return x

    @torch.no_grad()
    def conditional_sample(self, cond, *args, horizon=None, maze=None, **kwargs):
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        return self.p_sample_loop(shape, cond, maze=maze, *args, **kwargs)

    def forward(self, cond, *args, maze=None, **kwargs):
        return self.conditional_sample(cond=cond, *args, maze=maze, **kwargs)
