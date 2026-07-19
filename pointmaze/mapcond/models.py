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
        """
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


class MapConditionalGaussianDiffusion(GaussianDiffusion):
    """
    GaussianDiffusion that threads a maze grid through training and sampling to
    self.model(x, cond, t, maze). Every method below mirrors the parent body and
    only adds the `maze` passthrough; maze=None reproduces the parent exactly.
    """

    # -------- training --------
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
        return loss, info

    def loss(self, x, cond, maze=None):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t, maze=maze)

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
