"""
FiLM per-block local map conditioning ("feature grid v2").

Why: in LocalMapConditionalTemporalUnet the two local channels enter ONLY at
the UNet input; every downsample dilutes their per-timestep alignment and the
residual trunk is free to learn to ignore them (l3v2 at 600k: wall violations
reduced but plans still cross blocked corridors). This version gives the local
signal the same standing as the time embedding -- delivered to EVERY residual
block -- but stronger in two ways:
  * per-timestep, not shared across the horizon (the time embedding is a
    single per-channel bias broadcast over time; walls are not);
  * multiplicative FiLM (h * (1 + gamma) + beta) instead of additive bias, so
    "wall under this timestep" can gate feature channels off, a veto the trunk
    cannot cheaply ignore.

gamma/beta are a function of BOTH conditioning signals, not just the map:
the per-position local track and the block's own `t` (== time_mlp(time) +
map_mlp(global embedding), the same vector already driving the additive
conv1(x) + time_mlp(t) path) are each projected to a shared hidden width and
combined by addition -- the same "add two embeddings, then a shared MLP"
idiom this file's parent already uses to fold the global map embedding into
the time pathway. Without this, the gate could not depend on the diffusion
noise level at all: a wall violation at t=999 (mostly noise, position is
close to meaningless) and at t=1 (near-clean, position is real) would be
gated identically, which is backwards -- the veto should sharpen as the
trajectory resolves.

Alignment across depths: the input-resolution local track [B, C_local, H] is
adaptively average-pooled to each block's current temporal length right before
injection, so a coarse level receives "average wall density / average
distance-to-goal over the 2^k-step segment this position represents" -- the
right granularity for route-level decisions, and incidentally smoothing the
noisy-position jitter of early denoise steps.

Zero-init (ControlNet-style): only the final film_out projection is
zero-initialized (weight AND bias), so gamma=beta=0 at init regardless of
what the (normally-initialized) film_local/film_time branches feed into it --
at initialization this model is still bit-identical to
LocalMapConditionalTemporalUnet with the same trunk weights. Trained from
scratch it starts from the same place; alternatively an existing Local
checkpoint can warm-start it via the key remap in tests/film_smoke_test.py
(state-dict keys gain a `.block.` segment because blocks are wrapped, not
rebuilt).

Everything else (input-concat of local channels, global embedding + CFG,
bind_maze, coord-map buffers, dropout semantics) is inherited unchanged from
LocalMapConditionalTemporalUnet.
"""

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import LocalMapConditionalTemporalUnet


class FiLMBlockWrap(nn.Module):
    """
    Wraps an existing ResidualTemporalBlock, reusing its submodules verbatim,
    and inserts FiLM (conditioned on the local map track AND the time/global
    embedding `t` together) between conv1(+time) and conv2:

        out   = conv1(x) + time_mlp(t)
        h     = film_local(film) + film_time(t)          # map ⊕ time, shared hidden
        gamma, beta = film_out(mish(h)).chunk(2)          # <- inserted, zero-init
        out   = out * (1 + gamma) + beta
        out   = conv2(out)
        return out + residual(x)

    film_local: Conv1d(local_dim -> hidden, k=1), per-position (map).
    film_time:  Mish -> Linear(time_dim -> hidden), per-batch, broadcast over
                the horizon before the add (time/global-embedding).
    film_out:   Conv1d(hidden -> 2*out_channels, k=1), weight AND bias
                zero-initialized -- gamma=beta=0 at init no matter what
                film_local/film_time produce, so the wrap is still an exact
                identity at init. film=None skips the modulation entirely
                (used by the map-blind branch).
    """

    def __init__(self, block, local_dim, time_dim):
        super().__init__()
        self.block = block
        out_ch = block.blocks[1].block[0].in_channels
        hidden = out_ch
        self.film_local = nn.Conv1d(local_dim, hidden, kernel_size=1)
        self.film_time = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_dim, hidden),
        )
        self.film_act = nn.Mish()
        self.film_out = nn.Conv1d(hidden, 2 * out_ch, kernel_size=1)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)

    def forward(self, x, t, film=None):
        out = self.block.blocks[0](x) + self.block.time_mlp(t)
        if film is not None:
            h = self.film_local(film) + self.film_time(t).unsqueeze(-1)
            gamma, beta = self.film_out(self.film_act(h)).chunk(2, dim=1)
            out = out * (1 + gamma) + beta
        out = self.block.blocks[1](out)
        return out + self.block.residual_conv(x)


class FiLMLocalMapConditionalTemporalUnet(LocalMapConditionalTemporalUnet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        C = self.local_channels
        # time_dim: width of the `t` every block already receives, i.e.
        # time_mlp(time) + map_mlp(global embedding) -- see forward() below.
        # Read off the parent's own time_mlp rather than re-deriving `dim`.
        T = self.time_mlp[-1].out_features
        for level in self.downs:
            level[0] = FiLMBlockWrap(level[0], C, T)
            level[1] = FiLMBlockWrap(level[1], C, T)
        self.mid_block1 = FiLMBlockWrap(self.mid_block1, C, T)
        self.mid_block2 = FiLMBlockWrap(self.mid_block2, C, T)
        for level in self.ups:
            level[0] = FiLMBlockWrap(level[0], C, T)
            level[1] = FiLMBlockWrap(level[1], C, T)

    def _run_trunk(self, x, t, local=None):
        """local: [B, C_local, H] input-resolution track, or None (blind)."""

        def film_for(cur):
            if local is None:
                return None
            if local.shape[-1] == cur.shape[-1]:
                return local
            return F.adaptive_avg_pool1d(local, cur.shape[-1])

        h = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t, film_for(x))
            x = resnet2(x, t, film_for(x))
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, film_for(x))
        x = self.mid_block2(x, t, film_for(x))

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t, film_for(x))
            x = resnet2(x, t, film_for(x))
            x = upsample(x)

        return self.final_conv(x)

    def forward(self, x, cond, time, maze=None):
        B = x.shape[0]
        if maze is None and self._bound_maze is not None:
            maze = self._bound_maze.expand(B, *self._bound_maze.shape[1:])
        if maze is not None and maze.dim() == 3:
            pad = maze.new_zeros(B, self.local_channels - 1, *maze.shape[1:])
            maze = torch.cat([maze.unsqueeze(1), pad], dim=1)

        xy = x[..., 2:4]
        x_cf = einops.rearrange(x, "b h t -> b t h")
        t_base = self.time_mlp(time)

        if maze is None:
            local = x_cf.new_zeros(B, self.local_channels, x_cf.shape[-1])
            out = self._run_trunk(torch.cat([x_cf, local], dim=1), t_base,
                                  local=None)
            return einops.rearrange(out, "b t h -> b h t")

        emb = self.map_encoder(maze[:, :1])
        local = self.sample_local(maze, xy)
        if self.training and self.cfg_dropout > 0:
            drop = torch.rand(B, device=x.device) < self.cfg_dropout
            emb = torch.where(drop[:, None], self.null_map_emb.unsqueeze(0), emb)
            local = local * (~drop)[:, None, None].float()

        out = self._run_trunk(torch.cat([x_cf, local], dim=1),
                              t_base + self.map_mlp(emb), local=local)

        if (not self.training) and self.guidance_scale > 0:
            null = self.null_map_emb.unsqueeze(0).expand(B, -1)
            zeros = torch.zeros_like(local)
            out_null = self._run_trunk(torch.cat([x_cf, zeros], dim=1),
                                       t_base + self.map_mlp(null), local=zeros)
            out = (1 + self.guidance_scale) * out - self.guidance_scale * out_null

        return einops.rearrange(out, "b t h -> b h t")
