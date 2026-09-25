"""
Global-embedding FiLM conditioning ("global FiLM").

Why: MapConditionalTemporalUnet folds the global map embedding into the
diffusion time embedding by ADDITION --

    t = time_mlp(time) + map_mlp(map_encoder(maze))

-- and every block's own `time_mlp(t)` then turns that sum into one shared
additive bias, broadcast unchanged across the whole horizon. Addition can
only SHIFT feature values; it cannot scale or veto them. And because time and
map are pre-summed into a single vector before the per-block projection, the
network cannot even tell them apart once mixed -- it cannot learn "attenuate
this channel when the map is cluttered" independently of "attenuate it at
this noise level", because both facts arrive as one number.

This module keeps the two signals structurally separate from the start and
lets the global embedding modulate every residual block via FiLM instead,
mirroring the local per-position FiLM in film_models.py but for a per-batch
(not per-position) signal:

    t     = time_mlp(time)                              # pure time, no map
    h     = film_emb(emb) + film_time(t)                  # map ⊕ time, shared hidden
    gamma, beta = film_out(mish(h)).chunk(2)               # <- zero-init
    out   = (conv1(x) + block.time_mlp(t)) * (1+gamma) + beta

gamma/beta are the SAME for every horizon position within an episode --
`emb` carries no positional information, unlike the local track, so this is
an honest difference from film_models.py, not an oversight. What FiLM buys
here over addition is purely the scale/veto operation, plus the ability to
learn map- and time-dependent gating as two separate signals instead of one
pre-summed vector.

Zero-init (ControlNet-style): only film_out is zero-initialized (weight AND
bias), so gamma=beta=0 at init regardless of what film_emb/film_time produce.
Unlike MapConditionalTemporalUnet's *optional* zero_init_map, this is
unconditional: there is no other pathway left for map info to reach the
trunk, so a maze-bound GlobalFiLMTemporalUnet is bit-identical to a plain
map-blind TemporalUnet at init, with no flag needed to get there.

Not a subclass of MapConditionalTemporalUnet: existing L0/L1/L2 checkpoints
(all built on the additive t = time_mlp(time) + map_mlp(emb) mechanism) load
into their own classes unchanged; nothing about this file touches them. This
is a new, parallel tier so "global map: FiLM vs addition" stays a clean
single-variable ablation against MapConditionalTemporalUnet. Composing this
with the local track (film_models.py) is a deliberate follow-up, not implied
here -- train_multimap.py currently rejects --global_film together with
--local_channels > 0.
"""

import einops
import torch
import torch.nn as nn

from diffuser.models.temporal import TemporalUnet

from .map_encoder import MapEncoder


class GlobalFiLMBlockWrap(nn.Module):
    """
    Wraps an existing ResidualTemporalBlock and injects FiLM gated by the
    global map embedding (and time), in place of any additive map mixing:

        out   = conv1(x) + time_mlp(t)                # t is pure diffusion time
        h     = film_emb(emb) + film_time(t)             # global-map ⊕ time
        gamma, beta = film_out(mish(h)).chunk(2)          # <- zero-init
        out   = out * (1 + gamma) + beta                  # same gamma/beta at
                                                            # every horizon position
        out   = conv2(out)
        return out + residual(x)

    film_emb / film_time: Linear(emb_dim / time_dim -> hidden), per-batch --
    no Conv1d needed since neither input has a horizon axis (contrast
    film_models.FiLMBlockWrap, whose local track does).
    film_out: Linear(hidden -> 2*out_channels), weight AND bias zero-init.
    emb=None skips the modulation entirely (map-blind).
    """

    def __init__(self, block, emb_dim, time_dim):
        super().__init__()
        self.block = block
        out_ch = block.blocks[1].block[0].in_channels
        hidden = out_ch
        self.film_emb = nn.Linear(emb_dim, hidden)
        self.film_time = nn.Sequential(nn.Mish(), nn.Linear(time_dim, hidden))
        self.film_act = nn.Mish()
        self.film_out = nn.Linear(hidden, 2 * out_ch)
        nn.init.zeros_(self.film_out.weight)
        nn.init.zeros_(self.film_out.bias)

    def forward(self, x, t, emb=None):
        out = self.block.blocks[0](x) + self.block.time_mlp(t)
        if emb is not None:
            h = self.film_emb(emb) + self.film_time(t)          # [B, hidden]
            gamma, beta = self.film_out(self.film_act(h)).chunk(2, dim=-1)
            out = out * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)
        out = self.block.blocks[1](out)
        return out + self.block.residual_conv(x)


class GlobalFiLMTemporalUnet(TemporalUnet):
    """
    TemporalUnet + a MapEncoder whose global embedding modulates every
    residual block via FiLM (see module docstring), instead of being added
    into the time embedding the way MapConditionalTemporalUnet does.

    NOT built on MapConditionalTemporalUnet -- subclasses TemporalUnet
    directly, so `t = time_mlp(time)` stays pure diffusion time throughout;
    map info reaches the trunk ONLY through the FiLM gates below.
    """

    def __init__(self, horizon, transition_dim, cond_dim,
                 dim=32, dim_mults=(1, 2, 4, 8),
                 map_encoder_hidden=32, pool_size=4, cfg_dropout=0.1):
        super().__init__(horizon, transition_dim, cond_dim,
                         dim=dim, dim_mults=dim_mults)
        self.map_encoder = MapEncoder(out_dim=dim, hidden=map_encoder_hidden,
                                      pool_size=pool_size)
        # ---- classifier-free guidance (CFG), same semantics as
        # MapConditionalTemporalUnet: null_map_emb stands in for "maze
        # unknown" and is substituted in with prob cfg_dropout during
        # training; guidance_scale > 0 blends the conditional/null passes at
        # inference. Both passes now run through the FiLM gate, not the
        # additive path.
        self.cfg_dropout = float(cfg_dropout)
        self.guidance_scale = 0.0
        self.null_map_emb = nn.Parameter(torch.zeros(dim))

        # Wrap every residual block. time_dim: width of time_mlp(time),
        # read off the parent's own module rather than re-deriving `dim`.
        T = self.time_mlp[-1].out_features
        for level in self.downs:
            level[0] = GlobalFiLMBlockWrap(level[0], dim, T)
            level[1] = GlobalFiLMBlockWrap(level[1], dim, T)
        self.mid_block1 = GlobalFiLMBlockWrap(self.mid_block1, dim, T)
        self.mid_block2 = GlobalFiLMBlockWrap(self.mid_block2, dim, T)
        for level in self.ups:
            level[0] = GlobalFiLMBlockWrap(level[0], dim, T)
            level[1] = GlobalFiLMBlockWrap(level[1], dim, T)

        self._bound_maze = None

    def bind_maze(self, maze):
        """Bind an occupancy grid for the whole episode so the search
        pipeline can keep calling unet(x, cond, t) unchanged; None clears it.
        Same contract as MapConditionalTemporalUnet.bind_maze, including the
        map-blind refusal (set by mapcond.inference.load_mapcond_diffusion)."""
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

    def _run_trunk(self, x, t, emb=None):
        """U-shaped trunk on channels-first x [B, C, H] with time embedding t
        and global embedding emb (or None -> map-blind)."""
        h = []
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t, emb)
            x = resnet2(x, t, emb)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, emb)
        x = self.mid_block2(x, t, emb)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t, emb)
            x = resnet2(x, t, emb)
            x = upsample(x)

        return self.final_conv(x)

    def forward(self, x, cond, time, maze=None):
        x = einops.rearrange(x, "b h t -> b t h")

        if maze is None and self._bound_maze is not None:
            maze = self._bound_maze.expand(x.shape[0], *self._bound_maze.shape[1:])

        t = self.time_mlp(time)   # pure time -- no map folded in, ever

        if maze is None:
            # map-blind: skip the map pathway entirely. Bit-identical to
            # plain TemporalUnet at ANY point in training, not just init,
            # since t carries no map info to strip out.
            out = self._run_trunk(x, t, emb=None)
            return einops.rearrange(out, "b t h -> b h t")

        emb = self.map_encoder(maze)
        if self.training and self.cfg_dropout > 0:
            drop = torch.rand(emb.shape[0], device=emb.device) < self.cfg_dropout
            emb = torch.where(drop[:, None], self.null_map_emb.unsqueeze(0), emb)

        out = self._run_trunk(x, t, emb=emb)

        if (not self.training) and self.guidance_scale > 0:
            null = self.null_map_emb.unsqueeze(0).expand(emb.shape[0], -1)
            out_null = self._run_trunk(x, t, emb=null)
            out = (1 + self.guidance_scale) * out - self.guidance_scale * out_null

        return einops.rearrange(out, "b t h -> b h t")
