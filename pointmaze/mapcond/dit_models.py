"""
Map-conditional DiT1D: DiT1D backbone + MapEncoder, adaLN-Zero conditioning.

This is the DiT counterpart of mapcond.global_film_models.GlobalFiLMTemporalUnet,
not of mapcond.models.MapConditionalTemporalUnet's plain-addition scheme: adaLN-
Zero IS "global FiLM applied at every block" -- zero-init gates mean a freshly
bound maze has exactly zero effect until the conditioning pathway is trained in,
the same property GlobalFiLMBlockWrap gets from zero-initializing film_out. See
the Stage 2 writeup in conversation history for the full comparison across all
three existing U-Net conditioning tiers (plain addition / global FiLM / local
grid_sample) and why global FiLM is the one that maps onto DiT for free.

Only the global tier is implemented here. The Local tier (per-waypoint
grid_sample + concat/FiLM, mapcond.models.LocalMapConditionalTemporalUnet and
mapcond.film_models.FiLMLocalMapConditionalTemporalUnet) is a deliberate
follow-up, not implied by this file.

Same external contract as MapConditionalTemporalUnet / GlobalFiLMTemporalUnet
(see mapcond/models.py, mapcond/global_film_models.py) so mapcond/inference.py
and search/ need no changes beyond recognizing this class:
    forward(x, cond, time, maze=None) -> [B, horizon, transition_dim]
    bind_maze(maze) / guidance_scale / cfg_dropout / null_map_emb
"""
import torch
import torch.nn as nn

from .dit1d import DiT1D
from .map_encoder import MapEncoder


class MapConditionalDiT1D(DiT1D):
    def __init__(self, horizon, transition_dim, cond_dim,
                 hidden=256, heads=8, depth=4, patch_size=4, mlp_ratio=4.0,
                 map_encoder_hidden=32, pool_size=4, cfg_dropout=0.1):
        super().__init__(horizon, transition_dim, cond_dim,
                         hidden=hidden, heads=heads, depth=depth,
                         patch_size=patch_size, mlp_ratio=mlp_ratio)
        self.map_encoder = MapEncoder(out_dim=hidden, hidden=map_encoder_hidden,
                                      pool_size=pool_size)
        # ---- classifier-free guidance (CFG), same semantics as
        # MapConditionalTemporalUnet / GlobalFiLMTemporalUnet: null_map_emb
        # stands in for "maze unknown", substituted in with prob cfg_dropout
        # during training; guidance_scale > 0 blends the conditional/null
        # passes at inference (an inference-only knob, not a checkpoint field).
        self.cfg_dropout = float(cfg_dropout)
        self.guidance_scale = 0.0
        self.null_map_emb = nn.Parameter(torch.zeros(hidden))
        # map_mlp uses Mish (not the DiT blocks' SiLU/GELU) to stay consistent
        # with the rest of mapcond/*.py's map-pathway activations (map_encoder.py,
        # mapcond.models' map_mlp) -- the DiT-internal layers (adaLN_modulation,
        # block MLP) follow the literature's DiT convention instead.
        self.map_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(hidden, hidden),
        )
        nn.init.normal_(self.map_mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.map_mlp[-1].bias)

        # Optional episode-level bound grid (see bind_maze). None => map-blind
        # unless maze= is passed explicitly.
        self._bound_maze = None

    def bind_maze(self, maze):
        """Same contract as MapConditionalTemporalUnet.bind_maze /
        GlobalFiLMTemporalUnet.bind_maze, including the map-blind refusal (set
        by mapcond.inference.load_mapcond_diffusion)."""
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
               fall back to a grid bound via bind_maze() (if any); otherwise
               the model is map-blind (pure-timestep adaLN, no map term).
        """
        if maze is None and self._bound_maze is not None:
            maze = self._bound_maze.expand(x.shape[0], *self._bound_maze.shape[1:])

        c_time = self.time_mlp(time)

        if maze is None:
            return self._run_trunk(x, c_time)

        emb = self.map_encoder(maze)
        if self.training and self.cfg_dropout > 0:
            drop = torch.rand(emb.shape[0], device=emb.device) < self.cfg_dropout
            emb = torch.where(drop[:, None], self.null_map_emb.unsqueeze(0), emb)

        c = c_time + self.map_mlp(emb)
        out = self._run_trunk(x, c)

        if (not self.training) and self.guidance_scale > 0:
            null = self.null_map_emb.unsqueeze(0).expand(emb.shape[0], -1)
            c_null = c_time + self.map_mlp(null)
            out_null = self._run_trunk(x, c_null)
            out = (1 + self.guidance_scale) * out - self.guidance_scale * out_null

        return out
