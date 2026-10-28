"""
Map-conditional DiT1D with JOINT attention over trajectory tokens and per-cell
map tokens (MM-DiT style, Esser et al. 2024 / SD3).

Why not a global map vector (MapConditionalDiT1D): cond_sensitivity.py shows the
pooled-vector models respond to a cell being OPENED but not to one being CLOSED,
because one vector broadcast to every token cannot say "the cell under THIS
waypoint is a wall". Why joint attention rather than trajectory->map
cross-attention: with cross-attention the map tokens stay fixed conv features
with a ~7-cell receptive field and never talk to each other, while the decision
that fails -- which corridor to take -- is made at high noise and is a global
connectivity question. Here map tokens attend to each other (and to the
trajectory) in every layer.

Tokens
  trajectory : patchify(x_t) -> N = horizon/patch_size tokens, + 1D index pos
  map        : conv stem on the occupancy grid (no pooling) + raw occupancy
               -> Linear -> M = Hc*Wc tokens (one per canvas cell)
  Only occupancy goes in: no BFS / distance-to-goal channel, so inference needs
  nothing but the grid.

Each JointBlock1d (separate weights per stream, one shared softmax):
  hn = adaLN_t(LN(h)),  mn = adaLN_m(LN(m))
  q,k from (stream + 2D pos), v from stream            (DETR-style pos in q/k)
  attn over [h; m] with logits += -softplus(a_tm[head]) * d2(token, cell)
                                  -softplus(a_mm[head]) * d2(cell, cell)
                                  + offset[head]   (traj query -> map key only)
  h += gate * proj_t(attn_t);  h += gate * MLP_t(adaLN_t(LN(h)))
  m += gate * proj_m(attn_m);  m += gate * MLP_m(adaLN_m(LN(m)))
  The last block only reads the map (its map stream would be discarded).

  2D positions: trajectory token = mean normalized xy of its patch, mapped to
  canvas uv by the same affine as the local-feature track (set_coord_map,
  stored in buffers); map token = cell centre. One FourierPos2d serves both.
  NOTE the trajectory position comes from x_t, so it is noisy at high t
  (~3-4 cells off for t>=150 on giant) -- the self-conditioning fix for that
  is deliberately NOT in this model.

Conditioning vector c = time embedding only; the map enters solely as tokens.

CFG: dropping the map replaces every map token's CONTENT with the learned
null_map_emb (positions are kept); the guidance pass uses the same null.
maze=None means the same thing (map unknown). The name null_map_emb is kept so
mapcond.inference detects a CFG-trained checkpoint.

Self-conditioning (self_cond=True; Chen, Zhang & Hinton 2022, "Analog Bits"):
the model also receives a DRAFT -- its own earlier x0 estimate x_sc -- and
  * the draft is concatenated to x_t at the patch embedding (4 -> 8 channels),
  * trajectory-token 2D positions (Fourier pos + distance bias) are read from
    the draft instead of x_t. This is the point of it: on held-out demos of the
    training maps, mid-route token positions from x_t are 3.4 / 4.5 cells off
    at t=150 / 200, from the model's own x0 estimate 0.2-0.4 / 0.3-1.1 cells,
  * a learned flag (sc_flag) is added to c when a draft is present, so the
    network knows which position source it is looking at.
Samples without a draft (has_sc False) behave exactly as without
self-conditioning, except for the zero half of the input.
Training (two passes, first without gradient) lives in
MapConditionalGaussianDiffusion.p_losses. At inference the model caches its
own drafts when sc_cache_enabled is set (mapcond.inference does this): one per
CFG branch, each pinned at start/goal and clamped to [-1, 1]. The cache is used
only if the next call has the same batch size and a t that is not larger. Equal
t is the TFG recurrence (recur_steps=2 for the newvar mazes,
search/script_utils.py): the second recurrence re-noises back to the same t and
gets the first one's estimate as its draft. A larger t means a new sampling
chain or a DFS back-off to higher noise, which start draft-free;
reset_self_cond() clears it explicitly (DFS calls it on reject / forced_best).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffuser.models.helpers import SinusoidalPosEmb, apply_conditioning

from .dit1d import modulate, PatchEmbed1d, TokenPosEmb, FinalLayer1d


class FourierPos2d(nn.Module):
    """uv in [-1,1]^2 -> [.., hidden]. Fixed frequencies pi*2^k, k=0..n-1
    (k=5 -> period 1/16 in uv, finer than one cell on a 16-wide canvas)."""

    def __init__(self, hidden, n_freqs=6):
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(n_freqs).float(),
                             persistent=False)
        self.proj = nn.Linear(4 * n_freqs, hidden)

    def forward(self, uv):
        ang = uv.unsqueeze(-1) * self.freqs                    # [.., 2, n]
        feat = torch.cat([ang.sin(), ang.cos()], dim=-1)       # [.., 2, 2n]
        return self.proj(feat.flatten(-2))


def _inv_softplus(vals):
    v = torch.as_tensor(vals, dtype=torch.float32).clamp_min(1e-4)
    return torch.log(torch.expm1(v))


def _t2m_offset_init(loc, n_traj, n_map):
    """Per-head traj->map logit offset so that, at init, a trajectory token
    puts ~half its attention on map tokens in EVERY head. Without it the shared
    softmax starves the local heads: N trajectory keys at bias 0 against ~1-2
    map keys that survive a strong distance penalty -> ~1% on the map
    (measured). S(a) = effective number of map keys under exp(-a d^2), from an
    unbounded grid capped at the canvas size; offset = log(N / S)."""
    d = torch.arange(-20, 21, dtype=torch.float32)
    d2 = d[:, None] ** 2 + d[None, :] ** 2
    a = torch.as_tensor(loc, dtype=torch.float32)[:, None, None]
    S = torch.exp(-a * d2).sum((1, 2)).clamp(max=float(n_map))
    return torch.log(float(n_traj) / S)


class JointBlock1d(nn.Module):
    def __init__(self, hidden, heads, mlp_ratio, cond_dim, update_map,
                 loc_tm, loc_mm, t2m_offset):
        super().__init__()
        assert hidden % heads == 0
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim ** -0.5
        self.update_map = update_map
        mlp_hidden = int(hidden * mlp_ratio)

        def mlp():
            return nn.Sequential(nn.Linear(hidden, mlp_hidden),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(mlp_hidden, hidden))

        def ada(n):
            lin = nn.Linear(cond_dim, n * hidden)
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            return nn.Sequential(nn.SiLU(), lin)

        # trajectory stream
        self.t_norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.t_qk = nn.Linear(hidden, 2 * hidden)
        self.t_v = nn.Linear(hidden, hidden)
        self.t_proj = nn.Linear(hidden, hidden)
        self.t_norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.t_mlp = mlp()
        self.t_ada = ada(6)
        # map stream
        self.m_norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.m_v = nn.Linear(hidden, hidden)
        if update_map:
            self.m_qk = nn.Linear(hidden, 2 * hidden)
            self.m_proj = nn.Linear(hidden, hidden)
            self.m_norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
            self.m_mlp = mlp()
            self.m_ada = ada(6)
        else:                       # read-only: keys/values, no update
            self.m_k = nn.Linear(hidden, hidden)
            self.m_ada = ada(2)
        # per-head locality coefficients (softplus-parameterized)
        self.loc_tm = nn.Parameter(_inv_softplus(loc_tm))
        self.loc_mm = nn.Parameter(_inv_softplus(loc_mm))
        self.t2m_offset = nn.Parameter(torch.as_tensor(t2m_offset, dtype=torch.float32))

    def _heads(self, x):
        B, L, C = x.shape
        return x.reshape(B, L, self.heads, self.head_dim).transpose(1, 2)

    def attn_bias(self, D_tm, D_mm, T2M, n_traj):
        """Additive attention bias [B, heads, Lq, L], L = N+M (Lq = N in the
        read-only last block). D_tm [B,L,L]: token-cell d^2 in the traj<->map
        blocks, 0 elsewhere; D_mm [L,L]: cell-cell d^2 in the map<->map block;
        T2M [L,L]: 1 on traj-query/map-key entries. d in cells. The
        batch-independent part is formed at [1,h,..] and the batch-dependent
        term added in one addcmul (this was ~1/3 of forward time otherwise)."""
        rows = slice(None) if self.update_map else slice(0, n_traj)
        a_tm = F.softplus(self.loc_tm).view(1, -1, 1, 1)
        a_mm = F.softplus(self.loc_mm).view(1, -1, 1, 1)
        off = self.t2m_offset.view(1, -1, 1, 1)
        static = off * T2M[None, None, rows] - a_mm * D_mm[None, None, rows]
        return torch.addcmul(static, -a_tm, D_tm[:, None, rows])

    def forward(self, h, m, c, h_pos, m_pos, bias, return_attn=False):
        """h [B,N,C], m [B,M,C], c [B,C], h_pos [B,N,C], m_pos [1,M,C],
        bias from attn_bias()."""
        B, N, C = h.shape
        M = m.shape[1]
        ts1, tsc1, tg1, ts2, tsc2, tg2 = self.t_ada(c).chunk(6, dim=-1)
        hn = modulate(self.t_norm1(h), ts1, tsc1)
        if self.update_map:
            ms1, msc1, mg1, ms2, msc2, mg2 = self.m_ada(c).chunk(6, dim=-1)
        else:
            ms1, msc1 = self.m_ada(c).chunk(2, dim=-1)
        mn = modulate(self.m_norm1(m), ms1, msc1)

        tq, tk = self.t_qk(hn + h_pos).chunk(2, dim=-1)
        if self.update_map:
            mq, mk = self.m_qk(mn + m_pos).chunk(2, dim=-1)
            q = torch.cat([tq, mq], dim=1)
        else:
            mk = self.m_k(mn + m_pos)
            q = tq
        k = torch.cat([tk, mk], dim=1)
        v = torch.cat([self.t_v(hn), self.m_v(mn)], dim=1)
        q, k, v = self._heads(q), self._heads(k), self._heads(v)

        if return_attn:     # manual path: materializes the attention map
            attn = ((q @ k.transpose(-2, -1)) * self.scale + bias).softmax(dim=-1)
            out = attn @ v
        else:
            attn = None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(B, -1, C)

        h = h + tg1.unsqueeze(1) * self.t_proj(out[:, :N])
        h = h + tg2.unsqueeze(1) * self.t_mlp(modulate(self.t_norm2(h), ts2, tsc2))
        if self.update_map:
            m = m + mg1.unsqueeze(1) * self.m_proj(out[:, N:])
            m = m + mg2.unsqueeze(1) * self.m_mlp(modulate(self.m_norm2(m), ms2, msc2))
        return h, m, attn


class JointMapDiT1D(nn.Module):
    def __init__(self, horizon, transition_dim, cond_dim,
                 hidden=256, heads=8, depth=4, patch_size=4, mlp_ratio=4.0,
                 map_encoder_hidden=32, pool_size=None, cfg_dropout=0.1,
                 loc_init=None, self_cond=False):
        super().__init__()
        # pool_size is accepted for interface parity with the other backbones
        # (train_multimap / inference pass it) and is unused: nothing is pooled.
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.patch_size = patch_size
        self.action_dim = transition_dim - cond_dim
        self.self_cond = bool(self_cond)

        in_feat = transition_dim * (2 if self.self_cond else 1)   # [x_t, draft]
        self.patch_embed = PatchEmbed1d(horizon, in_feat, patch_size, hidden)
        self.n_tokens = self.patch_embed.n_tokens
        self.pos_emb = TokenPosEmb(hidden)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # map stem: MapEncoder's conv stack without the pooling, one token/cell
        mh = map_encoder_hidden
        self.map_conv = nn.Sequential(
            nn.Conv2d(1, mh, kernel_size=3, padding=1), nn.GroupNorm(8, mh), nn.Mish(),
            nn.Conv2d(mh, mh, kernel_size=3, padding=1), nn.GroupNorm(8, mh), nn.Mish(),
            nn.Conv2d(mh, 2 * mh, kernel_size=3, padding=1), nn.GroupNorm(8, 2 * mh), nn.Mish(),
        )
        self.map_tok_proj = nn.Linear(2 * mh + 1, hidden)
        self.null_map_emb = nn.Parameter(torch.zeros(hidden))
        self.pos2d = FourierPos2d(hidden)

        if loc_init is None:
            # log-spaced from ~global (exp(-0.01 d^2)) to ~one cell (exp(-2 d^2))
            loc_init = torch.logspace(-2, math.log10(2.0), heads).tolist()
        # offset init assumes the giant canvas (12x16); it is only a starting point
        t2m = _t2m_offset_init(loc_init, self.n_tokens, n_map=12 * 16)
        self.blocks = nn.ModuleList([
            JointBlock1d(hidden, heads, mlp_ratio, cond_dim=hidden,
                         update_map=(i < depth - 1), loc_tm=loc_init, loc_mm=loc_init,
                         t2m_offset=t2m)
            for i in range(depth)
        ])
        self.final_layer = FinalLayer1d(hidden, self.n_tokens, patch_size,
                                        transition_dim, cond_dim=hidden)

        self.cfg_dropout = float(cfg_dropout)
        self.guidance_scale = 0.0
        self._bound_maze = None
        self._negative_maze = None
        self.register_buffer("uv_scale", torch.zeros(2))
        self.register_buffer("uv_shift", torch.zeros(2))
        self.register_buffer("coord_map_ready", torch.zeros(1))
        self.register_buffer("canvas_hw", torch.zeros(2, dtype=torch.long))
        self.return_attn = False
        self._last_attn = None
        if self.self_cond:
            self.sc_flag = nn.Parameter(torch.zeros(hidden))
        self.sc_cache_enabled = False
        self._sc_cache = None

    # ------------------------------------------------------------------ #
    def reset_self_cond(self):
        """Forget the cached drafts (next call runs draft-free)."""
        self._sc_cache = None

    def _cached_drafts(self, time, B):
        cache = self._sc_cache
        t_now = int(time.max().item())
        if cache is None or cache["B"] != B or t_now > cache["t"]:
            self._sc_cache = None
            return None, None
        return cache["cond"], cache["null"]

    def _store_drafts(self, time, cond, out_cond, out_null):
        def pin(o):
            if o is None:
                return None
            o = o.detach().clamp(-1, 1)
            return apply_conditioning(o, cond, self.action_dim) if cond else o
        self._sc_cache = dict(t=int(time.max().item()), B=out_cond.shape[0],
                              cond=pin(out_cond), null=pin(out_null))

    # ------------------------------------------------------------------ #
    def set_coord_map(self, scale, shift, canvas_hw):
        """canvas_hw is stored too: the map-unknown pass (maze=None) still needs
        the cell grid for positions."""
        with torch.no_grad():
            self.uv_scale.copy_(torch.as_tensor(scale, dtype=torch.float32))
            self.uv_shift.copy_(torch.as_tensor(shift, dtype=torch.float32))
            self.canvas_hw.copy_(torch.as_tensor(canvas_hw, dtype=torch.long))
            self.coord_map_ready.fill_(1.0)

    def _as_grid(self, maze):
        maze = torch.as_tensor(maze, dtype=torch.float32)
        if maze.dim() == 2:
            maze = maze.unsqueeze(0)
        return maze.to(next(self.parameters()).device)

    def bind_maze(self, maze):
        """Same contract as MapConditionalDiT1D.bind_maze."""
        if getattr(self, "map_blind", False):
            if maze is not None:
                print("[mapcond] map-blind baseline: ignoring bind_maze()")
            self._bound_maze = None
            return
        self._bound_maze = None if maze is None else self._as_grid(maze)

    def bind_negative_maze(self, maze):
        """Grid for the CFG negative pass instead of the null map. Inference-only."""
        self._negative_maze = None if maze is None else self._as_grid(maze)

    # ------------------------------------------------------------------ #
    def _map_content(self, maze):
        """maze [B,Hc,Wc] -> token content [B,M,C] (no position)."""
        feat = self.map_conv(maze.unsqueeze(1))
        feat = torch.cat([feat, maze.unsqueeze(1)], dim=1)
        return self.map_tok_proj(feat.flatten(2).transpose(1, 2))

    def _geometry(self, x, Hc, Wc):
        """Cell positions and pairwise squared distances (in cells)."""
        assert bool(self.coord_map_ready.item()), \
            "coord map not installed; call set_coord_map() (train_multimap " \
            "does this) or load a checkpoint that contains it"
        dev = x.device
        ii, jj = torch.meshgrid(torch.arange(Hc, device=dev),
                                torch.arange(Wc, device=dev), indexing="ij")
        cell_ij = torch.stack([ii.flatten(), jj.flatten()], dim=-1).float()  # [M,2]
        cell_uv = torch.stack([2 * (cell_ij[:, 1] + 0.5) / Wc - 1,
                               2 * (cell_ij[:, 0] + 0.5) / Hc - 1], dim=-1)

        B = x.shape[0]
        xy = x[..., 2:4].reshape(B, self.n_tokens, self.patch_size, 2).mean(2)
        tok_uv = (xy * self.uv_scale + self.uv_shift).clamp(-1.5, 1.5)
        tok_ij = torch.stack([(tok_uv[..., 1] + 1) * Hc / 2 - 0.5,
                              (tok_uv[..., 0] + 1) * Wc / 2 - 0.5], dim=-1)

        N, M = self.n_tokens, Hc * Wc
        d2_tm = ((tok_ij.unsqueeze(2) - cell_ij.view(1, 1, -1, 2)) ** 2).sum(-1)
        d2_mm = ((cell_ij.unsqueeze(1) - cell_ij.unsqueeze(0)) ** 2).sum(-1)
        # full [L,L] layouts, built once per forward and shared by every block
        D_tm = x.new_zeros(B, N + M, N + M)
        D_tm[:, :N, N:] = d2_tm
        D_tm[:, N:, :N] = d2_tm.transpose(1, 2)
        D_mm = x.new_zeros(N + M, N + M)
        D_mm[N:, N:] = d2_mm
        T2M = x.new_zeros(N + M, N + M)
        T2M[:N, N:] = 1.0
        return (self.pos2d(tok_uv), self.pos2d(cell_uv).unsqueeze(0), D_tm, D_mm, T2M)

    def _biases(self, geo):
        _, _, D_tm, D_mm, T2M = geo
        return [b.attn_bias(D_tm, D_mm, T2M, self.n_tokens) for b in self.blocks]

    def _run(self, x, c, content, geo, biases=None):
        """x: patch-embedding input ([x_t, draft] with self_cond). biases: from
        _biases(geo); pass it in to share across the CFG passes."""
        h_pos, m_pos = geo[0], geo[1]
        if biases is None:
            biases = self._biases(geo)
        h = self.patch_embed(x) + self.pos_emb(self.n_tokens, x.device).unsqueeze(0)
        m = content
        attns = [] if self.return_attn else None
        for block, bias in zip(self.blocks, biases):
            h, m, attn = block(h, m, c, h_pos, m_pos, bias,
                               return_attn=self.return_attn)
            if self.return_attn:
                attns.append(attn)
        self._last_attn = attns
        return self.final_layer(h, c)

    def _sc_inputs(self, x, c_time, x_sc):
        """(patch-embedding input, tensor positions are read from, c) for a
        draft x_sc [B,H,T] or None. A draft row that is all zero = no draft."""
        if not self.self_cond:
            return x, x, c_time
        if x_sc is None:
            return torch.cat([x, torch.zeros_like(x)], dim=-1), x, c_time
        has = x_sc.abs().flatten(1).sum(1) > 0
        return (torch.cat([x, x_sc], dim=-1),
                torch.where(has[:, None, None], x_sc, x),
                c_time + has.float()[:, None] * self.sc_flag)

    def _branch(self, x, c_time, content, Hc, Wc, x_sc):
        """One full pass. Returns (output, geometry, biases) so another branch
        with the same draft can reuse the geometry."""
        inp, geo_src, c = self._sc_inputs(x, c_time, x_sc)
        geo = self._geometry(geo_src, Hc, Wc)
        biases = self._biases(geo)
        return self._run(inp, c, content, geo, biases), geo, biases

    def forward(self, x, cond, time, maze=None, x_sc=None, drop=None):
        """
        x    : [B, horizon, transition_dim]
        maze : [B, H, W] or [B, 1, H, W] occupancy grid, or None (falls back
               to the bound grid; with neither, the map is treated as unknown,
               i.e. the CFG null branch).
        x_sc : self-conditioning draft [B, horizon, transition_dim]; rows that
               are all zero mean "no draft". None: use the inference cache if
               enabled, otherwise run draft-free.
        drop : CFG-dropout mask [B] (bool), sampled by the caller so that the
               two self-conditioning passes drop the SAME samples. None: sample
               it here when training, as before.
        """
        B = x.shape[0]
        if maze is None and self._bound_maze is not None:
            maze = self._bound_maze.expand(B, *self._bound_maze.shape[1:])
        if maze is not None and maze.dim() == 4:
            maze = maze[:, 0]
        c_time = self.time_mlp(time)

        use_cache = (self.self_cond and x_sc is None and not self.training
                     and self.sc_cache_enabled)
        sc_cond, sc_null = self._cached_drafts(time, B) if use_cache else (x_sc, x_sc)

        if maze is None:
            Hc, Wc = (int(v) for v in self.canvas_hw.tolist())
            null = self.null_map_emb.view(1, 1, -1).expand(B, Hc * Wc, -1)
            out, _, _ = self._branch(x, c_time, null, Hc, Wc, sc_cond)
            if use_cache:
                self._store_drafts(time, cond, out, None)
            return out

        Hc, Wc = maze.shape[-2:]
        content = self._map_content(maze)
        null = self.null_map_emb.view(1, 1, -1).expand_as(content)
        if drop is None and self.training and self.cfg_dropout > 0:
            drop = torch.rand(B, device=x.device) < self.cfg_dropout
        if drop is not None:
            content = torch.where(drop[:, None, None], null, content)

        out, geo, biases = self._branch(x, c_time, content, Hc, Wc, sc_cond)
        out_null = None
        if (not self.training) and self.guidance_scale > 0:
            if self._negative_maze is None:
                content_neg = null
            else:
                content_neg = self._map_content(
                    self._negative_maze.expand(B, *self._negative_maze.shape[1:]))
            if sc_null is sc_cond:       # same draft (or none): reuse geometry
                inp, _, c = self._sc_inputs(x, c_time, sc_cond)
                out_null = self._run(inp, c, content_neg, geo, biases)
            else:                        # each CFG branch keeps its own draft
                out_null, _, _ = self._branch(x, c_time, content_neg, Hc, Wc, sc_null)
            if use_cache:
                self._store_drafts(time, cond, out, out_null)
            return (1 + self.guidance_scale) * out - self.guidance_scale * out_null
        if use_cache:
            self._store_drafts(time, cond, out, None)
        return out
