"""
Self-contained 1D DiT (Diffusion Transformer) backbone.

Custom implementation -- no CleanDiffuser dependency (not installed in this
env, and we need attention that is NOT a fused kernel, since a fused
scaled_dot_product_attention call never materializes the [B,heads,N,N] matrix,
which is required for the attention-weight diagnostics planned for the later
cross-attention work). Every block computes attention "by hand" (explicit
QK^T -> softmax -> @V) instead.

Design fixed by the Stage 2 writeup (see conversation history for the full
comparison of alternatives):
  * tokenization: patchify -- `patch_size` adjacent waypoints are concatenated
    and projected into one token, then unpatchified back to `patch_size`
    waypoints at the output. This keeps the diffusion state OUTSIDE this
    module at full [B, horizon, transition_dim] resolution (apply_conditioning
    in diffuser/models/helpers.py never has to change), unlike jump-step
    subsampling, which would throw away supervision on the skipped waypoints.
  * position embedding: sinusoidal (reusing diffuser.models.helpers.
    SinusoidalPosEmb), parameter-free and length-agnostic. The existing Conv1d
    TemporalUnet is sampled at horizon=2800 for the 'ultra' maze even though
    it is trained at horizon=600 (see search/configs.py: sampling_horizon)
    because Conv1d has no fixed sequence-length dependency; a learned absolute
    position embedding would silently break that. Sinusoidal keeps the same
    property for this backbone.
  * conditioning: adaLN-Zero (Peebles & Xie), matching this repo's existing
    zero-init/gated conditioning conventions (mapcond.models' zero_init_map,
    mapcond.global_film_models' film_out, mapcond.film_models' film_out are
    all zero-initialized so the conditioning pathway starts as a no-op and is
    learned in from zero).

This module is deliberately map-agnostic (mirrors diffuser.models.temporal.
TemporalUnet): map conditioning is added in mapcond/dit_models.py the same way
mapcond/models.py layers map conditioning on top of TemporalUnet.
"""
import torch
import torch.nn as nn

from diffuser.models.helpers import SinusoidalPosEmb


def modulate(x, shift, scale):
    """x: [B, N, C], shift/scale: [B, C] -> broadcast over the N tokens."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed1d(nn.Module):
    """[B, horizon, in_features] -> [B, horizon/patch_size, hidden]."""

    def __init__(self, horizon, in_features, patch_size, hidden):
        super().__init__()
        assert horizon % patch_size == 0, \
            f"horizon {horizon} not divisible by patch_size {patch_size}"
        self.horizon = horizon
        self.patch_size = patch_size
        self.n_tokens = horizon // patch_size
        self.proj = nn.Linear(in_features * patch_size, hidden)

    def forward(self, x):
        B, H, F_ = x.shape
        x = x.reshape(B, self.n_tokens, self.patch_size * F_)
        return self.proj(x)


class PatchUnembed1d(nn.Module):
    """Inverse of PatchEmbed1d. Zero-initialized (weight AND bias): output
    starts at exactly 0 regardless of the trunk, same zero-init philosophy as
    the rest of mapcond/*.py (see module docstring)."""

    def __init__(self, n_tokens, patch_size, out_features, hidden):
        super().__init__()
        self.n_tokens = n_tokens
        self.patch_size = patch_size
        self.out_features = out_features
        self.proj = nn.Linear(hidden, patch_size * out_features)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x)
        return x.reshape(B, self.n_tokens * self.patch_size, self.out_features)


class TokenPosEmb(nn.Module):
    """Fixed sinusoidal position embedding over token index (not raw env
    step -- see module docstring). No parameters; works for any n_tokens
    without retraining."""

    def __init__(self, hidden):
        super().__init__()
        self._emb = SinusoidalPosEmb(hidden)

    def forward(self, n_tokens, device):
        pos = torch.arange(n_tokens, device=device, dtype=torch.float32)
        return self._emb(pos)  # [n_tokens, hidden]


class Attention1d(nn.Module):
    """Manual multi-head self-attention (no fused SDPA kernel), so
    [B, heads, N, N] is always a real tensor we can hand back. return_attn
    defaults to False so ordinary training/rollout forward passes (millions
    of denoise steps across training + the 40-sample search rollouts) don't
    pay to keep it around; flip it on only for offline diagnostic runs.
    """

    def __init__(self, hidden, heads):
        super().__init__()
        assert hidden % heads == 0, f"hidden={hidden} not divisible by heads={heads}"
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale     # [B, heads, N, N]
        attn = attn.softmax(dim=-1)
        out = attn @ v                                     # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out, (attn if return_attn else None)


class DiTBlock1d(nn.Module):
    """Standard adaLN-Zero DiT block (Peebles & Xie), ported to 1D tokens:
        h1 = modulate(LN(x), shift1, scale1)
        x  = x + gate1 * Attn(h1)
        h2 = modulate(LN(x), shift2, scale2)
        x  = x + gate2 * MLP(h2)
    (shift1,scale1,gate1,shift2,scale2,gate2) = adaLN_modulation(c).chunk(6)
    adaLN_modulation's Linear is zero-init (weight AND bias) so gate=0 at
    init and the block starts as an exact identity, regardless of what c is.
    """

    def __init__(self, hidden, heads, mlp_ratio=4.0, cond_dim=None):
        super().__init__()
        cond_dim = cond_dim or hidden
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = Attention1d(hidden, heads)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c, return_attn=False):
        shift1, scale1, gate1, shift2, scale2, gate2 = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        attn_out, attn = self.attn(modulate(self.norm1(x), shift1, scale1),
                                   return_attn=return_attn)
        x = x + gate1.unsqueeze(1) * attn_out
        x = x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))
        return x, attn


class FinalLayer1d(nn.Module):
    """adaLN (shift/scale, no gate) + zero-init unpatchify -- standard DiT
    final layer, ported to 1D."""

    def __init__(self, hidden, n_tokens, patch_size, out_features, cond_dim=None):
        super().__init__()
        cond_dim = cond_dim or hidden
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        self.unembed = PatchUnembed1d(n_tokens, patch_size, out_features, hidden)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.unembed(x)


class DiT1D(nn.Module):
    """
    1D DiT backbone. Same external contract as diffuser.models.temporal.
    TemporalUnet:
        forward(x, cond, time) -> [B, horizon, transition_dim]
    `cond` is accepted and ignored, exactly like TemporalUnet -- goal/start
    conditioning happens via apply_conditioning() outside the backbone (see
    diffuser/models/helpers.py), never as a network input.

    Map-blind by construction (no MapEncoder here); mapcond/dit_models.py's
    MapConditionalDiT1D adds that on top, mirroring how mapcond/models.py
    layers map conditioning on top of the plain TemporalUnet.
    """

    def __init__(self, horizon, transition_dim, cond_dim,
                 hidden=256, heads=8, depth=4, patch_size=4, mlp_ratio=4.0):
        super().__init__()
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed1d(horizon, transition_dim, patch_size, hidden)
        self.n_tokens = self.patch_embed.n_tokens
        self.pos_emb = TokenPosEmb(hidden)

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.blocks = nn.ModuleList([
            DiTBlock1d(hidden, heads, mlp_ratio=mlp_ratio, cond_dim=hidden)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer1d(hidden, self.n_tokens, patch_size,
                                        transition_dim, cond_dim=hidden)

        # Diagnostic toggle (off by default -- see Attention1d docstring).
        # NOTE: if a subclass calls _run_trunk twice in one forward() (e.g.
        # MapConditionalDiT1D's CFG guidance double-pass), _last_attn ends up
        # holding whichever call ran LAST -- fine for diagnostics, which are
        # meant to be run with guidance_scale=0, but worth knowing if you ever
        # inspect it right after a guided sample.
        self.return_attn = False
        self._last_attn = None

    def forward(self, x, cond, time):
        c = self.time_mlp(time)
        return self._run_trunk(x, c)

    def _run_trunk(self, x, c):
        """x: [B, horizon, transition_dim], c: [B, hidden] conditioning
        vector (pure timestep, or timestep+map -- see MapConditionalDiT1D)."""
        h = self.patch_embed(x) + self.pos_emb(self.n_tokens, x.device).unsqueeze(0)

        attns = [] if self.return_attn else None
        for block in self.blocks:
            h, attn = block(h, c, return_attn=self.return_attn)
            if self.return_attn:
                attns.append(attn)
        self._last_attn = attns

        return self.final_layer(h, c)
