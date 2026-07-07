"""
MapEncoder: binary occupancy grid -> fixed-length embedding vector.

A small conv stack followed by adaptive average pooling makes the encoder
size-agnostic: any HxW grid collapses to the same [B, out_dim] embedding, so
one encoder serves medium (8x8) through giant (12x16) without reshaping. The
embedding is added to the diffusion time embedding (global FiLM-style
conditioning), which the existing ResidualTemporalBlock then broadcasts across
the whole horizon -- reusing the conditioning pathway already in the UNet.
"""
import torch
import torch.nn as nn


class MapEncoder(nn.Module):
    def __init__(self, out_dim, hidden=32, in_ch=1):
        super().__init__()
        self.out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.Mish(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.Mish(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden * 2),
            nn.Mish(),
        )
        # Global average pool -> size-agnostic; then project to out_dim.
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, out_dim),
            nn.Mish(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, maze):
        """
        maze: [B, H, W] or [B, 1, H, W] float tensor, 1 = wall / 0 = open.
        returns: [B, out_dim]
        """
        if maze.dim() == 3:
            maze = maze.unsqueeze(1)  # [B,1,H,W]
        h = self.conv(maze)
        h = self.pool(h).flatten(1)   # [B, hidden*2]
        return self.head(h)           # [B, out_dim]
