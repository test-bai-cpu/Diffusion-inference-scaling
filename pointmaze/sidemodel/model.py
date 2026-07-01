"""
model.py — Map-Conditioned Guidance Network (MCGN)
==================================================
A small side model that makes the diffusion planner map-aware WITHOUT
retraining the (frozen) diffusion backbone. It learns a purely local, map-
conditioned policy: given the occupancy patch around a waypoint and the
direction to the goal, predict the geodesic descent direction — the way a
shortest path would leave that cell on THIS map.

Why this transfers to unseen maps
----------------------------------
The network never sees a global map id or absolute coordinates. Its only maze
input is a K×K local occupancy crop centered on the query point (plus the
relative-goal unit vector and log-distance scalar). A shortest-path's local
behavior — hug the free side, turn at a wall, avoid a pocket — is a function of
that local neighborhood, so a model trained on one maze's crops predicts
sensible descent directions on crops it never saw, i.e. on new maps.

Output & use
------------
Predicts a 2D descent direction (unit-ish vector). At inference the MAFGS
composite adds  +w * <predicted_dir, waypoint_velocity>  style guidance (see
sidemodel_verifier.MCGNVerifier): the guidance gradient pushes each waypoint to
move along the predicted feasible descent, complementing the analytic
clearance + distance-field terms. It is an ADDITION to MAFGS, gated by the same
feasibility check, so a bad side-model prediction cannot cut a corner or enter a
dead-end that the analytic gate would reject.
"""
import torch
import torch.nn as nn


class MapConditionedGuidanceNet(nn.Module):
    def __init__(self, patch=15, cnn_ch=(16, 32), mlp_hidden=128):
        """
        patch     : side length K of the square occupancy crop (odd -> centered).
        Input per sample:
            occ  : (B,1,K,K) local occupancy (1=wall, 0=free), point at center.
            goal : (B,3)     [unit_gx, unit_gy, log1p(goal_dist)] relative to point.
        Output:
            dir  : (B,2) predicted descent direction (not normalized; caller may
                   normalize). Trained against the true geodesic descent dir.
        """
        super().__init__()
        self.patch = int(patch)
        c1, c2 = cnn_ch
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c1, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                       # (B,c2,1,1)
            nn.Flatten(),                                  # (B,c2)
        )
        self.head = nn.Sequential(
            nn.Linear(c2 + 3, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden), nn.ReLU(),
            nn.Linear(mlp_hidden, 2),
        )

    def forward(self, occ, goal):
        z = self.cnn(occ)                                  # (B,c2)
        h = torch.cat([z, goal], dim=-1)                   # (B,c2+3)
        return self.head(h)                                # (B,2)


def cosine_direction_loss(pred, target, eps=1e-6):
    """1 - cosine similarity between predicted and target directions, plus a mild
    magnitude term so the model does not collapse to a tiny vector.

    pred, target : (B,2). target is the (already unit) true geodesic descent dir;
    samples with a zero target (e.g. at the goal / unreachable) should be masked
    out by the caller before calling this."""
    pn = pred / (pred.norm(dim=-1, keepdim=True) + eps)
    cos = (pn * target).sum(dim=-1)                        # (B,)
    mag = (pred.norm(dim=-1) - 1.0) ** 2                   # encourage ~unit length
    return (1.0 - cos).mean() + 0.1 * mag.mean()
