"""
data_disruption.py
==================

Measure how much editing an OGBench pointmaze map disrupts the *training data
distribution*, rather than a single optimal path.

Motivation
----------
The diffusion planner learns from the demonstrated trajectories in
`pointmaze-<size>-navigate-v0`, not from one shortest path. So the faithful
notion of "out-of-distribution" for the model is: how much of the demonstrated
data does a maze edit invalidate? This module loads the real dataset (the
OGBench .npy arrays), maps every transition onto maze cells, builds a traffic
map, and scores any edited maze against that distribution.

Two complementary signals (mirroring the spirit of the single-path score, where
breaking a heavily-relied-on part of the solution was more severe than breaking
a backwater):

  invalid_fraction          : fraction of ALL demonstrated transitions that
                              become physically invalid after the edit (they now
                              pass through a wall or cross a now-blocked edge).
                              This is the distribution analog of "blocked".

  traffic_weighted_invalid  : same, but each invalidated transition is weighted
                              by how often the data uses it. Blocking the central
                              highway scores high; blocking a quiet dead-end
                              scores low. This captures severity.

  coverage_loss             : fraction of distinct demonstrated cells that the
                              edit walls off entirely (cells the data visited
                              that are now walls).

The headline `data_disruption` score (0..1) is a weighted blend, with the
traffic-weighted term dominating so that "how much the distribution relies on
what you broke" drives the score.

Data format (OGBench navigate datasets)
----------------------------------------
A directory containing:
  observations.npy (N, 2)  float32  xy positions
  actions.npy      (N, 2)
  terminals.npy    (N,)    bool      episode-end flags
  qpos.npy, qvel.npy       (optional for this analysis)

Coordinate conversion (from ogbench/locomaze/maze.py):
  maze_unit = 4.0, offset = 4.0
  i = int((y + offset + 0.5*maze_unit) / maze_unit)
  j = int((x + offset + 0.5*maze_unit) / maze_unit)
"""

from __future__ import annotations
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


MAZE_UNIT = 4.0
OFFSET = 4.0


def xy_to_ij(x, y):
    return (int((y + OFFSET + 0.5 * MAZE_UNIT) / MAZE_UNIT),
            int((x + OFFSET + 0.5 * MAZE_UNIT) / MAZE_UNIT))


# ---------------------------------------------------------------------------
# Dataset -> traffic model
# ---------------------------------------------------------------------------

@dataclass
class TrafficModel:
    """Cell- and transition-level traffic built from the demonstrated data."""
    cell_visits: dict            # (i,j) -> count
    transitions: dict            # ((i,j),(i2,j2)) -> count  (directed, moving only)
    total_transitions: int
    n_episodes: int
    grid_shape: tuple
    visited_cells: set = field(default_factory=set)

    def busiest(self, k=5):
        return Counter(self.cell_visits).most_common(k)


def build_traffic_model(data_dir: str, subsample: int = 1) -> TrafficModel:
    """Load an OGBench navigate dataset and build the traffic model.

    subsample: take every `subsample`-th step within each episode (1 = all).
    Transitions are counted between consecutive *retained* steps, only when the
    cell actually changes (moving transitions), matching how an edit can or
    cannot invalidate a move.
    """
    obs = np.load(os.path.join(data_dir, "observations.npy"))
    term = np.load(os.path.join(data_dir, "terminals.npy"))

    # map every observation to a cell
    cells = np.empty((len(obs), 2), dtype=int)
    for k in range(len(obs)):
        cells[k] = xy_to_ij(obs[k, 0], obs[k, 1])

    ep_end = np.where(term)[0]
    ep_start = np.concatenate([[0], ep_end[:-1] + 1])

    cell_visits = Counter()
    transitions = Counter()
    total = 0
    for s, e in zip(ep_start, ep_end):
        seg = cells[s:e + 1:subsample]
        for c in map(tuple, seg):
            cell_visits[c] += 1
        for a, b in zip(seg[:-1], seg[1:]):
            ta, tb = tuple(a), tuple(b)
            if ta != tb:
                transitions[(ta, tb)] += 1
                total += 1

    return TrafficModel(
        cell_visits=dict(cell_visits),
        transitions=dict(transitions),
        total_transitions=total,
        n_episodes=len(ep_end),
        grid_shape=None,
        visited_cells=set(cell_visits.keys()),
    )


# ---------------------------------------------------------------------------
# Disruption scoring against the data distribution
# ---------------------------------------------------------------------------

@dataclass
class DataDisruption:
    invalid_fraction: float = 0.0
    traffic_weighted_invalid: float = 0.0
    coverage_loss: float = 0.0
    data_disruption: float = 0.0
    n_invalid_transitions: int = 0
    n_total_transitions: int = 0
    cells_walled_off: int = 0
    label: str = ""


WEIGHTS = {"invalid": 0.30, "traffic": 0.55, "coverage": 0.15}


def _transition_invalid(a, b, new_grid):
    """A demonstrated move a->b is invalid on new_grid if either endpoint is now
    a wall, or (for adjacent cells) the step is no longer a legal 4-neighbour
    move. Non-adjacent jumps (rare, from subsampling) count invalid if either
    endpoint is a wall."""
    H, W = new_grid.shape
    for (i, j) in (a, b):
        if not (0 <= i < H and 0 <= j < W) or new_grid[i, j] == 1:
            return True
    # adjacent check
    if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
        return False  # both free and adjacent -> still legal
    return False


def score_data_disruption(traffic: TrafficModel, new_grid) -> DataDisruption:
    new_grid = np.asarray(new_grid)
    d = DataDisruption()
    d.n_total_transitions = traffic.total_transitions

    invalid_count = 0
    invalid_traffic = 0
    total_traffic = 0
    for (a, b), w in traffic.transitions.items():
        total_traffic += w
        if _transition_invalid(a, b, new_grid):
            invalid_count += w          # count weighted by frequency
            invalid_traffic += w
    # raw (unweighted by distinct transition) fraction of transition-instances
    d.n_invalid_transitions = invalid_count
    d.invalid_fraction = invalid_count / max(total_traffic, 1)
    d.traffic_weighted_invalid = invalid_traffic / max(total_traffic, 1)

    # coverage loss: visited cells that became walls
    walled = sum(1 for c in traffic.visited_cells
                 if new_grid[c] == 1)
    d.cells_walled_off = walled
    d.coverage_loss = walled / max(len(traffic.visited_cells), 1)

    d.data_disruption = float(min(max(
        WEIGHTS["invalid"] * d.invalid_fraction
        + WEIGHTS["traffic"] * d.traffic_weighted_invalid
        + WEIGHTS["coverage"] * d.coverage_loss, 0.0), 1.0))

    s = d.data_disruption
    d.label = ("minimal" if s < 0.10 else
               "minor" if s < 0.30 else
               "moderate" if s < 0.55 else
               "major" if s < 0.80 else
               "severe")
    return d


# Note: invalid_fraction and traffic_weighted_invalid are identical here because
# we weight both by frequency w; they are kept as separate fields so that an
# alternative unweighted variant (counting distinct transitions) can be slotted
# in without changing the score interface.