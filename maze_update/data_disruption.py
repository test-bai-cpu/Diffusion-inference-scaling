"""
data_disruption.py
==================

Traffic-model loader for the converged Reroute Disruption Index (RDI) pipeline.

Role in the converged pipeline
------------------------------
This module's ONLY job now is to turn an OGBench navigate dataset into a
``TrafficModel`` (per-cell visit counts and per-edge transition counts). That
model is consumed by ``reroute_disruption.py``:

  * ``build_data_model``  -> reads ``TrafficModel`` and builds the per-cell
                             reroute-severity ranking that biases WHERE edits land.
  * ``_route_traffic``    -> sums ``transitions`` along a candidate route to get
                             the route's demonstrated traffic, which feeds the
                             ``traffic_weighted_break`` term of RDI.

The earlier standalone "data disruption" scorer (``score_data_disruption``,
``DataDisruption``, ``WEIGHTS``, ``_transition_invalid``) has been REMOVED: it
measured global transition-collision, which the converged RDI metric replaced
with route-break + spatial-displacement scoring. Keeping it around only invited
confusion over which ``WEIGHTS`` to tune (the live ones are ``RDI_WEIGHTS`` in
``reroute_disruption.py``).

Data format (OGBench navigate datasets)
----------------------------------------
A directory containing:
  observations.npy (N, 2)  float32  xy positions
  terminals.npy    (N,)    bool      episode-end flags
  (actions.npy, qpos.npy, qvel.npy are not needed here)

Coordinate conversion (from ogbench/locomaze/maze.py):
  maze_unit = 4.0, offset = 4.0
  i = int((y + offset + 0.5*maze_unit) / maze_unit)
  j = int((x + offset + 0.5*maze_unit) / maze_unit)
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


MAZE_UNIT = 4.0
OFFSET = 4.0


def xy_to_ij(x, y):
    """Continuous (x, y) observation -> integer maze cell (i, j)."""
    return (int((y + OFFSET + 0.5 * MAZE_UNIT) / MAZE_UNIT),
            int((x + OFFSET + 0.5 * MAZE_UNIT) / MAZE_UNIT))


# ---------------------------------------------------------------------------
# Traffic model
# ---------------------------------------------------------------------------

@dataclass
class TrafficModel:
    """Cell- and transition-level traffic built from the demonstrated data.

    cell_visits  : (i, j)            -> how many observations landed in the cell
    transitions  : ((i,j), (i2,j2))  -> how many directed moves a->b occurred
                                        (directional; the RDI graph symmetrizes
                                        these when it needs an undirected graph)
    """
    cell_visits: dict
    transitions: dict
    total_transitions: int
    n_episodes: int
    grid_shape: tuple
    visited_cells: set = field(default_factory=set)

    def busiest(self, k=5):
        """The k most-visited cells, for quick inspection."""
        return Counter(self.cell_visits).most_common(k)

    def traffic_through_cell(self, cell):
        """Total directed transition traffic entering or leaving ``cell``.

        Useful if you want the per-cell ranking to weight by through-traffic
        instead of raw visit count (see the note in reroute_disruption's
        build_data_model). Sums both directions on every incident edge.
        """
        cell = tuple(cell)
        total = 0
        for (a, b), w in self.transitions.items():
            if a == cell or b == cell:
                total += w
        return total


def build_traffic_model(data_dir: str, subsample: int = 1) -> TrafficModel:
    """Load an OGBench navigate dataset and build the traffic model.

    subsample: take every ``subsample``-th step within each episode (1 = all).
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


if __name__ == "__main__":
    # Quick sanity check / dataset inspector.
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "../pointmaze/ogbench/data/pointmaze-giant-navigate-v0"
    if not os.path.isdir(data_dir):
        print(f"dataset dir not found: {data_dir}")
        print("usage: python3 data_disruption.py <path-to-navigate-dataset>")
    else:
        tm = build_traffic_model(data_dir)
        print(f"episodes            : {tm.n_episodes}")
        print(f"distinct cells      : {len(tm.visited_cells)}")
        print(f"distinct transitions: {len(tm.transitions)}")
        print(f"total transitions   : {tm.total_transitions}")
        print("busiest cells       :")
        for (i, j), c in tm.busiest(8):
            print(f"  ({i:>2},{j:>2})  visits={c}  through-traffic={tm.traffic_through_cell((i, j))}")
