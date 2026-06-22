"""
reroute_disruption.py
=====================

Converged maze-variant generator and scorer.

This unifies the two earlier iterations:

  * First iteration  : the per-cell "reroute severity" idea (severity comes from
                       the data's own connectivity graph) and the plot layout you
                       liked (original | traffic | variants, blue=original paths,
                       orange=new paths, red squares=broken cells).
  * Second iteration : the aligned xy drawing (ij_to_xy rectangles so walls,
                       paths and the traffic heatmap share one coordinate frame)
                       and the multi-task driver.

It REPLACES the second iteration's blind-planner / traffic-only pivot, because
in this setup the planner DOES see the map and replans. So the disruption axis
is rerouting, exactly as in your latest spec:

    For each task we ask a planner for up to K candidate routes (not just the
    optimal one). An edit BREAKS a route if any cell on it becomes a wall. The
    score grades by how many of the K routes break, and within that, by how hard
    the forced detour is (a cheap local patch is mild; needing a totally
    different route is severe), folded with how much demonstrated traffic used
    the broken routes.

Metric name
-----------
We call the headline metric the **Reroute Disruption Index (RDI)**, field name
``rdi``. Rename in ONE place: ``METRIC_NAME`` / the ``rdi`` field.

    break 1 of 3 routes -> ~0.3      break 2 of 3 -> ~0.6      break 3 of 3 -> ~0.9
    (with the severity-of-detour and traffic terms modulating within each band)

A fully disconnected goal saturates RDI to 1.0.

Requires the base maze map + tasks (maze_utils) and, for the traffic weighting,
an OGBench navigate dataset directory (observations.npy, terminals.npy). If no
dataset is supplied, RDI still works route-only (traffic weight falls back to
uniform).
"""

from __future__ import annotations

import heapq
import json
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict

import numpy as np

import data_disruption as dd
from maze_utils import MAZE_MAPS, MAZE_TASKS, ij_to_xy, shortest_path, _neighbors


METRIC_NAME = "Reroute Disruption Index (RDI)"   # rename here to rebrand


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# ===========================================================================
# Candidate routes: a planner-style set of up to K diverse routes per task
# ===========================================================================

def k_shortest_paths(grid, start, goal, k=5):
    """Up to k shortest start->goal paths via BFS with bounded node revisits.

    This is the "planner gives several possible trajectories, not only the
    optimal one" piece. Each node may appear on up to k paths so routes can
    diverge instead of all collapsing onto the single geodesic.
    """
    grid = np.asarray(grid)
    if grid[start] == 1 or grid[goal] == 1:
        return []
    heap = [(0, [start])]
    count = defaultdict(int)
    paths = []
    while heap and len(paths) < k:
        cost, path = heapq.heappop(heap)
        node = path[-1]
        count[node] += 1
        if count[node] > k:
            continue
        if node == goal:
            paths.append(path)
            continue
        for nb in _neighbors(node, grid):
            if count[nb] <= k:
                heapq.heappush(heap, (cost + 1, path + [nb]))
    return paths


def _path_via(grid, start, waypoint, goal):
    """Shortest start->waypoint->goal path (concatenated, dedup the seam).
    None if either leg is unreachable or the waypoint is a wall."""
    grid = np.asarray(grid)
    if grid[waypoint] == 1:
        return None
    p1 = shortest_path(grid, start, waypoint)
    p2 = shortest_path(grid, waypoint, goal)
    if p1 is None or p2 is None:
        return None
    return p1 + p2[1:]   # drop duplicated waypoint


def _waypoint_routes(grid, start, goal, region_div=4):
    """Force routes through different REGIONS of the maze so structurally
    distinct alternatives (e.g. the one that swings through the bottom-right
    corner) are guaranteed to appear, regardless of how much longer they are
    than the optimal route.

    The maze is split into a region_div x region_div grid of blocks; for each
    block we pick its most central free cell as a waypoint and build the
    shortest start->waypoint->goal route. These cover the whole maze, so every
    broad alternative corridor is represented. Returned sorted by length so the
    near-optimal ones come first.
    """
    grid = np.asarray(grid)
    H, W = grid.shape
    routes = []
    seen_keys = set()
    for bi in range(region_div):
        for bj in range(region_div):
            r0, r1 = bi * H // region_div, (bi + 1) * H // region_div
            c0, c1 = bj * W // region_div, (bj + 1) * W // region_div
            # candidate free cells in this block, prefer the most central
            cells = [(i, j) for i in range(r0, r1) for j in range(c0, c1)
                     if grid[i, j] == 0]
            if not cells:
                continue
            cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
            wp = min(cells, key=lambda c: (c[0] - cr) ** 2 + (c[1] - cc) ** 2)
            route = _path_via(grid, start, wp, goal)
            if route is None:
                continue
            key = tuple(route)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            routes.append(route)
    routes.sort(key=len)
    return routes


def diverse_routes(grid, start, goal, k=5, overlap_tol=0.75):
    """Up to k candidate routes that are *structurally distinct*, INCLUDING
    longer alternatives (e.g. a route that swings through the bottom-right
    corner), not just near-optimal ones.

    Sources, combined then de-overlapped:
      1. k-shortest pool  -> the near-optimal routes (length ~optimal..+2).
      2. waypoint routes  -> routes forced through every region of the maze,
                             which surfaces the visually-obvious longer
                             alternatives the k-shortest pool misses.

    A candidate is kept only if it overlaps every already-kept route by at most
    ``overlap_tol`` (fraction of the smaller route's cells). The optimal route is
    always seeded first so near-optimal structure is preferred, then longer
    distinct routes fill the remaining slots. No length cap: per your spec, any
    structurally-distinct route counts.
    """
    grid = np.asarray(grid)
    if grid[start] == 1 or grid[goal] == 1:
        return []
    # source pools: near-optimal first (so they seed), then region waypoints
    pool = k_shortest_paths(grid, start, goal, k=max(k * 8, 24))
    pool += _waypoint_routes(grid, start, goal)

    chosen = []
    for p in pool:
        ps = set(map(tuple, p))
        ok = True
        for q in chosen:
            qs = set(map(tuple, q))
            inter = len(ps & qs)
            denom = max(min(len(ps), len(qs)), 1)
            if inter / denom > overlap_tol:
                ok = False
                break
        if ok:
            chosen.append(p)
        if len(chosen) >= k:
            break
    if not chosen and pool:
        chosen = pool[:1]
    return chosen


# ===========================================================================
# Traffic model + per-cell reroute severity (first-iteration idea, retained)
# ===========================================================================

@dataclass
class DataModel:
    traffic: dd.TrafficModel            # may be a uniform stub if no dataset
    adj: dict                           # data connectivity graph: cell -> {nb: w}
    cell_severity: dict                 # cell -> reroute severity (detour cost)
    severity_norm: dict                 # cell -> severity*traffic, normalized
    max_cell_traffic: int
    base_grid: np.ndarray
    has_data: bool = True


def _build_adj(traffic: dd.TrafficModel):
    adj = defaultdict(dict)
    for (a, b), w in traffic.transitions.items():
        adj[a][b] = adj[a].get(b, 0) + w
        adj[b][a] = adj[b].get(a, 0) + w
    return adj


def _reroute_severity(adj, cell):
    """Extra data-graph hops needed to get between this cell's neighbors once the
    cell is removed. Disconnection -> large. (Verbatim spirit of iteration 1.)"""
    nbs = list(adj[cell].keys())
    if len(nbs) < 2:
        return 0.0
    src = nbs[0]
    dist = {src: 0}
    pq = [(0, src)]
    while pq:
        dc, c = heapq.heappop(pq)
        if dc > dist.get(c, 1e9):
            continue
        for nb in adj[c]:
            if nb == cell:
                continue
            nd = dc + 1
            if nd < dist.get(nb, 1e9):
                dist[nb] = nd
                heapq.heappush(pq, (nd, nb))
    incs = []
    for nb in nbs[1:]:
        d = dist.get(nb, None)
        incs.append(10.0 if d is None else max(d - 2, 0))
    return sum(incs) / len(incs)


def _uniform_traffic_stub(base_grid):
    """Build a TrafficModel-shaped stub from the maze graph itself when no OGBench
    dataset is available: every free-cell adjacency gets weight 1. Lets RDI run
    route-only with uniform traffic weighting."""
    base = np.asarray(base_grid)
    H, W = base.shape
    cell_visits, transitions = {}, {}
    free = [(i, j) for i in range(H) for j in range(W) if base[i, j] == 0]
    for c in free:
        cell_visits[c] = 1
    total = 0
    for c in free:
        for nb in _neighbors(c, base):
            transitions[(c, nb)] = 1
            total += 1
    return dd.TrafficModel(
        cell_visits=cell_visits, transitions=transitions,
        total_transitions=total, n_episodes=0,
        grid_shape=base.shape, visited_cells=set(cell_visits.keys()),
    )


def build_data_model(maze_type, data_dir=None, subsample=1) -> DataModel:
    base = np.array(MAZE_MAPS[maze_type])
    if data_dir and os.path.isdir(data_dir):
        traffic = dd.build_traffic_model(data_dir, subsample=subsample)
        has_data = True
    else:
        if data_dir:
            print(f"  [warn] dataset dir not found ({data_dir}); "
                  f"using uniform traffic stub.")
        traffic = _uniform_traffic_stub(base)
        has_data = False
    adj = _build_adj(traffic)
    cells = list(traffic.cell_visits.keys())
    sev = {c: _reroute_severity(adj, c) for c in cells}
    max_traffic = max(traffic.cell_visits.values()) if traffic.cell_visits else 1
    raw = {c: sev[c] * traffic.cell_visits[c] for c in cells}
    mx = max(raw.values()) if raw else 1.0
    mx = mx or 1.0
    sev_norm = {c: raw[c] / mx for c in cells}
    return DataModel(traffic, adj, sev, sev_norm, max_traffic, base, has_data)


# ===========================================================================
# RDI: route-break scoring (your latest spec)
# ===========================================================================

@dataclass
class RouteInfo:
    cells: list                 # the route as list of (i,j)
    traffic: float              # demonstrated traffic mass along the route
    broken: bool = False        # any cell now a wall
    detour_extra: float = 0.0   # spatial displacement of reroute from this route (cells; 1e9 if disconnected)
    detour_norm: float = 0.0    # displacement normalized to [0,1]
    severity: float = 0.0       # per-route break severity in [0,1]


@dataclass
class RDIScore:
    rdi: float = 0.0                    # headline Reroute Disruption Index [0,1]
    n_routes: int = 0
    n_broken: int = 0
    break_fraction: float = 0.0         # n_broken / n_routes
    mean_detour_severity: float = 0.0   # avg spatial displacement of reroutes from broken routes
    traffic_weighted_break: float = 0.0 # share of route-traffic that broke
    route_novelty: float = 0.0          # how far the NEW routes sit from the closest original (redesign-driven)
    disruption_term: float = 0.0        # the "old structure broke" half, [0,1]
    novelty_term: float = 0.0           # the "new structure appeared" half, [0,1]
    n_opened: int = 0                   # interior walls opened by redesign
    goal_reachable: bool = True
    broken_cells: list = field(default_factory=list)
    routes: list = field(default_factory=list)
    label: str = ""
    components: dict = field(default_factory=dict)


# RDI has two complementary halves, combined at the end (see score_rdi):
#
#   DISRUPTION half  -- the ORIGINAL routes break and the planner is forced onto
#                       a displaced reroute. Graded by break_fraction (how many
#                       original routes broke), detour_severity (how far the
#                       forced reroute drifts), and traffic (how used the broken
#                       routes were). This is "you destroyed familiar structure."
#
#   NOVELTY half     -- the EDITED map's own routes sit far from every original
#                       route. This rises when REDESIGN (opening new corridors)
#                       makes the environment genuinely different, EVEN IF no
#                       original route was walled. This is "you introduced
#                       unfamiliar structure." Without it, opening a dramatic new
#                       shortcut scored 0 because nothing broke.
#
# DISRUPTION_WEIGHTS blend the three signals inside the disruption half.
DISRUPTION_WEIGHTS = {"break_fraction": 0.55, "detour_severity": 0.30, "traffic": 0.15}
# Backward-compatible alias (older code / README referred to RDI_WEIGHTS).
RDI_WEIGHTS = DISRUPTION_WEIGHTS

# HALF_WEIGHTS tilts which half the GENERATOR's edits lean toward (used in
# _random_edit: higher novelty weight -> open more corridors). The SCORER
# combines the two halves with a soft-OR (see score_rdi), where both count fully.
HALF_WEIGHTS = {"disruption": 0.6, "novelty": 0.4}

# Displacement normalizer: a reroute / new route is "maximally displaced" when
# its mean-nearest-cell distance from the reference route reaches this fraction
# of (H + W). Smaller -> harsher (small moves already look severe) and raises the
# achievable RDI ceiling; larger -> softer. Used for detour_severity AND novelty.
DISPLACEMENT_SCALE = 0.14


def _route_traffic(model, route):
    """Total demonstrated traffic of the CELLS on this route.

    Sums ``cell_visits`` over every cell the route passes through. This is the
    "traffic of the cells on the route" notion: a route that runs through the
    busy part of the maze (e.g. the top-left highway in the traffic heatmap)
    sums to a large value, so breaking it weighs more in ``traffic_weighted_break``
    than breaking a route through a quiet backwater.

    (Earlier this summed transition-edge counts; cell-visit traffic is the more
    direct reading of "how busy are the cells this route uses" and matches the
    heatmap the variants are compared against.)
    """
    tm = model.traffic
    return float(sum(tm.cell_visits.get(tuple(c), 0) for c in route))


def _mean_nearest_distance(route_a, route_b):
    """Symmetric mean nearest-cell distance (in grid cells) between two routes.

    For every cell on A, find the closest cell on B; average those. Do the same
    from B to A; return the mean of the two directions. Small when the routes hug
    each other, large when they pass through different parts of the maze. This is
    a discrete Hausdorff-style displacement that is INSENSITIVE to length: a
    totally different corridor of similar length still scores high, which is the
    behavior we want.
    """
    A = np.asarray(route_a, dtype=float)
    B = np.asarray(route_b, dtype=float)
    if len(A) == 0 or len(B) == 0:
        return 0.0
    # pairwise Manhattan distances (cells are on a grid; Manhattan matches moves)
    # |A_i - B_j| summed over the 2 coords -> (len(A), len(B))
    d = np.abs(A[:, None, :] - B[None, :, :]).sum(axis=2)
    a_to_b = d.min(axis=1).mean()   # each A cell to nearest B cell
    b_to_a = d.min(axis=0).mean()   # each B cell to nearest A cell
    return float(0.5 * (a_to_b + b_to_a))


def _detour_displacement(new_grid, base_route, start, goal, maze_diag):
    """How far the forced reroute departs *spatially* from the broken route.

    Returns (raw_distance, norm in [0,1]). We compute the planner's new optimal
    path on the edited maze, then measure its mean-nearest-cell distance to the
    SPECIFIC route that broke. Length is irrelevant: a reroute through a
    completely different corridor scores high even if it is the same length.
    A disconnected goal saturates to 1.0. Normalized by the maze diagonal so the
    score is comparable across maze sizes.
    """
    new_path = shortest_path(np.asarray(new_grid), start, goal)
    if new_path is None:
        return np.inf, 1.0
    dist = _mean_nearest_distance(new_path, base_route)
    return dist, float(min(dist / max(maze_diag, 1e-9), 1.0))


def _route_novelty(new_grid, base_routes, start, goal, maze_diag, k=5):
    """How far the EDITED map's own candidate routes sit from the original ones.

    Finds the planner's candidate routes on the NEW map, then for each measures
    its displacement from the CLOSEST original route, and averages (normalized).
    Near 0 when new routes coincide with old corridors; climbs toward 1 when
    redesign (opened walls) creates genuinely new corridors the planner adopts.

    This is what lets "I opened something to make it very different" register,
    even when no original route was walled. A disconnected goal -> 1.0.
    """
    new_routes = diverse_routes(np.asarray(new_grid), start, goal, k=k)
    if not new_routes:
        return 1.0  # goal unreachable: maximally novel (nothing familiar survives)
    norms = []
    for nr in new_routes:
        nr_cells = [tuple(c) for c in nr]
        closest = min(_mean_nearest_distance(nr_cells, [tuple(c) for c in br])
                      for br in base_routes)
        norms.append(min(closest / max(maze_diag, 1e-9), 1.0))
    return float(np.mean(norms))


def score_rdi(model: DataModel, new_grid, start, goal, base_routes=None,
              k=5) -> RDIScore:
    """Reroute Disruption Index for one (start, goal) task on an edited maze."""
    new_grid = np.asarray(new_grid)
    base = model.base_grid
    if base_routes is None:
        base_routes = diverse_routes(base, start, goal, k=k)

    s = RDIScore(n_routes=len(base_routes))
    if not base_routes:
        return s  # task ill-defined on base maze

    # goal reachable on the edited maze?
    s.goal_reachable = shortest_path(new_grid, start, goal) is not None

    broken_cells = set()
    routes_info = []
    total_traffic = 0.0
    broken_traffic = 0.0
    detour_sevs = []

    # Normalize displacement by a fraction of the maze diagonal. Using the full
    # diagonal makes even big reroutes score low; ~1/3 of it means "a reroute
    # that averages a third of the maze away from the broken route is maximally
    # displaced." Tune DISPLACEMENT_SCALE if you want it harsher/softer.
    H, W = base.shape
    maze_diag = (H + W) * DISPLACEMENT_SCALE

    for route in base_routes:
        cells = [tuple(c) for c in route]
        rt_traffic = _route_traffic(model, cells)
        total_traffic += rt_traffic
        broken = any(new_grid[c] == 1 for c in cells)
        ri = RouteInfo(cells=cells, traffic=rt_traffic, broken=broken)
        if broken:
            broken_traffic += rt_traffic
            for c in cells:
                if new_grid[c] == 1:
                    broken_cells.add(c)
            raw, dnorm = _detour_displacement(new_grid, cells, start, goal, maze_diag)
            ri.detour_extra = (10 ** 9 if raw == np.inf else round(float(raw), 3))
            ri.detour_norm = dnorm
            ri.severity = dnorm
            detour_sevs.append(dnorm)
        routes_info.append(ri)

    s.n_broken = sum(1 for r in routes_info if r.broken)
    s.break_fraction = s.n_broken / max(s.n_routes, 1)
    s.mean_detour_severity = float(np.mean(detour_sevs)) if detour_sevs else 0.0
    s.traffic_weighted_break = broken_traffic / max(total_traffic, 1e-9)
    s.broken_cells = [list(map(int, c)) for c in sorted(broken_cells)]
    s.routes = routes_info
    s.n_opened = int(((new_grid == 0) & (base == 1)).sum())

    if not s.goal_reachable:
        s.rdi = 1.0
        s.disruption_term = 1.0
        s.novelty_term = 1.0
        s.route_novelty = 1.0
    else:
        # --- DISRUPTION half: original routes broke + forced displacement ---
        # When there are few distinct routes, break_fraction is coarse (0, 0.5,
        # 1.0 for 2 routes; 0/1 for a single-route task), so it cannot grade a
        # smooth sweep on single-exit tasks. We shift weight toward the CONTINUOUS
        # detour-severity term as n_routes shrinks: with many routes break_fraction
        # leads (your "1 of 3 -> 0.3" intuition); with one route detour hardness
        # carries the grade. Keeps every task sweepable.
        n = max(s.n_routes, 1)
        coarse_w = min((n - 1) / 4.0, 1.0)   # 0 for 1 route, ramps to 1 by 5 routes
        w_break = DISRUPTION_WEIGHTS["break_fraction"] * coarse_w
        w_detour = (DISRUPTION_WEIGHTS["detour_severity"]
                    + DISRUPTION_WEIGHTS["break_fraction"] * (1.0 - coarse_w))
        w_traffic = DISRUPTION_WEIGHTS["traffic"]
        s.disruption_term = float(min(max(
            w_break * s.break_fraction
            + w_detour * s.mean_detour_severity
            + w_traffic * s.traffic_weighted_break, 0.0), 1.0))

        # --- NOVELTY half: how DIFFERENT is the environment the planner now
        # actually traverses, vs the originals? Two parts:
        #   (a) route_novelty  -- the edited map's candidate routes vs originals
        #                         (rewards redesign-opened corridors).
        #   (b) survivor_novelty -- how far the planner's CHOSEN optimal route is
        #                         from the closest original. This is the fix for
        #                         "broke 3/4 but the survivor is an untouched
        #                         original": such a variant has high break_fraction
        #                         but LOW survivor_novelty, so it no longer counts
        #                         as fully disruptive. Transformation must be real.
        s.route_novelty = _route_novelty(new_grid, base_routes, start, goal,
                                         maze_diag, k=k)
        sp = shortest_path(new_grid, start, goal)
        if sp is None:
            survivor_novelty = 1.0
        else:
            sp_cells = [tuple(c) for c in sp]
            closest = min(_mean_nearest_distance(sp_cells, [tuple(c) for c in br])
                          for br in base_routes)
            survivor_novelty = float(min(closest / max(maze_diag, 1e-9), 1.0))
        s.route_novelty = max(s.route_novelty, survivor_novelty)
        s.novelty_term = s.route_novelty

        # --- combine the two halves into the headline RDI ---
        # We want TRANSFORMATION (the route the planner takes is genuinely
        # different) to matter as much as ELIMINATION (many routes broken). So
        # the disruption term is GATED by how much real change occurred: pure
        # elimination that leaves an identical survivor (novelty ~ 0) is damped,
        # while breaking routes AND displacing the survivor scores full value.
        a = s.disruption_term
        b = s.novelty_term
        # gate: elimination counts fully only when accompanied by some change.
        # Softened (0.7 + 0.3*b) because on dense mazes the achievable novelty is
        # modest, so a harsh gate would make high RDI unreachable. Pure
        # elimination (b=0) keeps 70% of its disruption; displacing the survivor
        # restores the rest. Transformation still clearly matters, but does not
        # make the high end impossible.
        gated_disruption = a * (0.7 + 0.3 * b)
        s.rdi = float(min(max(
            0.5 * max(gated_disruption, b)
            + 0.5 * (1.0 - (1.0 - gated_disruption) * (1.0 - b)), 0.0), 1.0))

    v = s.rdi
    s.label = ("minimal" if v < 0.10 else "minor" if v < 0.30 else
               "moderate" if v < 0.55 else "major" if v < 0.80 else "severe")
    s.components = {
        "disruption_term": round(s.disruption_term, 4),
        "novelty_term": round(s.novelty_term, 4),
        "break_fraction": round(s.break_fraction, 4),
        "mean_detour_severity": round(s.mean_detour_severity, 4),
        "traffic_weighted_break": round(s.traffic_weighted_break, 4),
        "route_novelty": round(s.route_novelty, 4),
        "n_broken": s.n_broken,
        "n_routes": s.n_routes,
        "n_opened": s.n_opened,
    }
    return s


# ===========================================================================
# Edits: block-only, and "redesign" (close one way, open another)
# ===========================================================================

def _interior_walls(grid):
    H, W = grid.shape
    return [(i, j) for i in range(1, H - 1) for j in range(1, W - 1)
            if grid[i, j] == 1]


def _random_edit(model, start, goal, target=0.0, redesign=True, n_open_max=4,
                 base_routes=None, survivor_overlap=0.6):
    """One random edit aimed loosely at `target` RDI.

    Higher target -> bias blocking toward high-severity (bottleneck) cells and
    block a few more of them, so more routes break and the forced detour is
    harder. With redesign=True we may also OPEN a wall (carve a new corridor)
    while closing an old one, so the map keeps a comparable amount of free
    space but the route SET changes -- "block one way, open a different new way".

    SURVIVOR DISPLACEMENT (high targets): killing N-1 routes and leaving one
    ORIGINAL route intact gives high break_fraction but the planner still drives
    a path it already knew (no transformation). To fix that, after the initial
    edits we iteratively inspect the current optimal route: while it still
    overlaps some original route by more than `survivor_overlap`, we wall a
    distinctive cell ON that surviving route (forcing it to reroute), so the
    route the planner ends up on is genuinely different from every original.
    Strength scales with `target`.
    """
    base = model.base_grid
    new = base.copy()
    visited = [c for c in model.traffic.visited_cells if c not in (start, goal)]
    if not visited:
        return new

    exponent = 1.0 + target * 3.0
    weights = np.array([model.severity_norm.get(c, 0.0) ** exponent for c in visited])
    if weights.sum() == 0:
        weights = np.ones(len(visited))
    weights /= weights.sum()

    n_block_max = max(1, round(1 + target * 4))
    n_block = random.randint(1, n_block_max)
    idx = np.random.choice(len(visited), size=min(n_block, len(visited)),
                           replace=False, p=weights)
    for i in idx:
        c = visited[i]
        new[c] = 1
        if shortest_path(new, start, goal) is None:
            new[c] = 0  # never fully disconnect during generation

    # Redesign: open interior walls to create alternative corridors. This now
    # DRIVES the novelty half of RDI, so high targets open MORE walls (the way to
    # reach high RDI is to make the map genuinely different, not just to wall off
    # routes -- which hits the ~0.7 break-only ceiling). Bias toward low-traffic
    # walls so we add genuinely new structure rather than reopening the highway.
    if redesign and random.random() < (0.4 + 0.55 * target):
        int_walls = _interior_walls(base)
        if int_walls:
            ow = np.array([1.0 / (1.0 + model.traffic.cell_visits.get(c, 0))
                           for c in int_walls])
            ow /= ow.sum()
            # high target -> open more walls (scale the cap with target)
            this_open_max = max(1, round(n_open_max + target * 6))
            n_open = random.randint(1, this_open_max)
            oidx = np.random.choice(len(int_walls), size=min(n_open, len(int_walls)),
                                    replace=False, p=ow)
            for i in oidx:
                new[int_walls[i]] = 0

    # --- survivor displacement: force the route the planner ACTUALLY takes to
    # differ from every original. On these mazes EVERY block-only route is one of
    # the originals (the diverse set already covers all corridors), so blocking
    # alone just hops the survivor to the next original. The only way to a truly
    # different survivor is REDESIGN: open a NEW corridor and block the original
    # the survivor is using, so the planner is pushed onto the new passage. ---
    if base_routes and redesign and target > 0.0:
        orig_sets = [set(map(tuple, p)) for p in base_routes]
        max_iters = max(0, round(target * 5))
        for _ in range(max_iters):
            sp = shortest_path(new, start, goal)
            if sp is None:
                break
            sp_set = set(map(tuple, sp))
            ov = max(len(sp_set & os) / max(min(len(sp_set), len(os)), 1)
                     for os in orig_sets)
            if ov <= survivor_overlap:
                break  # survivor already distinct enough
            # 1) open a few interior walls NEAR the survivor to create fresh
            #    corridors it could divert onto (prefer walls adjacent to the
            #    current route so the new passage actually competes).
            int_walls = _interior_walls(base)
            near = [w for w in int_walls
                    if any(abs(w[0] - c[0]) + abs(w[1] - c[1]) == 1 for c in sp)]
            pool_open = near if near else int_walls
            if pool_open:
                for w in random.sample(pool_open, min(len(pool_open),
                                                      random.randint(2, 5))):
                    new[w] = 0
            # 2) block a distinctive middle cell of the current (original)
            #    survivor so the planner must leave it.
            cand = [c for c in sp if c not in (start, goal)]
            cand.sort(key=lambda c: -min(sp.index(c), len(sp) - 1 - sp.index(c)))
            for c in cand:
                new[c] = 1
                if shortest_path(new, start, goal) is None:
                    new[c] = 0
                else:
                    break
    return new


# ===========================================================================
# Variant generation: target an RDI level, with randomness -> multiple variants
# ===========================================================================

def generate_variants_at_level(model, start, goal, target, n_variants=3,
                               tol=0.08, redesign=True, k=5,
                               allow_unreachable=True, max_attempts=20000):
    """Find n_variants DISTINCT edited mazes whose RDI is near `target`."""
    base_routes = diverse_routes(model.base_grid, start, goal, k=k)
    found, seen, attempts = [], set(), 0
    while len(found) < n_variants and attempts < max_attempts:
        attempts += 1
        new = _random_edit(model, start, goal, target, redesign=redesign,
                            base_routes=base_routes)
        sc = score_rdi(model, new, start, goal, base_routes=base_routes, k=k)
        if not sc.goal_reachable and not allow_unreachable:
            continue
        if attempts % 1000 == 0:
            print(f"  target={target:.1f} | attempts={attempts} "
                  f"| found={len(found)}/{n_variants} | last={sc.rdi:.3f}",
                  flush=True)
        if abs(sc.rdi - target) <= tol:
            key = new.tobytes()
            if key in seen:
                continue
            seen.add(key)
            found.append((new, sc, sc.goal_reachable))
            print(f"  target={target:.1f} | found {len(found)}/{n_variants} "
                  f"(rdi={sc.rdi:.3f}, broke {sc.n_broken}/{sc.n_routes}, "
                  f"attempts={attempts})", flush=True)
    if len(found) < n_variants:
        print(f"  [note] target={target:.1f}: only {len(found)}/{n_variants} "
              f"after {attempts} attempts (this RDI level may be hard to hit "
              f"with {sc.n_routes if 'sc' in dir() else k} routes).", flush=True)
    return found


# ===========================================================================
# Suite + output
# ===========================================================================

@dataclass
class Record:
    maze_type: str
    task: str
    start: tuple
    goal: tuple
    start_xy: tuple
    goal_xy: tuple
    target_level: float
    variant_index: int
    rdi: float
    components: dict
    break_fraction: float
    mean_detour_severity: float
    traffic_weighted_break: float
    route_novelty: float
    disruption_term: float
    novelty_term: float
    n_opened: int
    n_broken: int
    n_routes: int
    goal_reachable: bool
    broken_cells: list
    maze_map: list


def generate_suite(maze_type="giant", task="task5", data_dir=None,
                   levels=None, variants_per_level=3, redesign=True,
                   k=5, subsample=1):
    if levels is None:
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    model = build_data_model(maze_type, data_dir=data_dir, subsample=subsample)
    start, goal = MAZE_TASKS[maze_type][task]
    records = []
    for lvl in levels:
        print(f"\n[level {lvl}]", flush=True)
        hits = generate_variants_at_level(model, start, goal, lvl,
                                          n_variants=variants_per_level,
                                          redesign=redesign, k=k)
        for vi, (new, sc, reachable) in enumerate(hits):
            records.append(Record(
                maze_type=maze_type, task=task, start=start, goal=goal,
                start_xy=ij_to_xy(start), goal_xy=ij_to_xy(goal),
                target_level=lvl, variant_index=vi,
                rdi=round(sc.rdi, 4), components=sc.components,
                break_fraction=round(sc.break_fraction, 4),
                mean_detour_severity=round(sc.mean_detour_severity, 4),
                traffic_weighted_break=round(sc.traffic_weighted_break, 4),
                route_novelty=round(sc.route_novelty, 4),
                disruption_term=round(sc.disruption_term, 4),
                novelty_term=round(sc.novelty_term, 4),
                n_opened=sc.n_opened,
                n_broken=sc.n_broken, n_routes=sc.n_routes,
                goal_reachable=reachable,
                broken_cells=sc.broken_cells, maze_map=new.tolist(),
            ))
    return records, model


# ===========================================================================
# Visualization  (first-iteration layout, second-iteration aligned xy drawing)
# ===========================================================================

def plot_suite(records, model, outpath, maze_type, task, k=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    MAZE_UNIT = 4.0
    base = model.base_grid
    tm = model.traffic
    H, W = base.shape

    all_xs = [ij_to_xy((0, j))[0] for j in range(W)]
    all_ys = [ij_to_xy((i, 0))[1] for i in range(H)]
    xlim = (min(all_xs) - MAZE_UNIT / 2, max(all_xs) + MAZE_UNIT / 2)
    ylim = (min(all_ys) - MAZE_UNIT / 2, max(all_ys) + MAZE_UNIT / 2)

    def _setup(ax):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")

    def _bg(ax, grid):
        grid = np.array(grid)
        ax.set_facecolor("white")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 1:
                    x, y = ij_to_xy((i, j))
                    ax.add_patch(Rectangle((x - MAZE_UNIT / 2, y - MAZE_UNIT / 2),
                                           MAZE_UNIT, MAZE_UNIT, color="black", zorder=1))

    start, goal = tuple(records[0].start), tuple(records[0].goal)

    def _mark(ax):
        sx, sy = ij_to_xy(start); gx, gy = ij_to_xy(goal)
        ax.scatter([sx], [sy], c="limegreen", s=120, marker="*", edgecolors="k", zorder=6)
        ax.scatter([gx], [gy], c="gold", s=120, marker="X", edgecolors="k", zorder=6)

    def _paths(ax, paths, color, lw=1.8, base_alpha=0.8):
        for idx, path in enumerate(paths):
            if path and len(path) > 1:
                xs = [ij_to_xy(tuple(p))[0] for p in path]
                ys = [ij_to_xy(tuple(p))[1] for p in path]
                ax.plot(xs, ys, color=color, lw=lw,
                        alpha=base_alpha * (0.9 ** idx), zorder=4)

    by_level = {}
    for r in records:
        by_level.setdefault(r.target_level, []).append(r)
    levels = sorted(by_level)
    n_var = max((len(v) for v in by_level.values()), default=1)

    rows = max(len(levels), 1)
    cols = 2 + n_var
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.2 * rows))
    axes = np.atleast_2d(axes)
    if axes.shape != (rows, cols):
        axes = axes.reshape(rows, cols)

    heat = np.zeros((H, W))
    for (i, j), c in tm.cell_visits.items():
        heat[i, j] = np.log1p(c)
    heat_display = np.ma.masked_where(base == 1, heat)

    orig_routes = diverse_routes(base, start, goal, k=k)

    for ri, lvl in enumerate(levels):
        ax0 = axes[ri, 0]; _setup(ax0); _bg(ax0, base)
        _paths(ax0, orig_routes, "royalblue"); _mark(ax0)
        if ri == 0:
            ax0.set_title(f"original ({len(orig_routes)} routes)", fontsize=8)

        ax1 = axes[ri, 1]; _setup(ax1)
        ax1.imshow(base, cmap="binary", origin="lower",
                   extent=[*xlim, *ylim], alpha=0.3, zorder=1)
        ax1.imshow(heat_display, cmap="hot", origin="lower",
                   extent=[*xlim, *ylim], alpha=0.85, zorder=2)
        _mark(ax1)
        if ri == 0:
            ax1.set_title("traffic", fontsize=8)

        recs = by_level[lvl]
        for ci in range(n_var):
            ax = axes[ri, 2 + ci]
            if ci >= len(recs):
                ax.axis("off"); continue
            _setup(ax)
            r = recs[ci]; new = np.array(r.maze_map)
            _bg(ax, new)
            for (bi, bj) in r.broken_cells:
                t = tm.cell_visits.get((bi, bj), 0)
                bx, by_ = ij_to_xy((bi, bj))
                ax.scatter([bx], [by_], marker="s", s=120, c=[[1, 0.2, 0.2]],
                           alpha=0.4 + 0.6 * t / max(model.max_cell_traffic, 1),
                           edgecolors="darkred", zorder=3)
            opened = list(zip(*np.where((new == 0) & (base == 1))))
            if opened:
                oxs = [ij_to_xy((oi, oj))[0] for oi, oj in opened]
                oys = [ij_to_xy((oi, oj))[1] for oi, oj in opened]
                ax.scatter(oxs, oys, marker="+", s=60, c="dodgerblue", zorder=3)
            _paths(ax, orig_routes, "royalblue", lw=1.2, base_alpha=0.35)
            _paths(ax, diverse_routes(new, start, goal, k=k), "darkorange", lw=1.8)
            _mark(ax)
            tag = "reachable" if r.goal_reachable else "UNREACHABLE"
            ax.set_title(f"L{lvl:.1f} v{r.variant_index} | RDI {r.rdi:.2f}\n"
                         f"broke {r.n_broken}/{r.n_routes}  det {r.mean_detour_severity:.2f}  "
                         f"nov {r.novelty_term:.2f} (+{r.n_opened}w) | {tag}", fontsize=8)

    fig.suptitle(f"{METRIC_NAME}: pointmaze-{maze_type} {task}\n"
                 f"blue=original routes  orange=new routes  red sq=broken cell  "
                 f"blue +=opened wall  * start  X goal", fontsize=12, y=1.002)
    plt.tight_layout()
    plt.savefig(outpath, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return outpath


if __name__ == "__main__":
    maze_type = "giant"
    # task = "task1"
    for task in ["task1", "task2", "task3", "task4", "task5"]:
        print(f"\n=== {maze_type} {task} ===")
        seed = 1
        set_seed(seed)

        # Point this at your OGBench dataset dir; if missing, a uniform stub is used.
        DATA = f"../pointmaze/ogbench/data/pointmaze-{maze_type}-navigate-v0"

        recs, model = generate_suite(maze_type=maze_type, task=task, data_dir=DATA,
                                    levels=[0.2, 0.35, 0.5, 0.65], variants_per_level=3,
                                    redesign=True, k=5)

        print(f"\nData model: {'OGBench' if model.has_data else 'uniform stub'}, "
            f"{len(model.traffic.visited_cells)} cells")
        print(f"\n{'level':>6} {'var':>4} {'RDI':>6} {'broke':>7} {'detour':>7} "
            f"{'traf':>6} reachable")
        for r in recs:
            print(f"{r.target_level:>6.1f} {r.variant_index:>4} {r.rdi:>6.2f} "
                f"{r.n_broken}/{r.n_routes:<5} {r.mean_detour_severity:>7.2f} "
                f"{r.traffic_weighted_break:>6.2f} {r.goal_reachable}")

        os.makedirs("maze_variants", exist_ok=True)
        outpng = f"maze_variants/{maze_type}_{task}_rdi.png"
        plot_suite(recs, model, outpng, maze_type, task)
        print(f"\nSaved plot to {outpng}")

        outjson = f"maze_variants/{maze_type}_{task}_rdi.json"
        with open(outjson, "w") as f:
            json.dump({"maze_type": maze_type, "task": task, "seed": seed,
                    "metric": METRIC_NAME, "rdi_weights": RDI_WEIGHTS,
                    "base_maze": model.base_grid.tolist(),
                    "variants": [asdict(r) for r in recs]}, f, indent=2)
        print(f"Saved data to {outjson}")
