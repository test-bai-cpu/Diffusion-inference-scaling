"""
route_space.py
==============

Structurally-distinct start->goal route enumeration for maze grids.

Ported (dependency-free) from the k-shortest / waypoint / diverse-route
functions that used to live in ``reroute_disruption.py``. The only external
dependency is numpy plus ``_neighbors``/``shortest_path`` from ``maze_utils``.

The point of this module: a maze variant is not fully described by its single
optimal (geodesic) path. Two variants can share the same shortest path yet
differ in the *set* of near-optimal / structurally-distinct alternatives a
planner might follow. ``diverse_routes`` returns up to ``k`` such routes, and
``route_signature`` turns that set into a hashable, order-independent key so the
candidate pool can be de-duplicated (and the final ladder made distinct) by the
whole route SET rather than by one geodesic.
"""
from __future__ import annotations

import heapq
from collections import defaultdict

import numpy as np

from maze_utils import _neighbors, shortest_path


def k_shortest_paths(grid, start, goal, k=5):
    """Up to k shortest start->goal paths via BFS with bounded node revisits.

    Each node may appear on up to k paths so routes can diverge instead of all
    collapsing onto the single geodesic.
    """
    grid = np.asarray(grid)
    start = tuple(start); goal = tuple(goal)
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
    return p1 + p2[1:]


def _waypoint_routes(grid, start, goal, region_div=4):
    """Force routes through different REGIONS of the maze so structurally
    distinct alternatives (e.g. the one that swings through a far corner) are
    guaranteed to appear, regardless of how much longer they are.

    The maze is split into a region_div x region_div grid of blocks; for each
    block the most central free cell becomes a waypoint and we build the
    shortest start->waypoint->goal route. Returned sorted by length.
    """
    grid = np.asarray(grid)
    start = tuple(start); goal = tuple(goal)
    H, W = grid.shape
    routes = []
    seen_keys = set()
    for bi in range(region_div):
        for bj in range(region_div):
            r0, r1 = bi * H // region_div, (bi + 1) * H // region_div
            c0, c1 = bj * W // region_div, (bj + 1) * W // region_div
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
    longer alternatives, not just near-optimal ones.

    Sources, combined then de-overlapped:
      1. k-shortest pool  -> near-optimal routes (length ~optimal..+2).
      2. waypoint routes  -> routes forced through every region of the maze.

    A candidate is kept only if it overlaps every already-kept route by at most
    ``overlap_tol`` (fraction of the smaller route's cells). The optimal route is
    seeded first so near-optimal structure is preferred, then longer distinct
    routes fill remaining slots.
    """
    grid = np.asarray(grid)
    start = tuple(start); goal = tuple(goal)
    if grid[start] == 1 or grid[goal] == 1:
        return []
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


def route_signature(grid, start, goal, k=3, overlap_tol=0.75):
    """Order-independent hashable signature of a variant's *route set*.

    Instead of keying a candidate by its single optimal path, we key it by the
    (up to) k structurally-distinct routes returned by ``diverse_routes``. Each
    route is reduced to a frozenset of cells; the signature is the frozenset of
    those per-route frozensets, so two candidates collide only when their whole
    diverse-route SET matches (order-independent).
    """
    routes = diverse_routes(grid, start, goal, k=k, overlap_tol=overlap_tol)
    return frozenset(frozenset(map(tuple, r)) for r in routes)
