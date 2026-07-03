"""
maze_utils.py
=============

Maze map definitions, task definitions, coordinate utilities, and BFS
pathfinding. Used as a data/utility module by generate_data_ood.py.
"""

from __future__ import annotations
from collections import deque

import numpy as np


# ===========================================================================
# MAZE MAPS  (parsed live from ogbench/locomaze/maze.py -- the single source of
# truth -- instead of a hand-copied duplicate that could silently drift)
# ===========================================================================
import ast as _ast
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_REPO = _os.path.dirname(_HERE)
MAZE_PY = _os.path.join(_REPO, "pointmaze", "ogbench", "ogbench",
                        "locomaze", "maze.py")


def parse_maze_maps(maze_py_path=MAZE_PY):
    """Return {maze_type: [[...]]} for every maze type defined in ogbench
    maze.py. The literals are locals inside the env __init__ if/elif chain
    (`if self._maze_type == '<t>': ... maze_map = [[...]]`), so they can't be
    imported; we walk the AST and lift each maze_map assignment out of its
    matching branch. This keeps every downstream map locked to the upstream
    definition rather than a cached copy.
    """
    if not _os.path.exists(maze_py_path):
        raise FileNotFoundError(
            f"ogbench maze.py not found at {maze_py_path}; cannot read the "
            f"authoritative maze definitions.")
    tree = _ast.parse(open(maze_py_path).read())
    maps = {}
    for node in _ast.walk(tree):
        if isinstance(node, _ast.If) and isinstance(node.test, _ast.Compare):
            types = [c.value for c in node.test.comparators
                     if isinstance(c, _ast.Constant) and isinstance(c.value, str)]
            if not types:
                continue
            for stmt in node.body:
                if isinstance(stmt, _ast.Assign) and any(
                        getattr(t, "id", None) == "maze_map" for t in stmt.targets):
                    try:
                        maps[types[0]] = _ast.literal_eval(stmt.value)
                    except Exception:
                        pass
    if not maps:
        raise RuntimeError(f"no maze_map literals found in {maze_py_path}")
    return maps


MAZE_MAPS = parse_maze_maps()

# Back-compat named constants (same values, now sourced from maze.py).
MEDIUM_MAZE = MAZE_MAPS.get("medium")
LARGE_MAZE = MAZE_MAPS.get("large")
GIANT_MAZE = MAZE_MAPS.get("giant")
ULTRA_MAZE = MAZE_MAPS.get("ultra")

MAZE_TASKS = {
    "medium": {
        "task1": ((1, 1), (6, 6)), "task2": ((6, 1), (1, 6)),
        "task3": ((5, 3), (4, 2)), "task4": ((6, 5), (6, 1)),
        "task5": ((2, 6), (1, 1)),
    },
    "large": {
        "task1": ((1, 1), (7, 10)), "task2": ((5, 4), (7, 1)),
        "task3": ((7, 4), (1, 10)), "task4": ((3, 8), (5, 4)),
        "task5": ((1, 1), (5, 4)),
    },
    "giant": {
        "task1": ((1, 1),  (10, 14)), "task2": ((1, 14), (10, 1)),
        "task3": ((8, 14), (1, 1)),   "task4": ((8, 3),  (5, 12)),
        "task5": ((5, 9),  (3, 8)),
    },
    "ultra": {
        "task1": ((1, 1),  (19, 19)), "task2": ((1, 17), (17, 19)),
        "task3": ((1, 1),  (19, 1)),  "task4": ((19, 1), (1, 19)),
        "task5": ((1, 1),  (9, 19)),
    },
}


# ===========================================================================
# COORDINATE UTILITIES
# ===========================================================================

MAZE_UNIT = 4.0
OFFSET = 4.0


def ij_to_xy(ij):
    i, j = ij
    return (j * MAZE_UNIT - OFFSET, i * MAZE_UNIT - OFFSET)


# ===========================================================================
# PATHFINDING
# ===========================================================================

def _neighbors(cell, grid: np.ndarray):
    r, c = cell
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] == 0:
            yield (nr, nc)


def shortest_path(grid: np.ndarray, start, goal):
    if grid[start] == 1 or grid[goal] == 1:
        return None
    if start == goal:
        return [start]
    prev = {start: start}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nb in _neighbors(cur, grid):
            if nb not in prev:
                prev[nb] = cur
                q.append(nb)
    if goal not in prev:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]
