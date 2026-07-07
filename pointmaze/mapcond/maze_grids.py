"""
Parse pointmaze occupancy grids straight from the OGBench source
(ogbench/ogbench/locomaze/maze.py) without importing it -- importing would
pull up mujoco. Grids are binary: 0 = open cell, 1 = wall.

All pointmaze maps share the same world<->cell scale (maze_unit=4, offset=4),
so a given cell (i, j) maps to the SAME world (x, y) in every map. That is
what lets one shared normalizer + one map-conditional model span all maps:
the coordinates are common; only which cells are walls differs, and that is
exactly the signal the map encoder must supply.
"""
import os
import ast
import numpy as np

# Resolve the OGBench maze source relative to this file (pointmaze/mapcond/..).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PM = os.path.dirname(_HERE)  # .../pointmaze
MAZE_PY = os.path.join(_PM, "ogbench", "ogbench", "locomaze", "maze.py")

# Coordinate scale used by the OGBench pointmaze envs (see maze.py __init__).
MAZE_UNIT = 4.0
OFFSET_X = 4.0
OFFSET_Y = 4.0

# The pointmaze maps that have offline navigate datasets on disk.
KNOWN_MAZES = ("medium", "large", "giant", "teleport", "ultra")


def parse_grid(maze_type, src=None):
    """Return the HxW int occupancy grid for `maze_type` (0 open, 1 wall)."""
    if src is None:
        src = open(MAZE_PY).read()
    idx = src.find(f"maze_type == '{maze_type}'")
    if idx < 0:
        raise KeyError(f"maze_type {maze_type!r} not found in {MAZE_PY}")
    mm = src.find("maze_map = [", idx)
    if mm < 0:
        raise KeyError(f"no maze_map block after maze_type == {maze_type!r}")
    start = src.find("[", mm)
    depth = 0
    for k in range(start, len(src)):
        if src[k] == "[":
            depth += 1
        elif src[k] == "]":
            depth -= 1
            if depth == 0:
                grid = ast.literal_eval(src[start:k + 1])
                return np.asarray(grid, dtype=np.int64)
    raise ValueError(f"unbalanced brackets parsing maze_map for {maze_type!r}")


def all_grids(maze_types=KNOWN_MAZES):
    """dict maze_type -> HxW occupancy grid, parsed once from the shared source."""
    src = open(MAZE_PY).read()
    return {m: parse_grid(m, src=src) for m in maze_types}


def cell_to_xy(i, j):
    """Cell (row i, col j) -> world (x, y), matching maze.py ij_to_xy."""
    x = j * MAZE_UNIT - OFFSET_X
    y = i * MAZE_UNIT - OFFSET_Y
    return np.array([x, y], dtype=np.float64)


def grid_world_extent(grid):
    """World-coordinate bounding box (xmin, xmax, ymin, ymax) of a grid's cells."""
    H, W = grid.shape
    xmin, ymin = cell_to_xy(0, 0)
    xmax, ymax = cell_to_xy(H - 1, W - 1)
    return float(xmin), float(xmax), float(ymin), float(ymax)


def pad_to_canvas(grid, canvas_hw, pad_value=1, anchor="corner"):
    """
    Pad a grid into a fixed (H, W) canvas, filling the border with `pad_value`
    (default 1 = wall). A wall-filled border is the physically correct padding:
    cells outside a smaller maze are unreachable, i.e. walls.

    anchor="corner" (default) puts cell (0, 0) at canvas (0, 0). Because every
    pointmaze map shares the world origin at cell (0, 0), corner anchoring keeps
    a given world cell at a FIXED canvas pixel across all maps -- so the map
    encoder sees the same maze structure at the same location it occupies in the
    world. anchor="center" center-pads instead.

    Returns (padded_grid, (row_off, col_off)) so callers can map cells back.
    """
    H, W = grid.shape
    CH, CW = canvas_hw
    if H > CH or W > CW:
        raise ValueError(f"grid {grid.shape} exceeds canvas {canvas_hw}")
    out = np.full((CH, CW), pad_value, dtype=grid.dtype)
    if anchor == "corner":
        r0, c0 = 0, 0
    elif anchor == "center":
        r0 = (CH - H) // 2
        c0 = (CW - W) // 2
    else:
        raise ValueError(f"anchor must be 'corner' or 'center', got {anchor!r}")
    out[r0:r0 + H, c0:c0 + W] = grid
    return out, (r0, c0)


def canvas_size(grids):
    """Smallest (H, W) canvas that fits every grid in `grids` (dict or list)."""
    if isinstance(grids, dict):
        grids = list(grids.values())
    H = max(g.shape[0] for g in grids)
    W = max(g.shape[1] for g in grids)
    return (H, W)
