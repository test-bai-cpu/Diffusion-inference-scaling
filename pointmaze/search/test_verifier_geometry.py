"""
test_verifier_geometry.py
=========================
Geometry unit tests for the corner-safe verifier. No MuJoCo / diffusion needed:
a tiny mock env supplies a maze_map and the ij<->xy constants, exactly the
attributes get_wall_tensors reads.

Asserts:
  1. The OLD verifier gives ~0 collision loss at the diagonal pinch (the bug).
  2. The NEW verifier gives a strictly positive loss there (bug fixed).
  3. Centerline travel down a legal 4-wide corridor stays ~0 (no false positive).
  4. A straight segment that jumps across a wall is caught by segment sampling
     even when both endpoints are in free space.
  5. Deep inside a wall > grazing a wall face > clear (monotone).
"""
import os
import sys
import types
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maze_verifier_clearance as NEW

# The original maze_verifier.py imports search.utils (-> diffuser -> einops) at
# module level, which needs the full GPU stack. Its GEOMETRY functions, however,
# are defined ABOVE that import. Load just those into an isolated module so the
# "bug still present" assertion runs offline in the sandbox.
def _load_old_geometry():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maze_verifier.py")
    src = open(path).read()
    cut = src.index("from search.utils import")   # everything before the heavy import
    mod = types.ModuleType("maze_verifier_geom")
    exec(compile(src[:cut], path, "exec"), mod.__dict__)
    return mod

OLD = _load_old_geometry()


class MockEnv:
    """Minimal stand-in for the OGBench maze env used by wall extraction."""
    def __init__(self, maze_map, unit=4.0, offset=4.0, maze_type="giant"):
        self.maze_map = np.array(maze_map, dtype=int)
        self._maze_unit = unit
        self._offset_x = offset
        self._offset_y = offset
        self._maze_type = maze_type

    def ij_to_xy(self, ij):
        i, j = ij
        return (j * self._maze_unit - self._offset_x, i * self._maze_unit - self._offset_y)


def diagonal_gap_env():
    """
    5x5 map with two walls touching only at a corner, leaving a diagonal pinch.
    Walls at (2,2) and (3,3); orthogonal cells (2,3) and (3,2) are FREE.
    Shared corner world-xy = ((2+3)/2*4-4, ...) = (6, 6).
    """
    mm = np.zeros((5, 5), dtype=int)
    mm[0, :] = mm[-1, :] = mm[:, 0] = mm[:, -1] = 1  # border
    mm[2, 2] = 1
    mm[3, 3] = 1
    return MockEnv(mm)


def corridor_env():
    """1-cell-wide free corridor down column j=2, walls either side (j=1,3)."""
    mm = np.ones((7, 5), dtype=int)
    mm[1:6, 2] = 0        # vertical free corridor
    return MockEnv(mm)


def _old_loss_at(env, pts, device="cpu"):
    wb, adj = OLD.get_wall_boxes_and_adjacency(env, torch.device(device))
    pos = torch.tensor(pts, dtype=torch.float32).view(1, -1, 2)
    return OLD.batched_inside_wall_loss_non_adjacent(pos, wb, adj).view(-1)


def _new_loss_at(env, pts, **kw):
    v = NEW.CornerSafeMazeVerifier(margin=kw.pop("margin", 0.7),
                                   ball_radius=kw.pop("ball_radius", 0.5),
                                   n_sub=kw.pop("n_sub", 4),
                                   corner_radius=kw.pop("corner_radius", 0.9))
    v.update_env(env)
    pos = torch.tensor(pts, dtype=torch.float32).view(1, -1, 2)
    return v.collision_loss(pos).view(-1)


def test_diagonal_gap():
    env = diagonal_gap_env()
    # shared corner between wall (2,2)@xy(4,4) and wall (3,3)@xy(8,8) is (6,6)
    c00 = env.ij_to_xy((2, 2)); c11 = env.ij_to_xy((3, 3))
    gap = ((c00[0] + c11[0]) / 2, (c00[1] + c11[1]) / 2)
    assert gap == (6.0, 6.0), gap
    old = _old_loss_at(env, [gap]).item()
    new = _new_loss_at(env, [gap]).item()
    print(f"[diagonal gap {gap}] OLD loss = {old:.4f}   NEW loss = {new:.4f}")
    assert old < 1e-6, f"expected OLD~0 (the bug), got {old}"
    assert new > 0.1, f"expected NEW>0 (fixed), got {new}"


def test_corridor_free():
    env = corridor_env()
    # centerline of corridor: column j=2 -> x = 2*4-4 = 4; walk rows 2..4 -> y=4,8,12
    pts = [env.ij_to_xy((i, 2)) for i in (2, 3, 4)]
    new = _new_loss_at(env, pts)
    print(f"[corridor centerline] NEW loss per pt = {new.tolist()}")
    assert torch.all(new < 1e-4), f"legal corridor should be free, got {new.tolist()}"


def test_segment_tunnel():
    env = corridor_env()
    # jump straight across the corridor through the wall at (3,1)&(3,3):
    # from (3,2) free to a point on the far side, crossing a wall cell.
    p_free_a = env.ij_to_xy((3, 2))        # in corridor
    p_free_b = (env.ij_to_xy((3, 2))[0] + 8.0, env.ij_to_xy((3, 2))[1])  # 2 cells right, through wall
    # endpoints: a is free; b is inside/behind a wall. Test that the SEGMENT
    # midpoint (inside wall col j=3) is penalized via supersampling.
    seg = _new_loss_at(env, [p_free_a, p_free_b], n_sub=6)
    # point-only (n_sub=1) vs segment (n_sub=6): segment must be >= point loss
    pt = _new_loss_at(env, [p_free_a, p_free_b], n_sub=1)
    print(f"[segment tunnel] point-only={pt.tolist()}  with-segment={seg.tolist()}")
    assert seg[0].item() > pt[0].item() + 1e-6, "segment sampling must catch the wall crossing"


def test_monotone_depth():
    # A single interior wall in a large open field so the test points only ever
    # feel THAT wall (no nearby borders contaminating the "clear" reading).
    mm = np.zeros((11, 11), dtype=int)
    mm[5, 5] = 1
    env = MockEnv(mm)
    wall_c = env.ij_to_xy((5, 5))                 # wall center; half = 2 + 0.5 = 2.5
    deep = _new_loss_at(env, [wall_c]).item()                          # inside wall
    graze = _new_loss_at(env, [(wall_c[0] + 2.7, wall_c[1])]).item()   # just outside face
    clear = _new_loss_at(env, [(wall_c[0] + 8.0, wall_c[1])]).item()   # well clear
    print(f"[monotone] deep={deep:.3f} > graze={graze:.3f} > clear={clear:.3f}")
    assert deep > graze > clear, (deep, graze, clear)
    assert clear < 1e-4


if __name__ == "__main__":
    test_diagonal_gap()
    test_corridor_free()
    test_segment_tunnel()
    test_monotone_depth()
    print("\nALL GEOMETRY TESTS PASSED")
