"""
Per-instance test of the "generator door bias" hypothesis (see
SDFS_DOOR_BIAS_NOTES.md): does a plan's wall-tunnelling concentrate on the
SAME broken door across many independent episodes, and is that door actually
a shortcut worth wanting -- or is the model just replaying a fixed route
regardless of which door this particular OOD variant happened to close?

Uses the saved plan.npz files under
logs/pointmaze-giant-newvar-navigate-v0_sdfs-global-w2-omega05/inference/plans/...
(the completed sdfs 6-cell sweep) and each variant's own `broken_cells` field
in ../maze_update/maze_variants/giant_task1.json -- no re-run needed.

Run from pointmaze/:  python sdfs_door_bias_analysis.py
"""
import glob
import json
from collections import Counter, deque

import numpy as np

MAZE_UNIT, OFFSET_X, OFFSET_Y = 4.0, 4.0, 4.0   # mapcond/maze_grids.py constants

CELLS = [
    ('L3V1', 6, 'level3', '1'), ('L3V2', 7, 'level3', '2'), ('L3V3', 8, 'level3', '3'),
    ('L4V1', 9, 'level4', '1'), ('L4V2', 10, 'level4', '2'), ('L4V3', 11, 'level4', '3'),
]
PLAN_ROOT = ('logs/pointmaze-giant-newvar-navigate-v0_sdfs-global-w2-omega05/'
             'inference/plans/release_H400_T256_LimitsNormalizer_b1_condFalse')
VARIANT_JSON = '../maze_update/maze_variants/giant_task1.json'
# The checkpoint's OWN saved training config (state_500000.pt's ["config"]
# ["variant_json_dir"]) points here -- a DIFFERENT variant family from the
# one used at eval time above. This is what makes the check below "did
# training ever show this door closed" meaningful rather than circular.
TRAIN_VARIANT_JSON = '../maze_update/maze_variants_seed2/giant_task1.json'


def xy_to_ij(x, y):
    """Matches search/distance_field_verifier.py's DistanceFieldVerifier._xy_to_ij."""
    i = int(np.floor((y + OFFSET_Y + 0.5 * MAZE_UNIT) / MAZE_UNIT))
    j = int(np.floor((x + OFFSET_X + 0.5 * MAZE_UNIT) / MAZE_UNIT))
    return i, j


def bfs_dist(grid, start, goal):
    """Shortest free-space path length in cells, or None if unreachable."""
    H, W = grid.shape
    if grid[start] == 1 or grid[goal] == 1:
        return None
    dist = -np.ones((H, W), dtype=int)
    dist[start] = 0
    q = deque([start])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni, nj] == 0 and dist[ni, nj] < 0:
                dist[ni, nj] = dist[i, j] + 1
                q.append((ni, nj))
    d = dist[goal]
    return int(d) if d >= 0 else None


def door_hits_for_cell(vidx, level, variant, vjson):
    """(door_hit_counter, n_episodes_touching_any_door, n_episodes_total)."""
    v = vjson['variants'][vidx]
    grid = np.array(v['maze_map'])
    broken = [tuple(c) for c in v['broken_cells']]
    plan_files = sorted(glob.glob(
        f'{PLAN_ROOT}/sdfs-task1-{level}-variant{variant}/*/plan.npz'))

    door_hits = Counter()
    n_touch_door = 0
    for pf in plan_files:
        plan = np.load(pf)['plan'][0]
        ijs = {xy_to_ij(x, y) for x, y in plan}
        touched = ijs & set(broken)
        if touched:
            n_touch_door += 1
        for c in touched:
            door_hits[c] += 1
    return door_hits, n_touch_door, len(plan_files), broken, grid, v


def train_open_rate(door, train_vjson):
    """(n_open, n_total) for `door` across the base training map + all its
    variants -- n_total = 1 (base, always open by construction) + len(variants)."""
    n_variants = len(train_vjson['variants'])
    n_broken = sum(1 for v in train_vjson['variants']
                   if door in {tuple(c) for c in v['broken_cells']})
    return (1 + n_variants - n_broken), (1 + n_variants)


def main():
    vjson = json.load(open(VARIANT_JSON))
    train_vjson = json.load(open(TRAIN_VARIANT_JSON))
    print(f"{'cell':6s} {'doors':6s} {'door-ep':8s} {'top door':12s} {'top share':14s} "
          f"{'penalty':8s} {'train open rate':16s}")
    for name, vidx, level, variant in CELLS:
        door_hits, n_touch, n_ep, broken, grid, v = door_hits_for_cell(
            vidx, level, variant, vjson)
        if not door_hits:
            print(f"{name:6s} {len(broken):<6d} {n_touch}/{n_ep:<5d} "
                  f"{'(none hit)':12s}")
            continue
        top_door, top_n = door_hits.most_common(1)[0]
        top_share = top_n / n_touch if n_touch else 0.0

        start_ij = xy_to_ij(*v['start_xy'])
        goal_ij = xy_to_ij(*v['goal_xy'])
        d_closed = bfs_dist(grid, start_ij, goal_ij)
        grid_open = grid.copy(); grid_open[top_door] = 0
        d_open = bfs_dist(grid_open, start_ij, goal_ij)
        penalty = (d_closed - d_open) if (d_closed is not None and d_open is not None) else None

        n_open, n_total = train_open_rate(top_door, train_vjson)

        print(f"{name:6s} {len(broken):<6d} {n_touch}/{n_ep:<5d} "
              f"{str(top_door):12s} {top_n}/{n_touch} ({top_share:.0%}){'':1s} "
              f"{str(penalty):8s} {n_open}/{n_total} training maps open")


if __name__ == '__main__':
    main()
