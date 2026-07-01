"""
Tests for episode_metrics: each metric must isolate ONE failure mode on the real
giant maze. Standalone (numpy only): python search/test_episode_metrics.py
"""
import numpy as np, importlib.util, os, json

def _load(fn, nm):
    spec = importlib.util.spec_from_file_location(nm, os.path.join(os.path.dirname(os.path.abspath(__file__)), fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

em  = _load('episode_metrics.py', '_emt')
df2 = _load('distance_field_v2.py', '_df2e')

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = json.load(open(os.path.join(_HERE, '..', '..', 'maze_update',
                                    'maze_variants_v2', 'giant_task1.json')))
GRID = np.array(_BASE['base_maze'])
U, OX, OY = 4.0, 4.0, 4.0
START = tuple(_BASE['start']); GOAL = tuple(_BASE['goal'])


def _cell_xy(i, j): return (j*U - OX, i*U - OY)

def _optimal_path():
    D = df2.compute_distance_field(GRID, GOAL); cur = START; path = [cur]
    for _ in range(500):
        if cur == GOAL: break
        i, j = cur; bv = D[i, j]; best = None
        for di, dj in ((-1,0),(1,0),(0,-1),(0,1)):
            ni, nj = i+di, j+dj
            if 0 <= ni < GRID.shape[0] and 0 <= nj < GRID.shape[1] and GRID[ni,nj]==0 and D[ni,nj] < bv:
                bv = D[ni,nj]; best = (ni,nj)
        if best is None: break
        cur = best; path.append(cur)
    return path


def test_clean_path_scores_zero():
    path = _optimal_path()
    xy = np.array([_cell_xy(*c) for c in path])
    m = em.score_episode(xy, GRID, GOAL, U, OX, OY, success=(path[-1]==GOAL), start_ij=START)
    assert m['success'] == 1.0
    assert m['collision_rate'] == 0.0, m['collision_rate']
    assert m['cornercut_rate'] == 0.0
    assert m['deadend_frac'] == 0.0
    assert m['stalled_prog'] == 0.0
    assert m['final_gap'] == 0.0
    return "clean optimal path: all failure metrics 0, success 1"


def test_wall_hug_flags_collision():
    path = _optimal_path()
    xy = np.array([_cell_xy(*c) for c in path])
    centers, _ = em.wall_boxes(GRID, U, OX, OY)
    hug = xy.copy()
    for k, p in enumerate(xy):
        c = centers[np.linalg.norm(centers - p, axis=1).argmin()]
        v = c - p; hug[k] = p + v/(np.linalg.norm(v)+1e-9)*1.6
    m = em.score_episode(hug, GRID, GOAL, U, OX, OY, success=True, start_ij=START)
    assert m['collision_rate'] > 0.5, m['collision_rate']
    return f"wall-hug flagged: collision_rate={m['collision_rate']:.2f}"


def test_pinch_flags_cornercut():
    pinch = em.diagonal_pinch_points(GRID, U, OX, OY)
    assert len(pinch) > 0, "no diagonal pinches in base giant maze?"
    p = pinch[0]
    thru = np.array([p+[-2,-2], p+[-1,-1], p, p+[1,1], p+[2,2]])
    m = em.score_episode(thru, GRID, GOAL, U, OX, OY, success=False, start_ij=START)
    assert m['cornercut_rate'] > 0.0, m['cornercut_rate']
    return f"pinch traversal flagged: cornercut_rate={m['cornercut_rate']:.2f} ({len(pinch)} pinches)"


def test_deadend_wander_flags_deadend_and_stall():
    path = _optimal_path()
    dmap = df2.deadend_map(GRID, start=START, goal=GOAL)
    dead = list(zip(*np.where(dmap)))
    assert dead, "no dead-end cells?"
    half = path[:len(path)//2]
    wander = half + [dead[0]]*8 + half[::-1]
    xy = np.array([_cell_xy(*c) for c in wander])
    m = em.score_episode(xy, GRID, GOAL, U, OX, OY, success=False, start_ij=START)
    assert m['deadend_frac'] > 0.1, m['deadend_frac']
    assert m['stalled_prog'] > 0.1, m['stalled_prog']
    assert m['success'] == 0.0
    return f"dead-end wander flagged: dead={m['deadend_frac']:.2f} stall={m['stalled_prog']:.2f}"


if __name__ == '__main__':
    tests = [test_clean_path_scores_zero, test_wall_hug_flags_collision,
             test_pinch_flags_cornercut, test_deadend_wander_flags_deadend_and_stall]
    ok = 0
    for t in tests:
        try:
            print(f"PASS  {t.__name__}: {t()}"); ok += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{ok}/{len(tests)} tests passed")
