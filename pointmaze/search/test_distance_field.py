"""
Geometry tests for the corrected distance-to-goal field (distance_field_v2).
Run:  python -m search.test_distance_field   (from pointmaze/, needs GPU deps)
   or standalone:  python search/test_distance_field.py   (numpy only)
"""
import numpy as np
import importlib.util, os

def _load(fn, nm):
    spec = importlib.util.spec_from_file_location(nm, os.path.join(os.path.dirname(os.path.abspath(__file__)), fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

df2 = _load('distance_field_v2.py', '_df2t')
df1 = _load('distance_field.py', '_df1t')


def _barrier_violations(field, g):
    H, W = g.shape; viol = 0; checks = 0
    for i in range(H):
        for j in range(W):
            if g[i, j] != 0:
                continue
            for di, dj in ((-1,0),(1,0),(0,-1),(0,1)):
                ni, nj = i+di, j+dj
                if 0 <= ni < H and 0 <= nj < W and g[ni, nj] == 1:
                    checks += 1
                    if field[ni, nj] < field[i, j]:
                        viol += 1
    return viol, checks


def test_no_cross_wall_shortcut():
    """A wall touching the goal must NOT inherit the goal's low value: stepping
    into any wall from an adjacent free cell must be strictly uphill."""
    g = np.array([[0,0,0,0,0],
                  [0,1,1,1,0],
                  [0,1,0,0,0],
                  [0,0,0,0,0]], dtype=int)
    F = df2.build_field(g, (2, 2), iters=12, alpha=0.5, wall_penalty=2.0)
    v, c = _barrier_violations(F, g)
    assert v == 0, f"{v}/{c} downhill-into-wall steps (false shortcuts)"
    # the free cell above the wall (0,2) must be uphill into the wall below it
    assert F[1, 2] > F[0, 2], "wall below goal-adjacent free cell is a shortcut"
    return f"no-shortcut OK (0/{c} violations)"


def test_field_fidelity():
    """Masked smoothing must track the true BFS geodesic far better than the
    Gaussian blur (which leaks the inf-fill into free cells)."""
    g = np.array([[0,0,0,0,0,0],
                  [0,1,1,1,1,0],
                  [0,1,0,0,1,0],
                  [0,1,0,1,1,0],
                  [0,0,0,0,0,0]], dtype=int)
    goal = (2, 2)
    Draw = df2.compute_distance_field(g, goal)
    Fo = df1.smooth_distance_field(Draw, sigma=0.5, inf_replace=1000.0)
    Fn = df2.build_field(g, goal, iters=8, alpha=0.5, wall_penalty=2.0)
    reach = np.isfinite(Draw) & (g == 0)
    def rms(F):
        d = Draw[reach]; f = F[reach].astype(float)
        dn = (d-d.min())/(d.max()-d.min()+1e-9); fn = (f-f.min())/(f.max()-f.min()+1e-9)
        return float(np.sqrt(np.mean((dn-fn)**2)))
    ro, rn = rms(Fo), rms(Fn)
    assert rn < ro, f"new fidelity {rn:.3f} not better than old {ro:.3f}"
    return f"fidelity OK (OLD RMS={ro:.3f} -> NEW RMS={rn:.3f})"


def test_deadend_detection():
    """A tendril off a corridor is flagged; the corridor/loop is not."""
    #   col: 0 1 2 3 4
    g = np.array([[0,0,0,0,0],   # main corridor (row 0)
                  [1,1,0,1,1],   # (1,2) branches DOWN off the corridor
                  [1,1,0,1,1]], dtype=int)  # (2,2) is the dead tip
    dmap = df2.deadend_map(g, start=(0,0), goal=(0,4))
    assert dmap[2,2], "dead tip not flagged"
    assert dmap[1,2], "branch stem not flagged as dead-end"
    # main corridor (through-path) not flagged
    assert not dmap[0,2], "through-corridor cell wrongly flagged"
    return f"deadend OK ({int(dmap.sum())} pocket cells)"


def test_stalled_progress():
    good = np.array([10,8,6,4,2,0.])
    trap = np.array([10,8,10,12,10,8.])
    sg = df2.stalled_progress(good); st = df2.stalled_progress(trap)
    assert sg == 0.0, f"monotone approach should score 0, got {sg}"
    assert st > 0.3, f"reversing rollout should score high, got {st}"
    return f"stalled_progress OK (approach={sg:.2f}, trap={st:.2f})"


def test_four_connected_only():
    """Field must be 4-connected: a goal reachable only via a diagonal is
    unreachable (inf), never distance 1."""
    g = np.array([[0,1],
                  [1,0]], dtype=int)   # (0,0) and (1,1) touch only diagonally
    D = df2.compute_distance_field(g, (1,1))
    assert np.isinf(D[0,0]), "diagonal treated as traversable (should be inf)"
    return "4-connectivity OK (diagonal is not a move)"


if __name__ == '__main__':
    tests = [test_no_cross_wall_shortcut, test_field_fidelity,
             test_deadend_detection, test_stalled_progress, test_four_connected_only]
    ok = 0
    for t in tests:
        try:
            print(f"PASS  {t.__name__}: {t()}"); ok += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{ok}/{len(tests)} tests passed")
