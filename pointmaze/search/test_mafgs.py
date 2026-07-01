"""
test_mafgs.py — standalone tests for Map-Aware Feasibility-Guided Search.

MAFGS imports diffuser/diffusers (GPU-only) transitively, so we mock those two
symbols (randn_tensor, apply_conditioning) and load the torch-only pieces by
path. Everything exercised here — the map-aware composite, the feasibility gate,
and the guide_step backtracking control flow — is pure torch and runs on CPU.

Run:  python search/test_mafgs.py     (from the pointmaze dir, or anywhere;
      paths are resolved relative to this file)
"""
import os, sys, types, json, importlib.util
from collections import deque
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))            # .../Diffusion-inference-scaling
GIANT = os.path.join(REPO, 'maze_update', 'maze_variants_v2', 'giant_task1.json')
U = OX = OY = 4.0


# --------------------------------------------------------------------------- #
# mock GPU-only deps and load mafgs + real verifiers by path
# --------------------------------------------------------------------------- #
def _mock(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _load(dotted, path):
    spec = importlib.util.spec_from_file_location(dotted, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = m
    spec.loader.exec_module(m)
    return m


def _apply_cond(x, cond, adim):
    for t, val in cond.items():
        x[:, t, adim:] = torch.as_tensor(val, dtype=x.dtype)
    return x


def _setup():
    _mock('diffusers'); _mock('diffusers.utils')
    _mock('diffusers.utils.torch_utils',
          randn_tensor=lambda shape, generator=None, device=None, dtype=None: torch.randn(shape))
    _mock('diffuser'); _mock('diffuser.models')
    _mock('diffuser.models.helpers', apply_conditioning=_apply_cond)

    pkg = types.ModuleType('search'); pkg.__path__ = [HERE]; sys.modules['search'] = pkg
    meth = types.ModuleType('search.methods'); meth.__path__ = [os.path.join(HERE, 'methods')]
    sys.modules['search.methods'] = meth

    cfg = _load('search.configs', os.path.join(HERE, 'configs.py'))
    _load('search.distance_field_v2', os.path.join(HERE, 'distance_field_v2.py'))
    _load('search.distance_field', os.path.join(HERE, 'distance_field.py'))
    _load('search.maze_verifier_clearance', os.path.join(HERE, 'maze_verifier_clearance.py'))
    _load('search.distance_field_verifier_v2', os.path.join(HERE, 'distance_field_verifier_v2.py'))
    _mock('search.methods.tfg', TFGGuidance=object)

    class _BG:
        def __init__(self, args, **kw):
            self.args = args; self.guider = self._build_verifier(args)
        @staticmethod
        def _build_verifier(args):
            return None
        def update_env(self, env):
            self.guider.update_env(env)
        def _predict_xt(self, *a, **k):
            return a[0]
    _mock('search.methods.base_guidance', BaseGuidance=_BG)

    mafgs = _load('search.methods.mafgs', os.path.join(HERE, 'methods', 'mafgs.py'))
    return cfg.Arguments, mafgs


class _FakeEnv:
    def __init__(self, mm, gij):
        self.maze_map = mm; self._maze_unit = U
        self._offset_x = OX; self._offset_y = OY
        self.cur_task_info = {'goal_xy': (gij[1] * U - OX, gij[0] * U - OY)}


def _ij2xy(i, j):
    return (j * U - OX, i * U - OY)


def _bfs(mm, s, g):
    H, W = mm.shape; prev = {s: None}; q = deque([s])
    while q:
        c = q.popleft()
        if c == g:
            break
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + di, c[1] + dj)
            if 0 <= n[0] < H and 0 <= n[1] < W and mm[n] == 0 and n not in prev:
                prev[n] = c; q.append(n)
    path = []; c = g
    while c is not None:
        path.append(c); c = prev[c]
    return path[::-1]


def _resample(xy, T):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    s = np.linspace(0, d[-1], T)
    return np.c_[np.interp(s, d, xy[:, 0]), np.interp(s, d, xy[:, 1])]


def _make_x(xy2d):
    T = xy2d.shape[0]; x = torch.zeros(1, T, 4)
    x[0, :, 2:4] = torch.tensor(xy2d, dtype=torch.float32)
    return x


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
Arguments, mafgs = _setup()
_base = json.load(open(GIANT))
_mm = np.array(_base['base_maze'], dtype=int)
_start, _goal = tuple(_base['start']), tuple(_base['goal'])
_env = _FakeEnv(_mm, _goal)
_args = Arguments(device='cpu')
_comp = mafgs._MapAwareComposite(_args); _comp.update_env(_env)

_cells = _bfs(_mm, _start, _goal)
_clean = _resample(np.array([_ij2xy(i, j) for i, j in _cells]), 40)
_straight = _resample(np.array([_ij2xy(*_start), _ij2xy(*_goal)]), 40)
_pp = lambda z: z


def _feas(xy):
    inf, d = _comp.feasibility(_make_x(xy), post_process=_pp)
    return bool(inf.item()), d


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_clean_path_feasible():
    inf, d = _feas(_clean)
    assert not inf, f"clean geodesic flagged infeasible: {d}"
    assert d['corner'].item() == 0.0
    assert d['collision'].item() < _comp.collision_tol


def test_straight_tunnel_flags_collision():
    inf, d = _feas(_straight)
    assert inf, "straight wall-tunneling path should be infeasible"
    assert d['collision'].item() > _comp.collision_tol, "collision term must fire"


def test_cornercut_flags_corner():
    pc = _comp.clear.corner_pts.numpy()[len(_comp.clear.corner_pts) // 2]
    cut = _clean.copy(); m = len(cut) // 2
    cut[m - 1] = pc + np.array([-0.6, -0.6]); cut[m] = pc; cut[m + 1] = pc + np.array([0.6, 0.6])
    inf, d = _feas(cut)
    assert inf, "corner-cut path should be infeasible"
    assert d['corner'].item() > 0.0, "diagonal-pinch term must fire"


def test_reversal_flags_stall():
    rev_cells = _cells[:len(_cells) // 2] + _cells[len(_cells) // 2::-1]
    rev = _resample(np.array([_ij2xy(i, j) for i, j in rev_cells]), 40)
    inf, d = _feas(rev)
    assert inf, "goal-reversing path should be infeasible"
    assert d['stall'].item() > _comp.stall_tol, "stalled-progress term must fire"


def _make_guidance(budget=20):
    a = Arguments(device='cpu'); a.inference_steps = 16; a.budget = budget
    a.recur_depth = 12; a.start_step = 12; a.step_size = 1
    a.threshold = 6; a.threshold_schedule = 'increase'
    g = mafgs.MAFGSGuidance.__new__(mafgs.MAFGSGuidance); g.args = a
    g.guider = mafgs._MapAwareComposite(a); g.guider.update_env(_env)

    class _Local:
        def __init__(s): s.guider = None
        def update_env(s, e): pass
        def guide_step(s, x, i, unet, ts, apt, aptp, eta, **kw):
            return x.clone(), {"x0": x.clone()}
    g.local_search = _Local(); g.local_search.guider = g.guider
    g.reset()
    return g


_TS = torch.arange(16, 0, -1)
_APT = torch.linspace(0.99, 0.01, 16)
_APTP = torch.cat([_APT[1:], torch.tensor([1.0])])


def _step(g, xy, i):
    x = _make_x(xy)
    cond = {0: [x[0, 0, 2].item(), x[0, 0, 3].item()],
            x.shape[1] - 1: [x[0, -1, 2].item(), x[0, -1, 3].item()]}
    _, ex = g.guide_step(x, i, None, _TS, _APT, _APTP, 1.0, cond=cond, post_process=_pp)
    return ex['i_next']


def test_guide_step_accepts_clean():
    g = _make_guidance()
    assert _step(g, _clean, 12) == 13, "clean trajectory should advance"
    assert g.budget == 20, "no budget spent on a feasible candidate"


def test_guide_step_backtracks_on_infeasible():
    g = _make_guidance()
    nxt = _step(g, _straight, 12)
    assert nxt < 12, f"infeasible candidate must backtrack, got i_next={nxt}"
    assert g.budget == 19, "one budget unit spent on the backtrack"


def test_guide_step_noneval_always_accepts():
    g = _make_guidance()
    assert _step(g, _straight, 5) == 6, "non-eval steps do not gate"


def test_guide_step_budget_exhaustion_falls_back():
    g = _make_guidance(budget=0)
    assert _step(g, _straight, 12) == 13, "with no budget, accept buffered best and advance"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}"); passed += 1
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
