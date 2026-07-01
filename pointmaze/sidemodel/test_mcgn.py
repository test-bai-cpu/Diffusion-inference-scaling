"""
test_mcgn.py — CPU tests for the Map-Conditioned Guidance Network (Step 7).
==========================================================================
Runs entirely on CPU with no GPU/diffusion dependencies (the sidemodel path-
loads only distance_field_v2 + torch). Validates:

  1. occupancy_crop  : out-of-bounds padded as wall; center is the query cell.
  2. geodesic_descent_dir : points to a strictly-lower field neighbor (downhill),
                            (0,0) at the goal.
  3. build_samples   : array shapes/dtypes; goal[:, :2] unit; targets unit.
  4. model forward   : (B,2) output; cosine_direction_loss finite & positive.
  5. training        : loss decreases on a single map.
  6. D4 augment      : equivariance (rotated crop -> rotated target) and 8x size.
  7. transfer        : train on tasks {1,2,4,5} variants (D4-aug), holdout task3;
                       MCGN holdout cosine beats random (>0) and the goal-
                       direction baseline on must-detour cells.
  8. MCGNVerifier    : get_guidance returns per-sample logp and a position
                       gradient; a verifier with no checkpoint contributes zeros.

Run:  python -m sidemodel.test_mcgn      (from pointmaze/)
  or: python sidemodel/test_mcgn.py
"""
import os, sys, json, importlib.util
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_POINT = os.path.dirname(_HERE)
_MV = os.path.join(_POINT, os.pardir, 'maze_update', 'maze_variants_v2')
sys.path.insert(0, _POINT)


def _load(fn, nm):
    spec = importlib.util.spec_from_file_location(nm, os.path.join(_HERE, fn))
    m = importlib.util.module_from_spec(spec); sys.modules[nm] = m
    spec.loader.exec_module(m); return m


model = _load('model.py', '_t_mcgn_model')
dataset = _load('dataset.py', '_t_mcgn_ds')
train = _load('train.py', '_t_mcgn_train')
sv = _load('sidemodel_verifier.py', '_t_mcgn_sv')

_PASS = []


def check(name, cond, detail=""):
    _PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def maps_for_task(tk):
    d = json.load(open(os.path.join(_MV, f'giant_task{tk}.json')))
    return [(np.array(v['maze_map'], int), tuple(v['goal'])) for v in d['variants']]


# ---------------------------------------------------------------- 1. crop
def test_occupancy_crop():
    print("test_occupancy_crop")
    mm = np.zeros((5, 6), int); mm[2, 3] = 1
    K = 5
    crop = dataset.occupancy_crop(mm, 0, 0, K)   # corner -> half the crop is OOB
    r = K // 2
    check("shape (K,K)", crop.shape == (K, K))
    check("center is query cell (free)", crop[r, r] == 0.0)
    check("top row OOB = wall", np.all(crop[0, :] == 1.0))
    check("left col OOB = wall", np.all(crop[:, 0] == 1.0))
    # a crop centered so the wall at (2,3) lands inside
    crop2 = dataset.occupancy_crop(mm, 2, 3, K)
    check("interior wall captured", crop2[r, r] == 1.0)


# ------------------------------------------------------- 2. descent direction
def test_descent_dir():
    print("test_descent_dir")
    # synthetic field: value = distance to (0,0) along a corridor; downhill = toward (0,0)
    field = np.array([[0., 1., 2., 3.],
                      [1., 2., 3., 4.],
                      [2., 3., 4., 5.]], float)
    dx, dy = dataset.geodesic_descent_dir(field, 1, 1)  # should go toward lower: up or left
    check("unit length", abs((dx*dx+dy*dy)**0.5 - 1.0) < 1e-5, f"|d|={ (dx*dx+dy*dy)**0.5:.3f}")
    # lower neighbor is (0,1)=1 [up, dy=-1] or (1,0)=1 [left, dx=-1]; both strictly downhill
    ii, jj = 1 + int(round(dy)), 1 + int(round(dx))
    check("steps to a strictly-lower cell", field[ii, jj] < field[1, 1],
          f"from {field[1,1]:.0f} to {field[ii,jj]:.0f}")
    zx, zy = dataset.geodesic_descent_dir(field, 0, 0)   # the goal (min)
    check("zero at goal/min", (zx, zy) == (0.0, 0.0))


# ----------------------------------------------------------- 3. build_samples
def test_build_samples():
    print("test_build_samples")
    mm, gij = maps_for_task(1)[0]
    s = dataset.build_samples(mm, gij, K=15)
    N = s['occ'].shape[0]
    check("occ shape (N,1,15,15)", s['occ'].shape == (N, 1, 15, 15), str(s['occ'].shape))
    check("goal shape (N,3)", s['goal'].shape == (N, 3))
    check("target shape (N,2)", s['target'].shape == (N, 2))
    check("N>0 free cells", N > 0, f"N={N}")
    gn = np.linalg.norm(s['goal'][:, :2], axis=1)
    check("goal xy unit (or at-goal 0)", np.all((np.abs(gn - 1) < 1e-3) | (gn < 1e-3)))
    tn = np.linalg.norm(s['target'], axis=1)
    check("targets unit length", np.all(np.abs(tn - 1) < 1e-3))


# --------------------------------------------------------------- 4. model fwd
def test_model_forward():
    print("test_model_forward")
    net = model.MapConditionedGuidanceNet(patch=15)
    occ = torch.zeros(7, 1, 15, 15); goal = torch.randn(7, 3)
    out = net(occ, goal)
    check("forward shape (B,2)", tuple(out.shape) == (7, 2), str(tuple(out.shape)))
    tgt = torch.nn.functional.normalize(torch.randn(7, 2), dim=-1)
    loss = model.cosine_direction_loss(out, tgt)
    check("loss finite & >=0", torch.isfinite(loss) and loss.item() >= 0, f"loss={loss.item():.3f}")


# ---------------------------------------------------------------- 5. training
def test_training_reduces_loss():
    print("test_training_reduces_loss")
    mm, gij = maps_for_task(1)[0]
    s = dataset.build_samples(mm, gij, K=15)
    O = torch.tensor(s['occ']); G = torch.tensor(s['goal']); T = torch.tensor(s['target'])
    torch.manual_seed(0)
    net = model.MapConditionedGuidanceNet(patch=15)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    l0 = model.cosine_direction_loss(net(O, G), T).item()
    for _ in range(60):
        opt.zero_grad()
        loss = model.cosine_direction_loss(net(O, G), T); loss.backward(); opt.step()
    l1 = model.cosine_direction_loss(net(O, G), T).item()
    check("loss decreased", l1 < l0 - 0.05, f"{l0:.3f} -> {l1:.3f}")


# ------------------------------------------------------------- 6. D4 augment
def test_d4_augment():
    print("test_d4_augment")
    mm, gij = maps_for_task(1)[0]
    s = dataset.build_samples(mm, gij, K=15)
    O, G, T = s['occ'], s['goal'], s['target']
    OA, GA, TA = train.d4_augment(O, G, T)
    check("8x sample count", OA.shape[0] == 8 * O.shape[0], f"{O.shape[0]}->{OA.shape[0]}")
    # equivariance: the k=1 rotation block (indices [2N:3N] since order is
    # [rot0, flip0, rot1, flip1, ...]) — check one explicit 90deg rotation.
    Ork = np.rot90(O, 1, axes=(2, 3))
    Trk = train._rot2d(T, 1)
    check("rot90 crop matches augmented block", np.allclose(Ork, OA[2*O.shape[0]:3*O.shape[0]]))
    check("rot90 target matches augmented block", np.allclose(Trk, TA[2*O.shape[0]:3*O.shape[0]]))
    # rotating the crop must rotate the target identically (equivariance holds elementwise)
    check("targets stay unit after rotation",
          np.all(np.abs(np.linalg.norm(TA, axis=1) - 1) < 1e-3))


# --------------------------------------------------------- 7. transfer (holdout)
def _multi(tasks, K=15, augment=False):
    occ, goal, tgt = [], [], []
    for tk in tasks:
        for mm, gij in maps_for_task(tk):
            s = dataset.build_samples(mm, gij, K=K)
            if s['occ'].shape[0]:
                occ.append(s['occ']); goal.append(s['goal']); tgt.append(s['target'])
    O, G, T = np.concatenate(occ), np.concatenate(goal), np.concatenate(tgt)
    if augment:
        O, G, T = train.d4_augment(O, G, T)
    return O, G, T


def test_transfer():
    print("test_transfer (train {1,2,4,5} D4-aug, holdout task3 — unseen maps+goal)")
    Otr, Gtr, Ttr = _multi([1, 2, 4, 5], augment=True)
    Oho, Gho, Tho = _multi([3], augment=False)
    torch.manual_seed(0)
    net = model.MapConditionedGuidanceNet(patch=15)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    O = torch.tensor(Otr); G = torch.tensor(Gtr); T = torch.tensor(Ttr)
    for ep in range(40):
        idx = torch.randperm(len(O))
        for b in range(0, len(O), 512):
            bi = idx[b:b+512]; opt.zero_grad()
            loss = model.cosine_direction_loss(net(O[bi], G[bi]), T[bi]); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pm = net(torch.tensor(Oho), torch.tensor(Gho)).numpy()
    pmn = pm / (np.linalg.norm(pm, axis=1, keepdims=True) + 1e-6)
    cos = (pmn * Tho).sum(1).mean()
    # goal-direction baseline
    gdir = Gho[:, :2] / (np.linalg.norm(Gho[:, :2], axis=1, keepdims=True) + 1e-6)
    bcos = (gdir * Tho).sum(1).mean()
    det = (gdir * Tho).sum(1) < 0.5
    mcos = (pmn[det] * Tho[det]).sum(1).mean()
    bmcos = (gdir[det] * Tho[det]).sum(1).mean()
    print(f"    holdout cos: MCGN={cos:.3f}  goal-dir baseline={bcos:.3f}")
    print(f"    must-detour ({int(det.sum())}): baseline={bmcos:.3f} -> MCGN={mcos:.3f}")
    check("holdout transfer positive (beats random)", cos > 0.1, f"cos={cos:.3f}")
    check("MCGN beats goal-dir baseline on must-detour cells", mcos > bmcos + 0.1,
          f"{bmcos:.3f} -> {mcos:.3f}")


# ---------------------------------------------------------- 8. MCGNVerifier
class _FakeEnv:
    def __init__(self, mm, goal_ij, mu=4.0, ox=4.0, oy=4.0):
        self.maze_map = mm
        self._maze_unit = mu; self._offset_x = ox; self._offset_y = oy
        gi, gj = goal_ij
        gx = gj * mu - ox; gy = gi * mu - oy
        self.cur_task_info = {'goal_xy': (gx, gy)}


def test_verifier():
    print("test_verifier")
    mm, gij = maps_for_task(1)[0]
    env = _FakeEnv(mm, gij)
    # (a) no checkpoint -> identity (zeros)
    v0 = sv.MCGNVerifier(args=None, ckpt=None, weight=1.0)
    v0.update_env(env)
    x = torch.zeros(2, 5, 6, requires_grad=True)          # (B,T,adim>=4)
    x = x.clone(); x[..., 2:4] = torch.randn(2, 5, 2)
    lp0 = v0.get_guidance(x, return_logp=True)
    check("no-model logp is zeros", torch.allclose(lp0, torch.zeros(2)), str(lp0.tolist()))
    g0 = v0.get_guidance(x)
    check("no-model grad is zeros", torch.allclose(g0, torch.zeros_like(x)))
    # (b) with a (quickly) trained model -> nonzero logp + finite gradient
    s = dataset.build_samples(mm, gij, K=15)
    O = torch.tensor(s['occ']); G = torch.tensor(s['goal']); T = torch.tensor(s['target'])
    torch.manual_seed(0)
    net = model.MapConditionedGuidanceNet(patch=15)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(40):
        opt.zero_grad(); model.cosine_direction_loss(net(O, G), T).backward(); opt.step()
    v = sv.MCGNVerifier(args=None, ckpt=None, weight=1.0)
    v.net = net.eval(); v.patch = 15
    v.update_env(env)
    check("dir table built for free cells", v._dir_table is not None
          and np.abs(v._dir_table).sum() > 0)
    xx = torch.zeros(3, 8, 6)
    xx[..., 2:4] = torch.randn(3, 8, 2) * 2.0
    xx.requires_grad_(True)
    lp = v.get_guidance(xx, return_logp=True)
    check("logp shape (B,)", tuple(lp.shape) == (3,), str(tuple(lp.shape)))
    check("logp finite", torch.all(torch.isfinite(lp)))
    grad = v.get_guidance(xx)
    check("grad shape matches x", grad.shape == xx.shape)
    check("grad finite & nonzero", torch.all(torch.isfinite(grad)) and grad.abs().sum() > 0)
    # gradient only flows to the position channels (2:4)
    nonpos = grad[..., [0, 1, 4, 5]].abs().sum().item()
    check("grad only on position channels", nonpos < 1e-6, f"nonpos={nonpos:.2e}")


if __name__ == "__main__":
    for fn in (test_occupancy_crop, test_descent_dir, test_build_samples,
               test_model_forward, test_training_reduces_loss, test_d4_augment,
               test_transfer, test_verifier):
        fn()
    n = len(_PASS); k = sum(_PASS)
    print(f"\n{k}/{n} checks passed")
    sys.exit(0 if k == n else 1)
