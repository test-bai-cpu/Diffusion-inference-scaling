"""
train.py — train the Map-Conditioned Guidance Network (runs on the user's GPU).
===============================================================================
Builds map-conditioned samples from one or more mazes (default: the giant maze
free cells) and fits MCGN to predict the geodesic descent direction. Held-out
mazes measure generalization to unseen layouts.

Usage
-----
    python -m sidemodel.train \
        --maze_json ../maze_update/maze_variants_v2/giant_task1.json \
        --epochs 40 --patch 15 --out sidemodel/mcgn_giant.pt

To test map-transfer, pass several --maze_json (train on all but one) or use
--holdout_frac to hold out a random subset of cells. The saved checkpoint stores
{state_dict, patch} so the verifier can reconstruct the model.
"""
import argparse, json, os, sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                 # pointmaze/
try:
    from sidemodel.model import MapConditionedGuidanceNet, cosine_direction_loss
    from sidemodel.dataset import build_samples
except Exception:
    import importlib.util
    def _l(fn, nm):
        spec = importlib.util.spec_from_file_location(nm, os.path.join(_HERE, fn))
        m = importlib.util.module_from_spec(spec); sys.modules[nm] = m
        spec.loader.exec_module(m); return m
    _m = _l('model.py', '_mcgn_model'); _d = _l('dataset.py', '_mcgn_ds')
    MapConditionedGuidanceNet = _m.MapConditionedGuidanceNet
    cosine_direction_loss = _m.cosine_direction_loss
    build_samples = _d.build_samples


def load_maze_maps(path, use_variants=True):
    """Return a list of (maze_map, goal_ij). With use_variants, expand every
    variant layout in the JSON (each carries its own maze_map + goal), which is
    what forces genuine map-conditioning; otherwise just the base maze."""
    d = json.load(open(path))
    out = []
    if use_variants and d.get('variants'):
        for v in d['variants']:
            out.append((np.array(v['maze_map'], dtype=int), tuple(v['goal'])))
    else:
        out.append((np.array(d['base_maze'], dtype=int), tuple(d['goal'])))
    return out


def _rot2d(v, k):
    """Rotate 2D vectors (…,2) by k*90deg in image coords (x right, y down)."""
    x, y = v[..., 0], v[..., 1]
    for _ in range(k % 4):
        x, y = -y, x
    return np.stack([x, y], -1)


def d4_augment(occ, goal, tgt):
    """Apply the 8 dihedral symmetries. The descent-direction task is
    D4-equivariant: rotating/reflecting the occupancy crop rotates/reflects both
    the relative-goal vector and the target direction identically. This multiplies
    layout diversity 8x and forces the CNN to read occupancy instead of
    memorizing a global goal-direction prior."""
    oO, oG, oT = [], [], []
    for k in range(4):
        Ok = np.rot90(occ, k, axes=(2, 3)).copy()
        Gk = goal.copy(); Gk[:, :2] = _rot2d(goal[:, :2], k)
        Tk = _rot2d(tgt, k)
        oO.append(Ok); oG.append(Gk); oT.append(Tk)
        # horizontal flip (negate x component of both vectors)
        Of = Ok[:, :, :, ::-1].copy()
        Gf = Gk.copy(); Gf[:, 0] *= -1
        Tf = Tk.copy(); Tf[:, 0] *= -1
        oO.append(Of); oG.append(Gf); oT.append(Tf)
    return np.concatenate(oO), np.concatenate(oG), np.concatenate(oT)


def make_dataset(maze_paths, patch, use_variants=True, augment=False):
    occ, goal, tgt = [], [], []
    for p in maze_paths:
        for mm, goal_ij in load_maze_maps(p, use_variants):
            s = build_samples(mm, goal_ij, K=patch)
            if s['occ'].shape[0]:
                occ.append(s['occ']); goal.append(s['goal']); tgt.append(s['target'])
    O, G, T = np.concatenate(occ), np.concatenate(goal), np.concatenate(tgt)
    if augment:
        O, G, T = d4_augment(O, G, T)
    return O, G, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maze_json', nargs='+', required=True)
    ap.add_argument('--holdout', nargs='*', default=[],
                    help='maze json(s) held out for generalization eval')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patch', type=int, default=15)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--out', default='sidemodel/mcgn.pt')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--base_only', action='store_true',
                    help='train on base_maze only (default: expand all variant layouts)')
    ap.add_argument('--no_augment', action='store_true',
                    help='disable D4 augmentation (default: on; augmentation is '
                         'what drives unseen-map transfer)')
    args = ap.parse_args()

    use_variants = not args.base_only
    augment = not args.no_augment
    occ, goal, tgt = make_dataset(args.maze_json, args.patch,
                                  use_variants=use_variants, augment=augment)
    print(f"train samples: {len(occ)} from {len(args.maze_json)} maze file(s) "
          f"(variants={'on' if use_variants else 'off'}, "
          f"D4-augment={'on' if augment else 'off'})")
    ds = TensorDataset(torch.tensor(occ), torch.tensor(goal), torch.tensor(tgt))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)

    net = MapConditionedGuidanceNet(patch=args.patch).to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for ep in range(args.epochs):
        net.train(); tot = 0.0
        for o, g, t in dl:
            o, g, t = o.to(args.device), g.to(args.device), t.to(args.device)
            opt.zero_grad()
            loss = cosine_direction_loss(net(o, g), t)
            loss.backward(); opt.step()
            tot += loss.item() * len(o)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep:3d}  train cos-loss {tot/len(ds):.4f}")

    torch.save({'state_dict': net.state_dict(), 'patch': args.patch}, args.out)
    print(f"saved {args.out}")

    if args.holdout:
        ho, hg, ht = make_dataset(args.holdout, args.patch,
                                  use_variants=use_variants, augment=False)
        net.eval()
        with torch.no_grad():
            pred = net(torch.tensor(ho).to(args.device),
                       torch.tensor(hg).to(args.device)).cpu()
        pn = pred / (pred.norm(dim=-1, keepdim=True) + 1e-6)
        htt = torch.tensor(ht)
        cos = (pn * htt).sum(-1).mean().item()
        acc = ((pn * htt).sum(-1) > 0.707).float().mean().item()
        # goal-direction baseline: does the CNN beat "just head toward the goal"?
        gdir = torch.tensor(hg[:, :2])
        gdir = gdir / (gdir.norm(dim=-1, keepdim=True) + 1e-6)
        bcos = (gdir * htt).sum(-1).mean().item()
        # focus on 'must-detour' cells where goal-dir disagrees with the true path
        det = (gdir * htt).sum(-1) < 0.5
        mcos = (pn[det] * htt[det]).sum(-1).mean().item() if det.any() else float('nan')
        bmcos = (gdir[det] * htt[det]).sum(-1).mean().item() if det.any() else float('nan')
        print(f"HOLD-OUT ({len(ho)} samples): MCGN mean cos={cos:.3f}  "
              f"within-45deg acc={acc:.3f}")
        print(f"  goal-direction baseline mean cos={bcos:.3f}  "
              f"(MCGN {'beats' if cos > bcos else 'below'} baseline by {cos-bcos:+.3f})")
        print(f"  must-detour cells ({int(det.sum())}): baseline cos={bmcos:.3f} -> "
              f"MCGN cos={mcos:.3f}  ({mcos-bmcos:+.3f}); positive = turns around walls")


if __name__ == "__main__":
    main()
