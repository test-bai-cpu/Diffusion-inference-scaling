#!/usr/bin/env python
"""
calibrate_difficulty.py
=======================
Reproduce the difficulty-metric calibration against the logged DFS success rates
and emit `difficulty_calibration.png`.

What it does
------------
1. Parse per-(task, level, variant) DFS success rates from the results log.
2. Load the 5 giant-maze variant files and the OGBench training trajectories.
3. Compute DIRECTIONAL planner-difficulty features for every variant
   (difficulty_metric.compute_variant_features) and the calibrated RANK COMPOSITE
   (difficulty_metric.rank_composite).
4. Report Spearman rho of {nominal level, difficulty_score} vs success and
   the success-by-level ladder for nominal vs calibrated quantile bins.
5. Save the 2-panel calibration figure.

The headline metric written to each variant JSON is `difficulty_score`
(the calibrated rank composite from difficulty_metric.rank_composite).
Note: the DFS success rates must be logged on the SAME map version the
`--variants` directory points at; a stale log gives a meaningless rho.

Run:
    python maze_update/calibrate_difficulty.py
"""
import os, re, json, argparse, sys
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import difficulty_metric as dm

REPO = os.path.dirname(HERE)

# target_level float -> nominal level int (1..4); variant_index int -> variant (1..3)
LEVEL_OF = {0.2: 1, 0.35: 2, 0.5: 3, 0.65: 4}


def parse_success(log_path):
    """Return {(task, level, variant): success_rate} from the DFS results log."""
    pat = re.compile(r"dfs-(?:df-)?task(\d+)-level(\d+)-variant(\d+).*?Success Rate:\s*([\d.]+)")
    out = {}
    with open(log_path) as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                t, lv, vr, sr = m.groups()
                out[(int(t), int(lv), int(vr))] = float(sr)
    return out


def load_traffic(base_shape):
    obs_path = os.path.join(REPO, "pointmaze", "ogbench", "data",
                            "pointmaze-giant-navigate-v0", "observations.npy")
    if not os.path.exists(obs_path):
        print(f"[warn] training observations not found at {obs_path}; "
              f"corridor-novelty term will be 0.")
        return None
    obs = np.load(obs_path)
    return dm.traffic_map_from_observations(obs, base_shape)


def build_rows(variants_dir, success, traffic_cache):
    rows = []
    for task in range(1, 6):
        jf = os.path.join(variants_dir, f"giant_task{task}.json")
        if not os.path.exists(jf):
            continue
        data = json.load(open(jf))
        base = np.array(data["base_maze"], dtype=int)
        traffic = traffic_cache.get(base.shape)
        if traffic is None:
            traffic = load_traffic(base.shape)
            traffic_cache[base.shape] = traffic
        for v in data["variants"]:
            level = LEVEL_OF.get(round(float(v["target_level"]), 2))
            variant = int(v["variant_index"]) + 1
            key = (task, level, variant)
            if level is None or key not in success:
                continue
            grid = np.array(v["maze_map"], dtype=int)
            start = tuple(v["start"]); goal = tuple(v["goal"])
            res = dm.compute_variant_features(base, grid, start, goal, traffic)
            row = dict(res.features)
            row.update(task=task, level=level, variant=variant,
                       success=success[key],
                       difficulty_score=float(v.get("difficulty_score", np.nan)),
                       reachable=res.reachable)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.path.join(REPO, "pointmaze", "results_newvar_level.txt"))
    ap.add_argument("--variants", default=os.path.join(HERE, "maze_variants_v2"))
    ap.add_argument("--out", default=os.path.join(HERE, "difficulty_calibration.png"))
    args = ap.parse_args()

    success = parse_success(args.log)
    print(f"parsed {len(success)} logged (task,level,variant) success rates")
    rows = build_rows(args.variants, success, {})
    print(f"built features for {len(rows)} variants")

    succ = np.array([r["success"] for r in rows])
    level = np.array([r["level"] for r in rows])
    comp = dm.rank_composite(rows)

    rho_lvl = spearmanr(level, succ)[0]
    rho_comp = spearmanr(comp, succ)[0]
    print("\n== Spearman rho vs DFS success ==")
    print(f"  nominal level         {rho_lvl:+.3f}")
    print(f"  difficulty_score      {rho_comp:+.3f}   <-- calibrated")

    def curve(scores, k=4):
        binned = dm.quantile_levels(scores, k)
        return np.array([succ[binned == b].mean() for b in range(k)])
    nom = np.array([succ[level == l].mean() for l in (1, 2, 3, 4)])
    cal = curve(comp, 4)
    print("\n== success-by-level ladder (easy -> hard) ==")
    print(f"  nominal    {[f'{x:.1f}' for x in nom]}")
    print(f"  calibrated {[f'{x:.1f}' for x in cal]}")

    _plot(args.out, comp, succ, nom, cal, rho_comp)
    print(f"\nsaved {args.out}")


def _plot(out, comp, succ, nom, cal, rho_comp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 130})
    FOC, GREY = "#1f6feb", "#9aa0a6"
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.9))

    ax[0].scatter(comp, succ, s=34, c=FOC, edgecolor="white", linewidth=0.5, zorder=3)
    b = np.polyfit(comp, succ, 1); xs = np.linspace(comp.min(), comp.max(), 50)
    ax[0].plot(xs, np.polyval(b, xs), color=FOC, lw=2)
    ax[0].set_xlabel("difficulty_score (calibrated)"); ax[0].set_ylabel("DFS success rate (%)")
    ax[0].set_title(f"difficulty_score tracks success\nSpearman $\\rho$ = {rho_comp:+.2f}",
                    loc="left", fontsize=9.5); ax[0].margins(0.05)

    x = np.arange(4); w = 0.38
    ax[1].bar(x - w/2, nom, w, color=GREY, label="Nominal levels")
    ax[1].bar(x + w/2, cal, w, color=FOC, label="Calibrated quantile levels")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["L1\n(easy)", "L2", "L3", "L4\n(hard)"])
    ax[1].set_ylabel("Mean DFS success rate (%)")
    ax[1].set_title("Calibrated levels give a steeper,\nmore separated difficulty ladder",
                    loc="left", fontsize=9.5)
    ax[1].legend(frameon=False, fontsize=8, loc="upper right"); ax[1].margins(y=0.12)
    for xi, v in zip(x - w/2, nom): ax[1].text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5, color=GREY)
    for xi, v in zip(x + w/2, cal): ax[1].text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.5, color=FOC)

    fig.suptitle("Calibrating maze-variant difficulty_score to measured planner success "
                 "(giant maze)", fontsize=10.5, x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    main()
