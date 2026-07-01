# Evaluation protocol — map-aware inference-time search for the pointmaze planner

This document is the entry point for reproducing the full evaluation. Everything
below runs on the user's GPU machine with the `maze` conda environment and the
pretrained diffusion checkpoints; the geometry/metric/generator/verifier code it
calls is unit-tested in CI-style CPU tests (listed at the end).

## What is being compared

All four configurations share the **same frozen diffusion backbone**. They
differ only in the inference-time search / guidance:

| config | search | guidance | map-aware? |
|---|---|---|---|
| `dfs` | depth-first (paper baseline) | `MazeVerifier` (occupancy) | no |
| `field` | dfs | + BFS distance-field descent | goal-distance only |
| `mafgs` | **Map-Aware Feasibility-Guided Search** | clearance + distance-field composite, hard feasibility gate | yes (analytic) |
| `mafgs+mcgn` | mafgs | + learned Map-Conditioned Guidance Net | yes (analytic + learned) |

The two failure modes the project targets are measured directly, separated from
plain success, by `search/episode_metrics.py`:

- **collision / corner-cut** — the diagonal corner-cut bug (a path slipping
  between two diagonally-touching wall blocks the simulator treats as blocked)
  is caught by the corner-safe clearance verifier and reported as
  `collision_rate` / `cornercut_rate`.
- **dead-end trapping** — `deadend_frac` is the fraction of waypoints inside
  leaf-pruned pockets of the distance field (an undiluted trap detector, robust
  to a short detour in a long rollout).

## Difficulty ladders (calibrated)

`maze_update/maze_variants_v2/giant_task{1..5}.json` hold the calibrated variant
ladders: `n_levels` levels × `n_variants` variants, each variant carrying its own
`maze_map` and a monotone `difficulty` score. The difficulty metric was
calibrated against 60 logged DFS runs (composite Spearman −0.531 vs −0.295 for
the nominal RDI). Sweeping levels 0..L−1 is what demonstrates the **monotone
success drop** (Aim c) and lets us measure each method's margin as the maze gets
harder.

## Step 1 — (optional) train the MCGN side model

```bash
cd pointmaze
python -m sidemodel.train \
    --maze_json ../maze_update/maze_variants_v2/giant_task1.json \
                ../maze_update/maze_variants_v2/giant_task2.json \
                ../maze_update/maze_variants_v2/giant_task4.json \
                ../maze_update/maze_variants_v2/giant_task5.json \
    --holdout  ../maze_update/maze_variants_v2/giant_task3.json \
    --epochs 40 --patch 15 --out sidemodel/mcgn_giant.pt
```

Trains on tasks {1,2,4,5} variant layouts with **D4 augmentation** (the
equivariance that forces genuine map-conditioning) and reports held-out transfer
to the unseen task-3 layouts. Expected holdout: mean cosine ≈ 0.46 vs ≈ 0.33 for
a goal-direction baseline; on must-detour cells (where heading straight at the
goal points into a wall) the model swings from ≈ −0.22 to ≈ +0.22 — it turns
around obstacles on maps it never saw. The side model conditions **only** on a
local K×K occupancy crop + relative-goal vector (never a map id or absolute
coordinates), which is why one checkpoint transfers.

`mafgs+mcgn` adds the trained net as an extra guidance term. It never enters the
feasibility gate, so a bad prediction cannot cut a corner or enter a dead-end
that the analytic gate would reject; absent a checkpoint the term is exactly zero
and `mafgs+mcgn` reduces to `mafgs`.

## Step 2 — run the sweep

```bash
python run_eval.py \
    --maze_json_dir ../maze_update/maze_variants_v2 \
    --methods dfs field mafgs mafgs+mcgn \
    --tasks 1 2 3 4 5 --levels 0 1 2 3 --seeds 0 1 2 \
    --mcgn_ckpt sidemodel/mcgn_giant.pt \
    --device cuda:0 --out eval_results.csv
```

Writes one tidy row per (config, task, level, variant, seed) with the separated
metrics + difficulty bookkeeping. `--all_variants` sweeps every variant at each
level (default: the first, lowest-index variant). The CSV is append-only and
`flush`ed per run, so a long sweep is resumable and inspectable mid-flight.

## Step 3 — figures + report

```bash
python plot_degradation.py eval_results.csv degradation
```

Produces:

- `degradation_curves.png` — (a) success vs level per method with 95% CI;
  (b) collision vs level; (c) dead-end vs level; (d) success **margin over the
  dfs baseline** vs level (the margin widens where it matters — harder levels).
- `degradation_report.md` — monotonicity check (Spearman level vs success per
  method), aggregate success/collision/corner-cut/dead-end/steps/compute table,
  the MAFGS margin at the hardest level, and the MCGN map-transfer result.

`plot_metrics.py` (from Step 1) additionally renders the per-run failure
attribution bars from the same CSV schema.

### Illustrative output

`degradation_demo_curves.png` + `degradation_demo_report.md` are generated from
**synthetic** numbers (`eval_results_demo.csv`) to show the exact shape of the
deliverable before the GPU sweep is run. Replace with real `eval_results.csv`
output for reportable values.

## Graceful degradation — what "good" looks like

1. **Monotone ladder**: every method's success is non-increasing across levels
   (Spearman(level, success) strongly negative). If a method's curve inverts,
   the ladder — not the planner — is miscalibrated for that method.
2. **Margin widens with difficulty**: on easy levels all methods succeed, so
   margins are small; the map-aware methods should separate on the harder levels
   where the naive planner cuts corners or gets trapped (panel d).
3. **Safety monotone too**: collision and dead-end fractions should rise more
   slowly for the map-aware methods — they fail *gracefully* (longer paths,
   more backtracking) rather than *unsafely* (wall clipping, trapping).

## Tests (run in the sandbox, no GPU)

```bash
python search/test_episode_metrics.py      # metric separation (4/4)
python search/test_verifier_geometry.py    # corner-safe clearance (diagonal pinch)
python search/test_distance_field.py       # distance field v2 fidelity + dead-ends
python search/test_mafgs.py                # MAFGS gate + backtracking (8/8)
python sidemodel/test_mcgn.py              # MCGN transfer + verifier (31/31)
python test_run_eval.py                    # eval driver + plotter (23/23)
```
