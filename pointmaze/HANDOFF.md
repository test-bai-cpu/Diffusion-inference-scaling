# Project Handoff — Map-Aware Inference-Time Search for Diffusion Pointmaze Planning

**Branch:** `guidance-claude` · **Repo:** `Diffusion-inference-scaling` · **Head commit:** `dd2c986`
**Baseline paper:** arXiv 2505.23614v2 · **Benchmark:** OGBench `pointmaze-giant-navigate-v0`

---

## 0. Read this first — the one thing to know

Everything **geometric, metric, and side-model** was computed on **real maze data in the
sandbox (CPU)** and is trustworthy. Everything that requires the **diffusion planner to actually
roll out** — the end-to-end success / collision degradation curves — was **NOT run**, because the
sandbox has no GPU and no `maze` conda env. Those planner numbers are delivered as **tested code
for you to run**. The one figure that shows planner comparison numbers
(`degradation_demo_curves.png`) is **synthetic/illustrative** and is stamped as such.

So your job in taking over is essentially: **run the GPU sweep the code is ready for, and replace
the illustrative figure with the real one.**

---

## 1. The goal, in one paragraph

Take a *frozen, pretrained* diffusion trajectory planner and make it better purely at
**inference time** (plus one small learned side model), targeting two concrete failure modes that
the base planner exhibits in the giant maze:

- **Dead-end trapping** — the planner drives into a pocket and never recovers.
- **Diagonal corner-cutting** — the path slips diagonally between two wall blocks that only touch
  at a corner; the *verifier* used by the paper thinks that gap is free, but the *simulator*
  treats it as blocked, so the episode silently fails.

Three aims:
- **(a)** Raise success rate and lower collision rate on the maps the planner already knows.
- **(b)** Make the planner **map-aware** so it generalizes to *new, unseen* maps.
- **(c)** Build a **map-variant generator** whose difficulty *levels* produce a **monotone** drop in
  success — so "level 3 is harder than level 1" is actually true when measured.

---

## 2. Current stage — all 8 planned steps are DONE (as code)

| # | Step | Commit | Real result in-sandbox? |
|---|------|--------|--------------------------|
| 1 | Outcome instrumentation (separate success / collision / corner-cut / dead-end / stall) | `c7bd8c1` | ✅ validated on real giant maze |
| 2 | Calibrated difficulty metric | `1809fff`, `0539e69` | ✅ Spearman −0.53 on 60 real DFS runs |
| 3 | Variant generator with monotone difficulty ladders | `ade4159` | ✅ ladders built & calibrated |
| 4 | Corner-safe clearance verifier | `517079c` | ✅ fixes diagonal-gap bug on 4 real pinches |
| 5 | Corrected distance-to-goal field + dead-end detector | `67bdc16`, `bf43e64` | ✅ field RMS 0.28→0.03, 0% barrier violations |
| 6 | **MAFGS** — Map-Aware Feasibility-Guided Search | `b5fc881` | ✅ logic tested on real geometry (mocked GPU deps) |
| 7 | **MCGN** — learned map-conditioned guidance side model | `addc043` | ✅ trains & transfers on CPU |
| 8 | Evaluation protocol + graceful-degradation report | `dd2c986` | ⚠️ harness tested; **planner sweep not run** |

**67 CPU checks pass** across 6 test suites (see §7).

---

## 3. What each piece is, and where it lives

All paths relative to repo root.

### Instrumentation — `pointmaze/search/episode_metrics.py`
`score_episode()` turns one rollout into separated metrics instead of a single success bit:
`success` (total_reward>0), `collision_rate` (exact box-SDF contact with ball_radius=0.5,
contact_margin=0.15), `cornercut_rate` (proximity to diagonal pinches, radius 0.9),
`deadend_frac`, `stalled_prog`, `final_gap`, `steps`. Wired into `base_pipeline.sample()` (guarded
by `'pointmaze' in dataset`); `run.py` writes `results_detailed.csv`; `plot_metrics.py` renders the
breakdown. **This is what lets you see *why* an episode failed, not just that it did.**

### Difficulty metric — `maze_update/difficulty_metric.py`
Composite of 6 features (`spectral, sp_ratio, net_shift, deadend_delta, novelty_x_elong, n_blocked`).
Calibrated against measured planner success on 60 logged runs: **rank correlation −0.53**, vs the
naive "relative difficulty index" at −0.30 (which inverted 20% of level-pairs). `calibrate_difficulty.py`
+ `difficulty_calibration.png` reproduce it.

### Variant generator — `maze_update/generate_variants.py` → `maze_update/maze_variants_v2/giant_task{1..5}.json`
Block-only edits to the base maze (never opens walls into invalid mazes), grouped into **4 levels ×
3 variants**, with difficulty non-decreasing across levels. Drop-in OGBench schema. `variant_ladder.png`
is the contact sheet.

### Corner-safe verifier — `pointmaze/search/maze_verifier_clearance.py`
`CornerSafeMazeVerifier`: exact box signed-distance + clearance margin + segment supersampling +
an explicit **diagonal-pinch penalty**. This is the fix for corner-cutting: the canonical
diagonal gap that the old verifier scored 0.00 (thought free) now scores 3.69 (penalized). All 4
real giant-maze pinches fixed. Proven in `test_verifier_geometry.py`.

### Distance field v2 — `pointmaze/search/distance_field_v2.py` + `distance_field_verifier_v2.py`
4-connected BFS to goal + wall-respecting smoothing (no blurring *through* walls) + strictly-uphill
wall fill, plus a **dead-end pocket detector** (leaf-pruning). Field fidelity RMS improved 0.28 → 0.03
vs the true geodesic, with **0 false cross-wall shortcuts** (was 8%). This is the "which way is
actually toward the goal" signal that guides search and detects traps.

### MAFGS — `pointmaze/search/methods/mafgs.py`  (`--method mafgs`)
The main contribution for aim (a). During diffusion sampling it:
- **Guides** each denoising step with a weighted sum of clearance loss + goal-field descent
  (+ optional MCGN term, see below).
- **Gates** feasibility analytically: if a candidate sub-trajectory shows collision, corner-cut,
  stall, or dead-end occupancy above tolerance, it **backtracks** and re-samples.
- Key knob defaults (in `configs.py`): `clear_weight=1.0, field_weight=0.5, margin=0.7,
  corner_radius=0.9, collision_tol=0.05, stall_tol=0.10, deadend_tol=0.15`. Search budget:
  `recur_depth=12, budget=20, threshold=6, start_step=12`.
- **Design choice worth knowing:** the learned side model feeds only the *guidance*, never the
  *gate*. A bad neural prediction can nudge but can never override the analytic safety check.

### MCGN — `pointmaze/sidemodel/` (aim (b), the map-awareness)
Small CNN (`model.py`, patch=15 occupancy crop + goal → 2D direction) trained to predict the
**geodesic descent direction** at any cell. Trained **multi-map with D4 augmentation**
(`train.py`, `dataset.py`) — the D4 augmentation is what makes it generalize. `sidemodel_verifier.py`
wraps it as an MCGN guidance term for MAFGS.
- **Real transfer result (CPU):** trained on task {1,2,4,5} variant layouts, held out task 3
  (unseen layouts *and* goal): holdout direction-cosine **0.46 vs 0.33** goal-direction baseline;
  on "must-detour" cells (where naive goal-pointing is wrong) it swings **−0.22 → +0.22**. That is
  the map-awareness working on maps it never saw.

### Evaluation harness — `pointmaze/run_eval.py`, `plot_degradation.py`, `EVALUATION.md`
- `run_eval.py`: sweeps **4 configs × 5 tasks × 4 levels × N seeds**, writes a tidy resumable
  `eval_results.csv`. The 4 configs share the frozen backbone: `dfs` (paper baseline), `field`
  (dfs + distance field), `mafgs`, `mafgs+mcgn`.
- `plot_degradation.py`: produces `degradation_curves.png` (success / collision / dead-end vs
  level with 95% CI + a "margin over dfs baseline" panel) and `degradation_report.md`
  (monotonicity Spearman per method, aggregate table, hardest-level margin, transfer paragraph).
- `EVALUATION.md`: the standing protocol — exact commands, what "graceful degradation" should
  look like, and the full test list.

---

## 4. Real vs. synthetic — the honest ledger

**REAL (computed on real data in-sandbox, trust these):**
- Corner-cut fix: diagonal gap 0.00 → 3.69 penalty on 4 real giant pinches.
- Distance field: RMS 0.28 → 0.03 vs true geodesic; 0% cross-wall shortcuts (was 8%).
- Difficulty calibration: Spearman −0.53 (composite) vs −0.30 (RDI) on 60 real DFS runs.
- MCGN transfer: holdout cosine 0.46 vs 0.33; must-detour −0.22 → +0.22.
- Episode-metric behavior: clean path → all-zero failures + success; wall-hug → collision=1.0;
  through-pinch → cornercut>0; dead-end wander → deadend 0.21 / stall 0.49.

**SYNTHETIC (illustrative placeholders — DO NOT cite as results):**
- `degradation_demo_curves.png` — planner success/collision/dead-end curves. **Stamped with a red
  "ILLUSTRATIVE — synthetic numbers" banner.** Shows the *shape* of a good outcome only.
- `degradation_report_example.md` + `eval_results_demo.csv` — same synthetic source.

These synthetic files exist only so you can see exactly what `plot_degradation.py` will emit. The
moment you run the real sweep, they are replaced by real files of the same shape.

---

## 5. How to take over — environment + exact commands

Run from `pointmaze/` in your GPU `maze` conda env (the one the base planner already uses).

**Step A — Train the MCGN side model** (minutes on GPU; needed only for the `mafgs+mcgn` config):
```
python sidemodel/train.py \
  --maze_json ../maze_update/maze_variants_v2/giant_task1.json \
              ../maze_update/maze_variants_v2/giant_task2.json \
              ../maze_update/maze_variants_v2/giant_task4.json \
              ../maze_update/maze_variants_v2/giant_task5.json \
  --holdout task3 --epochs 40 --patch 15 --out sidemodel/mcgn_giant.pt
```
It prints holdout cosine + within-45° accuracy + the goal-direction baseline so you can confirm the
transfer number before spending sweep time.

**Step B — Run the evaluation sweep** (this is the real experiment; it drives the diffusion planner):
```
python run_eval.py \
  --methods dfs field mafgs mafgs+mcgn \
  --tasks 1 2 3 4 5 --levels 0 1 2 3 --seeds 0 1 2 \
  --maze_json_dir ../maze_update/maze_variants_v2 \
  --mcgn_ckpt sidemodel/mcgn_giant.pt \
  --device cuda:0 --out eval_results.csv
```
`eval_results.csv` is written **incrementally and is resumable** — safe to Ctrl-C and restart, or to
run one method at a time.

**Step C — Make the real figures + report:**
```
python plot_degradation.py eval_results.csv degradation
# -> degradation_curves.png  +  degradation_report.md   (these replace the demo files)
```

**Sanity check before the big sweep:** start with one method and one seed
(`--methods dfs mafgs --tasks 1 --seeds 0`) to confirm the planner loads and CSV rows appear, then
scale up.

---

## 6. Which experiments to run (prioritized)

1. **Headline comparison** — the full Step B sweep. This answers aim (a): does MAFGS raise success
   and cut collision/corner-cut vs the `dfs` paper baseline on known maps? Look at the aggregate
   table and the "margin over dfs" panel.
2. **Graceful degradation / monotonicity** — from the same CSV, `plot_degradation.py` reports the
   Spearman(level, success) per method. This validates aim (c): success should fall monotonically
   as level rises, for every method.
3. **Map transfer / map-awareness** — compare `mafgs` vs `mafgs+mcgn`, paying attention to the
   *harder* levels and to task 3 (the MCGN holdout). If MCGN helps most exactly where naive
   goal-pointing fails, that is aim (b) confirmed end-to-end.
4. **Ablations (optional, cheap given the harness):** `field` vs `dfs` isolates the distance-field
   contribution; `mafgs` vs `mafgs+mcgn` isolates the learned side model; toggling
   `mafgs_corner_weight` / `mafgs_deadend_tol` isolates each failure-mode gate.

---

## 7. How to verify nothing rotted before you start

All CPU-runnable, from `pointmaze/`:
```
python search/test_episode_metrics.py     # 4/4   outcome metrics
python search/test_verifier_geometry.py   # all   corner-safe verifier (diagonal-gap fix)
python search/test_distance_field.py      # 5/5   field fidelity + dead-end detector
python search/test_mafgs.py               # 8/8   MAFGS guide+gate logic (mocked GPU deps)
python sidemodel/test_mcgn.py             # 31/31 MCGN model/dataset/transfer/verifier
python test_run_eval.py                   # 23/23 sweep bookkeeping + plotter + report
```
Last full run: **all green (67 checks).**

---

## 8. Caveats & things to watch

- **The sweep is the only unrun part.** If the planner API in your `maze` env differs from what
  `run_eval.py` assumes (it calls `get_pipe(args).experiment()` via `search.script_utils`), that is
  the one integration point to check first. The harness is tested against a *mocked* pipe, so the
  contract is documented but not exercised against the real planner.
- **MCGN checkpoint path** must match between Step A `--out` and Step B `--mcgn_ckpt`
  (default `sidemodel/mcgn_giant.pt`). If `mcgn_ckpt` is empty, the `mafgs+mcgn` config silently
  degrades to plain `mafgs` (the MCGN guidance term is zero) — so an empty path won't crash, it'll
  just give you no side-model benefit.
- **Difficulty ladders** live in `maze_variants_v2/` (the calibrated v2), not the older
  `maze_variants/`. Point the sweep at v2.
- **Coordinate convention** (if you extend the geometry code): world `xy = (j*4 − 4, i*4 − 4)`;
  cell `i = round((y+4)/4)`, `j = round((x+4)/4)`; grid 0=free, 1=wall; giant is (12,16).

---

*Generated at handoff. All code committed on `guidance-claude` through `dd2c986`.*
