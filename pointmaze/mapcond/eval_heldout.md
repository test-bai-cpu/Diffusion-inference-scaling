# Held-out-map evaluation protocol

Goal: measure whether making the map an **input** lets one model plan on a map
it was **never trained on** — the real test of map generalization, not just
multi-map memorization.

The metric is the repo's existing one: **OGBench rollout success rate** (reach
the goal within the step budget), computed by `run.py`. Training is mujoco-free;
this evaluation needs the simulator, so run it on the GPU/SLURM cluster.

---

## 1. The split

Train on a subset of maps, hold one out entirely:

| Role      | Maps                              |
|-----------|-----------------------------------|
| held-in   | `medium`, `large`, `teleport`     |
| held-out  | `giant`                           |

`giant` is the natural held-out target: it *defines* the 12×16 canvas, so no
map at eval time exceeds the trained canvas, and the repo already ships
`giant_task{1..5}.json` OOD variants used by the paper's search experiments.

> Swap the held-out map to stress different axes: hold out `teleport` to test an
> unseen **topology** (wormholes), hold out `large` to test an unseen **size**
> between medium and giant.

For the new giant variant layouts, use an additional within-family split to
measure generalization to unseen giant maps:

| Role      | Giant variants                         |
|-----------|----------------------------------------|
| held-in   | task1-4, var0-9                        |
| held-out  | task1-4, var10-11                      |

Train with `--variant_tasks 1 2 3 4 --variant_vars 0 1 2 3 4 5 6 7 8 9`,
then evaluate on tasks/variants whose grids were not in the training map specs.
This keeps the coordinate frame, canvas, and normalizer shared while testing
whether the map encoder uses the supplied layout rather than memorizing a fixed
giant maze.

---

## 2. Train the held-in model (cluster, GPU)

```bash
cd /projects/bhlg/yzhu37/Diffusion-inference-scaling/pointmaze
# edit MAPS in the launcher to the held-in set, then:
sbatch mapcond/run_multimap.slurm
#   MAPS="medium large teleport"   SAVEPATH=logs/mapcond/heldout_giant
```

Produces `logs/mapcond/heldout_giant/state_1000000.pt` (model + EMA + shared
normalizer + geometry config embedded).

---

## 3. Three arms to compare

All three use the SAME DFS/BFS search and the SAME verifiers
(`MazeVerifier`, `DistanceFieldVerifier`) — the only thing that changes is the
generator, so any success-rate gap is attributable to map-conditioning.

| Arm | Generator | Command flag |
|-----|-----------|--------------|
| **A. search-only (paper baseline)** | single-map giant checkpoint, map-blind | *(existing run.py path, no map-cond flags)* |
| **B. multi-map, map-blind** | held-in multi-map checkpoint, map **not** bound | `--use_map_cond` on a checkpoint whose `bind_maze` is left unbound (bind an all-open grid), OR flip the one-line bind call off |
| **C. map-conditional (ours)** | same held-in checkpoint, env map **bound** at inference | `--use_map_cond --map_cond_ckpt <ckpt>` |

Arm C is the treatment; A and B are the controls that isolate *"does feeding the
map in actually help on an unseen map,"* separately from *"does training on more
maps help."* Arm B reuses arm C's checkpoint but withholds the bound grid, so the
only difference between B and C is whether the map is supplied — the map-cond
subclass is bit-identical to the map-blind parent when no map is bound (smoke
test C). If you only have time for two arms, run **A vs C**.

---

## 4. Run the held-out evaluation (cluster, GPU)

Map-conditional arm (C), evaluated on the held-out `giant` OOD variants:

```bash
MUJOCO_GL=egl python run.py \
    --dataset pointmaze-giant-navigate-v0 \
    --method dfs \
    --device cuda \
    --maze_json_dir ../maze_update/maze_variants \
    --task 1 2 3 4 5 \
    --use_map_cond \
    --map_cond_ckpt logs/mapcond/heldout_giant/state_1000000.pt \
    --run_tag mapcond_heldout_giant
```

Map-blind control (B) — identical command **without** the map-cond flags but
pointing `--model_dataset` at the same multi-map checkpoint's map-blind twin, or
arm A using the original single-map giant checkpoint.

`run.py` writes:
* `results_dfs_pointmaze-giant-navigate-v0_<tag>.txt` — success rate + compute
* `results_detailed_<tag>.csv` — per-task metrics (success separated from
  collision / cornercut / deadend / stalled failure modes)

---

## 5. What success looks like

Report success rate (mean over tasks 1–5) for the three arms:

```
arm A (search-only, map-blind giant ckpt) : baseline
arm B (multi-map, map-blind)              : isolates "more maps" effect
arm C (map-conditional, ours)             : isolates "map as input" effect
```

The claim *"making the map an input helps on unseen maps"* is supported if
**C > B** on the held-out map (feeding the map in beats not feeding it in, with
training data held fixed). C ≥ A shows the map-conditional model at least matches
the paper's search-only pipeline on the map it specializes in, while also
generalizing to held-out maps where a single-map checkpoint cannot be trained.

Secondary diagnostics (from the detailed CSV): the map-conditional arm should
lower the **collision_rate** and **deadend_frac** on held-out maps, since the
bound grid tells the generator where the walls are.

---

## 6. Sanity checks before trusting numbers

* **Normalizer parity** — arms B and C must use the shared normalizer baked into
  the checkpoint (loaded automatically by `load_mapcond_diffusion`). A mismatched
  normalizer silently shifts world coordinates and tanks success.
* **Canvas parity** — `bind_env_map` pads the env grid to the checkpoint's
  `canvas_hw` (12×16). Confirm the `[mapcond] task N: bound env map grid (12, 16)`
  log line appears once per task.
* **Bound-map ablation** — set `--use_map_cond` but with a checkpoint whose
  `bind_maze` is cleared (or bind an all-open grid) to confirm success degrades:
  this proves the model is *using* the map, not ignoring it.
