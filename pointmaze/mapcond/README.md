# `mapcond` — map-conditional diffusion planning

Extends the single-map pointmaze diffusion planner so that
**one** model plans on **many** maps, with the maze grid supplied as an
**input**. The original pipeline trains, denoises, and searches on a single fixed
map; its only out-of-distribution support is inference-time search steering on a
map-*blind* generator. This package makes the map a first-class conditioning
signal so the generator itself knows where the walls are — the prerequisite for
planning on a map it was never trained on.

**Design in one line:** a small CNN encodes the maze grid to a vector that is
**added to the diffusion time embedding** (global FiLM-style conditioning), and
the map is bound once per episode at inference so no network-call site in the
search code needs to change.

Everything here is **additive**: original files are untouched, new behavior lives
in subclasses, and every map argument is optional (`None` ⇒ bit-identical to the
original model). The map-blind pipeline keeps working exactly as before.

---

## Key decisions (locked)

| Choice | Decision | Why |
|---|---|---|
| Maps | Base OGBench maps `medium`, `large`, `giant`, `teleport`, plus optional JSON-backed giant task/variant maps | real datasets on disk, mujoco-free to load |
| Conditioning | **Global** map embedding added to time embedding (not spatial channels) | size-agnostic, cheap, one injection point |
| Normalizer | **One** shared normalizer across all maps | a map-conditional model must live in a single world frame |
| Padding | Corner-anchored to a fixed 12×16 canvas (giant-defined) | every trained map fits; consistent grid coordinates |
| Backward compat | Subclasses; `maze=None` reproduces the parent exactly | existing single-map runs unaffected |

---

## Package layout

```
mapcond/
├── maze_grids.py        # the 5 maze grids + canvas padding (pad_to_canvas, cell_to_xy)
├── data_utils.py        # mujoco-free loaders for ogbench/data/*.npy + giant variant .npz
├── global_normalizer.py # shared LimitsNormalizer fit across all train maps
├── global_normalizer.pkl# the fitted normalizer (baked into every checkpoint too)
├── map_encoder.py       # MapEncoder: grid [H,W] -> vector [dim] (~31k params, size-agnostic)
├── dataset.py           # MultiMazeGoalDataset + map_batch_collate (yields traj+cond+maze)
├── models.py            # MapConditional{TemporalUnet,GaussianDiffusion} subclasses
├── train_multimap.py    # mujoco-free multi-map training loop (EMA, checkpoints)
├── run_multimap.slurm   # SLURM launcher for the bhlg cluster
├── inference.py         # load checkpoint + bind env map into the generator
├── tests/smoke_test.py  # 4 CPU checks (shape/grad, map-swap, None-parity, overfit)
├── tests/variant_smoke_test.py # CPU checks for base+giant-variant data flow
├── artifacts/          # generated smoke reports
├── eval_heldout.md      # held-out-map success-rate evaluation protocol
└── README.md            # this file
```

---

## The model (`models.py`)

Two subclasses, diffuser core untouched:

* **`MapConditionalTemporalUnet(TemporalUnet)`**
  Adds `map_encoder` (CNN grid→vector) and `map_mlp`. In `forward(x, cond, time,
  maze=None)` the time embedding becomes `t = time_mlp(time) + map_mlp(map_encoder(maze))`.
  * `maze=None` **and** no bound map ⇒ identical to parent (structural skip).
  * `bind_maze(grid)` stashes a grid so the search code's plain
    `unet(x, cond, t)` calls automatically get the map — no kwarg threading.
  * Last map layer initialized `std=0.02` (not zero) so gradient reaches the
    encoder from step 0; `zero_init_map=True` opts into ControlNet-style zero init.

* **`MapConditionalGaussianDiffusion(GaussianDiffusion)`**
  Threads `maze=None` through `p_losses`/`loss`/`p_mean_variance`/`p_sample`/
  `p_sample_loop`/`conditional_sample`/`forward`. `maze=None` reproduces the parent.

**Smoke tests** (`python -m mapcond.tests.smoke_test` from `pointmaze/`, ~48 s CPU):

| Check | Result |
|---|---|
| A. forward/backward shape + grad into MapEncoder | PASS (grad norm 2.7e-4) |
| B. swapping the maze changes the predicted x₀ | PASS (Δ 8.3e-3) |
| C. `maze=None` **bit-identical** to plain `TemporalUnet` | PASS (Δ 0.0) |
| D. overfit one batch | PASS (loss 0.22→0.034, 6.5×) |

---

## Training (`train_multimap.py`) — mujoco-free

Base-map training remains unchanged:

```bash
cd pointmaze
python -m mapcond.train_multimap --maps medium large giant teleport \
    --savepath logs/mapcond/run1
# quick CPU check:
python -m mapcond.train_multimap --smoke
```

To train on the new giant layout variants, add the task/variant selectors:

```bash
python -m mapcond.train_multimap \
    --maps medium large giant teleport \
    --variant_tasks 1 2 3 4 5 \
    --variant_vars 0 1 2 3 4 5 6 7 8 9 10 11 \
    --variant_dir /home/yufei/research/diffusion/ogbench/data_gen_scripts/newdata_mazev2 \
    --variant_json_dir ../maze_update/maze_variants_v2 \
    --savepath logs/mapcond/base_plus_giant_variants
```

`--variant_vars` defaults to `0..11` whenever `--variant_tasks` is set, so the
full requested set can be written more compactly as `--variant_tasks 1 2 3 4 5`.
Each variant map is identified as `giant_task{task}_var{var}` and contributes
windows from its own `.npz` trajectory file while carrying its own 12x16 JSON
grid.

Hyperparameters match the repo (horizon 256, 256 diffusion steps, dim 32,
dim_mults [1,4,8], lr 2e-4, batch 32, grad-accum 2, L2 loss, EMA decay 0.995 with
2000-step warmup). Each `state_{step}.pt` bundles **model + EMA + pickled shared
normalizer + geometry config** (`maps/map_ids/map_specs, horizon,
n_diffusion_steps, dim, dim_mults, canvas_hw, pad_anchor,
transition/observation/action dims`) so inference is fully self-describing.

The shared normalizer is always fit over the exact selected training sources:
base maps only for the old path, or base maps union selected variants when
variant specs are supplied. The package-level `global_normalizer.pkl` has also
been refreshed over `medium large giant teleport` plus selected variant maps
(`observations` max includes y=37.318), but checkpoints are the authoritative
artifact because they embed the normalizer used for that run.

On the cluster:

```bash
sbatch mapcond/run_multimap.slurm   # bhlg paths, MUJOCO_GL=egl, env `maze`
```

Verified end-to-end mujoco-free: 4 maps → 11,940 windows, loss 0.60→0.21.

### Giant variant data source

Training trajectories live outside this repo at:

```
/home/yufei/research/diffusion/ogbench/data_gen_scripts/newdata/
  pointmaze-giant-navigate-v0-task{1..4}-var{0..11}.npz
```

The current mazev2 launcher uses:

```
/home/yufei/research/diffusion/ogbench/data_gen_scripts/newdata_mazev2/
  pointmaze-giant-navigate-v0-task{1..5}-var{0..11}.npz
```

Each file stores `observations`, `actions`, and `terminals`; `qpos/qvel` are
ignored. Per-variant grids come from:

```
maze_update/maze_variants/giant_task{1..4}.json
  variants[var_index]["maze_map"]
```

For mazev2 runs this is:

```
maze_update/maze_variants_v2/giant_task{1..5}.json
  variants[var_index]["maze_map"]
```

The `.npz` loader segments episodes using the same terminal-flag logic as the
base `.npy` loader. The 12x16 variant grids already match the training canvas,
so the existing corner-anchored padding path is unchanged.

Variant smoke checks:

```bash
cd pointmaze
python -m mapcond.tests.variant_smoke_test
```

The generated report is written to
`mapcond/artifacts/variant_smoke_report.md`.

---

## Inference — map threaded through the existing search (opt-in)

The map is **constant within an episode**, so instead of editing every network
call site in the DFS/BFS search, we **bind the map once per task** and let the
subclass inject it. Turning it on:

```bash
MUJOCO_GL=egl python run.py \
    --dataset pointmaze-giant-navigate-v0 --method dfs --device cuda \
    --use_map_cond \
    --map_cond_ckpt logs/mapcond/run1/state_1000000.pt
```

New `Arguments` fields (all default off — omitting them = original behavior):

| Flag | Default | Meaning |
|---|---|---|
| `--use_map_cond` | `False` | swap in the map-conditional generator + its shared normalizer |
| `--map_cond_ckpt` | `''` | path to a `mapcond` checkpoint (required when `use_map_cond`) |
| `--map_cond_use_ema` | `True` | load EMA weights (recommended) vs raw model weights |

What happens under the hood (`inference.py` + two hooks in
`search/base_pipeline.py`):

1. `setup()` — when `use_map_cond`, `load_mapcond_diffusion` rebuilds the
   map-conditional diffusion from the checkpoint config, loads EMA/model weights,
   restores the **shared normalizer**, and replaces the generator the `Policy`
   uses. (Map-blind path is unchanged.)
2. `experiment()` — for each task, after `guidance.update_env(env)`,
   `bind_env_map(unet, env)` reads `env.maze_map`, pads it to the checkpoint's
   `canvas_hw` (12×16), and binds it. Every subsequent denoising call in that
   episode sees the map with **zero changes to the search/verifier code**.

The verifiers (`MazeVerifier` wall-collision gradient, `DistanceFieldVerifier`
BFS on the env grid) stay active and untouched — map-conditioning is orthogonal
to and stacks with search steering.

Verified on the 250-step demo checkpoint: binding the giant grid yields canvas
(12,16); bound-vs-blind ε differ (0.30); medium-vs-giant bound ε differ (0.59) —
the map signal flows through.

---

## Evaluating map generalization

See **`eval_heldout.md`** for the held-out-map protocol: train on
`medium`+`large`+`teleport`, evaluate OGBench rollout **success rate** on
held-out `giant`, comparing three arms (search-only baseline / multi-map
map-blind / map-conditional) so the effect of *feeding the map in* is isolated
from the effect of *training on more maps*. Run on the GPU/SLURM cluster
(evaluation needs the simulator; training does not).

---

## Backward compatibility guarantee

* Original files unmodified except **additive** hooks (three opt-in `Arguments`
  fields; one `if use_map_cond:` block in `setup`; one `bind_env_map` call in
  `experiment`; three CLI pass-through lines in `run.py`).
* Without `--use_map_cond`, `run.py` behaves exactly as before.
* `maze=None` makes the subclasses bit-identical to their parents (smoke test C).
