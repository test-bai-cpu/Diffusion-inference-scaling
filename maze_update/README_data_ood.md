# Data-Distribution OOD Maze Variants

Generate maze variants whose disruption is measured against the **training data
distribution** of an OGBench navigate dataset, not a single optimal path.

## Why this differs from the path-based version

The diffusion planner learns from the demonstrated trajectories (500 episodes in
`pointmaze-giant-navigate-v0`), which show many different routes covering the
whole maze, not one path. So the faithful OOD question is: *how much of the
demonstrated data does a maze edit invalidate, and how hard is it for the data's
own routes to detour around the break?*

A concrete consequence: sealing the goal (path-disruption = 1.0) barely touches
the distribution (it invalidates <1% of transitions), so its **data**-disruption
is low. The two metrics measure different axes.

## The disruption definition

Breaking part of the maze is severe when BOTH conditions hold, matching the
intended notion:

1. **reroute severity** - the demonstrated data would need a very different route
   to get around the break (the data's own connectivity graph shows no easy
   nearby alternative). Computed by removing a cell from the data graph and
   measuring how many extra hops the demonstrated routes between its neighbors
   now require; weighted by how much traffic the cell carried; normalized to
   [0, 1].
2. **collision fraction** - what share of all demonstrated transitions become
   physically invalid (pass through a wall) after the edit.

Plus a small **coverage loss** term (visited cells walled off entirely).

```
data_disruption = 0.50 * reroute_severity
                + 0.35 * collision_fraction
                + 0.15 * coverage_loss          # all in [0,1], capped at 1
```

reroute_severity dominates, so "you forced the data onto a very different route"
drives the score, exactly the intended characteristic, with collision fraction
folding in "how much of the data actually collides."

## Level targeting with randomness

`generate_variants_at_level(model, start, goal, target, n_variants, seed=...)`
returns several DISTINCT mazes that all score near `target`. Randomness comes
from the seed and per-attempt random edits, so the same disruption level yields
different variants. `generate_suite` sweeps levels and collects `variants_per_level`
each.

## Reachable range (data-distribution ceiling)

On giant/task3, reachable data-disruption tops out around ~0.7. Beyond that, the
only way to invalidate enough of the distribution also disconnects the goal, so
0.8 generally yields unreachable mazes only. This mirrors the path-based ceiling
and is a real structural property of how spread-out the demonstrated data is.

## Files

- `generate_data_ood.py` - generator + scorer + plot (imports `data_disruption`
  and `generate_ood_mazes`).
- `data_disruption.py` - loads the dataset, builds the traffic model.
- `{maze}_{task}_data_variants.json` - per-variant records. Each has:
  - `maze_map` (ready to inject into `maze.py`)
  - `target_level`, `variant_index`, `data_disruption`
  - `components`: `reroute_severity`, `collision_fraction`, `coverage_loss`
  - `goal_reachable`, `broken_cells`, `start`/`goal`, `start_xy`/`goal_xy`
- `{maze}_{task}_data_variants.png` - one row per level, one column per variant;
  red squares = broken visited cells (opacity ~ traffic), blue + = opened walls.

## Usage

```python
import generate_data_ood as g

DATA = "path/to/pointmaze-giant-navigate-v0"   # dir with observations.npy etc.
recs, model = g.generate_suite(
    DATA, maze_type="giant", task="task3",
    levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    variants_per_level=3, seed=2,
)
g.plot_suite(recs, model, "out.png", "giant", "task3")

# score an arbitrary edited maze against the data:
import numpy as np
sc = g.score(model, np.array(some_maze_map))
print(sc.data_disruption, sc.reroute_severity, sc.collision_fraction)
```

To run on a different dataset/maze, point `DATA` at that dataset directory and
set `maze_type` accordingly (medium / large / giant / ultra). The dataset must
match the maze (same map the data was collected on).

## Plugging variants into the diffusion eval

Identical to the path-based version: each `maze_map` is in `maze.py` format. Set
it as the env's `maze_map` before the MuJoCo XML is built, keep the same task
(start/goal/conditioning fixed), and roll out `plan_pointmaze.py`. Use
`data_disruption` as the x-axis for "model performance vs. how much of the
training distribution the edit invalidated."
