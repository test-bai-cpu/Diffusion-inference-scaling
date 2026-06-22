# Reroute Disruption Index (RDI) — converged maze-variant generator

This is the converged version of the two earlier iterations. It keeps what you
liked from the **first** iteration (the per-cell reroute-severity idea and the
plot layout) and the aligned-coordinate drawing from the **second**, and it
**drops** the second iteration's blind-planner / traffic-only pivot — because in
this setup the planner *does* see the map and replans, so rerouting is the right
axis.

## The metric: Reroute Disruption Index (RDI)

For one task `(start, goal)` we ask a planner for up to **K diverse candidate
routes** on the *original* map (not just the optimal one). An edit **breaks** a
route if any of its cells becomes a wall. RDI grades disruption by three things:

| ingredient | meaning | your phrasing |
|---|---|---|
| `break_fraction` | how many of the K routes the edit breaks | "3 routes, break 1 → 0.3, break 2 → 0.6, break 3 → 0.9" |
| `mean_detour_severity` | how far the forced reroute departs *spatially* from the broken route | "totally different reroute → serious, even if same length" |
| `traffic_weighted_break` | share of demonstrated traffic on the broken routes | "compute traffic of the training data, use it as the basis" |

```
RDI = w_break * break_fraction
    + w_detour * mean_detour_severity
    + w_traffic * traffic_weighted_break        # all in [0,1], capped at 1
```

A **disconnected goal saturates RDI to 1.0**.

### Detour severity is *spatial displacement*, not length

`mean_detour_severity` measures how far the planner's forced reroute departs
**spatially** from the route that broke, using a symmetric mean-nearest-cell
(Hausdorff-style) distance. This is deliberately **insensitive to length**: a
reroute through a completely different corridor of the *same* length still scores
high, because the planner abandoned the original route's territory. (An earlier
version used `extra_hops ÷ length`, which scored a same-length reroute as ~0
disruption — wrong for a map-aware planner.) Each broken route is compared
against **itself** (the specific route that broke), not the surviving set.

Displacement is normalized by `DISPLACEMENT_SCALE * (H + W)`. Smaller scale =
harsher (small reroutes already look severe) and raises the achievable RDI
ceiling; larger = softer. Default `0.20`. On giant/task1 with the uniform stub,
scale 0.33 caps RDI ~0.63, 0.20 ~0.77, 0.12 ~0.89 — so tune this if a target
level can't be hit.

## Two halves: disruption AND novelty (added)

The reachable RDI ceiling from breaking alone is ~0.7 on giant (the edits that
would break routes harder also disconnect the goal and get reverted). To reach
higher, and to honor "I can redesign, open something, to make it very
different", RDI now has TWO halves:

- **disruption_term** [0,1]  the ORIGINAL routes break and the planner is forced
  onto a displaced reroute. Blend of `break_fraction`, `mean_detour_severity`,
  `traffic_weighted_break` (weights in `DISRUPTION_WEIGHTS`). "You destroyed
  familiar structure."
- **novelty_term** [0,1] = `route_novelty`  the EDITED map's own candidate routes
  sit far from every original route. Rises when REDESIGN opens corridors the
  planner now prefers, EVEN IF no original route was walled. "You introduced
  unfamiliar structure." Without this, opening a dramatic new shortcut scored 0.

They combine with a soft-OR, not a weighted average, so EITHER half can drive RDI
toward 1 and both together exceed either alone:

```
RDI = 0.5 * max(disruption, novelty)
    + 0.5 * (1 - (1 - disruption) * (1 - novelty))
```

The generator's `_random_edit` now opens MORE corridors at higher targets
(scaled by `target`), so high levels are reached by genuinely transforming the
map. A target of 0.8 is now achievable (rare, so raise `max_attempts` or widen
`tol` for the very top band). `DISPLACEMENT_SCALE` (default 0.14) still sets how
harsh displacement/novelty are and thus the achievable ceiling.

Per-variant fields now also include `disruption_term`, `novelty_term`,
`route_novelty`, and `n_opened` (walls opened). Plot titles show
`broke N/M  det X  nov Y (+Zw)`.

### Adaptive weighting for low route counts

Several PointMaze tasks have a single-exit start/goal, so the structurally
distinct route count is small (1–3). With few routes, `break_fraction` is coarse
(e.g. only 0 / 0.5 / 1.0 for two routes) and cannot grade a smooth sweep. So the
weight **shifts from `break_fraction` toward the continuous `mean_detour_severity`
as the route count shrinks**: with ~5 routes `break_fraction` leads (your "1 of 3"
intuition); with one route, detour hardness carries the grade. This makes every
task sweepable and resolves the single-exit cliff the second iteration hit,
without abandoning the reroute framing.

Rename the metric in one place: `METRIC_NAME` and the `rdi` field in
`reroute_disruption.py`.

## What "redesign" does

`_random_edit(..., redesign=True)` may **open interior walls** (carve new
corridors) while it closes old ones, biased toward low-traffic walls so it adds
genuinely new structure instead of reopening the busy highway. This is your
"block one way, open a different new way than before" — visible in the plot as
blue `+` markers, with orange (new) routes taking corridors the blue (original)
routes never used.

## Plot (first-iteration layout, retained)

`plot_suite` draws one row per target level:

- **col 0** original maze + its K diverse routes (blue)
- **col 1** training-traffic heatmap (log scale)
- **cols 2+** each variant: walls, broken cells (red squares, opacity ∝ traffic),
  opened walls (blue `+`), original routes (faint blue), planner's new routes
  (orange). Title shows `RDI`, `broke N/M`, `detour`, reachability.

All drawing is in physical `xy` coordinates (`ij_to_xy`) so walls, routes and the
heatmap share one coordinate frame.

## Usage

```python
import reroute_disruption as r
r.set_seed(1)

DATA = "path/to/pointmaze-giant-navigate-v0"   # dir with observations.npy, terminals.npy
                                               # omit / None -> uniform-traffic stub (route-only)

recs, model = r.generate_suite(
    maze_type="giant", task="task5", data_dir=DATA,
    levels=[0.3, 0.6, 0.9], variants_per_level=3,
    redesign=True, k=5,
)
r.plot_suite(recs, model, "out.png", "giant", "task5")

# score one arbitrary edited maze for a task:
import numpy as np
from maze_utils import MAZE_TASKS
s, g = MAZE_TASKS["giant"]["task5"]
sc = r.score_rdi(model, np.array(some_maze_map), s, g)
print(sc.rdi, sc.n_broken, sc.n_routes, sc.mean_detour_severity)
```

Run all five tasks by looping `task` over `MAZE_TASKS["giant"]`.

## Notes on achievable range

Some RDI levels may be hard to hit on a given task because the route structure
only admits certain break counts. The generator prints how many variants it
found per level; if a level under-fills, either widen `tol`, raise
`max_attempts`, or accept the achievable bands. (If you ever want the
"measure-don't-target" scan workflow from iteration 2, it slots on top of
`score_rdi` unchanged — but targeting is the primary mode here, as you preferred.)

## Files

- `reroute_disruption.py` — the converged generator + RDI scorer + plot.
- `data_disruption.py` — unchanged; loads the OGBench dataset, builds the traffic
  model (reused for the traffic-weighting term).
- `maze_utils.py` — unchanged; maze maps, tasks, `ij_to_xy`, BFS.


## Forcing transformation, not just elimination (added)

High RDI used to be reachable by pure ELIMINATION: wall N-1 of the candidate
routes and leave one ORIGINAL route untouched. The planner then drives a path it
already knew, so nothing about its traversed environment is new (survivor overlap
~1.0, novelty ~0). To fix this:

- **Generator (`_random_edit`, survivor-displacement step):** at higher targets,
  after the initial edits it inspects the planner's current optimal route; while
  that route is still ~identical to an original, it OPENS a new corridor near the
  route and BLOCKS a distinctive cell of the original, pushing the planner onto
  the fresh passage. (Blocking alone cannot help here: on these mazes every
  block-only route IS one of the diverse originals, so blocking just hops the
  survivor to the next original. Only redesign creates a genuinely different
  route.) Pass `base_routes=` so the step knows the originals.
- **Scorer:** the novelty term now also includes **survivor novelty** (how far
  the planner's chosen route sits from the closest original), and the disruption
  term is **gated** by novelty: `gated_disruption = disruption * (0.7 + 0.3*novelty)`.
  Pure elimination (novelty 0) keeps 70% of its value; displacing the survivor
  restores the rest. So transformation matters as much as elimination.

**Consequence on achievable range:** requiring real transformation lowers the
reachable ceiling (giant tops out ~0.68, because the maze geometry only permits
modest survivor displacement). That is honest, not a bug: target levels should be
set within the achievable range (e.g. up to ~0.65 on giant). Bigger/sparser mazes
allow higher.
