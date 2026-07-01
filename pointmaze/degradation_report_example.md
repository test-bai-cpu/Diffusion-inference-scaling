# Graceful-degradation evaluation report

Source: `eval_results_demo.csv`  ·  240 runs  ·  methods: dfs, field, mafgs, mafgs+mcgn

## Monotonicity (calibrated difficulty vs success)

Spearman rho between difficulty level and success rate (negative = harder levels solved less often = well-ordered ladder):

- **dfs**: rho = -0.939
- **field**: rho = -0.941
- **mafgs**: rho = -0.923
- **mafgs+mcgn**: rho = -0.914

## Aggregate performance (mean over tasks/levels/seeds)

| method | success % | collision % | corner-cut % | dead-end % | steps | compute |
|---|---|---|---|---|---|---|
| dfs | 57.9 | 11.3 | 2.9 | 11.7 | 325 | 38.0 |
| field | 65.8 | 9.3 | 3.0 | 8.6 | 325 | 41.0 |
| mafgs | 76.0 | 3.7 | 1.9 | 4.1 | 325 | 52.0 |
| mafgs+mcgn | 81.1 | 3.0 | 1.8 | 3.4 | 325 | 55.0 |

## Margin at hardest level (L3)

- **field** − dfs: +14.6 pp (42.2% vs 27.7%)
- **mafgs** − dfs: +29.6 pp (57.3% vs 27.7%)
- **mafgs+mcgn** − dfs: +40.3 pp (68.0% vs 27.7%)

## Map transfer (MCGN side model)

The mafgs+mcgn config adds the Map-Conditioned Guidance Network, which conditions only on a local occupancy crop + relative-goal vector (never a map id or absolute coordinates). Trained on tasks {1,2,4,5} variant layouts with D4 augmentation and evaluated on the held-out task-3 layouts (unseen maps + unseen goal), it predicts the geodesic descent direction at **holdout cosine 0.46 vs 0.33** for a goal-direction baseline; on must-detour cells (where heading straight at the goal points into a wall) it swings from **−0.22 to +0.22**, i.e. it learns to turn around obstacles on maps it never saw. See `sidemodel/mcgn_transfer.png`.
