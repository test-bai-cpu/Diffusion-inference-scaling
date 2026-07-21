# Project Summary and Task Brief

## What this project is

Base repo: `Diffusion-inference-scaling` (Zhang et al.) — diffusion planning on
OGBench pointmaze with inference-time search. The paper is in pointmaze/DFS_paper.pdf. And the original code is in https://github.com/XiangchengZhang/Diffusion-inference-scaling.

The basic model is:
A `TemporalUnet` diffusion model
(Diffuser-style: trajectories as 1D sequences, start/goal via inpainting)
generates plans; a search layer (`search/`) scales inference compute: DFS
denoises step by step, scores candidates with verifiers (MazeVerifier: wall
collision), rejects over-threshold candidates by re-noising (budget-
limited, `forced_best` fallback when exhausted).

Our extension: **generalization to unseen mazes** via map conditioning, built
in `pointmaze/mapcond/` (not upstream). Training pools 61 maps (base giant +
60 variants, `mazev1_seed2` family) with one shared `GlobalNormalizer` and a
fixed corner-anchored 12x16 canvas. Evaluation is on a *different* variant
family (`../maze_update/maze_variants`, mazev1 originals) — held-out maps.

And also enrich the verifiers (MazeVerifier: wall collision + corner-zone + diagonal-transition costs; optional DistanceFieldVerifier: BFS distance-to-goal preference, combined by CompositeVerifier)