#!/usr/bin/env bash
# Run maze-variant inference-time scaling experiments.
#
# For each difficulty level (easy / medium / hard) and each variant, both
# 'dfs' and 'hard-dfs' are evaluated using the pretrained giant-maze model.
# Results land under:
#   results/maze_variants/{difficulty}/variant_{N}/{method}/
#
# Usage:
#   bash run_maze_variants.sh [--device cuda] [--n_variants 3] [--base_seed 100]
#
# The script honours the MUJOCO_GL env var if already set; otherwise it
# defaults to egl (headless rendering).

# MUJOCO_GL=egl python run.py --dataset pointmaze-giant-navigate-v0 --method dfs

MUJOCO_GL=egl python run.py --dataset pointmaze-giant-navigate-v0 --method dfs --version all

MUJOCO_GL=egl python run.py \
  --dataset pointmaze-giant-newvar-navigate-v0 \
  --method dfs \
  --maze_json_dir ../maze_update/maze_variants \
  --maze_variant_idx 0
  --version level0.1