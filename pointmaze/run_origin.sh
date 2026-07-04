#!/usr/bin/env bash

# MUJOCO_GL=egl python run.py --dataset pointmaze-giant-navigate-v0 --method dfs --version all

for task in 1 2 3 4 5; do
  for level_idx in 0 1 2 3; do
    level=$((level_idx + 1))
    for v in 0 1 2; do
      maze_variant_idx=$((level_idx * 3 + v))
      method="dfs"
      echo "Running task ${task}, level ${level}, maze_variant_idx ${maze_variant_idx} with method ${method}"

      MUJOCO_GL=egl python run.py \
        --dataset pointmaze-giant-newvar-navigate-v0 \
        --method "${method}" \
        --maze_json_dir ../maze_update/maze_variants_v2 \
        --maze_variant_idx "${maze_variant_idx}" \
        --version "mazev2_${method}-task${task}-level${level}-variant$((v+1))" \
        --task "${task}" \
        --run_tag mazev2
    done
  done
done