#!/usr/bin/env bash
# Same sweep as run_origin.sh but with BFS distance-field guidance enabled.

for task in 1 2 3 4 5; do
  for level_idx in 0 1 2 3; do
    level=$((level_idx + 1))
    for v in 0 1 2; do
      variant_idx=$((level_idx * 3 + v))
      method="dfs"
      echo "Running task ${task}, level ${level}, variant_idx ${variant_idx} with method ${method} + distance-field guidance"

      MUJOCO_GL=egl python run.py \
        --dataset pointmaze-giant-newvar-navigate-v0 \
        --method "${method}" \
        --maze_json_dir ../maze_update/maze_variants \
        --maze_variant_idx "${variant_idx}" \
        --version "${method}-df-task${task}-level${level}-variant$((v+1))" \
        --task "${task}" \
        --use_distance_field \
        --dist_omega 0.1 \
        --dist_mode sum \
        --dist_smooth_sigma 0.5 \
        --dist_connectivity 4
    done
  done
done
