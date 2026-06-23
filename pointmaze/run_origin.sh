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

# MUJOCO_GL=egl python run.py --dataset pointmaze-giant-navigate-v0 --method dfs --version all

for task in 1 2 3 4 5; do
  for level_idx in 0 1 2 3; do
    level=$((level_idx + 1))
    for v in 0 1 2; do
      variant_idx=$((level_idx * 3 + v))
      method="dfs"
      echo "Running task ${task}, level ${level}, variant_idx ${variant_idx} with method ${method}"

      MUJOCO_GL=egl python run.py \
        --dataset pointmaze-giant-newvar-navigate-v0 \
        --method "${method}" \
        --maze_json_dir ../maze_update/maze_variants \
        --maze_variant_idx "${variant_idx}" \
        --version "${method}-task${task}-level${level}-variant$((v+1))" \
        --task "${task}"
    done
  done
done

# for task in 1 2 3 4 5; do
#   case "${task}" in
#     1) choices=(3 2 2 2) ;;
#     2) choices=(2 1 1 1) ;;
#     3) choices=(1 1 2 3) ;;
#     4) choices=(2 2 3 1) ;;
#     5) choices=(3 1 3 1) ;;
#   esac

#   for level_idx in "${!choices[@]}"; do
#     level=$((level_idx + 1))
#     choice="${choices[$level_idx]}"
#     variant_idx=$((level_idx * 3 + choice - 1))
#     method="dfs"
#     echo "Running task ${task}, level ${level}, variant ${variant_idx} with method ${method}"

#     MUJOCO_GL=egl python run.py \
#       --dataset pointmaze-giant-newvar-navigate-v0 \
#       --method "${method}" \
#       --maze_json_dir ../maze_update/maze_variants \
#       --maze_variant_idx "${variant_idx}" \
#       --version "${method}-task${task}-level${level}-variant${level_idx+1}" \
#       --task "${task}"
#   done
# done