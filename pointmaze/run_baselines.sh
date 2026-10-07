#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Mirrors the scope of run_origin_best_config.sh (the confirmed main-method
# config for this comparison): OOD giant-newvar variants, levels 3-4 only
# (maze_variant_idx 6-11), tasks 1-5, num_samples=40. Runs both non-diffusion
# baselines (bfs-greedy, bfs-cspace) so their rows land in the same
# results_detailed_<run_tag>.csv the main method already populates.

run_tag="baseline"
log_dir="logs/run_outputs"
mkdir -p "${log_dir}"

for method in bfs-greedy bfs-cspace; do
  for task in 1 2 3 4 5; do
    for level_idx in 0 1 2 3; do
      level=$((level_idx + 1))

      for v in 0 1 2; do
        maze_variant_idx=$((level_idx * 3 + v))
        variant=$((v + 1))
        version="${method}-task${task}-level${level}-variant${variant}"
        timestamp="$(date +%Y%m%d-%H%M%S)"
        terminal_log="${log_dir}/${run_tag}_task${task}_level${level}_variant${variant}_${method}_${timestamp}.log"

        echo "Running task ${task}, level ${level}, maze_variant_idx ${maze_variant_idx} with method ${method}"
        echo "Saving terminal output to ${terminal_log}"

        MUJOCO_GL=egl python run_baselines.py \
          --method "${method}" \
          --dataset pointmaze-giant-newvar-navigate-v0 \
          --maze_json_dir ../maze_update/maze_variants \
          --maze_variant_idx "${maze_variant_idx}" \
          --version "${version}" \
          --task "${task}" \
          --num_samples 40 \
          --run_tag "${run_tag}" \
          --device cuda \
          2>&1 | tee "${terminal_log}"
      done
    done
  done
done
