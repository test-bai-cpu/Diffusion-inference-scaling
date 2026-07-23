#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Best configuration selected by the three-stage DFS threshold sweep:
#   dfs_threshold_base=3
#   dfs_threshold_shape=noise_scaled
#   dfs_trans_threshold=40
#
# This currently runs levels 3 and 4, matching the requested experiment.
# To run every level, change `2 3` below to `0 1 2 3`.

method="dfs"
run_tag="film-w2-omega05-base3-noise-trans40"
log_dir="logs/run_outputs"
mkdir -p "${log_dir}"

for task in 1 2 3 4 5; do
  for level_idx in 2 3; do
    level=$((level_idx + 1))

    for v in 0 1 2; do
      maze_variant_idx=$((level_idx * 3 + v))
      variant=$((v + 1))
      version="${method}-task${task}-level${level}-variant${variant}"
      timestamp="$(date +%Y%m%d-%H%M%S)"
      terminal_log="${log_dir}/${run_tag}_task${task}_level${level}_variant${variant}_${method}_${timestamp}.log"

      echo "Running task ${task}, level ${level}, maze_variant_idx ${maze_variant_idx} with method ${method}"
      echo "Saving terminal output to ${terminal_log}"

      MUJOCO_GL=egl python run.py \
        --dataset pointmaze-giant-newvar-navigate-v0 \
        --method "${method}" \
        --maze_json_dir ../maze_update/maze_variants \
        --maze_variant_idx "${maze_variant_idx}" \
        --version "${version}" \
        --task "${task}" \
        --num_samples 40 \
        --use_map_cond \
        --map_cond_ckpt logs/mapcond-mazev1-seed2-film/variants_train_giant/state_500000.pt \
        --run_tag "${run_tag}" \
        --map_cond_guidance 2 \
        --device cuda \
        --corner_radius_frac 0.25 \
        --corner_transition_weight 10.0 \
        --verifier_monitor \
        --verifier_monitor_freq 1 \
        --use_distance_field \
        --dist_mode sum \
        --dist_omega 0.5 \
        --dfs_threshold_base 3 \
        --dfs_threshold_shape noise_scaled \
        --dfs_trans_threshold 40 \
        2>&1 | tee "${terminal_log}"
    done
  done
done
