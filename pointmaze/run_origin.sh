#!/usr/bin/env bash

# MUJOCO_GL=egl python run.py --dataset pointmaze-giant-navigate-v0 --method dfs --version all

for task in 1 2 3 4 5; do
  for level_idx in 2 3; do
    level=$((level_idx + 1))
    for v in 0 1 2; do
    maze_variant_idx=$((level_idx * 3 + v))
    method="dfs"
    run_tag="film-w2-omega-05"
    echo "Running task ${task}, level ${level}, maze_variant_idx ${maze_variant_idx} with method ${method}"

    log_dir="logs/run_outputs"
    mkdir -p "${log_dir}"
    timestamp="$(date +%Y%m%d-%H%M%S)"
    terminal_log="${log_dir}/${run_tag}_${task}_level${level}_variant$((v+1))_${method}_${timestamp}.log"
    echo "Saving terminal output to ${terminal_log}"

    MUJOCO_GL=egl python run.py \
      --dataset pointmaze-giant-newvar-navigate-v0 \
      --method "${method}" \
      --maze_json_dir ../maze_update/maze_variants \
      --maze_variant_idx "${maze_variant_idx}" \
      --version "${method}-task${task}-level${level}-variant$((v+1))" \
      --task "${task}" \
      --num_samples 40 \
      --use_map_cond \
      --map_cond_ckpt logs/mapcond-mazev1-seed2-film/variants_train_giant/state_200000.pt \
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
      2>&1 | tee "${terminal_log}"

    done
  done
done
