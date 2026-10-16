#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Full 60-maze rollout evaluation of the baseline DiT (no penalties), on the
# same protocol as run_global_film.sh: 5 tasks x 4 difficulty levels x 3
# variants of maze_update/maze_variants, 40 DFS samples each.
#
# Every argument except the checkpoint is copied from run_global_film.sh, so
# the two results files are directly comparable line by line. Verified against
# the saved args.json of both runs on task1/level3/variant2: 56 of 61 effective
# arguments identical, and the other 5 are the checkpoint, run_tag, version
# (naming only) and verifier_monitor (logging only).
#
# The checkpoint step defaults to 500k to match run_global_film.sh, which
# evaluates globalfilm at state_500000.pt. Compare arms at matched steps rather
# than at "best" -- picking a checkpoint by its score on these same 60 mazes and
# then reporting on them would be selecting on the test set.
#
# usage: run_dit_eval.sh [step] [guidance]
#   step      checkpoint step, default 500000
#   guidance  CFG scale (--map_cond_guidance), default 2, as in run_global_film.sh
#
# Output: results_dfs_pointmaze-giant-newvar-navigate-v0_<run_tag>.txt, one line
# per maze, APPENDED -- rename or delete it before re-running the same tag, or
# the two runs end up interleaved in one file.

step="${1:-1000000}"
guidance="${2:-2}"
ckpt="logs/mapcond-mazev1-seed2-dit/variants_train_giant/state_${step}.pt"

if [ ! -f "${ckpt}" ]; then
  echo "checkpoint not found: ${ckpt}" >&2
  exit 1
fi

method="dfs"
run_tag="dit-baseline-s${step}-w${guidance}"
log_dir="logs/run_outputs"
mkdir -p "${log_dir}"

echo "[eval] ckpt=${ckpt} guidance=${guidance} run_tag=${run_tag}"

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

      MUJOCO_GL=egl python run.py \
        --dataset pointmaze-giant-newvar-navigate-v0 \
        --method "${method}" \
        --maze_json_dir ../maze_update/maze_variants \
        --maze_variant_idx "${maze_variant_idx}" \
        --version "${version}" \
        --task "${task}" \
        --num_samples 40 \
        --use_map_cond \
        --map_cond_ckpt "${ckpt}" \
        --map_cond_guidance "${guidance}" \
        --device cuda \
        --corner_radius_frac 0.25 --corner_transition_weight 10.0 \
        --use_distance_field --dist_mode sum --dist_omega 0.5 \
        --dfs_threshold_base 3 --dfs_threshold_shape noise_scaled --dfs_trans_threshold 40 \
        --verifier_monitor --verifier_monitor_freq 10 \
        --run_tag "${run_tag}" \
        2>&1 | tee "${terminal_log}"
    done
  done
done
