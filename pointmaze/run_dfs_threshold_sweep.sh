#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Three-stage sweep of the new DFS acceptance flags (dfs_threshold_base/shape,
# dfs_trans_threshold) on two held-out giant maze variants, fixed at
# task=1, num_samples=20 for speed.
#
# Usage:
#   ./run_dfs_threshold_sweep.sh 1        # stage 1: sweep dfs_threshold_base in {6,3,2}, shape=current
#   BEST_BASE=3 ./run_dfs_threshold_sweep.sh 2   # stage 2: best base, shape=noise_scaled
#   BEST_BASE=3 BEST_SHAPE=current ./run_dfs_threshold_sweep.sh 3  # stage 3: sweep dfs_trans_threshold
#
# Inspect results_detailed_<run_tag>.csv (or results_dfs_<dataset>_<run_tag>.txt)
# between stages to pick BEST_BASE / BEST_SHAPE before moving on. Each stage
# writes to its own run_tag so results don't mix across stages.

dataset="pointmaze-giant-newvar-navigate-v0"
maze_json_dir="../maze_update/maze_variants"
map_cond_ckpt="logs/mapcond-mazev1-seed2-film/variants_train_giant/state_500000.pt"
task=1
num_samples=20
map_cond_guidance=2
dist_mode="sum"
dist_omega=0.5
corner_radius_frac=0.25
corner_transition_weight=10.0
variant_indices=(7 9)

# stage 2/3 depend on inspecting stage 1's (and stage 2's) results first;
# override via env vars once you know the winner, e.g. BEST_BASE=3 ./run...sh 2
BEST_BASE="${BEST_BASE:-3}"
BEST_SHAPE="${BEST_SHAPE:-current}"

log_dir="logs/run_outputs"
mkdir -p "${log_dir}"

run_variant() {
  local variant_idx="$1"; shift
  local run_tag="$1"; shift
  local version="$1"; shift
  # remaining args ($@) are extra --dfs_* flags appended verbatim
  local timestamp
  timestamp="$(date +%Y%m%d-%H%M%S)"
  local terminal_log="${log_dir}/${run_tag}_v${variant_idx}_${timestamp}.log"
  echo "== ${run_tag} | variant_idx=${variant_idx} | version=${version} | extra: $* =="
  echo "   log: ${terminal_log}"

  MUJOCO_GL=egl python run.py \
    --dataset "${dataset}" \
    --method dfs \
    --maze_json_dir "${maze_json_dir}" \
    --maze_variant_idx "${variant_idx}" \
    --task "${task}" \
    --num_samples "${num_samples}" \
    --use_map_cond --map_cond_ckpt "${map_cond_ckpt}" \
    --map_cond_guidance "${map_cond_guidance}" \
    --use_distance_field --dist_mode "${dist_mode}" --dist_omega "${dist_omega}" \
    --corner_radius_frac "${corner_radius_frac}" --corner_transition_weight "${corner_transition_weight}" \
    --verifier_monitor --verifier_monitor_freq 5 \
    --run_tag "${run_tag}" \
    --version "${version}" \
    --device cuda \
    "$@" \
    2>&1 | tee "${terminal_log}"
}

stage1_threshold_base() {
  echo "### stage 1: dfs_threshold_base in {6, 3, 2}, shape=current ###"
  for variant_idx in "${variant_indices[@]}"; do
    for base in 6 3 2; do
      run_variant "${variant_idx}" "stage1-threshold-base" \
        "base${base}-current-v${variant_idx}" \
        --dfs_threshold_base "${base}" --dfs_threshold_shape current
    done
  done
  echo "### stage 1 done. Inspect results_detailed_stage1-threshold-base.csv, set BEST_BASE, then run stage 2. ###"
}

stage2_noise_scaled() {
  echo "### stage 2: dfs_threshold_base=${BEST_BASE}, shape=noise_scaled (vs. same base, shape=current) ###"
  for variant_idx in "${variant_indices[@]}"; do
    run_variant "${variant_idx}" "stage2-noise-scaled" \
      "base${BEST_BASE}-noise_scaled-v${variant_idx}" \
      --dfs_threshold_base "${BEST_BASE}" --dfs_threshold_shape noise_scaled
  done
  echo "### stage 2 done. Inspect results_detailed_stage2-noise-scaled.csv (compare against stage 1's base=${BEST_BASE} row)."
  echo "### Set BEST_SHAPE accordingly, then run stage 3. ###"
}

stage3_trans_threshold() {
  echo "### stage 3: dfs_threshold_base=${BEST_BASE} shape=${BEST_SHAPE}, dfs_trans_threshold in {unset, 100, 40} ###"
  for variant_idx in "${variant_indices[@]}"; do
    run_variant "${variant_idx}" "stage3-trans-threshold" \
      "base${BEST_BASE}-${BEST_SHAPE}-transNone-v${variant_idx}" \
      --dfs_threshold_base "${BEST_BASE}" --dfs_threshold_shape "${BEST_SHAPE}"

    for trans in 100 40; do
      run_variant "${variant_idx}" "stage3-trans-threshold" \
        "base${BEST_BASE}-${BEST_SHAPE}-trans${trans}-v${variant_idx}" \
        --dfs_threshold_base "${BEST_BASE}" --dfs_threshold_shape "${BEST_SHAPE}" \
        --dfs_trans_threshold "${trans}"
    done
  done
  echo "### stage 3 done. Inspect results_detailed_stage3-trans-threshold.csv. ###"
}

stage="${1:-}"
case "${stage}" in
  1|stage1) stage1_threshold_base ;;
  2|stage2) stage2_noise_scaled ;;
  3|stage3) stage3_trans_threshold ;;
  all)      stage1_threshold_base; stage2_noise_scaled; stage3_trans_threshold ;;
  *)
    echo "usage: $0 {1|2|3|all}"
    echo "  stage 1: sweep dfs_threshold_base in {6,3,2}, shape=current"
    echo "  stage 2: BEST_BASE=<n> $0 2      # sweep shape=noise_scaled vs current at the best base"
    echo "  stage 3: BEST_BASE=<n> BEST_SHAPE=<current|noise_scaled> $0 3   # sweep dfs_trans_threshold"
    exit 1
    ;;
esac
