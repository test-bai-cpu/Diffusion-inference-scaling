#!/usr/bin/env bash
#
# Adaptive-backtracking DFS (--method adfs).
#
# Generator and verifier settings are copied verbatim from
# run_origin_best_config.sh's global-embedding arm
# (global-w2-omega05-base3-noise-trans40) so the ONLY difference is the
# search method. Do not swap in the FiLM checkpoint: the paired A/B in
# ADAPTIVE_DFS_NOTES.md §1 shows FiLM is -1.5/-2.7 pts vs the global
# embedding, so using it would confound the search result.
#
# All adfs knobs go in the single --adfs string; see
# search/methods/adaptive_dfs.py:AdaptiveDFSConfig for the field list.
# An unknown key is a hard error, not a silent no-op.
#
# Env overrides:
#   ADFS="local_max_depth=6,nfe_budget=480" ./run_adfs.sh
#   TASKS="1 2 3" LEVELS="0 1 2 3" VARIANTS="0 1 2" ./run_adfs.sh
#
set -euo pipefail

cd "$(dirname "$0")"

method="adfs"
run_tag="${RUN_TAG:-adfs-global-w2-omega05}"
ckpt="logs/mapcond-mazev1-seed2-feature/variants_train_giant/state_500000.pt"

# Default policy. route_depth=full keeps the legacy restart for genuine
# re-routes; local_max_depth=4 is what stops a corner graze from being thrown
# back to i=0..3 where sqrt(alpha_bar) is 0.000-0.306.
#
# nfe_budget is NOT set here: the config default (480) is legacy parity by
# construction (20 rejections x depth 12 x 2 recur steps). Run 1 used 320,
# which affords only 13 re-routes vs legacy's 20 -- a 35% cut in search on
# all-re-route cells -- and scored 0/40. See ADFS_RUN1_POSTMORTEM.md §5.
adfs_cfg="${ADFS:-route_depth=full,local_max_depth=4,dead_end_depth=8,escalate_after=3,verbose=true}"

# Default sweep: all of level 3 and level 4 (6 cells).
#
# Do NOT go back to the single-cell default (LEVELS=2 VARIANTS=1). That cell is
# the HARDEST of 15 in level 3 -- legacy scores 2.5% there, the next-hardest
# scores 55% -- and all 40 of its plans tunnel through walls, so it classifies
# 100% re-route and the adaptive branch never fires. Fisher exact on run 1 was
# p=1.0: that cell cannot show a difference in either direction.
TASKS="${TASKS:-1}"
LEVELS="${LEVELS:-2 3}"
VARIANTS="${VARIANTS:-0 1 2}"

log_dir="logs/run_outputs"
mkdir -p "${log_dir}"

if [[ ! -f "${ckpt}" ]]; then
  echo "checkpoint not found: ${ckpt}" >&2
  exit 1
fi

echo "method=${method}  run_tag=${run_tag}"
echo "adfs=${adfs_cfg}"

for task in ${TASKS}; do
  for level_idx in ${LEVELS}; do
    level=$((level_idx + 1))

    for v in ${VARIANTS}; do
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
        --run_tag "${run_tag}" \
        --adfs "${adfs_cfg}" \
        --map_cond_guidance 2 \
        --device cuda \
        --corner_radius_frac 0.25 \
        --corner_transition_weight 10.0 \
        --verifier_monitor \
        --verifier_monitor_freq 10 \
        --use_distance_field \
        --dist_mode sum \
        --dist_omega 0.5 \
        2>&1 | tee "${terminal_log}"
    done
  done
done

# Note: --dfs_threshold_base / --dfs_threshold_shape / --dfs_trans_threshold are
# deliberately absent. adfs does not read them; the equivalents live in the
# --adfs string as wall_threshold / threshold_shape / corner_threshold, whose
# defaults (3.0 / noise_scaled / 40.0) already match the global-w2 arm.
