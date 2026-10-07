#!/usr/bin/env bash
#
# Staged-acceptance DFS (--method sdfs).
#
# Generator and verifier settings are copied verbatim from run_adfs.sh, which
# copied them from run_origin_best_config.sh's global-embedding arm
# (global-w2-omega05-base3-noise-trans40), so the ONLY difference from BOTH the
# legacy baseline and the adfs arm is the search method. Do not swap in the FiLM
# checkpoint: the paired A/B in ADAPTIVE_DFS_NOTES.md §1 shows FiLM is
# -1.5/-2.7 pts vs the global embedding, so it would confound the search result.
#
# All sdfs knobs go in the single --sdfs string; see
# search/methods/staged_dfs.py:StagedDFSConfig for the field list. An unknown
# key is a hard error, not a silent no-op.
#
# Env overrides:
#   SDFS="local_max_depth=6,nfe_budget=480" ./run_sdfs.sh
#   TASKS="1 2 3" LEVELS="0 1 2 3" VARIANTS="0 1 2" ./run_sdfs.sh
#
set -euo pipefail

cd "$(dirname "$0")"

method="sdfs"
run_tag="${RUN_TAG:-sdfs-global-w2-omega05}"
ckpt="logs/mapcond-mazev1-seed2-feature/variants_train_giant/state_500000.pt"

# Defaults are the config defaults; this string only turns on logging. All three
# fixes over adfs are DEFAULTS in StagedDFSConfig, not flags set here:
#
#   route_depth=12         (adfs: 'full')  -- legacy landing point j=i-12
#   threshold_shape        now applies to the corner tolerance too
#   local_max_depth=8      (adfs: 4)       -- 27/28 LOCAL repairs hit the old cap
#   route_segment_count    DELETED         -- the cue measured sampler variance
#
# nfe_budget is NOT set here: the config default (480) is legacy parity by
# construction (20 rejections x depth 12 x 2 recur steps), and with
# route_depth=12 that arithmetic is now exact. Run 1 of adfs used 320, which
# afforded 13 re-routes vs legacy's 20, and scored 0/40. See
# ADFS_RUN1_POSTMORTEM.md §5.
sdfs_cfg="${SDFS:-verbose=true}"

# Default sweep: all of level 3 and level 4 (6 cells), same as the adfs sweep,
# so the two are directly pairable against the same legacy file.
#
# Do NOT go back to the single-cell default (LEVELS=2 VARIANTS=1). That cell is
# the HARDEST of 15 in level 3 -- legacy scores 2.5%, the next-hardest 55% --
# and all 40 of its plans tunnel through walls, so it classifies 100% re-route
# and the staged branch never fires. Fisher exact was p=1.0: that cell cannot
# show a difference in either direction.
#
# After this 6-cell run confirms the direction, the full 15-cell sweep
# (LEVELS="0 1 2 3 4") is what settles it -- 6 cells could not reach
# significance on the adfs sweep in either direction.
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
echo "sdfs=${sdfs_cfg}"

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
        --sdfs "${sdfs_cfg}" \
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
# deliberately absent. sdfs does not read them; the equivalents live in the
# --sdfs string as wall_threshold / threshold_shape / corner_threshold, whose
# defaults (3.0 / noise_scaled / 40.0) already match the global-w2 arm. The
# difference from adfs is that threshold_shape now governs BOTH tolerances.
