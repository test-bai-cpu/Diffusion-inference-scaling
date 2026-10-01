#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# DiT + differentiable wall-collision penalty (mapcond/collision.py).
#
# What this tests: whether the reason the model never produces a legal plan is
# that nothing ever ASKED it to. The diffusion objective has no negative
# examples -- every demo is a legal path on its own map -- so no gradient ever
# says "for this map, that waypoint is inside a wall". plan_quality.py on
# state_1000000.pt: clean% = 0.0% at every guidance setting, best of 64 plans
# still 4-8x above the DFS acceptance threshold.
#
# Warm-started from the finished DiT run, so this is a single-variable change:
# same architecture, same data, same weights at step 0, one added loss term.
# ~200k steps at ~30 it/s is under 2h on an A100, versus 9h from scratch.
#
# Success criterion is NOT training loss -- the added term makes the totals
# incomparable to the baseline run. Judge it with:
#   python plan_quality.py --ckpt logs/mapcond-mazev1-seed2-dit/variants_train_giant/state_1000000.pt \
#                                 logs/mapcond-dit-collision/variants_train_giant/state_200000.pt
# and look at clean% and `best`, then re-run cond_sensitivity.py to see whether
# the closed-cell response improved.
#
# Caveat to keep in mind when reading the result: the penalty is evaluated on
# x0 predicted from NOISED TRAINING trajectories, which are anchored to legal
# demos. It is possible to drive the term to ~0 in-distribution and still
# sample illegal plans on an OOD variant. A drop in the training `coll` column
# is therefore necessary but not sufficient -- only plan_quality.py settles it.

# usage: run_train_dit_collision.sh [mode] [wall_w] [corner_w] [neg_w] [steps]
#   mode      scratch (default) trains from random init, so the result is
#             directly comparable to the 1M-step baseline DiT run. warm starts
#             from that checkpoint instead: ~5x cheaper and a cleaner
#             single-variable attribution, but it inherits whatever basin the
#             baseline settled into.
#   corner_w  should be ~20x wall_w -- the two penalties have natural scales
#             four orders of magnitude apart; see MAPCOND_EXP.md section 1.
#   neg_w     > 0 turns on the map-perturbation negatives (section 5), the part
#             that supplies gradient at low noise. Costs ~2x per step, so
#             expect roughly 15 it/s rather than 30 on an A100.
mode="${1:-scratch}"
wall_w="${2:-1.0}"
corner_w="${3:-20.0}"
neg_w="${4:-1.0}"

case "$mode" in
  scratch) init_args=(); steps="${5:-1000000}" ;;
  warm)    init_args=(--init_from logs/mapcond-mazev1-seed2-dit/variants_train_giant/state_1000000.pt)
           steps="${5:-200000}" ;;
  *) echo "mode must be 'scratch' or 'warm', got '$mode'" >&2; exit 1 ;;
esac

# ogbench sits BESIDE the repo on the dev box and INSIDE it on the cluster, so
# resolve it rather than hardcoding one of the two. A wrong --variant_dir fails
# late, after the dataset has already spent minutes loading.
for cand in ../ogbench/data_gen_scripts ../../ogbench/data_gen_scripts; do
  if [ -d "$cand/newdata_mazev1_seed2" ]; then variant_root="$cand"; break; fi
done
if [ -z "${variant_root:-}" ]; then
  echo "cannot find ogbench/data_gen_scripts/newdata_mazev1_seed2 beside or inside the repo" >&2
  exit 1
fi
variant_dir="$variant_root/newdata_mazev1_seed2"
echo "[run] mode=$mode steps=$steps wall=$wall_w corner=$corner_w neg=$neg_w"
echo "[run] variant_dir=$variant_dir"

python -m mapcond.train_multimap \
  --maps giant --variant_tasks 1 2 3 4 5 --variant_vars 0 1 2 3 4 5 6 7 8 9 10 11 \
  --backbone dit --dit_hidden 256 --dit_heads 8 --dit_depth 4 \
  --dit_patch_size 4 --dit_mlp_ratio 4.0 \
  --pool_size 4 --cfg_dropout 0.1 \
  --collision_weight "${wall_w}" --corner_weight "${corner_w}" \
  --negative_weight "${neg_w}" \
  "${init_args[@]}" \
  --n_train_steps "${steps}" --save_freq 25000 \
  --savepath "logs/mapcond-dit-collision-${mode}-w${wall_w}-c${corner_w}-n${neg_w}/variants_train_giant" \
  --device cuda \
  --variant_dir "${variant_dir}" \
  --variant_json_dir ../maze_update/maze_variants_seed2
