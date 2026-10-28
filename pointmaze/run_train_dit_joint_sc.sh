#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Joint-attention DiT (run_train_dit_joint.sh) + self-conditioning: the model
# also gets its own earlier x0 estimate (a draft) and reads trajectory-token
# positions from it instead of x_t. Per sample, the draft is: none (50%), an
# estimate on the sample's own map (25%), or an estimate on another map of the
# same task family (25%, --sc_alt_frac) -- the last one is what teaches "the
# draft crosses a wall here -> go around". Drafts are estimated at
# t + U{0..16} (--sc_max_lag), the gap to the previous step of the 16-step DFS
# sampler. See MapConditionalGaussianDiffusion.set_self_cond and
# mapcond/dit_joint_models.py.
#
# Same data, shape and step budget as run_train_dit_joint.sh, so state_<step>.pt
# compares directly with logs/mapcond-mazev1-seed2-dit-joint at the same step.
# ~3.5 it/s on the RTX 3060 (joint without self-cond ~4.8).
#
# usage: run_train_dit_joint_sc.sh [n_train_steps]

steps="${1:-1000000}"

python -m mapcond.train_multimap \
  --maps giant --variant_tasks 1 2 3 4 5 --variant_vars 0 1 2 3 4 5 6 7 8 9 10 11 \
  --backbone dit --dit_joint --self_cond --sc_alt_frac 0.25 --sc_max_lag 16 \
  --dit_hidden 256 --dit_heads 8 --dit_depth 4 \
  --dit_patch_size 4 --dit_mlp_ratio 4.0 \
  --pool_size 4 --cfg_dropout 0.1 \
  --n_train_steps "${steps}" \
  --savepath logs/mapcond-mazev1-seed2-dit-joint-sc/variants_train_giant \
  --device cuda \
  --variant_dir /home/yufei/research/diffusion/ogbench/data_gen_scripts/newdata_mazev1_seed2 \
  --variant_json_dir ../maze_update/maze_variants_seed2
