#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# DiT with joint attention over trajectory tokens + one token per map cell
# (mapcond/dit_joint_models.py, MM-DiT style). Trained from scratch on the same
# data, DiT shape and step budget as run_train_dit.sh, so state_<step>.pt is
# directly comparable to logs/mapcond-mazev1-seed2-dit at the same step.
# Map input is the occupancy grid only (no BFS channel). No self-conditioning:
# trajectory token positions come from x_t.
#
# ~4.8 it/s on the RTX 3060 (plain DiT ~8), so 1M steps is ~2.5 days.
#
# Check before any rollout eval: cond_sensitivity.py's open->wall row (plain
# DiT and globalfilm both sit at +-5pp) and plan_quality.py at w=0 and w=2.
#
# usage: run_train_dit_joint.sh [n_train_steps]

steps=1000000

python -m mapcond.train_multimap \
  --maps giant --variant_tasks 1 2 3 4 5 --variant_vars 0 1 2 3 4 5 6 7 8 9 10 11 \
  --backbone dit --dit_joint --dit_hidden 256 --dit_heads 8 --dit_depth 4 \
  --dit_patch_size 4 --dit_mlp_ratio 4.0 \
  --pool_size 4 --cfg_dropout 0.1 \
  --n_train_steps "${steps}" \
  --savepath logs/mapcond-mazev1-seed2-dit-joint/variants_train_giant \
  --device cuda \
  --variant_dir /home/yufei/research/diffusion/ogbench/data_gen_scripts/newdata_mazev1_seed2 \
  --variant_json_dir ../maze_update/maze_variants_seed2
