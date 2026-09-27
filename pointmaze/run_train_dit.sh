#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# DiT1D backbone (mapcond.dit_models.MapConditionalDiT1D), Global-FiLM-
# equivalent tier (adaLN-Zero built into the class -- see its docstring).
#
# Every data/optimization arg here is copied verbatim from the --global_film
# command in run_train_maps.sh (the run that produced
# logs/mapcond-mazev1-seed2-globalfilm/variants_train_giant/state_500000.pt,
# the checkpoint run_global_film.sh evaluates) -- only the backbone flags
# differ. Keep it that way: this is meant to be a single-variable ablation
# against that checkpoint (backbone U-Net+FiLM vs DiT+adaLN-Zero, same data,
# same schedule), not a from-scratch hyperparameter search.
#
# To compare fairly against the globalfilm arm, evaluate this at the SAME
# training step it was evaluated at (run_global_film.sh currently uses
# state_500000.pt) rather than at final/best -- see that script's own TODO
# about matching step counts across arms.

python -m mapcond.train_multimap \
  --maps giant --variant_tasks 1 2 3 4 5 --variant_vars 0 1 2 3 4 5 6 7 8 9 10 11 \
  --backbone dit --dit_hidden 256 --dit_heads 8 --dit_depth 4 \
  --dit_patch_size 4 --dit_mlp_ratio 4.0 \
  --pool_size 4 --cfg_dropout 0.1 \
  --savepath logs/mapcond-mazev1-seed2-dit/variants_train_giant \
  --device cuda \
  --variant_dir /home/yufei/research/diffusion/ogbench/data_gen_scripts/newdata_mazev1_seed2 \
  --variant_json_dir ../maze_update/maze_variants_seed2
