#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# python -m mapcond.train_multimap \
#   --maps giant \
#   --variant_tasks 1 2 3 4 5 \
#   --variant_vars 0 1 2 3 4 5 6 7 8 9 10 11 \
#   --savepath logs/mapcond-H600-CFG-mazeseed2/variants_train_giant \
#   --device cuda \
#   --variant_dir /home/yufei/research/diffusion/ogbench/data_gen_scripts/newdata_mazev1_seed2 \
#   --variant_json_dir ../maze_update_v1/maze_variants_seed2


# python -m mapcond.train_multimap \
#   --maps giant --variant_tasks 1 2 3 4 5 --variant_vars 0 1 2 3 4 5 6 7 8 9 10 11 \
#   --local_channels 2 --pool_size 4 --cfg_dropout 0.1 \
#   --savepath logs/mapcond-mazev1-seed2-feature/variants_train_giant \
#   --device cuda \
#   --variant_dir /home/yufei/research/diffusion/ogbench/data_gen_scripts/newdata_mazev1_seed2 \
#   --variant_json_dir ../maze_update_v1/maze_variants_seed2


# python -m mapcond.train_multimap \
#   --maps giant --variant_tasks 1 2 3 4 5 --variant_vars 0 1 2 3 4 5 6 7 8 9 10 11 \
#   --map_blind \
#   --savepath logs/dfs-mazev1-seed2/variants_train_giant \
#   --device cuda \
#   --variant_dir /home/yufei/research/diffusion/ogbench/data_gen_scripts/newdata_mazev1_seed2 \
#   --variant_json_dir ../maze_update_v1/maze_variants_seed2

python -m mapcond.train_multimap \
  --maps giant --variant_tasks 1 2 3 4 5 --variant_vars 0 1 2 3 4 5 6 7 8 9 10 11 \
  --local_channels 2 --local_film --pool_size 4 --cfg_dropout 0.1 \
  --savepath logs/mapcond-mazev1-seed2-film/variants_train_giant \
  --device cuda \
  --variant_dir /home/yufei/research/diffusion/ogbench/data_gen_scripts/newdata_mazev1_seed2 \
  --variant_json_dir ../maze_update_v1/maze_variants_seed2