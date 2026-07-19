#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python -m mapcond.train_multimap \
  --maps giant \
  --variant_tasks 1 2 3 4 5 \
  --variant_vars 0 1 2 3 4 5 6 7 8 9 10 11 \
  # --savepath logs/mapcond-H600-CFG-mazev2/variants_train_giant \
  --savepath logs/mapcond-H600-CFG-mazeseed2/variants_train_giant \
  --device cuda