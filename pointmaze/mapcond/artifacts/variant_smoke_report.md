# Variant Smoke Test Report

Interpreter: `/home/yufei/anaconda3/envs/maze/bin/python`
Elapsed: `1.7s`

| Check | Result | Detail |
|---|---|---|
| A. variant npz + JSON grid load | PASS | obs(1000500, 2) act(1000500, 2) term(1000500,) grid(12, 16) y_max=37.27045 |
| B. pooled base+variant dataset | PASS | len=5235 canvas=(12, 16); giant@0 grid_match=True; giant_task1_var0@1745 grid_match=True; giant_task1_var1@3490 grid_match=True; variant_grid_L1=10.0 |
| C. one-batch overfit drops loss | PASS | loss 0.5192 -> 0.1159 (22.3%) |
| D. base vs variant changes predicted x0 | PASS | max\|delta\|=5.579e-05 |
