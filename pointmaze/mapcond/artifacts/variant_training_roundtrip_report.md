# Variant Training Round-Trip Report

Interpreter: `/home/yufei/anaconda3/envs/maze/bin/python`

Command:

```bash
cd pointmaze
MPLCONFIGDIR=/tmp/matplotlib-mapcond /home/yufei/anaconda3/envs/maze/bin/python -m mapcond.train_multimap \
  --maps giant \
  --variant_tasks 1 \
  --variant_vars 0 1 \
  --horizon 64 \
  --n_diffusion_steps 16 \
  --dim 8 \
  --dim_mults 1 2 \
  --n_train_steps 250 \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --gradient_accumulate_every 1 \
  --ema_decay 0.995 \
  --step_start_ema 10 \
  --update_ema_every 1 \
  --save_freq 250 \
  --log_freq 50 \
  --max_episodes_per_map 1 \
  --num_workers 0 \
  --seed 3 \
  --device cpu \
  --savepath logs/mapcond/variant_smoke_250
```

Result:

| Item | Value |
|---|---|
| Maps | `giant`, `giant_task1_var0`, `giant_task1_var1` |
| Dataset windows | `5811` |
| Canvas | `12x16` |
| Model params | `45,228` |
| Loss log | `0: 0.37803`, `50: 0.44853`, `100: 0.44065`, `150: 0.38252`, `200: 0.33232` |
| Checkpoint | `pointmaze/logs/mapcond/variant_smoke_250/state_250.pt` |
| Embedded map specs | `3` |
| Embedded observation y max | `37.270450592041016` |
| Embedded variants | `giant_task1_var0`, `giant_task1_var1` |

Artifacts:

```text
pointmaze/logs/mapcond/variant_smoke_250/global_normalizer.pkl
pointmaze/logs/mapcond/variant_smoke_250/loss_log.json
pointmaze/logs/mapcond/variant_smoke_250/state_250.pt
pointmaze/logs/mapcond/variant_smoke_250/train_loss.png
```
