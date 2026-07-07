#!/bin/bash
#SBATCH --job-name=mapcond_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-mapcond-%j.out

# ---------------------------------------------------------------------------
# Multi-map (map-conditional) diffusion training launcher.
#
# Trains ONE model on several pointmaze maps with the occupancy grid fed to the
# network as conditioning. Training is mujoco-FREE (offline .npy + ast-parsed
# grids), but we still export MUJOCO_GL=egl to stay consistent with the repo's
# environment on headless nodes -- harmless if mujoco is never imported.
#
# Submit from pointmaze/:   sbatch mapcond/run_multimap.slurm
# Edit REPO / ENV below to match your cluster paths (bhlg example given).
# ---------------------------------------------------------------------------

set -euo pipefail

REPO=/projects/bhlg/yzhu37/Diffusion-inference-scaling/pointmaze
ENV=maze                                  # conda env with torch+cuda on the cluster
MAPS="medium large giant teleport"        # held-in maps; hold one out for eval
HORIZON=256
NTRAIN=1000000
BATCH=32
SAVEPATH=logs/mapcond/H${HORIZON}_T256

export MUJOCO_GL=egl

source ~/.bashrc
conda activate ${ENV}
cd ${REPO}

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-none} maps=${MAPS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m mapcond.train_multimap \
    --maps ${MAPS} \
    --horizon ${HORIZON} \
    --n_diffusion_steps 256 \
    --n_train_steps ${NTRAIN} \
    --batch_size ${BATCH} \
    --learning_rate 2e-4 \
    --gradient_accumulate_every 2 \
    --ema_decay 0.995 \
    --save_freq 20000 \
    --log_freq 100 \
    --num_workers 8 \
    --device cuda \
    --savepath ${SAVEPATH}

echo "done -> ${SAVEPATH}"
