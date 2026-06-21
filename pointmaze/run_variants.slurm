#!/bin/bash
#SBATCH --job-name=maze-variants
#SBATCH --partition=gpuA40x4
#SBATCH --account=bhlg-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=48g
#SBATCH --time=18:00:00
#SBATCH --constraint="projects"
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module reset
module load pytorch-conda
conda activate /projects/bhlg/yzhu37/envs/maze

cd /projects/bhlg/yzhu37/Diffusion-inference-scaling/pointmaze
export MUJOCO_GL=egl
# bash run_maze_variants.sh --device cuda
bash run_origin.sh --device cuda
