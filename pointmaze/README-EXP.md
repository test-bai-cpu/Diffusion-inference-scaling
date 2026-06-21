


```bash
conda activate maze
```

```bash
MUJOCO_GL=egl python run.py --dataset pointmaze-giant-navigate-v0 --method dfs --device cuda

MUJOCO_GL=egl python run.py --dataset pointmaze-giant-navigate-v0 --method hard-dfs --device cuda
```

Add `MUJOCO_GL=egl` is for running it in remote ssh terminal.

```bash
cd /projects/bhlg/yzhu37/Diffusion-inference-scaling/pointmaze
sbatch run_variants.slurm
```

For check status:
squeue -u $USER 

To watch the live output while it runs:
tail -f slurm-<jobid>.out