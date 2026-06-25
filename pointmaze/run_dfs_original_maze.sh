#!/usr/bin/env bash

for task in 1 2 3 4 5; do
    method="dfs"
    echo "Running task ${task} with method ${method}"
    MUJOCO_GL=egl python run.py --dataset pointmaze-giant-navigate-v0 --method dfs --version "${method}-task${task}" --task "${task}"
done