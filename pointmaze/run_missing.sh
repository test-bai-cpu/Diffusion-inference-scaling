#!/usr/bin/env bash
# Rerun the 24 missing combinations from run_origin.sh:
#   - 12 that previously crashed due to the maze_verifier bug (now fixed)
#   - 12 that never ran because the slurm job hit the time limit

METHOD="dfs"
DATASET="pointmaze-giant-newvar-navigate-v0"
JSON_DIR="../maze_update/maze_variants"

run() {
    local task=$1 level=$2 v=$3
    local level_idx=$((level - 1))
    local variant_idx=$((level_idx * 3 + (v - 1)))
    echo "Running task ${task}, level ${level}, variant_idx ${variant_idx} with method ${METHOD}"
    MUJOCO_GL=egl python run.py \
        --dataset "${DATASET}" \
        --method "${METHOD}" \
        --maze_json_dir "${JSON_DIR}" \
        --maze_variant_idx "${variant_idx}" \
        --version "${METHOD}-task${task}-level${level}-variant${v}" \
        --task "${task}"
}

# task1 (level3, variants 1 and 3)
run 1 3 1
run 1 3 3

# task2 (level1 variants 1,3 | level2 variant3 | level4 variant3)
run 2 1 1
run 2 1 3
run 2 2 3
run 2 4 3

# task3 (level4, variant1)
run 3 4 1

# task4 (level1 variant3 | level3 variant3 | level4 all)
run 4 1 3
run 4 3 3
run 4 4 1
run 4 4 2
run 4 4 3

# task5 (all 12)
for level in 1 2 3 4; do
    for v in 1 2 3; do
        run 5 "${level}" "${v}"
    done
done