"""
run_eval.py — graceful-degradation evaluation protocol (Step 8)
================================================================
Sweeps  methods x tasks x calibrated difficulty levels x seeds  on the point-
maze planner and writes one tidy CSV (eval_results.csv) with the separated
success / collision / corner-cut / dead-end metrics from episode_metrics, plus
the difficulty-level bookkeeping needed for the degradation plots.

Methods compared (all share the SAME frozen diffusion backbone; they differ only
in the inference-time search / guidance):

    dfs         : paper baseline depth-first search (MazeVerifier).
    field       : dfs + BFS distance-field guidance (use_distance_field).
    mafgs       : Map-Aware Feasibility-Guided Search (clearance + distance-field
                  composite, hard feasibility gate).  [our method]
    mafgs+mcgn  : mafgs + the learned Map-Conditioned Guidance Network side model
                  (an added guidance term, still gated by the analytic check).

Difficulty levels come from the CALIBRATED variant ladders in
maze_update/maze_variants_v2/giant_task{N}.json (n_levels x n_variants, each
variant carrying its own maze_map + a monotone `difficulty` score).  Sweeping
levels 0..L-1 is what lets us show a MONOTONE success drop (Aim c) and measure
each method's margin as the maze gets harder.

This is the evaluation DRIVER. It needs the GPU diffusion stack + the `maze`
env + the pretrained checkpoints, so it is meant to run on the user's machine.
Everything it imports here (get_args/get_pipe/episode_metrics) is unit-tested in
the sandbox; the driver logic (arg construction, CSV schema, level bookkeeping)
is covered by test_run_eval.py with a mocked pipe.

Usage
-----
    python run_eval.py \
        --maze_json_dir ../maze_update/maze_variants_v2 \
        --methods dfs field mafgs mafgs+mcgn \
        --tasks 1 2 3 4 5 --levels 0 1 2 3 --seeds 0 1 2 \
        --mcgn_ckpt sidemodel/mcgn_giant.pt \
        --device cuda:0 --out eval_results.csv

Then:  python plot_degradation.py eval_results.csv degradation
"""
import argparse
import csv
import json
import os

# Metrics written per (method, task, level, variant, seed). Mirrors run.py's
# METRIC_COLS so plot_metrics.py stays compatible, plus the ladder bookkeeping.
METRIC_COLS = ['total_reward', 'success', 'collision_rate', 'cornercut_rate',
               'deadend_frac', 'stalled_prog', 'final_gap', 'steps', 'compute']
BOOK_COLS = ['dataset', 'method', 'config', 'task', 'level', 'variant_idx',
             'difficulty_score', 'seed']


# The four evaluation configurations. `config` is the display label; `method` is
# the actual search method string get_pipe dispatches on; the extra keys are
# Arguments fields toggled before get_args (so deepcopy carries them).
CONFIGS = {
    'dfs':        {'method': 'dfs',   'fields': {}},
    'field':      {'method': 'dfs',   'fields': {'use_distance_field': True}},
    'mafgs':      {'method': 'mafgs', 'fields': {}},
    'mafgs+mcgn': {'method': 'mafgs', 'fields': {'_needs_mcgn': True}},
}


def level_variants(maze_json_dir, task, levels):
    """Return [(level, variant_idx, difficulty), ...] for the requested levels of
    one task, read from the calibrated ladder JSON. If a level has several
    variants we take the FIRST (lowest-index) variant at that level, which the
    calibration ordered by difficulty; pass --all_variants to sweep them all."""
    path = os.path.join(maze_json_dir, f'giant_task{task}.json')
    d = json.load(open(path))
    by_level = {}
    for k, v in enumerate(d['variants']):
        by_level.setdefault(v['level_index'], []).append((k, float(v['difficulty_score'])))
    out = []
    for L in levels:
        if L in by_level:
            out.append((L, by_level[L]))
    return out


def build_args(base_cfg, dataset, method, device, task, maze_json_dir,
               variant_idx, seed, mcgn_ckpt, extra_fields):
    """Construct the per-run Arguments and expand via get_args (returns a grid;
    the giant branches yield a single element for dfs/mafgs)."""
    from search.configs import Arguments
    from search.script_utils import get_args
    a = Arguments()
    a.dataset = dataset
    a.method = method
    a.device = device
    a.task = [task]
    a.maze_json_dir = maze_json_dir
    a.maze_variant_idx = variant_idx
    a.seed = seed
    # per-config field toggles (applied BEFORE get_args so deepcopy carries them)
    for k, val in extra_fields.items():
        if k == '_needs_mcgn':
            a.mcgn_ckpt = mcgn_ckpt
        else:
            setattr(a, k, val)
    return get_args(a)


def run(args):
    from search.script_utils import get_pipe
    write_header = not os.path.exists(args.out)
    f = open(args.out, 'a', newline='')
    w = csv.writer(f)
    if write_header:
        w.writerow(BOOK_COLS + METRIC_COLS)

    n_done = 0
    for cfg_name in args.methods:
        spec = CONFIGS[cfg_name]
        for task in args.tasks:
            for (level, variants) in level_variants(args.maze_json_dir, task, args.levels):
                # first variant per level, or all if --all_variants
                chosen = variants if args.all_variants else variants[:1]
                for (variant_idx, difficulty) in chosen:
                    for seed in args.seeds:
                        grid = build_args(spec, args.dataset, spec['method'],
                                          args.device, task, args.maze_json_dir,
                                          variant_idx, seed, args.mcgn_ckpt,
                                          spec['fields'])
                        for a in grid:
                            pipe = get_pipe(a)
                            returns = pipe.experiment()
                            # per-task rows (task list has one entry here)
                            for tid in [t for t in returns if t != 'average']:
                                r = returns[tid]
                                w.writerow([args.dataset, spec['method'], cfg_name,
                                            tid, level, variant_idx,
                                            round(difficulty, 4), seed]
                                           + [r.get(c, '') for c in METRIC_COLS])
                            f.flush()
                            n_done += 1
                            print(f"[{n_done}] {cfg_name} task{task} L{level} "
                                  f"var{variant_idx} seed{seed} "
                                  f"succ={returns['average'].get('success','?')}")
    f.close()
    print(f"\nWrote {n_done} runs to {args.out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='pointmaze-giant-navigate-v0')
    p.add_argument('--methods', nargs='+',
                   default=['dfs', 'field', 'mafgs', 'mafgs+mcgn'],
                   choices=list(CONFIGS.keys()))
    p.add_argument('--tasks', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    p.add_argument('--levels', nargs='+', type=int, default=[0, 1, 2, 3])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--maze_json_dir', default='../maze_update/maze_variants_v2')
    p.add_argument('--mcgn_ckpt', default='sidemodel/mcgn_giant.pt')
    p.add_argument('--all_variants', action='store_true',
                   help='sweep every variant at each level (default: first only)')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default='eval_results.csv')
    run(p.parse_args())
