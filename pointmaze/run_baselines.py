"""
Entry point for the two non-diffusion BFS baselines (bfs-greedy, bfs-cspace)
-- see search/methods/bfs_baseline_pipes.py for what they do and why.

Deliberately a SEPARATE script from run.py, rather than new --method choices
wired into run.py / search/script_utils.py. Those two files dispatch on
exact-match ('dfs') or substring ('bfs' in args.method, 'dfs' in
args.method) checks tuned for the existing search methods; adding
'bfs-greedy'/'bfs-cspace' there risks silently falling into an unrelated
existing branch (e.g. get_pipe()'s `elif 'bfs' in args.method` would route a
literal 'bfs-greedy' through the diffusion-guided BFSGuidance class, not
this baseline). Keeping run.py and search/script_utils.py completely
untouched removes that risk entirely, at the cost of a separate CLI. Output
files (results_*.txt, results_detailed*.csv) use the identical format/
naming convention as run.py's, so the new rows drop straight into the same
tables.

Usage (mirrors run_origin_best_config.sh's scope: OOD giant-newvar variants,
levels 3-4, tasks 1-5, num_samples=40):

    python run_baselines.py --method bfs-greedy --dataset pointmaze-giant-newvar-navigate-v0 \\
        --maze_json_dir ../maze_update/maze_variants --maze_variant_idx 7 \\
        --task 3 --version bfs-greedy-task3-level3-variant2 \\
        --run_tag global-w2-omega05-base3-noise-trans40 --num_samples 40

See run_baselines.sh for the full sweep over tasks x levels 3-4 x variants,
for both methods.
"""
import argparse
import csv
import os
from os.path import join

from search.configs import Arguments
from search.search_policy import SearchPolicy
from search.methods.bfs_baseline_pipes import BFSGreedyPipe, BFSCspacePipe

PIPE_CLASSES = {
    'bfs-greedy': BFSGreedyPipe,
    'bfs-cspace': BFSCspacePipe,
}

# Same column list run.py uses for results_detailed*.csv, so appended rows
# match the existing header exactly. 'success', 'collision_rate',
# 'cornercut_rate', 'deadend_frac', 'stalled_prog', 'final_gap' are DFS
# backtracking failure-mode attributions specific to adaptive_dfs/
# staged_dfs -- not applicable here, left blank like any other method that
# doesn't emit them.
METRIC_COLS = ['total_reward', 'success', 'collision_rate', 'cornercut_rate',
               'deadend_frac', 'stalled_prog', 'final_gap', 'steps', 'compute']

# Baseline-specific metrics that do NOT go through BasePipe.eval()'s
# averaging (which rescales any average <= 1 by x100 -- correct for
# fractions like wall_hit_frac, but would corrupt sub-second wall-clock
# numbers). Written to their own CSV instead, one row per episode.
EXTRA_METRIC_COLS = ['task_id', 'maze_variant_idx', 'success', 'plan_time_sec',
                     'rollout_time_sec', 'wall_clock_sec', 'path_len_actual',
                     'path_len_optimal', 'path_efficiency', 'wall_hit_frac',
                     'unreachable']


def main():
    parser = argparse.ArgumentParser(
        description="Run a non-diffusion BFS baseline (bfs-greedy or bfs-cspace) "
                    "through the exact same real-env harness as run.py.")
    parser.add_argument('--method', type=str, required=True, choices=list(PIPE_CLASSES.keys()))
    parser.add_argument('--dataset', type=str, default='pointmaze-giant-newvar-navigate-v0',
                        choices=['pointmaze-giant-navigate-v0', 'pointmaze-giant-newvar-navigate-v0',
                                 'pointmaze-giant-newvar2-navigate-v0'])
    parser.add_argument('--model_dataset', type=str, default='pointmaze-giant-navigate-v0',
                        help='Checkpoint dataset key for the (unused, loaded only for '
                             'setup-path fidelity) diffusion model. Matches the '
                             'model_dataset the main method\'s own saved args.json uses.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--version', type=str, default='')
    parser.add_argument('--run_tag', type=str, default='')
    parser.add_argument('--task', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--maze_json_dir', type=str, default='')
    parser.add_argument('--maze_variant_idx', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=40)
    parser.add_argument('--write_detailed_csv', action='store_true', default=False,
                        help='Also write results_detailed<run_tag>.csv (per-task failure-mode '
                             'columns), matching run.py\'s --write_detailed_csv. Off by default.')
    parser.add_argument('--sampling_horizon', type=int, default=600,
                        help='Must match the main method\'s actual sampling_horizon '
                             '(confirmed 600 in its own saved args.json for the '
                             'giant-newvar dataset) so the reference path is '
                             'resampled to the same length.')
    parser.add_argument('--seed', type=int, default=42)
    cli = parser.parse_args()

    args = Arguments()
    args.method = cli.method
    args.dataset = cli.dataset
    args.model_dataset = cli.model_dataset
    args.device = cli.device
    args.version = cli.version
    args.run_tag = cli.run_tag
    args.task = cli.task
    args.maze_json_dir = cli.maze_json_dir
    args.maze_variant_idx = cli.maze_variant_idx
    args.num_samples = cli.num_samples
    args.sampling_horizon = cli.sampling_horizon
    args.seed = cli.seed
    # use_map_cond stays False (default): the diffusion model setup() loads
    # is never called by these baselines, so there is no reason to also pay
    # for loading the (much larger) map-conditional checkpoint.

    pipe_cls = PIPE_CLASSES[cli.method]
    policy = SearchPolicy(args)
    pipe = pipe_cls(args, policy, guidance=None)

    returns = pipe.experiment()

    success_rate = returns['average']['total_reward']
    average_compute = returns['average']['compute']
    print(f"Success Rate: {success_rate}, Average Compute: {average_compute}")

    _tag = f'_{args.run_tag}' if getattr(args, 'run_tag', '') else ''
    run_str = args.version if args.version else args.method

    output_file = f'results_{args.method}_{args.dataset}{_tag}.txt'
    with open(output_file, 'a') as f:
        f.write(f"Maze: {args.dataset} | Run: {run_str} | Compute: {average_compute} | "
                f"Success Rate: {success_rate}\n")

    detailed_file = None
    if cli.write_detailed_csv:
        detailed_file = f'results_detailed{_tag}.csv'
        write_header = not os.path.exists(detailed_file)
        tasks = [t for t in returns.keys() if t != 'average']
        with open(detailed_file, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['dataset', 'method', 'run', 'maze_json_dir',
                            'maze_variant_idx', 'task'] + METRIC_COLS)
            for task_id in tasks:
                r = returns[task_id]
                row = [args.dataset, args.method, run_str, args.maze_json_dir,
                      args.maze_variant_idx, task_id]
                row += [r.get(c, '') for c in METRIC_COLS]
                w.writerow(row)
            a = returns['average']
            w.writerow([args.dataset, args.method, run_str, args.maze_json_dir,
                        args.maze_variant_idx, 'average']
                       + [a.get(c, '') for c in METRIC_COLS])

    # ---- baseline-only metrics: wall-clock, path efficiency, wall-hits ----
    extra_file = f'results_baseline_extra_metrics{_tag}.csv'
    write_extra_header = not os.path.exists(extra_file)
    with open(extra_file, 'a', newline='') as f:
        w = csv.writer(f)
        if write_extra_header:
            w.writerow(['dataset', 'method', 'run'] + EXTRA_METRIC_COLS)
        for row in pipe.extra_metrics_log:
            w.writerow([args.dataset, args.method, run_str]
                       + [row.get(c, '') for c in EXTRA_METRIC_COLS])

    written = [output_file] + ([detailed_file] if detailed_file else []) + [extra_file]
    print(f"Wrote {', '.join(written)}")


if __name__ == "__main__":
    main()
