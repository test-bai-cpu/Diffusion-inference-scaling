from search.configs import Arguments
from search.script_utils import get_pipe, get_args

def main(dataset: str="pointmaze-giant-navigate-v0", method: str='dfs', device: str="cuda:7", version: str='',
         task=None, maze_json_dir: str='', maze_variant_idx: int=0, run_tag: str='',
         num_samples: int=40,
         use_distance_field: bool=False, dist_omega: float=1.0, dist_mode: str='sum',
         dist_smooth_sigma: float=0.5, dist_connectivity: int=4,
         maze_weight: float=1.0, dist_weight: float=1.0,
         corner_radius_frac: float=0.30, corner_transition_weight: float=20.0,
         verifier_monitor: bool=False, verifier_monitor_freq: int=1,
         use_map_cond: bool=False, map_cond_ckpt: str='', map_cond_use_ema: bool=True):
    args = Arguments()
    args.device = device
    args.dataset = dataset
    args.method = method
    args.version = version
    args.task = task if task is not None else [1, 2, 3, 4, 5]
    args.maze_json_dir = maze_json_dir
    args.maze_variant_idx = maze_variant_idx
    args.run_tag = run_tag
    args.num_samples = num_samples
    args.use_distance_field = use_distance_field
    args.dist_omega = dist_omega
    args.dist_mode = dist_mode
    args.dist_smooth_sigma = dist_smooth_sigma
    args.dist_connectivity = dist_connectivity
    args.maze_weight = maze_weight
    args.dist_weight = dist_weight
    args.corner_radius_frac = corner_radius_frac
    args.corner_transition_weight = corner_transition_weight
    args.verifier_monitor = verifier_monitor
    args.verifier_monitor_freq = verifier_monitor_freq
    args.use_map_cond = use_map_cond
    args.map_cond_ckpt = map_cond_ckpt
    args.map_cond_use_ema = map_cond_use_ema
    args_grid = get_args(args)

    import csv, os as _os
    METRIC_COLS = ['total_reward', 'success', 'collision_rate', 'cornercut_rate',
                   'deadend_frac', 'stalled_prog', 'final_gap', 'steps', 'compute']
    for args in args_grid:
        pipe = get_pipe(args)
        returns = pipe.experiment()
        success_rate = returns['average']['total_reward']
        average_compute = returns['average']['compute']
        print(f"Success Rate: {success_rate}, Average Compute: {average_compute}")
        _tag = f'_{args.run_tag}' if getattr(args, 'run_tag', '') else ''
        output_file = f'results_{args.method}_{args.dataset}{_tag}.txt'
        run_str = args.version if args.version else args.method
        with open(output_file, 'a') as f:
            f.write(f"Maze: {args.dataset} | Run: {run_str} | Compute: {average_compute} | Success Rate: {success_rate}\n")

        # ---- detailed per-task metrics CSV (success separated from failure modes) ----
        detailed_file = f'results_detailed{_tag}.csv'
        write_header = not _os.path.exists(detailed_file)
        tasks = [t for t in returns.keys() if t != 'average']
        with open(detailed_file, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['dataset', 'method', 'run', 'maze_json_dir',
                            'maze_variant_idx', 'task'] + METRIC_COLS)
            for task_id in tasks:
                r = returns[task_id]
                row = [args.dataset, args.method, run_str,
                       getattr(args, 'maze_json_dir', ''),
                       getattr(args, 'maze_variant_idx', 0), task_id]
                row += [r.get(c, '') for c in METRIC_COLS]
                w.writerow(row)
            # average row
            a = returns['average']
            w.writerow([args.dataset, args.method, run_str,
                        getattr(args, 'maze_json_dir', ''),
                        getattr(args, 'maze_variant_idx', 0), 'average']
                       + [a.get(c, '') for c in METRIC_COLS])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference scaling experiment.")
    parser.add_argument('--dataset',
                        type=str,
                        default='pointmaze-ultra-navigate-v0',
                        choices=['pointmaze-giant-navigate-v0', 'pointmaze-giant-newvar-navigate-v0', 'pointmaze-giant-newvar2-navigate-v0', 'pointmaze-ultra-navigate-v0'],
                        help='Maze env to use for the experiment.')
    parser.add_argument('--method', 
                        type=str, 
                        default='bon', 
                        choices=['dfs', 'bon', 'bfs-resampling', 'bfs-pruning'],
                        help='Search method to use')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        )
    parser.add_argument('--version',
                        type=str,
                        default='',
                        help='Optional version tag (e.g. ada) shown in results and folder names.')
    parser.add_argument('--run_tag',
                        type=str,
                        default='',
                        help='Optional label appended to the results txt filename (e.g. mazev2 -> results_dfs_<dataset>_mazev2.txt). All runs still append to this one file.')
    parser.add_argument('--task',
                        type=int,
                        nargs='+',
                        default=[1, 2, 3, 4, 5],
                        help='Task IDs to evaluate, e.g. --task 1 2 3 4 5.')
    parser.add_argument('--maze_json_dir',
                        type=str,
                        default='',
                        help='Dir containing giant_task{N}.json files. When set, each task uses its OOD map.')
    parser.add_argument('--maze_variant_idx',
                        type=int,
                        default=0,
                        help='Index into the variants list in the JSON file (same index used for all tasks).')
    parser.add_argument('--num_samples',
                        type=int,
                        default=40,
                        help='Number of DFS/BFS rollouts to run per task. Use 1 for a quick trajectory generation check.')
    
    # BFS distance field guidance
    parser.add_argument('--use_distance_field',
                        action='store_true', default=False,
                        help='Enable BFS distance-field guidance at inference time.')
    parser.add_argument('--dist_omega',
                        type=float, default=1.0,
                        help='Scale applied to distance-field cost (guidance strength).')
    parser.add_argument('--dist_mode',
                        type=str, default='sum',
                        choices=['endpoint', 'sum', 'weighted', 'monotonic'],
                        help='How per-timestep distances are aggregated into a cost. '
                             'endpoint is broken when conditioning pins t=T-1 to goal (D=0 always). '
                             'sum (mean per timestep) is recommended.')
    parser.add_argument('--dist_smooth_sigma',
                        type=float, default=0.5,
                        help='Gaussian blur sigma on the raw BFS field (0 = no blur).')
    parser.add_argument('--dist_connectivity',
                        type=int, default=4, choices=[4, 8],
                        help='BFS connectivity: 4 (N/S/E/W) or 8 (includes diagonals).')
    parser.add_argument('--maze_weight',
                        type=float, default=1.0,
                        help='Weight of MazeVerifier in CompositeVerifier.')
    parser.add_argument('--dist_weight',
                        type=float, default=1.0,
                        help='Weight of DistanceFieldVerifier in CompositeVerifier.')
    parser.add_argument('--corner_radius_frac',
                        type=float, default=0.30,
                        help='Forbidden-corner radius as a fraction of env._maze_unit. Set 0 to disable.')
    parser.add_argument('--corner_transition_weight',
                        type=float, default=20.0,
                        help='Hard DFS penalty for diagonal free-cell transitions blocked by two corner walls. Set 0 to disable.')
    parser.add_argument('--verifier_monitor',
                        action='store_true', default=False,
                        help='Print DFS verifier wall/transition cost decomposition at accept/reject checks.')
    parser.add_argument('--verifier_monitor_freq',
                        type=int, default=1,
                        help='Print every N DFS verifier checks when --verifier_monitor is set.')
    parser.add_argument('--use_map_cond',
                        action='store_true', default=False,
                        help='Use a map-conditional diffusion checkpoint from mapcond.train_multimap.')
    parser.add_argument('--map_cond_ckpt',
                        type=str, default='',
                        help='Path to a mapcond train_multimap state_*.pt checkpoint.')
    parser.add_argument('--map_cond_use_ema',
                        type=lambda x: str(x).lower() in ('1', 'true', 'yes', 'y'),
                        default=True,
                        help='Whether to load EMA weights from the mapcond checkpoint.')
    cli_args = parser.parse_args()

    main(dataset=cli_args.dataset, method=cli_args.method, device=cli_args.device, version=cli_args.version,
         task=cli_args.task,
         maze_json_dir=cli_args.maze_json_dir, maze_variant_idx=cli_args.maze_variant_idx,
         run_tag=cli_args.run_tag,
         num_samples=cli_args.num_samples,
         use_distance_field=cli_args.use_distance_field,
         dist_omega=cli_args.dist_omega,
         dist_mode=cli_args.dist_mode,
         dist_smooth_sigma=cli_args.dist_smooth_sigma,
         dist_connectivity=cli_args.dist_connectivity,
         maze_weight=cli_args.maze_weight,
         dist_weight=cli_args.dist_weight,
         corner_radius_frac=cli_args.corner_radius_frac,
         corner_transition_weight=cli_args.corner_transition_weight,
         verifier_monitor=cli_args.verifier_monitor,
         verifier_monitor_freq=cli_args.verifier_monitor_freq,
         use_map_cond=cli_args.use_map_cond,
         map_cond_ckpt=cli_args.map_cond_ckpt,
         map_cond_use_ema=cli_args.map_cond_use_ema)
