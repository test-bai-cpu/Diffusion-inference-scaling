"""
Step 8: Hyperparameter sweep for BFS distance-field guidance.

Sweeps omega and dist_mode on the OOD BON task (task 1, giant_task1.json variant 0).
Uses N=10 samples for better statistical coverage than the 4-sample smoke tests.

Saves results to outputs/sweep_results.csv and prints a summary table.
"""
import os, time, csv

os.makedirs('outputs/sweep', exist_ok=True)

from search.configs import Arguments
from search.script_utils import get_pipe, get_args

MAZE_JSON_DIR = '../maze_update/maze_variants'
N_SAMPLES = 10
TASK_IDS  = [1]


def make_args(method, use_df, dataset='pointmaze-giant-navigate-v0',
              maze_json_dir='', maze_variant_idx=0,
              dist_omega=1.0, dist_mode='sum',
              maze_weight=1.0, dist_weight=1.0):
    args = Arguments()
    args.device           = 'cuda'
    args.dataset          = dataset
    args.method           = method
    args.task             = TASK_IDS
    args.version          = ''
    args.maze_json_dir    = maze_json_dir
    args.maze_variant_idx = maze_variant_idx
    args.use_distance_field  = use_df
    args.dist_omega          = dist_omega
    args.dist_mode           = dist_mode
    args.dist_smooth_sigma   = 0.5
    args.dist_connectivity   = 4
    args.maze_weight         = maze_weight
    args.dist_weight         = dist_weight
    return args


def run_config(tag, args_base):
    args_grid = get_args(args_base)
    a = args_grid[0]
    a.num_samples = N_SAMPLES
    a.logging_dir = f'outputs/sweep/{tag}'
    os.makedirs(a.logging_dir, exist_ok=True)
    t0 = time.time()
    pipe = get_pipe(a)
    returns = pipe.experiment()
    elapsed = time.time() - t0
    sr  = returns['average']['total_reward']
    cmp = returns['average']['compute']
    df_on = getattr(a, 'use_distance_field', False)
    mode  = getattr(a, 'dist_mode', 'sum') if df_on else 'N/A'
    omega = getattr(a, 'dist_omega', 0.0)   if df_on else 0.0
    print(f'  [{tag}] sr={sr:.1f}% compute={cmp:.0f} mode={mode} omega={omega} ({elapsed:.0f}s)')
    return {'tag': tag, 'sr': sr, 'compute': cmp, 'mode': mode, 'omega': omega, 'time': elapsed}


results = []

# ─── Baselines ──────────────────────────────────────────────────────────────
print('=' * 70)
print('Baselines (N=%d)' % N_SAMPLES)

results.append(run_config('base_indist_nodf',
    make_args('bon', use_df=False)))

results.append(run_config('base_ood_nodf',
    make_args('bon', use_df=False, maze_json_dir=MAZE_JSON_DIR)))

# ─── Omega sweep (BON OOD, mode=sum) ────────────────────────────────────────
print()
print('Omega sweep: OOD / BON / sum mode')

for omega in [0.1, 0.3, 0.5, 1.0, 3.0, 5.0]:
    results.append(run_config(f'ood_bon_sum_o{omega}',
        make_args('bon', use_df=True, dist_mode='sum', dist_omega=omega,
                  maze_json_dir=MAZE_JSON_DIR)))

# ─── Mode sweep (BON OOD, omega=1.0) ────────────────────────────────────────
print()
print('Mode sweep: OOD / BON / omega=1.0')

for mode in ['sum', 'weighted', 'monotonic', 'endpoint']:
    results.append(run_config(f'ood_bon_{mode}_o1',
        make_args('bon', use_df=True, dist_mode=mode, dist_omega=1.0,
                  maze_json_dir=MAZE_JSON_DIR)))

# ─── DFS omega sweep (OOD) ──────────────────────────────────────────────────
print()
print('DFS omega sweep: OOD / DFS / sum mode')

for omega in [0.1, 0.3, 0.5, 1.0]:
    results.append(run_config(f'ood_dfs_sum_o{omega}',
        make_args('dfs', use_df=True, dist_mode='sum', dist_omega=omega,
                  maze_json_dir=MAZE_JSON_DIR)))

# ─── Save CSV ───────────────────────────────────────────────────────────────
csv_path = 'outputs/sweep_results.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['tag', 'sr', 'compute', 'mode', 'omega', 'time'])
    writer.writeheader()
    writer.writerows(results)

# ─── Summary ────────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('SWEEP SUMMARY')
print(f"{'tag':<30} {'sr':>6} {'mode':<10} {'omega':>6}")
print('-' * 60)
for r in results:
    print(f"  {r['tag']:<28} {r['sr']:>5.1f}%  {r['mode']:<10} {r['omega']:>5.1f}")
print()
print(f'Results saved to {csv_path}')
