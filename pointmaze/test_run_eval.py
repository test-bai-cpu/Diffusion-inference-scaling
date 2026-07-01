"""
test_run_eval.py — CPU tests for the Step-8 evaluation driver + plotter.
========================================================================
No GPU / diffusion stack: we mock get_pipe/get_args so the driver's arg
construction, level bookkeeping, and CSV schema are exercised end-to-end, and we
feed a synthetic eval_results.csv to plot_degradation to confirm it produces the
figure + report with the right monotonicity/margin arithmetic.

Run:  python test_run_eval.py     (from pointmaze/)
"""
import os, sys, csv, json, types, importlib.util, tempfile
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_MV = os.path.join(_HERE, os.pardir, 'maze_update', 'maze_variants_v2')

_PASS = []
def check(name, cond, detail=""):
    _PASS.append(bool(cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---- load run_eval with search.* mocked so no GPU imports fire ----
def _load_run_eval():
    # stub search package with get_args/get_pipe + Arguments
    sp = types.ModuleType('search'); sp.__path__ = []
    cfgmod = types.ModuleType('search.configs')
    class Arguments:
        def __init__(self):
            self.dataset = ''; self.method = ''; self.device = ''
            self.task = []; self.maze_json_dir = ''; self.maze_variant_idx = 0
            self.seed = 42; self.use_distance_field = False
            self.mcgn_ckpt = ''
    cfgmod.Arguments = Arguments
    sumod = types.ModuleType('search.script_utils')
    _calls = []
    def get_args(a):
        # echo a single-element grid, recording what the driver set
        _calls.append(dict(method=a.method, task=list(a.task),
                           variant=a.maze_variant_idx, seed=a.seed,
                           use_df=getattr(a, 'use_distance_field', False),
                           mcgn=getattr(a, 'mcgn_ckpt', '')))
        return [a]
    def get_pipe(a):
        class P:
            def experiment(self):
                # deterministic synthetic returns keyed off variant/seed
                base = 0.9 - 0.15 * a.maze_variant_idx
                return {a.task[0]: {'total_reward': base, 'success': base,
                                    'collision_rate': 0.05, 'cornercut_rate': 0.02,
                                    'deadend_frac': 0.03, 'stalled_prog': 0.1,
                                    'final_gap': 0.2, 'steps': 300, 'compute': 40.0},
                        'average': {'total_reward': base, 'success': base,
                                    'collision_rate': 0.05, 'cornercut_rate': 0.02,
                                    'deadend_frac': 0.03, 'stalled_prog': 0.1,
                                    'final_gap': 0.2, 'steps': 300, 'compute': 40.0}}
        return P()
    sumod.get_args = get_args; sumod.get_pipe = get_pipe
    sys.modules['search'] = sp
    sys.modules['search.configs'] = cfgmod
    sys.modules['search.script_utils'] = sumod
    spec = importlib.util.spec_from_file_location('_run_eval',
                                                  os.path.join(_HERE, 'run_eval.py'))
    m = importlib.util.module_from_spec(spec); sys.modules['_run_eval'] = m
    spec.loader.exec_module(m)
    return m, _calls


def test_level_variants():
    print("test_level_variants")
    m, _ = _load_run_eval()
    lv = m.level_variants(_MV, 1, [0, 1, 2, 3])
    check("returns 4 levels", len(lv) == 4, str([l for l, _ in lv]))
    # each level maps to a list of (variant_idx, difficulty)
    L0, vs0 = lv[0]
    check("level 0 first", L0 == 0)
    check("variants carry (idx,diff)", len(vs0[0]) == 2 and isinstance(vs0[0][0], int))
    # difficulty should be non-decreasing across levels (calibrated ladder)
    firstdiff = [vs[0][1] for _, vs in lv]
    check("difficulty non-decreasing across levels", all(
        firstdiff[i] <= firstdiff[i+1] + 1e-9 for i in range(len(firstdiff)-1)),
        str([round(x, 2) for x in firstdiff]))


def test_config_fields():
    print("test_config_fields")
    m, _ = _load_run_eval()
    check("4 configs defined", set(m.CONFIGS) == {'dfs', 'field', 'mafgs', 'mafgs+mcgn'})
    check("field toggles distance field", m.CONFIGS['field']['fields'].get('use_distance_field') is True)
    check("mafgs+mcgn dispatches mafgs method", m.CONFIGS['mafgs+mcgn']['method'] == 'mafgs')
    check("mafgs+mcgn flags mcgn need", m.CONFIGS['mafgs+mcgn']['fields'].get('_needs_mcgn') is True)


def test_build_args_toggles():
    print("test_build_args_toggles")
    m, calls = _load_run_eval()
    # field config must set use_distance_field on the args passed to get_args
    m.build_args(m.CONFIGS['field'], 'pointmaze-giant-navigate-v0', 'dfs', 'cpu',
                 1, _MV, 0, 7, 'ckpt.pt', m.CONFIGS['field']['fields'])
    check("field sets use_distance_field", calls[-1]['use_df'] is True)
    check("seed forwarded", calls[-1]['seed'] == 7)
    # mafgs+mcgn must forward the ckpt path
    m.build_args(m.CONFIGS['mafgs+mcgn'], 'pointmaze-giant-navigate-v0', 'mafgs', 'cpu',
                 1, _MV, 0, 0, '/tmp/x.pt', m.CONFIGS['mafgs+mcgn']['fields'])
    check("mcgn ckpt forwarded", calls[-1]['mcgn'] == '/tmp/x.pt')
    # dfs baseline must NOT set distance field or ckpt
    m.build_args(m.CONFIGS['dfs'], 'pointmaze-giant-navigate-v0', 'dfs', 'cpu',
                 1, _MV, 0, 0, '/tmp/x.pt', m.CONFIGS['dfs']['fields'])
    check("dfs leaves distance field off", calls[-1]['use_df'] is False)
    check("dfs leaves mcgn off", calls[-1]['mcgn'] == '')


def test_run_writes_csv():
    print("test_run_writes_csv")
    m, _ = _load_run_eval()
    td = tempfile.mkdtemp()
    out = os.path.join(td, 'eval.csv')
    ns = types.SimpleNamespace(
        dataset='pointmaze-giant-navigate-v0',
        methods=['dfs', 'mafgs'], tasks=[1], levels=[0, 1],
        seeds=[0, 1], maze_json_dir=_MV, mcgn_ckpt='ckpt.pt',
        all_variants=False, device='cpu', out=out)
    m.run(ns)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    # 2 methods x 1 task x 2 levels x 1 variant x 2 seeds = 8 rows
    check("row count = 8", len(rows) == 8, str(len(rows)))
    check("has book+metric columns", 'difficulty' in rows[0] and 'success' in rows[0]
          and 'level' in rows[0])
    check("levels present", sorted({r['level'] for r in rows}) == ['0', '1'])
    check("both configs present", {r['config'] for r in rows} == {'dfs', 'mafgs'})


def test_plot_degradation():
    print("test_plot_degradation")
    # synthesize a monotone-degradation CSV and confirm the plotter + report run
    td = tempfile.mkdtemp()
    csvp = os.path.join(td, 'eval_results.csv')
    BOOK = ['dataset', 'method', 'config', 'task', 'level', 'variant_idx',
            'difficulty', 'seed']
    MET = ['total_reward', 'success', 'collision_rate', 'cornercut_rate',
           'deadend_frac', 'stalled_prog', 'final_gap', 'steps', 'compute']
    rng = np.random.default_rng(0)
    with open(csvp, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(BOOK + MET)
        for cfg, boost in [('dfs', 0.0), ('field', 0.08), ('mafgs', 0.18), ('mafgs+mcgn', 0.24)]:
            for task in [1, 2, 3]:
                for L in [0, 1, 2, 3]:
                    for seed in [0, 1, 2]:
                        succ = np.clip(0.92 - 0.18 * L + boost + rng.normal(0, 0.03), 0, 1)
                        coll = np.clip(0.03 + 0.02 * L - 0.5 * boost, 0, 1)
                        dead = np.clip(0.02 + 0.03 * L - 0.4 * boost, 0, 1)
                        w.writerow(['giant', cfg.split('+')[0] if '+' not in cfg else 'mafgs',
                                    cfg, task, L, 0, round(0.1 + 0.2 * L, 3), seed,
                                    succ, succ, coll, 0.02, dead, 0.1, 0.2, 300, 40.0])
    pd_spec = importlib.util.spec_from_file_location('_plotdeg',
                                                     os.path.join(_HERE, 'plot_degradation.py'))
    pdm = importlib.util.module_from_spec(pd_spec); sys.modules['_plotdeg'] = pdm
    pd_spec.loader.exec_module(pdm)
    prefix = os.path.join(td, 'deg')
    pdm.main(csvp, prefix)
    check("figure written", os.path.exists(prefix + '_curves.png'))
    check("report written", os.path.exists(prefix + '_report.md'))
    rep = open(prefix + '_report.md').read()
    check("report has monotonicity section", 'Monotonicity' in rep)
    check("report has margin section", 'Margin at hardest level' in rep)
    check("report has transfer section", 'Map transfer' in rep)
    # sanity: spearman helper is negative for a monotone-decreasing ladder
    rho = pdm._spearman([0, 1, 2, 3], [0.9, 0.7, 0.5, 0.3])
    check("spearman negative on decreasing series", rho < -0.9, f"rho={rho:.3f}")


if __name__ == '__main__':
    for fn in (test_level_variants, test_config_fields, test_build_args_toggles,
               test_run_writes_csv, test_plot_degradation):
        fn()
    n = len(_PASS); k = sum(_PASS)
    print(f"\n{k}/{n} checks passed")
    sys.exit(0 if k == n else 1)
