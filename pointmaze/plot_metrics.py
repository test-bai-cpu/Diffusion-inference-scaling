"""
plot_metrics.py
===============
Visualize the separated success / collision / corner-cut / dead-end metrics from
results_detailed.csv (written by run.py). Produces metrics_breakdown.png:

  (a) grouped bars: success vs collision vs corner-cut vs dead-end, per method
  (b) success vs collision scatter (each point a task/variant), per method
  (c) stacked failure attribution among NON-successful episodes

Usage:  python plot_metrics.py [results_detailed.csv] [out.png]
"""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FOC = '#1f6feb'; GREY = '#9aa0a6'; RED = '#d1242f'; GREEN = '#1a7f37'; GOLD = '#d4a72c'
PAL = [FOC, RED, GOLD, GREEN, '#8250df', '#bf8700']


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get('task') == 'average':
                continue
            rows.append(r)
    return rows


def fnum(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError, TypeError):
        return np.nan


def main(csv_path='results_detailed.csv', out='metrics_breakdown.png'):
    rows = load(csv_path)
    if not rows:
        print('no rows'); return
    methods = sorted({r['method'] for r in rows})

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # (a) grouped bars of mean rates per method
    metrics = ['success', 'collision_rate', 'cornercut_rate', 'deadend_frac']
    labels = ['success', 'collision', 'corner-cut', 'dead-end']
    ax = axes[0]
    x = np.arange(len(metrics)); w = 0.8 / max(len(methods), 1)
    for mi, m in enumerate(methods):
        mr = [r for r in rows if r['method'] == m]
        vals = [np.nanmean([fnum(r, k) for r in mr]) for k in metrics]
        ax.bar(x + mi * w, vals, w, label=m, color=PAL[mi % len(PAL)])
    ax.set_xticks(x + w * (len(methods) - 1) / 2); ax.set_xticklabels(labels)
    ax.set_ylabel('mean rate (%)'); ax.set_title('(a) Outcome breakdown by method', fontsize=11)
    ax.legend(fontsize=9)

    # (b) success vs collision scatter
    ax = axes[1]
    for mi, m in enumerate(methods):
        mr = [r for r in rows if r['method'] == m]
        ax.scatter([fnum(r, 'collision_rate') for r in mr],
                   [fnum(r, 'success') for r in mr],
                   s=42, alpha=0.75, color=PAL[mi % len(PAL)], label=m, edgecolor='k', linewidth=0.4)
    ax.set_xlabel('collision rate (%)'); ax.set_ylabel('success (%)')
    ax.set_title('(b) Success vs collision (per task/variant)', fontsize=11); ax.legend(fontsize=9)

    # (c) failure attribution, each episode weighted by its failure mass (100-success)
    ax = axes[2]
    fail_kinds = ['collision_rate', 'cornercut_rate', 'deadend_frac']
    flabels = ['collision', 'corner-cut', 'dead-end']
    bottoms = np.zeros(len(methods))
    for ki, k in enumerate(fail_kinds):
        vals = []
        for m in methods:
            mr = [r for r in rows if r['method'] == m]
            wsum = 0.0; num = 0.0
            for r in mr:
                fw = max(0.0, 100.0 - fnum(r, 'success'))   # failure mass
                sig = fnum(r, k)
                if np.isfinite(fw) and np.isfinite(sig):
                    num += fw * sig; wsum += fw
            vals.append(num / wsum if wsum > 0 else 0.0)
        vals = np.nan_to_num(vals)
        ax.bar(range(len(methods)), vals, bottom=bottoms, label=flabels[ki], color=PAL[ki % len(PAL)])
        bottoms += vals
    ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel('failure-weighted mean signal (%)')
    ax.set_title('(c) Failure attribution (weighted by 100−success)', fontsize=11); ax.legend(fontsize=9)

    fig.suptitle('Instrumented outcome metrics: success separated from collision, corner-cut, dead-end',
                 fontsize=12.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f'saved {out}  ({len(rows)} rows, methods={methods})')


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'results_detailed.csv'
    out = sys.argv[2] if len(sys.argv) > 2 else 'metrics_breakdown.png'
    main(csv_path, out)
