#!/usr/bin/env python3
"""HyperAD v15: line-block bootstrap CIs and per-line mechanism traces.

No method change. Re-runs only the ERX family (official ERX unmodified, frozen,
delayed Trim05/Trim10 cc, Shadow cc) with the frozen protocol (line z-scores,
config_detection.yaml ERX settings) and saves:

* bootstrap_ci.csv      moving-block bootstrap over scan lines (paired: every
                        method is evaluated on the same resampled lines).
                        Point estimates equal Table II (common causal window).
* per_line_trace_seed*.csv  per scan line: Shadow action, GT anomaly fraction
                        (POST-HOC), unlabeled drift proxy, and rolling skip rate.
* mechanism_seed*.png   three-panel figure of the above.

Labels enter only metrics and the post-hoc columns. Projection seeds are not
independent scenes; the bootstrap resamples lines within one scene and seed,
so its CIs describe within-scene variability only.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from baseline_v3 import stable_md, normalise_line, reference_repo
from run_shadow import prepare_warmup, save_csv
from run_v12 import GatedPolicy, shadow_chooser, fixed_chooser
from run_v13 import build_official, simulate, OFFICIAL

METHODS = ('official_erx', 'frozen', 'delayed_trim05_cc', 'delayed_trim10_cc', 'shadow_cc')


# --------------------------------------------------------------------------
# Fast weighted metrics on a presorted score vector (ties ignored; scores are
# continuous line z-scores in the evaluated window).
# --------------------------------------------------------------------------
class Sorted:
    def __init__(self, scores, y, line_idx):
        order = np.argsort(-scores, kind='stable')
        self.y = y[order].astype(np.float64)
        self.line = line_idx[order]

    def metrics(self, line_w):
        w = line_w[self.line]
        wp = w * self.y
        tp = np.cumsum(wp)
        fp = np.cumsum(w - wp)
        P, N = tp[-1], fp[-1]
        tpr, fpr = tp / P, fp / N
        auroc = float(np.trapezoid(np.r_[0., tpr], np.r_[0., fpr])) if hasattr(np, 'trapezoid') \
            else float(np.trapz(np.r_[0., tpr], np.r_[0., fpr]))
        den = tp + fp
        prec = np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
        ap = float(np.sum(prec * wp) / P)
        k = int(np.searchsorted(tpr, .90, side='left'))
        return auroc, ap, float(fpr[min(k, len(fpr) - 1)])


def block_weights(n_lines, block, rng):
    n_blocks = int(np.ceil(n_lines / block))
    starts = rng.integers(0, n_lines - block + 1, size=n_blocks)
    w = np.zeros(n_lines)
    for s in starts:
        w[s:s + block] += 1
    # trim the overshoot uniformly so the total weight stays n_lines
    return w * (n_lines / w.sum())


# --------------------------------------------------------------------------
def run_seed(x, mask, seed, cfg, module, warmup, dims, momentum, core_fraction, penalty, tail):
    h, w, b = x.shape
    proj, mu0, cov0 = prepare_warmup(x, seed, warmup, dims, momentum)
    pols = {'delayed_trim05_cc': GatedPolicy(mu0, cov0, momentum, fixed_chooser('trim05'), True, ('trim05',)),
            'delayed_trim10_cc': GatedPolicy(mu0, cov0, momentum, fixed_chooser('trim10'), True, ('trim10',)),
            'shadow_cc': GatedPolicy(mu0, cov0, momentum, shadow_chooser(core_fraction, penalty), True)}
    maps = {k: np.zeros((h, w)) for k in METHODS}
    drift = np.full(w, np.nan)
    for t in range(warmup, w):
        z = np.asarray(proj.transform(np.asarray(x[:, t, :], dtype=np.float64)), dtype=np.float64)
        raw_frozen = stable_md(z, mu0, cov0)
        maps['frozen'][:, t] = normalise_line(raw_frozen)
        drift[t] = float(np.median(raw_frozen))  # unlabeled: distance of the line from the warm-up model
        for k, p in pols.items():
            maps[k][:, t] = normalise_line(p.step(z))
    np.random.seed(seed)
    det = build_official('erx', cfg, h, b, module)
    det.random_projector = proj
    maps['official_erx'], filled, _ = simulate(det, x)
    actions = np.full(w, '', dtype=object)
    acts = pols['shadow_cc'].actions  # first decision happens at line warmup+1
    actions[warmup + 1: warmup + 1 + len(acts)] = acts
    cols = np.arange(warmup, w - tail)
    if not filled[cols].all():
        raise AssertionError('Official ERX did not score the whole evaluation window')
    return maps, actions, drift, cols


def evaluate(maps, mask, cols, n_boot, block, rng):
    h = mask.shape[0]
    y = np.asarray(mask[:, cols] != 0).ravel()
    line_idx = np.tile(np.arange(len(cols)), h)  # ravel of [h, cols] is row-major: pixel-major
    srt = {m: Sorted(np.asarray(maps[m][:, cols]).ravel(), y, line_idx) for m in METHODS}
    ones = np.ones(len(cols))
    point = {m: srt[m].metrics(ones) for m in METHODS}
    # sanity: fast metrics equal sklearn on the full sample
    s = np.asarray(maps['shadow_cc'][:, cols]).ravel()
    assert abs(point['shadow_cc'][0] - roc_auc_score(y, s)) < 1e-6
    assert abs(point['shadow_cc'][1] - average_precision_score(y, s)) < 1e-6
    best_fixed = max(('delayed_trim05_cc', 'delayed_trim10_cc'), key=lambda m: point[m][1])
    boot = {m: [] for m in METHODS}
    for _ in range(n_boot):
        wts = block_weights(len(cols), block, rng)
        for m in METHODS:
            boot[m].append(srt[m].metrics(wts))
    boot = {m: np.array(v) for m, v in boot.items()}
    return point, boot, best_fixed


def write_rows(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=['seed', 'quantity', 'estimate', 'ci_low', 'ci_high',
                                           'p_boot_le_0', 'p_boot_ge_0'])
        wr.writeheader(); wr.writerows(rows)


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def plot_mechanism(trace, dest, title, window=25):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines = np.array([r['line'] for r in trace])
    gt = np.array([r['gt_anomaly_fraction_POSTHOC'] for r in trace])
    drift = np.array([r['drift_proxy_unlabeled'] for r in trace])
    skip = np.array([r['rolling_skip_rate'] for r in trace])
    act = [r['shadow_action'] for r in trace]
    fig, ax = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True)
    ax[0].fill_between(lines, gt, step='mid', color='#c0392b', alpha=.75)
    ax[0].set_ylabel('GT anomaly\nfraction (post hoc)', fontsize=8)
    ax[1].plot(lines, drift, color='#555', lw=.8)
    ax[1].set_ylabel('Median MD to\nwarm-up model', fontsize=8)
    ax[2].plot(lines, skip, color='#2f6f9f', lw=1.1)
    for a, col in (('trim05', '#d9a441'), ('trim10', '#2e8b57')):
        xs = [l for l, q in zip(lines, act) if q == a]
        ax[2].plot(xs, [-.06] * len(xs), '|', color=col, ms=5, label=a)
    ax[2].set_ylim(-.12, 1.05)
    ax[2].set_ylabel(f'Shadow skip rate\n(rolling {window} lines)', fontsize=8)
    ax[2].set_xlabel('Scan line', fontsize=9)
    ax[2].legend(fontsize=7, frameon=False, loc='upper right', ncol=2)
    for a in ax:
        a.spines[['top', 'right']].set_visible(False); a.tick_params(labelsize=8)
    ax[0].set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(dest, dpi=200); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hyperad-root', required=True)
    p.add_argument('--hsi', required=True)
    p.add_argument('--mask', required=True)
    p.add_argument('--dataset-name', default='Beach')
    p.add_argument('--reverse', action='store_true')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--core-fraction', type=float, default=.80)
    p.add_argument('--penalty', type=float, default=.05)
    p.add_argument('--tail', type=int, default=49, help='Columns excluded at the end (RX Baseline offset) to match Table II')
    p.add_argument('--n-boot', type=int, default=1000)
    p.add_argument('--block', type=int, default=50, help='Bootstrap block length in scan lines')
    p.add_argument('--output', default='results_v15')
    a = p.parse_args()
    import yaml, importlib
    root = reference_repo(a.hyperad_root)
    sys.path.insert(0, str(root))
    cfg = yaml.safe_load((root / 'test_detection' / 'config_detection.yaml').read_text(encoding='utf-8'))
    e = cfg['erx']
    warmup, dims, momentum = int(e['buffer_len']), int(e['projected_dimensions']), float(e['momentum'])
    module = importlib.import_module('detectors.erx')
    x = np.load(a.hsi, mmap_mode='r', allow_pickle=False)
    mask = np.load(a.mask, mmap_mode='r', allow_pickle=False)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if a.reverse:
        x, mask = x[:, ::-1, :], mask[:, ::-1]
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    (out / 'run_config.json').write_text(json.dumps({**vars(a), 'shape': list(x.shape), 'warmup': warmup,
        'dims': dims, 'momentum': momentum}, indent=2), encoding='utf-8')
    rng = np.random.default_rng(20261018)
    rows = []
    with threadpool_limits(limits=1):
        for seed in a.seeds:
            tic = time.perf_counter()
            maps, actions, drift, cols = run_seed(x, mask, seed, cfg, module, warmup, dims, momentum,
                                                  a.core_fraction, a.penalty, a.tail)
            point, boot, best = evaluate(maps, mask, cols, a.n_boot, a.block, rng)
            for m in METHODS:
                for j, k in enumerate(('auroc', 'ap', 'fpr90')):
                    lo, hi = ci(boot[m][:, j])
                    rows.append({'seed': seed, 'quantity': f'{m}_{k}', 'estimate': point[m][j], 'ci_low': lo, 'ci_high': hi})
            for ref in ('official_erx', best):
                for j, k in enumerate(('auroc', 'ap', 'fpr90')):
                    d = boot['shadow_cc'][:, j] - boot[ref][:, j]
                    lo, hi = ci(d)
                    rows.append({'seed': seed, 'quantity': f'delta_{k}_shadow_minus_{ref}',
                                 'estimate': point['shadow_cc'][j] - point[ref][j], 'ci_low': lo, 'ci_high': hi,
                                 'p_boot_le_0' if k != 'fpr90' else 'p_boot_ge_0':
                                     float(np.mean(d <= 0)) if k != 'fpr90' else float(np.mean(d >= 0))})
            # per-line trace
            gt = np.asarray(mask != 0).mean(axis=0)
            skip_flag = np.array([1. if q == 'skip' else (0. if q else np.nan) for q in actions])
            roll = np.full(len(skip_flag), np.nan)
            for t in range(len(skip_flag)):
                seg = skip_flag[max(0, t - 24):t + 1]
                seg = seg[np.isfinite(seg)]
                if len(seg): roll[t] = seg.mean()
            trace = [{'line': int(t), 'shadow_action': actions[t], 'gt_anomaly_fraction_POSTHOC': float(gt[t]),
                      'drift_proxy_unlabeled': float(drift[t]), 'rolling_skip_rate': float(roll[t])}
                     for t in range(warmup + 1, x.shape[1])]
            save_csv(out / f'per_line_trace_seed{seed}.csv', trace)
            plot_mechanism(trace, out / f'mechanism_seed{seed}.png',
                           f'{a.dataset_name}{" (reversed)" if a.reverse else ""}, projection seed {seed}')
            pos = np.array([r['gt_anomaly_fraction_POSTHOC'] > 0 for r in trace])
            sk = np.array([r['shadow_action'] == 'skip' for r in trace])
            dr = np.array([r['drift_proxy_unlabeled'] for r in trace])
            rows.append({'seed': seed, 'quantity': 'skip_rate_lines_with_gt_POSTHOC', 'estimate': float(sk[pos].mean()) if pos.any() else float('nan')})
            rows.append({'seed': seed, 'quantity': 'skip_rate_lines_without_gt_POSTHOC', 'estimate': float(sk[~pos].mean())})
            rows.append({'seed': seed, 'quantity': 'spearman_skip_vs_drift_unlabeled',
                         'estimate': float(spearmanr(sk.astype(float), dr).statistic)})
            print(f'seed={seed} done in {time.perf_counter() - tic:.1f}s; best fixed={best}', flush=True)
            write_rows(out / 'bootstrap_ci.csv', rows)
    print(f'\n{a.dataset_name}: 95% line-block bootstrap CIs (block={a.block}, B={a.n_boot})')
    for r in rows:
        if r['quantity'].startswith('delta') or r['quantity'].startswith('skip') or r['quantity'].startswith('spearman'):
            ci_txt = f" [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]" if 'ci_low' in r else ''
            print(f"  seed={r['seed']} {r['quantity']:48s} {r['estimate']:+.4f}{ci_txt}")
    print('Outputs saved:', out.resolve())


if __name__ == '__main__':
    main()
