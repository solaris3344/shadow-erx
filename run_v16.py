#!/usr/bin/env python3
"""HyperAD v16: rival gates, (rho, lambda) sensitivity and end-to-end timing.

Method and protocol are unchanged (line z-scores, config_detection.yaml ERX
settings, common causal window = columns warmup .. W-1-tail).

1. Rival update rules, all with the same one-line delay as the gate:
   * delayed_huber_q90   bounded-influence weights w = min(1, c / md), c = 90th
                         percentile of the line (as baseline_v3 'soft Huber').
   * ranksum_gate        line-level adaptation of a hypothesis-test update rule
                         (Ma et al., 2019): the candidate Trim10 update built from
                         line t-1 is rejected (skip) if a one-sided Mann-Whitney U
                         test says the distances of line t-1 are stochastically
                         larger than those of the last R accepted lines
                         (alpha = 0.01, R = 10; fixed a priori, not tuned).
2. Sensitivity of shadow_cc to the core fraction rho and penalty lambda
   (rho in {0.6,0.7,0.8,0.9}, lambda in {0,0.02,0.05,0.1,0.2}).
   POST HOC on all scenes: report, do not re-tune.
3. End-to-end single-thread time per line including float64 conversion and the
   random projection, for official ERX and every ERX-family rule (seed 0).

Labels are used only in metric code. Projection seeds are not independent scenes.
"""
from __future__ import annotations
import argparse, csv, json, platform, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits
from baseline_v3 import stable_md, normalise_line, reference_repo
from run_shadow import prepare_warmup
from run_v12 import GatedPolicy, shadow_chooser, fixed_chooser, proposal
from run_v13 import build_official, simulate

RHOS = (0.6, 0.7, 0.8, 0.9)
LAMBDAS = (0.0, 0.02, 0.05, 0.1, 0.2)
DEFAULT = (0.8, 0.05)


# --------------------------------------------------------------------------
# Rival rules
# --------------------------------------------------------------------------
def huber_proposal(z, mu, cov, momentum, q=0.90, reg=1e-5):
    d = z.shape[1]
    md = stable_md(z, mu, cov, reg)
    c = float(np.quantile(md, q))
    w = np.minimum(1.0, c / np.maximum(md, 1e-12))
    tot = float(w.sum())
    m = np.average(z, axis=0, weights=w)
    zc = z - m
    den = tot - float((w ** 2).sum()) / tot
    C = (zc.T * w) @ zc / max(den, 1.) + np.eye(d) * reg
    return (1 - momentum) * mu + momentum * m, (1 - momentum) * cov + momentum * C


class DelayedHuber:
    def __init__(self, mu, cov, momentum):
        self.mu, self.cov, self.m, self.pending = mu.copy(), cov.copy(), momentum, None

    def step(self, z):
        raw = stable_md(z, self.mu, self.cov)
        if self.pending is not None:
            self.mu, self.cov = self.pending
        self.pending = huber_proposal(z, self.mu, self.cov, self.m)
        return raw


class RankSumGate:
    """Skip the Trim10 candidate from line t-1 if its distances are stochastically
    larger than those of recently accepted lines (one-sided Mann-Whitney U)."""
    def __init__(self, mu, cov, momentum, alpha=0.01, memory=10, max_pool=5000, seed=0):
        self.mu, self.cov, self.m = mu.copy(), cov.copy(), momentum
        self.alpha, self.memory, self.max_pool = alpha, memory, max_pool
        self.pending, self.prev_raw, self.pool, self.actions = None, None, [], []
        self.rng = np.random.default_rng(seed)

    def step(self, z):
        raw = stable_md(z, self.mu, self.cov)
        if self.pending is not None:
            accept = True
            if self.pool:
                ref = np.concatenate(self.pool)
                if len(ref) > self.max_pool:
                    ref = self.rng.choice(ref, self.max_pool, replace=False)
                p = mannwhitneyu(self.prev_raw, ref, alternative='greater').pvalue
                accept = p >= self.alpha
            if accept:
                self.mu, self.cov = self.pending
                self.pool = (self.pool + [self.prev_raw])[-self.memory:]
            self.actions.append('trim10' if accept else 'skip')
        self.prev_raw = raw
        self.pending = proposal(z, self.mu, self.cov, .10, self.m, True)
        return raw


# --------------------------------------------------------------------------
def metrics(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    k = min(int(np.searchsorted(tpr, .90, side='left')), len(fpr) - 1)
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s)), float(fpr[k])


def project_all(x, proj, warmup):
    w = x.shape[1]
    return [np.asarray(proj.transform(np.asarray(x[:, t, :], dtype=np.float64)), dtype=np.float64)
            for t in range(warmup, w)]


def run_policy(policy, Z):
    return np.stack([normalise_line(policy.step(z)) for z in Z], axis=1)  # [pixels, lines]


def make_policies(mu0, cov0, mom, seed):
    return {
        'delayed_trim05_cc': GatedPolicy(mu0, cov0, mom, fixed_chooser('trim05'), True, ('trim05',)),
        'delayed_trim10_cc': GatedPolicy(mu0, cov0, mom, fixed_chooser('trim10'), True, ('trim10',)),
        'delayed_huber_q90': DelayedHuber(mu0, cov0, mom),
        'ranksum_gate': RankSumGate(mu0, cov0, mom, seed=seed),
        'shadow_cc': GatedPolicy(mu0, cov0, mom, shadow_chooser(*DEFAULT), True),
    }


def time_end_to_end(x, proj, mu0, cov0, mom, warmup, det):
    """ms/line from raw line to emitted score, single thread; line fetch excluded."""
    w = x.shape[1]
    out = {}

    def ours(factory):
        pol = factory()
        ts = []
        for t in range(warmup, w):
            line = np.array(x[:, t, :])  # fetch (and page-in) outside the timer
            tic = time.perf_counter()
            z = np.asarray(proj.transform(np.asarray(line, dtype=np.float64)), dtype=np.float64)
            normalise_line(pol.step(z))
            ts.append(time.perf_counter() - tic)
        return ts

    class Frozen:
        def step(self, z):
            return stable_md(z, mu0, cov0)

    facs = {'frozen': Frozen,
            **{k: (lambda k=k: make_policies(mu0, cov0, mom, 0)[k]) for k in
               ('delayed_trim10_cc', 'delayed_huber_q90', 'ranksum_gate', 'shadow_cc')}}
    for k, f in facs.items():
        out[k] = ours(f)
    ts = []
    for t in range(w):
        line = np.array(x[:, t, :])
        tic = time.perf_counter(); det.forward(line); dt = time.perf_counter() - tic
        if t >= warmup:
            ts.append(dt)
    out['official_erx'] = ts
    return {k: {'ms_median': 1000 * float(np.median(v)), 'ms_mean': 1000 * float(np.mean(v)),
                'lines_per_s_mean': float(len(v) / np.sum(v))} for k, v in out.items()}


def cpu_name():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0')
        return winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip()
    except Exception:
        pass
    try:
        for line in open('/proc/cpuinfo'):
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or 'unknown'


def heatmap(grid, ref, dest, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (j, name, cmap) in zip(axes, ((0, 'AUROC', 'Blues'), (1, 'AP', 'Blues'), (2, 'FPR90', 'Blues_r'))):
        M = np.array([[grid[(r, l)][j] for l in LAMBDAS] for r in RHOS])
        im = ax.imshow(M, cmap=cmap, aspect='auto')
        lo, hi = float(M.min()), float(M.max())
        for a in range(len(RHOS)):
            for b in range(len(LAMBDAS)):
                better = (M[a, b] < ref[j]) if j == 2 else (M[a, b] > ref[j])
                frac = 0.5 if hi == lo else (M[a, b] - lo) / (hi - lo)
                dark = frac < 0.45 if j == 2 else frac > 0.55
                ax.text(b, a, f'{M[a, b]:.3f}' + ('' if better else '*'), ha='center', va='center', fontsize=7,
                        color='white' if dark else 'black',
                        fontweight='bold' if (RHOS[a], LAMBDAS[b]) == DEFAULT else 'normal')
        ax.set_xticks(range(len(LAMBDAS)), [str(l) for l in LAMBDAS], fontsize=7)
        ax.set_yticks(range(len(RHOS)), [str(r) for r in RHOS], fontsize=7)
        ax.set_xlabel('penalty λ', fontsize=8); ax.set_ylabel('core fraction ρ', fontsize=8)
        ax.set_title(f'{name} (Trim10 = {ref[j]:.3f})', fontsize=8)
    fig.suptitle(title + '   bold = preregistered setting, * = not better than Delayed Trim10', fontsize=8)
    fig.tight_layout(); fig.savefig(dest, dpi=200); plt.close(fig)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hyperad-root', required=True)
    p.add_argument('--hsi', required=True)
    p.add_argument('--mask', required=True)
    p.add_argument('--dataset-name', default='Beach')
    p.add_argument('--reverse', action='store_true')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--tail', type=int, default=49)
    p.add_argument('--skip-sensitivity', action='store_true')
    p.add_argument('--skip-timing', action='store_true')
    p.add_argument('--output', default='results_v16')
    a = p.parse_args()
    import yaml, importlib
    root = reference_repo(a.hyperad_root)
    sys.path.insert(0, str(root))
    cfg = yaml.safe_load((root / 'test_detection' / 'config_detection.yaml').read_text(encoding='utf-8'))
    e = cfg['erx']
    warmup, dims, mom = int(e['buffer_len']), int(e['projected_dimensions']), float(e['momentum'])
    erx_mod = importlib.import_module('detectors.erx')
    x = np.load(a.hsi, mmap_mode='r', allow_pickle=False)
    mask = np.load(a.mask, mmap_mode='r', allow_pickle=False)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if a.reverse:
        x, mask = x[:, ::-1, :], mask[:, ::-1]
    h, w, b = x.shape
    label = a.dataset_name + (' (reversed)' if a.reverse else '')
    cols = np.arange(warmup, w - a.tail)
    sl = slice(0, len(cols))  # ours start at column warmup
    y = np.asarray(mask[:, cols] != 0).ravel()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    meta = {**vars(a), 'shape': list(x.shape), 'warmup': warmup, 'dims': dims, 'momentum': mom,
            'cpu': cpu_name(), 'python': platform.python_version(), 'numpy': np.__version__,
            'rival_settings': {'huber_q': 0.90, 'ranksum_alpha': 0.01, 'ranksum_memory': 10},
            'grid': {'rho': RHOS, 'lambda': LAMBDAS, 'preregistered': DEFAULT}}
    (out / 'run_config.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'V16 {label} shape={x.shape} seeds={a.seeds} CPU={meta["cpu"]}', flush=True)
    rows, sens, actions = [], [], []
    with threadpool_limits(limits=1):
        for seed in a.seeds:
            tic = time.perf_counter()
            proj, mu0, cov0 = prepare_warmup(x, seed, warmup, dims, mom)
            Z = project_all(x, proj, warmup)
            pols = make_policies(mu0, cov0, mom, seed)
            for k, pol in pols.items():
                s = run_policy(pol, Z)[:, sl]
                rows.append({'seed': seed, 'method': k, **dict(zip(('auroc', 'ap', 'fpr90'), metrics(y, s.ravel())))})
            for k in ('ranksum_gate', 'shadow_cc'):
                acts = pols[k].actions
                actions.append({'seed': seed, 'method': k, 'skip_rate': float(np.mean([q == 'skip' for q in acts]))})
            np.random.seed(seed)
            det = build_official('erx', cfg, h, b, erx_mod)
            det.random_projector = proj
            m, _, _ = simulate(det, x)
            rows.append({'seed': seed, 'method': 'official_erx',
                         **dict(zip(('auroc', 'ap', 'fpr90'), metrics(y, m[:, cols].ravel())))})
            if not a.skip_sensitivity:
                for rho in RHOS:
                    for lam in LAMBDAS:
                        pol = GatedPolicy(mu0, cov0, mom, shadow_chooser(rho, lam), True)
                        s = run_policy(pol, Z)[:, sl]
                        sens.append({'seed': seed, 'rho': rho, 'lambda': lam,
                                     **dict(zip(('auroc', 'ap', 'fpr90'), metrics(y, s.ravel()))),
                                     'skip_rate': float(np.mean([q == 'skip' for q in pol.actions]))})
            if not a.skip_timing and seed == a.seeds[0]:
                np.random.seed(seed)
                det = build_official('erx', cfg, h, b, erx_mod)
                det.random_projector = proj
                timing = time_end_to_end(x, proj, mu0, cov0, mom, warmup, det)
                write_csv(out / 'timing_end_to_end.csv',
                          [{'method': k, **v, 'cpu': meta['cpu']} for k, v in timing.items()])
            print(f'seed={seed} done in {time.perf_counter() - tic:.1f}s', flush=True)
            write_csv(out / 'rival_results_per_seed.csv', rows)
            write_csv(out / 'sensitivity_per_seed.csv', sens)
    # summaries
    summ = []
    for k in dict.fromkeys(r['method'] for r in rows):
        rs = [r for r in rows if r['method'] == k]
        summ.append({'method': k, **{m: float(np.mean([r[m] for r in rs])) for m in ('auroc', 'ap', 'fpr90')},
                     'skip_rate': float(np.mean([q['skip_rate'] for q in actions if q['method'] == k]))
                     if any(q['method'] == k for q in actions) else float('nan')})
    write_csv(out / 'rival_summary.csv', summ)
    L = [f'HyperAD v16 ({label}); line z-scores, common causal window, mean of {len(a.seeds)} seeds', '',
         f"{'method':22s} {'AUROC':>7s} {'AP':>7s} {'FPR90':>7s} {'skip':>6s}"]
    for r in summ:
        L.append(f"{r['method']:22s} {r['auroc']:7.4f} {r['ap']:7.4f} {r['fpr90']:7.4f} {r['skip_rate']:6.3f}")
    if sens:
        grid = {}
        for rho in RHOS:
            for lam in LAMBDAS:
                rs = [r for r in sens if r['rho'] == rho and r['lambda'] == lam]
                grid[(rho, lam)] = tuple(float(np.mean([r[m] for r in rs])) for m in ('auroc', 'ap', 'fpr90', 'skip_rate'))
        ref = next(r for r in summ if r['method'] == 'delayed_trim10_cc')
        refv = (ref['auroc'], ref['ap'], ref['fpr90'])
        heatmap(grid, refv, out / 'sensitivity_heatmap.png', f'{label}: shadow_cc sensitivity (POST HOC)')
        n_auc = sum(v[0] > refv[0] for v in grid.values()); n_ap = sum(v[1] > refv[1] for v in grid.values())
        n_f = sum(v[2] < refv[2] for v in grid.values())
        aps = [v[1] for v in grid.values()]
        L += ['', 'SENSITIVITY (POST HOC; preregistered setting rho=0.8, lambda=0.05):',
              f'  settings better than Delayed Trim10: AUROC {n_auc}/20, AP {n_ap}/20, FPR90 {n_f}/20',
              f'  AP range over grid: {min(aps):.4f} .. {max(aps):.4f} (Trim10 {refv[1]:.4f})',
              f"  skip-rate range: {min(v[3] for v in grid.values()):.3f} .. {max(v[3] for v in grid.values()):.3f}"]
        write_csv(out / 'sensitivity_summary.csv',
                  [{'rho': r, 'lambda': l, 'auroc': v[0], 'ap': v[1], 'fpr90': v[2], 'skip_rate': v[3]}
                   for (r, l), v in grid.items()])
    if (out / 'timing_end_to_end.csv').exists():
        with (out / 'timing_end_to_end.csv').open(encoding='utf-8') as f:
            tr = list(csv.DictReader(f))
        L += ['', f'END-TO-END TIMING (seed {a.seeds[0]}, single thread, projection included; CPU: {meta["cpu"]}):']
        for r in tr:
            L.append(f"  {r['method']:22s} median {float(r['ms_median']):7.3f} ms/line, "
                     f"mean {float(r['ms_mean']):7.3f} ms/line, {float(r['lines_per_s_mean']):8.0f} lines/s")
    (out / 'v16_report.txt').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))
    print('Outputs saved:', out.resolve())


if __name__ == '__main__':
    main()
