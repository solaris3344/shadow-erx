#!/usr/bin/env python3
"""HyperAD v12: protocol corrections and confound controls for Shadow.

This is NOT a new detector. It answers three reviewer-level questions about
the existing Shadow gate, on scenes that are already development data:

 1. Normalisation protocol: every method is reported with raw Mahalanobis
    scores AND with per-line z-scored scores (the HyperAD detection setting).
    ERX_original is run separately in each mode with its own flag.
 2. Trimmed-covariance consistency: every trimmed update exists in two forms,
    uncorrected (identical to v6..v11) and with the Gaussian consistency
    factor c = keep / F_{chi2(d+2)}(F^-1_{chi2(d)}(keep)).
 3. Matched-rate gate control: Shadow's own action sequence is randomly
    permuted (exact same skip/trim05/trim10 counts, timing destroyed) and
    replayed. If Shadow does not beat its permuted copies, its gain is a
    rate effect, not a timing/decision effect.

All model decisions are label-free. Labels are read only in metric code.
A Pareto table (AP up, AUROC up, FPR90 down) is produced per mode.
Projection seeds are NOT independent scenes. Do not tune anything here.
"""
from __future__ import annotations
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits
from baseline_v3 import stable_md, normalise_line, reference_repo, PrequentialERX
from run_shadow import (ACTIONS, TRIM, prepare_warmup, robust_predictive_cost,
                        save_csv, proposal as proposal_v6)

MODES = ('raw', 'line_z')


# --------------------------------------------------------------------------
# Update rules
# --------------------------------------------------------------------------
def consistency_factor(d: int, keep: float) -> float:
    """Gaussian consistency factor for a covariance of the `keep` fraction
    with the smallest Mahalanobis distances (Croux & Haesbroeck style)."""
    if not 0 < keep <= 1:
        raise ValueError('keep must lie in (0,1]')
    if keep == 1:
        return 1.0
    q = chi2.ppf(keep, d)
    return float(keep / chi2.cdf(q, d + 2))


def proposal(z, mu, cov, fraction, momentum, consistency=False, reg=1e-5):
    """v6 proposal; with consistency=True the kept covariance is rescaled."""
    if not consistency:
        return proposal_v6(z, mu, cov, fraction, momentum, reg)
    d = z.shape[1]
    scores = stable_md(z, mu, cov, reg)
    k = min(len(scores), max(d + 2, int(round(len(scores) * (1. - fraction)))))
    kept = z[np.argsort(scores, kind='stable')[:k]]
    m = kept.mean(axis=0)
    c = np.cov(kept, rowvar=False) * consistency_factor(d, k / len(scores)) + np.eye(d) * reg
    return ((1. - momentum) * mu + momentum * m,
            (1. - momentum) * cov + momentum * c)


class GatedPolicy:
    """One-line-delayed policy: score line t, commit a candidate built on t-1
    chosen by `chooser`, then build candidates on line t.

    chooser(z, raw, pending, mu_active, cov_active) -> action. The chooser sees
    unlabeled features only. With the Shadow chooser and consistency=False this
    is numerically identical to run_shadow.ShadowPolicy; with a constant
    'trim10' chooser it is identical to run_shadow.DelayedPolicy(.10).
    """
    def __init__(self, mu, cov, momentum, chooser, consistency=False, needed=ACTIONS):
        self.mu, self.cov = mu.copy(), cov.copy()
        self.momentum = momentum
        self.chooser = chooser
        self.consistency = consistency
        self.needed = tuple(needed)
        self.pending = None
        self.actions = []

    def step(self, z):
        raw = stable_md(z, self.mu, self.cov)
        if self.pending is not None:
            action = self.chooser(z, raw, self.pending, self.mu, self.cov)
            self.actions.append(action)
            self.mu, self.cov = self.pending[action]
        current = (self.mu.copy(), self.cov.copy())
        self.pending = {'skip': current}
        for action, q in TRIM.items():
            if action in self.needed:
                self.pending[action] = proposal(z, *current, q, self.momentum, self.consistency)
        return raw


def shadow_chooser(core_fraction, penalty):
    def choose(z, raw, pending, mu, cov):
        k = min(len(z), max(z.shape[1] + 2, int(round(len(z) * core_fraction))))
        ix = np.argsort(raw, kind='stable')[:k]
        costs = {a: robust_predictive_cost(z, ix, *pending[a], mu, cov, penalty) for a in ACTIONS}
        return min(ACTIONS, key=lambda a: costs[a])
    return choose


def fixed_chooser(action):
    return lambda *_: action


def sequence_chooser(seq):
    it = iter(seq)
    return lambda *_: next(it)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def to_mode(raw_lines, mode):
    if mode == 'raw':
        return raw_lines
    return np.stack([normalise_line(r) for r in raw_lines])


def metrics(y_flat, lines):
    pred = np.asarray(lines, dtype=np.float64).T.ravel()  # [lines,pix] -> pixel-major
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite scores')
    fpr, tpr, _ = roc_curve(y_flat, pred)
    ix = min(int(np.searchsorted(tpr, .90, side='left')), len(fpr) - 1)
    return {'auroc': float(roc_auc_score(y_flat, pred)),
            'ap': float(average_precision_score(y_flat, pred)),
            'fpr90': float(fpr[ix])}


def pareto_flags(rows):
    """Non-dominated set for (ap max, auroc max, fpr90 min)."""
    flags = []
    for r in rows:
        dominated = False
        for s in rows:
            if s is r:
                continue
            ge = s['ap_mean'] >= r['ap_mean'] and s['auroc_mean'] >= r['auroc_mean'] and s['fpr90_mean'] <= r['fpr90_mean']
            gt = s['ap_mean'] > r['ap_mean'] or s['auroc_mean'] > r['auroc_mean'] or s['fpr90_mean'] < r['fpr90_mean']
            if ge and gt:
                dominated = True
                break
        flags.append(int(not dominated))
    return flags


# --------------------------------------------------------------------------
# One projection seed
# --------------------------------------------------------------------------
def run_seed(x, seed, warmup, dims, momentum, core_fraction, penalty,
             n_perm=5, max_lines=None, erx_class=None):
    """Return {method: raw [lines,pixels]}, action logs, timings, permuted logs.

    Permuted-gate scores are returned as a list per Shadow variant because
    they are replicates of one control, not separate methods.
    """
    h, w, n_bands = x.shape
    n = w if max_lines is None else min(max_lines, w)
    projector, mu0, cov0 = prepare_warmup(x, seed, warmup, dims, momentum)
    z = np.empty((n - warmup, h, dims), dtype=np.float64)
    for t in range(warmup, n):
        z[t - warmup] = projector.transform(np.asarray(x[:, t, :], dtype=np.float64))

    out, timing, logs, perm = {}, {}, {}, {}

    def run_policy(name, policy):
        tic = time.perf_counter()
        out[name] = np.stack([policy.step(line) for line in z])
        timing[name] = 1000 * (time.perf_counter() - tic) / len(z)
        return policy

    out['frozen'] = np.stack([stable_md(line, mu0, cov0) for line in z])
    timing['frozen'] = float('nan')

    # Immediate (non-delayed) updates, identical to v6 'fast' detectors.
    for name, frac in (('immediate_full', 0.), ('immediate_trim05', .05), ('immediate_trim10', .10)):
        det = PrequentialERX(projector, warmup, momentum, trim_fraction=frac)
        for col in range(warmup):
            det.forward(np.asarray(x[:, col, :], dtype=np.float64))
        tic = time.perf_counter()
        out[name] = np.stack([det.forward(np.asarray(x[:, t, :], dtype=np.float64)) for t in range(warmup, n)])
        timing[name] = 1000 * (time.perf_counter() - tic) / len(z)

    for tag, cc in (('nc', False), ('cc', True)):
        for act in ('trim05', 'trim10'):
            run_policy(f'delayed_{act}_{tag}',
                       GatedPolicy(mu0, cov0, momentum, fixed_chooser(act), cc, needed=(act,)))
        sh = run_policy(f'shadow_{tag}',
                        GatedPolicy(mu0, cov0, momentum, shadow_chooser(core_fraction, penalty), cc))
        logs[f'shadow_{tag}'] = list(sh.actions)
        rng = np.random.default_rng(10_000 * seed + (0 if tag == 'nc' else 1))
        reps = []
        for _ in range(n_perm):
            seq = list(rng.permutation(np.array(sh.actions, dtype=object)))
            pol = GatedPolicy(mu0, cov0, momentum, sequence_chooser(seq), cc)
            reps.append(np.stack([pol.step(line) for line in z]))
            if sorted(pol.actions) != sorted(sh.actions):
                raise AssertionError('Permuted gate changed action counts')
        perm[f'shadow_{tag}'] = reps

    if erx_class is not None:
        for mode in MODES:
            det = erx_class(n_bands=n_bands, n_pixels=h, buffer_len=warmup,
                            n_projdims=dims, momentum=momentum, normalise_md=(mode == 'line_z'))
            det.random_projector = projector
            for col in range(warmup):
                det.forward(np.asarray(x[:, col, :], dtype=np.float64))
            tic = time.perf_counter()
            out[f'ERX_original@{mode}'] = np.stack(
                [np.asarray(det.forward(np.asarray(x[:, t, :], dtype=np.float64)), dtype=np.float64)
                 for t in range(warmup, n)])
            timing[f'ERX_original@{mode}'] = 1000 * (time.perf_counter() - tic) / len(z)
    return out, logs, perm, timing, n


# --------------------------------------------------------------------------
# Whole experiment
# --------------------------------------------------------------------------
def evaluate_all(x, mask, seeds, warmup, dims, momentum, core_fraction, penalty,
                 n_perm, max_lines=None, erx_class=None, log=print):
    per_seed, gate_rows, count_rows = [], [], []
    for seed in seeds:
        tic = time.perf_counter()
        out, logs, perm, timing, n = run_seed(x, seed, warmup, dims, momentum, core_fraction,
                                              penalty, n_perm, max_lines, erx_class)
        y = np.asarray(mask[:, warmup:n] != 0, dtype=bool).ravel()
        if y.all() or not y.any():
            raise ValueError('Evaluated mask needs both classes')
        for mode in MODES:
            for name, raw in out.items():
                if name.startswith('ERX_original@'):
                    if not name.endswith('@' + mode):
                        continue
                    scores, label = raw, 'ERX_original'  # ERX applies its own flag
                else:
                    scores, label = to_mode(raw, mode), name
                m = metrics(y, scores)
                per_seed.append({'seed': seed, 'mode': mode, 'method': label, **m,
                                 'ms_per_line': timing.get(name, float('nan'))})
            for variant, reps in perm.items():
                rep_m = [metrics(y, to_mode(r, mode)) for r in reps]
                ref = next(r for r in per_seed if r['seed'] == seed and r['mode'] == mode and r['method'] == variant)
                avg = {k: float(np.mean([q[k] for q in rep_m])) for k in ('auroc', 'ap', 'fpr90')}
                per_seed.append({'seed': seed, 'mode': mode, 'method': f'{variant}_permgate',
                                 **avg, 'ms_per_line': float('nan')})
                gate_rows.append({
                    'seed': seed, 'mode': mode, 'shadow_variant': variant, 'n_permutations': len(rep_m),
                    'shadow_ap': ref['ap'], 'permgate_ap_mean': avg['ap'],
                    'permgate_ap_min': min(q['ap'] for q in rep_m), 'permgate_ap_max': max(q['ap'] for q in rep_m),
                    'delta_ap_shadow_minus_perm': ref['ap'] - avg['ap'],
                    'delta_auroc_shadow_minus_perm': ref['auroc'] - avg['auroc'],
                    'delta_fpr90_shadow_minus_perm': ref['fpr90'] - avg['fpr90'],
                    'perm_reps_beaten_ap': int(sum(ref['ap'] > q['ap'] for q in rep_m)),
                    'perm_reps_beaten_fpr90': int(sum(ref['fpr90'] < q['fpr90'] for q in rep_m))})
        for variant, acts in logs.items():
            count_rows.append({'seed': seed, 'variant': variant, 'decisions': len(acts),
                               **{f'n_{a}': int(sum(v == a for v in acts)) for a in ACTIONS},
                               'skip_rate': float(np.mean([v == 'skip' for v in acts]))})
        log(f'seed={seed} done in {time.perf_counter() - tic:.1f}s  '
            + ' '.join(f"{r['variant']}:skip={r['skip_rate']:.3f}" for r in count_rows if r['seed'] == seed))
    summary = []
    for mode in MODES:
        names = list(dict.fromkeys(r['method'] for r in per_seed if r['mode'] == mode))
        block = []
        for name in names:
            rs = [r for r in per_seed if r['mode'] == mode and r['method'] == name]
            row = {'mode': mode, 'method': name, 'n_seeds': len(rs)}
            for k in ('auroc', 'ap', 'fpr90', 'ms_per_line'):
                v = np.array([r[k] for r in rs], dtype=float)
                row[k + '_mean'] = float(np.nanmean(v)) if np.isfinite(v).any() else float('nan')
                row[k + '_sd'] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float('nan')
            block.append(row)
        for row, flag in zip(block, pareto_flags(block)):
            row['pareto_nondominated'] = flag
        summary.extend(block)
    return per_seed, summary, gate_rows, count_rows


def plot_pareto(summary, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mode in zip(axes, MODES):
        rows = [r for r in summary if r['mode'] == mode]
        for r in rows:
            marker = 'o' if r['pareto_nondominated'] else 'x'
            ax.scatter(r['auroc_mean'], r['ap_mean'], marker=marker, s=40)
            ax.annotate(r['method'], (r['auroc_mean'], r['ap_mean']), fontsize=7,
                        xytext=(3, 3), textcoords='offset points')
        ax.set_xlabel('AUROC (seed mean)')
        ax.set_ylabel('AP (seed mean)')
        ax.set_title(f'{mode}: o = Pareto (AP, AUROC, FPR90)')
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(dest, dpi=170)
    plt.close(fig)


def write_report(out, summary, gate_rows, count_rows, dataset):
    L = [f'HyperAD v12 PROTOKOL DUZELTME VE KONTROL RAPORU ({dataset})',
         'Gelistirme sahnesi: sonuclar ayar icin kullanilmaz. Tohumlar bagimsiz sahne degildir.', '']
    for mode in MODES:
        L.append(f'[{mode}] tohum ortalamalari (P = Pareto-baskin olmayan):')
        for r in [r for r in summary if r['mode'] == mode]:
            L.append(f"  {'P' if r['pareto_nondominated'] else ' '} {r['method']:26s} "
                     f"AUROC={r['auroc_mean']:.4f} AP={r['ap_mean']:.4f} FPR90={r['fpr90_mean']:.4f}")
        L.append('')
    L.append('ESLESMIS ORANLI KAPI KONTROLU (Shadow - permuted gate, ayni aksiyon sayilari):')
    for mode in MODES:
        for v in ('shadow_nc', 'shadow_cc'):
            g = [r for r in gate_rows if r['mode'] == mode and r['shadow_variant'] == v]
            if not g:
                continue
            L.append(f"  {mode}/{v}: mean dAP={np.mean([r['delta_ap_shadow_minus_perm'] for r in g]):+.4f}, "
                     f"mean dFPR90={np.mean([r['delta_fpr90_shadow_minus_perm'] for r in g]):+.4f}, "
                     f"AP'de gecilen permutasyon={sum(r['perm_reps_beaten_ap'] for r in g)}/"
                     f"{sum(r['n_permutations'] for r in g)}")
    L += ['', 'SKIP ORANI:']
    for v in ('shadow_nc', 'shadow_cc'):
        c = [r for r in count_rows if r['variant'] == v]
        if c:
            L.append(f"  {v}: ortalama skip orani={np.mean([r['skip_rate'] for r in c]):.3f}")
    L += ['', 'YORUM KURALI (onceden sabit):',
          '- Shadow, permuted gate kopyalarinin cogunu her iki modda gecmiyorsa kazanc zamanlama degil oran etkisidir.',
          '- line_z modunda Frozen AP ustunlugu kayboluyorsa onceki Frozen bulgusu kalibrasyon artefaktidir.',
          '- nc ve cc arasindaki fark ana tablonun yonunu degistiriyorsa duzeltilmis surum raporlanir.',
          '- Bu kurallar SNP/Beach sonuclarina bakilarak degistirilmez.']
    (Path(out) / 'v12_report_TR.txt').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hyperad-root', default=None, help='Unmodified HyperAD-main (enables ERX_original)')
    p.add_argument('--hsi', required=True)
    p.add_argument('--mask', required=True)
    p.add_argument('--dataset-name', default='Beach')
    p.add_argument('--warmup', type=int, default=99)
    p.add_argument('--projected-dims', type=int, default=5)
    p.add_argument('--momentum', type=float, default=.10)
    p.add_argument('--core-fraction', type=float, default=.80)
    p.add_argument('--penalty', type=float, default=.05)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--permutations', type=int, default=5, help='Permuted-gate replicates per seed and variant')
    p.add_argument('--max-lines', type=int, default=None, help='Quick tests only')
    p.add_argument('--reverse', action='store_true', help='Scan lines in reverse order (pure flip of image and mask)')
    p.add_argument('--output', default='results_v12')
    a = p.parse_args()
    if not 0 < a.core_fraction < 1 or a.penalty < 0 or not 0 < a.momentum <= 1:
        raise ValueError('Invalid method settings')
    if a.permutations < 1 or len(set(a.seeds)) != len(a.seeds):
        raise ValueError('Invalid permutations or duplicate seeds')
    erx_class, erx_hash = None, None
    if a.hyperad_root:
        root = reference_repo(a.hyperad_root)
        sys.path.insert(0, str(root))
        from detectors.erx import ERX
        erx_class = ERX
        erx_hash = hashlib.sha256((root / 'detectors' / 'erx.py').read_bytes()).hexdigest()
    else:
        print('WARNING: --hyperad-root not given; ERX_original is skipped.', flush=True)
    x = np.load(a.hsi, mmap_mode='r', allow_pickle=False)
    mask = np.load(a.mask, mmap_mode='r', allow_pickle=False)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if a.reverse:
        x, mask = x[:, ::-1, :], mask[:, ::-1]
    if x.ndim != 3 or mask.shape != x.shape[:2]:
        raise ValueError(f'shape mismatch image {x.shape}, mask {mask.shape}')
    if a.projected_dims < 2 or a.projected_dims > x.shape[2]:
        raise ValueError('Bad projected dims')
    n = x.shape[1] if a.max_lines is None else min(a.max_lines, x.shape[1])
    if n <= a.warmup + 2:
        raise ValueError('Not enough lines after warmup')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    meta = vars(a).copy()
    meta.update(shape=list(x.shape), mask_shape=list(mask.shape), reference_erx_sha256=erx_hash,
                consistency_factor_trim05=consistency_factor(a.projected_dims, .95),
                consistency_factor_trim10=consistency_factor(a.projected_dims, .90),
                protocol='Label-free decisions; both normalisation modes; nc/cc trimmed covariance; '
                         'permuted-gate matched-rate control. Development scenes, not held-out test.')
    (out / 'run_config.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'V12 {a.dataset_name} shape={x.shape} seeds={a.seeds} perms={a.permutations} '
          f'c05={meta["consistency_factor_trim05"]:.4f} c10={meta["consistency_factor_trim10"]:.4f}', flush=True)
    with threadpool_limits(limits=1):
        per_seed, summary, gate, counts = evaluate_all(
            x, mask, a.seeds, a.warmup, a.projected_dims, a.momentum, a.core_fraction,
            a.penalty, a.permutations, a.max_lines, erx_class)
    save_csv(out / 'results_per_seed.csv', per_seed)
    save_csv(out / 'summary_pareto.csv', summary)
    save_csv(out / 'gate_vs_permuted.csv', gate)
    save_csv(out / 'shadow_action_counts.csv', counts)
    plot_pareto(summary, out / 'pareto_ap_auroc.png')
    write_report(out, summary, gate, counts, a.dataset_name)
    print('Outputs saved:', out.resolve())


if __name__ == '__main__':
    main()
