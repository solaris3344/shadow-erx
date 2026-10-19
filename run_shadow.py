#!/usr/bin/env python3
"""Exploratory LABEL-FREE delayed one-line shadow selection for ERX.

All decisions are made on unlabeled features. At scan line t, the prediction
is emitted using the model that was committed BEFORE viewing line t. Only then
is line t used to choose among pending hypotheses trained on line t-1.
Chosen hypothesis can influence the prediction starting at line t+1.
This is a research screen, NOT an independently validated new method.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits
from baseline_v3 import PrequentialERX, shared_projector, stable_md, normalise_line, reference_repo

ACTIONS = ('skip', 'trim05', 'trim10')
TRIM = {'trim05': .05, 'trim10': .10}


def proposal(z, mu, cov, fraction, momentum, reg=1e-5):
    """Apply a single line update with a pre-update anomaly ranking."""
    d = z.shape[1]
    scores = stable_md(z, mu, cov, reg)
    k = min(len(scores), max(d + 2, int(round(len(scores) * (1. - fraction)))))
    indices = np.argsort(scores, kind='stable')[:k]
    kept = z[indices]
    m = kept.mean(axis=0)
    c = np.cov(kept, rowvar=False) + np.eye(d) * reg
    return ((1. - momentum) * mu + momentum * m,
            (1. - momentum) * cov + momentum * c)


def robust_predictive_cost(z, core_ix, mu, cov, baseline_mu, baseline_cov, penalty):
    """Common held-out core, Gaussian log score and conservative movement penalty.

    The core is fixed using the previous ACTIVE (pre-selection) state.
    A candidate trained on the previous line is evaluated on the new line.
    """
    d = z.shape[1]
    sign, logdet = np.linalg.slogdet(cov + 1e-5 * np.eye(d))
    if sign <= 0:
        return float('inf')
    md2 = stable_md(z[core_ix], mu, cov) ** 2
    fit = (float(np.mean(md2)) + float(logdet)) / d
    cov_jump = float(np.linalg.norm(cov - baseline_cov, 'fro') /
                     max(np.linalg.norm(baseline_cov, 'fro'), 1e-12))
    mean_jump = float(np.linalg.norm(mu - baseline_mu) /
                      np.sqrt(max(np.trace(baseline_cov), 1e-12)))
    return fit + penalty * (cov_jump + mean_jump)


class DelayedPolicy:
    """Equal-latency fixed policy; predicted scores never use current features to update."""
    def __init__(self, mu, cov, momentum, fraction):
        self.mu = mu.copy()
        self.cov = cov.copy()
        self.momentum = momentum
        self.fraction = fraction
        self.pending = None

    def step(self, z, normalise=False):
        raw = stable_md(z, self.mu, self.cov)
        output = normalise_line(raw) if normalise else raw
        if self.pending is not None:
            self.mu, self.cov = self.pending
        self.pending = proposal(z, self.mu, self.cov, self.fraction, self.momentum)
        return output


class ShadowPolicy:
    """Select skip / Trim5 / Trim10 by unlabeled next-line shadow validation."""
    def __init__(self, mu, cov, momentum, core_fraction=.80, penalty=.05):
        self.mu = mu.copy()
        self.cov = cov.copy()
        self.momentum = momentum
        self.core_fraction = core_fraction
        self.penalty = penalty
        self.pending = None
        self.actions = []
        self.decision_rows = []

    def step(self, z, line_num, normalise=False):
        # Emit BEFORE observing this line to select or update a model.
        raw = stable_md(z, self.mu, self.cov)
        output = normalise_line(raw) if normalise else raw
        if self.pending is not None:
            k = max(z.shape[1] + 2, int(round(len(z) * self.core_fraction)))
            k = min(k, len(z))
            ix = np.argsort(raw, kind='stable')[:k]
            costs = {
                a: robust_predictive_cost(z, ix, *state, self.mu, self.cov, self.penalty)
                for a, state in self.pending.items()
            }
            winner = min(ACTIONS, key=lambda a: costs[a])
            self.actions.append(winner)
            self.decision_rows.append({'line': line_num, 'selected': winner,
                                       'cost_skip': costs['skip'],
                                       'cost_trim05': costs['trim05'],
                                       'cost_trim10': costs['trim10'],
                                       'selection_margin': costs['skip'] - costs[winner]})
            self.mu, self.cov = self.pending[winner]
        # Candidates for this line cannot affect this line's emitted scores.
        current = (self.mu.copy(), self.cov.copy())
        self.pending = {'skip': current}
        for action, q in TRIM.items():
            self.pending[action] = proposal(z, *current, q, self.momentum)
        return output


def prepare_warmup(x, seed, warmup, proj_dims, momentum):
    n_pixels, _, n_bands = x.shape
    projector = shared_projector(n_pixels, n_bands, proj_dims, seed)
    initializer = PrequentialERX(projector, warmup, momentum,
                                 trim_fraction=0, normalise_md=False)
    for col in range(warmup):
        line = np.asarray(x[:, col, :], dtype=np.float64)
        initializer.forward(line)
    return projector, initializer.mean.copy(), initializer.cov.copy()


def run_seed(x, seed, warmup, proj_dims, momentum, core_fraction, penalty,
             normalise_md=False, max_lines=None, original_erx_class=None):
    h, w, n_bands = x.shape
    n = w if max_lines is None else min(max_lines, w)
    if n <= warmup + 1:
        raise ValueError('Need warmup+2 lines for shadow selection')
    projector, mu, cov = prepare_warmup(x, seed, warmup, proj_dims, momentum)
    # Track actual full sequential policies as well as equal-latency baselines.
    fast = {q: PrequentialERX(projector, warmup, momentum, trim_fraction=f,
                              normalise_md=normalise_md)
            for q, f in [('prequential', 0), ('trim05', .05), ('trim10', .10)]}
    for col in range(warmup):
        data = np.asarray(x[:, col, :], dtype=np.float64)
        for det in fast.values():
            det.forward(data)
    delayed = {'delayed_trim05': DelayedPolicy(mu, cov, momentum, .05),
               'delayed_trim10': DelayedPolicy(mu, cov, momentum, .10)}
    shadow = ShadowPolicy(mu, cov, momentum, core_fraction, penalty)
    original = None
    if original_erx_class is not None:
        original = original_erx_class(n_bands=n_bands, n_pixels=h,
                        buffer_len=warmup, n_projdims=proj_dims,
                        momentum=momentum, normalise_md=normalise_md)
        original.random_projector = projector
        for col in range(warmup):
            original.forward(np.asarray(x[:, col, :], dtype=np.float64))
    methods = ['frozen', *fast, *delayed, 'shadow']
    if original is not None:
        methods = ['ERX_original', *methods]
    outputs = {q: np.empty((n-warmup, h), dtype=np.float64) for q in methods}
    t0 = time.perf_counter()
    for t in range(warmup, n):
        line = np.asarray(x[:, t, :], dtype=np.float64)
        z = np.asarray(projector.transform(line), dtype=np.float64)
        if original is not None:
            outputs['ERX_original'][t-warmup] = original.forward(line)
        frozen_score = stable_md(z, mu, cov)
        outputs['frozen'][t-warmup] = normalise_line(frozen_score) if normalise_md else frozen_score
        for key, det in fast.items():
            outputs[key][t-warmup] = det.forward(line)
        for key, det in delayed.items():
            outputs[key][t-warmup] = det.step(z, normalise_md)
        outputs['shadow'][t-warmup] = shadow.step(z, t, normalise_md)
    elapsed = time.perf_counter()-t0
    if any(not np.isfinite(arr).all() for arr in outputs.values()):
        raise ValueError('Nonfinite anomaly score encountered')
    return outputs, shadow.decision_rows, elapsed


def score_metrics(mask_eval, scores):
    m = np.asarray(mask_eval != 0, dtype=bool).ravel()
    if m.all() or not m.any():
        raise ValueError('Evaluation mask needs both classes')
    result = []
    for key, lines in scores.items():
        pred = lines.T.ravel()  # mask is [pixels, lines]
        fpr, tpr, _ = roc_curve(m, pred)
        ix = min(int(np.searchsorted(tpr, .90, side='left')), len(fpr)-1)
        result.append({'method': key, 'auroc': float(roc_auc_score(m,pred)),
                       'ap': float(average_precision_score(m,pred)),
                       'fpr90': float(fpr[ix])})
    return result


def save_csv(filepath, rows):
    if not rows: return
    with Path(filepath).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def cli():
    p = argparse.ArgumentParser(description='Exploratory delayed label-free ERX shadow validation')
    p.add_argument('--hyperad-root', required=True, help='Unmodified HyperAD-main folder')
    p.add_argument('--hsi', required=True)
    p.add_argument('--mask', required=True)
    p.add_argument('--dataset-name', default='Beach')
    p.add_argument('--warmup', type=int, default=99)
    p.add_argument('--projected-dims', type=int, default=5)
    p.add_argument('--momentum', type=float, default=.10)
    p.add_argument('--core-fraction', type=float, default=.80)
    p.add_argument('--penalty', type=float, default=.05)
    p.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    p.add_argument('--max-lines', type=int, default=None)
    p.add_argument('--normalise-md', action='store_true')
    p.add_argument('--output', default='results_shadow_v6')
    return p.parse_args()


def main(args):
    root = reference_repo(args.hyperad_root)
    sys.path.insert(0, str(root))
    from detectors.erx import ERX
    if not (0 < args.core_fraction < 1): raise ValueError('core-fraction must be in (0,1)')
    if args.penalty < 0: raise ValueError('penalty cannot be negative')
    if not (0 < args.momentum <= 1): raise ValueError('Invalid momentum')
    if args.warmup < 3: raise ValueError('warmup must be at least 3')
    if len(set(args.seeds)) != len(args.seeds): raise ValueError('seeds must be unique')
    x = np.load(args.hsi, mmap_mode='r', allow_pickle=False)
    mask = np.load(args.mask, mmap_mode='r', allow_pickle=False)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:,:,0]
    if x.ndim != 3 or mask.shape != x.shape[:2]:
        raise ValueError(f'Invalid shapes: image={x.shape}, mask={mask.shape}')
    if args.projected_dims > x.shape[2]: raise ValueError('Projection dimension exceeds bands')
    n = x.shape[1] if args.max_lines is None else min(args.max_lines, x.shape[1])
    if n <= args.warmup + 1: raise ValueError('Not enough lines')
    output = Path(args.output); output.mkdir(exist_ok=True, parents=True)
    info = {'dataset':args.dataset_name, 'shape':list(x.shape), 'warmup':args.warmup,
            'projected_dims':args.projected_dims, 'momentum':args.momentum,
            'core_fraction':args.core_fraction, 'penalty':args.penalty,
            'normalise_md':args.normalise_md, 'seeds':args.seeds,
            'max_lines':args.max_lines, 'reference_sha256':hashlib.sha256((root/'detectors'/'erx.py').read_bytes()).hexdigest(),
            'disclaimer':'SNP and Beach already used in research design; this is DEVELOPMENT, not independent test. No masks in updates/decisions.'}
    (output/'run_config.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    rows, decisions = [], []
    with threadpool_limits(limits=1):
        for seed in args.seeds:
            outputs, actions, elapsed = run_seed(x, seed, args.warmup,
                args.projected_dims, args.momentum, args.core_fraction,
                args.penalty, args.normalise_md, args.max_lines, ERX)
            results = score_metrics(mask[:,args.warmup:n], outputs)
            for row in results:
                row['seed']=seed
                row['ms_per_line_all_methods']=float(elapsed*1000/(n-args.warmup))
                rows.append(row)
                print(f"seed={seed} {row['method']:15s} AUROC={row['auroc']:.5f} "
                      f"AP={row['ap']:.5f} FPR90={row['fpr90']:.5f}", flush=True)
            for action in actions:
                decisions.append({'seed':seed, **action})
            counts = {key: sum(rec['selected']==key for rec in actions) for key in ACTIONS}
            print('  Shadow selections:',counts, '/',len(actions),flush=True)
            save_csv(output/'all_results.csv', rows)
            save_csv(output/'shadow_decisions.csv', decisions)
    means = []
    for method in ['ERX_original','frozen','prequential','trim05','trim10','delayed_trim05','delayed_trim10','shadow']:
        r = [k for k in rows if k['method']==method]
        row = {'method':method,'n_seeds':len(r)}
        for metric in ('auroc','ap','fpr90'):
            arr = np.array([q[metric] for q in r])
            row[metric+'_mean']=float(arr.mean())
            row[metric+'_sd']=float(arr.std(ddof=1)) if len(arr)>1 else float('nan')
        means.append(row)
    save_csv(output/'summary.csv',means)
    print('Saved:',output.resolve(),flush=True)

if __name__ == '__main__':
    main(cli())
