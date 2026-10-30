#!/usr/bin/env python3
"""HyperAD v20: the shadow gate on a second detector, RX-BIL (post hoc).

Question: is the delayed candidate selection specific to ERX, or does it also
help a different line-by-line detector?

RX-BIL (Du & Nekovei 2009, official HyperAD implementation) keeps a full-band,
uncentred correlation matrix R = sum of x x^T over all lines seen so far and
scores each pixel by sqrt(x^T R^-1 x). It has no forgetting and no projection.

The gate is transferred without changing its rule or parameters:
  state       S (sum of x x^T) and N (number of pixels); Sigma = S / N
  skip        S, N unchanged
  Trim-q      the pixels of line t-1 are ranked by their distance under the
              active state, the k = max(b+2, round(n(1-q))) closest are added,
              S += c_q * X_k^T X_k, N += k (c_q = Gaussian consistency factor
              for b dimensions; factor 1 for the nc variant)
  selection   on line t, robust core = rho*n pixels closest under the active
              state; cost(a) = [mean md^2_core(Sigma_a) + log det Sigma_a] / b
              + lambda * ||Sigma_a - Sigma||_F / ||Sigma||_F
              (RX-BIL has no mean, so the mean-jump term of Eq. (4) is zero)
  rho = 0.8, lambda = 0.05 as preregistered for ERX; nothing is tuned.

Methods (all line z-scored, warm-up 99 lines, evaluation lines warmup..W-1):
  RX_BIL_official     unmodified HyperAD RX-BIL (pixel dropout 0.5, seeded)
  rxbil_frozen        warm-up state, never updated
  rxbil_delayed_full  causal RX-BIL without dropout, one-line delayed update
  rxbil_delayed_trim05_cc / trim10_cc
  rxbil_gate_cc / rxbil_gate_nc
Our variants are deterministic (no projection, no dropout); only the official
detector depends on the seed. Labels are used only in metric code.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import chi2
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits

ACTIONS = ('skip', 'trim05', 'trim10')
TRIM = {'trim05': .05, 'trim10': .10}


def normalise_line(a):
    sd = float(a.std())
    return (a - float(a.mean())) / sd if sd > 1e-12 else np.zeros_like(a)


def consistency_factor(d, keep):
    if keep >= 1:
        return 1.0
    return float(keep / chi2.cdf(chi2.ppf(keep, d), d + 2))


class State:
    """Full-band uncentred second-moment model with a cached Cholesky factor."""
    __slots__ = ('S', 'N', '_L', '_logdet')

    def __init__(self, S, N):
        self.S, self.N, self._L, self._logdet = S, float(N), None, None

    def _chol(self):
        if self._L is None:
            sig = self.S / self.N
            b = sig.shape[0]
            jit = 1e-8 * float(np.trace(sig)) / b
            for _ in range(8):
                try:
                    self._L = np.linalg.cholesky(sig + jit * np.eye(b))
                    break
                except np.linalg.LinAlgError:
                    jit *= 10
            else:
                raise np.linalg.LinAlgError('Sigma not positive definite')
            self._logdet = 2.0 * float(np.log(np.diag(self._L)).sum())
        return self._L

    def md(self, X):
        return np.linalg.norm(np.linalg.solve(self._chol(), X.T), axis=0)

    def logdet(self):
        self._chol()
        return self._logdet

    def sigma(self):
        return self.S / self.N


MOMENTUM = None  # None: cumulative RX-BIL; float: exponential forgetting (ERX-style)


def _update(state, M, k):
    """Add second-moment sum M of k pixels to the state."""
    if MOMENTUM is None:
        return State(state.S + M, state.N + k)
    a = MOMENTUM
    return State((1 - a) * state.sigma() + a * M / k, 1.0)


def trimmed(state, X, q, cc):
    n, b = X.shape
    k = min(n, max(b + 2, int(round(n * (1 - q)))))
    kept = X[np.argsort(state.md(X), kind='stable')[:k]]
    c = consistency_factor(b, k / n) if cc else 1.0
    return _update(state, c * (kept.T @ kept), k)


def full(state, X):
    return _update(state, X.T @ X, len(X))


def gate_cost(X, core, cand, active, penalty):
    b = X.shape[1]
    md2 = cand.md(X[core]) ** 2
    fit = (float(md2.mean()) + cand.logdet()) / b
    sa, s0 = cand.sigma(), active.sigma()
    jump = float(np.linalg.norm(sa - s0, 'fro') / max(np.linalg.norm(s0, 'fro'), 1e-300))
    return fit + penalty * jump


class Delayed:
    """Score with the active state, commit the candidate built from line t-1,
    build the next candidate(s) from line t."""
    def __init__(self, s0, mode, cc=True, rho=.8, penalty=.05):
        self.state, self.mode, self.cc = s0, mode, cc
        self.rho, self.penalty = rho, penalty
        self.pending, self.actions = None, []

    def step(self, X):
        raw = self.state.md(X)
        if self.pending is not None:
            if self.mode == 'gate':
                k = min(len(X), max(X.shape[1] + 2, int(round(len(X) * self.rho))))
                core = np.argsort(raw, kind='stable')[:k]
                costs = {a: gate_cost(X, core, self.pending[a], self.state, self.penalty) for a in ACTIONS}
                act = min(ACTIONS, key=lambda a: costs[a])
                self.actions.append(act)
                self.state = self.pending[act]
            else:
                self.state = self.pending
        if self.mode == 'gate':
            self.pending = {'skip': self.state}
            for a, q in TRIM.items():
                self.pending[a] = trimmed(self.state, X, q, self.cc)
        elif self.mode == 'full':
            self.pending = full(self.state, X)
        else:
            self.pending = trimmed(self.state, X, TRIM[self.mode], self.cc)
        return raw


def metric_row(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    ix = min(int(np.searchsorted(tpr, .90, side='left')), len(fpr) - 1)
    return {'auroc': float(roc_auc_score(y, s)), 'ap': float(average_precision_score(y, s)),
            'fpr90': float(fpr[ix])}


def run_scene(x, mask, args, rx_class=None):
    h, w, b = x.shape
    if h < b + 2:
        raise ValueError('RX-BIL needs at least b+2 pixels per line')
    if MOMENTUM is None:
        S = np.zeros((b, b)); N = 0
        for t in range(args.warmup):
            X = np.asarray(x[:, t, :], dtype=np.float64)
            S += X.T @ X; N += h
        s0 = State(S, N)
    else:  # ERX-style warm-up: first line initialises, then momentum updates
        X = np.asarray(x[:, 0, :], dtype=np.float64)
        s0 = State(X.T @ X / h, 1.0)
        for t in range(1, args.warmup):
            s0 = full(s0, np.asarray(x[:, t, :], dtype=np.float64))
    pols = {'rxbil_delayed_full': Delayed(s0, 'full'),
            'rxbil_delayed_trim05_cc': Delayed(s0, 'trim05', True),
            'rxbil_delayed_trim10_cc': Delayed(s0, 'trim10', True),
            'rxbil_gate_cc': Delayed(s0, 'gate', True, args.core_fraction, args.penalty),
            'rxbil_gate_nc': Delayed(s0, 'gate', False, args.core_fraction, args.penalty)}
    n_eval = w - args.warmup
    out = {k: np.empty((n_eval, h)) for k in ['rxbil_frozen', *pols]}
    timing = {k: 0.0 for k in out}
    for t in range(args.warmup, w):
        X = np.asarray(x[:, t, :], dtype=np.float64)
        i = t - args.warmup
        tic = time.perf_counter(); out['rxbil_frozen'][i] = normalise_line(s0.md(X))
        timing['rxbil_frozen'] += time.perf_counter() - tic
        for k, p in pols.items():
            tic = time.perf_counter()
            out[k][i] = normalise_line(p.step(X))
            timing[k] += time.perf_counter() - tic
    official = []
    if rx_class is not None:
        for seed in args.seeds:
            np.random.seed(seed)
            det = rx_class(n_bands=b, n_pixels=h, buffer_len=args.warmup,
                           pixel_dropout=args.pixel_dropout, normalise_md=True)
            sc = np.zeros((w, h)); tic = time.perf_counter()
            for t in range(w):
                r = det.forward(np.asarray(x[:, t, :], dtype=np.float64))
                if r is not None:
                    sc[t] = r
            official.append((seed, sc[args.warmup:], 1000 * (time.perf_counter() - tic) / w))
    y = (np.asarray(mask[:, args.warmup:w]) != 0).T.ravel()
    rows = []
    for k, s in out.items():
        r = {'method': k, 'seed': 'deterministic', 'ms_per_line': 1000 * timing[k] / n_eval}
        r.update(metric_row(y, s.ravel())); rows.append(r)
    for seed, s, ms in official:
        r = {'method': 'RX_BIL_official', 'seed': seed, 'ms_per_line': ms}
        r.update(metric_row(y, s.ravel())); rows.append(r)
    acts = {k: p.actions for k, p in pols.items() if p.mode == 'gate'}
    return rows, acts


def summarise(rows):
    g = {}
    for r in rows:
        g.setdefault(r['method'], []).append(r)
    out = []
    for m, rs in g.items():
        s = {'method': m, 'n_runs': len(rs)}
        for k in ('auroc', 'ap', 'fpr90', 'ms_per_line'):
            s[k] = float(np.mean([r[k] for r in rs]))
            if len(rs) > 1:
                s[f'{k}_sd'] = float(np.std([r[k] for r in rs]))
        out.append(s)
    return out


def save_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)


def cli(argv=None):
    p = argparse.ArgumentParser(description='Shadow gate transferred to RX-BIL (post hoc)')
    p.add_argument('--hsi', required=True); p.add_argument('--mask', required=True)
    p.add_argument('--dataset-name', default='Scene')
    p.add_argument('--hyperad-root', default=None, help='adds the unmodified official RX-BIL')
    p.add_argument('--warmup', type=int, default=99)
    p.add_argument('--core-fraction', type=float, default=.80)
    p.add_argument('--penalty', type=float, default=.05)
    p.add_argument('--pixel-dropout', type=float, default=.5)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2], help='seeds of the official dropout')
    p.add_argument('--reverse', action='store_true', help='scan the scene in reverse direction')
    p.add_argument('--momentum', type=float, default=None,
                   help='if set (e.g. 0.1), our RX-BIL variants forget old lines like ERX; '
                        'the official RX-BIL is unchanged and is then skipped')
    p.add_argument('--output', default='results_v20')
    return p.parse_args(argv)


def main(args):
    global MOMENTUM
    MOMENTUM = args.momentum
    if MOMENTUM is not None:
        args.hyperad_root = None  # official RX-BIL has no forgetting; not comparable here
    x = np.load(args.hsi, mmap_mode='r', allow_pickle=False)
    mask = np.load(args.mask, mmap_mode='r', allow_pickle=False)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:, :, 0]
    if x.ndim != 3 or mask.shape != x.shape[:2]:
        raise ValueError(f'Invalid shapes {x.shape} {mask.shape}')
    if args.reverse:
        x, mask = x[:, ::-1, :], np.asarray(mask)[:, ::-1]
    rx_class = None
    if args.hyperad_root:
        root = Path(args.hyperad_root).resolve()
        if not (root / 'detectors' / 'rx_bil.py').is_file():
            raise FileNotFoundError(f'detectors/rx_bil.py not found in {root}')
        sys.path.insert(0, str(root))
        from detectors.rx_bil import RX_BIL as rx_class
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / 'run_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    tic = time.perf_counter()
    with threadpool_limits(limits=1):
        rows, acts = run_scene(x, mask, args, rx_class)
    summ = summarise(rows)
    save_csv(out / 'results.csv', rows); save_csv(out / 'summary.csv', summ)
    order = ['RX_BIL_official', 'rxbil_frozen', 'rxbil_delayed_full', 'rxbil_delayed_trim05_cc',
             'rxbil_delayed_trim10_cc', 'rxbil_gate_nc', 'rxbil_gate_cc']
    S = {r['method']: r for r in summ}
    tag = 'cumulative' if MOMENTUM is None else f'forgetting, momentum {MOMENTUM}'
    lines = [f'v20 shadow gate on RX-BIL ({tag}), {args.dataset_name}{" (reversed)" if args.reverse else ""}, '
             f'shape {list(x.shape)}, warm-up {args.warmup}', '',
             f'{"method":26s} AUROC    AP      FPR90   ms/line']
    for m in order:
        if m in S:
            r = S[m]
            lines.append(f'{m:26s} {r["auroc"]:.3f}   {r["ap"]:.3f}   {r["fpr90"]:.3f}   {r["ms_per_line"]:.2f}')
    lines.append('')
    for k, a in acts.items():
        a = np.array(a)
        lines.append(f'{k}: skip {np.mean(a == "skip"):.2f}, trim05 {np.mean(a == "trim05"):.2f}, '
                     f'trim10 {np.mean(a == "trim10"):.2f}')
    lines += ['', f'elapsed {time.perf_counter() - tic:.0f} s',
              'Post hoc transfer test; rho and lambda unchanged from the ERX preregistration.']
    (out / 'v20_report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(cli())
