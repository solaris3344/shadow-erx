#!/usr/bin/env python3
"""HyperAD v19: controlled contamination experiment (post hoc, not preregistered).

Question: does the advantage of the shadow gate over fixed trimming grow when
a target covers a larger share of a scan line than trimming can remove?

Design
------
* A real scene is used as background. After the warm-up, `--n-targets`
  rectangular targets are implanted at evenly spaced positions along the scan.
  Each target covers `--target-lines` consecutive lines and a contiguous block
  of round(f * n_pixels) pixels across the line, at a random cross-track offset.
* The implanted pixels are copies of annotated anomaly pixels of the SAME scene,
  drawn at random (labels are used only to build the test cube, never by any
  detector). With --source outlier, the spectra of the 1% pixels farthest from
  the warm-up mean are used instead, so no label is read at all.
* The per-line target fraction f is varied (default 0, 2, 5, 10, 15, 20, 30 %).
  f = 0 is the unmodified scene.
* Methods (all line z-scored, ERX projection of each seed, same warm-up):
  frozen, erx (immediate full update, re-implementation of ERX),
  delayed_trim05_cc, delayed_trim10_cc, shadow_cc. With --hyperad-root the
  unmodified official ERX is added as ERX_original.
* Metrics on lines warmup .. W-1:
  all       : original + implanted anomalies vs background
  original  : original anomalies vs background, implanted pixels excluded.
              This measures collateral damage: whether absorbing a large
              target into the background model hides the other anomalies.
* The shadow skip rate is reported separately for target lines and other lines.

Run once per scene. Nothing here is tuned; parameters are the preregistered ones.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits
from scipy.stats import chi2
from sklearn.random_projection import SparseRandomProjection

# ==========================================================================
# Embedded helpers, copied verbatim from baseline_v3.py, run_shadow.py and
# run_v12.py so that this file runs on its own (e.g. inside HyperAD-main).
# ==========================================================================
# --- from run_shadow.py (verbatim, renamed proposal_v6) ---
def proposal_v6(z, mu, cov, fraction, momentum, reg=1e-5):
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


# --- from baseline_v3.py (verbatim) ---
def reference_repo(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    if not (root / "detectors" / "erx.py").is_file():
        raise FileNotFoundError(f"Cannot find detectors/erx.py inside: {root}")
    return root


# --- from baseline_v3.py (verbatim) ---
def shared_projector(n_pixels: int, n_bands: int, n_proj: int, seed: int):
    p = SparseRandomProjection(n_components=n_proj, random_state=seed)
    p.fit(np.ones((n_pixels, n_bands), dtype=np.float64))
    return p


# --- from baseline_v3.py (verbatim) ---
def stable_md(z: np.ndarray, mu: np.ndarray, cov: np.ndarray, reg: float = 1e-5):
    delta = z - mu
    eye = np.eye(cov.shape[0], dtype=np.float64)
    jitter = float(reg)
    for _ in range(8):
        try:
            chol = np.linalg.cholesky(cov + jitter * eye)
            return np.linalg.norm(np.linalg.solve(chol, delta.T), axis=0)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("Covariance could not be stabilized")


# --- from baseline_v3.py (verbatim) ---
def normalise_line(a: np.ndarray) -> np.ndarray:
    sd = float(a.std())
    return (a - float(a.mean())) / sd if sd > 1e-12 else np.zeros_like(a)


# --- from baseline_v3.py (verbatim) ---
class PrequentialERX:
    """Score-then-update, with optional hard trim or bounded-influence weights.

    Reproduces the v2 update family. It is an ablation of robust statistics,
    NOT claimed to be a new estimator.
    """
    def __init__(self, projector, buffer_len: int, momentum: float,
                 trim_fraction: float = 0., huber_q: float | None = None,
                 normalise_md: bool = False):
        self.projector = projector
        self.buffer_len = int(buffer_len)
        self.momentum = float(momentum)
        self.trim_fraction = float(trim_fraction)
        self.huber_q = huber_q
        self.normalise_md = normalise_md
        self.line_count = 0
        self.mean = None
        self.cov = None
        self.reg = 1e-5
        if not (0 <= self.trim_fraction < 1):
            raise ValueError("trim_fraction must lie in [0, 1)")

    def forward(self, line):
        z = np.asarray(self.projector.transform(line), dtype=np.float64)
        d = z.shape[1]
        if self.line_count == 0:
            self.mean = z.mean(axis=0)
            self.cov = np.cov(z, rowvar=False) + np.eye(d) * self.reg
            self.line_count += 1
            return None
        raw_score = stable_md(z, self.mean, self.cov, self.reg)
        # Match v2: only perform robust updates after the 99-line warm-up.
        emit = self.line_count >= self.buffer_len
        update_z = z
        weights = None
        if emit and self.trim_fraction:
            k = max(d + 2, int(round(len(z) * (1. - self.trim_fraction))))
            k = min(k, len(z))
            ix = np.argpartition(raw_score, k - 1)[:k]
            update_z = z[ix]
        elif emit and self.huber_q is not None:
            cutoff = float(np.quantile(raw_score, self.huber_q))
            weights = np.minimum(1.0, cutoff / np.maximum(raw_score, 1e-12))
        if weights is None:
            current_mu = update_z.mean(axis=0)
            current_cov = np.cov(update_z, rowvar=False)
        else:
            total = float(weights.sum())
            current_mu = np.average(z, axis=0, weights=weights)
            centered = z - current_mu
            denominator = total - float((weights ** 2).sum()) / total
            current_cov = (centered.T * weights) @ centered / max(denominator, 1.)
        current_cov = current_cov + np.eye(d) * self.reg
        a = self.momentum
        self.mean = (1. - a) * self.mean + a * current_mu
        self.cov = (1. - a) * self.cov + a * current_cov
        self.line_count += 1
        if not emit:
            return None
        return normalise_line(raw_score) if self.normalise_md else raw_score


# --- from run_shadow.py (verbatim) ---
ACTIONS = ('skip', 'trim05', 'trim10')


# --- from run_shadow.py (verbatim) ---
TRIM = {'trim05': .05, 'trim10': .10}


# --- from run_shadow.py (verbatim) ---
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


# --- from run_shadow.py (verbatim) ---
def prepare_warmup(x, seed, warmup, proj_dims, momentum):
    n_pixels, _, n_bands = x.shape
    projector = shared_projector(n_pixels, n_bands, proj_dims, seed)
    initializer = PrequentialERX(projector, warmup, momentum,
                                 trim_fraction=0, normalise_md=False)
    for col in range(warmup):
        line = np.asarray(x[:, col, :], dtype=np.float64)
        initializer.forward(line)
    return projector, initializer.mean.copy(), initializer.cov.copy()


# --- from run_v12.py (verbatim) ---
def consistency_factor(d: int, keep: float) -> float:
    """Gaussian consistency factor for a covariance of the `keep` fraction
    with the smallest Mahalanobis distances (Croux & Haesbroeck style)."""
    if not 0 < keep <= 1:
        raise ValueError('keep must lie in (0,1]')
    if keep == 1:
        return 1.0
    q = chi2.ppf(keep, d)
    return float(keep / chi2.cdf(q, d + 2))


# --- from run_v12.py (verbatim) ---
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


# --- from run_v12.py (verbatim) ---
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


# --- from run_v12.py (verbatim) ---
def shadow_chooser(core_fraction, penalty):
    def choose(z, raw, pending, mu, cov):
        k = min(len(z), max(z.shape[1] + 2, int(round(len(z) * core_fraction))))
        ix = np.argsort(raw, kind='stable')[:k]
        costs = {a: robust_predictive_cost(z, ix, *pending[a], mu, cov, penalty) for a in ACTIONS}
        return min(ACTIONS, key=lambda a: costs[a])
    return choose


# --- from run_v12.py (verbatim) ---
def fixed_chooser(action):
    return lambda *_: action


# ==========================================================================

METHODS = ('frozen', 'erx', 'delayed_trim05_cc', 'delayed_trim10_cc', 'shadow_cc')


# --------------------------------------------------------------------------
# Implant plan
# --------------------------------------------------------------------------
def make_plan(n_pixels, n_lines, warmup, fraction, n_targets, target_lines, rng):
    """List of (first_line, last_line, first_pixel, last_pixel), inclusive."""
    if fraction <= 0:
        return []
    width = max(1, int(round(fraction * n_pixels)))
    usable = n_lines - warmup
    gap = usable // (n_targets + 1)
    if gap <= target_lines:
        raise ValueError('Scene too short for the requested number of targets')
    plan = []
    for k in range(1, n_targets + 1):
        start = warmup + k * gap - target_lines // 2
        off = int(rng.integers(0, n_pixels - width + 1))
        plan.append((start, start + target_lines - 1, off, off + width - 1))
    return plan


def target_spectra(x, mask, warmup, source, rng, max_pool=20000):
    if source == 'mask':
        ii, jj = np.nonzero(mask)
        if len(ii) == 0:
            raise ValueError('Mask has no anomalies; use --source outlier')
        pick = rng.choice(len(ii), size=min(max_pool, len(ii)), replace=False)
        return np.stack([np.asarray(x[ii[p], jj[p], :], dtype=np.float64) for p in pick])
    # label-free: pixels farthest from the warm-up mean in full-band Euclidean distance
    warm = np.asarray(x[:, :warmup, :], dtype=np.float64).reshape(-1, x.shape[2])
    m = warm.mean(axis=0)
    cols = rng.choice(np.arange(warmup, x.shape[1]), size=min(200, x.shape[1] - warmup), replace=False)
    pool = np.concatenate([np.asarray(x[:, c, :], dtype=np.float64) for c in cols])
    d = np.linalg.norm(pool - m, axis=1)
    return pool[d >= np.quantile(d, 0.99)]


def implant_line(line, t, plan, spectra, rng):
    """Return a modified copy of `line` and the implanted-pixel boolean vector."""
    hit = np.zeros(line.shape[0], dtype=bool)
    for a, b, p0, p1 in plan:
        if a <= t <= b:
            idx = np.arange(p0, p1 + 1)
            line[idx] = spectra[rng.integers(0, len(spectra), size=len(idx))]
            hit[idx] = True
    return line, hit


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metric_row(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    ix = min(int(np.searchsorted(tpr, .90, side='left')), len(fpr) - 1)
    return {'auroc': float(roc_auc_score(y, s)), 'ap': float(average_precision_score(y, s)),
            'fpr90': float(fpr[ix])}


# --------------------------------------------------------------------------
# One seed, one fraction
# --------------------------------------------------------------------------
def run_one(x, mask, seed, fraction, args, spectra, erx_class=None):
    h, w, nb = x.shape
    rng_plan = np.random.default_rng(1000 * seed + int(round(1000 * fraction)) + args.implant_seed)
    plan = make_plan(h, w, args.warmup, fraction, args.n_targets, args.target_lines, rng_plan)
    rng_pix = np.random.default_rng(7 + seed)
    proj, mu0, cov0 = prepare_warmup(x, seed, args.warmup, args.projected_dims, args.momentum)
    pol = {
        'delayed_trim05_cc': GatedPolicy(mu0, cov0, args.momentum, fixed_chooser('trim05'), True, ('trim05',)),
        'delayed_trim10_cc': GatedPolicy(mu0, cov0, args.momentum, fixed_chooser('trim10'), True, ('trim10',)),
        'shadow_cc': GatedPolicy(mu0, cov0, args.momentum,
                                 shadow_chooser(args.core_fraction, args.penalty), True),
    }
    erx = PrequentialERX(proj, args.warmup, args.momentum, trim_fraction=0)
    official = None
    if erx_class is not None:
        official = erx_class(n_bands=nb, n_pixels=h, buffer_len=args.warmup,
                             n_projdims=args.projected_dims, momentum=args.momentum,
                             normalise_md=(args.score_mode == 'line_z'))
        official.random_projector = proj
    for col in range(args.warmup):
        line = np.asarray(x[:, col, :], dtype=np.float64)
        erx.forward(line)
        if official is not None:
            official.forward(line)
    names = list(METHODS) + (['ERX_original'] if official is not None else [])
    n_eval = w - args.warmup
    out = {k: np.empty((n_eval, h)) for k in names}
    implanted = np.zeros((n_eval, h), dtype=bool)
    target_line = np.zeros(n_eval, dtype=bool)
    nrm = normalise_line if args.score_mode == 'line_z' else (lambda a: a)
    for t in range(args.warmup, w):
        line, hit = implant_line(np.array(x[:, t, :], dtype=np.float64), t, plan, spectra, rng_pix)
        i = t - args.warmup
        implanted[i], target_line[i] = hit, hit.any()
        z = np.asarray(proj.transform(line), dtype=np.float64)
        out['frozen'][i] = nrm(stable_md(z, mu0, cov0))
        out['erx'][i] = nrm(erx.forward(line))
        for k, p in pol.items():
            out[k][i] = nrm(p.step(z))
        if official is not None:
            out['ERX_original'][i] = official.forward(line)
    orig = np.asarray(mask[:, args.warmup:w] != 0).T  # [lines, pixels]
    y_all = (orig | implanted).ravel()
    keep = ~implanted.ravel()
    y_orig = orig.ravel()[keep]
    rows = []
    for k in names:
        s = out[k].ravel()
        r = {'fraction': fraction, 'seed': seed, 'method': k}
        r.update({f'{m}_all': v for m, v in metric_row(y_all, s).items()})
        if y_orig.any():
            r.update({f'{m}_orig': v for m, v in metric_row(y_orig, s[keep]).items()})
        rows.append(r)
    # Shadow decision i applies to the candidate built from line i-1 (decisions start at eval line 1)
    acts = np.array(pol['shadow_cc'].actions)
    cand_target = target_line[:len(acts)]  # candidate line index = decision index
    skip = acts == 'skip'
    info = {'fraction': fraction, 'seed': seed, 'n_targets': len(plan),
            'target_width_px': (plan[0][3] - plan[0][2] + 1) if plan else 0,
            'skip_rate_target_lines': float(skip[cand_target].mean()) if cand_target.any() else float('nan'),
            'skip_rate_other_lines': float(skip[~cand_target].mean())}
    return rows, info


def summarise(rows):
    keys = [k for k in rows[0] if k not in ('fraction', 'seed', 'method')]
    groups = {}
    for r in rows:
        groups.setdefault((r['fraction'], r['method']), []).append(r)
    out = []
    for (f, m), g in sorted(groups.items()):
        s = {'fraction': f, 'method': m, 'n_seeds': len(g)}
        for k in keys:
            v = [r[k] for r in g if k in r]
            if v:
                s[f'{k}_mean'], s[f'{k}_sd'] = float(np.mean(v)), float(np.std(v))
        out.append(s)
    return out


def save_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader(); wr.writerows(rows)


def plot(summary, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    style = {'frozen': ('Frozen ERX', '#8c8c8c', ':'), 'erx': ('ERX', '#d62728', '--'),
             'ERX_original': ('ERX (official)', '#d62728', '--'),
             'delayed_trim05_cc': ('Delayed Trim05', '#ff7f0e', '-.'),
             'delayed_trim10_cc': ('Delayed Trim10', '#2ca02c', '-.'),
             'shadow_cc': ('Shadow gate', '#1f77b4', '-')}
    show = [m for m in style if any(r['method'] == m for r in summary)]
    if 'ERX_original' in show and 'erx' in show:
        show.remove('erx')
    panels = [('auroc_all', 'AUROC (all anomalies)'), ('ap_all', 'AP (all anomalies)'),
              ('auroc_orig', 'AUROC (original anomalies only)')]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, (k, lab) in zip(axes, panels):
        for m in show:
            g = [r for r in summary if r['method'] == m and f'{k}_mean' in r]
            if not g:
                continue
            f = [100 * r['fraction'] for r in g]
            ax.plot(f, [r[f'{k}_mean'] for r in g], style[m][2], marker='o', ms=3.5,
                    color=style[m][1], label=style[m][0], lw=1.4)
        for b in (5, 10):
            ax.axvline(b, color='0.85', lw=0.8, zorder=0)
        ax.set_xlabel('Target share of a scan line (%)'); ax.set_ylabel(lab)
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--hsi', required=True); p.add_argument('--mask', required=True)
    p.add_argument('--dataset-name', default='Scene')
    p.add_argument('--hyperad-root', default=None, help='optional, adds unmodified official ERX')
    p.add_argument('--warmup', type=int, default=99)
    p.add_argument('--projected-dims', type=int, default=5)
    p.add_argument('--momentum', type=float, default=.10)
    p.add_argument('--core-fraction', type=float, default=.80)
    p.add_argument('--penalty', type=float, default=.05)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--fractions', type=float, nargs='+', default=[0, .02, .05, .10, .15, .20, .30])
    p.add_argument('--n-targets', type=int, default=6)
    p.add_argument('--target-lines', type=int, default=40)
    p.add_argument('--source', choices=('mask', 'outlier'), default='mask')
    p.add_argument('--implant-seed', type=int, default=0)
    p.add_argument('--score-mode', choices=('line_z', 'raw'), default='line_z',
                   help='line_z = official per-line z-scores; raw = Mahalanobis distances (diagnostic)')
    p.add_argument('--output', default='results_v19')
    return p.parse_args(argv)


def main(args):
    x = np.load(args.hsi, mmap_mode='r', allow_pickle=False)
    mask = np.load(args.mask, mmap_mode='r', allow_pickle=False)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[:, :, 0]
    if x.ndim != 3 or mask.shape != x.shape[:2]:
        raise ValueError(f'Invalid shapes {x.shape} {mask.shape}')
    erx_class = None
    if args.hyperad_root:
        root = reference_repo(args.hyperad_root); sys.path.insert(0, str(root))
        from detectors.erx import ERX as erx_class
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / 'run_config.json').write_text(json.dumps(vars(args), indent=2), encoding='utf-8')
    spectra = target_spectra(x, np.asarray(mask), args.warmup, args.source,
                             np.random.default_rng(args.implant_seed))
    rows, infos = [], []
    tic = time.perf_counter()
    with threadpool_limits(limits=1):
        for f in args.fractions:
            for s in args.seeds:
                r, info = run_one(x, mask, s, f, args, spectra, erx_class)
                rows += r; infos.append(info)
                sh = [q for q in r if q['method'] == 'shadow_cc'][0]
                t10 = [q for q in r if q['method'] == 'delayed_trim10_cc'][0]
                print(f'f={f:.2f} seed={s}  AUROC gate {sh["auroc_all"]:.3f} vs Trim10 {t10["auroc_all"]:.3f}'
                      f'  | skip on target lines {info["skip_rate_target_lines"]:.2f}', flush=True)
    summary = summarise(rows)
    save_csv(out / 'results_per_seed.csv', rows)
    save_csv(out / 'summary.csv', summary)
    save_csv(out / 'shadow_skip_rates.csv', infos)
    plot(summary, out / f'contamination_{args.dataset_name}.png',
         f'{args.dataset_name}: implanted targets, mean of {len(args.seeds)} seeds')
    # short report
    lines = [f'v19 controlled contamination, {args.dataset_name}, source={args.source}, scores={args.score_mode}, '
             f'{args.n_targets} targets x {args.target_lines} lines, seeds={args.seeds}', '',
             'fraction | AUROC_all gate / Trim10 / ERX | AP_all gate / Trim10 / ERX | AUROC_orig gate / Trim10 / ERX | skip target / other lines']
    for f in args.fractions:
        g = {r['method']: r for r in summary if r['fraction'] == f}
        sk = np.nanmean([i['skip_rate_target_lines'] for i in infos if i['fraction'] == f])
        so = np.nanmean([i['skip_rate_other_lines'] for i in infos if i['fraction'] == f])
        G = lambda m, k: g[m].get(k, float('nan'))
        tri = lambda k: ' / '.join(f'{G(m, k):.3f}' for m in ('shadow_cc', 'delayed_trim10_cc', 'erx'))
        lines.append(f'{100*f:5.1f}% | {tri("auroc_all_mean")} | {tri("ap_all_mean")} | '
                     f'{tri("auroc_orig_mean")} | {sk:.2f} / {so:.2f}')
    lines += ['', f'elapsed {time.perf_counter() - tic:.0f} s',
              'Post hoc controlled experiment; not part of either preregistration.']
    (out / 'v19_report.txt').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main(cli())
