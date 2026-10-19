#!/usr/bin/env python3
"""HyperAD v13: official-protocol comparison against the HyperAD baselines.

Method is frozen. This script only connects Shadow to the official benchmark:

* Official detectors (erx, rx_baseline, lbl_ad, rt_ck_rxd, rx_bil, and
  cdlss_ad if included) are imported UNMODIFIED from HyperAD-main and built
  with test_detection/config_detection.yaml parameters (normalise_md=True).
* Score maps are assembled exactly like utils.simulation_core.DetectionSimulator:
  column i - line_offset receives the output of line i; unscored columns are 0.
* Our methods (frozen, delayed_trim05/10_cc, shadow_cc, shadow_nc) share ERX's
  projection for each seed and are line z-scored, matching the official setting.

Two evaluation windows are reported:
  official_full : whole image incl. zero-filled warm-up columns (HyperAD style)
  common_scored : only columns scored by EVERY method (fair causal window)

Threshold-AUC metrics are reported three ways:
  auc_dt/ft_hyperad_code   : HyperAD utils.metrics as called by
                             show_detection_results.py with the installed sklearn
  auc_dt/ft_hyperad_legacy : same code with sklearn<1.3 first threshold (max+1)
  auc_dt/ft                : correct value. For min-max normalised scores s,
                             AUC(D,tau)=mean s over anomalies, AUC(F,tau)=mean s
                             over background (exact integral of P(s>tau)).
AUC_TD=(AUROC+AUC_dt)/2 and AUC_BS=(AUROC-AUC_ft+1)/2 use the correct values.
Labels are used only in metric code. Projection seeds are not independent scenes.
"""
from __future__ import annotations
import argparse, hashlib, importlib, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from threadpoolctl import threadpool_limits
from baseline_v3 import stable_md, normalise_line, reference_repo, shared_projector
from run_shadow import prepare_warmup, save_csv
from run_v12 import GatedPolicy, shadow_chooser, fixed_chooser

_trapz = getattr(np, 'trapezoid', None) or np.trapz
OFFICIAL = {  # config key -> (module, class)
    'erx': ('erx', 'ERX'), 'rx_baseline': ('rx_baseline', 'RXBaseline'),
    'rt_ck_rxd': ('rt_ck_rxd', 'RT_CK_RXD'), 'rx_bil': ('rx_bil', 'RX_BIL'),
    'cdlss_ad': ('cdlss_ad', 'CDLSS_AD'), 'lbl_ad': ('lbl_ad', 'LblAD'),
}
OURS = ('frozen', 'delayed_trim05_cc', 'delayed_trim10_cc', 'shadow_cc', 'shadow_nc')


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def hyperad_auc(rate, thresholds):
    """Verbatim logic of HyperAD utils.metrics.calc_auc_dt / calc_auc_ft."""
    with np.errstate(invalid='ignore', divide='ignore'):
        n = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())
        return float(_trapz(rate, n[::-1]))


def all_metrics(y, s):
    s = np.nan_to_num(np.asarray(s, dtype=np.float64))
    fpr, tpr, thr = roc_curve(y, s)
    legacy = thr.copy()
    if np.isinf(legacy[0]):
        legacy[0] = s.max() + 1  # sklearn < 1.3 behaviour
    ix = min(int(np.searchsorted(tpr, .90, side='left')), len(fpr) - 1)
    span = s.max() - s.min()
    sn = (s - s.min()) / span if span > 0 else np.zeros_like(s)
    auroc = float(roc_auc_score(y, s))
    dt, ft = float(sn[y].mean()), float(sn[~y].mean())
    return {'auroc': auroc, 'ap': float(average_precision_score(y, s)), 'fpr90': float(fpr[ix]),
            'auc_dt': dt, 'auc_ft': ft, 'auc_td': (auroc + dt) / 2, 'auc_bs': (auroc - ft + 1) / 2,
            'auc_dt_hyperad_code': hyperad_auc(tpr, thr), 'auc_ft_hyperad_code': hyperad_auc(fpr, thr),
            'auc_dt_hyperad_legacy': hyperad_auc(tpr, legacy),
            'auc_ft_hyperad_legacy': hyperad_auc(fpr, legacy)}


# --------------------------------------------------------------------------
# Runners
# --------------------------------------------------------------------------
def build_official(name, cfg, n_pixels, n_bands, module):
    c = cfg[name]
    common = dict(n_bands=n_bands, n_pixels=n_pixels, buffer_len=c['buffer_len'],
                  normalise_md=c['normalise_md'])
    cls = getattr(module, OFFICIAL[name][1])
    if name == 'erx':
        return cls(n_projdims=c['projected_dimensions'], momentum=c['momentum'], **common)
    if name == 'rx_baseline':
        return cls(line_offset=c['buffer_len'] // 2, **common)
    if name == 'rx_bil':
        return cls(pixel_dropout=c['pixel_dropout'], **common)
    if name == 'lbl_ad':
        return cls(pca_dims=c['pca_dims'], **common)
    return cls(**common)


def simulate(det, x):
    """DetectionSimulator.run semantics on a freshly constructed detector."""
    h, w, _ = x.shape
    out = np.zeros((h, w)); filled = np.zeros(w, dtype=bool)
    off = getattr(det, 'line_offset', 0)
    elapsed = 0.
    for i in range(w):
        line = np.asarray(x[:, i, :])
        tic = time.perf_counter()
        y = det.forward(line)
        elapsed += time.perf_counter() - tic
        if y is not None:
            out[:, i - off] = y
            filled[i - off] = True
    return out, filled, 1000 * elapsed / w


def run_ours(x, projector_seed, warmup, dims, momentum, core_fraction, penalty):
    h, w, _ = x.shape
    proj, mu0, cov0 = prepare_warmup(x, projector_seed, warmup, dims, momentum)
    pols = {'delayed_trim05_cc': GatedPolicy(mu0, cov0, momentum, fixed_chooser('trim05'), True, ('trim05',)),
            'delayed_trim10_cc': GatedPolicy(mu0, cov0, momentum, fixed_chooser('trim10'), True, ('trim10',)),
            'shadow_cc': GatedPolicy(mu0, cov0, momentum, shadow_chooser(core_fraction, penalty), True),
            'shadow_nc': GatedPolicy(mu0, cov0, momentum, shadow_chooser(core_fraction, penalty), False)}
    maps = {k: np.zeros((h, w)) for k in OURS}
    times = {k: 0. for k in OURS}
    for t in range(warmup, w):
        z = np.asarray(proj.transform(np.asarray(x[:, t, :], dtype=np.float64)), dtype=np.float64)
        tic = time.perf_counter(); maps['frozen'][:, t] = normalise_line(stable_md(z, mu0, cov0))
        times['frozen'] += time.perf_counter() - tic
        for k, p in pols.items():
            tic = time.perf_counter(); maps[k][:, t] = normalise_line(p.step(z))
            times[k] += time.perf_counter() - tic
    filled = np.zeros(w, dtype=bool); filled[warmup:] = True
    skip = {k: float(np.mean([a == 'skip' for a in pols[k].actions])) for k in ('shadow_cc', 'shadow_nc')}
    # ms/line excludes projection (shared by all ERX-family methods)
    return {k: (maps[k], filled, 1000 * times[k] / w) for k in OURS}, proj, skip


def evaluate(x, mask, seeds, cfg, repo, warmup, dims, momentum, core_fraction, penalty, log=print):
    h, w, b = x.shape
    names = [n for n in ('erx', 'rx_baseline', 'lbl_ad', 'rt_ck_rxd', 'rx_bil', 'cdlss_ad')
             if cfg.get(n, {}).get('include')]
    modules = {n: importlib.import_module(f'detectors.{OFFICIAL[n][0]}') for n in names}
    rows, skips = [], []
    for seed in seeds:
        tic = time.perf_counter()
        results, proj, skip = run_ours(x, seed, warmup, dims, momentum, core_fraction, penalty)
        skips.append({'seed': seed, **{f'skip_rate_{k}': v for k, v in skip.items()}})
        for n in names:
            np.random.seed(seed)  # rx_bil pixel dropout and any np.random use
            det = build_official(n, cfg, h, b, modules[n])
            if n == 'erx':
                det.random_projector = proj  # same projection as our ERX-family methods
            results[f'official_{n}'] = simulate(det, x)
        common = np.logical_and.reduce([f for _, f, _ in results.values()])
        for window, cols in (('official_full', np.ones(w, dtype=bool)), ('common_scored', common)):
            y = np.asarray(mask[:, cols] != 0, dtype=bool).ravel()
            for name, (m, _, ms) in results.items():
                rows.append({'seed': seed, 'window': window, 'method': name,
                             'n_columns': int(cols.sum()), **all_metrics(y, m[:, cols].ravel()),
                             'ms_per_line': ms, 'line_offset': 49 if name == 'official_rx_baseline' else 0})
        log(f'seed={seed} done in {time.perf_counter() - tic:.1f}s, common window={int(common.sum())} cols, '
            f'skip cc={skip["shadow_cc"]:.3f} nc={skip["shadow_nc"]:.3f}')
    keys = ['auroc', 'ap', 'fpr90', 'auc_dt', 'auc_ft', 'auc_td', 'auc_bs',
            'auc_dt_hyperad_code', 'auc_ft_hyperad_code', 'auc_dt_hyperad_legacy',
            'auc_ft_hyperad_legacy', 'ms_per_line']
    summary = []
    for window in ('common_scored', 'official_full'):
        for name in dict.fromkeys(r['method'] for r in rows):
            rs = [r for r in rows if r['window'] == window and r['method'] == name]
            s = {'window': window, 'method': name, 'n_seeds': len(rs), 'n_columns': rs[0]['n_columns']}
            for k in keys:
                v = np.array([r[k] for r in rs], dtype=float)
                fin = v[np.isfinite(v)]
                s[k + '_mean'] = float(fin.mean()) if len(fin) else float('nan')
                s[k + '_sd'] = float(fin.std(ddof=1)) if len(fin) > 1 else float('nan')
            summary.append(s)
    return rows, summary, skips


def write_report(out, summary, dataset):
    L = [f'HyperAD v13 RESMI PROTOKOL KARSILASTIRMASI ({dataset})',
         'Tum skorlar satir bazli z-skor (config_detection.yaml). Tohumlar bagimsiz sahne degildir.',
         'rx_baseline 49 satir gecikmeli ve nedensel degildir.', '']
    for window in ('common_scored', 'official_full'):
        L.append(f'[{window}]')
        L.append(f"  {'yontem':24s} {'AUROC':>7s} {'AP':>7s} {'FPR90':>7s} {'AUCdt':>7s} {'AUCft':>7s} "
                 f"{'dt_kod':>7s} {'dt_eski':>7s} {'ms/sat':>7s}")
        for r in [r for r in summary if r['window'] == window]:
            L.append(f"  {r['method']:24s} {r['auroc_mean']:7.4f} {r['ap_mean']:7.4f} {r['fpr90_mean']:7.4f} "
                     f"{r['auc_dt_mean']:7.4f} {r['auc_ft_mean']:7.4f} {r['auc_dt_hyperad_code_mean']:7.4f} "
                     f"{r['auc_dt_hyperad_legacy_mean']:7.4f} {r['ms_per_line_mean']:7.3f}")
        L.append('')
    L += ['AUCdt/AUCft: dogru deger. dt_kod: HyperAD kodu (yuklu sklearn). dt_eski: HyperAD kodu, sklearn<1.3.',
          'dt_kod nan ise yuklu sklearn ilk esigi inf veriyor. dt_eski ile AUCdt farki, kod hatasinin buyuklugudur.',
          'ms/satir: tek is parcacigi; bizim yontemlerde ortak projeksiyon suresi haric.']
    (Path(out) / 'v13_report_TR.txt').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hyperad-root', required=True)
    p.add_argument('--config', default=None, help='Default: <root>/test_detection/config_detection.yaml')
    p.add_argument('--hsi', required=True)
    p.add_argument('--mask', required=True)
    p.add_argument('--dataset-name', default='Beach')
    p.add_argument('--core-fraction', type=float, default=.80)
    p.add_argument('--penalty', type=float, default=.05)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--reverse', action='store_true', help='Scan lines in reverse order (pure flip of image and mask)')
    p.add_argument('--output', default='results_v13')
    a = p.parse_args()
    import yaml
    root = reference_repo(a.hyperad_root)
    sys.path.insert(0, str(root))
    cfg_path = Path(a.config) if a.config else root / 'test_detection' / 'config_detection.yaml'
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    e = cfg['erx']
    warmup, dims, momentum = int(e['buffer_len']), int(e['projected_dimensions']), float(e['momentum'])
    if not all(cfg[n]['normalise_md'] for n in OFFICIAL if cfg.get(n, {}).get('include')):
        raise ValueError('Official detection config expected normalise_md=True for included detectors')
    x = np.load(a.hsi, mmap_mode='r', allow_pickle=False)
    mask = np.load(a.mask, mmap_mode='r', allow_pickle=False)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if a.reverse:
        x, mask = x[:, ::-1, :], mask[:, ::-1]
    if x.ndim != 3 or mask.shape != x.shape[:2]:
        raise ValueError(f'shape mismatch image {x.shape}, mask {mask.shape}')
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    hashes = {n: hashlib.sha256((root / 'detectors' / f'{OFFICIAL[n][0]}.py').read_bytes()).hexdigest()
              for n in OFFICIAL if (root / 'detectors' / f'{OFFICIAL[n][0]}.py').exists()}
    meta = vars(a).copy()
    meta.update(shape=list(x.shape), config_used=str(cfg_path), config=cfg, detector_sha256=hashes,
                warmup=warmup, projected_dims=dims, momentum=momentum)
    (out / 'run_config.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f'V13 {a.dataset_name} shape={x.shape} seeds={a.seeds} warmup={warmup} dims={dims} momentum={momentum}', flush=True)
    with threadpool_limits(limits=1):
        rows, summary, skips = evaluate(x, mask, a.seeds, cfg, root, warmup, dims, momentum,
                                        a.core_fraction, a.penalty)
    save_csv(out / 'results_per_seed.csv', rows)
    save_csv(out / 'summary.csv', summary)
    save_csv(out / 'shadow_skip_rates.csv', skips)
    write_report(out, summary, a.dataset_name)
    print('Outputs saved:', out.resolve())


if __name__ == '__main__':
    main()
