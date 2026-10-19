#!/usr/bin/env python3
"""HyperAD v18: frozen method on a suite of standard HAD scenes (PREREG_v2).

For every eligible scene in manifest.csv (written by convert_had.py):
  * v13 protocol: official HyperAD detectors (unmodified) and our ERX-family
    methods, line z-scores, common causal window, 3 projection seeds;
  * v12 matched-rate permutation control (line z-scores).
The warm-up / buffer length of EVERY method is set to --warmup (default 20)
because the scenes have only 100-150 lines; nothing else is changed.
Cross-scene tests (one value per scene = mean over seeds) follow PREREG_v2.
Labels are used only in metric code.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from threadpoolctl import threadpool_limits
from baseline_v3 import reference_repo
import run_v13
import run_v12

KEY = {'gate': 'shadow_cc', 'erx': 'official_erx', 'fixed': ('delayed_trim05_cc', 'delayed_trim10_cc')}
METRICS = ('auroc', 'ap', 'fpr90')


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.
    for r, i in enumerate(order):
        run = max(run, (m - r) * pvals[i]); adj[i] = min(1., run)
    return adj


def one_sided(d):
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(d, alternative='greater', zero_method='wilcox').pvalue)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hyperad-root', required=True)
    p.add_argument('--data-dir', required=True, help='Folder with *_hsi.npy, *_mask.npy and manifest.csv')
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--permutations', type=int, default=5)
    p.add_argument('--reverse', action='store_true', help='Secondary analysis: reversed scan direction')
    p.add_argument('--output', default='results_v18_suite')
    a = p.parse_args()
    import yaml
    root = reference_repo(a.hyperad_root)
    sys.path.insert(0, str(root))
    cfg0 = yaml.safe_load((root / 'test_detection' / 'config_detection.yaml').read_text(encoding='utf-8'))
    cfg = copy.deepcopy(cfg0)
    for k, v in cfg.items():
        if isinstance(v, dict) and 'buffer_len' in v:
            v['buffer_len'] = a.warmup
    dims, mom = int(cfg['erx']['projected_dimensions']), float(cfg['erx']['momentum'])
    data = Path(a.data_dir)
    manifest = list(csv.DictReader((data / 'manifest.csv').open(encoding='utf-8')))
    scenes = [r['scene'] for r in manifest if r.get('eligible') == '1']
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    meta = {**vars(a), 'scenes': scenes, 'manifest_sha256': hashlib.sha256((data / 'manifest.csv').read_bytes()).hexdigest(),
            'config_overrides': {'buffer_len (all detectors)': a.warmup}, 'dims': dims, 'momentum': mom}
    (out / 'run_config.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'V18 suite: {len(scenes)} scenes, warmup={a.warmup}, reverse={a.reverse}', flush=True)
    per_scene, all_rows, perm_rows = [], [], []
    with threadpool_limits(limits=1):
        for sc in scenes:
            tic = time.perf_counter()
            x = np.load(data / f'{sc}_hsi.npy', mmap_mode='r')
            m = np.load(data / f'{sc}_mask.npy')
            if a.reverse:
                x, m = x[:, ::-1, :], m[:, ::-1]
            rows, summary, _ = run_v13.evaluate(x, m, a.seeds, cfg, root, a.warmup, dims, mom, .8, .05, log=lambda *_: None)
            for r in summary:
                all_rows.append({'scene': sc, **r})
            com = {r['method']: r for r in summary if r['window'] == 'common_scored'}
            _, _, gate, _ = run_v12.evaluate_all(x, m, a.seeds, a.warmup, dims, mom, .8, .05, a.permutations,
                                                 log=lambda *_: None)
            g = [r for r in gate if r['mode'] == 'line_z' and r['shadow_variant'] == 'shadow_cc']
            perm_rows.append({'scene': sc, 'perm_won_ap': sum(r['perm_reps_beaten_ap'] for r in g),
                              'perm_total': sum(r['n_permutations'] for r in g),
                              'delta_ap_vs_perm': float(np.mean([r['delta_ap_shadow_minus_perm'] for r in g])),
                              'delta_fpr90_vs_perm': float(np.mean([r['delta_fpr90_shadow_minus_perm'] for r in g]))})
            row = {'scene': sc}
            for meth, r in com.items():
                for k in METRICS:
                    row[f'{meth}_{k}'] = r[f'{k}_mean']
            per_scene.append(row)
            print(f"{sc:22s} gate AUROC {row['shadow_cc_auroc']:.4f}  ERX {row['official_erx_auroc']:.4f}  "
                  f"({time.perf_counter() - tic:.1f}s)", flush=True)
            for path, rr in ((out / 'per_scene.csv', per_scene), (out / 'all_methods.csv', all_rows), (out / 'permutation.csv', perm_rows)):
                keys = list(dict.fromkeys(k for r in rr for k in r))
                with path.open('w', newline='', encoding='utf-8') as fh:
                    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rr)
    # ---------------- cross-scene analysis (PREREG_v2) ----------------
    S = len(per_scene)
    def col(meth, k):
        return np.array([r[f'{meth}_{k}'] for r in per_scene])
    sign = {'auroc': 1, 'ap': 1, 'fpr90': -1}
    L = [f'HyperAD v18 suite ({"reversed" if a.reverse else "forward"}), {S} scenes, warm-up {a.warmup} lines', '']
    # H1: gate vs official ERX
    d1 = {k: sign[k] * (col(KEY['gate'], k) - col(KEY['erx'], k)) for k in METRICS}
    p1 = {k: one_sided(d1[k]) for k in METRICS}
    adj = dict(zip(('ap', 'fpr90'), holm(np.array([p1['ap'], p1['fpr90']]))))
    L += ['H1 gate vs official ERX (one-sided Wilcoxon signed-rank over scenes; positive = gate better):',
          f"  AUROC (primary): median diff {np.median(d1['auroc']):+.4f}, wins {int((d1['auroc'] > 0).sum())}/{S}, p = {p1['auroc']:.4f}"]
    for k in ('ap', 'fpr90'):
        L.append(f"  {k.upper()} (secondary): median diff {np.median(d1[k]):+.4f}, wins {int((d1[k] > 0).sum())}/{S}, "
                 f"p = {p1[k]:.4f}, Holm p = {adj[k]:.4f}")
    # H2: gate vs per-scene best fixed rule (chosen post hoc per scene, conservative)
    L += ['', 'H2 gate vs best fixed rule per scene (best chosen per scene and metric; conservative):']
    for k in METRICS:
        f05, f10 = col(KEY['fixed'][0], k), col(KEY['fixed'][1], k)
        best = np.maximum(f05, f10) if sign[k] > 0 else np.minimum(f05, f10)
        d = sign[k] * (col(KEY['gate'], k) - best)
        pg, pr = one_sided(d), one_sided(-d)
        verdict = 'gain' if pg < .05 else ('regret' if pr < .05 else 'no difference')
        L.append(f"  {k.upper():5s}: median diff {np.median(d):+.4f}, wins {int((d > 0).sum())}/{S}, "
                 f"p(gain) = {pg:.4f}, p(regret) = {pr:.4f} -> {verdict}")
    # H3 descriptive
    won = sum(1 for r in perm_rows if r['perm_won_ap'] >= 0.8 * r['perm_total'])
    L += ['', f'H3 (descriptive) scenes where the gate beats >= 80% of permuted gates on AP: {won}/{S}']
    # H4 ranks
    meths = [m for m in dict.fromkeys(r['method'] for r in all_rows)]
    ranks = {m: [] for m in meths}
    for sc in [r['scene'] for r in per_scene]:
        sub = [r for r in all_rows if r['scene'] == sc and r['window'] == 'common_scored']
        order = sorted(sub, key=lambda r: -r['auroc_mean'])
        for i, r in enumerate(order):
            ranks[r['method']].append(i + 1)
    L += ['', 'H4 (descriptive) mean AUROC rank over scenes (1 = best):']
    for m_, rk in sorted(ranks.items(), key=lambda t: np.mean(t[1])):
        L.append(f'  {m_:24s} {np.mean(rk):5.2f}')
    (out / 'cross_scene_report.txt').write_text('\n'.join(L) + '\n', encoding='utf-8')
    print('\n' + '\n'.join(L))
    print('Outputs saved:', out.resolve())


if __name__ == '__main__':
    main()
