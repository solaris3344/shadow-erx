#!/usr/bin/env python3
"""HyperAD v17: qualitative detection maps (figure only, no new numbers).

For one projection seed, renders pseudo-RGB, ground truth and the line z-scored
score maps of official ERX, Delayed Trim10 (cc) and the Shadow gate (cc), with the
frozen protocol. Also saves the maps (float16) so the figure can be re-styled
without re-running.  Optional --lines START END crops a scan-line window.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from baseline_v3 import normalise_line, reference_repo
from run_shadow import prepare_warmup
from run_v12 import GatedPolicy, shadow_chooser, fixed_chooser
from run_v13 import build_official, simulate

METHODS = (('official_erx', 'ERX (official)'), ('delayed_trim10_cc', 'Delayed Trim10 (cc)'),
           ('shadow_cc', 'Shadow gate (cc), proposed'))


def pseudo_rgb(x, step=4):
    b = x.shape[2]
    idx = [int(round(f * (b - 1))) for f in (0.75, 0.45, 0.15)]
    img = np.stack([np.asarray(x[:, ::step, i], dtype=np.float64) for i in idx], axis=-1)
    lo, hi = np.percentile(img, 2, axis=(0, 1)), np.percentile(img, 98, axis=(0, 1))
    return np.clip((img - lo) / np.maximum(hi - lo, 1e-12), 0, 1)


def compute_maps(x, seed, cfg, erx_mod, warmup, dims, mom):
    h, w, b = x.shape
    proj, mu0, cov0 = prepare_warmup(x, seed, warmup, dims, mom)
    pols = {'delayed_trim10_cc': GatedPolicy(mu0, cov0, mom, fixed_chooser('trim10'), True, ('trim10',)),
            'shadow_cc': GatedPolicy(mu0, cov0, mom, shadow_chooser(0.8, 0.05), True)}
    maps = {k: np.zeros((h, w), dtype=np.float32) for k in pols}
    for t in range(warmup, w):
        z = np.asarray(proj.transform(np.asarray(x[:, t, :], dtype=np.float64)), dtype=np.float64)
        for k, p in pols.items():
            maps[k][:, t] = normalise_line(p.step(z))
    np.random.seed(seed)
    det = build_official('erx', cfg, h, b, erx_mod)
    det.random_projector = proj
    m, _, _ = simulate(det, x)
    maps['official_erx'] = m.astype(np.float32)
    return maps


def render(rgb, mask, maps, step, warmup, title, dest, lines=None, clip=99.5):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    h, w = mask.shape
    a, b = (warmup, w) if lines is None else lines
    sub = lambda M: M[:, a:b]
    rows = [('Pseudo-RGB', rgb[:, a // step:max(a // step + 1, b // step)], None),
            ('Ground truth', sub(mask).astype(float), 'gray')]
    # common display scale: clip each map at its own high percentile (scores are line z-scores)
    for k, name in METHODS:
        M = sub(maps[k]).astype(np.float64)
        vmax = np.percentile(M, clip)
        rows.append((name, np.clip(M, 0, vmax) / max(vmax, 1e-12), 'inferno'))
    aspect = (b - a) / h
    fig_w = 10.0
    row_h = max(0.55, min(2.2, fig_w / aspect))
    fig, axes = plt.subplots(len(rows), 1, figsize=(fig_w, row_h * len(rows) + 0.6))
    for ax, (name, img, cmap) in zip(axes, rows):
        ax.imshow(img, cmap=cmap, aspect='auto', interpolation='nearest',
                  extent=(a, b, h, 0), vmin=0, vmax=1)
        ax.set_ylabel(name, rotation=0, ha='right', va='center', fontsize=8)
        ax.set_yticks([]); ax.tick_params(labelsize=7)
        for s in ax.spines.values():
            s.set_linewidth(0.5)
    for ax in axes[:-1]:
        ax.set_xticks([])
    axes[-1].set_xlabel('Scan line', fontsize=8)
    axes[0].set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(dest, dpi=220); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hyperad-root', required=True)
    p.add_argument('--hsi', required=True)
    p.add_argument('--mask', required=True)
    p.add_argument('--dataset-name', default='Beach')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--lines', type=int, nargs=2, default=None, help='Optional crop START END (scan lines)')
    p.add_argument('--output', default='results_v17_maps')
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
    mask = np.asarray(mask != 0)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        maps = compute_maps(x, a.seed, cfg, erx_mod, warmup, dims, mom)
    step = 4
    rgb = pseudo_rgb(x, step)
    np.savez_compressed(out / f'maps_{a.dataset_name}_seed{a.seed}.npz',
                        mask=mask, rgb=rgb.astype(np.float16), step=step, warmup=warmup,
                        **{k: v.astype(np.float16) for k, v in maps.items()})
    render(rgb, mask, maps, step, warmup, f'{a.dataset_name}, projection seed {a.seed}',
           out / f'detection_maps_{a.dataset_name}.png')
    if a.lines:
        render(rgb, mask, maps, step, warmup, f'{a.dataset_name}, lines {a.lines[0]}-{a.lines[1]}',
               out / f'detection_maps_{a.dataset_name}_lines{a.lines[0]}_{a.lines[1]}.png', tuple(a.lines))
    print('Saved:', out.resolve())


if __name__ == '__main__':
    main()
