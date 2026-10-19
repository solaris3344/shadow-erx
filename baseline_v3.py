#!/usr/bin/env python3
"""Hold-out validation of HyperAD/ERX update policies on a new data cube.

SNP was explored before designing this script. Trim05 is the frozen primary
candidate. This script never uses test labels for selection or updating.
Reference ERX is imported unmodified from the separately supplied HyperAD repo.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.random_projection import SparseRandomProjection
from threadpoolctl import threadpool_limits


def reference_repo(path: str) -> Path:
    root = Path(path).expanduser().resolve()
    if not (root / "detectors" / "erx.py").is_file():
        raise FileNotFoundError(f"Cannot find detectors/erx.py inside: {root}")
    return root


def file_info(path: str) -> dict:
    p = Path(path).expanduser().resolve()
    return {"path": str(p), "size_bytes": p.stat().st_size}


def shared_projector(n_pixels: int, n_bands: int, n_proj: int, seed: int):
    p = SparseRandomProjection(n_components=n_proj, random_state=seed)
    p.fit(np.ones((n_pixels, n_bands), dtype=np.float64))
    return p


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


def normalise_line(a: np.ndarray) -> np.ndarray:
    sd = float(a.std())
    return (a - float(a.mean())) / sd if sd > 1e-12 else np.zeros_like(a)


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


def evaluate(detector, x, flat_y: np.ndarray, warmup: int) -> dict:
    n_pixels, n_lines, _ = x.shape
    n_eval_lines = n_lines - warmup
    scores = np.empty((n_eval_lines, n_pixels), dtype=np.float64)
    start = time.perf_counter()
    for col in range(n_lines):
        # Only a single line is converted to float64; avoids copying 600MB cube.
        line = np.asarray(x[:, col, :], dtype=np.float64)
        response = detector.forward(line)
        if col >= warmup:
            if response is None or len(response) != n_pixels:
                raise RuntimeError(f"Detector returned no scores at line {col}")
            scores[col - warmup] = np.asarray(response, dtype=np.float64)
    seconds = time.perf_counter() - start
    flat_score = scores.T.ravel()  # same flatten order as mask[:,warmup:].ravel()
    if not np.isfinite(flat_score).all():
        raise ValueError("Nonfinite detection scores; review numerical condition")
    fpr, tpr, _ = roc_curve(flat_y, flat_score)
    ix = np.searchsorted(tpr, 0.90, side="left")
    return {
        "auroc": float(roc_auc_score(flat_y, flat_score)),
        "ap": float(average_precision_score(flat_y, flat_score)),
        "fpr_at_90_tpr": float(fpr[min(ix, len(fpr) - 1)]),
        "ms_per_line": float(1000.0 * seconds / n_lines),
    }


def run(args):
    if args.warmup < 2 or not (0 < args.projected_dims):
        raise ValueError("warmup>=2 and projected-dims>0 required")
    if not (0. < args.momentum <= 1.):
        raise ValueError("momentum must lie in (0,1]")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be distinct")
    root = reference_repo(args.hyperad_root)
    sys.path.insert(0, str(root))
    from detectors.erx import ERX  # authors' original unmodified source

    x = np.load(args.hsi, mmap_mode="r", allow_pickle=False)
    mask = np.load(args.mask, mmap_mode="r", allow_pickle=False)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]  # SNP provides H,W,1; Beach may provide H,W
    if x.ndim != 3 or mask.shape != x.shape[:2]:
        raise ValueError(f"Image shape {x.shape}, mask shape {mask.shape}; expected H,W,B and H,W")
    n_pixels, n_lines, n_bands = x.shape
    if args.projected_dims > n_bands:
        raise ValueError(f"projected-dims={args.projected_dims} > bands={n_bands}")
    if args.warmup >= n_lines:
        raise ValueError("warmup must be smaller than number of lines")
    y = np.asarray(mask[:, args.warmup:] != 0, dtype=np.uint8).ravel()
    positives = int(y.sum())
    if positives == 0 or positives == y.size:
        raise ValueError("Evaluated mask needs positive and negative pixels")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (root / 'detectors' / 'erx.py').open('rb') as f:
        code_hash = hashlib.sha256(f.read()).hexdigest()
    conf = {
        "dataset_name": args.dataset_name, "hsi": file_info(args.hsi),
        "mask": file_info(args.mask), "hsi_shape": list(x.shape), "mask_shape": list(mask.shape),
        "warmup": args.warmup, "projected_dims": args.projected_dims,
        "momentum": args.momentum, "seeds": args.seeds,
        "normalise_md": args.normalise_md, "reference_repo": str(root),
        "reference_erx_sha256": code_hash, "n_eval_pixels": int(y.size),
        "n_positive": positives, "prevalence": float(y.mean()),
        "note": "Beach evaluation must not be used to tune the preselected Trim05; projection seeds are not independent scenes.",
    }
    (out / 'run_config.json').write_text(json.dumps(conf, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"Dataset: {args.dataset_name} | image={x.shape}, mask={mask.shape} "
          f"| evaluation positives={positives} / {y.size} ({y.mean():.6f})", flush=True)
    print(f"Normalise MD per line: {args.normalise_md} | reference SHA-256: {code_hash[:16]}", flush=True)
    methods = ["ERX_original", "ERX_prequential", "ERX_trim05", "ERX_trim10", "ERX_soft_huber_q90"]
    raw_rows = []
    with threadpool_limits(limits=1):
        for seed in args.seeds:
            p = shared_projector(n_pixels, n_bands, args.projected_dims, seed)
            src = ERX(n_bands=n_bands, n_pixels=n_pixels, buffer_len=args.warmup,
                      n_projdims=args.projected_dims, momentum=args.momentum,
                      normalise_md=args.normalise_md)
            src.random_projector = p
            models = [
                (methods[0], src),
                (methods[1], PrequentialERX(p, args.warmup, args.momentum,
                                           normalise_md=args.normalise_md)),
                (methods[2], PrequentialERX(p, args.warmup, args.momentum,
                                           trim_fraction=0.05, normalise_md=args.normalise_md)),
                (methods[3], PrequentialERX(p, args.warmup, args.momentum,
                                           trim_fraction=0.10, normalise_md=args.normalise_md)),
                (methods[4], PrequentialERX(p, args.warmup, args.momentum,
                                           huber_q=0.90, normalise_md=args.normalise_md)),
            ]
            for name, detector in models:
                result = evaluate(detector, x, y, args.warmup)
                row = {"seed": seed, "model": name, **result}
                raw_rows.append(row)
                print(f"seed={seed:2d} {name:22s} "
                      f"AUROC={result['auroc']:.5f} AP={result['ap']:.5f} "
                      f"FPR90={result['fpr_at_90_tpr']:.5f} "
                      f"time={result['ms_per_line']:.3f} ms/line", flush=True)
            # Save each seed so a partial run is not lost.
            with (out / 'beach_results.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
                w.writeheader(); w.writerows(raw_rows)

    columns = ['auroc', 'ap', 'fpr_at_90_tpr', 'ms_per_line']
    summary = []
    for name in methods:
        rs = [r for r in raw_rows if r['model'] == name]
        s = {'model': name, 'n_projection_seeds': len(rs)}
        for col in columns:
            vals = np.array([r[col] for r in rs], dtype=float)
            s[col + '_mean'] = float(vals.mean())
            s[col + '_sd'] = float(vals.std(ddof=1)) if len(vals)>1 else 0.
        summary.append(s)
    with (out / 'beach_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader(); w.writerows(summary)
    reference = {r['seed']: r for r in raw_rows if r['model'] == 'ERX_original'}
    deltas = []
    for r in raw_rows:
        if r['model'] == 'ERX_original':
            continue
        base = reference[r['seed']]
        deltas.append({"seed": r['seed'], "model": r['model'], **{
            'delta_' + c: r[c] - base[c] for c in columns}})
    with (out / 'paired_seed_deltas_vs_ERX.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(deltas[0])); w.writeheader(); w.writerows(deltas)

    print('\nMEAN +/- SD OVER PROJECTION SEEDS (same scene; NOT independent samples):', flush=True)
    for r in summary:
        print(f"{r['model']:22s} AUROC={r['auroc_mean']:.4f}±{r['auroc_sd']:.4f} "
              f"AP={r['ap_mean']:.4f}±{r['ap_sd']:.4f} "
              f"FPR90={r['fpr_at_90_tpr_mean']:.4f}±{r['fpr_at_90_tpr_sd']:.4f} "
              f"ms/line={r['ms_per_line_mean']:.3f}")
    print(f'Output: {out.resolve()}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hyperad-root', required=True)
    parser.add_argument('--hsi', required=True)
    parser.add_argument('--mask', required=True)
    parser.add_argument('--output', default='results_beach')
    parser.add_argument('--dataset-name', default='Beach')
    parser.add_argument('--warmup', type=int, default=99)
    parser.add_argument('--projected-dims', type=int, default=5)
    parser.add_argument('--momentum', type=float, default=0.1)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--normalise-md', action='store_true',
                        help='Use per-line z-scored detection scores (original HyperAD detection config uses this).')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
