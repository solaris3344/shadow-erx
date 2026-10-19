#!/usr/bin/env python3
"""Convert standard HAD .mat scenes to the line-scan .npy format and write a manifest.

* Finds the image (largest 3-D numeric array) and the mask (2-D array with the
  same spatial size and at most a few distinct values) in each .mat file,
  for MATLAB v5 (scipy) and v7.3 (h5py) files.
* Saves <name>_hsi.npy (H, W, B, float32) and <name>_mask.npy (H, W, uint8).
  Scan lines are axis 1 (columns), as in the HyperAD benchmark.
* Detects duplicate scenes across sources (same mask and same image up to scale).
* Applies the preregistered eligibility rule mechanically: a scene is eligible if
  it has >= --min-eval-lines evaluation lines after the warm-up and at least one
  anomalous pixel in the evaluation window. No detector is run here.
"""
from __future__ import annotations
import argparse, csv, hashlib
from pathlib import Path
import numpy as np


def load_mat(path):
    try:
        from scipy.io import loadmat
        d = loadmat(path)
        return {k: np.asarray(v) for k, v in d.items() if not k.startswith('__')}
    except Exception:  # MATLAB v7.3 files are HDF5
        import h5py
        out = {}
        with h5py.File(path, 'r') as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    a = np.asarray(obj)
                    out[name] = a.transpose(tuple(range(a.ndim))[::-1])  # MATLAB column-major
            f.visititems(visit)
        return out


def pick(arrays):
    cube = [(k, a) for k, a in arrays.items() if a.ndim == 3 and np.issubdtype(a.dtype, np.number)]
    if not cube:
        raise ValueError('no 3-D numeric array found')
    k_img, img = max(cube, key=lambda t: t[1].size)
    # make bands the last axis if needed
    if img.shape[0] < min(img.shape[1:]) and img.shape[0] < 400 and img.shape[2] > img.shape[0]:
        img = np.moveaxis(img, 0, -1)
    h, w = img.shape[:2]
    masks = []
    for k, a in arrays.items():
        a2 = np.squeeze(a)
        if a2.ndim == 2 and a2.shape in ((h, w), (w, h)) and len(np.unique(a2)) <= 3:
            masks.append((k, a2 if a2.shape == (h, w) else a2.T))
    if not masks:
        raise ValueError(f'no mask with spatial size {(h, w)} found; keys: {list(arrays)}')
    k_m, m = masks[0]
    return k_img, img, k_m, (m != 0).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input-dir', required=True, nargs='+')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--min-eval-lines', type=int, default=50)
    a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted({f for d in a.input_dir for f in Path(d).rglob('*.mat')})
    rows, seen = [], []
    for f in files:
        name = f.stem.lower().replace(' ', '-')
        try:
            k_img, img, k_m, mask = pick(load_mat(f))
        except Exception as e:
            rows.append({'scene': name, 'source_file': str(f), 'status': f'ERROR: {e}'})
            print(f'{name:22s} ERROR: {e}'); continue
        img = np.asarray(img, dtype=np.float32)
        h, w, b = img.shape
        dup = None
        for other, om, oi in seen:
            if om.shape == mask.shape and np.array_equal(om, mask) and oi.shape == img.shape:
                c = np.corrcoef(oi.reshape(-1)[::97], img.reshape(-1)[::97])[0, 1]
                if c > 0.999:
                    dup = other; break
        eval_pos = int(mask[:, a.warmup:].sum())
        eligible = (w - a.warmup >= a.min_eval_lines) and eval_pos > 0 and dup is None
        row = {'scene': name, 'source_file': str(f), 'image_key': k_img, 'mask_key': k_m,
               'pixels_per_line': h, 'lines': w, 'bands': b, 'anomalous_pixels': int(mask.sum()),
               'anomalous_pixels_in_eval_window': eval_pos, 'anomaly_fraction': float(mask.mean()),
               'duplicate_of': dup or '', 'eligible': int(eligible), 'status': 'ok'}
        if dup is None:
            np.save(out / f'{name}_hsi.npy', img)
            np.save(out / f'{name}_mask.npy', mask)
            seen.append((name, mask, img))
            row['hsi_sha256'] = hashlib.sha256((out / f'{name}_hsi.npy').read_bytes()).hexdigest()
            row['mask_sha256'] = hashlib.sha256((out / f'{name}_mask.npy').read_bytes()).hexdigest()
        rows.append(row)
        print(f"{name:22s} {h:4d} px x {w:4d} lines x {b:3d} bands  anomalies {row['anomaly_fraction']:.4f}"
              f"  eval-window positives {eval_pos:5d}  {'DUPLICATE of ' + dup if dup else ('eligible' if eligible else 'NOT eligible')}")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (out / 'manifest.csv').open('w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    sha = hashlib.sha256((out / 'manifest.csv').read_bytes()).hexdigest()
    print(f"\n{sum(int(r.get('eligible', 0)) for r in rows)} eligible scenes; manifest SHA-256: {sha}")


if __name__ == '__main__':
    main()
