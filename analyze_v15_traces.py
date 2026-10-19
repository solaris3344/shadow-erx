#!/usr/bin/env python3
"""POST-HOC, exploratory: Shadow action rates by the annotated anomaly fraction of
the line whose candidate update is being accepted or rejected (line t-1).

Reads per_line_trace_seed*.csv written by run_v15.py; no model is re-run.
The decision logged at line t chooses among candidates built from line t-1,
so the relevant label is the anomaly fraction of line t-1.
"""
import argparse, csv, glob
from pathlib import Path
import numpy as np

BINS = [(-1, 0, '0'), (0, .05, '(0, 5%]'), (.05, .10, '(5%, 10%]'), (.10, 1.01, '>10%')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folders', nargs='+')
    p.add_argument('--output', default='v15_mechanism_bins.csv')
    a = p.parse_args()
    rows = []
    for folder in a.folders:
        for f in sorted(glob.glob(str(Path(folder) / 'per_line_trace_seed*.csv'))):
            seed = Path(f).stem.split('seed')[-1]
            with open(f, encoding='utf-8') as fh:
                tr = list(csv.DictReader(fh))
            gt = np.array([float(r['gt_anomaly_fraction_POSTHOC']) for r in tr])
            act = np.array([r['shadow_action'] for r in tr])
            gt_prev, act_now = gt[:-1], act[1:]            # candidate line t-1, decision at t
            ok = act_now != ''
            gt_prev, act_now = gt_prev[ok], act_now[ok]
            for lo, hi, name in BINS:
                sel = (gt_prev > lo) & (gt_prev <= hi) if lo >= 0 else (gt_prev == 0)
                n = int(sel.sum())
                r = {'folder': Path(folder).name, 'seed': seed, 'gt_fraction_line_t_minus_1': name, 'n_decisions': n}
                for k in ('skip', 'trim05', 'trim10'):
                    r[f'rate_{k}'] = float(np.mean(act_now[sel] == k)) if n else float('nan')
                rows.append(r)
    with open(a.output, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # seed-averaged print
    print(f"{'folder':28s} {'GT(t-1) bin':12s} {'n':>6s} {'skip':>7s} {'trim05':>7s} {'trim10':>7s}")
    for folder in dict.fromkeys(r['folder'] for r in rows):
        for _, _, name in BINS:
            sub = [r for r in rows if r['folder'] == folder and r['gt_fraction_line_t_minus_1'] == name and r['n_decisions']]
            if not sub:
                continue
            n = sum(r['n_decisions'] for r in sub)
            m = {k: np.mean([r[f'rate_{k}'] for r in sub]) for k in ('skip', 'trim05', 'trim10')}
            print(f"{folder:28s} {name:12s} {n:6d} {m['skip']:7.3f} {m['trim05']:7.3f} {m['trim10']:7.3f}")
    print('Saved', a.output)


if __name__ == '__main__':
    main()
