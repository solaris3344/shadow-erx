"""V15 checks: fast weighted metrics, bootstrap weights, CLI end-to-end with stub HyperAD."""
import subprocess, sys, csv
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from run_v15 import Sorted, block_weights
import smoke_test_v13 as st


def main():
    rng = np.random.default_rng(5)
    nl, h = 120, 30
    y = (rng.random((h, nl)) < .05)
    s = rng.normal(size=(h, nl)) + 1.5 * y
    li = np.tile(np.arange(nl), h)
    srt = Sorted(s.ravel(), y.ravel(), li)
    a, ap, f = srt.metrics(np.ones(nl))
    fpr, tpr, _ = roc_curve(y.ravel(), s.ravel())
    k = min(int(np.searchsorted(tpr, .9)), len(fpr) - 1)
    assert abs(a - roc_auc_score(y.ravel(), s.ravel())) < 1e-9
    assert abs(ap - average_precision_score(y.ravel(), s.ravel())) < 1e-9
    assert abs(f - fpr[k]) < 1e-9
    w = rng.integers(0, 3, nl).astype(float)
    pw = np.repeat(w[None], h, 0).ravel()
    a2, ap2, _ = srt.metrics(w)
    assert abs(a2 - roc_auc_score(y.ravel(), s.ravel(), sample_weight=pw)) < 1e-9
    assert abs(ap2 - average_precision_score(y.ravel(), s.ravel(), sample_weight=pw)) < 1e-9
    print('PASS: fast weighted AUROC/AP/FPR90 equal sklearn (unweighted and weighted)')
    bw = block_weights(500, 50, rng)
    assert abs(bw.sum() - 500) < 1e-9 and (bw > 0).sum() < 500
    print('PASS: block bootstrap weights')

    pixels, lines, bands = 40, 140, 9
    x = rng.normal(size=(pixels, lines, bands)).astype(np.float32)
    mask = np.zeros((pixels, lines), dtype=np.int8)
    for a_, b_ in [(30, 37), (60, 70), (95, 104)]:
        x[5:14, a_:b_ + 1, :4] += 3.; mask[5:14, a_:b_ + 1] = 1
    x[:, :, 5] += np.arange(lines)[None] * .02
    here = Path(__file__).resolve().parent
    with TemporaryDirectory() as d:
        d = Path(d); det = d / 'repo' / 'detectors'; det.mkdir(parents=True)
        (d / 'repo' / 'test_detection').mkdir()
        (d / 'repo' / 'test_detection' / 'config_detection.yaml').write_text(st.CONFIG)
        (d / 'repo' / 'utils_stub.py').write_text(st.UTILS)
        (d / 'repo' / 'baseline_v3.py').write_text((here / 'baseline_v3.py').read_text())
        (det / '__init__.py').write_text('')
        (det / 'erx.py').write_text(st.ERX)
        np.save(d / 'hsi.npy', x); np.save(d / 'mask.npy', mask)
        for extra in ([], ['--reverse']):
            res = subprocess.run([sys.executable, str(here / 'run_v15.py'), '--hyperad-root', str(d / 'repo'),
                                  '--hsi', str(d / 'hsi.npy'), '--mask', str(d / 'mask.npy'), '--seeds', '0', '1',
                                  '--n-boot', '30', '--block', '10', '--tail', '7', '--output', str(d / 'out'), *extra],
                                 capture_output=True, text=True, cwd=here)
            assert res.returncode == 0, res.stderr[-3000:]
        rows = list(csv.DictReader((d / 'out' / 'bootstrap_ci.csv').open()))
        q = {r['quantity'] for r in rows}
        assert 'delta_auroc_shadow_minus_official_erx' in q and 'shadow_cc_ap' in q
        for f in ('per_line_trace_seed0.csv', 'mechanism_seed0.png', 'mechanism_seed1.png'):
            assert (d / 'out' / f).exists(), f
        r = next(r for r in rows if r['quantity'] == 'shadow_cc_auroc')
        assert float(r['ci_low']) <= float(r['estimate']) + 0.05
    print('PASS: CLI end-to-end (forward and reverse), CI table, traces and figures written')


if __name__ == '__main__':
    main()
