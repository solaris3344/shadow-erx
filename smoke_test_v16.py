"""V16 checks: rival rules, sensitivity grid, timing, CLI end-to-end with stub HyperAD."""
import subprocess, sys, csv
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from run_shadow import prepare_warmup
from run_v12 import GatedPolicy, shadow_chooser
from run_v16 import RankSumGate, DelayedHuber, huber_proposal, RHOS, LAMBDAS
import smoke_test_v13 as st


def main():
    rng = np.random.default_rng(7)
    pixels, lines, bands = 40, 150, 9
    x = rng.normal(size=(pixels, lines, bands)).astype(np.float32)
    mask = np.zeros((pixels, lines), dtype=np.int8)
    for a_, b_ in [(30, 37), (60, 70), (95, 104)]:
        x[5:14, a_:b_ + 1, :4] += 3.; mask[5:14, a_:b_ + 1] = 1
    x[:, :, 5] += np.arange(lines)[None] * .02
    proj, mu, cov = prepare_warmup(x, 0, 15, 5, .1)
    Z = [proj.transform(x[:, t].astype('float64')) for t in range(15, lines)]
    # Huber with all-equal weights reduces to the plain momentum update
    z = Z[0]; m1, c1 = huber_proposal(z, mu, cov, .1, q=1.0)
    m2 = .9 * mu + .1 * z.mean(0); c2 = .9 * cov + .1 * (np.cov(z, rowvar=False) + 1e-5 * np.eye(5))
    assert np.allclose(m1, m2) and np.allclose(c1, c2)
    print('PASS: Huber update reduces to plain update when no pixel is down-weighted')
    g = RankSumGate(mu, cov, .1)
    for zz in Z:
        g.step(zz)
    assert len(g.actions) == len(Z) - 1 and set(g.actions) <= {'skip', 'trim10'}
    a_hi = RankSumGate(mu, cov, .1, alpha=1.0)  # p >= 1 never: always skip after first accept
    for zz in Z:
        a_hi.step(zz)
    assert a_hi.actions[0] == 'trim10' and set(a_hi.actions[1:]) <= {'skip'}
    print('PASS: rank-sum gate actions and alpha extremes')
    here = Path(__file__).resolve().parent
    with TemporaryDirectory() as d:
        d = Path(d); det = d / 'repo' / 'detectors'; det.mkdir(parents=True)
        (d / 'repo' / 'test_detection').mkdir()
        (d / 'repo' / 'test_detection' / 'config_detection.yaml').write_text(st.CONFIG)
        (d / 'repo' / 'utils_stub.py').write_text(st.UTILS)
        (d / 'repo' / 'baseline_v3.py').write_text((here / 'baseline_v3.py').read_text())
        (det / '__init__.py').write_text('')
        (det / 'erx.py').write_text(st.ERX)
        np.save(d / 'hsi.npy', x); np.save(d / 'mask.npy', mask[..., None])
        res = subprocess.run([sys.executable, str(here / 'run_v16.py'), '--hyperad-root', str(d / 'repo'),
                              '--hsi', str(d / 'hsi.npy'), '--mask', str(d / 'mask.npy'), '--seeds', '0', '1',
                              '--tail', '7', '--output', str(d / 'out')], capture_output=True, text=True, cwd=here)
        assert res.returncode == 0, res.stderr[-3000:]
        o = d / 'out'
        for f in ('rival_summary.csv', 'sensitivity_summary.csv', 'sensitivity_heatmap.png',
                  'timing_end_to_end.csv', 'v16_report.txt', 'run_config.json'):
            assert (o / f).exists(), f
        rs = list(csv.DictReader((o / 'rival_summary.csv').open()))
        assert {r['method'] for r in rs} >= {'official_erx', 'shadow_cc', 'ranksum_gate', 'delayed_huber_q90'}
        sg = list(csv.DictReader((o / 'sensitivity_summary.csv').open()))
        assert len(sg) == len(RHOS) * len(LAMBDAS)
        # default grid point equals shadow_cc row
        dflt = next(r for r in sg if float(r['rho']) == .8 and float(r['lambda']) == .05)
        sh = next(r for r in rs if r['method'] == 'shadow_cc')
        assert abs(float(dflt['ap']) - float(sh['ap'])) < 1e-12
        tm = list(csv.DictReader((o / 'timing_end_to_end.csv').open()))
        assert len(tm) == 6 and all(float(r['ms_mean']) > 0 for r in tm)
    print('PASS: CLI end-to-end; grid default equals shadow_cc; timing for 6 methods')


if __name__ == '__main__':
    main()
