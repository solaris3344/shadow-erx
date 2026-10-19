"""V12 invariants: equivalence with v6 policies, consistency factor, matched-rate
permutation, label independence, both modes, end-to-end CLI with a stub ERX."""
import subprocess, sys
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from run_shadow import ShadowPolicy, DelayedPolicy, prepare_warmup
from run_v12 import (GatedPolicy, shadow_chooser, fixed_chooser, consistency_factor,
                     evaluate_all, pareto_flags, proposal)

STUB_ERX = '''
import numpy as np
from baseline_v3 import PrequentialERX
class ERX:
    def __init__(self, n_bands, n_pixels, buffer_len, n_projdims, momentum, normalise_md):
        self.args = (buffer_len, momentum, normalise_md); self.random_projector = None; self.det = None
    def forward(self, line):
        if self.det is None:
            b, m, nm = self.args
            self.det = PrequentialERX(self.random_projector, b, m, normalise_md=nm)
        return self.det.forward(line)
'''


def synthetic(seed=20261018):
    rng = np.random.default_rng(seed)
    pixels, lines, bands = 48, 120, 9
    x = rng.normal(size=(pixels, lines, bands)).astype(np.float32)
    y = np.zeros((pixels, lines), dtype=np.int8)
    for a, b in [(30, 38), (55, 64), (90, 96)]:
        x[5:16, a:b + 1, :4] += 3.0
        y[5:16, a:b + 1] = 1
    x[:, :, 4] += np.arange(lines)[None, :] * .02  # slow drift
    return x, y


def main():
    x, y = synthetic()
    warmup, dims, mom = 15, 5, .1
    proj, mu, cov = prepare_warmup(x, 0, warmup, dims, mom)
    z = [proj.transform(x[:, t].astype('float64')) for t in range(warmup, x.shape[1])]

    ref = ShadowPolicy(mu, cov, mom, .8, .05)
    new = GatedPolicy(mu, cov, mom, shadow_chooser(.8, .05), consistency=False)
    for t, line in enumerate(z):
        assert np.array_equal(ref.step(line, t), new.step(line))
    assert [r['selected'] for r in ref.decision_rows] == new.actions
    print('PASS: uncorrected Shadow gate is identical to run_shadow.ShadowPolicy')

    ref = DelayedPolicy(mu, cov, mom, .10)
    new = GatedPolicy(mu, cov, mom, fixed_chooser('trim10'), False, needed=('trim10',))
    for line in z:
        assert np.array_equal(ref.step(line), new.step(line))
    print('PASS: uncorrected fixed gate is identical to run_shadow.DelayedPolicy(.10)')

    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 5)); S = A @ A.T + np.eye(5)
    sample = rng.multivariate_normal(np.zeros(5), S, size=400_000)
    for frac in (.05, .10):
        _, c_nc = proposal(sample, np.zeros(5), S, frac, 1.0, False)
        _, c_cc = proposal(sample, np.zeros(5), S, frac, 1.0, True)
        r_nc = np.trace(c_nc) / np.trace(S); r_cc = np.trace(c_cc) / np.trace(S)
        assert r_nc < .93 and abs(r_cc - 1) < .02, (frac, r_nc, r_cc)
        print(f'PASS: trim{int(frac*100):02d} covariance scale nc={r_nc:.3f} cc={r_cc:.3f} '
              f'(factor={consistency_factor(5, 1 - frac):.4f})')

    rows = [{'ap_mean': .5, 'auroc_mean': .9, 'fpr90_mean': .1},
            {'ap_mean': .4, 'auroc_mean': .8, 'fpr90_mean': .2},
            {'ap_mean': .3, 'auroc_mean': .95, 'fpr90_mean': .1}]
    assert pareto_flags(rows) == [1, 0, 1]
    print('PASS: Pareto dominance')

    msgs = []
    p1, s1, g1, c1 = evaluate_all(x, y, [0, 1], warmup, dims, mom, .8, .05, 3, log=msgs.append)
    y_alt = np.zeros_like(y); y_alt[0, warmup + 1] = 1
    _, _, _, c2 = evaluate_all(x, y_alt, [0, 1], warmup, dims, mom, .8, .05, 3, log=msgs.append)
    assert c1 == c2
    print('PASS: Shadow decisions and counts unchanged when the GT mask changes')
    assert {r['mode'] for r in s1} == {'raw', 'line_z'}
    frozen_raw = next(r for r in p1 if r['mode'] == 'raw' and r['method'] == 'frozen' and r['seed'] == 0)
    frozen_z = next(r for r in p1 if r['mode'] == 'line_z' and r['method'] == 'frozen' and r['seed'] == 0)
    assert frozen_raw['ap'] != frozen_z['ap']
    assert all(r['n_permutations'] == 3 for r in g1)
    print('PASS: both normalisation modes, permuted-gate replicates and summaries produced')

    here = Path(__file__).resolve().parent
    with TemporaryDirectory() as d:
        d = Path(d)
        (d / 'repo' / 'detectors').mkdir(parents=True)
        (d / 'repo' / 'detectors' / 'erx.py').write_text(STUB_ERX)
        (d / 'repo' / 'baseline_v3.py').write_text((here / 'baseline_v3.py').read_text())
        np.save(d / 'hsi.npy', x); np.save(d / 'mask.npy', y[..., None])
        cmd = [sys.executable, str(here / 'run_v12.py'), '--hyperad-root', str(d / 'repo'),
               '--hsi', str(d / 'hsi.npy'), '--mask', str(d / 'mask.npy'), '--dataset-name', 'Synthetic',
               '--warmup', str(warmup), '--seeds', '0', '1', '--permutations', '2',
               '--output', str(d / 'out')]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=here)
        assert res.returncode == 0, res.stderr
        for f in ('results_per_seed.csv', 'summary_pareto.csv', 'gate_vs_permuted.csv',
                  'shadow_action_counts.csv', 'pareto_ap_auroc.png', 'v12_report_TR.txt', 'run_config.json'):
            assert (d / 'out' / f).exists(), f
        txt = (d / 'out' / 'summary_pareto.csv').read_text()
        assert 'ERX_original' in txt and 'shadow_cc_permgate' in txt
    print('PASS: CLI end-to-end with stub ERX (H,W,1 mask), all outputs written')


if __name__ == '__main__':
    main()
