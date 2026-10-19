"""V13 checks with stub HyperAD detectors (real ones are not bundled)."""
import subprocess, sys
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from run_v13 import all_metrics, hyperad_auc

STUB_COMMON = '''
import numpy as np
from utils_stub import line_md
class _Base:
    def __init__(self, n_bands, n_pixels, buffer_len, normalise_md=False, **kw):
        self.buffer_len, self.normalise_md, self.buf = buffer_len, normalise_md, []
        for k, v in kw.items(): setattr(self, k, v)
    def reset(self): self.buf = []
    def forward(self, line):
        line = np.asarray(line, dtype=np.float64); self.buf.append(line)
        if len(self.buf) <= self.buffer_len: return None
        self.buf = self.buf[-self.buffer_len:]
        return line_md(line, np.concatenate(self.buf), self.normalise_md)
'''
UTILS = '''
import numpy as np
def line_md(line, ref, norm):
    mu = ref.mean(0); c = np.cov(ref, rowvar=False) + 1e-3*np.eye(ref.shape[1])
    d = line - mu; md = np.sqrt(np.einsum('ij,jk,ik->i', d, np.linalg.inv(c), d))
    return (md - md.mean())/md.std() if norm else md
'''
STUBS = {
    'rt_ck_rxd': 'class RT_CK_RXD(_Base): pass',
    'cdlss_ad': 'class CDLSS_AD(_Base): pass',
    'lbl_ad': 'class LblAD(_Base): pass',
    'rx_bil': 'class RX_BIL(_Base):\n    def forward(self, line):\n        np.random.rand(); return super().forward(line)',
    'rx_baseline': ('class RXBaseline(_Base):\n    def forward(self, line):\n'
                    '        line = np.asarray(line, dtype=np.float64); self.buf.append(line)\n'
                    '        if len(self.buf) < self.buffer_len: return None\n'
                    '        self.buf = self.buf[-self.buffer_len:]\n'
                    '        return line_md(self.buf[-1 - self.line_offset], np.concatenate(self.buf), self.normalise_md)'),
}
ERX = '''
import numpy as np
from baseline_v3 import PrequentialERX
class ERX:
    def __init__(self, n_bands, n_pixels, buffer_len, n_projdims, momentum, normalise_md):
        self.args = (buffer_len, momentum, normalise_md); self.random_projector = None; self.det = None
    def reset(self): self.det = None
    def forward(self, line):
        if self.det is None:
            b, m, nm = self.args
            self.det = PrequentialERX(self.random_projector, b, m, normalise_md=nm)
        return self.det.forward(np.asarray(line, dtype=np.float64))
'''
CONFIG = '''
erx: {include: True, buffer_len: 15, projected_dimensions: 5, momentum: 0.1, line_offset: 0, normalise_md: True}
rx_baseline: {include: True, buffer_len: 15, normalise_md: True}
lbl_ad: {include: True, buffer_len: 15, pca_dims: 3, normalise_md: True}
rt_ck_rxd: {include: True, buffer_len: 15, normalise_md: True}
rx_bil: {include: True, buffer_len: 15, pixel_dropout: 0.5, normalise_md: True}
cdlss_ad: {include: False, buffer_len: 15, normalise_md: True}
'''


def main():
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(5000), np.ones(200)].astype(bool)
    s = np.r_[rng.normal(0, 1, 5000), rng.normal(2, 1, 200)]
    m = all_metrics(y, s)
    sn = (s - s.min()) / (s.max() - s.min())
    grid = np.linspace(0, 1, 20001)
    num_dt = _num = np.trapezoid([np.mean(sn[y] > t) for t in grid], grid) if hasattr(np, 'trapezoid') \
        else np.trapz([np.mean(sn[y] > t) for t in grid], grid)
    assert abs(m['auc_dt'] - num_dt) < 1e-3, (m['auc_dt'], num_dt)
    print(f"PASS: closed-form AUC(D,tau)={m['auc_dt']:.4f} matches numeric integral {num_dt:.4f}")
    assert np.isnan(m['auc_dt_hyperad_code'])
    assert abs(m['auc_dt_hyperad_legacy'] - m['auc_dt']) > .05
    print(f"PASS: HyperAD code gives nan with current sklearn; legacy={m['auc_dt_hyperad_legacy']:.4f} "
          f"vs correct={m['auc_dt']:.4f}")

    pixels, lines, bands = 40, 110, 9
    x = rng.normal(size=(pixels, lines, bands)).astype(np.float32)
    mask = np.zeros((pixels, lines), dtype=np.int8)
    for a, b in [(30, 37), (60, 70), (85, 92)]:
        x[5:14, a:b + 1, :4] += 3.; mask[5:14, a:b + 1] = 1
    here = Path(__file__).resolve().parent
    with TemporaryDirectory() as d:
        d = Path(d); det = d / 'repo' / 'detectors'; det.mkdir(parents=True)
        (d / 'repo' / 'test_detection').mkdir()
        (d / 'repo' / 'test_detection' / 'config_detection.yaml').write_text(CONFIG)
        (d / 'repo' / 'utils_stub.py').write_text(UTILS)
        (d / 'repo' / 'baseline_v3.py').write_text((here / 'baseline_v3.py').read_text())
        (det / '__init__.py').write_text('')
        (det / 'erx.py').write_text(ERX)
        for name, body in STUBS.items():
            (det / f'{name}.py').write_text(STUB_COMMON + '\n' + body + '\n')
        np.save(d / 'hsi.npy', x); np.save(d / 'mask.npy', mask[..., None])
        res = subprocess.run([sys.executable, str(here / 'run_v13.py'), '--hyperad-root', str(d / 'repo'),
                              '--hsi', str(d / 'hsi.npy'), '--mask', str(d / 'mask.npy'),
                              '--dataset-name', 'Synthetic', '--seeds', '0', '1', '--output', str(d / 'out')],
                             capture_output=True, text=True, cwd=here)
        assert res.returncode == 0, res.stderr[-3000:]
        txt = (d / 'out' / 'summary.csv').read_text()
        for name in ('official_erx', 'official_rx_baseline', 'official_lbl_ad', 'official_rt_ck_rxd',
                     'official_rx_bil', 'shadow_cc', 'frozen'):
            assert name in txt, name
        assert 'official_cdlss_ad' not in txt
        import csv
        rows = list(csv.DictReader((d / 'out' / 'summary.csv').open()))
        n_common = {r['n_columns'] for r in rows if r['window'] == 'common_scored'}
        assert n_common == {str(lines - 15 - 7)}, n_common  # warm-up and rx_baseline offset excluded
        erx = next(r for r in rows if r['window'] == 'official_full' and r['method'] == 'official_erx')
        assert float(erx['auroc_mean']) > .5
    print('PASS: CLI end-to-end; 5 included official detectors, cdlss_ad excluded by config')
    print('PASS: common window excludes warm-up and rx_baseline offset columns')


if __name__ == '__main__':
    main()
