"""V18 checks: .mat conversion (v5 and v7.3), duplicate detection, eligibility, suite end-to-end."""
import subprocess, sys, csv
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from scipy.io import savemat
import smoke_test_v13 as st


def scene(rng, h=40, w=90, b=12, shift=0.0):
    x = rng.normal(size=(h, w, b)).astype(np.float32)
    m = np.zeros((h, w), np.uint8)
    for c in (35, 55, 75):
        x[5:9, c:c + 4, :4] += 3.5; m[5:9, c:c + 4] = 1
    x[:, :, 5] += np.linspace(0, 2 + shift, w)[None]
    return x, m


def main():
    rng = np.random.default_rng(11)
    here = Path(__file__).resolve().parent
    with TemporaryDirectory() as d:
        d = Path(d); raw = d / 'raw'; raw.mkdir()
        s = [scene(rng, shift=k) for k in range(5)]
        savemat(raw / 'a-scene.mat', {'data': s[0][0], 'map': s[0][1]})
        savemat(raw / 'b-scene.mat', {'hsi': s[1][0], 'gt': s[1][1].astype(float)})
        savemat(raw / 'dup-of-a.mat', {'data': s[0][0] * 2.0, 'map': s[0][1]})
        xs, ms = s[2]; ms2 = ms.copy(); ms2[:, 20:] = 0
        savemat(raw / 'no-eval-pos.mat', {'data': xs, 'map': ms2})
        import h5py
        with h5py.File(raw / 'v73-scene.mat', 'w') as f:  # MATLAB stores transposed
            f['data'] = s[3][0].transpose(2, 1, 0); f['map'] = s[3][1].T
        res = subprocess.run([sys.executable, str(here / 'convert_had.py'), '--input-dir', str(raw),
                              '--output-dir', str(d / 'npy'), '--warmup', '20', '--min-eval-lines', '50'],
                             capture_output=True, text=True, cwd=here)
        assert res.returncode == 0, res.stderr[-2000:]
        man = {r['scene']: r for r in csv.DictReader((d / 'npy' / 'manifest.csv').open())}
        assert man['dup-of-a']['duplicate_of'] == 'a-scene' and man['dup-of-a']['eligible'] == '0'
        assert man['no-eval-pos']['eligible'] == '0'
        assert man['v73-scene']['eligible'] == '1' and man['v73-scene']['bands'] == '12'
        assert np.array_equal(np.load(d / 'npy' / 'v73-scene_mask.npy'), s[3][1])
        print('PASS: conversion (v5 + v7.3), duplicate detection and eligibility rule')
        rp = d / 'repo'; det = rp / 'detectors'; det.mkdir(parents=True); (rp / 'test_detection').mkdir()
        (rp / 'test_detection' / 'config_detection.yaml').write_text(st.CONFIG)
        (rp / 'utils_stub.py').write_text(st.UTILS)
        (rp / 'baseline_v3.py').write_text((here / 'baseline_v3.py').read_text())
        (det / '__init__.py').write_text(''); (det / 'erx.py').write_text(st.ERX)
        for name, body in st.STUBS.items():
            (det / f'{name}.py').write_text(st.STUB_COMMON + '\n' + body + '\n')
        for extra in ([], ['--reverse']):
            res = subprocess.run([sys.executable, str(here / 'run_v18_suite.py'), '--hyperad-root', str(rp),
                                  '--data-dir', str(d / 'npy'), '--warmup', '20', '--seeds', '0', '1',
                                  '--permutations', '2', '--output', str(d / 'out'), *extra],
                                 capture_output=True, text=True, cwd=here)
            assert res.returncode == 0, res.stderr[-3000:]
        rep = (d / 'out' / 'cross_scene_report.txt').read_text()
        assert 'H1 gate vs official ERX' in rep and 'H2' in rep and 'H4' in rep
        rows = list(csv.DictReader((d / 'out' / 'per_scene.csv').open()))
        assert len(rows) == 3
    print('PASS: suite end-to-end (forward and reverse), cross-scene report written')


if __name__ == '__main__':
    main()
