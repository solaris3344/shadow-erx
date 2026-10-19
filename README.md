# Shadow update gate for ERX

Code, preregistrations and result records for the paper

> S. Keser, "# Preregistered Evaluation of a Label-Free Update Gate for Real-Time Line-Scan Hyperspectral Anomaly Detection," submitted to *Journal of Real-Time Image Processing*, 2026.

ERX [1] keeps a momentum-updated Gaussian background model in a sparse random projection of each scan line. Every update can absorb anomalous pixels, and every skipped update lets the model fall behind the scene. The shadow gate prepares three candidate updates from line t−1 (skip, Trim05, Trim10, the trimmed covariances corrected for consistency), scores them on the unseen line t with a robust Gaussian cost, and commits the best one. Scores for a line are always produced by the model committed before that line was seen, and no label enters any decision.

The repository also contains the reproduction of the official HyperAD benchmark of five line-by-line detectors and a correction of its threshold-based AUC metrics (see [Metric note](#metric-note)).

## Contents

| File | Purpose | Paper |
|---|---|---|
| `baseline_v3.py` | Prequential ERX re-implementation, line normalization, projection, loader for the unmodified HyperAD repository | all |
| `run_shadow.py` | Candidate updates, robust predictive cost, warm-up | all |
| `run_v12.py` | Gate, consistency correction, raw and line-normalized scores, matched-rate permuted gate | Tables V, VII, Fig. 4 |
| `run_v13.py` | Official HyperAD detectors run unmodified with `config_detection.yaml`, common causal window, corrected threshold AUC | Tables I, II, XI |
| `run_v15.py` | Paired line-block bootstrap, per-line decision traces | Table III |
| `analyze_v15_traces.py` | Gate decisions grouped by the annotated anomaly fraction of line t−1 (post hoc) | Fig. 5 |
| `run_v16.py` | Rival rules (Delayed Huber, rank-sum gate), ρ × λ sensitivity grid, end-to-end timing | Tables VI, VIII, Fig. 6 |
| `run_v17_maps.py` | Score maps | Figs. 2, 3 |
| `convert_had.py` | Converts standard HAD `.mat` scenes to line-scan `.npy`, detects duplicates, applies the eligibility rule, writes `manifest.csv` | Stage 2 |
| `run_v18_suite.py` | Frozen method on the stage-2 scene suite, cross-scene Wilcoxon tests with Holm correction | Tables IX, X |
| `smoke_test_v*.py` | Self-contained tests with a stub HyperAD repository and random data | |
| `preregistration/` | Both preregistrations (hashed Turkish originals and English translations) and the result records | Section V |
| `development/` | Exploratory versions v10 and v11 used on the development scenes before the first preregistration | |

Version numbers in the file names follow the development history and are kept because the preregistrations refer to them.

## Installation

Python 3.10 or newer.

```
pip install -r requirements.txt
python smoke_test_v12.py
python smoke_test_v13.py
python smoke_test_v15.py
python smoke_test_v16.py
python smoke_test_v18.py
```

Every test prints only `PASS` lines. The tests need neither the data nor the HyperAD repository.

The official detectors are taken from the authors' HyperAD repository, which is not included here. Clone or download https://github.com/WiseGamgee/HyperAD and leave it unmodified. Each run writes the SHA-256 of every detector file it loaded to `run_config.json`.

## Data

Stage 1 uses the three scenes released with ERX [2]: Zenodo, doi:10.5281/zenodo.13370800 (CC BY 4.0). Download `beach_hsi.npy`, `beach_mask.npy`, `snp_msi.npy`, `snp_mask.npy`, `synthetic_hsi.npy` and `synthetic_mask.npy`.

Stage 2 uses the six `.mat` scenes in the `data` folder of the GT-HAD repository [3] (https://github.com/NanWangAC/GT-HAD): Cat Island, Gulfport, Los Angeles 1 and 2, Pavia and Texas Coast.

## Reproducing the results

Below, `HA` is the HyperAD folder and `DATA` the folder with the Zenodo files. Run each command once per scene, replacing the scene file names and `--dataset-name` (Beach, SNP, Synthetic). Add `--reverse` for the reversed scan direction and change `--output` accordingly. All runs use three projection seeds (0, 1, 2) and one CPU thread.

Official benchmark, reproduction and main comparison (Tables I, II, XI):

```
python run_v13.py --hyperad-root HA --hsi DATA/beach_hsi.npy --mask DATA/beach_mask.npy --dataset-name Beach --output results_beach_v13
```

Permutation control and ablations (Tables V, VII, Fig. 4):

```
python run_v12.py --hyperad-root HA --hsi DATA/beach_hsi.npy --mask DATA/beach_mask.npy --dataset-name Beach --output results_beach_v12
```

Bootstrap and decision traces (Table III, Fig. 5):

```
python run_v15.py --hyperad-root HA --hsi DATA/beach_hsi.npy --mask DATA/beach_mask.npy --dataset-name Beach --output results_beach_v15
python analyze_v15_traces.py results_beach_v15 results_snp_v15 results_synthetic_v15 results_synthetic_rev_v15
```

Rival rules, sensitivity and timing (Tables VI, VIII, Fig. 6):

```
python run_v16.py --hyperad-root HA --hsi DATA/beach_hsi.npy --mask DATA/beach_mask.npy --dataset-name Beach --output results_beach_v16
```

Score maps (Figs. 2, 3):

```
python run_v17_maps.py --hyperad-root HA --hsi DATA/beach_hsi.npy --mask DATA/beach_mask.npy --dataset-name Beach --output results_beach_v17_maps
```

Stage 2 (Tables IX, X):

```
python convert_had.py --input-dir GT-HAD/data --output-dir had_npy
python run_v18_suite.py --hyperad-root HA --data-dir had_npy --output results_v18_suite
python run_v18_suite.py --hyperad-root HA --data-dir had_npy --reverse --output results_v18_suite_rev
```

The `results/` folder holds the summary outputs (CSV, TXT, JSON and PNG files) of the runs reported in the paper, with one subfolder per run.

The large scenes take from minutes (Beach, Synthetic) to several hours (SNP with `run_v15.py`) on a laptop core. `run_v16.py --skip-sensitivity` and `--skip-timing` shorten a run.

## Preregistration

| Stage | Scenes | Protocol | SHA-256 |
|---|---|---|---|
| 1 | Synthetic (held out) | `preregistration/PREREG_TR.md` | `668169E7906C8137D686033EF6DD4F9049DF81FE35AB13F0A615D522252BFCE2` |
| 2 | six standard HAD scenes | `preregistration/PREREG_v2_TR.md` | `55E265EED1564C08A57DF9270967D7BE3E288B118F9C0E7C1B01C9A4C8BCAE0C` |
| 2 | scene list | `preregistration/manifest.csv` | `F6B7DC5FCE82D2F68E366D0F69BF4B355C3B561B90572B0F4CFB8AA4C86B351C` |

The Turkish files are the originals whose hashes were recorded before the corresponding scenes were analyzed. The `_EN.md` files are translations made for readers and are not hashed. Check a hash with `sha256sum` (Linux, macOS) or `Get-FileHash` (Windows PowerShell). Beach and SNP were used during development and are not validation scenes.

## Metric note

In the HyperAD repository, `calc_auc_dt` and `calc_auc_ft` reverse the threshold array but not the TPR and FPR arrays, so the areas are computed from misaligned arrays (and the functions return `nan` with scikit-learn 1.3 or newer). The published AUC_TD values are reproduced exactly with this behaviour. With min-max normalized scores, the correct values have a closed form: AUC(D,τ) is the mean normalized score of the anomalous pixels and AUC(F,τ) the mean normalized score of the background. `run_v13.py` reports the correct, the buggy and the legacy variant side by side.

## References

[1] S. Garske, B. Evans, C. Artlett, and K. C. Wong, "ERX: A fast real-time anomaly detection algorithm for hyperspectral line scanning," *IEEE Trans. Geosci. Remote Sens.*, vol. 63, Art. no. 5503617, 2025, doi:10.1109/TGRS.2025.3532225.

[2] S. Garske and Y. Mao, "Three annotated anomaly detection datasets for line-scan algorithms," Zenodo, 2024, doi:10.5281/zenodo.13370800.

[3] J. Lian, L. Wang, H. Sun, and H. Huang, "GT-HAD: Gated transformer for hyperspectral anomaly detection," *IEEE Trans. Neural Netw. Learn. Syst.*, 2024, doi:10.1109/TNNLS.2024.3355166.

## License

The code is released under the MIT License (see `LICENSE`). The datasets and the HyperAD code are not redistributed here and remain under their own licenses.
