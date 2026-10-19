# Preregistration v2: the frozen shadow gate on standard HAD scenes

*English translation of `PREREG_v2_TR.md`. Only the Turkish file was hashed (SHA-256 `55E265EED1564C08A57DF9270967D7BE3E288B118F9C0E7C1B01C9A4C8BCAE0C`); if the two differ, the Turkish file is authoritative.*

**Date:** 19 October 2026
**Status:** This note was written before any detector was run on the scenes below. It is not changed after the results are seen. The SHA-256 values of this file and of `manifest.csv` are stored in `PREREG_v2_SHA256.txt`.

## 1. History

- Beach and SNP are development scenes. Synthetic was tested under the first preregistration (PREREG_TR.md, SHA-256 668169E7…252BFCE2).
- The ρ/λ sensitivity analysis carried out after Synthetic showed that ρ ≤ 0.7 is better on some scenes. **The method is nevertheless used unchanged.**
- No method has been run on any of the scenes in this note so far. (On 19 October 2026 `run_v18_suite.py` was started once with an empty manifest and processed 0 scenes; no result was produced.)

## 2. Scenes

- **Source:** the 6 scenes in the GT-HAD repository (github.com/NanWangAC/GT-HAD, `data` folder). Because the ABU repository could not be reached, no further scenes are used. The sizes and band counts of these scenes match a subset of the ABU series, and the paper states this.
- **Scene set (fixed, hashed together with `manifest.csv`):**

| Scene | Pixels per line | Lines | Bands | Anomalous pixels in the evaluation window |
|---|---|---|---|---|
| cat-island | 150 | 150 | 188 | 19 |
| gulfport | 100 | 100 | 191 | 60 |
| los-angeles-1 | 100 | 100 | 205 | 127 |
| los-angeles-2 | 100 | 100 | 205 | 87 |
| pavia | 150 | 150 | 102 | 31 |
| texas-goast | 100 | 100 | 204 | 67 |

- **Eligibility rule (mechanical, `convert_had.py`):** at least 50 evaluation lines after the warm-up and at least one anomalous pixel in the evaluation window. All six scenes are eligible. The anomalous pixel counts were read only for this rule; no method was run.
- **Scan direction:** columns as stored in the file (axis 1). The reversed direction is a secondary analysis.
- **Power warning:** with 6 scenes, a one-sided Wilcoxon test reaches p < 0.05 only if the signed rank sum is at least 19 of 21. The gate must therefore win on all six scenes or lose only on one or two scenes with the smallest absolute differences. A rejection of H1 is therefore read as "could not be shown with 6 scenes", not as "no effect".

## 3. Frozen method and protocol

| Item | Value |
|---|---|
| Main method | `shadow_cc`, ρ = 0.8, λ = 0.05 (unchanged) |
| Warm-up / buffer length | **20 lines for all methods** (99 cannot be used because the scenes have 100 to 150 lines). For the official detectors only `buffer_len` changes; the RX Baseline latency becomes 10 lines accordingly. |
| Other ERX settings | `config_detection.yaml`: d = 5, momentum 0.1, `normalise_md = True` |
| References | official `erx`, `rx_baseline`, `lbl_ad`, `rt_ck_rxd`, `rx_bil`; `frozen`, `delayed_trim05_cc`, `delayed_trim10_cc`, `shadow_nc` |
| Evaluation window | `common_scored`: the columns scored by all methods |
| Projection seeds | 0, 1, 2; scene value = mean over seeds |
| Permutation control | 5 permuted gates per seed |
| Code | `run_v18_suite.py` (the rules in this note are implemented in the code) |

## 4. Hypotheses and decision rules (across scenes)

The unit is the **scene**. One value is used per scene (the mean over 3 seeds).

**H1 (primary). Improvement over ERX.**
- Test: one-sided Wilcoxon signed-rank test on the AUROC difference (gate − official ERX).
- Decision: supported if p < 0.05, otherwise not supported.
- The secondary metrics AP and FPR90 (lower is better for FPR90) are reported with the same test, with Holm correction.
- Note: with 6 scenes the smallest attainable p value is 1/64 = 0.0156 (power warning in Section 2).

**H2. Relative to the best fixed trimming rule.**
- For each scene and metric, the better of Delayed Trim05 and Delayed Trim10 is selected. This choice favors the fixed rule.
- Decision (AUROC primary):
  - **gain** if the one-sided test in favor of the gate gives p < 0.05,
  - **regret** if the test in favor of the fixed rule gives p < 0.05,
  - **no difference** otherwise.

**H3 (descriptive). Timing.** The number of scenes on which the shadow gate beats at least 80% of the permuted gates on AP is reported.

**H4 (descriptive). Position.** The mean AUROC rank across scenes is reported.

## 5. Secondary analyses

- Reversed scan direction, with the same rules.
- AP, FPR90 and the corrected AUC(D,τ) and AUC(F,τ) are reported descriptively per scene.
- Published results of offline methods are given only in a separate "offline upper bound" column and are not ranked in the same table as the streaming methods.

## 6. Reporting commitment

- The outcome of H1 to H4 is reported in the paper whatever it is.
- All per-scene results are given in the supplementary material.
- If a code error appears, only the error is fixed, and the fix is recorded with date and reason in `DEVIATIONS_v2.md`.
