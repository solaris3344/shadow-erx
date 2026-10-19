# Result record: Synthetic scene (preregistered primary analysis)

*English translation of `RESULTS_PREREG_TR.md`.*

**Date:** 18 October 2026
**Preregistration:** `PREREG_TR.md`, SHA-256 `668169E7906C8137D686033EF6DD4F9049DF81FE35AB13F0A615D522252BFCE2`. The hash was taken before the analysis.
**Deviations:** None. No parameter was changed. The first download attempt produced a corrupt file; the file was downloaded again (600 × 2400 × 90, float64), and no analysis was completed with the corrupt file.
**Outputs:** `results_synthetic_v13`, `results_synthetic_v12`

## 1. Primary results (v13, `common_scored`, line_z, mean of 3 seeds)

| Method | AUROC | AP | FPR90 | ms/line |
|---|---|---|---|---|
| shadow_cc | **0.8609** | **0.0479** | **0.3409** | 0.654 |
| shadow_nc | 0.8520 | 0.0460 | 0.3727 | 0.455 |
| delayed_trim10_cc | 0.8183 | 0.0409 | 0.4840 | 0.266 |
| delayed_trim05_cc | 0.7958 | 0.0366 | 0.5459 | 0.283 |
| official_erx | 0.7861 | 0.0366 | 0.5486 | 0.291 |
| official_rt_ck_rxd | 0.5821 | 0.0361 | 0.8806 | 10.089 |
| official_lbl_ad | 0.5731 | 0.0155 | 0.9968 | 0.605 |
| official_rx_bil | 0.5672 | 0.0397 | 0.8894 | 3.343 |
| frozen | 0.5591 | 0.0259 | 0.9987 | 0.077 |
| official_rx_baseline | 0.5327 | 0.0420 | 0.9140 | 36.784 (49-line latency) |

ms/line: single thread. For our methods the shared projection time is excluded, for official ERX it is included. End-to-end timings with the projection included are produced by `run_v16.py` (Table VIII of the paper).

## 2. Hypothesis decisions

| Hypothesis | Rule | Observation | Decision |
|---|---|---|---|
| H1: improvement over ERX | improvement in the same direction on 3 metrics | AUROC +0.075, AP +0.011, FPR90 −0.208 | **Supported** |
| H2: regret relative to fixed trimming | Δ AP > +0.02 gain; ±0.02 no regret | Δ = +0.0070 | **No regret** |
| H3: timing (conditional) | if S ≥ 0.02, ≥ 12/15 permutations | S = 0.0042 (v12 line_z) | **Uninformative** |
| H4: position (descriptive) | ranks reported | 1st of 10 methods in AUROC, AP and FPR90 | Descriptive |

## 3. Descriptive and exploratory findings (not confirmatory)

**3.1 Permutation control (v12).** The shadow gate beat 15/15 shuffled copies with the same action counts on AP, in both modes.

| Mode | shadow_cc | shadow_cc_permgate |
|---|---|---|
| line_z | AUROC 0.860, AP 0.047, FPR90 0.343 | AUROC 0.758, AP 0.041, FPR90 0.989 |
| raw | AUROC 0.917, AP 0.263, FPR90 0.201 | AUROC 0.888, AP 0.114, FPR90 0.201 |

In line_z mode the permuted gate collapses to the frozen level in FPR90 (0.999) while the shadow gate stays at 0.343. This shows that, at a fixed update rate, the timing of the accepted updates is decisive. Because H3 was tied to the condition on S in the preregistration, this finding is reported as **exploratory**.

**3.2 Lesson for preregistration design.** The moderator S in H3 (spread between Trim05 and Trim10) did not anticipate a scene with a high skip rate. On Synthetic the gate skips the update on 77 to 80% of the lines; the real contrast is not between Trim05 and Trim10 but between updating and skipping. S did not capture the value of timing on this scene. A better moderator for future scenes may be the difference between the frozen model and the best fixed trimming rule. This suggestion was made after the results were seen and may only be used in a new preregistration.

**3.3 Limitation of the H2 threshold.** The absolute AP is very low on this scene (0.03 to 0.05). The absolute threshold of 0.02 corresponds to about 50% in relative terms. The observed relative difference is +17%. There are also differences of +0.043 in AUROC and −0.143 in FPR90. These are descriptive; the H2 decision does not change.

**3.4 Consistency correction.** `shadow_cc` beat `shadow_nc` on all three metrics (v13 and v12). The direction is the same as on Beach and supports choosing `cc` as the main method.

**3.5 Frozen model.** The frozen model collapses on Synthetic (line_z AUROC 0.559). Updating the background is necessary on this scene, yet the gate accepts an update on only about 20% of the lines.

## 4. Reproduction check

The `official_full` AUROC values of the official detectors agree with the Synthetic table of the ERX paper: RX Baseline 0.531 (paper 0.531), RT-CK-RXD 0.581 (0.581), RX-BIL 0.566 (0.566 ± 0.002), LBL-AD 0.575 (0.576), ERX 0.787 (0.751 ± 0.032). On Beach and SNP the four deterministic detectors had also matched within ±0.001.

## 5. Summary of the three scenes (line_z, `common_scored`)

| Scene | Status | shadow_cc vs ERX | shadow_cc vs best fixed trimming (AP) | Permutation (AP) | Among 10 methods |
|---|---|---|---|---|---|
| Beach | development | 3/3 improved | +0.100 | 15/15 | 1st on 3 metrics |
| SNP | development | 3/3 improved | −0.004 | 10/15 | middle ranks |
| Synthetic | **preregistered** | 3/3 improved (H1) | +0.007 (H2 no regret) | 15/15 (exploratory) | 1st on 3 metrics |

## 6. Secondary analysis: reversed Synthetic scan (`--reverse`)

The reversed scan was defined as secondary evidence in the preregistration. The same H1 to H3 rules were applied.
Outputs: `results_synthetic_rev_v13`, `results_synthetic_rev_v12`.

**v13, `common_scored`, line_z:**

| Method | AUROC | AP | FPR90 |
|---|---|---|---|
| shadow_cc | **0.8987** | **0.0539** | **0.2367** |
| shadow_nc | 0.8883 | 0.0514 | 0.2627 |
| delayed_trim10_cc | 0.8564 | 0.0478 | 0.3750 |
| delayed_trim05_cc | 0.8344 | 0.0421 | 0.4328 |
| official_erx | 0.8074 | 0.0396 | 0.4903 |
| official_rt_ck_rxd | 0.6328 | 0.0401 | 0.7989 |
| official_rx_bil | 0.6171 | 0.0400 | 0.8229 |
| official_lbl_ad | 0.5503 | 0.0154 | 0.9966 |
| official_rx_baseline | 0.5328 | 0.0417 | 0.9145 |
| frozen | 0.1098 | 0.0174 | 0.9792 |

| Hypothesis | Observation | Decision |
|---|---|---|
| H1 | AUROC +0.091, AP +0.014, FPR90 −0.254 | **Supported** |
| H2 | Δ = 0.0539 − 0.0478 = +0.0061 | **No regret** |
| H3 | S = \|0.0412 − 0.0467\| = 0.0055 (v12 line_z) | **Uninformative** |
| H4 | 1st in AUROC, AP and FPR90 | Descriptive |

**Permutation control (v12, descriptive):**

| Mode | shadow_cc | shadow_cc_permgate | Beaten on AP |
|---|---|---|---|
| line_z | AUROC 0.899, AP 0.0526, FPR90 0.237 | AUROC 0.864, AP 0.0509, FPR90 0.322 | 11/15 |
| raw | AUROC 0.937, AP 0.249, FPR90 0.136 | AUROC 0.908, AP 0.101, FPR90 0.176 | 15/15 |

In the reversed scan the timing effect on AP in line_z mode is weak and mixed (11/15, ΔAP +0.002). AUROC (+0.035) and FPR90 (−0.085) favor the gate. In raw mode the effect is strong (15/15, ΔAP +0.148). Compared with the forward line_z result (FPR90 0.343 against 0.989), the size of the timing effect depends on the scan direction. This supports keeping the timing finding exploratory.

**Other observations:**

- Frozen line_z AUROC 0.110: the background drift is much stronger in the reversed direction.
- The skip rate of the gate is 78 to 80%, the same level as in the forward scan.
- The official detectors reproduce the "Flipped" column of the ERX paper: RT-CK-RXD 0.631 (paper 0.632), RX-BIL 0.615 (0.615), LBL-AD 0.552 (0.550).
- Because the centered window of RX Baseline is symmetric, our forward and reversed results are identical (0.531). The paper's value of 0.527 is consistent with the official `Sampler` taking the first line from line 0 in reverse mode, which shifts it by one line relative to the mask.
- ERX gives 0.810 here and 0.760 ± 0.022 in the paper. The difference probably comes from the projection seed.

## 7. Overall assessment (Synthetic, forward and reversed)

- **H1** is supported in both directions: the gate consistently beats official ERX in AUROC, AP and FPR90.
- **H2** is "no regret" in both directions: the AP difference to the best fixed trimming rule is +0.006/+0.007, below the gain threshold.
- **H3** is uninformative in both directions: the moderator condition of the preregistration was not met.
- **H4:** the gate ranks first of 10 methods on all three metrics in both directions.
- The timing effect (permutation) is strong in raw mode in both directions (15/15). In line_z mode it is strong forward and weak reversed. It should be reported as exploratory.
