# Preregistration: testing the shadow gate on an unseen scene

*English translation of `PREREG_TR.md`. Only the Turkish file was hashed (SHA-256 `668169E7906C8137D686033EF6DD4F9049DF81FE35AB13F0A615D522252BFCE2`); if the two differ, the Turkish file is authoritative.*

**Date:** 18 October 2026
**Status:** This note was written before the Synthetic scene was downloaded and before any method was run on it. It is not changed after the results are seen. Its hash is stored in `PREREG_SHA256.txt`.

## 1. Development history (disclosure)

- Beach and SNP were used to design the method throughout v1 to v13. They are development scenes and do not count as validation.
- The Synthetic scene (Zenodo 13370799, AVIRIS, 600 pixels × 2400 lines × 90 bands) has not been used in any analysis up to this date.
- The pipeline reproduced the Beach and SNP AUROC values of the ERX paper (Garske et al., TGRS 2025) within ±0.001 for the five official detectors. For ERX the difference lies within the variance due to the projection seed.

## 2. Frozen method and settings

| Item | Value |
|---|---|
| Main method | `shadow_cc` (trimming with consistency correction) |
| Candidate updates | skip, Trim05, Trim10 (one-line delay) |
| core_fraction / penalty | 0.80 / 0.05 |
| ERX settings | `config_detection.yaml`: buffer 99, d = 5, momentum 0.1, `normalise_md=True` |
| Projection seeds | 0, 1, 2 |
| Permutation replicates (v12) | 5 |
| Main evaluation window | `common_scored` (v13) |
| Main normalization | line_z (official detection protocol) |
| References | official `erx`, `rx_baseline`, `lbl_ad`, `rt_ck_rxd`, `rx_bil` (as in the config); `frozen`, `delayed_trim05_cc`, `delayed_trim10_cc` |

No parameter will be changed in response to the Synthetic results. If a run stops because of a code error, only the error is fixed, and the fix is recorded with date and reason in `DEVIATIONS.md`.

## 3. Hypotheses and decision rules fixed in advance

All values are means over the three seeds. The forward Synthetic scan (no `--reverse`) is the primary analysis.

**H1. Improvement over ERX.** `shadow_cc` raises AUROC and AP and lowers FPR90 relative to `official_erx`.
- Supported: improvement in the same direction on all three metrics.
- Partial: improvement on one or two metrics.
- Rejected: no improvement on any metric.

**H2. Regret relative to fixed trimming.** Δ = AP(`shadow_cc`) − max(AP(`delayed_trim05_cc`), AP(`delayed_trim10_cc`)).
- Gain: Δ > +0.02
- No regret: −0.02 ≤ Δ ≤ +0.02
- Regret: Δ < −0.02

**H3. The timing effect is conditional.** Candidate spread S = |AP(`delayed_trim05_cc`) − AP(`delayed_trim10_cc`)|, from the v12 line_z output.
- If S ≥ 0.02, prediction: `shadow_cc` beats at least 12 of the 15 permutations on AP. Supported if it does, rejected otherwise.
- If S < 0.02, H3 is uninformative and no claim about timing is made.

**H4. Position relative to the reference methods (descriptive).** The rank of `shadow_cc` in AUROC, AP and FPR90 and its ms per line are reported. No threshold or superiority claim is set in advance.

## 4. Secondary analyses (not confirmatory)

- Reversed Synthetic scan (`--reverse`): the same H1 to H3 rules, as secondary evidence.
- Reversed Beach and SNP scans: robustness description only, since these are development scenes.
- `official_full` window and raw normalization: supplementary material.

## 5. Reporting commitment

- The outcome of H1 to H4 is reported in the paper whatever it is.
- A negative result is discussed as a limitation of the method and is not "rescued" by further tuning.
- Because of the HyperAD `calc_auc_dt` / `calc_auc_ft` error, AUC(D,τ), AUC(F,τ), AUC_TD and AUC_BS are given with the corrected definition, and the difference from the published values is shown in a separate section. The main comparison uses AUROC, AP and FPR90.
