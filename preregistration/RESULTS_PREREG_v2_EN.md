# Result record v2: standard HAD scenes (preregistered)

*English translation of `RESULTS_PREREG_v2_TR.md`.*

**Date:** 19 October 2026
**Preregistration:** `PREREG_v2_TR.md`, SHA-256 `55E265EED1564C08A57DF9270967D7BE3E288B118F9C0E7C1B01C9A4C8BCAE0C`
**Manifest:** `manifest.csv`, SHA-256 `F6B7DC5FCE82D2F68E366D0F69BF4B355C3B561B90572B0F4CFB8AA4C86B351C`
**Deviations:** None. Method, parameters and scene set are as preregistered. Before the analysis, the preregistration file in the working folder had been overwritten by an earlier draft; the correct version was restored and hashed before the analysis.
**Outputs:** `results_v18_suite` (forward, primary), `results_v18_suite_rev` (reversed, secondary)

## 1. AUROC per scene (shadow cc / official ERX, mean of 3 seeds)

| Scene | Forward: shadow | Forward: ERX | Difference | Reversed: shadow | Reversed: ERX | Difference |
|---|---|---|---|---|---|---|
| cat-island | 0.8803 | 0.8686 | +0.012 | 0.8564 | 0.8883 | −0.032 |
| gulfport | 0.9463 | 0.9268 | +0.020 | 0.9206 | 0.9125 | +0.008 |
| los-angeles-1 | 0.8252 | 0.8069 | +0.018 | 0.8025 | 0.8050 | −0.003 |
| los-angeles-2 | 0.9106 | 0.8997 | +0.011 | 0.8758 | 0.8732 | +0.003 |
| pavia | 0.8662 | 0.8715 | −0.005 | 0.8704 | 0.8581 | +0.012 |
| texas-goast | 0.9712 | 0.9614 | +0.010 | 0.9668 | 0.9553 | +0.012 |

## 2. Hypothesis decisions

| Hypothesis | Forward (primary) | Decision | Reversed (secondary) | Decision |
|---|---|---|---|---|
| **H1 AUROC (primary)** | 5/6 wins, median +0.011, p = 0.031 | **Supported** | 4/6, median +0.005, p = 0.34 | Not supported |
| H1 AP (secondary) | 2/6, median −0.011, Holm p = 0.66 | Not supported | 3/6, median −0.007, Holm p = 0.58 | Not supported |
| H1 FPR90 (secondary) | 5/6, p = 0.031, Holm p = 0.063 | Not significant after Holm | 5/6, p = 0.031, Holm p = 0.063 | Not significant after Holm |
| **H2 AUROC (primary)** | 3/6, median 0.000 | **No difference** | 3/6, median +0.000 | No difference |
| H2 AP | 1/6, median −0.004, p(regret) = 0.031 | **Regret** | 0/6, median −0.007, p(regret) = 0.016 | **Regret** |
| H2 FPR90 | 3/6 | No difference | 3/6 | No difference |
| H3 timing | 1/6 scenes | Weak | 0/6 scenes | None |
| H4 mean AUROC rank | shadow 3.67 (tied with Trim10, best); ERX 6.0 | Descriptive | shadow 4.0 (second after RT-CK-RXD at 3.5); ERX 5.67 | Descriptive |

## 3. Interpretation

1. **The primary hypothesis is supported in the primary analysis** (p = 0.031, gains on 5 of 6 scenes). **It is not replicated in the reversed scan** (p = 0.34). The largest loss in the reversed scan is on cat-island (−0.032), a scene with only 19 anomalous pixels in its evaluation window. The evidence should therefore be described as **weak and direction-dependent**. The AUROC improvement over ERX is small (median about +0.01).
2. **FPR90** improves over ERX on 5 of 6 scenes in both directions. The direction is consistent, but not significant after the preregistered Holm correction.
3. **On AP** the gate does not beat ERX and falls slightly but significantly behind the best fixed trimming rule chosen per scene, in both directions (median −0.004 / −0.007). Because this choice favors the fixed rule, the regret may be somewhat overstated; by the preregistered rule it is nevertheless reported as "regret".
4. **The gate's own contribution** (gain over the best fixed rule, timing effect) is not visible on these scenes. On short scenes with small anomalies the gain comes mainly from the trimmed update.
5. **Post hoc explanation (untested):** on these scenes the contamination per line is usually below the 10% trimming budget. The situation in which the gate helps, contamination above the trimming budget (the large panel on Beach), is rare here. Moreover, with 100 to 150 lines, a 20-line warm-up and 80 to 130 evaluation lines may not represent the long scans ERX was designed for.

## 4. Consequences for the paper

- The claim "a delayed trimmed update improves ERX in AUROC" is consistent across 9 scenes. It passes the preregistered primary test on an independent source but is not replicated in the reversed direction.
- The claim "the gate is better than fixed trimming" is supported only on Beach. On the other scenes there is no difference, and on the standard scenes there is a small cost in AP.
- In line with the preregistration commitment, these results are reported in full in the paper.
