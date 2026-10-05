# Multi-Dataset Model Card (Phase 5)

**Status: EXPERIMENTALLY VALIDATED for two tabular datasets.** Research prototype —
not a medical device; not for patient care.

All numbers below are read from persisted artefacts
(`outputs/phase5/model_scorecard.json`, `outputs/phase5/*/metrics/*.json`) —
nothing here is hand-typed from memory. Regenerate with
`python scripts/run_phase5.py analyze`.

## Intended use

Decision-support **research** on pediatric diagnostic reasoning: compare ML
backbones and measure confirmation-bias behaviour of an information-isolated
multi-agent engine. Not a diagnostic device; no clinical deployment.

## Datasets (never concatenated)

| Dataset | Status | Split | Train / Val / Test |
| --- | --- | --- | --- |
| REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR | **PASS** (blocker) | Phase 2 persisted split, reused verbatim (no re-splitting) | 545 / 118 / 117 |
| NEONATAL_SEPSIS_REGISTRY (DOI 10.17632/5vdz5cftz7.1) | **CONDITIONAL** — target verified as culture-positive vs culture-negative evaluation; post-outcome fields excluded | persisted stratified group split (patient IDs never cross partitions), seed 20261002 | 1337 / 309 / 300 |

Class mix (test): Regensburg 48 negative / 69 positive; sepsis 279 negative /
21 positive (severe imbalance — see limitations).

## Training recipe (identical for both)

- 17 tabular algorithms + 4 ensembles (`config/model_zoo.yaml`); scaling fitted on
  TRAIN only for LogReg/SVM/KNN/MLP/LDA/QDA; trees unscaled; CatBoost native cats.
- Cross-validation and all hyperparameter search inside TRAIN.
- Calibration (raw/Platt/isotonic) compared on VALIDATION, best Brier/ECE wins.
- Threshold: `argmax Macro F0.5 s.t. sensitivity ≥ 0.90` on VALIDATION; fallback
  = max sensitivity first. Locked before test (`locked_before_test: true`).
- Ensembles (uniform soft, validation-weighted soft, hard voting, stacking) fitted
  on VALIDATION only.
- Sealed test opened **once** (`test_evaluation_lock.json`).

## Winners (selected on validation only)

Selection rule: floor compliance → validation Macro F0.5 → AUPRC → calibration →
simpler model when indistinguishable. Winner = **gradient_boosting** on both datasets.

### Regensburg appendicitis (reference benchmark)

| Field | Value |
| --- | --- |
| Winner | gradient_boosting (isotonic calibration) |
| Locked threshold | 0.19 (validation; sensitivity floor 0.90, **pass**) |
| Macro F0.5 (test) | 0.911 (95% CI 0.854–0.958) |
| Positive-class F0.5 | 0.935 |
| Accuracy / Balanced acc. | 0.915 / 0.915 |
| Precision / Sensitivity / Specificity | 0.940 / 0.913 / 0.917 |
| AUROC (test) | 0.952 (95% CI 0.907–0.986) |
| AUPRC / Brier / ECE | 0.951 / 0.062 / 0.054 |
| MCC | 0.825 |
| Validation Macro F0.5 | 0.981 (gap to test: 0.070 — honest generalization gap) |

Note: the Phase 2 LightGBM headline result (AUROC 0.9716, threshold 0.34) is a
**separate, earlier experiment** on the same split and is never overwritten here.

### Neonatal sepsis registry (separate experiment)

| Field | Value |
| --- | --- |
| Winner | gradient_boosting (isotonic calibration) |
| Locked threshold | 0.04 (floor 0.90, **pass on validation**) |
| Macro F0.5 (test) | 0.485 (95% CI 0.448–0.525) |
| Positive-class F0.5 | 0.138 |
| Accuracy / Balanced acc. | 0.547 / 0.668 |
| Precision / Sensitivity / Specificity | 0.114 / 0.810 / 0.527 |
| AUROC (test) | 0.697 (95% CI 0.601–0.789) |
| AUPRC / Brier / ECE | 0.145 / 0.064 / 0.035 |
| MCC | 0.172 |
| Validation Macro F0.5 | 0.500 (gap: 0.015) |

This is a **hard, imbalanced task** (21 positives in 300 test cases). AUROC ≈ 0.70
with wide CIs; precision at the sensitivity-first operating point is low. The result
is reported exactly as measured — it is evidence that the pipeline runs honestly on
a second domain, **not** a claim of clinical utility.

## Ensemble reference points (test, locked thresholds)

- Regensburg validation-weighted soft voting: Macro F0.5 **0.929**, sensitivity 0.942
  (slightly above the best single model; paired Δ significance — see
  `outputs/phase5/statistical_comparison.json`).
- Sepsis validation-weighted soft voting: Macro F0.5 0.471, sensitivity 0.905.
- Hard voting: no probabilities → AUROC/AUPRC/Brier = **null (n/a), never 0**.

## Not evaluated as models

- QDA on Regensburg: **UNAVAILABLE** — class-0 covariance rank-deficient
  (`n=176 < 212` features). Not scored, not imputed.
- Image models (ResNet/DenseNet/EfficientNet/ConvNeXt on Kermany CXR, Regensburg US):
  **NOT_EXECUTED** — PyTorch unavailable in this environment.
- PHYSIONet PIC, PECARN: **NOT_AVAILABLE** (credentialed/absent data — no sample
  counts invented).

## Known limitations

Class imbalance (sepsis), one-hospital tabular cohorts, no external appendicitis
cohort executed, image pipeline not run, deterministic (non-LLM) agent backend in
this environment. Full list: `docs/LIMITATIONS.md`.

**Research prototype — not a medical device; not for patient care.**
