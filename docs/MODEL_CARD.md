# Model Card — Pediatric Appendicitis Tabular Model (planned, NOT YET TRAINED)

> **No model has been trained in this repository as of Phase 1.** This card specifies the intended
> model so its evaluation and limits are fixed *before* training (pre-registration discipline).
> Update the Status row when training happens.

## Model details

| Field | Value |
|---|---|
| Status | **PLANNED — training blocked until dataset-audit conditions are accepted** |
| Intended use | Research prototype: producing candidate probabilities + feature attributions for the multi-agent bias experiments. **Not a medical device; not for patient care.** |
| Model family | LightGBM (histogram gradient boosting) as primary candidate |
| Benchmark models | Logistic Regression, Random Forest, HistGradientBoosting/XGBoost-class boosting, CatBoost (categorical handling) |
| Task | Binary classification: `Diagnosis` ∈ {appendicitis, no appendicitis}; separate later models for `Management` and `Severity` |
| Training data | Regensburg Pediatric Appendicitis (UCI 938 / Zenodo 7669442, CC-BY-NC-4.0, n=782), tabular only for Phase A |
| Feature set | Demographics, symptoms/examination, vitals, labs, ultrasound findings, scores — **excluding** `Diagnosis_Presumptive`, `Management`, `Severity`, `Length_of_Stay`, `US_Number` (see docs/DATASET_AUDIT.md §7) |
| Preprocessing | Explicit missingness indicators; native categorical handling; no imputation fit on the full dataset; no oversampling before splitting |
| Split policy | Patient-level, stratified: 70/15/15 train/val/test (or repeated stratified CV on train+val with a locked test partition, chosen after class-count inspection). Splitting happens **before** any preprocessing fit |
| Calibration | Platt and isotonic compared on validation data only; selected by Brier/ECE on validation; **never** on test |
| Threshold | Chosen on validation against a stated utility (e.g., F2 or sensitivity floor), locked before test |
| Explainability | SHAP (global + per-patient), reported as contributions to the model prediction, never as causation |

## Evaluation protocol (locked before training)

Report with bootstrap 95% CIs: AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, F2,
balanced accuracy, Brier score, ECE, calibration curve, confusion matrix — plus an explicit
false-negative review for safety-sensitive analysis. Accuracy alone is never reported in isolation.

## Training-data analysis (from the audit)

- 782 patients; 59.2% positive; 30.9% cells missing; 0 duplicates.
- Label circularity: for conservatively managed patients the `Diagnosis` label was defined partly
  via Alvarado/PAS ≥ 4 **and** appendix diameter ≥ 6 mm → mandatory sensitivity analysis excluding
  `Alvarado_Score`, `Paedriatic_Appendicitis_Score`, `Appendix_Diameter`.
- `Appendix_Diameter` alone reaches univariate AUC ≈ 0.95 against `Diagnosis` — expected, and a
  warning about circularity, not a feature to brag about.

## Known limitations / risks

- Single centre, n=782, ~53 candidate features → overfitting risk; wide CIs; no external cohort.
- Informative missingness (ultrasound fields recorded only when something was seen) — models may
  learn "was visualised" rather than pathology; mitigated with missingness indicators + ablations.
- Referral/verification bias (histology only for operated patients).
- Population age skew: only 5 patients under 2 years → infant predictions are out-of-distribution.
- Calibration measured in-distribution only; must be re-measured if applied to another site.
- CC-BY-NC-4.0: outputs derived for commercial purposes are out of license scope.

## Intended vs out-of-scope use

- In scope: reproducible research on confirmation-bias mitigation; baseline comparisons.
- Out of scope: any clinical deployment, triage, treatment recommendation, or claims of clinical
  validation. The multi-agent confidence produced downstream is **not** a clinical risk estimate.
