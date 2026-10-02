# Phase 2 Report — ML Training, Calibration and Explainability

**Status: GATE PASSED (2026-10-02).** Every number below was produced by
`python scripts/run_phase2.py` on this machine (run time 467.7 s, completed
2026-10-02T15:02:07Z, config hash `cb7198448d25`). Nothing here is estimated,
copied from documentation, or hand-entered.

> Research prototype — not a medical device. Test metrics are single-use
> held-out estimates from one split of one small dataset.

---

## 1. Dataset

| Item | Value |
|---|---|
| Dataset | Regensburg Pediatric Appendicitis dataset |
| Source | `data/raw/regensburg_pediatric_appendicitis/app_data.xlsx` (+ `Data Summary` sheet) |
| DOI | 10.5281/zenodo.7669442 |
| License | CC-BY-NC-4.0 |
| Raw rows | 782 |
| Rows used | **780** (2 rows with missing `Diagnosis` target dropped and logged) |
| Source columns | 58 |
| Target | `Diagnosis` → binary `appendicitis` (1) / `no appendicitis` (0) |
| Class balance | 463 positive / 317 negative |

## 2. Features

- **Included predictors: 52** (numeric + categorical), plus **52 missing-value
  indicators** → model matrix **X = (780, 104)** (68 numeric, 36 categorical
  after encoding columns are planned; indicators are 0/1 numeric).
- Fitted only on the training partition (categorical vocabulary, imputers,
  scalers); validation/test only receive `transform`.

**Excluded columns (6) and reasons** — machine-readable in
`outputs/metadata/regensburg_feature_manifest.json`:

| Excluded | Reason |
|---|---|
| `Diagnosis_Presumptive` | anchor variable reserved for bias experiments (config `feature_policy.exclude`) |
| `Management` | co-target, never a predictor (config exclude) |
| `Severity` | co-target, never a predictor (config exclude) |
| `Length_of_Stay` | post-outcome leakage, recorded at discharge (config exclude) |
| `US_Number` | identifier (patient/image linkage), not a predictor (config exclude) |
| `Diagnosis` | target column |

No other column was silently dropped. Conditional-include circular features
(`Alvarado_Score`, `Paedriatic_Appendicitis_Score`, `Appendix_Diameter`) are
retained per config and analysed separately in a sensitivity run (§7).

## 3. Split (patient-level, seed 20261002)

Strategy `stratified_patient_level` — 70/15/15, stratified on the target;
persisted under `data/interim/splits/*.json` with row indices **and**
`US_Number` patient identifiers. The config fallback
(`repeated_stratified_cv_on_train_val_with_locked_test`) was verified by test
on a synthetic n=60 set; the real test set satisfies the ≥40-per-class rule so
the primary split stands.

| Partition | n | positive | negative |
|---|---|---|---|
| train | 545 | 324 | 221 |
| validation | 118 | 70 | 48 |
| test (locked, single use) | 117 | 69 | 48 |

Test set influenced nothing: no preprocessing fits, no feature selection, no
hyperparameter search, no calibration, no threshold selection.

## 4. LightGBM training

`RandomizedSearchCV` (40 candidates × 5-fold stratified CV, scoring AUPRC) on
the training partition with validation as monitor.

Best CV AUPRC **0.9917**; final model validation AUPRC **0.9934**.

Best hyperparameters:

```
subsample=0.7, reg_lambda=0.5, num_leaves=31, n_estimators=300,
min_split_gain=0.01, min_child_samples=40, max_depth=4,
learning_rate=0.02, colsample_bytree=1.0
```

Artifacts: `outputs/models/lightgbm_appendicitis.joblib`,
`outputs/models/lightgbm_appendicitis_config.json`,
`outputs/models/lightgbm_appendicitis_feature_order.json` (104 ordered
features). Reload verified programmatically.

## 5. Baselines (same splits)

| Model | val AUPRC | test AUROC | test AUPRC | F2 | Brier | threshold |
|---|---|---|---|---|---|---|
| **lightgbm_appendicitis** | 0.9934 | **0.9716** | **0.9815** | 0.9456 | 0.0589 | 0.34 |
| logistic_regression | 0.9764 | 0.9475 | 0.9629 | 0.9129 | 0.0832 | 0.10 |
| random_forest | 0.9859 | 0.9571 | 0.9726 | 0.9306 | 0.0955 | 0.32 |
| hist_gradient_boosting | 0.9966 | 0.9728 | 0.9846 | 0.9538 | 0.0548 | 0.20 |
| catboost | 0.9928 | 0.9777 | 0.9875 | 0.9448 | 0.0473 | 0.37 |
| lightgbm_no_circular_features (sensitivity) | — | 0.8687 | 0.9121 | 0.8757 | 0.1484 | 0.34 |

All baselines return the standardized prediction interface
(`src/models/prediction_interface.py`, Pydantic-validated). Full table:
`outputs/metrics/model_comparison.json` / `.csv`.

Note honestly: on this single split, CatBoost and HistGradientBoosting have
slightly higher test AUPRC than LightGBM; LightGBM remains the designated
primary model per `config/model.yaml`, and the differences are within
bootstrap uncertainty (§6).

## 6. Calibration + threshold (validation only)

| Candidate | Brier (val) | ECE (val) |
|---|---|---|
| uncalibrated | 0.0405 | 0.0732 |
| Platt | 0.0478 | 0.1041 |
| **isotonic (selected)** | **0.0299** | 0.0000* |

\* ECE 0.0 is on the same validation data used to fit isotonic — an
in-sample-of-calibrator figure, not a guarantee on new data.

Selected calibrator persisted to `outputs/models/lightgbm_calibrator.joblib`
(+ `_meta.json`). Threshold selected by F2 on validation: **0.34**
(val F2 0.9687, val sensitivity 0.9714). Raw and calibrated probabilities are
stored in separate columns in `outputs/predictions/test_predictions.csv` and
remain distinguishable in the prediction interface.

## 7. Held-out test results (n=117, 2000-sample bootstrap 95% CIs)

Calibrated probabilities, threshold 0.34:

| Metric | Point | 95% CI |
|---|---|---|
| AUROC | 0.9716 | [0.938, 0.997] |
| AUPRC | 0.9815 | [0.959, 0.997] |
| Sensitivity | 0.9565 | [0.902, 1.000] |
| Specificity | 0.8542 | [0.745, 0.942] |
| PPV | 0.9041 | [0.833, 0.962] |
| NPV | 0.9318 | [0.850, 1.000] |
| F1 | 0.9296 | [0.880, 0.967] |
| F2 | 0.9456 | [0.899, 0.980] |
| Balanced accuracy | 0.9053 | [0.845, 0.955] |
| Brier | 0.0589 | [0.030, 0.096] |
| ECE | 0.0801 | [0.044, 0.126] |

Confusion matrix: TN 41, FP 7, FN 3, TP 66 (from `model_comparison.json`).

**Label-circularity sensitivity** (retrained without `Alvarado_Score`,
`Paedriatic_Appendicitis_Score`, `Appendix_Diameter`): test AUROC 0.8687,
AUPRC 0.9121, sensitivity 0.8986, specificity 0.6667, F2 0.8757. The primary
model's headline numbers therefore depend meaningfully on score/diameter
features — reported as a descriptive comparison, same protocol, different
feature set.

**Subgroups (test):** female n=57 AUROC 0.944 / sens 0.903; male n=60 AUROC
0.998 / sens 1.000; age 6–12 n=76 AUROC 0.968; age 13–18 n=32 AUROC 0.980;
age bands 0–1 (n=1) and 2–5 (n=8) have too few events for metrics and are
reported as such, never silently aggregated.

## 8. False-negative review (safety-critical direction)

3 false negatives of 69 positives (FNR 4.35%), each persisted with clinical
features (research analysis only):

| row | p(cal) | Age | Sex | WBC | CRP | Appendix Ø | Alvarado | PAS |
|---|---|---|---|---|---|---|---|---|
| 717 | 0.067 | 15.9 | female | 7.6 | 0.0 | missing | 6 | 7 |
| 764 | 0.000 | 8.0 | female | 7.0 | 0.0 | 5.0 | 5 | 4 |
| 686 | 0.000 | 12.3 | female | 17.3 | 4.0 | missing | 7 | 8 |

Pattern: two of three are female adolescents with normal-to-mild inflammatory
markers; one has clearly abnormal WBC 17.3 yet scored 0.0 — a hard failure
worth carrying into Phase 3's watchdog/opponent analysis. Full data in
`model_comparison.json → false_negative_review`.

## 9. SHAP explainability

- `shap.TreeExplainer` on the fitted LightGBM; computed for all 117 test rows.
- Global top features (mean |SHAP|): `Appendix_Diameter` 2.457,
  `Surrounding_Tissue_Reaction` 0.465, `Ipsilateral_Rebound_Tenderness__missing`
  0.397, `WBC_Count` 0.364, `Ipsilateral_Rebound_Tenderness` 0.319.
- Figures written: `shap_summary.png` (beeswarm),
  `shap_waterfall_case_test_row_289.png`, `shap_dependence_Appendix_Diameter.png`,
  `shap_dependence_Ipsilateral_Rebound_Tenderness__missing.png`.
- Metadata: `outputs/metadata/shap_explainability.json`;
  patient helper `explain_patient()` →
  `outputs/metadata/shap_explain_patient_test_row0.json`.
- Wording enforced everywhere: *"SHAP indicates the feature's contribution to
  this model prediction, not clinical causation."* No causal phrasing exists in
  code or outputs.
- Implementation note: SHAP's automatic interaction selection crashes on
  mixed str/float categorical columns; `plot_dependence` now resolves the
  interaction partner among numeric columns only and skips (rather than
  fabricates) a plot if rendering fails.

## 10. Artifacts produced

```
outputs/models/lightgbm_appendicitis.joblib
outputs/models/lightgbm_appendicitis_config.json
outputs/models/lightgbm_appendicitis_feature_order.json
outputs/models/lightgbm_calibrator.joblib
outputs/models/lightgbm_calibrator_meta.json
outputs/metrics/model_comparison.json
outputs/metrics/model_comparison.csv
outputs/metadata/regensburg_feature_manifest.json
outputs/metadata/shap_explainability.json
outputs/metadata/shap_explain_patient_test_row0.json
outputs/predictions/test_predictions.csv
outputs/predictions/test_prediction_sample.json
outputs/figures/{roc_curve,pr_curve,confusion_matrix,calibration_curve,
                 shap_summary,shap_waterfall_case_test_row_289,
                 shap_dependence_Appendix_Diameter,
                 shap_dependence_Ipsilateral_Rebound_Tenderness__missing}.png
data/interim/splits/{train,validation,test,split_metadata}.json
```

## 11. Gate checklist

| Requirement | Status |
|---|---|
| Dataset loads + audited | PASS (audit gate enforced by driver) |
| Split reproducibility verified | PASS (deterministic rerun + tests) |
| LightGBM trains | PASS |
| Baselines train (LR, RF, HGB, CatBoost) | PASS |
| Calibration completes | PASS (isotonic selected on validation) |
| Test metrics + bootstrap CIs generated | PASS |
| SHAP works (beeswarm/waterfall/dependence) | PASS |
| Prediction interface works | PASS (pydantic sample validated) |
| Saved models reload | PASS (programmatic reload checked) |
| Tests pass | PASS — **58 passed in 6.06 s** (23 Phase 2 + 35 pre-existing) |

Reproduce: `python scripts/run_phase2.py` (add `--fast --skip-shap` for a
quick pass; the persisted split is reused, delete `data/interim/splits/` to
recreate it).

## 12. Limitations

1. Single 70/15/15 split of n=780; no external validation cohort.
2. Test set is n=117 — CIs are wide (e.g. specificity [0.745, 0.942]).
3. Isotonic ECE on validation is in-sample for the calibrator.
4. Score/diameter features are partially label-circular; sensitivity run shows
   AUROC dropping 0.97 → 0.87 without them.
5. Age bands <6 have too few test events for any metric.
6. Threshold tuned for F2, not for a clinically mandated sensitivity floor;
   choosing operating points is out of scope for this prototype.
7. No LLM/agent claims are made in this phase.
