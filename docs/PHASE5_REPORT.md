# Phase 5 Report — Multi-Dataset, Multi-Algorithm Robustness & External Validation

**Status: EXECUTED for two tabular datasets; BLOCKED / NOT_AVAILABLE / NOT_EXECUTED
elsewhere (reported honestly below).** Every number in this report is read from
persisted artefacts under `outputs/phase5/` and regenerable with
`python scripts/run_phase5.py analyze`. Research prototype — not a medical device.

## 1. Executive summary

- 7 candidate datasets audited by the 27-check blocker: **1 PASS, 3 CONDITIONAL,
  1 BLOCKED, 2 NOT_AVAILABLE** (`outputs/phase5/dataset_blocking_report.json`).
- 2 datasets executed end-to-end as **separate** experiments (never concatenated):
  Regensburg appendicitis tabular (PASS, reused Phase 2 split verbatim) and the
  neonatal sepsis registry (CONDITIONAL, patient-group split).
- **17 tabular algorithms + 4 ensembles** trained per dataset (QDA unavailable on
  Regensburg — rank-deficient covariance, recorded, not faked).
- Thresholds/calibration/ensemble weights fitted **on validation only** and locked
  before each sealed test was opened exactly once.
- Validation-selected winner: **gradient_boosting on both datasets**
  (Regensburg test Macro F0.5 0.911 [0.854, 0.958]; sepsis 0.485 [0.448, 0.525]).
- Post-hoc statistics: paired bootstrap Δ, exact McNemar, DeLong CIs, BH correction
  (116 rows in `statistical_comparison.json`).
- **35/35 required figures generated** (67 PNGs) from persisted outputs, with a
  manifest that says NOT GENERATED + reproduce command for anything absent.
- **Cross-backbone bias experiments executed** (6 backbones × 160 paired runs):
  the isolated engine's CBR stays ≤ 0.025 across all classifiers vs 0.475–0.575 for
  the single-agent baseline — H5-c holds on this subset.
- Backend `/api/phase5/*` + 3 new UI pages (Dataset Registry, Cross-Dataset
  Benchmark, Algorithm Lab) verified in the browser.

## 2. Dataset blocking outcomes (exact reasons abridged; full text in
`docs/PHASE5_DATASET_AUDIT.md`)

| Dataset | Status | Key reason |
| --- | --- | --- |
| REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR | PASS | reference benchmark; existing persisted split reused, no re-split |
| NEONATAL_SEPSIS_REGISTRY | CONDITIONAL | license not exposed by API; post-outcome/treatment fields excluded from predictors; target = culture-positive vs culture-negative evaluation (verified) |
| REG_ENSBURG_ULTRASOUND_IMAGES | CONDITIONAL | shares patients with the tabular cohort → internal multimodal extension, **not** an external population; image training not run |
| KERMANY_PEDIATRIC_PNEUMONIA | CONDITIONAL | Kaggle copy is a mirror (ONE dataset); OCT excluded; image training not run |
| PHYSIONET_PIC | NOT_AVAILABLE | NOT_AVAILABLE_REQUIRES_PHYSIONET_CREDENTIALING — no local data, no invented counts |
| PECARN | NOT_AVAILABLE | access approval unavailable; no invented counts |
| CHILD_PNEUMONIA_MENDELEY | BLOCKED | 2,644 pre-augmented `aug_*` files, 279 duplicate filenames, no patient grouping → split-leakage risk; gate refuses training |

## 3. Splits (persisted)

| Dataset | Train | Validation | Test | Split source |
| --- | --- | --- | --- | --- |
| Regensburg appendicitis | 545 (221/324) | 118 (48/70) | 117 (48/69) | `phase2_persisted_existing_split` |
| Neonatal sepsis | 1,337 (1253/84) | 309 (286/23) | 300 (279/21) | `phase5_group_split_persisted` (patients never cross) |

## 4. Winners (selected on validation only; locked before test)

| Dataset | Winner | thr | Macro F0.5 (test) | pos F0.5 | Acc | Bal Acc | Prec | Sens | Spec | AUROC | AUPRC | Brier | ECE | MCC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Regensburg | gradient_boosting (isotonic) | 0.19 | 0.911 | 0.935 | 0.915 | 0.915 | 0.940 | 0.913 | 0.917 | 0.952 | 0.951 | 0.062 | 0.054 | 0.825 |
| Neonatal sepsis | gradient_boosting (isotonic) | 0.04 | 0.485 | 0.138 | 0.547 | 0.668 | 0.114 | 0.810 | 0.527 | 0.697 | 0.145 | 0.064 | 0.035 | 0.172 |

Bootstrap 95% CIs (2,000 replicates): Regensburg Macro F0.5 [0.854, 0.958],
AUROC [0.907, 0.986]; sepsis Macro F0.5 [0.448, 0.525], AUROC [0.601, 0.789].
Both winners satisfy the sensitivity floor **on validation** (Regensburg test
sensitivity 0.913; sepsis test sensitivity 0.810 at a prevalence of 7% — the floor
is a validation-selection constraint, not a test-set guarantee).

Test-set leaders (reporting only, **not** selection): Regensburg stacking
Macro F0.5 0.941; sepsis adaboost 0.492. Full 20-row tables with every metric and
CI: `outputs/phase5/<ds>/metrics/metrics_test.json`; consolidated 42-row scorecard:
`outputs/phase5/model_scorecard.{csv,json}` (unavailable metrics = null, never 0).

## 5. Statistics (post-hoc, test labels used for inference only)

`outputs/phase5/statistical_comparison.{json,csv}` — 116 rows:

- paired bootstrap Δ Macro F0.5, winner vs each competitor (B = 2,000);
- exact McNemar on discordant hard predictions;
- DeLong AUROC 95% CI per model;
- bootstrap CI for the ensemble probabilities;
- BH q-values per dataset family (e.g., Regensburg: winner vs gaussian_nb
  significant at q < 0.05; winner vs adaboost not — Δ = −0.008, p = 0.75).

The vectorized paired bootstrap is asserted numerically identical to the reference
implementation (`tests/test_phase5.py::test_fast_paired_bootstrap_equals_reference`).

## 6. Figures — 35/35

`python scripts/run_phase5.py analyze` → 67 PNGs + `figures_manifest.json`
(mapping all 35 required figures to paths). Includes 8 leaderboards, ROC/PR,
confusion matrices (raw + normalized), reliability curves, four threshold curves +
operating point (all watermarked **DEVELOPMENT / VALIDATION**), learning curves,
native + permutation importance, SHAP summary/waterfall (non-causal wording),
RF tree-count / KNN K / SVM C curves, boosting comparison,
ensemble-vs-individual, times, performance-vs-complexity, cross-dataset view,
validation→test gaps, CI chart for validation-ranked leaders, CBR comparison,
BCR-vs-HFR, and the cross-backbone CBR/BCR figure.
**NOT generated:** Grad-CAM / image-model figures (torch missing — manifest carries
the reproduce command, never a fake image).

## 7. Cross-backbone bias experiments (H5-c)

Design: same 40 persisted test rows × 2 conditions (control, incorrect anchor) ×
2 profiles (single-agent, full isolated engine) × 6 backbones = **960 case-runs**;
uniform information budget (no SHAP for any backbone), ground truth never shown to
agents, deterministic LLM backend, seed 20261002.
Results (`outputs/phase5/bias_backbones_summary.json`, incorrect-anchor condition,
engine profile): CBR = 0.025 (LR), 0.000 (RF), 0.000 (XGB), 0.025 (LGBM),
0.000 (CatBoost), 0.025 (ensemble) vs single-agent CBR 0.475–0.575;
BCR 0.875–0.975. **Conclusion on this subset:** the debiasing finding does not
depend on the underlying classifier. HFR is undefined (excluded) here, so HFR
claims come only from the Phase 4 headline experiments.

## 8. API / UI

- `GET /api/phase5/{status,datasets,scorecard,cross-dataset,external-validation,
  statistics,figures,figures/{id},figure-file,algorithms,dataset/{id}/metrics}` —
  each serves real persisted JSON or `available: false` + reproduce hint.
- UI pages: `/datasets` (registry + exact blocking reasons), `/benchmark`
  (scorecard with n/a-not-0, gaps, paired statistics), `/lab` (dataset × algorithm ×
  metric, formulas, hyperparameters, locked threshold, val/test tables with CIs,
  figure viewer incl. NOT GENERATED fallback). Verified in the browser; console
  error-free; `npm run build` green.

## 9. Tests

- Historical Phase 1–4 suite: **125 tests** — still passing.
- New Phase 5 suite (`tests/test_phase5.py`): Macro F0.5 vs sklearn (binary,
  multiclass, zero-division), threshold floor + fallback + validation-only source,
  blocker statuses/gate/mirror dedup, split disjointness + patient grouping,
  17-model zoo without Linear Regression, scaling policy, scorecard fields + null
  policy + validation-only winner, statistics artifacts + BH invariant, fast
  paired-bootstrap equivalence, 35/35 figure manifest, calibration validation-only
  + test lock, domain packs, CLI dry-runs/NOT_EXECUTED exits, cross-backbone
  summary, and API endpoints incl. traversal guard.
- Full suite: **174 passed, 0 failed, exit 0**
  (`python -m pytest tests/ -p no:warnings`) = 125 historical + 49 Phase 5 tests.
- `npm run build` (tsc + vite): **PASS**.

## 10. Not executed / blocked (honest list)

| Item | Status | Reason |
| --- | --- | --- |
| Kermany/US CNNs, Grad-CAM, late fusion | NOT_EXECUTED | torch not installed |
| PHYSIONET_PIC, PECARN | NOT_AVAILABLE | credentials/access |
| CHILD_PNEUMONIA_MENDELEY | BLOCKED | pre-augmented leakage, duplicates, no grouping |
| Multi-agent runner on sepsis/pneumonia domains | PLANNED | domain packs exist; runner not implemented — CLI exits with explicit NOT_EXECUTED |
| True external appendicitis cohort | NOT EXECUTED | none available (see `docs/EXTERNAL_VALIDATION.md`) |

## 11. Reproduce

```bash
python scripts/run_phase5.py audit
python scripts/run_phase5.py train --dataset REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR
python scripts/run_phase5.py evaluate --dataset REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR
python scripts/run_phase5.py train --dataset NEONATAL_SEPSIS_REGISTRY
python scripts/run_phase5.py evaluate --dataset NEONATAL_SEPSIS_REGISTRY
python scripts/run_phase5.py analyze                      # scorecard + statistics + 35 figures
python scripts/run_phase5.py bias --dataset REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR --backbone all --rows 40
python -m pytest tests/ -p no:warnings
cd frontend && npm run build
```

**Research prototype — not a medical device; not for patient care.**
