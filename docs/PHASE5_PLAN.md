# Phase 5 Plan — Multi-Dataset, Multi-Algorithm External Validation and Robustness

**Status: EXECUTED for two tabular datasets; BLOCKED / NOT_AVAILABLE / NOT_EXECUTED for the rest (see `docs/PHASE5_DATASET_AUDIT.md`).**
Research prototype — not a medical device.

## 1. Purpose

Phase 5 does **not** merge unrelated pediatric datasets to inflate `N`. Each dataset
is a separate experiment. The phase answers five questions:

1. Does the ML layer stay performant across pediatric diagnostic domains?
2. Does the adversarial, information-isolated multi-agent architecture generalize
   beyond Regensburg appendicitis?
3. Do conclusions hold across algorithms and modalities?
4. Does confirmation-bias reduction persist across datasets **and model backbones**?
5. Do conclusions survive leakage controls, dataset blocking, external-validation
   discipline and calibration analysis?

## 2. Pre-registered hypothesis (before new test-set results were opened)

> **H5 (primary).** Adversarial, information-isolated multi-agent diagnostic reasoning
> reduces confirmation bias (lower CBR than single-agent/seeing-the-prediction
> baselines) consistently across distinct pediatric diagnostic datasets, modalities,
> diseases and underlying machine-learning classifiers.

Secondary hypotheses:

- **H5-a (robustness).** The Phase 2 reference quality (validation Macro F0.5 ≥ 0.90
  at sensitivity ≥ 0.90) replicates on the Regensburg persisted split for at least
  one model family beyond LightGBM.
- **H5-b (floor policy).** Every deployed operating point satisfies
  sensitivity ≥ 0.90 chosen on validation only.
- **H5-c (backbone independence).** CBR/BCR/HFR conclusions do not flip sign when the
  underlying classifier changes (LR/RF/XGB/LGBM/CatBoost/ensemble).

No hypothesis is revised after test results are seen; deviations are recorded in
`docs/PHASE5_REPORT.md`.

## 3. Non-negotiable rules (enforced in code + tests)

- Never concatenate diseases; mirrors are one dataset (Kermany Mendeley = Kaggle).
- Test data never fits preprocessing, imputation, features, hyperparameters,
  calibration, thresholds, ensemble weights or early stopping.
- Patient/group-level splits wherever identifiers exist; augmentation only on train.
- No silent missing-value drops, no fabricated labels, no targets invented from
  "useful-looking" columns.
- BLOCKED datasets cannot reach the trainer (`assert_trainable` gate).
- Every displayed number comes from `outputs/` — no hard-coded UI metrics.

## 4. Components and where they live

| Component | Artefact |
| --- | --- |
| Dataset blocker (27 checks) | `src/data/dataset_blocker.py`, `config/dataset_blocking.yaml` |
| Dataset registry / loaders | `src/data/phase5_datasets.py`, `config/phase5.yaml` |
| Model zoo (17 algorithms + 4 ensembles) | `config/model_zoo.yaml`, `src/models/model_registry.py` |
| Train → calibrate → lock threshold → seal test | `src/models/phase5_pipeline.py` |
| Macro F0.5 + full metrics + CIs + tests | `src/evaluation/phase5_metrics.py` |
| Sensitivity-floor threshold policy | `src/evaluation/threshold_policy.py` |
| Calibration (Platt/isotonic, val-only) | `src/calibration/phase5.py` |
| Scorecard + winner selection (validation only) | `src/evaluation/scorecard.py` |
| Post-hoc statistics (paired Δ, McNemar, DeLong, BH) | `src/evaluation/phase5_stats.py` |
| 35-figure suite + manifest | `src/evaluation/phase5_figures.py` |
| Domain packs for agents | `src/agents/domains.py` |
| Cross-backbone bias experiments | `src/bias/backbones.py` |
| CLI | `scripts/run_phase5.py {audit,train,evaluate,agents,bias,analyze,all}` |
| API | `backend/api/phase5.py` (`/api/phase5/*`) |
| UI | Dataset Registry, Cross-Dataset Benchmark, Algorithm Lab pages |
| Tests | `tests/test_phase5.py` |

## 5. Execution order (as run)

1. Inspect repository (baseline `7f7e04f`).
2. Configs + blocker → audit all 7 candidates → persist reports.
3. Freeze eligible datasets (Regensburg tabular PASS; sepsis CONDITIONAL with a
   verified culture-positive target).
4. Macro F0.5 + metric battery (verified against sklearn).
5. Model registry (17 algorithms; scaling only where required; CatBoost native cats).
6. Splits: Regensburg reuses the persisted Phase 2 split verbatim; sepsis uses a
   persisted stratified **group** split (patients never cross partitions).
7. Train on train only → calibrate on validation → select thresholds on validation
   (sensitivity floor 0.90) → fit ensembles on validation → lock.
8. Open sealed test exactly once per dataset; persist lock + per-case predictions.
9. Scorecard, validation-only winner selection, cross-dataset summary.
10. Post-hoc statistics on the test partition (inference only).
11. Figures (35/35) from persisted artefacts.
12. Multi-agent domain packs + cross-backbone bias experiments.
13. Backend + frontend (3 new pages) + docs + full test suite.

## 6. Reproduce

```bash
python scripts/run_phase5.py audit                      # 27-check blocker
python scripts/run_phase5.py train --dataset REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR
python scripts/run_phase5.py evaluate --dataset REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR
python scripts/run_phase5.py analyze                    # scorecard + stats + 35 figures
python scripts/run_phase5.py bias --dataset REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR \
    --backbone all --rows 40                            # cross-backbone CBR/BCR/HFR
python -m pytest tests/ -p no:warnings                  # full suite
```

**Research prototype — not a medical device; not for patient care.**
