# External Validation Status (Phase 5)

**Status: NOT EXECUTED for a true external cohort.** This document exists to keep
the word "external validation" honest. Research prototype — not a medical device.

## Definition used in this repository

**External validation** = evaluating a *locked* model (features, preprocessing,
calibrator, threshold all frozen) on an **independent patient cohort for the same
clinical task**, from a different source, without any refitting.

Anything less is labelled precisely:

| Situation | Correct label | Status here |
| --- | --- | --- |
| Same Regensburg patients, ultrasound views | **Internal / multimodal extension** — NOT external (shares patients with the tabular cohort) | NOT_EXECUTED (torch unavailable) |
| Regensburg validation → sealed test split | **Internal hold-out test** | EXECUTED |
| Neonatal sepsis cohort, different disease + target | **Cross-domain replication of the pipeline** — NOT external validation of the appendicitis model | EXECUTED |
| Another appendicitis cohort (e.g., independent hospital) | **True external validation** | **NOT EXECUTED — no such cohort available** |
| Kermany Mendeley vs its Kaggle mirror | **One dataset, two mirrors** — never two cohorts | mirror deduplicated |

Machine-readable status: `outputs/phase5/external_validation.json`
(`status: NOT_EXECUTED`, with the reason recorded).

## What WAS executed (and what it proves)

1. **Regensburg appendicitis, persisted Phase 2 split** — 17 algorithms + 4
   ensembles, validation-locked thresholds, sealed test opened once.
   Proves: the ML layer's robustness *within* the reference benchmark across model
   families (not external validity).
2. **Neonatal sepsis registry (DOI 10.17632/5vdz5cftz7.1)** — independent patients,
   different disease, group-split by patient, full pipeline.
   Proves: the *pipeline and the experimental protocol* transfer to a second
   pediatric domain. It says **nothing** about appendicitis-model transportability.
3. **Cross-backbone bias experiments** (`outputs/phase5/bias_backbones_summary.json`)
   — same engine, same cases, six classifiers.
   Proves: (or refutes) backbone-dependence of the debiasing finding — a robustness
   claim about the *architecture*, not external clinical validation.

## What would be required for a real external validation (Phase 6+)

- An independent appendicitis-suspected cohort with the same feature semantics;
- frozen preprocessing artefacts (the Phase 2 encoder state) applied without refit;
- the locked LightGBM/gradient-boosting calibrator and threshold applied unchanged;
- pre-registered acceptance bounds (e.g., AUROC CI lower bound ≥ 0.80, sensitivity
  ≥ 0.90 at the frozen threshold);
- patient-level deduplication proof against the Regensburg source.

Until such a cohort is downloaded, audited and run, this repository will keep
reporting **NOT EXECUTED** — no fabricated external numbers.

## Credentialed / blocked datasets

- **PHYSIONET_PIC** — `NOT_AVAILABLE_REQUIRES_PHYSIONET_CREDENTIALING`; the pipeline
  must not fail without credentials and must not invent sample counts.
- **PECARN** — `NOT_AVAILABLE` (access/absence; no data on disk).
- **CHILD_PNEUMONIA_MENDELEY** — `BLOCKED` (pre-augmented `aug_*` files, duplicate
  filenames, no patient grouping → split leakage risk). Not unlocked.

**Research prototype — not a medical device; not for patient care.**
