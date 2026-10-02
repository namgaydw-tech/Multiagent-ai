# Reporting Checklist

Maps project components to clinical-AI reporting frameworks. Status values: **done (Phase 1)**,
**planned**, **N/A**. Verified guideline sources are in `research/literature_review.csv`.

## TRIPOD+AI (prediction model) — DOI:10.1136/bmj-2023-078378

| Item | Where addressed | Status |
|---|---|---|
| Title/abstract identifies model development + bias study aim | README, future paper | planned |
| Data source, setting, dates, eligibility | docs/DATASET_CARD.md | **done** |
| Sample size & events-per-parameter reasoning | docs/DATASET_AUDIT.md, MODEL_CARD.md | **done** (audit) / planned (EPV calc) |
| Target definition & measurement (reference standard) | docs/DATASET_CARD.md (incl. label circularity) | **done** |
| Missing data handling | audit §6 + MODEL_CARD (indicators, no silent drop) | **done** (spec) |
| Candidate predictors & exclusions (leakage) | DATASET_AUDIT §7 leakage screen | **done** |
| Model type, hyperparameters, tuning procedure | config/model.yaml | planned |
| Fairness/subgroup evaluation (age, sex) | evaluation module | planned |
| Performance: discrimination + calibration + CIs | evaluation module (AUROC/AUPRC/Brier/ECE + bootstrap) | planned |
| Model availability & code | this repository | **done** |
| Harms/limitations of deployment use | docs/LIMITATIONS.md, MODEL_CARD.md | **done** |

## STARD-AI (diagnostic accuracy) — DOI:10.1038/s41591-025-03953-8

| Item | Where addressed | Status |
|---|---|---|
| Index test definition (agents/system as index "test") | docs/ARCHITECTURE.md stages | **done** (design) |
| Reference standard (histology / documented diagnosis) | DATASET_CARD.md | **done** |
| Flow & timing of patients/images | split by patient (REPRODUCIBILITY) | planned |
| Threshold selection & pre-specification | MODEL_CARD (locked on validation) | **done** (spec) |
| Distribution of test results / 2×2 table | confusion matrices in outputs/figures | planned |
| Adverse/unsafe outputs reporting | bias metrics + critical-miss rate | planned |
| Subgroup accuracy (age bands) | evaluation | planned |
| AI-specific: model provenance, versioning | model_version in normalized interface | **done** (interface) / planned (implementation) |

## DECIDE-AI (early live clinical evaluation) — DOI:10.1038/s41591-022-01772-9

| Item | Status |
|---|---|
| Live clinical evaluation of the system | **N/A for this project** — no clinical deployment, no patients, no clinicians in the loop |
| Clinical usability, human-AI interaction measures | planned only if a future study with IRB approval happens |
| Clinical-safety incident reporting | N/A (research simulations only) |

## TRIPOD-LLM (LLM/agent studies) — DOI:10.1038/s41591-024-03425-5

| Item | Where addressed | Status |
|---|---|---|
| Model identity, version, provider, access date | config/agents.yaml + audit `model_version` | **done** (config) / planned (logging) |
| Prompt text (all agent prompts) | config/agents.yaml (agent-specific prompts) | planned |
| Sampling parameters (temperature, seeds) | REPRODUCIBILITY.md determinism rules | planned |
| Repeated-run stability | bias metric "Diagnostic Stability" | planned |
| Hallucination / fabrication controls | SAFETY.md (no invented evidence, schema validation) | **done** (policy) |
| Cost and token accounting | architecture logging requirements | planned |
| Human/automated evaluation protocol | evaluation module | planned |
| Full per-stage transcript availability | outputs/audits per-case JSON (project brief §19) | **done** (schema) / planned (runs) |

## CONSORT-AI / SPIRIT-AI — DOI:10.1038/s41591-020-1034-x / DOI:10.1038/s41591-020-1037-7

**N/A** — no clinical trial or interventional protocol is conducted. Retained here so the omission
is explicit rather than accidental.
