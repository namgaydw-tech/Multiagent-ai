# Mitigating Confirmation Bias in Automated Pediatric Diagnostics via Adversarial Multi-Agent AI Frameworks

> **Research prototype. Not a medical device. Not clinically validated. Not a replacement for
> qualified pediatric clinicians. No output of this system may be used for patient care.**

## What this project actually is

An academic clinical-AI research platform whose **primary objective is not accuracy**. It
experimentally tests whether an adversarial, information-isolated multi-agent reasoning framework
reduces:

1. confirmation bias,
2. diagnostic anchoring,
3. premature diagnostic closure,
4. inappropriate adherence to an incorrect preliminary diagnosis,
5. failure to respond to contradictory clinical evidence,

— compared with a conventional single-model diagnostic pipeline and with conventional (cooperative)
multi-agent pipelines.

Core hypothesis (falsifiable): *explicit adversarial diagnostic reasoning with information
isolation reduces confirmation bias (CBR) and increases beneficial corrections (BCR) without
raising the harmful flip rate (HFR) beyond a pre-registered margin.* See
[research/RESEARCH_GAP.md](research/RESEARCH_GAP.md).

## Status: Phases 1–4 complete, UI complete, all tests passing

| Phase | What ran | Status |
|---|---|---|
| Phase 1 | Dataset audit, literature review (46 verified entries), configs, safety knowledge | **COMPLETED** (commit `47ca7d9`) |
| Phase 2 | LightGBM training + calibration + SHAP + evaluation on Regensburg | **EXPERIMENTALLY VALIDATED** (commit `e57cdd7`) |
| Phase 3 | Five-agent debiasing engine + RAG + safety orchestration | **IMPLEMENTED + EXECUTED** (commit `2173778`) |
| Phase 4 | Anchor experiments, 11 ablations, statistics, error analysis, figures | **EXPERIMENTALLY VALIDATED** (commit `7568b08`) |
| UI | FastAPI backend + React research UI (9 pages, responsive, dark/light) | **IMPLEMENTED + VERIFIED IN BROWSER** |

Test suite: **125 passed** (`python -m pytest tests/ -p no:warnings`).
The repository clearly distinguishes **IMPLEMENTED** from **PLANNED** from
**EXPERIMENTALLY VALIDATED** — see the *Implementation status* section and
`outputs/metrics/final_results.json → implementation_status`.

---

## Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │  Layer A — Disease-specific tabular model (Phase 2) │
                    │  LightGBM → isotonic calibrator → threshold 0.34    │
                    │  + SHAP explainer + uncertainty estimator           │
                    │  Normalized JSON prediction interface               │
                    └───────────────────────┬─────────────────────────────┘
                                            │  model_output + shap + uncertainty
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│  Layer B — Common multi-agent debiasing engine (Phase 3), 8-stage state machine   │
│                                                                                   │
│  1_data_cleansing ──► 2_independent_differential ──► 3_proponent ──► 4_opponent    │
│        (blind)              (blind)                    (sees p)        (sees p)   │
│              │                                                               │    │
│              ▼                                                               ▼    │
│  5_high_acuity_watchdog ──► 6_evidence_retrieval ──► 7_arbitration ──► 8_bias_audit│
│        (blind)                  (RAG, blind)          (sees p)         (sees all) │
│                                                                                   │
│  Visibility policy is config-driven (config/agents.yaml + ablation profiles):     │
│  prediction, anchor and ground truth are withheld per stage and recorded.         │
│  Every stage persists input/output/meta JSON under outputs/audits/<case_id>/.      │
└───────────────────────┬───────────────────────────────────────────────────────────┘
                        │  final working diagnosis + confidence + audit trail
                        ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│  Layer C — Bias experiments (Phase 4): 7 anchor conditions × 11 ablation          │
│  profiles × 117 test cases × 3 repetitions, all with seed 20261002                │
│  CBR/AOR/BCR/HFR/CRR/CMR/DR + McNemar + paired bootstrap + Benjamini–Hochberg     │
└───────────────────────────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│  Layer D — Serving: FastAPI (backend/) on :8765 + React/Vite research UI          │
│  (frontend/) on :5199 — run the model, run the agents, run quick experiments,     │
│  browse audits and persisted metrics. The legacy static dashboard/ is untouched.  │
└───────────────────────────────────────────────────────────────────────────────────┘
```

Detailed design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## How this is a multi-agent AI system

Five specialist agents, each with its own prompt, Pydantic output schema, and — critically —
its own **visibility policy** (what it is allowed to see):

| # | Agent | Role | Sees model prediction? | Sees anchor? | Sees ground truth? |
|---|---|---|---|---|---|
| 1 | **Data Cleanser** | Normalizes the record, flags missingness, never imputes outcomes | no | no | no |
| 2 | **Independent Differential** | Builds a differential *before* any prediction is shown (anti-anchoring) | **no** | no | no |
| 3 | **Proponent** | Argues the leading diagnosis — steelmans the case *for* | yes | yes (if injected) | no |
| 4 | **Opponent / Devil's Advocate** | Argues against the leading diagnosis — the adversarial counterweight | yes | yes (if injected) | no |
| 5 | **High-Acuity Watchdog** | Pediatric safety net: enforces fail-closed rules from `knowledge/*.yaml` (missed surgical target ⇒ escalate) | no | no | no |
| 6 | **Evidence Retrieval (RAG)** | Hybrid TF-IDF + SQLite FTS5 over 72 offline knowledge chunks; **provenance mandatory** — every claim carries its source | no | no | no |
| 7 | **Arbitrator (Pediatric Consultant)** | Reads both sides + watchdog + evidence and issues the final working diagnosis with confidence | yes | yes | no |
| 8 | **Bias Audit** | Deterministic computation of anchor-following, flips and contradictions for this case | yes | yes | **yes** |

Why this is *multi-agent* and not just a prompt chain:

- **Adversarial by construction** — agent 4 exists solely to attack the diagnosis agent 3 defends.
  The ablation matrix shows profile 5 (*generic multi-agent, no opponent*) has **zero harmful
  flips**: the opponent is the mechanism that can both correct and flip a case.
- **Information isolation** — agents 1, 2, 5, 6 are *blind* to the model's prediction and to the
  injected anchor. Stage 2 always completes before the prediction is revealed, so the differential
  cannot be contaminated by the anchor. Isolation is data-driven (`VisibilityPolicy`), recorded per
  stage in `withheld_from_this_stage`, and verified by tests.
- **Schema-validated state machine** — every stage returns a typed Pydantic payload
  (`SCHEMA_REGISTRY`); invalid output fails closed and is recorded in `audit.json`, never
  silently coerced.
- **Safety layer** — the watchdog applies pre-registered pediatric fail-closed rules independent of
  the debate; the arbitration stage cannot overrule an unresolved high-acuity flag without
  recording it.
- **Full auditability** — every stage's exact input/output/prompt-hash/latency/token counts are
  persisted; the UI's Audit Viewer reads those files directly, nothing is regenerated for display.

Agent configuration: [config/agents.yaml](config/agents.yaml). Orchestrator:
`src/orchestration/{state,stages,engine,persistence}.py`.

> **LLM backend disclosure:** no API key is configured in this environment, so every agent runs on
> the **deterministic rule-based provider** (`LLM_BACKEND` unset → `deterministic`). All Phase 3/4
> numbers therefore describe the framework *under a deterministic backend* — this is stated in
> every report and in `final_results.json`. Live-LLM conditions are **not validated** here.

---

## Formulas used

### Phase 2 — model metrics (`src/evaluation/model_metrics.py`)

All computed on the held-out test partition (n = 117), with **2000-sample bootstrap 95% CIs**
(seed 20261002) where computationally practical.

| Metric | Definition |
|---|---|
| AUROC | Rank-based area under the ROC curve (Mann–Whitney U formulation) |
| AUPRC | Average precision = Σₖ (Rₖ − Rₖ₋₁)·Pₖ — the primary optimization target (positive class ≈ 41%) |
| Sensitivity | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| PPV | TP / (TP + FP) |
| NPV | TN / (TN + FN) |
| F1 | 2·PPV·Sens / (PPV + Sens) |
| F2 | (1 + β²)·PPV·Sens / (β²·PPV + Sens), β = 2 — threshold selected on **validation** set |
| Balanced accuracy | (Sensitivity + Specificity) / 2 |
| Brier score | (1/n)·Σᵢ (pᵢ − yᵢ)² |
| ECE | Σₘ (|Bₘ|/n)·\|acc(Bₘ) − conf(Bₘ)\| over 10 equal-width confidence bins |
| Confusion matrix | (TN, FP, FN, TP) at the calibrated threshold 0.34 |

**Calibration** (`src/calibration/calibrate.py`): uncalibrated vs. Platt (sigmoid) vs. isotonic
regression, fitted on **validation data only** (never test). Selected: **isotonic**
(Brier 0.0589, ECE 0.0801). Raw and calibrated probabilities remain separately addressable
(`p_raw`, `p_calibrated`).

**Uncertainty** (`src/models/uncertainty.py`) — probability is *not* mislabelled as uncertainty:

```
H(p) = −p·ln p − (1−p)·ln(1−p)            # predictive entropy in nats (max ln 2 ≈ 0.6931)
margin = min(1, 2·|p_calibrated − threshold|)  # separation from the decision boundary
level  = LOW      if H ≤ 0.45
         MODERATE if H ≤ 0.63
         HIGH     otherwise                  # margin < 0.10 elevates to at least MODERATE
```

**SHAP** (`src/explainability/shap_engine.py`): TreeExplainer global importance, beeswarm,
per-patient waterfall and dependence plots. Wording rule enforced in code and UI:
*"Feature X **contributed to this model prediction**"* — never *"caused the diagnosis"*.

### Phase 4 — confirmation-bias metrics (`src/bias/metrics.py`)

Definitions taken verbatim from `config/experiment.yaml → bias_metrics`:

| Metric | Formula | Meaning |
|---|---|---|
| **CBR** | `followed_incorrect_anchor / incorrect_anchor_cases` | Confirmation-bias rate: the system adopted a wrong anchor. **Lower is better.** |
| **AOR** | `rejected_incorrect_anchor / incorrect_anchor_cases` | Anchor-override rate: the system resisted a wrong anchor |
| **BCR** | `incorrect_initial_corrected / incorrect_initial` | Beneficial corrections: the debate fixed a wrong initial call |
| **HFR** | `initially_correct_changed_to_incorrect / initially_correct` | Harmful flip rate: the debate broke a correct call. **Lower is better.** |
| **CRR** | `cases_revised_appropriately_on_new_contradictory_evidence / cases_with_decisive_contradictory_evidence` | Responsiveness to contradiction |
| **CMR** | `high_acuity_targets_missed / high_acuity_targets_present` | Missed surgical-target rate (safety) |
| **DR** | `identical_conclusion_rate_across_repeated_runs` | Diagnostic stability across the 3 repetitions |
| **DDR** | `unique_plausible_hypotheses_before_consensus` | Differential-diversity count |
| **Bias reduction** | `CBR_single_agent − CBR_multi_agent` | Pre-registered primary effect size |

Statistical machinery (all seeded, all persisted):

- **95% CIs** — case-level percentile bootstrap, 2000 draws, seed 20261002.
- **Exact McNemar test** — two-sided binomial tail over discordant pairs
  `p = 2·Σᵢ₌₀^min(b,c) C(n,i)·(½)ⁿ` with `n = b + c`.
- **Paired bootstrap difference** — resample *paired* cases, difference of rates, two-sided p.
- **Benjamini–Hochberg** — FDR correction across each reported family, α = 0.05, q-values
  reported as `q`.
- **Tri-state rule** — a rate whose denominator is 0 returns `value: null` (**n/a**, never a
  disguised 0); unjudgeable cases are excluded from numerator *and* denominator and counted in
  `n_excluded`.

---

## Graphs (generated figures)

All figures are produced from real computed values by `scripts/run_phase2.py` and
`scripts/run_phase4.py analyze` → `outputs/figures/`. Nothing is hard-coded; a missing figure
renders as an explicit "not generated" placeholder in the UI.

### Phase 2 (`src/evaluation/model_metrics.py`, `src/explainability/shap_engine.py`)

| Figure | Contents |
|---|---|
| `roc_curve.png` | ROC per model (LightGBM + 4 baselines), test n=117, AUROC in legend |
| `pr_curve.png` | Precision–recall per model with no-skill baseline (≈41% positives) |
| `confusion_matrix.png` | LightGBM TN41 / FP7 / FN3 / TP66 at threshold 0.34 |
| `calibration_curve.png` | Reliability diagram: uncalibrated vs. Platt vs. isotonic, with ECE/Brier |
| `shap_summary.png` | Global SHAP beeswarm — Appendix_Diameter leads (mean\|SHAP\| 2.4565) |
| `shap_waterfall_case_test_row_289.png` | Patient-level waterfall for one test case |
| `shap_dependence_*.png` | SHAP dependence plots (Appendix_Diameter, Ipsilateral_Rebound_Tenderness) |

### Phase 4 (`src/evaluation/publication_figures.py`)

| Figure | Contents |
|---|---|
| `fig_anchor_cbr_aor.png` | CBR and AOR per anchor condition, multi-agent vs. single-agent baselines |
| `fig_anchor_outcomes.png` | Followed / resisted / induced-error outcomes per condition |
| `fig_bias_reduction.png` | Paired bias reduction (single − multi) with bootstrap CI |
| `fig_ablation_matrix.png` | Accuracy / CBR / HFR across all 11 ablation profiles |
| `fig_error_taxonomy.png` | Error-tag counts across the 117 test cases |

---

## Headline results (all values read from `outputs/metrics/*.json`)

### Phase 2 — model (test partition, n = 117; split 545 / 118 / 117, seed 20261002)

| Model | AUROC (95% CI) | AUPRC | Sens | Spec | Brier | ECE |
|---|---|---|---|---|---|---|
| **lightgbm_appendicitis (primary)** | **0.972 [0.938, 0.997]** | **0.982** | 95.7% | 85.4% | 0.059 | 0.080 |
| catboost | 0.978 [0.951, 0.996] | 0.987 | 94.2% | 93.8% | 0.047 | 0.061 |
| hist_gradient_boosting | 0.973 [0.942, 0.995] | 0.985 | 95.7% | 91.7% | 0.055 | 0.063 |
| random_forest | 0.957 [0.925, 0.984] | 0.973 | 97.1% | 64.6% | 0.096 | 0.089 |
| logistic_regression | 0.947 [0.904, 0.981] | 0.963 | 94.2% | 68.8% | 0.083 | 0.080 |

Confusion matrix (LightGBM): **TN 41 / FP 7 / FN 3 / TP 66**. False negatives (rows 717, 764, 686)
are reviewed explicitly in `docs/MODEL_CARD.md` — never hidden.

### Phase 4 — bias experiment (117 test cases × 3 repetitions = 351 rows per condition)

| Condition | CBR | AOR | BCR | HFR | Final accuracy |
|---|---|---|---|---|---|
| control | n/a (no incorrect anchor) | n/a | 0.000 (n=30) | 0.028 (n=321) | 0.889 |
| **incorrect_anchor** | **0.051 [0.017, 0.094]** | **0.949** | 0.915 | n/a | 0.915 |
| correct_anchor | n/a | n/a | n/a | 0.111 (n=351) | 0.889 |

**Bias reduction vs. single agents (pre-registered primary effect):**

| Comparison | CBR | Δ vs multi | p |
|---|---|---|---|
| Single LLM, record only (profile 2) | 0.974 [0.940, 1.000] | — | — |
| Single LLM + model (profile 3) | 0.521 [0.427, 0.615] | — | — |
| **Full five-agent + RAG (profile 9)** | **0.051 [0.017, 0.094]** | — | — |
| Paired difference (single₃ − multi) | — | **0.470 [0.376, 0.564]** | **p = 0.0009995** (paired bootstrap, 2000 draws) |
| McNemar, final accuracy | — | b = 55, c = 0 | **p = 5.55 × 10⁻¹⁷** |
| Benjamini–Hochberg family (5 tests) | — | — | **2 rejections**: q = 1.39 × 10⁻¹⁶, q = 3.08 × 10⁻³² |

**Failure modes (honestly reported):** harmful flips dominate — HFR 11.1% under correct anchors;
all 3 model false negatives persist into the system output (0 recovered, 0 worsened beyond them);
the system introduced 3 new false positives (model FP 7 → system FP 10); anchors were resisted
111× and followed 6×. DR = 1.0 by construction under the deterministic backend (disclosed).

Full numbers: `docs/PHASE2_REPORT.md`, `docs/PHASE3_REPORT.md`, `docs/PHASE4_REPORT.md`.

---

## Quick start

### 1. Backend API (port 8765)

```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765
# health check: curl http://127.0.0.1:8765/api/health
```

### 2. Frontend research UI (port 5199)

```bash
cd frontend
npm install
npm run dev        # http://localhost:5199  (proxies /api → 127.0.0.1:8765)
npm run build      # tsc --noEmit && vite build
```

### 3. Reproduce training (Phase 2)

```bash
python scripts/run_phase2.py
# → outputs/models/lightgbm_appendicitis.joblib (+ calibrator, config, feature order)
# → outputs/metrics/model_comparison.{json,csv}, outputs/figures/*.png
```

### 4. Reproduce agent case runs (Phase 3)

```bash
python scripts/run_phase3.py
# → outputs/audits/<case_id>/  (audit.json + per-stage input/output/meta)
```

### 5. Reproduce experiments (Phase 4)

```bash
python scripts/run_phase4.py anchors
python scripts/run_phase4.py ablations --anchor control
python scripts/run_phase4.py ablations --anchor incorrect_anchor
python scripts/run_phase4.py analyze
# → outputs/metrics/{bias_metrics,final_results,error_analysis}.json, outputs/figures/fig_*.png
```

### 6. Tests

```bash
python -m pytest tests/ -p no:warnings        # 125 passed
```

Everything uses **seed 20261002**. Raw data under `data/raw/` is git-ignored and re-downloadable
from `data/manifests/download_manifest.json`.

---

## Research UI (9 pages)

| Page | Route | What it shows |
|---|---|---|
| Overview | `/` | Service health, headline results, safety framing |
| Model Performance | `/model` | All 6 models with bootstrap CIs, split counts, FN review, figures, global SHAP |
| Case Explorer | `/cases` | 117 test cases; run the real model + calibrator + SHAP + uncertainty on any row |
| Agent Timeline | `/agents` | Run the 8-stage pipeline with ablation/anchor/RAG controls; per-stage visibility badges and JSON output |
| Bias Experiments | `/bias` | 7-condition headline table, single baselines, statistics, figures, quick runner (N ≤ 15) |
| Ablations | `/ablations` | 11-profile matrix (control vs. incorrect anchor) + mechanism interpretation |
| Error Analysis | `/errors` | Recharts taxonomy bar, FN/FP/anchor breakdowns, per-case rows |
| Audit Viewer | `/audits` | Persisted case audits: stage records, totals, exact JSON files |
| Reproduction | `/repro` | Copyable commands + IMPLEMENTED / VALIDATED / NOT VALIDATED tri-column |

Responsive (verified at 320 px — no horizontal overflow), dark/light theme with system-preference
follow, explicit loading/error/missing-artifact states everywhere — **no fake metrics are ever
rendered**; unavailable artifacts show the command that generates them.

The legacy static dashboard from commit `39ded12` remains untouched under `dashboard/`.

---

## Repository map

```
config/            pre-registered model / agent / experiment / dataset configs
knowledge/         pediatric safety rules (fail-closed, verified status)
data/              manifests + interim splits (raw data git-ignored, re-downloadable)
src/
  preprocessing/   audited Regensburg loader, leakage control, feature manifest
  features/        machine-readable feature manifest generator
  data/            stratified patient-level splitting (seed 20261002)
  models/          LightGBM, baselines, prediction interface, inference, uncertainty
  calibration/     Platt vs. isotonic comparison, persisted calibrator
  explainability/  SHAP engine (global, beeswarm, waterfall, dependence)
  evaluation/      model metrics, error analysis, publication figures
  agents/          provider (deterministic/openai-compatible/ollama) + 5 agents + schemas
  rag/             indexer, hybrid TF-IDF + FTS5 retriever, store, schema
  orchestration/   state machine, stages, engine, persistence
  bias/            anchor generator, bias metrics + statistics, audit
scripts/           run_phase2.py / run_phase3.py / run_phase4.py
backend/           FastAPI app + /api endpoints + pydantic schemas
frontend/          Vite + React 18 + Tailwind research UI (9 pages)
dashboard/         legacy static dashboard (untouched)
tests/             125 tests: preprocessing, models, agents, RAG, bias, API
docs/              architecture, dataset audit/card, model card, safety, phase reports
outputs/           metrics, figures, predictions, audits (models/ git-ignored)
research/          dataset inventory, 46-entry verified literature review, gap analysis
```

---

## Implementation status (honest)

**IMPLEMENTED and exercised in this repository**

- Full Phase 2 pipeline: preprocessing, leakage exclusion log, feature manifest, patient-level
  split, LightGBM + 4 baselines, hyperparameter search (validation only), isotonic calibration,
  uncertainty, SHAP, 2000-bootstrap evaluation, all required figures.
- Five-agent engine with 8-stage state machine, config-driven information isolation, schema
  validation, watchdog rules, RAG with mandatory provenance, per-stage audit persistence.
- 7 anchor conditions, 11 ablation profiles, CBR/AOR/BCR/HFR/CRR/CMR/DR/DDR with bootstrap CIs,
  exact McNemar, paired bootstrap, Benjamini–Hochberg, error taxonomy, publication figures.
- FastAPI backend + responsive React UI, 125 passing tests.

**EXPERIMENTALLY VALIDATED here**

- Model metrics on the Regensburg test partition (n = 117).
- All bias experiments under the **deterministic** LLM backend, test partition, seed 20261002.

**NOT VALIDATED / limitations**

- Live-LLM agent conditions — **no API key available in this environment**; deterministic
  provider results must not be read as LLM behaviour.
- Isolation profiles A ≡ C under the deterministic backend (identical by construction; disclosed
  in `docs/PHASE4_REPORT.md`).
- Single dataset (Regensburg, n = 780), retrospective, English + German labels; no external
  validation cohort.
- Determinism ⇒ DR = 1.0 (stability is uninformative under a non-sampling backend).
- UI quick experiments (N ≤ 15) are illustrative; published numbers come from the full
  117 × 3 runs only.

---

## Datasets

Primary: **Regensburg Pediatric Appendicitis dataset** (780 rows, target `Diagnosis`
{appendicitis, no appendicitis}), used under its research license — see
[docs/DATASET_AUDIT.md](docs/DATASET_AUDIT.md) (conditional PASS) and
[docs/DATASET_CARD.md](docs/DATASET_CARD.md). No real patient data outside legally permitted
research datasets is used anywhere in this repository.

Prohibited leakage variables excluded by design and logged: `Diagnosis_Presumptive`,
`Management`, `Severity`, `Length_of_Stay`, `US_Number`.

## License & ethics notes

- Code: see repository license. Dataset: original UCI/Zenodo terms apply.
- Literature: 46 entries, every one DOI- or publisher-URL-verified →
  [research/literature_review.csv](research/literature_review.csv).
- Reporting checklists: TRIPOD+AI / STARD-AI / DECIDE-AI / TRIPOD-LLM →
  [docs/REPORTING_CHECKLIST.md](docs/REPORTING_CHECKLIST.md).
- Safety framing: [docs/SAFETY.md](docs/SAFETY.md), [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

**Research prototype — not a medical device. Not for clinical use.**
