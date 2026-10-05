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

## Status: Phases 1–4 complete, UI complete, Phase 5 (multi-dataset/multi-algorithm) executed for 2 tabular datasets

| Phase | What ran | Status |
|---|---|---|
| Phase 1 | Dataset audit, literature review (46 verified entries), configs, safety knowledge | **COMPLETED** (commit `47ca7d9`) |
| Phase 2 | LightGBM training + calibration + SHAP + evaluation on Regensburg | **EXPERIMENTALLY VALIDATED** (commit `e57cdd7`) |
| Phase 3 | Five-agent debiasing engine + RAG + safety orchestration | **IMPLEMENTED + EXECUTED** (commit `2173778`) |
| Phase 4 | Anchor experiments, 11 ablations, statistics, error analysis, figures | **EXPERIMENTALLY VALIDATED** (commit `7568b08`) |
| UI | FastAPI backend + React research UI (12 pages, responsive, dark/light) | **IMPLEMENTED + VERIFIED IN BROWSER** |
| **Phase 5** | Dataset blocker (27 checks × 7 datasets), 17-algorithm model zoo, Macro F0.5 + safety floor, calibration, ensembles, sealed-test evaluation of 2 tabular datasets, 35/35 figures, statistics, scorecard, cross-backbone bias runs, 3 new UI pages, cross-domain agent packs | **EXECUTED (tabular)** — images NOT_EXECUTED (torch), credentialed datasets NOT_AVAILABLE, blocked dataset locked; see *Phase 5* section |

Phase 1–4 state above is what existed **before** Phase 5 began (baseline commit `7f7e04f`,
`origin/main`); everything Phase 5 added is described in the *Phase 5* section below.

Test suite: **174 passed** (`python -m pytest tests/ -p no:warnings`) — 125 historical
Phases 1–4 tests + 49 Phase 5 tests, exit 0.
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

### Phase 5 — decision metrics: Macro F0.5 (`src/evaluation/phase5_metrics.py`)

Macro F0.5 is **not** accuracy: it is the unweighted mean of the per-class F0.5 scores and
weights precision twice as heavily as recall (β = 0.5 → precision-focused), while the sensitivity
floor keeps recall safety-critical.

| Metric | Formula |
|---|---|
| Precision P | TP / (TP + FP) |
| Sensitivity / Recall / TPR | TP / (TP + FN) |
| Specificity / TNR | TN / (TN + FP) |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| NPV | TN / (TN + FN) |
| False Positive Rate | FP / (FP + TN) |
| False Negative Rate | FN / (FN + TP) |
| F1 | 2·P·R / (P + R) |
| F_β | (1 + β²)·P·R / (β²·P + R) |
| **F0.5** (β = 0.5) | 1.25·P·R / (0.25·P + R) |
| **Macro F0.5** | (1/K) · Σ_{k=1..K} F0.5_k  — unweighted mean over the K classes |
| Balanced accuracy | (Sensitivity + Specificity) / 2 |
| MCC | (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) |
| Brier score | (1/N) · Σ_i (p_i − y_i)² |

Cross-checked against `sklearn.metrics.fbeta_score(beta=0.5, average="macro", zero_division=0)`
for both binary and multiclass cases (unit test in `tests/test_phase5.py`). Undefined metrics
return **null / n/a — never 0** (e.g. hard-voting ensembles have no probabilities ⇒ AUROC,
AUPRC, Brier, ECE are null).

**Safety-constrained selection** (`src/evaluation/threshold_policy.py`) — model choice is never
"highest accuracy":

```
t* = argmax_t  MacroF0.5(validation, t)   subject to   Sens(validation, t) ≥ 0.90
    if no t satisfies the floor:  t* = argmax_t Sens(validation, t)
                                  (Macro F0.5 as secondary ranking)
```

Fitted on the **validation partition only**, then locked; the sealed test is opened exactly once
(`metrics/test_evaluation_lock.json`). Winner selection also uses validation only
(floor-compliant → rank by validation Macro F0.5 → AUPRC/calibration as tie-breakers → prefer
simpler models when statistically indistinguishable).

**Calibration** (`src/calibration/phase5.py`): raw vs. Platt vs. isotonic on validation only,
selected by Brier + ECE; multiclass uses one-vs-rest sigmoid calibrators.

- Platt: `p̂ = 1 / (1 + e^{−(A·logit(p) + B)})`
- Isotonic: non-decreasing step fit `p̂ = f_iso(p)` minimizing squared error on validation.

### Phase 5 — algorithm formulas (`src/models/model_registry.py`, `config/model_zoo.yaml`)

**Logistic Regression** (linear classification model; no Linear Regression anywhere as a
classifier)

```
z = β0 + β1·x1 + … + βn·xn
P(y = 1 | x) = 1 / (1 + e^{−z})
Odds ratio:   OR_j = e^{β_j}
Loss:  min −Σ [ y_i·log p_i + (1−y_i)·log(1−p_i) ] + λ·‖β‖²
```

**Decision Tree** (CART)

```
Gini(S)   = 1 − Σ_k p_k²
Entropy   H(S) = −Σ_k p_k·log2(p_k)
Information Gain:
IG = H(parent) − Σ_children (n_child / n_parent)·H(child)
Prediction: ŷ = argmax_k p_k at the leaf
```

**Random Forest** (B bootstrap-sampled trees, majority/mean aggregation)

```
ŷ        = mode{ h1(x), …, hB(x) }
P(y = c | x) = (1/B) · Σ_b P_b(y = c | x)
```

**Extra Trees** — same aggregation, but splits are chosen from the *full* training set with
randomly drawn thresholds (no bootstrap), i.e. maximal randomization at each node.

**K-Nearest Neighbors** (scaled with TRAIN-fitted scaler)

```
d(x, z) = sqrt( Σ_i (x_i − z_i)² )          # Euclidean
P(y = 1 | x) = (1/k) · Σ_{i ∈ N_k(x)} 1[y_i = 1]
```

**Gaussian Naive Bayes**

```
P(C | X) = P(X | C)·P(C) / P(X)
P(X | C) = Π_j  N(x_j ; μ_{C,j}, σ²_{C,j})
```

**LDA / QDA** (class centroids with shared / class-specific covariance)

```
LDA:  δ_k(x) = xᵀΣ⁻¹μ_k − ½·μ_kᵀΣ⁻¹μ_k + log π_k
QDA:  δ_k(x) = −½·log|Σ_k| − ½·(x−μ_k)ᵀΣ_k⁻¹(x−μ_k) + log π_k
ŷ = argmax_k δ_k(x)
```

**Support Vector Machine** (linear and RBF kernels; probability via validation-fitted calibration)

```
f(x) = wᵀx + b
min  ½·‖w‖² + C·Σ_i ξ_i
RBF kernel:  K(x_i, x_j) = exp( −γ·‖x_i − x_j‖² )
```

**Gradient Boosting / HistGradientBoosting** (additive stage-wise fitting of negative gradients)

```
F_m(x) = F_{m−1}(x) + η·h_m(x)
h_m fits the negative gradients:  −∂L(y_i, F(x_i)) / ∂F(x_i)
```

**AdaBoost** (exponential loss, reweighting)

```
F(x) = Σ_t α_t·h_t(x),    α_t = ½·ln( (1 − ε_t) / ε_t )
w_i ← w_i·exp(α_t)  if misclassified,  exp(−α_t) otherwise
```

**XGBoost** (regularized objective)

```
Obj = Σ_i L(y_i, ŷ_i) + Σ_k Ω(f_k),    Ω(f) = γ·T + ½·λ·‖w‖²
```

**LightGBM** — gradient boosting with leaf-wise (best-first) growth and histogram binning;
**CatBoost** — gradient boosting with ordered statistics (permutation-based target encoding for
categoricals) and native categorical handling; both follow `F_m = F_{m−1} + η·h_m`.

**Multi-Layer Perceptron**

```
z^(l) = W^(l)·a^(l−1) + b^(l)
a^(l) = φ(z^(l))                 # ReLU hidden, softmax output
backprop:  ∂L/∂W^(l) via the chain rule
```

**Soft Voting ensemble** (uniform and validation-weighted variants)

```
P_ensemble(y = c | x) = Σ_m w_m·P_m(y = c | x),    Σ_m w_m = 1
validation-weighted:  w_m ∝ MacroF0.5_val(model m) among floor-satisfying models
```

**Hard Voting**: `ŷ = mode of the per-model predicted labels` (no probabilities ⇒ probability
metrics reported as null).

**Stacking**: a logistic-regression meta-learner `ŷ = g(P_1(x), …, P_M(x))` fitted on
**out-of-fold validation** probabilities only — never on test data.

**SHAP** (feature attribution, global and per-case)

```
f(x) = φ0 + Σ_i φ_i
```

SHAP values are **attributions, never causal effects** — wording enforced in code and UI:
"feature X contributed to this model prediction", never "caused the diagnosis". The same rule
applies to permutation importance and Grad-CAM.

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

## Phase 5 — Multi-Dataset, Multi-Algorithm External Validation & Robustness (added now)

**Research question preserved:** *Does adversarial, information-isolated multi-agent diagnostic
reasoning reduce confirmation bias consistently across distinct pediatric diagnostic datasets,
modalities, diseases and underlying ML classifiers?* The expanded classifier benchmark exists to
test whether the Phase 3/4 debiasing finding depends on a particular underlying model — it is
**not** a "which classifier wins" project.

Scientific rules enforced in code (not by convention): unrelated diseases are **never
concatenated** into one training table; website mirrors are **never** counted as independent
cohorts (Kermany on Mendeley = the Kaggle mirror = ONE dataset); test data is never used for
fitting preprocessing, tuning, calibration, threshold selection or ensemble weighting; splits are
patient/group-level wherever identifiers exist.

### What already existed vs. what Phase 5 added

| Area | Before Phase 5 (baseline `7f7e04f`) | After Phase 5 (this working tree) |
|---|---|---|
| Datasets | 1 dataset (Regensburg tabular, Phase 2 split) | 7 candidate datasets audited; 2 executed as separate experiments |
| Dataset blocker | none (docs-level audit only) | `src/data/dataset_blocker.py`: 27 pre-training checks → PASS / CONDITIONAL / BLOCKED / NOT_AVAILABLE, hard gate `assert_trainable()` blocks the trainer |
| Models | LightGBM + 4 baselines (Phase 2) | **17 tabular algorithms + 4 ensembles** (`config/model_zoo.yaml`, `src/models/model_registry.py`) |
| Selection metric | F2 on validation (Phase 2) | **Macro F0.5** with sensitivity floor 0.90, selected on validation only, threshold locked before the sealed test is opened |
| Calibration | binary Platt/isotonic (Phase 2) | + multiclass OvR-sigmoid, Brier+ECE based selection, validation only |
| Statistics | bootstrap CI, McNemar, BH (Phase 4) | + per-model bootstrap CIs (Macro F0.5, sens, spec, AUROC, AUPRC, Brier, ECE, MCC), paired comparisons, DeLong (AUROC) |
| Agents | appendicitis domain only | + **domain packs** (`src/agents/domains.py`): neonatal sepsis, pediatric pneumonia — same isolation policy, no ground truth ever shown to decision agents |
| Test-set protocol | one Phase 2 evaluation (preserved, untouched) | sealed-test lock per Phase 5 dataset (`metrics/test_evaluation_lock.json`, evaluated once after threshold lock) |

### Dataset blocking results (executed — `outputs/phase5/dataset_blocking_report.{json,csv}`)

| Dataset | Status | Reason (abridged; full reasons in `docs/PHASE5_DATASET_AUDIT.md`) |
|---|---|---|
| REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR | **PASS** | reference benchmark; reuses the existing persisted Phase 2 split (no resplit) |
| NEONATAL_SEPSIS_REGISTRY (Mendeley DOI 10.17632/5vdz5cftz7.1) | **CONDITIONAL** | license not exposed by API; post-outcome/treatment columns excluded from predictors; executed as separate experiment |
| REG_ENSBURG_ULTRASOUND_IMAGES (Zenodo) | **CONDITIONAL** | shares patients with the Regensburg cohort — an *internal multimodal extension*, **not** an external population; pending image training |
| KERMANY_PEDIATRIC_PNEUMONIA (Mendeley DOI 10.17632/rscbjbr9sj.2) | **CONDITIONAL** | Kaggle copy treated as a mirror, not a second cohort; pending image training |
| PHYSIONET_PIC (credentialed) | **NOT_AVAILABLE** | requires PhysioNet credentialing — no local data, no fabricated counts |
| PECARN | **NOT_AVAILABLE** | access requires approval/credentials unavailable here |
| CHILD_PNEUMONIA_MENDELEY | **BLOCKED** | pre-augmented image leakage (2,644 `aug_` files), 279 duplicate filenames, no patient grouping — must never reach the trainer |

### Model zoo (17 tabular models + 4 ensemble methods — `config/model_zoo.yaml`)

Baselines: Logistic Regression, Decision Tree, Gaussian NB, KNN, LDA, QDA ·
Tree ensembles: Random Forest, Extra Trees ·
Boosting: Gradient Boosting, HistGradientBoosting, AdaBoost, XGBoost, LightGBM, CatBoost ·
Margin: SVM (linear + RBF) · Neural: MLP ·
Ensembles: soft voting (uniform), **validation-weighted soft voting**, hard voting, stacking.
There is deliberately **no Linear Regression classifier** — Logistic Regression is the linear
model. Scaling is fitted on TRAIN only for LogReg/SVM/KNN/MLP/LDA/QDA; tree models are unscaled;
CatBoost keeps native categorical handling.

### Executed results — sealed test partitions (all values read from `outputs/phase5/*/metrics/metrics_test.json`)

**Regensburg appendicitis (tabular)** — test n = 117 (48 neg / 69 pos), 16/17 models trained
(QDA unavailable: rank-deficient covariance, recorded as UNAVAILABLE, never silently skipped):

| Rank | Model | Macro F0.5 | Sens | Spec | AUROC | AUPRC | Brier | thr |
|---|---|---|---|---|---|---|---|---|
| 1 | **stacking (ensemble)** | **0.941** | 0.971 | 0.896 | 0.966 | 0.973 | 0.050 | 0.35 |
| 2 | catboost | 0.938 | 0.942 | 0.938 | 0.965 | 0.968 | 0.046 | 0.17 |
| 3 | ensemble_soft_uniform | 0.933 | 0.971 | 0.875 | 0.972 | 0.982 | 0.062 | 0.42 |
| 4 | ensemble_hard_voting | 0.931 | 0.957 | 0.896 | n/a (hard labels) | n/a | n/a | 0.5 (maj.) |
| 5 | hist_gradient_boosting | 0.928 | 0.928 | 0.938 | 0.938 | 0.936 | 0.066 | 0.19 |

Full 20-row table incl. all metrics (accuracy, balanced accuracy, NPV, F1, F2, MCC, ECE,
bootstrap 95% CIs): `outputs/phase5/regensburg_appendicitis_tabular/metrics/metrics_test.json`.
Phase 2 headline numbers (LightGBM AUROC 0.9716, CM TN41/FP7/FN3/TP66, threshold 0.34) are
untouched — Phase 5 thresholds differ because Phase 5 optimizes Macro F0.5 with a sensitivity
floor on validation, a different pre-registered policy.

**Neonatal sepsis registry (Mendeley)** — train 1,337 / val 309 / test 300 (test: 279 neg /
21 pos — a genuinely hard, imbalanced task; results honestly reported, no cherry-picking):

| Rank | Model | Macro F0.5 | Sens | Spec | AUROC | AUPRC | thr |
|---|---|---|---|---|---|---|---|
| 1 | adaboost | 0.492 | 0.857 | 0.530 | 0.700 | 0.119 | 0.03 |
| 2 | stacking (ensemble) | 0.491 | 0.857 | 0.527 | 0.703 | 0.223 | 0.06 |
| 3 | gradient_boosting | 0.485 | 0.810 | 0.527 | 0.697 | 0.145 | 0.04 |
| 4 | validation_weighted soft voting | 0.471 | 0.905 | 0.466 | 0.691 | 0.201 | 0.06 |

The sepsis target is weakly separable from the released features (best AUROC ≈ 0.70); this is a
**limitation of the released dataset**, reported as such — not a tuned-away result.

### Winner selection — VALIDATION only, never the test set

`python scripts/run_phase5.py analyze` writes `outputs/phase5/model_scorecard.{csv,json}`
(42 rows = 2 datasets × (17 models + 4 ensembles), the ~28 required fields, unavailable
metrics as null), `outputs/phase5/cross_dataset_summary.{json,csv}` and
`outputs/phase5/external_validation.json`. The winner is locked on **validation**:
floor-compliant first → max validation Macro F0.5 → validation AUPRC → simpler model on ties;
test metrics never enter the selection (each test partition is then opened exactly once,
recorded in `metrics/test_evaluation_lock.json`).

| Dataset | Winner (validation-selected) | val Macro F0.5 | test Macro F0.5 | generalization gap (val − test) |
|---|---|---|---|---|
| Regensburg appendicitis | gradient_boosting | 0.981 | 0.911 | 0.070 |
| Neonatal sepsis | gradient_boosting | 0.500 | 0.485 | 0.015 |

The test-leaderboard tables above are **reporting**, not selection — the stacked ensemble
scores highest on the Regensburg test partition, but it was not the validation winner.

### Statistics — post-hoc on the sealed test (inference only, never selection)

`python scripts/run_phase5.py analyze` → `outputs/phase5/statistical_comparison.{json,csv}`
(116 comparison rows): paired bootstrap Δ Macro F0.5 (winner vs each of 16 competitors per
dataset, B = 2,000), exact McNemar on discordant errors, DeLong AUROC 95% CIs per model,
bootstrap CIs for the ensemble probabilities, and Benjamini–Hochberg q-values per dataset
family. Example (Regensburg): winner vs gaussian_nb is significant after BH
(q < 0.05), winner vs adaboost is not (Δ = −0.008, p = 0.75) — tiny point differences
without CI separation are never called meaningful.

### Figures — 35/35 target visualizations generated from persisted outputs

`src/evaluation/phase5_figures.py` writes 66 PNGs under `outputs/phase5/*/figures/` and
`outputs/phase5/figures/`, plus `outputs/phase5/figures_manifest.json` mapping each of the 35
required figures to its path (or to an explicit NOT_GENERATED + reproduce command). Sealed-test
panels are watermarked "SEALED TEST"; threshold/learning/hyperparameter panels are watermarked
"DEVELOPMENT / VALIDATION". This includes leaderboards (8), ROC/PR, confusion matrices,
reliability curves, the four threshold curves + operating point, learning curves, native +
permutation importance, SHAP summary/waterfall (LightGBM, non-causal wording), RF tree-count and
KNN K curves, SVM kernel/C sweep, boosting comparison, ensemble-vs-individual, training/inference
times, performance-vs-complexity, cross-dataset gaps, CIs for validation-ranked leaders, and the
CBR/BCR/HFR views. Image-model figures (Grad-CAM) remain NOT_GENERATED with the torch install
command — never invented.

### Cross-backbone bias experiments — EXECUTED (the core Phase 5 question)

`python scripts/run_phase5.py bias --dataset REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR
--backbone all --rows 40` runs the **same** 40 test rows × 2 anchor conditions × 2 profiles
(single + full isolated engine) with only the classifier swapped — 6 backbones × 160 case-runs
= 960 runs under a uniform information budget (no SHAP for any backbone; ground truth never
shown to agents). Results in `outputs/phase5/bias_backbones_summary.json`:

| Backbone | engine CBR (incorrect anchor) | single-agent CBR | engine BCR |
|---|---|---|---|
| logistic_regression | 0.025 | 0.550 | 0.950 |
| random_forest | 0.000 | 0.575 | 0.875 |
| xgboost | 0.000 | 0.525 | 0.950 |
| lightgbm | 0.025 | 0.525 | 0.950 |
| catboost | 0.000 | 0.475 | 0.975 |
| ensemble | 0.025 | 0.550 | 0.950 |

**H5-c holds on this subset:** confirmation-bias rate stays ≤ 0.025 for the isolated engine
across all six classifiers while the single-agent baseline sits at 0.475–0.575 — the debiasing
finding is not an artifact of LightGBM. HFR is undefined (excluded) on this subset, so HFR
claims come only from the Phase 4 headline experiments.

### Phase 5 API + UI

Backend: `/api/phase5/{status,datasets,scorecard,cross-dataset,external-validation,
statistics,figures,figures/{id},algorithms,dataset/{id}/metrics}` — every endpoint serves the
persisted JSON or an explicit `available: false` + reproduce hint (never placeholder numbers).
Frontend: three new pages — **Dataset Registry** (7 datasets, statuses + exact reasons),
**Cross-Dataset Benchmark** (scorecard with n/a for unavailable metrics, gaps, paired
statistics), **Algorithm Lab** (dataset × algorithm × metric selectors, formula display,
hyperparameters, locked threshold, val vs sealed-test table with CIs, figure selector with
NOT GENERATED fallback + reproduce command).

### Not executed yet (reported honestly, never fabricated)

- **Image experiments** (Kermany CXR, Regensburg ultrasound, Grad-CAM, late fusion): data
  extracted/downloaded, but `torch`/`torchvision` are not installed in this environment →
  **NOT_EXECUTED** until the install succeeds.
- **PHYSIONET_PIC / PECARN**: NOT_AVAILABLE (credentials) — configuration only, no fake counts.
- **CHILD_PNEUMONIA_MENDELEY**: BLOCKED — will not enter training until its blocker is resolved.
- **Multi-agent runner on the sepsis domain** (beyond domain packs): PLANNED — the CLI exits
  with an explicit NOT_EXECUTED instead of inventing results.

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

### 6. Phase 5 (multi-dataset, multi-algorithm)

```bash
python scripts/run_phase5.py audit                    # dataset blocker -> outputs/phase5/dataset_blocking_report.{json,csv} + docs/PHASE5_DATASET_AUDIT.md
python scripts/run_phase5.py train    --dataset REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR
python scripts/run_phase5.py evaluate --dataset REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR   # sealed test, once, after threshold lock
python scripts/run_phase5.py train    --dataset NEONATAL_SEPSIS_REGISTRY
python scripts/run_phase5.py evaluate --dataset NEONATAL_SEPSIS_REGISTRY
python scripts/run_phase5.py agents   --dataset <dataset>   # multi-agent runs with domain pack
python scripts/run_phase5.py bias     --dataset <dataset>   # anchor/bias experiments per backbone
python scripts/run_phase5.py bias     --dataset REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR --backbone all --rows 40   # cross-backbone CBR/BCR
python scripts/run_phase5.py analyze                      # scorecard + statistics + 35 figures
python scripts/run_phase5.py all                          # everything above in order
# -> outputs/phase5/<dataset>/{splits,calibration,models,predictions,metrics,figures}/*
```

### 7. Tests

```bash
python -m pytest tests/ -p no:warnings        # 174 passed = 125 historical + 49 Phase 5
```

Everything uses **seed 20261002**. Raw data under `data/raw/` is git-ignored and re-downloadable
from `data/manifests/download_manifest.json`.

---

## Research UI (12 pages)

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
| Dataset Registry | `/datasets` | 7 Phase 5 candidates: PASS / CONDITIONAL / BLOCKED / NOT_AVAILABLE + exact reasons |
| Cross-Dataset Benchmark | `/benchmark` | scorecard (n/a for unavailable metrics, never 0), val→test gaps, paired statistics |
| Algorithm Lab | `/lab` | dataset × algorithm × metric selectors, formulas, locked threshold, val vs test + CIs, figure viewer with NOT GENERATED fallback |

Responsive (verified at 320 px — no horizontal overflow), dark/light theme with system-preference
follow, explicit loading/error/missing-artifact states everywhere — **no fake metrics are ever
rendered**; unavailable artifacts show the command that generates them.

The legacy static dashboard from commit `39ded12` remains untouched under `dashboard/`.

---

## Repository map

```
config/            pre-registered model / agent / experiment / dataset configs
                   + phase5.yaml, dataset_blocking.yaml, model_zoo.yaml (Phase 5)
knowledge/         pediatric safety rules (fail-closed, verified status)
data/              manifests + interim splits (raw data git-ignored, re-downloadable)
src/
  preprocessing/   audited Regensburg loader, leakage control, feature manifest
  features/        machine-readable feature manifest generator
  data/            stratified patient-level splitting (seed 20261002)
                   + dataset_blocker.py, phase5_datasets.py (Phase 5 adapters)
  models/          LightGBM, baselines, prediction interface, inference, uncertainty
                   + model_registry.py (17-algorithm zoo), phase5_pipeline.py (Phase 5)
  calibration/     Platt vs. isotonic comparison, persisted calibrator + phase5.py
  explainability/  SHAP engine (global, beeswarm, waterfall, dependence)
  evaluation/      model metrics, error analysis, publication figures
                   + phase5_metrics.py (Macro F0.5, DeLong, BH), threshold_policy.py
                   + scorecard.py (winner selection), phase5_stats.py, phase5_figures.py
  agents/          provider + 5 agents + schemas + domains.py (domain packs, Phase 5)
  rag/             indexer, hybrid TF-IDF + FTS5 retriever, store, schema
  orchestration/   state machine, stages, engine, persistence (domain-aware, Phase 5)
  bias/            anchor generator, bias metrics + statistics, audit
                   + backbones.py (cross-backbone CBR/BCR experiments, Phase 5)
scripts/           run_phase2.py / run_phase3.py / run_phase4.py / run_phase5.py
backend/           FastAPI app + /api endpoints (incl. api/phase5.py) + schemas
frontend/          Vite + React 18 + Tailwind research UI (12 pages)
dashboard/         legacy static dashboard (untouched)
tests/             174 tests (125 historical + 49 Phase 5: blocker, floor policy,
                   scorecard nulls, figures manifest, API, CLI, cross-backbone)
docs/              architecture, dataset audit/card, model card, safety, phase reports
                   + PHASE5_DATASET_AUDIT.md (auto-generated by the blocker)
                   + PHASE5_PLAN/REPORT, ML_ALGORITHMS_AND_MATHEMATICS, EXTERNAL_VALIDATION
outputs/           metrics, figures, predictions, audits (models/ git-ignored)
                   + phase5/<dataset>/{splits,calibration,models,predictions,metrics}
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
- **Phase 5 (this working tree, added now):** 27-check dataset blocker with hard trainer gate
  and persisted reports; 7-dataset audit (1 PASS, 3 CONDITIONAL, 1 BLOCKED, 2 NOT_AVAILABLE);
  17-algorithm tabular zoo + 4 ensembles; Macro F0.5 with validation-only sensitivity-floor
  threshold policy; binary + multiclass calibration; patient/group-aware adapters for the
  neonatal sepsis registry (Mendeley) and Kermany/ultrasound inventories; sealed-test
  evaluation executed for Regensburg tabular (16 models) and neonatal sepsis (16 models);
  cross-domain agent packs (appendicitis / neonatal sepsis / pediatric pneumonia) with the
  original isolation policy unchanged.

**EXPERIMENTALLY VALIDATED here**

- Model metrics on the Regensburg test partition (n = 117).
- All bias experiments under the **deterministic** LLM backend, test partition, seed 20261002.
- Phase 5: sealed-test metrics for 20 entries (16 models + 4 ensembles) on Regensburg tabular
  and neonatal sepsis, each with bootstrap 95% CIs, persisted under `outputs/phase5/*/metrics/`.

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

**Phase 5 candidates** (every one gated by the 27-check blocker, statuses and exact reasons in
[docs/PHASE5_DATASET_AUDIT.md](docs/PHASE5_DATASET_AUDIT.md)): neonatal sepsis registry
(Mendeley, CONDITIONAL, executed as a separate experiment), Regensburg ultrasound images
(CONDITIONAL — same cohort, internal extension, **not** an external population), Kermany pediatric
pneumonia (Mendeley, CONDITIONAL, mirror deduplicated), PhysioNet PIC + PECARN (NOT_AVAILABLE —
credentialed), child pneumonia Mendeley (BLOCKED — pre-augmented leakage). Datasets stay
separate experiments; mirrors are never counted as independent cohorts; blocked datasets cannot
reach the trainer.

## License & ethics notes

- Code: see repository license. Dataset: original UCI/Zenodo terms apply.
- Literature: 46 entries, every one DOI- or publisher-URL-verified →
  [research/literature_review.csv](research/literature_review.csv).
- Reporting checklists: TRIPOD+AI / STARD-AI / DECIDE-AI / TRIPOD-LLM →
  [docs/REPORTING_CHECKLIST.md](docs/REPORTING_CHECKLIST.md).
- Safety framing: [docs/SAFETY.md](docs/SAFETY.md), [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

**Research prototype — not a medical device. Not for clinical use.**
