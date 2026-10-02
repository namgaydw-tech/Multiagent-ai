# PHASE 3 REPORT — Multi-Agent Debiasing Engine + RAG + Safety Orchestration

> **Status: IMPLEMENTED and EXPERIMENTALLY VALIDATED (deterministic backend).**
> All numbers below were produced by actually running the code in this repository
> (`python scripts/run_phase3.py`, `python -m pytest tests/`). Nothing is projected,
> simulated or copied from documentation.
>
> Research prototype — **not a medical device**, not for clinical use.

---

## 1. LLM backend used (honest statement)

`LLM_BACKEND` is **not set** in this environment and no LLM API is configured.
Every stage therefore ran on the deterministic rule-based backend
(`deterministic-rules-v1 (no LLM inference)`), which is a first-class backend of
`src/agents/provider.py`, not a mock: stage payloads are computed from the real
patient record, knowledge files and Layer-A model artifacts by documented
functions, then validated against the same Pydantic schemas an LLM would have
to satisfy.

Consequences, stated plainly:

- The real-LLM code paths (OpenAI-compatible / Ollama HTTP calls, retries,
  validation-error re-prompting, token accounting) are **IMPLEMENTED and unit
  tested for failure behavior** (fallback + loud failure) but were **not
  exercised against a live model** — no API key exists here. See
  `test_provider_llm_failure_falls_back_to_deterministic_builder`.
- With the deterministic backend, visibility ablations that only change an
  LLM's *prompt context* (e.g. A vs C for the watchdog) change the recorded
  inputs and `prediction_seen` flag but not the computed red-flag rules —
  those differences become measurable only with a real LLM. This is documented
  here rather than disguised as a measured effect.
- Provider provenance (`backend`, `model_version`, `prompt_hash`, tokens,
  latency, attempts, `fallback_reason`) is recorded per stage regardless of
  backend, so switching `LLM_BACKEND` requires no code change.

## 2. Files created (Phase 3)

| Area | Files |
|---|---|
| Agents (5 + contract) | `src/agents/provider.py`, `schemas.py`, `data_cleanser.py`, `independent_differential.py`, `proponent.py`, `opponent.py`, `watchdog.py`, `arbitrator.py` |
| RAG | `src/rag/schema.py`, `store.py`, `indexer.py`, `retriever.py` |
| Orchestration | `src/orchestration/state.py`, `stages.py`, `engine.py`, `persistence.py` |
| Layer-A bridge | `src/models/inference.py` (single-case model + calibrator + SHAP + uncertainty) |
| Bias audit | `src/bias/audit.py` |
| Runners / tests / docs | `scripts/run_phase3.py`, `tests/test_phase3.py` (28 tests), this report |

Pre-existing from Phase 1–2 and reused unchanged: `config/agents.yaml`,
`knowledge/*.yaml`, `research/literature_review.csv`, Phase 2 model artifacts.

## 3. Agents implemented

| # | Stage | Agent | Role | Validation |
|---|---|---|---|---|
| 1 | 1_data_cleansing | Data cleanser | objective extraction, missing-data report, data-quality warnings, objective red flags with citations; **never diagnoses** (`diagnoses_made: Literal[False]` guard) | `DataCleanserOutput` |
| 2 | 2_independent_differential | Independent differential | 8-candidate differential from raw evidence only; **`anchor_seen: Literal[False]`** hard guard | `IndependentDifferentialOutput` |
| 3 | 3_proponent | Proponent | strongest evidence-based case FOR the model's top class; must acknowledge contradictions and missing findings; SHAP cited with mandated non-causal wording | `ProponentOutput` |
| 4 | 4_opponent | Opponent | two-stage: independent alternatives first, then cited challenges of the revealed leading hypothesis; anchoring-risk rating; disconfirmatory test menu | `OpponentOutput` |
| 5 | 5_high_acuity_watchdog | Watchdog | rule-first safety screen over `knowledge/*.yaml`; prediction-blind by default | `WatchdogOutput` |
| 6 | 6_evidence_retrieval | RAG stage | offline hybrid retrieval with provenance | `EvidenceRetrievalOutput` |
| 7 | 7_arbitration | Arbitrator (Agent 5) | weighted blend (0.45 evidence agreement + 0.25 model support + 0.15 completeness + 0.10 contradiction + 0.05 guideline) × risk × uncertainty multipliers; explicit `confidence_derivation`; never copies the model probability | `ArbitratorOutput` |
| 8 | 8_bias_audit | Bias audit | before/after diagnosis, anchor-followed, beneficial/harmful change, contradiction handling (stage 8 is the only stage that sees ground truth) | `BiasAuditOutput` |

## 4. Orchestration status — state machine + information isolation

- Deterministic transition order 1→8 (`STAGE_ORDER`), no branching on model output.
- **Isolation is data flow, not prompt wording**: each stage receives only
  `CaseState.stage_slice(n)`; the record is stripped of
  `Diagnosis / Diagnosis_Presumptive / Management / Severity / Length_of_Stay / US_Number`
  before the run starts.
- Verified by tests (all pass):
  - stages 1/2/5/6 receive no `model_output`, no `anchor`, no `ground_truth`;
  - stage 2 is prediction- and anchor-blind **in every named ablation**
    (`VisibilityPolicy.from_ablation` test loop);
  - deliberately corrupting the policy so stage 2 would see the prediction makes
    the engine raise `StageError` and record the error in the audit — it does not run;
  - ablation A exposes the prediction to the watchdog (`prediction_seen: true`),
    ablation B hides it from opponent/watchdog, `all_see_anchor` never un-blinds stage 2.
- **Resumable**: every stage is written to disk before the next starts;
  `resume=True` reloads + re-validates all completed stages and executes nothing
  (test asserts the provider log stays empty on resume).
- Fail-loudly: schema violation or stage exception → `StageError`, error persisted
  in `audit.json`; missing fields are never invented.

## 5. RAG status

- Index: **72 chunks** in `data/interim/rag_index.sqlite` (gitignored, rebuilt by
  `python -m src.rag.indexer`): 46 literature rows (DOI-verified metadata),
  14 knowledge rules, 12 source-manifest entries.
- Retrieval: deterministic offline hybrid — TF-IDF word 1–2 gram cosine +
  SQLite FTS5 keyword fusion (0.55 keyword-only weight, documented). No model
  downloads, fully reproducible (test asserts identical results across runs).
- Provenance: `SourceChunk` **rejects any chunk without DOI, URL or citation** at
  storage time; every hit projected into `RetrievedSource` carries
  title/year/DOI-or-URL/score/method.
- Honesty: empty query → zero hits; with `rag.enabled=false` stage 6 returns an
  empty `retrieved` list with an explicit note — it never fabricates citations.
- Used by: opponent at stage 4 (pre-retrieval, cited challenges) and arbitrator
  at stage 6 (guideline support term).

## 6. Safety-rule status (watchdog, actually executed)

From the real run of row 746 (`outputs/audits/regensburg_row_746/stage_05_output.json`):

- `risk_level: MODERATE` with evidence: `Peritonitis=local` → "focal peritonitis —
  urgent surgical review"; perforation/abscess correctly reported **absent** with
  the record values that prove it (`Perforation=no`, `Appendicular_Abscess=no`).
- **Fail-closed refusals**: `hypotension_sbp_by_age`, `tachycardia_by_age`,
  `phoenix_sepsis_2024` all refused with "pending_full_text_transcription — fail closed".
- **Cannot-assess** (dataset has no such field, never invented): petechial rash,
  meningeal signs, IMCI danger signs, Kawasaki features; HR/RR/BP/SpO2 absent →
  sepsis scoring refused and escalated as "urgent clinical rule-out required".
- `prediction_seen: false` under the default condition; passing the prediction
  changes nothing in the computed assessment (test-verified).
- HIGH risk is granted **only** when a complication flag is present in the record
  (test finds such rows dynamically: `Perforation=yes` / `Peritonitis=generalized`
  → HIGH; benign rows stay LOW/MODERATE — never alarmist without evidence).

## 7. Executed cases (seed 20261002, default condition C, RAG on)

`python scripts/run_phase3.py` — 5 real Regensburg rows, **0 failures**
(full data in `outputs/metadata/phase3_case_runs.json`):

| Row | GT | P(calibrated) | Uncertainty | Model top | Final working dx (stage 7) | Confidence | Changed? | Watchdog | Retrieved | Contra-evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 469 | no appendicitis | 0.1667 | MODERATE | no appendicitis | **Acute appendicitis** | 0.5627 | **Yes → harmful** | LOW | 4 | 5 |
| 217 | no appendicitis | 0.0000 | LOW | no appendicitis | no appendicitis | 0.6710 | No | LOW | 4 | 5 |
| 746 | appendicitis | 1.0000 | LOW | appendicitis | appendicitis | 0.7062 | No | MODERATE | 4 | 2 |
| 393 | appendicitis | 0.6430 | **HIGH** | appendicitis | appendicitis | 0.4451 | No | LOW | 4 | 6 |
| 717 | appendicitis | 0.0673 | LOW | no appendicitis | no appendicitis | 0.6052 | No (miss retained) | MODERATE | 4 | 5 |

Honest observations (failure modes, not hidden):

1. **Row 469 — harmful flip**: the debate overturned a correct model decision.
   Stage 8 records `diagnosis_changed=true`, `change_harmful=true`,
   `model_vs_final_disagreement=true` with the note *"debate overturned an
   initially correct decision — safety review required"*. This is a genuine
   observed failure mode and is a target of the Phase 4 error analysis.
2. **Row 717 — uncorrected false negative**: the model is confidently wrong
   (P=0.067, LOW uncertainty) and the debate did not revise it. The watchdog
   still escalated MODERATE risk (focal peritonitis recorded) — the safety net
   flagged the case even though the final working diagnosis stayed wrong.
3. **Row 393 — uncertainty tempering works as designed**: HIGH predictive
   uncertainty pulls multi-agent confidence (0.4451) well below the calibrated
   probability (0.6430), per the documented formula.
4. With the deterministic backend all outputs are exactly reproducible:
   re-running the batch produces byte-identical stage payloads.

## 8. Example audit location

**`outputs/audits/regensburg_row_469/`** (committed as the example — see `.gitignore`
exception; 25 files, ~74 KB). Every case run writes the same structure:

```
outputs/audits/<case_id>/
  stage_01_input.json … stage_08_input.json    # truthful inputs incl. visibility + withheld lists
  stage_01_output.json … stage_08_output.json  # schema-validated payloads
  stage_01_meta.json  … stage_08_meta.json     # prompt_hash, backend, model_version, tokens, latency, attempts
  audit.json                                     # case-level trail (project brief §19)
```

`audit.json` contains: case_id, config_hash (`cb7198448d25`), seed, ablation,
provider backend/model version, per-stage visibility policy, and per-stage
status/prompt_hash/tokens/latency/attempts/fallback_reason/schema/files.

## 9. Phase 3 gate checklist

- [x] Five agents implemented with strict Pydantic schemas — **and runnable**
- [x] Information isolation verified (unit tests + fail-loud engine guard)
- [x] RAG working (72 chunks, hybrid retrieval, provenance enforced, tested)
- [x] Watchdog rules functioning on real records (fail-closed + cannot-assess + tiers)
- [x] State machine working (8/8 stages, resumable, per-stage persistence)
- [x] Audit files generated (`outputs/audits/<case_id>/`, example committed)
- [x] End-to-end runs on real dataset cases (5 cases, 0 failures)
- [x] Tests: **28 new Phase 3 tests; 86/86 total pass**
  (`python -m pytest tests/ -p no:warnings` → `86 passed`)
- [ ] Live-LLM execution — **blocked: no LLM backend/key in this environment**
      (documented in §1; code path implemented + failure-tested)

## 10. Reproduction commands

```bash
python -m src.rag.indexer                 # rebuild the RAG index (72 chunks)
python scripts/run_phase3.py              # 5 real cases → outputs/audits/ + metadata
python scripts/run_phase3.py --rows 469 --ablation A_all_see_prediction
python scripts/run_phase3.py --no-rag     # RAG-disabled condition (honest empty stage 6)
python -m pytest tests/ -p no:warnings    # 86 passed
```

## 11. Known limitations

1. Deterministic backend only (§1) — real-LLM behavior unmeasured here.
2. Offline TF-IDF retrieval instead of neural embeddings (deliberate: no model
   downloads, reproducible; upgrade path documented in `src/rag/retriever.py`).
3. Only 5 cases executed in this phase (full N-case experiments are Phase 4).
4. Two observed disagreement failure modes (§7) are reported, not tuned away —
   adjusting the arbitrator against held-out labels would be test-set fitting.
5. `outputs/audits/*` stays gitignored except the one committed example.
