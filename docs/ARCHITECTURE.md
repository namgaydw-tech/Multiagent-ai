# Architecture

> Status: **design + Phase 1 infrastructure**. The pipeline described here is the target
> architecture; only the data/audit layer is implemented and verified so far.

## Layer separation

```
┌──────────────────────────────────────────────────────────────────────────┐
│ LAYER A — disease-specific statistical prediction (LightGBM et al.)      │
│   raw tabular data → preprocessing → calibrated probabilities + SHAP    │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │  normalized prediction interface (JSON schema)
┌───────────────────────────────▼──────────────────────────────────────────┐
│ NORMALIZED PREDICTION INTERFACE                                          │
│   {model_name, clinical_domain, target_classes, class_probabilities,     │
│    predicted_class, calibrated_probability, uncertainty,                  │
│    important_features, model_version}                                     │
└───────────────────────────────┬──────────────────────────────────────────┘
                                │
┌───────────────────────────────▼──────────────────────────────────────────┐
│ COMMON MULTI-AGENT DEBIASING ENGINE (identical for every domain)         │
│   staged state machine with information isolation + per-stage audit      │
└──────────────────────────────────────────────────────────────────────────┘
```

An optional Layer B (image branch: pretrained CNN → probability/embedding → fusion) joins at the
normalized interface; images never enter LightGBM directly.

## Staged execution (information isolation)

Each stage is persisted separately under `outputs/audits/<case_id>/` before the next stage starts,
so runs are inspectable and resumable.

| Stage | Name | Sees classifier prediction? | Purpose |
|---:|---|---|---|
| 1 | Data cleansing (Agent 1) | no | objective extraction, missing-data report, no diagnosis |
| 2 | Independent differential generation | **no** | Opponent + differential formed from raw evidence only |
| 3 | Proponent argument (Agent 2) | **yes** | strongest evidence-based case for the top prediction (with SHAP wording rules) |
| 4 | Opponent attack (Agent 3) | reveals now (if ablation says so) | challenges the leading hypothesis citing patient/retrieved evidence |
| 5 | High-acuity watchdog (Agent 4) | no (rule-first) | evidence-grounded red-flag screening vs cited pediatric criteria |
| 6 | Evidence retrieval (RAG) | n/a | guideline/abstract chunks with provenance |
| 7 | Arbitration (Agent 5) | yes, all of the above | weighted synthesis — explicitly not majority voting |
| 8 | Bias audit | n/a | computes CBR/AOR/BCR/HFR/CRR/stability, writes audit JSON |

Control flow is a **deterministic state machine** (custom Python orchestrator by default; LangGraph
is an optional adapter). Requirements: deterministic transitions, per-agent prompts, schema
validation of every JSON payload, retry/failure handling, token/cost logging, reproducibility by
seed + config hash. No framework is adopted for fashion; the orchestrator is deliberately thin.

## Information-isolation ablations (config-driven, not code forks)

| Condition | A1 cleanse | independent differential | proponent | opponent | watchdog | arbitrator |
|---|---|---|---|---|---|---|
| A — all see prediction | ✓ | hidden | ✓ | ✓ | ✓ | ✓ |
| B — only proponent sees it | ✓ | hidden | ✓ | hidden | hidden | ✓ |
| C — hidden until differential complete (default) | ✓ | hidden | ✓ | reveal after Stage 2 | hidden | ✓ |
| D — single LLM baseline | ✓ | — | — | — | — | one-shot |
| E — LLM + self-reflection | ✓ | — | — | — | — | self-reflection loop |
| F — multi-agent, no adversarial roles | ✓ | ✓ | cooperative roles only | | ✓ | ✓ |
| G — full five-agent adversarial framework | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

(plus RAG on/off, anchor-source variants, and the anchor-injection conditions from
experiments/confirmation_bias/.)

## LLM provider modality

`config/agents.yaml` selects a backend per agent: OpenAI-compatible endpoints, Hugging Face
models, Ollama, or any configurable cloud API — never a hard-coded vendor. Constraints enforced in
prompts + validation: agents may not invent lab values, history or evidence absent from the record;
all claims that touch safety thresholds must carry provenance from `knowledge/`.

## Data flow for a single case

```
case_id + raw features + config hash
  → Stage 1..8, each {input_snapshot, prompt_hash, output_json, tokens, latency, model_version}
  → outputs/audits/<case_id>/audit.json   (schema in §19 of the project brief)
  → bias metrics aggregated over cases → outputs/metrics/ + figures
```
