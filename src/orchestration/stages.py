"""Stage functions for the deterministic 8-stage orchestrator (Phase 3, section 3.5).

Each ``run_stageN`` executes exactly one transition of the state machine defined in
``docs/ARCHITECTURE.md`` / ``config/agents.yaml``:

  1 data cleansing → 2 independent differential → 3 proponent → 4 opponent →
  5 high-acuity watchdog → 6 evidence retrieval (RAG) → 7 arbitration → 8 bias audit

Isolation is enforced by data flow, not prompt wording: every stage receives its
inputs from ``CaseState.stage_slice(n)`` (policy-visible fields only) and the
record it sees has already been stripped of target/anchor/identifier columns by
``strip_record()``. The ``inputs`` dict returned in :class:`StageResult` is a
truthful snapshot of what the stage actually received (the arguments its prompt
was built from), so audit files never overstate or understate exposure.

Design notes:

* stages 1/2/5 are prediction-blind by default (condition C); stage 2 is also
  anchor-blind in **every** condition (``anchor_seen: Literal[False]`` guard) —
  the independent differential is the unbiased reference measurement;
* stage 4's "leading hypothesis" is the anchor when one is present, otherwise
  the model's top class, otherwise the top differential candidate when the
  policy hides the prediction (ablation B) — it is always a *revealed* input,
  never something the opponent infers;
* stage 6 packages deterministic offline retrieval (TF-IDF + FTS5) with full
  provenance; stages 6 and 8 are compute stages and synthesize their own audit
  call records (no LLM involved);
* stage 8 computes the per-case bias audit against ground truth (stage 8 is the
  only stage whose policy exposes the target label).

Research prototype — not a medical device.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from src.agents.arbitrator import run_arbitrator
from src.agents.data_cleanser import run_cleanser
from src.agents.independent_differential import run_differential
from src.agents.opponent import run_opponent
from src.agents.proponent import run_proponent
from src.agents.schemas import (
    DataCleanserOutput,
    EvidenceRetrievalOutput,
    IndependentDifferentialOutput,
    RetrievedSource,
    validate_stage,
)
from src.agents.watchdog import run_watchdog
from src.bias.audit import compute_bias_audit
from src.orchestration.state import CaseState

RETRIEVAL_BACKEND = "retrieval_tfidf+fts5"
RETRIEVAL_MODEL_VERSION = "offline-tfidf-v1 (no LLM inference)"
COMPUTE_MODEL_VERSION = "deterministic-compute-v1 (no LLM inference)"

# stage number -> (SCHEMA_REGISTRY key, human name)
STAGE_KEYS: dict[int, tuple[str, str]] = {
    1: ("data_cleanser", "1_data_cleansing"),
    2: ("independent_differential", "2_independent_differential"),
    3: ("proponent", "3_proponent"),
    4: ("opponent", "4_opponent"),
    5: ("watchdog", "5_high_acuity_watchdog"),
    6: ("evidence_retrieval", "6_evidence_retrieval"),
    7: ("arbitrator", "7_arbitration"),
    8: ("bias_audit", "8_bias_audit"),
}


@dataclass
class StageResult:
    """Outcome of one stage transition: validated payload + truthful provenance."""

    stage: int
    key: str
    name: str
    payload: BaseModel
    inputs: dict[str, Any]
    call: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _synthesized_call(stage_name: str, backend: str, model_version: str,
                      prompt_text: str, latency_s: float,
                      schema_name: str) -> dict[str, Any]:
    """Audit record for compute stages (retrieval / bias audit) — no LLM call."""
    return {
        "stage": stage_name,
        "backend": backend,
        "model_version": model_version,
        "prompt_hash": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_s": round(latency_s, 4),
        "attempts": 1,
        "fallback_reason": None,
        "error": None,
        "timestamp": _now(),
        "schema_name": schema_name,
    }


def _provider_call_dict(result: Any) -> dict[str, Any]:
    """ProviderCall dataclass → plain dict (prompt hash, tokens, latency, retries)."""
    from dataclasses import asdict
    return asdict(result.call)


def _typed(state: CaseState, key: str) -> Any:
    """Typed payload of an already-completed stage (validates dicts after resume)."""
    value = state.stage_outputs[key]
    if isinstance(value, BaseModel):
        return value
    return validate_stage(key, value)


def _visibility_meta(state: CaseState, stage: int) -> dict[str, Any]:
    """Isolation statement attached to every stage's input snapshot."""
    return {"visibility": state.policy.description(stage),
            "withheld_from_this_stage": state.stage_slice(stage).get("_withheld", [])}


# --------------------------------------------------------------------------- stage 1
def run_stage1(state: CaseState, provider: Any, retriever: Any = None) -> StageResult:
    """Agent 1 — objective extraction; never diagnoses (schema-guarded)."""
    t0 = time.perf_counter()
    sl = state.stage_slice(1)
    result = run_cleanser(sl["record"], provider, domain=state.domain)
    inputs = {"record": sl["record"], **_visibility_meta(state, 1)}
    return StageResult(1, *STAGE_KEYS[1], result.payload, inputs,
                       _provider_call_dict(result), time.perf_counter() - t0)


# --------------------------------------------------------------------------- stage 2
def run_stage2(state: CaseState, provider: Any, retriever: Any = None) -> StageResult:
    """Independent differential — anchor-blind and prediction-blind by construction."""
    t0 = time.perf_counter()
    sl = state.stage_slice(2)
    if "model_output" in sl or "anchor" in sl:
        raise RuntimeError(
            "stage 2 isolation violated: differential must never receive model_output/anchor")
    cleanser = _typed(state, "data_cleanser")
    result = run_differential(sl["record"], cleanser, provider, domain=state.domain)
    inputs = {"record": sl["record"],
              "cleanser_summary": {"missing_critical_information":
                                   cleanser.missing_critical_information},
              **_visibility_meta(state, 2)}
    return StageResult(2, *STAGE_KEYS[2], result.payload, inputs,
                       _provider_call_dict(result), time.perf_counter() - t0)


# --------------------------------------------------------------------------- stage 3
def run_stage3(state: CaseState, provider: Any, retriever: Any = None) -> StageResult:
    """Proponent — defends the model's top class with SHAP-grounded evidence."""
    t0 = time.perf_counter()
    sl = state.stage_slice(3)
    if sl.get("model_output") is None:
        raise RuntimeError("stage 3 requires model_output (proponent role); policy misconfigured")
    cleanser = _typed(state, "data_cleanser")
    result = run_proponent(sl["record"], cleanser, sl["model_output"],
                           sl.get("shap_explain"), sl.get("uncertainty"), provider,
                           domain=state.domain)
    inputs = {"record": sl["record"],
              "model_output": sl["model_output"],
              "shap_explain": sl.get("shap_explain"),
              "uncertainty": sl.get("uncertainty"),
              **_visibility_meta(state, 3)}
    return StageResult(3, *STAGE_KEYS[3], result.payload, inputs,
                       _provider_call_dict(result), time.perf_counter() - t0)


# --------------------------------------------------------------------------- stage 4
def _leading_hypothesis(state: CaseState, sl: dict[str, Any]) -> str | None:
    """Reveal rule: anchor > model top > differential top (policy-dependent).

    Always derived from *visible* inputs only — if the policy hides the
    prediction and no anchor exists, the opponent attacks the evidence-derived
    differential leader instead of the model's answer (ablation B).
    """
    anchor = sl.get("anchor")
    if anchor and anchor.get("value"):
        return str(anchor["value"])
    model_output = sl.get("model_output")
    if model_output and model_output.get("predicted_class"):
        return str(model_output["predicted_class"])
    differential = _typed(state, "independent_differential")
    return differential.candidate_diagnoses[0] if differential.candidate_diagnoses else None


def run_stage4(state: CaseState, provider: Any, retriever: Any = None) -> StageResult:
    """Opponent — two-stage attack; Stage A stays independent of the model/anchor."""
    t0 = time.perf_counter()
    sl = state.stage_slice(4)
    cleanser = _typed(state, "data_cleanser")
    differential = _typed(state, "independent_differential")
    leading = _leading_hypothesis(state, sl)

    retrieved: list[dict[str, Any]] = []
    retrieval_note = ""
    if state.rag_enabled and retriever is not None:
        from src.rag.retriever import build_query
        query = build_query(differential.candidate_diagnoses, [],
                            leading_hypothesis=leading)
        retrieved = retriever.retrieve_for_agents(query, k=4, used_by=["opponent"])
        retrieval_note = f"pre-stage-6 retrieval for the opponent (k=4, query={query!r})"
    else:
        retrieval_note = "RAG disabled — opponent cites patient-record evidence only"

    stage_a_sees_anchor = state.ablation == "all_see_anchor"
    result = run_opponent(sl["record"], cleanser, differential, leading, retrieved,
                          provider, stage_a_sees_anchor=stage_a_sees_anchor,
                          domain=state.domain)
    inputs = {"record": sl["record"],
              "leading_hypothesis": leading,
              "leading_hypothesis_source": (
                  "anchor" if (sl.get("anchor") or {}).get("value") else
                  "model_top" if sl.get("model_output") else "differential_top"),
              "retrieved_evidence": retrieved,
              "retrieval_note": retrieval_note,
              "stage_a_sees_anchor": stage_a_sees_anchor,
              **_visibility_meta(state, 4)}
    return StageResult(4, *STAGE_KEYS[4], result.payload, inputs,
                       _provider_call_dict(result), time.perf_counter() - t0)


# --------------------------------------------------------------------------- stage 5
def run_stage5(state: CaseState, provider: Any, retriever: Any = None) -> StageResult:
    """Watchdog — rule-first, prediction-blind unless an ablation exposes it."""
    t0 = time.perf_counter()
    sl = state.stage_slice(5)
    prediction = sl.get("model_output")  # present only when policy grants it (ablation A)
    result = run_watchdog(sl["record"], provider, prediction=prediction,
                          domain=state.domain)
    inputs = {"record": sl["record"],
              "prediction_passed": prediction is not None,
              **_visibility_meta(state, 5)}
    return StageResult(5, *STAGE_KEYS[5], result.payload, inputs,
                       _provider_call_dict(result), time.perf_counter() - t0)


# --------------------------------------------------------------------------- stage 6
def run_stage6(state: CaseState, provider: Any, retriever: Any = None) -> StageResult:
    """Evidence retrieval — deterministic offline RAG with full provenance."""
    t0 = time.perf_counter()
    differential = _typed(state, "independent_differential")
    watchdog = (state.stage_outputs.get("watchdog"))
    if watchdog is not None and not isinstance(watchdog, BaseModel):
        watchdog = _typed(state, "watchdog")
    red_flags = ([e.quote for e in watchdog.red_flags_present if e.quote][:2]
                 if watchdog is not None else [])

    from src.rag.retriever import build_query
    query = build_query(differential.candidate_diagnoses, red_flags)

    hits: list[dict[str, Any]] = []
    note = ""
    if not state.rag_enabled:
        note = "RAG disabled by experiment config (rag.enabled=false) — no chunks cited"
    elif retriever is None:
        note = "no retriever supplied to the engine — stage returns an empty, honest result"
    elif not retriever.chunks:
        note = "retrieval index empty — nothing to cite (never fabricated)"
    else:
        hits = retriever.retrieve_for_agents(query, k=4, used_by=["arbitrator"])
        note = "hybrid TF-IDF cosine + FTS5 keyword fusion; every hit carries DOI/URL provenance"

    payload = EvidenceRetrievalOutput.model_validate({
        "query": query,
        "retrieved": [RetrievedSource.model_validate(h) for h in hits],
        "used_by_agents": ["arbitrator"] if hits else [],
        "note": note,
    })
    latency = time.perf_counter() - t0
    call = _synthesized_call("6_evidence_retrieval", RETRIEVAL_BACKEND,
                             RETRIEVAL_MODEL_VERSION, query, latency,
                             "EvidenceRetrievalOutput")
    inputs = {"query": query,
              "candidates": differential.candidate_diagnoses,
              "red_flags": red_flags,
              "rag_enabled": state.rag_enabled,
              **_visibility_meta(state, 6)}
    return StageResult(6, *STAGE_KEYS[6], payload, inputs, call, latency)


# --------------------------------------------------------------------------- stage 7
def run_stage7(state: CaseState, provider: Any, retriever: Any = None) -> StageResult:
    """Arbitrator — weighted synthesis over all prior outputs (never majority voting)."""
    t0 = time.perf_counter()
    sl = state.stage_slice(7)
    if sl.get("model_output") is None:
        raise RuntimeError("stage 7 requires model_output (arbitrator integrates Layer A)")
    result = run_arbitrator(
        cleanser=_typed(state, "data_cleanser"),
        differential=_typed(state, "independent_differential"),
        proponent=_typed(state, "proponent"),
        opponent=(_typed(state, "opponent")
                  if "opponent" in state.stage_outputs else None),
        watchdog=(_typed(state, "watchdog")
                  if "watchdog" in state.stage_outputs else None),
        retrieval=(_typed(state, "evidence_retrieval")
                   if "evidence_retrieval" in state.stage_outputs else None),
        model=sl["model_output"],
        uncertainty=sl.get("uncertainty"),
        anchor=sl.get("anchor"),
        record=sl["record"],
        provider=provider,
        domain=state.domain,
    )
    inputs = {"record": sl["record"],
              "model_output": sl["model_output"],
              "uncertainty": sl.get("uncertainty"),
              "anchor": sl.get("anchor"),
              "prior_stage_keys": [k for _, (k, _) in sorted(STAGE_KEYS.items()) if k != "arbitrator"],
              **_visibility_meta(state, 7)}
    return StageResult(7, *STAGE_KEYS[7], result.payload, inputs,
                       _provider_call_dict(result), time.perf_counter() - t0)


# --------------------------------------------------------------------------- stage 8
def run_stage8(state: CaseState, provider: Any, retriever: Any = None) -> StageResult:
    """Bias audit — the only stage allowed to see the ground-truth label."""
    t0 = time.perf_counter()
    sl = state.stage_slice(8)
    if not sl.get("_visibility", {}).get("sees_ground_truth"):
        raise RuntimeError("stage 8 must see ground truth (policy error)")

    payload = compute_bias_audit(
        case_id=state.case_id,
        anchor=state.anchor,
        ground_truth=sl.get("ground_truth"),
        model_output=state.model_output,
        opponent=((state.stage_outputs.get("opponent").model_dump()
                   if isinstance(state.stage_outputs.get("opponent"), BaseModel)
                   else _typed(state, "opponent").model_dump())
                  if "opponent" in state.stage_outputs else None),
        arbitrator=_typed(state, "arbitrator").model_dump(),
        proponent=_typed(state, "proponent").model_dump(),
    )
    validated = validate_stage("bias_audit", payload)
    latency = time.perf_counter() - t0
    prompt_text = f"{state.case_id}|{state.anchor}|{sl.get('ground_truth')}"
    call = _synthesized_call("8_bias_audit", "deterministic_compute",
                             COMPUTE_MODEL_VERSION, prompt_text, latency,
                             "BiasAuditOutput")
    inputs = {"anchor": state.anchor, "ground_truth": sl.get("ground_truth"),
              "model_top": (state.model_output or {}).get("predicted_class"),
              **_visibility_meta(state, 8)}
    return StageResult(8, *STAGE_KEYS[8], validated, inputs, call, latency)


STAGE_RUNNERS = {
    1: run_stage1, 2: run_stage2, 3: run_stage3, 4: run_stage4,
    5: run_stage5, 6: run_stage6, 7: run_stage7, 8: run_stage8,
}
