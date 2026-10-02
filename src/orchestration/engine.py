"""Deterministic, resumable 8-stage orchestrator (Phase 3, section 3.6).

``run_case`` drives one patient case through the state machine:

    stage 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

with these guarantees (docs/ARCHITECTURE.md):

* **deterministic transitions** — fixed order, no branching on model output;
* **information isolation** — each stage only ever receives
  ``CaseState.stage_slice(n)`` (see ``state.py``); the engine refuses to run if
  a stage would receive withheld fields it is not entitled to;
* **schema validation** — every payload is a validated Pydantic model before it
  is persisted or handed to the next stage;
* **fail-loudly errors** — a stage exception is recorded in the case audit and
  re-raised as :class:`StageError`; no silent defaulting of missing fields;
* **resumability** — every completed stage is on disk before the next starts;
  ``resume=True`` reloads and validates completed stages instead of re-running
  them (a crash mid-run continues where it stopped);
* **audit trail** — per-stage prompt hash / model version / tokens / latency /
  retry info, plus a case-level ``audit.json`` (project brief section 19).

LLM backend resolution is delegated to :class:`src.agents.provider.LLMProvider`
(env ``LLM_BACKEND`` > ``config/agents.yaml``); with no LLM configured the
deterministic rule-based backend is used and recorded honestly in the audit.

Research prototype — not a medical device.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.orchestration import persistence
from src.orchestration.state import CaseState
from src.orchestration.stages import STAGE_KEYS, STAGE_RUNNERS

try:
    from src.agents.provider import LLMProvider, ProviderError
except ImportError:  # pragma: no cover - provider is a hard dependency in practice
    LLMProvider = None  # type: ignore

    class ProviderError(RuntimeError):  # type: ignore
        pass


class StageError(RuntimeError):
    """One stage failed; carries case/stage context. The audit records it too."""

    def __init__(self, case_id: str, stage_no: int, key: str, cause: BaseException):
        self.case_id, self.stage_no, self.key = case_id, stage_no, key
        super().__init__(f"case {case_id!r} failed at stage {stage_no} ({key}): "
                         f"{type(cause).__name__}: {cause}")


def _make_retriever(state: CaseState) -> Any:
    """Offline retriever for stage 4/6; builds the index on first use if missing."""
    if not state.rag_enabled:
        return None
    from src.rag.retriever import Retriever
    retriever = Retriever()
    if not retriever.chunks:
        from src.rag.indexer import build_index
        build_index()  # deterministic: literature CSV + knowledge YAML + manifest
        retriever.refresh()
    return retriever


def _restore(directory: Path, state: CaseState) -> int:
    """Populate state from previously completed stage files (resume path)."""
    completed = persistence.load_completed(directory)
    for stage_no in sorted(completed):
        record = completed[stage_no]
        key, _ = STAGE_KEYS[stage_no]
        state.stage_outputs[key] = record["payload"]
        state.stage_inputs[key] = record["inputs"]
        state.stage_meta[key] = record["meta"]
        if stage_no not in state.completed_stages:
            state.completed_stages.append(stage_no)
    return len(completed)


def run_case(state: CaseState,
             provider: Any = None,
             retriever: Any = None,
             *,
             resume: bool = False,
             persist: bool = True,
             out_root: Path | None = None) -> CaseState:
    """Execute stages 1–8 for one case and return the fully-populated state.

    Parameters
    ----------
    provider:
        an :class:`LLMProvider`; constructed from config/env when omitted.
    retriever:
        optional pre-built RAG retriever (built automatically when
        ``state.rag_enabled`` and one is not supplied).
    resume:
        reload completed stages from the audit directory instead of re-running.
    persist:
        write per-stage files and ``audit.json`` (disable for in-memory tests).
    out_root:
        override ``outputs/audits/`` (used by tests and experiment batches).
    """
    if provider is None:
        provider = LLMProvider()
    directory = persistence.case_dir(state.case_id, root=out_root)
    started_utc = persistence.utc_now()
    t0 = time.perf_counter()

    restored = 0
    if resume:
        restored = _restore(directory, state)

    need_rag = state.rag_enabled and (
        retriever is None and (4 not in state.completed_stages
                               or 6 not in state.completed_stages))
    if need_rag:
        retriever = _make_retriever(state)

    try:
        for stage_no, (key, name) in sorted(STAGE_KEYS.items()):
            if stage_no in state.completed_stages:
                continue
            runner = STAGE_RUNNERS[stage_no]
            try:
                result = runner(state, provider, retriever)
            except (ProviderError, RuntimeError, ValueError, KeyError,
                    TypeError, AttributeError) as exc:
                state.errors[key] = f"{type(exc).__name__}: {exc}"
                raise StageError(state.case_id, stage_no, key, exc) from exc

            state.stage_outputs[key] = result.payload
            state.stage_inputs[key] = result.inputs
            files: dict[str, str] = {}
            if persist:
                files = persistence.save_stage(directory, stage_no, key,
                                               result.inputs, result.payload,
                                               {"call": result.call,
                                                "latency_s": round(result.latency_s, 4)})
            state.stage_meta[key] = {"call": result.call,
                                     "latency_s": round(result.latency_s, 4),
                                     "files": files}
            state.completed_stages.append(stage_no)
    finally:
        # persist an audit even on failure so the run is always inspectable
        if persist and (state.completed_stages or state.errors):
            audit = persistence.build_audit(
                state, started_utc, persistence.utc_now(), time.perf_counter() - t0,
                getattr(provider, "backend", "unknown"),
                getattr(provider, "model_version", "unknown"))
            audit["resumed_stages"] = restored
            persistence.save_audit(directory, audit)

    return state


def run_case_summary(state: CaseState, out_root: Path | None = None) -> dict[str, Any]:
    """Compact machine-readable summary of a finished run (CLI/UI convenience)."""
    arb = state.stage_outputs.get("arbitrator")
    aud = state.stage_outputs.get("bias_audit")
    wd = state.stage_outputs.get("watchdog")
    return {
        "case_id": state.case_id,
        "completed_stages": sorted(state.completed_stages),
        "errors": state.errors,
        "model_top": (state.model_output or {}).get("predicted_class"),
        "model_probability": (state.model_output or {}).get("calibrated_probability"),
        "primary_working_diagnosis": getattr(arb, "primary_working_diagnosis", None),
        "multi_agent_confidence": getattr(arb, "multi_agent_confidence", None),
        "watchdog_risk": getattr(wd, "risk_level", None),
        "anchor_followed": getattr(aud, "anchor_followed", None),
        "diagnosis_changed": getattr(aud, "diagnosis_changed", None),
        "audit_dir": str(persistence.case_dir(state.case_id, root=out_root)),
    }
