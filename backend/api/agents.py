"""Agent pipeline endpoints: run the real 8-stage engine from the UI.

``POST /api/agents/run`` executes the genuine state machine (isolation policy,
schema validation, RAG, watchdog, arbitration, bias audit) for one dataset row
and persists the full per-stage transcript under ``outputs/audits/<case_id>/``
so every UI run is inspectable afterwards.

Runs are serialized behind the shared lock and take ~0.1–0.3 s with the
deterministic backend.

Research prototype — not a medical device.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from backend.api._shared import (DISCLAIMER, ApiUnavailable, get_predictor,
                                  get_retriever, run_lock)
from backend.schemas.requests import AgentRunRequest
from backend.schemas.responses import AgentRunResponse, StageSummary

router = APIRouter(prefix="/agents", tags=["agents"])


@router.post("/run", response_model=AgentRunResponse)
def run_agents(req: AgentRunRequest) -> AgentRunResponse:
    """Execute stages 1–8 for one case with the requested visibility policy."""
    from src.agents.provider import LLMProvider
    from src.bias.anchor_generator import ANCHOR_CONDITIONS, anchor_for_case, anchor_vocabulary
    from src.orchestration.engine import StageError, run_case, run_case_summary
    from src.orchestration.stages import STAGE_KEYS
    from src.orchestration.state import make_state

    try:
        predictor = get_predictor()
    except ApiUnavailable as exc:
        raise HTTPException(status_code=503, detail={"detail": str(exc),
                                                     "hint": exc.hint})
    if not predictor.has_row(req.row_index):
        raise HTTPException(
            status_code=404,
            detail={"detail": f"row {req.row_index} is not in the modelling matrix",
                    "hint": "Pick a row_index from GET /api/cases"})
    if req.anchor_condition is not None and req.anchor_condition not in ANCHOR_CONDITIONS:
        raise HTTPException(
            status_code=422,
            detail={"detail": f"unknown anchor condition {req.anchor_condition!r}",
                    "hint": f"use one of: {', '.join(ANCHOR_CONDITIONS)}"})

    case_suffix = f"ui_row_{req.row_index}"
    case_suffix += f"_{req.ablation}" if req.ablation != "default" else ""
    case_suffix += f"_anchor-{req.anchor_condition}" if req.anchor_condition else ""
    case_suffix += "" if req.rag_enabled else "_norag"

    t0 = time.perf_counter()
    with run_lock():
        try:
            bundle = predictor.bundle(req.row_index)
            record = predictor.df.loc[req.row_index].to_dict()
            anchor = None
            if req.anchor_condition:
                anchor = anchor_for_case(
                    req.anchor_condition, req.row_index, predictor.df,
                    str(record.get("Diagnosis")),
                    bundle[0]["predicted_class"], anchor_vocabulary(predictor.df))
            state = make_state(case_suffix, record, anchor=anchor,
                               ablation=req.ablation, rag_enabled=req.rag_enabled,
                               config_hash=predictor.config_hash, seed=20261002)
            state.model_output, state.shap_explain, state.uncertainty = bundle
            try:
                state = run_case(state, provider=LLMProvider(backend="deterministic"),
                                 retriever=(get_retriever() if req.rag_enabled else None),
                                 persist=True, stage_files=True)
            except StageError as exc:
                raise HTTPException(
                    status_code=500,
                    detail={"detail": str(exc),
                            "hint": "See outputs/audits/<case_id>/audit.json"})
        except ApiUnavailable as exc:
            raise HTTPException(status_code=503,
                                detail={"detail": str(exc), "hint": exc.hint})

    from src.orchestration import persistence
    stages: list[StageSummary] = []
    for stage_no, (key, name) in sorted(STAGE_KEYS.items()):
        if stage_no not in state.completed_stages:
            continue
        meta = state.stage_meta.get(key, {})
        call = meta.get("call") or {}
        inputs = state.stage_inputs.get(key) or {}
        stages.append(StageSummary(
            stage=stage_no, key=key, name=name,
            backend=call.get("backend"), model_version=call.get("model_version"),
            prompt_hash=call.get("prompt_hash"), latency_s=call.get("latency_s"),
            visibility=inputs.get("visibility", {}),
            withheld=inputs.get("withheld_from_this_stage", []),
            output=state.stage_outputs[key].model_dump()
            if hasattr(state.stage_outputs[key], "model_dump")
            else dict(state.stage_outputs[key])))

    return AgentRunResponse(
        case_id=state.case_id, row_index=req.row_index,
        ground_truth=state.ground_truth, anchor=state.anchor,
        ablation=req.ablation, rag_enabled=req.rag_enabled,
        summary=run_case_summary(state), stages=stages,
        audit_path=str(persistence.case_dir(state.case_id)),
        elapsed_s=round(time.perf_counter() - t0, 3),
        disclaimer=DISCLAIMER)
