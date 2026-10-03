"""Pydantic response schemas for the research UI backend.

These document the API contract consumed by ``frontend/src/lib``. Optional
fields stay ``None`` when a source artifact is missing — the UI shows an
explicit "not available" state instead of fabricated values.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    name: str
    version: str
    available: bool
    calibrator: str | None = None
    threshold: float | None = None
    config_hash: str | None = None


class HealthResponse(BaseModel):
    """GET /api/health — service + artifact status for the UI banner."""

    status: str = "ok"
    seed: int = 20261002
    llm_backend: str = "deterministic"
    llm_configured: bool = False
    model: ModelInfo
    rag_chunks: int | None = None
    metrics_files: dict[str, bool] = {}
    figures: list[str] = []
    experiments_available: dict[str, bool] = {}
    disclaimer: str


class PredictResponse(BaseModel):
    """POST /api/predict — normalized Layer-A output for one case."""

    row_index: int
    model_output: dict[str, Any]
    shap_explain: dict[str, Any]
    uncertainty: dict[str, Any]


class StageSummary(BaseModel):
    stage: int
    key: str
    name: str
    backend: str | None = None
    model_version: str | None = None
    prompt_hash: str | None = None
    latency_s: float | None = None
    visibility: dict[str, Any] = {}
    withheld: list[str] = []
    output: dict[str, Any] = {}


class AgentRunResponse(BaseModel):
    """POST /api/agents/run — one executed pipeline with isolation metadata."""

    case_id: str
    row_index: int
    ground_truth: str | None = None
    anchor: dict[str, Any] | None = None
    ablation: str
    rag_enabled: bool
    summary: dict[str, Any] = {}
    stages: list[StageSummary] = []
    audit_path: str
    elapsed_s: float
    disclaimer: str


class QuickExperimentCondition(BaseModel):
    condition: str
    headline: dict[str, Any] = {}
    n_rows: int = 0


class QuickExperimentResponse(BaseModel):
    """POST /api/experiments/quick — small executed experiment for the UI."""

    n_cases: int
    conditions: list[str]
    profiles: list[str]
    rows: list[dict[str, Any]] = []
    headline: dict[str, Any] = {}
    elapsed_s: float
    note: str
    disclaimer: str


class ErrorDetail(BaseModel):
    detail: str
    hint: str | None = None
