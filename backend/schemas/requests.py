"""Pydantic request schemas for the research UI backend.

Validation happens here so endpoints never receive malformed input; every
schema documents its bounds explicitly (research tool — not a clinical device).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Run the Phase 2 model (+ calibration, SHAP, uncertainty) for one case."""

    row_index: int = Field(..., ge=0, description="Regensburg dataset row index")


class AgentRunRequest(BaseModel):
    """Run the 8-stage multi-agent pipeline for one case."""

    row_index: int = Field(..., ge=0)
    ablation: Literal["default", "A_all_see_prediction", "B_only_proponent",
                      "C_hidden_until_differential_complete",
                      "all_see_anchor"] = "default"
    rag_enabled: bool = True
    anchor_condition: str | None = Field(
        default=None,
        description="Anchor condition name from config/experiment.yaml, or null for none")


class QuickExperimentRequest(BaseModel):
    """Small synchronous bias experiment for the UI (full runs use the CLI)."""

    conditions: list[str] = Field(
        default_factory=lambda: ["control", "incorrect_anchor"],
        min_length=1, max_length=7,
        description="Anchor conditions to execute (subset of the 7)")
    profiles: list[str] = Field(
        default_factory=lambda: ["9_full_with_rag"],
        min_length=1, max_length=4,
        description="Ablation profiles to execute (subset of the 11)")
    n_cases: int = Field(default=5, ge=1, le=15,
                         description="First N test cases (UI cap; CLI runs all 117)")


class AuditFileRequest(BaseModel):
    """Reserved for future POST-based audit queries (kept for schema parity)."""

    case_id: str = Field(default="", max_length=120)
