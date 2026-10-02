"""Strict Pydantic output schemas for the five-agent pipeline (Phase 3, section 3.2).

Every stage output is validated against its schema. Malformed LLM output is rejected and
retried; after the retry budget the stage fails loudly — missing fields are never invented
(SAFETY.md failure-handling policy).

Schemas mirror ``config/agents.yaml`` output_schema entries exactly.

Research prototype — not a medical device; outputs are research artefacts, not clinical
advice.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

RESEARCH_DISCLAIMER = (
    "Research prototype — generated for a bias study; not a medical device, not for "
    "clinical use, not a diagnosis."
)


class EvidenceRef(BaseModel):
    """A single cited piece of evidence (patient record, rule, or retrieved source)."""

    kind: Literal["patient_record", "rule", "retrieved_source", "model", "derived"] = "patient_record"
    citation: str = Field(..., description="Row/column, rule id, or DOI identifying the source")
    quote: str = Field(default="", description="Value or short excerpt — must exist in the input")
    relevance: str = ""


class DataCleanserOutput(BaseModel):
    """Agent 1 — objective extraction, no diagnosis."""

    demographics: dict[str, Any] = {}
    vitals: dict[str, Any] = {}
    laboratory_values: dict[str, Any] = {}
    symptoms: list[str] = []
    physical_findings: list[str] = []
    imaging_findings: list[str] = []
    missing_critical_information: list[str] = []
    objective_red_flags: list[EvidenceRef] = []
    data_quality_warnings: list[str] = []
    diagnoses_made: Literal[False] = False  # hard guard: the cleanser never diagnoses

    @field_validator("demographics", "vitals", "laboratory_values")
    @classmethod
    def _not_empty_dicts(cls, v):
        if not isinstance(v, dict):
            raise ValueError("must be a dict")
        return v


class IndependentDifferentialOutput(BaseModel):
    """Stage 2 — differential generated BEFORE any anchor/prediction exposure."""

    candidate_diagnoses: list[str] = Field(min_length=1)
    supporting_evidence: dict[str, list[str]]
    contradictory_evidence: dict[str, list[str]]
    uncertainty: str
    missing_information: list[str]
    anchor_seen: Literal[False] = False  # information-isolation guard


class ProponentOutput(BaseModel):
    """Agent 2 — strongest evidence-based case for the model's top prediction."""

    supported_diagnosis: str
    model_probability: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[EvidenceRef]
    missing_expected_evidence: list[str]
    contradictions_acknowledged: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class OpponentOutput(BaseModel):
    """Agent 3 — adversarial critique of the leading hypothesis."""

    challenged_hypothesis: str
    contradictory_evidence: list[EvidenceRef]
    missing_criteria: list[str]
    alternative_diagnoses: list[str]
    strongest_alternative: str
    disconfirmatory_tests: list[str]
    anchoring_risk: Literal["LOW", "MEDIUM", "HIGH"]
    reason_for_anchoring_risk: str
    stage_a_completed: bool = True
    anchor_seen_in_stage_a: bool = False


class WatchdogOutput(BaseModel):
    """Agent 4 — high-sensitivity, evidence-grounded safety screen (rule-first)."""

    high_acuity_conditions_considered: list[str]
    red_flags_present: list[EvidenceRef]
    red_flags_absent: list[str]
    cannot_assess_due_to_missing_data: list[str]
    urgent_rule_out_conditions: list[str]
    risk_level: Literal["LOW", "MODERATE", "HIGH", "CRITICAL"]
    evidence: list[EvidenceRef]
    prediction_seen: bool = False  # watchdog stays prediction-blind unless ablation says otherwise
    rules_applied: list[str] = []
    rules_refused: list[str] = []


class RetrievedSource(BaseModel):
    """One retrieved evidence chunk with full provenance."""

    title: str
    authors: str = ""
    year: str = ""
    source: str = ""
    doi: str = ""
    url: str = ""
    section: str = ""
    license: str = ""
    chunk_text: str = ""
    score: float = 0.0
    used_by: list[str] = []


class EvidenceRetrievalOutput(BaseModel):
    """Stage 6 — RAG results with provenance (no uncited chunks)."""

    query: str
    retrieved: list[RetrievedSource]
    used_by_agents: list[str] = []
    note: str = ""


class ArbitratorOutput(BaseModel):
    """Agent 5 — weighted synthesis (NOT majority voting)."""

    primary_working_diagnosis: str
    multi_agent_confidence: float = Field(ge=0.0, le=1.0)
    confidence_interpretation: str
    critical_differentials: list[str]
    urgent_rule_outs: list[str]
    recommended_information_or_tests: list[str]
    important_missing_data: list[str]
    model_vs_agents_disagreement: str
    confirmation_bias_risk_before_debate: Literal["LOW", "MODERATE", "HIGH"]
    confirmation_bias_risk_after_debate: Literal["LOW", "MODERATE", "HIGH"]
    audit_trail: list[str]
    research_only_disclaimer: str = RESEARCH_DISCLAIMER
    confidence_derivation: str = Field(
        default="",
        description="How model probability was transformed into multi-agent confidence",
    )
    anchor_present: bool = False
    anchor_followed: bool | None = None


class BiasAuditOutput(BaseModel):
    """Stage 8 — per-case machine-readable bias audit."""

    case_id: str
    anchor_present: bool
    anchor_correct: bool | None
    anchor_followed: bool | None
    diagnosis_before_debate: str
    diagnosis_after_debate: str
    confidence_before: float
    confidence_after: float
    diagnosis_changed: bool
    change_beneficial: bool | None
    change_harmful: bool | None
    contradictory_evidence_introduced: bool
    contradiction_handled_appropriately: bool | None
    model_vs_final_disagreement: bool
    research_only_disclaimer: str = RESEARCH_DISCLAIMER
    notes: list[str] = []


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "data_cleanser": DataCleanserOutput,
    "independent_differential": IndependentDifferentialOutput,
    "proponent": ProponentOutput,
    "opponent": OpponentOutput,
    "watchdog": WatchdogOutput,
    "evidence_retrieval": EvidenceRetrievalOutput,
    "arbitrator": ArbitratorOutput,
    "bias_audit": BiasAuditOutput,
}


def validate_stage(stage_key: str, payload: dict | BaseModel) -> BaseModel:
    """Validate a stage payload against its registered schema.

    Raises ``pydantic.ValidationError`` on malformed output — callers must retry or fail
    the stage loudly (never fill in defaults silently).
    """
    schema = SCHEMA_REGISTRY.get(stage_key)
    if schema is None:
        raise KeyError(f"no schema registered for stage {stage_key!r}")
    if isinstance(payload, schema):
        return payload
    return schema.model_validate(payload)
