"""Agent 3 — Opponent (Phase 3, section 3.6).

Two-stage adversarial critique with information isolation:

* **Stage A** (runs inside the anchor-blind part of the pipeline, before the
  leading hypothesis is revealed): independent hypothesis generation and
  challenge from patient evidence + the independent differential only. The
  ``anchor_seen_in_stage_a`` field records whether an ablation exposed the
  prediction (config condition A); by default it did not.
* **Stage B** (after reveal): explicitly attacks the leading hypothesis —
  contradictory evidence, missing criteria, alternative diagnoses, strongest
  alternative, recommended disconfirmatory tests, anchoring risk.

Constraints (schema + builder enforced):

* every challenge cites a patient value or a retrieved source — challenges
  without citations cannot be constructed;
* no disagreement for its own sake: when no present value contradicts the
  leading hypothesis the output says so explicitly instead of inventing one.

Research prototype — not a medical device.
"""

from __future__ import annotations

from typing import Any

from src.agents.independent_differential import (
    _CANDIDATES,
    _contra_appendicitis,
    _has,
    _is_num,
)
from src.agents.provider import LLMProvider, ProviderResult
from src.agents.schemas import DataCleanserOutput, EvidenceRef, IndependentDifferentialOutput, OpponentOutput

# Disconfirmatory tests grounded in what the record does/does not contain.
_TEST_MENU = [
    ("Appendix_Diameter missing", "Specialist-repeated ultrasound with appendix measurement "
                                  "if the appendix was not measured"),
    ("WBC_Count missing", "Repeat complete blood count with differential"),
    ("CRP missing", "Repeat CRP as inflammatory marker"),
    ("Dysuria workup absent", "Urinalysis if urinary symptoms persist"),
    ("Peritonitis equivocal", "Serial abdominal examination by an independent examiner"),
]


def _stage_a(record: dict[str, Any],
             differential: IndependentDifferentialOutput) -> list[str]:
    """Anchor-blind initial alternatives (Stage A evidence)."""
    return list(differential.candidate_diagnoses)


def extract(record: dict[str, Any], cleanser: DataCleanserOutput,
            differential: IndependentDifferentialOutput,
            leading_hypothesis: str | None,
            retrieved: list[dict[str, Any]],
            stage_a_sees_anchor: bool = False) -> dict[str, Any]:
    """Deterministic two-stage opponent payload."""
    alternatives = [c for c in _stage_a(record, differential)
                    if leading_hypothesis is None
                    or c.split("(")[0].strip().lower() not in leading_hypothesis.lower()
                    and leading_hypothesis.lower() not in c.lower()]
    leading = leading_hypothesis or "not revealed at this stage (information isolation)"

    contra_refs: list[dict[str, Any]] = []
    if leading_hypothesis:
        contra_fn = _contra_appendicitis if "appendicitis" in leading_hypothesis.lower() else None
        if contra_fn:
            for stmt in contra_fn(record):
                # statement text embeds the record value that serves as the citation
                col = stmt.split("=")[0].split()[0]
                contra_refs.append(EvidenceRef(
                    kind="patient_record", citation=f"column={col}", quote=stmt,
                    relevance=f"present record value contradicting {leading_hypothesis!r}").model_dump())
        if _is_num(record, "CRP") and float(record.get("CRP") or 0) == 0.0 and "appendicitis" in leading_hypothesis.lower():
            contra_refs.append(EvidenceRef(
                kind="patient_record", citation="column=CRP", quote="CRP=0.0",
                relevance="no inflammatory response despite the leading surgical hypothesis").model_dump())
        for src in retrieved[:2]:
            contra_refs.append(EvidenceRef(
                kind="retrieved_source", citation=src.get("doi") or src.get("title", ""),
                quote=(src.get("chunk_text") or "")[:240],
                relevance="retrieved source considered when challenging the leading hypothesis").model_dump())

    missing_criteria = [
        f"{c} — diagnostic criterion unavailable in the record"
        for c in cleanser.missing_critical_information[:5]
    ]
    missing_criteria.append("Histopathology (reference standard) unavailable pre-operatively")

    disconfirmatory = [test for need, test in _TEST_MENU
                       if any(need.split()[0] in m for m in cleanser.missing_critical_information)]

    # Anchoring risk from evidence strength: leading hypothesis with thin
    # independent support while alternatives have comparable support = HIGH.
    if leading_hypothesis is None:
        anchoring_risk, reason = "LOW", ("Stage A only: the leading hypothesis was not revealed, "
                                         "so anchoring to a model/anchor prediction is impossible by construction")
    else:
        support_leading = next((len(differential.supporting_evidence.get(c, []))
                                for c in differential.candidate_diagnoses
                                if c.lower().startswith(leading_hypothesis.lower()[:12])), 0)
        best_alt_support = max((len(differential.supporting_evidence.get(c, []))
                                for c in alternatives), default=0)
        if support_leading <= 1 and best_alt_support >= 2:
            anchoring_risk = "HIGH"
            reason = (f"leading hypothesis has {support_leading} supporting evidence statements "
                      f"while strongest alternative has {best_alt_support} — challenge is required")
        elif support_leading < best_alt_support + 2:
            anchoring_risk = "MEDIUM"
            reason = (f"leading hypothesis support ({support_leading}) is not clearly stronger "
                      f"than alternatives ({best_alt_support})")
        else:
            anchoring_risk = "LOW"
            reason = (f"independent record evidence ({support_leading} statements) clearly "
                      f"supports the leading hypothesis; challenging it further would be "
                      f"disagreement without evidence")

    if not contra_refs and leading_hypothesis:
        contra_refs.append(EvidenceRef(
            kind="patient_record", citation="record-scan",
            quote="No present record value directly contradicts the leading hypothesis",
            relevance="explicit negative finding: no evidence-free challenge is manufactured").model_dump())

    return {
        "challenged_hypothesis": leading,
        "contradictory_evidence": contra_refs,
        "missing_criteria": missing_criteria,
        "alternative_diagnoses": alternatives,
        "strongest_alternative": (alternatives[0] if alternatives
                                  else "none supported by record evidence at Stage A"),
        "disconfirmatory_tests": disconfirmatory or [
            "None identified from record gaps — serial clinical review remains standard"],
        "anchoring_risk": anchoring_risk,
        "reason_for_anchoring_risk": reason,
        "stage_a_completed": True,
        "anchor_seen_in_stage_a": bool(stage_a_sees_anchor),
    }


def run_opponent(record: dict[str, Any], cleanser: DataCleanserOutput,
                 differential: IndependentDifferentialOutput,
                 leading_hypothesis: str | None,
                 retrieved: list[dict[str, Any]],
                 provider: LLMProvider,
                 stage_a_sees_anchor: bool = False) -> ProviderResult:
    """Execute Agent 3 (Stage A anchor-blind unless ablation exposes the anchor)."""
    system = (
        "You are the Opponent in a pediatric appendicitis research debate. "
        "Stage A: generate independent challenges from patient evidence only "
        "(the leading hypothesis may or may not be revealed to you now). "
        "Stage B: attack the revealed leading hypothesis using record values or "
        "retrieved sources as citations for every claim. Never disagree without "
        "evidence; never invent findings. Emit JSON."
    )
    user = ("Inputs:\n"
            f"record={record}\n"
            f"cleansed_missing={cleanser.missing_critical_information}\n"
            f"independent_differential={differential.model_dump()}\n"
            f"leading_hypothesis_revealed={leading_hypothesis is not None}\n"
            f"leading_hypothesis={leading_hypothesis}\n"
            f"retrieved_evidence={retrieved}\n"
            f"stage_a_sees_anchor={stage_a_sees_anchor}")
    return provider.complete_json(
        stage="4_opponent", system=system, user=user, schema=OpponentOutput,
        deterministic=lambda: extract(record, cleanser, differential,
                                      leading_hypothesis, retrieved, stage_a_sees_anchor))
