"""Agent 2 — Proponent (Phase 3, section 3.5).

Builds the strongest **evidence-based** case for the classifier's top
prediction. The proponent is the only agent that intentionally sees the model
output alongside the patient record (visibility per ``config/agents.yaml``).

Hard constraints (schema-enforced):

* ``supporting_evidence`` may only cite values actually present in the record
  or SHAP contributions actually produced by the model;
* ``contradictory_evidence`` may never be omitted — every present value that
  argues against the supported hypothesis is listed (no cherry-picking);
* ``missing_expected_evidence`` reports absent expected findings instead of
  pretending they were negative;
* ``confidence`` is *derived* from the calibrated probability and tempered by
  model uncertainty and data completeness — it is not a copy of the raw
  model probability (the copy restriction applies to the arbitrator; the
  proponent documents the same derivation in its prompt/audit trail).

SHAP values are quoted with the mandated wording: features *contributed to the
model prediction*, never caused the diagnosis.

Research prototype — not a medical device.
"""

from __future__ import annotations

from typing import Any

from src.agents.independent_differential import _contra_appendicitis, _has, _is_num
from src.agents.provider import LLMProvider, ProviderResult
from src.agents.schemas import DataCleanserOutput, EvidenceRef, ProponentOutput
from src.explainability.shap_engine import WORDING as SHAP_WORDING

# Map model predicted class → contradiction extractor for that hypothesis.
_CONTRADICTION_EXTRACTORS = {
    "appendicitis": _contra_appendicitis,
}


def _model_confidence(calibrated_prob: float, uncertainty: dict[str, Any] | None,
                      n_missing_critical: int) -> float:
    """Deterministic confidence derivation (documented, bounded).

    Start from the calibrated probability of the predicted class, then temper:
    * HIGH uncertainty band → pull toward 0.5 (cap 0.65);
    * MODERATE band → cap 0.80;
    * 3+ missing critical fields → −0.05 (data incompleteness).
    Never exceeds [0.05, 0.99].
    """
    p = float(calibrated_prob)
    level = (uncertainty or {}).get("uncertainty_level", "MODERATE")
    if level == "HIGH":
        p = min(p, 0.65)
        p = 0.5 + (p - 0.5) * 0.6
    elif level == "MODERATE":
        p = min(p, 0.80)
    if n_missing_critical >= 3:
        p -= 0.05
    return round(min(max(p, 0.05), 0.99), 4)


def extract(record: dict[str, Any], cleanser: DataCleanserOutput,
            model: dict[str, Any], shap_explain: dict[str, Any] | None,
            uncertainty: dict[str, Any] | None,
            domain: str = "appendicitis") -> dict[str, Any]:
    """Deterministic proponent payload from record + model + SHAP values only."""
    if domain != "appendicitis":
        return _extract_domain(record, cleanser, model, shap_explain, uncertainty, domain)
    predicted = str(model.get("predicted_class", ""))
    probs = model.get("class_probabilities", {}) or {}
    calibrated = model.get("calibrated_probability")
    if calibrated is None:
        calibrated = probs.get(predicted, 0.5)

    supporting: list[EvidenceRef] = []
    for col in ("Lower_Right_Abd_Pain", "Migratory_Pain", "Ipsilateral_Rebound_Tenderness",
                "Psoas_Sign", "Coughing_Pain", "Loss_of_Appetite", "Nausea",
                "Peritonitis", "Target_Sign", "Surrounding_Tissue_Reaction"):
        v = record.get(col)
        if _has(record, col, "yes") or (col == "Peritonitis" and _has(record, col, "local", "generalized")):
            supporting.append(EvidenceRef(kind="patient_record", citation=f"column={col}",
                                          quote=f"{col}={v}",
                                          relevance="present finding supporting the supported hypothesis"))
    wbc, crp = record.get("WBC_Count"), record.get("CRP")
    if _is_num(record, "WBC_Count") and float(wbc) >= 10.0:
        supporting.append(EvidenceRef(kind="patient_record", citation="column=WBC_Count",
                                      quote=f"WBC_Count={wbc}",
                                      relevance="elevated white cell count"))
    if _is_num(record, "CRP") and float(crp) > 5.0:
        supporting.append(EvidenceRef(kind="patient_record", citation="column=CRP",
                                      quote=f"CRP={crp}",
                                      relevance="elevated inflammatory marker"))
    if not supporting:
        supporting.append(EvidenceRef(
            kind="model", citation=str(model.get("model_name", "model")),
            quote="No individual record value independently supports the model's top class; "
                  "the case rests on the model score alone",
            relevance="weak-evidence disclosure (no cherry-picking in the opposite direction)"))

    for c in (shap_explain or {}).get("top_contributors", [])[:5]:
        if (c.get("direction") == "positive"
                and str(c.get("feature", "")).rstrip("__missing") not in ("Alvarado_Score",)):
            supporting.append(EvidenceRef(
                kind="model",
                citation=f"{model.get('model_name')} {model.get('model_version')} SHAP",
                quote=f"{c.get('feature')}={c.get('value')} — contributed positively "
                      f"(shap={c.get('contribution')})",
                relevance=SHAP_WORDING))

    contra_fn = _CONTRADICTION_EXTRACTORS.get(predicted.lower(), _contra_appendicitis)
    contradictions = [f"{c}: value present in record argues against the supported hypothesis"
                      for c in contra_fn(record)]
    if _is_num(record, "CRP") and float(crp or 0) == 0.0 and predicted.lower() == "appendicitis":
        contradictions.append(f"CRP=0.0 (no inflammatory response recorded) despite model "
                              f"supporting {predicted!r}")
    contradiction_refs = [EvidenceRef(kind="patient_record", citation="column-values",
                                      quote=c, relevance="counter-evidence the proponent must acknowledge")
                          for c in contradictions]

    missing = [f"{c} missing — expected finding unavailable for the supported hypothesis"
               for c in cleanser.missing_critical_information[:6]]
    if _is_num(record, "WBC_Count") is False:
        missing.append("WBC_Count unavailable — differentiating value absent")

    confidence = _model_confidence(float(calibrated), uncertainty,
                                   len(cleanser.missing_critical_information))
    return {
        "supported_diagnosis": predicted,
        "model_probability": round(float(calibrated), 4),
        "supporting_evidence": [e.model_dump() for e in supporting],
        "missing_expected_evidence": missing,
        "contradictory_evidence_acknowledged_flag": True,
        "contradictions_acknowledged": contradictions or
            ["No present record value directly contradicts the supported hypothesis; "
             "absence of expected findings is listed under missing_expected_evidence"],
        "confidence": confidence,
    }


def _extract_domain(record: dict[str, Any], cleanser: DataCleanserOutput,
                     model: dict[str, Any], shap_explain: dict[str, Any] | None,
                     uncertainty: dict[str, Any] | None, domain: str) -> dict[str, Any]:
    """Domain-pack proponent: evidence from the pack's candidates, no invented columns."""
    from src.agents.domains import get_pack
    pack = get_pack(domain)
    predicted = str(model.get("predicted_class", ""))
    probs = model.get("class_probabilities", {}) or {}
    calibrated = model.get("calibrated_probability")
    if calibrated is None:
        calibrated = probs.get(predicted, 0.5)

    supp_fn = next((s for name, s, _ in pack.candidates if name == predicted), None)
    contra_fn = next((c for name, _, c in pack.candidates if name == predicted), None)
    supporting_stmts = supp_fn(record) if supp_fn else []
    contradictions = list(contra_fn(record)) if contra_fn else []

    supporting = [EvidenceRef(kind="patient_record", citation="record-value",
                              quote=s,
                              relevance="present record value supporting the model's top class")
                  for s in supporting_stmts]
    for c in (shap_explain or {}).get("top_contributors", [])[:5]:
        if c.get("direction") == "positive":
            supporting.append(EvidenceRef(
                kind="model", citation=f"{model.get('model_name')} SHAP",
                quote=f"{c.get('feature')}={c.get('value')} — contributed positively "
                      f"(shap={c.get('contribution')})",
                relevance=SHAP_WORDING))
    if not supporting:
        supporting.append(EvidenceRef(
            kind="model", citation=str(model.get("model_name", "model")),
            quote="No individual record value independently supports the model's top class; "
                  "the case rests on the model score alone",
            relevance="weak-evidence disclosure (no cherry-picking in the opposite direction)"))

    missing = [f"{c} missing — expected finding unavailable for the supported hypothesis"
               for c in cleanser.missing_critical_information[:6]]
    confidence = _model_confidence(float(calibrated), uncertainty,
                                   len(cleanser.missing_critical_information))
    return {
        "supported_diagnosis": predicted,
        "model_probability": round(float(calibrated), 4),
        "supporting_evidence": [e.model_dump() for e in supporting],
        "missing_expected_evidence": missing,
        "contradictory_evidence_acknowledged_flag": True,
        "contradictions_acknowledged": contradictions or
            ["No present record value directly contradicts the supported hypothesis; "
             "absence of expected findings is listed under missing_expected_evidence"],
        "confidence": confidence,
    }


def run_proponent(record: dict[str, Any], cleanser: DataCleanserOutput,
                  model: dict[str, Any], shap_explain: dict[str, Any] | None,
                  uncertainty: dict[str, Any] | None,
                  provider: LLMProvider, domain: str = "appendicitis") -> ProviderResult:
    """Execute Agent 2 (sees patient data + model probabilities + SHAP + uncertainty)."""
    system = (
        "You are the Proponent in a pediatric appendicitis research debate. "
        "Defend the classifier's top prediction with the strongest honest "
        "evidence-based case. You MUST acknowledge every contradictory finding "
        "and every missing expected finding — no cherry-picking. Cite only "
        "values present in the record. SHAP wording: features contributed to "
        "the model prediction, never caused the diagnosis. Emit JSON."
    )
    if domain != "appendicitis":
        from src.agents.domains import get_pack
        system = (f"You are the Proponent in a pediatric {get_pack(domain).display} "
                  "research debate. Defend the classifier's top prediction with the "
                  "strongest honest evidence-based case. You MUST acknowledge every "
                  "contradictory finding and every missing expected finding — no "
                  "cherry-picking. Cite only values present in the record. SHAP wording: "
                  "features contributed to the model prediction, never caused the "
                  "diagnosis. Emit JSON.")
    user = ("Inputs:\n"
            f"record={record}\n"
            f"cleansed={cleanser.model_dump()}\n"
            f"model_output={model}\n"
            f"shap={shap_explain}\n"
            f"uncertainty={uncertainty}")
    return provider.complete_json(
        stage="3_proponent", system=system, user=user, schema=ProponentOutput,
        deterministic=lambda: {k: v for k, v in
                               extract(record, cleanser, model, shap_explain, uncertainty,
                                       domain=domain).items()
                               if k in ProponentOutput.model_fields})
