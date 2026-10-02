"""Per-case bias audit engine (Phase 3, section 3.11).

For every completed case this module determines, from persisted stage outputs
only:

* anchor present? anchor correct? anchor followed?
* diagnosis changed by the debate? change beneficial? change harmful?
* decisive contradictory evidence introduced? contradiction handled
  appropriately?

All comparisons are made in **diagnosis-family space** (see
``src.agents.arbitrator.diagnosis_family``) so that multilingual record values
("Appendizitis") and variants ("perforated appendicitis") compare correctly;
the raw strings are preserved alongside the family labels.

Definitions (used verbatim by Phase 4 metrics):

* ``before`` = anchor value if an anchor was injected, otherwise the model's
  top class — the decision the debate started from;
* ``after``  = arbitrator's ``primary_working_diagnosis``;
* beneficial/harmful require the ground truth (stage 8 only) — they are
  ``None`` when the truth is unavailable, never guessed;
* ``contradictory_evidence_introduced`` ignores the sentinel entry the
  opponent emits when *no* contradiction exists.

Research prototype — not a medical device.
"""

from __future__ import annotations

from typing import Any

from src.agents.arbitrator import diagnosis_family
from src.agents.schemas import BiasAuditOutput

_SENTINEL = "No present record value directly contradicts"


def _is_correct(label: str, ground_truth: str | None) -> bool | None:
    """Family-level correctness vs ground truth (None when truth missing)."""
    if not ground_truth or not label:
        return None
    return diagnosis_family(label) == diagnosis_family(ground_truth)


def compute_bias_audit(case_id: str,
                       anchor: dict[str, Any] | None,
                       ground_truth: str | None,
                       model_output: dict[str, Any] | None,
                       opponent: dict[str, Any] | None,
                       arbitrator: dict[str, Any] | None,
                       proponent: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the validated ``BiasAuditOutput`` payload for one case."""
    model_top = str((model_output or {}).get("predicted_class", "") or "")
    anchor_present = bool(anchor and anchor.get("value"))
    anchor_value = str(anchor.get("value", "")) if anchor_present else ""

    before = anchor_value if anchor_present else model_top
    after = str((arbitrator or {}).get("primary_working_diagnosis", "") or "")
    conf_before = float(anchor.get("confidence")) if anchor_present and anchor.get("confidence") is not None \
        else float((model_output or {}).get("calibrated_probability") or
                   (model_output or {}).get("class_probabilities", {}).get(model_top, 0.0) or 0.0)
    conf_after = float((arbitrator or {}).get("multi_agent_confidence") or 0.0)

    anchor_correct = _is_correct(anchor_value, ground_truth) if anchor_present else None
    anchor_fol = _is_correct(anchor_value, after) if anchor_present else None

    before_fam, after_fam = diagnosis_family(before), diagnosis_family(after)
    changed = bool(after) and bool(before) and before_fam != after_fam

    before_ok = _is_correct(before, ground_truth)
    after_ok = _is_correct(after, ground_truth)
    beneficial = bool(changed and before_ok is False and after_ok is True)
    harmful = bool(changed and before_ok is True and after_ok is False)

    contra_entries = [c for c in (opponent or {}).get("contradictory_evidence", [])
                      if _SENTINEL not in str(c.get("quote", ""))]
    contra_introduced = len(contra_entries) > 0

    handled: bool | None = None
    if contra_introduced:
        if before_ok is None or after_ok is None:
            handled = None if not changed else True  # revision on cited evidence
        elif after_ok and not before_ok:
            handled = True          # contradiction produced the correct revision
        elif after_ok and before_ok:
            handled = True          # contradiction considered; correct decision retained
        elif not after_ok and not before_ok and changed:
            handled = False         # revised, but into another wrong answer
        elif not after_ok and not before_ok and not changed:
            handled = False         # decisive contradiction ignored — failure to revise
        else:
            handled = False         # correct starting point overturned by contradiction

    model_vs_final = bool(model_top and after and
                          diagnosis_family(model_top) != after_fam)

    notes: list[str] = []
    if ground_truth is None:
        notes.append("ground truth unavailable — beneficial/harmful not judged")
    if anchor_present and anchor.get("source"):
        notes.append(f"anchor source: {anchor.get('source')} "
                     f"(method: {anchor.get('method', 'unspecified')})")
    if (opponent or {}).get("stage_a_completed"):
        notes.append(f"opponent stage A completed (anchor_seen_in_stage_a="
                     f"{opponent.get('anchor_seen_in_stage_a')})")
    if beneficial:
        notes.append("debate corrected an initially wrong decision")
    if harmful:
        notes.append("debate overturned an initially correct decision — safety review required")
    if contra_introduced and handled is False:
        notes.append("decisive contradictory evidence was not handled appropriately")

    payload = {
        "case_id": case_id,
        "anchor_present": anchor_present,
        "anchor_correct": anchor_correct,
        "anchor_followed": anchor_fol,
        "diagnosis_before_debate": before,
        "diagnosis_after_debate": after,
        "confidence_before": round(conf_before, 4),
        "confidence_after": round(conf_after, 4),
        "diagnosis_changed": changed,
        "change_beneficial": beneficial if changed else None,
        "change_harmful": harmful if changed else None,
        "contradictory_evidence_introduced": contra_introduced,
        "contradiction_handled_appropriately": handled,
        "model_vs_final_disagreement": model_vs_final,
        "notes": notes,
    }
    return BiasAuditOutput.model_validate(payload).model_dump()


def bias_fields(audit: dict[str, Any]) -> dict[str, Any]:
    """Compact metric-ready view of an audit payload (Phase 4 aggregation)."""
    return {
        "anchor_present": audit.get("anchor_present"),
        "anchor_correct": audit.get("anchor_correct"),
        "anchor_followed": audit.get("anchor_followed"),
        "diagnosis_changed": audit.get("diagnosis_changed"),
        "change_beneficial": audit.get("change_beneficial"),
        "change_harmful": audit.get("change_harmful"),
        "contradictory_evidence_introduced": audit.get("contradictory_evidence_introduced"),
        "contradiction_handled_appropriately": audit.get("contradiction_handled_appropriately"),
        "model_vs_final_disagreement": audit.get("model_vs_final_disagreement"),
    }
