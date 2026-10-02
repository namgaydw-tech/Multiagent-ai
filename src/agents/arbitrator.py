"""Agent 5 — Pediatric Consultant / Arbitrator (Phase 3, section 3.9).

Weighted synthesis of all preceding stages — **not** majority voting, and the
model probability is never copied into ``multi_agent_confidence``.

Deterministic weighting (documented in ``confidence_derivation`` of every
output):

* ``model_support``      — calibrated probability of the model's top class
  (quality-adjusted by calibration being validation-selected);
* ``contradiction_strength`` — opponent counter-evidence density plus the
  strongest alternative's independent support;
* ``data_completeness``  — fraction of critical fields present (Agent 1);
* ``guideline_support``  — retrieved evidence coverage (capped, stage 6);
* ``clinical_risk``      — watchdog risk level (raises urgency, lowers
  comfortable confidence);
* ``model uncertainty``  — LOW/MODERATE/HIGH band multiplies the final blend.

The blend starts from an evidence-weighted neutral base (0.5), moves toward
the candidate with the strongest total evidence, and is then scaled by
uncertainty — so a 0.98 model probability can legitimately end as 0.62
multi-agent confidence (or higher, if independent evidence concurs).

Confirmation-bias risk before/after the debate is derived from anchor
presence, whether the anchor was followed, and whether decisive contradictory
evidence was handled — never asserted.

Research prototype — not a medical device; the output is a research-system
working hypothesis, not a diagnosis.
"""

from __future__ import annotations

from typing import Any

from src.agents.provider import LLMProvider, ProviderResult
from src.agents.schemas import (
    ArbitratorOutput,
    DataCleanserOutput,
    EvidenceRetrievalOutput,
    IndependentDifferentialOutput,
    OpponentOutput,
    ProponentOutput,
    WatchdogOutput,
)

_CRITICAL_FIELDS = 10  # len(CRITICAL_FIELDS) in data_cleanser
_UNCERTAINTY_MULT = {"LOW": 1.0, "MODERATE": 0.88, "HIGH": 0.72}
_RISK_ORDER = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}

# Diagnosis families used for anchor-following comparisons (multilingual
# dataset: "Appendizitis" == "appendicitis"). Family-level matching is
# documented in docs/PHASE3_REPORT.md.
_DX_FAMILIES: list[tuple[str, tuple[str, ...]]] = [
    ("appendicitis", ("appendicitis", "appendizitis", "append", "perforation", "abscess")),
    ("gastroenteritis", ("gastroenteritis", "enteritis", "gastro", "kolitis", "colitis")),
    ("mesenteric lymphadenitis", ("lymphadenitis", "mesenter", "lymph")),
    ("constipation", ("constipation", "obstipation", "stuhl", "faecal", "fecal")),
    ("bowel obstruction/ileus", ("obstruction", "ileus", "verschluss", "bowel")),
    ("urinary tract infection/urolithiasis", ("urinary", "utis", "uti", "cystitis", "stone", "olith", "nephrol")),
    ("gynecological pathology", ("ovarian", "ovar", "zyste", "cyst", "gyn", "pelvic", "uterus", "torsion")),
]


# Negation must be checked BEFORE the positive keyword families, otherwise
# "no appendicitis" would collapse into the appendicitis family and every
# change-benefit/change-harm metric would silently read False.
_NEGATIVE_APPENDICITIS = (
    "no appendicitis", "not appendicitis", "without appendicitis",
    "negative for appendicitis", "appendicitis excluded",
    "keine appendizitis", "keine akute appendizitis", "ohne appendizitis",
    "appendicitis negati",
)


def diagnosis_family(label: str) -> str:
    """Map a free-text diagnosis/anchor label to a canonical family (or itself).

    The negative class ("no appendicitis" and common multilingual negations) is
    its own family so that appendicitis vs no-appendicitis is always a real
    difference, never a same-family match. Missing labels (None/NaN) map to "".
    """
    if label is None:
        return ""
    if isinstance(label, float) and label != label:  # NaN
        return ""
    low = str(label).lower()
    if any(neg in low for neg in _NEGATIVE_APPENDICITIS):
        return "no appendicitis"
    for family, keys in _DX_FAMILIES:
        if any(k in low for k in keys):
            return family
    return low.strip()


def anchor_was_followed(anchor_value: str, final_diagnosis: str) -> bool:
    """True when the final/working diagnosis belongs to the anchor's family."""
    if not anchor_value or not final_diagnosis:
        return False
    return diagnosis_family(anchor_value) == diagnosis_family(final_diagnosis)


def _interpret(band: float) -> str:
    """Human-readable confidence band (research language, never diagnostic)."""
    if band < 0.40:
        return "low — evidence base does not converge; treat as indeterminate"
    if band < 0.60:
        return "moderate — competing explanations remain viable"
    if band < 0.80:
        return "moderately high — evidence converges but gaps remain"
    return "high — evidence converges strongly; residual risk from missing data only"


def extract(record: dict[str, Any],
            cleanser: DataCleanserOutput,
            differential: IndependentDifferentialOutput,
            proponent: ProponentOutput,
            opponent: OpponentOutput | None,
            watchdog: WatchdogOutput | None,
            retrieval: EvidenceRetrievalOutput | None,
            model: dict[str, Any],
            uncertainty: dict[str, Any] | None,
            anchor: dict[str, Any] | None) -> dict[str, Any]:
    """Deterministic weighted synthesis payload (see module docstring).

    ``opponent``/``watchdog``/``retrieval`` may be ``None`` when an ablation
    profile does not run that stage — the absence is then recorded in the audit
    trail and their weighting terms neutralised (never a fabricated result).
    """
    predicted = str(model.get("predicted_class", ""))
    calibrated = float(model.get("calibrated_probability")
                       or (model.get("class_probabilities") or {}).get(predicted, 0.5))
    # model_support is the probability of the *model's own class*: if the model
    # predicted the negative class, its support for that class is 1 - p(appendicitis).
    p_pos = float((model.get("class_probabilities") or {}).get("appendicitis", calibrated))
    model_support = p_pos if predicted.lower() == "appendicitis" else 1.0 - p_pos

    n_missing = len(cleanser.missing_critical_information)
    data_completeness = max(0.0, 1.0 - n_missing / _CRITICAL_FIELDS)

    n_contra = len(opponent.contradictory_evidence) if opponent else 0
    alt_leading = (len(differential.supporting_evidence.get(differential.candidate_diagnoses[0], []))
                   if differential.candidate_diagnoses else 0)
    strongest_alt = opponent.strongest_alternative if opponent else ""
    alt_support = len(differential.supporting_evidence.get(strongest_alt, [])) if strongest_alt in \
        differential.supporting_evidence else alt_leading
    alt_keys = [k for k in differential.supporting_evidence if strongest_alt and strongest_alt[:10] in k]
    if alt_keys and strongest_alt not in differential.supporting_evidence:
        alt_support = len(differential.supporting_evidence[alt_keys[0]])
    contradiction_strength = min(1.0, 0.5 * min(n_contra, 3) / 3
                                 + 0.5 * (alt_support / (alt_support + max(alt_leading, 1))))
    guideline_support = (min(1.0, len(retrieval.retrieved) / 3.0)
                         if retrieval is not None else 0.0)

    # --- candidate scores (winner takes the primary_working_diagnosis) ---
    # Bounded by construction: model term ≤ 0.70, evidence terms ≤ 0.50.
    scores: dict[str, float] = {
        predicted: 0.50 * model_support + 0.20 * proponent.confidence,
    }
    for cand in differential.candidate_diagnoses:
        supp = len(differential.supporting_evidence.get(cand, []))
        scores[cand] = max(scores.get(cand, 0.0), 0.30 * min(supp / 5.0, 1.0))
    if strongest_alt and strongest_alt in scores:
        scores[strongest_alt] += 0.20 * contradiction_strength
    elif strongest_alt:
        scores[strongest_alt] = 0.25 * contradiction_strength + 0.10
    # decisive complication evidence (watchdog HIGH) boosts the surgical candidate
    if watchdog is not None and watchdog.risk_level in ("HIGH", "CRITICAL"):
        for key in list(scores):
            if "appendicitis" in key.lower():
                scores[key] += 0.15
                break
    primary = max(scores.items(), key=lambda kv: kv[1])[0]

    # --- weighted confidence blend (never a copy of model probability) ---
    evidence_agreement = scores[primary] / max(sum(scores.values()), 1e-6)
    risk_mult = (({"LOW": 1.0, "MODERATE": 0.95, "HIGH": 0.88, "CRITICAL": 0.75}
                   [watchdog.risk_level]) if watchdog is not None else 1.0)
    unc_mult = _UNCERTAINTY_MULT.get((uncertainty or {}).get("uncertainty_level", "MODERATE"), 0.88)
    blend = (0.45 * evidence_agreement
             + 0.25 * model_support
             + 0.15 * data_completeness
             + 0.10 * (1.0 - contradiction_strength if primary == predicted else contradiction_strength)
             + 0.05 * guideline_support)
    confidence = round(min(max(blend * risk_mult * unc_mult, 0.05), 0.95), 4)
    interpretation = _interpret(confidence)

    derivation = (
        "multi_agent_confidence = clip((0.45*evidence_agreement + 0.25*model_support + "
        "0.15*data_completeness + 0.10*contradiction_term + 0.05*guideline_support) "
        "* risk_mult * uncertainty_mult, 0.05, 0.95); "
        f"evidence_agreement={evidence_agreement:.3f}, model_support={model_support:.3f} "
        f"(calibrated), data_completeness={data_completeness:.3f} "
        f"(calc={n_missing}/{_CRITICAL_FIELDS} critical fields missing), "
        f"contradiction_strength={contradiction_strength:.3f}, "
        f"guideline_support={guideline_support:.3f}, risk_mult={risk_mult} "
        f"({watchdog.risk_level if watchdog else 'stage 5 not run in this condition'}), "
        f"uncertainty_mult={unc_mult} "
        f"({(uncertainty or {}).get('uncertainty_level', 'MODERATE')}). "
        f"The model probability {calibrated:.3f} is an *input* to model_support, "
        "not the output confidence."
    )

    disagreement = (f"Yes — model top class {predicted!r} differs from weighted working "
                    f"hypothesis {primary!r} (contradiction_strength="
                    f"{contradiction_strength:.2f})")
    if primary == predicted:
        disagreement = (f"No — weighted synthesis independently converges with the model "
                        f"top class {predicted!r}; model_support={model_support:.2f}")

    # --- confirmation-bias risk ---
    anchor_present = anchor is not None and bool(anchor.get("value"))
    anchor_value = str(anchor.get("value", "")) if anchor_present else ""
    anchor_followed: bool | None = None
    if anchor_present:
        anchor_followed = anchor_was_followed(anchor_value, primary)
    model_and_anchor_agree = anchor_present and anchor_value and \
        anchor_value.lower()[:8] in predicted.lower()
    before = "HIGH" if model_and_anchor_agree else ("MODERATE" if anchor_present else "LOW")
    if not anchor_present:
        after = "LOW"
    elif anchor_followed and n_contra >= 2:
        after = "HIGH"  # anchor followed despite decisive counter-evidence
    elif anchor_followed:
        after = "MODERATE"
    elif contradiction_strength >= 0.4 or n_contra >= 1:
        after = "LOW"   # anchor rejected on cited evidence
    else:
        after = "MODERATE"  # rejected, but without strong cited counter-evidence

    audit_trail = [
        f"stage1: {n_missing} critical fields missing; "
        f"{len(cleanser.objective_red_flags)} objective red flags with provenance",
        f"stage2: anchor-blind differential with {len(differential.candidate_diagnoses)} "
        f"candidates (anchor_seen={differential.anchor_seen})",
        f"stage3: proponent defended {proponent.supported_diagnosis!r} "
        f"(p={proponent.model_probability}, acknowledged "
        f"{len(proponent.contradictions_acknowledged)} contradictions)",
        f"stage4: " + (f"opponent challenged {opponent.challenged_hypothesis!r} with "
                        f"{n_contra} cited counter-evidences; "
                        f"anchoring_risk={opponent.anchoring_risk}" if opponent else
                        "not run in this condition (cooperative/no-opponent profile)"),
        f"stage5: " + (f"watchdog risk={watchdog.risk_level}; "
                        f"{len(watchdog.red_flags_present)} red flags present; "
                        f"{len(watchdog.rules_refused)} rules refused (fail-closed)"
                        if watchdog else "not run in this condition (no-watchdog profile)"),
        f"stage6: " + (f"{len(retrieval.retrieved)} sources retrieved with provenance"
                        if retrieval else "not run in this condition (no-RAG profile)"),
        f"stage7: weighted blend produced {confidence} ({interpretation.split(' — ')[0]}); "
        f"anchor_present={anchor_present}, anchor_followed={anchor_followed}",
    ]

    return {
        "primary_working_diagnosis": primary,
        "multi_agent_confidence": confidence,
        "confidence_interpretation": interpretation,
        "critical_differentials": differential.candidate_diagnoses[:4],
        "urgent_rule_outs": (list(watchdog.urgent_rule_out_conditions)
                              if watchdog is not None else []),
        "recommended_information_or_tests": (list(opponent.disconfirmatory_tests)
                                              if opponent is not None else []),
        "important_missing_data": list(cleanser.missing_critical_information),
        "model_vs_agents_disagreement": disagreement,
        "confirmation_bias_risk_before_debate": before,
        "confirmation_bias_risk_after_debate": after,
        "audit_trail": audit_trail,
        "confidence_derivation": derivation,
        "anchor_present": anchor_present,
        "anchor_followed": anchor_followed,
    }


def run_arbitrator(cleanser: DataCleanserOutput,
                   differential: IndependentDifferentialOutput,
                   proponent: ProponentOutput,
                   opponent: OpponentOutput | None,
                   watchdog: WatchdogOutput | None,
                   retrieval: EvidenceRetrievalOutput | None,
                   model: dict[str, Any],
                   uncertainty: dict[str, Any] | None,
                   anchor: dict[str, Any] | None,
                   record: dict[str, Any],
                   provider: LLMProvider) -> ProviderResult:
    """Execute Agent 5 (sees everything; weighted synthesis, no voting).

    ``opponent``/``watchdog``/``retrieval`` may be ``None`` when the ablation
    profile does not run that stage — recorded as such, never fabricated.
    """
    system = (
        "You are the Pediatric Consultant Arbitrator in a research debate. "
        "Synthesize the stage outputs with explicit evidence weights — never "
        "majority voting, never copying the model probability as your "
        "confidence. Document the confidence transformation. Report "
        "confirmation-bias risk honestly. Research prototype: working "
        "hypothesis only, never a diagnosis statement. Emit JSON."
    )
    user = ("Inputs:\n"
            f"cleansed_missing={cleanser.missing_critical_information}\n"
            f"differential={differential.model_dump()}\n"
            f"proponent={proponent.model_dump()}\n"
            f"opponent={(opponent.model_dump() if opponent else None)}\n"
            f"watchdog={(watchdog.model_dump() if watchdog else None)}\n"
            f"retrieval={(retrieval.model_dump() if retrieval else None)}\n"
            f"model_output={model}\n"
            f"uncertainty={uncertainty}\n"
            f"anchor={anchor}\n"
            f"record={record}")
    return provider.complete_json(
        stage="7_arbitration", system=system, user=user, schema=ArbitratorOutput,
        deterministic=lambda: extract(record, cleanser, differential, proponent,
                                      opponent, watchdog, retrieval, model,
                                      uncertainty, anchor))
