"""Agent 4 — High-Acuity Watchdog (Phase 3, section 3.7).

Prediction-blind, rule-first safety screen over the record. Rules come only
from ``knowledge/emergency_red_flags.yaml``, ``knowledge/pediatric_thresholds.yaml``
and ``knowledge/sepsis_rules.yaml``:

* **fail-closed**: any rule whose ``verification.status`` is
  ``pending_full_text_transcription`` is *refused* (recorded under
  ``rules_refused``) and never fires numerically;
* **provenance required**: every present/absent flag carries the citation of
  the knowledge entry it came from — thresholds are never invented;
* **no catastrophic guessing**: red flags with no corresponding field in the
  dataset (rash, mental state, respiratory rate, …) go to
  ``cannot_assess_due_to_missing_data``, never to ``red_flags_absent``;
* **risk level** follows the knowledge file's own definition: CRITICAL only
  for immediate life-threatening findings *present in the record*, HIGH for any
  present red flag, MODERATE for concerning-but-non-emergent or equivocal
  findings, LOW only when assessable red flags are absent;
* an optional model prediction may be passed for ablation condition A — it is
  recorded (``prediction_seen=True``) but **never used by any rule**.

LLM interpretation is optional and may only *supplement* the rules; with the
default deterministic backend it does not run.

Research prototype — not a medical device.
"""

from __future__ import annotations

import math
from typing import Any

import yaml

from src.agents.provider import LLMProvider, ProviderResult
from src.agents.schemas import EvidenceRef, WatchdogOutput
from src.preprocessing.regensburg import ROOT

THRESHOLDS_PATH = ROOT / "knowledge/pediatric_thresholds.yaml"
RED_FLAGS_PATH = ROOT / "knowledge/emergency_red_flags.yaml"
SEPSIS_PATH = ROOT / "knowledge/sepsis_rules.yaml"


def _load(path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def _low(v: Any) -> str:
    return str(v).strip().lower() if not _missing(v) else ""


def _eval_red_flags(record: dict[str, Any]) -> tuple[list[EvidenceRef], list[str],
                                                     list[str], list[str]]:
    """Evaluate emergency red flags against the record.

    Returns (present_refs, absent, cannot_assess, refused_rules).
    """
    flags = _load(RED_FLAGS_PATH)["red_flags"]
    thr = _load(THRESHOLDS_PATH)["thresholds"]
    present: list[EvidenceRef] = []
    absent: list[str] = []
    cannot: list[str] = []
    refused: list[str] = []

    # --- appendicitis complication flags (record-level US/exam fields exist) ---
    comp = flags["appendicitis_complication_flags"]
    comp_ver = comp["verification"]["status"]
    if comp_ver == "confirmed_from_authoritative_summary":
        cite = comp["citation"]
        peritonitis = _low(record.get("Peritonitis"))
        if peritonitis == "generalized":
            present.append(EvidenceRef(kind="rule", citation=cite, quote="Peritonitis=generalized",
                                       relevance="generalized peritonitis — urgent surgical red flag"))
        elif peritonitis == "no":
            absent.append("generalized peritonitis (Peritonitis=no)")
        elif peritonitis == "local":
            absent.append("generalized peritonitis (Peritonitis=local — local peritonitis recorded, "
                          "a concerning but distinct finding)")
        else:
            cannot.append("peritonitis status (Peritonitis not recorded)")
        perf = _low(record.get("Perforation"))
        if perf == "yes":
            present.append(EvidenceRef(kind="rule", citation=cite, quote="Perforation=yes",
                                       relevance="ultrasound-confirmed perforation — urgent surgical red flag"))
        elif perf == "no":
            absent.append("perforation (Perforation=no)")
        elif perf in ("suspected", "not excluded"):
            cannot.append(f"perforation (equivocal: Perforation={record.get('Perforation')})")
        else:
            cannot.append("perforation (Perforation not recorded)")
        abscess = _low(record.get("Appendicular_Abscess"))
        if abscess == "yes":
            present.append(EvidenceRef(kind="rule", citation=cite, quote="Appendicular_Abscess=yes",
                                       relevance="ultrasound-confirmed abscess — urgent surgical red flag"))
        elif abscess == "no":
            absent.append("appendicular abscess (Appendicular_Abscess=no)")
        else:
            cannot.append("appendicular abscess (Appendicular_Abscess not recorded)")
    else:
        refused.append("appendicitis_complication_flags (verification not confirmed — fail closed)")

    # --- red flags with no corresponding dataset field: never fabricated ---
    for key, field_note in (
        ("petechial_or_purpuric_rash_with_fever", "non-blanching/petechial rash (no rash field in dataset)"),
        ("meningeal_signs", "meningeal signs (no neck-stiffness/photophobia field in dataset)"),
        ("imci_danger_signs", "IMCI danger signs (no drinking/lethargy/convulsion fields in dataset)"),
        ("kawasaki_disease_suspicion", "Kawasaki features (no mucosal/conjunctival/lymphadenopathy fields in dataset)"),
    ):
        rf = flags[key]
        if rf["verification"]["status"] == "confirmed_from_authoritative_summary":
            cannot.append(field_note)
        else:
            refused.append(key)

    # --- fever (threshold knowledge, confirmed) ---
    fever_rule = thr["fever"]
    temp = record.get("Body_Temperature")
    if fever_rule["verification"]["status"] != "confirmed_from_authoritative_summary":
        refused.append("fever")
    elif not _missing(temp):
        if float(temp) >= 38.0:
            present.append(EvidenceRef(kind="rule", citation=fever_rule["citation"],
                                       quote=f"Body_Temperature={float(temp):.1f} degC ≥ 38.0",
                                       relevance="fever (NICE CG160 threshold)"))
        else:
            absent.append(f"fever (Body_Temperature={float(temp):.1f} degC < 38.0)")
    else:
        cannot.append("fever (Body_Temperature not recorded)")

    # --- fields absent from the dataset entirely ---
    cannot.extend([
        "hypoxaemia (no SpO2 column in dataset)",
        "tachypnea/WHO-IMCI fast breathing (no respiratory rate column in dataset)",
        "bradycardia with poor perfusion (no heart rate column in dataset)",
        "severe dehydration signs (no lethargy/skin-turgor/drinking fields in dataset)",
    ])
    return present, absent, cannot, refused


def _eval_thresholds_and_sepsis(record: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Fail-closed evaluation of pending threshold/sepsis rules.

    Returns (refused_rules, urgent_rule_outs, evidence_notes).
    """
    thr = _load(THRESHOLDS_PATH)["thresholds"]
    sepsis = _load(SEPSIS_PATH)
    refused: list[str] = []
    urgent: list[str] = []
    notes: list[str] = []

    for key in ("hypotension_sbp_by_age", "tachycardia_by_age"):
        if thr[key]["verification"]["status"] == "pending_full_text_transcription":
            refused.append(f"{key} (pending_full_text_transcription — fail closed)")

    phoenix = sepsis["rule_sets"]["phoenix_sepsis_2024"]
    if phoenix["verification"]["status"] == "pending_full_text_transcription":
        refused.append("phoenix_sepsis_2024 (component score tables pending transcription — fail closed)")

    # Infection-suspicion gate for the sepsis rule-out (evidence required by
    # watchdog_policy.trigger_evidence_required — never fabricated).
    infection_suspicion = (
        _low(record.get("Peritonitis")) in ("local", "generalized")
        or _low(record.get("Appendicular_Abscess")) in ("suspected", "yes")
        or _low(record.get("Perforation")) in ("suspected", "yes", "not excluded")
        or (not _missing(record.get("Body_Temperature")) and float(record["Body_Temperature"]) >= 38.0)
    )
    if infection_suspicion:
        urgent.append(
            "sepsis — cannot be scored: Phoenix 2024 components pending transcription "
            "(fail closed) and heart rate/respiratory rate absent from the dataset; "
            "infection-suspicion evidence present, urgent clinical rule-out required")
        notes.append("Sepsis rule-out triggered by infection-suspicion evidence only "
                     "(peritonitis/abscess/perforation/fever present in record); "
                     "no sepsis alert was computed from model output.")
    return refused, urgent, notes


def _extract_domain(record: dict[str, Any], prediction: dict[str, Any] | None,
                     domain: str) -> dict[str, Any]:
    """Domain-pack watchdog: only verified knowledge rules; fail-closed elsewhere."""
    from src.agents.domains import get_pack
    pack = get_pack(domain)
    present: list[EvidenceRef] = []
    absent: list[str] = []
    cannot: list[str] = []
    refused: list[str] = []

    thr = _load(THRESHOLDS_PATH)["thresholds"]
    fever_rule = thr["fever"]
    temp = record.get("Body_Temperature")
    if fever_rule["verification"]["status"] != "confirmed_from_authoritative_summary":
        refused.append("fever")
    elif not _missing(temp):
        if float(temp) >= 38.0:
            present.append(EvidenceRef(kind="rule", citation=fever_rule["citation"],
                                       quote=f"Body_Temperature={float(temp):.1f} degC ≥ 38.0",
                                       relevance="fever (NICE CG160 threshold)"))
        else:
            absent.append(f"fever (Body_Temperature={float(temp):.1f} degC < 38.0)")
            if float(temp) < 36.0:
                cannot.append(f"hypothermia (Body_Temperature={float(temp):.1f} degC < 36.0 — "
                              "no confirmed neonatal hypothermia threshold rule in the "
                              "knowledge files, fail closed)")
    else:
        cannot.append("fever (Body_Temperature not recorded)")

    sepsis = _load(SEPSIS_PATH)
    phoenix = sepsis["rule_sets"]["phoenix_sepsis_2024"]
    if phoenix["verification"]["status"] == "pending_full_text_transcription":
        refused.append("phoenix_sepsis_2024 (component score tables pending transcription — "
                       "fail closed)")
    cannot.extend(pack.known_absent)

    urgent: list[str] = []
    notes: list[str] = []
    if not _missing(temp) and (float(temp) >= 38.0 or float(temp) < 36.0):
        urgent.append("sepsis — cannot be scored: Phoenix 2024 components pending "
                      "transcription (fail closed); temperature derangement present, "
                      "urgent clinical rule-out required")
        notes.append("Sepsis rule-out triggered by temperature evidence only; no sepsis "
                     "alert was computed from model output.")

    conditions = list(pack.red_flag_rules) + ["fever (NICE CG160)"]
    fever_refs = [r for r in present if "fever" in r.relevance.lower()]
    risk = "MODERATE" if (fever_refs or urgent) else "LOW"
    evidence = list(present) + [
        EvidenceRef(kind="rule", citation="knowledge/pediatric_thresholds.yaml + sepsis_rules.yaml",
                    quote=n, relevance="rule-screen note") for n in notes]
    return {
        "high_acuity_conditions_considered": conditions,
        "red_flags_present": [r.model_dump() for r in present],
        "red_flags_absent": absent,
        "cannot_assess_due_to_missing_data": cannot,
        "urgent_rule_out_conditions": urgent,
        "risk_level": risk,
        "evidence": [e.model_dump() for e in evidence],
        "prediction_seen": prediction is not None,
        "rules_applied": sorted({"fever",
                                 "infection-suspicion gate (watchdog_policy."
                                 "trigger_evidence_required)"} - set(refused)),
        "rules_refused": sorted(set(refused)),
    }


def extract(record: dict[str, Any], prediction: dict[str, Any] | None = None,
            domain: str = "appendicitis") -> dict[str, Any]:
    """Deterministic watchdog payload (rules only; prediction never consulted)."""
    if domain != "appendicitis":
        return _extract_domain(record, prediction, domain)
    present, absent, cannot, refused = _eval_red_flags(record)
    refused2, urgent, notes = _eval_thresholds_and_sepsis(record)

    conditions_considered = [
        "appendicitis complications (perforation/abscess/peritonitis)",
        "fever (NICE CG160)",
        "sepsis (Phoenix 2024 / IPSCC 2005)",
        "meningococcal disease (NICE CG160)",
        "IMCI general danger signs",
        "Kawasaki disease (AHA)",
        "hypoxaemia, tachypnea, hypotension, tachycardia, bradycardia, severe dehydration",
    ]
    rule_outs = list(urgent)
    perf = _low(record.get("Perforation"))
    if perf in ("suspected", "not excluded"):
        rule_outs.append(f"appendicitis with perforation (equivocal: Perforation="
                         f"{record.get('Perforation')}) — urgent surgical review")
    if _low(record.get("Peritonitis")) == "local":
        rule_outs.append("focal peritonitis (Peritonitis=local) — urgent surgical review")

    all_refused = refused + refused2
    complication_refs = [r for r in present if "urgent surgical red flag" in r.relevance]
    fever_refs = [r for r in present if "fever" in r.relevance.lower()]
    equivocal = any("equivocal" in c for c in cannot)

    # Risk tiers follow knowledge/emergency_red_flags.yaml `risk_levels`:
    # HIGH only for present complication red flags; fever/equivocal findings/
    # infection-suspicion rule-outs are MODERATE; LOW requires no present red
    # flag (unobservable flags stay in cannot_assess and never lower risk silently).
    if complication_refs:
        risk = "HIGH"
    elif fever_refs or equivocal or urgent or _low(record.get("Peritonitis")) == "local":
        risk = "MODERATE"
    else:
        risk = "LOW"

    evidence = list(present) + [
        EvidenceRef(kind="rule", citation="knowledge/pediatric_thresholds.yaml + sepsis_rules.yaml",
                    quote=n, relevance="rule-screen note")
        for n in notes
    ]
    return {
        "high_acuity_conditions_considered": conditions_considered,
        "red_flags_present": [r.model_dump() for r in present],
        "red_flags_absent": absent,
        "cannot_assess_due_to_missing_data": cannot,
        "urgent_rule_out_conditions": rule_outs,
        "risk_level": risk,
        "evidence": [e.model_dump() for e in evidence],
        "prediction_seen": prediction is not None,
        "rules_applied": sorted({
            "appendicitis_complication_flags", "fever",
            "infection-suspicion gate (watchdog_policy.trigger_evidence_required)",
        } - set(all_refused)),
        "rules_refused": sorted(set(all_refused)),
    }


def run_watchdog(record: dict[str, Any],
                 provider: LLMProvider,
                 prediction: dict[str, Any] | None = None,
                 domain: str = "appendicitis") -> ProviderResult:
    """Execute Agent 4 (prediction-blind unless an ablation passes ``prediction``)."""
    system = (
        "You are the High-Acuity Watchdog in a pediatric appendicitis research "
        "pipeline. Apply ONLY the supplied knowledge rules to the supplied "
        "record. Fail closed on pending rules. Never fabricate findings; "
        "unobservable red flags go to cannot_assess. Do not label a case "
        "critical without record evidence. Emit JSON."
    )
    user = ("Record:\n" + str({k: v for k, v in sorted(record.items()) if k != "Diagnosis"}))
    if prediction is not None:
        user += ("\n\nABLATION: model prediction below is provided for logging only; "
                 f"no rule may depend on it:\n{prediction}")
    return provider.complete_json(
        stage="5_high_acuity_watchdog", system=system, user=user,
        schema=WatchdogOutput,
        deterministic=lambda: extract(record, prediction, domain=domain))
