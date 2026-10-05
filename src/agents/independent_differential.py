"""Stage 2 — Independent Differential generation (Phase 3, section 3.4).

Generates a differential diagnosis from **sanitized patient evidence only**,
*before* any model prediction, probability, SHAP value or anchor is revealed.
This is the central anti-anchoring mechanism of the framework:

* the function signature accepts only the raw record (anchor/target columns
  already removed by the orchestrator) and the Agent 1 cleansing output;
* the schema carries an ``anchor_seen: False`` literal guard;
* the deterministic builder scores literature-standard pediatric abdominal-pain
  differentials against values **actually present in the record** — supporting
  evidence cites present values, contradictory evidence cites present values
  that argue against the candidate, and absent fields become
  ``missing_information`` rather than fabricated negative findings.

Research prototype — not a medical device; outputs are a research-system
working differential, not a diagnosis.
"""

from __future__ import annotations

from typing import Any

from src.agents.provider import LLMProvider, ProviderResult
from src.agents.schemas import DataCleanserOutput, IndependentDifferentialOutput


def _has(record: dict, col: str, *values: str) -> bool:
    """True when a categorical column holds one of the given values."""
    v = record.get(col)
    if v is None or (isinstance(v, float) and v != v):
        return False
    return str(v).strip().lower() in values


def _is_num(record: dict, col: str) -> bool:
    v = record.get(col)
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def _supp_appendicitis(r: dict) -> list[str]:
    ev = []
    if _has(r, "Migratory_Pain", "yes"):
        ev.append("Migratory_Pain=yes (migratory course typical of appendicitis)")
    if _has(r, "Lower_Right_Abd_Pain", "yes"):
        ev.append("Lower_Right_Abd_Pain=yes (localization to right lower quadrant)")
    if _has(r, "Ipsilateral_Rebound_Tenderness", "yes"):
        ev.append("Ipsilateral_Rebound_Tenderness=yes (peritoneal irritation)")
    if _has(r, "Psoas_Sign", "yes"):
        ev.append("Psoas_Sign=yes (retrocaecal irritation)")
    if _has(r, "Coughing_Pain", "yes"):
        ev.append("Coughing_Pain=yes (peritoneal irritation on strain)")
    if _has(r, "Loss_of_Appetite", "yes"):
        ev.append("Loss_of_Appetite=yes")
    if _has(r, "Nausea", "yes"):
        ev.append("Nausea=yes")
    wbc = r.get("WBC_Count")
    if _is_num(r, "WBC_Count") and float(wbc) >= 10.0:
        ev.append(f"WBC_Count={wbc} (elevated vs typical pediatric reference range)")
    crp = r.get("CRP")
    if _is_num(r, "CRP") and float(crp) > 5.0:
        ev.append(f"CRP={crp} (elevated inflammatory marker)")
    if _has(r, "Target_Sign", "yes") or _has(r, "Appendix_on_US", "yes"):
        ev.append("Ultrasound shows appendix target/visibility signs")
    if _has(r, "Surrounding_Tissue_Reaction", "yes"):
        ev.append("Surrounding_Tissue_Reaction=yes on ultrasound")
    return ev


def _contra_appendicitis(r: dict) -> list[str]:
    ev = []
    if _has(r, "Lower_Right_Abd_Pain", "no"):
        ev.append("Lower_Right_Abd_Pain=no (no localizing pain)")
    if _has(r, "Ipsilateral_Rebound_Tenderness", "no"):
        ev.append("Ipsilateral_Rebound_Tenderness=no")
    if _has(r, "Psoas_Sign", "no"):
        ev.append("Psoas_Sign=no")
    wbc, crp = r.get("WBC_Count"), r.get("CRP")
    if _is_num(r, "WBC_Count") and float(wbc) < 10.0 and _is_num(r, "CRP") and float(crp) <= 5.0:
        ev.append(f"WBC_Count={wbc} and CRP={crp} both non-inflammatory")
    if str(r.get("Stool", "")).lower() == "diarrhea":
        ev.append("Stool=diarrhea (points toward gastroenteritis)")
    return ev


def _supp_gastro(r: dict) -> list[str]:
    ev = []
    if str(r.get("Stool", "")).lower() in ("diarrhea", "constipation, diarrhea"):
        ev.append(f"Stool={r.get('Stool')} (enteric presentation)")
    if _has(r, "Nausea", "yes"):
        ev.append("Nausea=yes")
    if _has(r, "Lower_Right_Abd_Pain", "no"):
        ev.append("Lower_Right_Abd_Pain=no (no right-lower-quadrant localization)")
    if _has(r, "Enteritis", "yes"):
        ev.append("Enteritis=yes on ultrasound")
    return ev


def _contra_gastro(r: dict) -> list[str]:
    ev = []
    if _has(r, "Migratory_Pain", "yes"):
        ev.append("Migratory_Pain=yes (atypical for simple gastroenteritis)")
    if _has(r, "Ipsilateral_Rebound_Tenderness", "yes"):
        ev.append("Ipsilateral_Rebound_Tenderness=yes (peritoneal sign not expected)")
    if _has(r, "Peritonitis", "local", "generalized"):
        ev.append(f"Peritonitis={r.get('Peritonitis')} (peritoneal inflammation argues against)")
    return ev


def _supp_lymph(r: dict) -> list[str]:
    ev = []
    if _has(r, "Pathological_Lymph_Nodes", "yes"):
        ev.append("Pathological_Lymph_Nodes=yes on ultrasound (mesenteric lymphadenopathy)")
    if _has(r, "Loss_of_Appetite", "yes"):
        ev.append("Loss_of_Appetite=yes")
    return ev


def _contra_lymph(r: dict) -> list[str]:
    ev = []
    if _has(r, "Pathological_Lymph_Nodes", "no"):
        ev.append("Pathological_Lymph_Nodes=no on ultrasound")
    if _has(r, "Peritonitis", "generalized"):
        ev.append("Peritonitis=generalized (would expect surgical peritonitis)")
    return ev


def _supp_constipation(r: dict) -> list[str]:
    ev = []
    if str(r.get("Stool", "")).lower() == "constipation":
        ev.append("Stool=constipation")
    if _has(r, "Meteorism", "yes"):
        ev.append("Meteorism=yes on ultrasound")
    return ev


def _contra_constipation(r: dict) -> list[str]:
    ev = []
    if _has(r, "Peritonitis", "local", "generalized"):
        ev.append(f"Peritonitis={r.get('Peritonitis')} (not explained by constipation)")
    wbc = r.get("WBC_Count")
    if _is_num(r, "WBC_Count") and float(wbc) >= 15.0:
        ev.append(f"WBC_Count={wbc} markedly elevated (not explained by constipation)")
    return ev


def _supp_uti(r: dict) -> list[str]:
    ev = []
    if _has(r, "Dysuria", "yes"):
        ev.append("Dysuria=yes")
    if _has(r, "WBC_in_Urine", "yes"):
        ev.append("WBC_in_Urine=yes (urine leukocytes)")
    if _has(r, "RBC_in_Urine", "yes"):
        ev.append("RBC_in_Urine=yes")
    return ev


def _contra_uti(r: dict) -> list[str]:
    ev = []
    if _has(r, "Dysuria", "no"):
        ev.append("Dysuria=no")
    if _has(r, "Lower_Right_Abd_Pain", "yes") and _has(r, "Ipsilateral_Rebound_Tenderness", "yes"):
        ev.append("RLQ pain with rebound tenderness not explained by UTI")
    return ev


def _supp_gyn(r: dict) -> list[str]:
    ev = []
    sex = str(r.get("Sex", "")).lower()
    gyn = r.get("Gynecological_Findings")
    if sex == "female" and gyn is not None and str(gyn).strip() and str(gyn).strip().lower() != "no":
        ev.append(f"Sex=female with Gynecological_Findings={gyn} (recorded gynecological pathology)")
    if _has(r, "Free_Fluids", "yes") and sex == "female":
        ev.append("Free_Fluids=yes on ultrasound in a female patient")
    return ev


def _contra_gyn(r: dict) -> list[str]:
    ev = []
    sex = str(r.get("Sex", "")).lower()
    if sex == "male":
        ev.append("Sex=male (gynecological pathology not applicable)")
    gyn = r.get("Gynecological_Findings")
    if sex == "female" and (gyn is None or str(gyn).strip() == ""):
        ev.append("No gynecological findings recorded")
    return ev


def _supp_perf(r: dict) -> list[str]:
    ev = []
    if _has(r, "Perforation", "yes"):
        ev.append("Perforation=yes on ultrasound")
    if _has(r, "Appendicular_Abscess", "yes"):
        ev.append("Appendicular_Abscess=yes on ultrasound")
    if _has(r, "Peritonitis", "generalized"):
        ev.append("Peritonitis=generalized")
    if _has(r, "Peritonitis", "local"):
        ev.append("Peritonitis=local")
    return ev


def _contra_perf(r: dict) -> list[str]:
    ev = []
    if _has(r, "Perforation", "no") and _has(r, "Appendicular_Abscess", "no"):
        ev.append("Perforation=no and Appendicular_Abscess=no on ultrasound")
    if _has(r, "Peritonitis", "no"):
        ev.append("Peritonitis=no on examination")
    return ev


def _supp_bowel(r: dict) -> list[str]:
    ev = []
    for col in ("Ileus", "Bowel_Wall_Thickening", "Conglomerate_of_Bowel_Loops"):
        if _has(r, col, "yes"):
            ev.append(f"{col}=yes on ultrasound (bowel-process alternative)")
    if str(r.get("Stool", "")) == "constipation, diarrhea":
        ev.append("Stool=constipation, diarrhea (mixed bowel habit)")
    return ev


def _contra_bowel(r: dict) -> list[str]:
    ev = []
    if _has(r, "Target_Sign", "yes"):
        ev.append("Target_Sign=yes (appendiceal origin rather than generic bowel process)")
    return ev


_CANDIDATES: list[tuple[str, Any, Any]] = [
    # (name, supporting-evidence extractor, contradictory-evidence extractor)
    # Candidate differentials for pediatric suspected appendicitis (standard
    # abdominal-pain differential; evidence patterns reference record columns only).
    ("Acute appendicitis", _supp_appendicitis, _contra_appendicitis),
    ("Complicated appendicitis (perforation/abscess)", _supp_perf, _contra_perf),
    ("Gastroenteritis / enteritis", _supp_gastro, _contra_gastro),
    ("Mesenteric lymphadenitis", _supp_lymph, _contra_lymph),
    ("Constipation / fecal loading", _supp_constipation, _contra_constipation),
    ("Urinary tract infection / urolithiasis", _supp_uti, _contra_uti),
    ("Gynecological pathology (ovarian/uterine)", _supp_gyn, _contra_gyn),
    ("Bowel obstruction / ileus process", _supp_bowel, _contra_bowel),
]

_CANDIDATE_LIST = [name for name, _, _ in _CANDIDATES]

# Columns whose absence materially limits the differential (record-level fact:
# the Regensburg dataset has no respiratory rate / blood pressure / SpO2 fields).
_KNOWN_ABSENT_IN_DATASET = [
    "Respiratory rate (no column in dataset — WHO/IMCI tachypnea rule cannot be assessed)",
    "Blood pressure (no column in dataset — hypotension rule cannot be assessed)",
    "SpO2 (no column in dataset — hypoxaemia rule cannot be assessed)",
    "Heart rate (no column in dataset — tachycardia/bradycardia rules cannot be assessed)",
]


def _extract_domain(record: dict[str, Any], cleanser: Any, pack: Any) -> dict[str, Any]:
    """Domain-pack differential: candidates and evidence from the knowledge pack only."""
    if isinstance(cleanser, DataCleanserOutput):
        missing_critical = list(cleanser.missing_critical_information)
    else:
        missing_critical = list(cleanser.get("missing_critical_information", []))

    supporting: dict[str, list[str]] = {}
    contradictory: dict[str, list[str]] = {}
    scored: list[tuple[int, int, str]] = []
    for name, supp_fn, contra_fn in pack.candidates:
        supp = supp_fn(record)
        contra = contra_fn(record)
        supporting[name] = supp
        contradictory[name] = contra
        scored.append((len(supp) - len(contra), len(supp), name))
    scored.sort(reverse=True)
    top = [name for diff, supp_n, name in scored if supp_n > 0 and diff > 0][:4]
    if not top:  # never emit an empty differential (schema min_length=1)
        top = [name for _, _, name in scored[:2]] or [pack.label_space[0]]
    n_evidence = sum(len(v) for v in supporting.values())
    uncertainty = (
        f"Anchor-blind differential for domain '{pack.name}' from {n_evidence} "
        f"record-derived evidence statements; {len(missing_critical)} critical fields "
        "missing; no model prediction, probability, SHAP value or anchor was visible at "
        "this stage. Candidates ranked by (supporting − contradictory) evidence counts, "
        "not by model output."
    )
    return {
        "candidate_diagnoses": top,
        "supporting_evidence": {k: supporting[k] for k in top},
        "contradictory_evidence": {k: contradictory[k] for k in top},
        "uncertainty": uncertainty,
        "missing_information": [f"{c} missing from record" for c in missing_critical]
                                + list(pack.known_absent),
        "anchor_seen": False,
    }


def extract(record: dict[str, Any], cleanser: DataCleanserOutput | dict[str, Any],
            domain: str = "appendicitis") -> dict[str, Any]:
    """Deterministic anchor-blind differential built from record values only."""
    if domain != "appendicitis":
        from src.agents.domains import get_pack
        return _extract_domain(record, cleanser, get_pack(domain))
    if isinstance(cleanser, DataCleanserOutput):
        missing_critical = list(cleanser.missing_critical_information)
    else:
        missing_critical = list(cleanser.get("missing_critical_information", []))

    supporting: dict[str, list[str]] = {}
    contradictory: dict[str, list[str]] = {}
    scored: list[tuple[int, int, str]] = []
    for name, supp_fn, contra_fn in _CANDIDATES:
        supp = supp_fn(record)
        contra = contra_fn(record)
        supporting[name] = supp
        contradictory[name] = contra
        scored.append((len(supp) - len(contra), len(supp), name))
    scored.sort(reverse=True)
    top = [name for diff, supp_n, name in scored if supp_n > 0 and diff > 0][:4]
    if not top:  # never emit an empty differential (schema min_length=1)
        top = ["Acute appendicitis", "Gastroenteritis / enteritis"]
        supporting.setdefault("Acute appendicitis", [])
        supporting["Acute appendicitis"] = supporting.get("Acute appendicitis", [])
        contradictory.setdefault("Acute appendicitis", [])

    missing_info = [f"{c} missing from record" for c in missing_critical]
    missing_info += _KNOWN_ABSENT_IN_DATASET

    n_evidence = sum(len(v) for v in supporting.values())
    uncertainty = (
        f"Anchor-blind differential from {n_evidence} record-derived evidence statements; "
        f"{len(missing_critical)} critical fields missing; no model prediction, probability, "
        "SHAP value or anchor was visible at this stage. Candidates ranked by "
        "(supporting − contradictory) evidence counts, not by model output."
    )

    return {
        "candidate_diagnoses": top,
        "supporting_evidence": {k: supporting[k] for k in top},
        "contradictory_evidence": {k: contradictory[k] for k in top},
        "uncertainty": uncertainty,
        "missing_information": missing_info,
        "anchor_seen": False,
    }


def run_differential(record: dict[str, Any], cleanser: DataCleanserOutput,
                      provider: LLMProvider, domain: str = "appendicitis") -> ProviderResult:
    """Execute Stage 2 with strict information isolation.

    Only ``record`` (anchor/target removed upstream) and the Agent 1 output are
    passed in — no model fields exist in this call path by construction.
    """
    system = (
        "You are the Independent Differential stage in a pediatric appendicitis "
        "research pipeline. Generate a differential diagnosis from the supplied "
        "patient evidence ONLY. You have NOT seen any model prediction, "
        "probability, SHAP explanation or anchor. Cite only values present in "
        "the record; report absent fields as missing information. Emit JSON."
    )
    if domain != "appendicitis":
        from src.agents.domains import get_pack
        system = (f"You are the Independent Differential stage in a pediatric "
                  f"{get_pack(domain).display} research pipeline. Generate a differential "
                  "from the supplied patient evidence ONLY. You have NOT seen any model "
                  "prediction, probability, SHAP explanation or anchor. Cite only values "
                  "present in the record; report absent fields as missing information. "
                  "Emit JSON.")
    user = ("Patient evidence (sanitized, anchor-blind):\n"
            + str({"record": {k: v for k, v in sorted(record.items())},
                   "cleansed": cleanser.model_dump()}))
    return provider.complete_json(
        stage="2_independent_differential", system=system, user=user,
        schema=IndependentDifferentialOutput,
        deterministic=lambda: extract(record, cleanser, domain=domain))
