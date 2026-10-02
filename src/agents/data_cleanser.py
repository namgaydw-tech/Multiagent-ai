"""Agent 1 — Data Cleanser (Phase 3, section 3.3).

Objective extraction and standardization of the raw patient record. This agent
**never diagnoses**: the output schema carries a hard ``diagnoses_made: False``
guard, and the anchor/target columns are removed by the orchestrator before the
record reaches this stage.

Responsibilities
----------------
* group raw columns into demographics / vitals / laboratory / symptoms /
  physical findings / imaging findings using the audited feature manifest
  clinical groups (no re-design of the dataset);
* standardize string values (trim, normalize yes/no) without inventing data;
* preserve missingness — NaN values are *not* imputed, they are reported under
  ``missing_critical_information``;
* flag **objective red flags** only when a confirmed rule in
  ``knowledge/pediatric_thresholds.yaml`` or ``knowledge/emergency_red_flags.yaml``
  matches a value actually present in the record — every red flag carries its
  citation provenance; rules whose ``verification.status`` is
  ``pending_full_text_transcription`` are never fired (fail-closed).

LLM use: with the default ``deterministic`` backend the extraction itself is the
payload; with a configured LLM the same record is sent for language-level
normalization, still validated against the strict schema.

Research prototype — not a medical device.
"""

from __future__ import annotations

import json
import math
from typing import Any

import yaml

from src.agents.provider import LLMProvider, ProviderResult
from src.agents.schemas import DataCleanserOutput, EvidenceRef
from src.preprocessing.regensburg import ROOT

THRESHOLDS_PATH = ROOT / "knowledge/pediatric_thresholds.yaml"
RED_FLAGS_PATH = ROOT / "knowledge/emergency_red_flags.yaml"

# Column → bucket. Groups follow the audited dataset card / feature manifest.
DEMOGRAPHICS = ["Age", "Sex", "Height", "Weight", "BMI"]
VITALS = ["Body_Temperature"]
LABORATORY = ["WBC_Count", "Neutrophil_Percentage", "Segmented_Neutrophils",
              "Neutrophilia", "RBC_Count", "Hemoglobin", "RDW", "Thrombocyte_Count",
              "Ketones_in_Urine", "RBC_in_Urine", "WBC_in_Urine", "CRP",
              "Alvarado_Score", "Paedriatic_Appendicitis_Score"]
SYMPTOMS = ["Migratory_Pain", "Lower_Right_Abd_Pain", "Coughing_Pain", "Nausea",
            "Loss_of_Appetite", "Dysuria", "Stool"]
EXAMINATION = ["Peritonitis", "Psoas_Sign", "Ipsilateral_Rebound_Tenderness",
               "Contralateral_Rebound_Tenderness"]
IMAGING = ["US_Performed", "Appendix_on_US", "Appendix_Diameter", "Free_Fluids",
           "Appendix_Wall_Layers", "Target_Sign", "Appendicolith", "Perfusion",
           "Perforation", "Surrounding_Tissue_Reaction", "Appendicular_Abscess",
           "Abscess_Location", "Pathological_Lymph_Nodes", "Lymph_Nodes_Location",
           "Bowel_Wall_Thickening", "Conglomerate_of_Bowel_Loops", "Ileus",
           "Coprostasis", "Meteorism", "Enteritis", "Gynecological_Findings"]

# Critical information for the appendicitis question — absence is reported, never filled.
CRITICAL_FIELDS = ["Age", "Sex", "Body_Temperature", "WBC_Count", "CRP",
                   "Peritonitis", "US_Performed", "Migratory_Pain",
                   "Lower_Right_Abd_Pain", "Appendix_Diameter"]

_PLAUSIBLE_TEMP_C = (30.0, 43.0)
_YES = {"yes", "true", "ja"}


def _load_yaml(path):
    """Load a knowledge YAML file (single documented read point)."""
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _is_missing(value: Any) -> bool:
    """True for NaN/None/empty-string record values."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _standardize(value: Any) -> Any:
    """Trim strings and lowercase boolean-ish tokens; leave numbers untouched."""
    if isinstance(value, str):
        v = value.strip()
        low = v.lower()
        if low in ("yes", "no"):
            return low
        return v
    return value


def _build_prompt(record: dict[str, Any]) -> tuple[str, str]:
    """Role + record prompt (hashable; identical inputs → identical hash)."""
    system = (
        "You are Data Cleanser in a pediatric appendicitis research pipeline. "
        "Group and standardize the supplied record ONLY. Never diagnose, never "
        "invent values, never impute missing data. Emit the JSON schema provided."
    )
    user = "Record (anchor and target columns already removed):\n" + json.dumps(
        {k: (None if _is_missing(v) else _standardize(v)) for k, v in record.items()},
        default=str, sort_keys=True)
    return system, user


def _objective_red_flags(record: dict[str, Any]) -> list[EvidenceRef]:
    """Confirmed-rule red flags with provenance (fail-closed on pending rules)."""
    thr = _load_yaml(THRESHOLDS_PATH)
    flags = _load_yaml(RED_FLAGS_PATH)
    out: list[EvidenceRef] = []

    temp = record.get("Body_Temperature")
    if not _is_missing(temp):
        fever_rule = thr["thresholds"]["fever"]
        if float(temp) >= 38.0 and fever_rule["verification"]["status"] == "confirmed_from_authoritative_summary":
            out.append(EvidenceRef(
                kind="rule",
                citation=fever_rule["citation"],
                quote=f"Body_Temperature={float(temp):.1f} degC ≥ 38.0 degC",
                relevance="fever — objective red-flag screen input",
            ))

    comp = flags["red_flags"]["appendicitis_complication_flags"]
    if comp["verification"]["status"] != "confirmed_from_authoritative_summary":
        return out  # fail closed: no numeric firing from pending rules
    peritonitis = str(record.get("Peritonitis", "")).lower()
    if peritonitis == "generalized":
        out.append(EvidenceRef(kind="rule", citation=comp["citation"],
                               quote="Peritonitis=generalized",
                               relevance="generalized peritonitis — urgent surgical review"))
    perforation = str(record.get("Perforation", "")).lower()
    if perforation == "yes":
        out.append(EvidenceRef(kind="rule", citation=comp["citation"],
                               quote="Perforation=yes",
                               relevance="ultrasound perforation — urgent surgical review"))
    abscess = str(record.get("Appendicular_Abscess", "")).lower()
    if abscess == "yes":
        out.append(EvidenceRef(kind="rule", citation=comp["citation"],
                               quote="Appendicular_Abscess=yes",
                               relevance="palpable/ultrasound abscess — urgent surgical review"))
    return out


def _data_quality_warnings(record: dict[str, Any]) -> list[str]:
    """Recording anomalies — reported, never corrected in place."""
    warnings: list[str] = []
    temp = record.get("Body_Temperature")
    if not _is_missing(temp):
        t = float(temp)
        if not (_PLAUSIBLE_TEMP_C[0] <= t <= _PLAUSIBLE_TEMP_C[1]):
            warnings.append(
                f"Body_Temperature={t} degC outside plausible human range "
                f"{_PLAUSIBLE_TEMP_C[0]}–{_PLAUSIBLE_TEMP_C[1]} degC — verify recording/units; "
                "value retained as-is, not corrected")
    if str(record.get("US_Performed", "")).lower() == "no" and not _is_missing(record.get("Appendix_Diameter")):
        warnings.append("US_Performed=no but Appendix_Diameter present — source inconsistency retained")
    if str(record.get("Perforation", "")).lower() in ("suspected", "not excluded"):
        warnings.append(f"Perforation={record.get('Perforation')!r} is equivocal — "
                        "kept as stated, downstream agents must not treat it as confirmed")
    gyn = record.get("Gynecological_Findings")
    if isinstance(gyn, str) and gyn.strip() and any(ord(ch) > 127 for ch in gyn):
        warnings.append("Gynecological_Findings contains non-English free text — retained verbatim")
    return warnings


def extract(record: dict[str, Any]) -> dict[str, Any]:
    """Deterministic grouping/standardization/red-flag extraction payload.

    Uses only values present in ``record``; missing values are reported, never
    imputed. This is the payload used by the default ``deterministic`` backend
    and the fallback when an LLM fails schema validation.
    """

    def present(cols: list[str]) -> dict[str, Any]:
        return {c: _standardize(record[c]) for c in cols
                if c in record and not _is_missing(record[c])}

    missing = [f for f in CRITICAL_FIELDS if f not in record or _is_missing(record.get(f))]
    findings = []
    for c in EXAMINATION:
        if c in record and not _is_missing(record.get(c)):
            findings.append(f"{c}: {_standardize(record[c])}")
    imaging = []
    for c in IMAGING:
        if c in record and not _is_missing(record.get(c)):
            imaging.append(f"{c}: {_standardize(record[c])}")

    return {
        "demographics": present(DEMOGRAPHICS),
        "vitals": present(VITALS),
        "laboratory_values": present(LABORATORY),
        "symptoms": [f"{c}: {_standardize(record[c])}" for c in SYMPTOMS
                     if c in record and not _is_missing(record.get(c))],
        "physical_findings": findings,
        "imaging_findings": imaging,
        "missing_critical_information": missing,
        "objective_red_flags": [ref.model_dump() for ref in _objective_red_flags(record)],
        "data_quality_warnings": _data_quality_warnings(record),
        "diagnoses_made": False,
    }


def run_cleanser(record: dict[str, Any], provider: LLMProvider) -> ProviderResult:
    """Execute Agent 1 and validate against :class:`DataCleanserOutput`."""
    system, user = _build_prompt(record)
    return provider.complete_json(
        stage="1_data_cleansing", system=system, user=user,
        schema=DataCleanserOutput, deterministic=lambda: extract(record))
