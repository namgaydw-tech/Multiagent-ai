"""Cross-domain knowledge packs for the multi-agent engine (Phase 5, section K).

The 8-stage architecture, information isolation and schema validation are
unchanged; a **domain pack** supplies the disease-specific pieces the agents
need:

* cleanser column buckets + critical fields (what "the record" means here),
* differential candidates with supporting/contradictory evidence functions
  driven ONLY by values present in the record (never the label, never the
  anchor, never the model probability),
* domain red-flag screening rules (fail-closed on unverified knowledge),
* the canonical label space used for anchors / model outputs / ground truth so
  that diagnosis-family comparisons stay exact across domains.

``appendicitis`` packs reproduce the original Phase 3 behaviour byte-for-byte
(the legacy functions are re-used directly), so Phase 3/4 results and their
tests are untouched.

Research prototype — not a medical device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.agents.provider import LLMProvider  # noqa: F401  (typing)
from src.agents.schemas import EvidenceRef

# --------------------------------------------------------------------- helpers
def _has(record: dict, col: str, *values: str) -> bool:
    v = record.get(col)
    if v is None or (isinstance(v, float) and v != v):
        return False
    return str(v).strip().lower() in values


def _num(record: dict, col: str) -> float | None:
    v = record.get(col)
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v:
        return float(v)
    return None


# ---------------------------------------------------- neonatal sepsis evidence
def _supp_culture_pos(r: dict) -> list[str]:
    ev = []
    temp = _num(r, "Body_Temperature")
    if temp is not None and temp >= 38.0:
        ev.append(f"Body_Temperature={temp:.1f} degC (fever at sepsis evaluation)")
    if temp is not None and temp < 36.0:
        ev.append(f"Body_Temperature={temp:.1f} degC (hypothermia at sepsis evaluation)")
    ga = _num(r, "gestational_age_at_birth_weeks")
    if ga is not None and ga < 32:
        ev.append(f"gestational_age_at_birth_weeks={ga} (extremely preterm)")
    bw = _num(r, "birth_weight_kg")
    if bw is not None and bw < 1.5:
        ev.append(f"birth_weight_kg={bw} (very low birth weight)")
    if _has(r, "comorbidity_surgical", "1"):
        ev.append("comorbidity_surgical=1 (surgical comorbidity raises invasive-infection risk)")
    if _has(r, "comorbidity_ivh_or_shunt", "1"):
        ev.append("comorbidity_ivh_or_shunt=1 (CSF shunt/IVH raises infection risk)")
    if _has(r, "central_venous_line", "1"):
        ev.append("central_venous_line=1 (central access — device-associated infection risk)")
    if _has(r, "inotrope_at_time_of_sepsis_eval", "1"):
        ev.append("inotrope_at_time_of_sepsis_eval=1 (hemodynamic support at evaluation)")
    onset = _num(r, "onset_age_in_days")
    if onset is not None and onset <= 2:
        ev.append(f"onset_age_in_days={onset} (early-onset window)")
    return ev


def _contra_culture_pos(r: dict) -> list[str]:
    ev = []
    temp = _num(r, "Body_Temperature")
    if temp is not None and 36.0 <= temp < 38.0:
        ev.append(f"Body_Temperature={temp:.1f} degC (afebrile at evaluation)")
    if _has(r, "central_venous_line", "0"):
        ev.append("central_venous_line=0 (no central access)")
    onset = _num(r, "onset_age_in_days")
    if onset is not None and onset > 28:
        ev.append(f"onset_age_in_days={onset} (late-onset, well beyond early-onset window)")
    ga = _num(r, "gestational_age_at_birth_weeks")
    if ga is not None and ga >= 37:
        ev.append(f"gestational_age_at_birth_weeks={ga} (term gestation)")
    return ev


def _supp_culture_neg(r: dict) -> list[str]:
    ev = []
    temp = _num(r, "Body_Temperature")
    if temp is not None and 36.0 <= temp < 38.0:
        ev.append(f"Body_Temperature={temp:.1f} degC (normal temperature at evaluation)")
    if _has(r, "central_venous_line", "0") and _has(r, "inotrope_at_time_of_sepsis_eval", "0"):
        ev.append("no central line and no inotrope at evaluation")
    ga = _num(r, "gestational_age_at_birth_weeks")
    if ga is not None and ga >= 37:
        ev.append(f"gestational_age_at_birth_weeks={ga} (term infant)")
    ev.extend(_contra_culture_pos_terms(r))
    return ev


def _contra_culture_pos_terms(r: dict) -> list[str]:
    """Terms that argue AGAINST culture positivity (shared with the negative candidate)."""
    ev = []
    temp = _num(r, "Body_Temperature")
    if temp is not None and 36.0 <= temp < 38.0:
        ev.append("afebrile at evaluation")
    return list(dict.fromkeys(ev))  # de-duplicated


def _supp_noninfectious(r: dict) -> list[str]:
    ev = []
    if _has(r, "comorbidity_chronic_lung_disease", "1"):
        ev.append("comorbidity_chronic_lung_disease=1 (respiratory deterioration mimics sepsis)")
    if _has(r, "comorbidity_cardiac", "1"):
        ev.append("comorbidity_cardiac=1 (cardiac deterioration mimics sepsis)")
    if _has(r, "comorbidity_surgical", "1"):
        ev.append("comorbidity_surgical=1 (post-surgical inflammatory response)")
    if _has(r, "intubated_at_time_of_sepsis_eval", "1"):
        ev.append("intubated_at_time_of_sepsis_eval=1 (respiratory failure as primary process)")
    return ev


def _contra_noninfectious(r: dict) -> list[str]:
    ev = []
    temp = _num(r, "Body_Temperature")
    if temp is not None and (temp >= 38.0 or temp < 36.0):
        ev.append(f"Body_Temperature={temp:.1f} degC (temperature derangement favors infection)")
    return ev


# ------------------------------------------------------- pneumonia evidence
def _supp_bacterial(r: dict) -> list[str]:
    # image-only record: the only legitimate evidence is the model attribution itself,
    # which the proponent/arbitrator supply; the differential stays evidence-empty here.
    return []


def _contra_bacterial(r: dict) -> list[str]:
    return []


# ------------------------------------------------------------------- packs
@dataclass
class DomainPack:
    name: str
    display: str
    label_space: list[str]                      # canonical vocabulary (anchors/answers)
    target_id_to_label: dict[int, str]
    cleanser_buckets: dict[str, list[str]]
    critical_fields: list[str]
    candidates: list[tuple[str, Callable[[dict], list[str]], Callable[[dict], list[str]]]]
    red_flag_rules: list[str] = field(default_factory=list)
    record_aliases: dict[str, str] = field(default_factory=dict)  # domain col -> agent col
    knowledge_files: list[str] = field(default_factory=list)
    known_absent: list[str] = field(default_factory=list)


def _appendicitis_pack() -> DomainPack:
    from src.agents import data_cleanser as DC
    from src.agents import independent_differential as ID
    return DomainPack(
        name="appendicitis",
        display="Pediatric suspected appendicitis (Regensburg)",
        label_space=["appendicitis", "no appendicitis"],
        target_id_to_label={0: "no appendicitis", 1: "appendicitis"},
        cleanser_buckets={"demographics": DC.DEMOGRAPHICS, "vitals": DC.VITALS,
                          "laboratory": DC.LABORATORY, "symptoms": DC.SYMPTOMS,
                          "examination": DC.EXAMINATION, "imaging": DC.IMAGING},
        critical_fields=list(DC.CRITICAL_FIELDS),
        candidates=[(n, s, c) for n, s, c in ID._CANDIDATES],
        red_flag_rules=["appendicitis_complication_flags", "fever"],
        knowledge_files=["knowledge/pediatric_thresholds.yaml",
                         "knowledge/emergency_red_flags.yaml"],
        known_absent=list(ID._KNOWN_ABSENT_IN_DATASET),
    )


def _sepsis_pack() -> DomainPack:
    return DomainPack(
        name="neonatal_sepsis",
        display="Neonatal sepsis evaluation (culture positivity)",
        label_space=["culture-positive sepsis", "culture-negative sepsis evaluation",
                     "non-infectious clinical deterioration"],
        target_id_to_label={0: "culture-negative sepsis evaluation",
                            1: "culture-positive sepsis"},
        cleanser_buckets={
            "demographics": ["sex", "race", "gestational_age_at_birth_weeks",
                             "birth_weight_kg", "period"],
            "vitals": ["Body_Temperature"],
            "laboratory": [],
            "symptoms": ["onset_age_in_days", "onset_hour_of_day"],
            "examination": ["intubated_at_time_of_sepsis_eval",
                            "inotrope_at_time_of_sepsis_eval", "central_venous_line",
                            "umbilical_arterial_line", "ecmo"],
            "imaging": [],
            "comorbidities": ["comorbidity_necrotizing_enterocolitis",
                              "comorbidity_chronic_lung_disease", "comorbidity_cardiac",
                              "comorbidity_surgical", "comorbidity_ivh_or_shunt"],
        },
        critical_fields=["Body_Temperature", "gestational_age_at_birth_weeks",
                         "birth_weight_kg", "onset_age_in_days", "central_venous_line"],
        candidates=[
            ("culture-positive sepsis", _supp_culture_pos, _contra_culture_pos),
            ("culture-negative sepsis evaluation", _supp_culture_neg, _contra_culture_pos),
            ("non-infectious clinical deterioration", _supp_noninfectious,
             _contra_noninfectious),
        ],
        red_flag_rules=["fever", "hypothermia_unverified_fail_closed",
                        "phoenix_sepsis_2024 (pending transcription — fail closed)"],
        record_aliases={"Body_Temperature": "temp_celsius"},  # canonical <- domain column
        knowledge_files=["knowledge/pediatric_thresholds.yaml", "knowledge/sepsis_rules.yaml"],
        known_absent=["heart rate (no column)", "respiratory rate (no column)",
                      "blood pressure (no column)", "SpO2 (no column)",
                      "blood culture result timing (post-evaluation, target only)"],
    )


def _pneumonia_pack() -> DomainPack:
    return DomainPack(
        name="pediatric_pneumonia",
        display="Pediatric chest X-ray pneumonia (Kermany)",
        label_space=["normal chest radiograph", "bacterial pneumonia", "viral pneumonia"],
        target_id_to_label={0: "normal chest radiograph", 1: "bacterial pneumonia",
                            2: "viral pneumonia"},
        cleanser_buckets={
            "demographics": [],
            "vitals": [],
            "laboratory": [],
            "symptoms": [],
            "examination": [],
            "imaging": ["study_type", "modality", "view"],
        },
        critical_fields=["clinical_history", "temperature", "white_blood_cell_count",
                         "respiratory_rate"],
        candidates=[
            ("bacterial pneumonia", _supp_bacterial, _contra_bacterial),
            ("viral pneumonia", _supp_bacterial, _contra_bacterial),
            ("normal chest radiograph", _supp_bacterial, _contra_bacterial),
        ],
        red_flag_rules=["hypoxaemia (no SpO2 field — cannot assess)",
                        "respiratory distress (no work-of-breathing fields — cannot assess)"],
        knowledge_files=["knowledge/emergency_red_flags.yaml"],
        known_absent=["all clinical examination fields (image-only dataset)",
                      "fever, heart rate, respiratory rate, SpO2 (not present)"],
    )


_PACKS: dict[str, DomainPack] = {
    "appendicitis": _appendicitis_pack(),
    "neonatal_sepsis": _sepsis_pack(),
    "pediatric_pneumonia": _pneumonia_pack(),
}


def get_pack(domain: str) -> DomainPack:
    if domain not in _PACKS:
        raise KeyError(f"unknown domain {domain!r}; expected one of {sorted(_PACKS)}")
    return _PACKS[domain]


def known_domains() -> list[str]:
    return sorted(_PACKS)
