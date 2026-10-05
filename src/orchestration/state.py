"""Case state + visibility policy for the staged orchestrator (Phase 3, section 3.10).

``CaseState`` carries everything a run needs, but each stage only ever receives
the slice permitted by :class:`VisibilityPolicy` — information isolation is a
data-flow property, not a prompt instruction:

* ``record``      — raw features with anchor/target/identifier columns removed
                    by ``strip_record()`` before the run starts;
* ``model_output``/``shap_explain``/``uncertainty`` — Layer-A artifacts,
                    hidden from stages 1, 2 and 5 by default (condition C);
* ``anchor``      — experimental anchor (natural or synthetic), hidden from
                    stages 1, 2 and 5, revealed to 3/4-from-4/7/8;
* ``ground_truth`` — target label, visible **only** to stage 8 (bias audit).

Default = ablation condition C (``config/agents.yaml``:
``C_hidden_until_differential_complete``). Ablations are config-only overrides
of this policy — never code forks.

Research prototype — not a medical device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Columns never shown to any agent (target / anchor / identifier).
RESERVED_COLUMNS = ("Diagnosis", "Diagnosis_Presumptive", "US_Number",
                    "Management", "Severity", "Length_of_Stay")

STAGE_ORDER: list[tuple[int, str, str]] = [
    (1, "1_data_cleansing", "Data cleansing"),
    (2, "2_independent_differential", "Independent differential"),
    (3, "3_proponent", "Proponent"),
    (4, "4_opponent", "Opponent"),
    (5, "5_high_acuity_watchdog", "High-acuity watchdog"),
    (6, "6_evidence_retrieval", "Evidence retrieval"),
    (7, "7_arbitration", "Arbitration"),
    (8, "8_bias_audit", "Bias audit"),
]


def strip_record(record: dict[str, Any]) -> dict[str, Any]:
    """Remove target/anchor/identifier columns before any agent sees the record."""
    return {k: v for k, v in record.items() if k not in RESERVED_COLUMNS}


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and v != v:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def clean_json(value: Any) -> Any:
    """Make a value JSON-serializable (NaN → None) for stage snapshots."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and value != value:
        return None
    if hasattr(value, "model_dump"):  # pydantic v2
        return clean_json(value.model_dump())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class VisibilityPolicy:
    """What each stage may receive (per-stage booleans; ablations override)."""

    stage_prediction: dict[int, bool] = field(default_factory=lambda: {
        1: False, 2: False, 3: True, 4: True, 5: False, 6: False, 7: True, 8: True,
    })
    stage_anchor: dict[int, bool] = field(default_factory=lambda: {
        1: False, 2: False, 3: True, 4: True, 5: False, 6: False, 7: True, 8: True,
    })
    stage_ground_truth: dict[int, bool] = field(default_factory=lambda: {
        i: (i == 8) for i in range(1, 9)
    })
    opponent_stage_a_sees_prediction: bool = False  # ablation A overrides to True

    @classmethod
    def from_ablation(cls, ablation: str | None) -> "VisibilityPolicy":
        """Config-only ablation switch (docs/ARCHITECTURE.md table)."""
        pol = cls()
        if ablation in ("A_all_see_prediction", "A"):
            # NOTE: stage 2 (independent differential) stays prediction-blind in EVERY
            # condition — docs/ARCHITECTURE.md marks it hidden under A and B, and its
            # independence is the measurement baseline. "All see" = proponent,
            # opponent, watchdog, arbitrator (+ opponent stage A reveal flag).
            pol.stage_prediction.update({4: True, 5: True})
            pol.opponent_stage_a_sees_prediction = True
        elif ablation in ("B_only_proponent", "B"):
            pol.stage_prediction.update({4: False, 5: False})
        elif ablation in ("C_hidden_until_differential_complete", "C", None, "default"):
            pass
        elif ablation in ("all_see_anchor",):
            # Stage 2 never sees the anchor either (anchor_seen: Literal[False] guard):
            # the independent differential is the unbiased reference for CBR/AOR.
            pol.stage_anchor.update({1: True, 5: True, 6: True})
        return pol

    def description(self, stage: int) -> dict[str, Any]:
        """Machine-readable isolation statement for the stage snapshot/UI."""
        return {
            "sees_prediction": self.stage_prediction.get(stage, False),
            "sees_anchor": self.stage_anchor.get(stage, False),
            "sees_ground_truth": self.stage_ground_truth.get(stage, False),
        }


@dataclass
class CaseState:
    """All data for one end-to-end case run (see module docstring)."""

    case_id: str
    record: dict[str, Any]                      # already stripped of reserved columns
    domain: str = "appendicitis"               # Phase 5 knowledge pack (src/agents/domains.py)
    ground_truth: str | None = None             # stage 8 only
    anchor: dict[str, Any] | None = None        # {value, confidence, source, method}
    model_output: dict[str, Any] | None = None  # normalized prediction interface dict
    shap_explain: dict[str, Any] | None = None
    uncertainty: dict[str, Any] | None = None
    config_hash: str = ""
    seed: int = 20261002
    ablation: str = "default"
    pipeline_stages: tuple[int, ...] | None = None  # ablation profiles; None = stages 1-8
    rag_enabled: bool = True
    policy: VisibilityPolicy = field(default_factory=VisibilityPolicy)
    stage_outputs: dict[str, Any] = field(default_factory=dict)
    stage_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    stage_inputs: dict[str, Any] = field(default_factory=dict)
    completed_stages: list[int] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    def stage_slice(self, stage: int) -> dict[str, Any]:
        """Exactly what stage ``stage`` is allowed to receive (isolation core)."""
        sl: dict[str, Any] = {"case_id": self.case_id, "record": self.record}
        if self.policy.stage_prediction.get(stage):
            sl["model_output"] = self.model_output
            sl["shap_explain"] = self.shap_explain
            sl["uncertainty"] = self.uncertainty
        if self.policy.stage_anchor.get(stage):
            sl["anchor"] = self.anchor
        if self.policy.stage_ground_truth.get(stage):
            sl["ground_truth"] = self.ground_truth
        sl["_visibility"] = self.policy.description(stage)
        sl["_withheld"] = sorted(
            ({"model_output", "shap_explain", "uncertainty"}
             if not self.policy.stage_prediction.get(stage) else set())
            | ({"anchor"} if not self.policy.stage_anchor.get(stage) else set())
            | ({"ground_truth"} if not self.policy.stage_ground_truth.get(stage) else set())
        )
        return clean_json(sl)


def make_state(case_id: str, record: dict[str, Any], **kwargs: Any) -> CaseState:
    """Build a state from a *raw* record (strips reserved columns itself)."""
    raw = dict(record)
    gt = kwargs.pop("ground_truth", None)
    if gt is None and "Diagnosis" in raw:
        gt = raw.get("Diagnosis")
    # the visibility policy is derived from the ablation label unless overridden
    kwargs.setdefault("policy", VisibilityPolicy.from_ablation(kwargs.get("ablation")))
    return CaseState(case_id=case_id, record=strip_record(raw), ground_truth=gt, **kwargs)
