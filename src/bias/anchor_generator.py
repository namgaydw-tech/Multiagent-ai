"""Anchor injection for the confirmation-bias experiments (Phase 4, section 4.1).

Seven conditions from ``config/experiment.yaml``:

===========================  ==========================================================
condition                    anchor injected
===========================  ==========================================================
control                      none
correct_anchor               preliminary diagnosis that matches ground truth
incorrect_anchor             preliminary diagnosis that contradicts ground truth
high_confidence_incorrect    same incorrect anchor, assertive, confidence 0.95
low_confidence_incorrect     same incorrect anchor, tentative, confidence 0.55
model_anchor                 the LightGBM top prediction presented as the anchor
no_model_anchor              none; model revealed only after the differential
                             (default isolation C — verifies no pre-stage-2 leak)
===========================  ==========================================================

Anchor sources (config ``anchor_sources``):

* **observed** — the real ``Diagnosis_Presumptive`` value of the same case
  (a genuine clinical preliminary diagnosis, the natural anchor);
* **synthetic** — a *label-swapped* **real** ``Diagnosis_Presumptive`` value taken
  from another case with the appropriate family correctness. Plausible by
  construction because it comes from the dataset's own vocabulary; anchor strings
  are never invented by this code.

Hard policy: an anchor is a **diagnosis-level statement only** — this module
returns ``None`` or an anchor dict and never mutates patient features, so all
paired conditions see byte-identical clinical records (the experiment's core
paired-design requirement).

Deterministic: selection depends only on ``seed``, condition name and row index.

Research prototype — not a medical device.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import pandas as pd

from src.agents.arbitrator import diagnosis_family

SEED = 20261002

ANCHOR_CONDITIONS: tuple[str, ...] = (
    "control",
    "correct_anchor",
    "incorrect_anchor",
    "high_confidence_incorrect_anchor",
    "low_confidence_incorrect_anchor",
    "model_anchor",
    "no_model_anchor",
)

# confidence/phrasing per condition (config: confidence high/low, phrasing assertive/tentative)
_CONFIDENCE = {
    "correct_anchor": 0.80,
    "incorrect_anchor": 0.80,
    "high_confidence_incorrect_anchor": 0.95,
    "low_confidence_incorrect_anchor": 0.55,
}
_PHRASING = {
    "correct_anchor": "neutral",
    "incorrect_anchor": "neutral",
    "high_confidence_incorrect_anchor": "assertive",
    "low_confidence_incorrect_anchor": "tentative",
}


def _rng(*parts: Any) -> random.Random:
    """Deterministic RNG keyed by seed + parts (stable across processes)."""
    key = "|".join(str(p) for p in (SEED,) + parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def anchor_vocabulary(df: pd.DataFrame) -> dict[str, list[str]]:
    """Real ``Diagnosis_Presumptive`` values grouped by diagnosis family.

    This is the synthetic-anchor pool: every synthetic anchor is a genuine
    presumptive diagnosis recorded for some case in the permitted research
    dataset (label swap, never an invented string).
    """
    vocab: dict[str, list[str]] = {}
    for value in df["Diagnosis_Presumptive"].dropna().astype(str).unique():
        family = diagnosis_family(value)
        if not family:
            continue
        vocab.setdefault(family, []).append(value)
    return {fam: sorted(vals) for fam, vals in vocab.items()}


def _pick(family: str, vocab: dict[str, list[str]], *rng_key: Any) -> str | None:
    """Deterministic pick of a real presumptive value from ``family``."""
    pool = vocab.get(family) or []
    if not pool:
        return None
    return pool[_rng(*rng_key).randrange(len(pool))]


def natural_anchor(row: pd.Series) -> dict[str, Any] | None:
    """Observed anchor for one case (its own preliminary diagnosis), if present."""
    value = row.get("Diagnosis_Presumptive")
    if value is None or (isinstance(value, float) and value != value):
        return None
    return {"value": str(value), "confidence": 0.80, "source": "observed",
            "method": "Diagnosis_Presumptive (natural preliminary diagnosis)"}


def anchor_for_case(condition: str,
                    row_index: int,
                    df: pd.DataFrame,
                    ground_truth: str,
                    model_top: str,
                    vocab: dict[str, list[str]] | None = None) -> dict[str, Any] | None:
    """Build the anchor dict for one case under one condition (or ``None``).

    Parameters
    ----------
    condition: one of :data:`ANCHOR_CONDITIONS`.
    row_index: dataset row index (also seeds the deterministic synthetic pick).
    df: full audited dataframe (used for ``Diagnosis_Presumptive`` / vocabulary).
    ground_truth: the case's true label (family space).
    model_top: the LightGBM top class for the ``model_anchor`` condition.
    vocab: pre-computed :func:`anchor_vocabulary` (computed on first use).
    """
    if condition not in ANCHOR_CONDITIONS:
        raise ValueError(f"unknown anchor condition {condition!r}; "
                         f"expected one of {ANCHOR_CONDITIONS}")
    if condition in ("control", "no_model_anchor"):
        # no_model_anchor: model reveal timing is handled by the visibility policy
        # (default C reveals after stage 2) — recorded, never an anchor value.
        return None

    gt_family = diagnosis_family(ground_truth)
    natural = natural_anchor(df.loc[row_index])
    natural_family = diagnosis_family(natural["value"]) if natural else ""
    natural_correct = bool(natural) and natural_family == gt_family
    vocab = vocab if vocab is not None else anchor_vocabulary(df)

    if condition == "model_anchor":
        p = model_top
        return {"value": p, "confidence": 0.80, "source": "model",
                "method": "lightgbm_top_prediction (config: visible_to proponent+arbitrator)"}

    if condition == "correct_anchor":
        if natural_correct:
            anchor = dict(natural)
            anchor.update(condition=condition, correct=True, phrasing="neutral")
            return anchor
        value = _pick(gt_family, vocab, "correct", row_index)
        if value is None:            # no real presumptive value for this family
            return None
        return {"value": value, "confidence": _CONFIDENCE[condition],
                "source": "synthetic",
                "method": "label_swapped_real_preliminary_diagnosis (correct family)",
                "condition": condition, "correct": True, "phrasing": "neutral"}

    # incorrect-anchor family (incl. high/low confidence variants): pick the
    # largest wrong-family vocabulary (ties broken alphabetically) so synthetic
    # incorrect anchors are the most plausible real alternative for this ground
    # truth — and the SAME value is reused across the confidence variants
    # (paired design), because the rng key excludes the condition name.
    wrong_families = sorted(
        (fam for fam in vocab if fam and fam != gt_family),
        key=lambda fam: (-len(vocab[fam]), fam))
    incorrect_family = wrong_families[0] if wrong_families else None
    if natural and not natural_correct:
        anchor = dict(natural)
        source, value = "observed", natural["value"]
    else:
        value = _pick(incorrect_family or "", vocab, "incorrect", row_index)
        source = "synthetic"
        if value is None:
            return None
    confidence = _CONFIDENCE.get(condition, 0.80)
    return {"value": value, "confidence": confidence, "source": source,
            "method": ("Diagnosis_Presumptive (natural preliminary diagnosis)"
                       if source == "observed" else
                       "label_swapped_real_preliminary_diagnosis (incorrect family)"),
            "condition": condition, "correct": False,
            "phrasing": _PHRASING.get(condition, "neutral")}


def anchor_summary(anchor: dict[str, Any] | None) -> dict[str, Any]:
    """Compact anchor description for audit rows (never the raw record)."""
    if not anchor:
        return {"present": False}
    return {"present": True, "value": anchor.get("value"),
            "source": anchor.get("source"), "confidence": anchor.get("confidence"),
            "phrasing": anchor.get("phrasing"), "method": anchor.get("method")}
