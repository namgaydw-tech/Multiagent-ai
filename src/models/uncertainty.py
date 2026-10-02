"""Uncertainty estimation for the tabular appendicitis model (Phase 2, section 2.6).

Produces an explicit, honestly-labelled uncertainty record:

    confidence, predictive_entropy, margin, uncertainty_level (LOW|MODERATE|HIGH)

Design rules:

* predictive entropy is computed from the (calibrated, if available) probability,
* the margin is |p - threshold| style separation between the top two classes,
* ``uncertainty_level`` derives from entropy bands — **probability itself is not
  mislabelled as uncertainty**; a confident-but-miscalibrated model can still be
  wrong, which is why entropy + margin are reported separately from the raw p.

Not a clinical confidence statement. Research prototype only.
"""

from __future__ import annotations

import math
from typing import Any

# Entropy bands for a binary outcome (max entropy = ln 2 ≈ 0.6931).
LOW_MAX = 0.45       # entropy <= 0.45 -> LOW
MODERATE_MAX = 0.63  # entropy <= 0.63 -> MODERATE, else HIGH
MARGIN_FLOOR = 0.10  # margin below this always elevates at least to MODERATE


def binary_entropy(p: float) -> float:
    """Predictive entropy in nats for Bernoulli(p)."""
    p = min(max(float(p), 1e-12), 1.0 - 1e-12)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def estimate_uncertainty(
    positive_probability: float,
    threshold: float = 0.5,
    calibrated_probability: float | None = None,
    ensemble_std: float | None = None,
) -> dict[str, Any]:
    """Build the uncertainty record for one prediction.

    Parameters
    ----------
    positive_probability:
        raw model P(appendicitis).
    threshold:
        decision threshold (validation-locked).
    calibrated_probability:
        optional calibrated value; entropy is computed on it when present.
    ensemble_std:
        optional disagreement term (e.g. bootstrap/ensemble std); raises the level
        when model disagreement is large.
    """
    p_cal = calibrated_probability if calibrated_probability is not None else positive_probability
    entropy = binary_entropy(p_cal)
    margin = abs(float(p_cal) - float(threshold)) * 2.0  # 0..1 separation from the decision boundary
    margin = min(margin, 1.0)

    if entropy <= LOW_MAX:
        level = "LOW"
    elif entropy <= MODERATE_MAX:
        level = "MODERATE"
    else:
        level = "HIGH"

    # A small margin means the decision sits near the boundary: never call it LOW.
    if margin < MARGIN_FLOOR and level == "LOW":
        level = "MODERATE"
    # Ensemble disagreement (if supplied) only ever escalates the level, never lowers it.
    if ensemble_std is not None:
        if ensemble_std >= 0.15:
            level = "HIGH"
        elif ensemble_std >= 0.10 and level == "LOW":
            level = "MODERATE"

    return {
        "confidence": round(1.0 - entropy, 4),
        "predictive_entropy": round(entropy, 4),
        "margin": round(margin, 4),
        "uncertainty_level": level,
        "entropy_source": "calibrated" if calibrated_probability is not None else "raw",
        "ensemble_std": None if ensemble_std is None else round(float(ensemble_std), 4),
        "note": "Entropy/margin describe model uncertainty, not clinical certainty; "
                "research prototype — not a clinical confidence statement.",
    }
