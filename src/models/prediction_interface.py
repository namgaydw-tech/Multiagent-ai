"""Normalized prediction interface shared by every model and the agent layer (Phase 2).

All models in this project — LightGBM, baselines, future domain models — emit the same
JSON-serialisable structure so the multi-agent debiasing engine consumes one contract:

    model_name, model_version, clinical_domain, target_classes, class_probabilities,
    predicted_class, calibrated_probability, uncertainty, important_features,
    timestamp, config_hash

Validated with Pydantic. ``calibrated_probability`` stays ``None`` until a calibrator has
been applied, so raw model probability and calibrated probability are always distinguishable.

This is a research interface, not a clinical output. Not a medical device.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

CLINICAL_DOMAIN = "pediatric_appendicitis"
TARGET_CLASSES = ["no appendicitis", "appendicitis"]


class NormalizedPrediction(BaseModel):
    """Schema for a single model prediction (normalized cross-model contract)."""

    model_name: str
    model_version: str
    clinical_domain: str
    target_classes: list[str]
    class_probabilities: dict[str, float]
    predicted_class: str
    calibrated_probability: float | None = None
    uncertainty: dict[str, Any] | None = None
    important_features: list[dict[str, Any]] = []
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    config_hash: str = ""

    @field_validator("class_probabilities")
    @classmethod
    def _probabilities_valid(cls, v: dict[str, float]) -> dict[str, float]:
        if not v:
            raise ValueError("class_probabilities must not be empty")
        for k, p in v.items():
            if not 0.0 <= float(p) <= 1.0:
                raise ValueError(f"probability for {k!r} outside [0,1]: {p}")
        total = sum(float(p) for p in v.values())
        if abs(total - 1.0) > 1e-3:
            raise ValueError(f"class_probabilities must sum to 1 (got {total})")
        return {k: float(p) for k, p in v.items()}

    @field_validator("predicted_class")
    @classmethod
    def _predicted_known(cls, v: str, info) -> str:
        classes = info.data.get("target_classes") or []
        if classes and v not in classes:
            raise ValueError(f"predicted_class {v!r} not in target_classes {classes}")
        return v


def build_prediction(
    *,
    model_name: str,
    model_version: str,
    positive_probability: float,
    important_features: list[dict[str, Any]],
    config_hash: str,
    calibrated_probability: float | None = None,
    uncertainty: dict[str, Any] | None = None,
    threshold: float = 0.5,
    clinical_domain: str = CLINICAL_DOMAIN,
) -> NormalizedPrediction:
    """Assemble a validated :class:`NormalizedPrediction` from raw model outputs.

    ``positive_probability`` is P(appendicitis) from the raw model; the predicted class is
    decided by ``threshold`` (validation-locked), independent of calibration.
    """
    p = float(positive_probability)
    probs = {TARGET_CLASSES[0]: round(1.0 - p, 6), TARGET_CLASSES[1]: round(p, 6)}
    predicted = TARGET_CLASSES[1] if p >= threshold else TARGET_CLASSES[0]
    return NormalizedPrediction(
        model_name=model_name,
        model_version=model_version,
        clinical_domain=clinical_domain,
        target_classes=list(TARGET_CLASSES),
        class_probabilities=probs,
        predicted_class=predicted,
        calibrated_probability=None if calibrated_probability is None else float(calibrated_probability),
        uncertainty=uncertainty,
        important_features=important_features,
        config_hash=config_hash,
    )


def apply_calibration(prediction: NormalizedPrediction, calibrated_p: float) -> NormalizedPrediction:
    """Attach a calibrated probability without disturbing the raw class probabilities."""
    out = prediction.model_copy(deep=True)
    out.calibrated_probability = float(calibrated_p)
    return out
