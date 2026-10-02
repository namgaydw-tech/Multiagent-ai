"""Single-case inference over the persisted Phase 2 artifacts (Phase 3, section 3.4).

The multi-agent engine consumes the *normalized prediction interface* for one patient
row at a time, so this module reloads everything Phase 2 persisted and reproduces the
exact training-time transformation for a single row:

* the audited Regensburg dataset + feature spec from ``config/model.yaml``;
* the categorical vocabulary re-fit **on the training partition only** (identical to
  training — the encoding state was not serialized, and fitting it on the same split
  reproduces it deterministically; no test/validation row ever contributes);
* ``outputs/models/lightgbm_appendicitis.joblib`` (fitted LightGBM),
  ``lightgbm_calibrator.joblib`` (isotonic calibrator) and the validation-locked
  threshold from ``lightgbm_calibrator_meta.json``;
* per-case SHAP attributions via ``shap.TreeExplainer`` (raw model output — SHAP
  explains the model, the calibrator is a monotone post-hoc map).

Every returned prediction follows the Phase 2 contract
(``src/models/prediction_interface.py``): raw class probabilities, separately
reported ``calibrated_probability``, and an uncertainty record computed by
``src/models/uncertainty.py``. Nothing here invents clinical content.

Research prototype — not a medical device.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.split import load_splits
from src.models.prediction_interface import TARGET_CLASSES, apply_calibration, build_prediction
from src.models.uncertainty import estimate_uncertainty
from src.preprocessing.regensburg import (
    ROOT,
    build_feature_spec,
    config_hash,
    load_audited_dataset,
    load_config,
    prepare_matrices,
    train_feature_arrays,
)

MODEL_DIR = ROOT / "outputs/models"
CAL_META_PATH = MODEL_DIR / "lightgbm_calibrator_meta.json"

SHAP_WORDING = (
    "SHAP value: this feature contributed to this model prediction; "
    "it did not cause the diagnosis."
)


def _load_shap_values(explainer: Any, X_row: pd.DataFrame) -> np.ndarray:
    """Normalize TreeExplainer output (list-of-2 or single array) to shape (n, f)."""
    raw = explainer.shap_values(X_row)
    if isinstance(raw, (list, tuple)):            # [class0, class1] (older shap)
        arr = np.asarray(raw[1])
    else:
        arr = np.asarray(raw)
        if arr.ndim == 3 and arr.shape[0] == 2:   # (2, n, f) defensive branch
            arr = arr[1]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


class CasePredictor:
    """Predict and explain one dataset row with the persisted Phase 2 model.

    Construction loads the dataset, refits the training categorical vocabulary and
    deserializes the model/calibrator — do this once per process and reuse
    (:func:`get_predictor` returns a cached singleton).
    """

    def __init__(self) -> None:
        self.config = load_config()
        self.config_hash = config_hash(self.config)
        self.df = load_audited_dataset()
        self.spec = build_feature_spec(self.df, self.config)
        self.X, self.y = prepare_matrices(self.df, self.spec)

        splits = load_splits(ROOT / "data/interim/splits")
        train_rows = splits["train"]
        X_train_raw = self.X.loc[train_rows]
        # vocabulary fit on TRAIN only — reproduces the Phase 2 encoding exactly
        _, self.cat_state = train_feature_arrays(X_train_raw, self.spec, fit=True)

        payload = joblib.load(MODEL_DIR / "lightgbm_appendicitis.joblib")
        self.model = payload["model"]
        self.feature_order: list[str] = list(payload["feature_order"])
        self.model_name = str(payload.get("model_name", "lightgbm_appendicitis"))
        self.model_version = str(payload.get("model_version", "1.0.0"))

        cal = joblib.load(MODEL_DIR / "lightgbm_calibrator.joblib")
        self.calibrator = cal["calibrator"]
        self.calibrator_name = str(cal.get("name", "unknown"))
        meta = json.loads(CAL_META_PATH.read_text(encoding="utf-8"))
        self.threshold = float(meta["threshold"]["threshold"])

        self._explainer: Any = None
        self._shap_version: str = ""

    # ------------------------------------------------------------------ helpers
    def case_id(self, row_index: int) -> str:
        """Stable case identifier for a dataset row."""
        return f"regensburg_row_{int(row_index)}"

    def row_index_from_case(self, case_id: str) -> int:
        """Inverse of :meth:`case_id` (raises on malformed ids)."""
        return int(str(case_id).rsplit("_", 1)[-1])

    def has_row(self, row_index: int) -> bool:
        """True when the row exists in the modelling matrix (target present)."""
        return int(row_index) in self.X.index

    def _matrix_one(self, row_index: int) -> pd.DataFrame:
        """One-row model matrix in training column order."""
        if not self.has_row(row_index):
            raise KeyError(f"row {row_index} not in modelling matrix (target missing?)")
        one = self.X.loc[[row_index]]
        one = self.cat_state.apply(one)
        return one.reindex(columns=self.feature_order)

    def _get_explainer(self) -> Any:
        if self._explainer is None:
            import shap  # local import: keeps module import cheap
            self._shap_version = getattr(shap, "__version__", "unknown")
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    # ------------------------------------------------------------------ public
    def predict(self, row_index: int) -> dict[str, Any]:
        """Normalized prediction dict for one row (raw + calibrated + uncertainty)."""
        one = self._matrix_one(row_index)
        p_raw = float(self.model.predict_proba(one)[0, 1])
        p_cal = float(self.calibrator.predict(np.asarray([p_raw]))[0])
        unc = estimate_uncertainty(p_raw, threshold=self.threshold, calibrated_probability=p_cal)
        pred = build_prediction(
            model_name=self.model_name,
            model_version=self.model_version,
            positive_probability=p_raw,
            important_features=self._shap_contributors(row_index, one, top_k=15),
            config_hash=self.config_hash,
            calibrated_probability=p_cal,
            uncertainty=unc,
            threshold=self.threshold,
        )
        return pred.model_dump()

    def explain(self, row_index: int, top_k: int = 10) -> dict[str, Any]:
        """Structured per-case SHAP explanation (non-causal wording enforced)."""
        one = self._matrix_one(row_index)
        contributors = self._shap_contributors(row_index, one, top_k=top_k)
        p_raw = float(self.model.predict_proba(one)[0, 1])
        return {
            "case_id": self.case_id(row_index),
            "row_index": int(row_index),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "shap_version": self._shap_version or None,
            "explainer": "shap.TreeExplainer (LightGBM) — raw model output",
            "raw_probability": round(p_raw, 6),
            "top_contributors": contributors,
            "wording_rule": SHAP_WORDING,
        }

    def bundle(self, row_index: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """(model_output, shap_explain, uncertainty) for the engine's Layer-A input."""
        model_output = self.predict(row_index)
        shap_explain = self.explain(row_index)
        uncertainty = model_output["uncertainty"]
        return model_output, shap_explain, uncertainty

    # ------------------------------------------------------------------ internal
    def _shap_contributors(self, row_index: int, one: pd.DataFrame,
                           top_k: int = 10) -> list[dict[str, Any]]:
        """Top-k |SHAP| features for this row, direction signed toward appendicitis."""
        explainer = self._get_explainer()
        values = _load_shap_values(explainer, one)[0]
        row = one.iloc[0]
        order = np.argsort(np.abs(values))[::-1][: max(1, int(top_k))]
        out: list[dict[str, Any]] = []
        for i in order:
            fname = str(self.feature_order[int(i)])
            val = row.iloc[int(i)]
            contribution = float(values[int(i)])
            out.append({
                "feature": fname,
                "value": (None if val is None or (isinstance(val, float) and np.isnan(val))
                          else (val.item() if hasattr(val, "item") else val)),
                "importance": round(abs(contribution), 6),
                "contribution": round(contribution, 6),
                "direction": "positive" if contribution >= 0 else "negative",
                "wording": SHAP_WORDING,
            })
        return out


_PREDICTOR: CasePredictor | None = None
_LOCK = threading.Lock()


def get_predictor() -> CasePredictor:
    """Process-wide cached :class:`CasePredictor` (heavy artifacts loaded once)."""
    global _PREDICTOR
    with _LOCK:
        if _PREDICTOR is None:
            _PREDICTOR = CasePredictor()
        return _PREDICTOR


def predict_case(row_index: int) -> dict[str, Any]:
    """Convenience wrapper: normalized prediction for one dataset row."""
    return get_predictor().predict(row_index)


def explain_case(row_index: int, top_k: int = 10) -> dict[str, Any]:
    """Convenience wrapper: SHAP explanation for one dataset row."""
    return get_predictor().explain(row_index, top_k=top_k)
