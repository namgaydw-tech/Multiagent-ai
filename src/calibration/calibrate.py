"""Probability calibration for the LightGBM model (Phase 2, section 2.5).

Candidates: uncalibrated (baseline), Platt scaling, isotonic regression.
All calibrators are fitted on **validation** predictions (non-test data only) and
selected by Brier score then ECE on the same validation partition. The threshold is
selected on validation (F2 objective) and locked before the test evaluation.

Persisted artifact: ``outputs/models/lightgbm_calibrator.joblib``.
Raw model probability and calibrated probability remain distinguishable: calibration
never overwrites the raw ``class_probabilities`` in the normalized interface.

Research prototype — not a medical device.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.preprocessing.regensburg import ROOT

MODEL_DIR = ROOT / "outputs/models"
CALIBRATOR_PATH = MODEL_DIR / "lightgbm_calibrator.joblib"


class PlattCalibrator:
    """Logistic (Platt) scaling of raw probabilities, fitted on validation predictions."""

    def __init__(self) -> None:
        self._clf = LogisticRegression(solver="lbfgs", random_state=20261002)

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self._clf.fit(np.asarray(probs).reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(np.asarray(probs).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Isotonic regression mapping, fitted on validation predictions."""

    def __init__(self) -> None:
        self._ir = IsotonicRegression(out_of_bounds="clip")

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._ir.fit(np.asarray(probs), np.asarray(y))
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        return self._ir.predict(np.asarray(probs))


def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-width bins."""
    y_true = np.asarray(y_true).astype(float)
    probs = np.asarray(probs).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        total += (mask.sum() / len(y_true)) * abs(y_true[mask].mean() - probs[mask].mean())
    return float(total)


def compare_calibrators(y_val: np.ndarray, p_val: np.ndarray) -> dict[str, dict[str, float]]:
    """Fit Platt and isotonic on validation predictions; report Brier/ECE for all candidates."""
    from sklearn.metrics import brier_score_loss

    out: dict[str, dict[str, float]] = {}
    out["uncalibrated"] = {
        "brier": float(brier_score_loss(y_val, p_val)),
        "ece": ece(y_val, p_val),
    }
    platt = PlattCalibrator().fit(p_val, y_val)
    p_platt = platt.predict(p_val)
    out["platt"] = {
        "brier": float(brier_score_loss(y_val, p_platt)),
        "ece": ece(y_val, p_platt),
    }
    iso = IsotonicCalibrator().fit(p_val, y_val)
    p_iso = iso.predict(p_val)
    out["isotonic"] = {
        "brier": float(brier_score_loss(y_val, p_iso)),
        "ece": ece(y_val, p_iso),
    }
    return out


def select_calibrator(comparison: dict[str, dict[str, float]]) -> str:
    """Pick the candidate with the lowest validation Brier, breaking ties with ECE."""
    candidates = {k: v for k, v in comparison.items() if k != "uncalibrated"}
    return min(candidates, key=lambda k: (round(candidates[k]["brier"], 6),
                                          round(candidates[k]["ece"], 6)))


def fit_selected(name: str, y_val: np.ndarray, p_val: np.ndarray):
    """Fit the selected calibrator on validation predictions."""
    if name == "platt":
        return PlattCalibrator().fit(p_val, y_val)
    if name == "isotonic":
        return IsotonicCalibrator().fit(p_val, y_val)
    raise ValueError(f"unknown calibrator: {name}")


def select_threshold_f2(y_val: np.ndarray, p_val: np.ndarray) -> dict[str, Any]:
    """Choose the decision threshold on validation maximising F2 (sensitivity-weighted).

    Locked before the test evaluation, per config/model.yaml.
    """
    from sklearn.metrics import fbeta_score

    grid = np.arange(0.05, 0.951, 0.01)
    best_t, best_f2, best_sens = 0.5, -1.0, 0.0
    for t in grid:
        yhat = (p_val >= t).astype(int)
        if yhat.sum() == 0:
            continue
        f2 = fbeta_score(y_val, yhat, beta=2)
        if f2 > best_f2:
            best_t, best_f2 = float(t), float(f2)
            sens = float(((yhat == 1) & (y_val == 1)).sum() / max((y_val == 1).sum(), 1))
            best_sens = sens
    return {"threshold": round(best_t, 3), "val_f2": round(best_f2, 4),
            "val_sensitivity_at_threshold": round(best_sens, 4),
            "objective": "f2", "fitted_on": "validation"}


def run_calibration(y_val: np.ndarray, p_val: np.ndarray) -> dict[str, Any]:
    """Full calibration step: compare → select → fit → persist → return metadata."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    comparison = compare_calibrators(y_val, p_val)
    selected = select_calibrator(comparison)
    calibrator = fit_selected(selected, y_val, p_val)
    threshold_info = select_threshold_f2(y_val, p_val)

    payload = {
        "calibrator": selected,
        "comparison": comparison,
        "threshold": threshold_info,
        "fitted_on": "validation",
        "n_validation": int(len(y_val)),
        "fitted_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - t0, 2),
        "note": "Calibration fitted on validation only; test labels never used.",
    }
    joblib.dump({"calibrator": calibrator, "name": selected, "meta": payload}, CALIBRATOR_PATH)
    (MODEL_DIR / "lightgbm_calibrator_meta.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_calibrator() -> dict:
    """Reload the persisted calibrator artifact."""
    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(f"Calibrator artifact missing: {CALIBRATOR_PATH}")
    return joblib.load(CALIBRATOR_PATH)
