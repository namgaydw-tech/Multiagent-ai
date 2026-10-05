"""Phase 5 calibration (section I): raw / Platt / isotonic, validation-only.

Binary datasets reuse :mod:`src.calibration.calibrate` (Platt + isotonic, fitted
on validation predictions, selected by Brier then ECE). This module adds:

* ``fit_multiclass_sigmoid`` — one-vs-rest logistic (Platt) per class for
  multiclass probability calibration (documented as the only multiclass method
  used here; isotonic per class is deliberately NOT used on small n — its
  instability is reported as a limitation in docs/PHASE5_REPORT.md);
* ``compare_and_select`` — candidate comparison on validation with the exact
  selection rule (lowest Brier, ties broken by ECE), never touching test data;
* reliability-diagram data (bin accuracies/confidences) generated from
  validation or test predictions by the caller.

Every fitted calibrator here was trained on validation predictions only.

Research prototype — not a medical device.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.calibration.calibrate import (
    IsotonicCalibrator,
    PlattCalibrator,
    ece,
)


def apply_calibrator(name: str, fitted: Any, probs: np.ndarray) -> np.ndarray:
    return np.asarray(fitted.predict(np.asarray(probs, dtype=float)), dtype=float)


def fit_binary(name: str, y_val: np.ndarray, p_val: np.ndarray) -> Any:
    """Fit one binary calibrator on validation predictions (raw → no fit needed)."""
    if name in ("raw", "uncalibrated", "none"):
        return None
    if name == "platt":
        return PlattCalibrator().fit(np.asarray(p_val, dtype=float),
                                     np.asarray(y_val).astype(int))
    if name == "isotonic":
        return IsotonicCalibrator().fit(np.asarray(p_val, dtype=float),
                                        np.asarray(y_val).astype(int))
    raise ValueError(f"unknown calibrator {name!r}")


def compare_binary(y_val: np.ndarray, p_val: np.ndarray) -> dict[str, dict[str, float]]:
    """{candidate: {brier, ece}} for raw/platt/isotonic — validation only."""
    from sklearn.metrics import brier_score_loss

    y = np.asarray(y_val).astype(int)
    p = np.asarray(p_val, dtype=float)
    out: dict[str, dict[str, float]] = {}
    for name in ("raw", "platt", "isotonic"):
        cal = fit_binary(name, y, p)
        pc = p if cal is None else apply_calibrator(name, cal, p)
        out[name] = {"brier": float(brier_score_loss(y, pc)), "ece": ece(y, pc)}
    return out


def select_binary(comparison: dict[str, dict[str, float]]) -> str:
    """Lowest validation Brier, ties broken by ECE (test never consulted)."""
    return min(comparison, key=lambda k: (round(comparison[k]["brier"], 6),
                                          round(comparison[k]["ece"], 6)))


class OVRSigmoidCalibrator:
    """One-vs-rest Platt (logistic) calibration for K-class probabilities."""

    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression
        self._clfs = [LogisticRegression(solver="lbfgs", random_state=20261002)
                      for _ in range(0)]
        self.k = 0

    def fit(self, probs: np.ndarray, y: np.ndarray) -> "OVRSigmoidCalibrator":
        from sklearn.linear_model import LogisticRegression
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y).astype(int)
        self.k = probs.shape[1]
        self._clfs = []
        for c in range(self.k):
            clf = LogisticRegression(solver="lbfgs", random_state=20261002)
            clf.fit(probs[:, [c]], (y == c).astype(int))
            self._clfs.append(clf)
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        out = np.column_stack([c.predict_proba(probs[:, [i]])[:, 1]
                               for i, c in enumerate(self._clfs)])
        row = out.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        return out / row


def compare_multiclass(y_val: np.ndarray, probs_val: np.ndarray) -> dict[str, dict[str, float]]:
    """Raw vs one-vs-rest sigmoid on validation (Brier + top-label ECE)."""
    from sklearn.metrics import brier_score_loss

    y = np.asarray(y_val).astype(int)
    P = np.asarray(probs_val, dtype=float)
    k = P.shape[1]
    onehot = np.eye(k)[y]

    def _brier(probs: np.ndarray) -> float:
        return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))

    def _ece_top(probs: np.ndarray) -> float:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        acc = (pred == y).astype(float)
        return ece(acc, conf)

    sigmoid = OVRSigmoidCalibrator().fit(P, y)
    P_sig = sigmoid.predict(P)
    return {"raw": {"brier": _brier(P), "ece": _ece_top(P)},
            "ovr_sigmoid": {"brier": _brier(P_sig), "ece": _ece_top(P_sig)}}


def select_multiclass(comparison: dict[str, dict[str, float]]) -> str:
    return min(comparison, key=lambda k: (round(comparison[k]["brier"], 6),
                                          round(comparison[k]["ece"], 6)))


def reliability_bins(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Equal-width bin records for a reliability diagram (figure 6)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        out.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                    "n": int(mask.sum()),
                    "confidence": float(p[mask].mean()),
                    "accuracy": float(y[mask].mean())})
    return out
