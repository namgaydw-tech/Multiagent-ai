"""Phase 5 metric suite — Macro F0.5 as a first-class metric (section F).

Binary and multiclass implementations, all threshold-explicit, all computed
from arrays actually produced by executed model runs:

* ``macro_f0_5``  = unweighted mean of per-class F0.5, exactly equal to
  ``fbeta_score(y, yhat, beta=0.5, average="macro", zero_division=0)``
  (asserted by ``tests/test_phase5.py``);
* per-class F0.5, macro precision/recall/F1/F2, accuracy, balanced accuracy,
  sensitivity/specificity/PPV/NPV, MCC, AUROC, AUPRC, Brier, ECE, confusion
  matrix — binary and multiclass (macro + per-class AUROC/AUPRC where valid);
* 95% bootstrap CIs (percentile, seeded), paired bootstrap difference in
  Macro F0.5, an exact McNemar wrapper, DeLong AUROC CI, Benjamini-Hochberg
  FDR via ``src.bias.metrics.benjamini_hochberg``.

Zero-division rule: any undefined ratio (e.g. recall of an absent class)
returns 0.0 — never NaN — matching sklearn's ``zero_division=0``.

Research prototype — not a medical device.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
)

SEED = 20261002
N_BOOT = 2000
ALPHA = 0.05
BETA = 0.5
ECE_BINS = 10


def _f_beta(precision: float, recall: float, beta: float = BETA) -> float:
    """F_beta from precision/recall with zero-division = 0 (sklearn-equivalent)."""
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0:
        return 0.0
    return float((1 + b2) * precision * recall / denom)


def _prf(y_true: np.ndarray, y_pred: np.ndarray, cls: int) -> tuple[float, float, float]:
    """(precision, recall, f1) for one class, zero_division = 0."""
    tp = int(((y_pred == cls) & (y_true == cls)).sum())
    fp = int(((y_pred == cls) & (y_true != cls)).sum())
    fn = int(((y_pred != cls) & (y_true == cls)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = _f_beta(p, r, beta=1.0)
    return p, r, f1


def macro_f05(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F0.5 = unweighted mean of per-class F0.5 (zero_division=0)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    vals = [_f_beta(*_prf(y_true, y_pred, c)[:2], beta=BETA) for c in classes]
    return float(np.mean(vals)) if vals else 0.0


def per_class_f05(y_true: np.ndarray, y_pred: np.ndarray,
                  classes: list[int] | None = None) -> dict[int, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if classes is None:
        classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {int(c): _f_beta(*_prf(y_true, y_pred, int(c))[:2], beta=BETA) for c in classes}


def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Expected calibration error, equal-width bins (same definition as Phase 2)."""
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            total += (mask.sum() / len(y_true)) * abs(y_true[mask].mean() - probs[mask].mean())
    return float(total)


def _binary(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    yhat = (probs >= threshold).astype(int)
    classes = [0, 1]
    per = {c: _prf(y_true, yhat, c) for c in classes}
    p1, r1, f1_1 = per[1]
    p0, r0, f1_0 = per[0]
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_true, yhat, labels=[0, 1]).ravel())
    two_class = len(np.unique(y_true)) > 1
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "n_positive": int((y_true == 1).sum()),
        "n_negative": int((y_true == 0).sum()),
        "threshold": float(threshold),
        "macro_f0_5": float(np.mean([_f_beta(p0, r0), _f_beta(p1, r1)])),
        "positive_class_f0_5": _f_beta(p1, r1),
        "negative_class_f0_5": _f_beta(p0, r0),
        "per_class_f0_5": {int(c): _f_beta(*per[c][:2]) for c in classes},
        "macro_precision": float(np.mean([p0, p1])),
        "macro_recall": float(np.mean([r0, r1])),
        "macro_f1": float(np.mean([f1_0, f1_1])),
        "macro_f2": float(np.mean([_f_beta(p0, r0, beta=2.0), _f_beta(p1, r1, beta=2.0)])),
        "accuracy": float((yhat == y_true).mean()),
        "balanced_accuracy": float(np.mean([r0, r1])),
        "sensitivity": r1,
        "specificity": r0,
        "ppv": p1,
        "npv": p0,
        "auroc": (float(roc_auc_score(y_true, probs)) if two_class else None),
        "auprc": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": ece(y_true, probs),
        "mcc": float(matthews_corrcoef(y_true, yhat)) if two_class else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "labels": {0: "negative", 1: "positive"},
    }
    return out


def _multiclass(y_true: np.ndarray, probs: np.ndarray, target_names: dict[int, str]
                | None = None) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    k = probs.shape[1]
    yhat = probs.argmax(axis=1)
    classes = list(range(k))
    per = {c: _prf(y_true, yhat, c) for c in classes}
    cm = confusion_matrix(y_true, yhat, labels=classes)
    per_auroc: dict[int, float | None] = {}
    per_auprc: dict[int, float | None] = {}
    for c in classes:
        ybin = (y_true == c).astype(int)
        if ybin.sum() == 0 or ybin.sum() == len(ybin):
            per_auroc[c] = None
            per_auprc[c] = None
        else:
            per_auroc[c] = float(roc_auc_score(ybin, probs[:, c]))
            per_auprc[c] = float(average_precision_score(ybin, probs[:, c]))
    auroc_vals = [v for v in per_auroc.values() if v is not None]
    auprc_vals = [v for v in per_auprc.values() if v is not None]
    onehot = np.eye(k)[y_true]
    labels = target_names or {c: str(c) for c in classes}
    return {
        "n": int(len(y_true)),
        "threshold": None,
        "multiclass": True,
        "macro_f0_5": float(np.mean([_f_beta(*per[c][:2]) for c in classes])),
        "positive_class_f0_5": None,
        "per_class_f0_5": {int(c): _f_beta(*per[c][:2]) for c in classes},
        "macro_precision": float(np.mean([per[c][0] for c in classes])),
        "macro_recall": float(np.mean([per[c][1] for c in classes])),
        "macro_f1": float(np.mean([per[c][2] for c in classes])),
        "macro_f2": float(np.mean([_f_beta(*per[c][:2], beta=2.0) for c in classes])),
        "accuracy": float((yhat == y_true).mean()),
        "balanced_accuracy": float(np.mean([per[c][1] for c in classes])),
        "sensitivity": None, "specificity": None, "ppv": None, "npv": None,
        "auroc": (float(np.mean(auroc_vals)) if auroc_vals else None),
        "auroc_macro": (float(np.mean(auroc_vals)) if auroc_vals else None),
        "auprc": (float(np.mean(auprc_vals)) if auprc_vals else None),
        "auprc_macro": (float(np.mean(auprc_vals)) if auprc_vals else None),
        "per_class_auroc": per_auroc,
        "per_class_auprc": per_auprc,
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "ece": ece((yhat == y_true).astype(float), probs.max(axis=1)),
        "mcc": float(matthews_corrcoef(y_true, yhat)),
        "confusion_matrix": cm.tolist(),
        "labels": {int(c): str(labels.get(c, c)) for c in classes},
    }


def compute_full_metrics(y_true: np.ndarray, probs: np.ndarray, *,
                         threshold: float = 0.5,
                         multiclass: bool = False,
                         target_names: dict[int, str] | None = None) -> dict[str, Any]:
    """Full Phase 5 metric dict (section F). ``probs``: (n,) for binary, (n, K) multiclass."""
    arr = np.asarray(probs)
    if multiclass or arr.ndim == 2:
        if arr.ndim == 1:
            raise ValueError("multiclass probabilities must be a (n, K) matrix")
        return _multiclass(y_true, arr, target_names=target_names)
    return _binary(y_true, arr, threshold)


def metrics_from_hard(y_true: np.ndarray, y_pred: np.ndarray,
                      labels: dict[int, str] | None = None) -> dict[str, Any]:
    """Threshold-dependent metrics from hard labels only.

    Probability-dependent metrics (AUROC/AUPRC/Brier/ECE) are mathematically
    undefined for a pure vote — they are reported as ``None`` (n/a), never as 0.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    per = {c: _prf(y_true, y_pred, c) for c in (0, 1)}
    p1, r1, _ = per[1]
    p0, r0, _ = per[0]
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())
    return {
        "n": int(len(y_true)),
        "threshold": None,
        "macro_f0_5": float(np.mean([_f_beta(p0, r0), _f_beta(p1, r1)])),
        "positive_class_f0_5": _f_beta(p1, r1),
        "per_class_f0_5": {0: _f_beta(p0, r0), 1: _f_beta(p1, r1)},
        "macro_precision": float(np.mean([p0, p1])),
        "macro_recall": float(np.mean([r0, r1])),
        "macro_f1": float(np.mean([per[0][2], per[1][2]])),
        "macro_f2": float(np.mean([_f_beta(p0, r0, beta=2.0), _f_beta(p1, r1, beta=2.0)])),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(np.mean([r0, r1])),
        "sensitivity": r1, "specificity": r0, "ppv": p1, "npv": p0,
        "auroc": None, "auprc": None, "brier": None, "ece": None,
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "labels": labels or {0: "negative", 1: "positive"},
        "probability_metrics_note": ("AUROC/AUPRC/Brier/ECE not defined for a hard vote "
                                     "— reported as n/a, never 0"),
    }


# ------------------------------------------------------------------ statistics
def bootstrap_ci(y_true: np.ndarray, probs: np.ndarray, *, threshold: float = 0.5,
                 n_boot: int = N_BOOT, seed: int = SEED,
                 multiclass: bool = False,
                 metrics: tuple[str, ...] = ("macro_f0_5", "accuracy", "auroc", "auprc",
                                             "sensitivity", "specificity", "balanced_accuracy",
                                             "brier", "ece", "mcc")) -> dict[str, Any]:
    """Seeded percentile 95% CIs for the requested metrics (point + lo + hi)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    p = np.asarray(probs)
    n = len(y)
    samples: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pt = y[idx], p[idx]
        try:
            m = compute_full_metrics(yt, pt, threshold=threshold, multiclass=multiclass)
        except Exception:
            continue
        for key in metrics:
            val = m.get(key)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                samples[key].append(float(val))
    point = compute_full_metrics(y, p, threshold=threshold, multiclass=multiclass)
    lo_q, hi_q = 100 * (ALPHA / 2), 100 * (1 - ALPHA / 2)
    out: dict[str, Any] = {}
    for key, vals in samples.items():
        if not vals:
            continue
        out[key] = {"point": float(point.get(key)),
                    "lo": float(np.percentile(vals, lo_q)),
                    "hi": float(np.percentile(vals, hi_q)),
                    "n_boot": int(len(vals))}
    return out


def paired_bootstrap_delta_f05(y_true: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray,
                               *, threshold_a: float = 0.5, threshold_b: float = 0.5,
                               n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, Any]:
    """Paired bootstrap difference in Macro F0.5 (A − B) with a two-sided p-value.

    Both predictions must come from the SAME cases (paired design).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    a, b = np.asarray(probs_a), np.asarray(probs_b)
    n = len(y)
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        ma = compute_full_metrics(yt, a[idx], threshold=threshold_a)
        mb = compute_full_metrics(yt, b[idx], threshold=threshold_b)
        deltas.append(ma["macro_f0_5"] - mb["macro_f0_5"])
    point = (compute_full_metrics(y, a, threshold=threshold_a)["macro_f0_5"]
             - compute_full_metrics(y, b, threshold=threshold_b)["macro_f0_5"])
    p_two = float(2 * min(np.mean(np.array(deltas) <= 0), np.mean(np.array(deltas) >= 0)))
    return {"delta_macro_f0_5": float(point),
            "ci95": [float(np.percentile(deltas, 100 * ALPHA / 2)),
                     float(np.percentile(deltas, 100 * (1 - ALPHA / 2)))],
            "p_value": min(1.0, p_two), "n_boot": n_boot, "seed": seed}


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """Exact two-sided McNemar test on paired hard predictions (b/c discordant counts)."""
    from src.bias.metrics import mcnemar_exact
    y = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    b_count = int(((a == y) & (b != y)).sum())     # A right, B wrong
    c_count = int(((a != y) & (b == y)).sum())     # A wrong, B right
    res = mcnemar_exact(b_count, c_count)
    return {"b": b_count, "c": c_count, "p_value": res.get("p_value"),
            "test": "exact McNemar (two-sided binomial over discordant pairs)"}


def delong_roc(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """DeLong AUROC estimate with variance and 95% CI (standard U-statistic form)."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return {"auc": None, "var": None, "ci95": [None, None],
                "note": "degenerate classes — DeLong not computable"}

    def _psi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # pairwise comparison kernel with tie handling (0.5)
        diff = a[:, None] - b[None, :]
        return (diff > 0).astype(float) + 0.5 * (diff == 0)

    theta = float(_psi(pos, neg).mean())
    v01 = _psi(pos, neg).mean(axis=1)          # structural components, positives
    v10 = _psi(pos, neg).mean(axis=0)          # structural components, negatives
    s01 = float(np.var(v01, ddof=1)) if m > 1 else 0.0
    s10 = float(np.var(v10, ddof=1)) if n > 1 else 0.0
    var = s01 / m + s10 / n
    se = float(np.sqrt(var)) if var > 0 else 0.0
    ci = [theta - 1.959963985 * se, theta + 1.959963985 * se]
    z = (theta - 0.5) / se if se > 0 else None
    from math import erfc, sqrt
    p = float(erfc(abs(z) / sqrt(2))) if z is not None else None
    return {"auc": theta, "var": float(var), "se": se,
            "ci95": [float(ci[0]), float(ci[1])],
            "z_vs_0.5": (float(z) if z is not None else None), "p_value": p,
            "method": "DeLong (Mann-Whitney U structural components)"}


def bh_family(p_values: list[float | None]) -> list[dict[str, Any]]:
    """Benjamini-Hochberg corrected q-values (wraps src.bias.metrics)."""
    from src.bias.metrics import benjamini_hochberg
    return benjamini_hochberg(p_values)
