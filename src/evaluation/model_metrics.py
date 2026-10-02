"""Model evaluation metrics, bootstrap confidence intervals, and figures (Phase 2, section 2.8).

Computes the full metric suite required by ``config/model.yaml``:

AUROC, AUPRC, sensitivity, specificity, PPV, NPV, F1, F2, balanced accuracy,
Brier score, ECE, confusion matrix — with 2000-sample bootstrap 95% CIs where
computationally practical, plus an explicit false-negative review and subgroup
reporting (age band, sex).

All figures are generated from actual computed values; nothing is hard-coded.

Research prototype — not a medical device.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocessing.regensburg import ROOT

FIG_DIR = ROOT / "outputs/figures"
METRIC_DIR = ROOT / "outputs/metrics"
BOOTSTRAP_N = 2000
BOOTSTRAP_ALPHA = 0.05
ECE_BINS = 10

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def ece_score(y_true: np.ndarray, probs: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Expected Calibration Error (equal-width bins)."""
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


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Full metric dict for binary probabilities at a fixed threshold."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    yhat = (probs >= threshold).astype(int)

    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_true, yhat, labels=[0, 1]).ravel())
    out = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y_true, probs)),
        "sensitivity": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "f2": float(fbeta_score(y_true, yhat, beta=2)) if (tp + fn) else 0.0,
        "balanced_accuracy": 0.5 * (_safe_div(tp, tp + fn) + _safe_div(tn, tn + fp)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": ece_score(y_true, probs),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    return out


def bootstrap_ci(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5,
                 n_boot: int = BOOTSTRAP_N, seed: int = 20261002,
                 metrics: tuple[str, ...] = ("auroc", "auprc", "sensitivity", "specificity",
                                             "ppv", "npv", "f1", "f2", "balanced_accuracy",
                                             "brier", "ece")) -> dict[str, dict[str, float]]:
    """Bootstrap 95% CIs (percentile) for the requested metrics."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    n = len(y_true)
    samples: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pr = y_true[idx], probs[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = compute_metrics(yt, pr, threshold=threshold)
        for key in metrics:
            samples[key].append(m[key])
    lo_q, hi_q = 100 * (BOOTSTRAP_ALPHA / 2), 100 * (1 - BOOTSTRAP_ALPHA / 2)
    out: dict[str, dict[str, float]] = {}
    for key, vals in samples.items():
        if not vals:
            continue
        out[key] = {
            "point": float(compute_metrics(y_true, probs, threshold=threshold)[key]),
            "lo": float(np.percentile(vals, lo_q)),
            "hi": float(np.percentile(vals, hi_q)),
            "n_boot": int(len(vals)),
        }
    return out


def false_negative_review(y_true: np.ndarray, probs: np.ndarray, threshold: float,
                          indices: list[int], rows: Any) -> dict[str, Any]:
    """Explicit review of false negatives (safety-sensitive analysis)."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    yhat = (probs >= threshold).astype(int)
    fn_positions = np.where((yhat == 0) & (y_true == 1))[0]
    cases = []
    for pos in fn_positions:
        rec: dict[str, Any] = {
            "row_index": int(indices[pos]),
            "true_label": "appendicitis",
            "predicted_probability": round(float(probs[pos]), 4),
            "threshold": threshold,
        }
        if rows is not None:
            row = rows.loc[indices[pos]]
            for col in ("Age", "Sex", "WBC_Count", "CRP", "Appendix_Diameter",
                        "Alvarado_Score", "Paedriatic_Appendicitis_Score"):
                if col in rows.columns:
                    v = row[col]
                    rec[col] = None if (v is None or (isinstance(v, float) and np.isnan(v))) else (
                        float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else str(v))
        cases.append(rec)
    return {
        "n_false_negatives": int(len(fn_positions)),
        "false_negative_rate": float(len(fn_positions) / max((y_true == 1).sum(), 1)),
        "cases": cases,
        "note": "False negatives are the safety-critical error direction for this screen; "
                "each case is reviewed against clinical features (research analysis only).",
    }


def subgroup_report(y_true: np.ndarray, probs: np.ndarray, threshold: float,
                    df_any, indices: list[int], age_col: str = "Age",
                    sex_col: str = "Sex") -> dict[str, Any]:
    """Metrics by age band and sex (fixed, auditable bands)."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    idx = np.asarray(indices)
    sub: dict[str, Any] = {}
    if df_any is None:
        return sub
    frame = df_any.loc[idx]
    age = pd_to_numeric(frame[age_col]) if age_col in frame.columns else None
    sex = frame[sex_col] if sex_col in frame.columns else None

    bands = {}
    if age is not None:
        bins = [(0, 2, "0-1"), (2, 6, "2-5"), (6, 13, "6-12"), (13, 19, "13-18")]
        for lo, hi, label in bins:
            mask = ((age >= lo) & (age < hi)).to_numpy()
            if mask.sum() >= 10 and len(np.unique(y_true[mask])) > 1:
                bands[f"age_{label}"] = compute_metrics(y_true[mask], probs[mask], threshold)
            else:
                bands[f"age_{label}"] = {"n": int(mask.sum()), "note": "too few events for metrics"}
    if sex is not None:
        for value in sorted(sex.dropna().astype(str).unique()):
            mask = (sex.astype(str) == value).to_numpy()
            if mask.sum() >= 10 and len(np.unique(y_true[mask])) > 1:
                sub[f"sex_{value}"] = compute_metrics(y_true[mask], probs[mask], threshold)
            else:
                sub[f"sex_{value}"] = {"n": int(mask.sum()), "note": "too few events for metrics"}
    sub.update(bands)
    return sub


def pd_to_numeric(series):
    import pandas as pd

    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------- figures
def plot_roc(y_true, model_probas: dict[str, np.ndarray], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for name, probs in model_probas.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUROC {roc_auc_score(y_true, probs):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — test partition (held-out)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_pr(y_true, model_probas: dict[str, np.ndarray], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for name, probs in model_probas.items():
        prec, rec, _ = precision_recall_curve(y_true, probs)
        ap = average_precision_score(y_true, probs)
        ax.plot(rec, prec, label=f"{name} (AUPRC {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall — test partition (held-out)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_confusion(y_true, probs, threshold: float, path: Path, title: str = "Confusion matrix") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    yhat = (np.asarray(probs) >= threshold).astype(int)
    cm = confusion_matrix(np.asarray(y_true).astype(int), yhat, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["no appendicitis", "appendicitis"])
    ax.set_yticks([0, 1], labels=["no appendicitis", "appendicitis"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center",
                color="white" if val > cm.max() / 2 else "#0f172a")
    ax.set_title(f"{title}\nthreshold={threshold:.2f}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_calibration(y_true, raw_probs, calibrated_probs: dict[str, np.ndarray],
                     threshold: float, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", linewidth=1, label="perfect")

    def _curve(p, label):
        bins = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.any():
                xs.append(float(np.mean(p[m])))
                ys.append(float(np.mean(np.asarray(y_true)[m])))
        ax.plot(xs, ys, marker="o", label=label)

    _curve(raw_probs, f"raw LightGBM (Brier {brier_score_loss(y_true, raw_probs):.3f})")
    for name, p in calibrated_probs.items():
        _curve(p, f"{name} (Brier {brier_score_loss(y_true, p):.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration — test partition (held-out)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_model_comparison(results: dict[str, dict]) -> tuple[Path, Path]:
    """Persist outputs/metrics/model_comparison.{json,csv}."""
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "partition": "test (held-out, single use)",
        "models": results,
        "note": "All values computed from actual model outputs; bootstrap CIs use "
                f"{BOOTSTRAP_N} resamples.",
    }
    jpath = METRIC_DIR / "model_comparison.json"
    jpath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    import csv as _csv

    rows = []
    for name, rec in results.items():
        m = rec.get("test_metrics", {})
        row = {
            "model": name,
            "auroc": m.get("auroc", ""),
            "auprc": m.get("auprc", ""),
            "sensitivity": m.get("sensitivity", ""),
            "specificity": m.get("specificity", ""),
            "ppv": m.get("ppv", ""),
            "npv": m.get("npv", ""),
            "f1": m.get("f1", ""),
            "f2": m.get("f2", ""),
            "balanced_accuracy": m.get("balanced_accuracy", ""),
            "brier": m.get("brier", ""),
            "ece": m.get("ece", ""),
            "threshold": m.get("threshold", ""),
            "calibrator": rec.get("calibrator", "none"),
        }
        rows.append(row)
    cpath = METRIC_DIR / "model_comparison.csv"
    with cpath.open("w", encoding="utf-8", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return jpath, cpath
