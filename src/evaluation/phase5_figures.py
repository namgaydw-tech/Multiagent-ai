"""Phase 5 figure generation — every number comes from persisted artifacts.

Sources (never re-tuned, never hard-coded):

* ``outputs/phase5/<ds>/metrics/metrics_{val,test}.json`` — leaderboards, CIs, times;
* ``outputs/phase5/<ds>/predictions/*_test.csv`` — per-case sealed-test predictions;
* ``outputs/phase5/<ds>/calibration/*`` — calibration choices, locked thresholds;
* ``outputs/phase5/<ds>/models/*.joblib`` — fitted models (importance, SHAP);
* ``outputs/phase5/cross_dataset_summary.json`` — cross-dataset view;
* ``outputs/metrics/bias_metrics.json`` — Phase 4 bias headlines (CBR/BCR/HFR).

Labels: sealed-test panels are marked "SEALED TEST"; anything computed on
development/validation data is marked "DEVELOPMENT / VALIDATION".
Reproduce with: ``python scripts/run_phase5.py analyze``.
Research prototype — not a medical device.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P5 = ROOT / "outputs" / "phase5"
SEED = 20261002
REPRODUCE = "python scripts/run_phase5.py analyze"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "figure.autolayout": True})

PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f59e0b",
           "#0891b2", "#db2777", "#65a30d", "#7c3aed", "#ea580c",
           "#0d9488", "#4b5563", "#b91c1c", "#1d4ed8", "#15803d", "#a21caf"]

# The 35 required visualizations (spec section 9) -> manifest entries.
TARGETS: list[tuple[int, str]] = [
    (1, "Macro F0.5 leaderboard"), (2, "accuracy comparison"),
    (3, "balanced accuracy"), (4, "sensitivity comparison"),
    (5, "specificity comparison"), (6, "precision comparison"),
    (7, "AUROC comparison"), (8, "AUPRC comparison"),
    (9, "ROC curves"), (10, "precision-recall curves"),
    (11, "confusion matrices"), (12, "normalized confusion matrices"),
    (13, "calibration/reliability curves"), (14, "threshold vs Macro F0.5"),
    (15, "threshold vs sensitivity"), (16, "threshold vs specificity"),
    (17, "threshold vs precision"),
    (18, "sensitivity-specificity operating point"), (19, "learning curves"),
    (20, "native feature importance"), (21, "permutation importance"),
    (22, "SHAP summary"), (23, "SHAP waterfall"),
    (24, "training-time comparison"), (25, "inference-time comparison"),
    (26, "performance-vs-complexity"), (27, "Random Forest n_estimators curve"),
    (28, "KNN K curve"), (29, "boosting-model comparison"),
    (30, "ensemble-vs-individual comparison"), (31, "cross-dataset comparison"),
    (32, "CBR comparison"), (33, "BCR vs HFR trade-off"),
    (34, "external-validation generalization gap"),
    (35, "confidence intervals for leading models"),
]

_DATASET_DIRS: dict[str, str] = {}   # dataset_id -> directory name (filled lazily)


def log(msg: str) -> None:
    print(f"[phase5-figures] {msg}", flush=True)


# --------------------------------------------------------------------- loading
def _j(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _datasets() -> list[tuple[str, Path]]:
    """(dataset_id, directory) for every dataset with a sealed-test evaluation."""
    out = []
    if not P5.exists():
        return out
    for d in sorted(P5.iterdir()):
        mt = d / "metrics" / "metrics_test.json"
        if d.is_dir() and mt.exists():
            ds_id = str(_j(mt).get("dataset_id") or d.name)
            _DATASET_DIRS[ds_id] = d.name
            out.append((ds_id, d))
    return out


def fig_dir(ds_dir: Path) -> Path:
    p = ds_dir / "figures"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _badge(ax, text: str) -> None:
    ax.text(0.99, 0.01, text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="#444", style="italic")


def metric_table(ds_dir: Path, partition: str) -> dict[str, dict[str, Any]]:
    """{model_or_ensemble: metrics} for 'val' or 'test', available entries only."""
    d = _j(ds_dir / "metrics" / f"metrics_{partition}.json")
    out: dict[str, dict[str, Any]] = {}
    # metrics_test nests under "test_metrics"; metrics_val.json nests under "metrics"
    if partition == "test":
        for name, e in (d.get("models") or {}).items():
            if isinstance(e, dict) and e.get("available") and e.get("test_metrics"):
                out[name] = e["test_metrics"]
        for name, e in (d.get("ensembles") or {}).items():
            if isinstance(e, dict) and e.get("available") and e.get("test_metrics"):
                out[name] = e["test_metrics"]
    else:
        for name, e in (d.get("metrics") or {}).items():
            if isinstance(e, dict):
                out[name] = e
    return out


def test_detail(ds_dir: Path) -> dict[str, Any]:
    return _j(ds_dir / "metrics" / "metrics_test.json")


def val_metrics(ds_dir: Path) -> dict[str, dict[str, Any]]:
    return _j(ds_dir / "metrics" / "metrics_val.json").get("metrics", {})


def preds(ds_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    pdir = ds_dir / "predictions"
    if pdir.exists():
        for f in sorted(pdir.glob("*_test.csv")):
            out[f.name[: -len("_test.csv")]] = pd.read_csv(f)
    return out


def scorecard() -> dict[str, Any]:
    p = P5 / "model_scorecard.json"
    return _j(p) if p.exists() else {"rows": []}


def winner_of(dataset_id: str) -> str | None:
    for r in scorecard().get("rows", []):
        if r.get("dataset") == dataset_id and r.get("winner"):
            return str(r["algorithm"])
    return None


def _leaderboard(ds_dir: Path, key: str, fname: str, title: str,
                 out: dict[int, list[str]], tid: int) -> None:
    tab = metric_table(ds_dir, "test")
    rows = [(n, m.get(key)) for n, m in tab.items()
            if isinstance(m.get(key), (int, float)) and m.get(key) is not None]
    if not rows:
        return
    rows.sort(key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(rows) + 1.6))
    ax.barh([r[0] for r in rows], [r[1] for r in rows],
            color=[PALETTE[i % len(PALETTE)] for i in range(len(rows))])
    for i, (_, v) in enumerate(rows):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=7.5)
    ax.set_title(f"{title} — SEALED TEST ({ds_dir.name})", fontsize=10)
    ax.set_xlabel(key)
    ax.set_xlim(0, 1.05 if max(r[1] for r in rows) <= 1.0 else None)
    _badge(ax, "SEALED TEST — thresholds locked on validation")
    _save(fig, fig_dir(ds_dir) / fname)
    out.setdefault(tid, []).append(str(fig_dir(ds_dir) / fname))


# ---------------------------------------------------------- per-case figures
def _roc_pr(ds_dir: Path, out: dict[int, list[str]], tid_base: int) -> None:
    from sklearn.metrics import auc, precision_recall_curve, roc_curve
    P = preds(ds_dir)
    if not P:
        return
    fig_r, ax_r = plt.subplots(figsize=(6.4, 5))
    fig_p, ax_p = plt.subplots(figsize=(6.4, 5))
    for i, (name, df) in enumerate(sorted(P.items())):
        col = "p_ens" if "p_ens" in df.columns else (
            "p_calibrated" if "p_calibrated" in df.columns else "p_raw")
        y, s = df["y_true"].to_numpy(int), df[col].to_numpy(float)
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        rec, pre, _ = precision_recall_curve(y, s)
        c = PALETTE[i % len(PALETTE)]
        lw = 2.2 if name in ("ensemble", "gradient_boosting") else 1.1
        ax_r.plot(fpr, tpr, color=c, lw=lw,
                  label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        ax_p.plot(rec, pre, color=c, lw=lw)
    ax_r.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax_r.set(xlabel="False positive rate", ylabel="True positive rate",
             title=f"ROC — SEALED TEST ({ds_dir.name})")
    ax_r.legend(fontsize=6.5, loc="lower right")
    _badge(ax_r, "SEALED TEST (descriptive; thresholds not tuned on test)")
    ax_p.set(xlabel="Recall", ylabel="Precision",
             title=f"Precision-recall — SEALED TEST ({ds_dir.name})")
    _badge(ax_p, "SEALED TEST (descriptive)")
    out.setdefault(tid_base, []).append(str(_save(fig_r, fig_dir(ds_dir) / "roc_curves.png")))
    out.setdefault(tid_base + 1, []).append(str(_save(fig_p, fig_dir(ds_dir) / "precision_recall_curves.png")))


def _confusion(ds_dir: Path, dataset_id: str, out: dict[int, list[str]]) -> None:
    win = winner_of(dataset_id)
    P = preds(ds_dir)
    if not win or win not in P:
        return
    df = P[win]
    y, pred = df["y_true"].to_numpy(int), df["pred"].to_numpy(int)
    labels = [0, 1]
    cm = np.array([[int(((y == a) & (pred == b)).sum()) for b in labels] for a in labels])
    for norm, fname, tid, title in (
            (False, "confusion_matrix_winner.png", 11, "Confusion matrix"),
            (True, "confusion_matrix_normalized_winner.png", 12,
             "Normalized confusion matrix")):
        data = cm.astype(float)
        if norm:
            data = data / np.maximum(data.sum(axis=1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=1 if norm else None)
        for a in range(2):
            for b in range(2):
                txt = f"{data[a, b]:.2f}" if norm else str(int(data[a, b]))
                ax.text(b, a, txt, ha="center", va="center",
                        color="white" if data[a, b] > (0.6 if norm else cm.max() / 2) else "black")
        ax.set_xticks([0, 1], ["no", "yes"])
        ax.set_yticks([0, 1], ["no", "yes"])
        ax.set(xlabel="predicted", ylabel="actual",
               title=f"{title}: {win} — SEALED TEST")
        fig.colorbar(im, fraction=0.046)
        _badge(ax, "SEALED TEST")
        out.setdefault(tid, []).append(str(_save(fig, fig_dir(ds_dir) / fname)))


def _calibration_curves(ds_dir: Path, dataset_id: str, out: dict[int, list[str]]) -> None:
    """Reliability curves for the winner + 3 runners-up (sealed test, descriptive)."""
    P = preds(ds_dir)
    win = winner_of(dataset_id)
    val = val_metrics(ds_dir)
    ranked = sorted((n for n in P if n in val),
                    key=lambda n: val[n].get("macro_f0_5") or -1, reverse=True)
    names = [n for n in ([win] + ranked) if n][:4]
    fig, ax = plt.subplots(figsize=(6, 5))
    edges = np.linspace(0, 1, 11)
    for i, name in enumerate(names):
        df = P[name]
        col = "p_ens" if "p_ens" in df.columns else "p_calibrated"
        y, p = df["y_true"].to_numpy(float), df[col].to_numpy(float)
        idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
        xs, ys, ns = [], [], []
        for b in range(10):
            m = idx == b
            if m.any():
                xs.append(float(p[m].mean())); ys.append(float(y[m].mean())); ns.append(int(m.sum()))
        ax.plot(xs, ys, "o-", color=PALETTE[i], label=f"{name} (ECE bins)", ms=4)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="mean predicted probability", ylabel="observed frequency",
           title=f"Calibration / reliability — SEALED TEST ({ds_dir.name})")
    ax.legend(fontsize=7)
    _badge(ax, "SEALED TEST (descriptive; calibration fitted on validation)")
    out.setdefault(13, []).append(str(_save(fig, fig_dir(ds_dir) / "calibration_reliability.png")))


# ------------------------------------------------- validation threshold curves
def _val_probs(ds_dir: Path, dataset_id: str, names: list[str]):
    """Rebuild calibrated validation probabilities from persisted artifacts."""
    import joblib
    from src.models.model_registry import rebuild_predict
    from src.models.phase5_pipeline import prepare
    prep = prepare(dataset_id)
    rows = prep.splits["validation"]
    X_val, y_val = prep.X.loc[rows], prep.y.loc[rows]
    cal_sel = _j(ds_dir / "calibration" / "calibration.json")
    out: dict[str, np.ndarray] = {}
    for name in names:
        f_path = ds_dir / "models" / f"{name}.joblib"
        if not f_path.exists():
            continue
        p = np.asarray(rebuild_predict(joblib.load(f_path))(X_val), dtype=float)
        if p.ndim == 1 and (cal_sel.get(name) or {}).get("selected", "raw") != "raw":
            cal = joblib.load(ds_dir / "calibration" / f"{name}_calibrator.joblib").get("calibrator")
            if cal is not None:
                p = np.asarray(cal.predict(p), dtype=float)
        out[name] = p
    return prep, X_val, y_val, out


def _threshold_curves(ds_dir: Path, dataset_id: str, out: dict[int, list[str]]) -> None:
    from src.evaluation.phase5_metrics import compute_full_metrics
    win = winner_of(dataset_id)
    val = val_metrics(ds_dir)
    if not win or not val:
        return
    ranked = sorted(val, key=lambda n: val[n].get("macro_f0_5") or -1, reverse=True)
    names = [n for n in ([win] + ranked) if n][:4]
    _, _, y_val, probs = _val_probs(ds_dir, dataset_id, names)
    if not probs:
        return
    grid = np.linspace(0.01, 0.99, 99)
    curves: dict[str, dict[str, np.ndarray]] = {}
    for name, p in probs.items():
        acc = {k: [] for k in ("macro_f0_5", "sensitivity", "specificity", "macro_precision")}
        for t in grid:
            m = compute_full_metrics(y_val, p, threshold=float(t))
            for k in acc:
                acc[k].append(m.get(k))
        curves[name] = {k: np.asarray(v, dtype=float) for k, v in acc.items()}
    locked = (_j(ds_dir / "calibration" / "thresholds.json").get("thresholds", {})
              .get(win, {}) or {}).get("threshold")
    specs = ((14, "macro_f0_5", "Macro F0.5", "threshold_vs_macro_f0_5"),
             (15, "sensitivity", "Sensitivity", "threshold_vs_sensitivity"),
             (16, "specificity", "Specificity", "threshold_vs_specificity"),
             (17, "macro_precision", "Macro precision", "threshold_vs_precision"))
    for tid, key, label, fname in specs:
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        for i, name in enumerate(curves):
            ax.plot(grid, curves[name][key], color=PALETTE[i],
                    lw=2.2 if name == win else 1.1,
                    label=name + (" (winner)" if name == win else ""))
        if locked is not None:
            ax.axvline(float(locked), color="black", ls=":", lw=1.2,
                       label=f"locked threshold ({locked})")
        if key in ("sensitivity", "macro_f0_5"):
            ax.axhline(0.90, color="#dc2626", ls="--", lw=1,
                       label="sensitivity floor 0.90")
        ax.set(xlabel="decision threshold", ylabel=label,
               title=f"Threshold vs {label} — DEVELOPMENT / VALIDATION ({dataset_id})")
        ax.legend(fontsize=6.5)
        _badge(ax, "DEVELOPMENT / VALIDATION — test never used")
        out.setdefault(tid, []).append(str(_save(fig, fig_dir(ds_dir) / f"{fname}.png")))
    # 18: sensitivity-specificity operating point (validation)
    fig, ax = plt.subplots(figsize=(6, 4.6))
    for i, name in enumerate(curves):
        ax.plot(curves[name]["specificity"], curves[name]["sensitivity"],
                color=PALETTE[i], lw=2.2 if name == win else 1.1,
                label=name + (" (winner)" if name == win else ""))
        if name == win and locked is not None:
            j = int(np.argmin(np.abs(grid - float(locked))))
            ax.scatter([curves[name]["specificity"][j]], [curves[name]["sensitivity"][j]],
                       marker="*", s=180, color="black", zorder=5,
                       label=f"locked operating point ({locked})")
    ax.axhline(0.90, color="#dc2626", ls="--", lw=1, label="sensitivity floor 0.90")
    ax.set(xlabel="specificity", ylabel="sensitivity",
           title=f"Operating point — DEVELOPMENT / VALIDATION ({dataset_id})")
    ax.legend(fontsize=6.5)
    _badge(ax, "DEVELOPMENT / VALIDATION")
    out.setdefault(18, []).append(str(_save(fig, fig_dir(ds_dir) / "operating_point_validation.png")))


# ------------------------------------------- times, complexity, ensembles, CIs
def _times_complexity(ds_dir: Path, out: dict[int, list[str]]) -> None:
    det = test_detail(ds_dir)
    models = {n: e for n, e in (det.get("models") or {}).items()
              if isinstance(e, dict) and e.get("available")}
    if not models:
        return
    for tid, fname, key, title in (
            (24, "training_time_comparison.png", "training_seconds",
             "Training time"),
            (25, "inference_time_comparison.png", "inference_ms_per_case",
             "Inference time (ms/case)")):
        rows = [(n, e.get(key)) for n, e in models.items()
                if isinstance(e.get(key), (int, float))]
        if not rows:
            continue
        rows.sort(key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.4))
        ax.barh([r[0] for r in rows], [r[1] for r in rows], color="#4b5563")
        ax.set_title(f"{title} — SEALED TEST ({ds_dir.name})", fontsize=10)
        ax.set_xlabel("seconds" if key == "training_seconds" else "ms per case")
        _badge(ax, "SEALED TEST")
        out.setdefault(tid, []).append(str(_save(fig, fig_dir(ds_dir) / fname)))
    # 26: performance vs complexity
    tab = metric_table(ds_dir, "test")
    pts = [(n, e.get("training_seconds"), tab.get(n, {}).get("macro_f0_5"))
           for n, e in models.items()]
    pts = [(n, t, m) for n, t, m in pts if t and m is not None]
    if pts:
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        xs = [max(float(t), 1e-3) for _, t, _ in pts]
        ys = [float(m) for _, _, m in pts]
        ax.scatter(xs, ys, s=46, color="#2563eb")
        for (n, _, _), x, y in zip(pts, xs, ys):
            ax.annotate(n, (x, y), fontsize=6, xytext=(4, 3),
                        textcoords="offset points")
        ax.set_xscale("log")
        ax.set(xlabel="training time (s, log scale)", ylabel="Macro F0.5",
               title=f"Performance vs complexity — SEALED TEST ({ds_dir.name})")
        _badge(ax, "SEALED TEST")
        out.setdefault(26, []).append(str(_save(fig, fig_dir(ds_dir) / "performance_vs_complexity.png")))


def _boosting_and_ensembles(ds_dir: Path, dataset_id: str, out: dict[int, list[str]]) -> None:
    tab = metric_table(ds_dir, "test")
    val = val_metrics(ds_dir)
    boosters = ["gradient_boosting", "hist_gradient_boosting", "adaboost",
                "xgboost", "lightgbm", "catboost"]
    boosters = [b for b in boosters if b in tab]
    if boosters:
        metrics3 = ["macro_f0_5", "auroc", "auprc"]
        x = np.arange(len(metrics3))
        w = 0.8 / len(boosters)
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        for i, b in enumerate(boosters):
            vals = [tab[b].get(m) or 0.0 for m in metrics3]
            ax.bar(x + i * w, vals, w, label=b, color=PALETTE[i % len(PALETTE)])
        ax.set_xticks(x + w * (len(boosters) - 1) / 2, metrics3)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Boosting-model comparison — SEALED TEST ({ds_dir.name})")
        ax.legend(fontsize=6.5, ncol=3)
        _badge(ax, "SEALED TEST")
        out.setdefault(29, []).append(str(_save(fig, fig_dir(ds_dir) / "boosting_models_comparison.png")))
    # 30: ensembles vs top individuals (ranked by VALIDATION, plotted on test)
    ens = [n for n, e in (test_detail(ds_dir).get("ensembles") or {}).items()
           if isinstance(e, dict) and e.get("available") and n in tab]
    top = sorted((n for n in val if n in tab),
                 key=lambda n: val[n].get("macro_f0_5") or -1, reverse=True)[:5]
    names = ens + [n for n in top if n not in ens]
    if len(ens) >= 1 and names:
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        vals = [tab[n].get("macro_f0_5") or 0.0 for n in names]
        colors = ["#dc2626" if n in ens else "#2563eb" for n in names]
        ax.bar(range(len(names)), vals, color=colors)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(names)), names, rotation=25, ha="right", fontsize=7.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Macro F0.5")
        ax.set_title(f"Ensemble (red) vs individual models (blue) — SEALED TEST ({ds_dir.name})")
        _badge(ax, "SEALED TEST; individuals ranked by VALIDATION")
        out.setdefault(30, []).append(str(_save(fig, fig_dir(ds_dir) / "ensemble_vs_individual.png")))


def _confidence_intervals(ds_dir: Path, dataset_id: str, out: dict[int, list[str]]) -> None:
    """Top-6 by VALIDATION Macro F0.5 with their sealed-test bootstrap CIs."""
    det = test_detail(ds_dir)
    val = val_metrics(ds_dir)
    ranked = sorted((n for n in (det.get("models") or {}) if n in val),
                    key=lambda n: val[n].get("macro_f0_5") or -1, reverse=True)[:6]
    rows = []
    for n in ranked:
        e = det["models"][n]
        ci = (e.get("bootstrap_ci") or {}).get("macro_f0_5")
        if ci:
            rows.append((n, ci["point"], ci["lo"], ci["hi"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(rows) + 1.5))
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], color="#2563eb", alpha=0.85)
    ax.errorbar([r[1] for r in rows], list(ys),
                xerr=[[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]],
                fmt="none", ecolor="black", capsize=3)
    ax.set_yticks(list(ys), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Macro F0.5 with bootstrap 95% CI")
    ax.set_title(f"Leading models (ranked by VALIDATION) — SEALED TEST ({ds_dir.name})")
    _badge(ax, "SEALED TEST — 2000-replicate bootstrap CI")
    out.setdefault(35, []).append(str(_save(fig, fig_dir(ds_dir) / "confidence_intervals_leading_models.png")))


def _generalization_gap(ds_dir: Path, dataset_id: str, out: dict[int, list[str]]) -> None:
    """Validation vs sealed-test Macro F0.5 per model (generalization gap = 34)."""
    val = val_metrics(ds_dir)
    tab = metric_table(ds_dir, "test")
    names = sorted((n for n in val if n in tab),
                   key=lambda n: val[n].get("macro_f0_5") or -1, reverse=True)
    if not names:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(names) + 1.7))
    ys = np.arange(len(names))
    v = [val[n].get("macro_f0_5") or np.nan for n in names]
    t = [tab[n].get("macro_f0_5") or np.nan for n in names]
    for i in range(len(names)):
        ax.plot([v[i], t[i]], [i, i], color="#94a3b8", lw=2, zorder=1)
    ax.scatter(v, ys, color="#16a34a", label="validation", zorder=3, s=34)
    ax.scatter(t, ys, color="#dc2626", label="sealed test", zorder=3, s=34)
    ax.set_yticks(ys, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Macro F0.5")
    ax.set_title(f"Generalization gap: validation -> sealed test ({dataset_id})")
    ax.legend(fontsize=7)
    _badge(ax, "validation vs SEALED TEST — gaps are descriptive, never re-tuned")
    out.setdefault(34, []).append(str(_save(fig, fig_dir(ds_dir) / "generalization_gap_validation_vs_test.png")))


# ------------------------------------------------ cross-dataset + bias figures
def _cross_dataset(out: dict[int, list[str]]) -> None:
    p = P5 / "cross_dataset_summary.json"
    if not p.exists():
        return
    data = _j(p)
    ds = [d for d in data.get("datasets", []) if d.get("executed")]
    if not ds:
        return
    labels = [d["dataset"].replace("REG_ENSBURG_", "").replace("_REGISTRY", "")
              for d in ds]
    x = np.arange(len(ds))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - w / 2, [d.get("winner_val_macro_f0_5") or 0 for d in ds], w,
           label="winner — validation", color="#16a34a")
    ax.bar(x + w / 2, [d.get("winner_test_macro_f0_5") or 0 for d in ds], w,
           label="winner — sealed test", color="#dc2626")
    for i, d in enumerate(ds):
        gap = d.get("generalization_gap_val_minus_test")
        if gap is not None:
            ax.text(i, max(d.get("winner_val_macro_f0_5") or 0,
                           d.get("winner_test_macro_f0_5") or 0) + 0.03,
                    f"gap {gap:+.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Macro F0.5")
    ax.set_title("Cross-dataset comparison — separate experiments, never concatenated")
    ax.legend(fontsize=8)
    _badge(ax, "validation vs SEALED TEST per dataset")
    target = P5 / "figures"
    out.setdefault(31, []).append(str(_save(fig, target / "cross_dataset_comparison.png")))
    # 34 also gets a cross-dataset gap view when >1 dataset executed
    if len(ds) > 1:
        fig2, ax2 = plt.subplots(figsize=(6.4, 4))
        gaps = [d.get("generalization_gap_val_minus_test") or 0 for d in ds]
        ax2.bar(labels, gaps, color="#f59e0b")
        for i, g in enumerate(gaps):
            ax2.text(i, g, f"{g:+.3f}", ha="center", fontsize=8)
        ax2.set_ylabel("validation - sealed test Macro F0.5")
        ax2.set_title("External-validation generalization gap per dataset")
        _badge(ax2, "descriptive gap — no re-tuning after test opening")
        out.setdefault(34, []).append(str(_save(fig2, target / "external_validation_gap.png")))


def _bias_figures(out: dict[int, list[str]]) -> None:
    """CBR comparison + BCR vs HFR from persisted Phase 4 bias_metrics.json."""
    p = ROOT / "outputs" / "metrics" / "bias_metrics.json"
    if not p.exists():
        return
    bm = _j(p)
    target = P5 / "figures"
    # profile -> condition -> headline
    series: dict[str, dict[str, dict]] = {}
    anchor = bm.get("anchor_experiment", {})
    if anchor.get("conditions"):
        series[str(anchor.get("profile", "multi_agent_engine"))] = anchor["conditions"]
    for prof, conds in (bm.get("single_baselines") or {}).items():
        series[f"single: {prof}"] = conds
    for prof, conds in (bm.get("ablations") or {}).items():
        for cond, head in (conds or {}).items():
            series.setdefault(f"ablation: {prof}", {})[cond] = head
    if not series:
        return
    # 32: CBR by profile for control vs incorrect_anchor
    profs = [k for k in series
             if all(c in series[k] for c in ("control", "incorrect_anchor"))]
    if profs:
        cbr_c = [(series[k]["control"].get("CBR") or {}).get("value") for k in profs]
        cbr_i = [(series[k]["incorrect_anchor"].get("CBR") or {}).get("value") for k in profs]
        x = np.arange(len(profs))
        w = 0.38
        fig, ax = plt.subplots(figsize=(8.6, 4.6))
        ax.bar(x - w / 2, [v or 0 for v in cbr_c], w, label="control anchor", color="#64748b")
        ax.bar(x + w / 2, [v if v is not None else 0 for v in cbr_i], w,
               label="incorrect anchor", color="#dc2626",
               hatch=["" if v is not None else "//" for v in cbr_i])
        ax.set_xticks(x, profs, rotation=20, ha="right", fontsize=7.5)
        ax.set_ylabel("CBR (confirmation-bias rate)")
        ax.set_title("CBR comparison across profiles (LightGBM backbone, appendicitis)")
        ax.legend(fontsize=8)
        _badge(ax, "persisted outputs/metrics/bias_metrics.json — cross-backbone CBR: NOT GENERATED")
        out.setdefault(32, []).append(str(_save(fig, target / "cbr_comparison_profiles.png")))
    # 33: BCR vs HFR pairs where both exist
    pairs = []
    for prof, conds in series.items():
        for cond, head in conds.items():
            bcr = (head.get("BCR") or {}).get("value")
            hfr = (head.get("HFR") or {}).get("value")
            if bcr is not None and hfr is not None:
                pairs.append((prof, cond, float(bcr), float(hfr)))
    if len(pairs) >= 3:
        fig, ax = plt.subplots(figsize=(7, 4.8))
        for i, (prof, cond, b, h) in enumerate(pairs):
            ax.scatter(b, h, s=54, color=PALETTE[i % len(PALETTE)])
            ax.annotate(f"{prof.split(':')[-1].strip()} / {cond}", (b, h),
                        fontsize=6, xytext=(4, 3), textcoords="offset points")
        ax.set(xlabel="BCR (beneficial correction rate)", ylabel="HFR (harmful flip rate)",
               title="BCR vs HFR trade-off across profiles/conditions")
        _badge(ax, "persisted bias_metrics.json — profiles with non-null HFR only")
        out.setdefault(33, []).append(str(_save(fig, target / "bcr_vs_hfr_tradeoff.png")))
    else:
        # not enough joint (BCR, HFR) points — plot side-by-side where available
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        for ax, metric in zip(axes, ("BCR", "HFR")):
            vals, labs = [], []
            for prof, conds in series.items():
                v = (conds.get("incorrect_anchor", {}).get(metric) or {}).get("value")
                if v is not None:
                    labs.append(prof); vals.append(float(v))
            ax.barh(labs, vals, color="#0891b2")
            ax.set_title(f"{metric} (incorrect-anchor condition)")
            ax.tick_params(labelsize=6.5)
        fig.suptitle("BCR vs HFR across profiles (separate panels: joint points too few)")
        _badge(axes[1], "persisted bias_metrics.json")
        out.setdefault(33, []).append(str(_save(fig, target / "bcr_vs_hfr_tradeoff.png")))


# ---------------------------------------------------- model-specific analyses
def _quick_pipe(name: str, X_train, y_train, numeric, categorical, *,
                seed: int = SEED, **params):
    """Fit one zoo model without CV search (development diagnostics only)."""
    from sklearn.pipeline import Pipeline
    from src.models.model_registry import _make_estimator, _onehot, load_zoo
    spec = load_zoo()["tabular"][name]
    pre = _onehot(numeric, categorical,
                  scale=(spec.get("preprocess") == "onehot_scale"))
    clf = _make_estimator(name, spec, False, spec.get("class_weight"), seed)
    if params:
        clf.set_params(**params)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    return pipe


def _f05_at_half(y_true, proba) -> float:
    from src.evaluation.phase5_metrics import macro_f05
    return macro_f05(np.asarray(y_true), (np.asarray(proba) >= 0.5).astype(int))


def _stratified_subset(y: pd.Series, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = []
    for cls in np.unique(y.to_numpy()):
        cls_idx = np.flatnonzero(y.to_numpy() == cls)
        k = max(1, int(round(frac * len(cls_idx))))
        idx.append(rng.choice(cls_idx, size=k, replace=False))
    return np.sort(np.concatenate(idx))


def _feature_names(pipe) -> list[str]:
    try:
        return [str(x) for x in pipe.named_steps["pre"].get_feature_names_out()]
    except Exception:
        return []


def _learning_curves(prep, out: dict[int, list[str]]) -> None:
    """19: training-fraction vs validation Macro F0.5 (DEVELOPMENT / VALIDATION)."""
    from src.models.model_registry import load_zoo
    tr = prep.splits["train"]; va = prep.splits["validation"]
    X_tr, y_tr = prep.X.loc[tr], prep.y.loc[tr]
    X_va, y_va = prep.X.loc[va], prep.y.loc[va]
    fracs = [0.25, 0.5, 0.75, 1.0]
    models = [m for m in ("logistic_regression", "random_forest",
                          "gradient_boosting", "knn")
              if m in load_zoo()["tabular"]]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for i, m in enumerate(models):
        ys = []
        for f in fracs:
            sub = _stratified_subset(y_tr, f, SEED)
            pipe = _quick_pipe(m, X_tr.iloc[sub], y_tr.iloc[sub],
                               prep.numeric, prep.categorical)
            ys.append(_f05_at_half(y_va, pipe.predict_proba(X_va)[:, 1]))
        ax.plot([int(f * len(tr)) for f in fracs], ys, "o-", color=PALETTE[i],
                label=m)
    ax.set(xlabel="training samples used", ylabel="Macro F0.5 @0.5",
           title="Learning curves — DEVELOPMENT / VALIDATION")
    ax.legend(fontsize=7)
    _badge(ax, "DEVELOPMENT / VALIDATION — default hyperparameters, no search")
    out.setdefault(19, []).append(str(_save(fig, fig_dir(prep.out_dir) / "learning_curves.png")))


def _native_importance(prep, out: dict[int, list[str]]) -> None:
    """20: persisted tree-model importances (native)."""
    import joblib
    entries = []
    for m in ("random_forest", "extra_trees", "gradient_boosting"):
        p = prep.out_dir / "models" / f"{m}.joblib"
        if not p.exists():
            continue
        est = joblib.load(p).get("model")
        if est is None or not hasattr(est, "named_steps"):
            continue
        imp = est.named_steps["clf"].feature_importances_
        names = _feature_names(est)
        if not names or len(names) != len(imp):
            names = [f"f{i}" for i in range(len(imp))]
        order = np.argsort(imp)[::-1][:12]
        entries.append((m, [names[j] for j in order], imp[order]))
    if not entries:
        return
    fig, axes = plt.subplots(1, len(entries), figsize=(4.4 * len(entries), 4.6),
                             squeeze=False)
    for ax, (m, names, imp) in zip(axes[0], entries):
        ax.barh(range(len(names))[::-1], imp, color="#16a34a")
        ax.set_yticks(range(len(names))[::-1], names, fontsize=6.5)
        ax.set_title(f"{m} (native importance)", fontsize=9)
    fig.suptitle("Native feature importance — persisted models (top-12)")
    _badge(axes[0][-1], "persisted outputs/phase5/*/models/*.joblib")
    out.setdefault(20, []).append(str(_save(fig, fig_dir(prep.out_dir) / "native_feature_importance.png")))


def _permutation_importance(prep, out: dict[int, list[str]]) -> None:
    """21: permutation importance on VALIDATION for RF + LogReg pipelines."""
    import joblib
    from sklearn.inspection import permutation_importance
    va = prep.splits["validation"]
    X_va, y_va = prep.X.loc[va], prep.y.loc[va]
    entries = []
    for m in ("random_forest", "logistic_regression"):
        p = prep.out_dir / "models" / f"{m}.joblib"
        if not p.exists():
            continue
        est = joblib.load(p).get("model")
        if est is None:
            continue
        r = permutation_importance(est, X_va, y_va, n_repeats=10,
                                   random_state=SEED, scoring="roc_auc", n_jobs=-1)
        names = _feature_names(est)
        if not names or len(names) != len(r.importances_mean):
            names = [f"f{i}" for i in range(len(r.importances_mean))]
        order = np.argsort(r.importances_mean)[::-1][:12]
        entries.append((m, [names[j] for j in order],
                        r.importances_mean[order], r.importances_std[order]))
    if not entries:
        return
    fig, axes = plt.subplots(1, len(entries), figsize=(4.6 * len(entries), 4.6),
                             squeeze=False)
    for ax, (m, names, mean, std) in zip(axes[0], entries):
        ax.barh(range(len(names))[::-1], mean, xerr=std, color="#2563eb", alpha=0.9)
        ax.set_yticks(range(len(names))[::-1], names, fontsize=6.5)
        ax.set_title(f"{m} (permutation, ROC AUC)", fontsize=9)
    fig.suptitle("Permutation importance — VALIDATION partition, 10 repeats")
    _badge(axes[0][-1], "DEVELOPMENT / VALIDATION")
    out.setdefault(21, []).append(str(_save(fig, fig_dir(prep.out_dir) / "permutation_importance.png")))


# ------------------------------------------------------------- SHAP figures
def _shap_figures(prep, out: dict[int, list[str]]) -> None:
    """22/23: SHAP summary + waterfall for the persisted LightGBM model.

    SHAP explains the model's prediction function — it is never a causal claim.
    """
    import joblib
    import shap
    p = prep.out_dir / "models" / "lightgbm.joblib"
    if not p.exists():
        return
    payload = joblib.load(p)
    model = payload.get("model")
    art = payload.get("artifacts") or {}
    state = art.get("state")
    if model is None or state is None:
        return
    va = prep.splits["validation"]
    X_enc = state.apply(prep.X.loc[va])
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_enc)
    if isinstance(raw, (list, tuple)):
        vals = np.asarray(raw[1] if len(raw) > 1 else raw[0])
        ev = float(explainer.expected_value[1])
    else:
        arr = np.asarray(raw)
        vals = arr[0] if arr.ndim == 3 and arr.shape[0] == 1 else (
            arr[..., 1] if arr.ndim == 3 else arr)
        ev_arr = np.asarray(explainer.expected_value)
        ev = float(ev_arr.reshape(-1)[-1]) if ev_arr.ndim else float(ev_arr)
    wording = "SHAP value: feature contribution to this model prediction; not causal."
    # 22 summary
    try:
        fig = plt.gcf()
        shap.summary_plot(vals, X_enc, show=False, max_display=12)
        ax = plt.gca()
        ax.set_title("SHAP summary — persisted LightGBM, VALIDATION cases")
        _badge(ax, "model attribution (non-causal) — DEVELOPMENT / VALIDATION")
        out.setdefault(22, []).append(str(_save(plt.gcf(), fig_dir(prep.out_dir) / "shap_summary.png")))
        plt.close("all")
    except Exception as exc:
        log(f"shap summary failed: {exc}")
    # 23 waterfall for the first validation case
    i = 0
    try:
        exp = shap.Explanation(values=vals[i], base_values=ev,
                               data=X_enc.iloc[i].to_numpy(float),
                               feature_names=list(X_enc.columns))
        shap.plots.waterfall(exp, max_display=12, show=False)
        ax = plt.gca()
        ax.set_title(f"SHAP waterfall — validation case {va[i]} (non-causal)")
        out.setdefault(23, []).append(str(_save(plt.gcf(), fig_dir(prep.out_dir) / "shap_waterfall.png")))
        plt.close("all")
    except Exception:
        plt.close("all")
        order = np.argsort(np.abs(vals[i]))[::-1][:12]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.barh(range(len(order))[::-1], vals[i][order],
                color=["#dc2626" if vals[i][j] > 0 else "#2563eb" for j in order])
        ax.set_yticks(range(len(order))[::-1], [X_enc.columns[j] for j in order],
                      fontsize=6.5)
        ax.set_xlabel("SHAP value (red = toward appendicitis)")
        ax.set_title(f"SHAP waterfall (bar fallback) — validation case {va[i]}")
        _badge(ax, wording)
        out.setdefault(23, []).append(str(_save(fig, fig_dir(prep.out_dir) / "shap_waterfall.png")))


# ------------------------------------------------- hyperparameter response curves
def _param_curves(prep, out: dict[int, list[str]]) -> None:
    """27 RF n_estimators, 28 KNN K, SVM C/kernel — validation responses."""
    tr, va = prep.splits["train"], prep.splits["validation"]
    X_tr, y_tr = prep.X.loc[tr], prep.y.loc[tr]
    X_va, y_va = prep.X.loc[va], prep.y.loc[va]
    # 27 RF tree count
    ns = [50, 100, 200, 400]
    ys = []
    for n in ns:
        pipe = _quick_pipe("random_forest", X_tr, y_tr, prep.numeric,
                           prep.categorical, n_estimators=n)
        ys.append(_f05_at_half(y_va, pipe.predict_proba(X_va)[:, 1]))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(ns, ys, "o-", color="#16a34a")
    for x, y in zip(ns, ys):
        ax.annotate(f"{y:.3f}", (x, y), fontsize=7, xytext=(0, 5),
                    textcoords="offset points", ha="center")
    ax.set(xlabel="Random Forest n_estimators", ylabel="Macro F0.5 @0.5",
           title="Random Forest tree-count vs validation Macro F0.5")
    _badge(ax, "DEVELOPMENT / VALIDATION")
    out.setdefault(27, []).append(str(_save(fig, fig_dir(prep.out_dir) / "random_forest_n_estimators_curve.png")))
    # 28 KNN K
    ks = [1, 3, 5, 11, 21, 31]
    f05, sens = [], []
    from src.evaluation.phase5_metrics import macro_f05, compute_full_metrics
    for k in ks:
        pipe = _quick_pipe("knn", X_tr, y_tr, prep.numeric, prep.categorical,
                           n_neighbors=k)
        pr = pipe.predict_proba(X_va)[:, 1]
        f05.append(macro_f05(y_va.to_numpy(int), (pr >= 0.5).astype(int)))
        sens.append(compute_full_metrics(y_va.to_numpy(int), pr,
                                         threshold=0.5).get("sensitivity"))
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(ks, f05, "o-", color="#2563eb", label="Macro F0.5 @0.5")
    ax.plot(ks, sens, "s--", color="#dc2626", label="sensitivity @0.5")
    ax.set(xlabel="K (n_neighbors)", ylabel="score",
           title="KNN K curve — DEVELOPMENT / VALIDATION")
    ax.legend()
    _badge(ax, "DEVELOPMENT / VALIDATION")
    out.setdefault(28, []).append(str(_save(fig, fig_dir(prep.out_dir) / "knn_k_curve.png")))
    # SVM kernel + C
    cs = [0.01, 0.1, 1.0, 10.0]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for i, m in enumerate(("svm_linear", "svm_rbf")):
        ys = []
        for c in cs:
            pipe = _quick_pipe(m, X_tr, y_tr, prep.numeric, prep.categorical, C=c)
            ys.append(_f05_at_half(y_va, pipe.predict_proba(X_va)[:, 1]))
        ax.plot(cs, ys, "o-", color=PALETTE[i], label=m)
    ax.set_xscale("log")
    ax.set(xlabel="C (log scale)", ylabel="Macro F0.5 @0.5",
           title="SVM kernel / C comparison — DEVELOPMENT / VALIDATION")
    ax.legend()
    _badge(ax, "DEVELOPMENT / VALIDATION")
    out.setdefault(0, []).append(str(_save(fig, fig_dir(prep.out_dir) / "svm_kernel_c_curve.png")))


# -------------------------------------------- linear / tree interpretability
def _logreg_coefficients(prep, out: dict[int, list[str]]) -> None:
    tr = prep.splits["train"]
    pipe = _quick_pipe("logistic_regression", prep.X.loc[tr], prep.y.loc[tr],
                       prep.numeric, prep.categorical)
    names = _feature_names(pipe)
    coef = pipe.named_steps["clf"].coef_.ravel()
    if not names or len(names) != len(coef):
        names = [f"f{i}" for i in range(len(coef))]
    order = np.argsort(np.abs(coef))[::-1][:15][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
    c = coef[order]
    axes[0].barh(range(len(order)), c, color=["#dc2626" if v > 0 else "#2563eb" for v in c])
    axes[0].set_yticks(range(len(order)), [names[j] for j in order], fontsize=7)
    axes[0].set_xlabel("coefficient (log-odds)")
    axes[0].set_title("Logistic Regression coefficients (top-15 |β|)")
    orr = np.exp(c)
    axes[1].barh(range(len(order)), orr, color=["#dc2626" if v > 1 else "#16a34a" for v in orr])
    axes[1].set_yticks(range(len(order)), [names[j] for j in order], fontsize=7)
    axes[1].axvline(1.0, color="black", ls="--", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("odds ratio exp(β), log scale")
    axes[1].set_title("Odds ratios (OR = e^β)")
    fig.suptitle("Logistic Regression — fitted on TRAIN only (development)")
    _badge(axes[1], "DEVELOPMENT — training-partition fit")
    out.setdefault(0, []).append(str(_save(fig, fig_dir(prep.out_dir) / "logreg_coefficients_odds_ratios.png")))


def _decision_tree_analysis(prep, out: dict[int, list[str]]) -> None:
    import joblib
    from sklearn.tree import plot_tree
    tr = prep.splits["train"]
    X_tr, y_tr = prep.X.loc[tr], prep.y.loc[tr]
    shallow = _quick_pipe("decision_tree", X_tr, y_tr, prep.numeric,
                          prep.categorical, max_depth=3)
    names = _feature_names(shallow)
    fig = plt.figure(figsize=(13.5, 5.4))
    ax1 = fig.add_subplot(1, 2, 1)
    plot_tree(shallow.named_steps["clf"], max_depth=3, class_names=["no", "yes"],
              feature_names=(names or None), filled=True, rounded=True,
              fontsize=6, ax=ax1)
    ax1.set_title("Illustrative shallow tree (max_depth=3, TRAIN fit)")
    ax2 = fig.add_subplot(1, 2, 2)
    p = prep.out_dir / "models" / "decision_tree.joblib"
    if p.exists():
        est = joblib.load(p).get("model")
        imp = est.named_steps["clf"].feature_importances_
        fn = _feature_names(est) or [f"f{i}" for i in range(len(imp))]
        order = np.argsort(imp)[::-1][:12]
        ax2.barh(range(len(order))[::-1], imp[order], color="#f59e0b")
        ax2.set_yticks(range(len(order))[::-1], [fn[j] for j in order], fontsize=7)
        ax2.set_title("Persisted Decision Tree — native importance (top-12)")
    _badge(ax2, "left: DEVELOPMENT tree; right: persisted model")
    out.setdefault(0, []).append(str(_save(fig, fig_dir(prep.out_dir) / "decision_tree_analysis.png")))


def _extra_trees_vs_rf(prep, out: dict[int, list[str]]) -> None:
    val = val_metrics(prep.out_dir)
    tab = metric_table(prep.out_dir, "test")
    det = test_detail(prep.out_dir)
    pair = [m for m in ("random_forest", "extra_trees") if m in val and m in tab]
    if len(pair) != 2:
        return
    metrics = ["macro_f0_5", "auroc", "auprc"]
    x = np.arange(len(metrics))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for i, m in enumerate(pair):
        axes[0].bar(x + (i - 0.5) * w, [val[m].get(k) or 0 for k in metrics], w,
                    label=f"{m} — validation")
        axes[1].bar(x + (i - 0.5) * w, [tab[m].get(k) or 0 for k in metrics], w,
                    label=f"{m} — sealed test")
    for ax, title in zip(axes, ("VALIDATION", "SEALED TEST")):
        ax.set_xticks(x, metrics)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(fontsize=7)
    times = [(m, (det.get("models", {}).get(m) or {}).get("training_seconds"))
             for m in pair]
    fig.suptitle("Extra Trees vs Random Forest — "
                 + ", ".join(f"{m}: {t}s" for m, t in times if t is not None))
    out.setdefault(0, []).append(str(_save(fig, fig_dir(prep.out_dir) / "extra_trees_vs_random_forest.png")))


def _cross_backbone(out: dict[int, list[str]]) -> None:
    """32: cross-backbone CBR/BCR from bias_backbones_summary.json (paired runs)."""
    p = P5 / "bias_backbones_summary.json"
    if not p.exists():
        return
    data = _j(p)
    bbs = data.get("backbones") or {}
    if not bbs:
        return
    engine = "9_full_with_rag"
    single = "3_lightgbm_plus_single_llm"
    cond = "incorrect_anchor"
    names, e_cbr, s_cbr, e_bcr = [], [], [], []
    for name, profs in bbs.items():
        try:
            ec = (profs[engine][cond]["CBR"] or {}).get("value")
            sc = (profs[single][cond]["CBR"] or {}).get("value")
            bc = (profs[engine][cond]["BCR"] or {}).get("value")
        except KeyError:
            continue
        names.append(name)
        e_cbr.append(0.0 if ec is None else float(ec))
        s_cbr.append(0.0 if sc is None else float(sc))
        e_bcr.append(None if bc is None else float(bc))
    if not names:
        return
    x = np.arange(len(names))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(x - w / 2, s_cbr, w, label="single-agent (profile 3)", color="#dc2626")
    axes[0].bar(x + w / 2, e_cbr, w, label="isolated engine (profile 9)", color="#2563eb")
    axes[0].set_xticks(x, names, rotation=20, ha="right", fontsize=7.5)
    axes[0].set_ylabel("CBR (confirmation-bias rate)")
    axes[0].set_title("CBR across model backbones — incorrect anchor")
    axes[0].legend(fontsize=8)
    bcr_vals = [v if v is not None else 0.0 for v in e_bcr]
    axes[1].bar(names, bcr_vals, color="#16a34a")
    for i, v in enumerate(bcr_vals):
        axes[1].text(i, v, f"{v:.3f}", ha="center", fontsize=7.5)
    axes[1].set_xticks(np.arange(len(names)), names, rotation=20, ha="right", fontsize=7.5)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("BCR (beneficial correction rate)")
    axes[1].set_title("BCR of the isolated engine across backbones")
    fig.suptitle("Cross-backbone bias experiment — same rows/conditions/engine, "
                 "only the classifier changes (n=40 test rows, seed 20261002)")
    _badge(axes[1], "outputs/phase5/bias_backbones_summary.json — uniform information budget")
    out.setdefault(32, []).append(str(_save(fig, P5 / "figures" /
                                            "cross_backbone_cbr_bcr.png")))
    # extra: model accuracy per backbone in this experiment (context for the CBR view)
    fig2, ax2 = plt.subplots(figsize=(7, 4.2))
    init_acc, fin_acc = [], []
    for name in names:
        h = bbs[name][engine][cond]
        init_acc.append((h.get("initial_accuracy") or {}).get("value") or 0.0)
        fin_acc.append((h.get("final_accuracy") or {}).get("value") or 0.0)
    ax2.bar(x - w / 2, init_acc, w, label="model initial accuracy", color="#f59e0b")
    ax2.bar(x + w / 2, fin_acc, w, label="engine final accuracy", color="#2563eb")
    ax2.set_xticks(x, names, rotation=20, ha="right", fontsize=7.5)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("accuracy")
    ax2.set_title("Initial vs final accuracy across backbones (incorrect anchor)")
    ax2.legend(fontsize=8)
    _badge(ax2, "persisted bias_backbones_summary.json")
    out.setdefault(0, []).append(str(_save(fig2, P5 / "figures" /
                                           "cross_backbone_accuracy.png")))


# ----------------------------------------------------------------- entry point
_LEADERBOARDS = (
    (1, "macro_f0_5", "leaderboard_macro_f0_5.png", "Macro F0.5"),
    (2, "accuracy", "leaderboard_accuracy.png", "Accuracy"),
    (3, "balanced_accuracy", "leaderboard_balanced_accuracy.png", "Balanced accuracy"),
    (4, "sensitivity", "leaderboard_sensitivity.png", "Sensitivity"),
    (5, "specificity", "leaderboard_specificity.png", "Specificity"),
    (6, "macro_precision", "leaderboard_precision.png", "Precision (macro)"),
    (7, "auroc", "leaderboard_auroc.png", "AUROC"),
    (8, "auprc", "leaderboard_auprc.png", "AUPRC"),
)

NOT_GENERATED_NOTES = {
    32: ("Cross-backbone CBR comparison (LR/RF/XGB/LGBM/CatBoost backbones) — "
         "bias experiments executed only for the Phase 4 LightGBM backbone. "
         "Reproduce when the cross-backbone runner is available: "
         "python scripts/run_phase5.py bias --dataset REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR "
         "--backbone <logistic_regression|random_forest|xgboost|lightgbm|catboost|ensemble>"),
}


def generate_all() -> int:
    """Generate every Phase 5 figure from persisted outputs; write the manifest."""
    t0 = time.time()
    out: dict[int, list[str]] = {}
    errors: list[str] = []
    datasets = _datasets()
    for ds_id, ds_dir in datasets:
        log(f"{ds_id}: core figures")
        for tid, key, fname, title in _LEADERBOARDS:
            try:
                _leaderboard(ds_dir, key, fname, title, out, tid)
            except Exception as exc:
                errors.append(f"{ds_id}/{fname}: {type(exc).__name__}: {exc}")
        steps = ((_roc_pr, (ds_dir, out, 9), {}),
                 (_confusion, (ds_dir, ds_id, out), {}),
                 (_calibration_curves, (ds_dir, ds_id, out), {}),
                 (_threshold_curves, (ds_dir, ds_id, out), {}),
                 (_times_complexity, (ds_dir, out), {}),
                 (_boosting_and_ensembles, (ds_dir, ds_id, out), {}),
                 (_confidence_intervals, (ds_dir, ds_id, out), {}),
                 (_generalization_gap, (ds_dir, ds_id, out), {}))
        for fn, args, kw in steps:
            try:
                fn(*args, **kw)
            except Exception as exc:
                errors.append(f"{ds_id}/{fn.__name__}: {type(exc).__name__}: {exc}")
        # model-specific analyses only on the reference (blocker PASS) dataset
        blocker = str((test_detail(ds_dir).get("blocker_status") or "")).upper()
        if blocker == "PASS":
            log(f"{ds_id}: model-specific analyses")
            try:
                from src.models.phase5_pipeline import prepare
                prep = prepare(ds_id)
            except Exception as exc:
                errors.append(f"{ds_id}/prepare: {type(exc).__name__}: {exc}")
                prep = None
            if prep is not None:
                for fn in (_learning_curves, _native_importance,
                           _permutation_importance, _shap_figures,
                           _param_curves, _logreg_coefficients,
                           _decision_tree_analysis, _extra_trees_vs_rf):
                    try:
                        fn(prep, out)
                    except Exception as exc:
                        errors.append(f"{ds_id}/{fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            errors.append(f"{ds_id}: model-specific analyses skipped "
                          f"(blocker={blocker or 'unknown'} — reference dataset only)")
    try:
        _cross_dataset(out)
    except Exception as exc:
        errors.append(f"_cross_dataset: {type(exc).__name__}: {exc}")
    try:
        _bias_figures(out)
    except Exception as exc:
        errors.append(f"_bias_figures: {type(exc).__name__}: {exc}")
    try:
        _cross_backbone(out)
    except Exception as exc:
        errors.append(f"_cross_backbone: {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------- manifest
    figures = []
    for tid, name in TARGETS:
        paths = [p for p in out.get(tid, []) if p]
        entry: dict[str, Any] = {
            "id": tid, "name": name,
            "status": "GENERATED" if paths else "NOT_GENERATED",
            "paths": paths,
            "reproduce": REPRODUCE,
        }
        if not paths and tid in NOT_GENERATED_NOTES:
            entry["note"] = NOT_GENERATED_NOTES[tid]
        elif not paths:
            entry["note"] = "not generated in this environment — see errors[]"
        figures.append(entry)
    extras = [p for p in out.get(0, []) if p]
    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "persisted outputs/phase5 + outputs/metrics artifacts only",
        "reproduce": REPRODUCE,
        "datasets_executed": [d for d, _ in datasets],
        "figures": figures,
        "model_specific_analysis_extras": extras,
        "already_existed_pre_phase5": [
            str(ROOT / "outputs/figures" / f)
            for f in ("fig_anchor_cbr_aor.png", "fig_bias_reduction.png",
                      "shap_summary.png", "shap_waterfall_case_test_row_289.png")
            if (ROOT / "outputs/figures" / f).exists()],
        "not_executed": [
            "Image-model training/Grad-CAM (Kermany CXR, Regensburg US): "
            "torch not installed in this environment — NOT_EXECUTED, no figures invented.",
        ],
        "cross_backbone_bias": (
            "EXECUTED: 6 backbones × 160 paired case-runs — "
            "outputs/phase5/bias_backbones_summary.json" if (P5 / "bias_backbones_summary.json").exists()
            else "NOT_EXECUTED — reproduce: python scripts/run_phase5.py bias "
                 "--dataset REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR --backbone all --rows 40"),
        "errors": errors,
        "elapsed_seconds": round(time.time() - t0, 1),
        "disclaimer": "Research prototype — not a medical device.",
    }
    P5.mkdir(parents=True, exist_ok=True)
    (P5 / "figures_manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    n = sum(len([p for p in v if p]) for v in out.values())
    n_gen = sum(1 for f in figures if f["status"] == "GENERATED")
    log(f"{n} files written; {n_gen}/35 target figures GENERATED; "
        f"{len(errors)} notes/errors")
    if errors:
        for e in errors:
            log(f"  note: {e}")
    return n


if __name__ == "__main__":
    raise SystemExit(0 if generate_all() >= 0 else 1)

