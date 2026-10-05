"""Phase 5 cross-model statistical comparison (executed from persisted artifacts).

Every number produced here comes from the per-case test predictions persisted under
``outputs/phase5/<dataset>/predictions/*_test.csv`` during the sealed-test evaluation.
Test labels are used **only for post-hoc inference** (confidence intervals, paired
tests) — never for model selection, threshold tuning or calibration, all of which were
locked on validation before the test partition was opened (see
``metrics/test_evaluation_lock.json``).

Methods (implementations in ``src/evaluation/phase5_metrics.py``):

* paired bootstrap difference in Macro F0.5 (winner vs each competitor), two-sided p;
* exact McNemar test on paired hard predictions;
* DeLong AUROC 95% CI per model (U-statistic form);
* bootstrap 95% CI for the ensemble probabilities (ensembles have no persisted CI);
* Benjamini-Hochberg correction over each dataset's family of paired comparisons.

Outputs::

    outputs/phase5/statistical_comparison.json
    outputs/phase5/statistical_comparison.csv

Research prototype — not a medical device.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.phase5_metrics import (
    bh_family,
    bootstrap_ci,
    compute_full_metrics,
    delong_roc,
    mcnemar,
)

ROOT = Path(__file__).resolve().parents[2]
P5 = ROOT / "outputs/phase5"
SEED = 20261002
N_BOOT = 2000

_DISCLAIMER = (
    "Post-hoc inference on the sealed test partition: test labels were used ONLY for "
    "confidence intervals and paired tests after all models, calibrators and thresholds "
    "were locked on validation. No selection or tuning used test data. Research "
    "prototype — not a medical device."
)


def _macro_f05_rows(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Vectorized binary Macro F0.5 for (B, n) batches.

    Identical definition to ``compute_full_metrics()['macro_f0_5']``: fixed classes
    [0, 1], zero-division = 0 (see src/evaluation/phase5_metrics._binary).
    """
    tp = ((y == 1) & (pred == 1)).sum(axis=1).astype(float)
    fp = ((y == 0) & (pred == 1)).sum(axis=1).astype(float)
    fn = ((y == 1) & (pred == 0)).sum(axis=1).astype(float)
    tn = ((y == 0) & (pred == 0)).sum(axis=1).astype(float)

    def _f(tp_: np.ndarray, fp_: np.ndarray, fn_: np.ndarray) -> np.ndarray:
        p = np.divide(tp_, tp_ + fp_, out=np.zeros_like(tp_), where=(tp_ + fp_) > 0)
        r = np.divide(tp_, tp_ + fn_, out=np.zeros_like(tp_), where=(tp_ + fn_) > 0)
        den = 0.25 * p + r
        return np.divide(1.25 * p * r, den, out=np.zeros_like(den), where=den > 0)

    f_pos = _f(tp, fp, fn)
    f_neg = _f(tn, fn, fp)
    return (f_pos + f_neg) / 2.0


def _paired_delta_fast(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, *,
                       threshold_a: float, threshold_b: float,
                       n_boot: int, seed: int) -> dict[str, Any]:
    """Vectorized paired bootstrap difference in Macro F0.5 (A − B).

    Mathematically identical to
    ``phase5_metrics.paired_bootstrap_delta_f05`` (same seeded resample stream,
    same metric definition) but computed with batched confusion counts — verified
    equal against the reference implementation in tests/test_phase5.py.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n = len(y)
    idx = rng.integers(0, n, size=(n_boot, n))
    yb = y[idx]
    pred_a = (np.asarray(pa)[idx] >= threshold_a).astype(int)
    pred_b = (np.asarray(pb)[idx] >= threshold_b).astype(int)
    deltas = _macro_f05_rows(yb, pred_a) - _macro_f05_rows(yb, pred_b)
    point = float(compute_full_metrics(y, pa, threshold=threshold_a)["macro_f0_5"]
                  - compute_full_metrics(y, pb, threshold=threshold_b)["macro_f0_5"])
    p_two = float(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0)))
    return {"delta_macro_f0_5": point,
            "ci95": [float(np.percentile(deltas, 2.5)),
                     float(np.percentile(deltas, 97.5))],
            "p_value": min(1.0, p_two), "n_boot": n_boot, "seed": seed,
            "implementation": "vectorized (equal to paired_bootstrap_delta_f05)"}


def _scorecard_winner(dataset: str) -> str | None:
    sc_path = P5 / "model_scorecard.json"
    if not sc_path.exists():
        return None
    rows = json.loads(sc_path.read_text(encoding="utf-8")).get("rows", [])
    for r in rows:
        if r.get("dataset") == dataset and r.get("winner"):
            return str(r.get("algorithm"))
    return None


def _load_predictions(dataset_dir: Path) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    pred_dir = dataset_dir / "predictions"
    if not pred_dir.exists():
        return out
    for csv_path in sorted(pred_dir.glob("*_test.csv")):
        name = csv_path.name[: -len("_test.csv")]
        out[name] = pd.read_csv(csv_path)
    return out


def _prob_column(df: pd.DataFrame, metrics_test: dict, model: str) -> tuple[str, bool]:
    """Pick the probability column that reproduces the reported test metrics.

    The evaluation used calibrated probabilities at the locked threshold (models) or
    the ensemble probability column; verify rather than assume.
    """
    entry = (metrics_test.get("models", {}).get(model)
             or metrics_test.get("ensembles", {}).get(model) or {})
    reported = (entry.get("test_metrics") or {}).get("macro_f0_5")
    y = df["y_true"].to_numpy()
    if "threshold" in df.columns:
        thr = float(df["threshold"].iloc[0])
    else:
        # ensembles: the locked threshold lives in metrics_test.json, not the CSV
        thr = float(entry.get("threshold") or 0.5)
    for col in ("p_calibrated", "p_raw", "p_ens"):
        if col not in df.columns or reported is None:
            continue
        val = compute_full_metrics(y, df[col].to_numpy(), threshold=thr)["macro_f0_5"]
        if abs(val - float(reported)) < 1e-6:
            return col, True
    col = "p_calibrated" if "p_calibrated" in df.columns else (
        "p_raw" if "p_raw" in df.columns else "p_ens")
    return col, False


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["dataset", "method", "model_a", "model_b", "metric", "estimate",
              "ci_lo", "ci_hi", "p_value", "q_value", "n", "note"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def write_statistics(dataset_ids: list[str] | None = None, *,
                     n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, Any]:
    """Run all Phase 5 statistical comparisons from persisted predictions."""
    t0 = time.time()
    datasets = dataset_ids or sorted(
        p.name for p in P5.iterdir()
        if p.is_dir() and (p / "metrics" / "metrics_test.json").exists())
    out_rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "generated_from": "outputs/phase5/*/predictions/*_test.csv",
        "n_boot": n_boot, "seed": seed,
        "methods": ["paired_bootstrap_delta_macro_f0_5", "mcnemar_exact",
                    "delong_auroc_ci", "bootstrap_ci_ensemble",
                    "benjamini_hochberg"],
        "test_labels_used_for": "post-hoc inference only (no selection/tuning)",
        "disclaimer": _DISCLAIMER,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "datasets": {},
    }

    for ds in datasets:
        ds_dir = P5 / ds
        metrics_path = ds_dir / "metrics" / "metrics_test.json"
        if not metrics_path.exists():
            continue
        metrics_test = json.loads(metrics_path.read_text(encoding="utf-8"))
        ds_id = str(metrics_test.get("dataset_id") or ds)
        preds = _load_predictions(ds_dir)
        if not preds:
            continue
        winner = _scorecard_winner(ds) or ("gradient_boosting"
                                           if "gradient_boosting" in preds else None)
        if winner is None or winner not in preds:
            continue

        checked: dict[str, str] = {}
        cols: dict[str, str] = {}
        thresholds: dict[str, float] = {}
        for name, df in preds.items():
            col, ok = _prob_column(df, metrics_test, name)
            cols[name] = col
            checked[name] = "reproduces_reported_macro_f0_5" if ok else "unverified"
            entry = (metrics_test.get("models", {}).get(name)
                     or metrics_test.get("ensembles", {}).get(name) or {})
            thresholds[name] = (float(df["threshold"].iloc[0])
                                if "threshold" in df.columns
                                else float(entry.get("threshold") or 0.5))

        wdf = preds[winner]
        y = wdf["y_true"].to_numpy(int)
        w_thr = thresholds[winner]
        w_prob = wdf[cols[winner]].to_numpy(float)
        w_pred = (w_prob >= w_thr).astype(int)

        ds_entry: dict[str, Any] = {
            "winner_selected_on": "validation (see outputs/phase5/model_scorecard.json)",
            "winner": winner, "n_test": int(len(y)),
            "probability_source_check": checked,
            "comparisons": [], "delong_auroc_ci": {}, "ensemble_bootstrap_ci": {},
            "benjamini_hochberg": None}

        # ---- DeLong AUROC CI for every model with probabilities
        for name, df in sorted(preds.items()):
            if cols[name] == "p_ens":
                continue
            res = delong_roc(y, df[cols[name]].to_numpy(float))
            ds_entry["delong_auroc_ci"][name] = res
            out_rows.append({"dataset": ds_id, "method": "delong_auroc_ci",
                             "model_a": name, "model_b": "", "metric": "AUROC",
                             "estimate": res.get("auc"),
                             "ci_lo": (res.get("ci95") or [None, None])[0],
                             "ci_hi": (res.get("ci95") or [None, None])[1],
                             "p_value": None, "q_value": None, "n": len(y),
                             "note": "DeLong 95% CI (single model)"})

        # ---- paired comparisons: winner vs every other model
        p_values: list[float | None] = []
        comp_names: list[str] = []
        for name, df in sorted(preds.items()):
            if name == winner:
                continue
            thr = thresholds[name]
            prob = df[cols[name]].to_numpy(float)
            pred = (prob >= thr).astype(int) if cols[name] != "p_ens" else pred_from(df)
            pair = _paired_delta_fast(y, w_prob, prob,
                                      threshold_a=w_thr, threshold_b=thr,
                                      n_boot=n_boot, seed=seed)
            mcn = mcnemar(y, w_pred, pred)
            comp = {"competitor": name, "paired_bootstrap_delta_macro_f0_5": pair,
                    "mcnemar": mcn}
            ds_entry["comparisons"].append(comp)
            comp_names.append(name)
            p_values.append(pair.get("p_value"))
            out_rows.append({"dataset": ds_id, "method": "paired_bootstrap_delta_macro_f0_5",
                             "model_a": winner, "model_b": name,
                             "metric": "Macro_F0.5_delta",
                             "estimate": pair.get("delta_macro_f0_5"),
                             "ci_lo": (pair.get("ci95") or [None, None])[0],
                             "ci_hi": (pair.get("ci95") or [None, None])[1],
                             "p_value": pair.get("p_value"), "q_value": None,
                             "n": len(y), "note": "A - B, paired test cases"})
            out_rows.append({"dataset": ds_id, "method": "mcnemar_exact",
                             "model_a": winner, "model_b": name,
                             "metric": "discordant_errors",
                             "estimate": f"b={mcn.get('b')},c={mcn.get('c')}",
                             "ci_lo": None, "ci_hi": None,
                             "p_value": mcn.get("p_value"), "q_value": None,
                             "n": len(y), "note": str(mcn.get("test", ""))})

        # ---- Benjamini-Hochberg over this dataset's paired-comparison family
        bh = bh_family(p_values)          # {q_values, rejected, alpha, n_tests}
        q_list = list(bh.get("q_values") or [])
        rej_list = list(bh.get("rejected") or [])
        q_map = dict(zip(comp_names, q_list))
        for name, q in q_map.items():
            for row in out_rows:
                if (row["dataset"] == ds_id and row["method"] == "paired_bootstrap_delta_macro_f0_5"
                        and row["model_b"] == name):
                    row["q_value"] = q
        ds_entry["benjamini_hochberg"] = {
            "family": f"{ds_id}: paired bootstrap Macro F0.5 (winner vs each competitor)",
            "n_tests_raw": len(p_values), "n_tests": bh.get("n_tests"),
            "alpha": bh.get("alpha"), "q_values": q_list, "rejected": rej_list,
            "rejected_q_lt_0_05": [n for n, r in zip(comp_names, rej_list) if r]}

        # ---- bootstrap CI for the ensemble probabilities (no persisted CI)
        if "ensemble" in preds:
            edf = preds["ensemble"]
            e_thr = thresholds["ensemble"]
            ens_ci = bootstrap_ci(y, edf["p_ens"].to_numpy(float),
                                  threshold=e_thr, n_boot=n_boot, seed=seed)
            ds_entry["ensemble_bootstrap_ci"] = {"model": "ensemble", **ens_ci}
            for metric_name, stat in (ens_ci.get("metrics") or ens_ci).items():
                if isinstance(stat, dict) and "lo" in stat:
                    out_rows.append({"dataset": ds_id, "method": "bootstrap_ci",
                                     "model_a": "ensemble", "model_b": "",
                                     "metric": metric_name, "estimate": stat.get("point"),
                                     "ci_lo": stat.get("lo"), "ci_hi": stat.get("hi"),
                                     "p_value": None, "q_value": None, "n": len(y),
                                     "note": "ensemble (validation-weighted soft voting)"})

        payload["datasets"][ds_id] = ds_entry

    payload["elapsed_seconds"] = round(time.time() - t0, 2)
    P5.mkdir(parents=True, exist_ok=True)
    (P5 / "statistical_comparison.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    _write_csv(out_rows, P5 / "statistical_comparison.csv")
    return {"datasets": list(payload["datasets"]), "rows": len(out_rows),
            "json": str(P5 / "statistical_comparison.json")}


def pred_from(df: pd.DataFrame) -> np.ndarray:
    """Hard prediction column for frames without a threshold (hard voting)."""
    if "pred" in df.columns:
        return df["pred"].to_numpy(int)
    if "p_ens" in df.columns:
        return (df["p_ens"].to_numpy(float) >= 0.5).astype(int)
    raise KeyError(f"no prediction column in {list(df.columns)}")
