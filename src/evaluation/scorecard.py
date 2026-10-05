"""Phase 5 model scorecard + cross-dataset summary (sections 11–12 of the spec).

Reads ONLY persisted Phase 5 artifacts — never recomputes, never fabricates:

* ``outputs/phase5/<ds>/metrics/train_summary.json``  (status, hyperparameters, val lock)
* ``outputs/phase5/<ds>/metrics/metrics_val.json``    (validation metrics)
* ``outputs/phase5/<ds>/metrics/metrics_test.json``   (sealed test metrics, if evaluated)

Writes:

* ``outputs/phase5/model_scorecard.csv`` / ``.json`` — one row per (dataset, algorithm)
* ``outputs/phase5/cross_dataset_summary.csv`` / ``.json``
* ``outputs/phase5/external_validation.json`` — honest status of external-cohort validation

Rules enforced here:

* the WINNER is chosen on VALIDATION only (floor compliance → val Macro F0.5 → val AUPRC →
  simplicity tie-break). Test metrics never enter the selection.
* a metric that is mathematically unavailable is ``null`` (empty in CSV) — never 0.

Research prototype — not a medical device.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PHASE5_OUT = ROOT / "outputs/phase5"

SCORECARD_FIELDS = [
    "dataset", "algorithm", "model_family", "status", "hyperparameters",
    "threshold", "Macro_F0.5", "positive_F0.5", "accuracy", "balanced_accuracy",
    "precision", "sensitivity", "specificity", "NPV", "F1", "F2", "MCC",
    "AUROC", "AUPRC", "Brier", "ECE", "training_time", "inference_time",
    "calibration_method", "sensitivity_floor", "sensitivity_floor_pass", "winner",
    "note",
]

# Simpler is preferred on ties (spec 12.6: "if statistically indistinguishable, prefer simpler").
_SIMPLICITY = [
    "logistic_regression", "gaussian_nb", "lda", "decision_tree", "knn",
    "svm_linear", "qda", "svm_rbf", "mlp", "adaboost", "gradient_boosting",
    "hist_gradient_boosting", "xgboost", "lightgbm", "random_forest",
    "extra_trees", "catboost",
    "ensemble_hard_voting", "ensemble_soft_uniform",
    "validation_weighted_soft_voting", "stacking",
]

# Field-name mapping note persisted alongside the scorecard:
#   precision        = positive-class PPV
#   F1 / F2          = macro-averaged F1 / F2
#   Macro_F0.5       = unweighted mean of per-class F0.5
#   AUROC/AUPRC/Brier/ECE for hard voting are mathematically undefined -> null (never 0)


def _load(dataset_id: str) -> dict[str, Any]:
    mdir = PHASE5_OUT / dataset_id / "metrics"
    out: dict[str, Any] = {}
    for name in ("train_summary", "metrics_val", "metrics_test"):
        p = mdir / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _f(x: Any) -> float | None:
    """Null-safe float: unavailable metrics stay None, never 0."""
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _metric_row(dataset_id: str, algo: str, family: str, status: str,
                train_entry: dict, val: dict | None, test: dict | None,
                floor: float, note: str | None = None) -> dict[str, Any]:
    t = test or {}
    return {
        "dataset": dataset_id,
        "algorithm": algo,
        "model_family": family,
        "status": status,
        "hyperparameters": train_entry.get("best_params")
            or train_entry.get("weights") or train_entry.get("method") or None,
        "threshold": _f(t.get("threshold", train_entry.get("threshold"))),
        "Macro_F0.5": _f(t.get("macro_f0_5")),
        "positive_F0.5": _f(t.get("positive_class_f0_5")),
        "accuracy": _f(t.get("accuracy")),
        "balanced_accuracy": _f(t.get("balanced_accuracy")),
        "precision": _f(t.get("ppv")),
        "sensitivity": _f(t.get("sensitivity")),
        "specificity": _f(t.get("specificity")),
        "NPV": _f(t.get("npv")),
        "F1": _f(t.get("macro_f1")),
        "F2": _f(t.get("macro_f2")),
        "MCC": _f(t.get("mcc")),
        "AUROC": _f(t.get("auroc")),
        "AUPRC": _f(t.get("auprc")),
        "Brier": _f(t.get("brier")),
        "ECE": _f(t.get("ece")),
        "training_time": _f(train_entry.get("fit_seconds")
                            or train_entry.get("training_seconds")),
        "inference_time": _f(t.get("inference_seconds")),
        "calibration_method": train_entry.get("calibrator")
            or train_entry.get("calibration_method"),
        "sensitivity_floor": _f(floor),
        "sensitivity_floor_pass": train_entry.get("floor_satisfied"),
        "winner": False,                      # set by select_winner() on VALIDATION
        "note": note or train_entry.get("note"),
        # internal (stripped before write): validation metrics for winner selection
        "_val_macro_f0_5": _f((val or {}).get("macro_f0_5")),
        "_val_auprc": _f((val or {}).get("auprc")),
        "_val_sens": _f((val or {}).get("sensitivity")),
    }


def build_scorecard_rows() -> list[dict[str, Any]]:
    """One scorecard row per (dataset, algorithm) for every executed dataset."""
    rows: list[dict[str, Any]] = []
    for ds_dir in sorted(PHASE5_OUT.iterdir()):
        if not ds_dir.is_dir() or not (ds_dir / "metrics" / "train_summary.json").exists():
            continue
        slug = ds_dir.name
        blob = _load(slug)
        train = blob.get("train_summary", {})
        dataset_id = train.get("dataset_id") or slug
        val_metrics = (blob.get("metrics_val") or {}).get("metrics", {})
        test_blob = blob.get("metrics_test")
        test_models = (test_blob or {}).get("models", {})
        test_ens = (test_blob or {}).get("ensembles", {})
        floor = train.get("sensitivity_floor", 0.90)
        zoo = _zoo_families()

        for algo, entry in (train.get("models") or {}).items():
            status = "AVAILABLE" if entry.get("available") else "UNAVAILABLE"
            tm = test_models.get(algo) or {}
            if tm.get("test_metrics"):
                status = "EVALUATED"
            rows.append(_metric_row(
                dataset_id, algo, zoo.get(algo, "other"), status, entry,
                val_metrics.get(algo), tm.get("test_metrics"), floor,
                note=None if entry.get("available") else entry.get("note")))

        ens_train = train.get("ensemble") or {}
        ens_val_thr = ens_train.get("thresholds") or {}
        # threshold-file keys → test-file keys (same methods, different names)
        ens_name_map = {
            "ensemble_soft_voting": "validation_weighted_soft_voting",
            "ensemble_soft_uniform": "ensemble_soft_uniform",
            "ensemble_stacking": "stacking",
            "ensemble_hard_voting": "ensemble_hard_voting",
        }
        seen: set[str] = set()
        for vkey, algo in ens_name_map.items():
            val_e = ens_val_thr.get(vkey) or ens_val_thr.get(algo) or None
            tm = test_ens.get(algo) or {}
            entry = {
                "available": True,
                "method": (ens_train.get("method") if algo == "validation_weighted_soft_voting"
                           else algo),
                "weights": ens_train.get("weights") if algo == "validation_weighted_soft_voting"
                    else None,
                "threshold": (val_e or {}).get("threshold"),
                "floor_satisfied": (val_e or {}).get("floor_satisfied"),
                "calibrator": "n/a (ensemble probabilities)",
            }
            val_metrics_e = None
            if val_e:
                val_metrics_e = {
                    "macro_f0_5": val_e.get("val_macro_f0_5"),
                    "sensitivity": val_e.get("val_sensitivity"),
                    "auprc": None,   # not computed for ensembles on validation
                }
            seen.add(algo)
            rows.append(_metric_row(
                dataset_id, algo, "ensemble",
                "EVALUATED" if tm.get("test_metrics") else "TRAINED_ONLY",
                entry, val_metrics_e, tm.get("test_metrics"), floor,
                note=entry["method"]))
        for algo in test_ens:                       # anything else persisted at test time
            if algo not in seen:
                tm = test_ens[algo] or {}
                rows.append(_metric_row(
                    dataset_id, algo, "ensemble",
                    "EVALUATED" if tm.get("test_metrics") else "TRAINED_ONLY",
                    {"available": True, "method": algo}, None,
                    tm.get("test_metrics"), floor))
    return rows


def _zoo_families() -> dict[str, str]:
    try:
        import yaml
        zoo = yaml.safe_load((ROOT / "config/model_zoo.yaml").read_text(encoding="utf-8"))
        return {k: str(v.get("family", "other")) for k, v in (zoo.get("tabular") or {}).items()}
    except Exception:
        return {}


def select_winner(rows: list[dict[str, Any]]) -> None:
    """Mark exactly one winner per dataset — VALIDATION performance only (spec 12).

    Order: drop unavailable → require floor compliance (where configured) →
    rank by validation Macro F0.5 → validation AUPRC → simpler model on ties.
    Test metrics are never consulted.
    """
    by_ds: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for dataset_rows in by_ds.values():
        for r in dataset_rows:
            r["winner"] = False
        eligible = [r for r in dataset_rows
                    if r["status"] != "UNAVAILABLE"
                    and r["_val_macro_f0_5"] is not None
                    and (r["sensitivity_floor_pass"] is not False)]
        if not eligible:
            # fall back: floor-infeasible → max validation sensitivity (same as threshold policy)
            eligible = [r for r in dataset_rows
                        if r["status"] != "UNAVAILABLE" and r["_val_sens"] is not None]
            eligible.sort(key=lambda r: (
                -(round(r["_val_sens"], 6) if r["_val_sens"] is not None else -1.0),
                -(round(r["_val_macro_f0_5"], 6)
                  if r["_val_macro_f0_5"] is not None else -1.0)))
        else:
            eligible.sort(key=lambda r: (
                -(round(r["_val_macro_f0_5"], 6) if r["_val_macro_f0_5"] is not None else -1.0),
                -(round(r["_val_auprc"], 6) if r["_val_auprc"] is not None else -1.0),
                _SIMPLICITY.index(r["algorithm"]) if r["algorithm"] in _SIMPLICITY
                else len(_SIMPLICITY),
                r["algorithm"]))
        if eligible:
            eligible[0]["winner"] = True


def write_scorecard() -> dict[str, Any]:
    rows = build_scorecard_rows()
    select_winner(rows)
    public = [{k: r.get(k) for k in SCORECARD_FIELDS} for r in rows]
    payload = {
        "generated_from": "outputs/phase5/*/metrics/*.json (persisted only)",
        "winner_selection": ("validation-only: floor compliance → validation Macro F0.5 → "
                             "validation AUPRC → simpler model on ties; test never used"),
        "field_notes": {
            "precision": "positive-class PPV",
            "F1": "macro-averaged F1",
            "F2": "macro-averaged F2",
            "Macro_F0.5": "unweighted mean of per-class F0.5 (NOT accuracy)",
            "unavailable_metrics": "null / n/a — never 0",
        },
        "sensitivity_floor_default": 0.90,
        "n_rows": len(public),
        "rows": public,
        "disclaimer": "Research prototype — not a medical device.",
    }
    PHASE5_OUT.mkdir(parents=True, exist_ok=True)
    (PHASE5_OUT / "model_scorecard.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    with (PHASE5_OUT / "model_scorecard.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SCORECARD_FIELDS)
        w.writeheader()
        for r in public:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in SCORECARD_FIELDS})
    return payload


def write_cross_dataset_summary() -> dict[str, Any]:
    """Per-dataset winner + generalization gap (validation → sealed test), persisted only."""
    rows = build_scorecard_rows()
    select_winner(rows)
    per_ds: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = per_ds.setdefault(r["dataset"], {
            "dataset": r["dataset"],
            "n_algorithms": 0, "n_evaluated": 0, "winner": None,
            "winner_val_macro_f0_5": None, "winner_test_macro_f0_5": None,
            "generalization_gap_val_minus_test": None,
            "blocker_status": None, "executed": True,
        })
        d["n_algorithms"] += 1
        if r["status"] == "EVALUATED":
            d["n_evaluated"] += 1
        if r["winner"]:
            d["winner"] = r["algorithm"]
            d["winner_val_macro_f0_5"] = r["_val_macro_f0_5"]
            d["winner_test_macro_f0_5"] = r["Macro_F0.5"]
            if r["_val_macro_f0_5"] is not None and r["Macro_F0.5"] is not None:
                d["generalization_gap_val_minus_test"] = round(
                    r["_val_macro_f0_5"] - r["Macro_F0.5"], 6)
        if r.get("note") and r["status"] == "UNAVAILABLE":
            d.setdefault("unavailable", {})[r["algorithm"]] = r["note"]

    # attach blocker statuses from the blocking report
    rep = PHASE5_OUT / "dataset_blocking_report.json"
    if rep.exists():
        try:
            info = json.loads(rep.read_text(encoding="utf-8"))
            dsinfo = info.get("datasets")
            it = dsinfo.items() if isinstance(dsinfo, dict) else (
                (d.get("dataset_id"), d) for d in dsinfo)
            for did, d in it:
                if did in per_ds:
                    per_ds[did]["blocker_status"] = d.get("status")
        except Exception:
            pass

    payload = {
        "generated_from": "outputs/phase5/*/metrics/*.json",
        "note": ("Each dataset is a separate experiment — never concatenated. "
                 "Regensburg ultrasound shares patients with Regensburg tabular and is NOT "
                 "an external population. Gaps are validation → sealed-test Macro F0.5."),
        "datasets": list(per_ds.values()),
        "external_cohort_validation": "see outputs/phase5/external_validation.json",
        "disclaimer": "Research prototype — not a medical device.",
    }
    (PHASE5_OUT / "cross_dataset_summary.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    fields = ["dataset", "n_algorithms", "n_evaluated", "winner",
              "winner_val_macro_f0_5", "winner_test_macro_f0_5",
              "generalization_gap_val_minus_test", "blocker_status"]
    with (PHASE5_OUT / "cross_dataset_summary.csv").open("w", newline="",
                                                          encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for d in per_ds.values():
            w.writerow({k: ("" if d.get(k) is None else d.get(k)) for k in fields})
    return payload


def write_external_validation_status() -> dict[str, Any]:
    """Honest status of EXTERNAL cohort validation — never fabricate one."""
    payload = {
        "status": "NOT_EXECUTED",
        "definition": ("An independent cohort for the SAME task, different population, "
                       "evaluated with a locked model. Mirrors are not cohorts; different "
                       "diseases are different tasks, not external validation."),
        "candidates": [
            {"candidate": "Kermany pneumonia as external cohort for appendicitis",
             "status": "NOT_APPLICABLE",
             "reason": "different disease/task — not an external cohort for appendicitis"},
            {"candidate": "Neonatal sepsis registry as external cohort for appendicitis",
             "status": "NOT_APPLICABLE",
             "reason": "different disease/task — executed as its own separate experiment"},
            {"candidate": "Regensburg ultrasound images",
             "status": "NOT_EXTERNAL",
             "reason": "shares patients with the Regensburg tabular cohort (internal extension)"},
            {"candidate": "Kaggle mirror of Kermany",
             "status": "NOT_A_COHORT",
             "reason": "mirror of the Mendeley dataset — one underlying dataset"},
            {"candidate": "PhysioNet PIC / PECARN pediatric cohorts",
             "status": "NOT_AVAILABLE",
             "reason": "credentialed access unavailable in this environment"},
        ],
        "disclaimer": ("No dataset is claimed externally validated. Research prototype — "
                       "not a medical device."),
    }
    (PHASE5_OUT / "external_validation.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    return payload


def write_all() -> dict[str, Any]:
    sc = write_scorecard()
    cd = write_cross_dataset_summary()
    ev = write_external_validation_status()
    return {"scorecard_rows": sc["n_rows"],
            "datasets": len(cd["datasets"]),
            "external_validation": ev["status"]}
