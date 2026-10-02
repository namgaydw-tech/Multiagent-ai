"""Phase 2 end-to-end driver: train, calibrate, evaluate, explain (research only).

Execution order (mirrors config/model.yaml and the pre-registered protocol):

    audit gate -> feature manifest -> patient-level split -> preprocess (fit on train)
    -> LightGBM tuning (train CV, validation monitoring) -> baselines
    -> calibration + threshold on validation -> LOCK -> single test evaluation
    -> bootstrap CIs -> FN review -> subgroups -> figures -> SHAP

Run::

    python scripts/run_phase2.py            # full phase
    python scripts/run_phase2.py --fast     # reduced search (CI/smoke)

Nothing is printed as a result unless it was actually computed in this run.
Research prototype — not a medical device.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import calibrate as calibration  # noqa: E402
from src.data.split import load_splits, make_splits, save_splits  # noqa: E402
from src.evaluation import model_metrics as M  # noqa: E402
from src.explainability.shap_engine import ShapEngine, persist_metadata  # noqa: E402
from src.features.regensburg_features import write_manifest  # noqa: E402
from src.models import baselines as baseline_models  # noqa: E402
from src.models.lightgbm_model import MODEL_NAME, MODEL_VERSION, load_lightgbm, train_lightgbm  # noqa: E402
from src.models.prediction_interface import build_prediction, apply_calibration  # noqa: E402
from src.models.uncertainty import estimate_uncertainty  # noqa: E402
from src.preprocessing.regensburg import (  # noqa: E402
    ROOT as PROOT, build_feature_spec, config_hash, load_audited_dataset, load_config,
    train_feature_arrays,
)

AUDIT_JSON = PROOT / "outputs/metrics/regensburg_audit.json"
SPLIT_DIR = PROOT / "data/interim/splits"
PRED_DIR = PROOT / "outputs/predictions"


def log(msg: str) -> None:
    print(f"[phase2] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true", help="reduced hyperparameter search")
    ap.add_argument("--skip-shap", action="store_true", help="skip SHAP figures (smoke runs)")
    args = ap.parse_args()

    t_start = time.time()
    if not AUDIT_JSON.exists():
        log("ABORT: dataset audit missing (docs/DATASET_AUDIT.md). Run: python -m src.data.audit_regensburg")
        return 2
    audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    if not audit.get("leakage_screen", {}).get("flagged_columns"):
        log("ABORT: audit incomplete")
        return 2
    log("audit gate OK (conditional PASS conditions will be enforced below)")

    # ---------------------------------------------------------------- 1. manifest + data
    manifest = write_manifest()
    log(f"feature manifest: {manifest['n_included_features']} features, "
        f"{manifest['n_missing_indicators']} indicators, {manifest['n_excluded_columns']} excluded")

    config = load_config()
    cfg_hash = config_hash(config)
    df = load_audited_dataset()
    spec = build_feature_spec(df, config)

    from src.preprocessing.regensburg import prepare_matrices

    X, y = prepare_matrices(df, spec)
    # numeric column set must include the derived missing indicators (built from X itself)
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    log(f"prepared matrices: X={X.shape} (numeric={len(numeric_cols)}, "
        f"categorical={len(spec.categorical)}), target balance={y.value_counts().to_dict()}")

    # ---------------------------------------------------------------- 2. split
    if (SPLIT_DIR / "test.json").exists() and not args.fast:
        splits_meta = json.loads((SPLIT_DIR / "split_metadata.json").read_text(encoding="utf-8"))
        idx_map = load_splits(SPLIT_DIR)
        log(f"reusing persisted split ({splits_meta['strategy']}) seed={splits_meta['seed']}")
    else:
        splits = make_splits(y, config)
        save_splits(splits, df, SPLIT_DIR)
        idx_map = {p: splits[p]["row_indices"] for p in ("train", "validation", "test")}
        log(f"split created: strategy={splits['strategy']} "
            f"train={splits['train']['n']} val={splits['validation']['n']} test={splits['test']['n']}")

    tr, va, te = idx_map["train"], idx_map["validation"], idx_map["test"]
    for part, rows in (("train", tr), ("validation", va), ("test", te)):
        counts = y.loc[rows].value_counts().to_dict()
        log(f"  {part}: n={len(rows)} pos={counts.get(1, 0)} neg={counts.get(0, 0)}")

    X_train, y_train = X.loc[tr], y.loc[tr]
    X_val, y_val = X.loc[va], y.loc[va]
    X_test, y_test = X.loc[te], y.loc[te]

    # fit categorical vocabulary on TRAIN ONLY
    X_train, cat_state = train_feature_arrays(X_train, spec, fit=True)
    X_val = cat_state.apply(X_val)
    X_test = cat_state.apply(X_test)

    # ---------------------------------------------------------------- 3. LightGBM
    log("training LightGBM (RandomizedSearchCV, AUPRC, train CV + validation monitor)...")
    lgb = train_lightgbm(X_train, y_train, X_val, y_val, config,
                         n_iter=(10 if args.fast else 40))
    log(f"  best CV AUPRC={lgb['config']['search']['cv_best_average_precision']:.4f} "
        f"params={lgb['config']['best_params']}")

    # ---------------------------------------------------------------- 4. baselines
    log("training baselines (LR, RF, HistGradientBoosting, CatBoost)...")
    base_results = baseline_models.train_baselines(
        X_train, y_train, X_val, y_val, numeric_cols, spec.categorical)
    for r in base_results:
        note = "" if r.get("available", True) else f" SKIPPED: {r.get('note')}"
        log(f"  {r['model_name']}: val AUPRC={r['val_metrics']['auprc']:.4f}{note}")

    # ---------------------------------------------------------------- 5. calibration + threshold (val only)
    log("calibrating on validation partition (Platt vs isotonic vs none)...")
    cal_meta = calibration.run_calibration(y_val.to_numpy(), lgb["val_proba"])
    calibrator = calibration.load_calibrator()
    cal_name = calibrator["name"]
    threshold = float(cal_meta["threshold"]["threshold"])
    log(f"  selected calibrator={cal_name} "
        f"(Brier {cal_meta['comparison'][cal_name]['brier']:.4f} vs uncal "
        f"{cal_meta['comparison']['uncalibrated']['brier']:.4f}); threshold={threshold}")

    # per-model validation-locked thresholds for the baselines
    base_thresholds: dict[str, float] = {}
    for r in base_results:
        if not r.get("available", True):
            continue
        info = calibration.select_threshold_f2(y_val.to_numpy(), r["val_proba"])
        base_thresholds[r["model_name"]] = float(info["threshold"])

    # ---------------------------------------------------------------- 6. TEST (single use)
    log("LOCKED: evaluating on the held-out test partition (single use)...")
    lgb_model = lgb["model"]
    test_raw = lgb_model.predict_proba(X_test)[:, 1]
    test_cal = np.asarray(calibrator["calibrator"].predict(test_raw))

    results: dict[str, dict] = {}
    model_probas: dict[str, np.ndarray] = {}

    lgb_test_metrics_raw = M.compute_metrics(y_test.to_numpy(), test_raw, threshold=threshold)
    lgb_test_metrics_cal = M.compute_metrics(y_test.to_numpy(), test_cal, threshold=threshold)
    lgb_ci = M.bootstrap_ci(y_test.to_numpy(), test_cal, threshold=threshold)
    results[MODEL_NAME] = {
        "test_metrics": lgb_test_metrics_cal,
        "test_metrics_raw_probabilities": lgb_test_metrics_raw,
        "bootstrap_ci": lgb_ci,
        "calibrator": cal_name,
        "threshold": threshold,
        "best_params": lgb["config"]["best_params"],
        "val_auprc": lgb["config"]["validation_metrics_at_0_5"]["auprc"],
    }
    model_probas[f"{MODEL_NAME} (calibrated)"] = test_cal
    model_probas[f"{MODEL_NAME} (raw)"] = test_raw

    for r in base_results:
        if not r.get("available", True):
            results[r["model_name"]] = {"available": False, "note": r.get("note")}
            continue
        # baselines may need their own test-time transform (ordinal/cat missing-fill)
        pre = r.get("preprocess")
        Xte_used = pre(X_test) if pre is not None else X_test
        proba = r["model"].predict_proba(Xte_used)[:, 1]
        thr = base_thresholds[r["model_name"]]
        results[r["model_name"]] = {
            "test_metrics": M.compute_metrics(y_test.to_numpy(), proba, threshold=thr),
            "bootstrap_ci": M.bootstrap_ci(y_test.to_numpy(), proba, threshold=thr,
                                           n_boot=(200 if args.fast else 2000)),
            "calibrator": "none",
            "threshold": thr,
            "best_params": r["best_params"],
            "val_auprc": r["val_metrics"]["auprc"],
        }
        model_probas[r["model_name"]] = proba

    # sensitivity analysis: label-circular features excluded (MODEL_CARD requirement)
    circular = [c for c in ("Alvarado_Score", "Paedriatic_Appendicitis_Score", "Appendix_Diameter")
                if c in X_test.columns]
    sens_note = {"circular_features_excluded": circular, "status": "not_run"}
    if circular:
        Xtr_c = X_train.drop(columns=circular)
        Xva_c = X_val.drop(columns=circular)
        Xte_c = X_test.drop(columns=circular)
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        from src.models.lightgbm_model import PARAM_DIST
        search = RandomizedSearchCV(
            LGBMClassifier(objective="binary", random_state=20261002, n_jobs=-1, verbose=-1),
            PARAM_DIST, n_iter=(6 if args.fast else 16), scoring="average_precision",
            random_state=20261002, cv=StratifiedKFold(5, shuffle=True, random_state=20261002),
            n_jobs=-1, error_score="raise")
        search.fit(Xtr_c, y_train)
        p_c = search.best_estimator_.predict_proba(Xte_c)[:, 1]
        sens_note = {
            "circular_features_excluded": circular,
            "test_metrics": M.compute_metrics(y_test.to_numpy(), p_c, threshold=threshold),
            "note": "Retrained without label-circular features; comparison is descriptive "
                    "(different feature set, same protocol).",
        }
        results["lightgbm_no_circular_features"] = {
            "test_metrics": sens_note["test_metrics"],
            "calibrator": "none (raw probabilities)",
            "threshold": threshold,
            "role": "label-circularity sensitivity analysis",
        }
        model_probas["lightgbm_no_circular (raw)"] = p_c
        log(f"  sensitivity (no {circular}): AUPRC={sens_note['test_metrics']['auprc']:.4f}")

    # ---------------------------------------------------------------- 7. FN review + subgroups
    fn_review = M.false_negative_review(y_test.to_numpy(), test_cal, threshold, te, df)
    subgroups = M.subgroup_report(y_test.to_numpy(), test_cal, threshold, df, te)
    log(f"false negatives: {fn_review['n_false_negatives']} "
        f"(FNR {fn_review['false_negative_rate']:.3f})")

    # ---------------------------------------------------------------- 8. figures
    FIG = PROOT / "outputs/figures"
    M.plot_roc(y_test.to_numpy(), model_probas, FIG / "roc_curve.png")
    M.plot_pr(y_test.to_numpy(), model_probas, FIG / "pr_curve.png")
    M.plot_confusion(y_test.to_numpy(), test_cal, threshold, FIG / "confusion_matrix.png",
                     title=f"LightGBM + {cal_name} (test)")
    M.plot_calibration(y_test.to_numpy(), test_raw, {"calibrated (test)": test_cal},
                       threshold, FIG / "calibration_curve.png")
    log("figures: roc_curve, pr_curve, confusion_matrix, calibration_curve written")

    # ---------------------------------------------------------------- 9. prediction interface + uncertainty
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    imp = [{"feature": f, "importance": float(v)} for f, v in
           zip(lgb["feature_order"],
               lgb_model.named_steps["clf"].feature_importances_
               if hasattr(lgb_model, "named_steps") else lgb_model.feature_importances_)][:15]
    rows = []
    for pos, row_idx in enumerate(te):
        unc = estimate_uncertainty(float(test_raw[pos]), threshold=threshold,
                                   calibrated_probability=float(test_cal[pos]))
        pred = build_prediction(
            model_name=MODEL_NAME, model_version=MODEL_VERSION,
            positive_probability=float(test_raw[pos]), important_features=imp,
            config_hash=cfg_hash, calibrated_probability=float(test_cal[pos]),
            uncertainty=unc, threshold=threshold)
        rows.append({
            "row_index": int(row_idx),
            "y_true": int(y_test.iloc[pos]),
            "p_raw": round(float(test_raw[pos]), 6),
            "p_calibrated": round(float(test_cal[pos]), 6),
            "threshold": threshold,
            "pred_class": pred.predicted_class,
            "uncertainty_level": unc["uncertainty_level"],
            "entropy": unc["predictive_entropy"],
            "margin": unc["margin"],
            "model_json": pred.model_dump_json(),
        })
    pd.DataFrame(rows).to_csv(PRED_DIR / "test_predictions.csv", index=False)
    sample = build_prediction(model_name=MODEL_NAME, model_version=MODEL_VERSION,
                              positive_probability=float(test_raw[0]), important_features=imp,
                              config_hash=cfg_hash, calibrated_probability=float(test_cal[0]),
                              uncertainty=estimate_uncertainty(float(test_raw[0]),
                                                               threshold=threshold,
                                                               calibrated_probability=float(test_cal[0])),
                              threshold=threshold)
    (PRED_DIR / "test_prediction_sample.json").write_text(sample.model_dump_json(indent=2),
                                                          encoding="utf-8")
    log("prediction interface: normalized JSON written (sample validated by pydantic)")

    # ---------------------------------------------------------------- 10. SHAP
    shap_status = {"status": "skipped", "reason": "--skip-shap"}
    if not args.skip_shap:
        engine = ShapEngine(lgb_model, list(lgb["feature_order"]))
        engine.compute(X_test)
        engine.plot_beeswarm()
        engine.plot_waterfall(0, case_id=f"test_row_{te[0]}")
        numeric_feats = [c for c in X_test.columns
                         if pd.api.types.is_numeric_dtype(X_test[c])]
        top2 = [f["feature"] for f in engine.global_importance()
                if f["feature"] in numeric_feats][:2]
        dep_paths = []
        for feat in top2:
            p = engine.plot_dependence(feat, FIG / f"shap_dependence_{feat}.png")
            if p:
                dep_paths.append(str(p.relative_to(PROOT)))
            else:
                log(f"  dependence plot skipped for {feat}")
        persist_metadata(engine, n_rows=len(X_test), split_name="test")
        explain = engine.explain_patient(case_id=f"test_row_{te[0]}", position=0)
        (PROOT / "outputs/metadata/shap_explain_patient_test_row0.json").write_text(
            json.dumps(explain, indent=2), encoding="utf-8")
        shap_status = {"status": "ok", "n_rows": int(len(X_test)),
                       "dependence_plots": dep_paths,
                       "global_top": engine.global_importance()[:10]}
        log("SHAP: beeswarm + waterfall + dependence written")

    # ---------------------------------------------------------------- 11. comparison outputs
    jpath, cpath = M.save_model_comparison(results)
    extra = {
        "false_negative_review": fn_review,
        "subgroups": subgroups,
        "label_circularity_sensitivity": sens_note,
        "split_strategy": json.loads((SPLIT_DIR / "split_metadata.json").read_text(encoding="utf-8"))["strategy"],
        "split_counts": {p: {"n": len(rows_i),
                             "n_positive": int(y.loc[rows_i].sum()),
                             "n_negative": int(len(rows_i) - y.loc[rows_i].sum())}
                         for p, rows_i in (("train", tr), ("validation", va), ("test", te))},
        "shap": shap_status,
        "manifest_path": "outputs/metadata/regensburg_feature_manifest.json",
        "config_hash": cfg_hash,
        "elapsed_seconds": round(time.time() - t_start, 1),
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": "Research prototype — not a medical device; test metrics are single-use "
                      "held-out estimates with bootstrap CIs.",
    }
    payload = json.loads(jpath.read_text(encoding="utf-8"))
    payload.update(extra)
    jpath.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    log(f"comparison written: {jpath.name}, {cpath.name}")
    log("PRIMARY MODEL TEST RESULTS:")
    for k in ("auroc", "auprc", "sensitivity", "specificity", "ppv", "npv", "f1", "f2",
              "balanced_accuracy", "brier", "ece"):
        v = lgb_test_metrics_cal[k]
        ci = lgb_ci.get(k, {})
        lo, hi = ci.get("lo"), ci.get("hi")
        ci_str = f"  95% CI [{lo:.3f}, {hi:.3f}]" if isinstance(lo, float) else ""
        log(f"  {k:18s} {v:.4f}{ci_str}")
    log(f"total elapsed: {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
