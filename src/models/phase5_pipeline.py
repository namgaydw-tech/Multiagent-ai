"""Phase 5 tabular pipeline (sections G–J, W steps 8–13).

Stage contract (each stage persists before the next may run):

``train_dataset``     gate (blocker) → prepare partitions → zoo training with
                      CV on the TRAIN partition only → calibration candidates
                      compared on VALIDATION → threshold locked on VALIDATION
                      (Macro F0.5 s.t. sensitivity floor) → ensemble weights +
                      stacking meta-learner fitted WITHOUT any test prediction.
``evaluate_dataset``  loads the locked artifacts and evaluates the SEALED TEST
                      partition exactly once: full metrics + bootstrap CIs +
                      persisted per-case predictions.

Guarantees enforced here (and asserted by ``tests/test_phase5.py``):

* BLOCKED / NOT_AVAILABLE datasets raise :class:`DatasetBlockedError` before
  any data is touched (the blocker gate);
* the Regensburg reference dataset reuses the EXISTING persisted Phase 2 split
  — never re-split, never re-tuned against the test set;
* test labels are never passed to calibration, threshold or ensemble-weight
  functions (they simply have no parameter for them);
* every artifact lands under ``outputs/phase5/<dataset>/`` only — Phase 2-4
  directories are never written to.

Research prototype — not a medical device.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.calibration import phase5 as P5CAL
from src.data.dataset_blocker import DatasetBlockedError, assert_trainable
from src.data.phase5_datasets import load_sepsis, make_group_split, phase5_out
from src.evaluation.phase5_metrics import bootstrap_ci, compute_full_metrics
from src.evaluation.threshold_policy import select_threshold_macro_f05
from src.models.model_registry import load_zoo, rebuild_predict, train_model
from src.preprocessing.regensburg import ROOT

SEED = 20261002


@dataclass
class Prepared:
    """One dataset's modelling partitions (never merged with another dataset)."""

    dataset_id: str
    out_dir: Path
    X: pd.DataFrame
    y: pd.Series
    splits: dict[str, list[int]]
    numeric: list[str]
    categorical: list[str]
    multiclass: bool
    target_names: dict[int, str]
    sensitivity_floor: float
    split_source: str
    groups: pd.Series | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _phase5_cfg() -> dict:
    from src.data.phase5_datasets import load_phase5_config
    return load_phase5_config()


def _split_counts(prepared: Prepared) -> dict[str, dict[str, int]]:
    out = {}
    for part, rows in prepared.splits.items():
        y = prepared.y.loc[rows]
        vc = y.value_counts().to_dict()
        out[part] = {"n": len(rows),
                     **{f"class_{int(k)}": int(v) for k, v in sorted(vc.items())}}
    return out


# --------------------------------------------------------------------- prepare
def prepare(dataset_id: str) -> Prepared:
    """Build partitions/features for one dataset (tabular datasets only here)."""
    cfg = _phase5_cfg()
    ds_cfg = cfg["datasets"][dataset_id]
    out_dir = phase5_out(ds_cfg["dir"])
    floor = float(ds_cfg.get("sensitivity_floor",
                             cfg["threshold_policy"]["sensitivity_floor"]))
    multiclass = bool(ds_cfg.get("multiclass") and dataset_id == "KERMANY_PEDIATRIC_PNEUMONIA")

    if dataset_id == "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR":
        from src.data.split import load_splits
        from src.preprocessing.regensburg import (
            build_feature_spec, load_audited_dataset, load_config, prepare_matrices,
            train_feature_arrays,
        )
        config = load_config()
        df = load_audited_dataset()
        spec = build_feature_spec(df, config)
        X, y = prepare_matrices(df, spec)
        splits = {p: [int(r) for r in rows] for p, rows in load_splits(
            ROOT / "data/interim/splits").items()}
        # reproduce the Phase 2 training-only categorical vocabulary
        Xtr_raw = X.loc[splits["train"]]
        Xtr, state = train_feature_arrays(Xtr_raw, spec, fit=True)
        X = X.copy()
        X.loc[splits["train"]] = Xtr
        for part in ("validation", "test"):
            X.loc[splits[part]] = state.apply(X.loc[splits[part]])
        numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        return Prepared(
            dataset_id=dataset_id, out_dir=out_dir, X=X, y=y, splits=splits,
            numeric=numeric, categorical=list(spec.categorical), multiclass=False,
            target_names={0: "no_appendicitis", 1: "appendicitis"},
            sensitivity_floor=floor, split_source="phase2_persisted_existing_split",
            groups=None,
            meta={"resplit_allowed": False,
                  "note": "existing Phase 2 split reused verbatim; test never re-tuned"})

    if dataset_id == "NEONATAL_SEPSIS_REGISTRY":
        ds = load_sepsis()
        if not ds.available:
            raise DatasetBlockedError(
                f"{dataset_id} loader returned no data — cannot train "
                f"({ds.meta.get('unavailable', 'unknown reason')})")
        split_path = out_dir / "splits" / "split.json"
        if split_path.exists():
            splits = json.loads(split_path.read_text(encoding="utf-8"))
            splits = {k: [int(i) for i in v] for k, v in splits.items()}
        else:
            splits = make_group_split(ds.y, ds.groups, seed=SEED)
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text(json.dumps(splits, indent=1), encoding="utf-8")
            meta = {"strategy": "stratified_group_shuffle (patients never cross partitions)",
                    "seed": SEED, "group_column": "unique_patient_id",
                    "n_patients": int(ds.meta["n_patients"]),
                    "counts": {p: len(v) for p, v in splits.items()},
                    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            (out_dir / "splits" / "split_metadata.json").write_text(
                json.dumps(meta, indent=1), encoding="utf-8")
        numeric = [c for c in ds.X.columns if pd.api.types.is_numeric_dtype(ds.X[c])]
        categorical = [c for c in ds.X.columns if c not in numeric]
        return Prepared(
            dataset_id=dataset_id, out_dir=out_dir, X=ds.X, y=ds.y, splits=splits,
            numeric=numeric, categorical=categorical, multiclass=False,
            target_names=ds.target_names, sensitivity_floor=floor,
            split_source="phase5_group_split_persisted", groups=ds.groups,
            meta=ds.meta)

    raise NotImplementedError(
        f"{dataset_id} is not a tabular Phase 5 dataset — use the image pipeline "
        "(src/models/image_pipeline.py) or the dataset is NOT_AVAILABLE")


# ------------------------------------------------------------------------ train
def train_dataset(dataset_id: str, *, models: list[str] | None = None,
                  seed: int = SEED, n_iter_scale: float = 1.0,
                  allow_conditional: bool = True,
                  with_ensembles: bool = True) -> dict[str, Any]:
    """Train + calibrate + lock thresholds for one dataset (test stays sealed)."""
    t0 = time.time()
    gate = assert_trainable(dataset_id, allow_conditional=allow_conditional)
    prepared = prepare(dataset_id)
    zoo = load_zoo()
    wanted = models or [m for m in zoo["tabular"] if zoo["tabular"][m].get("enabled", True)]

    tr, va = prepared.splits["train"], prepared.splits["validation"]
    X_train, y_train = prepared.X.loc[tr], prepared.y.loc[tr]
    X_val, y_val = prepared.X.loc[va], prepared.y.loc[va]
    train_groups = prepared.groups.loc[tr] if prepared.groups is not None else None

    summary: dict[str, Any] = {
        "dataset_id": dataset_id, "seed": seed, "stage": "train",
        "blocker_status": gate.status, "blocker_reasons": gate.reasons,
        "split_source": prepared.split_source,
        "counts": _split_counts(prepared),
        "multiclass": prepared.multiclass,
        "target_names": prepared.target_names,
        "sensitivity_floor": prepared.sensitivity_floor,
        "models": {}, "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_partition": "SEALED (never read in this stage)",
        "disclaimer": "Research prototype — not a medical device.",
    }

    trained: dict[str, dict] = {}
    raw_val_probs: dict[str, np.ndarray] = {}
    val_probs: dict[str, np.ndarray] = {}
    calibrators: dict[str, dict] = {}
    calibrator_objs: dict[str, Any] = {}
    thresholds: dict[str, dict] = {}
    val_metrics: dict[str, dict] = {}

    for name in wanted:
        if prepared.multiclass and name in ("svm_rbf", "mlp"):
            # still allowed; SVC/MLP handle multiclass natively
            pass
        res = train_model(name, X_train, y_train, X_val, y_val,
                          numeric=prepared.numeric, categorical=prepared.categorical,
                          multiclass=prepared.multiclass,
                          groups=np.asarray(train_groups) if train_groups is not None else None,
                          seed=seed, n_iter_scale=n_iter_scale, zoo=zoo)
        if not res.get("available"):
            summary["models"][name] = {"available": False, "note": res.get("note")}
            continue

        p_val_raw = np.asarray(res["predict"](X_val))
        if not prepared.multiclass:
            comparison = P5CAL.compare_binary(y_val.to_numpy(), p_val_raw)
            sel = P5CAL.select_binary(comparison)
            cal = P5CAL.fit_binary(sel, y_val.to_numpy(), p_val_raw)
            p_val_cal = p_val_raw if cal is None else P5CAL.apply_calibrator(sel, cal, p_val_raw)
            thr = select_threshold_macro_f05(y_val.to_numpy(), p_val_cal,
                                             sensitivity_floor=prepared.sensitivity_floor)
            vm = compute_full_metrics(y_val.to_numpy(), p_val_cal,
                                      threshold=thr["threshold"])
            thresholds[name] = thr
            calibrators[name] = {"selected": sel, "comparison": comparison,
                                 "fit_on": "validation"}
            calibrator_objs[name] = cal
        else:
            comparison = P5CAL.compare_multiclass(y_val.to_numpy(), p_val_raw)
            sel = P5CAL.select_multiclass(comparison)
            cal = (None if sel == "raw"
                   else P5CAL.OVRSigmoidCalibrator().fit(p_val_raw, y_val.to_numpy()))
            p_val_cal = p_val_raw if cal is None else cal.predict(p_val_raw)
            thr = {"threshold": None, "rule": "argmax (multiclass)",
                   "fitted_on": "validation",
                   "sensitivity_floor": None, "floor_satisfied": None}
            thresholds[name] = thr
            calibrators[name] = {"selected": sel, "comparison": comparison,
                                 "fit_on": "validation"}
            calibrator_objs[name] = cal
            vm = compute_full_metrics(y_val.to_numpy(), p_val_cal, multiclass=True,
                                      target_names=prepared.target_names)

        trained[name] = res
        raw_val_probs[name] = p_val_raw
        val_probs[name] = p_val_cal
        val_metrics[name] = vm
        summary["models"][name] = {
            "available": True, "cv_score": res["cv_score"],
            "cv_scoring": res["cv_scoring"], "best_params": res["best_params"],
            "fit_seconds": res["fit_seconds"],
            "calibrator": calibrators[name]["selected"],
            "val_macro_f0_5": vm["macro_f0_5"],
            "val_auroc": vm.get("auroc"), "val_auprc": vm.get("auprc"),
            "threshold": (thr.get("threshold") if not prepared.multiclass else None),
            "floor_satisfied": thr.get("floor_satisfied"),
        }

    # ---------------------------------------------------- ensembles (validation only)
    ensemble_info: dict[str, Any] = {"fitted_on": "validation", "test_used": False}
    if with_ensembles and trained:
        ensemble_info.update(
            _fit_ensembles(prepared, trained, val_probs, val_metrics, thresholds))
        for ens_name, ens_thr in (ensemble_info.get("thresholds") or {}).items():
            thresholds[ens_name] = ens_thr

    # ---------------------------------------------------------------- persistence
    models_dir = prepared.out_dir / "models"
    cal_dir = prepared.out_dir / "calibration"
    met_dir = prepared.out_dir / "metrics"
    for res in trained.values():
        blob = {k: v for k, v in res.items() if k not in ("predict", "preprocess")}
        joblib.dump(blob, models_dir / f"{res['model_name']}.joblib")
    for name, cinfo in calibrators.items():
        cal = calibrator_objs.get(name)
        if cal is not None:
            joblib.dump({"calibrator": cal, "name": cinfo["selected"]},
                        cal_dir / f"{name}_calibrator.joblib")
    (cal_dir / "calibration.json").write_text(
        json.dumps(calibrators, indent=1, default=str), encoding="utf-8")
    (cal_dir / "thresholds.json").write_text(
        json.dumps({"fitted_on": "validation", "locked_before_test": True,
                    "sensitivity_floor": prepared.sensitivity_floor,
                    "thresholds": thresholds}, indent=1, default=str), encoding="utf-8")
    (met_dir / "metrics_val.json").write_text(
        json.dumps({"partition": "validation", "metrics": val_metrics,
                    "note": "validation metrics at locked thresholds — used for model "
                            "selection; test is evaluated separately"},
                   indent=1, default=str), encoding="utf-8")
    summary["ensemble"] = ensemble_info
    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (met_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


# ------------------------------------------------------------------ ensembles
def _fit_ensembles(prepared: Prepared, trained: dict[str, dict],
                   val_probs: dict[str, np.ndarray], val_metrics: dict[str, dict],
                   thresholds: dict[str, dict]) -> dict[str, Any]:
    """Validation-weighted soft voting + stacking on held-out validation (section J).

    Weight rule: among models whose locked threshold satisfies the sensitivity
    floor, weights are proportional to validation Macro F0.5; if no model
    satisfies the floor, weights fall back to all models with an explicit note.
    """
    names = list(trained)
    info: dict[str, Any] = {"fitted_on": "validation", "test_used": False,
                           "models": names}
    floor_ok = [n for n in names
                if thresholds.get(n, {}).get("floor_satisfied") is True]
    weighted_names = floor_ok or names
    ens_thresholds: dict[str, dict] = {}
    info["floor_satisfied_models"] = floor_ok
    info["weight_note"] = (
        "weights ∝ validation Macro F0.5 among floor-satisfied models"
        if floor_ok else
        "no model satisfied the sensitivity floor on validation — weights over ALL "
        "models (recorded, not hidden)")
    raw_scores = {n: max(float(val_metrics[n]["macro_f0_5"]), 1e-9)
                  for n in weighted_names}
    total = sum(raw_scores.values())
    weights = {n: round(raw_scores[n] / total, 6) for n in weighted_names}
    info["method"] = "validation_weighted_soft_voting"
    info["weights"] = weights

    y_val = _val_labels(prepared).to_numpy()
    if not prepared.multiclass:
        p_val_ens = sum(weights[n] * val_probs[n] for n in weights)
        ens_thr = select_threshold_macro_f05(
            y_val, np.asarray(p_val_ens),
            sensitivity_floor=prepared.sensitivity_floor)
        ens_thresholds["ensemble_soft_voting"] = ens_thr
        info["threshold"] = ens_thr["threshold"]
        # uniform soft voting over ALL trained models (no weights learned at all)
        p_val_uniform = np.mean([val_probs[n] for n in names], axis=0)
        ens_thresholds["ensemble_soft_uniform"] = select_threshold_macro_f05(
            y_val, np.asarray(p_val_uniform),
            sensitivity_floor=prepared.sensitivity_floor)
        info["uniform_weights"] = {n: round(1.0 / len(names), 6) for n in names}

    # Stacking meta-learner fitted on held-out validation predictions
    # (out-of-fold relative to base-model fitting; the test set is never used).
    try:
        from sklearn.linear_model import LogisticRegression
        names_list = list(trained)
        if prepared.multiclass:
            K = len(prepared.target_names)
            M_tr = np.hstack([val_probs[n] for n in names_list])
        else:
            M_tr = np.column_stack([val_probs[n] for n in names_list])
        y_meta = np.asarray([y_val for y_val in _val_labels(prepared)], dtype=int)
        meta = LogisticRegression(max_iter=2000, random_state=SEED)
        meta.fit(M_tr, y_meta)
        if not prepared.multiclass:
            p_meta_val = meta.predict_proba(M_tr)[:, 1]
            ens_thresholds["ensemble_stacking"] = select_threshold_macro_f05(
                y_meta, p_meta_val, sensitivity_floor=prepared.sensitivity_floor)
        joblib.dump({"meta": meta, "models": names_list,
                     "fit_on": "out-of-fold validation predictions (held-out from base fitting)",
                     "multiclass": prepared.multiclass,
                     "n_fit": int(len(y_meta))},
                    prepared.out_dir / "models" / "stacking_meta.joblib")
        info["stacking"] = {"method": "stacking",
                            "meta_learner": "logistic_regression",
                            "fit_on": "out-of-fold validation predictions",
                            "n_fit": int(len(y_meta)),
                            "test_used": False}
    except Exception as exc:
        info["stacking"] = {"status": "not_fitted",
                            "reason": f"{type(exc).__name__}: {exc}"}
    info["thresholds"] = ens_thresholds
    return info


def _val_labels(prepared: Prepared):
    return prepared.y.loc[prepared.splits["validation"]]


# ------------------------------------------------------------------- evaluate
def evaluate_dataset(dataset_id: str, *, n_boot: int = 2000,
                     seed: int = SEED) -> dict[str, Any]:
    """Evaluate the SEALED TEST partition once, with locked thresholds (section W.12).

    Loads persisted models/calibrators/thresholds — nothing is re-fit here. Test
    labels are used ONLY for final metric computation (never for any fitting),
    and the first evaluation is recorded in ``metrics/test_evaluation_lock.json``.
    """
    t0 = time.time()
    gate = assert_trainable(dataset_id)
    prepared = prepare(dataset_id)
    te = prepared.splits["test"]
    X_test, y_test = prepared.X.loc[te], prepared.y.loc[te]

    thr_blob = json.loads((prepared.out_dir / "calibration" / "thresholds.json")
                          .read_text(encoding="utf-8"))
    thresholds = thr_blob["thresholds"]

    results: dict[str, dict] = {}
    test_probs: dict[str, np.ndarray] = {}
    models_dir = prepared.out_dir / "models"
    for path in sorted(models_dir.glob("*.joblib")):
        if path.name == "stacking_meta.joblib":
            continue
        blob = joblib.load(path)
        name = blob.get("model_name", path.stem)
        try:
            predict = rebuild_predict(blob)
        except Exception as exc:
            results[name] = {"available": False,
                             "note": f"predict rebuild failed: {exc}"}
            continue
        t_inf = time.perf_counter()
        p_raw = np.asarray(predict(X_test))
        infer_s = time.perf_counter() - t_inf
        cal_path = prepared.out_dir / "calibration" / f"{name}_calibrator.joblib"
        if cal_path.exists():
            cblob = joblib.load(cal_path)
            p_cal = P5CAL.apply_calibrator(cblob["name"], cblob["calibrator"], p_raw)
            cal_name = cblob["name"]
        else:
            p_cal, cal_name = p_raw, "raw"
        test_probs[name] = p_cal
        thr = thresholds.get(name, {})
        if prepared.multiclass:
            metrics = compute_full_metrics(y_test.to_numpy(), p_cal, multiclass=True,
                                           target_names=prepared.target_names)
        else:
            metrics = compute_full_metrics(y_test.to_numpy(), p_cal,
                                           threshold=float(thr["threshold"]))
        ci = bootstrap_ci(y_test.to_numpy(), p_cal,
                          threshold=(float(thr["threshold"])
                                     if not prepared.multiclass else 0.5),
                          n_boot=n_boot, seed=seed, multiclass=prepared.multiclass)
        results[name] = {"available": True, "calibrator": cal_name,
                         "threshold": thr.get("threshold"),
                         "floor_satisfied": thr.get("floor_satisfied"),
                         "training_seconds": blob.get("fit_seconds"),
                         "inference_seconds": round(infer_s, 5),
                         "inference_ms_per_case": round(1000 * infer_s / max(len(te), 1), 4),
                         "test_metrics": metrics, "bootstrap_ci": ci}
        # per-case predictions (persisted, auditable)
        rows = {"row_index": [int(r) for r in te],
                "y_true": [int(v) for v in y_test.to_numpy()]}
        if not prepared.multiclass:
            rows["p_raw"] = np.round(p_raw, 6)
            rows["p_calibrated"] = np.round(p_cal, 6)
            rows["threshold"] = float(thr["threshold"])
            rows["pred"] = (p_cal >= float(thr["threshold"])).astype(int)
        else:
            rows["pred"] = p_cal.argmax(axis=1)
            for k in range(p_cal.shape[1]):
                rows[f"p_class_{k}"] = np.round(p_cal[:, k], 6)
        pd.DataFrame(rows).to_csv(
            prepared.out_dir / "predictions" / f"{name}_test.csv", index=False)

    # ------------------------------------------------------------ ensembles on test
    ens_summary: dict[str, Any] = {}
    train_summary_path = prepared.out_dir / "metrics" / "train_summary.json"
    if train_summary_path.exists() and test_probs:
        tsum = json.loads(train_summary_path.read_text(encoding="utf-8"))
        ens = tsum.get("ensemble") or {}
        weights = ens.get("weights") or {}
        common = [n for n in weights if n in test_probs]
        if common:
            w = np.array([weights[n] for n in common], dtype=float)
            w = w / w.sum()
            p_ens = sum(wi * test_probs[n] for wi, n in zip(w, common))
            thr_ens = thresholds.get("ensemble_soft_voting", {}).get("threshold", 0.5)
            metrics = (compute_full_metrics(y_test.to_numpy(), p_ens, multiclass=True,
                                            target_names=prepared.target_names)
                       if prepared.multiclass else
                       compute_full_metrics(y_test.to_numpy(), p_ens,
                                            threshold=float(thr_ens)))
            ens_summary["validation_weighted_soft_voting"] = {
                "available": True, "weights": {n: weights[n] for n in common},
                "threshold": (None if prepared.multiclass else thr_ens),
                "threshold_selected_on": "validation (same Macro F0.5 + floor policy)",
                "test_metrics": metrics}
            pd.DataFrame({"row_index": [int(r) for r in te],
                          "y_true": [int(v) for v in y_test.to_numpy()],
                          "p_ens": np.round(p_ens, 6) if not prepared.multiclass else None}
                         ).to_csv(prepared.out_dir / "predictions" / "ensemble_test.csv",
                                  index=False)

        # uniform soft voting (no learned weights) over ALL models
        all_common = [n for n in test_probs]
        if all_common and not prepared.multiclass:
            p_uni = np.mean([test_probs[n] for n in all_common], axis=0)
            thr_uni = thresholds.get("ensemble_soft_uniform", {}).get("threshold", 0.5)
            ens_summary["ensemble_soft_uniform"] = {
                "available": True, "test_metrics": compute_full_metrics(
                    y_test.to_numpy(), p_uni, threshold=float(thr_uni)),
                "threshold": thr_uni,
                "threshold_selected_on": "validation (Macro F0.5 + floor policy)"}
            # hard voting: majority of locked-threshold individual predictions
            hard_preds = np.column_stack([
                (test_probs[n] >= float(thresholds.get(n, {}).get("threshold", 0.5)))
                .astype(int) for n in all_common])
            majority = (hard_preds.sum(axis=1) * 2 > len(all_common)).astype(int)
            from src.evaluation.phase5_metrics import metrics_from_hard
            ens_summary["ensemble_hard_voting"] = {
                "available": True,
                "test_metrics": metrics_from_hard(y_test.to_numpy(), majority),
                "rule": "majority vote over locked-threshold individual predictions",
                "n_voters": len(all_common)}
        stack_path = models_dir / "stacking_meta.joblib"
        if stack_path.exists():
            sblob = joblib.load(stack_path)
            names_list = [n for n in sblob["models"] if n in test_probs]
            if len(names_list) == len(sblob["models"]):
                M_te = (np.hstack([test_probs[n] for n in names_list])
                        if prepared.multiclass else
                        np.column_stack([test_probs[n] for n in names_list]))
                p_meta = sblob["meta"].predict_proba(M_te)
                thr_stack = thresholds.get("ensemble_stacking", {}).get("threshold", 0.5)
                metrics = (compute_full_metrics(y_test.to_numpy(), p_meta, multiclass=True,
                                                target_names=prepared.target_names)
                           if prepared.multiclass else
                           compute_full_metrics(y_test.to_numpy(), p_meta[:, 1],
                                                threshold=float(thr_stack)))
                ens_summary["stacking"] = {"available": True, "test_metrics": metrics,
                                           "fit_on": sblob.get("fit_on"),
                                           "threshold": (None if prepared.multiclass
                                                         else thr_stack)}

    # ---------------------------------------------------------------- persistence
    met_dir = prepared.out_dir / "metrics"
    payload = {"dataset_id": dataset_id, "partition": "test",
               "split_source": prepared.split_source,
               "counts": {"test": {"n": int(len(te)),
                                    **{f"class_{int(k)}": int(v) for k, v in
                                       prepared.y.loc[te].value_counts().sort_index().items()}}},
               "blocker_status": gate.status,
               "models": results, "ensembles": ens_summary,
               "thresholds_locked_on": "validation",
               "n_boot": n_boot, "seed": seed,
               "evaluated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "disclaimer": "Research prototype — not a medical device; single-use "
                             "held-out estimates with bootstrap CIs."}
    (met_dir / "metrics_test.json").write_text(
        json.dumps(payload, indent=1, default=str), encoding="utf-8")

    lock_path = met_dir / "test_evaluation_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        lock["n_evaluations"] = int(lock.get("n_evaluations", 1)) + 1
        lock["last_evaluation_utc"] = payload["evaluated_utc"]
        lock["note"] = "re-evaluation for reproduction — the first evaluation is recorded"
    else:
        lock = {"first_evaluation_utc": payload["evaluated_utc"],
                "n_evaluations": 1,
                "note": "test partition evaluated once after threshold lock"}
    lock_path.write_text(json.dumps(lock, indent=1), encoding="utf-8")

    summary = {"dataset_id": dataset_id,
               "n_models": sum(1 for r in results.values() if r.get("available")),
               "elapsed_seconds": round(time.time() - t0, 1),
               "lock": lock, "blocker_status": gate.status}
    (met_dir / "evaluate_summary.json").write_text(
        json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary
