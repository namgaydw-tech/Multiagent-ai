"""LightGBM model training for pediatric appendicitis (Phase 2, section 2.3).

* Binary classifier for ``Diagnosis`` (appendicitis vs no appendicitis).
* Hyperparameter search with ``RandomizedSearchCV`` (5-fold stratified) on the
  **training partition only**, optimising ``average_precision`` (AUPRC); the
  validation partition monitors AUROC/F2/sensitivity/specificity/Brier.
* Test partition is never touched here.
* Deterministic seed (20261002); artifacts persisted under ``outputs/models/``.

Artifacts: ``lightgbm_appendicitis.joblib``, ``lightgbm_config.json``,
``lightgbm_feature_order.json``.

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
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.preprocessing.regensburg import ROOT, config_hash

MODEL_DIR = ROOT / "outputs/models"
MODEL_NAME = "lightgbm_appendicitis"
MODEL_VERSION = "1.0.0"

PARAM_DIST: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300, 400, 600, 800],
    "learning_rate": [0.01, 0.02, 0.05, 0.1],
    "num_leaves": [15, 31, 50, 70],
    "max_depth": [-1, 4, 6, 8],
    "min_child_samples": [10, 20, 40],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_lambda": [0.0, 0.5, 1.0, 5.0],
    "min_split_gain": [0.0, 0.01, 0.05],
}
N_ITER = 40
SEED = 20261002


def make_estimator(seed: int = SEED) -> LGBMClassifier:
    """Deterministic LightGBM classifier for binary targets."""
    return LGBMClassifier(
        objective="binary",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        importance_type="gain",
    )


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    n_iter: int = N_ITER,
    seed: int = SEED,
) -> dict:
    """Tune on train CV (AUPRC), monitor on validation, persist artifacts.

    Returns a dict with the fitted search object, best params, val metrics, and paths.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    search = RandomizedSearchCV(
        estimator=make_estimator(seed),
        param_distributions=PARAM_DIST,
        n_iter=int(n_iter),
        scoring="average_precision",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    best: LGBMClassifier = search.best_estimator_

    # Validation monitoring (never used for fitting).
    val_proba = best.predict_proba(X_val)[:, 1]
    from src.evaluation.model_metrics import compute_metrics  # local import avoids cycle

    val_metrics = compute_metrics(y_val.to_numpy(), val_proba, threshold=0.5)

    # Calibration-friendly attributes for later stages.
    feature_order = list(X_train.columns)
    model_path = MODEL_DIR / f"{MODEL_NAME}.joblib"
    joblib.dump(
        {
            "model": best,
            "feature_order": feature_order,
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        model_path,
    )

    config_record = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else
                            float(v) if isinstance(v, (np.floating,)) else v)
                        for k, v in search.best_params_.items()},
        "search": {
            "method": "RandomizedSearchCV",
            "n_iter": int(n_iter),
            "folds": 5,
            "scoring": "average_precision",
            "seed": seed,
            "cv_best_average_precision": float(search.best_score_),
        },
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "validation_metrics_at_0_5": val_metrics,
        "fit_seconds": round(time.time() - t0, 2),
        "config_hash": config_hash(config),
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
    }
    (MODEL_DIR / f"{MODEL_NAME}_config.json").write_text(
        json.dumps(config_record, indent=2, default=str), encoding="utf-8")
    (MODEL_DIR / f"{MODEL_NAME}_feature_order.json").write_text(
        json.dumps({"feature_order": feature_order}, indent=2), encoding="utf-8")

    return {
        "search": search,
        "model": best,
        "val_proba": val_proba,
        "config": config_record,
        "model_path": model_path,
        "feature_order": feature_order,
    }


def load_lightgbm() -> dict:
    """Reload the persisted LightGBM artifact (verifies reload works)."""
    path = MODEL_DIR / f"{MODEL_NAME}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact missing: {path}")
    payload = joblib.load(path)
    payload["config"] = json.loads(
        (MODEL_DIR / f"{MODEL_NAME}_config.json").read_text(encoding="utf-8"))
    return payload
