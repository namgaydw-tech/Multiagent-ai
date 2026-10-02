"""Baseline benchmark models for pediatric appendicitis (Phase 2, section 2.4).

Trains on the same persisted patient-level partitions as LightGBM:

* Logistic Regression  (median imputation + scaling + one-hot)
* Random Forest         (median imputation + one-hot)
* HistGradientBoosting  (native NaN + ordinal-encoded categoricals)
* CatBoost              (native NaN handling + native categoricals) — if installed

Every model tunes only on the training partition (stratified CV), monitors on the
validation partition, and never touches the test partition.

Research prototype — not a medical device.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SEED = 20261002


# --------------------------------------------------------------------------- preprocessor helpers
def onehot_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    """Median-impute numerics (+ optional scaling) and one-hot encode categoricals."""
    num_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    cat_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        transformers=[("num", Pipeline(num_steps), numeric),
                      ("cat", Pipeline(cat_steps), categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# --------------------------------------------------------------------------- individual trainers
def train_logistic_regression(X_train, y_train, X_val, y_val, numeric, categorical) -> dict:
    """Tuned logistic regression baseline (AUPRC-optimised CV on train)."""
    pipe = Pipeline([
        ("pre", onehot_preprocessor(numeric, categorical, scale=True)),
        ("clf", LogisticRegression(max_iter=2000, solver="lbfgs", random_state=SEED)),
    ])
    grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    search = RandomizedSearchCV(
        pipe, grid, n_iter=4, scoring="average_precision", random_state=SEED,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1, error_score="raise",
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    return _result("logistic_regression", search, X_val, y_val, time.time() - t0)


def train_random_forest(X_train, y_train, X_val, y_val, numeric, categorical) -> dict:
    """Tuned random forest baseline."""
    pipe = Pipeline([
        ("pre", onehot_preprocessor(numeric, categorical, scale=False)),
        ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1)),
    ])
    params = {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [None, 12, 24],
        "clf__min_samples_leaf": [1, 3, 5],
    }
    search = RandomizedSearchCV(
        pipe, params, n_iter=9, scoring="average_precision", random_state=SEED,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1, error_score="raise",
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    return _result("random_forest", search, X_val, y_val, time.time() - t0)


def _ordinal_encode(X: pd.DataFrame, categorical: list[str], encoder=None):
    """Ordinal-encode categoricals with missing/unknown handling (fit on train)."""
    if encoder is None:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-2, encoded_missing_value=-1
        )
        encoder.fit(X[categorical].astype("object"))
    codes = encoder.transform(X[categorical].astype("object"))
    cat_df = pd.DataFrame(codes, columns=categorical, index=X.index).astype("int64")
    X_out = X.drop(columns=categorical).copy()
    for col in categorical:
        X_out[col] = pd.Categorical(cat_df[col])
    return X_out, encoder


def train_hist_gradient_boosting(X_train, y_train, X_val, y_val, numeric, categorical) -> dict:
    """Tuned sklearn HistGradientBoosting (native missing values)."""
    Xtr, enc = _ordinal_encode(X_train, categorical)
    Xva, _ = _ordinal_encode(X_val, categorical, encoder=enc)
    clf = HistGradientBoostingClassifier(
        categorical_features="from_dtype", random_state=SEED
    )
    params = {
        "learning_rate": [0.05, 0.1],
        "max_iter": [200, 300],
        "max_leaf_nodes": [15, 31],
        "min_samples_leaf": [10, 20],
        "l2_regularization": [0.0, 1.0],
    }
    search = RandomizedSearchCV(
        clf, params, n_iter=8, scoring="average_precision", random_state=SEED,
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1, error_score="raise",
    )
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(Xtr, y_train)
    val_proba = search.best_estimator_.predict_proba(Xva)[:, 1]
    result = _pack("hist_gradient_boosting", search, search.best_estimator_, val_proba,
                   y_val, time.time() - t0)
    # ordinal encoding must be re-applied at test time with the TRAIN-fitted encoder
    result["preprocess"] = (lambda X, _enc=enc: _ordinal_encode(X, categorical, encoder=_enc)[0])
    return result


def train_catboost(X_train, y_train, X_val, y_val, numeric, categorical) -> dict | None:
    """Tuned CatBoost baseline (skipped with a note if unavailable)."""
    try:
        from catboost import CatBoostClassifier
    except Exception:
        return {"model_name": "catboost", "available": False,
                "note": "catboost package not installed"}

    def prepare(X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in categorical:
            out[col] = out[col].astype("object").fillna("__MISSING__").astype(str)
        return out

    Xtr, Xva = prepare(X_train), prepare(X_val)
    base_params = dict(loss_function="Logloss", eval_metric="PRAUC", random_seed=SEED,
                       cat_features=list(categorical), verbose=False, allow_writing_files=False,
                       thread_count=4)

    # CatBoostClassifier cannot be cloned by sklearn's search (constructor parameter
    # mutation), so tuning uses an explicit stratified-CV loop instead of RandomizedSearchCV.
    # Search budget is deliberately small (documented in docs/PHASE2_REPORT.md): this host
    # runs ~0.065 s/iteration, so 4 candidates x 3 folds keeps the benchmark affordable.
    rng = np.random.default_rng(SEED)
    grid = {
        "iterations": [200, 300, 400],
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.06, 0.1],
        "l2_leaf_reg": [1.0, 3.0, 6.0],
    }
    n_iter = 4
    combos = [{k: grid[k][int(rng.integers(0, len(grid[k])))] for k in grid} for _ in range(n_iter)]

    from sklearn.model_selection import StratifiedKFold

    Xcb = Xtr
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    best_score, best_params, cv_scores = -1.0, None, []
    t0 = time.time()
    for params in combos:
        scores = []
        for tr_pos, te_pos in skf.split(Xcb, y_train):
            from catboost import CatBoostClassifier as _CBC
            est = _CBC(**{**base_params, **params})
            est.fit(Xcb.iloc[tr_pos], y_train.iloc[tr_pos])
            p = est.predict_proba(Xcb.iloc[te_pos])[:, 1]
            from sklearn.metrics import average_precision_score as _ap
            scores.append(float(_ap(y_train.iloc[te_pos], p)))
        mean_score = float(np.mean(scores))
        cv_scores.append({**params, "cv_auprc": round(mean_score, 5)})
        if mean_score > best_score:
            best_score, best_params = mean_score, params

    from catboost import CatBoostClassifier
    final = CatBoostClassifier(**{**base_params, **best_params})
    final.fit(Xtr, y_train)
    val_proba = final.predict_proba(Xva)[:, 1]
    from src.evaluation.model_metrics import compute_metrics
    result = {
        "model_name": "catboost",
        "model": final,
        "best_params": best_params,
        "cv_best_average_precision": float(best_score),
        "cv_candidates": cv_scores,
        "val_proba": np.asarray(val_proba),
        "val_metrics": compute_metrics(np.asarray(y_val), np.asarray(val_proba), threshold=0.5),
        "fit_seconds": round(time.time() - t0, 2),
        "available": True,
        "categorical": list(categorical),
        "preprocess": prepare,
    }
    return result


# --------------------------------------------------------------------------- helpers / API
def _result(name: str, search, X_val, y_val, elapsed: float) -> dict:
    val_proba = search.best_estimator_.predict_proba(X_val)[:, 1]
    return _pack(name, search, search.best_estimator_, val_proba, y_val, elapsed)


def _pack(name, search, estimator, val_proba, y_val, elapsed, **extra) -> dict:
    from src.evaluation.model_metrics import compute_metrics

    return {
        "model_name": name,
        "model": estimator,
        "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else
                            float(v) if isinstance(v, (np.floating,)) else v)
                        for k, v in search.best_params_.items()},
        "cv_best_average_precision": float(search.best_score_),
        "val_proba": np.asarray(val_proba),
        "val_metrics": compute_metrics(np.asarray(y_val), np.asarray(val_proba), threshold=0.5),
        "fit_seconds": round(elapsed, 2),
        "available": True,
        **extra,
    }


def train_baselines(X_train, y_train, X_val, y_val, numeric, categorical) -> list[dict]:
    """Train all baseline models on the shared partitions.

    Returns a list of result dicts (each with fitted model + validation metrics).
    """
    results = [
        train_logistic_regression(X_train, y_train, X_val, y_val, numeric, categorical),
        train_random_forest(X_train, y_train, X_val, y_val, numeric, categorical),
        train_hist_gradient_boosting(X_train, y_train, X_val, y_val, numeric, categorical),
        train_catboost(X_train, y_train, X_val, y_val, numeric, categorical),
    ]
    return [r for r in results if r is not None]
