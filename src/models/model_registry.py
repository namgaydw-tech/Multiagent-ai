"""Phase 5 model registry — the multi-algorithm zoo (section E).

``config/model_zoo.yaml`` declares the candidates; this module resolves them to
trainers. Contract for every tabular trainer::

    result = train_model(name, X_train, y_train, X_val, y_val, feature_types, ...)
    result["predict"](X)  ->  positive-class probabilities (n,)  [binary]
                          or  class probability matrix (n, K)    [multiclass]

Rules honoured:

* deterministic seeds (``seed`` everywhere: CV folds, model constructors, RNG);
* hyperparameter search only over training data with stratified (group) CV —
  validation is for calibration/threshold, test stays sealed;
* class weighting where configured (``class_weight`` in the zoo);
* an unavailable package disables its model with ``available: False`` and a
  note — never an exception, never a silent skip without a record.

Research prototype — not a medical device.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from src.models.baselines import _ordinal_encode
from src.preprocessing.regensburg import ROOT

ZOO_PATH = ROOT / "config/model_zoo.yaml"
SEED = 20261002


def load_zoo(path: Path | str = ZOO_PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def list_tabular_models(zoo: dict | None = None) -> list[str]:
    zoo = zoo or load_zoo()
    return [name for name, spec in zoo["tabular"].items()
            if isinstance(spec, dict) and spec.get("enabled", True)]


def image_model_specs(zoo: dict | None = None) -> dict[str, dict]:
    zoo = zoo or load_zoo()
    return {name: spec for name, spec in zoo["image"]["models"].items()
            if spec.get("enabled", True)}


def availability() -> dict[str, dict[str, Any]]:
    """Runtime availability of every zoo entry (import checks, no side effects)."""
    out: dict[str, dict[str, Any]] = {}
    for name in list_tabular_models():
        pkg = {"lightgbm": "lightgbm", "catboost": "catboost", "xgboost": "xgboost"}.get(name)
        if pkg is None:
            out[name] = {"kind": "tabular", "available": True}
            continue
        try:
            __import__(pkg)
            out[name] = {"kind": "tabular", "available": True}
        except Exception as exc:
            out[name] = {"kind": "tabular", "available": False,
                         "note": f"{pkg} not importable: {exc}"}
    return out


# ------------------------------------------------------------------ preprocess
def _onehot(numeric: list[str], categorical: list[str], scale: bool):
    from src.models.baselines import onehot_preprocessor
    return onehot_preprocessor(numeric, categorical, scale=scale)


def _ordinal_fit_transform(X: pd.DataFrame, categorical: list[str]):
    from src.models.baselines import _ordinal_encode
    return _ordinal_encode(X, categorical)


def _catboost_prepare(X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    out = X.copy()
    for col in categorical:
        out[col] = out[col].astype("object").fillna("__MISSING__").astype(str)
    return out


def _categorical_state_fit(X_train: pd.DataFrame, categorical: list[str]):
    from src.preprocessing.regensburg import CategoricalEncodingState
    state = CategoricalEncodingState()
    if categorical:
        state.fit(X_train[categorical], categorical)
    return state


# --------------------------------------------------------------------- trainers
def _search_cv(multiclass: bool, groups=None, seed: int = SEED, folds: int = 5):
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    if groups is not None:
        return StratifiedGroupKFold(n_splits=min(folds, 3), shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def _fit_pipeline_model(name: str, spec: dict, X_train, y_train, X_val, y_val,
                        numeric, categorical, *, multiclass: bool, groups,
                        seed: int, n_iter_scale: float) -> dict:
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.pipeline import Pipeline

    prep_kind = spec["preprocess"]
    if prep_kind == "onehot_scale":
        pre = _onehot(numeric, categorical, scale=True)
    else:
        pre = _onehot(numeric, categorical, scale=False)
    cw = spec.get("class_weight")
    clf = _make_estimator(name, spec, multiclass, cw, seed)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    grid = dict(spec.get("grid") or {})
    n_iter = max(1, int(round(spec.get("n_iter", 4) * n_iter_scale)))
    scoring = "average_precision" if not multiclass else "roc_auc_ovr_weighted"
    from sklearn.model_selection import ParameterSampler
    combos = list(ParameterSampler(grid, n_iter=min(n_iter, _grid_size(grid)),
                                   random_state=seed))
    cv = _search_cv(multiclass, groups, seed)
    t0 = time.time()
    best_score, best_params, best_est = -1.0, {}, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for params in combos:
            est = pipe.set_params(**params)
            scores = []
            for tr_idx, va_idx in cv.split(X_train, y_train,
                                           groups if groups is not None else None):
                est.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
                proba = est.predict_proba(X_train.iloc[va_idx])
                scores.append(_scoring_value(y_train.iloc[va_idx], proba, scoring, multiclass))
            mean = float(np.mean(scores))
            if mean > best_score:
                best_score, best_params, best_est = mean, params, None
        # refit best on the full training partition
        pipe.set_params(**best_params)
        pipe.fit(X_train, y_train)
        best_est = pipe
    return {
        "model_name": name, "model": best_est, "predict": _pipe_predict(best_est),
        "best_params": _jsonable(best_params), "cv_score": best_score,
        "cv_scoring": scoring, "fit_seconds": round(time.time() - t0, 2),
        "available": True, "kind": "tabular", "preprocess": None,
        "artifacts": {"type": "pipeline"},
    }


def _grid_size(grid: dict) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return max(n, 1)


def _scoring_value(y_true, proba, scoring: str, multiclass: bool) -> float:
    from sklearn.metrics import average_precision_score, roc_auc_score
    if not multiclass:
        return float(average_precision_score(y_true, proba[:, 1]))
    try:
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted"))
    except Exception:
        return 0.0


def _make_estimator(name: str, spec: dict, multiclass: bool, class_weight, seed: int):
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed,
                                  class_weight=class_weight)
    if name == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(random_state=seed, class_weight=class_weight)
    if name == "gaussian_nb":
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier()
    if name == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis(solver="lsqr")
    if name == "qda":
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        return QuadraticDiscriminantAnalysis()
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=seed)
    if name == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(random_state=seed)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, n_jobs=-1,
                                      class_weight=class_weight)
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(random_state=seed, n_jobs=-1, class_weight=class_weight)
    if name in ("svm_linear", "svm_rbf"):
        from sklearn.svm import SVC
        return SVC(kernel=("linear" if name == "svm_linear" else "rbf"),
                   probability=True, class_weight=class_weight, random_state=seed)
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier
        arch = tuple(spec.get("architecture", [64, 32]))
        return MLPClassifier(hidden_layer_sizes=arch, early_stopping=True,
                             validation_fraction=0.15, max_iter=400, random_state=seed,
                             learning_rate="adaptive")
    raise KeyError(f"no estimator mapping for {name!r}")


def _pipe_predict(est) -> Callable[[Any], np.ndarray]:
    def _predict(X) -> np.ndarray:
        proba = est.predict_proba(X if not isinstance(X, pd.DataFrame) else X)
        return proba[:, 1] if proba.shape[1] == 2 else proba
    return _predict


def _jsonable(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _fit_hist_gb(spec, X_train, y_train, X_val, multiclass, seed, n_iter_scale) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import ParameterSampler
    from src.models.baselines import _ordinal_encode

    cat = [c for c in X_train.columns if not pd.api.types.is_numeric_dtype(X_train[c])]
    Xtr, enc = _ordinal_encode(X_train, cat)
    grid = dict(spec["grid"])
    combos = list(ParameterSampler(grid, n_iter=max(1, int(spec.get("n_iter", 6) * n_iter_scale)),
                                   random_state=seed))
    scoring = "average_precision" if not multiclass else "roc_auc_ovr_weighted"
    cv = _search_cv(multiclass, None, seed)
    t0 = time.time()
    best_score, best_params = -1.0, {}
    for params in combos:
        clf = HistGradientBoostingClassifier(random_state=seed, **params)
        scores = []
        for tr_idx, va_idx in cv.split(Xtr, y_train):
            clf.fit(Xtr.iloc[tr_idx], y_train.iloc[tr_idx])
            proba = clf.predict_proba(Xtr.iloc[va_idx])
            scores.append(_scoring_value(y_train.iloc[va_idx], proba, scoring, multiclass))
        mean = float(np.mean(scores))
        if mean > best_score:
            best_score, best_params = mean, params
    clf = HistGradientBoostingClassifier(random_state=seed, **best_params)
    clf.fit(Xtr, y_train)

    def _predict(X) -> np.ndarray:
        Xp, _ = _ordinal_encode(X, cat, encoder=enc)
        proba = clf.predict_proba(Xp)
        return proba[:, 1] if proba.shape[1] == 2 else proba

    return {"model_name": "hist_gradient_boosting", "model": clf, "predict": _predict,
            "best_params": _jsonable(best_params), "cv_score": best_score,
            "cv_scoring": scoring, "fit_seconds": round(time.time() - t0, 2),
            "available": True, "kind": "tabular", "preprocess": _predict_preprocess(enc, cat),
            "artifacts": {"type": "hist_gb", "encoder": enc, "categorical": cat}}


def _predict_preprocess(enc, cat):
    def _pre(X):
        from src.models.baselines import _ordinal_encode
        return _ordinal_encode(X, cat, encoder=enc)[0]
    return _pre


def _fit_lightgbm(spec, X_train, y_train, X_val, multiclass, seed, n_iter_scale,
                  categorical: list[str]) -> dict:
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import ParameterSampler
    from src.preprocessing.regensburg import CategoricalEncodingState

    state = CategoricalEncodingState()
    if categorical:
        state.fit(X_train[categorical], categorical)
    Xtr = state.apply(X_train)

    obj = "multiclass" if multiclass else "binary"
    base = dict(objective=obj, random_state=seed, n_jobs=-1, verbose=-1,
                **({"num_class": int(y_train.max()) + 1} if multiclass else {}))
    grid = dict(spec["grid"])
    combos = list(ParameterSampler(grid, n_iter=max(1, int(spec.get("n_iter", 8) * n_iter_scale)),
                                   random_state=seed))
    scoring = "average_precision" if not multiclass else "roc_auc_ovr_weighted"
    cv = _search_cv(multiclass, None, seed)
    t0 = time.time()
    best_score, best_params = -1.0, {}
    for params in combos:
        clf = LGBMClassifier(**base, **params)
        scores = []
        for tr_idx, va_idx in cv.split(Xtr, y_train):
            clf.fit(Xtr.iloc[tr_idx], y_train.iloc[tr_idx])
            proba = clf.predict_proba(Xtr.iloc[va_idx])
            scores.append(_scoring_value(y_train.iloc[va_idx], proba, scoring, multiclass))
        mean = float(np.mean(scores))
        if mean > best_score:
            best_score, best_params = mean, params
    clf = LGBMClassifier(**base, **best_params)
    clf.fit(Xtr, y_train)

    def _predict(X) -> np.ndarray:
        proba = clf.predict_proba(state.apply(X))
        return proba[:, 1] if proba.shape[1] == 2 else proba

    return {"model_name": "lightgbm", "model": clf, "predict": _predict,
            "best_params": _jsonable(best_params), "cv_score": best_score,
            "cv_scoring": scoring, "fit_seconds": round(time.time() - t0, 2),
            "available": True, "kind": "tabular", "preprocess": _state_preprocess(state),
            "artifacts": {"type": "lightgbm", "state": state, "categorical": categorical}}


def _state_preprocess(state):
    def _pre(X):
        return state.apply(X)
    return _pre


def _fit_catboost(spec, X_train, y_train, X_val, multiclass, seed, n_iter_scale,
                  categorical: list[str]) -> dict:
    try:
        from catboost import CatBoostClassifier, Pool
    except Exception as exc:
        return {"model_name": "catboost", "available": False,
                "note": f"catboost not installed: {exc}"}

    Xtr = _catboost_prepare(X_train, categorical)
    base = dict(loss_function="MultiClass" if multiclass else "Logloss",
                eval_metric="TotalF1" if multiclass else "PRAUC",
                random_seed=seed, verbose=False, allow_writing_files=False,
                cat_features=list(categorical), thread_count=4)
    grid = dict(spec["grid"])
    keys = list(grid)
    rng = np.random.default_rng(seed)
    n_iter = max(1, int(spec.get("n_iter", 4) * n_iter_scale))
    combos = [{k: grid[k][int(rng.integers(0, len(grid[k])))] for k in keys}
              for _ in range(n_iter)]
    folds = int(spec.get("cv_folds", 3))
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scoring = "average_precision" if not multiclass else "roc_auc_ovr_weighted"
    t0 = time.time()
    best_score, best_params = -1.0, {}
    for params in combos:
        scores = []
        for tr_idx, va_idx in skf.split(Xtr, y_train):
            est = CatBoostClassifier(**{**base, **params})
            est.fit(Xtr.iloc[tr_idx], y_train.iloc[tr_idx])
            proba = est.predict_proba(Xtr.iloc[va_idx])
            scores.append(_scoring_value(y_train.iloc[va_idx], proba, scoring, multiclass))
        mean = float(np.mean(scores))
        if mean > best_score:
            best_score, best_params = mean, params
    final = CatBoostClassifier(**{**base, **best_params})
    final.fit(Xtr, y_train)

    def _predict(X) -> np.ndarray:
        proba = final.predict_proba(_catboost_prepare(X, categorical))
        return proba[:, 1] if proba.shape[1] == 2 else proba

    return {"model_name": "catboost", "model": final, "predict": _predict,
            "best_params": _jsonable(best_params), "cv_score": best_score,
            "cv_scoring": scoring, "fit_seconds": round(time.time() - t0, 2),
            "available": True, "kind": "tabular",
            "preprocess": None,
            "artifacts": {"type": "catboost", "categorical": categorical}}


def _fit_xgboost(spec, X_train, y_train, X_val, multiclass, seed, n_iter_scale) -> dict:
    try:
        import xgboost as xgb
    except Exception as exc:
        return {"model_name": "xgboost", "available": False,
                "note": f"xgboost not installed: {exc}"}
    from sklearn.model_selection import ParameterSampler
    from sklearn.pipeline import Pipeline
    numeric = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical = [c for c in X_train.columns if c not in numeric]
    pre = _onehot(numeric, categorical, scale=False)
    n_classes = int(y_train.max()) + 1
    base = dict(objective=("multi:softprob" if multiclass else "binary:logistic"),
                eval_metric="logloss", tree_method="hist", random_state=seed,
                n_jobs=4, **({"num_class": n_classes} if multiclass else {}))
    grid = dict(spec["grid"])
    combos = list(ParameterSampler(grid, n_iter=max(1, int(spec.get("n_iter", 6) * n_iter_scale)),
                                   random_state=seed))
    scoring = "average_precision" if not multiclass else "roc_auc_ovr_weighted"
    cv = _search_cv(multiclass, None, seed)
    t0 = time.time()
    best_score, best_params = -1.0, {}
    for params in combos:
        pipe = Pipeline([("pre", pre), ("clf", xgb.XGBClassifier(**base, **params))])
        scores = []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            pipe.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            proba = pipe.predict_proba(X_train.iloc[va_idx])
            scores.append(_scoring_value(y_train.iloc[va_idx], proba, scoring, multiclass))
        mean = float(np.mean(scores))
        if mean > best_score:
            best_score, best_params = mean, params
    pipe = Pipeline([("pre", pre), ("clf", xgb.XGBClassifier(**base, **best_params))])
    pipe.fit(X_train, y_train)
    return {"model_name": "xgboost", "model": pipe,
            "predict": _pipe_predict(pipe), "best_params": _jsonable(best_params),
            "cv_score": best_score, "cv_scoring": scoring,
            "fit_seconds": round(time.time() - t0, 2), "available": True,
            "kind": "tabular", "preprocess": None,
            "artifacts": {"type": "pipeline"}}


_PIPE_MODELS = {"logistic_regression", "decision_tree", "gaussian_nb", "knn", "lda",
                "qda", "random_forest", "extra_trees", "gradient_boosting", "adaboost",
                "svm_linear", "svm_rbf", "mlp"}


def rebuild_predict(saved: dict) -> Callable[[Any], np.ndarray]:
    """Recreate the predict closure from persisted, picklable artifacts."""
    art = saved.get("artifacts") or {"type": "pipeline"}
    kind = art["type"]
    est = saved["model"]
    if kind == "pipeline":
        def _f(X) -> np.ndarray:
            P = est.predict_proba(X)
            return P[:, 1] if P.shape[1] == 2 else P
        return _f
    if kind == "hist_gb":
        enc, cat = art["encoder"], art["categorical"]
        def _f(X) -> np.ndarray:
            Xp, _ = _ordinal_encode(X, cat, encoder=enc)
            P = est.predict_proba(Xp)
            return P[:, 1] if P.shape[1] == 2 else P
        return _f
    if kind == "lightgbm":
        state = art["state"]
        def _f(X) -> np.ndarray:
            P = est.predict_proba(state.apply(X))
            return P[:, 1] if P.shape[1] == 2 else P
        return _f
    if kind == "catboost":
        cat = art["categorical"]
        def _f(X) -> np.ndarray:
            P = est.predict_proba(_catboost_prepare(X, cat))
            return P[:, 1] if P.shape[1] == 2 else P
        return _f
    raise KeyError(f"unknown artifact type {kind!r}")


def train_model(name: str, X_train: pd.DataFrame, y_train: pd.Series,
                X_val: pd.DataFrame, y_val: pd.Series, *,
                numeric: list[str], categorical: list[str],
                multiclass: bool = False, groups=None, seed: int = SEED,
                n_iter_scale: float = 1.0, zoo: dict | None = None) -> dict:
    """Train one zoo entry on the training partition (CV over train only)."""
    zoo = zoo or load_zoo()
    spec = zoo["tabular"].get(name)
    if spec is None:
        raise KeyError(f"{name!r} not in config/model_zoo.yaml")
    if not spec.get("enabled", True):
        return {"model_name": name, "available": False, "note": "disabled in model_zoo.yaml"}
    try:
        if name in _PIPE_MODELS:
            return _fit_pipeline_model(name, spec, X_train, y_train, X_val, y_val,
                                       numeric, categorical, multiclass=multiclass,
                                       groups=groups, seed=seed, n_iter_scale=n_iter_scale)
        if name == "hist_gradient_boosting":
            return _fit_hist_gb(spec, X_train, y_train, X_val, multiclass, seed, n_iter_scale)
        if name == "lightgbm":
            return _fit_lightgbm(spec, X_train, y_train, X_val, multiclass, seed,
                                 n_iter_scale, categorical)
        if name == "catboost":
            return _fit_catboost(spec, X_train, y_train, X_val, multiclass, seed,
                                 n_iter_scale, categorical)
        if name == "xgboost":
            return _fit_xgboost(spec, X_train, y_train, X_val, multiclass, seed, n_iter_scale)
        raise KeyError(f"no trainer implemented for {name!r}")
    except Exception as exc:
        return {"model_name": name, "available": False,
                "note": f"training failed: {type(exc).__name__}: {exc}"}
