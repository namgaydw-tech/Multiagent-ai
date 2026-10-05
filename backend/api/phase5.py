"""Phase 5 endpoints: dataset registry/blocking, scorecard, statistics, figures,
cross-dataset summary, and the Algorithm Lab's algorithm/formula catalogue.

Every endpoint serves the real persisted artifacts under ``outputs/phase5/`` and
returns an explicit ``available: false`` state with a reproduce hint when an
artifact has not been generated — the UI never invents numbers.

Research prototype — not a medical device.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from backend.api._shared import DISCLAIMER, ApiUnavailable, load_json

router = APIRouter(prefix="/phase5", tags=["phase5"])

ROOT = Path(__file__).resolve().parents[2]
P5 = ROOT / "outputs" / "phase5"
REPRO = "python scripts/run_phase5.py analyze"


def _available(rel: str, hint: str = REPRO) -> dict:
    path = P5 / rel
    if not path.exists():
        return {"available": False, "reason": f"{rel} not generated",
                "hint": hint, "disclaimer": DISCLAIMER}
    try:
        payload = load_json(path, hint=hint)
    except ApiUnavailable as exc:
        return {"available": False, "reason": str(exc),
                "hint": exc.hint or hint, "disclaimer": DISCLAIMER}
    return {"available": True, "data": payload, "disclaimer": DISCLAIMER}


# --------------------------------------------------------------------- status
@router.get("/status")
def status() -> dict:
    """One-call overview: blocker summary + which Phase 5 artifacts exist."""
    files = {
        "blocking_report": P5 / "dataset_blocking_report.json",
        "scorecard": P5 / "model_scorecard.json",
        "cross_dataset_summary": P5 / "cross_dataset_summary.json",
        "external_validation": P5 / "external_validation.json",
        "statistical_comparison": P5 / "statistical_comparison.json",
        "figures_manifest": P5 / "figures_manifest.json",
    }
    availability = {k: p.exists() for k, p in files.items()}
    blocker = None
    if files["blocking_report"].exists():
        try:
            blocker = load_json(files["blocking_report"],
                                hint="python scripts/run_phase5.py audit").get("summary")
        except ApiUnavailable:
            blocker = None
    executed = []
    for p in sorted(P5.iterdir()):
        mt = p / "metrics" / "metrics_test.json"
        if p.is_dir() and mt.exists():
            try:
                executed.append(str(load_json(mt, hint=REPRO).get("dataset_id") or p.name))
            except ApiUnavailable:
                executed.append(p.name)
    return {"available": True, "artifacts": availability,
            "blocker_summary": blocker, "datasets_executed": executed,
            "reproduce": REPRO, "disclaimer": DISCLAIMER}


# ---------------------------------------------------------- dataset registry
@router.get("/datasets")
def datasets() -> dict:
    """Dataset registry: every candidate with status + exact blocking reasons."""
    return _available("dataset_blocking_report.json",
                      hint="python scripts/run_phase5.py audit")


@router.get("/dataset/{dataset_id}/metrics")
def dataset_metrics(dataset_id: str) -> dict:
    """Validation + sealed-test metric tables for one dataset (Algorithm Lab)."""
    out: dict[str, Any] = {"dataset_id": dataset_id}
    found = False
    for d in P5.iterdir():
        mt = d / "metrics" / "metrics_test.json"
        if not d.is_dir() or not mt.exists():
            continue
        try:
            test = load_json(mt, hint=REPRO)
        except ApiUnavailable:
            continue
        if str(test.get("dataset_id")) != dataset_id:
            continue
        found = True
        out["dir"] = d.name          # artifacts directory (used to pick figures)
        out["test"] = test
        mv = d / "metrics" / "metrics_val.json"
        if mv.exists():
            out["validation"] = load_json(mv, hint=REPRO)
        ts = d / "metrics" / "train_summary.json"
        if ts.exists():
            train = load_json(ts, hint=REPRO)
            out["counts"] = train.get("counts")
            out["split_source"] = train.get("split_source")
            out["sensitivity_floor"] = train.get("sensitivity_floor")
        th = d / "calibration" / "thresholds.json"
        if th.exists():
            out["thresholds"] = load_json(th, hint=REPRO)
        cal = d / "calibration" / "calibration.json"
        if cal.exists():
            out["calibration"] = load_json(cal, hint=REPRO)
        break
    if not found:
        raise HTTPException(status_code=404, detail={
            "detail": f"no executed Phase 5 evaluation for {dataset_id}",
            "hint": "python scripts/run_phase5.py train --dataset <ID> && "
                    "python scripts/run_phase5.py evaluate --dataset <ID>"})
    out["disclaimer"] = DISCLAIMER
    return out


# ----------------------------------------------------- scorecard + summaries
@router.get("/scorecard")
def scorecard() -> dict:
    """Model scorecard: 28 fields per algorithm, unavailable metrics = null."""
    return _available("model_scorecard.json")


@router.get("/cross-dataset")
def cross_dataset() -> dict:
    return _available("cross_dataset_summary.json")


@router.get("/external-validation")
def external_validation() -> dict:
    return _available("external_validation.json")


@router.get("/statistics")
def statistics() -> dict:
    return _available("statistical_comparison.json")


# ------------------------------------------------------------------- figures
@router.get("/figures")
def figures() -> dict:
    """Manifest of the 35 required visualizations (GENERATED / NOT_GENERATED)."""
    return _available("figures_manifest.json")


@router.get("/figures/{figure_id}")
def figure_file(figure_id: int, dataset: str | None = Query(default=None)) -> dict:
    """Serve one persisted PNG (by target id, optionally filtered by dataset)."""
    man_path = P5 / "figures_manifest.json"
    if not man_path.exists():
        raise HTTPException(status_code=503, detail={
            "detail": "figures manifest not generated",
            "hint": REPRO})
    manifest = load_json(man_path, hint=REPRO)
    entry = next((f for f in manifest.get("figures", []) if f.get("id") == figure_id),
                 None)
    if entry is None:
        raise HTTPException(status_code=404, detail={
            "detail": f"unknown figure id {figure_id}", "hint": REPRO})
    paths = [p for p in entry.get("paths", [])]
    if dataset:
        paths = [p for p in paths if dataset.lower() in p.lower()]
    if not paths:
        raise HTTPException(status_code=404, detail={
            "detail": f"figure {figure_id} ({entry.get('name')}) NOT GENERATED"
                      + (f" for {dataset}" if dataset else ""),
            "note": entry.get("note"),
            "hint": entry.get("reproduce", REPRO)})
    path = Path(paths[0]).resolve()
    # traversal guard: only serve files inside the repository outputs tree
    if not path.is_relative_to((ROOT / "outputs").resolve()) or not path.exists():
        raise HTTPException(status_code=404, detail={
            "detail": "figure file missing on disk", "hint": REPRO})
    return {"available": True, "id": figure_id, "name": entry.get("name"),
            "url": f"/api/phase5/figure-file?path={path.relative_to(ROOT / 'outputs')}",
            "all_paths": paths, "disclaimer": DISCLAIMER}


@router.get("/figure-file")
def figure_binary(path: str = Query(...)) -> FileResponse:
    """Binary PNG delivery (relative to outputs/, traversal-checked)."""
    target = (ROOT / "outputs" / path).resolve()
    if not target.is_relative_to((ROOT / "outputs").resolve()) or not target.exists():
        raise HTTPException(status_code=404, detail={
            "detail": "figure file not found", "hint": REPRO})
    if target.suffix.lower() != ".png":
        raise HTTPException(status_code=400, detail={"detail": "only .png served"})
    return FileResponse(target, media_type="image/png")


# -------------------------------------------------- Algorithm Lab catalogue
ALGORITHM_FORMULAS: dict[str, dict[str, str]] = {
    "logistic_regression": {
        "family": "linear / baseline",
        "formula": "z = β0 + Σ βj·xj ;  P(y=1|x) = 1/(1+e^(-z)) ;  OR_j = e^(βj)",
        "why": "Linear classifier with probabilistic output; coefficients give odds ratios."},
    "decision_tree": {
        "family": "tree / baseline",
        "formula": "Gini = 1 − Σ pk² ;  H(S) = −Σ pk·log2(pk) ;  "
                   "IG = H(parent) − Σ (nchild/nparent)·H(child)",
        "why": "Axis-aligned recursive partitioning; interpretable rules."},
    "gaussian_nb": {
        "family": "probabilistic / baseline",
        "formula": "P(C|X) = P(X|C)·P(C) / P(X)  with P(xj|C) = N(μj,σj)",
        "why": "Conditional-independence generative classifier; strong baseline."},
    "knn": {
        "family": "instance-based / baseline",
        "formula": "d(x,z) = sqrt(Σi (xi−zi)²) ;  ŷ = vote of k nearest training points",
        "why": "Non-parametric; distance-based, so scaling is fit on TRAIN only."},
    "lda": {
        "family": "linear / baseline",
        "formula": "δk(x) = xᵀΣ⁻¹μk − ½ μkᵀΣ⁻¹μk + log πk",
        "why": "Class-mean separation under a shared covariance."},
    "qda": {
        "family": "quadratic / baseline",
        "formula": "δk(x) = −½ log|Σk| − ½ (x−μk)ᵀΣk⁻¹(x−μk) + log πk",
        "why": "Per-class covariance; fails when a class covariance is rank-deficient "
               "(recorded UNAVAILABLE on Regensburg)."},
    "random_forest": {
        "family": "tree ensemble",
        "formula": "ŷ = mode{h1(x),…,hB(x)} ;  P(y=c|x) = (1/B) Σb Pb(y=c|x)",
        "why": "Bagged randomized trees; robust nonlinear baseline."},
    "extra_trees": {
        "family": "tree ensemble",
        "formula": "split thresholds drawn uniformly at random, then averaged like RF",
        "why": "More randomization than RF → faster, lower variance/higher bias."},
    "gradient_boosting": {
        "family": "boosting",
        "formula": "Fm(x) = F(m−1)(x) + η·hm(x)  (gradient step on the loss)",
        "why": "Sequential additive correction of residuals."},
    "hist_gradient_boosting": {
        "family": "boosting",
        "formula": "Fm = F(m−1) + η·hm  with histogram-binned efficient splits",
        "why": "Scalable GBM variant with native missing-value support."},
    "adaboost": {
        "family": "boosting",
        "formula": "F(x) = Σt αt·ht(x) ,  αt = ½·ln((1−εt)/εt) ,  "
                   "wi ← wi·e^(−αt·yi·hti)",
        "why": "Exponential-loss reweighting of hard-to-classify cases."},
    "xgboost": {
        "family": "boosting",
        "formula": "Obj = Σi L(yi,ŷi) + Σk Ω(fk) ,  Ω(f) = γT + ½λ‖w‖²",
        "why": "Regularized gradient boosting with second-order optimization."},
    "lightgbm": {
        "family": "boosting",
        "formula": "Fm = F(m−1) + η·hm  (leaf-wise growth, histogram splitting)",
        "why": "Fast leaf-wise GBM; the Phase 2 reference backbone."},
    "catboost": {
        "family": "boosting",
        "formula": "Fm = F(m−1) + η·hm  with ordered boosting (no target leakage) "
                   "and native categorical features",
        "why": "Ordered boosting; keeps categoricals native (no OHE)."},
    "svm_linear": {
        "family": "margin",
        "formula": "f(x) = wᵀx + b ;  min ½‖w‖² + C·Σ max(0, 1−yi f(xi))",
        "why": "Maximum-margin linear separator with hinge loss."},
    "svm_rbf": {
        "family": "margin",
        "formula": "K(xi,xj) = exp(−γ‖xi−xj‖²) ;  f(x) = Σi αi·K(xi,x) + b",
        "why": "Kernel margin model for nonlinear boundaries."},
    "mlp": {
        "family": "neural",
        "formula": "z^(l) = W^(l)·a^(l−1) + b^(l) ;  a^(l) = φ(z^(l))",
        "why": "Shallow neural baseline; scaling fit on TRAIN only."},
    "ensemble_soft_uniform": {
        "family": "ensemble",
        "formula": "P(y=c|x) = Σm wm·Pm(y=c|x) ,  Σm wm = 1 ,  wm = 1/M",
        "why": "Uniform average of calibrated member probabilities."},
    "validation_weighted_soft_voting": {
        "family": "ensemble",
        "formula": "wm ∝ validation Macro F0.5 of member m (normalized, Σwm = 1) ;  "
                   "P = Σm wm·Pm",
        "why": "Weights fitted on VALIDATION only; test never consulted."},
    "ensemble_hard_voting": {
        "family": "ensemble",
        "formula": "ŷ = mode{ŷ1(x),…,ŷM(x)}  (no probabilities → AUROC = null)",
        "why": "Majority vote of member hard predictions."},
    "stacking": {
        "family": "ensemble",
        "formula": "Pmeta(y|x) = g( P1(x),…,PM(x) )  fitted on out-of-fold "
                   "validation predictions",
        "why": "Meta-learner on held-out member probabilities (validation only)."},
}

METRIC_FORMULAS: dict[str, str] = {
    "precision": "TP/(TP+FP)",
    "sensitivity": "TP/(TP+FN)",
    "specificity": "TN/(TN+FP)",
    "accuracy": "(TP+TN)/(TP+TN+FP+FN)",
    "NPV": "TN/(TN+FN)",
    "FPR": "FP/(FP+TN)",
    "FNR": "FN/(FN+TP)",
    "F1": "2PR/(P+R)",
    "F_beta": "(1+β²)PR/(β²P+R)",
    "F0.5": "1.25PR/(0.25P+R)",
    "Macro_F0.5": "(1/K) Σk F0.5k",
    "balanced_accuracy": "(sensitivity+specificity)/2",
    "MCC": "(TP·TN − FP·FN)/sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))",
    "Brier": "(1/N) Σi (pi − yi)²",
}


@router.get("/algorithms")
def algorithms() -> dict:
    """Algorithm catalogue for the Algorithm Lab: family, explanation, formula.

    Static mathematical documentation — never performance numbers (those come
    from /api/phase5/scorecard and /api/phase5/dataset/{id}/metrics).
    """
    return {"available": True, "algorithms": ALGORITHM_FORMULAS,
            "metrics": METRIC_FORMULAS,
            "selection_rule": (
                "winner = argmax validation Macro F0.5 subject to positive-class "
                "sensitivity ≥ floor (default 0.90); fallback = max sensitivity "
                "first. Thresholds locked on validation, test opened once."),
            "disclaimer": DISCLAIMER}
