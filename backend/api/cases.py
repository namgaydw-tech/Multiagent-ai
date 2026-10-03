"""Case listing and single-case prediction endpoints (Layer A from the UI).

``GET /api/cases`` lists the test-partition cases with their persisted Phase 2
predictions (read from ``outputs/predictions/test_predictions.csv`` — no
recomputation); ``POST /api/predict`` runs the real model + isotonic calibrator
+ SHAP explainer + uncertainty estimator for one row under a global lock.

Research prototype — not a medical device.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.api._shared import (DISCLAIMER, METRICS_DIR, ApiUnavailable,
                                  get_predictor, load_json, run_lock)
from backend.schemas.requests import PredictRequest
from backend.schemas.responses import PredictResponse

router = APIRouter(tags=["cases"])


@router.get("/cases")
def list_cases(limit: int = 100) -> dict:
    """Test-partition cases with persisted model predictions (no recompute)."""
    limit = max(1, min(limit, 780))
    csv_path = METRICS_DIR.parent / "predictions/test_predictions.csv"
    if not csv_path.exists():
        return {"available": False, "rows": [], "n": 0,
                "reason": "test_predictions.csv missing",
                "hint": "Run: python scripts/run_phase2.py",
                "disclaimer": DISCLAIMER}
    import csv as csv_mod
    rows = []
    with csv_path.open(encoding="utf-8") as fh:
        for rec in csv_mod.DictReader(fh):
            model_json = {}
            try:
                import json as _json
                model_json = _json.loads(rec.get("model_json") or "{}")
            except ValueError:
                model_json = {}
            rows.append({
                "row_index": int(rec["row_index"]),
                "y_true": int(rec["y_true"]),
                "p_raw": float(rec["p_raw"]),
                "p_calibrated": float(rec["p_calibrated"]),
                "threshold": float(rec["threshold"]),
                "pred_class": rec["pred_class"],
                "uncertainty_level": rec["uncertainty_level"],
                "entropy": float(rec["entropy"]),
                "model_version": model_json.get("model_version"),
                "config_hash": model_json.get("config_hash"),
            })
            if len(rows) >= limit:
                break
    return {"available": True, "n": len(rows), "rows": rows,
            "disclaimer": DISCLAIMER}


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Normalized prediction for one row: probabilities, calibration, SHAP, uncertainty."""
    try:
        predictor = get_predictor()
    except ApiUnavailable as exc:
        raise HTTPException(status_code=503, detail={"detail": str(exc),
                                                     "hint": exc.hint})
    if not predictor.has_row(req.row_index):
        raise HTTPException(
            status_code=404,
            detail={"detail": f"row {req.row_index} is not in the modelling matrix",
                    "hint": "Pick a row_index from GET /api/cases"})
    with run_lock():
        try:
            model_output = predictor.predict(req.row_index)
            shap_explain = predictor.explain(req.row_index)
        except Exception as exc:                     # surface the real failure
            raise HTTPException(
                status_code=500,
                detail={"detail": f"{type(exc).__name__}: {exc}",
                        "hint": "Check that Phase 2 artifacts are current"})
    uncertainty = model_output.get("uncertainty") or {}
    return PredictResponse(row_index=req.row_index, model_output=model_output,
                           shap_explain=shap_explain, uncertainty=uncertainty)


@router.get("/cases/{row_index}/record")
def case_record(row_index: int) -> dict:
    """Raw research record for one row (reserved columns stripped for display)."""
    try:
        predictor = get_predictor()
    except ApiUnavailable as exc:
        raise HTTPException(status_code=503, detail={"detail": str(exc),
                                                     "hint": exc.hint})
    if not predictor.has_row(row_index):
        raise HTTPException(status_code=404,
                            detail={"detail": f"row {row_index} not found",
                                    "hint": "Use GET /api/cases"})
    from src.orchestration.state import strip_record
    record = strip_record(predictor.df.loc[row_index].to_dict())
    clean = {}
    for key, value in record.items():
        if value is None or (isinstance(value, float) and value != value):
            clean[key] = None
        elif hasattr(value, "item"):
            clean[key] = value.item()
        else:
            clean[key] = value
    return {"row_index": row_index, "fields": clean,
            "n_fields": len(clean),
            "ground_truth": str(predictor.df.loc[row_index, "Diagnosis"]),
            "disclaimer": DISCLAIMER}
