"""Health, capability and reproduction endpoints (``/api/health``, ``/api/repro``).

The health payload is what the UI uses to render honest availability states:
every metrics file, figure and experiment bundle reports ``exists`` so the
frontend can show "not generated yet — run phase X" instead of empty charts.

Research prototype — not a medical device.
"""

from __future__ import annotations

import os

from fastapi import APIRouter

from backend.api._shared import (DISCLAIMER, FIGURES_DIR, METRICS_DIR,
                                  MODELS_DIR, load_json_optional)
from backend.schemas.responses import HealthResponse, ModelInfo

router = APIRouter(tags=["meta"])

METRIC_FILES = [
    "model_comparison.json", "bias_metrics.json", "final_results.json",
    "error_analysis.json", "error_taxonomy.csv", "anchor_experiment.json",
    "ablation_experiment.json", "single_baseline.json",
]

FIGURE_NAMES = [
    "fig_anchor_cbr_aor.png", "fig_bias_reduction.png",
    "fig_ablation_matrix.png", "fig_error_taxonomy.png",
    "fig_anchor_outcomes.png", "roc_curve.png", "pr_curve.png",
    "confusion_matrix.png", "calibration_curve.png", "shap_summary.png",
]


def _model_info() -> ModelInfo:
    bundle = MODELS_DIR / "lightgbm_appendicitis.joblib"
    meta, _ = load_json_optional(MODELS_DIR / "lightgbm_calibrator_meta.json")
    cfg, _ = load_json_optional(MODELS_DIR / "lightgbm_appendicitis_config.json")
    if not bundle.exists():
        return ModelInfo(name="lightgbm_appendicitis", version="unknown",
                         available=False)
    return ModelInfo(
        name="lightgbm_appendicitis",
        version=str((cfg or {}).get("model_version", "unknown")),
        available=True,
        calibrator=(meta or {}).get("calibrator"),
        threshold=(meta or {}).get("threshold", {}).get("threshold"),
        config_hash=(meta or {}).get("config_hash") or (cfg or {}).get("config_hash"),
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service status: backend, model, RAG, metrics, figures, experiments."""
    metrics = {name: (METRICS_DIR / name).exists() for name in METRIC_FILES}
    figures = [n for n in FIGURE_NAMES if (FIGURES_DIR / n).exists()]
    rag_chunks: int | None = None
    try:
        from src.rag.store import DEFAULT_DB, ChunkStore
        if DEFAULT_DB.exists():
            store = ChunkStore(DEFAULT_DB)
            try:
                rag_chunks = len(store.all_chunks())
            finally:
                store.close()
    except Exception:                                        # pragma: no cover
        rag_chunks = None
    return HealthResponse(
        status="ok",
        llm_backend=os.environ.get("LLM_BACKEND", "").strip() or "deterministic",
        llm_configured=bool(os.environ.get("LLM_API_KEY", "").strip()),
        model=_model_info(),
        rag_chunks=rag_chunks,
        metrics_files=metrics,
        figures=figures,
        experiments_available={
            "anchor": metrics["anchor_experiment.json"],
            "ablations": metrics["ablation_experiment.json"],
            "bias_metrics": metrics["bias_metrics.json"],
            "final_results": metrics["final_results.json"],
        },
        disclaimer=DISCLAIMER,
    )


@router.get("/repro")
def repro() -> dict:
    """Exact commands to reproduce every phase (rendered on the Repro page)."""
    return {
        "train": ["python scripts/run_phase2.py"],
        "agents_cases": ["python scripts/run_phase3.py"],
        "experiments": [
            "python scripts/run_phase4.py anchors",
            "python scripts/run_phase4.py ablations --anchor control",
            "python scripts/run_phase4.py ablations --anchor incorrect_anchor",
            "python scripts/run_phase4.py analyze",
        ],
        "tests": ["python -m pytest tests/ -p no:warnings"],
        "backend": ["python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765"],
        "frontend": ["cd frontend", "npm install", "npm run dev   # http://localhost:5199"],
        "seed": 20261002,
        "disclaimer": DISCLAIMER,
    }
