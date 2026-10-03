"""FastAPI application for the pediatric appendicitis research UI.

Launch (from the repository root)::

    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765

Routers (all under ``/api``):

* ``meta``        — /health, /repro (artifact availability + repro commands)
* ``metrics``     — Phase 2/4 results, error taxonomy, figures
* ``cases``       — test-partition case list, single-case prediction + SHAP
* ``agents``      — run the real 8-stage multi-agent pipeline
* ``experiments`` — experiment status + small synchronous bias experiments
* ``audits``      — browse persisted per-stage audit transcripts

Error policy: structured ``{"detail": …, "hint": …}`` responses with real
reasons; missing artifacts yield explicit availability states, never
placeholder data. CORS allows the local Vite dev server.

This API exposes a **research prototype** — not a medical device, not for
clinical use.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api import agents, audits, cases, experiments, meta, metrics
from backend.api._shared import DISCLAIMER, ApiUnavailable

app = FastAPI(
    title="Multiagent Pediatric Appendicitis Research API",
    version="1.0.0",
    description=(
        "Research prototype API: LightGBM predictions, the five-agent debiasing "
        "pipeline, confirmation-bias experiments and audit transcripts. "
        "NOT a medical device; not for clinical use."),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5199", "http://127.0.0.1:5199",
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:4173", "http://127.0.0.1:4173",
        "null",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(meta.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(cases.router, prefix="/api")
app.include_router(agents.router, prefix="/api")
app.include_router(experiments.router, prefix="/api")
app.include_router(audits.router, prefix="/api")


@app.exception_handler(ApiUnavailable)
async def unavailable_handler(_request: Request, exc: ApiUnavailable) -> JSONResponse:
    """Missing artifact → 503 with an actionable hint (never a placeholder)."""
    return JSONResponse(status_code=503,
                        content={"detail": str(exc), "hint": exc.hint or None})


@app.get("/")
def root() -> dict:
    """Service root: what this API is (and is not)."""
    return {
        "service": "multiagent-pediatric-appendicitis-research-api",
        "status": "ok",
        "endpoints": ["/api/health", "/api/repro", "/api/metrics/summary",
                      "/api/metrics/bias", "/api/metrics/final-results",
                      "/api/metrics/error-analysis", "/api/metrics/experiments",
                      "/api/metrics/figures/{name}", "/api/cases",
                      "/api/cases/{row_index}/record", "/api/predict",
                      "/api/agents/run", "/api/experiments/status",
                      "/api/experiments/quick", "/api/audits"],
        "seed": 20261002,
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8765, log_level="info")
