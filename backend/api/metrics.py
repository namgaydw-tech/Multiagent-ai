"""Metrics endpoints serving the real persisted Phase 2/4 results.

Every endpoint returns ``available: false`` with a hint when its artifact has
not been generated yet — the frontend renders an explicit empty state instead
of inventing numbers. When present, payloads are the genuine JSON/CSV files
written by ``scripts/run_phase2.py`` / ``scripts/run_phase4.py``.

Research prototype — not a medical device.
"""

from __future__ import annotations

import csv
import re

from fastapi import APIRouter
from fastapi.responses import FileResponse

from backend.api._shared import (DISCLAIMER, FIGURES_DIR, METRICS_DIR,
                                  ApiUnavailable, load_json)

router = APIRouter(prefix="/metrics", tags=["metrics"])


def _available(path_name: str, hint: str) -> dict:
    """Uniform availability wrapper so missing files are a UI state, not a 500."""
    path = METRICS_DIR / path_name
    if not path.exists():
        return {"available": False, "reason": f"{path_name} not generated",
                "hint": hint, "disclaimer": DISCLAIMER}
    try:
        payload = load_json(path, hint=hint)
    except ApiUnavailable as exc:
        return {"available": False, "reason": str(exc),
                "hint": exc.hint or hint, "disclaimer": DISCLAIMER}
    return {"available": True, "data": payload, "disclaimer": DISCLAIMER}


@router.get("/summary")
def summary() -> dict:
    """Phase 2 model comparison (AUROC/AUPRC/etc., bootstrap CIs, subgroups)."""
    return _available("model_comparison.json",
                      "Run: python scripts/run_phase2.py")


@router.get("/bias")
def bias() -> dict:
    """Phase 4 bias metrics: per-condition headline + bias reduction + stats."""
    return _available("bias_metrics.json",
                      "Run: python scripts/run_phase4.py anchors && "
                      "python scripts/run_phase4.py analyze")


@router.get("/final-results")
def final_results() -> dict:
    """Combined headline table (final_results.json)."""
    return _available("final_results.json",
                      "Run: python scripts/run_phase4.py analyze")


@router.get("/error-analysis")
def error_analysis() -> dict:
    """Error taxonomy summary + per-case rows (CSV parsed, capped)."""
    payload = _available("error_analysis.json",
                         "Run: python scripts/run_phase4.py analyze")
    if not payload.get("available"):
        return payload
    rows: list[dict] = []
    csv_path = METRICS_DIR / "error_taxonomy.csv"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            rows = [dict(r) for r in csv.DictReader(fh)][:500]
    payload["rows"] = rows
    return payload


@router.get("/experiments")
def experiments() -> dict:
    """Condensed experiment summaries (anchor headline + ablation matrix)."""
    bias, err = _maybe("bias_metrics.json")
    if bias is None:
        return {"available": False, "reason": err,
                "hint": "Run: python scripts/run_phase4.py anchors && "
                        "python scripts/run_phase4.py ablations --anchor control && "
                        "python scripts/run_phase4.py ablations --anchor incorrect_anchor && "
                        "python scripts/run_phase4.py analyze",
                "disclaimer": DISCLAIMER}
    return {
        "available": True,
        "anchor_conditions": bias["anchor_experiment"],
        "single_baselines": {k: v for k, v in bias["single_baselines"].items()},
        "bias_reduction": bias["bias_reduction"],
        "statistics": bias["statistics"],
        "ablations": bias["ablations"],
        "ablation_anchor_setting": bias.get("ablation_anchor_setting"),
        "disclaimer": DISCLAIMER,
    }


def _maybe(name: str) -> tuple[dict | None, str]:
    path = METRICS_DIR / name
    if not path.exists():
        return None, f"{name} not generated"
    try:
        return load_json(path), ""
    except ApiUnavailable as exc:
        return None, str(exc)


_FIGURE_RE = re.compile(r"^[a-z0-9_]+\.png$")


@router.get("/figures/{name}")
def figure(name: str) -> FileResponse:
    """Serve one generated figure (whitelist-validated, 404 with hint if absent)."""
    if not _FIGURE_RE.match(name):
        raise ApiUnavailable(f"invalid figure name {name!r}",
                             "Use a name from GET /api/health figures list")
    path = FIGURES_DIR / name
    if not path.exists():
        raise ApiUnavailable(
            f"figure {name} not generated",
            "Run: python scripts/run_phase2.py (Phase 2 figures) or "
            "python scripts/run_phase4.py analyze (Phase 4 figures)")
    return FileResponse(path, media_type="image/png")
