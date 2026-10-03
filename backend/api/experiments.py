"""Quick bias experiment endpoint (synchronous, UI-sized).

Runs real anchor-condition cases through the real framework on the first N test
cases and returns per-condition headline metrics computed by
``src/bias/metrics.py``. Caps (N ≤ 15 cases, ≤ 4 profiles) keep the request
synchronous (~5–30 s); the full 117-case × 3-repetition experiments remain a
CLI operation, and the endpoint says so in its ``note``.

Every executed case still writes its ``audit.json`` (stopping rule:
``no_results_without_persisted_audit_json``) under
``outputs/audits/experiments/ui_quick/``.

Research prototype — not a medical device.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException

from backend.api._shared import (DISCLAIMER, METRICS_DIR, get_predictor,
                                  get_retriever, load_json_optional, run_lock)
from backend.schemas.requests import QuickExperimentRequest
from backend.schemas.responses import QuickExperimentResponse

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("/status")
def status() -> dict:
    """Which full experiments already exist (drives the UI availability states)."""
    final, final_err = load_json_optional(METRICS_DIR / "final_results.json")
    anchor, anchor_err = load_json_optional(METRICS_DIR / "anchor_experiment.json")
    ablation, ablation_err = load_json_optional(METRICS_DIR / "ablation_experiment.json")
    return {
        "full_results": {"available": final is not None, "reason": final_err},
        "anchor_experiment": {
            "available": anchor is not None, "reason": anchor_err,
            "n_rows": len((anchor or {}).get("rows", [])),
            "meta": (anchor or {}).get("meta"),
        },
        "ablation_experiment": {
            "available": ablation is not None, "reason": ablation_err,
            "runs": sorted((ablation or {}).get("runs", {})),
        },
        "hint": "Full reproduction: python scripts/run_phase4.py anchors / "
                "ablations --anchor … / analyze",
        "disclaimer": DISCLAIMER,
    }


@router.post("/quick", response_model=QuickExperimentResponse)
def quick_experiment(req: QuickExperimentRequest) -> QuickExperimentResponse:
    """Execute a small anchor experiment now and return real rows + headline metrics."""
    from src.bias import experiment as ex
    from src.bias import metrics as M
    from src.bias.anchor_generator import ANCHOR_CONDITIONS, anchor_vocabulary
    from src.data.split import load_splits
    from src.preprocessing.regensburg import ROOT as PROOT

    unknown = [c for c in req.conditions if c not in ANCHOR_CONDITIONS]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={"detail": f"unknown anchor conditions: {unknown}",
                    "hint": f"use subsets of: {', '.join(ANCHOR_CONDITIONS)}"})
    unknown = [p for p in req.profiles if p not in ex.PROFILES]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail={"detail": f"unknown profiles: {unknown}",
                    "hint": f"use subsets of: {', '.join(sorted(ex.PROFILES))}"})

    try:
        predictor = get_predictor()
    except Exception as exc:
        raise HTTPException(status_code=503,
                            detail={"detail": str(exc),
                                    "hint": "Run: python scripts/run_phase2.py"})
    rows = [int(r) for r in load_splits(PROOT / "data/interim/splits")["test"]
            if predictor.has_row(int(r))][:req.n_cases]

    t0 = time.time()
    from src.agents.provider import LLMProvider
    vocab = anchor_vocabulary(predictor.df)
    provider = LLMProvider(backend="deterministic")
    all_rows: list[dict] = []
    with run_lock():
        bundles = {r: predictor.bundle(r) for r in rows}
        retriever = get_retriever()
        for profile_name in req.profiles:
            profile = ex.PROFILES[profile_name]
            runner = ex.make_runner(
                predictor, profile, vocab, bundles, retriever=retriever,
                provider=provider, experiment="ui_quick",
                audit_root_base=ex.EXPERIMENT_ROOT / "ui_quick")
            for condition in req.conditions:
                for r in rows:
                    all_rows.append(runner(condition, r, 0))

    headline: dict[str, dict] = {}
    for profile_name in req.profiles:
        for condition in req.conditions:
            subset = [r for r in all_rows
                      if r["profile"] == profile_name and r["condition"] == condition]
            headline[f"{profile_name}|{condition}"] = M.headline_metrics(subset)

    return QuickExperimentResponse(
        n_cases=len(rows), conditions=req.conditions, profiles=req.profiles,
        rows=all_rows, headline=headline,
        elapsed_s=round(time.time() - t0, 2),
        note=("UI quick run: first N test cases, repetition 0 only. The published "
              "numbers use all 117 test cases × 3 repetitions — reproduce with: "
              "python scripts/run_phase4.py anchors / ablations / analyze"),
        disclaimer=DISCLAIMER)
