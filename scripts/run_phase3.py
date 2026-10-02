#!/usr/bin/env python
"""Phase 3 end-to-end execution: run the 8-stage multi-agent engine on real cases.

For each selected Regensburg dataset row this script:

1. builds the normalized Layer-A prediction (LightGBM + isotonic calibrator +
   validation-locked threshold + SHAP + uncertainty) via ``src/models/inference``;
2. runs stages 1-8 with information isolation (default condition C);
3. persists the per-case audit trail under ``outputs/audits/<case_id>/``;
4. collects run summaries into ``outputs/metadata/phase3_case_runs.json``.

Default case set (deterministic, seed 20261002):

* row 469 — model true negative, MODERATE uncertainty (discordant debate case);
* row 217 — model true negative, calibrated probability 0.0;
* row 746 — model true positive, calibrated probability ~1.0;
* row 393 — model true positive with HIGH predictive uncertainty;
* row 717 — a documented model false negative (from Phase 2 FN review).

Usage::

    python scripts/run_phase3.py               # default 5 cases
    python scripts/run_phase3.py --rows 469 746
    python scripts/run_phase3.py --ablation A_all_see_prediction
    python scripts/run_phase3.py --no-rag

Research prototype — not a medical device.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.inference import get_predictor                      # noqa: E402
from src.orchestration import persistence                           # noqa: E402
from src.orchestration.engine import StageError, run_case, run_case_summary  # noqa: E402
from src.orchestration.state import make_state                      # noqa: E402

DEFAULT_ROWS = [469, 217, 746, 393, 717]
OUT_META = ROOT / "outputs/metadata/phase3_case_runs.json"


def log(msg: str) -> None:
    print(f"[phase3] {msg}", flush=True)


def run_one(predictor, row: int, ablation: str, rag: bool, fresh: bool) -> dict:
    """Run stages 1-8 for one dataset row and return its summary dict."""
    case_id = predictor.case_id(row)
    directory = persistence.case_dir(case_id)
    if fresh and directory.exists():
        shutil.rmtree(directory)

    record = predictor.df.loc[row].to_dict()
    state = make_state(case_id, record, ablation=ablation, rag_enabled=rag,
                       config_hash=predictor.config_hash, seed=20261002)
    model_output, shap_explain, uncertainty = predictor.bundle(row)
    state.model_output = model_output
    state.shap_explain = shap_explain
    state.uncertainty = uncertainty

    t0 = time.time()
    state = run_case(state, resume=not fresh)
    elapsed = time.time() - t0

    summary = run_case_summary(state)
    summary.update({
        "row_index": row,
        "ground_truth": state.ground_truth,
        "ablation": ablation,
        "rag_enabled": rag,
        "elapsed_s": round(elapsed, 3),
        "uncertainty_level": uncertainty["uncertainty_level"],
        "watchdog_red_flags_present": len(
            state.stage_outputs["watchdog"].red_flags_present),
        "watchdog_cannot_assess": len(
            state.stage_outputs["watchdog"].cannot_assess_due_to_missing_data),
        "retrieved_sources": len(
            state.stage_outputs["evidence_retrieval"].retrieved),
        "opponent_contradictions": len(
            state.stage_outputs["opponent"].contradictory_evidence),
    })
    # isolation spot-checks (fail loudly if the data flow regressed)
    s2 = state.stage_inputs.get("independent_differential", {})
    if "model_output" in s2.get("record", {}) or s2.get("model_output") is not None:
        raise RuntimeError(f"isolation violated at stage 2 for {case_id}")
    if s2.get("withheld_from_this_stage") is None:
        raise RuntimeError(f"stage 2 snapshot missing isolation metadata for {case_id}")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, nargs="+", default=DEFAULT_ROWS)
    ap.add_argument("--ablation", default="default",
                    help="visibility ablation (default | A_all_see_prediction | "
                         "B_only_proponent | C_hidden_until_differential_complete | all_see_anchor)")
    ap.add_argument("--no-rag", action="store_true", help="disable retrieval stages")
    ap.add_argument("--no-fresh", action="store_true",
                    help="resume from existing audit files instead of re-running")
    args = ap.parse_args()

    if not (ROOT / "outputs/models/lightgbm_appendicitis.joblib").exists():
        log("ABORT: Phase 2 model artifact missing — run scripts/run_phase2.py first")
        return 2

    predictor = get_predictor()
    log(f"model={predictor.model_name} v{predictor.model_version} "
        f"threshold={predictor.threshold} calibrator={predictor.calibrator_name} "
        f"config_hash={predictor.config_hash}")

    summaries: list[dict] = []
    failures: list[str] = []
    for row in args.rows:
        if not predictor.has_row(row):
            failures.append(f"row {row}: not in modelling matrix")
            continue
        log(f"running case row {row} "
            f"(gt={predictor.df.loc[row, 'Diagnosis']!r}, ablation={args.ablation}, "
            f"rag={not args.no_rag}) ...")
        try:
            summary = run_one(predictor, row, args.ablation, not args.no_rag,
                              fresh=not args.no_fresh)
        except StageError as exc:
            failures.append(str(exc))
            log(f"  FAILED: {exc}")
            continue
        summaries.append(summary)
        log(f"  done in {summary['elapsed_s']}s: model_top={summary['model_top']!r} "
            f"final={summary['primary_working_diagnosis']!r} "
            f"conf={summary['multi_agent_confidence']} "
            f"risk={summary['watchdog_risk']} gt={summary['ground_truth']!r} "
            f"changed={summary['diagnosis_changed']} "
            f"retrieved={summary['retrieved_sources']}")

    OUT_META.parent.mkdir(parents=True, exist_ok=True)
    OUT_META.write_text(json.dumps({
        "generated_utc": persistence.utc_now(),
        "ablation": args.ablation,
        "rag_enabled": not args.no_rag,
        "seed": 20261002,
        "provider_backend": "deterministic (LLM_BACKEND unset — see PHASE3_REPORT)",
        "cases": summaries,
        "failures": failures,
        "disclaimer": "Research prototype — not a medical device.",
    }, indent=2), encoding="utf-8")
    log(f"wrote {OUT_META.relative_to(ROOT)} ({len(summaries)} cases, "
        f"{len(failures)} failures)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
