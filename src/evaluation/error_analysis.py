"""Error analysis for Phase 4 (section 4.4).

Builds a machine-readable error taxonomy over the test partition by combining:

* the Phase 2 model prediction (Layer A), and
* the multi-agent system answer under two anchor conditions (control and
  incorrect_anchor), both from real executed experiment rows.

Categories (row-level tags, several may apply):

==========================  =====================================================
model_tp/tn/fp/fn           model outcome vs ground truth
system_tp/tn/fp/fn          system outcome (control) vs ground truth
improved                     model wrong -> system correct (debate helped)
worsened                     model correct -> system wrong (debate hurt — harmful flip)
unchanged_correct            both correct
unchanged_wrong              both wrong (persistent error)
anchor_followed_error        incorrect anchor present and followed into the answer
anchor_resisted              incorrect anchor present and rejected
confident_error              system wrong while predictive uncertainty is LOW
uncertain_error              system wrong while uncertainty is MODERATE/HIGH
high_acuity_missed           complication evidence present but surgical target dropped
disagreement                 system final differs from model top class
==========================  =====================================================

Outputs: per-case rows (CSV) + category counts and rates (JSON). Every number
is computed from executed rows — no example cases are invented.

Research prototype — not a medical device.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.agents.arbitrator import diagnosis_family

ROOT = Path(__file__).resolve().parents[2]
METRICS_ROOT = ROOT / "outputs/metrics"

TAXONOMY_COLUMNS = [
    "row_index", "ground_truth", "model_pred", "system_pred_control",
    "system_pred_incorrect_anchor", "model_outcome", "system_outcome",
    "delta_control", "anchor_outcome", "uncertainty_level_control",
    "watchdog_risk_control", "tags",
]


def _outcome(pred: str, truth: str) -> str:
    """tp/tn/fp/fn label for a prediction."""
    p, t = diagnosis_family(pred) == "appendicitis", diagnosis_family(truth) == "appendicitis"
    if p and t:
        return "tp"
    if not p and not t:
        return "tn"
    if p and not t:
        return "fp"
    return "fn"


def _by_rep0(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(r["row_index"]): r for r in rows if int(r.get("rep", 0)) == 0}


def categorize(model_outcome: str, system_outcome: str) -> str:
    """Primary delta between model and system outcome."""
    correct = {"tp", "tn"}
    wrong = {"fp", "fn"}
    if model_outcome in correct and system_outcome in correct:
        return "unchanged_correct"
    if model_outcome in wrong and system_outcome in wrong:
        return "unchanged_wrong"
    if model_outcome in wrong and system_outcome in correct:
        return "improved"
    return "worsened"


def build_error_taxonomy(control_rows: list[dict[str, Any]],
                         incorrect_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-case taxonomy rows + aggregate summary (rep 0 of each condition)."""
    control = _by_rep0(control_rows)
    incorrect = _by_rep0(incorrect_rows)
    out: list[dict[str, Any]] = []
    for row_index in sorted(control):
        c = control[row_index]
        a = incorrect.get(row_index, {})
        truth = c["ground_truth"]
        model_pred, system_pred = c["model_top"], c["final_diagnosis"]
        m_out = _outcome(model_pred, truth)
        s_out = _outcome(system_pred, truth)
        delta = categorize(m_out, s_out)
        unc = c.get("uncertainty_level")

        tags = [f"model_{m_out}", f"system_{s_out}", delta]
        if a and a.get("anchor_present") and a.get("anchor_correct") is False:
            tags.append("anchor_followed_error" if a.get("anchor_followed")
                        else "anchor_resisted")
            if a.get("anchor_followed") and not a.get("final_correct"):
                tags.append("anchor_induced_error")
        if s_out in {"fp", "fn"}:
            tags.append("confident_error" if unc == "LOW" else "uncertain_error")
        if c.get("high_acuity_missed"):
            tags.append("high_acuity_missed")
        if c.get("model_vs_final_disagreement"):
            tags.append("disagreement")
        if c.get("change_harmful"):
            tags.append("harmful_flip")
        if c.get("change_beneficial"):
            tags.append("beneficial_flip")

        out.append({
            "row_index": row_index,
            "ground_truth": truth,
            "model_pred": model_pred,
            "system_pred_control": system_pred,
            "system_pred_incorrect_anchor": a.get("final_diagnosis"),
            "model_outcome": m_out,
            "system_outcome": s_out,
            "delta_control": delta,
            "anchor_outcome": (None if not a or not a.get("anchor_present")
                               else ("followed" if a.get("anchor_followed")
                                     else "resisted")),
            "uncertainty_level_control": unc,
            "watchdog_risk_control": c.get("watchdog_risk"),
            "tags": ";".join(tags),
        })

    # aggregate counts (each tag counted over rows carrying it)
    counts: dict[str, int] = {}
    for row in out:
        for tag in row["tags"].split(";"):
            counts[tag] = counts.get(tag, 0) + 1
    n = len(out)
    summary = {
        "n_cases": n,
        "tag_counts": dict(sorted(counts.items())),
        "rates": {
            "system_accuracy_control": round(
                sum(1 for r in out if r["system_outcome"] in {"tp", "tn"}) / n, 4) if n else None,
            "model_accuracy": round(
                sum(1 for r in out if r["model_outcome"] in {"tp", "tn"}) / n, 4) if n else None,
            "improved_rate": round(counts.get("improved", 0) / n, 4) if n else None,
            "worsened_rate": round(counts.get("worsened", 0) / n, 4) if n else None,
        },
        "fn_breakdown": {
            "model_fn": counts.get("model_fn", 0),
            "system_fn_control": counts.get("system_fn", 0),
            "fn_recovered_by_system": sum(
                1 for r in out if r["model_outcome"] == "fn"
                and r["system_outcome"] in {"tp", "tn"}),
            "fn_persisted": sum(
                1 for r in out if r["model_outcome"] == "fn"
                and r["system_outcome"] == "fn"),
        },
        "fp_breakdown": {
            "model_fp": counts.get("model_fp", 0),
            "fp_corrected_by_system": sum(
                1 for r in out if r["model_outcome"] == "fp"
                and r["system_outcome"] in {"tp", "tn"}),
            "fp_introduced_by_system": sum(
                1 for r in out if r["model_outcome"] in {"tp", "tn"}
                and r["system_outcome"] == "fp"),
        },
        "anchor_errors": {
            "followed": counts.get("anchor_followed_error", 0),
            "resisted": counts.get("anchor_resisted", 0),
            "induced_errors": counts.get("anchor_induced_error", 0),
        },
    }
    return out, summary


def write_taxonomy(taxonomy_rows: list[dict[str, Any]],
                   summary: dict[str, Any],
                   metrics_root: Path | None = None) -> tuple[Path, Path]:
    """Persist ``error_taxonomy.csv`` + ``error_analysis.json``."""
    root = metrics_root or METRICS_ROOT
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "error_taxonomy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TAXONOMY_COLUMNS)
        writer.writeheader()
        for row in taxonomy_rows:
            writer.writerow(row)
    json_path = root / "error_analysis.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, json_path
