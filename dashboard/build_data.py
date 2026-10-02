"""Assemble dashboard/data/dashboard.json from real repository artifacts.

Sources (all committed to the repo, none fabricated):
  - outputs/metrics/regensburg_audit.json        (executed dataset audit)
  - research/dataset_inventory.csv               (verified dataset inventory)
  - research/literature_review.csv               (DOI-verified literature)
  - research/LITERATURE_REVIEW.md                (category grouping, parsed by DOI)
  - config/*.yaml                                (agents, stages, ablations, metrics)
  - git-derived metadata is NOT included (no git at deploy time)

Run:  python dashboard/build_data.py
Output: dashboard/data/dashboard.json  (committed so Vercel needs no Python at deploy time)
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "data" / "dashboard.json"


def load_audit() -> dict:
    p = ROOT / "outputs/metrics/regensburg_audit.json"
    if not p.exists():
        raise SystemExit("missing outputs/metrics/regensburg_audit.json - run: python -m src.data.audit_regensburg")
    return json.loads(p.read_text(encoding="utf-8"))


def load_inventory() -> list[dict]:
    with (ROOT / "research/dataset_inventory.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_literature() -> list[dict]:
    with (ROOT / "research/literature_review.csv").open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_categories(md_text: str) -> dict[str, list[str]]:
    """Map category heading -> list of DOIs mentioned under it (before the next heading)."""
    cats: dict[str, list[str]] = {}
    current = None
    for line in md_text.splitlines():
        h = line.strip()
        if h.startswith("## "):
            current = h[3:].strip()
            cats.setdefault(current, [])
        elif current:
            cats[current].extend(re.findall(r"DOI:(10\.[^\s\)\;]+)", line))
    return {k: v for k, v in cats.items() if v}


def main() -> int:
    audit = load_audit()
    inventory = load_inventory()
    lit = load_literature()
    md = (ROOT / "research/LITERATURE_REVIEW.md").read_text(encoding="utf-8")
    cats = parse_categories(md)

    doi_to_cat: dict[str, str] = {}
    for cat, dois in cats.items():
        for d in dois:
            doi_to_cat.setdefault(d.lower().rstrip("."), cat)

    literature = []
    for r in lit:
        doi = r["DOI"].strip()
        literature.append(
            {
                "title": r["title"],
                "authors": r["authors"],
                "year": r["year"],
                "venue": r["journal_or_conference"],
                "doi": doi,
                "url": r["URL"],
                "category": doi_to_cat.get(doi.lower(), "Other / methods"),
                "open_access": r["open_access"],
                "relevance": r["relevance_to_project"],
            }
        )

    missingness = audit["missingness"][:12]
    flagged = audit["leakage_screen"]["flagged_columns"]

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": {
            "title": "Mitigating Confirmation Bias in Automated Pediatric Diagnostics via Adversarial Multi-Agent AI Frameworks",
            "status": "Phase 1 complete — no model trained, no experiment executed",
            "disclaimer": "Research prototype. Not a medical device. Not clinically validated. Not a replacement for qualified pediatric clinicians.",
            "phase": 1,
        },
        "audit": {
            "verdict": "PASS (conditional)",
            "dimensions": audit["dimensions"],
            "dtypes": audit["dtypes"],
            "missing_cells_pct": audit["pct_missing_cells"],
            "missing_cells_total": audit["n_missing_cells_total"],
            "class_balance": audit["class_balance_diagnosis"],
            "targets": audit["target_distributions"],
            "age": audit["age_distribution"],
            "duplicates": audit["duplicates"],
            "missingness_top": missingness,
            "flagged_columns": flagged,
            "anchor": audit["anchor_variable"],
            "sources": audit["sources"],
            "limitations": audit["known_limitations"],
            "timestamp": audit["audit_timestamp_utc"],
        },
        "datasets": inventory,
        "literature": {
            "count": len(literature),
            "categories": sorted(cats.keys()),
            "entries": literature,
        },
        "pipeline": {
            "stages": [
                {"n": 1, "id": "data_cleansing", "name": "Data Cleansing (Agent 1)",
                 "purpose": "Objective extraction, missing-data report, red flags. No diagnosis.",
                 "sees_prediction": False},
                {"n": 2, "id": "independent_differential", "name": "Independent Differential",
                 "purpose": "Opponent forms a differential from raw evidence BEFORE seeing any anchor.",
                 "sees_prediction": False},
                {"n": 3, "id": "proponent", "name": "Proponent Argument (Agent 2)",
                 "purpose": "Strongest evidence-based case for the classifier's top prediction.",
                 "sees_prediction": True},
                {"n": 4, "id": "opponent", "name": "Opponent Attack (Agent 3)",
                 "purpose": "Attacks the leading hypothesis citing patient/retrieved evidence only.",
                 "sees_prediction": "after stage 2 (config A–C)"},
                {"n": 5, "id": "watchdog", "name": "High-Acuity Watchdog (Agent 4)",
                 "purpose": "Rule-first, prediction-blind red-flag screening vs cited pediatric criteria.",
                 "sees_prediction": False},
                {"n": 6, "id": "retrieval", "name": "Evidence Retrieval (RAG)",
                 "purpose": "Guideline/abstract chunks with provenance (license-aware).",
                 "sees_prediction": "n/a"},
                {"n": 7, "id": "arbitration", "name": "Arbitration (Agent 5)",
                 "purpose": "Weighted synthesis — explicitly not majority voting; confidence ≠ model probability.",
                 "sees_prediction": True},
                {"n": 8, "id": "bias_audit", "name": "Bias Audit",
                 "purpose": "CBR / AOR / BCR / HFR / CRR / stability computed and persisted per case.",
                 "sees_prediction": "n/a"},
            ],
            "ablations": [
                {"id": 1, "name": "LightGBM alone", "status": "planned"},
                {"id": 2, "name": "Single LLM", "status": "planned"},
                {"id": 3, "name": "LightGBM + single LLM", "status": "planned"},
                {"id": 4, "name": "LightGBM + self-reflection", "status": "planned"},
                {"id": 5, "name": "Generic multi-agent (no adversary)", "status": "planned"},
                {"id": 6, "name": "Proponent / Opponent debate", "status": "planned"},
                {"id": 7, "name": "Full five-agent framework", "status": "planned"},
                {"id": 8, "name": "Full framework without RAG", "status": "planned"},
                {"id": 9, "name": "Full framework with RAG", "status": "planned"},
                {"id": 10, "name": "All agents see the anchor", "status": "planned"},
                {"id": 11, "name": "Information-isolated agents", "status": "planned"},
            ],
            "conditions": [
                {"id": "control", "name": "Control", "anchor": "none"},
                {"id": "correct_anchor", "name": "Correct anchor", "anchor": "preliminary diagnosis (correct)"},
                {"id": "incorrect_anchor", "name": "Incorrect anchor", "anchor": "preliminary diagnosis (wrong)"},
                {"id": "high_conf_incorrect", "name": "High-confidence incorrect", "anchor": "assertive wrong diagnosis"},
                {"id": "low_conf_incorrect", "name": "Low-confidence incorrect", "anchor": "tentative wrong diagnosis"},
                {"id": "model_anchor", "name": "Model anchor", "anchor": "LightGBM top prediction"},
                {"id": "no_model_anchor", "name": "No model anchor", "anchor": "hidden until differential complete"},
            ],
        },
        "agents": [
            {"id": "A1", "name": "Data Cleanser", "sees": "raw patient features only",
             "output": "structured JSON: demographics, vitals, labs, symptoms, findings, missing critical info, red flags, data-quality warnings",
             "rule": "Must not diagnose. Must not invent pediatric thresholds."},
            {"id": "A2", "name": "Proponent", "sees": "patient data + probabilities + SHAP + uncertainty",
             "output": "supported_diagnosis, model_probability, supporting_evidence, missing_expected_evidence, contradictions_acknowledged, confidence",
             "rule": "Must acknowledge contradictions; SHAP wording is non-causal."},
            {"id": "A3", "name": "Opponent / Devil's Advocate", "sees": "stage 2: raw evidence only; stage 4: hypothesis to attack",
             "output": "challenged_hypothesis, contradictory_evidence, missing_criteria, alternative_diagnoses, strongest_alternative, disconfirmatory_tests, anchoring_risk",
             "rule": "Every challenge must cite actual evidence; no disagreement for its own sake."},
            {"id": "A4", "name": "High-Acuity Watchdog", "sees": "patient evidence + cited rules (prediction-blind)",
             "output": "conditions_considered, red_flags_present/absent, cannot_assess, urgent_rule_outs, risk_level, evidence",
             "rule": "High sensitivity, evidence-grounded; fail-closed on pending rules."},
            {"id": "A5", "name": "Pediatric Consultant / Arbitrator", "sees": "everything above + calibration + retrieval",
             "output": "primary_working_diagnosis, multi_agent_confidence, critical_differentials, urgent_rule_outs, disagreement, bias-risk before/after, audit_trail, disclaimer",
             "rule": "Not majority voting; confidence must not reuse the model probability."},
        ],
        "bias_metrics": [
            {"code": "CBR", "name": "Confirmation Bias Rate", "formula": "followed incorrect anchor / incorrect-anchor cases"},
            {"code": "AOR", "name": "Anchor Override Rate", "formula": "rejected incorrect anchor / incorrect-anchor cases"},
            {"code": "BCR", "name": "Beneficial Correction Rate", "formula": "incorrect initial diagnoses corrected / incorrect initial diagnoses"},
            {"code": "HFR", "name": "Harmful Flip Rate", "formula": "correct → incorrect after debate / initially correct"},
            {"code": "CRR", "name": "Contradiction Recovery Rate", "formula": "appropriate revisions on decisive new contradicting evidence / such cases"},
            {"code": "DDR", "name": "Differential Diversity", "formula": "unique plausible hypotheses before consensus"},
            {"code": "CMR", "name": "Critical Miss Rate", "formula": "high-acuity targets missed / high-acuity targets present"},
            {"code": "DR", "name": "Diagnostic Stability", "formula": "identical conclusions across repeated runs"},
        ],
        "docs": [
            {"path": "docs/DATASET_AUDIT.md", "label": "Dataset audit"},
            {"path": "docs/DATASET_CARD.md", "label": "Dataset card"},
            {"path": "docs/ARCHITECTURE.md", "label": "Architecture"},
            {"path": "docs/MODEL_CARD.md", "label": "Model card (planned)"},
            {"path": "docs/SAFETY.md", "label": "Safety"},
            {"path": "docs/LIMITATIONS.md", "label": "Limitations"},
            {"path": "docs/REPRODUCIBILITY.md", "label": "Reproducibility"},
            {"path": "docs/REPORTING_CHECKLIST.md", "label": "Reporting checklist"},
            {"path": "research/LITERATURE_REVIEW.md", "label": "Literature review"},
            {"path": "research/RESEARCH_GAP.md", "label": "Research gap"},
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")
    print(f"  audit rows={payload['audit']['dimensions']['n_rows']} cols={payload['audit']['dimensions']['n_cols']}")
    print(f"  datasets={len(inventory)} literature={len(literature)} categories={len(cats)}")

    # Copy linked markdown docs into the dashboard so the static bundle is self-contained
    # (identical behaviour on localhost and Vercel, where only dashboard/ is deployed).
    copied = 0
    for d in payload["docs"]:
        src = ROOT / d["path"]
        if not src.exists():
            print(f"  WARNING missing doc: {d['path']}")
            continue
        dest = Path(__file__).resolve().parent / d["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        copied += 1
    print(f"  copied {copied} linked docs into dashboard/ for static hosting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
