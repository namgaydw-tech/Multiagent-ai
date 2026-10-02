#!/usr/bin/env python
"""Phase 4: confirmation-bias experiments, ablations, evaluation, figures.

Phases (run in order, each bounded well under 10 minutes)::

    python scripts/run_phase4.py anchors                 # 7 anchor conditions
    python scripts/run_phase4.py ablations --anchor control
    python scripts/run_phase4.py ablations --anchor incorrect_anchor
    python scripts/run_phase4.py analyze                 # stats + errors + figures
    python scripts/run_phase4.py all                     # everything in order

What each phase does (config/experiment.yaml is authoritative):

* ``anchors``    — test partition (117 cases) × 7 anchor conditions × 3
                   repetitions on the complete framework (9_full_with_rag),
                   plus the single-agent baselines (2/3/4) on the same rows for
                   the paired ``bias_reduction = CBR_single − CBR_multi``.
* ``ablations``  — the 11 ablation profiles × the same cases × 3 repetitions
                   under one anchor setting (control or incorrect_anchor).
* ``analyze``    — headline bias metrics with bootstrap CIs, exact McNemar +
                   paired bootstrap + Benjamini-Hochberg, error taxonomy,
                   publication figures, ``final_results.json``/``.csv``.

Every executed case persists an ``audit.json`` under
``outputs/audits/experiments/`` before its result row is accepted.

Research prototype — not a medical device.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents.provider import LLMProvider                          # noqa: E402
from src.bias.anchor_generator import ANCHOR_CONDITIONS              # noqa: E402
from src.bias.anchor_generator import anchor_vocabulary              # noqa: E402
from src.bias import experiment as ex                                # noqa: E402
from src.bias import metrics as M                                    # noqa: E402
from src.data.split import load_splits                               # noqa: E402
from src.evaluation import error_analysis as EA                      # noqa: E402
from src.models.inference import get_predictor                       # noqa: E402
from src.preprocessing.regensburg import ROOT as PROOT               # noqa: E402

METRICS = PROOT / "outputs/metrics"
META = PROOT / "outputs/metadata"
ANCHOR_JSON = METRICS / "anchor_experiment.json"
SINGLE_JSON = METRICS / "single_baseline.json"
ABLATION_JSON = METRICS / "ablation_experiment.json"
BIAS_JSON = METRICS / "bias_metrics.json"
FINAL_JSON = METRICS / "final_results.json"
FINAL_CSV = METRICS / "final_results.csv"
SINGLE_PROFILES = ("2_single_llm", "3_lightgbm_plus_single_llm",
                   "4_lightgbm_self_reflection")
REPS_DEFAULT = 3


def log(msg: str) -> None:
    print(f"[phase4] {msg}", flush=True)


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False),
                    encoding="utf-8")
    log(f"wrote {path.relative_to(PROOT)} "
        f"({path.stat().st_size // 1024} KB)")


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def test_rows(predictor, limit: int | None) -> list[int]:
    rows = load_splits(PROOT / "data/interim/splits")["test"]
    rows = [int(r) for r in rows if predictor.has_row(int(r))]
    return rows[:limit] if limit else rows


def _meta(args, n_rows, phase) -> dict:
    return {"phase": phase, "seed": 20261002, "n_cases": n_rows,
            "reps": args.reps, "partition": "test",
            "llm_backend": "deterministic (LLM_BACKEND unset — see PHASE3/4 reports)",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_hash": None,
            "disclaimer": "Research prototype — not a medical device."}


# --------------------------------------------------------------------- phases
def phase_anchors(args) -> None:
    t0 = time.time()
    predictor = get_predictor()
    rows = test_rows(predictor, args.limit)
    vocab = anchor_vocabulary(predictor.df)
    bundles = {r: predictor.bundle(r) for r in rows}
    provider = LLMProvider(backend="deterministic")
    meta = _meta(args, len(rows), "anchors")
    meta["config_hash"] = predictor.config_hash
    meta["profiles"] = [ex.ANCHOR_PROFILE, *SINGLE_PROFILES]
    meta["anchor_conditions"] = list(ANCHOR_CONDITIONS)

    multi_rows: list[dict] = []
    single_rows: list[dict] = []
    for profile_name in (ex.ANCHOR_PROFILE, *SINGLE_PROFILES):
        profile = ex.PROFILES[profile_name]
        runner = ex.make_runner(
            predictor, profile, vocab, bundles,
            retriever=None, provider=provider, experiment="anchor",
            audit_root_base=ex.EXPERIMENT_ROOT / "anchor")
        for condition in ANCHOR_CONDITIONS:
            c0 = time.time()
            for rep in range(args.reps):
                for r in rows:
                    row = runner(condition, r, rep)
                    (multi_rows if profile.kind == "engine"
                     else single_rows).append(row)
            log(f"{profile_name:30s} {condition:36s} n={len(rows)*args.reps} "
                f"({time.time()-c0:.1f}s)")
    _save(ANCHOR_JSON, {"meta": meta, "rows": multi_rows})
    _save(SINGLE_JSON, {"meta": meta, "rows": single_rows})
    log(f"anchors phase done in {time.time()-t0:.1f}s "
        f"(multi {len(multi_rows)} + single {len(single_rows)} rows)")


def phase_ablations(args) -> None:
    if not args.anchor:
        raise SystemExit("--anchor {control,incorrect_anchor} is required")
    t0 = time.time()
    predictor = get_predictor()
    rows = test_rows(predictor, args.limit)
    vocab = anchor_vocabulary(predictor.df)
    bundles = {r: predictor.bundle(r) for r in rows}
    provider = LLMProvider(backend="deterministic")
    existing = _load(ABLATION_JSON) or {"meta": {}, "runs": {}}
    existing["meta"] = existing.get("meta") or {}

    for profile_name in sorted(ex.PROFILES):
        key = f"{profile_name}||{args.anchor}"
        if key in existing["runs"] and not args.force:
            log(f"skip {key} (already run; use --force to redo)")
            continue
        profile = ex.PROFILES[profile_name]
        runner = ex.make_runner(
            predictor, profile, vocab, bundles,
            retriever=None, provider=provider, experiment="ablation",
            audit_root_base=ex.EXPERIMENT_ROOT / "ablation")
        c0 = time.time()
        out: list[dict] = []
        for rep in range(args.reps):
            for r in rows:
                out.append(runner(args.anchor, r, rep))
        existing["runs"][key] = out
        log(f"{profile_name:30s} anchor={args.anchor:16s} n={len(out)} "
            f"({time.time()-c0:.1f}s)")
    existing["meta"].update({
        "seed": 20261002, "reps": args.reps, "partition": "test",
        "anchor_settings": sorted({k.split("||")[1] for k in existing["runs"]}),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "llm_backend": "deterministic (LLM_BACKEND unset)",
        "disclaimer": "Research prototype — not a medical device."})
    _save(ABLATION_JSON, existing)
    log(f"ablations phase done in {time.time()-t0:.1f}s")


def phase_analyze(args) -> None:
    t0 = time.time()
    predictor = get_predictor()
    anchor_payload = _load(ANCHOR_JSON)
    single_payload = _load(SINGLE_JSON)
    ablation_payload = _load(ABLATION_JSON) or {"runs": {}}
    if not anchor_payload or not single_payload:
        raise SystemExit("missing anchor/single results — run the `anchors` phase first")

    multi_rows = anchor_payload["rows"]
    single_rows = single_payload["rows"]
    by_condition = {c: [r for r in multi_rows if r["condition"] == c]
                    for c in ANCHOR_CONDITIONS}
    multi_headline = {c: M.headline_metrics(rows) for c, rows in by_condition.items()}
    single_headline: dict[str, dict] = {}
    for profile_name in SINGLE_PROFILES:
        single_headline[profile_name] = {
            c: M.headline_metrics([r for r in single_rows
                                   if r["condition"] == c
                                   and r["profile"] == profile_name])
            for c in ANCHOR_CONDITIONS}

    # ---- paired bias reduction (incorrect-anchor cases, rep 0, same anchors)
    def _followed(r):
        if r.get("anchor_present") and r.get("anchor_correct") is False:
            return bool(r.get("anchor_followed"))
        return None

    multi_inc = [r for r in multi_rows
                 if r["condition"] == "incorrect_anchor" and r["rep"] == 0]
    flags_multi = [_followed(r) for r in multi_inc]
    cbr_multi = M.rate_record(flags_multi)

    singles_for_reduction = {}
    for profile_name in ("2_single_llm", "3_lightgbm_plus_single_llm"):
        single_inc = [r for r in single_rows
                      if r["condition"] == "incorrect_anchor"
                      and r["profile"] == profile_name and r["rep"] == 0]
        flags_single = [_followed(r) for r in single_inc]
        paired = M.paired_bootstrap_difference(flags_single, flags_multi)
        b = sum(1 for s, m in zip(flags_single, flags_multi)
                if s is True and m is not True)
        c = sum(1 for s, m in zip(flags_single, flags_multi)
                if s is not True and m is True)
        mcn = M.mcnemar_exact(b, c)
        singles_for_reduction[profile_name] = {
            "cbr_single": M.rate_record(flags_single),
            "paired_diff_single_minus_multi": paired,
            "mcnemar": mcn,
        }
    primary = singles_for_reduction["3_lightgbm_plus_single_llm"]
    bias_reduction = {
        "definition": "CBR_single_agent - CBR_multi_agent (config/experiment.yaml)",
        "cbr_single_2": singles_for_reduction["2_single_llm"]["cbr_single"],
        "cbr_single_3": primary["cbr_single"],
        "cbr_multi": cbr_multi,
        "paired_diff": primary["paired_diff_single_minus_multi"],
        "mcnemar": primary["mcnemar"],
        "secondary_vs_single_2": singles_for_reduction["2_single_llm"],
    }

    # ---- planned comparison family -> Benjamini-Hochberg
    planned: list[tuple[str, str, dict]] = []
    planned.append(("CBR_single3_vs_multi (incorrect anchor)", "mcnemar_exact",
                    primary["mcnemar"]))
    planned.append(("CBR_single2_vs_multi (incorrect anchor)", "mcnemar_exact",
                    singles_for_reduction["2_single_llm"]["mcnemar"]))
    # high- vs low-confidence incorrect anchor (same anchor value -> paired)
    hi = [r for r in multi_rows if r["condition"] == "high_confidence_incorrect_anchor"
          and r["rep"] == 0]
    lo = [r for r in multi_rows if r["condition"] == "low_confidence_incorrect_anchor"
          and r["rep"] == 0]
    planned.append(("CBR high vs low confidence anchor (multi)",
                    "paired_bootstrap",
                    M.paired_bootstrap_difference([_followed(r) for r in hi],
                                                  [_followed(r) for r in lo])))
    # control vs incorrect anchor: final accuracy (multi)
    ctrl = [r for r in multi_rows if r["condition"] == "control" and r["rep"] == 0]
    inc = [r for r in multi_rows if r["condition"] == "incorrect_anchor"
           and r["rep"] == 0]
    planned.append(("final accuracy control vs incorrect anchor (multi)",
                    "paired_bootstrap",
                    M.paired_bootstrap_difference(
                        [bool(r["final_correct"]) for r in ctrl],
                        [bool(r["final_correct"]) for r in inc])))
    # full framework vs LightGBM-only accuracy under control anchor
    abl = ablation_payload.get("runs", {})
    lgb = abl.get("1_lightgbm_only||control")
    full = abl.get("9_full_with_rag||control")
    if lgb and full:
        planned.append(("final accuracy full framework vs LightGBM-only (control)",
                        "paired_bootstrap",
                        M.paired_bootstrap_difference(
                            [bool(r["final_correct"]) for r in full if r["rep"] == 0],
                            [bool(r["final_correct"]) for r in lgb if r["rep"] == 0])))
    comparisons = [{"name": n, "test": t, **rec} for n, t, rec in planned]
    ps = [rec.get("p_value") for _, _, rec in planned]
    idx_ok = [i for i, p in enumerate(ps) if p is not None]
    bh = M.benjamini_hochberg([ps[i] for i in idx_ok])
    for slot, i in enumerate(idx_ok):
        comparisons[i]["q_value"] = bh["q_values"][slot]
        comparisons[i]["rejected_bh"] = bh["rejected"][slot]

    # ---- ablation summaries
    ablation_summary: dict[str, dict] = {}
    anchor_settings = sorted({k.split("||")[1] for k in abl}) or []
    for key, rows in abl.items():
        profile_name, setting = key.split("||")
        head = M.headline_metrics(rows)
        ablation_summary.setdefault(profile_name, {})[setting] = {
            k: head[k] for k in ("n_rows", "CBR", "AOR", "BCR", "HFR", "CRR",
                                 "CMR", "DR", "final_accuracy",
                                 "initial_accuracy")}

    # ---- error taxonomy (rep 0: control + incorrect anchor, multi profile)
    taxonomy_rows, taxonomy_summary = EA.build_error_taxonomy(
        by_condition["control"], by_condition["incorrect_anchor"])
    EA.write_taxonomy(taxonomy_rows, taxonomy_summary, METRICS)

    bias = {
        "meta": {**anchor_payload["meta"],
                 "ablation_anchor_settings": anchor_settings,
                 "statistics": {"ci_method": "bootstrap", "n_boot": 2000,
                                "alpha": 0.05,
                                "tests": ["mcnemar_exact",
                                          "paired_bootstrap_difference"],
                                "multiple_comparison_correction":
                                    "benjamini_hochberg"}},
        "anchor_experiment": {"profile": ex.ANCHOR_PROFILE,
                              "n_cases": len({r["row_index"] for r in multi_rows}),
                              "reps": anchor_payload["meta"].get("reps"),
                              "conditions": multi_headline},
        "single_baselines": single_headline,
        "bias_reduction": bias_reduction,
        "statistics": {"comparisons": comparisons, "benjamini_hochberg": bh},
        "ablations": ablation_summary,
        "ablation_anchor_setting": (anchor_settings[0]
                                    if "incorrect_anchor" in anchor_settings
                                    else (anchor_settings[0] if anchor_settings else None)),
        "error_analysis": taxonomy_summary,
    }
    n_cases = len({r["row_index"] for r in multi_rows})
    _save(BIAS_JSON, bias)

    # ---- final results (combined headline table)
    try:
        m2 = json.loads((METRICS / "model_comparison.json").read_text(encoding="utf-8"))
        lgb_metrics = m2["models"]["lightgbm_appendicitis"]["test_metrics"]
        phase2_ref = {k: lgb_metrics.get(k) for k in
                      ("auroc", "auprc", "sensitivity", "specificity", "ppv",
                       "npv", "f1", "f2", "brier", "ece")}
    except Exception:                                     # pragma: no cover
        phase2_ref = None
    figures = []
    try:
        from src.evaluation.publication_figures import generate_all_figures
        figures = [str(p.relative_to(PROOT))
                   for p in generate_all_figures(METRICS, PROOT / "outputs/figures")]
    except FileNotFoundError as exc:
        log(f"figures skipped: {exc}")
    final = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": 20261002,
        "n_test_cases": n_cases,
        "repetitions": anchor_payload["meta"].get("reps"),
        "llm_backend": "deterministic (no LLM configured — disclosed limitation)",
        "phase2_reference_test_metrics": phase2_ref,
        "anchor_experiment": bias["anchor_experiment"],
        "bias_reduction": bias_reduction,
        "statistics": bias["statistics"],
        "ablations": ablation_summary,
        "error_analysis": taxonomy_summary,
        "figures": figures,
        "implementation_status": {
            "implemented": ["anchor conditions (7)", "ablations (11)",
                            "bias metrics with bootstrap CIs", "McNemar + paired "
                            "bootstrap + Benjamini-Hochberg", "error taxonomy",
                            "publication figures"],
            "experimentally_validated_here": "deterministic backend, test partition",
            "not_validated": ["live-LLM conditions (no API key available)"],
        },
        "disclaimer": "Research prototype — not a medical device; "
                      "results are from a bias study, not clinical evidence.",
    }
    _save(FINAL_JSON, final)

    # ---- flat CSV of every headline metric
    with FINAL_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["section", "condition", "metric", "value",
                         "ci_low", "ci_high", "n_den", "note"])
        def _emit(section, condition, name, rec):
            if isinstance(rec, dict) and "value" in rec:
                ci = rec.get("ci95") or [None, None]
                writer.writerow([section, condition, name, rec.get("value"),
                                 ci[0], ci[1], rec.get("n_den"), ""])
            elif isinstance(rec, dict) and "mean_candidates_per_case" in rec:
                writer.writerow([section, condition, name,
                                 rec.get("mean_candidates_per_case"), "", "",
                                 rec.get("n_cases"), "DDR"])
        for cond, head in multi_headline.items():
            for name, rec in head.items():
                _emit("anchor_multi", cond, name, rec)
        for prof, conds in single_headline.items():
            for cond, head in conds.items():
                for name in ("CBR", "AOR", "HFR", "BCR", "final_accuracy"):
                    _emit("single_baseline", f"{prof}|{cond}", name, head[name])
        for prof, settings in ablation_summary.items():
            for setting, head in settings.items():
                for name, rec in head.items():
                    _emit("ablation", f"{prof}|{setting}", name, rec)
        for comp in comparisons:
            writer.writerow(["statistics", "", comp["name"],
                             comp.get("p_value"), "", "", "",
                             f"test={comp['test']}; q={comp.get('q_value')}; "
                             f"rejected={comp.get('rejected_bh')}"])
        writer.writerow(["bias_reduction", "incorrect_anchor",
                         "CBR_single_minus_multi",
                         bias_reduction["paired_diff"].get("diff"),
                         *(bias_reduction["paired_diff"].get("ci95") or [None, None]),
                         bias_reduction["paired_diff"].get("n_pairs"),
                         f"mcnemar_p={bias_reduction['mcnemar'].get('p_value')}"])
    log(f"figures: {figures}")
    log(f"analyze done in {time.time()-t0:.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["anchors", "ablations", "analyze", "all"])
    ap.add_argument("--anchor", choices=["control", "incorrect_anchor"],
                    help="anchor setting for the ablations phase")
    ap.add_argument("--reps", type=int, default=REPS_DEFAULT)
    ap.add_argument("--limit", type=int, default=None,
                    help="debug: only the first N test cases")
    ap.add_argument("--force", action="store_true",
                    help="redo ablation runs that already exist")
    args = ap.parse_args()

    if not (PROOT / "outputs/models/lightgbm_appendicitis.joblib").exists():
        log("ABORT: Phase 2 model missing — run scripts/run_phase2.py first")
        return 2

    if args.phase in ("anchors", "all"):
        phase_anchors(args)
    if args.phase in ("ablations", "all"):
        if not args.anchor and args.phase == "ablations":
            raise SystemExit("--anchor is required for the ablations phase")
        for setting in ([args.anchor] if args.anchor else ["control",
                                                           "incorrect_anchor"]):
            args.anchor = setting
            phase_ablations(args)
    if args.phase in ("analyze", "all"):
        phase_analyze(args)
    META.mkdir(parents=True, exist_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
