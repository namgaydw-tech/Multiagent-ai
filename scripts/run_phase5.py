#!/usr/bin/env python
"""Phase 5 CLI — multi-dataset, multi-algorithm external validation & robustness.

Research prototype — not a medical device.

Usage::

    python scripts/run_phase5.py audit                     # dataset blocker (27 checks)
    python scripts/run_phase5.py train    --dataset <ID>   # train + calibrate + lock thresholds
    python scripts/run_phase5.py evaluate --dataset <ID>   # sealed test, once (after lock)
    python scripts/run_phase5.py agents   --dataset <ID>   # multi-agent runs with domain pack
    python scripts/run_phase5.py bias     --dataset <ID>   # anchor/bias experiments per backbone
    python scripts/run_phase5.py analyze                   # scorecard + summaries + figures
    python scripts/run_phase5.py all                       # everything above, in order

Flags: --models a,b (train subset), --seed N (default 20261002), --n-boot N (evaluate),
--dry-run (print the plan, execute nothing).

Subcommands whose runner is not implemented for a dataset print an explicit
``NOT_EXECUTED`` status (exit code 0 for `all`; a standalone call exits 3) — never a
fabricated result.

Dataset IDs::

    REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR   NEONATAL_SEPSIS_REGISTRY
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SEED = 20261002


def log(msg: str) -> None:
    print(f"[phase5] {msg}", flush=True)


# --------------------------------------------------------------------------- audit
def cmd_audit(args) -> int:
    if args.dry_run:
        log("DRY RUN — would run the 27-check dataset blocker for all candidates")
        return 0
    from src.data.dataset_blocker import run_blocking
    report = run_blocking(write=True)
    summary = report.get("summary", {})
    log(f"blocker: {summary}")
    log("wrote outputs/phase5/dataset_blocking_report.{json,csv} + docs/PHASE5_DATASET_AUDIT.md")
    return 0


# --------------------------------------------------------------------------- train
def cmd_train(args) -> int:
    if not args.dataset:
        log("ERROR: --dataset is required for train")
        return 2
    models = args.models.split(",") if args.models else None
    if args.dry_run:
        log(f"DRY RUN — would train {models or 'the full enabled zoo'} "
            f"on {args.dataset} (seed {args.seed}); test stays sealed")
        return 0
    from src.models.phase5_pipeline import train_dataset
    summary = train_dataset(args.dataset, models=models, seed=args.seed)
    n_ok = sum(1 for v in (summary.get("models") or {}).values()
               if isinstance(v, dict) and v.get("available"))
    log(f"{args.dataset}: {n_ok} models available, ensembles fitted on validation; "
        "thresholds locked before test")
    return 0


# ------------------------------------------------------------------------ evaluate
def cmd_evaluate(args) -> int:
    if not args.dataset:
        log("ERROR: --dataset is required for evaluate")
        return 2
    if args.dry_run:
        log(f"DRY RUN — would evaluate the sealed test of {args.dataset} once "
            f"(n_boot={args.n_boot}); thresholds already locked")
        return 0
    from src.models.phase5_pipeline import evaluate_dataset
    result = evaluate_dataset(args.dataset, n_boot=args.n_boot, seed=args.seed)
    lock = result.get("lock", {})
    log(f"{args.dataset}: n_models={result.get('n_models')} "
        f"first_evaluation={lock.get('first_evaluation_utc')} "
        f"n_evaluations={lock.get('n_evaluations')}")
    return 0


# -------------------------------------------------------------------------- agents
def cmd_agents(args) -> int:
    if not args.dataset:
        log("ERROR: --dataset is required for agents")
        return 2
    if args.dry_run:
        log(f"DRY RUN — would run the 8-stage information-isolated engine "
            f"on eligible {args.dataset} cases (domain pack, no ground truth shown)")
        return 0
    # Appendicitis: the proven Phase 3 runner already implements this exact flow.
    if args.dataset == "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR":
        import subprocess
        cmd = [sys.executable, str(ROOT / "scripts/run_phase3.py")]
        log("delegating to scripts/run_phase3.py (appendicitis domain pack is the default)")
        return subprocess.call(cmd, cwd=str(ROOT))
    # Other domains: domain packs exist (src/agents/domains.py) but the end-to-end
    # runner is NOT implemented yet — say so, never fabricate.
    log(f"NOT_EXECUTED: multi-agent runner for {args.dataset} is not implemented yet "
        "(domain packs exist in src/agents/domains.py; runner PLANNED) — no results invented")
    return 0 if args.called_from_all else 3


# ---------------------------------------------------------------------------- bias
def cmd_bias(args) -> int:
    if not args.dataset:
        log("ERROR: --dataset is required for bias")
        return 2
    if args.dry_run:
        log(f"DRY RUN — would run anchor/bias experiments on {args.dataset} "
            "across model backbones (LR/RF/XGB/LGBM/CatBoost/ensemble)")
        return 0
    if args.backbone:
        if args.dataset and args.dataset != "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR":
            log(f"NOT_EXECUTED: cross-backbone bias runs exist only for the Regensburg "
                f"appendicitis experiment (got {args.dataset}) — no results invented")
            return 0 if args.called_from_all else 3
        from src.bias.backbones import BACKBONES, run_all
        bbs = BACKBONES if args.backbone == "all" else [args.backbone]
        n_rows = int(getattr(args, "rows", 40))
        summary = run_all(bbs, n_rows=n_rows)
        log(f"cross-backbone bias: {list(summary['backbones'])} — "
            "outputs/phase5/bias_backbones_summary.json")
        return 0
    if args.dataset == "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR" and not args.backbone:
        log("delegating baseline (LightGBM backbone) bias experiments to run_phase4.py analyze "
            "-> headline CBR/AOR/BCR/HFR already persisted under outputs/metrics/")
        from src.evaluation.scorecard import write_all
        log(f"scorecard refreshed: {write_all()}")
        return 0
    log(f"NOT_EXECUTED: cross-backbone bias experiment runner for {args.dataset} "
        "is not implemented yet (PLANNED) — no results invented")
    return 0 if args.called_from_all else 3


# ------------------------------------------------------------------------- analyze
def cmd_analyze(args) -> int:
    if args.dry_run:
        log("DRY RUN — would build model scorecard, cross-dataset summary, "
            "external-validation status and Phase 5 figures from persisted outputs")
        return 0
    from src.evaluation.scorecard import write_all
    stats = write_all()
    log(f"scorecard: {stats['scorecard_rows']} rows -> "
        "outputs/phase5/model_scorecard.csv + .json")
    log(f"cross-dataset summary: {stats['datasets']} datasets; "
        f"external validation: {stats['external_validation']}")
    from src.evaluation.phase5_stats import write_statistics
    st = write_statistics(n_boot=args.n_boot if hasattr(args, "n_boot") else 2000)
    log(f"statistics: {st['rows']} comparison rows -> "
        "outputs/phase5/statistical_comparison.{json,csv}")
    # figures: only if the generator exists (never fabricate a figure)
    try:
        from src.evaluation import phase5_figures  # type: ignore
    except Exception:
        log("figures: NOT_GENERATED (phase5 figure generator not implemented yet) "
            "| reproduce with: python scripts/run_phase5.py analyze")
    else:
        n = phase5_figures.generate_all()
        log(f"figures: {n} written under outputs/phase5/*/figures/")
    return 0


# ---------------------------------------------------------------------------- all
def cmd_all(args) -> int:
    from src.data.dataset_blocker import run_blocking
    if not args.dry_run:
        report = run_blocking(write=True)
        log(f"audit: {report.get('summary', {})}")
    from src.models.phase5_pipeline import train_dataset, evaluate_dataset
    datasets = ([args.dataset] if args.dataset else
                ["REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR", "NEONATAL_SEPSIS_REGISTRY"])
    for ds in datasets:
        if args.dry_run:
            log(f"DRY RUN — would train+evaluate {ds}")
            continue
        log(f"=== {ds}: train ===")
        train_dataset(ds, seed=args.seed)
        log(f"=== {ds}: sealed-test evaluate ===")
        evaluate_dataset(ds, n_boot=args.n_boot, seed=args.seed)
        log(f"=== {ds}: agents/bias ===")
        a = argparse.Namespace(dataset=ds, dry_run=False, backbone=None, called_from_all=True)
        cmd_agents(a)
        cmd_bias(a)
    if not args.dry_run:
        cmd_analyze(argparse.Namespace(dry_run=False))
    log("done")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="run_phase5",
                                description="Phase 5 multi-dataset, multi-algorithm pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **kw):
        sp = sub.add_parser(name, **kw)
        sp.add_argument("--dataset", default=None)
        sp.add_argument("--dry-run", action="store_true")
        sp.add_argument("--seed", type=int, default=DEFAULT_SEED)
        sp.set_defaults(fn=fn, called_from_all=False)
        return sp

    add("audit", cmd_audit, help="run the 27-check dataset blocker")
    tr = add("train", cmd_train, help="train + calibrate + lock thresholds (test stays sealed)")
    tr.add_argument("--models", default=None, help="comma-separated subset of zoo models")
    ev = add("evaluate", cmd_evaluate, help="evaluate the sealed test partition once")
    ev.add_argument("--n-boot", type=int, default=2000)
    ag = add("agents", cmd_agents, help="run the information-isolated multi-agent engine")
    bi = add("bias", cmd_bias, help="anchor/bias experiments (cross-backbone)")
    bi.add_argument("--backbone", default=None,
                    help="model backbone: logistic_regression|random_forest|xgboost|"
                         "lightgbm|catboost|ensemble|all")
    bi.add_argument("--rows", type=int, default=40,
                    help="test rows per cross-backbone run (default 40)")
    an = add("analyze", cmd_analyze, help="scorecard, cross-dataset summary, figures")
    al = add("all", cmd_all, help="audit -> train -> evaluate -> agents -> bias -> analyze")
    al.add_argument("--n-boot", type=int, default=2000)

    args = p.parse_args()
    rc = args.fn(args)
    if args.cmd in ("agents", "bias") and rc == 3:
        print(f"[phase5] NOT_EXECUTED — {args.cmd} runner not available for "
              f"{args.dataset}; see docs/PHASE5_DATASET_AUDIT.md", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
