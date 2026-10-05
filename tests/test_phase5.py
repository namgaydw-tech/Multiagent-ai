"""Phase 5 tests: Macro F0.5, threshold policy, dataset blocker, leakage controls,
model zoo, scorecard nulls, statistics, figures manifest, domain packs, CLI.

These tests assert the *safety rules* of the Phase 5 spec: validation-only
selection, the sensitivity floor, BLOCKED-dataset gating, mirror dedup, group
isolation, and that unavailable metrics are never reported as zero.

Research prototype — not a medical device.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
P5 = ROOT / "outputs" / "phase5"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# =========================================================== Macro F0.5 formulas
def test_macro_f05_matches_sklearn_binary():
    from sklearn.metrics import fbeta_score
    from src.evaluation.phase5_metrics import macro_f05
    rng = np.random.default_rng(3)
    for _ in range(20):
        y = rng.integers(0, 2, 60)
        p = rng.integers(0, 2, 60)
        want = fbeta_score(y, p, beta=0.5, average="macro", zero_division=0)
        assert macro_f05(y, p) == pytest.approx(want, abs=1e-12)


def test_macro_f05_matches_sklearn_multiclass():
    from sklearn.metrics import fbeta_score
    from src.evaluation.phase5_metrics import macro_f05
    rng = np.random.default_rng(5)
    y = rng.integers(0, 3, 90)
    p = rng.integers(0, 3, 90)
    want = fbeta_score(y, p, beta=0.5, average="macro", zero_division=0)
    assert macro_f05(y, p) == pytest.approx(want, abs=1e-12)


def test_macro_f05_zero_division_is_zero_not_nan():
    from src.evaluation.phase5_metrics import macro_f05
    # class 1 never predicted: precision undefined -> 0 (sklearn zero_division=0)
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0])
    v = macro_f05(y, pred)
    assert not np.isnan(v)
    # class-1 F0.5 = 0 (precision undefined -> 0); class 0: P=2/4, R=2/2
    f0 = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0)
    assert v == pytest.approx(f0 / 2, abs=1e-12)


def test_f05_closed_form():
    """F0.5 = 1.25*P*R / (0.25*P + R)."""
    from src.evaluation.phase5_metrics import _f_beta
    p, r = 0.8, 0.6
    want = 1.25 * p * r / (0.25 * p + r)
    assert _f_beta(p, r, beta=0.5) == pytest.approx(want, abs=1e-12)
    # F1 = 2PR/(P+R) when beta = 1
    assert _f_beta(p, r, beta=1.0) == pytest.approx(2 * p * r / (p + r), abs=1e-12)


def test_full_metrics_contains_required_keys():
    from src.evaluation.phase5_metrics import compute_full_metrics
    rng = np.random.default_rng(9)
    y = rng.integers(0, 2, 80)
    p = rng.random(80)
    m = compute_full_metrics(y, p, threshold=0.4)
    for key in ("macro_f0_5", "positive_class_f0_5", "accuracy", "balanced_accuracy",
                "sensitivity", "specificity", "ppv", "npv", "auroc", "auprc",
                "brier", "ece", "mcc"):
        assert key in m, key
    # balanced accuracy = (sens + spec) / 2
    assert m["balanced_accuracy"] == pytest.approx(
        (m["sensitivity"] + m["specificity"]) / 2, abs=1e-12)


# ================================================== safety-constrained selection
def test_threshold_policy_satisfies_floor_when_feasible():
    from src.evaluation.threshold_policy import select_threshold_macro_f05
    rng = np.random.default_rng(11)
    y = np.array([0] * 40 + [1] * 60)
    p = np.where(y == 1, rng.uniform(0.6, 1.0, 100), rng.uniform(0.0, 0.5, 100))
    res = select_threshold_macro_f05(y, p, sensitivity_floor=0.90)
    yhat = (p >= res["threshold"]).astype(int)
    sens = ((yhat == 1) & (y == 1)).sum() / (y == 1).sum()
    assert sens >= 0.90
    assert res["floor_satisfied"] is True
    assert res["fitted_on"] == "validation"
    assert res["locked_before_test"] is True


def test_threshold_policy_fallback_prefers_sensitivity():
    """When NO grid threshold reaches the floor: max sensitivity first."""
    from src.evaluation.threshold_policy import select_threshold_macro_f05
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.random.default_rng(1).uniform(0.0, 0.5, 50),
                        np.random.default_rng(0).uniform(0.1, 0.6, 50)])
    grid = np.array([0.6, 0.7, 0.8, 0.9])   # floor 0.99 unreachable on this grid
    res = select_threshold_macro_f05(y, p, sensitivity_floor=0.99, grid=grid)
    assert res["floor_satisfied"] is False
    yhat = (p >= res["threshold"]).astype(int)
    sens = ((yhat == 1) & (y == 1)).sum() / (y == 1).sum()
    best = max(((p >= t).astype(int) * y == 1).sum() / (y == 1).sum() for t in grid)
    assert sens == pytest.approx(best), "fallback must maximize sensitivity first"
    # and the chosen threshold came from the supplied grid
    assert res["threshold"] in grid


def test_threshold_policy_rejects_other_betas():
    from src.evaluation.threshold_policy import select_threshold_macro_f05
    with pytest.raises(ValueError, match="beta=0.5"):
        select_threshold_macro_f05(np.array([0, 1]), np.array([0.2, 0.8]), beta=2.0)


def test_threshold_policy_never_reads_test():
    import inspect
    from src.evaluation import threshold_policy as tp
    src = inspect.getsource(tp)
    assert "test" not in inspect.signature(tp.select_threshold_macro_f05).parameters
    assert "validation" in src


# ================================================================= dataset blocker
def test_blocker_statuses_and_reasons():
    from src.data.dataset_blocker import evaluate_dataset, _load_config
    cfg = _load_config()
    statuses = {}
    for ds in cfg["datasets"]:
        rep = evaluate_dataset(ds, cfg)
        statuses[ds] = rep.status
        assert rep.status in {"PASS", "CONDITIONAL", "BLOCKED", "NOT_AVAILABLE"}
        if rep.status in {"CONDITIONAL", "BLOCKED", "NOT_AVAILABLE"}:
            reasons = getattr(rep, "reasons", None) or [
                c for c in getattr(rep, "checks", []) if getattr(c, "ok", True) is False]
            assert reasons, f"{ds}: {rep.status} without a stated reason"
    assert statuses["REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR"] == "PASS"
    assert statuses["CHILD_PNEUMONIA_MENDELEY"] == "BLOCKED"
    assert statuses["PHYSIONET_PIC"] == "NOT_AVAILABLE"


def test_blocked_dataset_cannot_reach_trainer():
    from src.data.dataset_blocker import DatasetBlockedError, assert_trainable
    with pytest.raises(DatasetBlockedError, match="BLOCKED"):
        assert_trainable("CHILD_PNEUMONIA_MENDELEY")


def test_not_available_dataset_cannot_reach_trainer():
    from src.data.dataset_blocker import DatasetBlockedError, assert_trainable
    with pytest.raises(DatasetBlockedError):
        assert_trainable("PHYSIONET_PIC")


def test_pass_dataset_is_trainable():
    from src.data.dataset_blocker import assert_trainable
    rep = assert_trainable("REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR")
    assert rep.status == "PASS"


def test_blocker_reports_persisted():
    for rel in ("outputs/phase5/dataset_blocking_report.json",
                "outputs/phase5/dataset_blocking_report.csv",
                "docs/PHASE5_DATASET_AUDIT.md"):
        assert (ROOT / rel).exists(), rel


def test_no_mirror_counted_as_separate_dataset():
    """Kaggle/Mendeley mirrors of Kermany are ONE dataset id, not two."""
    cfg = yaml.safe_load((ROOT / "config/dataset_blocking.yaml").read_text(encoding="utf-8"))
    ids = list(cfg["datasets"])
    assert ids.count("KERMANY_PEDIATRIC_PNEUMONIA") == 1
    assert not any("kaggle" in i.lower() for i in ids)


def test_datasets_never_concatenated():
    """Each dataset trains separately — no merged cross-disease table exists."""
    cfg = yaml.safe_load((ROOT / "config/phase5.yaml").read_text(encoding="utf-8"))
    ds = cfg["datasets"]
    assert isinstance(ds, dict)
    targets = {v.get("target") for v in ds.values() if isinstance(v, dict)}
    assert len(targets) >= 2  # separate targets per experiment, never one union target


# ==================================================================== leakage/splits
def test_regensburg_split_reused_and_disjoint():
    from src.data.split import load_splits
    from src.preprocessing.regensburg import ROOT as RROOT
    sp = load_splits(RROOT / "data/interim/splits")
    tr, va, te = set(sp["train"]), set(sp["validation"]), set(sp["test"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    # persisted counts documented in train summary
    if (P5 / "regensburg_appendicitis_tabular/metrics/train_summary.json").exists():
        ts = _load(P5 / "regensburg_appendicitis_tabular/metrics/train_summary.json")
        assert ts["split_source"] == "phase2_persisted_existing_split"
        assert set(ts["counts"]) == {"train", "validation", "test"}


def test_sepsis_group_split_keeps_patients_whole():
    split_path = P5 / "neonatal_sepsis/splits/split.json"
    if not split_path.exists():
        pytest.skip("sepsis split not persisted in this checkout")
    sp = _load(split_path)
    sets = {k: set(v) for k, v in sp.items()}
    assert not (sets["train"] & sets["validation"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["validation"] & sets["test"])
    from src.data.phase5_datasets import load_sepsis
    ds = load_sepsis()
    if ds.available and ds.groups is not None:
        g = ds.groups
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            ga = set(g.loc[g.index.isin(sets[a])].unique())
            gb = set(g.loc[g.index.isin(sets[b])].unique())
            assert not (ga & gb), f"patient crossed {a}/{b}"


def test_make_group_split_no_group_leakage():
    from src.data.phase5_datasets import make_group_split
    rng = np.random.default_rng(2)
    y = pd.Series(rng.integers(0, 2, 200))
    groups = pd.Series([f"g{i // 5}" for i in range(200)])  # 40 groups of 5
    sp = make_group_split(y, groups, seed=7)
    sets = {k: set(v) for k, v in sp.items()}
    assert not (sets["train"] & sets["validation"])
    assert not (sets["train"] & sets["test"])
    assert not (sets["validation"] & sets["test"])
    for part, rows in sets.items():
        crossed = groups.loc[list(rows)].unique()
        for other, orows in sets.items():
            if other <= part:
                continue
            others = groups.loc[list(orows)].unique()
            assert not (set(crossed) & set(others))


# ====================================================================== model zoo
def test_zoo_covers_requested_algorithms_without_linear_regression():
    from src.models.model_registry import load_zoo
    zoo = load_zoo()
    tab = set(zoo["tabular"])
    required = {"logistic_regression", "decision_tree", "gaussian_nb", "knn", "lda",
                "qda", "random_forest", "extra_trees", "gradient_boosting",
                "hist_gradient_boosting", "adaboost", "xgboost", "lightgbm",
                "catboost", "svm_linear", "svm_rbf", "mlp"}
    assert required <= tab, required - tab
    assert "linear_regression" not in tab  # never a disease classifier
    assert "qda" in tab  # present even though it fails on Regensburg at runtime


def test_scaling_only_for_margin_models():
    from src.models.model_registry import load_zoo
    zoo = load_zoo()["tabular"]
    scaled = {n for n, s in zoo.items() if s.get("preprocess") == "onehot_scale"}
    assert {"logistic_regression", "svm_linear", "svm_rbf", "knn", "mlp"} <= scaled
    for tree in ("random_forest", "extra_trees", "decision_tree",
                 "gradient_boosting", "xgboost", "lightgbm"):
        assert tree not in scaled, tree


def test_ensembles_configured_validation_only():
    zoo = yaml.safe_load((ROOT / "config/model_zoo.yaml").read_text(encoding="utf-8"))
    ens = zoo.get("ensembles") or {}
    methods = set(ens.get("methods") or ens)
    assert {"soft_voting_uniform", "validation_weighted_soft_voting",
            "hard_voting", "stacking"} <= methods
    assert "validation" in json.dumps(ens).lower()


def test_availability_covers_enabled_zoo():
    from src.models.model_registry import availability
    av = availability()
    for name in ("logistic_regression", "lightgbm", "xgboost", "catboost", "mlp"):
        assert name in av


# ================================================================ scorecard/nulls
REQUIRED_FIELDS = [
    "dataset", "algorithm", "model_family", "status", "hyperparameters", "threshold",
    "Macro_F0.5", "positive_F0.5", "accuracy", "balanced_accuracy", "precision",
    "sensitivity", "specificity", "NPV", "F1", "F2", "MCC", "AUROC", "AUPRC",
    "Brier", "ECE", "training_time", "inference_time", "calibration_method",
    "sensitivity_floor", "sensitivity_floor_pass", "winner"]


def test_scorecard_required_fields_and_null_policy():
    if not (P5 / "model_scorecard.json").exists():
        pytest.skip("scorecard not generated")
    sc = _load(P5 / "model_scorecard.json")
    rows = sc["rows"]
    assert rows
    for row in rows:
        for f in REQUIRED_FIELDS:
            assert f in row, f"missing field {f}"
    # hard voting has no probabilities — must be null, never 0
    hard = [r for r in rows if r["algorithm"] == "ensemble_hard_voting"]
    assert hard
    for r in hard:
        assert r["AUROC"] is None and r["AUPRC"] is None, "hard voting has no probabilities"
    # unavailable models must not carry fabricated metrics
    for r in rows:
        if r["status"] in {"UNAVAILABLE", "NOT_EXECUTED", "BLOCKED", "FAILED"}:
            assert r["Macro_F0.5"] in (None, "n/a"), r["algorithm"]


def test_scorecard_winner_selected_on_validation_only():
    if not (P5 / "model_scorecard.json").exists():
        pytest.skip("scorecard not generated")
    sc = _load(P5 / "model_scorecard.json")
    assert "validation" in sc["winner_selection"].lower()
    assert "test" not in sc["winner_selection"].lower().split("validation")[0]
    winners = [r for r in sc["rows"] if r["winner"]]
    assert winners, "one winner per executed dataset expected"
    import re
    norm = lambda s: re.sub(r"[^A-Z0-9]", "", s.upper())
    executed = {p.name for p in P5.iterdir()
                if p.is_dir() and (p / "metrics" / "metrics_test.json").exists()}
    for r in winners:
        assert any(norm(r["dataset"])[:10] in norm(d) for d in executed), \
            f"winner dataset {r['dataset']} has no executed artifacts"


def test_scorecard_csv_json_parity():
    if not (P5 / "model_scorecard.json").exists():
        pytest.skip("scorecard not generated")
    import csv
    n_json = _load(P5 / "model_scorecard.json")["n_rows"]
    with (P5 / "model_scorecard.csv").open(encoding="utf-8") as fh:
        n_csv = sum(1 for _ in csv.DictReader(fh))
    assert n_json == n_csv


# ======================================================= statistics + figures
def test_fast_paired_bootstrap_equals_reference():
    from src.evaluation.phase5_metrics import paired_bootstrap_delta_f05
    from src.evaluation.phase5_stats import _paired_delta_fast
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 80)
    pa, pb = rng.random(80), rng.random(80)
    ref = paired_bootstrap_delta_f05(y, pa, pb, threshold_a=0.4, threshold_b=0.6,
                                     n_boot=300, seed=7)
    fast = _paired_delta_fast(y, pa, pb, threshold_a=0.4, threshold_b=0.6,
                              n_boot=300, seed=7)
    assert fast["delta_macro_f0_5"] == pytest.approx(ref["delta_macro_f0_5"], abs=1e-12)
    assert fast["ci95"][0] == pytest.approx(ref["ci95"][0], abs=1e-9)
    assert fast["ci95"][1] == pytest.approx(ref["ci95"][1], abs=1e-9)
    assert fast["p_value"] == pytest.approx(ref["p_value"], abs=1e-9)


def test_statistical_comparison_artifacts():
    if not (P5 / "statistical_comparison.json").exists():
        pytest.skip("statistics not generated")
    d = _load(P5 / "statistical_comparison.json")
    assert "post-hoc" in d["test_labels_used_for"] or "inference" in d["test_labels_used_for"]
    assert d["datasets"]
    for ds, entry in d["datasets"].items():
        bh = entry["benjamini_hochberg"]
        assert bh["n_tests"] >= 1
        for p, q in zip(
                [c["paired_bootstrap_delta_macro_f0_5"]["p_value"]
                 for c in entry["comparisons"]], bh["q_values"]):
            if p is not None and q is not None:
                assert q >= p - 1e-12, "BH q must not be smaller than p"
        assert entry["delong_auroc_ci"], "DeLong CIs expected per model"


def test_figures_manifest_all_targets_generated():
    if not (P5 / "figures_manifest.json").exists():
        pytest.skip("figures manifest not generated")
    m = _load(P5 / "figures_manifest.json")
    assert len(m["figures"]) == 35
    gen = [f for f in m["figures"] if f["status"] == "GENERATED"]
    assert len(gen) == 35, [f["name"] for f in m["figures"] if f["status"] != "GENERATED"]
    for f in gen:
        for p in f["paths"]:
            assert Path(p).exists(), p


def test_figures_manifest_reports_unexecuted_work_honestly():
    m = _load(P5 / "figures_manifest.json")
    joined = " ".join(m["not_executed"]).lower()
    assert "torch" in joined and "not_executed" in joined
    # cross-backbone status is stated explicitly either way — never silent
    cb = m.get("cross_backbone_bias", "")
    assert cb
    has_runs = (P5 / "bias_backbones_summary.json").exists()
    assert cb.lower().startswith("executed") == has_runs


def test_persisted_figures_nonempty():
    pngs = list(P5.rglob("*.png"))
    assert len(pngs) >= 35
    for p in pngs:
        assert p.stat().st_size > 2000, f"suspiciously small figure: {p}"


# ================================================================ calibration/locks
def test_calibration_fitted_on_validation_only():
    for ds_dir in P5.iterdir():
        cal = ds_dir / "calibration" / "calibration.json"
        if not cal.exists():
            continue
        data = _load(cal)
        for name, entry in data.items():
            assert entry.get("fit_on") == "validation", f"{ds_dir.name}/{name}"
        th = _load(ds_dir / "calibration" / "thresholds.json")
        assert th["fitted_on"] == "validation"
        assert th["locked_before_test"] is True


def test_test_partition_locked_once():
    for lock in P5.glob("*/metrics/test_evaluation_lock.json"):
        d = _load(lock)
        assert d.get("n_evaluations", 1) >= 1
        assert d.get("first_evaluation_utc")


# ========================================================== domain packs/agents
def test_domain_packs_exist_for_both_domains():
    from src.agents.domains import get_pack, known_domains
    assert {"appendicitis", "neonatal_sepsis", "pediatric_pneumonia"} <= set(known_domains())
    app = get_pack("appendicitis")
    sepsis = get_pack("neonatal_sepsis")
    assert app.label_space != sepsis.label_space
    assert len(sepsis.candidates) == 3
    assert sepsis.record_aliases.get("Body_Temperature") == "temp_celsius"
    # image-only pack fails closed instead of fabricating evidence
    pn = get_pack("pediatric_pneumonia")
    assert any("cannot assess" in r for r in pn.red_flag_rules)
    assert pn.known_absent


def test_domain_packs_have_no_ground_truth_leakage_fields():
    from src.agents.domains import get_pack
    for did in ("appendicitis", "neonatal_sepsis", "pediatric_pneumonia"):
        pack = get_pack(did)
        text = json.dumps({k: v for k, v in pack.__dict__.items()
                           if k != "candidates"}, default=str).lower()
        assert "ground_truth" not in text and "ground truth" not in text


# ============================================================================= CLI
def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "scripts/run_phase5.py", *args],
                          cwd=str(ROOT), capture_output=True, text=True, timeout=300)


def test_cli_dry_runs_succeed():
    for args in (("audit", "--dry-run"), ("train", "--dry-run", "--dataset",
                                          "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR"),
                 ("evaluate", "--dry-run", "--dataset",
                  "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR"),
                 ("agents", "--dry-run", "--dataset", "NEONATAL_SEPSIS_REGISTRY"),
                 ("bias", "--dry-run", "--dataset", "NEONATAL_SEPSIS_REGISTRY"),
                 ("analyze", "--dry-run")):
        r = _cli(*args)
        assert r.returncode == 0, (args, r.stdout[-500:], r.stderr[-500:])


def test_cli_bias_unimplemented_dataset_reports_not_executed():
    r = _cli("bias", "--dataset", "NEONATAL_SEPSIS_REGISTRY")
    assert r.returncode == 3
    assert "NOT_EXECUTED" in (r.stdout + r.stderr)


def test_cli_train_requires_dataset():
    r = _cli("train")
    assert r.returncode == 2


# ==================================================== cross-backbone bias runs
def test_cross_backbone_bias_summary():
    p = P5 / "bias_backbones_summary.json"
    if not p.exists():
        pytest.skip("cross-backbone bias runs not executed in this checkout")
    s = _load(p)
    assert set(s["backbones"]) == {"logistic_regression", "random_forest", "xgboost",
                                   "lightgbm", "catboost", "ensemble"}
    engine, single = "9_full_with_rag", "3_lightgbm_plus_single_llm"
    for bb, profs in s["backbones"].items():
        eng = (profs[engine]["incorrect_anchor"]["CBR"] or {}).get("value")
        sgl = (profs[single]["incorrect_anchor"]["CBR"] or {}).get("value")
        if eng is not None and sgl is not None:
            assert eng <= sgl, f"{bb}: engine CBR {eng} not below single {sgl}"
    assert "uniform" in s["design"].lower()
    # paired design: identical row counts across backbones
    counts = set()
    for bb in s["backbones"]:
        rows_f = P5 / "bias_backbones" / f"{bb}.json"
        if rows_f.exists():
            meta = _load(rows_f)["meta"]
            assert meta["ground_truth_shown_to_agents"] is False
            assert meta["n_rows"] == 40
            counts.add(meta["n_rows"])
    assert len(counts) == 1


def test_cross_backbone_figure_generated():
    p = P5 / "figures" / "cross_backbone_cbr_bcr.png"
    if not (P5 / "bias_backbones_summary.json").exists():
        pytest.skip("cross-backbone runs not executed")
    assert p.exists() and p.stat().st_size > 6000
    man = _load(P5 / "figures_manifest.json")
    entry = next(f for f in man["figures"] if f["id"] == 32)
    assert entry["status"] == "GENERATED"
    assert any("cross_backbone" in pth for pth in entry["paths"])


# ================================================================ backend API
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from backend.main import app
    return TestClient(app)


def test_api_phase5_status(client):
    j = client.get("/api/phase5/status").json()
    assert j["available"] is True
    assert j["artifacts"]["scorecard"] is True
    if isinstance(j["blocker_summary"], dict):
        assert sum(j["blocker_summary"].values()) == 7  # seven candidate datasets
        assert j["blocker_summary"].get("BLOCKED") == 1
    assert j["datasets_executed"]


def test_api_phase5_datasets_registry(client):
    j = client.get("/api/phase5/datasets").json()
    assert j["available"] is True
    data = j["data"]
    text = json.dumps(data)
    assert "CHILD_PNEUMONIA_MENDELEY" in text
    assert "BLOCKED" in text and "NOT_AVAILABLE" in text


def test_api_phase5_scorecard_nulls_survive_api(client):
    j = client.get("/api/phase5/scorecard").json()
    assert j["available"] is True
    rows = j["data"]["rows"]
    hard = [r for r in rows if r["algorithm"] == "ensemble_hard_voting"]
    assert hard and all(r["AUROC"] is None for r in hard)


def test_api_phase5_algorithms_catalogue(client):
    j = client.get("/api/phase5/algorithms").json()
    algs = j["algorithms"]
    for name in ("logistic_regression", "random_forest", "knn", "svm_rbf",
                 "xgboost", "lightgbm", "catboost", "mlp", "stacking"):
        assert name in algs, name
        assert algs[name]["formula"], name
    assert "e^(" in algs["logistic_regression"]["formula"]  # odds ratio
    assert "Σwm = 1" in algs["validation_weighted_soft_voting"]["formula"]
    assert j["metrics"]["Macro_F0.5"] == "(1/K) Σk F0.5k"
    assert "sensitivity ≥ floor" in j["selection_rule"]
    # formulas are documentation, not fabricated results
    assert "accuracy" in j["metrics"] and j["metrics"]["F0.5"].startswith("1.25")


def test_api_phase5_dataset_metrics(client):
    r = client.get("/api/phase5/dataset/REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR/metrics")
    assert r.status_code == 200
    j = r.json()
    assert j["counts"]["test"]["n"] == 117
    assert j["thresholds"]["fitted_on"] == "validation"
    assert j["split_source"] == "phase2_persisted_existing_split"
    assert "gradient_boosting" in j["test"]["models"]


def test_api_phase5_dataset_metrics_404(client):
    r = client.get("/api/phase5/dataset/NOT_A_DATASET/metrics")
    assert r.status_code == 404
    assert "hint" in r.json()["detail"]


def test_api_phase5_figure_serving(client):
    man = client.get("/api/phase5/figures").json()
    assert man["available"] is True
    entry = next(f for f in man["data"]["figures"] if f["id"] == 1)
    assert entry["status"] == "GENERATED"
    j = client.get(f"/api/phase5/figures/1").json()
    assert j["available"] is True
    png = client.get(j["url"])
    assert png.status_code == 200
    assert png.headers["content-type"] == "image/png"
    assert png.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_api_phase5_figure_traversal_blocked(client):
    r = client.get("/api/phase5/figure-file?path=../secrets.txt")
    assert r.status_code in (404, 400)


def test_api_phase5_unknown_figure_404_with_hint(client):
    r = client.get("/api/phase5/figures/999")
    assert r.status_code == 404
    assert "hint" in r.json()["detail"]
