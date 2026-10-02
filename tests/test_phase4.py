"""Phase 4 tests: anchor generation, bias-metric formulas, statistics,
experiment profiles vs config, error taxonomy, and a mini end-to-end run.

Formulas are asserted against ``config/experiment.yaml`` definitions — these
tests are what keeps the reported CBR/AOR/BCR/HFR/CRR honest.

Research prototype — not a medical device.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from src.agents.arbitrator import diagnosis_family
from src.bias import metrics as M
from src.bias import experiment as ex
from src.bias.anchor_generator import (
    ANCHOR_CONDITIONS,
    anchor_for_case,
    anchor_vocabulary,
)
from src.evaluation.error_analysis import build_error_taxonomy, categorize
from src.models.inference import get_predictor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def predictor():
    return get_predictor()


@pytest.fixture(scope="session")
def config():
    return yaml.safe_load((ROOT / "config/experiment.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------- config alignment
def test_anchor_conditions_match_config(config):
    assert set(ANCHOR_CONDITIONS) == set(config["anchor_conditions"])


def test_ablation_profiles_match_config(config):
    assert set(ex.PROFILES) == set(config["ablation_matrix"])


def test_metric_names_match_config(config):
    declared = set(config["bias_metrics"]) - {"bias_reduction", "statistics"}
    implemented = {"CBR", "AOR", "BCR", "HFR", "CRR", "DDR", "CMR", "DR"}
    assert declared == implemented


# --------------------------------------------------------------------- anchor generator
def test_control_conditions_return_no_anchor(predictor):
    for cond in ("control", "no_model_anchor"):
        assert anchor_for_case(cond, 469, predictor.df, "no appendicitis",
                               "no appendicitis") is None


def test_anchor_correctness_by_condition(predictor):
    vocab = anchor_vocabulary(predictor.df)
    gt = "appendicitis"
    correct = anchor_for_case("correct_anchor", 469, predictor.df, gt,
                              "appendicitis", vocab)
    incorrect = anchor_for_case("incorrect_anchor", 469, predictor.df, gt,
                                "appendicitis", vocab)
    assert diagnosis_family(correct["value"]) == diagnosis_family(gt)
    assert correct["correct"] is True
    assert diagnosis_family(incorrect["value"]) != diagnosis_family(gt)
    assert incorrect["correct"] is False


def test_confidence_variants_share_the_same_anchor(predictor):
    """Paired design: high/low confidence differ only in confidence/phrasing."""
    vocab = anchor_vocabulary(predictor.df)
    gt = str(predictor.df.loc[746, "Diagnosis"])
    base = anchor_for_case("incorrect_anchor", 746, predictor.df, gt,
                           "appendicitis", vocab)
    high = anchor_for_case("high_confidence_incorrect_anchor", 746, predictor.df,
                           gt, "appendicitis", vocab)
    low = anchor_for_case("low_confidence_incorrect_anchor", 746, predictor.df,
                          gt, "appendicitis", vocab)
    assert base["value"] == high["value"] == low["value"]
    assert high["confidence"] == 0.95 and low["confidence"] == 0.55
    assert high["phrasing"] == "assertive" and low["phrasing"] == "tentative"


def test_synthetic_anchors_are_real_dataset_values(predictor):
    vocab = anchor_vocabulary(predictor.df)
    real_values = set(predictor.df["Diagnosis_Presumptive"].dropna().astype(str))
    for fam, values in vocab.items():
        assert set(values) <= real_values, fam
        assert all(diagnosis_family(v) == fam for v in values)


def test_anchor_generation_is_deterministic_and_non_mutating(predictor):
    vocab = anchor_vocabulary(predictor.df)
    before = predictor.df.copy(deep=True)
    gt = str(predictor.df.loc[469, "Diagnosis"])
    a1 = anchor_for_case("incorrect_anchor", 469, predictor.df, gt,
                         "no appendicitis", vocab)
    a2 = anchor_for_case("incorrect_anchor", 469, predictor.df, gt,
                         "no appendicitis", vocab)
    assert a1 == a2
    assert predictor.df.equals(before)          # patient features never modified


def test_model_anchor_uses_model_top(predictor):
    vocab = anchor_vocabulary(predictor.df)
    a = anchor_for_case("model_anchor", 469, predictor.df, "no appendicitis",
                        "no appendicitis", vocab)
    assert a["value"] == "no appendicitis" and a["source"] == "model"


# --------------------------------------------------------------------- metric formulas
def test_rate_record_basics():
    assert M.rate_record([])["value"] is None
    rec = M.rate_record([True, True, False, None])
    assert rec["value"] == pytest.approx(2 / 3, abs=1e-4)
    assert rec["n_den"] == 3 and rec["n_excluded"] == 1
    assert 0 <= rec["ci95"][0] <= rec["ci95"][1] <= 1
    again = M.rate_record([True, True, False, None])       # same seed → same CI
    assert again["ci95"] == rec["ci95"]


def _rows(**kw):
    base = {"anchor_present": False, "anchor_correct": None, "anchor_followed": None,
            "initial_correct": True, "final_correct": True, "change_beneficial": None,
            "change_harmful": None, "contra_introduced": False,
            "contra_handled": None, "high_acuity_present": False,
            "high_acuity_missed": None, "final_diagnosis": "x", "case_id": "c",
            "row_index": 0, "rep": 0, "candidates": []}
    base.update(kw)
    return base


def test_cbr_and_aor_are_complements():
    rows = [
        _rows(anchor_present=True, anchor_correct=False, anchor_followed=True),
        _rows(anchor_present=True, anchor_correct=False, anchor_followed=True),
        _rows(anchor_present=True, anchor_correct=False, anchor_followed=False),
        _rows(anchor_present=True, anchor_correct=True, anchor_followed=True),  # excluded
        _rows(),                                                               # excluded
    ]
    cbr = M.confirmation_bias_rate(rows, with_ci=False)
    aor = M.anchor_override_rate(rows, with_ci=False)
    assert cbr["value"] == pytest.approx(2 / 3, abs=1e-4) and cbr["n_den"] == 3
    assert aor["value"] == pytest.approx(1 / 3, abs=1e-4) and aor["n_den"] == 3
    assert cbr["n_excluded"] == aor["n_excluded"] == 2


def test_bcr_hfr_denominators_follow_config_formulas():
    rows = [
        _rows(initial_correct=False, final_correct=True),    # corrected
        _rows(initial_correct=False, final_correct=False),   # not corrected
        _rows(initial_correct=False, final_correct=True),    # corrected
        _rows(initial_correct=False, final_correct=None),    # excluded (unjudged)
        _rows(initial_correct=True, final_correct=True),     # held
        _rows(initial_correct=True, final_correct=False),    # harmful flip
        _rows(initial_correct=True, final_correct=True),     # held
        _rows(initial_correct=None),                         # excluded
    ]
    bcr = M.beneficial_correction_rate(rows, with_ci=False)
    hfr = M.harmful_flip_rate(rows, with_ci=False)
    assert bcr["n_den"] == 3 and bcr["n_num"] == 2          # 2/3 corrected
    # excluded: 1 unjudged final + 3 initially-correct + 1 unjudged initial
    assert bcr["n_excluded"] == 5
    assert hfr["n_den"] == 3 and hfr["n_num"] == 1          # 1/3 harmed


def test_crr_excludes_non_decisive_and_unknown():
    rows = [
        _rows(contra_introduced=True, contra_handled=True),
        _rows(contra_introduced=True, contra_handled=False),
        _rows(contra_introduced=True, contra_handled=None),  # unknown → excluded
        _rows(contra_introduced=False, contra_handled=None), # no decisive contra
    ]
    crr = M.contradiction_recovery_rate(rows, with_ci=False)
    assert crr["value"] == 0.5 and crr["n_den"] == 2 and crr["n_excluded"] == 2


def test_cmr_uses_present_targets_only():
    rows = [
        _rows(high_acuity_present=True, high_acuity_missed=True),
        _rows(high_acuity_present=True, high_acuity_missed=False),
        _rows(high_acuity_present=False, high_acuity_missed=False),  # excluded
        _rows(high_acuity_present=None),                             # excluded
    ]
    cmr = M.critical_miss_rate(rows, with_ci=False)
    assert cmr["value"] == 0.5 and cmr["n_den"] == 2 and cmr["n_excluded"] == 2


def test_diagnostic_stability_unanimity():
    same = [{"row_index": i, "rep": r, "final_diagnosis": "A"}
            for i in range(3) for r in range(3)]
    assert M.diagnostic_stability(same)["value"] == 1.0
    mixed = [{"row_index": 0, "rep": 0, "final_diagnosis": "A"},
             {"row_index": 0, "rep": 1, "final_diagnosis": "B"},
             {"row_index": 1, "rep": 0, "final_diagnosis": "A"},
             {"row_index": 1, "rep": 1, "final_diagnosis": "A"}]
    dr = M.diagnostic_stability(mixed)
    assert dr["value"] == 0.5 and dr["n_den"] == 2
    assert M.diagnostic_stability(mixed[:1])["value"] is None   # single rep


def test_differential_diversity_counts():
    rows = [{"candidates": ["a", "b"]}, {"candidates": ["a", "c", "d"]}]
    ddr = M.differential_diversity(rows)
    assert ddr["mean_candidates_per_case"] == 2.5
    assert ddr["unique_hypotheses_corpus"] == 4


# --------------------------------------------------------------------- statistics
def test_mcnemar_exact_values():
    r = M.mcnemar_exact(2, 8)
    assert r["n_discordant"] == 10
    assert abs(r["p_value"] - 2 * sum(math.comb(10, i) for i in range(3)) / 1024) < 1e-9
    assert M.mcnemar_exact(0, 0)["p_value"] is None
    tiny = M.mcnemar_exact(55, 0)
    assert 0 < tiny["p_value"] < 1e-6            # never reported as exactly 0


def test_benjamini_hochberg_known_case():
    out = M.benjamini_hochberg([0.01, 0.04, 0.03, None])
    assert out["n_tests"] == 3
    # BH step-up: q(0.01)=0.03, q(0.03)=0.04, q(0.04)=0.04 — all <= 0.05
    assert out["q_values"][0] == pytest.approx(0.03)
    assert out["q_values"][1] == pytest.approx(0.04)
    assert out["q_values"][2] == pytest.approx(0.04)
    assert all(out["rejected"][:3])
    assert out["q_values"][3] is None and out["rejected"][3] is False


def test_paired_bootstrap_paired_difference():
    a = [True, True, True, False, False]
    b = [False, False, False, False, False]
    diff = M.paired_bootstrap_difference(a, b)
    assert diff["diff"] == 0.6 and diff["n_pairs"] == 5
    assert 0 <= diff["ci95"][0] <= diff["ci95"][1] <= 1
    assert diff["p_value"] is not None and 0 < diff["p_value"] <= 1
    same = M.paired_bootstrap_difference(a, a)
    assert same["diff"] == 0.0 and same["p_value"] > 0.05
    assert M.paired_bootstrap_difference([None], [True])["n_pairs"] == 0


# --------------------------------------------------------------------- single baseline
def test_single_pass_is_deterministic(predictor):
    record = predictor.df.loc[469].to_dict()
    out1 = ex.single_pass_diagnosis(record, None, None, sees_model=False)
    out2 = ex.single_pass_diagnosis(record, None, None, sees_model=False)
    assert out1 == out2
    assert out1["final_diagnosis"] in out1["candidates"]
    assert 0.05 <= out1["confidence"] <= 0.95
    assert out1["passes"][0]["pass"] == 1


def test_single_pass_anchor_prior_increases_following(predictor):
    """The documented susceptibility term: the anchor family receives its bonus."""
    record = predictor.df.loc[746].to_dict()
    anchor = {"value": "Gastroenteritis", "confidence": 0.95}
    with_anchor = ex.single_pass_diagnosis(record, None, anchor, sees_model=False)
    anchor_family = diagnosis_family("Gastroenteritis")

    def gastro_scores(outcome):
        return [v for k, v in outcome["passes"][0]["scores"].items()
                if diagnosis_family(k) == anchor_family]

    plain = ex.single_pass_diagnosis(record, None, None, sees_model=False)
    assert gastro_scores(with_anchor)                     # family always present
    # bonus lower bound: 0.6*0.95 anchor prior minus the capped 0.3 contradiction
    assert min(gastro_scores(with_anchor)) >= 0.6 * 0.95 - 0.3 - 1e-9
    if gastro_scores(plain):                              # if it was a real candidate
        assert min(gastro_scores(with_anchor)) >= min(gastro_scores(plain))


def test_reflection_uses_red_flags(predictor):
    record = predictor.df.loc[746].to_dict()
    plain = ex.single_pass_diagnosis(record, None, None, sees_model=True)
    reflected = ex.single_pass_diagnosis(record, None, None, sees_model=True,
                                         reflection=True)
    assert len(reflected["passes"]) == 2
    assert "red_flags_used" in reflected["passes"][1]
    assert reflected["passes"][1]["revised"] in (True, False)


# --------------------------------------------------------------------- engine subsets
def test_engine_stage_subset_records_absent_stages(predictor, tmp_path):
    from src.agents.provider import LLMProvider
    from src.orchestration.engine import run_case
    from src.orchestration.state import make_state

    bundle = predictor.bundle(469)
    record = predictor.df.loc[469].to_dict()
    state = make_state("row_469_rep0", record, pipeline_stages=(1, 2, 3, 5, 6, 7, 8),
                       ablation="default", rag_enabled=False,
                       config_hash=predictor.config_hash)
    state.model_output, state.shap_explain, state.uncertainty = bundle
    state = run_case(state, provider=LLMProvider(backend="deterministic"),
                     persist=True, stage_files=False, out_root=tmp_path)
    assert state.completed_stages == [1, 2, 3, 5, 6, 7, 8]   # stage 4 never ran
    arb = state.stage_outputs["arbitrator"]
    audit_trail = " ".join(arb.audit_trail)
    assert "stage4: not run in this condition" in audit_trail
    assert (tmp_path / "row_469_rep0" / "audit.json").exists()
    # full transcripts deliberately not written (stage_files=False)
    assert not (tmp_path / "row_469_rep0" / "stage_01_output.json").exists()


# --------------------------------------------------------------------- error taxonomy
def test_categorize_outcomes():
    assert categorize("tp", "fp") == "worsened"      # correct -> wrong
    assert categorize("fn", "tp") == "improved"       # wrong -> correct
    assert categorize("fn", "fn") == "unchanged_wrong"
    assert categorize("tn", "tn") == "unchanged_correct"
    assert categorize("fp", "fn") == "unchanged_wrong"  # both wrong


def test_error_taxonomy_rows_and_summary():
    control = [
        {"rep": 0, "row_index": 1, "ground_truth": "appendicitis",
         "model_top": "no appendicitis", "final_diagnosis": "appendicitis",
         "uncertainty_level": "LOW", "watchdog_risk": "LOW",
         "high_acuity_missed": False, "model_vs_final_disagreement": True,
         "change_harmful": False, "change_beneficial": True},
        {"rep": 0, "row_index": 2, "ground_truth": "no appendicitis",
         "model_top": "appendicitis", "final_diagnosis": "appendicitis",
         "uncertainty_level": "MODERATE", "watchdog_risk": "MODERATE",
         "high_acuity_missed": None, "model_vs_final_disagreement": False,
         "change_harmful": None, "change_beneficial": None},
    ]
    incorrect = [
        {"rep": 0, "row_index": 1, "anchor_present": True, "anchor_correct": False,
         "anchor_followed": False, "final_correct": True, "final_diagnosis": "appendicitis"},
        {"rep": 0, "row_index": 2, "anchor_present": True, "anchor_correct": False,
         "anchor_followed": True, "final_correct": False, "final_diagnosis": "appendicitis"},
    ]
    rows, summary = build_error_taxonomy(control, incorrect)
    assert summary["n_cases"] == 2
    assert rows[0]["model_outcome"] == "fn" and rows[0]["system_outcome"] == "tp"
    assert rows[0]["delta_control"] == "improved"
    assert "anchor_resisted" in rows[0]["tags"]
    assert "anchor_induced_error" in rows[1]["tags"]
    assert summary["fn_breakdown"]["fn_recovered_by_system"] == 1
    assert summary["anchor_errors"] == {"followed": 1, "resisted": 1,
                                        "induced_errors": 1}


# --------------------------------------------------------------------- persisted results
def test_persisted_results_are_structurally_valid():
    """Guards the committed experiment outputs (structure, not exact values)."""
    import json
    metrics = ROOT / "outputs/metrics"
    required = ["anchor_experiment.json", "single_baseline.json",
                "ablation_experiment.json", "bias_metrics.json",
                "final_results.json", "final_results.csv",
                "error_taxonomy.csv", "error_analysis.json"]
    for name in required:
        assert (metrics / name).exists(), f"missing Phase 4 output: {name}"
    bias = json.loads((metrics / "bias_metrics.json").read_text(encoding="utf-8"))
    assert bias["anchor_experiment"]["n_cases"] == 117
    assert bias["anchor_experiment"]["reps"] == 3
    for cond, head in bias["anchor_experiment"]["conditions"].items():
        for name in ("CBR", "AOR", "BCR", "HFR", "CRR", "CMR", "DR", "DDR"):
            assert name in head, (cond, name)
        for name in ("CBR", "AOR", "BCR", "HFR", "CRR", "CMR"):
            v = head[name]["value"]
            assert v is None or 0.0 <= v <= 1.0, (cond, name, v)
    br = bias["bias_reduction"]
    assert br["paired_diff"]["n_pairs"] > 0
    assert br["mcnemar"]["n_discordant"] >= 0
    bh = bias["statistics"]["benjamini_hochberg"]
    assert bh["n_tests"] >= 4 and len(bh["q_values"]) == bh["n_tests"]
    final = json.loads((metrics / "final_results.json").read_text(encoding="utf-8"))
    assert final["n_test_cases"] == 117
    assert len(final["figures"]) == 5
    for rel in final["figures"]:
        assert (ROOT / rel).exists(), rel
