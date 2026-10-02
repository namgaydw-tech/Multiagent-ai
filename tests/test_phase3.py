"""Phase 3 tests: agent schemas, provider fallback, information isolation,
watchdog safety rules, RAG provenance, orchestrator state machine, audit
persistence/resume, and the bias audit.

All tests run against the real Regensburg research dataset and the persisted
Phase 2 artifacts — no fabricated clinical inputs (synthetic payloads appear
only in pure unit tests of metric bookkeeping, using class labels, not patients).

Research prototype — not a medical device.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agents.arbitrator import anchor_was_followed, diagnosis_family
from src.agents.provider import LLMProvider, ProviderError
from src.agents.schemas import (
    SCHEMA_REGISTRY,
    DataCleanserOutput,
    IndependentDifferentialOutput,
    RetrievedSource,
    WatchdogOutput,
    validate_stage,
)
from src.agents.watchdog import extract as watchdog_extract
from src.bias.audit import compute_bias_audit
from src.models.inference import get_predictor
from src.orchestration import persistence
from src.orchestration.engine import StageError, run_case, run_case_summary
from src.orchestration.state import (
    RESERVED_COLUMNS,
    VisibilityPolicy,
    make_state,
    strip_record,
)

ROOT = Path(__file__).resolve().parents[1]
E2E_ROW = 469          # model true negative, MODERATE uncertainty, discordant debate
ANCHOR_ROW = 746       # model true positive (appendicitis)


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def predictor():
    """Process-wide cached Phase 2 artifacts (dataset + model + calibrator + SHAP)."""
    return get_predictor()


def _fresh_state(predictor, row: int, **kwargs):
    record = predictor.df.loc[row].to_dict()
    state = make_state(predictor.case_id(row), record,
                       config_hash=predictor.config_hash, **kwargs)
    model_output, shap_explain, uncertainty = predictor.bundle(row)
    state.model_output = model_output
    state.shap_explain = shap_explain
    state.uncertainty = uncertainty
    return state


@pytest.fixture(scope="session")
def finished(predictor):
    """One full 8-stage run (in-memory, default condition C, RAG on)."""
    state = _fresh_state(predictor, E2E_ROW, rag_enabled=True, ablation="default")
    return run_case(state, persist=False)


# --------------------------------------------------------------------------- schemas
def test_schema_registry_covers_all_stages(finished):
    assert set(SCHEMA_REGISTRY) == {
        "data_cleanser", "independent_differential", "proponent", "opponent",
        "watchdog", "evidence_retrieval", "arbitrator", "bias_audit"}
    for key, payload in finished.stage_outputs.items():
        assert key in SCHEMA_REGISTRY
        validated = validate_stage(key, payload)
        assert isinstance(validated, SCHEMA_REGISTRY[key])
    with pytest.raises(KeyError):
        validate_stage("not_a_stage", {})


def test_cleanser_schema_guards(finished):
    cleanser = finished.stage_outputs["data_cleanser"]
    assert isinstance(cleanser, DataCleanserOutput)
    assert cleanser.diagnoses_made is False          # hard guard: never diagnoses
    with pytest.raises(ValidationError):
        DataCleanserOutput.model_validate(
            cleanser.model_dump() | {"diagnoses_made": True})


def test_differential_isolation_schema_guard(finished):
    diff = finished.stage_outputs["independent_differential"]
    assert isinstance(diff, IndependentDifferentialOutput)
    assert diff.anchor_seen is False                 # anchor-blind in every condition
    assert diff.candidate_diagnoses                  # non-empty differential
    with pytest.raises(ValidationError):
        IndependentDifferentialOutput.model_validate(
            diff.model_dump() | {"anchor_seen": True})


# --------------------------------------------------------------------------- provider
def test_provider_deterministic_call_record():
    provider = LLMProvider(backend="deterministic")
    result = provider.complete_json(
        stage="3_proponent", system="s", user="u",
        schema=SCHEMA_REGISTRY["proponent"],
        deterministic=lambda: {
            "supported_diagnosis": "appendicitis", "model_probability": 0.5,
            "supporting_evidence": [], "missing_expected_evidence": [],
            "contradictions_acknowledged": [], "confidence": 0.5})
    call = result.call
    assert call.backend == "deterministic"
    assert len(call.prompt_hash) == 64
    assert call.attempts == 1 and call.latency_s >= 0
    assert call.input_tokens == 0 and call.output_tokens == 0
    assert "deterministic" in call.model_version
    assert provider.total_tokens() == 0
    assert provider.calls_for("3_proponent")


def test_provider_llm_failure_falls_back_to_deterministic_builder():
    """No LLM configured → retries fail fast → documented deterministic fallback."""
    provider = LLMProvider(backend="openai_compatible", retries=2, backoff_base_s=0)
    result = provider.complete_json(
        stage="1_data_cleansing", system="s", user="u",
        schema=SCHEMA_REGISTRY["data_cleanser"],
        deterministic=lambda: {"demographics": {}, "vitals": {}, "laboratory_values": {}})
    assert result.call.backend.endswith("+deterministic_fallback")
    assert result.call.fallback_reason and result.call.fallback_reason.startswith(
        "llm_failed_after_2_attempts")
    assert result.call.attempts == 2


def test_provider_no_builder_fails_loudly():
    provider = LLMProvider(backend="openai_compatible", retries=1, backoff_base_s=0)
    with pytest.raises(ProviderError):
        provider.complete_json(stage="3_proponent", system="s", user="u",
                               schema=SCHEMA_REGISTRY["proponent"])


# --------------------------------------------------------------------------- isolation
def test_strip_record_removes_reserved_columns(predictor):
    record = predictor.df.loc[E2E_ROW].to_dict()
    stripped = strip_record(record)
    for col in RESERVED_COLUMNS:
        assert col not in stripped
    assert set(stripped) == set(record) - set(RESERVED_COLUMNS)


def test_visibility_default_policy(finished):
    for stage in (1, 2, 5, 6):
        sl = finished.stage_slice(stage)
        assert "model_output" not in sl and "anchor" not in sl
        assert "ground_truth" not in sl
        assert sl["_visibility"]["sees_prediction"] is False
        assert "model_output" in sl["_withheld"]
    for stage in (3, 4, 7):
        assert finished.stage_slice(stage).get("model_output") is not None
    assert finished.stage_slice(8)["ground_truth"] == finished.ground_truth
    for stage in range(1, 9):
        record = finished.stage_slice(stage)["record"]
        for col in RESERVED_COLUMNS:
            assert col not in record


def test_differential_never_sees_prediction_or_anchor_any_condition(predictor):
    """Stage 2 blindness is an invariant across every named ablation."""
    for ablation in (None, "default", "A_all_see_prediction", "B_only_proponent",
                     "C_hidden_until_differential_complete", "all_see_anchor"):
        policy = VisibilityPolicy.from_ablation(ablation)
        assert policy.stage_prediction[2] is False, ablation
        assert policy.stage_anchor[2] is False, ablation


def test_ablation_visibility_switches(predictor):
    a = VisibilityPolicy.from_ablation("A_all_see_prediction")
    assert a.stage_prediction[4] and a.stage_prediction[5]
    assert a.opponent_stage_a_sees_prediction is True
    b = VisibilityPolicy.from_ablation("B_only_proponent")
    assert b.stage_prediction[4] is False and b.stage_prediction[5] is False
    assert b.stage_prediction[3] is True
    anchor_all = VisibilityPolicy.from_ablation("all_see_anchor")
    assert anchor_all.stage_anchor[5] and anchor_all.stage_anchor[6]
    assert anchor_all.stage_anchor[2] is False


# --------------------------------------------------------------------------- watchdog
def test_watchdog_is_rule_first_and_prediction_blind(predictor):
    record = predictor.df.loc[E2E_ROW].to_dict()
    blind = WatchdogOutput.model_validate(watchdog_extract(record))
    with_pred = WatchdogOutput.model_validate(
        watchdog_extract(record, prediction={"predicted_class": "appendicitis"}))
    # the prediction is logged but never changes the safety assessment
    assert blind.prediction_seen is False
    assert with_pred.prediction_seen is True
    assert with_pred.risk_level == blind.risk_level
    assert ([e.quote for e in with_pred.red_flags_present]
            == [e.quote for e in blind.red_flags_present])


def test_watchdog_fever_threshold_applied(predictor):
    fever_rows = predictor.df.index[predictor.df["Body_Temperature"] >= 38.0]
    afebrile_rows = predictor.df.index[(predictor.df["Body_Temperature"] < 38.0)
                                       & predictor.df["Body_Temperature"].notna()]
    assert len(fever_rows) and len(afebrile_rows)
    fever = WatchdogOutput.model_validate(
        watchdog_extract(predictor.df.loc[int(fever_rows[0])].to_dict()))
    afebrile = WatchdogOutput.model_validate(
        watchdog_extract(predictor.df.loc[int(afebrile_rows[0])].to_dict()))
    fever_text = " ".join(e.quote + e.citation for e in fever.red_flags_present).lower()
    assert "fever" in fever_text
    absent_text = " ".join(afebrile.red_flags_absent).lower()
    assert "fever" in absent_text


def test_watchdog_fail_closed_on_pending_rules(finished):
    wd = finished.stage_outputs["watchdog"]
    refused = " ".join(wd.rules_refused).lower()
    assert "fail closed" in refused              # hypotension / tachycardia / Phoenix
    assert any("hypotension" in r for r in (x.lower() for x in wd.rules_refused))
    cannot = " ".join(wd.cannot_assess_due_to_missing_data).lower()
    assert "rash" in cannot                      # dataset-absent red flag → cannot assess
    assert wd.risk_level in {"LOW", "MODERATE", "HIGH", "CRITICAL"}


def test_watchdog_high_risk_only_with_complication_evidence(predictor):
    crit = predictor.df[(predictor.df["Perforation"] == "yes")
                        | (predictor.df["Peritonitis"] == "generalized")]
    if len(crit) == 0:
        pytest.skip("no complication-positive rows in dataset")
    out = WatchdogOutput.model_validate(
        watchdog_extract(crit.iloc[0].to_dict()))
    assert out.risk_level == "HIGH"
    assert out.red_flags_present               # HIGH requires cited present evidence
    benign = predictor.df[(predictor.df["Perforation"].isin(["no"]))
                          & (predictor.df["Peritonitis"] == "no")
                          & (predictor.df["Body_Temperature"] < 38.0)
                          & predictor.df["Body_Temperature"].notna()]
    if len(benign):
        out2 = WatchdogOutput.model_validate(watchdog_extract(benign.iloc[0].to_dict()))
        assert out2.risk_level in {"LOW", "MODERATE"}   # never alarmist without evidence


# --------------------------------------------------------------------------- RAG
def test_rag_index_provenance_and_retrieval():
    from src.rag.indexer import build_index
    from src.rag.retriever import Retriever
    retriever = Retriever()
    if not retriever.chunks:
        build_index()
        retriever.refresh()
    assert len(retriever.chunks) >= 70
    for chunk in retriever.chunks:                     # mandatory provenance
        assert chunk.doi.strip() or chunk.url.strip() or chunk.citation.strip()
        assert chunk.title.strip()
    hits = retriever.retrieve_for_agents(
        "appendicitis fever white blood cell count", k=4, used_by=["arbitrator"])
    assert 1 <= len(hits) <= 4
    for hit in hits:
        src = RetrievedSource.model_validate(hit)
        assert src.title and 0.0 <= src.score <= 1.0
        assert src.used_by == ["arbitrator"]
    assert any("appendicitis" in h["title"].lower() for h in hits)
    # deterministic retrieval: identical queries give identical results
    again = retriever.retrieve_for_agents(
        "appendicitis fever white blood cell count", k=4, used_by=["arbitrator"])
    assert [(h["title"], h["score"]) for h in again] == \
           [(h["title"], h["score"]) for h in hits]
    # no query, no hits — never fabricated
    assert retriever.retrieve_for_agents("", k=4) == []


def test_stage6_honest_when_rag_disabled(predictor):
    state = _fresh_state(predictor, E2E_ROW, rag_enabled=False)
    run_case(state, persist=False)
    retrieval = state.stage_outputs["evidence_retrieval"]
    assert retrieval.retrieved == []
    assert "disabled" in retrieval.note
    assert retrieval.query                      # query still documented, evidence empty


# --------------------------------------------------------------------------- engine
def test_engine_full_case_information_isolation(finished):
    assert sorted(finished.completed_stages) == list(range(1, 9))
    assert finished.errors == {}
    # stage 2 input: no model, no anchor, isolation metadata present
    s2 = finished.stage_inputs["independent_differential"]
    assert "model_output" not in s2 and "anchor" not in s2
    assert "model_output" in s2["withheld_from_this_stage"]
    assert "anchor" in s2["withheld_from_this_stage"]
    # stage 5 (watchdog) prediction-blind under default condition
    assert finished.stage_inputs["watchdog"]["prediction_passed"] is False
    # stage 8 sees ground truth
    assert finished.stage_inputs["bias_audit"]["ground_truth"] == finished.ground_truth
    summary = run_case_summary(finished)
    assert summary["primary_working_diagnosis"]
    assert 0.0 <= summary["multi_agent_confidence"] <= 1.0


def test_engine_ablation_a_exposes_prediction_to_watchdog(predictor):
    state = _fresh_state(predictor, E2E_ROW, rag_enabled=False,
                         ablation="A_all_see_prediction")
    run_case(state, persist=False)
    assert state.stage_inputs["watchdog"]["prediction_passed"] is True
    assert state.stage_outputs["watchdog"].prediction_seen is True
    # differential still blind even under A
    assert "model_output" not in state.stage_inputs["independent_differential"]


def test_engine_fails_loudly_on_isolation_violation(predictor):
    state = _fresh_state(predictor, E2E_ROW, rag_enabled=False)
    state.policy.stage_prediction[2] = True   # corrupt the policy deliberately
    with pytest.raises(StageError) as excinfo:
        run_case(state, persist=False)
    assert "independent_differential" in state.errors
    assert "stage 2" in str(excinfo.value)


def test_anchor_injection_flow(predictor):
    """Anchor is revealed to the debate (stage 4/7/8) but never to stages 1/2/5."""
    state = _fresh_state(predictor, ANCHOR_ROW, rag_enabled=False)
    state.anchor = {"value": "Gastroenteritis", "confidence": 0.9,
                    "source": "unit_test", "method": "synthetic_test_only"}
    run_case(state, persist=False)
    s4 = state.stage_inputs["opponent"]
    assert s4["leading_hypothesis"] == "Gastroenteritis"
    assert s4["leading_hypothesis_source"] == "anchor"
    for stage in (1, 2, 5):
        assert "anchor" not in state.stage_inputs[
            {1: "data_cleanser", 2: "independent_differential", 5: "watchdog"}[stage]]
    assert state.stage_outputs["arbitrator"].anchor_present is True
    audit = state.stage_outputs["bias_audit"]
    assert audit.anchor_present is True
    assert audit.anchor_correct is False      # gastroenteritis vs true appendicitis
    # differential output still claims anchor blindness (schema guard held)
    assert state.stage_outputs["independent_differential"].anchor_seen is False


# --------------------------------------------------------------------------- persistence
def test_persistence_and_resume(predictor, tmp_path):
    state1 = _fresh_state(predictor, 217, rag_enabled=False)
    run_case(state1, persist=True, out_root=tmp_path)

    directory = persistence.case_dir(state1.case_id, root=tmp_path)
    assert (directory / "audit.json").exists()
    for n in range(1, 9):
        for kind in ("input", "output", "meta"):
            assert persistence.stage_paths(directory, n)[kind].exists(), (n, kind)

    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    assert audit["case_id"] == state1.case_id
    assert audit["seed"] == 20261002
    assert audit["config_hash"] == predictor.config_hash
    assert audit["provider_backend"] == "deterministic"
    assert audit["totals"]["provider_calls"] == 8
    assert sorted(audit["completed_stages"]) == list(range(1, 9))
    assert set(audit["visibility_policy"]) == {str(n) for n in range(1, 9)}
    for entry in audit["stages"]:
        assert entry["status"] == "ok"
        assert len(entry["prompt_hash"]) == 64
        assert entry["model_version"]
        assert entry["latency_s"] >= 0
        assert entry["input_tokens"] == 0 and entry["output_tokens"] == 0
        assert entry["schema_name"]

    # resume: every stage restored from disk, nothing re-executed
    state2 = _fresh_state(predictor, 217, rag_enabled=False)
    provider = LLMProvider(backend="deterministic")
    run_case(state2, provider=provider, resume=True, persist=True, out_root=tmp_path)
    assert sorted(state2.completed_stages) == list(range(1, 9))
    assert len(state2.stage_outputs) == 8
    assert provider.log == []                 # no stage was re-run
    assert state2.stage_outputs["arbitrator"].multi_agent_confidence == \
        state1.stage_outputs["arbitrator"].multi_agent_confidence


def test_persisted_stage_payloads_match_outputs(predictor, tmp_path):
    state = _fresh_state(predictor, 393, rag_enabled=False)
    run_case(state, persist=True, out_root=tmp_path)
    directory = persistence.case_dir(state.case_id, root=tmp_path)
    for stage_no, (key, _) in sorted(persistence.STAGE_KEYS.items()):
        paths = persistence.stage_paths(directory, stage_no)
        disk = json.loads(paths["output"].read_text(encoding="utf-8"))
        assert disk == state.stage_outputs[key].model_dump()
        snap = json.loads(paths["input"].read_text(encoding="utf-8"))
        assert "visibility" in snap and "withheld_from_this_stage" in snap


# --------------------------------------------------------------------------- bias audit
def test_diagnosis_family_negation():
    assert diagnosis_family("no appendicitis") == "no appendicitis"
    assert diagnosis_family("appendicitis") == "appendicitis"
    assert diagnosis_family("acute appendicitis") == "appendicitis"
    assert diagnosis_family("Appendizitis") == "appendicitis"
    assert diagnosis_family("keine Appendizitis") == "no appendicitis"
    assert diagnosis_family("Gastroenteritis") == "gastroenteritis"
    assert anchor_was_followed("Appendizitis", "Acute appendicitis") is True
    assert anchor_was_followed("Gastroenteritis", "Acute appendicitis") is False
    assert anchor_was_followed("no appendicitis", "acute appendicitis") is False


def test_compute_bias_audit_bookkeeping():
    model_pos = {"predicted_class": "appendicitis", "calibrated_probability": 0.9,
                 "class_probabilities": {"appendicitis": 0.9, "no appendicitis": 0.1}}
    # anchor followed (anchor wrong, final wrong-way) — judged vs ground truth
    a1 = compute_bias_audit(
        "c1", {"value": "Gastroenteritis", "confidence": 0.9,
               "source": "unit_test", "method": "test"},
        "appendicitis", model_pos,
        opponent={"contradictory_evidence": []},
        arbitrator={"primary_working_diagnosis": "appendicitis",
                    "multi_agent_confidence": 0.7},
        proponent={})
    assert a1["anchor_present"] is True
    assert a1["anchor_correct"] is False
    assert a1["anchor_followed"] is False     # final=appendicitis != anchor family

    # harmful flip: correct model answer overturned by the debate
    a2 = compute_bias_audit(
        "c2", None, "appendicitis", model_pos,
        opponent={"contradictory_evidence": [{"quote": "CRP=0.0",
                                              "citation": "column=CRP"}]},
        arbitrator={"primary_working_diagnosis": "no appendicitis",
                    "multi_agent_confidence": 0.6},
        proponent={})
    assert a2["diagnosis_changed"] is True
    assert a2["change_harmful"] is True and a2["change_beneficial"] is False
    assert a2["contradictory_evidence_introduced"] is True
    assert a2["model_vs_final_disagreement"] is True

    # beneficial flip: debate corrects a wrong model answer
    a3 = compute_bias_audit(
        "c3", None, "appendicitis",
        {"predicted_class": "no appendicitis", "calibrated_probability": 0.4},
        opponent={"contradictory_evidence": []},
        arbitrator={"primary_working_diagnosis": "appendicitis",
                    "multi_agent_confidence": 0.7},
        proponent={})
    assert a3["diagnosis_changed"] is True
    assert a3["change_beneficial"] is True and a3["change_harmful"] is False


def test_e2e_bias_audit_matches_known_run(finished):
    """The session e2e case (row 469) is a documented harmful flip."""
    audit = finished.stage_outputs["bias_audit"]
    assert audit.diagnosis_before_debate == "no appendicitis"
    assert audit.diagnosis_after_debate == "Acute appendicitis"
    assert audit.diagnosis_changed is True
    assert audit.change_harmful is True
    assert audit.change_beneficial is False
    assert audit.model_vs_final_disagreement is True


# --------------------------------------------------------------------------- Layer A
def test_inference_reproduces_phase2_predictions(predictor):
    import pandas as pd
    csv_path = ROOT / "outputs/predictions/test_predictions.csv"
    if not csv_path.exists():
        pytest.skip("Phase 2 prediction file not present")
    csv = pd.read_csv(csv_path)
    row = csv[csv.row_index == E2E_ROW]
    if row.empty:
        pytest.skip(f"row {E2E_ROW} not in test partition file")
    expected = row.iloc[0]
    pred = predictor.predict(E2E_ROW)
    assert abs(pred["class_probabilities"]["appendicitis"] - expected.p_raw) < 1e-5
    assert pred["predicted_class"] == expected.pred_class
    assert predictor.threshold == float(expected.threshold) == 0.34
    assert pred["calibrated_probability"] is not None   # raw vs calibrated distinguishable
    assert pred["clinical_domain"] == "pediatric_appendicitis"


def test_uncertainty_is_not_probability(finished):
    unc = finished.uncertainty
    assert {"confidence", "predictive_entropy", "margin",
            "uncertainty_level"} <= set(unc)
    assert unc["uncertainty_level"] in {"LOW", "MODERATE", "HIGH"}
    assert 0.0 <= unc["predictive_entropy"] <= math.log(2) + 1e-9
    assert abs(unc["confidence"] - (1.0 - unc["predictive_entropy"])) < 1e-3
    assert "not clinical certainty" in unc["note"]


def test_shap_explanation_wording_and_content(finished):
    shap_explain = finished.shap_explain
    assert "contributed to this model prediction" in shap_explain["wording_rule"]
    assert shap_explain["top_contributors"]
    for contrib in shap_explain["top_contributors"][:3]:
        assert contrib["feature"] and contrib["direction"] in {"positive", "negative"}
        assert "contributed to this model prediction" in contrib["wording"]
    # SHAP feeds the proponent with the mandated non-causal wording
    proponent = finished.stage_outputs["proponent"]
    model_refs = [e.relevance for e in proponent.supporting_evidence
                  if e.kind == "model"]
    assert model_refs
    assert any("to this model prediction" in r for r in model_refs)
    assert not any("caused the diagnosis" in r.lower() for r in model_refs)
