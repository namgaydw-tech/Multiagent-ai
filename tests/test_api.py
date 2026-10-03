"""API → frontend integration tests (FastAPI TestClient).

These cover the contract the React frontend consumes: availability states for
missing metrics, real model predictions, a real 8-stage agent run with
isolation metadata, the quick bias experiment, audit browsing, figure serving
and structured error responses.

Research prototype — not a medical device.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# --------------------------------------------------------------------- meta
def test_root_and_repro(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["seed"] == 20261002
    assert "not a medical device" in r.json()["disclaimer"].lower()
    repro = client.get("/api/repro").json()
    assert any("run_phase2" in cmd for cmd in repro["train"])
    assert any("run_phase4" in cmd for cmd in repro["experiments"])
    assert repro["frontend"][0].startswith("cd frontend")


def test_health_reports_artifacts(client):
    j = client.get("/api/health").json()
    assert j["status"] == "ok"
    assert j["model"]["available"] is True
    assert j["model"]["threshold"] == 0.34
    assert j["model"]["config_hash"]
    assert sum(j["metrics_files"].values()) == len(j["metrics_files"])
    assert all(j["experiments_available"].values())
    assert j["llm_backend"] == "deterministic"
    assert isinstance(j["rag_chunks"], int) and j["rag_chunks"] >= 70


# --------------------------------------------------------------------- metrics
def test_metrics_endpoints_available_and_shaped(client):
    summary = client.get("/api/metrics/summary").json()
    assert summary["available"] is True
    assert "lightgbm_appendicitis" in summary["data"]["models"]
    assert summary["data"]["models"]["lightgbm_appendicitis"]["test_metrics"]["auroc"] > 0.9

    bias = client.get("/api/metrics/bias").json()
    assert bias["available"] is True
    cond = bias["data"]["anchor_experiment"]["conditions"]
    assert set(cond) >= {"control", "incorrect_anchor"}
    assert bias["data"]["bias_reduction"]["paired_diff"]["n_pairs"] == 117

    final = client.get("/api/metrics/final-results").json()
    assert final["available"] and final["data"]["n_test_cases"] == 117
    assert len(final["data"]["figures"]) == 5

    errs = client.get("/api/metrics/error-analysis").json()
    assert errs["available"] and len(errs["rows"]) == 117
    assert errs["data"]["n_cases"] == 117

    exp = client.get("/api/metrics/experiments").json()
    assert exp["available"] and len(exp["ablations"]) == 11


def test_metrics_missing_file_returns_availability_state(client, tmp_path, monkeypatch):
    """A missing artifact is an availability state with a hint, not fake data."""
    from backend.api import metrics as metrics_api
    monkeypatch.setattr(metrics_api, "METRICS_DIR", tmp_path)
    j = client.get("/api/metrics/summary").json()
    assert j["available"] is False
    assert "run_phase2" in j["hint"]
    assert "data" not in j


def test_figure_serving_and_missing(client):
    r = client.get("/api/metrics/figures/fig_bias_reduction.png")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert len(r.content) > 1000
    r = client.get("/api/metrics/figures/does_not_exist.png")
    assert r.status_code == 503
    assert "analyze" in r.json()["hint"]
    r = client.get("/api/metrics/figures/evil%2Epng")
    assert r.status_code in (404, 503)


# --------------------------------------------------------------------- cases + predict
def test_case_list_matches_phase2_predictions(client):
    j = client.get("/api/cases?limit=5").json()
    assert j["available"] and j["n"] == 5
    row = j["rows"][0]
    assert {"row_index", "y_true", "p_raw", "p_calibrated", "pred_class",
            "uncertainty_level"} <= set(row)
    csv = ROOT / "outputs/predictions/test_predictions.csv"
    expected = next(r for r in csv.read_text(encoding="utf-8").splitlines()[1:]
                    if r.startswith(f"{row['row_index']},"))
    assert f"{row['p_raw']:.6f}" in expected


def test_predict_real_case_and_error_paths(client):
    r = client.post("/api/predict", json={"row_index": 469})
    assert r.status_code == 200
    body = r.json()
    assert body["model_output"]["predicted_class"] == "no appendicitis"
    assert abs(body["model_output"]["class_probabilities"]["appendicitis"]
               - 0.269645) < 1e-5
    assert body["model_output"]["calibrated_probability"] is not None
    assert body["uncertainty"]["uncertainty_level"] in {"LOW", "MODERATE", "HIGH"}
    assert body["shap_explain"]["top_contributors"]
    assert "contributed to this model prediction" in body["shap_explain"]["wording_rule"]

    r = client.post("/api/predict", json={"row_index": 999999})
    assert r.status_code == 404
    assert "hint" in r.json()["detail"]

    r = client.post("/api/predict", json={"row_index": -5})
    assert r.status_code == 422          # pydantic validation


def test_case_record_strips_reserved_columns(client):
    j = client.get("/api/cases/469/record").json()
    assert "Diagnosis_Presumptive" not in j["fields"]
    assert "US_Number" not in j["fields"]
    assert j["ground_truth"] in {"appendicitis", "no appendicitis"}


# --------------------------------------------------------------------- agents
def test_agent_run_end_to_end_with_isolation(client):
    r = client.post("/api/agents/run", json={"row_index": 469, "rag_enabled": False})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [s["stage"] for s in body["stages"]] == list(range(1, 9))
    assert body["summary"]["errors"] == {}
    assert body["summary"]["primary_working_diagnosis"]
    assert 0.0 <= body["summary"]["multi_agent_confidence"] <= 1.0

    stage2 = next(s for s in body["stages"] if s["stage"] == 2)
    assert stage2["visibility"]["sees_prediction"] is False
    assert stage2["visibility"]["sees_anchor"] is False
    assert "model_output" in stage2["withheld"]
    assert "anchor" in stage2["withheld"]
    stage8 = next(s for s in body["stages"] if s["stage"] == 8)
    assert stage8["visibility"]["sees_ground_truth"] is True

    # the audit transcript was really persisted
    audit_path = Path(body["audit_path"]) / "audit.json"
    assert audit_path.exists()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["completed_stages"] == list(range(1, 9))
    assert all(len(s["prompt_hash"]) == 64 for s in audit["stages"])


def test_agent_run_with_anchor_and_validation_errors(client):
    r = client.post("/api/agents/run",
                    json={"row_index": 469, "anchor_condition": "incorrect_anchor",
                          "rag_enabled": True})
    assert r.status_code == 200
    body = r.json()
    assert body["anchor"]["source"] in {"observed", "synthetic"}
    stage4 = next(s for s in body["stages"] if s["stage"] == 4)
    assert stage4["output"]["challenged_hypothesis"]  # anchor revealed at stage 4

    r = client.post("/api/agents/run",
                    json={"row_index": 469, "anchor_condition": "bogus"})
    assert r.status_code == 422
    assert "unknown anchor condition" in r.json()["detail"]["detail"]

    r = client.post("/api/agents/run", json={"row_index": 424242})
    assert r.status_code == 404


# --------------------------------------------------------------------- experiments
def test_experiment_status_and_quick_run(client):
    status = client.get("/api/experiments/status").json()
    assert status["full_results"]["available"] is True
    assert status["anchor_experiment"]["n_rows"] == 2457
    assert len(status["ablation_experiment"]["runs"]) == 22

    r = client.post("/api/experiments/quick",
                    json={"conditions": ["incorrect_anchor"],
                          "profiles": ["9_full_with_rag"], "n_cases": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_cases"] == 2 and len(body["rows"]) == 2
    head = body["headline"]["9_full_with_rag|incorrect_anchor"]
    cbr = head["CBR"]
    assert cbr["value"] is None or 0.0 <= cbr["value"] <= 1.0
    assert cbr["n_den"] == 2
    assert "117" in body["note"]           # honest scoping of UI runs
    # quick runs persist audits (stopping rule)
    assert client.get("/api/audits").json()["n"] >= 1

    r = client.post("/api/experiments/quick",
                    json={"conditions": ["bogus"], "n_cases": 1})
    assert r.status_code == 422
    r = client.post("/api/experiments/quick",
                    json={"conditions": ["control"], "profiles": ["nope"],
                          "n_cases": 1})
    assert r.status_code == 422
    r = client.post("/api/experiments/quick",
                    json={"conditions": ["control"], "n_cases": 999})
    assert r.status_code == 422            # UI cap (le 15)


# --------------------------------------------------------------------- audits
def test_audit_browsing(client):
    j = client.get("/api/audits").json()
    assert j["available"] and j["n"] >= 1
    case = next(c for c in j["cases"] if c["case_id"].startswith("regensburg_row"))
    detail = client.get(f"/api/audits/{case['case_id']}").json()
    assert detail["audit"]["completed_stages"] == list(range(1, 9))
    assert "stage_07_output.json" in detail["files"]

    stage = client.get(f"/api/audits/{case['case_id']}/stage_07_output.json").json()
    assert "primary_working_diagnosis" in stage["content"]

    assert client.get(f"/api/audits/{case['case_id']}/secrets.json").status_code == 422
    assert client.get("/api/audits/does_not_exist").status_code == 404
    assert client.get("/api/audits/..%2F..%2Fetc").status_code in (404, 422)


def test_structured_error_shape(client):
    r = client.get("/api/audits/not_a_case/stage_01_output.json")
    assert r.status_code == 404
    body = r.json()
    # FastAPI wraps HTTPException detail: {"detail": {"detail": …, "hint": …}}
    assert "detail" in body and "hint" in body["detail"]
    assert "not_a_case" in body["detail"]["detail"]
