"""Phase 1 integrity tests: audit output, inventories, citations, config validity.

Run: python -m pytest tests/ -q
These tests never train models; they verify that Phase 1 artifacts are internally consistent
and that no citation or rule exists without a verifiable source.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIT_JSON = ROOT / "outputs/metrics/regensburg_audit.json"
LIT_CSV = ROOT / "research/literature_review.csv"
INV_CSV = ROOT / "research/dataset_inventory.csv"
MANIFEST = ROOT / "knowledge/source_manifest.json"
REQUIRED_COLS = [
    "title", "authors", "year", "journal_or_conference", "publisher", "DOI", "URL",
    "database/index", "open_access", "research_question", "method", "dataset",
    "sample_size", "model", "major_results", "limitations", "relevance_to_project", "citation",
]


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.mark.skipif(not AUDIT_JSON.exists(), reason="run python -m src.data.audit_regensburg first")
class TestAudit:
    def test_dimensions(self):
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        assert audit["dimensions"]["n_rows"] == 782
        assert audit["dimensions"]["n_cols"] == 58

    def test_anchor_variable_is_flagged_and_excluded(self):
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        flagged = audit["leakage_screen"]["flagged_columns"]
        assert "Diagnosis_Presumptive" in flagged
        assert any("ANCHOR_VARIABLE" in f for f in flagged["Diagnosis_Presumptive"]["flags"])
        assert "Length_of_Stay" in flagged  # post-outcome leakage
        assert "Alvarado_Score" in flagged  # label-circularity risk

    def test_no_duplicate_patients(self):
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        assert audit["duplicates"]["exact_duplicate_rows"] == 0

    def test_anchor_analysis_present(self):
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        av = audit["anchor_variable"]
        assert av["column"] == "Diagnosis_Presumptive"
        assert av["incorrect_anchor_rate_pct"] is not None
        assert 0 < av["incorrect_anchor_rate_pct"] < 100

    def test_class_balance_plausible(self):
        audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
        cb = audit["class_balance_diagnosis"]
        assert cb["n_positive"] + cb["n_negative"] <= 782
        assert cb["prevalence"] > 0.3


class TestLiteratureReview:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.rows = _read_csv(LIT_CSV)

    def test_row_count(self):
        assert len(self.rows) >= 40

    def test_columns(self):
        for r in self.rows:
            assert set(REQUIRED_COLS) <= set(r.keys())
            assert None not in r.values()  # no ragged rows

    def test_every_entry_verifiable(self):
        for r in self.rows:
            assert (r["DOI"].strip() and r["DOI"] != "(none - proceedings paper)") or r["URL"], r["title"]
            assert r["URL"].startswith("http"), r["title"]

    def test_no_empty_citation(self):
        for r in self.rows:
            assert r["citation"].strip(), r["title"]

    def test_categories_cover_required_groups(self):
        # group coverage is documented in LITERATURE_REVIEW.md; the gap analysis lives next to it
        md = (ROOT / "research/LITERATURE_REVIEW.md").read_text(encoding="utf-8")
        for heading in [
            "Clinical confirmation bias", "Automation and anchoring", "Multi-agent",
            "Pediatric machine learning", "Explainable AI", "Calibration",
            "Clinical AI safety", "Medical imaging", "Reporting guidelines",
        ]:
            assert heading.lower() in md.lower(), heading
        assert (ROOT / "research/RESEARCH_GAP.md").exists()


class TestDatasetInventory:
    def test_parse_and_roles(self):
        rows = _read_csv(INV_CSV)
        assert len(rows) >= 3
        roles = " ".join(r["Role"] for r in rows)
        assert "Primary training" in roles
        assert "Phase B" in roles
        assert "Phase C" in roles

    def test_no_concatenation_claim(self):
        doc = (ROOT / "docs/DATASETS.md").read_text(encoding="utf-8").lower()
        assert "concatenated" in doc and "never" in doc
        assert "separate" in doc


class TestKnowledgeIntegrity:
    def test_source_manifest_valid(self):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert manifest["sources"], "manifest must list sources"
        for s in manifest["sources"]:
            assert s.get("doi") or s.get("url"), s["id"]

    @pytest.mark.skipif(
        pytest.importorskip("yaml") is None, reason="PyYAML required"
    )
    def test_rule_files_parse_and_cite(self):
        import yaml

        for name in ["pediatric_thresholds.yaml", "sepsis_rules.yaml", "emergency_red_flags.yaml"]:
            data = yaml.safe_load((ROOT / "knowledge" / name).read_text(encoding="utf-8"))
            blob = json.dumps(data, default=str)
            assert "citation" in blob or "source" in blob, name
            assert "doi" in blob or "http" in blob, name
            assert "pending_full_text_transcription" in blob or "verification" in blob, name

    def test_phoenix_criteria_present_and_cited(self):
        text = (ROOT / "knowledge/sepsis_rules.yaml").read_text(encoding="utf-8")
        assert "10.1001/jama.2024.0179" in text
        assert "10.1001/jama.2024.0196" in text
        assert "fail_closed: true" in text


class TestConfig:
    @pytest.fixture(autouse=True)
    def _yaml(self):
        self.yaml = pytest.importorskip("yaml")

    def test_configs_parse(self):
        for p in (ROOT / "config").glob("*.yaml"):
            self.yaml.safe_load(p.read_text(encoding="utf-8"))

    def test_information_isolation_in_agents_config(self):
        agents = self.yaml.safe_load((ROOT / "config/agents.yaml").read_text(encoding="utf-8"))
        stages = agents["stages"]
        assert stages.index("2_independent_differential") < stages.index("3_proponent")
        default = agents["ablations"]["C_hidden_until_differential_complete"]
        assert default["agent_4"]["sees_prediction"] is False
        assert default["agent_3"]["reveal_prediction_after_stage"] == 2

    def test_anchor_excluded_in_model_config(self):
        model = self.yaml.safe_load((ROOT / "config/model.yaml").read_text(encoding="utf-8"))
        excl = model["feature_policy"]["exclude"]
        for col in ["Diagnosis_Presumptive", "Management", "Severity", "Length_of_Stay", "US_Number"]:
            assert col in excl, col

    def test_experiment_conditions(self):
        exp = self.yaml.safe_load((ROOT / "config/experiment.yaml").read_text(encoding="utf-8"))
        conds = exp["anchor_conditions"]
        for key in ["control", "correct_anchor", "incorrect_anchor",
                    "high_confidence_incorrect_anchor", "low_confidence_incorrect_anchor",
                    "model_anchor", "no_model_anchor"]:
            assert key in conds, key
        metrics = exp["bias_metrics"]
        for key in ["CBR", "AOR", "BCR", "HFR", "CRR", "CMR"]:
            assert key in metrics, key


class TestSafetyDocs:
    def test_disclaimer_present(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        assert "not a medical device" in readme
        safety = (ROOT / "docs/SAFETY.md").read_text(encoding="utf-8")
        for phrase in ["MODEL PREDICTION", "RULE-BASED SAFETY ALERT", "LLM INTERPRETATION"]:
            assert phrase in safety
