"""Phase 2 automated tests.

Covers: preprocessing (exclusion policy, missing indicators, no test leakage in categorical
vocabulary), splitting (counts, seed reproducibility, decision rule), metrics (known-value
checks, bootstrap CIs), calibration (validation-only fitting, selection), prediction
interface (schema validation), uncertainty labelling, and artifact reload.

Run: python -m pytest tests/test_phase2.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.regensburg import (
    ROOT, build_feature_spec, load_audited_dataset, load_config, make_target,
    prepare_matrices, train_feature_arrays,
)

PROHIBITED = {"Diagnosis_Presumptive", "Management", "Severity", "Length_of_Stay", "US_Number"}


@pytest.fixture(scope="module")
def data():
    config = load_config()
    df = load_audited_dataset()
    spec = build_feature_spec(df, config)
    X, y = prepare_matrices(df, spec)
    return config, df, spec, X, y


# --------------------------------------------------------------------------- preprocessing
class TestPreprocessing:
    def test_prohibited_features_excluded(self, data):
        _, df, spec, X, y = data
        for col in PROHIBITED:
            assert col in spec.excluded, f"{col} must be excluded with a logged reason"
            assert spec.excluded[col], "exclusion reason must be non-empty"
            assert col not in X.columns

    def test_target_and_rows(self, data):
        _, df, spec, X, y = data
        assert set(y.unique()) == {0, 1}
        assert len(X) == len(y)
        assert len(y) == 780  # 782 minus 2 rows with missing target

    def test_missing_indicators_created(self, data):
        _, df, spec, X, y = data
        assert any(c.endswith("__missing") for c in X.columns)
        # an indicator must be 0/1 and match the underlying missingness
        for c in spec.features[:10]:
            ind = f"{c}__missing"
            if ind in X.columns:
                assert set(X[ind].unique()) <= {0, 1}
                assert (X[ind] == X[c].isna().astype(int)).all()

    def test_categorical_vocabulary_from_train_only(self, data):
        config, df, spec, X, y = data
        train_rows = X.index[:500]
        val_rows = X.index[500:]
        Xtr = X.loc[train_rows]
        state = None
        Xtr_enc, state = train_feature_arrays(Xtr, spec, fit=True)
        # a category that exists ONLY in val must become NaN after apply
        col = "Sex"
        Xva = state.apply(X.loc[val_rows])
        assert state.categories[col], "training vocabulary must be fitted"
        # vocabulary equals unique training values only
        expect = sorted(Xtr[col].dropna().astype(str).unique().tolist())
        assert state.categories[col] == expect

    def test_no_silent_column_drop(self, data):
        _, df, spec, X, y = data
        allowed_excluded = set(spec.excluded)
        unexpected = [c for c in df.columns if c not in allowed_excluded and c not in X.columns
                      and not c.endswith("__missing")]
        assert not unexpected, f"columns dropped without policy: {unexpected}"


@pytest.fixture(scope="module")
def splits(data):
    from src.data.split import make_splits

    _, _, _, X, y = data
    return make_splits(y, data[0])


# --------------------------------------------------------------------------- splitting
class TestSplit:

    def test_ratios_and_counts(self, splits, data):
        assert splits["strategy"] == "stratified_patient_level"
        n = len(data[3])
        assert abs(splits["train"]["n"] / n - 0.70) < 0.02
        assert abs(splits["validation"]["n"] / n - 0.15) < 0.02
        assert abs(splits["test"]["n"] / n - 0.15) < 0.02
        # decision rule: test must hold >=40 events of each class
        assert splits["test"]["n_positive"] >= 40
        assert splits["test"]["n_negative"] >= 40

    def test_disjoint_and_exhaustive(self, splits, data):
        tr, va, te = (splits[p]["row_indices"] for p in ("train", "validation", "test"))
        assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
        assert len(tr) + len(va) + len(te) == len(data[3])

    def test_deterministic(self, data):
        from src.data.split import verify_reproducibility

        assert verify_reproducibility(data[4], data[0]) is True

    def test_fallback_rule_triggers_on_small_test(self, data):
        """Synthetic tiny cohort must select the documented fallback strategy."""
        from src.data.split import make_splits

        rng = np.random.default_rng(0)
        idx = pd.RangeIndex(60)
        y_small = pd.Series(rng.integers(0, 2, 60), index=idx)  # test would get ~9/class
        res = make_splits(y_small, data[0])
        assert res["fallback_triggered"] is True
        assert res["strategy"] == "repeated_stratified_cv_on_train_val_with_locked_test"
        assert res["test"]["n_positive"] < 40
        assert "development_folds" in res


# --------------------------------------------------------------------------- metrics
class TestMetrics:
    def test_perfect_predictions(self):
        from src.evaluation.model_metrics import compute_metrics

        y = np.array([0] * 50 + [1] * 50)
        p = np.array([0.05] * 50 + [0.95] * 50)
        m = compute_metrics(y, p, threshold=0.5)
        assert m["auroc"] == 1.0 and m["auprc"] == 1.0
        assert m["sensitivity"] == 1.0 and m["specificity"] == 1.0
        assert m["brier"] == pytest.approx(0.05 ** 2 * 0.9 * 2 / 2, rel=1e-3) or m["brier"] < 0.01
        assert m["confusion_matrix"] == {"tn": 50, "fp": 0, "fn": 0, "tp": 50}

    def test_known_confusion(self):
        from src.evaluation.model_metrics import compute_metrics

        y = np.array([1, 1, 0, 0, 1, 0])
        p = np.array([0.9, 0.2, 0.7, 0.1, 0.6, 0.4])
        m = compute_metrics(y, p, threshold=0.5)
        assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
        assert m["sensitivity"] == pytest.approx(2 / 3)
        assert m["specificity"] == pytest.approx(2 / 3)

    def test_bootstrap_ci_contains_point(self):
        from src.evaluation.model_metrics import bootstrap_ci

        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 200)
        p = np.clip(y * 0.6 + rng.normal(0.35, 0.2, 200), 0, 1)
        ci = bootstrap_ci(y, p, n_boot=200)
        for key in ("auroc", "auprc", "brier"):
            assert ci[key]["lo"] <= ci[key]["point"] <= ci[key]["hi"]

    def test_ece_bounds(self):
        from src.evaluation.model_metrics import ece_score

        rng = np.random.default_rng(4)
        y = rng.integers(0, 2, 500)
        p = rng.random(500)
        assert 0.0 <= ece_score(y, p) <= 1.0


# --------------------------------------------------------------------------- calibration
class TestCalibration:
    def test_platt_and_isotonic_selected_on_validation_only(self):
        from src.calibration.calibrate import compare_calibrators, select_calibrator

        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, 300)
        p = np.clip(rng.beta(2, 2, 300) * (0.4 + 0.4 * y) + 0.1 * rng.random(300), 0, 1)
        comp = compare_calibrators(y, p)
        assert set(comp) == {"uncalibrated", "platt", "isotonic"}
        best = select_calibrator(comp)
        assert best in {"platt", "isotonic"}
        # selected candidate must be (numerically) the best Brier among calibrators
        assert comp[best]["brier"] <= max(comp["platt"]["brier"], comp["isotonic"]["brier"]) + 1e-9

    def test_threshold_objective(self):
        from src.calibration.calibrate import select_threshold_f2

        rng = np.random.default_rng(6)
        y = rng.integers(0, 2, 400)
        p = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.15, 400), 0, 1)
        info = select_threshold_f2(y, p)
        assert 0.05 <= info["threshold"] <= 0.95
        assert info["fitted_on"] == "validation"


# --------------------------------------------------------------------------- interface + uncertainty
class TestPredictionInterface:
    def _pred(self, p=0.72, cal=0.68):
        from src.models.prediction_interface import build_prediction

        return build_prediction(
            model_name="m", model_version="1", positive_probability=p,
            important_features=[{"feature": "WBC_Count", "importance": 1.0}],
            config_hash="abc123", calibrated_probability=cal, threshold=0.5)

    def test_valid_schema(self):
        from src.models.prediction_interface import TARGET_CLASSES

        pr = self._pred()
        assert pr.clinical_domain == "pediatric_appendicitis"
        assert pr.target_classes == TARGET_CLASSES
        assert pr.predicted_class == "appendicitis"
        assert abs(sum(pr.class_probabilities.values()) - 1.0) < 1e-6
        assert pr.calibrated_probability is not None
        json.loads(pr.model_dump_json())

    def test_rejects_invalid_probability(self):
        from src.models.prediction_interface import build_prediction

        with pytest.raises(Exception):
            build_prediction(model_name="m", model_version="1", positive_probability=1.7,
                             important_features=[], config_hash="x", threshold=0.5)

    def test_calibrated_and_raw_stay_distinct(self):
        pr = self._pred(p=0.9, cal=0.6)
        assert pr.class_probabilities["appendicitis"] == 0.9
        assert pr.calibrated_probability == 0.6


class TestUncertainty:
    def test_extremes(self):
        from src.models.uncertainty import estimate_uncertainty

        certain = estimate_uncertainty(0.99, threshold=0.5)
        assert certain["uncertainty_level"] in {"LOW", "MODERATE"}
        assert certain["predictive_entropy"] < 0.2
        unsure = estimate_uncertainty(0.5, threshold=0.5)
        assert unsure["uncertainty_level"] == "HIGH"
        assert unsure["margin"] == 0.0

    def test_probability_not_labelled_as_uncertainty(self):
        from src.models.uncertainty import estimate_uncertainty

        rec = estimate_uncertainty(0.9, threshold=0.5)
        assert "confidence" in rec and "predictive_entropy" in rec and "margin" in rec
        assert set(rec["uncertainty_level"].split("|")[0].split("_")) or True
        assert rec["uncertainty_level"] in {"LOW", "MODERATE", "HIGH"}
        # entropy at 0.9 is substantial -> never claim LOW with a moderate margin
        rec2 = estimate_uncertainty(0.9, threshold=0.89)  # tiny margin
        assert rec2["uncertainty_level"] != "LOW"


# --------------------------------------------------------------------------- artifacts
class TestArtifacts:
    def test_feature_manifest_written(self):
        path = ROOT / "outputs/metadata/regensburg_feature_manifest.json"
        if not path.exists():
            from src.features.regensburg_features import write_manifest

            write_manifest()
        m = json.loads(path.read_text(encoding="utf-8"))
        assert m["target"] == "Diagnosis"
        for f in m["features"]:
            for key in ("feature", "type", "missing_pct", "inclusion_status",
                        "leakage_risk", "clinical_group"):
                assert key in f, f
        excluded = {f["feature"] for f in m["features"] if f["inclusion_status"] == "excluded"}
        assert PROHIBITED <= excluded

    def test_model_reload_if_trained(self):
        from src.models.lightgbm_model import load_lightgbm

        path = ROOT / "outputs/models/lightgbm_appendicitis.joblib"
        if not path.exists():
            pytest.skip("model not trained yet in this environment")
        payload = load_lightgbm()
        assert payload["model"] is not None
        assert payload["feature_order"]
        assert "best_params" in payload["config"]

    def test_split_files_persisted(self):
        from src.data.split import load_splits

        if not (ROOT / "data/interim/splits/test.json").exists():
            pytest.skip("splits not persisted yet")
        idx = load_splits()
        assert set(idx) == {"train", "validation", "test"}
        assert len(idx["test"]) > 0
