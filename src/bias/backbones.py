"""Cross-backbone confirmation-bias experiments (Phase 5, section 15).

Research question under test: *does the debiasing finding depend on the
underlying ML classifier?* Each backbone is one **persisted Phase 5 model**
(or the validation-weighted ensemble) feeding the same information-isolated
multi-agent engine on the same Regensburg test rows under the same anchor
conditions — only the classifier changes.

Design decisions (pre-registered here, not after seeing results):

* decision probability = validation-calibrated probability at the locked
  threshold (identical to ``evaluate_dataset``: ``pred = p_cal >= t``);
* **uniform information budget**: ``shap_explain`` is an empty contributor list
  for every backbone, so no backbone receives extra attribution evidence;
* ground truth is never shown to the decision agents (existing isolation
  preserved — the runners are the Phase 4 ones, unchanged);
* rows/conditions/profiles are identical across backbones (paired design).

Outputs::

    outputs/phase5/bias_backbones/<backbone>.json   (per-case rows + meta)
    outputs/phase5/bias_backbones_summary.json      (CBR/AOR/BCR/HFR/... per backbone)

Research prototype — not a medical device.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.agents.provider import LLMProvider
from src.bias.anchor_generator import anchor_for_case, anchor_vocabulary
from src.bias import experiment as ex
from src.bias import metrics as M
from src.data.split import load_splits
from src.models.model_registry import rebuild_predict
from src.models.prediction_interface import build_prediction
from src.models.uncertainty import estimate_uncertainty
from src.preprocessing.regensburg import (ROOT, config_hash, load_audited_dataset,
                                          load_config)

P5 = ROOT / "outputs" / "phase5"
DS_DIR = P5 / "regensburg_appendicitis_tabular"
REG_ID = "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR"
SEED = 20261002
BACKBONES = ["logistic_regression", "random_forest", "xgboost", "lightgbm",
             "catboost", "ensemble"]
PROFILE_NAMES = ["3_lightgbm_plus_single_llm", "9_full_with_rag"]
CONDITIONS = ["control", "incorrect_anchor"]


def log(msg: str) -> None:
    print(f"[phase5-bias] {msg}", flush=True)


class BackbonePredictor:
    """Phase 5 analogue of :class:`CasePredictor` for one persisted backbone."""

    def __init__(self, backbone: str) -> None:
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}; expected one of {BACKBONES}")
        from src.models.phase5_pipeline import prepare
        prep = prepare(REG_ID)
        self.backbone = backbone
        self.df = load_audited_dataset()
        self.config_hash = config_hash(load_config())
        self.X = prep.X
        self.splits = prep.splits
        self._bundles: dict[int, tuple[dict, dict, dict]] = {}

        thresholds = json.loads(
            (DS_DIR / "calibration" / "thresholds.json").read_text(encoding="utf-8")
        )["thresholds"]
        cal_sel = json.loads(
            (DS_DIR / "calibration" / "calibration.json").read_text(encoding="utf-8"))
        test = json.loads(
            (DS_DIR / "metrics" / "metrics_test.json").read_text(encoding="utf-8"))

        def member(name: str):
            predict = rebuild_predict(joblib.load(DS_DIR / "models" / f"{name}.joblib"))

            def _p(X) -> np.ndarray:
                raw = np.asarray(predict(X), dtype=float)
                if raw.ndim > 1:
                    raw = raw[:, 1]
                if (cal_sel.get(name) or {}).get("selected", "raw") == "raw":
                    return raw
                cblob = joblib.load(DS_DIR / "calibration" / f"{name}_calibrator.joblib")
                cal = cblob.get("calibrator")
                if cal is None:
                    return raw
                return np.asarray(cal.predict(raw), dtype=float)

            return _p

        ens_cfg = (test.get("ensembles") or {}).get("validation_weighted_soft_voting") or {}
        if backbone == "ensemble":
            weights = ens_cfg.get("weights") or {}
            self._members = [(n, float(w), member(n)) for n, w in weights.items()]
            self.threshold = float(ens_cfg.get("threshold")
                                   or thresholds.get("ensemble", {}).get("threshold", 0.5))
        else:
            self._single = member(backbone)
            self.threshold = float(thresholds[backbone]["threshold"])

    # ------------------------------------------------------------------ API
    def has_row(self, row_index: int) -> bool:
        return int(row_index) in self.X.index

    def _probability(self, row_index: int) -> float:
        X_one = self.X.loc[[row_index]]
        if self.backbone == "ensemble":
            total_w = sum(w for _, w, _ in self._members)
            p = sum(w * float(fn(X_one)[0]) for _, w, fn in self._members) / max(total_w, 1e-12)
            return float(p)
        return float(self._single(X_one)[0])

    def bundle(self, row_index: int) -> tuple[dict, dict, dict]:
        if row_index in self._bundles:
            return self._bundles[row_index]
        p = self._probability(row_index)
        unc = estimate_uncertainty(p, threshold=self.threshold, calibrated_probability=p)
        model_output = build_prediction(
            model_name=f"phase5_{self.backbone}",
            model_version="5.0.0",
            positive_probability=p,               # calibrated decision prob (matches test)
            important_features=[],                # uniform budget: no SHAP for any backbone
            config_hash=self.config_hash,
            calibrated_probability=p,
            uncertainty=unc,
            threshold=self.threshold,
        ).model_dump()
        # uniform attribution budget across backbones (documented, never fabricated)
        shap_explain = {"top_contributors": [], "wording_rule":
                        "cross-backbone run: attributions intentionally not provided "
                        "to any backbone (uniform information budget)"}
        out = (model_output, shap_explain, unc)
        self._bundles[row_index] = out
        return out


def run_backbone(backbone: str, *, n_rows: int = 40, reps: int = 1,
                 seed: int = SEED) -> dict[str, Any]:
    """Run the anchor/bias profiles for one backbone and persist the rows."""
    t0 = time.time()
    predictor = BackbonePredictor(backbone)
    test_rows = [int(r) for r in load_splits(ROOT / "data/interim/splits")["test"]
                 if predictor.has_row(int(r))][:n_rows]
    vocab = anchor_vocabulary(predictor.df)
    bundles = {r: predictor.bundle(r) for r in test_rows}
    provider = LLMProvider(backend="deterministic")
    audit_base = ROOT / "outputs" / "audits" / "phase5_backbones" / backbone

    rows_out: list[dict] = []
    per_profile: dict[str, list[dict]] = {}
    for pname in PROFILE_NAMES:
        profile = ex.PROFILES[pname]
        runner = ex.make_runner(predictor, profile, vocab, bundles,
                                retriever=None, provider=provider,
                                experiment="bias_backbone",
                                audit_root_base=audit_base)
        prof_rows: list[dict] = []
        for cond in CONDITIONS:
            for rep in range(reps):
                for r in test_rows:
                    prof_rows.append(runner(cond, r, rep))
        per_profile[pname] = prof_rows
        rows_out.extend(prof_rows)

    payload = {
        "meta": {
            "experiment": "bias_backbone", "backbone": backbone,
            "dataset": REG_ID, "seed": seed, "reps": reps,
            "n_rows": len(test_rows), "row_selection":
                "first n_rows of the persisted Phase 2 test split (identical across backbones)",
            "conditions": CONDITIONS, "profiles": PROFILE_NAMES,
            "information_budget": "shap_explain empty for every backbone (uniform)",
            "decision_probability": "validation-calibrated p at locked threshold",
            "ground_truth_shown_to_agents": False,
            "llm_backend": "deterministic (LLM_BACKEND unset)",
            "elapsed_seconds": round(time.time() - t0, 1),
            "disclaimer": "Research prototype — not a medical device.",
        },
        "headline": {
            pname: {cond: M.headline_metrics([r for r in rows if r["condition"] == cond])
                    for cond in CONDITIONS}
            for pname, rows in per_profile.items()},
        "rows": rows_out,
    }
    out_dir = P5 / "bias_backbones"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{backbone}.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8")
    log(f"{backbone}: {len(rows_out)} case-runs "
        f"({payload['meta']['elapsed_seconds']}s) -> bias_backbones/{backbone}.json")
    return payload


def run_all(backbones: list[str] | None = None, **kw) -> dict[str, Any]:
    """Run every backbone, then write the cross-backbone summary."""
    summary: dict[str, Any] = {
        "generated_from": "outputs/phase5/bias_backbones/*.json",
        "design": ("paired: same test rows, anchor conditions, profiles and engine; "
                   "only the underlying classifier changes; uniform information budget"),
        "backbones": {}, "seed": SEED,
        "disclaimer": "Research prototype — not a medical device.",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for bb in (backbones or BACKBONES):
        payload = run_backbone(bb, **kw)
        summary["backbones"][bb] = payload["headline"]
    (P5 / "bias_backbones_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False, default=str),
        encoding="utf-8")
    log(f"summary -> outputs/phase5/bias_backbones_summary.json "
        f"({len(summary['backbones'])} backbones)")
    return summary


if __name__ == "__main__":
    run_all()
