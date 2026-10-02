"""SHAP explainability engine for the LightGBM appendicitis model (Phase 2, section 2.7).

Outputs:

* global SHAP importance (mean |SHAP|),
* SHAP beeswarm (``outputs/figures/shap_summary.png``),
* patient-level waterfall (``outputs/figures/shap_waterfall_case_<id>.png``),
* dependence plots for the top features.

Wording constraint (enforced throughout, see docs/SAFETY.md):

    "Feature X contributed positively/negatively to this model prediction."

never "Feature X caused the diagnosis".

Research prototype — not a medical device.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.regensburg import ROOT

FIG_DIR = ROOT / "outputs/figures"
META_PATH = ROOT / "outputs/metadata/shap_explainability.json"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WORDING = ("SHAP indicates the feature's contribution to this model prediction, "
           "not clinical causation.")


class ShapEngine:
    """Wraps a fitted LightGBM classifier with a SHAP TreeExplainer."""

    def __init__(self, fitted_estimator, feature_names: list[str]):
        import shap  # local import keeps module import cheap

        self._shap = shap
        self.estimator = fitted_estimator
        self.feature_names = list(feature_names)
        self.explainer = shap.TreeExplainer(fitted_estimator)
        self._values: np.ndarray | None = None
        self._X: pd.DataFrame | None = None

    # ------------------------------------------------------------------ computation
    def compute(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for a feature matrix (returns [n, k] array)."""
        values = self.explainer.shap_values(X)
        if isinstance(values, list):          # older shap: list per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:                  # [n, k, 2] layout from some versions
            values = values[..., 1]
        self._values, self._X = values, X
        return values

    @property
    def values(self) -> np.ndarray:
        if self._values is None:
            raise RuntimeError("call compute() before accessing SHAP values")
        return self._values

    # ------------------------------------------------------------------ outputs
    def global_importance(self) -> list[dict[str, Any]]:
        """Mean |SHAP| per feature, sorted descending."""
        mean_abs = np.abs(self.values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        return [
            {"feature": self.feature_names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in order
            if mean_abs[i] > 0
        ]

    def plot_beeswarm(self, path: Path | None = None) -> Path:
        """Global beeswarm summary plot."""
        path = path or (FIG_DIR / "shap_summary.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(8.5, 6.5))
        self._shap.summary_plot(self.values, self._X, feature_names=self.feature_names,
                                show=False, max_display=20)
        plt.title("SHAP feature contributions (test partition)")
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_waterfall(self, position: int, case_id: str, path: Path | None = None) -> Path:
        """Patient-level waterfall for one row of the computed matrix."""
        path = path or (FIG_DIR / f"shap_waterfall_case_{case_id}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        row = self._shap.Explanation(
            values=self.values[position],
            base_values=float(self.explainer.expected_value)
            if np.isscalar(self.explainer.expected_value) else float(np.asarray(self.explainer.expected_value).ravel()[-1]),
            data=self._X.iloc[position].to_numpy(),
            feature_names=self.feature_names,
        )
        fig = plt.figure(figsize=(8.5, 6.5))
        self._shap.plots.waterfall(row, max_display=15, show=False)
        plt.title(f"SHAP contributions — {case_id}\n{WORDING}")
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return path

    def _auto_interaction(self, feature: str) -> int | None:
        """Strongest interaction partner for ``feature`` among *numeric* columns.

        SHAP's built-in ``interaction_index='auto'`` scans every column and
        crashes on categorical columns holding mixed str/float values
        (``np.unique`` cannot sort them).  We therefore restrict the search to
        numeric columns, which keeps the automatic selection while remaining
        deterministic and crash-free.
        """
        from shap.utils._general import approximate_interactions

        idx_map = [i for i, name in enumerate(self.feature_names)
                   if pd.api.types.is_numeric_dtype(self._X[name])]
        if feature not in self.feature_names or not idx_map:
            return None
        ind = self.feature_names.index(feature)
        if ind not in idx_map:
            return None
        sub_pos = idx_map.index(ind)
        sub_vals = self.values[:, idx_map]
        sub_feats = self._X.iloc[:, idx_map].to_numpy()
        partner = int(approximate_interactions(sub_pos, sub_vals, sub_feats)[0])
        return idx_map[partner]

    def plot_dependence(self, feature: str, path: Path) -> Path | None:
        """Dependence plot for a single feature (returns ``None`` if skipped).

        Only numeric features are plotted (categorical dependence plots are not
        meaningful on the mixed dtype matrix), and the interaction colour is
        resolved explicitly so SHAP never guesses against categorical columns.
        """
        if feature not in self.feature_names:
            return None
        if not pd.api.types.is_numeric_dtype(self._X[feature]):
            return None
        interaction = self._auto_interaction(feature)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(7.5, 5.5))
        try:
            self._shap.dependence_plot(feature, self.values, self._X,
                                       feature_names=self.feature_names, show=False,
                                       interaction_index=interaction)
        except Exception as exc:  # pragma: no cover - defensive, never fabricate plots
            plt.close(fig)
            print(f"[shap] dependence plot skipped for {feature!r}: {exc}")
            return None
        plt.title(f"SHAP dependence — {feature}")
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return path

    def explain_patient(self, case_id: str, position: int,
                        top_k: int = 10) -> dict[str, Any]:
        """Structured, non-causal explanation for a single patient row."""
        row = self.values[position]
        order = np.argsort(np.abs(row))[::-1][:top_k]
        contributors = []
        for i in order:
            contribution = float(row[i])
            raw = self._X.iloc[position, i]
            contributors.append({
                "feature": self.feature_names[i],
                "contribution": round(contribution, 5),
                "direction": "positive" if contribution >= 0 else "negative",
                "value": None if (raw is None or (isinstance(raw, float) and np.isnan(raw))) else
                (float(raw) if isinstance(raw, (int, float, np.floating, np.integer))
                 and not isinstance(raw, bool) else str(raw)),
                "wording": (f"Feature {self.feature_names[i]} contributed "
                            f"{'positively' if contribution >= 0 else 'negatively'} "
                            f"to this model prediction"),
            })
        base = self.explainer.expected_value
        base = float(np.asarray(base).ravel()[-1]) if not np.isscalar(base) else float(base)
        return {
            "case_id": case_id,
            "model_output_scale": "raw LightGBM margin (log-odds-like); contributions sum with base to the model score",
            "base_value": round(base, 5),
            "disclaimer": WORDING,
            "research_only": True,
            "top_contributors": contributors,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


def persist_metadata(engine: ShapEngine, n_rows: int, split_name: str) -> Path:
    """Write explainability metadata (global importance + provenance)."""
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "shap.TreeExplainer (LightGBM)",
        "shap_version": engine._shap.__version__,
        "split": split_name,
        "n_rows_scored": int(n_rows),
        "n_features": len(engine.feature_names),
        "global_importance": engine.global_importance()[:30],
        "wording_constraint": WORDING,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    META_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return META_PATH
