"""Data pipeline for the Regensburg Pediatric Appendicitis dataset (Phase 2).

Implements the audited feature policy from ``config/model.yaml``:

* loads the audited workbook and validates expected columns/target,
* removes the prohibited leakage/anchor variables **explicitly, with logged reasons**,
* preserves missingness and adds explicit missing-value indicators,
* never fits a transformation on data outside the training partition.

Nothing in this module invents values: missing data stays missing. Categorical vocabularies are
derived from the training partition only (``CategoricalEncodingState``), so no test information
leaks into preprocessing.

This is a research pipeline. Not a medical device.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "model.yaml"
AUDIT_JSON = ROOT / "outputs/metrics/regensburg_audit.json"

TARGET = "Diagnosis"
POSITIVE_CLASS = "appendicitis"
NEGATIVE_CLASS = "no appendicitis"

# Fallback clinical grouping when the workbook data dictionary has no entry for a column.
GROUP_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("demographic", ("age", "sex", "height", "weight", "bmi")),
    ("scoring", ("alvarado", "appendicitis_score")),
    ("laboratory", ("wbc", "rbc", "hemoglobin", "rdw", "thrombocyte", "neutrophil", "crp",
                    "ketone", "urine", "segmented")),
    ("ultrasound", ("us_", "appendix", "free_fluid", "target_sign", "perfusion", "perforation",
                    "abscess", "lymph", "bowel", "ileus", "coprostasis", "meteorism", "enteritis",
                    "appendicolith", "apendicolith", "conglomerate", "gynecological", "wall_layers",
                    "surrounding", "tissue_reaction")),
    ("vitals", ("temperature", "heart", "respiratory", "oxygen", "spo2", "blood_pressure")),
    ("symptom", ("pain", "nausea", "appetite", "dysuria", "stool", "migratory", "coughing")),
    ("examination", ("peritonitis", "rebound", "psoas")),
    ("workflow", ("length_of_stay", "us_number", "management", "severity")),
]


class DataValidationError(Exception):
    """Raised when the audited dataset does not match the expected schema."""


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    """Load ``config/model.yaml`` (authoritative feature policy)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_audited_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the audited workbook's ``All cases`` sheet."""
    if path is None:
        cfg = load_config()
        # dataset file path is registered in config/datasets.yaml; default to audited location
        path = ROOT / "data/raw/regensburg_pediatric_appendicitis/app_data.xlsx"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Audited dataset not found: {path}. Phase 1 download required (see docs/DATASETS.md)."
        )
    df = pd.read_excel(path, sheet_name="All cases")
    return df


def load_audit() -> dict:
    """Load the Phase-1 audit JSON (leakage labels). Returns {} if absent."""
    if AUDIT_JSON.exists():
        return json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    return {}


def clinical_group(column: str, summary: pd.DataFrame | None) -> str:
    """Map a data column to a clinical grouping using the workbook data dictionary."""
    if summary is not None and len(summary):
        col_map = {
            "Paedriatic_Appendicitis_Score": "Pediatric_Appendicitis_Score",
            "Appendicolith": "Apendicolith",
            "Lymph_Nodes_Location": "Lymph_Node_Location",
        }
        lookup = col_map.get(column, column)
        hit = summary[summary["Variable Name in Data Files"] == lookup]
        if len(hit):
            group = str(hit.iloc[0]["Variable Group"]).strip()
            if group and group.lower() != "nan":
                return group
    low = column.lower()
    for name, keys in GROUP_HINTS:
        if any(k in low for k in keys):
            return name
    return "other"


def leakage_label(column: str, audit: dict) -> str:
    """Return the leakage/anchor label for a column from the Phase-1 audit screen."""
    flagged = audit.get("leakage_screen", {}).get("flagged_columns", {})
    if column in flagged:
        flags = " ".join(flagged[column]["flags"])
        if "ANCHOR_VARIABLE" in flags:
            return "anchor_variable"
        if "POST_OUTCOME" in flags:
            return "post_outcome"
        if "LABEL_DERIVED_RISK" in flags:
            return "label_derived"
        if "PROCESS_VARIABLE" in flags:
            return "process_variable"
        if "HIGH_UNIVARIATE_AUC" in flags:
            return "high_univariate_auc"
    return "none"


def load_data_summary() -> pd.DataFrame | None:
    """Load the workbook's ``Data Summary`` dictionary (best effort)."""
    path = ROOT / "data/raw/regensburg_pediatric_appendicitis/app_data.xlsx"
    if not path.exists():
        return None
    try:
        return pd.read_excel(path, sheet_name="Data Summary")
    except Exception:
        return None


@dataclass
class FeatureSpec:
    """Declarative description of the modelling feature table."""

    features: list[str]
    excluded: dict[str, str]          # column -> reason
    numeric: list[str]
    categorical: list[str]
    indicators: list[str]             # derived __missing columns
    target: str = TARGET


@dataclass
class CategoricalEncodingState:
    """Training-derived categorical vocabulary (no test/val leakage)."""

    categories: dict[str, list[str]] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, categorical: Iterable[str]) -> "CategoricalEncodingState":
        for col in categorical:
            values = X[col].dropna().astype(str).unique().tolist()
            self.categories[col] = sorted(values)
        return self

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cast categoricals to ``category`` dtype using training vocabularies.

        Unseen values become NaN (missing), which LightGBM handles natively.
        """
        out = X.copy()
        for col, cats in self.categories.items():
            if col in out.columns:
                values = out[col].astype("object")
                # keep only values seen during training; everything else -> NaN
                known = values.where(values.isin(cats))
                out[col] = pd.Categorical(known.astype(str).where(known.notna()), categories=cats)
        return out


def validate_dataset(df: pd.DataFrame, config: dict) -> None:
    """Validate schema/target of the audited dataset. Raises ``DataValidationError``."""
    required = {TARGET, "Age", "Sex", "WBC_Count", "CRP", "Appendix_Diameter"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Dataset missing expected columns: {sorted(missing)}")
    if TARGET not in df.columns:
        raise DataValidationError("Target column 'Diagnosis' absent")
    allowed = {POSITIVE_CLASS, NEGATIVE_CLASS, None}
    values = set(df[TARGET].dropna().unique())
    unexpected = values - allowed
    if unexpected:
        raise DataValidationError(f"Unexpected target values: {sorted(map(str, unexpected))}")
    if config.get("target") != TARGET:
        raise DataValidationError(f"config target mismatch: {config.get('target')!r}")


def build_feature_spec(df: pd.DataFrame, config: dict) -> FeatureSpec:
    """Apply the exclusion policy from ``config/model.yaml`` and classify columns."""
    policy = config["feature_policy"]
    excluded: dict[str, str] = {}
    exclude_cols = set(policy.get("exclude", [])) | {TARGET, "Diagnosis_Presumptive"}
    for col in df.columns:
        if col in exclude_cols:
            reason = {
                "Diagnosis_Presumptive": "anchor variable reserved for bias experiments (config feature_policy.exclude)",
                "Management": "co-target, never a predictor (config feature_policy.exclude)",
                "Severity": "co-target, never a predictor (config feature_policy.exclude)",
                "Length_of_Stay": "post-outcome leakage, recorded at discharge (config feature_policy.exclude)",
                "US_Number": "identifier (patient/image linkage), not a predictor (config feature_policy.exclude)",
                TARGET: "target column",
            }.get(col, "excluded by config feature_policy.exclude")
            excluded[col] = reason

    features = [c for c in df.columns if c not in excluded]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]

    # Explicit missing-value indicators for every feature that has missing values in the data.
    missing_mask = df[features].isna().any()
    indicators = [f"{c}__missing" for c in features if bool(missing_mask[c])]
    return FeatureSpec(features=features, excluded=excluded, numeric=numeric,
                       categorical=categorical, indicators=indicators)


def make_target(df: pd.DataFrame) -> pd.Series:
    """Binary target: appendicitis -> 1, no appendicitis -> 0; rows with missing target dropped."""
    y = df[TARGET].map({POSITIVE_CLASS: 1, NEGATIVE_CLASS: 0})
    if y.isna().any():
        n = int(y.isna().sum())
        y = y.dropna()
    return y.astype(int)


def prepare_matrices(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) with missing indicators; rows lacking a valid target are dropped."""
    y = make_target(df)
    X = df.loc[y.index, spec.features].copy()
    for col in spec.features:
        if f"{col}__missing" in spec.indicators:
            X[f"{col}__missing"] = X[col].isna().astype(int)
    return X, y


def split_column_roles(df: pd.DataFrame, spec: FeatureSpec, summary: pd.DataFrame | None,
                       audit: dict) -> list[dict]:
    """Per-column manifest records (feature name, type, missing %, status, reason, group)."""
    records = []
    n = len(df)
    for col in df.columns:
        missing_pct = float(round(100.0 * float(df[col].isna().mean()), 3)) if n else 0.0
        if col in spec.excluded:
            status, reason = "excluded", spec.excluded[col]
            ftype = "target" if col == TARGET else "excluded"
        else:
            status, reason = "included", ""
            ftype = "numeric" if col in spec.numeric else "categorical"
        rec = {
            "feature": col,
            "type": ftype,
            "missing_pct": missing_pct,
            "inclusion_status": status,
            "exclusion_reason": reason,
            "categories": (sorted(str(v) for v in df[col].dropna().unique())[:50]
                           if col in spec.categorical else None),
            "leakage_risk": leakage_label(col, audit),
            "clinical_group": clinical_group(col, summary),
        }
        records.append(rec)
    for ind in spec.indicators:
        records.append({
            "feature": ind,
            "type": "missing_indicator",
            "missing_pct": 0.0,
            "inclusion_status": "included",
            "exclusion_reason": "",
            "categories": ["0", "1"],
            "leakage_risk": "none",
            "clinical_group": "derived_missingness",
            "derived_from": ind[: -len("__missing")],
        })
    return records


def config_hash(config: dict) -> str:
    """Stable short hash of a config dict for run metadata."""
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def train_feature_arrays(X: pd.DataFrame, spec: FeatureSpec,
                         state: CategoricalEncodingState | None = None,
                         fit: bool = False) -> tuple[pd.DataFrame, CategoricalEncodingState]:
    """Return X with categorical columns encoded against the training vocabulary.

    Parameters
    ----------
    fit:
        when True the vocabulary is (re)fit on ``X`` (training partition only).
    """
    state = CategoricalEncodingState() if state is None else state
    if fit:
        state.fit(X[spec.categorical], spec.categorical)
    return state.apply(X), state
