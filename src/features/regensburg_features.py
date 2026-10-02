"""Feature-manifest generator for the Regensburg tabular model (Phase 2).

Produces the machine-readable manifest required by the project brief:

    outputs/metadata/regensburg_feature_manifest.json

One record per original column plus one record per derived missingness indicator:
name, type, missing percentage, inclusion status, exclusion reason, categorical
levels, leakage-risk label, and clinical grouping.

Run standalone::

    python -m src.features.regensburg_features
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.preprocessing.regensburg import (
    ROOT, build_feature_spec, config_hash, load_audit, load_audited_dataset,
    load_config, load_data_summary, split_column_roles,
)

MANIFEST_PATH = ROOT / "outputs/metadata/regensburg_feature_manifest.json"


def build_manifest(df: pd.DataFrame | None = None) -> dict:
    """Build the feature manifest dict from the audited dataset + config."""
    config = load_config()
    audit = load_audit()
    df = load_audited_dataset() if df is None else df
    summary = load_data_summary()

    spec = build_feature_spec(df, config)
    records = split_column_roles(df, spec, summary, audit)

    included = [r for r in records if r["inclusion_status"] == "included"]
    excluded = [r for r in records if r["inclusion_status"] == "excluded"]
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": "regensburg_pediatric_appendicitis",
        "dataset_doi": "10.5281/zenodo.7669442",
        "license": "CC-BY-NC-4.0",
        "target": config["target"],
        "config_hash": config_hash(config),
        "n_rows": int(len(df)),
        "n_source_columns": int(len(df.columns)),
        "n_included_features": len([r for r in included if r["type"] != "missing_indicator"]),
        "n_missing_indicators": len([r for r in included if r["type"] == "missing_indicator"]),
        "n_excluded_columns": len(excluded),
        "features": records,
        "excluded_summary": {r["feature"]: r["exclusion_reason"] for r in excluded},
    }


def write_manifest(path: Path = MANIFEST_PATH) -> dict:
    """Build and persist the manifest; returns the manifest dict."""
    manifest = build_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> int:
    m = write_manifest()
    print(f"Wrote {MANIFEST_PATH}")
    print(f"  rows={m['n_rows']} included={m['n_included_features']} "
          f"indicators={m['n_missing_indicators']} excluded={m['n_excluded_columns']}")
    for col, reason in m["excluded_summary"].items():
        print(f"  - excluded {col}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
