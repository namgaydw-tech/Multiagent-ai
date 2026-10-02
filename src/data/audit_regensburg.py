"""Dataset audit for the Regensburg Pediatric Appendicitis Dataset (UCI 938 / Zenodo 7669442).

This script performs *only* a data audit. It does not train, tune, or evaluate any
predictive model. Training is explicitly deferred until the audit passes review
(see docs/DATASET_AUDIT.md).

Outputs:
  - outputs/metrics/regensburg_audit.json
  - docs/DATASET_AUDIT.md

Usage:
  python -m src.data.audit_regensburg \
      --input data/raw/regensburg_pediatric_appendicitis/app_data.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["Diagnosis", "Management", "Severity"]
ANCHOR_CANDIDATE = "Diagnosis_Presumptive"
POST_OUTCOME_COLUMNS = {
    # recorded at discharge / after the clinical decision -> temporal leakage
    "Length_of_Stay",
}
KNOWN_PROCESS_COLUMNS = {
    "US_Performed",  # whether clinicians ordered ultrasound (encodes suspicion)
    "US_Number",  # image subject identifier, not a clinical predictor
}
# For conservatively managed patients the dataset documentation states the
# Diagnosis label was *defined* using Alvarado/PAS >= 4 AND appendix diameter >= 6 mm.
LABEL_DERIVED_COLUMNS = {"Alvarado_Score", "Paedriatic_Appendicitis_Score", "Appendix_Diameter"}

AGE_BINS = [0, 2, 6, 13, 19]
AGE_LABELS = ["0-1y (infant/toddler)", "2-5y (preschool)", "6-12y (school age)", "13-18y (adolescent)"]


def _univariate_auc(x: pd.Series, y: pd.Series) -> float | None:
    """Return max(AUC, 1-AUC) of a single feature as a leakage / dominance signal."""
    mask = x.notna() & y.notna()
    if mask.sum() < 30:
        return None
    xv, yv = x[mask], y[mask]
    if not pd.api.types.is_numeric_dtype(xv):
        codes, _ = pd.factorize(xv, sort=True)
        xv = pd.Series(codes, index=xv.index)
    if yv.nunique() < 2:
        return None
    pos = xv[yv == yv.unique()[1]]
    neg = xv[yv == yv.unique()[0]]
    if len(pos) == 0 or len(neg) == 0:
        return None
    try:
        ranks = pd.concat([pos, neg]).rank()
        ranks_pos = ranks.iloc[: len(pos)]
        auc = (ranks_pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    except Exception:
        return None
    if np.isnan(auc):
        return None
    return float(max(auc, 1 - auc))


def run_audit(input_path: Path) -> dict:
    xl = pd.ExcelFile(input_path)
    df = xl.parse("All cases")
    summary = xl.parse("Data Summary")

    # --- shape / types -----------------------------------------------------
    dtypes = {str(k): int(v) for k, v in df.dtypes.astype(str).value_counts().items()}

    # --- missingness -------------------------------------------------------
    missing = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)
    missing_table = pd.DataFrame(
        {"n_missing": missing, "pct_missing": missing_pct}
    ).sort_values("n_missing", ascending=False)
    missing_table = missing_table[missing_table.n_missing > 0]

    # --- duplicates --------------------------------------------------------
    dup_exact = int(df.duplicated().sum())
    feature_cols = [c for c in df.columns if c not in TARGETS]
    dup_on_features = int(df.duplicated(subset=feature_cols).sum())

    # --- target distributions --------------------------------------------
    target_distributions = {}
    for t in TARGETS:
        vc = df[t].value_counts(dropna=False)
        target_distributions[t] = {
            str(k): {"n": int(v), "pct": round(100 * v / len(df), 2)} for k, v in vc.items()
        }

    # --- presumptive diagnosis (anchor variable) --------------------------
    pres = df[ANCHOR_CANDIDATE]
    presumptive_counts = pres.value_counts(dropna=False)
    presumptive_top = {str(k): int(v) for k, v in presumptive_counts.head(20).items()}
    # coarse collapse to the two dominant categories for anchor experiments
    pres_norm = pres.astype("string").str.strip().str.lower()
    coarse = pres_norm.where(pres_norm.isin(["appendicitis", "no appendicitis"]), "other_free_text")
    anchor_collapse = {str(k): int(v) for k, v in coarse.value_counts(dropna=False).items()}

    # agreement between presumptive diagnosis and final diagnosis (anchor quality)
    agree_mask = pres_norm.isin(["appendicitis", "no appendicitis"]) & df["Diagnosis"].notna()
    agree = float((pres_norm[agree_mask] == df.loc[agree_mask, "Diagnosis"]).mean()) if agree_mask.sum() else None
    incorrect_anchor_rate = round(100 * (1 - agree), 2) if agree is not None else None

    # --- age distribution --------------------------------------------------
    age = pd.to_numeric(df["Age"], errors="coerce")
    age_stats = {
        "n": int(age.notna().sum()),
        "mean": round(float(age.mean()), 2),
        "std": round(float(age.std()), 2),
        "min": round(float(age.min()), 2),
        "q25": round(float(age.quantile(0.25)), 2),
        "median": round(float(age.median()), 2),
        "q75": round(float(age.quantile(0.75)), 2),
        "max": round(float(age.max()), 2),
        "n_missing": int(age.isna().sum()),
        "age_bins": {
            str(lbl): int(n)
            for lbl, n in pd.cut(age, bins=AGE_BINS, labels=AGE_LABELS, right=False).value_counts().sort_index().items()
        },
    }

    # --- class balance -----------------------------------------------------
    diag = df["Diagnosis"]
    n_pos = int((diag == "appendicitis").sum())
    n_neg = int((diag == "no appendicitis").sum())
    imbalance = {
        "positive_class": "appendicitis",
        "n_positive": n_pos,
        "n_negative": n_neg,
        "prevalence": round(n_pos / max(n_pos + n_neg, 1), 4),
        "imbalance_ratio_neg_pos": round(n_neg / max(n_pos, 1), 3),
    }

    # --- leakage screen ----------------------------------------------------
    y = diag.map({"appendicitis": 1, "no appendicitis": 0})
    leakage = {}
    for c in df.columns:
        if c in TARGETS:
            continue
        auc = _univariate_auc(df[c], y)
        flags = []
        if c == ANCHOR_CANDIDATE:
            flags.append("ANCHOR_VARIABLE: presumptive diagnosis recorded at admission; intentional bias-injection candidate, excluded from baseline predictors")
        if c in POST_OUTCOME_COLUMNS:
            flags.append("POST_OUTCOME: recorded at discharge, after the clinical decision")
        if c in KNOWN_PROCESS_COLUMNS:
            flags.append("PROCESS_VARIABLE: encodes clinician workflow/suspicion or is an identifier")
        if c in LABEL_DERIVED_COLUMNS:
            flags.append("LABEL_DERIVED_RISK: used in the documented definition of the Diagnosis label for conservatively managed patients")
        if auc is not None and auc >= 0.90 and not flags:
            flags.append("HIGH_UNIVARIATE_AUC: verify temporal availability and semantics before use")
        leakage[c] = {
            "univariate_auc_max": round(auc, 4) if auc is not None else None,
            "flags": flags,
        }
    flagged = {k: v for k, v in leakage.items() if v["flags"]}

    # --- data dictionary cross-check --------------------------------------
    dict_vars = set(summary["Variable Name in Data Files"].dropna().astype(str))
    cols = set(df.columns)
    missing_from_dict = sorted(cols - dict_vars - {"Paedriatic_Appendicitis_Score"})
    dict_not_in_data = sorted(v for v in dict_vars if v not in cols and v != "Pediatric_Appendicitis_Score")

    audit = {
        "dataset": "Regensburg Pediatric Appendicitis Dataset",
        "sources": {
            "uci_id": 938,
            "uci_url": "https://archive.ics.uci.edu/dataset/938/regensburg+pediatric+appendicitis",
            "zenodo_doi": "10.5281/zenodo.7669442",
            "zenodo_url": "https://zenodo.org/records/7669442",
            "license": "CC-BY-NC-4.0",
            "audited_file": str(input_path),
            "audited_file_bytes": input_path.stat().st_size,
        },
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dimensions": {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1]), "sheets": xl.sheet_names},
        "dtypes": dtypes,
        "n_missing_cells_total": int(df.isna().sum().sum()),
        "pct_missing_cells": round(100 * float(df.isna().mean().mean()), 2),
        "columns_with_missing": int((missing > 0).sum()),
        "missingness": missing_table.reset_index().rename(columns={"index": "column"}).to_dict(orient="records"),
        "duplicates": {"exact_duplicate_rows": dup_exact, "duplicate_rows_on_feature_columns": dup_on_features},
        "target_distributions": target_distributions,
        "class_balance_diagnosis": imbalance,
        "age_distribution": age_stats,
        "anchor_variable": {
            "column": ANCHOR_CANDIDATE,
            "n_missing": int(pres.isna().sum()),
            "top_values": presumptive_top,
            "collapsed_distribution": anchor_collapse,
            "presumptive_vs_final_diagnosis_agreement": round(agree, 4) if agree is not None else None,
            "incorrect_anchor_rate_pct": incorrect_anchor_rate,
            "usage": "NOT a baseline predictor; used as experimental preliminary-diagnosis anchor (correct/incorrect anchor conditions)",
        },
        "leakage_screen": {
            "method": "univariate ROC-AUC max(AUC,1-AUC) per column vs Diagnosis + documented measurement timing",
            "flagged_columns": flagged,
            "all_columns": leakage,
        },
        "data_dictionary_crosscheck": {
            "n_summary_rows": int(len(summary)),
            "columns_missing_from_dictionary": missing_from_dict,
            "dictionary_rows_absent_from_data": dict_not_in_data,
        },
        "known_limitations": [
            "Single-centre cohort (Children's Hospital St. Hedwig, Regensburg, 2016-2021); external generalisability unknown.",
            "Small sample size (n=782) with ~53 features; high risk of overfitting and wide confidence intervals.",
            "Diagnosis label for conservatively managed patients is partly defined by Alvarado/PAS and appendix diameter -> circularity risk for those columns.",
            "Many ultrasound findings are >70% missing because they were only recorded when the appendix or a complication was visualised; missingness is informative, not MCAR.",
            "Free-text German columns (Diagnosis_Presumptive, Lymph_Nodes_Location, Abscess_Location, Gynecological_Findings) require controlled normalisation.",
            "Ultrasound images (523 MB zip) are optional for the tabular Phase A model and are not downloaded in Phase 1.",
            "No protected health information expected (de-identified research release), but no independent DPIA/IRB review has been performed in this project.",
        ],
    }
    return audit


def render_markdown(audit: dict) -> str:
    ls = audit["leakage_screen"]
    flagged = ls["flagged_columns"]
    miss = audit["missingness"][:15]
    L = []
    L.append("# DATASET_AUDIT: Regensburg Pediatric Appendicitis Dataset\n")
    L.append(f"_Generated by `python -m src.data.audit_regensburg` on {audit['audit_timestamp_utc']}._\n")
    L.append("## 1. Provenance\n")
    src = audit["sources"]
    L.append(f"- UCI: {src['uci_url']} (uci_id {src['uci_id']})")
    L.append(f"- Zenodo DOI: {src['zenodo_doi']} ({src['zenodo_url']})")
    L.append(f"- License: **{src['license']}** (non-commercial restriction — see docs/LIMITATIONS.md)")
    L.append(f"- Audited file: `{src['audited_file']}` ({src['audited_file_bytes']:,} bytes)\n")

    d = audit["dimensions"]
    L.append("## 2. Dimensions\n")
    L.append(f"- Rows: **{d['n_rows']}** (one row per patient) | Columns: **{d['n_cols']}** | Sheets: {', '.join(d['sheets'])}")
    L.append(f"- Dtypes: {audit['dtypes']}")
    L.append(f"- Missing cells: {audit['n_missing_cells_total']:,} ({audit['pct_missing_cells']}% of all cells); columns with ≥1 missing value: {audit['columns_with_missing']}\n")

    L.append("## 3. Targets\n")
    for t, dist in audit["target_distributions"].items():
        L.append(f"- **{t}**: " + "; ".join(f"{k}={v['n']} ({v['pct']}%)" for k, v in dist.items()))
    cb = audit["class_balance_diagnosis"]
    L.append(f"- Diagnosis balance: {cb['n_positive']} appendicitis vs {cb['n_negative']} no appendicitis "
             f"(prevalence {cb['prevalence']}, neg:pos ratio {cb['imbalance_ratio_neg_pos']})\n")

    L.append("## 4. Age distribution\n")
    a = audit["age_distribution"]
    L.append(f"- n={a['n']}, mean {a['mean']}±{a['std']} y, median {a['median']} y (IQR {a['q25']}–{a['q75']}), range {a['min']}–{a['max']}, missing {a['n_missing']}")
    L.append("- Bins: " + "; ".join(f"{k}: {v}" for k, v in a["age_bins"].items()) + "\n")

    L.append("## 5. Duplicates\n")
    L.append(f"- Exact duplicate rows: {audit['duplicates']['exact_duplicate_rows']}")
    L.append(f"- Duplicates over all non-target feature columns: {audit['duplicates']['duplicate_rows_on_feature_columns']}\n")

    L.append("## 6. Missingness (top 15 columns)\n")
    L.append("| Column | n missing | % missing |")
    L.append("|---|---:|---:|")
    for m in miss:
        L.append(f"| {m['column']} | {m['n_missing']} | {m['pct_missing']} |")
    L.append("")

    L.append("## 7. Leakage / anchor screen\n")
    L.append(f"Method: {ls['method']}.\n")
    L.append("| Column | univariate AUC | flags |")
    L.append("|---|---:|---|")
    for k, v in flagged.items():
        L.append(f"| {k} | {v['univariate_auc_max']} | {'; '.join(v['flags'])} |")
    L.append("")
    L.append("**Decision rules for the baseline model:**\n")
    L.append(f"- `{ANCHOR_CANDIDATE}` is excluded from all baseline predictors; it is reserved for the anchoring experiments.")
    L.append("- `Management` and `Severity` are co-targets, never features of the `Diagnosis` model.")
    L.append("- `Length_of_Stay` is excluded (recorded at discharge).")
    L.append("- `US_Number` is an identifier; images are split by patient/subject, never by image.")
    L.append("- `Alvarado_Score`, `Paedriatic_Appendicitis_Score`, `Appendix_Diameter` are retained only with an explicit circularity caveat and a sensitivity analysis excluding them.\n")

    L.append("## 8. Anchor-variable (presumptive diagnosis) analysis\n")
    av = audit["anchor_variable"]
    L.append(f"- Missing: {av['n_missing']}; agreement with final diagnosis (coarse 2-class values only): "
             f"{av['presumptive_vs_final_diagnosis_agreement']}")
    L.append(f"- Estimated incorrect-anchor rate for `appendicitis`/`no appendicitis` anchors: **{av['incorrect_anchor_rate_pct']}%**")
    L.append(f"- Collapsed distribution: {av['collapsed_distribution']}")
    L.append(f"- Usage: {av['usage']}\n")

    L.append("## 9. Data dictionary cross-check\n")
    dd = audit["data_dictionary_crosscheck"]
    L.append(f"- Data Summary rows: {dd['n_summary_rows']}")
    L.append(f"- Data columns without a dictionary row: {dd['columns_missing_from_dictionary'] or 'none'}")
    L.append(f"- Dictionary rows absent from data: {dd['dictionary_rows_absent_from_data'] or 'none'}\n")

    L.append("## 10. Known limitations\n")
    for x in audit["known_limitations"]:
        L.append(f"- {x}")
    L.append("")
    L.append("## 11. Audit verdict\n")
    L.append("- **PASS (conditional)** for Phase A tabular model development, subject to: anchor/co-target exclusions above, "
             "patient-level splitting, informative-missingness handling, and a leakage sensitivity analysis. "
             "No model may be trained until this file is reviewed and this verdict is confirmed.")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=Path("data/raw/regensburg_pediatric_appendicitis/app_data.xlsx"))
    ap.add_argument("--json-out", type=Path, default=Path("outputs/metrics/regensburg_audit.json"))
    ap.add_argument("--md-out", type=Path, default=Path("docs/DATASET_AUDIT.md"))
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: input file not found: {args.input}", file=sys.stderr)
        return 2

    audit = run_audit(args.input)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    args.md_out.write_text(render_markdown(audit), encoding="utf-8")
    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
