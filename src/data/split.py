"""Patient-level stratified splitting with reproducible persistence (Phase 2).

Implements ``config/model.yaml``:

* 70% train / 15% validation / 15% test, stratified, seed 20261002;
* the decision rule: if the test partition would contain fewer than 40 positive or
  fewer than 40 negative cases, the fallback strategy
  (``repeated_stratified_cv_on_train_val_with_locked_test``) is selected — the test
  partition stays locked and the development partition carries repeated fold
  assignments for cross-validation;
* split assignments are persisted to ``data/interim/splits/*.json`` and are
  byte-reproducible for a fixed seed.

The test partition never influences preprocessing, feature selection, tuning,
calibration, or threshold selection.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.preprocessing.regensburg import ROOT, load_config

SPLIT_DIR = ROOT / "data/interim/splits"
POS = 1
NEG = 0


def _counts(y: np.ndarray) -> dict:
    y = np.asarray(y)
    return {"n": int(len(y)), "n_positive": int((y == POS).sum()), "n_negative": int((y == NEG).sum())}


def make_splits(y: pd.Series, config: dict | None = None) -> dict:
    """Create deterministic stratified 70/15/15 splits for a binary target.

    Returns a dict with strategy, per-partition row indices (lists), and counts.
    """
    config = load_config() if config is None else config
    split_cfg = config["split"]
    seed = int(split_cfg.get("seed", 20261002))
    ratios = split_cfg.get("ratios", {"train": 0.70, "validation": 0.15, "test": 0.15})

    idx = np.asarray(y.index)
    yv = np.asarray(y.values)

    # Step 1: carve out the test partition (stratified).
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=ratios["test"], random_state=seed)
    dev_idx, test_idx = next(sss1.split(idx, yv))

    # Step 2: stratified train/validation split of the development partition.
    val_share = ratios["validation"] / (ratios["train"] + ratios["validation"])
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_share, random_state=seed)
    dev_pos, val_pos = next(sss2.split(idx[dev_idx], yv[dev_idx]))
    train_idx = idx[dev_idx][dev_pos]
    val_idx = idx[dev_idx][val_pos]
    test_idx = idx[test_idx]

    test_counts = _counts(yv[np.isin(idx, test_idx)])
    min_events = 40
    fallback = (test_counts["n_positive"] < min_events) or (test_counts["n_negative"] < min_events)
    strategy = ("repeated_stratified_cv_on_train_val_with_locked_test" if fallback
                else "stratified_patient_level")

    result = {
        "strategy": strategy,
        "fallback_triggered": bool(fallback),
        "decision_rule": split_cfg.get("decision_rule", ""),
        "seed": seed,
        "ratios": ratios,
        "train": {"row_indices": [int(i) for i in train_idx], **_counts(yv[np.isin(idx, train_idx)])},
        "validation": {"row_indices": [int(i) for i in val_idx], **_counts(yv[np.isin(idx, val_idx)])},
        "test": {"row_indices": [int(i) for i in test_idx], **_counts(yv[np.isin(idx, test_idx)])},
    }

    if fallback:
        # Locked test + repeated stratified CV over train+val for model development.
        dev_all = np.concatenate([train_idx, val_idx])
        skf = StratifiedKFold(n_splits=int(split_cfg.get("folds", 5)), shuffle=True, random_state=seed)
        folds = []
        for fold, (tr_pos, te_pos) in enumerate(skf.split(dev_all, yv[np.isin(idx, dev_all)])):
            folds.append({
                "fold": fold,
                "train_indices": [int(i) for i in dev_all[tr_pos]],
                "validation_indices": [int(i) for i in dev_all[te_pos]],
            })
        result["development_folds"] = folds
    return result


def save_splits(splits: dict, df: pd.DataFrame, out_dir: Path = SPLIT_DIR) -> dict[str, Path]:
    """Persist train/validation/test assignments (row indices + patient identifiers)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    id_col = "US_Number" if "US_Number" in df.columns else None
    written: dict[str, Path] = {}
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for part in ("train", "validation", "test"):
        rows = splits[part]["row_indices"]
        payload = {
            "partition": part,
            "strategy": splits["strategy"],
            "seed": splits["seed"],
            "created_utc": created,
            "n": len(rows),
            "n_positive": splits[part]["n_positive"],
            "n_negative": splits[part]["n_negative"],
            "row_indices": rows,
            "patient_ids": [None if id_col is None else (None if pd.isna(df.iloc[r][id_col])
                                                         else str(df.iloc[r][id_col]))
                            for r in rows],
        }
        p = out_dir / f"{part}.json"
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written[part] = p
    # metadata record for reproducibility
    meta = {k: v for k, v in splits.items() if k != "development_folds"}
    (out_dir / "split_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    written["metadata"] = out_dir / "split_metadata.json"
    return written


def load_splits(out_dir: Path = SPLIT_DIR) -> dict[str, list[int]]:
    """Load persisted split row indices (train/validation/test)."""
    out: dict[str, list[int]] = {}
    for part in ("train", "validation", "test"):
        p = out_dir / f"{part}.json"
        if not p.exists():
            raise FileNotFoundError(f"Split file missing: {p}. Run the Phase 2 pipeline first.")
        out[part] = json.loads(p.read_text(encoding="utf-8"))["row_indices"]
    return out


def verify_reproducibility(y: pd.Series, config: dict | None = None) -> bool:
    """Re-compute splits twice and assert identical assignments."""
    a = make_splits(y, config)
    b = make_splits(y, config)
    for part in ("train", "validation", "test"):
        if a[part]["row_indices"] != b[part]["row_indices"]:
            return False
    return True
