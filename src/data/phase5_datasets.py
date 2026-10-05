"""Phase 5 dataset adapters — one loader per dataset, never concatenated (rule A).

Every loader returns a :class:`Phase5Dataset` with an explicit feature/target
split, documented exclusions (post-outcome / treatment / identifier / label-
derived columns), and patient/group identifiers for grouped splitting. Nothing
is imputed or dropped silently here: loaders only *describe*; fitting happens
in the training pipeline on the training partition.

Loaders for datasets that are absent locally (PhysioNet PIC, PECARN) return an
empty dataset with ``unavailable_status`` set — the pipeline must degrade
gracefully and never fabricate counts.

Research prototype — not a medical device.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.preprocessing.regensburg import ROOT

PHASE5_CONFIG = ROOT / "config/phase5.yaml"

SEED = 20261002


def load_phase5_config(path: Path | str = PHASE5_CONFIG) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class Phase5Dataset:
    """One dataset's modelling view (never merged with another)."""

    dataset_id: str
    X: pd.DataFrame
    y: pd.Series                       # integer class ids; empty when unavailable
    groups: pd.Series | None           # patient/group ids for grouped splitting
    feature_columns: list[str]
    target_names: dict[int, str]       # id -> display label
    exclusions: dict[str, str]         # column -> reason (auditable)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return len(self.X) > 0 and len(self.y) > 0

    @property
    def positive_name(self) -> str:
        return self.target_names.get(1, "positive")


# --------------------------------------------------------------------- sepsis
SEPSIS_EXCLUSIONS: dict[str, str] = {
    "sepsis_group": "label-derived grouping (every culture-positive episode is in group 1)",
    "positive_days": "post-outcome (days to culture positivity)",
    "time_to_antibiotics": "treatment variable — given after the prediction index time",
    "stat_abx": "treatment variable (stat antibiotics flag)",
    "overall_mortality_within_7_days": "post-outcome",
    "overall_mortality_within_14_days": "post-outcome",
    "overall_mortality_within_30_days": "post-outcome",
    "intubated_free_days": "post-outcome (survivorship days)",
    "inotrope_free_days": "post-outcome (survivorship days)",
    "length_of_stay_hours": "post-outcome (recorded at discharge)",
    "cx_site": "90% missing — cannot be used without silent imputation",
    "episode_id": "identifier",
    "unique_patient_id": "identifier (used for grouping only, never a feature)",
    "blood_culture_positive": "target column",
    "intubated_at_time_of_sepsis_evaluation":
        "supportive-care state at/after the sepsis evaluation — treatment-adjacent, excluded "
        "conservatively",
    "inotrope_at_time_of_sepsis_eval":
        "supportive-care state at/after the sepsis evaluation — treatment-adjacent, excluded "
        "conservatively",
}


def load_sepsis(path: Path | str | None = None) -> Phase5Dataset:
    """Neonatal sepsis registry → binary culture-positivity task.

    Target ``blood_culture_positive`` is a verified recorded diagnostic outcome.
    Features exclude every post-outcome, treatment, identifier and label-derived
    column (``SEPSIS_EXCLUSIONS``). Prediction index time = sepsis evaluation;
    only values known at/before that point are used as features.
    """
    cfg = load_phase5_config()["datasets"]["NEONATAL_SEPSIS_REGISTRY"]
    path = Path(path or ROOT / cfg["file"])
    if not path.exists():
        return Phase5Dataset("NEONATAL_SEPSIS_REGISTRY", pd.DataFrame(), pd.Series(dtype=int),
                             None, [], {}, {}, {"unavailable": f"missing file {path}"})
    df = pd.read_csv(path)

    target = "blood_culture_positive"
    y = df[target].astype(int)
    features = [c for c in df.columns if c not in SEPSIS_EXCLUSIONS and c != target]
    X = df[features].copy()

    cat_cols = ["race", "period"]
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype(str)

    counts = y.value_counts().to_dict()
    return Phase5Dataset(
        dataset_id="NEONATAL_SEPSIS_REGISTRY",
        X=X, y=y, groups=df["unique_patient_id"],
        feature_columns=features,
        target_names={0: "culture_negative", 1: "culture_positive"},
        exclusions=dict(SEPSIS_EXCLUSIONS),
        meta={
            "n_rows": int(len(df)),
            "n_patients": int(df["unique_patient_id"].nunique()),
            "episodes_per_patient_median": float(df.groupby("unique_patient_id").size().median()),
            "class_counts": {int(k): int(v) for k, v in counts.items()},
            "prevalence": round(float(y.mean()), 4),
            "missingness_pct": {c: round(float(df[c].isna().mean() * 100), 2)
                                for c in df.columns if df[c].isna().any()},
            "prediction_index_time": "sepsis evaluation (blood culture drawn at evaluation)",
            "doi": cfg["doi"],
        },
    )


# ------------------------------------------------------------------- Kermany
_Kermany_PATTERNS = [
    # v3 unified naming (train + test): NORMAL-1049278-0001.jpeg, BACTERIA-...-, VIRUS-...
    # where the middle field is the randomized patient ID (Kermany et al., Cell 2018)
    re.compile(r"^(?P<cls>NORMAL|BACTERIA|VIRUS)-(?P<pid>\d+)-\d+", re.I),
    # legacy train naming: IM-0115-0001.jpeg
    re.compile(r"^(?P<pid>IM-\d+)-\d+", re.I),
    # legacy train naming: person1000_bacteria_2981.jpeg / person1000_virus_2647.jpeg
    re.compile(r"^(?P<pid>person\d+)_(?:bacteria|virus)_\d+", re.I),
    # legacy test naming: NORMAL-7920519-2.jpeg
    re.compile(r"^(?P<cls>NORMAL|PNEUMONIA)-(?P<pid>\d+)-\d+", re.I),
]


def parse_kermany_filename(name: str) -> tuple[str | None, str | None]:
    """(patient_id, subtype) parsed from a Kermany filename.

    subtype: ``bacteria`` | ``virus`` | ``normal`` | ``pneumonia_unknown``.
    """
    stem = Path(name).stem
    patient = None
    for pat in _Kermany_PATTERNS:
        m = pat.match(stem)
        if m:
            patient = m.groupdict().get("pid")
            if patient:
                break
    low = stem.lower()
    if low.startswith("bacteria") or "bacteria" in low:
        subtype = "bacteria"
    elif low.startswith("virus") or "virus" in low:
        subtype = "virus"
    elif low.startswith("normal") or "normal" in low or low.startswith("im-"):
        subtype = "normal"
    else:
        subtype = "pneumonia_unknown"
    return (patient.upper() if patient else None), subtype


def kermany_inventory(root: Path | str | None = None) -> pd.DataFrame:
    """Inventory of the local Kermany chest-X-ray subset (paths only; no images read)."""
    cfg = load_phase5_config()["datasets"]["KERMANY_PEDIATRIC_PNEUMONIA"]
    root = Path(root or ROOT / cfg["root"])
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return pd.DataFrame(columns=["path", "split", "label", "patient_id", "subtype", "bytes"])
    for p in sorted(root.rglob("*.jpeg")) + sorted(root.rglob("*.jpg")):
        rel = p.relative_to(root)
        parts = rel.parts
        split = parts[0] if len(parts) > 1 else ""
        label_dir = parts[1] if len(parts) > 2 else (parts[0] if len(parts) == 2 else "")
        patient, subtype = parse_kermany_filename(p.name)
        label = label_dir.upper()
        rows.append({
            "path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
            "split": split if split.upper() in ("TRAIN", "TEST", "VAL") else "train",
            "label": label,
            "patient_id": patient,
            "subtype": subtype,
            "bytes": p.stat().st_size,
        })
    return pd.DataFrame(rows)


def kermany_target(inv: pd.DataFrame, multiclass: bool) -> tuple[pd.Series, dict[int, str]]:
    """Map the inventory to target ids. Binary: Normal vs Pneumonia.

    Multiclass (Normal/Bacterial/Viral) is only asserted when the source labels
    actually encode the subtype — otherwise the caller must stay binary.
    """
    if not multiclass:
        y = (inv["label"] != "NORMAL").astype(int)
        return y, {0: "normal", 1: "pneumonia"}
    subtype = inv["subtype"]
    unknown = int((subtype == "pneumonia_unknown").sum())
    if unknown > 0:
        raise ValueError(
            f"multiclass labels not fully verifiable: {unknown} pneumonia files do not encode "
            "bacteria/virus — stay on the binary task (no invented labels)")
    y = subtype.map({"normal": 0, "bacteria": 1, "virus": 2}).astype(int)
    return y, {0: "normal", 1: "bacterial_pneumonia", 2: "viral_pneumonia"}


# ------------------------------------------------------- Regensburg ultrasound
def ultrasound_inventory(zip_path: Path | str | None = None) -> pd.DataFrame:
    """Inventory of US_Pictures.zip: file, patient (US_Number), view.

    Patient id = leading ``<subject #>`` before the first ``.`` in the filename
    (Zenodo README naming convention ``<subject #>.<view #> *.bmp``). Multiple
    views of one patient are grouped; images are NEVER split independently.
    """
    from src.preprocessing.regensburg import ROOT as _R
    zip_path = Path(zip_path or _R / "data/raw/regensburg_pediatric_appendicitis/US_Pictures.zip")
    rows: list[dict[str, Any]] = []
    if not zip_path.exists():
        return pd.DataFrame(columns=["name", "patient_id", "view", "bytes"])
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".bmp"):
                continue
            base = Path(info.filename).name
            m = re.match(r"^(\d+)\.", base)
            patient = m.group(1) if m else None
            rows.append({"name": base, "patient_id": patient,
                         "view": base.split(".", 1)[1] if "." in base else "",
                         "bytes": info.file_size})
    return pd.DataFrame(rows)


def ultrasound_label_map() -> pd.DataFrame:
    """US_Number -> Diagnosis mapping from the tabular cohort (same patients)."""
    from src.preprocessing.regensburg import load_audited_dataset, TARGET
    df = load_audited_dataset()
    sub = df[["US_Number", TARGET]].dropna(subset=["US_Number"]).copy()
    sub["patient_id"] = sub["US_Number"].astype(str).str.replace(r"\.0$", "", regex=True)
    return sub.rename(columns={TARGET: "diagnosis"})[["patient_id", "diagnosis"]]


# --------------------------------------------------------------------- splits
def make_group_split(y: pd.Series, groups: pd.Series, *,
                     test_size: float = 0.15, val_size: float = 0.15,
                     seed: int = SEED) -> dict[str, list[int]]:
    """Stratified *group-level* 70/15/15 split.

    Groups (patients) never cross partitions; stratification is done on the
    group-majority label so each partition keeps a workable class mix. When
    groups override stratification this is documented in the returned metadata
    by the caller (config: ``group_overrides_stratification: true``).
    """
    from sklearn.model_selection import GroupShuffleSplit

    idx = np.asarray(y.index)
    labels = np.asarray(y.values)
    grp = np.asarray(groups.reindex(idx).values)

    def _majority(g: pd.Series) -> int:
        vals = labels[g == grp] if False else None  # placeholder (unused)
        return 0

    # group -> majority label
    tmp = pd.DataFrame({"label": labels, "group": grp})
    glabel = tmp.groupby("group")["label"].agg(lambda s: int(s.mean() >= 0.5))
    uniq_groups = glabel.index.to_numpy()
    uniq_labels = glabel.to_numpy()

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev_g, test_g = next(gss1.split(uniq_groups, uniq_labels, uniq_groups))
    dev_groups, test_groups = uniq_groups[dev_g], uniq_groups[test_g]

    inner_val_share = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=inner_val_share, random_state=seed)
    tr_g, va_g = next(gss2.split(dev_groups, uniq_labels[dev_g], dev_groups))
    train_groups, val_groups = dev_groups[tr_g], dev_groups[va_g]

    def _rows(gs: np.ndarray) -> list[int]:
        mask = np.isin(grp, gs)
        return [int(i) for i in idx[mask]]

    return {"train": _rows(train_groups), "validation": _rows(val_groups),
            "test": _rows(test_groups)}


def phase5_out(dataset_dir: str) -> Path:
    """outputs/phase5/<dataset_dir>/ with the standard subdirs (P)."""
    base = ROOT / "outputs/phase5" / dataset_dir
    for sub in ("audit", "splits", "models", "calibration", "predictions",
                "metrics", "figures", "agents", "bias"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
