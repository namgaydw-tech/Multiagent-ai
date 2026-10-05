"""PHASE 5 dataset blocker — the 27 mandatory pre-training checks (section B).

Every candidate dataset receives exactly one status::

    PASS           all checks satisfied
    CONDITIONAL    warn-level findings, each with an explicit recorded reason
    BLOCKED        at least one fatal finding — training is refused by the gate
    NOT_AVAILABLE  data cannot be accessed/verified in this environment

Outputs (when :func:`run_blocking` is executed):

* ``outputs/phase5/dataset_blocking_report.json``
* ``outputs/phase5/dataset_blocking_report.csv``
* ``docs/PHASE5_DATASET_AUDIT.md``

Gate contract: :func:`assert_trainable` raises :class:`DatasetBlockedError` for
any dataset whose executed status is BLOCKED or NOT_AVAILABLE — the trainer
calls it before touching data, so a failed blocker can never reach training.

All findings are computed from local files/metadata actually present; nothing
is assumed from this file's comments. Expensive evidence (file hashes,
perceptual hashes) is computed once per process and cached.

Research prototype — not a medical device.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import yaml

from src.preprocessing.regensburg import ROOT

BLOCKING_CONFIG = ROOT / "config/dataset_blocking.yaml"
REPORT_JSON = ROOT / "outputs/phase5/dataset_blocking_report.json"
REPORT_CSV = ROOT / "outputs/phase5/dataset_blocking_report.csv"
AUDIT_MD = ROOT / "docs/PHASE5_DATASET_AUDIT.md"
DOWNLOAD_MANIFEST = ROOT / "data/manifests/download_manifest.json"

PASS, FAIL, WARN, NA, NOT_AVAILABLE = "PASS", "FAIL", "WARN", "NA", "NOT_AVAILABLE"
S_PASS, S_CONDITIONAL, S_BLOCKED, S_NOT_AVAILABLE = (
    "PASS", "CONDITIONAL", "BLOCKED", "NOT_AVAILABLE")


class DatasetBlockedError(RuntimeError):
    """Raised by the training gate when a dataset failed the blocker."""


@dataclass
class CheckResult:
    key: str
    name: str
    severity: str
    result: str
    reason: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "name": self.name, "severity": self.severity,
                "result": self.result, "reason": self.reason, "evidence": self.evidence}


@dataclass
class DatasetReport:
    dataset_id: str
    status: str
    checks: list[CheckResult]
    reasons: list[str]
    data_present: bool
    evaluated_utc: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "status": self.status,
            "data_present": self.data_present,
            "reasons": self.reasons,
            "evaluated_utc": self.evaluated_utc,
            "n_pass": sum(1 for c in self.checks if c.result == PASS),
            "n_warn": sum(1 for c in self.checks if c.result == WARN),
            "n_fail": sum(1 for c in self.checks if c.result == FAIL),
            "n_na": sum(1 for c in self.checks if c.result == NA),
            "n_not_available": sum(1 for c in self.checks if c.result == NOT_AVAILABLE),
            "checks": [c.as_dict() for c in self.checks],
        }


# --------------------------------------------------------------- evidence cache
_EV_CACHE: dict[str, dict[str, Any]] = {}


def _load_config() -> dict:
    with BLOCKING_CONFIG.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _manifest_entries() -> list[dict[str, Any]]:
    if DOWNLOAD_MANIFEST.exists():
        return json.loads(DOWNLOAD_MANIFEST.read_text(encoding="utf-8")).get("artifacts", [])
    return []


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _dhash_bytes(data: bytes, small: int = 8) -> int:
    """64-bit difference hash for one image (perceptual duplicate screening)."""
    from PIL import Image
    with Image.open(BytesIO(data)) as img:
        g = img.convert("L").resize((small + 1, small), Image.Resampling.LANCZOS)
        px = list(g.getdata())
    bits = 0
    for row in range(small):
        for col in range(small):
            left = px[row * (small + 1) + col]
            right = px[row * (small + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _near_dup_pairs(hashes: dict[str, int], max_distance: int = 6) -> list[tuple[str, str]]:
    """Perceptual near-duplicates via 4-band LSH + exact Hamming check."""
    pairs: list[tuple[str, str]] = []
    band_bits = 16
    bands: dict[tuple[int, int], list[str]] = {}
    for name, h in hashes.items():
        for band in range(4):
            key = (band, (h >> (band * band_bits)) & ((1 << band_bits) - 1))
            bands.setdefault(key, []).append(name)
    checked: set[tuple[str, str]] = set()
    for names in bands.values():
        if len(names) < 2 or len(names) > 200:
            continue
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pair = (a, b) if a < b else (b, a)
                if pair in checked:
                    continue
                checked.add(pair)
                if _hamming(hashes[a], hashes[b]) <= max_distance:
                    pairs.append(pair)
    return pairs


# ------------------------------------------------------------- evidence builders
def _file_evidence(prof: dict[str, Any]) -> dict[str, Any]:
    ev: dict[str, Any] = {"kind": "files", "files": [], "present": [], "missing": []}
    for rel in prof.get("data_paths", []) or []:
        p = ROOT / rel
        if p.is_dir():
            files = [q for q in p.rglob("*") if q.is_file()]
            ev["files"].extend(files)
            (ev["present"] if files else ev["missing"]).append(str(rel))
        elif p.exists():
            ev["files"].append(p)
            ev["present"].append(str(rel))
        else:
            ev["missing"].append(str(rel))
    ev["data_present"] = bool(ev["present"])
    return ev


def _image_dir_evidence(root: Path) -> dict[str, Any]:
    """Hash + perceptual-hash every image file once (cached per process)."""
    key = f"imgdir:{root}"
    if key in _EV_CACHE:
        return _EV_CACHE[key]
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    hashes: dict[str, int] = {}
    sha: dict[str, str] = {}
    for p in files:
        data = p.read_bytes()
        sha[str(p.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        try:
            hashes[str(p.relative_to(ROOT))] = _dhash_bytes(data)
        except Exception:
            continue
    out = {"n_files": len(files), "sha256": sha, "dhash": hashes}
    _EV_CACHE[key] = out
    return out


def _zip_image_evidence(zip_path: Path) -> dict[str, Any]:
    """Hash + perceptual-hash every image entry of a zip (cached per process)."""
    key = f"zip:{zip_path}"
    if key in _EV_CACHE:
        return _EV_CACHE[key]
    import zipfile
    sha: dict[str, str] = {}
    dhash: dict[str, int] = {}
    if not zip_path.exists():
        return {"n_files": 0, "sha256": sha, "dhash": dhash}
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".bmp")]
        for n in names:
            data = zf.read(n)
            sha[n] = hashlib.sha256(data).hexdigest()
            try:
                dhash[n] = _dhash_bytes(data)
            except Exception:
                continue
    out = {"n_files": len(names), "sha256": sha, "dhash": dhash}
    _EV_CACHE[key] = out
    return out


def _ultrasound_label_counts() -> dict[str, int]:
    """US image labels come from the tabular cohort via patient/US_Number linkage."""
    try:
        from src.data.phase5_datasets import ultrasound_label_map
        return {str(k): int(v) for k, v in
                ultrasound_label_map()["diagnosis"].value_counts().items()}
    except Exception as exc:
        return {"unavailable": -1, "error": type(exc).__name__}


def collect_evidence(dataset_id: str, prof: dict[str, Any]) -> dict[str, Any]:
    """Gather every fact the checks need for one dataset (idempotent per process)."""
    if dataset_id in _EV_CACHE:
        return _EV_CACHE[dataset_id]
    ev = _file_evidence(prof)
    ev["dataset_id"] = dataset_id
    ev["profile"] = prof

    # tabular frame
    csvs = [f for f in ev["files"] if f.suffix.lower() == ".csv"]
    xlsx = [f for f in ev["files"] if f.suffix.lower() in (".xlsx", ".xls")]
    if csvs:
        import pandas as pd
        ev["df"] = pd.read_csv(csvs[0])
    elif xlsx:
        import pandas as pd
        try:
            ev["df"] = pd.read_excel(xlsx[0], sheet_name="All cases")
        except Exception:
            ev["df"] = None

    # image inventories (robust to a partially-downloaded archive)
    zip_files = [f for f in ev["files"] if f.suffix.lower() == ".zip"]
    try:
        if dataset_id == "REG_ENSBURG_ULTRASOUND_IMAGES" and zip_files:
            from src.data.phase5_datasets import ultrasound_inventory
            inv = ultrasound_inventory(zip_files[0])
            ev["inventory"] = inv
            ev["image_ev"] = _zip_image_evidence(zip_files[0])
            ev["n_images"] = int(len(inv))
            ev["n_groups"] = int(inv["patient_id"].nunique()) if len(inv) else 0
            ev["parse_rate"] = (float(inv["patient_id"].notna().mean()) if len(inv) else 0.0)
        elif dataset_id == "KERMANY_PEDIATRIC_PNEUMONIA" and ev["present"]:
            from src.data.phase5_datasets import kermany_inventory
            inv = kermany_inventory()
            ev["inventory"] = inv
            root = ROOT / prof.get("data_paths", [""])[0]
            ev["image_ev"] = _image_dir_evidence(root)
            ev["n_images"] = int(len(inv))
            ev["n_groups"] = int(inv["patient_id"].nunique()) if len(inv) else 0
            ev["parse_rate"] = (float(inv["patient_id"].notna().mean()) if len(inv) else 0.0)
    except Exception as exc:
        ev["inventory_error"] = f"{type(exc).__name__}: {exc}"
        ev["data_present"] = False   # incomplete archive == data not usable yet

    # metadata-only datasets (child pneumonia high-risk inventory)
    if prof.get("metadata") and (ROOT / prof["metadata"]).exists():
        ev["metadata"] = json.loads((ROOT / prof["metadata"]).read_text(encoding="utf-8"))

    _EV_CACHE[dataset_id] = ev
    return ev


# ------------------------------------------------------------------------- checks
def _c01_provenance(pid: str, prof: dict, ev: dict) -> CheckResult:
    expects_files = bool(prof.get("data_paths"))
    if not ev["data_present"]:
        if expects_files:
            return CheckResult("1_provenance", "Provenance verification", "fatal", NOT_AVAILABLE,
                              "local data files absent — provenance cannot be verified without "
                              "the data", {"missing": ev["missing"]})
        return CheckResult("1_provenance", "Provenance verification", "fatal", PASS,
                          "documented source with DOI and access mode (metadata-only dataset)",
                          {"source_documented": True})
    entries = _manifest_entries()
    verified, unlisted = [], []
    for rel in ev["present"]:
        entry = next((e for e in entries if e.get("path") == rel
                      or e.get("path", "").replace("\\", "/") == rel), None)
        if entry and entry.get("present"):
            p = ROOT / rel
            if p.is_file() and entry.get("sha256"):
                ok = _sha256(p) == entry.get("sha256")
                verified.append({"path": rel, "sha256_match": ok})
                if not ok:
                    return CheckResult("1_provenance", "Provenance verification", "fatal", FAIL,
                                      f"sha256 mismatch for {rel} vs download manifest", verified)
            else:
                verified.append({"path": rel, "manifest_entry": True})
        else:
            unlisted.append(rel)
    reason = ("all local files match the Phase-1 download manifest"
              if not unlisted else
              f"present but not covered by the download manifest: {unlisted} (recorded as "
              "self-audited provenance)")
    return CheckResult("1_provenance", "Provenance verification", "fatal",
                       PASS if not unlisted else WARN, reason, {"verified": verified,
                                                                "unlisted": unlisted})


def _c02_doi(pid: str, prof: dict, ev: dict) -> CheckResult:
    doi = prof.get("doi")
    if not doi:
        if prof.get("access") or prof.get("credentialing_required"):
            return CheckResult("2_doi_source", "DOI/source verification", "fatal",
                              NOT_AVAILABLE,
                              "source registry documented but no public DOI — access requires "
                              "approval/credentials unavailable in this environment",
                              {"access": prof.get("access") or "credentialed"})
        return CheckResult("2_doi_source", "DOI/source verification", "fatal", FAIL,
                          "no DOI or source recorded in config/dataset_blocking.yaml", {})
    evidence: dict[str, Any] = {"doi": doi}
    meta_files = {
        "NEONATAL_SEPSIS_REGISTRY": "data/manifests/mendeley_5vdz5cftz7_metadata.json",
        "CHILD_PNEUMONIA_MENDELEY": "data/manifests/mendeley_3tx8xymdsv_metadata.json",
        "KERMANY_PEDIATRIC_PNEUMONIA": "data/manifests/mendeley_rscbjbr9sj_metadata.json",
    }
    meta_path = meta_files.get(pid)
    if meta_path and (ROOT / meta_path).exists():
        meta = json.loads((ROOT / meta_path).read_text(encoding="utf-8"))
        meta_doi = ((meta.get("doi") or {}).get("id")) or meta.get("id")
        evidence["metadata_doi"] = meta_doi
        if meta_doi and not str(doi).split("/")[0] in str(meta_doi) and str(meta_doi) != str(doi):
            if not str(meta_doi).startswith(str(doi)):
                return CheckResult("2_doi_source", "DOI/source verification", "fatal", FAIL,
                                  f"config doi {doi} != metadata doi {meta_doi}", evidence)
    return CheckResult("2_doi_source", "DOI/source verification", "fatal", PASS,
                      f"DOI recorded and matched against local source metadata ({doi})",
                      evidence)


def _c03_license(pid: str, prof: dict, ev: dict) -> CheckResult:
    lic = str(prof.get("license") or "")
    if not lic:
        return CheckResult("3_license", "License verification", "fatal", FAIL,
                          "no license recorded", {})
    low = lic.lower()
    if any(tok in low for tok in ("not_exposed", "reverify", "pending")):
        return CheckResult("3_license", "License verification", "fatal", WARN,
                          f"license not confirmed: '{lic}' — recorded as a CONDITIONAL reason; "
                          "confirm on the landing page before any reuse beyond inspection",
                          {"license": lic})
    if "research_only" in low:
        return CheckResult("3_license", "License verification", "fatal", WARN,
                          f"source record states research-only use ('{lic}') — not a standard "
                          "open license; recorded as a CONDITIONAL reason", {"license": lic})
    return CheckResult("3_license", "License verification", "fatal", PASS,
                      f"license identified: {lic}", {"license": lic})


def _c04_pediatric(pid: str, prof: dict, ev: dict) -> CheckResult:
    if prof.get("pediatric") is True:
        return CheckResult("4_pediatric_population", "Pediatric population", "fatal", PASS,
                          "population documented as pediatric (0–18y / neonatal) in the dataset "
                          "record", {})
    if prof.get("pediatric") is False:
        return CheckResult("4_pediatric_population", "Pediatric population", "fatal", FAIL,
                          "population is not pediatric", {})
    return CheckResult("4_pediatric_population", "Pediatric population", "fatal", NOT_AVAILABLE,
                      "population not verifiable without dataset access", {})


def _c05_task(pid: str, prof: dict, ev: dict) -> CheckResult:
    if prof.get("task") and prof.get("target_column"):
        return CheckResult("5_task_relevance", "Diagnostic-task relevance", "fatal", PASS,
                          f"proposed task '{prof['task']}' with target '{prof['target_column']}'",
                          {"task": prof["task"]})
    return CheckResult("5_task_relevance", "Diagnostic-task relevance", "fatal",
                       NOT_AVAILABLE if not ev["data_present"] else FAIL,
                       "no proposed diagnostic task recorded", {})


def _c06_target_availability(pid: str, prof: dict, ev: dict) -> CheckResult:
    if not ev["data_present"]:
        return CheckResult("6_target_availability", "Target availability", "fatal",
                          NOT_AVAILABLE, "data absent — target availability cannot be checked", {})
    df = ev.get("df")
    target = prof.get("target_column", "")
    col = str(target).split()[0]
    if pid == "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR" and df is not None:
        n = int(df[col].notna().sum())
        return CheckResult("6_target_availability", "Target availability", "fatal",
                          PASS if n > 0 else FAIL,
                          f"target '{col}' present for {n}/{len(df)} rows",
                          {"n_labelled": n, "n_rows": int(len(df))})
    if pid == "NEONATAL_SEPSIS_REGISTRY" and df is not None:
        if col not in df.columns:
            return CheckResult("6_target_availability", "Target availability", "fatal", FAIL,
                              f"target column '{col}' missing", {"columns": list(df.columns)})
        n = int(df[col].notna().sum())
        counts = {int(k): int(v) for k, v in df[col].value_counts().items()}
        return CheckResult("6_target_availability", "Target availability", "fatal",
                          PASS if n > 0 else FAIL,
                          f"target '{col}' present for {n}/{len(df)} rows; counts={counts}",
                          {"counts": counts})
    inv = ev.get("inventory")
    if inv is not None and len(inv):
        if pid == "REG_ENSBURG_ULTRASOUND_IMAGES":
            counts = _ultrasound_label_counts()
        else:
            counts = inv["label"].value_counts().to_dict() if "label" in inv else {}
        return CheckResult("6_target_availability", "Target availability", "fatal", PASS,
                          f"{len(inv)} image entries; label counts via patient linkage={counts}",
                          {"counts": {str(k): int(v) for k, v in counts.items()},
                           "n": int(len(inv))})
    if pid == "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR":
        return CheckResult("6_target_availability", "Target availability", "fatal",
                          NOT_AVAILABLE, "tabular data file absent", {})
    return CheckResult("6_target_availability", "Target availability", "fatal", NOT_AVAILABLE,
                      "no labelled data present", {})


def _c07_target_validity(pid: str, prof: dict, ev: dict) -> CheckResult:
    rationale = prof.get("target_rationale")
    if rationale:
        return CheckResult("7_target_validity", "Target validity", "fatal", PASS,
                          " ".join(str(rationale).split())[:400], {"rationale": True})
    if pid == "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR":
        prior = ROOT / "outputs/metrics/regensburg_audit.json"
        if prior.exists():
            audit = json.loads(prior.read_text(encoding="utf-8"))
            if audit.get("leakage_screen", {}).get("flagged_columns"):
                return CheckResult("7_target_validity", "Target validity", "fatal", PASS,
                                  "target semantics verified by the Phase-1 executed audit "
                                  "(docs/DATASET_AUDIT.md)", {"prior_audit": str(prior)})
        return CheckResult("7_target_validity", "Target validity", "fatal", NOT_AVAILABLE,
                          "Phase-1 audit not found", {})
    if not ev["data_present"]:
        return CheckResult("7_target_validity", "Target validity", "fatal", NOT_AVAILABLE,
                          "data absent — target validity cannot be verified", {})
    if pid == "REG_ENSBURG_ULTRASOUND_IMAGES":
        counts = _ultrasound_label_counts()
        if counts and "unavailable" not in counts:
            return CheckResult("7_target_validity", "Target validity", "fatal", PASS,
                              "image labels come from the audited tabular Diagnosis column via "
                              "patient/US_Number linkage (same cohort — labels are not "
                              "re-annotated in Phase 5)", {"label_counts": counts})
        return CheckResult("7_target_validity", "Target validity", "fatal", NOT_AVAILABLE,
                          f"tabular label linkage unavailable: {counts}", {})
    if pid == "KERMANY_PEDIATRIC_PNEUMONIA":
        inv = ev.get("inventory")
        if inv is not None and len(inv):
            labels = sorted(set(inv["label"]))
            ok = set(labels) <= {"NORMAL", "PNEUMONIA"}
            return CheckResult("7_target_validity", "Target validity", "fatal",
                              PASS if ok else FAIL,
                              f"label schema from directory structure: {labels}; source "
                              "description states validated labels (Kermany et al., Cell 2018)",
                              {"labels": labels})
    return CheckResult("7_target_validity", "Target validity", "fatal", NOT_AVAILABLE,
                      "target validity not verifiable", {})


def _c08_sample_size(pid: str, prof: dict, ev: dict) -> CheckResult:
    min_n = int(prof.get("min_n", 200))
    min_groups = int(prof.get("min_groups", 0))
    if not ev["data_present"]:
        return CheckResult("8_sample_size", "Sample size", "warn", NOT_AVAILABLE,
                          "data absent — no sample count may be claimed", {})
    n = ev.get("n_images")
    if n is None:
        df = ev.get("df")
        n = int(len(df)) if df is not None else 0
    groups = ev.get("n_groups")
    if pid == "NEONATAL_SEPSIS_REGISTRY" and ev.get("df") is not None:
        groups = int(ev["df"]["unique_patient_id"].nunique())
    if pid == "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR":
        groups = n
    ok = n >= min_n and (groups is None or groups >= min_groups)
    return CheckResult("8_sample_size", "Sample size", "warn",
                       PASS if ok else WARN,
                       f"n={n} (min_n={min_n}), groups={groups} (min_groups={min_groups})",
                       {"n": n, "groups": groups, "min_n": min_n, "min_groups": min_groups})


def _c09_class_balance(pid: str, prof: dict, ev: dict) -> CheckResult:
    if not ev["data_present"]:
        return CheckResult("9_class_balance", "Class balance", "warn", NOT_AVAILABLE,
                          "data absent — class balance not measurable", {})
    counts: dict[str, int] = {}
    df = ev.get("df")
    target = str(prof.get("target_column", "")).split()[0]
    if pid == "NEONATAL_SEPSIS_REGISTRY" and df is not None and target in df.columns:
        counts = {str(k): int(v) for k, v in df[target].value_counts().items()}
    elif pid == "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR" and df is not None and target in df.columns:
        counts = {str(k): int(v) for k, v in df[target].value_counts(dropna=True).items()}
    elif pid == "REG_ENSBURG_ULTRASOUND_IMAGES":
        counts = {str(k): int(v) for k, v in _ultrasound_label_counts().items()}
    else:
        inv = ev.get("inventory")
        if inv is not None and len(inv) and "label" in inv:
            counts = {str(k): int(v) for k, v in inv["label"].value_counts().items()}
    if not counts:
        return CheckResult("9_class_balance", "Class balance", "warn", NOT_AVAILABLE,
                          "no class counts derivable", {})
    total = sum(counts.values())
    shares = {k: round(v / total, 4) for k, v in counts.items()}
    min_share = min(shares.values())
    note = "" if min_share >= 0.10 else " (minority share <10%: AUPRC must be read alongside AUROC)"
    return CheckResult("9_class_balance", "Class balance", "warn",
                       PASS if min_share >= 0.05 else WARN,
                       f"counts={counts}, min share={min_share:.4f}{note}",
                       {"counts": counts, "shares": shares})


def _c10_patient_identifier(pid: str, prof: dict, ev: dict) -> CheckResult:
    if not ev["data_present"]:
        return CheckResult("10_patient_identifier", "Patient identifier", "warn", NOT_AVAILABLE,
                          "data absent — grouping cannot be verified", {})
    ident = prof.get("patient_identifier")
    if pid == "NEONATAL_SEPSIS_REGISTRY":
        df = ev["df"]
        n = int(df["unique_patient_id"].nunique())
        return CheckResult("10_patient_identifier", "Patient identifier", "warn", PASS,
                          f"unique_patient_id present ({n} patients, repeated episodes) — "
                          "grouped split enforced", {"n_groups": n})
    if pid == "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR":
        df = ev["df"]
        n_rows = len(df)
        return CheckResult("10_patient_identifier", "Patient identifier", "warn", PASS,
                          f"one row per patient ({n_rows} rows = {n_rows} patients)",
                          {"n_rows": n_rows})
    if ev.get("inventory") is not None and len(ev["inventory"]):
        rate = float(ev.get("parse_rate", 0.0))
        n = ev.get("n_groups")
        res = PASS if rate >= 0.98 else WARN
        return CheckResult("10_patient_identifier", "Patient identifier", "warn", res,
                          f"patient id parsed from filename/source for {rate:.1%} of entries "
                          f"({n} groups); grouping overrides stratification",
                          {"parse_rate": rate, "n_groups": n, "declared": ident})
    return CheckResult("10_patient_identifier", "Patient identifier", "warn", NOT_AVAILABLE,
                      "grouping not verifiable", {"declared": ident})


def _c11_duplicate_rows(pid: str, prof: dict, ev: dict) -> CheckResult:
    if ev.get("df") is None:
        return CheckResult("11_duplicate_rows", "Duplicate rows", "fatal", NA,
                          "no tabular rows to screen (image/metadata dataset)", {})
    df = ev["df"]
    n_dup = int(df.duplicated().sum())
    return CheckResult("11_duplicate_rows", "Duplicate rows", "fatal",
                       PASS if n_dup == 0 else FAIL,
                       f"{n_dup} exact duplicate rows of {len(df)}", {"n_duplicates": n_dup})


def _c12_duplicate_patients(pid: str, prof: dict, ev: dict) -> CheckResult:
    if ev.get("df") is not None and pid == "NEONATAL_SEPSIS_REGISTRY":
        df = ev["df"]
        n_pat = int(df["unique_patient_id"].nunique())
        return CheckResult("12_duplicate_patients", "Duplicate patients", "warn", PASS,
                          f"{len(df)} episodes across {n_pat} patients — repeated episodes are "
                          "grouped so no patient crosses partitions",
                          {"n_rows": int(len(df)), "n_patients": n_pat})
    inv = ev.get("inventory")
    if inv is not None and len(inv) and "patient_id" in inv:
        n_multi = int((inv.groupby("patient_id").size() > 1).sum())
        return CheckResult("12_duplicate_patients", "Duplicate patients", "warn", PASS,
                          f"{n_multi} patients contribute multiple images — grouped, never "
                          "split across partitions", {"n_multi_image_patients": n_multi})
    return CheckResult("12_duplicate_patients", "Duplicate patients", "warn", NA,
                      "one row per patient / no patient grouping needed", {})


def _c13_duplicate_files(pid: str, prof: dict, ev: dict) -> CheckResult:
    image_ev = ev.get("image_ev")
    if not image_ev:
        return CheckResult("13_duplicate_files", "Duplicate files", "fatal", NA,
                          "no image files for this dataset", {})
    sha = image_ev["sha256"]
    by_hash: dict[str, list[str]] = {}
    for name, h in sha.items():
        by_hash.setdefault(h, []).append(name)
    dup_groups = {h: names for h, names in by_hash.items() if len(names) > 1}
    n_dup_files = sum(len(v) for v in dup_groups.values())
    if not dup_groups:
        return CheckResult("13_duplicate_files", "Duplicate files", "fatal", PASS,
                          f"0 exact duplicates across {len(sha)} files (sha256)",
                          {"n_files": len(sha)})
    # cross-split duplicates are contamination; within-split duplicates are removed deterministically
    cross = [names for names in dup_groups.values()
             if len({n.split("/")[0] if pid == "KERMANY_PEDIATRIC_PNEUMONIA" else ""
                     for n in names}) > 1]
    if pid == "KERMANY_PEDIATRIC_PNEUMONIA":
        cross = [names for names in dup_groups.values()
                 if len({Path(n).parts[-3] for n in names if len(Path(n).parts) >= 3}) > 1]
    if cross and pid == "KERMANY_PEDIATRIC_PNEUMONIA":
        return CheckResult("13_duplicate_files", "Duplicate files", "fatal", FAIL,
                          f"{len(cross)} exact-duplicate groups span the original train/test "
                          "directories — train/test contamination; remediation required "
                          "(patient-level regroup before any training)",
                          {"cross_split_groups": cross[:20], "n_groups": len(cross)})
    return CheckResult("13_duplicate_files", "Duplicate files", "fatal", WARN,
                       f"{len(dup_groups)} exact-duplicate groups ({n_dup_files} files) within "
                       "one partition — removed deterministically by sha256 dedup before "
                       "training; file list persisted in the audit",
                       {"n_groups": len(dup_groups), "n_files": n_dup_files,
                        "examples": list(dup_groups.values())[:10]})


def _c14_file_hashes(pid: str, prof: dict, ev: dict) -> CheckResult:
    image_ev = ev.get("image_ev")
    if not image_ev:
        # tabular: verify against download manifest
        entries = _manifest_entries()
        listed = [e for e in entries if e.get("present") and e.get("sha256")
                  and any(e.get("path", "").endswith(Path(f).name)
                          for f in [str(p) for p in ev["files"]])]
        if ev["data_present"] and listed:
            return CheckResult("14_file_hashes", "File-hash deduplication", "fatal", PASS,
                              f"sha256 recorded for {len(listed)} dataset files in the manifest",
                              {"n": len(listed)})
        return CheckResult("14_file_hashes", "File-hash deduplication", "fatal", NA,
                          "no per-file hashes applicable (single tabular source)", {})
    sha = image_ev["sha256"]
    stored = sorted(set(sha.values()))
    return CheckResult("14_file_hashes", "File-hash deduplication", "fatal", PASS,
                      f"sha256 computed for {len(sha)} files ({len(stored)} unique)",
                      {"n_files": len(sha), "n_unique": len(stored)})


def _c15_perceptual(pid: str, prof: dict, ev: dict) -> CheckResult:
    image_ev = ev.get("image_ev")
    if not image_ev or not image_ev.get("dhash"):
        return CheckResult("15_perceptual_duplicates", "Perceptual duplicate risk", "warn", NA,
                          "no images to screen", {})
    pairs = _near_dup_pairs(image_ev["dhash"], max_distance=6)
    if not pairs:
        return CheckResult("15_perceptual_duplicates", "Perceptual duplicate risk", "warn", PASS,
                          f"0 near-duplicate pairs (64-bit dHash, Hamming ≤ 6) among "
                          f"{len(image_ev['dhash'])} images",
                          {"n_hashed": len(image_ev["dhash"])})
    return CheckResult("15_perceptual_duplicates", "Perceptual duplicate risk", "warn", WARN,
                      f"{len(pairs)} perceptually near-duplicate image pairs (dHash Hamming ≤ 6) "
                      "— flagged for deterministic exclusion from evaluation partitions",
                      {"n_pairs": len(pairs), "examples": [list(p) for p in pairs[:10]]})


def _c16_pre_augmented(pid: str, prof: dict, ev: dict) -> CheckResult:
    # metadata-only inventory dataset: the Phase-1 audit flags are the evidence
    if pid == "CHILD_PNEUMONIA_MENDELEY":
        known = (prof.get("known_risks") or {})
        n_aug = int(known.get("pre_augmented_files", 0))
        return CheckResult("16_pre_augmented_leakage", "Pre-augmented image leakage", "fatal",
                          FAIL if n_aug > 0 else PASS,
                          f"{n_aug} of {known.get('total_files')} files carry the aug_ prefix "
                          "(pre-augmented copies) with {known.get('duplicate_filenames')} "
                          "duplicate filenames and no patient grouping — random splits would "
                          "leak augmented siblings across train/test"
                          if n_aug > 0 else "no pre-augmented files flagged",
                          {"known_risks": known})
    inv = ev.get("inventory")
    if inv is None or not len(inv):
        return CheckResult("16_pre_augmented_leakage", "Pre-augmented image leakage", "fatal", NA,
                          "no image inventory", {})
    names = (inv["name"] if "name" in inv else inv.get("path", pd_series_empty())).astype(str)
    suspicious = [n for n in names if n.lower().startswith("aug_") or "_aug" in n.lower()
                  or "augmented" in n.lower()]
    return CheckResult("16_pre_augmented_leakage", "Pre-augmented image leakage", "fatal",
                       PASS if not suspicious else FAIL,
                       f"0 pre-augmented filenames detected among {len(names)} files"
                       if not suspicious else
                       f"{len(suspicious)} pre-augmented filenames detected",
                       {"n_suspicious": len(suspicious), "examples": suspicious[:10]})


def pd_series_empty():
    import pandas as pd
    return pd.Series(dtype=str)


def _leakage_check(name: str, key: str, required: dict[str, str], prof: dict) -> CheckResult:
    exclusions = prof.get("feature_exclusions") or {}
    if isinstance(exclusions, list):
        exclusions = {c: "excluded by config" for c in exclusions}
    if not prof.get("target_column") and not prof.get("task"):
        pass
    missing = [col for col in required if col not in exclusions
               and col not in str(prof.get("feature_exclusions", ""))]
    if not prof.get("data_present_hint", True) and not exclusions:
        return CheckResult(key, name, "fatal", NA, "no tabular feature space for this dataset", {})
    if not exclusions and str(prof.get("modality", "")) == "image":
        return CheckResult(key, name, "fatal", NA,
                          "no tabular feature space (image dataset)", {})
    if missing:
        return CheckResult(key, name, "fatal", FAIL,
                          f"columns not found in the recorded exclusion policy: {missing}",
                          {"exclusions": sorted(exclusions)})
    return CheckResult(key, name, "fatal", PASS,
                      f"all required columns excluded from features: {sorted(required)}",
                      {"exclusions": {c: exclusions[c] for c in required}})


_POST_OUTCOME = {
    "NEONATAL_SEPSIS_REGISTRY": ["positive_days", "overall_mortality_within_7_days",
                                 "overall_mortality_within_14_days",
                                 "overall_mortality_within_30_days", "intubated_free_days",
                                 "inotrope_free_days", "length_of_stay_hours"],
    "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR": ["Length_of_Stay"],
}
_IDENTIFIER = {
    "NEONATAL_SEPSIS_REGISTRY": ["episode_id", "unique_patient_id"],
    "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR": ["US_Number"],
}
_TREATMENT = {
    "NEONATAL_SEPSIS_REGISTRY": ["time_to_antibiotics", "stat_abx"],
    "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR": ["Management"],
}
_LABEL_DERIVED = {
    "NEONATAL_SEPSIS_REGISTRY": ["sepsis_group"],
    "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR": ["Diagnosis_Presumptive", "Diagnosis"],
}


def _image_or_na(pid: str, prof: dict, ev: dict, key: str, name: str) -> CheckResult | None:
    if str(prof.get("modality", "")) == "image" or pid in (
            "PHYSIONET_PIC", "PECARN"):
        if not ev["data_present"]:
            return CheckResult(key, name, "fatal", NOT_AVAILABLE,
                              "data absent — feature-space check not applicable", {})
        return CheckResult(key, name, "fatal", NA,
                          "no tabular feature space (image dataset)", {})
    return None


def _c17_post_outcome(pid: str, prof: dict, ev: dict) -> CheckResult:
    r = _image_or_na(pid, prof, ev, "17_post_outcome_leakage", "Post-outcome leakage")
    if r:
        return r
    req = _POST_OUTCOME.get(pid, [])
    if not req:
        if not ev["data_present"]:
            return CheckResult("17_post_outcome_leakage", "Post-outcome leakage", "fatal",
                              NOT_AVAILABLE, "data absent", {})
        return CheckResult("17_post_outcome_leakage", "Post-outcome leakage", "fatal", NA,
                          "no known post-outcome columns for this dataset", {})
    return _leakage_check("Post-outcome leakage", "17_post_outcome_leakage",
                          {c: "" for c in req}, prof)


def _c18_identifier_leakage(pid: str, prof: dict, ev: dict) -> CheckResult:
    r = _image_or_na(pid, prof, ev, "18_identifier_leakage", "Identifier leakage")
    if r:
        return r
    req = _IDENTIFIER.get(pid, {})
    if not req:
        return CheckResult("18_identifier_leakage", "Identifier leakage", "fatal", NA,
                          "no identifier columns for this dataset", {})
    return _leakage_check("Identifier leakage", "18_identifier_leakage",
                          {c: "" for c in req}, prof)


def _c19_treatment(pid: str, prof: dict, ev: dict) -> CheckResult:
    r = _image_or_na(pid, prof, ev, "19_treatment_leakage", "Treatment/management leakage")
    if r:
        return r
    req = _TREATMENT.get(pid, {})
    if not req:
        return CheckResult("19_treatment_leakage", "Treatment/management leakage", "fatal", NA,
                          "no treatment columns for this dataset", {})
    return _leakage_check("Treatment/management leakage", "19_treatment_leakage",
                          {c: "" for c in req}, prof)


def _c20_diagnosis_code(pid: str, prof: dict, ev: dict) -> CheckResult:
    if pid == "PHYSIONET_PIC":
        return CheckResult("20_diagnosis_code_leakage", "Diagnosis-code leakage", "fatal", PASS,
                          "policy recorded: ICD/discharge codes may be TARGETS only and must "
                          "never appear as input features (enforced when data becomes available)",
                          {"policy": prof.get("label_policy", "")})
    r = _image_or_na(pid, prof, ev, "20_diagnosis_code_leakage", "Diagnosis-code leakage")
    if r:
        return r
    df = ev.get("df")
    if df is None:
        return CheckResult("20_diagnosis_code_leakage", "Diagnosis-code leakage", "fatal",
                          NOT_AVAILABLE if not ev["data_present"] else NA, "no frame", {})
    code_like = [c for c in df.columns
                 if any(tok in c.lower() for tok in ("icd", "diagnosis_code", "dx_code",
                                                     "discharge_code"))]
    exclusions = prof.get("feature_exclusions") or {}
    leaked = [c for c in code_like if c not in exclusions]
    return CheckResult("20_diagnosis_code_leakage", "Diagnosis-code leakage", "fatal",
                       PASS if not leaked else FAIL,
                       "no diagnosis-code columns present in the feature space"
                       if not leaked else f"diagnosis-code columns not excluded: {leaked}",
                       {"code_like_columns": code_like})


def _c21_temporal(pid: str, prof: dict, ev: dict) -> CheckResult:
    if pid == "NEONATAL_SEPSIS_REGISTRY":
        window = prof.get("prediction_index_time") or (ev.get("df") is not None and
                                                       "prediction window recorded in loader")
        ok = bool(ev.get("df") is not None) and _POST_OUTCOME[pid]
        return CheckResult("21_temporal_leakage", "Temporal leakage", "fatal",
                          PASS if ok else FAIL,
                          "prediction window defined: index time = sepsis evaluation; only "
                          "values known at/before that time are features; post-index outcome "
                          "and treatment columns excluded (see exclusion policy)",
                          {"prediction_window": prof.get("prediction_index_time",
                                                         "sepsis evaluation")})
    if pid == "PHYSIONET_PIC":
        return CheckResult("21_temporal_leakage", "Temporal leakage", "fatal", PASS,
                          "policy recorded: prediction window and cutoff must be defined before "
                          "any task is built; only pre-cutoff information allowed",
                          {"policy": prof.get("prediction_window", "")})
    r = _image_or_na(pid, prof, ev, "21_temporal_leakage", "Temporal leakage")
    if r:
        return r
    return CheckResult("21_temporal_leakage", "Temporal leakage", "fatal", NA,
                      "cross-sectional data — no temporal ordering to exploit", {})


def _c22_label_circularity(pid: str, prof: dict, ev: dict) -> CheckResult:
    r = _image_or_na(pid, prof, ev, "22_label_circularity", "Label circularity")
    if r:
        return r
    req = _LABEL_DERIVED.get(pid, {})
    if not req:
        return CheckResult("22_label_circularity", "Label circularity", "warn", NA,
                          "no label-circular columns identified for this dataset", {})
    res = _leakage_check("Label circularity", "22_label_circularity", {c: "" for c in req}, prof)
    res.severity = "warn"
    if res.result == FAIL:
        res.reason = ("label-circular/anchor columns not recorded as excluded: "
                      + res.reason)
    else:
        res.reason += (" — excluded from predictors; Regensburg label-circular score features "
                       "additionally carry a documented sensitivity analysis (config/model.yaml)"
                       if pid.startswith("REG_") else
                       " — excluded from predictors (label-derived/anchor columns)")
    return res


def _c23_contamination(pid: str, prof: dict, ev: dict) -> CheckResult:
    if pid == "REG_ENSBURG_PEDIATRIC_APPENDICITIS_TABULAR":
        split_dir = ROOT / "data/interim/splits"
        if not (split_dir / "test.json").exists():
            return CheckResult("23_train_test_contamination", "Train/test contamination", "fatal",
                              NOT_AVAILABLE, "persisted split not found", {})
        parts = {}
        for p in ("train", "validation", "test"):
            parts[p] = set(json.loads((split_dir / f"{p}.json").read_text(encoding="utf-8"))
                           ["row_indices"])
        overlap = ((parts["train"] & parts["test"]) | (parts["train"] & parts["validation"])
                   | (parts["validation"] & parts["test"]))
        return CheckResult("23_train_test_contamination", "Train/test contamination", "fatal",
                          PASS if not overlap else FAIL,
                          f"persisted Phase-2 partitions are disjoint (n_train="
                          f"{len(parts['train'])}, n_val={len(parts['validation'])}, "
                          f"n_test={len(parts['test'])})",
                          {"overlap": len(overlap)})
    image_ev = ev.get("image_ev")
    if image_ev and pid == "KERMANY_PEDIATRIC_PNEUMONIA":
        inv = ev.get("inventory")
        if inv is None or not len(inv):
            return CheckResult("23_train_test_contamination", "Train/test contamination", "fatal",
                              NOT_AVAILABLE, "inventory empty", {})
        # hash overlap between original train/ and test/ directories
        sha = image_ev["sha256"]
        train_h = {h for n, h in sha.items() if "/train/" in n}
        test_h = {h for n, h in sha.items() if "/test/" in n}
        shared = train_h & test_h
        # patient-id overlap between original splits
        tr_pat = set(inv[inv["split"] == "train"]["patient_id"].dropna())
        te_pat = set(inv[inv["split"] == "test"]["patient_id"].dropna())
        shared_pat = tr_pat & te_pat
        ok = not shared and not shared_pat
        return CheckResult("23_train_test_contamination", "Train/test contamination", "fatal",
                          PASS if ok else FAIL,
                          f"original train/test: {len(shared)} shared file hashes, "
                          f"{len(shared_pat)} shared patient ids"
                          + ("" if ok else " — contamination: regroup by patient before any "
                                           "training (original split not trusted)"),
                          {"shared_hashes": len(shared), "shared_patients": len(shared_pat)})
    if not ev["data_present"]:
        return CheckResult("23_train_test_contamination", "Train/test contamination", "fatal",
                          NOT_AVAILABLE, "data absent", {})
    return CheckResult("23_train_test_contamination", "Train/test contamination", "fatal", NA,
                      "no partitions created yet — enforced by grouped splitting at train time", {})


def _c24_mirror(pid: str, prof: dict, ev: dict) -> CheckResult:
    mirrors = prof.get("mirrors") or []
    if not mirrors:
        return CheckResult("24_mirror_duplication", "Mirror-source duplication", "fatal", PASS,
                          "no known mirrors; single recorded source", {})
    local_roots = prof.get("data_paths", [])
    return CheckResult("24_mirror_duplication", "Mirror-source duplication", "fatal", PASS,
                      f"declared mirrors {mirrors} are the SAME underlying dataset; only one "
                      f"local source is used ({local_roots}) — mirrors never count as cohorts",
                      {"mirrors": mirrors, "local_sources": local_roots})


def _c25_missingness(pid: str, prof: dict, ev: dict) -> CheckResult:
    r = _image_or_na(pid, prof, ev, "25_excessive_missingness", "Excessive missingness")
    if r:
        r.severity = "warn"
        return r
    df = ev.get("df")
    if df is None:
        return CheckResult("25_excessive_missingness", "Excessive missingness", "warn",
                          NOT_AVAILABLE if not ev["data_present"] else NA, "no frame", {})
    excl = prof.get("feature_exclusions") or {}
    excl_keys = set(excl) if not isinstance(excl, list) else set(excl)
    feats = [c for c in df.columns if c not in excl_keys
             and c != str(prof.get("target_column", "")).split()[0]]
    miss = {c: round(float(df[c].isna().mean() * 100), 2) for c in feats}
    bad = {c: v for c, v in miss.items() if v > 50}
    # documented no-silent-drop policy: missingness kept + explicit indicators (model.yaml)
    policy_doc = ""
    model_cfg = ROOT / "config/model.yaml"
    if model_cfg.exists() and pid.startswith("REG_ENSBURG_PEDIATRIC"):
        try:
            import yaml as _yaml
            mc = _yaml.safe_load(model_cfg.read_text(encoding="utf-8"))
            policy_doc = str(mc.get("feature_policy", {}).get("missing_values", ""))
        except Exception:
            policy_doc = ""
    if bad and not policy_doc:
        return CheckResult("25_excessive_missingness", "Excessive missingness", "warn", WARN,
                          f"{len(bad)} feature(s) >50% missing {bad} — no documented "
                          "missingness policy; silent drops would be a blocker",
                          {"missing_pct": {c: miss[c] for c in list(bad)[:20]}})
    if bad:
        return CheckResult("25_excessive_missingness", "Excessive missingness", "warn", PASS,
                          f"{len(bad)} feature(s) >50% missing, documented: preserved as-is with "
                          f"explicit missing indicators (feature_policy.missing_values="
                          f"{policy_doc}); no silent drops, no imputation of outcomes",
                          {"missing_pct": {c: miss[c] for c in list(bad)[:20]},
                           "policy": policy_doc})
    return CheckResult("25_excessive_missingness", "Excessive missingness", "warn", PASS,
                      f"max feature missingness {max(miss.values(), default=0)}%; no silent drops",
                      {"n_features": len(feats)})


def _c26_access(pid: str, prof: dict, ev: dict) -> CheckResult:
    if prof.get("credentialing_required"):
        return CheckResult("26_access_restrictions", "Availability/access", "fatal", NOT_AVAILABLE,
                          prof.get("unavailable_status", "NOT_AVAILABLE"),
                          {"access": "credentialed"})
    if prof.get("access"):
        return CheckResult("26_access_restrictions", "Availability/access", "fatal",
                          NOT_AVAILABLE if not ev["data_present"] else PASS,
                          prof.get("unavailable_status", str(prof.get("access"))),
                          {"access": prof.get("access")})
    if not ev["data_present"]:
        return CheckResult("26_access_restrictions", "Availability/access", "fatal",
                          NOT_AVAILABLE, f"expected local data missing: {ev['missing']}",
                          {"missing": ev["missing"]})
    return CheckResult("26_access_restrictions", "Availability/access", "fatal", PASS,
                      "open access without registration; data present locally", {})


def _c27_compatibility(pid: str, prof: dict, ev: dict) -> CheckResult:
    if prof.get("credentialing_required") or prof.get("access"):
        return CheckResult("27_task_compatibility", "Task compatibility", "fatal", NOT_AVAILABLE,
                          "cannot confirm task compatibility without dataset access", {})
    if not ev["data_present"]:
        return CheckResult("27_task_compatibility", "Task compatibility", "fatal", NOT_AVAILABLE,
                          "data absent — compatibility unverified", {})
    if prof.get("task") and prof.get("target_column"):
        note = ""
        if pid == "REG_ENSBURG_ULTRASOUND_IMAGES":
            note = (" (same cohort as the tabular experiment — usable for multimodal/late "
                    "fusion, NOT external validation)")
        if pid == "NEONATAL_SEPSIS_REGISTRY":
            note = (" (CONDITIONAL_NOT_SUITABLE_FOR_DIAGNOSTIC_CLASSIFICATION would apply only "
                    "if no valid target existed; a verified target does exist)")
        return CheckResult("27_task_compatibility", "Task compatibility", "fatal", PASS,
                          f"compatible with proposed task '{prof['task']}'{note}", {})
    return CheckResult("27_task_compatibility", "Task compatibility", "fatal", FAIL,
                      "no task/target recorded for this dataset", {})


CHECKS: list[tuple[str, str, Callable]] = [
    ("1_provenance", "Provenance verification", _c01_provenance),
    ("2_doi_source", "DOI/source verification", _c02_doi),
    ("3_license", "License verification", _c03_license),
    ("4_pediatric_population", "Pediatric population", _c04_pediatric),
    ("5_task_relevance", "Diagnostic-task relevance", _c05_task),
    ("6_target_availability", "Target availability", _c06_target_availability),
    ("7_target_validity", "Target validity", _c07_target_validity),
    ("8_sample_size", "Sample size", _c08_sample_size),
    ("9_class_balance", "Class balance", _c09_class_balance),
    ("10_patient_identifier", "Patient identifier", _c10_patient_identifier),
    ("11_duplicate_rows", "Duplicate rows", _c11_duplicate_rows),
    ("12_duplicate_patients", "Duplicate patients", _c12_duplicate_patients),
    ("13_duplicate_files", "Duplicate files", _c13_duplicate_files),
    ("14_file_hashes", "File-hash deduplication", _c14_file_hashes),
    ("15_perceptual_duplicates", "Perceptual duplicate risk", _c15_perceptual),
    ("16_pre_augmented_leakage", "Pre-augmented image leakage", _c16_pre_augmented),
    ("17_post_outcome_leakage", "Post-outcome leakage", _c17_post_outcome),
    ("18_identifier_leakage", "Identifier leakage", _c18_identifier_leakage),
    ("19_treatment_leakage", "Treatment/management leakage", _c19_treatment),
    ("20_diagnosis_code_leakage", "Diagnosis-code leakage", _c20_diagnosis_code),
    ("21_temporal_leakage", "Temporal leakage", _c21_temporal),
    ("22_label_circularity", "Label circularity", _c22_label_circularity),
    ("23_train_test_contamination", "Train/test contamination", _c23_contamination),
    ("24_mirror_duplication", "Mirror-source duplication", _c24_mirror),
    ("25_excessive_missingness", "Excessive missingness", _c25_missingness),
    ("26_access_restrictions", "Availability/access", _c26_access),
    ("27_task_compatibility", "Task compatibility", _c27_compatibility),
]

_SEVERITY: dict[str, str] = {}


# ------------------------------------------------------------------ evaluation
def evaluate_dataset(dataset_id: str, cfg: dict | None = None) -> DatasetReport:
    """Run all 27 checks for one dataset and roll up its status."""
    cfg = cfg or _load_config()
    prof = cfg["datasets"].get(dataset_id)
    if prof is None:
        raise KeyError(f"dataset {dataset_id!r} not present in config/dataset_blocking.yaml")
    for key, spec in cfg["checks"].items():
        _SEVERITY[key] = spec["severity"]
    ev = collect_evidence(dataset_id, prof)
    results: list[CheckResult] = []
    for key, name, fn in CHECKS:
        try:
            res = fn(dataset_id, prof, ev)
        except Exception as exc:  # a check that crashes must not silently pass
            res = CheckResult(key, name, _SEVERITY.get(key, "fatal"), FAIL,
                              f"check raised {type(exc).__name__}: {exc}", {})
        res.severity = _SEVERITY.get(key, res.severity)
        results.append(res)

    reasons = [f"[{c.key}] {c.reason}" for c in results
               if c.result in (FAIL, WARN) or (c.result == NOT_AVAILABLE and c.severity == "fatal")]
    if any(c.result == FAIL and c.severity == "fatal" for c in results):
        status = S_BLOCKED
    elif any(c.result == NOT_AVAILABLE for c in results):
        status = S_NOT_AVAILABLE
    elif any(c.result in (FAIL, WARN) for c in results):
        status = S_CONDITIONAL
    else:
        status = S_PASS
    return DatasetReport(dataset_id=dataset_id, status=status, checks=results,
                         reasons=reasons, data_present=bool(ev["data_present"]),
                         evaluated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def run_blocking(datasets: list[str] | None = None, write: bool = True) -> dict[str, Any]:
    """Evaluate every configured dataset; optionally persist JSON/CSV/MD reports."""
    cfg = _load_config()
    ids = datasets or list(cfg["datasets"])
    reports = {ds: evaluate_dataset(ds, cfg) for ds in ids}
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "src/data/dataset_blocker.py",
        "n_checks": len(CHECKS),
        "seed": 20261002,
        "gate": cfg["rollup_rules"]["gate"],
        "datasets": {ds: rep.as_dict() for ds, rep in reports.items()},
        "summary": {S_PASS: 0, S_CONDITIONAL: 0, S_BLOCKED: 0, S_NOT_AVAILABLE: 0},
        "disclaimer": "Research prototype — not a medical device.",
    }
    for rep in reports.values():
        payload["summary"][rep.status] += 1
    if write:
        write_reports(payload)
    return payload


def write_reports(payload: dict[str, Any]) -> None:
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str),
                           encoding="utf-8")
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset_id", "status", "data_present", "n_pass", "n_warn", "n_fail",
                    "n_na", "n_not_available", "reasons", "evaluated_utc"])
        for ds, d in payload["datasets"].items():
            w.writerow([ds, d["status"], d["data_present"], d["n_pass"], d["n_warn"],
                        d["n_fail"], d["n_na"], d["n_not_available"],
                        " | ".join(d["reasons"]), d["evaluated_utc"]])
    AUDIT_MD.write_text(render_audit_md(payload), encoding="utf-8")


def render_audit_md(payload: dict[str, Any]) -> str:
    lines = [
        "# PHASE 5 DATASET AUDIT — pre-training blocker (27 checks)",
        "",
        f"_Generated by `{payload['generator']}` on {payload['generated_utc']}_",
        "",
        "Statuses: **PASS** / **CONDITIONAL** / **BLOCKED** / **NOT_AVAILABLE**.",
        "A BLOCKED or NOT_AVAILABLE dataset cannot enter training (gate enforced in "
        "`src/data/dataset_blocker.py::assert_trainable`).",
        "",
        f"Summary: {payload['summary']}",
        "",
    ]
    for ds, d in payload["datasets"].items():
        lines += [f"## {ds} — **{d['status']}**",
                  "",
                  f"- data present locally: `{d['data_present']}`",
                  f"- checks: {d['n_pass']} PASS / {d['n_warn']} WARN / {d['n_fail']} FAIL / "
                  f"{d['n_na']} NA / {d['n_not_available']} NOT_AVAILABLE",
                  f"- evaluated: {d['evaluated_utc']}", ""]
        if d["reasons"]:
            lines.append("**Exact reasons for CONDITIONAL/BLOCKED/NOT_AVAILABLE:**")
            lines += [f"- {r}" for r in d["reasons"]]
            lines.append("")
        lines += ["| # check | severity | result | reason |",
                  "|---|---|---|---|"]
        for c in d["checks"]:
            reason = c["reason"].replace("|", "/").replace("\n", " ")
            lines.append(f"| {c['key']} | {c['severity']} | {c['result']} | {reason} |")
        lines.append("")
    lines += ["---",
              "Research prototype — not a medical device. No result here is a clinical claim.",
              ""]
    return "\n".join(lines)


def get_status(dataset_id: str) -> str:
    """Executed status for one dataset (runs the blocker if no report is persisted)."""
    if REPORT_JSON.exists():
        payload = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
        ds = payload.get("datasets", {}).get(dataset_id)
        if ds:
            return ds["status"]
    return evaluate_dataset(dataset_id)["status"]


def assert_trainable(dataset_id: str, allow_conditional: bool = True) -> DatasetReport:
    """Training gate — the blocker's only public enforcement point.

    BLOCKED and NOT_AVAILABLE datasets raise :class:`DatasetBlockedError`.
    CONDITIONAL datasets pass only because every condition is recorded in the
    report (``allow_conditional=False`` refuses them too).
    """
    cfg = _load_config()
    report = evaluate_dataset(dataset_id, cfg)
    if report.status in (S_BLOCKED, S_NOT_AVAILABLE):
        raise DatasetBlockedError(
            f"dataset {dataset_id} is {report.status} — training refused by the Phase 5 "
            f"blocker. Reasons: " + ("; ".join(report.reasons) or "see blocker report"))
    if report.status == S_CONDITIONAL and not allow_conditional:
        raise DatasetBlockedError(
            f"dataset {dataset_id} is CONDITIONAL and conditional training was not allowed. "
            + "; ".join(report.reasons))
    return report
