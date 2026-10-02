"""Build data/manifests/download_manifest.json from downloaded artifacts.

Run: python research/_scripts/make_download_manifest.py
Deterministic (hashing only, no network).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS = [
    {
        "path": "data/raw/regensburg_pediatric_appendicitis/app_data.xlsx",
        "dataset": "regensburg_pediatric_appendicitis",
        "url": "https://zenodo.org/api/records/7669442/files/app_data.xlsx/content",
        "doi": "10.5281/zenodo.7669442",
        "license": "CC-BY-NC-4.0",
    },
    {
        "path": "data/raw/regensburg_pediatric_appendicitis/ZENODO_README.md",
        "dataset": "regensburg_pediatric_appendicitis",
        "url": "https://zenodo.org/api/records/7669442/files/README.md/content",
        "doi": "10.5281/zenodo.7669442",
        "license": "CC-BY-NC-4.0",
    },
    {
        "path": "data/manifests/uci_dataset_938_metadata.json",
        "dataset": "regensburg_pediatric_appendicitis",
        "url": "https://archive.ics.uci.edu/api/dataset?id=938",
        "doi": "10.5281/zenodo.7669442",
        "license": "metadata",
    },
    {
        "path": "data/manifests/mendeley_5vdz5cftz7_metadata.json",
        "dataset": "neonatal_sepsis_registry",
        "url": "https://data.mendeley.com/public-api/datasets/5vdz5cftz7",
        "doi": "10.17632/5vdz5cftz7.1",
        "license": "not_exposed_by_api",
    },
    {
        "path": "data/raw/neonatal_sepsis/deid-nicu-sepsis-tta.csv",
        "dataset": "neonatal_sepsis_registry",
        "url": "https://data.mendeley.com/public-files/datasets/5vdz5cftz7/files/cac64abf-244f-46dd-976b-dd585ca89589/file_downloaded",
        "doi": "10.17632/5vdz5cftz7.1",
        "license": "not_exposed_by_api",
        "published_sha256": "54aa1dd81d40ed171282fdcca57de55bc2d8b8a3e2b3eec49d67ba1c1ba4c457",
    },
    {
        "path": "data/manifests/mendeley_3tx8xymdsv_metadata.json",
        "dataset": "child_pneumonia_dataset",
        "url": "https://data.mendeley.com/public-api/datasets/3tx8xymdsv",
        "doi": "10.17632/3tx8xymdsv.1",
        "license": "cc_by_4_0_per_repository_description",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    records = []
    for a in ARTIFACTS:
        p = ROOT / a["path"]
        rec = dict(a)
        if p.exists():
            rec.update(
                present=True,
                bytes=p.stat().st_size,
                sha256=sha256(p),
            )
            if "published_sha256" in rec:
                rec["sha256_matches_published"] = rec["sha256"] == rec["published_sha256"]
        else:
            rec["present"] = False
        records.append(rec)

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": "research/_scripts/make_download_manifest.py",
        "artifacts": records,
        "not_downloaded": [
            {
                "path": "data/raw/regensburg_pediatric_appendicitis/US_Pictures.zip",
                "url": "https://zenodo.org/api/records/7669442/files/US_Pictures.zip/content",
                "bytes_upstream": 523034730,
                "reason": "imaging branch deferred; Phase 1 is tabular-only",
            },
            {
                "dataset": "child_pneumonia_dataset",
                "url": "https://data.mendeley.com/datasets/3tx8xymdsv/1",
                "reason": "Phase C blocked pending image audit (pre-augmented files + duplicate filenames)",
            },
        ],
    }
    dest = ROOT / "data/manifests/download_manifest.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {dest}")
    for r in records:
        flag = "OK" if r.get("present") else "MISSING"
        match = r.get("sha256_matches_published")
        extra = f" sha256==published:{match}" if match is not None else ""
        print(f"  [{flag}] {r['path']}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
