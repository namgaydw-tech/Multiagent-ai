# Reproducibility

## Environment

- Python 3.11+ (verified locally on Python 3.13.11, Windows/Git Bash).
- Dependencies pinned in `requirements.txt`; project metadata in `pyproject.toml`.
- No notebook is required for the audited pipeline; notebooks are for exploration only and must
  re-run the same functions from `src/`.

```bash
python -m venv .venv && source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Data provenance (Phase 1, verifiable)

| Artifact | Source URL | Integrity |
|---|---|---|
| `data/raw/regensburg_pediatric_appendicitis/app_data.xlsx` | `https://zenodo.org/api/records/7669442/files/app_data.xlsx/content` | byte size 212,771 recorded in `data/manifests/download_manifest.json` |
| `data/raw/regensburg_pediatric_appendicitis/ZENODO_README.md` | Zenodo record README | size 2,235 |
| `data/manifests/uci_dataset_938_metadata.json` | `https://archive.ics.uci.edu/api/dataset?id=938` | machine metadata, cached |
| `data/raw/neonatal_sepsis/deid-nicu-sepsis-tta.csv` | Mendeley public files API for 5vdz5cftz7 v1 | SHA-256 published by the API: `54aa1dd81d40ed171282fdcca57de55bc2d8b8a3e2b3eec49d67ba1c1ba4c457` |
| Mendeley metadata JSONs | `https://data.mendeley.com/public-api/datasets/<id>` | cached in `data/manifests/` |

Regenerate the checksum locally: `python research/_scripts/make_download_manifest.py`
(re-runs hashing only; no re-download).

## Rerunning Phase 1

```bash
# 1. dataset audit (writes outputs/metrics/regensburg_audit.json + docs/DATASET_AUDIT.md)
python -m src.data.audit_regensburg

# 2. literature verification against OpenAlex (network; results cached)
python research/_scripts/verify_literature.py

# 3. tests (audit + inventory + citation integrity)
python -m pytest tests/ -q
```

## Determinism rules for future phases

1. Every experiment writes `config hash`, `git commit`, `random seed`, `LLM model id`,
   `prompt hash`, `temperature=0 where supported` into the audit JSON.
2. Splits are created once, saved to `data/interim/splits/`, and reused; no re-splitting per run.
3. Preprocessing is fit on training data only; artifacts saved with fitted state.
4. Calibration uses validation data only; test labels are touched exactly once at final evaluation.
5. LLM calls are logged with token counts, latency, and raw responses under `outputs/audits/`.
6. Figures are generated exclusively from stored metric files under `outputs/metrics/` — a figure
   can always be traced to a run ID.
7. The audit trail schema (project brief §19) is validated per case; incomplete stages fail loudly.

## What "verified" means here

- Dataset facts come from running `src/data/audit_regensburg.py` on the downloaded file, not from
  memory or abstracts.
- Citations are DOI-verified via OpenAlex/Crossref with a cached response
  (`research/_scripts/openalex_cache.json`).
- Anything not verified is either absent or explicitly labelled as planned/blocked.
