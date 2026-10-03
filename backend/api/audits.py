"""Audit browsing endpoints (the UI's inspector for persisted run transcripts).

Only the top-level case directories of ``outputs/audits/`` are listed (the
experiment batch tree with thousands of runs is addressed by the metrics files
instead). Case ids and file names are strictly validated — no path traversal.

Research prototype — not a medical device.
"""

from __future__ import annotations

import re

from fastapi import APIRouter, HTTPException

from backend.api._shared import (AUDITS_DIR, DISCLAIMER, ApiUnavailable,
                                  load_json, safe_case_dir)

router = APIRouter(prefix="/audits", tags=["audits"])

_STAGE_FILE_RE = re.compile(r"^stage_\d{2}_(input|output|meta)\.json$")


@router.get("")
def list_audits(limit: int = 100) -> dict:
    """Top-level case audit directories (newest first, bounded)."""
    limit = max(1, min(limit, 200))
    if not AUDITS_DIR.exists():
        return {"available": True, "cases": [], "n": 0, "disclaimer": DISCLAIMER}
    entries = []
    for path in AUDITS_DIR.iterdir():
        if not path.is_dir() or path.name == "experiments":
            continue
        audit_path = path / "audit.json"
        if not audit_path.exists():
            continue
        info = {"case_id": path.name, "audit_available": True,
                "files": sorted(p.name for p in path.glob("*.json")),
                "mtime": int(path.stat().st_mtime)}
        try:
            audit = load_json(audit_path)
            info.update(
                completed_stages=audit.get("completed_stages"),
                provider_backend=audit.get("provider_backend"),
                ablation=audit.get("ablation"),
                rag_enabled=audit.get("rag_enabled"),
                started_utc=audit.get("started_utc"),
                has_errors=bool(audit.get("errors")))
        except ApiUnavailable as exc:
            info.update(audit_available=False, reason=str(exc))
        entries.append(info)
    entries.sort(key=lambda e: -e["mtime"])
    return {"available": True, "n": len(entries), "cases": entries[:limit],
            "disclaimer": DISCLAIMER}


@router.get("/{case_id}")
def get_audit(case_id: str) -> dict:
    """Full ``audit.json`` for one case plus its file inventory."""
    try:
        directory = safe_case_dir(case_id)
    except ApiUnavailable as exc:
        raise HTTPException(status_code=404,
                            detail={"detail": str(exc),
                                    "hint": exc.hint})
    audit_path = directory / "audit.json"
    if not audit_path.exists():
        raise HTTPException(status_code=404,
                            detail={"detail": f"no audit.json for {case_id!r}",
                                    "hint": "Run a case first"})
    return {"case_id": case_id, "audit": load_json(audit_path),
            "files": sorted(p.name for p in directory.glob("*.json")),
            "disclaimer": DISCLAIMER}


@router.get("/{case_id}/{file_name}")
def get_stage_file(case_id: str, file_name: str) -> dict:
    """One stage transcript (input/output/meta) or the audit itself."""
    if file_name != "audit.json" and not _STAGE_FILE_RE.match(file_name):
        raise HTTPException(
            status_code=422,
            detail={"detail": f"invalid file name {file_name!r}",
                    "hint": "Use audit.json or stage_0N_input|output|meta.json"})
    try:
        directory = safe_case_dir(case_id)
    except ApiUnavailable as exc:
        raise HTTPException(status_code=404,
                            detail={"detail": str(exc), "hint": exc.hint})
    path = directory / file_name
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail={"detail": f"{file_name} not found for {case_id!r}",
                    "hint": "stage files are only written by full runs "
                            "(scripts/run_phase3.py or the UI agent runner)"})
    return {"case_id": case_id, "file": file_name, "content": load_json(path),
            "disclaimer": DISCLAIMER}
