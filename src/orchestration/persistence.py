"""Per-case audit persistence for the orchestrator (Phase 3, section 3.9).

Every stage is written to disk **before** the next stage starts, so runs are
inspectable, resumable and schema-auditable (docs/ARCHITECTURE.md):

    outputs/audits/<case_id>/stage_0N_input.json    truthful input snapshot
    outputs/audits/<case_id>/stage_0N_output.json   validated stage payload
    outputs/audits/<case_id>/stage_0N_meta.json     prompt_hash, model_version,
                                                    tokens, latency, retries,
                                                    backend, fallback/error, timing
    outputs/audits/<case_id>/audit.json             case-level audit trail
                                                    (project brief section 19)

``load_completed()`` powers resume: completed stages are reloaded from disk and
validated against their Pydantic schema instead of being re-executed.

All content is derived from the case run itself — nothing here invents fields.
Generated audit files are gitignored (repository cleanliness) but always written
locally by the engine.

Research prototype — not a medical device.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.agents.schemas import validate_stage
from src.orchestration.state import CaseState, clean_json
from src.orchestration.stages import STAGE_KEYS

ROOT = Path(__file__).resolve().parents[2]
AUDITS_ROOT = ROOT / "outputs/audits"

_SAFE = re.compile(r"[^A-Za-z0-9_.\-]+")


def case_dir(case_id: str, root: Path | None = None) -> Path:
    """Sanitized per-case audit directory (path-traversal safe)."""
    safe = _SAFE.sub("_", str(case_id)).strip("._") or "case"
    return (root or AUDITS_ROOT) / safe


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean_json(payload), indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(path)  # atomic-ish: readers never see a half-written snapshot


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_paths(directory: Path, stage_no: int) -> dict[str, Path]:
    """The three files for one stage (input / output / meta)."""
    stem = f"stage_{stage_no:02d}"
    return {
        "input": directory / f"{stem}_input.json",
        "output": directory / f"{stem}_output.json",
        "meta": directory / f"{stem}_meta.json",
    }


def save_stage(directory: Path, stage_no: int, key: str, inputs: dict[str, Any],
               payload: BaseModel | dict[str, Any], meta: dict[str, Any]) -> dict[str, str]:
    """Persist one completed stage (input snapshot + payload + provenance meta)."""
    paths = stage_paths(directory, stage_no)
    body = payload.model_dump() if isinstance(payload, BaseModel) else payload
    _write_json(paths["input"], inputs)
    _write_json(paths["output"], body)
    _write_json(paths["meta"], {"stage": stage_no, "key": key, **meta})
    return {k: str(v.relative_to(directory)) for k, v in paths.items()}


def load_completed(directory: Path) -> dict[int, dict[str, Any]]:
    """Reload completed stages for resume: ``{stage_no: {inputs, payload, meta}}``.

    A stage counts as complete only when all three files exist; the payload is
    re-validated against its schema (a corrupted file fails loudly, it is never
    silently skipped or defaulted).
    """
    out: dict[int, dict[str, Any]] = {}
    for stage_no, (key, _) in sorted(STAGE_KEYS.items()):
        paths = stage_paths(directory, stage_no)
        if not all(p.exists() for p in paths.values()):
            continue
        payload = validate_stage(key, _read_json(paths["output"]))
        out[stage_no] = {
            "inputs": _read_json(paths["input"]),
            "payload": payload,
            "meta": _read_json(paths["meta"]),
        }
    return out


def build_audit(state: CaseState, started_utc: str, completed_utc: str,
                total_latency_s: float, provider_backend: str,
                provider_model_version: str) -> dict[str, Any]:
    """Case-level audit trail (project brief section 19) from the executed state."""
    stages = []
    total_tokens = 0
    for stage_no, (key, name) in sorted(STAGE_KEYS.items()):
        meta = state.stage_meta.get(key)
        if meta is None:
            stages.append({"stage": stage_no, "key": key, "name": name,
                           "status": "not_run",
                           "error": state.errors.get(key)})
            continue
        call = meta.get("call", {})
        tokens = int(call.get("input_tokens", 0) or 0) + int(call.get("output_tokens", 0) or 0)
        total_tokens += tokens
        stages.append({
            "stage": stage_no, "key": key, "name": name,
            "status": "error" if key in state.errors else "ok",
            "backend": call.get("backend"),
            "model_version": call.get("model_version"),
            "prompt_hash": call.get("prompt_hash"),
            "input_tokens": call.get("input_tokens"),
            "output_tokens": call.get("output_tokens"),
            "latency_s": call.get("latency_s"),
            "attempts": call.get("attempts"),
            "fallback_reason": call.get("fallback_reason"),
            "error": state.errors.get(key) or call.get("error"),
            "schema_name": call.get("schema_name"),
            "timestamp": call.get("timestamp"),
            "files": meta.get("files"),
        })
    return {
        "schema_version": 1,
        "case_id": state.case_id,
        "config_hash": state.config_hash,
        "seed": state.seed,
        "ablation": state.ablation,
        "rag_enabled": state.rag_enabled,
        "provider_backend": provider_backend,
        "provider_model_version": provider_model_version,
        "visibility_policy": {str(n): state.policy.description(n) for n in range(1, 9)},
        "stage_order": [name for _, (_, name) in sorted(STAGE_KEYS.items())],
        "completed_stages": sorted(state.completed_stages),
        "errors": state.errors,
        "totals": {"latency_s": round(total_latency_s, 4), "tokens": total_tokens,
                   "provider_calls": sum(1 for m in state.stage_meta.values() if m.get("call"))},
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "stages": stages,
        "disclaimer": "Research prototype — not a medical device; audit trail of a "
                      "bias study, not a clinical record.",
    }


def save_audit(directory: Path, audit: dict[str, Any]) -> Path:
    """Write ``audit.json`` for the case (atomic replace)."""
    path = directory / "audit.json"
    _write_json(path, audit)
    return path


def audit_summary(directory: Path) -> dict[str, Any] | None:
    """Load ``audit.json`` if present (used by tests, CLI and the UI backend)."""
    path = directory / "audit.json"
    return _read_json(path) if path.exists() else None


class StageTimer:
    """Context helper: measures wall-clock latency of one stage transition."""

    def __enter__(self) -> "StageTimer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.latency_s = time.perf_counter() - self.t0


def utc_now() -> str:
    """ISO-8601 UTC timestamp for audit metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
