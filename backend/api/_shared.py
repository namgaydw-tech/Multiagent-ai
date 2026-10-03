"""Shared helpers for backend endpoints (lazy heavy imports + safe file IO).

The Phase 2 artifacts (dataset, LightGBM, SHAP) are loaded on first use and
cached process-wide; a lock serializes model/agent execution so concurrent UI
requests cannot interleave a SHAP explanation with a pipeline run.

All JSON readers either return the parsed payload or raise ``ApiUnavailable``
with an actionable hint — endpoints translate that into a structured
``available: false`` response or a 500 with the real reason. Nothing is ever
substituted with placeholder values.

Research prototype — not a medical device.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]   # repo root (…/backend/api/_shared.py)
METRICS_DIR = ROOT / "outputs/metrics"
FIGURES_DIR = ROOT / "outputs/figures"
MODELS_DIR = ROOT / "outputs/models"
AUDITS_DIR = ROOT / "outputs/audits"

DISCLAIMER = ("Research prototype — not a medical device; outputs are from a "
              "bias study, not clinical evidence.")


class ApiUnavailable(RuntimeError):
    """A required artifact is missing or unreadable; carries a user hint."""

    def __init__(self, message: str, hint: str = ""):
        super().__init__(message)
        self.hint = hint


# Reentrant: endpoint handlers hold run_lock() while lazily fetching the
# predictor/retriever, which acquire the same lock (same thread → RLock).
_lock = threading.RLock()
_local = threading.local()          # per-thread caches (sqlite connections)
_predictor: Any = None


def get_predictor():
    """Cached Phase 2 :class:`CasePredictor` (loads artifacts on first call)."""
    global _predictor
    with _lock:
        if _predictor is None:
            try:
                from src.models.inference import get_predictor as _gp
                _predictor = _gp()
            except FileNotFoundError as exc:
                raise ApiUnavailable(
                    f"Phase 2 artifacts missing: {exc}",
                    "Run: python scripts/run_phase2.py") from exc
        return _predictor


def get_retriever():
    """Per-request RAG retriever (thread-local: SQLite connections are not
    shared across the FastAPI threadpool; building the TF-IDF index for 72
    chunks takes milliseconds)."""
    retriever = getattr(_local, "retriever", None)
    if retriever is None:
        from src.rag.retriever import Retriever
        retriever = Retriever()
        if not retriever.chunks:
            from src.rag.indexer import build_index
            build_index()
            retriever.refresh()
        _local.retriever = retriever
    return retriever


def run_lock() -> threading.Lock:
    """Global lock for model inference and agent runs."""
    return _lock


def load_json(path: Path, *, hint: str = "") -> Any:
    """Read a JSON artifact or raise :class:`ApiUnavailable`."""
    if not path.exists():
        raise ApiUnavailable(f"{path.name} not found at {path}",
                             hint or "Run the phase that produces this file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiUnavailable(f"{path.name} is not valid JSON: {exc}",
                             "Regenerate the file") from exc


def load_json_optional(path: Path) -> tuple[Any | None, str | None]:
    """``(payload, None)`` when present/readable, else ``(None, reason)``."""
    try:
        return load_json(path), None
    except ApiUnavailable as exc:
        return None, str(exc)


def safe_case_dir(case_id: str) -> Path:
    """Resolve an audit case directory with traversal protection."""
    safe = "".join(ch for ch in case_id if ch.isalnum() or ch in "_.-").strip("._")
    if not safe or safe != case_id:
        raise ApiUnavailable(f"invalid case_id {case_id!r}",
                             "Use the exact id shown in the audit list")
    path = AUDITS_DIR / safe
    if not path.is_dir():
        raise ApiUnavailable(f"audit directory not found: {case_id}",
                             "Pick a case from GET /api/audits")
    return path
