"""RAG indexer (Phase 3, section 3.8).

Builds ``data/interim/rag_index.sqlite`` from legally permitted sources only:

1. ``research/literature_review.csv`` — Phase 1 verified literature inventory
   (title, authors, year, journal, DOI, URL, method, results, limitations —
   metadata + short factual summaries, never full copyrighted text);
2. ``knowledge/*.yaml`` — public clinical guidance rules with their citations;
3. ``knowledge/source_manifest.json`` — DOI-verified source manifest entries;
4. user-supplied documents (``index_document``) — .md/.txt with mandatory
   provenance supplied by the caller.

Chunking is metadata-aware and deterministic (seedless by design: no random
splits). Every chunk must carry DOI/URL/citation — ``SourceChunk`` enforces it,
so *uncited chunks are never stored*. Full papers are not ingested.

Research prototype — not a medical device.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.preprocessing.regensburg import ROOT
from src.rag.schema import SourceChunk
from src.rag.store import ChunkStore, DEFAULT_DB

LITERATURE_CSV = ROOT / "research/literature_review.csv"
SOURCE_MANIFEST = ROOT / "knowledge/source_manifest.json"
KNOWLEDGE_DIR = ROOT / "knowledge"

_MAX_CHUNK_CHARS = 700


def _tid(*parts: Any) -> str:
    """Deterministic short id from parts (stable across processes)."""
    key = "|".join(str(p) for p in parts)
    return "chunk_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def _clip(text: str, limit: int = _MAX_CHUNK_CHARS) -> str:
    text = re.sub(r"\s+", " ", (text or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def chunk_literature(path: Path = LITERATURE_CSV) -> list[SourceChunk]:
    """One chunk per literature row: factual summary fields only."""
    if not path.exists():
        return []
    chunks: list[SourceChunk] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            title = (row.get("title") or "").strip()
            if not title:
                continue
            doi = (row.get("DOI") or "").strip()
            url = (row.get("URL") or "").strip()
            summary = " | ".join(
                f"{label}: {row.get(field, '').strip()}"
                for label, field in (("Question", "research_question"),
                                     ("Method", "method"),
                                     ("Result", "major_results"),
                                     ("Limitation", "limitations"))
                if (row.get(field) or "").strip())
            chunks.append(SourceChunk(
                chunk_id=_tid("lit", i, title),
                source_id=f"literature_review:{i}",
                text=_clip(summary or title),
                title=title,
                authors=(row.get("authors") or "").strip(),
                year=(row.get("year") or "").strip(),
                journal=(row.get("journal_or_conference") or "").strip(),
                doi=doi, url=url or (f"https://doi.org/{doi}" if doi else ""),
                section="inventory summary",
                license=("open-access record" if "Yes" == (row.get("open_access") or "").strip()
                         else "metadata/abstract summary only — not full text"),
                source_type="literature_metadata",
                citation=(row.get("citation") or "").strip(),
            ))
    return chunks


def chunk_knowledge(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[SourceChunk]:
    """One chunk per knowledge rule/threshold entry, with its citation."""
    chunks: list[SourceChunk] = []
    for path in sorted(knowledge_dir.glob("*.yaml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

        def walk(node: Any, trail: str) -> Iterable[tuple[str, dict[str, Any]]]:
            if isinstance(node, dict):
                for k, v in node.items():
                    if isinstance(v, dict) and ("rule" in v or "definition" in v):
                        yield f"{trail}.{k}" if trail else str(k), v
                    else:
                        yield from walk(v, f"{trail}.{k}" if trail else str(k))
            elif isinstance(node, list):
                for idx, v in enumerate(node):
                    yield from walk(v, f"{trail}[{idx}]")

        for trail, entry in walk(doc, ""):
            text_bits = [str(entry.get(k)) for k in ("rule", "definition", "sepsis", "septic_shock")
                         if entry.get(k)]
            if not text_bits:
                continue
            citation = str(entry.get("citation") or "")
            url = str(entry.get("url") or "")
            doi = str(entry.get("doi") or "")
            if not (citation or url or doi):
                continue  # uncited rule → never stored
            chunks.append(SourceChunk(
                chunk_id=_tid("knowledge", path.name, trail),
                source_id=f"knowledge:{path.name}",
                text=_clip(" ".join(text_bits)),
                title=f"{path.name} :: {trail}",
                year=str(entry.get("publication_year") or ""),
                journal=str(entry.get("source") or ""),
                doi=doi, url=url,
                section=trail,
                license="public guideline summary (see citation)",
                source_type="guideline_rule",
                citation=citation,
            ))
    return chunks


def chunk_source_manifest(path: Path = SOURCE_MANIFEST) -> list[SourceChunk]:
    """One chunk per ``knowledge/source_manifest.json`` entry."""
    if not path.exists():
        return []
    doc = json.loads(path.read_text(encoding="utf-8"))
    chunks: list[SourceChunk] = []
    for entry in doc.get("sources", []):
        title = str(entry.get("title") or "").strip()
        if not title:
            continue
        doi = str(entry.get("doi") or "")
        url = str(entry.get("url") or "")
        if not (doi or url):
            continue
        used_by = ", ".join(entry.get("used_by", []) or [])
        chunks.append(SourceChunk(
            chunk_id=_tid("manifest", entry.get("id", title)),
            source_id=f"manifest:{entry.get('id', '')}",
            text=_clip(f"{title}. {entry.get('verification', '')} "
                       f"Access: {entry.get('access', '')}. Used by: {used_by}"),
            title=title,
            authors=str(entry.get("authors") or ""),
            year=str(entry.get("publication_year") or ""),
            journal=str(entry.get("journal") or ""),
            doi=doi, url=url or (f"https://doi.org/{doi}" if doi else ""),
            section="source manifest",
            license=str(entry.get("access") or ""),
            source_type="manifest_entry",
            citation=f"{title} ({entry.get('publication_year', '')}). doi:{doi}" if doi else f"{title}. {url}",
        ))
    return chunks


def chunk_user_document(path: Path | str, source_id: str, title: str,
                        citation: str, doi: str = "", url: str = "",
                        license: str = "user-supplied (rights confirmed by supplier)",
                        section: str = "") -> list[SourceChunk]:
    """Paragraph-chunk a user-supplied .md/.txt document.

    The caller MUST supply provenance (citation and/or DOI/URL); without it the
    ``SourceChunk`` validator rejects every chunk and nothing is stored.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    paragraphs = [pa.strip() for pa in re.split(r"\n\s*\n", text) if pa.strip()]
    out: list[SourceChunk] = []
    buf = ""
    for para in paragraphs:
        if len(buf) + len(para) <= _MAX_CHUNK_CHARS:
            buf = f"{buf}\n{para}".strip()
            continue
        if buf:
            out.append(SourceChunk(chunk_id=_tid(source_id, len(out)), source_id=source_id,
                                   text=_clip(buf), title=title, doi=doi, url=url,
                                   section=section, license=license,
                                   source_type="user_document", citation=citation))
        buf = para
    if buf:
        out.append(SourceChunk(chunk_id=_tid(source_id, len(out)), source_id=source_id,
                               text=_clip(buf), title=title, doi=doi, url=url,
                               section=section, license=license,
                               source_type="user_document", citation=citation))
    return out


def build_index(db_path: Path | str = DEFAULT_DB, clear: bool = True) -> dict[str, Any]:
    """(Re)build the full index; returns provenance-count summary."""
    store = ChunkStore(db_path)
    if clear:
        store.clear()
    counts = {"literature": 0, "knowledge": 0, "manifest": 0}
    for chunk in chunk_literature():
        store.upsert(chunk)
        counts["literature"] += 1
    for chunk in chunk_knowledge():
        store.upsert(chunk)
        counts["knowledge"] += 1
    for chunk in chunk_source_manifest():
        store.upsert(chunk)
        counts["manifest"] += 1
    summary = {"db": str(db_path), "counts": counts, "total": store.count()}
    store.close()
    return summary


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    print(json.dumps(build_index(), indent=2))
