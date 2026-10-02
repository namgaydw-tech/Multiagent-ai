"""SQLite chunk store with FTS5 keyword search (Phase 3, section 3.8).

Lightweight, dependency-free persistence for the RAG index (no external
vector DB — the architecture brief asks for a lightweight store; FAISS/Chroma
remain optional adapters). The database lives at ``data/interim/rag_index.sqlite``
(regenerable, git-ignored).

* ``chunks`` table holds the full provenance columns;
* ``chunks_fts`` (FTS5, external-content) powers keyword retrieval — the
  fallback path when vector scores are weak;
* every write goes through :class:`src.rag.schema.SourceChunk`, so uncited
  chunks cannot enter the store even via raw SQL helpers.

Research prototype — not a medical device.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.preprocessing.regensburg import ROOT
from src.rag.schema import SourceChunk

DEFAULT_DB = ROOT / "data/interim/rag_index.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    source_id TEXT DEFAULT '',
    text TEXT NOT NULL,
    title TEXT NOT NULL,
    authors TEXT DEFAULT '',
    year TEXT DEFAULT '',
    journal TEXT DEFAULT '',
    doi TEXT DEFAULT '',
    url TEXT DEFAULT '',
    section TEXT DEFAULT '',
    license TEXT DEFAULT '',
    source_type TEXT DEFAULT 'literature_metadata',
    citation TEXT DEFAULT ''
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, title, authors, journal,
    content='chunks', content_rowid='rowid'
);
"""


class ChunkStore:
    """SQLite-backed chunk store (FTS5 with graceful LIKE fallback)."""

    def __init__(self, db_path: Path | str = DEFAULT_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._fts_enabled = True
        try:
            self.conn.executescript(_SCHEMA)
        except sqlite3.OperationalError:  # pragma: no cover - FTS5 unavailable build
            self._fts_enabled = False
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, "
                "source_id TEXT, text TEXT, title TEXT, authors TEXT, year TEXT, "
                "journal TEXT, doi TEXT, url TEXT, section TEXT, license TEXT, "
                "source_type TEXT, citation TEXT)")
        self.conn.commit()

    # ------------------------------------------------------------------ writes
    def upsert(self, chunk: SourceChunk) -> None:
        """Insert or replace one validated (hence cited) chunk."""
        old_row = self.conn.execute("SELECT rowid FROM chunks WHERE chunk_id=?",
                                    (chunk.chunk_id,)).fetchone()
        if old_row and self._fts_enabled:
            self.conn.execute("DELETE FROM chunks_fts WHERE rowid=?", (old_row[0],))
        self.conn.execute(
            "INSERT OR REPLACE INTO chunks (chunk_id, source_id, text, title, authors, "
            "year, journal, doi, url, section, license, source_type, citation) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (chunk.chunk_id, chunk.source_id, chunk.text, chunk.title, chunk.authors,
             chunk.year, chunk.journal, chunk.doi, chunk.url, chunk.section,
             chunk.license, chunk.source_type, chunk.citation))
        if self._fts_enabled:
            self.conn.execute(
                "INSERT INTO chunks_fts (rowid, text, title, authors, journal) "
                "SELECT rowid, text, title, authors, journal FROM chunks WHERE chunk_id=?",
                (chunk.chunk_id,))
        self.conn.commit()

    def clear(self) -> None:
        """Drop all chunks (used when rebuilding the index)."""
        self.conn.execute("DELETE FROM chunks")
        if self._fts_enabled:
            self.conn.execute("DELETE FROM chunks_fts")
        self.conn.commit()

    # ------------------------------------------------------------------ reads
    def count(self) -> int:
        """Number of stored chunks."""
        return int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def all_chunks(self) -> list[SourceChunk]:
        """Every chunk (vector index builds from this)."""
        rows = self.conn.execute("SELECT * FROM chunks ORDER BY chunk_id").fetchall()
        return [SourceChunk(**dict(r)) for r in rows]

    def get(self, chunk_id: str) -> SourceChunk | None:
        """Fetch one chunk by id."""
        row = self.conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        return SourceChunk(**dict(row)) if row else None

    def search_keyword(self, query: str, limit: int = 10) -> list[tuple[str, float]]:
        """Keyword search: FTS5 MATCH with a sanitized LIKE fallback.

        Returns ``(chunk_id, score)`` pairs; score in [0, 1].
        """
        terms = [t for t in "".join(ch if ch.isalnum() or ch.isspace() else " "
                                    for ch in query).split() if len(t) > 1]
        if not terms:
            return []
        if self._fts_enabled:
            match = " OR ".join(f'"{t}"' for t in terms)
            try:
                rows = self.conn.execute(
                    "SELECT c.chunk_id AS chunk_id, bm25(chunks_fts) AS rank "
                    "FROM chunks_fts JOIN chunks c ON c.rowid = chunks_fts.rowid "
                    "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?", (match, limit)).fetchall()
                if rows:
                    ranks = [-float(r["rank"]) for r in rows]  # bm25 lower=better → negate
                    top = max(ranks) if ranks and max(ranks) > 0 else 1.0
                    return [(r["chunk_id"], max(0.0, min(1.0, rk / top)))
                            for r, rk in zip(rows, ranks)]
            except sqlite3.OperationalError:  # pragma: no cover - malformed MATCH
                pass
        # LIKE fallback over concatenated searchable text
        hits: list[tuple[str, float]] = []
        for term in terms:
            rows = self.conn.execute(
                "SELECT chunk_id FROM chunks WHERE lower(text || ' ' || title) "
                "LIKE ? LIMIT ?", (f"%{term.lower()}%", limit)).fetchall()
            for r in rows:
                cid = r["chunk_id"]
                existing = dict(hits)
                existing[cid] = min(1.0, existing.get(cid, 0.0) + 1.0 / len(terms))
                hits = list(existing.items())
        return sorted(hits, key=lambda x: -x[1])[:limit]

    def close(self) -> None:
        """Close the underlying connection."""
        self.conn.close()


def ensure_store(db_path: Path | str = DEFAULT_DB) -> ChunkStore:
    """Convenience factory (used by orchestrator and CLI scripts)."""
    return ChunkStore(db_path)
