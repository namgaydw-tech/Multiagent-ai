"""Hybrid retriever: vector search + keyword fallback (Phase 3, section 3.8).

* **Vector path** — TF-IDF (word 1–2 grams, deterministic, offline) over all
  stored chunks with cosine similarity. A neural sentence-transformer would be
  the production upgrade, but this project runs offline with no model
  downloads; TF-IDF is honest, reproducible and dependency-free (scikit-learn
  is already required by Phase 2).
* **Keyword path** — SQLite FTS5/BM25 (or LIKE fallback) from the store.
* **Fusion** — best-normalised cosine score, with keyword-only hits entering
  at 0.55× their normalised keyword score (documented constant), so exact
  term matches survive even when vector scores are flat.

Every hit carries full provenance (title/authors/year/DOI/URL/license) and the
retrieval method, so the orchestrator can show *why* a source was retrieved.

Research prototype — not a medical device.
"""

from __future__ import annotations

import math
import re
from typing import Any

from src.rag.schema import QueryHit, SourceChunk
from src.rag.store import ChunkStore, DEFAULT_DB

_KEYWORD_ONLY_WEIGHT = 0.55
_VECTOR_TERMS = (1, 2)


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric unigrams+bigrams (deterministic mini-vectorizer)."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


class Retriever:
    """In-memory vector index over a :class:`ChunkStore` (rebuilt per process)."""

    def __init__(self, store: ChunkStore | None = None,
                 db_path: Any = DEFAULT_DB) -> None:
        self.store = store or ChunkStore(db_path)
        self.chunks: list[SourceChunk] = []
        self._tfs: list[dict[str, float]] = []
        self._df: dict[str, int] = {}
        self._norms: list[float] = []
        self._idf: dict[str, float] = {}
        self.refresh()

    def refresh(self) -> int:
        """(Re)load all chunks and rebuild the TF-IDF vectors. Returns n chunks."""
        self.chunks = self.store.all_chunks()
        self._tfs, self._df = [], {}
        for chunk in self.chunks:
            terms = _tokenize(chunk.prompt_text())
            tf: dict[str, int] = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1
            self._tfs.append({t: c for t, c in tf.items()})
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1
        n = max(1, len(self.chunks))
        self._idf = {t: math.log((1 + n) / (1 + d)) + 1.0 for t, d in self._df.items()}
        self._norms = []
        for tf in self._tfs:
            vec = {t: (1 + math.log(c)) * self._idf.get(t, 0.0) for t, c in tf.items()}
            self._norms.append(math.sqrt(sum(v * v for v in vec.values())) or 1.0)
        return len(self.chunks)

    # ------------------------------------------------------------------ search
    def _vector_scores(self, query: str, top: int) -> list[tuple[int, float]]:
        q_tf: dict[str, int] = {}
        for t in _tokenize(query):
            q_tf[t] = q_tf.get(t, 0) + 1
        q_vec = {t: (1 + math.log(c)) * self._idf.get(t, 0.0) for t, c in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0
        scored: list[tuple[int, float]] = []
        for i, (tf, dnorm) in enumerate(zip(self._tfs, self._norms)):
            dot = 0.0
            small, big = (tf, q_vec) if len(tf) <= len(q_vec) else (q_vec, tf)
            for t, v in small.items():
                w = big.get(t)
                if w is not None:
                    dot += v * w
            score = dot / (dnorm * q_norm)
            if score > 0:
                scored.append((i, score))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top]

    def retrieve(self, query: str, k: int = 4) -> list[QueryHit]:
        """Top-k hybrid hits with method + normalised scores in [0, 1]."""
        query = (query or "").strip()
        if not query or not self.chunks:
            return []
        vec = self._vector_scores(query, top=k * 3)
        max_cos = vec[0][1] if vec else 1.0
        best: dict[int, tuple[float, str]] = {
            i: (score / max_cos, "vector") for i, score in vec[:k]
        }
        # keyword fallback: exact-term matches that vector search may have missed
        for cid, kscore in self.store.search_keyword(query, limit=k * 3):
            idx = next((i for i, c in enumerate(self.chunks) if c.chunk_id == cid), None)
            if idx is None:
                continue
            kscore_n = max(0.0, min(1.0, kscore))
            if idx in best:
                prev, method = best[idx]
                best[idx] = (max(prev, kscore_n), "both")
            else:
                best[idx] = (min(0.99, kscore_n * _KEYWORD_ONLY_WEIGHT), "keyword_fallback")
        ordered = sorted(best.items(), key=lambda kv: (-kv[1][0], kv[0]))[:k]
        return [QueryHit(chunk=self.chunks[i], score=round(min(max(s, 0.0), 1.0), 4),
                         method=method, rank=rank)  # type: ignore[arg-type]
                for rank, (i, (s, method)) in enumerate(ordered, start=1)]

    def retrieve_for_agents(self, query: str, k: int = 4,
                            used_by: list[str] | None = None) -> list[dict[str, Any]]:
        """Hits projected into the agents' ``RetrievedSource`` dicts."""
        return [h.as_retrieved(used_by=used_by) for h in self.retrieve(query, k)]

    def close(self) -> None:
        """Close the underlying store connection."""
        self.store.close()


def build_query(differential_candidates: list[str], red_flags: list[str],
                leading_hypothesis: str | None = None) -> str:
    """Deterministic retrieval query assembled from stage outputs only."""
    parts = list(differential_candidates[:3]) + list(red_flags[:2])
    if leading_hypothesis:
        parts.append(leading_hypothesis)
    return "; ".join(p for p in parts if p)
