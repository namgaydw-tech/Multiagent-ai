"""RAG data schemas (Phase 3, section 3.8).

Every stored chunk must be **cited**: ``CitationRequiredChunk`` refuses any
text without at least one of DOI / URL / explicit citation, so the store can
never hold an uncited passage. Metadata preserved per the project brief:
title, authors, year, DOI, source, section, license, URL.

Only legally permitted material is indexed: open-access papers, abstracts and
metadata summaries, public clinical guidance (``knowledge/``), and
user-supplied documents. Full copyrighted texts are never ingested.

Research prototype — not a medical device.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class SourceChunk(BaseModel):
    """One retrievable passage with complete provenance."""

    chunk_id: str
    source_id: str = ""
    text: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: str = ""
    year: str = ""
    journal: str = ""
    doi: str = ""
    url: str = ""
    section: str = ""
    license: str = ""
    source_type: Literal["literature_metadata", "guideline_rule", "manifest_entry",
                         "user_document", "dataset_document"] = "literature_metadata"
    citation: str = ""

    @model_validator(mode="after")
    def _require_citation(self) -> "SourceChunk":
        """Reject uncited chunks — provenance is mandatory at storage time."""
        if not (self.doi.strip() or self.url.strip() or self.citation.strip()):
            raise ValueError(
                f"chunk {self.chunk_id!r} has no DOI, URL or citation — uncited chunks "
                "must never be stored (rag provenance policy)")
        if not self.citation.strip():
            self.citation = f"{self.title} ({self.year}). {('doi:' + self.doi) if self.doi else self.url}"
        return self

    def prompt_text(self) -> str:
        """Text actually indexed/shown: passage + compact provenance line."""
        prov = " | ".join(x for x in [self.title, self.year, self.journal,
                                      f"doi:{self.doi}" if self.doi else "",
                                      self.url] if x)
        return f"{self.text}\n[{prov}]"


class QueryHit(BaseModel):
    """Retrieval result: chunk + score + how it was found."""

    chunk: SourceChunk
    score: float = Field(ge=0.0, le=1.0)
    method: Literal["vector", "keyword_fallback", "both"] = "vector"
    rank: int = 0

    def as_retrieved(self, used_by: list[str] | None = None) -> dict[str, Any]:
        """Project into the agents' ``RetrievedSource`` schema shape."""
        c = self.chunk
        return {
            "title": c.title, "authors": c.authors, "year": c.year,
            "source": c.journal or c.source_type, "doi": c.doi, "url": c.url,
            "section": c.section, "license": c.license,
            "chunk_text": c.text, "score": round(self.score, 4),
            "used_by": list(used_by or []),
        }
