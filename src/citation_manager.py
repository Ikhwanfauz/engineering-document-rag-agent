"""Create consistent document and page citations from retrieved evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.retriever import RetrievedChunk


@dataclass(frozen=True, slots=True)
class Citation:
    """One citation shown with a generated answer."""

    document_id: str
    source_name: str
    page_number: int
    page_label: str
    excerpt: str

    @property
    def label(self) -> str:
        """Return the human-readable physical-page citation."""
        return f"{self.source_name}, page {self.page_number}"


def build_citations(
    chunks: Iterable[RetrievedChunk],
    *,
    excerpt_max_chars: int = 300,
) -> tuple[Citation, ...]:
    """Create unique page citations while preserving retrieval order."""
    if excerpt_max_chars < 4:
        raise ValueError("excerpt_max_chars must be at least 4")

    citations: list[Citation] = []
    seen_pages: set[tuple[str, int]] = set()

    for chunk in chunks:
        page_key = (chunk.document_id, chunk.page_number)

        if page_key in seen_pages:
            continue

        seen_pages.add(page_key)
        citations.append(
            Citation(
                document_id=chunk.document_id,
                source_name=chunk.source_name,
                page_number=chunk.page_number,
                page_label=chunk.page_label,
                excerpt=_create_excerpt(
                    chunk.text,
                    max_chars=excerpt_max_chars,
                ),
            )
        )

    return tuple(citations)


def _create_excerpt(text: str, *, max_chars: int) -> str:
    """Normalize whitespace and shorten one evidence excerpt."""
    normalized = " ".join(text.split())

    if len(normalized) <= max_chars:
        return normalized

    return normalized[: max_chars - 3].rstrip() + "..."
