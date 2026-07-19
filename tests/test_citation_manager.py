"""Tests for document and page citation handling."""

import pytest

from src.citation_manager import build_citations
from src.retriever import RetrievedChunk


def make_chunk(
    *,
    chunk_id: str = "chunk-50-0",
    document_id: str = "document-123",
    source_name: str = "service-manual.pdf",
    page_number: int = 50,
    page_label: str = "50",
    text: str = "Support the joint while removing the clamp.",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_name=source_name,
        page_number=page_number,
        page_label=page_label,
        chunk_index=0,
        text=text,
        distance=0.12,
        similarity_score=0.88,
    )


def test_citation_preserves_source_page_and_excerpt() -> None:
    citation = build_citations([make_chunk()])[0]

    assert citation.document_id == "document-123"
    assert citation.source_name == "service-manual.pdf"
    assert citation.page_number == 50
    assert citation.page_label == "50"
    assert citation.excerpt == "Support the joint while removing the clamp."
    assert citation.label == "service-manual.pdf, page 50"


def test_duplicate_document_pages_are_removed() -> None:
    citations = build_citations(
        [
            make_chunk(chunk_id="chunk-50-0"),
            make_chunk(
                chunk_id="chunk-50-1",
                text="Another chunk from the same physical page.",
            ),
        ]
    )

    assert len(citations) == 1


def test_different_pages_remain_in_retrieval_order() -> None:
    citations = build_citations(
        [
            make_chunk(page_number=50, page_label="50"),
            make_chunk(
                chunk_id="chunk-49-0",
                page_number=49,
                page_label="49",
            ),
        ]
    )

    assert [citation.page_number for citation in citations] == [50, 49]


def test_long_excerpt_is_normalized_and_shortened() -> None:
    citation = build_citations(
        [make_chunk(text="Support   the\njoint while removing the clamp.")],
        excerpt_max_chars=25,
    )[0]

    assert citation.excerpt == "Support the joint whil..."
    assert len(citation.excerpt) == 25


def test_invalid_excerpt_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be at least 4"):
        build_citations([make_chunk()], excerpt_max_chars=3)
