"""Tests for reusable retrieval-evaluation metrics."""

from __future__ import annotations

import pytest

from evaluation.metrics import (
    RetrievalCaseMetrics,
    aggregate_retrieval_metrics,
    evaluate_retrieval_case,
    filter_by_similarity,
)
from src.retriever import RetrievedChunk


def _make_chunk(
    *,
    page_number: int,
    similarity_score: float,
    source_name: str = "e-Series_Service_Manual_en.pdf",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{page_number}",
        document_id="e-series-service-manual",
        source_name=source_name,
        page_number=page_number,
        page_label=str(page_number),
        chunk_index=0,
        text="Example engineering evidence.",
        distance=1.0 - similarity_score,
        similarity_score=similarity_score,
    )


def _make_example() -> dict[str, object]:
    return {
        "expected_documents": ["e-Series_Service_Manual_en.pdf"],
        "expected_pages": [50, 51],
    }


def test_filter_by_similarity_without_threshold_keeps_all_results() -> None:
    results = (
        _make_chunk(page_number=50, similarity_score=0.80),
        _make_chunk(page_number=51, similarity_score=0.55),
    )

    assert filter_by_similarity(results, None) == results


def test_filter_by_similarity_removes_results_below_threshold() -> None:
    results = (
        _make_chunk(page_number=50, similarity_score=0.80),
        _make_chunk(page_number=51, similarity_score=0.55),
    )

    filtered = filter_by_similarity(results, 0.60)

    assert filtered == (results[0],)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_filter_by_similarity_rejects_invalid_threshold(
    threshold: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="Similarity threshold must be between 0 and 1",
    ):
        filter_by_similarity((), threshold)


def test_evaluate_retrieval_case_measures_hit_and_page_recall() -> None:
    results = (
        _make_chunk(page_number=50, similarity_score=0.82),
        _make_chunk(page_number=70, similarity_score=0.76),
    )

    metrics = evaluate_retrieval_case(_make_example(), results)

    assert metrics.hit is True
    assert metrics.retrieved_chunks == 2
    assert metrics.expected_pages == 2
    assert metrics.matched_pages == 1
    assert metrics.expected_page_recall == pytest.approx(0.5)
    assert metrics.highest_similarity == pytest.approx(0.82)


def test_evaluate_retrieval_case_applies_similarity_threshold() -> None:
    results = (
        _make_chunk(page_number=50, similarity_score=0.59),
        _make_chunk(page_number=70, similarity_score=0.75),
    )

    metrics = evaluate_retrieval_case(
        _make_example(),
        results,
        threshold=0.60,
    )

    assert metrics.hit is False
    assert metrics.retrieved_chunks == 1
    assert metrics.matched_pages == 0
    assert metrics.expected_page_recall == 0.0
    assert metrics.highest_similarity == pytest.approx(0.75)


def test_evaluate_retrieval_case_requires_matching_document() -> None:
    results = (
        _make_chunk(
            page_number=50,
            similarity_score=0.85,
            source_name="different_manual.pdf",
        ),
    )

    metrics = evaluate_retrieval_case(_make_example(), results)

    assert metrics.hit is False
    assert metrics.matched_pages == 0


def test_aggregate_retrieval_metrics_combines_cases() -> None:
    cases = (
        RetrievalCaseMetrics(
            hit=True,
            retrieved_chunks=3,
            expected_pages=2,
            matched_pages=1,
            expected_page_recall=0.5,
            highest_similarity=0.82,
        ),
        RetrievalCaseMetrics(
            hit=False,
            retrieved_chunks=3,
            expected_pages=1,
            matched_pages=0,
            expected_page_recall=0.0,
            highest_similarity=0.71,
        ),
    )

    aggregate = aggregate_retrieval_metrics(cases)

    assert aggregate.question_count == 2
    assert aggregate.hit_count == 1
    assert aggregate.hit_rate == pytest.approx(0.5)
    assert aggregate.expected_page_count == 3
    assert aggregate.matched_page_count == 1
    assert aggregate.expected_page_recall == pytest.approx(1 / 3)


def test_aggregate_retrieval_metrics_handles_empty_cases() -> None:
    aggregate = aggregate_retrieval_metrics(())

    assert aggregate.question_count == 0
    assert aggregate.hit_count == 0
    assert aggregate.hit_rate == 0.0
    assert aggregate.expected_page_count == 0
    assert aggregate.matched_page_count == 0
    assert aggregate.expected_page_recall == 0.0