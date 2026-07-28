"""Reusable metrics for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.retriever import RetrievedChunk


@dataclass(frozen=True, slots=True)
class RetrievalCaseMetrics:
    """Metrics for one answerable evaluation question."""

    hit: bool
    retrieved_chunks: int
    expected_pages: int
    matched_pages: int
    expected_page_recall: float
    highest_similarity: float | None


@dataclass(frozen=True, slots=True)
class RetrievalAggregateMetrics:
    """Aggregated metrics across answerable questions."""

    question_count: int
    hit_count: int
    hit_rate: float
    expected_page_count: int
    matched_page_count: int
    expected_page_recall: float


def filter_by_similarity(
    results: Sequence[RetrievedChunk],
    threshold: float | None,
) -> tuple[RetrievedChunk, ...]:
    """Keep retrieval results meeting an optional similarity threshold."""
    if threshold is None:
        return tuple(results)

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Similarity threshold must be between 0 and 1")

    return tuple(
        result
        for result in results
        if result.similarity_score >= threshold
    )


def evaluate_retrieval_case(
    example: Mapping[str, Any],
    results: Sequence[RetrievedChunk],
    *,
    threshold: float | None = None,
) -> RetrievalCaseMetrics:
    """Evaluate retrieved chunks against one labeled dataset example."""
    filtered_results = filter_by_similarity(results, threshold)

    expected_documents = {
        str(document)
        for document in example["expected_documents"]
    }
    expected_pages = {
        int(page)
        for page in example["expected_pages"]
    }

    matched_pages = {
        result.page_number
        for result in filtered_results
        if result.source_name in expected_documents
        and result.page_number in expected_pages
    }

    expected_page_count = len(expected_pages)
    matched_page_count = len(matched_pages)

    page_recall = (
        matched_page_count / expected_page_count
        if expected_page_count
        else 0.0
    )

    highest_similarity = (
        max(result.similarity_score for result in filtered_results)
        if filtered_results
        else None
    )

    return RetrievalCaseMetrics(
        hit=matched_page_count > 0,
        retrieved_chunks=len(filtered_results),
        expected_pages=expected_page_count,
        matched_pages=matched_page_count,
        expected_page_recall=page_recall,
        highest_similarity=highest_similarity,
    )


def aggregate_retrieval_metrics(
    cases: Sequence[RetrievalCaseMetrics],
) -> RetrievalAggregateMetrics:
    """Aggregate retrieval hit rate and expected-page recall."""
    question_count = len(cases)
    hit_count = sum(case.hit for case in cases)
    expected_page_count = sum(case.expected_pages for case in cases)
    matched_page_count = sum(case.matched_pages for case in cases)

    hit_rate = hit_count / question_count if question_count else 0.0
    page_recall = (
        matched_page_count / expected_page_count
        if expected_page_count
        else 0.0
    )

    return RetrievalAggregateMetrics(
        question_count=question_count,
        hit_count=hit_count,
        hit_rate=hit_rate,
        expected_page_count=expected_page_count,
        matched_page_count=matched_page_count,
        expected_page_recall=page_recall,
    )