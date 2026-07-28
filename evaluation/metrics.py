"""Reusable metrics for retrieval evaluation."""

from __future__ import annotations
import re

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

@dataclass(frozen=True, slots=True)
class AnswerCaseMetrics:
    """Answer-quality metrics for one evaluation question."""

    answerable: bool
    abstention_correct: bool
    citation_correctness: float | None
    answer_point_coverage: float | None
    evidence_grounding_score: float | None
    latency_seconds: float


@dataclass(frozen=True, slots=True)
class AnswerAggregateMetrics:
    """Aggregated answer-quality metrics."""

    question_count: int
    abstention_correct_count: int
    abstention_accuracy: float
    citation_correctness: float
    answer_point_coverage: float
    evidence_grounding_score: float
    average_latency_seconds: float


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "answer",
    "as",
    "be",
    "before",
    "by",
    "described",
    "during",
    "every",
    "explain",
    "for",
    "from",
    "how",
    "identify",
    "in",
    "instructs",
    "is",
    "it",
    "must",
    "of",
    "on",
    "or",
    "properly",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "using",
    "when",
    "which",
    "with",
}


def _normalize_text(text: str) -> str:
    """Normalize text for deterministic comparison."""
    return " ".join(text.casefold().split())


def _content_tokens(text: str) -> set[str]:
    """Return meaningful lowercase tokens used by lexical metrics."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in _STOP_WORDS and len(token) > 1
    }


def _score_answer_points(
    required_points: Sequence[str],
    answer_text: str,
) -> float | None:
    """Measure lexical coverage of labeled answer requirements."""
    if not required_points:
        return None

    answer_tokens = _content_tokens(answer_text)
    point_scores: list[float] = []

    for point in required_points:
        expected_tokens = _content_tokens(point)

        if not expected_tokens:
            point_scores.append(1.0)
            continue

        matched_tokens = expected_tokens & answer_tokens
        point_scores.append(
            len(matched_tokens) / len(expected_tokens)
        )

    return sum(point_scores) / len(point_scores)


def _score_citations(
    example: Mapping[str, Any],
    answer: Any,
) -> float | None:
    """Compare generated citations with expected document-page labels."""
    expected_documents = {
        str(document)
        for document in example["expected_documents"]
    }
    expected_pages = [
        int(page)
        for page in example["expected_pages"]
    ]
    expected_labels = [
        str(label)
        for label in example["expected_page_labels"]
    ]

    if not expected_documents or not expected_pages:
        return None

    if len(expected_labels) != len(expected_pages):
        expected_labels = [
            str(page)
            for page in expected_pages
        ]

    expected_citations = {
        (document, page, label)
        for document in expected_documents
        for page, label in zip(
            expected_pages,
            expected_labels,
            strict=True,
        )
    }
    actual_citations = {
        (
            citation.source_name,
            citation.page_number,
            citation.page_label,
        )
        for citation in answer.citations
    }

    combined_citations = expected_citations | actual_citations

    if not combined_citations:
        return 1.0

    return (
        len(expected_citations & actual_citations)
        / len(combined_citations)
    )


def _score_grounding(answer: Any) -> float | None:
    """Measure how much answer wording is supported by accepted evidence."""
    if answer.abstained:
        return None

    answer_tokens = _content_tokens(answer.answer)

    if not answer_tokens:
        return 0.0

    evidence_tokens: set[str] = set()

    for chunk in answer.evidence:
        evidence_tokens.update(_content_tokens(chunk.text))

    if not evidence_tokens:
        return 0.0

    supported_tokens = answer_tokens & evidence_tokens
    return len(supported_tokens) / len(answer_tokens)


def evaluate_answer_case(
    example: Mapping[str, Any],
    answer: Any,
    *,
    latency_seconds: float,
) -> AnswerCaseMetrics:
    """Evaluate one generated answer against its labeled example."""
    if latency_seconds < 0:
        raise ValueError("Latency cannot be negative")

    answerable = bool(example["answerable"])
    expected_abstention = not answerable
    abstention_correct = answer.abstained == expected_abstention

    if expected_abstention and answer.abstained:
        expected_response = example.get("expected_response")

        if expected_response:
            abstention_correct = (
                _normalize_text(answer.answer)
                == _normalize_text(str(expected_response))
            )

    citation_correctness = None
    answer_point_coverage = None
    grounding_score = None

    if answerable:
        if answer.abstained:
            citation_correctness = 0.0
            answer_point_coverage = 0.0
            grounding_score = 0.0
        else:
            citation_correctness = _score_citations(
                example,
                answer,
            )
            answer_point_coverage = _score_answer_points(
                example["required_answer_points"],
                answer.answer,
            )
            grounding_score = _score_grounding(answer)

    return AnswerCaseMetrics(
        answerable=answerable,
        abstention_correct=abstention_correct,
        citation_correctness=citation_correctness,
        answer_point_coverage=answer_point_coverage,
        evidence_grounding_score=grounding_score,
        latency_seconds=latency_seconds,
    )


def _average_optional(
    values: Sequence[float | None],
) -> float:
    """Average available metric values, ignoring non-applicable cases."""
    available_values = [
        value
        for value in values
        if value is not None
    ]

    if not available_values:
        return 0.0

    return sum(available_values) / len(available_values)


def aggregate_answer_metrics(
    cases: Sequence[AnswerCaseMetrics],
) -> AnswerAggregateMetrics:
    """Aggregate answer-quality metrics across all questions."""
    question_count = len(cases)
    abstention_correct_count = sum(
        case.abstention_correct
        for case in cases
    )

    abstention_accuracy = (
        abstention_correct_count / question_count
        if question_count
        else 0.0
    )
    average_latency = (
        sum(case.latency_seconds for case in cases)
        / question_count
        if question_count
        else 0.0
    )

    return AnswerAggregateMetrics(
        question_count=question_count,
        abstention_correct_count=abstention_correct_count,
        abstention_accuracy=abstention_accuracy,
        citation_correctness=_average_optional(
            [
                case.citation_correctness
                for case in cases
            ]
        ),
        answer_point_coverage=_average_optional(
            [
                case.answer_point_coverage
                for case in cases
            ]
        ),
        evidence_grounding_score=_average_optional(
            [
                case.evidence_grounding_score
                for case in cases
            ]
        ),
        average_latency_seconds=average_latency,
    )