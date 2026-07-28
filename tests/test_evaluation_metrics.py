"""Tests for reusable retrieval-evaluation metrics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation.metrics import (
    RetrievalCaseMetrics,
    aggregate_retrieval_metrics,
    evaluate_retrieval_case,
    filter_by_similarity,
    AnswerCaseMetrics,
    aggregate_answer_metrics,
    evaluate_answer_case,
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

def _make_answer(
    *,
    text: str,
    abstained: bool,
    citations: tuple[object, ...] = (),
    evidence: tuple[RetrievedChunk, ...] = (),
) -> SimpleNamespace:
    return SimpleNamespace(
        answer=text,
        abstained=abstained,
        citations=citations,
        evidence=evidence,
    )


def _make_answer_example(
    *,
    answerable: bool = True,
) -> dict[str, object]:
    return {
        "answerable": answerable,
        "expected_documents": [
            "e-Series_Service_Manual_en.pdf"
        ]
        if answerable
        else [],
        "expected_pages": [51] if answerable else [],
        "expected_page_labels": ["51"] if answerable else [],
        "required_answer_points": [
            "Replace seals and rings."
        ]
        if answerable
        else [],
        "expected_response": (
            None
            if answerable
            else "I don't know based on the uploaded documents."
        ),
    }


def test_evaluate_answer_case_scores_correct_answer() -> None:
    citation = SimpleNamespace(
        source_name="e-Series_Service_Manual_en.pdf",
        page_number=51,
        page_label="51",
    )
    evidence = _make_chunk(
        page_number=51,
        similarity_score=0.85,
    )
    evidence = RetrievedChunk(
        chunk_id=evidence.chunk_id,
        document_id=evidence.document_id,
        source_name=evidence.source_name,
        page_number=evidence.page_number,
        page_label=evidence.page_label,
        chunk_index=evidence.chunk_index,
        text="Replace seals and rings.",
        distance=evidence.distance,
        similarity_score=evidence.similarity_score,
    )
    answer = _make_answer(
        text="Replace seals and rings.",
        abstained=False,
        citations=(citation,),
        evidence=(evidence,),
    )

    metrics = evaluate_answer_case(
        _make_answer_example(),
        answer,
        latency_seconds=1.25,
    )

    assert metrics.answerable is True
    assert metrics.abstention_correct is True
    assert metrics.citation_correctness == pytest.approx(1.0)
    assert metrics.answer_point_coverage == pytest.approx(1.0)
    assert metrics.evidence_grounding_score == pytest.approx(1.0)
    assert metrics.latency_seconds == pytest.approx(1.25)


def test_evaluate_answer_case_penalizes_answerable_abstention() -> None:
    answer = _make_answer(
        text="I don't know based on the uploaded documents.",
        abstained=True,
    )

    metrics = evaluate_answer_case(
        _make_answer_example(),
        answer,
        latency_seconds=0.5,
    )

    assert metrics.abstention_correct is False
    assert metrics.citation_correctness == 0.0
    assert metrics.answer_point_coverage == 0.0
    assert metrics.evidence_grounding_score == 0.0


def test_evaluate_answer_case_accepts_expected_abstention() -> None:
    answer = _make_answer(
        text="I don't know based on the uploaded documents.",
        abstained=True,
    )

    metrics = evaluate_answer_case(
        _make_answer_example(answerable=False),
        answer,
        latency_seconds=0.4,
    )

    assert metrics.answerable is False
    assert metrics.abstention_correct is True
    assert metrics.citation_correctness is None
    assert metrics.answer_point_coverage is None
    assert metrics.evidence_grounding_score is None


def test_evaluate_answer_case_rejects_wrong_abstention_text() -> None:
    answer = _make_answer(
        text="I cannot answer.",
        abstained=True,
    )

    metrics = evaluate_answer_case(
        _make_answer_example(answerable=False),
        answer,
        latency_seconds=0.4,
    )

    assert metrics.abstention_correct is False


def test_evaluate_answer_case_rejects_negative_latency() -> None:
    answer = _make_answer(
        text="Replace seals and rings.",
        abstained=False,
    )

    with pytest.raises(ValueError, match="Latency cannot be negative"):
        evaluate_answer_case(
            _make_answer_example(),
            answer,
            latency_seconds=-0.1,
        )


def test_aggregate_answer_metrics_combines_cases() -> None:
    cases = (
        AnswerCaseMetrics(
            answerable=True,
            abstention_correct=True,
            citation_correctness=1.0,
            answer_point_coverage=0.8,
            evidence_grounding_score=0.6,
            latency_seconds=2.0,
        ),
        AnswerCaseMetrics(
            answerable=False,
            abstention_correct=False,
            citation_correctness=None,
            answer_point_coverage=None,
            evidence_grounding_score=None,
            latency_seconds=4.0,
        ),
    )

    aggregate = aggregate_answer_metrics(cases)

    assert aggregate.question_count == 2
    assert aggregate.abstention_correct_count == 1
    assert aggregate.abstention_accuracy == pytest.approx(0.5)
    assert aggregate.citation_correctness == pytest.approx(1.0)
    assert aggregate.answer_point_coverage == pytest.approx(0.8)
    assert aggregate.evidence_grounding_score == pytest.approx(0.6)
    assert aggregate.average_latency_seconds == pytest.approx(3.0)


def test_aggregate_answer_metrics_handles_empty_cases() -> None:
    aggregate = aggregate_answer_metrics(())

    assert aggregate.question_count == 0
    assert aggregate.abstention_correct_count == 0
    assert aggregate.abstention_accuracy == 0.0
    assert aggregate.citation_correctness == 0.0
    assert aggregate.answer_point_coverage == 0.0
    assert aggregate.evidence_grounding_score == 0.0
    assert aggregate.average_latency_seconds == 0.0