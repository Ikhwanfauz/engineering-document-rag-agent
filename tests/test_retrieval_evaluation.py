"""Tests for the retrieval-evaluation runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import argparse

from evaluation.evaluate import (
    _failure_reason,
    _load_dataset,
    _parse_threshold,
    _result_summary,
)
from src.retriever import RetrievedChunk


def _make_chunk(
    *,
    page_number: int,
    similarity_score: float = 0.80,
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
        "expected_pages": [50],
    }


def test_parse_threshold_accepts_none() -> None:
    assert _parse_threshold("none") is None
    assert _parse_threshold("NONE") is None


def test_parse_threshold_accepts_valid_number() -> None:
    assert _parse_threshold("0.6") == pytest.approx(0.6)


@pytest.mark.parametrize("value", ["-0.1", "1.1"])
def test_parse_threshold_rejects_out_of_range_value(
    value: str,
) -> None:
    with pytest.raises(
    argparse.ArgumentTypeError,
        match="Similarity thresholds must be between 0 and 1",
    ):
        _parse_threshold(value)


def test_load_dataset_reads_json(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    expected = {
        "schema_version": "1.0",
        "dataset_version": "7A",
        "examples": [],
    }
    dataset_path.write_text(
        json.dumps(expected),
        encoding="utf-8",
    )

    assert _load_dataset(dataset_path) == expected


def test_load_dataset_rejects_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(
        ValueError,
        match="Evaluation dataset does not exist",
    ):
        _load_dataset(missing_path)


def test_result_summary_keeps_machine_readable_fields() -> None:
    result = _make_chunk(
        page_number=50,
        similarity_score=0.82,
    )

    assert _result_summary(result) == {
        "chunk_id": "chunk-50",
        "source_name": "e-Series_Service_Manual_en.pdf",
        "page_number": 50,
        "page_label": "50",
        "similarity_score": pytest.approx(0.82),
    }


def test_failure_reason_detects_threshold_removal() -> None:
    expected_result = _make_chunk(
        page_number=50,
        similarity_score=0.59,
    )

    reason = _failure_reason(
        _make_example(),
        (expected_result,),
        (),
        (expected_result,),
    )

    assert reason == "expected_page_below_similarity_threshold"


def test_failure_reason_detects_expected_page_outside_top_k() -> None:
    unrelated_result = _make_chunk(page_number=70)
    expected_result = _make_chunk(page_number=50)

    reason = _failure_reason(
        _make_example(),
        (unrelated_result,),
        (unrelated_result,),
        (unrelated_result, expected_result),
    )

    assert reason == "expected_page_outside_top_k"


def test_failure_reason_detects_missing_candidate() -> None:
    unrelated_result = _make_chunk(page_number=70)

    reason = _failure_reason(
        _make_example(),
        (unrelated_result,),
        (unrelated_result,),
        (unrelated_result,),
    )

    assert reason == "expected_page_not_in_retrieved_candidates"