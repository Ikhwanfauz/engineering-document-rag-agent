"""Tests for the SQLite question-answering persistence layer."""

from __future__ import annotations

import sqlite3

import pytest

from database.db import (
    CitationReference,
    get_database_path,
    get_interaction,
    initialize_database,
    store_interaction,
)


def test_get_database_path_uses_environment_variable(
    tmp_path,
    monkeypatch,
) -> None:
    database_path = tmp_path / "configured" / "interactions.db"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))

    assert get_database_path() == database_path


def test_initialize_database_creates_parent_directory_and_tables(
    tmp_path,
) -> None:
    database_path = tmp_path / "nested" / "interactions.db"

    initialized_path = initialize_database(database_path)

    assert initialized_path == database_path
    assert database_path.is_file()

    with sqlite3.connect(database_path) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }

    assert "qa_interactions" in table_names
    assert "interaction_citations" in table_names


def test_initialize_database_can_run_repeatedly(tmp_path) -> None:
    database_path = tmp_path / "interactions.db"

    initialize_database(database_path)
    initialize_database(database_path)

    assert database_path.is_file()


def test_store_and_retrieve_complete_interaction(tmp_path) -> None:
    database_path = tmp_path / "interactions.db"
    citations = (
        CitationReference(
            document_id="manual-001",
            source_name="service_manual.pdf",
            page_number=13,
            page_label="5",
        ),
        CitationReference(
            document_id="manual-001",
            source_name="service_manual.pdf",
            page_number=16,
            page_label="8",
        ),
    )

    interaction_id = store_interaction(
        question="How should the robot be handled safely?",
        answer="The required safety procedure must be followed.",
        status="ANSWERED",
        latency_seconds=1.25,
        citations=citations,
        database_path=database_path,
    )

    stored = get_interaction(
        interaction_id,
        database_path=database_path,
    )

    assert stored is not None
    assert stored.id == interaction_id
    assert stored.question == "How should the robot be handled safely?"
    assert stored.answer == "The required safety procedure must be followed."
    assert stored.status == "ANSWERED"
    assert stored.latency_seconds == pytest.approx(1.25)
    assert stored.created_at
    assert stored.citations == citations


def test_get_interaction_returns_none_for_unknown_id(tmp_path) -> None:
    database_path = tmp_path / "interactions.db"

    assert get_interaction(999, database_path=database_path) is None


@pytest.mark.parametrize(
    ("question", "answer", "status", "latency_seconds", "message"),
    (
        (" ", "Answer", "ANSWERED", 0.5, "question cannot be empty"),
        ("Question", " ", "ANSWERED", 0.5, "answer cannot be empty"),
        ("Question", "Answer", " ", 0.5, "status cannot be empty"),
        (
            "Question",
            "Answer",
            "ANSWERED",
            -0.1,
            "latency_seconds cannot be negative",
        ),
    ),
)
def test_store_interaction_rejects_invalid_values(
    tmp_path,
    question,
    answer,
    status,
    latency_seconds,
    message,
) -> None:
    database_path = tmp_path / "interactions.db"

    with pytest.raises(ValueError, match=message):
        store_interaction(
            question=question,
            answer=answer,
            status=status,
            latency_seconds=latency_seconds,
            database_path=database_path,
        )