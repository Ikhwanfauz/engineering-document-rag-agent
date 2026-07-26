"""Tests for the SQLite question-answering persistence layer."""

from __future__ import annotations

import sqlite3

import pytest

from database.db import (
    CitationReference,
    get_database_path,
    get_interaction,
    initialize_database,
    list_recent_interactions,
    store_feedback,
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
    assert "interaction_feedback" in table_names


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

def test_store_feedback_links_to_interaction(tmp_path) -> None:
    database_path = tmp_path / "interactions.db"
    interaction_id = store_interaction(
        question="Was this answer useful?",
        answer="Follow the documented safety procedure.",
        status="ANSWERED",
        latency_seconds=0.5,
        database_path=database_path,
    )

    feedback_id = store_feedback(
        interaction_id=interaction_id,
        feedback="POSITIVE",
        database_path=database_path,
    )

    with sqlite3.connect(database_path) as connection:
        stored_feedback = connection.execute(
            """
            SELECT id, interaction_id, feedback
            FROM interaction_feedback
            WHERE id = ?
            """,
            (feedback_id,),
        ).fetchone()

    assert stored_feedback == (
        feedback_id,
        interaction_id,
        "POSITIVE",
    )

def test_store_feedback_rejects_invalid_value(tmp_path) -> None:
    with pytest.raises(
        ValueError,
        match="feedback must be POSITIVE or NEGATIVE",
    ):
        store_feedback(
            interaction_id=1,
            feedback="MAYBE",
            database_path=tmp_path / "interactions.db",
        )

def test_store_feedback_rejects_duplicate_submission(tmp_path) -> None:
    database_path = tmp_path / "interactions.db"
    interaction_id = store_interaction(
        question="Was this answer useful?",
        answer="Follow the documented safety procedure.",
        status="ANSWERED",
        latency_seconds=0.5,
        database_path=database_path,
    )

    store_feedback(
        interaction_id=interaction_id,
        feedback="POSITIVE",
        database_path=database_path,
    )

    with pytest.raises(sqlite3.IntegrityError):
        store_feedback(
            interaction_id=interaction_id,
            feedback="NEGATIVE",
            database_path=database_path,
        )

def test_store_feedback_rejects_unknown_interaction(tmp_path) -> None:
    with pytest.raises(sqlite3.IntegrityError):
        store_feedback(
            interaction_id=999,
            feedback="POSITIVE",
            database_path=tmp_path / "interactions.db",
        )

def test_get_interaction_includes_submitted_feedback(tmp_path) -> None:
    database_path = tmp_path / "interactions.db"
    interaction_id = store_interaction(
        question="Was this answer useful?",
        answer="Follow the documented safety procedure.",
        status="ANSWERED",
        latency_seconds=0.5,
        database_path=database_path,
    )

    store_feedback(
        interaction_id=interaction_id,
        feedback="POSITIVE",
        database_path=database_path,
    )

    stored = get_interaction(
        interaction_id,
        database_path=database_path,
    )

    assert stored is not None
    assert stored.feedback == "POSITIVE"

def test_list_recent_interactions_returns_newest_first(tmp_path) -> None:
    database_path = tmp_path / "interactions.db"
    interaction_ids = [
        store_interaction(
            question=f"Question {number}",
            answer=f"Answer {number}",
            status="ANSWERED",
            latency_seconds=0.5,
            database_path=database_path,
        )
        for number in range(1, 4)
    ]

    recent = list_recent_interactions(
        limit=2,
        database_path=database_path,
    )

    assert [interaction.id for interaction in recent] == [
        interaction_ids[2],
        interaction_ids[1],
    ]

def test_list_recent_interactions_rejects_invalid_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="limit must be positive"):
        list_recent_interactions(
            limit=0,
            database_path=tmp_path / "interactions.db",
        )