"""SQLite persistence for question-answering interactions."""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_DATABASE_PATH = Path("database/engineering_document_ai.db")
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@dataclass(frozen=True, slots=True)
class CitationReference:
    """Lightweight reference to one cited document page."""

    document_id: str
    source_name: str
    page_number: int
    page_label: str


@dataclass(frozen=True, slots=True)
class StoredInteraction:
    """One stored question-answering interaction."""

    id: int
    question: str
    answer: str
    status: str
    latency_seconds: float
    created_at: str
    citations: tuple[CitationReference, ...]


def get_database_path(database_path: str | Path | None = None) -> Path:
    """Return the configured SQLite database path."""
    if database_path is not None:
        return Path(database_path)

    return Path(os.getenv("DATABASE_PATH", DEFAULT_DATABASE_PATH))


def initialize_database(database_path: str | Path | None = None) -> Path:
    """Create the database directory and initialize its schema safely."""
    resolved_path = get_database_path(database_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    schema = SCHEMA_PATH.read_text(encoding="utf-8")

    with sqlite3.connect(resolved_path) as connection:
        connection.executescript(schema)

    return resolved_path


def store_interaction(
    *,
    question: str,
    answer: str,
    status: str,
    latency_seconds: float,
    citations: Iterable[CitationReference] = (),
    database_path: str | Path | None = None,
) -> int:
    """Store one complete interaction and return its database ID."""
    if not question.strip():
        raise ValueError("question cannot be empty")
    if not answer.strip():
        raise ValueError("answer cannot be empty")
    if not status.strip():
        raise ValueError("status cannot be empty")
    if latency_seconds < 0:
        raise ValueError("latency_seconds cannot be negative")

    resolved_path = initialize_database(database_path)
    citation_references = tuple(citations)

    with sqlite3.connect(resolved_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        cursor = connection.execute(
            """
            INSERT INTO qa_interactions (
                question,
                answer,
                status,
                latency_seconds
            )
            VALUES (?, ?, ?, ?)
            """,
            (question, answer, status, latency_seconds),
        )
        interaction_id = cursor.lastrowid

        if interaction_id is None:
            raise RuntimeError("SQLite did not return an interaction ID")

        connection.executemany(
            """
            INSERT INTO interaction_citations (
                interaction_id,
                citation_order,
                document_id,
                source_name,
                page_number,
                page_label
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                (
                    interaction_id,
                    citation_order,
                    citation.document_id,
                    citation.source_name,
                    citation.page_number,
                    citation.page_label,
                )
                for citation_order, citation in enumerate(
                    citation_references,
                    start=1,
                )
            ),
        )

    return interaction_id


def get_interaction(
    interaction_id: int,
    *,
    database_path: str | Path | None = None,
) -> StoredInteraction | None:
    """Retrieve one interaction and its ordered citation references."""
    resolved_path = initialize_database(database_path)

    with sqlite3.connect(resolved_path) as connection:
        connection.row_factory = sqlite3.Row
        interaction = connection.execute(
            """
            SELECT
                id,
                question,
                answer,
                status,
                latency_seconds,
                created_at
            FROM qa_interactions
            WHERE id = ?
            """,
            (interaction_id,),
        ).fetchone()

        if interaction is None:
            return None

        citation_rows = connection.execute(
            """
            SELECT
                document_id,
                source_name,
                page_number,
                page_label
            FROM interaction_citations
            WHERE interaction_id = ?
            ORDER BY citation_order
            """,
            (interaction_id,),
        ).fetchall()

    citations = tuple(
        CitationReference(
            document_id=row["document_id"],
            source_name=row["source_name"],
            page_number=row["page_number"],
            page_label=row["page_label"],
        )
        for row in citation_rows
    )

    return StoredInteraction(
        id=interaction["id"],
        question=interaction["question"],
        answer=interaction["answer"],
        status=interaction["status"],
        latency_seconds=interaction["latency_seconds"],
        created_at=interaction["created_at"],
        citations=citations,
    )
