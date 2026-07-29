"""Tests for duplicate-safe ChromaDB document indexing."""

from uuid import uuid4

import chromadb
import pytest

from src.embedding_manager import EmbeddingConfig, EmbeddingServiceError
from src.text_chunker import ChunkingConfig, process_document
from src.vector_store import (
    VectorStoreConfig,
    VectorStoreManager,
    VectorStoreServiceError,
)
from tests.test_text_chunker import _make_document


class FakeEmbeddingManager:
    """Predictable local embedder for vector-store unit tests."""

    def __init__(self) -> None:
        self.config = EmbeddingConfig(model_name="test-embedding-model")
        self.embedded_batches: list[list[str]] = []

    def embed_texts(
        self,
        texts: list[str],
        *,
        show_progress: bool = False,
    ) -> list[list[float]]:
        self.embedded_batches.append(list(texts))

        return [
            [float(len(text)), float(index + 1), 1.0]
            for index, text in enumerate(texts)
        ]


def _create_store(
    embedding_manager: FakeEmbeddingManager,
) -> VectorStoreManager:
    config = VectorStoreConfig(
        collection_name=f"test-{uuid4().hex}",
        write_batch_size=2,
    )

    return VectorStoreManager(
        embedding_manager=embedding_manager,
        config=config,
        client=chromadb.EphemeralClient(),
    )


def test_document_chunks_are_indexed_with_citation_metadata() -> None:
    embedding_manager = FakeEmbeddingManager()
    store = _create_store(embedding_manager)
    document = process_document(
        _make_document(
            [
                "Disconnect power before maintenance.",
                "Support the joint before removing the clamp.",
            ]
        )
    )

    report = store.index_document(document)
    stored = store.collection.get(ids=[chunk.chunk_id for chunk in document.chunks])

    assert report.total_chunks == 2
    assert report.added_chunks == 2
    assert report.existing_chunks == 0
    assert report.collection_count == 2
    assert store.document_chunk_count(document.document_id) == 2

    assert stored["documents"] == [chunk.text for chunk in document.chunks]

    for metadata, chunk in zip(
        stored["metadatas"],
        document.chunks,
        strict=True,
    ):
        assert metadata["document_id"] == document.document_id
        assert metadata["source_name"] == "manual.pdf"
        assert metadata["page_number"] == chunk.page_number
        assert metadata["page_label"] == chunk.page_label
        assert metadata["chunk_index"] == chunk.chunk_index
        assert metadata["embedding_model"] == "test-embedding-model"


def test_reindexing_same_document_skips_existing_chunks() -> None:
    embedding_manager = FakeEmbeddingManager()
    store = _create_store(embedding_manager)
    document = process_document(_make_document(["Inspect the emergency stop."]))

    first_report = store.index_document(document)
    first_embedding_calls = len(embedding_manager.embedded_batches)

    second_report = store.index_document(document)

    assert first_report.added_chunks == 1
    assert second_report.added_chunks == 0
    assert second_report.existing_chunks == 1
    assert second_report.collection_count == 1
    assert len(embedding_manager.embedded_batches) == first_embedding_calls


def test_changed_chunking_replaces_stale_document_chunks() -> None:
    embedding_manager = FakeEmbeddingManager()
    store = _create_store(embedding_manager)
    loaded_document = _make_document(["Clamp maintenance instruction. " * 20])

    first_document = process_document(
        loaded_document,
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            margin_line_count=0,
        ),
    )
    updated_document = process_document(
        loaded_document,
        ChunkingConfig(
            chunk_size=180,
            chunk_overlap=30,
            margin_line_count=0,
        ),
    )

    first_report = store.index_document(first_document)
    updated_report = store.index_document(updated_document)

    assert first_document.document_id == updated_document.document_id
    assert first_report.added_chunks == len(first_document.chunks)
    assert updated_report.removed_chunks > 0
    assert updated_report.collection_count == len(updated_document.chunks)
    assert store.document_chunk_count(updated_document.document_id) == len(
        updated_document.chunks
    )


def test_delete_document_removes_only_its_indexed_chunks() -> None:
    embedding_manager = FakeEmbeddingManager()
    store = _create_store(embedding_manager)

    first_document = process_document(
        _make_document(["Disconnect power before maintenance."])
    )
    second_document = process_document(
        _make_document(["Inspect the emergency stop before operation."])
    )

    store.index_document(first_document)
    store.index_document(second_document)

    removed_chunks = store.delete_document(first_document.document_id)

    assert removed_chunks == len(first_document.chunks)
    assert store.document_chunk_count(first_document.document_id) == 0
    assert store.document_chunk_count(second_document.document_id) == len(
        second_document.chunks
    )
    assert store.collection.count() == len(second_document.chunks)


def test_delete_document_returns_zero_when_document_is_not_indexed() -> None:
    embedding_manager = FakeEmbeddingManager()
    store = _create_store(embedding_manager)

    removed_chunks = store.delete_document("missing-document-id")

    assert removed_chunks == 0
    assert store.collection.count() == 0

def test_embedding_failure_preserves_existing_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_manager = FakeEmbeddingManager()
    store = _create_store(embedding_manager)
    loaded_document = _make_document(["Clamp maintenance instruction. " * 20])

    first_document = process_document(
        loaded_document,
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            margin_line_count=0,
        ),
    )
    updated_document = process_document(
        loaded_document,
        ChunkingConfig(
            chunk_size=180,
            chunk_overlap=30,
            margin_line_count=0,
        ),
    )

    store.index_document(first_document)
    original_ids = store._document_chunk_ids(first_document.document_id)

    def raise_embedding_error(
        *_: object,
        **__: object,
    ) -> list[list[float]]:
        raise EmbeddingServiceError("embedding unavailable")

    monkeypatch.setattr(
        embedding_manager,
        "embed_texts",
        raise_embedding_error,
    )

    with pytest.raises(
        EmbeddingServiceError,
        match="embedding unavailable",
    ):
        store.index_document(updated_document)

    remaining_ids = store._document_chunk_ids(first_document.document_id)

    assert remaining_ids == original_ids
    assert store.collection.count() == len(original_ids)

def test_chromadb_write_failure_raises_service_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_manager = FakeEmbeddingManager()
    store = _create_store(embedding_manager)
    document = process_document(
        _make_document(["Disconnect power before maintenance."])
    )
    original_error = RuntimeError("database unavailable")

    def raise_write_error(*_: object, **__: object) -> None:
        raise original_error

    monkeypatch.setattr(
        store.collection,
        "upsert",
        raise_write_error,
    )

    with pytest.raises(
        VectorStoreServiceError,
        match="vector database could not write chunks",
    ) as error_info:
        store.index_document(document)

    assert error_info.value.__cause__ is original_error

def test_partial_write_failure_removes_new_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_manager = FakeEmbeddingManager()
    store = _create_store(embedding_manager)
    loaded_document = _make_document(
        ["Clamp maintenance instruction. " * 40]
    )

    first_document = process_document(
        loaded_document,
        ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            margin_line_count=0,
        ),
    )
    updated_document = process_document(
        loaded_document,
        ChunkingConfig(
            chunk_size=180,
            chunk_overlap=30,
            margin_line_count=0,
        ),
    )

    store.index_document(first_document)
    original_ids = store._document_chunk_ids(first_document.document_id)
    original_upsert = store.collection.upsert
    write_calls = 0

    def fail_second_write(*args: object, **kwargs: object) -> None:
        nonlocal write_calls
        write_calls += 1

        if write_calls == 2:
            raise RuntimeError("second batch failed")

        original_upsert(*args, **kwargs)

    monkeypatch.setattr(
        store.collection,
        "upsert",
        fail_second_write,
    )

    with pytest.raises(
        VectorStoreServiceError,
        match="vector database could not write chunks",
    ):
        store.index_document(updated_document)

    remaining_ids = store._document_chunk_ids(first_document.document_id)

    assert write_calls == 2
    assert remaining_ids == original_ids
