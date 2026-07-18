"""Tests for duplicate-safe ChromaDB document indexing."""

from uuid import uuid4

import chromadb

from src.embedding_manager import EmbeddingConfig
from src.text_chunker import process_document
from src.vector_store import VectorStoreConfig, VectorStoreManager

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
