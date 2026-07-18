"""Tests for citation-aware semantic retrieval."""

from typing import Any

import pytest

from src.retriever import DocumentRetriever


class FakeEmbeddingManager:
    def __init__(self) -> None:
        self.last_query: str | None = None

    def embed_query(self, query: str) -> list[float]:
        self.last_query = query
        return [0.1, 0.2, 0.3]


class FakeCollection:
    def __init__(self, count: int = 2) -> None:
        self._count = count
        self.last_query_arguments: dict[str, Any] = {}

    def count(self) -> int:
        return self._count

    def query(self, **arguments: Any) -> dict[str, Any]:
        self.last_query_arguments = arguments

        return {
            "ids": [["chunk-50-0", "chunk-49-1"]],
            "documents": [
                [
                    "Support the joint while removing the clamp.",
                    "Remove the clamp screws.",
                ]
            ],
            "metadatas": [
                [
                    {
                        "document_id": "document-123",
                        "source_name": "service-manual.pdf",
                        "page_number": 50,
                        "page_label": "50",
                        "chunk_index": 0,
                    },
                    {
                        "document_id": "document-123",
                        "source_name": "service-manual.pdf",
                        "page_number": 49,
                        "page_label": "49",
                        "chunk_index": 1,
                    },
                ]
            ],
            "distances": [[0.12, 0.35]],
        }


class FakeVectorStore:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection


def test_retrieval_returns_ranked_text_scores_and_citations() -> None:
    embedding_manager = FakeEmbeddingManager()
    collection = FakeCollection()
    retriever = DocumentRetriever(
        embedding_manager=embedding_manager,
        vector_store=FakeVectorStore(collection),
    )

    results = retriever.retrieve(
        "How should the joint be supported?",
        top_k=2,
    )

    assert embedding_manager.last_query == ("How should the joint be supported?")
    assert len(results) == 2

    first = results[0]
    assert first.chunk_id == "chunk-50-0"
    assert first.document_id == "document-123"
    assert first.source_name == "service-manual.pdf"
    assert first.page_number == 50
    assert first.page_label == "50"
    assert first.chunk_index == 0
    assert first.distance == pytest.approx(0.12)
    assert first.similarity_score == pytest.approx(0.88)
    assert first.citation == "service-manual.pdf, page 50"
    assert "Support the joint" in first.text


def test_retrieval_passes_document_filter_to_chroma() -> None:
    collection = FakeCollection()
    retriever = DocumentRetriever(
        embedding_manager=FakeEmbeddingManager(),
        vector_store=FakeVectorStore(collection),
    )

    retriever.retrieve(
        "How is the clamp removed?",
        document_id="document-123",
    )

    assert collection.last_query_arguments["where"] == {"document_id": "document-123"}


def test_empty_collection_returns_no_results_without_embedding() -> None:
    embedding_manager = FakeEmbeddingManager()
    retriever = DocumentRetriever(
        embedding_manager=embedding_manager,
        vector_store=FakeVectorStore(FakeCollection(count=0)),
    )

    assert retriever.retrieve("Any maintenance instruction?") == ()
    assert embedding_manager.last_query is None


@pytest.mark.parametrize(
    ("query", "top_k", "message"),
    [
        ("", 5, "query cannot be empty"),
        ("   ", 5, "query cannot be empty"),
        ("valid question", 0, "top_k must be greater than zero"),
        ("valid question", -1, "top_k must be greater than zero"),
    ],
)
def test_invalid_retrieval_inputs_are_rejected(
    query: str,
    top_k: int,
    message: str,
) -> None:
    retriever = DocumentRetriever(
        embedding_manager=FakeEmbeddingManager(),
        vector_store=FakeVectorStore(FakeCollection()),
    )

    with pytest.raises(ValueError, match=message):
        retriever.retrieve(query, top_k=top_k)
