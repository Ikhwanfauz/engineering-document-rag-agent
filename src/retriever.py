"""Retrieve semantically relevant, citation-aware document chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.embedding_manager import EmbeddingManager
from src.vector_store import VectorStoreManager


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """One semantically retrieved evidence chunk."""

    chunk_id: str
    document_id: str
    source_name: str
    page_number: int
    page_label: str
    chunk_index: int
    text: str
    distance: float
    similarity_score: float

    @property
    def citation(self) -> str:
        """Return a simple human-readable physical-page citation."""
        return f"{self.source_name}, page {self.page_number}"


class DocumentRetriever:
    """Embed questions and retrieve matching chunks from ChromaDB."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        vector_store: VectorStoreManager,
    ) -> None:
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        """Retrieve the nearest evidence chunks for one question."""
        if not query.strip():
            raise ValueError("Retrieval query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        collection = self.vector_store.collection
        collection_count = collection.count()

        if collection_count == 0:
            return ()

        query_embedding = self.embedding_manager.embed_query(query)

        query_arguments: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, collection_count),
            "include": ["documents", "metadatas", "distances"],
        }

        if document_id is not None:
            query_arguments["where"] = {
                "document_id": document_id,
            }

        raw_results = collection.query(**query_arguments)

        ids = (raw_results.get("ids") or [[]])[0]
        documents = (raw_results.get("documents") or [[]])[0]
        metadatas = (raw_results.get("metadatas") or [[]])[0]
        distances = (raw_results.get("distances") or [[]])[0]

        retrieved: list[RetrievedChunk] = []

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            distance_value = float(distance)

            retrieved.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    document_id=str(metadata["document_id"]),
                    source_name=str(metadata["source_name"]),
                    page_number=int(metadata["page_number"]),
                    page_label=str(metadata["page_label"]),
                    chunk_index=int(metadata["chunk_index"]),
                    text=str(text),
                    distance=distance_value,
                    similarity_score=1.0 - distance_value,
                )
            )

        return tuple(retrieved)
