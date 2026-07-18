"""Persist citation-safe document embeddings in a local ChromaDB collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import chromadb

from src.embedding_manager import EmbeddingManager
from src.text_chunker import DocumentChunk, ProcessedDocument


@dataclass(frozen=True, slots=True)
class VectorStoreConfig:
    """Configuration for the persistent local vector database."""

    persist_directory: Path = Path("data/vector_store")
    collection_name: str = "engineering_documents"
    distance_metric: Literal["cosine", "l2", "ip"] = "cosine"
    write_batch_size: int = 64

    def __post_init__(self) -> None:
        if len(self.collection_name.strip()) < 3:
            raise ValueError("collection_name must contain at least 3 characters")
        if self.write_batch_size <= 0:
            raise ValueError("write_batch_size must be greater than zero")


@dataclass(frozen=True, slots=True)
class IndexingReport:
    """Summary of one document-indexing operation."""

    document_id: str
    total_chunks: int
    added_chunks: int
    existing_chunks: int
    collection_count: int


DEFAULT_VECTOR_STORE_CONFIG = VectorStoreConfig()


class VectorStoreManager:
    """Store chunk embeddings and citation metadata in ChromaDB."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        config: VectorStoreConfig = DEFAULT_VECTOR_STORE_CONFIG,
        client: Any | None = None,
    ) -> None:
        self.embedding_manager = embedding_manager
        self.config = config

        if client is None:
            config.persist_directory.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(config.persist_directory))

        self.client = client
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            embedding_function=None,
            configuration={
                "hnsw": {
                    "space": config.distance_metric,
                }
            },
        )

    def _metadata_for_chunk(
        self,
        chunk: DocumentChunk,
    ) -> dict[str, str | int]:
        return {
            "document_id": chunk.document_id,
            "source_name": chunk.source_name,
            "page_number": chunk.page_number,
            "page_label": chunk.page_label,
            "chunk_index": chunk.chunk_index,
            "char_count": chunk.char_count,
            "embedding_model": (self.embedding_manager.config.model_name),
        }

    def _existing_chunk_ids(
        self,
        chunk_ids: Sequence[str],
    ) -> set[str]:
        if not chunk_ids:
            return set()

        result = self.collection.get(ids=list(chunk_ids))
        return set(result["ids"] or [])

    def index_document(
        self,
        document: ProcessedDocument,
        *,
        show_progress: bool = False,
    ) -> IndexingReport:
        """Embed and index only chunks that are not already stored."""
        all_chunks = list(document.chunks)
        all_chunk_ids = [chunk.chunk_id for chunk in all_chunks]
        existing_ids = self._existing_chunk_ids(all_chunk_ids)

        new_chunks = [
            chunk for chunk in all_chunks if chunk.chunk_id not in existing_ids
        ]

        for start in range(
            0,
            len(new_chunks),
            self.config.write_batch_size,
        ):
            batch = new_chunks[start : start + self.config.write_batch_size]
            batch_embeddings = self.embedding_manager.embed_texts(
                [chunk.text for chunk in batch],
                show_progress=show_progress,
            )

            self.collection.upsert(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=batch_embeddings,
                metadatas=[self._metadata_for_chunk(chunk) for chunk in batch],
            )

        return IndexingReport(
            document_id=document.document_id,
            total_chunks=len(all_chunks),
            added_chunks=len(new_chunks),
            existing_chunks=len(existing_ids),
            collection_count=self.collection.count(),
        )

    def document_chunk_count(self, document_id: str) -> int:
        """Count indexed chunks belonging to one document."""
        result = self.collection.get(
            where={"document_id": document_id},
        )
        return len(result["ids"] or [])
