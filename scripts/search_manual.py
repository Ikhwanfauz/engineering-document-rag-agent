"""Search indexed engineering-document chunks using semantic similarity."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.embedding_manager import EmbeddingConfig, EmbeddingManager
from src.retriever import DocumentRetriever
from src.vector_store import VectorStoreConfig, VectorStoreManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve cited evidence from indexed documents."
    )
    parser.add_argument("query", help="Technical question to search for")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of evidence chunks",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("data/vector_store"),
        help="Local ChromaDB directory",
    )
    parser.add_argument(
        "--collection",
        default="engineering_documents",
        help="ChromaDB collection name",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformers model name",
    )
    parser.add_argument(
        "--device",
        help="Embedding device such as cpu or cuda",
    )
    parser.add_argument(
        "--document-id",
        help="Optionally search only one indexed document",
    )
    return parser


def _preview(text: str, limit: int = 400) -> str:
    single_line = " ".join(text.split())

    if len(single_line) <= limit:
        return single_line

    return f"{single_line[:limit].rstrip()}..."


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    started_at = time.perf_counter()

    try:
        embedding_config = EmbeddingConfig(
            model_name=args.embedding_model,
            device=args.device,
        )
        embedding_manager = EmbeddingManager(embedding_config)

        vector_config = VectorStoreConfig(
            persist_directory=args.persist_dir,
            collection_name=args.collection,
        )
        vector_store = VectorStoreManager(
            embedding_manager=embedding_manager,
            config=vector_config,
        )

        retriever = DocumentRetriever(
            embedding_manager=embedding_manager,
            vector_store=vector_store,
        )
        results = retriever.retrieve(
            args.query,
            top_k=args.top_k,
            document_id=args.document_id,
        )
    except ValueError as exc:
        parser.error(str(exc))

    elapsed_seconds = time.perf_counter() - started_at

    print("SEMANTIC RETRIEVAL RESULTS")
    print(f"Query: {args.query}")
    print(f"Results: {len(results)}")
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")

    if not results:
        print("No indexed evidence was found.")
        return 0

    for rank, result in enumerate(results, start=1):
        print()
        print(f"Rank {rank}")
        print(f"Citation: {result.citation}")
        print(f"Similarity: {result.similarity_score:.4f}")
        print(f"Distance: {result.distance:.4f}")
        print(f"Chunk ID: {result.chunk_id}")
        print(f"Evidence: {_preview(result.text)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
