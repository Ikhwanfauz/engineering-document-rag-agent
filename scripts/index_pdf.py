"""Process a PDF and index its page-aware chunks in local ChromaDB."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.document_loader import PDFIngestionError, load_pdf
from src.embedding_manager import EmbeddingConfig, EmbeddingManager
from src.text_chunker import ChunkingConfig, process_document
from src.vector_store import VectorStoreConfig, VectorStoreManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed and index an engineering PDF in ChromaDB."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("data/vector_store"),
        help="Local ChromaDB storage directory",
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
        help="Embedding device such as cpu, cuda, or mps",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum characters per chunk",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Character overlap between neighboring chunks",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Display embedding progress",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    started_at = time.perf_counter()

    try:
        loaded_document = load_pdf(args.pdf_path)

        chunking_config = ChunkingConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        processed_document = process_document(
            loaded_document,
            chunking_config,
        )

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

        report = vector_store.index_document(
            processed_document,
            show_progress=args.show_progress,
        )
    except (
        FileNotFoundError,
        ValueError,
        PDFIngestionError,
    ) as exc:
        parser.error(str(exc))

    elapsed_seconds = time.perf_counter() - started_at

    print("PDF INDEXING SUMMARY")
    print(f"Source: {processed_document.source_name}")
    print(f"Document ID: {processed_document.document_id}")
    print(f"Physical pages: {processed_document.page_count}")
    print(f"Processed chunks: {report.total_chunks}")
    print(f"New chunks indexed: {report.added_chunks}")
    print(f"Existing chunks skipped: {report.existing_chunks}")
    print(f"Collection total: {report.collection_count}")
    print(f"Embedding model: {embedding_config.model_name}")
    print(f"Vector store: {vector_config.persist_directory}")
    print(f"Collection: {vector_config.collection_name}")
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
