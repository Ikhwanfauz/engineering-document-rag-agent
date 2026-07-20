"""Ask grounded questions over indexed engineering documents."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.embedding_manager import EmbeddingConfig, EmbeddingManager
from src.llm_provider import OllamaLLMProvider
from src.rag_pipeline import (
    DEFAULT_MINIMUM_SIMILARITY,
    GroundingValidationError,
    RAGPipeline,
)
from src.retriever import DocumentRetriever
from src.vector_store import VectorStoreConfig, VectorStoreManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a grounded answer from indexed documents."
    )
    parser.add_argument("question", help="Technical question to answer")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of retrieved evidence chunks",
    )
    parser.add_argument(
        "--minimum-similarity",
        type=float,
        default=DEFAULT_MINIMUM_SIMILARITY,
        help="Minimum similarity required to use retrieved evidence",
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
    parser.add_argument(
        "--llm-model",
        default="llama3.2",
        help="Local Ollama model used for answer generation",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama server address",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM response temperature",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    started_at = time.perf_counter()

    try:
        embedding_manager = EmbeddingManager(
            EmbeddingConfig(
                model_name=args.embedding_model,
                device=args.device,
            )
        )

        vector_store = VectorStoreManager(
            embedding_manager=embedding_manager,
            config=VectorStoreConfig(
                persist_directory=args.persist_dir,
                collection_name=args.collection,
            ),
        )

        retriever = DocumentRetriever(
            embedding_manager=embedding_manager,
            vector_store=vector_store,
        )

        llm_provider = OllamaLLMProvider(
            model=args.llm_model,
            temperature=args.temperature,
            base_url=args.ollama_base_url,
        )

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_provider=llm_provider,
            minimum_similarity=args.minimum_similarity,
        )

        result = pipeline.answer(
            args.question,
            top_k=args.top_k,
            document_id=args.document_id,
        )
    except (ValueError, GroundingValidationError) as exc:
        parser.error(str(exc))

    elapsed_seconds = time.perf_counter() - started_at

    print("GROUNDED DOCUMENT ANSWER")
    print(f"Question: {result.question}")
    print(f"LLM model: {args.llm_model}")
    print(f"LLM generation: {'skipped' if result.abstained else 'completed'}")
    print(f"Status: {'ABSTAINED' if result.abstained else 'ANSWERED'}")
    print(f"Minimum similarity: {args.minimum_similarity:.2f}")
    print(f"Accepted evidence chunks: {len(result.evidence)}")
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")

    print()
    print("ANSWER")
    print(result.answer)

    print()
    print("CITATIONS")
    if result.citations:
        for number, citation in enumerate(result.citations, start=1):
            print(f"[{number}] {citation.label}")
            print(f"    PDF page label: {citation.page_label}")
            print(f"    Excerpt: {citation.excerpt}")
    else:
        print("None — insufficient evidence.")

    print()
    print("RETRIEVAL DETAILS")
    if result.evidence:
        for rank, evidence in enumerate(result.evidence, start=1):
            print(
                f"Rank {rank}: {evidence.citation} | "
                f"similarity={evidence.similarity_score:.4f} | "
                f"chunk={evidence.chunk_id}"
            )
    else:
        print(
            f"No evidence met the minimum similarity of {args.minimum_similarity:.2f}."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
