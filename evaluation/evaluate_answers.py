"""Evaluate answer quality, citations, abstention, and response latency."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.metrics import (
    aggregate_answer_metrics,
    evaluate_answer_case,
)
from src.embedding_manager import EmbeddingConfig, EmbeddingManager
from src.llm_provider import LLMServiceError, OllamaLLMProvider
from src.rag_pipeline import GroundedAnswer, RAGPipeline
from src.retriever import DocumentRetriever, RetrievedChunk
from src.vector_store import VectorStoreConfig, VectorStoreManager


def build_parser() -> argparse.ArgumentParser:
    """Create the Version 7C command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generated answers, citations, abstention, "
            "groundedness, and response latency."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/evaluation_dataset.json"),
        help="Labeled evaluation dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "evaluation/results/answer_evaluation.json"
        ),
        help="Machine-readable evaluation output",
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
        "--ollama-model",
        default="qwen3:8b",
        help="Ollama model used for answer generation",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
        help="Ollama service URL",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of chunks retrieved per question",
    )
    parser.add_argument(
        "--minimum-similarity",
        type=float,
        default=0.60,
        help="Minimum accepted evidence similarity",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=256,
        help="Maximum generated response tokens",
    )
    return parser


def _load_dataset(path: Path) -> dict[str, Any]:
    """Load one labeled evaluation dataset."""
    if not path.is_file():
        raise ValueError(
            f"Evaluation dataset does not exist: {path}"
        )

    return json.loads(path.read_text(encoding="utf-8"))


def _citation_summary(answer: GroundedAnswer) -> list[dict[str, Any]]:
    """Convert answer citations into JSON-compatible dictionaries."""
    return [
        {
            "document_id": citation.document_id,
            "source_name": citation.source_name,
            "page_number": citation.page_number,
            "page_label": citation.page_label,
            "excerpt": citation.excerpt,
        }
        for citation in answer.citations
    ]


def _evidence_summary(
    evidence: tuple[RetrievedChunk, ...],
) -> list[dict[str, Any]]:
    """Convert accepted evidence into JSON-compatible dictionaries."""
    return [
        {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "source_name": chunk.source_name,
            "page_number": chunk.page_number,
            "page_label": chunk.page_label,
            "similarity_score": chunk.similarity_score,
            "text": chunk.text,
        }
        for chunk in evidence
    ]


def _validate_arguments(
    parser: argparse.ArgumentParser,
    *,
    top_k: int,
    minimum_similarity: float,
    num_predict: int,
) -> None:
    """Reject invalid evaluation configuration."""
    if top_k <= 0:
        parser.error("--top-k must be greater than zero")

    if not 0.0 <= minimum_similarity <= 1.0:
        parser.error(
            "--minimum-similarity must be between 0 and 1"
        )

    if num_predict <= 0:
        parser.error("--num-predict must be greater than zero")


def main() -> int:
    """Run the complete Version 7C evaluation."""
    parser = build_parser()
    args = parser.parse_args()

    _validate_arguments(
        parser,
        top_k=args.top_k,
        minimum_similarity=args.minimum_similarity,
        num_predict=args.num_predict,
    )

    evaluation_started_at = time.perf_counter()

    try:
        dataset = _load_dataset(args.dataset)

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
            model=args.ollama_model,
            temperature=0.0,
            base_url=args.ollama_base_url,
            reasoning=False,
            num_predict=args.num_predict,
        )
        pipeline = RAGPipeline(
            retriever=retriever,
            llm_provider=llm_provider,
            minimum_similarity=args.minimum_similarity,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    case_metrics = []
    case_results: list[dict[str, Any]] = []

    for example in dataset["examples"]:
        question_started_at = time.perf_counter()

        try:
            answer = pipeline.answer(
                example["question"],
                top_k=args.top_k,
            )
        except LLMServiceError as exc:
            parser.error(
                f"Question {example['id']} failed: {exc}"
            )

        latency_seconds = (
            time.perf_counter() - question_started_at
        )
        metrics = evaluate_answer_case(
            example,
            answer,
            latency_seconds=latency_seconds,
        )
        case_metrics.append(metrics)

        case_results.append(
            {
                "id": example["id"],
                "question": example["question"],
                "expected": {
                    "answerable": example["answerable"],
                    "documents": example["expected_documents"],
                    "pages": example["expected_pages"],
                    "page_labels": example[
                        "expected_page_labels"
                    ],
                    "required_answer_points": example[
                        "required_answer_points"
                    ],
                    "response": example.get(
                        "expected_response"
                    ),
                },
                "actual": {
                    "answer": answer.answer,
                    "abstained": answer.abstained,
                    "citations": _citation_summary(answer),
                    "accepted_evidence": _evidence_summary(
                        answer.evidence
                    ),
                },
                "metrics": asdict(metrics),
            }
        )

        status = (
            "ABSTAINED"
            if answer.abstained
            else "ANSWERED"
        )
        print(
            f"{example['id']}: {status} "
            f"({latency_seconds:.2f} seconds)"
        )

    aggregate = aggregate_answer_metrics(case_metrics)
    elapsed_seconds = (
        time.perf_counter() - evaluation_started_at
    )

    output = {
        "schema_version": "1.0",
        "evaluation_version": "7C",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "dataset": {
            "path": str(args.dataset),
            "schema_version": dataset["schema_version"],
            "dataset_version": dataset["dataset_version"],
        },
        "configuration": {
            "embedding_model": args.embedding_model,
            "collection": args.collection,
            "ollama_model": args.ollama_model,
            "ollama_base_url": args.ollama_base_url,
            "temperature": 0.0,
            "reasoning": False,
            "num_predict": args.num_predict,
            "top_k": args.top_k,
            "minimum_similarity": args.minimum_similarity,
        },
        "question_count": len(case_results),
        "elapsed_seconds": elapsed_seconds,
        "aggregate_metrics": asdict(aggregate),
        "cases": case_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print()
    print("ANSWER EVALUATION COMPLETE")
    print(f"Questions: {len(case_results)}")
    print(
        "Abstention accuracy: "
        f"{aggregate.abstention_accuracy:.2%}"
    )
    print(
        "Citation correctness: "
        f"{aggregate.citation_correctness:.2%}"
    )
    print(
        "Answer-point coverage: "
        f"{aggregate.answer_point_coverage:.2%}"
    )
    print(
        "Evidence grounding score: "
        f"{aggregate.evidence_grounding_score:.2%}"
    )
    print(
        "Average response latency: "
        f"{aggregate.average_latency_seconds:.2f} seconds"
    )
    print(f"Total elapsed time: {elapsed_seconds:.2f} seconds")
    print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())