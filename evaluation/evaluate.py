"""Run reproducible retrieval evaluation against the labeled dataset."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.metrics import (
    aggregate_retrieval_metrics,
    evaluate_retrieval_case,
    filter_by_similarity,
)
from src.embedding_manager import EmbeddingConfig, EmbeddingManager
from src.retriever import DocumentRetriever, RetrievedChunk
from src.vector_store import VectorStoreConfig, VectorStoreManager


def _parse_threshold(value: str) -> float | None:
    if value.lower() == "none":
        return None

    threshold = float(value)

    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError(
            "Similarity thresholds must be between 0 and 1"
        )

    return threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval against labeled questions."
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
            "evaluation/results/retrieval_evaluation.json"
        ),
        help="Machine-readable evaluation output",
    )
    parser.add_argument(
        "--top-k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Top-k values to evaluate",
    )
    parser.add_argument(
        "--thresholds",
        type=_parse_threshold,
        nargs="+",
        default=[None, 0.50, 0.60, 0.70],
        help="Similarity thresholds, using 'none' for no threshold",
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
    return parser


def _load_dataset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Evaluation dataset does not exist: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def _result_summary(result: RetrievedChunk) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "source_name": result.source_name,
        "page_number": result.page_number,
        "page_label": result.page_label,
        "similarity_score": result.similarity_score,
    }


def _failure_reason(
    example: dict[str, Any],
    top_k_results: tuple[RetrievedChunk, ...],
    filtered_results: tuple[RetrievedChunk, ...],
    all_results: tuple[RetrievedChunk, ...],
) -> str:
    expected_documents = set(example["expected_documents"])
    expected_pages = set(example["expected_pages"])

    def contains_expected(
        results: tuple[RetrievedChunk, ...],
    ) -> bool:
        return any(
            result.source_name in expected_documents
            and result.page_number in expected_pages
            for result in results
        )

    if contains_expected(top_k_results) and not contains_expected(
        filtered_results
    ):
        return "expected_page_below_similarity_threshold"

    if contains_expected(all_results) and not contains_expected(
        top_k_results
    ):
        return "expected_page_outside_top_k"

    return "expected_page_not_in_retrieved_candidates"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if any(top_k <= 0 for top_k in args.top_k_values):
        parser.error("All top-k values must be greater than zero")

    top_k_values = sorted(set(args.top_k_values))
    maximum_top_k = max(top_k_values)
    started_at = time.perf_counter()

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
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    answerable_examples = [
        example
        for example in dataset["examples"]
        if example["answerable"]
    ]

    retrieved_by_id: dict[str, tuple[RetrievedChunk, ...]] = {}

    for example in answerable_examples:
        retrieved_by_id[example["id"]] = retriever.retrieve(
            example["question"],
            top_k=maximum_top_k,
        )

    configurations: list[dict[str, Any]] = []
    overall_failure_counts: Counter[str] = Counter()

    for top_k in top_k_values:
        for threshold in args.thresholds:
            case_metrics = []
            case_results = []
            failure_counts: Counter[str] = Counter()

            for example in answerable_examples:
                all_results = retrieved_by_id[example["id"]]
                top_k_results = all_results[:top_k]
                filtered_results = filter_by_similarity(
                    top_k_results,
                    threshold,
                )
                metrics = evaluate_retrieval_case(
                    example,
                    top_k_results,
                    threshold=threshold,
                )
                case_metrics.append(metrics)

                failure_reason = None

                if not metrics.hit:
                    failure_reason = _failure_reason(
                        example,
                        top_k_results,
                        filtered_results,
                        all_results,
                    )
                    failure_counts[failure_reason] += 1
                    overall_failure_counts[failure_reason] += 1

                case_results.append(
                    {
                        "id": example["id"],
                        "question": example["question"],
                        "expected_documents": example[
                            "expected_documents"
                        ],
                        "expected_pages": example["expected_pages"],
                        "metrics": asdict(metrics),
                        "failure_reason": failure_reason,
                        "retrieved_results": [
                            _result_summary(result)
                            for result in filtered_results
                        ],
                    }
                )

            aggregate = aggregate_retrieval_metrics(case_metrics)

            configurations.append(
                {
                    "top_k": top_k,
                    "similarity_threshold": threshold,
                    "aggregate_metrics": asdict(aggregate),
                    "failure_counts": dict(failure_counts),
                    "cases": case_results,
                }
            )

    elapsed_seconds = time.perf_counter() - started_at

    output = {
        "schema_version": "1.0",
        "evaluation_version": "7B",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(args.dataset),
            "schema_version": dataset["schema_version"],
            "dataset_version": dataset["dataset_version"],
        },
        "retrieval_configuration": {
            "embedding_model": args.embedding_model,
            "collection": args.collection,
            "top_k_values": top_k_values,
            "similarity_thresholds": args.thresholds,
        },
        "answerable_question_count": len(answerable_examples),
        "elapsed_seconds": elapsed_seconds,
        "common_retrieval_failures": dict(
            overall_failure_counts.most_common()
        ),
        "configurations": configurations,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print("RETRIEVAL EVALUATION COMPLETE")
    print(f"Answerable questions: {len(answerable_examples)}")
    print(f"Configurations: {len(configurations)}")
    print(f"Elapsed time: {elapsed_seconds:.2f} seconds")
    print(f"Results saved to: {args.output}")

    for configuration in configurations:
        aggregate = configuration["aggregate_metrics"]
        threshold = configuration["similarity_threshold"]
        threshold_label = "none" if threshold is None else threshold

        print()
        print(
            f"top_k={configuration['top_k']}, "
            f"threshold={threshold_label}"
        )
        print(f"Hit rate: {aggregate['hit_rate']:.2%}")
        print(
            "Expected-page recall: "
            f"{aggregate['expected_page_recall']:.2%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

