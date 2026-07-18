"""Process a PDF into cleaned, page-aware chunks and save them as JSON."""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

from src.document_loader import PDFIngestionError, load_pdf
from src.text_chunker import (
    ChunkingConfig,
    process_document,
    save_processed_document,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean and chunk a PDF while preserving page citations."
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the source PDF")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed JSON output",
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
        help="Overlapping characters between neighboring chunks",
    )
    parser.add_argument(
        "--sample-page",
        type=int,
        help="Show a short chunk sample from this physical page",
    )
    return parser


def _preview(text: str, limit: int = 240) -> str:
    single_line = " ".join(text.split())
    if len(single_line) <= limit:
        return single_line
    return f"{single_line[:limit].rstrip()}..."


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = ChunkingConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        loaded_document = load_pdf(args.pdf_path)
        processed_document = process_document(loaded_document, config)
        output_path = save_processed_document(
            processed_document,
            args.output_dir,
        )
    except (
        FileNotFoundError,
        ValueError,
        PDFIngestionError,
    ) as exc:
        parser.error(str(exc))

    chunk_sizes = [chunk.char_count for chunk in processed_document.chunks]
    removed_line_count = sum(
        len(page.removed_margin_lines) for page in processed_document.pages
    )
    removed_unique_lines = sorted(
        {
            line
            for page in processed_document.pages
            for line in page.removed_margin_lines
        }
    )

    original_non_whitespace = sum(
        len(re.sub(r"\s+", "", page.text)) for page in loaded_document.pages
    )
    cleaned_non_whitespace = sum(
        len(re.sub(r"\s+", "", page.cleaned_text)) for page in processed_document.pages
    )
    content_retention = cleaned_non_whitespace / original_non_whitespace * 100

    print("PDF processing completed.")
    print(f"Source: {processed_document.source_name}")
    print(f"Document ID: {processed_document.document_id}")
    print(f"Physical pages: {processed_document.page_count}")
    print(f"Original characters: {loaded_document.total_characters:,}")
    print(f"Cleaned characters: {processed_document.cleaned_character_count:,}")
    print(f"Non-whitespace content retained: {content_retention:.2f}%")
    print(f"Removed margin occurrences: {removed_line_count}")
    print(f"Unique removed margin lines: {len(removed_unique_lines)}")
    print(f"Chunk size: {config.chunk_size}")
    print(f"Chunk overlap: {config.chunk_overlap}")
    print(f"Total chunks: {processed_document.chunk_count}")

    if chunk_sizes:
        print(f"Smallest chunk: {min(chunk_sizes)} characters")
        print(f"Average chunk: {statistics.mean(chunk_sizes):.1f} characters")
        print(f"Median chunk: {statistics.median(chunk_sizes):.1f} characters")
        print(f"Largest chunk: {max(chunk_sizes)} characters")

    if removed_unique_lines:
        print("Removed margin-line examples:")
        for line in removed_unique_lines[:5]:
            print(f"- {line}")

    print(f"Saved output: {output_path}")

    if args.sample_page is not None:
        if not 1 <= args.sample_page <= processed_document.page_count:
            parser.error(
                f"sample page must be between 1 and {processed_document.page_count}"
            )

        page_chunks = [
            chunk
            for chunk in processed_document.chunks
            if chunk.page_number == args.sample_page
        ]

        print(f"Sample physical page: {args.sample_page}")
        print(f"Chunks on sample page: {len(page_chunks)}")

        for chunk in page_chunks[:3]:
            print(
                f"- chunk {chunk.chunk_index}, "
                f"PDF label {chunk.page_label}, "
                f"{chunk.char_count} characters"
            )
            print(f"  {_preview(chunk.text)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
