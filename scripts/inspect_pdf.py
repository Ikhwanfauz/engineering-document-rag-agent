"""Inspect page-level PDF extraction from the command line."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean, median

from src.document_loader import PDFIngestionError, load_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a PDF before chunking or vector indexing."
    )
    parser.add_argument("pdf", type=Path, help="Path to a local PDF manual or SOP")
    parser.add_argument(
        "--sample-page",
        type=int,
        help="Show a short text sample from one physical page number",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    try:
        document = load_pdf(args.pdf)
    except (FileNotFoundError, ValueError, PDFIngestionError) as exc:
        print(f"PDF inspection failed: {exc}")
        return 1

    character_counts = [page.char_count for page in document.pages]
    empty_pages = document.empty_page_numbers

    print("PDF INGESTION SUMMARY")
    print(f"File: {document.source_name}")
    print(f"Pages: {document.page_count}")
    print(f"Pages with text: {document.text_page_count}")
    print(f"Empty or image-only pages: {len(empty_pages)}")
    print(f"Total extracted characters: {document.total_characters:,}")
    print(f"Average characters per page: {mean(character_counts):,.1f}")
    print(f"Median characters per page: {median(character_counts):,.1f}")
    print(f"Title: {document.metadata.get('title') or '(not provided)'}")
    print(f"Author: {document.metadata.get('author') or '(not provided)'}")

    if empty_pages:
        page_list = ", ".join(str(number) for number in empty_pages)
        print(f"Pages requiring later OCR review: {page_list}")

    if args.sample_page is not None:
        if not 1 <= args.sample_page <= document.page_count:
            print(
                f"Sample page must be between 1 and {document.page_count}, "
                f"received {args.sample_page}."
            )
            return 1

        page = document.pages[args.sample_page - 1]
        preview = page.text[:500] if page.text else "(no extractable text)"
        print("\nPAGE SAMPLE")
        print(f"Physical page: {page.page_number}")
        print(f"PDF label: {page.page_label}")
        print(preview)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
