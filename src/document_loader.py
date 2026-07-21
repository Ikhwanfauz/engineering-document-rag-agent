"""Load PDF manuals while preserving page-level citation metadata.

Version 1A deliberately stops before chunking, embeddings, and OCR. Its only
job is to turn a readable PDF into a predictable document model without losing
the source filename, physical page number, or PDF page label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pymupdf


class PDFIngestionError(RuntimeError):
    """Raised when a PDF cannot be opened or safely extracted."""


@dataclass(frozen=True, slots=True)
class PDFPage:
    """Extracted text and citation metadata for one physical PDF page."""

    source_name: str
    page_number: int
    page_label: str
    text: str

    @property
    def char_count(self) -> int:
        """Number of extracted characters after whitespace normalization."""
        return len(self.text)

    @property
    def has_text(self) -> bool:
        """Whether the page contains any extractable text."""
        return bool(self.text)


@dataclass(frozen=True, slots=True)
class LoadedPDF:
    """A PDF manual represented as ordered, page-aware extracted text."""

    source_name: str
    page_count: int
    metadata: Mapping[str, str]
    pages: tuple[PDFPage, ...]

    @property
    def text_page_count(self) -> int:
        """Number of pages with at least one extracted character."""
        return sum(page.has_text for page in self.pages)

    @property
    def empty_page_numbers(self) -> tuple[int, ...]:
        """Physical page numbers that have no extractable text."""
        return tuple(page.page_number for page in self.pages if not page.has_text)

    @property
    def total_characters(self) -> int:
        """Total extracted character count across the complete document."""
        return sum(page.char_count for page in self.pages)


def _normalize_text(text: str) -> str:
    """Normalize newlines and trailing whitespace without changing wording."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def _validate_pdf_path(path: str | Path) -> Path:
    """Resolve and validate a local PDF path before opening it."""
    pdf_path = Path(path).expanduser()

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, received: {pdf_path.name}")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")
    if not pdf_path.is_file():
        raise ValueError(f"PDF path is not a file: {pdf_path}")

    return pdf_path


def load_pdf(path: str | Path) -> LoadedPDF:
    """Extract a PDF into ordered pages with stable citation metadata.

    Physical page numbers are one-based. ``page_label`` preserves a label
    embedded in the PDF when one exists and otherwise falls back to the
    physical page number. Password-protected documents are rejected in this
    version so the caller receives a clear failure instead of partial output.
    """
    pdf_path = _validate_pdf_path(path)

    try:
        document = pymupdf.open(
            stream=pdf_path.read_bytes(),
            filetype="pdf",
        )
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise PDFIngestionError(f"Could not open PDF: {pdf_path.name}") from exc

    with document:
        if document.needs_pass:
            raise PDFIngestionError(
                f"Password-protected PDFs are not supported: {pdf_path.name}"
            )
        if document.page_count == 0:
            raise PDFIngestionError(f"PDF contains no pages: {pdf_path.name}")

        metadata = {
            str(key): str(value or "") for key, value in document.metadata.items()
        }
        pages: list[PDFPage] = []

        for page_index, page in enumerate(document):
            page_number = page_index + 1
            try:
                text = _normalize_text(page.get_text("text", sort=True))
            except RuntimeError as exc:
                raise PDFIngestionError(
                    f"Failed to extract page {page_number} from {pdf_path.name}"
                ) from exc

            page_label = (page.get_label() or str(page_number)).strip()
            pages.append(
                PDFPage(
                    source_name=pdf_path.name,
                    page_number=page_number,
                    page_label=page_label,
                    text=text,
                )
            )

    return LoadedPDF(
        source_name=pdf_path.name,
        page_count=len(pages),
        metadata=metadata,
        pages=tuple(pages),
    )
