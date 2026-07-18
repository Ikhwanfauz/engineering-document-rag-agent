"""Clean and split extracted PDF pages into citation-safe chunks.

Every chunk belongs to exactly one physical PDF page. This prevents retrieved
evidence from receiving an incorrect or ambiguous page citation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_loader import LoadedPDF, PDFPage


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configuration for document cleaning and chunking."""

    chunk_size: int = 1000
    chunk_overlap: int = 150
    margin_line_count: int = 3
    repeated_line_page_ratio: float = 0.25
    repeated_line_min_pages: int = 3

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.margin_line_count < 0:
            raise ValueError("margin_line_count cannot be negative")
        if not 0 < self.repeated_line_page_ratio <= 1:
            raise ValueError("repeated_line_page_ratio must be between 0 and 1")
        if self.repeated_line_min_pages <= 0:
            raise ValueError("repeated_line_min_pages must be greater than zero")


@dataclass(frozen=True, slots=True)
class ProcessedPage:
    """Cleaned text and cleaning diagnostics for one PDF page."""

    source_name: str
    page_number: int
    page_label: str
    original_char_count: int
    cleaned_text: str
    removed_margin_lines: tuple[str, ...]

    @property
    def cleaned_char_count(self) -> int:
        return len(self.cleaned_text)


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """One citation-safe text chunk from exactly one PDF page."""

    chunk_id: str
    document_id: str
    source_name: str
    page_number: int
    page_label: str
    chunk_index: int
    text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    """Normalized pages and chunks produced from one loaded PDF."""

    document_id: str
    source_name: str
    page_count: int
    config: ChunkingConfig
    pages: tuple[ProcessedPage, ...]
    chunks: tuple[DocumentChunk, ...]

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def cleaned_character_count(self) -> int:
        return sum(page.cleaned_char_count for page in self.pages)


DEFAULT_CONFIG = ChunkingConfig()

_PROTECTED_MARGIN_TERMS = (
    "warning",
    "danger",
    "caution",
    "safety",
    "important",
    "note",
)


def compute_document_id(document: LoadedPDF) -> str:
    """Create a stable SHA-256 ID from the extracted document content.

    The filename is deliberately excluded. Renaming an identical PDF therefore
    does not make the system treat it as a different document.
    """
    digest = hashlib.sha256()
    digest.update(str(document.page_count).encode("utf-8"))

    for page in document.pages:
        digest.update(b"\x1ePAGE\x1f")
        digest.update(str(page.page_number).encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(page.page_label.encode("utf-8"))
        digest.update(b"\x1f")
        digest.update(page.text.encode("utf-8"))

    return digest.hexdigest()


def _normalize_line(line: str) -> str:
    """Collapse spacing inside one line while preserving its wording."""
    return re.sub(r"[ \t]+", " ", line).strip()


def _margin_signature(raw_line: str) -> str:
    """Create a comparison form for repeated margin lines.

    Numbers are replaced only in lines containing wide layout spacing. This
    detects page-number footers without treating numbered section headings as
    repeated noise.
    """
    normalized = _normalize_line(raw_line)

    if re.search(r"[ \t]{3,}", raw_line):
        normalized = re.sub(r"\d+", "<page-number>", normalized)

    return normalized


def _nonempty_raw_lines(page: PDFPage) -> list[str]:
    return [line for line in page.text.splitlines() if _normalize_line(line)]


def _is_protected_margin_line(line: str) -> bool:
    """Protect safety-related labels from automatic removal."""
    lowered = line.casefold()
    return any(term in lowered for term in _PROTECTED_MARGIN_TERMS)


def _is_copyright_fragment(text: str) -> bool:
    """Return whether one line is clearly part of a copyright footer."""
    normalized = _normalize_line(text).casefold()

    patterns = (
        r"^copyright(?:\s+.*)?$",
        r"^©(?:\s+.*)?$",
        r"^all rights reserved\.?$",
        r"^reserved\.?$",
        r"^rights$",
        r"^all$",
        r"^by$",
        r"^(?:universal\s+robots(?:\s+a/s\.?)?)$",
        r"^universal$",
        r"^robots$",
        r"^a/s\.?$",
        r"^\d{4}\s*[–-]\s*\d{4}$",
    )

    return any(re.fullmatch(pattern, normalized) for pattern in patterns)


def _copyright_block_positions(raw_lines: list[str]) -> set[int]:
    """Find only clear copyright fragments near the page bottom."""
    nonempty_positions = [
        index for index, line in enumerate(raw_lines) if _normalize_line(line)
    ]

    removable_positions: set[int] = set()

    # First detect a clear Copyright or © marker near the page bottom.
    for rank, position in enumerate(nonempty_positions):
        normalized = _normalize_line(raw_lines[position]).casefold()

        is_marker = (
            "copyright" in normalized or "©" in normalized
        ) and _is_copyright_fragment(raw_lines[position])
        is_near_bottom = rank >= len(nonempty_positions) - 15

        if not is_marker or not is_near_bottom:
            continue

        removable_positions.add(position)

        previous_rank = rank - 1
        while previous_rank >= 0:
            previous_position = nonempty_positions[previous_rank]

            if not _is_copyright_fragment(raw_lines[previous_position]):
                break

            removable_positions.add(previous_position)
            previous_rank -= 1

        next_rank = rank + 1
        while next_rank < len(nonempty_positions):
            next_position = nonempty_positions[next_rank]

            if not _is_copyright_fragment(raw_lines[next_position]):
                break

            removable_positions.add(next_position)
            next_rank += 1

    # Some pages contain the vertical footer words without an attached
    # Copyright marker. Detect only this exact unusual sequence.
    footer_sequence = [
        "reserved.",
        "rights",
        "all",
        "a/s.",
        "robots",
        "universal",
        "by",
    ]
    normalized_nonempty = [
        _normalize_line(raw_lines[position]).casefold()
        for position in nonempty_positions
    ]
    sequence_length = len(footer_sequence)

    for start in range(len(normalized_nonempty) - sequence_length + 1):
        end = start + sequence_length

        if normalized_nonempty[start:end] != footer_sequence:
            continue

        for sequence_rank in range(start, end):
            removable_positions.add(nonempty_positions[sequence_rank])

        previous_rank = start - 1
        while previous_rank >= 0:
            previous_position = nonempty_positions[previous_rank]

            if not _is_copyright_fragment(raw_lines[previous_position]):
                break

            removable_positions.add(previous_position)
            previous_rank -= 1

        next_rank = end
        while next_rank < len(nonempty_positions):
            next_position = nonempty_positions[next_rank]

            if not _is_copyright_fragment(raw_lines[next_position]):
                break

            removable_positions.add(next_position)
            next_rank += 1

    known_footer_fragments = {
        "reserved.",
        "rights",
        "all",
        "a/s.",
        "robots",
        "universal",
        "by",
    }
    detected_footer_fragments = {
        _normalize_line(raw_lines[position]).casefold()
        for position in nonempty_positions
        if (_normalize_line(raw_lines[position]).casefold() in known_footer_fragments)
    }

    if len(detected_footer_fragments) >= 5:
        for position in nonempty_positions:
            if _is_copyright_fragment(raw_lines[position]):
                removable_positions.add(position)

    return removable_positions


def _strip_embedded_copyright_noise(
    raw_line: str,
    *,
    strip_vertical_fragments: bool = False,
) -> tuple[str, tuple[str, ...]]:
    """Strip attached copyright markers without deleting body text."""
    cleaned = _normalize_line(raw_line)
    removed_fragments: list[str] = []

    cleaned, copyright_count = re.subn(
        r"\s*copyright\s*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    if copyright_count:
        removed_fragments.append("Copyright")

    if "©" in cleaned:
        cleaned = cleaned.replace("©", " ")
        removed_fragments.append("©")

        cleaned, year_count = re.subn(
            r"^\d{4}\s*[–-]\s*\d{4}\s+",
            "",
            cleaned,
        )
        if year_count:
            removed_fragments.append("copyright year range")

    if strip_vertical_fragments:
        fragment_patterns = (
            ("reserved.", r"(?:^|\s+)reserved\.\s*$"),
            ("rights", r"(?:^|\s+)rights\s*$"),
            ("All", r"(?:^|\s+)all\s*$"),
            ("A/S.", r"(?:^|\s+)a/s\.\s*$"),
            ("Robots", r"(?:^|\s+)robots\s*$"),
            ("Universal", r"(?:^|\s+)universal\s*$"),
            ("by", r"(?:^|\s+)by\s*$"),
            (
                "copyright year range",
                r"(?:^|\s+)\d{4}\s*[–-]\s*\d{4}\s*$",
            ),
        )

        for fragment_name, pattern in fragment_patterns:
            cleaned, fragment_count = re.subn(
                pattern,
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            if fragment_count:
                removed_fragments.append(fragment_name)

    return _normalize_line(cleaned), tuple(removed_fragments)


def _comparison_text(text: str) -> str:
    """Normalize text only for duplicate comparison."""
    normalized = re.sub(r"\s+", " ", text).strip()
    normalized = re.sub(r"-\s+", "-", normalized)
    return normalized.casefold()


def _remove_duplicate_wrapped_lines(
    lines: list[str],
) -> tuple[list[str], list[str]]:
    """Remove an adjacent heading repeated as one wrapped or unwrapped line."""
    cleaned: list[str] = []
    removed: list[str] = []
    index = 0

    while index < len(lines):
        current = lines[index]

        if not current:
            cleaned.append(current)
            index += 1
            continue

        if (
            index + 2 < len(lines)
            and lines[index + 1]
            and lines[index + 2]
            and _comparison_text(current)
            == _comparison_text(f"{lines[index + 1]} {lines[index + 2]}")
        ):
            cleaned.append(current)
            removed.extend([lines[index + 1], lines[index + 2]])
            index += 3
            continue

        if (
            index + 1 < len(lines)
            and lines[index + 1]
            and _comparison_text(current) == _comparison_text(lines[index + 1])
        ):
            cleaned.append(current)
            removed.append(lines[index + 1])
            index += 2
            continue

        cleaned.append(current)
        index += 1

    return cleaned, removed


def find_repeated_margin_lines(
    document: LoadedPDF,
    config: ChunkingConfig = DEFAULT_CONFIG,
) -> frozenset[str]:
    """Find repeated lines occurring near page tops or bottoms."""
    text_pages = [page for page in document.pages if page.has_text]
    if not text_pages or config.margin_line_count == 0:
        return frozenset()

    occurrences: Counter[str] = Counter()

    for page in text_pages:
        raw_lines = _nonempty_raw_lines(page)
        margin_lines = (
            raw_lines[: config.margin_line_count]
            + raw_lines[-config.margin_line_count :]
        )

        signatures = {
            _margin_signature(line) for line in margin_lines if _margin_signature(line)
        }
        occurrences.update(signatures)

    required_pages = max(
        config.repeated_line_min_pages,
        math.ceil(len(text_pages) * config.repeated_line_page_ratio),
    )

    repeated = {
        line
        for line, count in occurrences.items()
        if count >= required_pages
        and len(line) >= 3
        and not _is_protected_margin_line(line)
    }

    return frozenset(repeated)


def clean_page(
    page: PDFPage,
    repeated_margin_lines: frozenset[str],
    config: ChunkingConfig = DEFAULT_CONFIG,
) -> ProcessedPage:
    """Remove confirmed layout noise and duplicate headings from one page."""
    raw_lines = page.text.splitlines()
    nonempty_positions = [
        index for index, line in enumerate(raw_lines) if _normalize_line(line)
    ]

    margin_positions = set(
        nonempty_positions[: config.margin_line_count]
        + nonempty_positions[-config.margin_line_count :]
    )
    copyright_positions = _copyright_block_positions(raw_lines)

    kept_lines: list[str] = []
    removed_lines: list[str] = []

    for index, raw_line in enumerate(raw_lines):
        normalized = _normalize_line(raw_line)

        if index in copyright_positions:
            if normalized:
                removed_lines.append(normalized)
            continue

        if (
            index in margin_positions
            and _margin_signature(raw_line) in repeated_margin_lines
        ):
            if normalized:
                removed_lines.append(normalized)
            continue

        cleaned_line, removed_fragments = _strip_embedded_copyright_noise(
            raw_line,
            strip_vertical_fragments=bool(copyright_positions),
        )
        kept_lines.append(cleaned_line)
        removed_lines.extend(removed_fragments)

    kept_lines, duplicate_lines = _remove_duplicate_wrapped_lines(kept_lines)
    removed_lines.extend(duplicate_lines)

    cleaned_text = "\n".join(kept_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()

    return ProcessedPage(
        source_name=page.source_name,
        page_number=page.page_number,
        page_label=page.page_label,
        original_char_count=page.char_count,
        cleaned_text=cleaned_text,
        removed_margin_lines=tuple(removed_lines),
    )


def _create_chunk_id(
    document_id: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> str:
    identity = f"{document_id}:{page_number}:{chunk_index}:{text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def process_document(
    document: LoadedPDF,
    config: ChunkingConfig = DEFAULT_CONFIG,
) -> ProcessedDocument:
    """Clean and chunk a loaded PDF without crossing page boundaries."""
    document_id = compute_document_id(document)
    repeated_lines = find_repeated_margin_lines(document, config)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    processed_pages: list[ProcessedPage] = []
    chunks: list[DocumentChunk] = []

    for page in document.pages:
        processed_page = clean_page(page, repeated_lines, config)
        processed_pages.append(processed_page)

        if not processed_page.cleaned_text:
            continue

        page_chunks = splitter.split_text(processed_page.cleaned_text)

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunks.append(
                DocumentChunk(
                    chunk_id=_create_chunk_id(
                        document_id,
                        page.page_number,
                        chunk_index,
                        chunk_text,
                    ),
                    document_id=document_id,
                    source_name=page.source_name,
                    page_number=page.page_number,
                    page_label=page.page_label,
                    chunk_index=chunk_index,
                    text=chunk_text,
                )
            )

    return ProcessedDocument(
        document_id=document_id,
        source_name=document.source_name,
        page_count=document.page_count,
        config=config,
        pages=tuple(processed_pages),
        chunks=tuple(chunks),
    )


def save_processed_document(
    document: ProcessedDocument,
    output_directory: str | Path,
) -> Path:
    """Save normalized pages, chunks, and citation metadata as JSON."""
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_stem = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        Path(document.source_name).stem,
    ).strip("_")

    output_path = output_dir / (
        f"{safe_stem}_{document.document_id[:12]}"
        f"_c{document.config.chunk_size}"
        f"_o{document.config.chunk_overlap}.json"
    )

    output_path.write_text(
        json.dumps(asdict(document), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_path
