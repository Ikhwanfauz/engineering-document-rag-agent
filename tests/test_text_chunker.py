"""Tests for citation-safe PDF cleaning and chunking."""

import json

import pytest

from src.document_loader import LoadedPDF, PDFPage
from src.text_chunker import (
    ChunkingConfig,
    compute_document_id,
    find_repeated_margin_lines,
    process_document,
    save_processed_document,
)


def _make_document(
    page_texts: list[str],
    source_name: str = "manual.pdf",
) -> LoadedPDF:
    pages = tuple(
        PDFPage(
            source_name=source_name,
            page_number=index,
            page_label=str(index),
            text=text,
        )
        for index, text in enumerate(page_texts, start=1)
    )

    return LoadedPDF(
        source_name=source_name,
        page_count=len(pages),
        metadata={},
        pages=pages,
    )


def test_chunks_never_cross_page_boundaries() -> None:
    document = _make_document(
        [
            "Alpha maintenance procedure. " * 20,
            "Beta calibration procedure. " * 20,
        ]
    )
    config = ChunkingConfig(
        chunk_size=100,
        chunk_overlap=20,
        margin_line_count=0,
    )

    result = process_document(document, config)

    page_one_chunks = [chunk for chunk in result.chunks if chunk.page_number == 1]
    page_two_chunks = [chunk for chunk in result.chunks if chunk.page_number == 2]

    assert len(page_one_chunks) > 1
    assert len(page_two_chunks) > 1

    assert all("Beta" not in chunk.text for chunk in page_one_chunks)
    assert all("Alpha" not in chunk.text for chunk in page_two_chunks)

    assert [chunk.chunk_index for chunk in page_one_chunks] == list(
        range(len(page_one_chunks))
    )
    assert [chunk.chunk_index for chunk in page_two_chunks] == list(
        range(len(page_two_chunks))
    )

    assert all(chunk.source_name == "manual.pdf" for chunk in result.chunks)
    assert all(chunk.char_count <= config.chunk_size for chunk in result.chunks)


def test_repeated_margin_cleaning_preserves_safety_labels() -> None:
    document = _make_document(
        [
            (
                "Service Manual\n"
                f"Section {page_number}\n"
                f"Maintenance instruction {page_number}\n"
                "WARNING\n"
                "Universal Robots"
            )
            for page_number in range(1, 5)
        ]
    )
    config = ChunkingConfig(
        chunk_size=200,
        chunk_overlap=20,
        margin_line_count=2,
        repeated_line_page_ratio=0.5,
        repeated_line_min_pages=2,
    )

    repeated_lines = find_repeated_margin_lines(document, config)
    result = process_document(document, config)

    assert "Service Manual" in repeated_lines
    assert "Universal Robots" in repeated_lines
    assert "WARNING" not in repeated_lines

    for page in result.pages:
        assert "Service Manual" not in page.cleaned_text
        assert "Universal Robots" not in page.cleaned_text
        assert "WARNING" in page.cleaned_text
        assert set(page.removed_margin_lines) == {
            "Service Manual",
            "Universal Robots",
        }


def test_document_id_is_stable_when_file_is_renamed() -> None:
    first_document = _make_document(
        ["Disconnect the electrical supply."],
        source_name="original-name.pdf",
    )
    renamed_document = _make_document(
        ["Disconnect the electrical supply."],
        source_name="renamed-document.pdf",
    )
    changed_document = _make_document(
        ["Keep the electrical supply connected."],
        source_name="original-name.pdf",
    )

    assert compute_document_id(first_document) == compute_document_id(renamed_document)
    assert compute_document_id(first_document) != compute_document_id(changed_document)


def test_processed_document_can_be_saved_as_json(tmp_path) -> None:
    document = _make_document(["Check the emergency stop before operation."])
    processed = process_document(document)

    output_path = save_processed_document(processed, tmp_path)
    saved_data = json.loads(output_path.read_text(encoding="utf-8"))

    assert output_path.exists()
    assert output_path.name.startswith("manual_")
    assert saved_data["document_id"] == processed.document_id
    assert saved_data["source_name"] == "manual.pdf"
    assert saved_data["page_count"] == 1
    assert len(saved_data["chunks"]) == 1
    assert saved_data["chunks"][0]["page_number"] == 1
    assert saved_data["chunks"][0]["source_name"] == "manual.pdf"

    def test_embedded_copyright_noise_does_not_delete_body_text() -> None:
        document = _make_document(
            [
                (
                    "3. Remove the USB stick after completion.Copyright\n"
                    "2009–2025 Disassembling Clamp Connection ©"
                )
            ]
        )

    result = process_document(document)
    cleaned_text = result.pages[0].cleaned_text

    assert "Remove the USB stick after completion." in cleaned_text
    assert "Disassembling Clamp Connection" in cleaned_text
    assert "Copyright" not in cleaned_text
    assert "©" not in cleaned_text
    assert "2009–2025" not in cleaned_text


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("chunk_size", 0),
        ("chunk_overlap", -1),
        ("repeated_line_page_ratio", 0),
        ("repeated_line_min_pages", 0),
    ],
)
def test_invalid_chunking_configuration_is_rejected(
    field_name: str,
    field_value: int | float,
) -> None:
    values = {
        "chunk_size": 1000,
        "chunk_overlap": 150,
        "margin_line_count": 3,
        "repeated_line_page_ratio": 0.25,
        "repeated_line_min_pages": 3,
    }
    values[field_name] = field_value

    with pytest.raises(ValueError):
        ChunkingConfig(**values)
