"""Tests for page-aware PDF ingestion."""

from pathlib import Path

import pymupdf
import pytest

from src.document_loader import PDFIngestionError, extract_text_with_ocr, load_pdf


def _create_test_pdf(path: Path) -> None:
    document = pymupdf.open()
    first_page = document.new_page()
    first_page.insert_text((72, 72), "Safety procedure\nDisconnect power first.")
    document.new_page()
    document.set_metadata({"title": "Fixture Service Manual", "author": "Test Team"})
    document.save(path)
    document.close()


def test_load_pdf_preserves_page_metadata_and_detects_empty_pages(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "fixture.pdf"
    _create_test_pdf(pdf_path)

    result = load_pdf(pdf_path)

    assert result.source_name == "fixture.pdf"
    assert result.page_count == 2
    assert len(result.pages) == 2
    assert result.metadata["title"] == "Fixture Service Manual"
    assert result.metadata["author"] == "Test Team"

    first_page = result.pages[0]
    assert first_page.page_number == 1
    assert first_page.page_label == "1"
    assert first_page.source_name == "fixture.pdf"
    assert "Disconnect power first." in first_page.text
    assert first_page.has_text is True
    assert first_page.is_scanned is False
    assert first_page.extraction_method == "digital_text"
    assert first_page.char_count == len(first_page.text)

    second_page = result.pages[1]
    assert second_page.page_number == 2
    assert second_page.page_label == "2"
    assert second_page.has_text is False
    assert second_page.is_scanned is False
    assert second_page.extraction_method == "none"
    assert result.text_page_count == 1
    assert result.empty_page_numbers == (2,)
    assert result.total_characters == first_page.char_count


def test_load_pdf_rejects_non_pdf_extension(tmp_path: Path) -> None:
    text_path = tmp_path / "manual.txt"
    text_path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Expected a \.pdf file"):
        load_pdf(text_path)


def test_load_pdf_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_pdf(tmp_path / "missing.pdf")


def test_load_pdf_rejects_corrupt_pdf(tmp_path: Path) -> None:
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"this is not valid PDF data")

    with pytest.raises(PDFIngestionError, match="Could not open PDF"):
        load_pdf(corrupt_pdf)


def test_load_pdf_detects_image_only_scanned_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned.pdf"
    image = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 200, 200), False)
    image.clear_with(255)

    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(page.rect, stream=image.tobytes("png"))
    document.save(pdf_path)
    document.close()

    result = load_pdf(pdf_path)

    scanned_page = result.pages[0]
    assert scanned_page.text == ""
    assert scanned_page.has_text is False
    assert scanned_page.is_scanned is True
    assert scanned_page.extraction_method == "none"


def test_load_pdf_detects_scanned_page_with_little_digital_text(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "scanned_with_little_text.pdf"
    image = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 200, 200), False)
    image.clear_with(255)

    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(page.rect, stream=image.tobytes("png"))
    page.insert_text((72, 72), "1")
    document.save(pdf_path)
    document.close()

    result = load_pdf(pdf_path)

    scanned_page = result.pages[0]
    assert scanned_page.text == "1"
    assert scanned_page.has_text is True
    assert scanned_page.is_scanned is True
    assert scanned_page.extraction_method == "digital_text"


def test_extract_text_with_ocr_returns_clean_text() -> None:
    class FakeOCRReader:
        def readtext(
            self,
            _image: object,
            detail: int,
            paragraph: bool,
        ) -> list[str]:
            assert detail == 0
            assert paragraph is True
            return [
                " Disconnect power first. ",
                "",
                "Wear protective gloves.",
            ]

    document = pymupdf.open()
    page = document.new_page()

    try:
        text = extract_text_with_ocr(page, FakeOCRReader())
    finally:
        document.close()

    assert text == "Disconnect power first.\nWear protective gloves."

def test_load_pdf_uses_ocr_fallback_for_scanned_page(tmp_path: Path) -> None:
    class FakeOCRReader:
        def readtext(
            self,
            _image: object,
            detail: int,
            paragraph: bool,
        ) -> list[str]:
            assert detail == 0
            assert paragraph is True
            return ["Emergency stop procedure."]

    pdf_path = tmp_path / "scanned.pdf"
    image = pymupdf.Pixmap(pymupdf.csRGB, (0, 0, 200, 200), False)
    image.clear_with(255)

    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(page.rect, stream=image.tobytes("png"))
    document.save(pdf_path)
    document.close()

    result = load_pdf(pdf_path, ocr_reader=FakeOCRReader())

    scanned_page = result.pages[0]
    assert scanned_page.text == "Emergency stop procedure."
    assert scanned_page.has_text is True
    assert scanned_page.is_scanned is True
    assert scanned_page.extraction_method == "ocr"
