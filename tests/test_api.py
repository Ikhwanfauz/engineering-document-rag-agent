"""Tests for the FastAPI backend."""

from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app

client = TestClient(app)


@pytest.fixture
def upload_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Use temporary storage instead of the real manuals directory."""
    monkeypatch.setattr(api_main, "UPLOAD_DIRECTORY", tmp_path)
    return tmp_path


def create_pdf_bytes() -> bytes:
    """Create a small readable PDF for upload tests."""
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Engineering safety instructions")
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "4A",
    }


def test_upload_endpoint_saves_valid_pdf(upload_directory: Path) -> None:
    pdf_bytes = create_pdf_bytes()

    response = client.post(
        "/documents/upload",
        files={"file": ("manual.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "manual.pdf"
    assert response.json()["size_bytes"] == len(pdf_bytes)
    assert response.json()["page_count"] == 1
    assert response.json()["text_page_count"] == 1
    assert len(response.json()["document_id"]) == 64
    assert (upload_directory / "manual.pdf").read_bytes() == pdf_bytes


def test_upload_rejects_non_pdf_file(upload_directory: Path) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("manual.txt", b"Not a PDF", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "invalid_file_type"
    assert list(upload_directory.iterdir()) == []


def test_upload_rejects_empty_pdf(upload_directory: Path) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "empty_file"
    assert list(upload_directory.iterdir()) == []


def test_upload_rejects_oversized_pdf(
    upload_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "MAX_UPLOAD_SIZE_BYTES", 10)

    response = client.post(
        "/documents/upload",
        files={"file": ("large.pdf", b"x" * 11, "application/pdf")},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "file_too_large"
    assert list(upload_directory.iterdir()) == []


def test_upload_rejects_unreadable_pdf(upload_directory: Path) -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("broken.pdf", b"Not genuine PDF data", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_pdf"
    assert list(upload_directory.iterdir()) == []
