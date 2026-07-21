"""Tests for the FastAPI backend."""

from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from src.vector_store import IndexingReport

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


class FakeEmbeddingManager:
    """Avoid loading the real embedding model during API tests."""

    def __init__(self, config: object) -> None:
        self.config = config


class FakeVectorStoreManager:
    """Return a predictable indexing report without using ChromaDB."""

    def __init__(self, embedding_manager: object, config: object) -> None:
        self.embedding_manager = embedding_manager
        self.config = config

    def index_document(self, document: object) -> IndexingReport:
        return IndexingReport(
            document_id=document.document_id,
            total_chunks=len(document.chunks),
            added_chunks=len(document.chunks),
            existing_chunks=0,
            removed_chunks=0,
            collection_count=len(document.chunks),
        )


class FakeExistingVectorStoreManager(FakeVectorStoreManager):
    """Simulate indexing a document whose chunks already exist."""

    def index_document(self, document: object) -> IndexingReport:
        return IndexingReport(
            document_id=document.document_id,
            total_chunks=len(document.chunks),
            added_chunks=0,
            existing_chunks=len(document.chunks),
            removed_chunks=0,
            collection_count=len(document.chunks),
        )


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "4B",
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


def test_index_endpoint_indexes_uploaded_pdf(
    upload_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = upload_directory / "manual.pdf"
    pdf_path.write_bytes(create_pdf_bytes())

    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(api_main, "VectorStoreManager", FakeVectorStoreManager)

    response = client.post("/documents/manual.pdf/index")

    assert response.status_code == 200
    assert response.json()["filename"] == "manual.pdf"
    assert response.json()["page_count"] == 1
    assert response.json()["total_chunks"] == 1
    assert response.json()["added_chunks"] == 1
    assert response.json()["existing_chunks"] == 0
    assert response.json()["removed_chunks"] == 0
    assert response.json()["collection_count"] == 1
    assert len(response.json()["document_id"]) == 64
    assert response.json()["elapsed_seconds"] >= 0


def test_index_endpoint_rejects_missing_document(
    upload_directory: Path,
) -> None:
    response = client.post("/documents/missing.pdf/index")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "document_not_found"


def test_index_endpoint_rejects_unreadable_pdf(
    upload_directory: Path,
) -> None:
    pdf_path = upload_directory / "broken.pdf"
    pdf_path.write_bytes(b"Not genuine PDF data")

    response = client.post("/documents/broken.pdf/index")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "indexing_failed"


def test_index_endpoint_reports_existing_chunks(
    upload_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = upload_directory / "manual.pdf"
    pdf_path.write_bytes(create_pdf_bytes())

    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(
        api_main,
        "VectorStoreManager",
        FakeExistingVectorStoreManager,
    )

    response = client.post("/documents/manual.pdf/index")

    assert response.status_code == 200
    assert response.json()["total_chunks"] == 1
    assert response.json()["added_chunks"] == 0
    assert response.json()["existing_chunks"] == 1
    assert response.json()["removed_chunks"] == 0
    assert response.json()["collection_count"] == 1
