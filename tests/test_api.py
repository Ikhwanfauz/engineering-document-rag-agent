"""Tests for the FastAPI backend."""

from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from src.llm_provider import LLMServiceError
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

    def document_chunk_count(self, document_id: str) -> int:
        """Simulate one indexed chunk for an uploaded document."""
        return 1

    def delete_document(self, document_id: str) -> int:
        """Simulate deleting one indexed chunk."""
        return 1


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


class FakeDocumentRetriever:
    """Avoid real vector retrieval during API tests."""

    def __init__(self, embedding_manager: object, vector_store: object) -> None:
        self.embedding_manager = embedding_manager
        self.vector_store = vector_store


class FakeLLMProvider:
    """Avoid connecting to Ollama during API tests."""

    def __init__(
        self,
        model: str,
        temperature: float,
        base_url: str,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.base_url = base_url


class FakeAnsweredRAGPipeline:
    """Return a predictable grounded answer during API tests."""

    def __init__(
        self,
        retriever: object,
        llm_provider: object,
        *,
        minimum_similarity: float,
    ) -> None:
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.minimum_similarity = minimum_similarity

    def answer(
        self,
        question: str,
        *,
        top_k: int,
        document_id: str | None,
    ) -> SimpleNamespace:
        citation = SimpleNamespace(
            document_id=document_id or "a" * 64,
            source_name="manual.pdf",
            page_number=50,
            page_label="50",
            label="manual.pdf, page 50",
            excerpt="Support the joint before removing the clamp.",
        )

        evidence = SimpleNamespace(
            chunk_id="chunk-001",
            document_id=document_id or "a" * 64,
            source_name="manual.pdf",
            page_number=50,
            page_label="50",
            chunk_index=0,
            text="Support the joint before removing the clamp.",
            distance=0.16,
            similarity_score=0.84,
            citation="manual.pdf, page 50",
        )

        return SimpleNamespace(
            question=question,
            answer="The joint must be supported before removing the clamp.",
            citations=(citation,),
            evidence=(evidence,),
            abstained=False,
        )


class FakeAbstainedRAGPipeline(FakeAnsweredRAGPipeline):
    """Return a predictable don't-know response during API tests."""

    def answer(
        self,
        question: str,
        *,
        top_k: int,
        document_id: str | None,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            question=question,
            answer="I don't know based on the uploaded documents.",
            citations=(),
            evidence=(),
            abstained=True,
        )


class FakeInvalidRAGPipeline(FakeAnsweredRAGPipeline):
    """Simulate request validation inside the RAG pipeline."""

    def answer(
        self,
        question: str,
        *,
        top_k: int,
        document_id: str | None,
    ) -> SimpleNamespace:
        raise ValueError("Question cannot be empty")

class FakeLLMServiceFailureRAGPipeline(FakeAnsweredRAGPipeline):
    """Simulate an unavailable language-model service."""

    def answer(
        self,
        question: str,
        *,
        top_k: int,
        document_id: str | None,
    ) -> SimpleNamespace:
        raise LLMServiceError(
            "The language-model service could not generate a response"
        )


class FakeGroundingFailureRAGPipeline(FakeAnsweredRAGPipeline):
    """Simulate an answer that fails grounding validation."""

    def answer(
        self,
        question: str,
        *,
        top_k: int,
        document_id: str | None,
    ) -> SimpleNamespace:
        raise api_main.GroundingValidationError(
            "Generated answer softened a mandatory document instruction"
        )


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "4C",
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


def test_ask_endpoint_returns_grounded_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(api_main, "VectorStoreManager", FakeVectorStoreManager)
    monkeypatch.setattr(api_main, "DocumentRetriever", FakeDocumentRetriever)
    monkeypatch.setattr(api_main, "OllamaLLMProvider", FakeLLMProvider)
    monkeypatch.setattr(api_main, "RAGPipeline", FakeAnsweredRAGPipeline)

    response = client.post(
        "/questions/ask",
        json={
            "question": "What must be done before removing the clamp?",
            "document_id": "a" * 64,
            "top_k": 3,
            "minimum_similarity": 0.60,
        },
    )

    assert response.status_code == 200
    assert response.json()["question"] == (
        "What must be done before removing the clamp?"
    )
    assert response.json()["answer"] == (
        "The joint must be supported before removing the clamp."
    )
    assert response.json()["status"] == "ANSWERED"
    assert response.json()["abstained"] is False
    assert response.json()["accepted_evidence_count"] == 1
    evidence = response.json()["evidence"][0]
    assert evidence["chunk_id"] == "chunk-001"
    assert evidence["document_id"] == "a" * 64
    assert evidence["source_name"] == "manual.pdf"
    assert evidence["page_number"] == 50
    assert evidence["page_label"] == "50"
    assert evidence["chunk_index"] == 0
    assert evidence["text"] == "Support the joint before removing the clamp."
    assert evidence["distance"] == 0.16
    assert evidence["similarity_score"] == 0.84
    assert evidence["citation"] == "manual.pdf, page 50"
    assert response.json()["citations"][0]["document_id"] == "a" * 64
    assert response.json()["citations"][0]["source_name"] == "manual.pdf"
    assert response.json()["citations"][0]["page_number"] == 50
    assert response.json()["citations"][0]["page_label"] == "50"
    assert response.json()["citations"][0]["label"] == "manual.pdf, page 50"
    assert response.json()["elapsed_seconds"] >= 0


def test_ask_endpoint_returns_abstained_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(api_main, "VectorStoreManager", FakeVectorStoreManager)
    monkeypatch.setattr(api_main, "DocumentRetriever", FakeDocumentRetriever)
    monkeypatch.setattr(api_main, "OllamaLLMProvider", FakeLLMProvider)
    monkeypatch.setattr(api_main, "RAGPipeline", FakeAbstainedRAGPipeline)

    response = client.post(
        "/questions/ask",
        json={
            "question": "What is the robot Wi-Fi password?",
            "top_k": 3,
            "minimum_similarity": 0.60,
        },
    )

    assert response.status_code == 200
    assert response.json()["question"] == "What is the robot Wi-Fi password?"
    assert response.json()["answer"] == (
        "I don't know based on the uploaded documents."
    )
    assert response.json()["status"] == "ABSTAINED"
    assert response.json()["abstained"] is True
    assert response.json()["citations"] == []
    assert response.json()["evidence"] == []
    assert response.json()["accepted_evidence_count"] == 0
    assert response.json()["elapsed_seconds"] >= 0


def test_ask_endpoint_rejects_invalid_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(api_main, "VectorStoreManager", FakeVectorStoreManager)
    monkeypatch.setattr(api_main, "DocumentRetriever", FakeDocumentRetriever)
    monkeypatch.setattr(api_main, "OllamaLLMProvider", FakeLLMProvider)
    monkeypatch.setattr(api_main, "RAGPipeline", FakeInvalidRAGPipeline)

    response = client.post(
        "/questions/ask",
        json={
            "question": "",
            "top_k": 3,
            "minimum_similarity": 0.60,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_question_request"
    assert response.json()["detail"]["message"] == "Question cannot be empty"

def test_ask_endpoint_reports_llm_service_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(api_main, "VectorStoreManager", FakeVectorStoreManager)
    monkeypatch.setattr(api_main, "DocumentRetriever", FakeDocumentRetriever)
    monkeypatch.setattr(api_main, "OllamaLLMProvider", FakeLLMProvider)
    monkeypatch.setattr(
        api_main,
        "RAGPipeline",
        FakeLLMServiceFailureRAGPipeline,
    )

    response = client.post(
        "/questions/ask",
        json={
            "question": "How should the joint be supported?",
            "top_k": 3,
            "minimum_similarity": 0.60,
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "llm_service_unavailable"
    assert response.json()["detail"]["message"] == (
        "The language-model service is unavailable."
    )


def test_ask_endpoint_reports_grounding_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(api_main, "VectorStoreManager", FakeVectorStoreManager)
    monkeypatch.setattr(api_main, "DocumentRetriever", FakeDocumentRetriever)
    monkeypatch.setattr(api_main, "OllamaLLMProvider", FakeLLMProvider)
    monkeypatch.setattr(
        api_main,
        "RAGPipeline",
        FakeGroundingFailureRAGPipeline,
    )

    response = client.post(
        "/questions/ask",
        json={
            "question": "What mandatory action is required?",
            "top_k": 3,
            "minimum_similarity": 0.60,
        },
    )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "grounding_validation_failed"
    assert response.json()["detail"]["message"] == (
        "The generated answer failed grounding validation."
    )


def test_list_documents_returns_uploaded_pdfs(
    upload_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_bytes = create_pdf_bytes()
    (upload_directory / "manual.pdf").write_bytes(pdf_bytes)

    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(api_main, "VectorStoreManager", FakeVectorStoreManager)

    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json()["total_documents"] == 1

    document = response.json()["documents"][0]
    assert document["filename"] == "manual.pdf"
    assert document["size_bytes"] == len(pdf_bytes)
    assert document["page_count"] == 1
    assert document["text_page_count"] == 1
    assert document["indexed"] is True
    assert document["indexed_chunk_count"] == 1
    assert len(document["document_id"]) == 64


def test_get_document_returns_document_details(
    upload_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (upload_directory / "manual.pdf").write_bytes(create_pdf_bytes())

    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(api_main, "VectorStoreManager", FakeVectorStoreManager)

    response = client.get("/documents/manual.pdf")

    assert response.status_code == 200
    assert response.json()["filename"] == "manual.pdf"
    assert response.json()["page_count"] == 1
    assert response.json()["text_page_count"] == 1
    assert response.json()["indexed"] is True
    assert response.json()["indexed_chunk_count"] == 1
    assert len(response.json()["document_id"]) == 64


def test_delete_document_removes_pdf_and_indexed_chunks(
    upload_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = upload_directory / "manual.pdf"
    pdf_path.write_bytes(create_pdf_bytes())

    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(api_main, "VectorStoreManager", FakeVectorStoreManager)

    response = client.delete("/documents/manual.pdf")

    assert response.status_code == 200
    assert response.json()["filename"] == "manual.pdf"
    assert response.json()["removed_chunks"] == 1
    assert len(response.json()["document_id"]) == 64
    assert not pdf_path.exists()


def test_get_document_rejects_missing_document(
    upload_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(api_main, "VectorStoreManager", FakeVectorStoreManager)

    response = client.get("/documents/missing.pdf")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "document_not_found"
