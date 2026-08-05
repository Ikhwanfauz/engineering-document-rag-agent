"""Tests for the FastAPI backend."""

from pathlib import Path
from types import SimpleNamespace

import sqlite3
import pymupdf
import pytest
import fitz
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from src.llm_provider import LLMServiceError
from src.vector_store import IndexingReport
from src.checklist_agent import EvidenceCategory

client = TestClient(app)


@pytest.fixture
def upload_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Use temporary storage instead of the real manuals directory."""
    monkeypatch.setattr(api_main, "UPLOAD_DIRECTORY", tmp_path)
    return tmp_path

@pytest.fixture(autouse=True)
def stored_interactions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    """Capture API interaction logs without writing to the real database."""
    interactions: list[dict[str, object]] = []

    def fake_store_interaction(**kwargs: object) -> int:
        interactions.append(kwargs)
        return len(interactions)

    monkeypatch.setattr(api_main, "store_interaction", fake_store_interaction)
    return interactions


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

class FakeOCRReader:
    """Avoid loading the real EasyOCR model during API tests."""

    def readtext(
        self,
        _image: object,
        detail: int,
        paragraph: bool,
    ) -> list[str]:
        assert detail == 0
        assert paragraph is True
        return []


@pytest.fixture(autouse=True)
def fake_ocr_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent EasyOCR from loading during API tests."""
    monkeypatch.setattr(api_main, "get_ocr_reader", FakeOCRReader)


class FakeLLMProvider:
    """Avoid connecting to Ollama during API tests."""

    def __init__(
        self,
        model: str,
        temperature: float,
        base_url: str,
        num_predict: int = 256,
        response_format: object | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.num_predict = num_predict
        self.response_format = response_format


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

class FakeCompletedChecklistWorkflow:
    """Return a predictable checklist awaiting human review."""

    def invoke(self, state: dict[str, object]) -> dict[str, object]:
        citation = SimpleNamespace(
            document_id=state.get("document_id") or "a" * 64,
            source_name="manual.pdf",
            page_number=50,
            page_label="50",
            label="manual.pdf, page 50",
            excerpt="Support the joint before removing the clamp.",
        )
        checklist_item = SimpleNamespace(
            text="Support the joint before removing the clamp.",
            citations=(citation,),
        )
        checklist = SimpleNamespace(
            prerequisites=(checklist_item,),
            tools=(checklist_item,),
            parts=(checklist_item,),
            safety_warnings=(checklist_item,),
            procedure_steps=(checklist_item,),
            review_notes=("Human review is required before use.",),
        )

        return {
            **state,
            "stage": api_main.ChecklistStage.AWAITING_HUMAN_REVIEW,
            "evidence_sufficient": True,
            "missing_evidence_categories": (),
            "generated_checklist": checklist,
        }

class FakeAbstainedChecklistWorkflow:
    """Return a safe abstention when required evidence is missing."""

    def invoke(self, state: dict[str, object]) -> dict[str, object]:
        return {
            **state,
            "stage": api_main.ChecklistStage.ABSTAINED,
            "evidence_sufficient": False,
            "missing_evidence_categories": (
                EvidenceCategory.TOOLS,
                EvidenceCategory.PARTS,
            ),
        }

class FakeInvalidChecklistWorkflow:
    """Reject an invalid checklist request."""

    def invoke(self, state: dict[str, object]) -> dict[str, object]:
        raise ValueError("Checklist request cannot be empty")


def test_health_endpoint_returns_ok() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": "11D",
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

@pytest.mark.parametrize(
    "filename",
    [
        "../manual.pdf",
        "..\\manual.pdf",
        "documents/manual.pdf",
        "manual:stream.pdf",
        "CON.pdf",
    ],
)
def test_upload_rejects_unsafe_filename(
    upload_directory: Path,
    filename: str,
) -> None:
    response = client.post(
        "/documents/upload",
        files={
            "file": (
                filename,
                create_pdf_bytes(),
                "application/pdf",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unsafe_file_path"
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


def test_index_endpoint_uses_ocr_for_scanned_pdf(
    upload_directory: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ScannedOCRReader:
        def readtext(
            self,
            _image: object,
            detail: int,
            paragraph: bool,
        ) -> list[tuple[object, str, float]]:
            assert detail == 1
            assert paragraph is False
            return [
                (
                    [],
                    "Emergency stop maintenance procedure.",
                    0.95,
                )
            ]

    source_document = fitz.open()
    source_page = source_document.new_page()
    source_page.insert_text(
        (72, 72),
        "Emergency stop maintenance procedure.",
    )
    image_bytes = source_page.get_pixmap().tobytes("png")
    source_document.close()

    scanned_document = fitz.open()
    scanned_page = scanned_document.new_page()
    scanned_page.insert_image(scanned_page.rect, stream=image_bytes)
    pdf_bytes = scanned_document.tobytes()
    scanned_document.close()

    upload_response = client.post(
        "/documents/upload",
        files={
            "file": (
                "scanned_manual.pdf",
                pdf_bytes,
                "application/pdf",
            )
        },
    )

    assert upload_response.status_code == 201
    assert upload_response.json()["text_page_count"] == 0

    monkeypatch.setattr(
        api_main,
        "get_ocr_reader",
        lambda: ScannedOCRReader(),
    )
    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(
        api_main,
        "VectorStoreManager",
        FakeVectorStoreManager,
    )

    index_response = client.post(
        "/documents/scanned_manual.pdf/index"
    )

    assert index_response.status_code == 200
    assert index_response.json()["filename"] == "scanned_manual.pdf"
    assert index_response.json()["page_count"] == 1
    assert index_response.json()["total_chunks"] == 1
    assert index_response.json()["added_chunks"] == 1
    assert (upload_directory / "scanned_manual.pdf").exists()


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
    stored_interactions: list[dict[str, object]],
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


def test_ask_endpoint_returns_answer_when_database_logging_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(api_main, "VectorStoreManager", FakeVectorStoreManager)
    monkeypatch.setattr(api_main, "DocumentRetriever", FakeDocumentRetriever)
    monkeypatch.setattr(api_main, "OllamaLLMProvider", FakeLLMProvider)
    monkeypatch.setattr(api_main, "RAGPipeline", FakeAnsweredRAGPipeline)

    def raise_database_error(**_: object) -> int:
        raise sqlite3.OperationalError("database unavailable")

    monkeypatch.setattr(
        api_main,
        "store_interaction",
        raise_database_error,
    )
    caplog.set_level("WARNING", logger=api_main.__name__)

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
    assert response.json()["interaction_id"] is None
    assert response.json()["answer"] == (
        "The joint must be supported before removing the clamp."
    )
    assert response.json()["status"] == "ANSWERED"
    assert response.json()["abstained"] is False
    assert (
        "Database logging failed; returning answer without interaction ID."
        in caplog.text
    )
    assert "What must be done before removing the clamp?" not in caplog.text
    assert (
        "The joint must be supported before removing the clamp."
        not in caplog.text
    )
    assert "Support the joint before removing the clamp." not in caplog.text
    assert "database unavailable" not in caplog.text


def test_ask_endpoint_returns_abstained_answer(
    monkeypatch: pytest.MonkeyPatch,
    stored_interactions: list[dict[str, object]],
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

    assert len(stored_interactions) == 1

    stored_interaction = stored_interactions[0]
    assert stored_interaction["question"] == (
        "What is the robot Wi-Fi password?"
    )
    assert stored_interaction["answer"] == (
        "I don't know based on the uploaded documents."
    )
    assert stored_interaction["status"] == "ABSTAINED"
    assert stored_interaction["latency_seconds"] >= 0
    assert tuple(stored_interaction["citations"]) == ()


def test_ask_endpoint_rejects_invalid_question(
    monkeypatch: pytest.MonkeyPatch,
    stored_interactions: list[dict[str, object]],
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
    assert stored_interactions == []

def test_ask_endpoint_reports_llm_service_failure(
    monkeypatch: pytest.MonkeyPatch,
    stored_interactions: list[dict[str, object]],
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
    assert stored_interactions == []


def test_ask_endpoint_reports_grounding_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    stored_interactions: list[dict[str, object]],
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
    assert stored_interactions == []

def test_checklist_endpoint_returns_grounded_checklist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return a cited checklist that requires human review."""
    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(
        api_main,
        "VectorStoreManager",
        FakeVectorStoreManager,
    )
    monkeypatch.setattr(
        api_main,
        "DocumentRetriever",
        FakeDocumentRetriever,
    )
    monkeypatch.setattr(api_main, "OllamaLLMProvider", FakeLLMProvider)
    monkeypatch.setattr(
        api_main,
        "build_checklist_workflow",
        lambda **kwargs: FakeCompletedChecklistWorkflow(),
    )

    document_id = "a" * 64
    response = client.post(
        "/checklists/generate",
        json={
            "request": "Create a clamp maintenance checklist.",
            "document_id": document_id,
            "top_k": 5,
            "minimum_similarity": 0.6,
        },
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["request"] == "Create a clamp maintenance checklist."
    assert payload["stage"] == "awaiting_human_review"
    assert payload["evidence_sufficient"] is True
    assert payload["missing_evidence_categories"] == []
    assert payload["human_review_required"] is True
    assert payload["checklist"] is not None
    assert payload["checklist"]["procedure_steps"][0]["text"] == (
        "Support the joint before removing the clamp."
    )
    assert payload["checklist"]["procedure_steps"][0]["citations"][0] == {
        "document_id": document_id,
        "source_name": "manual.pdf",
        "page_number": 50,
        "page_label": "50",
        "label": "manual.pdf, page 50",
        "excerpt": "Support the joint before removing the clamp.",
    }

def test_checklist_endpoint_abstains_when_evidence_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Abstain safely when required evidence categories are missing."""
    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(
        api_main,
        "VectorStoreManager",
        FakeVectorStoreManager,
    )
    monkeypatch.setattr(
        api_main,
        "DocumentRetriever",
        FakeDocumentRetriever,
    )
    monkeypatch.setattr(api_main, "OllamaLLMProvider", FakeLLMProvider)
    monkeypatch.setattr(
        api_main,
        "build_checklist_workflow",
        lambda **kwargs: FakeAbstainedChecklistWorkflow(),
    )

    response = client.post(
        "/checklists/generate",
        json={
            "request": "Create a maintenance checklist.",
            "top_k": 5,
            "minimum_similarity": 0.6,
        },
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["stage"] == "abstained"
    assert payload["evidence_sufficient"] is False
    assert payload["missing_evidence_categories"] == ["tools", "parts"]
    assert payload["checklist"] is None
    assert payload["human_review_required"] is False

def test_checklist_endpoint_rejects_invalid_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return HTTP 400 when checklist request validation fails."""
    monkeypatch.setattr(api_main, "EmbeddingManager", FakeEmbeddingManager)
    monkeypatch.setattr(
        api_main,
        "VectorStoreManager",
        FakeVectorStoreManager,
    )
    monkeypatch.setattr(
        api_main,
        "DocumentRetriever",
        FakeDocumentRetriever,
    )
    monkeypatch.setattr(api_main, "OllamaLLMProvider", FakeLLMProvider)
    monkeypatch.setattr(
        api_main,
        "build_checklist_workflow",
        lambda **kwargs: FakeInvalidChecklistWorkflow(),
    )

    response = client.post(
        "/checklists/generate",
        json={
            "request": "",
            "top_k": 5,
            "minimum_similarity": 0.6,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": {
            "code": "invalid_checklist_request",
            "message": "Checklist request cannot be empty",
        }
    }


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

@pytest.mark.parametrize(
    ("method", "url"),
    [
        ("GET", "/documents/CON.pdf"),
        ("POST", "/documents/CON.pdf/index"),
        ("DELETE", "/documents/CON.pdf"),
    ],
)
def test_document_operations_reject_unsafe_filename(
    method: str,
    url: str,
) -> None:
    response = client.request(method, url)

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unsafe_file_path"

def test_submit_feedback_stores_positive_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_feedback: dict[str, object] = {}

    def fake_store_feedback(
        *,
        interaction_id: int,
        feedback: str,
    ) -> int:
        stored_feedback["interaction_id"] = interaction_id
        stored_feedback["feedback"] = feedback
        return 7

    monkeypatch.setattr(api_main, "store_feedback", fake_store_feedback)

    response = client.post(
        "/interactions/3/feedback",
        json={"feedback": "POSITIVE"},
    )

    assert response.status_code == 201
    assert stored_feedback == {
        "interaction_id": 3,
        "feedback": "POSITIVE",
    }
    assert response.json() == {
        "feedback_id": 7,
        "interaction_id": 3,
        "feedback": "POSITIVE",
    }

def test_submit_feedback_rejects_invalid_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_store_feedback(**kwargs: object) -> int:
        raise ValueError("feedback must be POSITIVE or NEGATIVE")

    monkeypatch.setattr(api_main, "store_feedback", fake_store_feedback)

    response = client.post(
        "/interactions/3/feedback",
        json={"feedback": "MAYBE"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "invalid_feedback",
        "message": "feedback must be POSITIVE or NEGATIVE",
    }

def test_submit_feedback_reports_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_store_feedback(**kwargs: object) -> int:
        raise api_main.sqlite3.IntegrityError("feedback conflict")

    monkeypatch.setattr(api_main, "store_feedback", fake_store_feedback)

    response = client.post(
        "/interactions/3/feedback",
        json={"feedback": "POSITIVE"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "feedback_conflict",
        "message": (
            "Feedback already exists or the interaction was not found."
        ),
    }

def test_interaction_history_returns_recent_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = SimpleNamespace(
        id=5,
        question="How should the joint be supported?",
        answer="Support the joint using the documented fixture.",
        status="ANSWERED",
        latency_seconds=1.25,
        created_at="2026-07-26T10:00:00+00:00",
        citations=(
            SimpleNamespace(
                document_id="manual-123",
                source_name="service_manual.pdf",
                page_number=50,
                page_label="50",
            ),
        ),
        feedback="POSITIVE",
    )

    monkeypatch.setattr(
        api_main,
        "list_recent_interactions",
        lambda *, limit: (stored,),
    )

    response = client.get("/interactions?limit=5")

    assert response.status_code == 200
    assert response.json() == {
        "interactions": [
            {
                "interaction_id": 5,
                "question": "How should the joint be supported?",
                "answer": "Support the joint using the documented fixture.",
                "status": "ANSWERED",
                "latency_seconds": 1.25,
                "created_at": "2026-07-26T10:00:00+00:00",
                "citations": [
                    {
                        "document_id": "manual-123",
                        "source_name": "service_manual.pdf",
                        "page_number": 50,
                        "page_label": "50",
                    }
                ],
                "feedback": "POSITIVE",
            }
        ],
        "total_interactions": 1,
    }

def test_interaction_history_rejects_invalid_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_list_recent_interactions(*, limit: int) -> tuple[object, ...]:
        raise ValueError("limit must be positive")

    monkeypatch.setattr(
        api_main,
        "list_recent_interactions",
        fake_list_recent_interactions,
    )

    response = client.get("/interactions?limit=0")

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "invalid_history_limit",
        "message": "limit must be positive",
    }