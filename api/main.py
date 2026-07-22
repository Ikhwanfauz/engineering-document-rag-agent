"""FastAPI backend for the Engineering Document RAG Agent."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.document_loader import PDFIngestionError, load_pdf
from src.embedding_manager import EmbeddingConfig, EmbeddingManager
from src.llm_provider import OllamaLLMProvider
from src.rag_pipeline import (
    DEFAULT_MINIMUM_SIMILARITY,
    GroundingValidationError,
    RAGPipeline,
)
from src.retriever import DocumentRetriever
from src.text_chunker import ChunkingConfig, process_document
from src.vector_store import VectorStoreConfig, VectorStoreManager

UPLOAD_DIRECTORY = Path("data/manuals")
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024
VECTOR_STORE_DIRECTORY = Path("data/vector_store")
VECTOR_COLLECTION_NAME = "engineering_documents"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
LLM_MODEL_NAME = "qwen3:8b"
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_TEMPERATURE = 0.0
DEFAULT_TOP_K = 3

app = FastAPI(
    title="Engineering Document RAG Agent API",
    version="4C",
)


class UploadedDocument(BaseModel):
    """Metadata returned after a successful PDF upload."""

    document_id: str
    filename: str
    size_bytes: int
    page_count: int
    text_page_count: int


class IndexedDocument(BaseModel):
    """Summary returned after indexing an uploaded PDF."""

    document_id: str
    filename: str
    page_count: int
    total_chunks: int
    added_chunks: int
    existing_chunks: int
    removed_chunks: int
    collection_count: int
    elapsed_seconds: float


class QuestionRequest(BaseModel):
    """Question and optional retrieval settings."""

    question: str
    document_id: str | None = None
    top_k: int = DEFAULT_TOP_K
    minimum_similarity: float = DEFAULT_MINIMUM_SIMILARITY


class AnswerCitation(BaseModel):
    """Citation returned with a grounded answer."""

    document_id: str
    source_name: str
    page_number: int
    page_label: str
    label: str
    excerpt: str


class QuestionAnswer(BaseModel):
    """Grounded answer returned by the question-answering endpoint."""

    question: str
    answer: str
    status: str
    abstained: bool
    citations: list[AnswerCitation]
    accepted_evidence_count: int
    elapsed_seconds: float


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Return the current API health status."""
    return {
        "status": "ok",
        "version": "4C",
    }


@app.post(
    "/documents/upload",
    response_model=UploadedDocument,
    status_code=status.HTTP_201_CREATED,
    tags=["documents"],
)
async def upload_document(
    file: Annotated[UploadFile, File(description="Engineering manual in PDF format")],
) -> UploadedDocument:
    """Validate and save an uploaded PDF manual."""
    original_filename = file.filename or ""
    filename = Path(original_filename.replace("\\", "/")).name

    if not filename or Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={
                "code": "invalid_file_type",
                "message": "Only PDF files are supported.",
            },
        )

    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = UPLOAD_DIRECTORY / filename

    if destination.exists():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": "document_already_exists",
                "message": f"A document named '{filename}' already exists.",
            },
        )

    temporary_path = UPLOAD_DIRECTORY / f".{uuid4().hex}.upload.pdf"
    size_bytes = 0
    content_hash = hashlib.sha256()

    try:
        with temporary_path.open("wb") as output:
            while chunk := await file.read(UPLOAD_CHUNK_SIZE):
                size_bytes += len(chunk)

                if size_bytes > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail={
                            "code": "file_too_large",
                            "message": "PDF exceeds the 25 MB upload limit.",
                        },
                    )

                content_hash.update(chunk)
                output.write(chunk)

        if size_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "empty_file",
                    "message": "The uploaded PDF is empty.",
                },
            )

        try:
            loaded_pdf = load_pdf(temporary_path)
        except (PDFIngestionError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "code": "invalid_pdf",
                    "message": "The uploaded file is not a readable PDF.",
                },
            ) from exc

        temporary_path.replace(destination)

    except HTTPException:
        temporary_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "storage_error",
                "message": "The uploaded PDF could not be saved.",
            },
        ) from exc
    finally:
        await file.close()

    return UploadedDocument(
        document_id=content_hash.hexdigest(),
        filename=filename,
        size_bytes=size_bytes,
        page_count=loaded_pdf.page_count,
        text_page_count=loaded_pdf.text_page_count,
    )


@app.post(
    "/documents/{filename}/index",
    response_model=IndexedDocument,
    tags=["documents"],
)
def index_uploaded_document(filename: str) -> IndexedDocument:
    """Process and index an uploaded PDF in ChromaDB."""
    safe_filename = Path(filename.replace("\\", "/")).name
    pdf_path = UPLOAD_DIRECTORY / safe_filename

    if safe_filename != filename or not pdf_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "document_not_found",
                "message": f"Document '{filename}' was not found.",
            },
        )

    started_at = time.perf_counter()

    try:
        loaded_document = load_pdf(pdf_path)

        processed_document = process_document(
            loaded_document,
            ChunkingConfig(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            ),
        )

        embedding_manager = EmbeddingManager(
            EmbeddingConfig(model_name=EMBEDDING_MODEL_NAME),
        )

        vector_store = VectorStoreManager(
            embedding_manager=embedding_manager,
            config=VectorStoreConfig(
                persist_directory=VECTOR_STORE_DIRECTORY,
                collection_name=VECTOR_COLLECTION_NAME,
            ),
        )

        report = vector_store.index_document(processed_document)

    except (PDFIngestionError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "indexing_failed",
                "message": "The stored PDF could not be processed.",
            },
        ) from exc

    elapsed_seconds = time.perf_counter() - started_at

    return IndexedDocument(
        document_id=report.document_id,
        filename=safe_filename,
        page_count=processed_document.page_count,
        total_chunks=report.total_chunks,
        added_chunks=report.added_chunks,
        existing_chunks=report.existing_chunks,
        removed_chunks=report.removed_chunks,
        collection_count=report.collection_count,
        elapsed_seconds=round(elapsed_seconds, 3),
    )


@app.post(
    "/questions/ask",
    response_model=QuestionAnswer,
    tags=["questions"],
)
def ask_question(request: QuestionRequest) -> QuestionAnswer:
    """Answer a question using indexed engineering documents."""
    started_at = time.perf_counter()

    try:
        embedding_manager = EmbeddingManager(
            EmbeddingConfig(model_name=EMBEDDING_MODEL_NAME),
        )

        vector_store = VectorStoreManager(
            embedding_manager=embedding_manager,
            config=VectorStoreConfig(
                persist_directory=VECTOR_STORE_DIRECTORY,
                collection_name=VECTOR_COLLECTION_NAME,
            ),
        )

        retriever = DocumentRetriever(
            embedding_manager=embedding_manager,
            vector_store=vector_store,
        )

        llm_provider = OllamaLLMProvider(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            base_url=OLLAMA_BASE_URL,
        )

        pipeline = RAGPipeline(
            retriever=retriever,
            llm_provider=llm_provider,
            minimum_similarity=request.minimum_similarity,
        )

        result = pipeline.answer(
            request.question,
            top_k=request.top_k,
            document_id=request.document_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "invalid_question_request",
                "message": str(exc),
            },
        ) from exc
    except GroundingValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "code": "grounding_validation_failed",
                "message": "The generated answer failed grounding validation.",
            },
        ) from exc

    citations = [
        AnswerCitation(
            document_id=citation.document_id,
            source_name=citation.source_name,
            page_number=citation.page_number,
            page_label=citation.page_label,
            label=citation.label,
            excerpt=citation.excerpt,
        )
        for citation in result.citations
    ]

    elapsed_seconds = time.perf_counter() - started_at

    return QuestionAnswer(
        question=result.question,
        answer=result.answer,
        status="ABSTAINED" if result.abstained else "ANSWERED",
        abstained=result.abstained,
        citations=citations,
        accepted_evidence_count=len(result.evidence),
        elapsed_seconds=round(elapsed_seconds, 3),
    )
