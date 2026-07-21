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

app = FastAPI(
    title="Engineering Document RAG Agent API",
    version="4B",
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


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Return the current API health status."""
    return {
        "status": "ok",
        "version": "4B",
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
