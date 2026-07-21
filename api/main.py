"""FastAPI backend for the Engineering Document RAG Agent."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from src.document_loader import PDFIngestionError, load_pdf

UPLOAD_DIRECTORY = Path("data/manuals")
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
UPLOAD_CHUNK_SIZE = 1024 * 1024

app = FastAPI(
    title="Engineering Document RAG Agent API",
    version="4A",
)


class UploadedDocument(BaseModel):
    """Metadata returned after a successful PDF upload."""

    document_id: str
    filename: str
    size_bytes: int
    page_count: int
    text_page_count: int


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Return the current API health status."""
    return {
        "status": "ok",
        "version": "4A",
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
