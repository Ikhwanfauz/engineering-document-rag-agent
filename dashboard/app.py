"""Streamlit dashboard for the Engineering Document RAG Agent."""

import os
from typing import Any
from urllib.parse import quote

import requests
import streamlit as st


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL = os.getenv("ENGDOC_API_URL", DEFAULT_API_BASE_URL).rstrip("/")


def fetch_api_health() -> dict[str, Any]:
    """Request the current health status from the FastAPI backend."""
    response = requests.get(f"{API_BASE_URL}/health", timeout=5)
    response.raise_for_status()
    return response.json()

def upload_pdf(uploaded_file: Any) -> dict[str, Any]:
    """Upload and validate a PDF through the FastAPI backend."""
    response = requests.post(
        f"{API_BASE_URL}/documents/upload",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf",
            ),
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def index_document(filename: str) -> dict[str, Any]:
    """Ask FastAPI to process and index an uploaded PDF."""
    encoded_filename = quote(filename, safe="")
    response = requests.post(
        f"{API_BASE_URL}/documents/{encoded_filename}/index",
        timeout=300,
    )
    response.raise_for_status()
    return response.json()

def get_api_error(
    error: requests.HTTPError,
    default_message: str,
) -> tuple[str | None, str]:
    """Extract the error code and message returned by FastAPI."""
    if error.response is None:
        return None, default_message

    try:
        payload = error.response.json()
    except requests.exceptions.JSONDecodeError:
        return None, default_message

    detail = payload.get("detail", {})

    if not isinstance(detail, dict):
        return None, default_message

    return detail.get("code"), detail.get("message", default_message)

def render_document_upload() -> None:
    """Render the PDF upload and indexing interface."""
    st.subheader("Upload and index a document")

    uploaded_file = st.file_uploader(
        "Choose an engineering manual",
        type=["pdf"],
        help="PDF files up to 25 MB are supported.",
    )

    if uploaded_file is None:
        st.caption("Select a PDF to make it searchable by the RAG system.")
        return

    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    st.write(f"Selected: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")

    if not st.button("Upload and index", type="primary"):
        return

    filename = uploaded_file.name

    try:
        with st.spinner("Uploading and validating the PDF..."):
            upload_result = upload_pdf(uploaded_file)

    except requests.HTTPError as exc:
        error_code, message = get_api_error(
            exc,
            "The backend rejected the document.",
        )

        if error_code != "document_already_exists":
            st.error(message)
            return

        st.info(
            f"{filename} already exists in backend storage. "
            "Indexing the existing copy instead."
        )

    except requests.RequestException:
        st.error(
            "Could not communicate with FastAPI. "
            "Check that the backend is running."
        )
        return

    else:
        filename = upload_result["filename"]
        st.success(
            f"Uploaded {filename} successfully — "
            f"{upload_result['page_count']} pages detected."
        )

    try:
        with st.spinner("Extracting, chunking, embedding, and indexing..."):
            index_result = index_document(filename)

    except requests.HTTPError as exc:
        _, message = get_api_error(
            exc,
            "The document could not be indexed.",
        )
        st.error(message)

    except requests.RequestException:
        st.error(
            "Could not communicate with FastAPI. "
            "Check that the backend is running."
        )

    else:
        st.success("Document indexed successfully.")

        column_1, column_2, column_3 = st.columns(3)
        column_1.metric("Pages", index_result["page_count"])
        column_2.metric("Total chunks", index_result["total_chunks"])
        column_3.metric("New chunks", index_result["added_chunks"])

        st.caption(
            f"Indexing completed in {index_result['elapsed_seconds']:.3f} seconds."
        )


def main() -> None:
    """Render the Streamlit dashboard."""
    st.set_page_config(
        page_title="Engineering Document RAG Agent",
        page_icon="📘",
        layout="wide",
    )

    st.title("Engineering Document RAG Agent")
    st.caption(
        "Upload engineering documents and receive grounded answers "
        "with page-level citations."
    )

    st.subheader("Backend status")

    try:
        health = fetch_api_health()
    except requests.RequestException:
        st.error(
            "FastAPI backend is unavailable. "
            "Start the API server and refresh this page."
        )
    else:
        status = health.get("status")
        version = health.get("version", "unknown")

        if status == "ok":
            st.success(f"FastAPI backend connected — Version {version}")
        else:
            st.warning("FastAPI responded, but its status is not healthy.")

    st.divider()
    render_document_upload()


if __name__ == "__main__":
    main()
