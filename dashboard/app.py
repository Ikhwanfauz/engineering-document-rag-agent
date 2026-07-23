"""Streamlit dashboard for the Engineering Document RAG Agent."""

import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
API_BASE_URL = os.getenv("ENGDOC_API_URL", DEFAULT_API_BASE_URL).rstrip("/")


def fetch_api_health() -> dict[str, Any]:
    """Request the current health status from the FastAPI backend."""
    response = requests.get(f"{API_BASE_URL}/health", timeout=5)
    response.raise_for_status()
    return response.json()


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

    st.info(
        "PDF upload and question-answer features will be added "
        "in the next Version 5 checkpoints."
    )


if __name__ == "__main__":
    main()
