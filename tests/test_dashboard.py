"""Tests for the Streamlit dashboard."""

from unittest.mock import Mock, patch

import requests
from streamlit.testing.v1 import AppTest


def test_dashboard_shows_error_when_backend_is_unavailable() -> None:
    """Show a controlled message instead of crashing without FastAPI."""
    with patch(
        "requests.get",
        side_effect=requests.ConnectionError,
    ):
        app = AppTest.from_file("dashboard/app.py").run(timeout=10)

    assert len(app.exception) == 0
    assert len(app.error) == 1
    assert app.error[0].value == (
        "FastAPI backend is unavailable. "
        "Start the API server and refresh this page."
    )

def test_dashboard_displays_controlled_abstention() -> None:
    """Display an abstention warning when evidence is insufficient."""
    health_response = Mock()
    health_response.raise_for_status.return_value = None
    health_response.json.return_value = {
        "status": "ok",
        "version": "5D",
    }

    answer_response = Mock()
    answer_response.raise_for_status.return_value = None
    answer_response.json.return_value = {
        "answer": "I don't know based on the indexed documents.",
        "status": "insufficient_evidence",
        "abstained": True,
        "accepted_evidence_count": 0,
        "elapsed_seconds": 0.125,
        "citations": [],
    }

    with (
        patch("requests.get", return_value=health_response),
        patch("requests.post", return_value=answer_response),
    ):
        app = AppTest.from_file("dashboard/app.py").run(timeout=10)
        app.text_area[0].input("What is the warranty period?")
        app.button[0].click().run(timeout=10)

    assert len(app.exception) == 0
    assert len(app.warning) == 1
    assert app.warning[0].value == (
        "I don't know based on the indexed documents."
    )

def test_dashboard_displays_structured_api_error() -> None:
    """Display the structured error message returned by FastAPI."""
    health_response = Mock()
    health_response.raise_for_status.return_value = None
    health_response.json.return_value = {
        "status": "ok",
        "version": "5D",
    }

    error_response = Mock()
    error_response.json.return_value = {
        "detail": {
            "code": "llm_unavailable",
            "message": "The Ollama service is unavailable.",
        }
    }

    api_error = requests.HTTPError(response=error_response)

    with (
        patch("requests.get", return_value=health_response),
        patch("requests.post", side_effect=api_error),
    ):
        app = AppTest.from_file("dashboard/app.py").run(timeout=10)
        app.text_area[0].input("How do I replace the control box?")
        app.button[0].click().run(timeout=10)

    assert len(app.exception) == 0
    assert len(app.error) == 1
    assert app.error[0].value == "The Ollama service is unavailable."