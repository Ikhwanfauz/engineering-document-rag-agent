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

def test_dashboard_submits_positive_feedback() -> None:
    """Submit positive feedback for the latest answer."""
    health_response = Mock()
    health_response.raise_for_status.return_value = None
    health_response.json.return_value = {
        "status": "ok",
        "version": "6C",
    }

    answer_response = Mock()
    answer_response.raise_for_status.return_value = None
    answer_response.json.return_value = {
        "interaction_id": 3,
        "answer": "Support the joint using the documented fixture.",
        "status": "ANSWERED",
        "abstained": False,
        "accepted_evidence_count": 1,
        "elapsed_seconds": 1.25,
        "citations": [],
    }

    feedback_response = Mock()
    feedback_response.raise_for_status.return_value = None
    feedback_response.json.return_value = {
        "feedback_id": 7,
        "interaction_id": 3,
        "feedback": "POSITIVE",
    }

    def fake_post(url: str, **kwargs: object) -> Mock:
        if url.endswith("/questions/ask"):
            return answer_response
        return feedback_response

    with (
        patch("requests.get", return_value=health_response),
        patch("requests.post", side_effect=fake_post) as mocked_post,
    ):
        app = AppTest.from_file("dashboard/app.py").run(timeout=10)
        app.text_area[0].input("How should the joint be supported?")
        app.button[0].click().run(timeout=10)
        app.button[1].click().run(timeout=10)

    assert len(app.exception) == 0
    assert len(app.success) == 2
    assert app.success[1].value == "Thank you—your feedback was saved."

    feedback_call = mocked_post.call_args_list[-1]
    assert feedback_call.args[0].endswith("/interactions/3/feedback")
    assert feedback_call.kwargs["json"] == {"feedback": "POSITIVE"}

def test_dashboard_submits_negative_feedback() -> None:
    """Submit negative feedback for the latest answer."""
    health_response = Mock()
    health_response.raise_for_status.return_value = None
    health_response.json.return_value = {
        "status": "ok",
        "version": "6C",
    }

    answer_response = Mock()
    answer_response.raise_for_status.return_value = None
    answer_response.json.return_value = {
        "interaction_id": 4,
        "answer": "Support the joint using the documented fixture.",
        "status": "ANSWERED",
        "abstained": False,
        "accepted_evidence_count": 1,
        "elapsed_seconds": 1.25,
        "citations": [],
    }

    feedback_response = Mock()
    feedback_response.raise_for_status.return_value = None
    feedback_response.json.return_value = {
        "feedback_id": 8,
        "interaction_id": 4,
        "feedback": "NEGATIVE",
    }

    def fake_post(url: str, **kwargs: object) -> Mock:
        if url.endswith("/questions/ask"):
            return answer_response
        return feedback_response

    with (
        patch("requests.get", return_value=health_response),
        patch("requests.post", side_effect=fake_post) as mocked_post,
    ):
        app = AppTest.from_file("dashboard/app.py").run(timeout=10)
        app.text_area[0].input("How should the joint be supported?")
        app.button[0].click().run(timeout=10)
        app.button[2].click().run(timeout=10)

    assert len(app.exception) == 0
    assert len(app.success) == 2
    assert app.success[1].value == "Thank you—your feedback was saved."

    feedback_call = mocked_post.call_args_list[-1]
    assert feedback_call.args[0].endswith("/interactions/4/feedback")
    assert feedback_call.kwargs["json"] == {"feedback": "NEGATIVE"}

def test_dashboard_displays_recent_interaction_history() -> None:
    """Display a recent question-and-answer interaction."""
    health_response = Mock()
    health_response.raise_for_status.return_value = None
    health_response.json.return_value = {
        "status": "ok",
        "version": "6C",
    }

    history_response = Mock()
    history_response.raise_for_status.return_value = None
    history_response.json.return_value = {
        "interactions": [
            {
                "question": "How should the joint be supported?",
                "answer": "Use the documented support fixture.",
                "status": "ANSWERED",
                "latency_seconds": 1.25,
                "feedback": "POSITIVE",
                "citations": [],
            }
        ]
    }

    def fake_get(url: str, **kwargs: object) -> Mock:
        if url.endswith("/health"):
            return health_response
        return history_response

    with patch("requests.get", side_effect=fake_get):
        app = AppTest.from_file("dashboard/app.py").run(timeout=10)

    assert len(app.exception) == 0
    assert len(app.expander) == 1
    assert app.expander[0].label == "How should the joint be supported?"
    assert app.metric[0].value == "ANSWERED"
    assert app.metric[2].value == "POSITIVE"