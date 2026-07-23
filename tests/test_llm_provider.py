"""Tests for configurable language-model providers."""

from types import SimpleNamespace
from typing import Any

import pytest

from src.llm_provider import LLMServiceError, OllamaLLMProvider


class FakeChatModel:
    def __init__(self, response_content: Any = " Grounded answer. ") -> None:
        self.response_content = response_content
        self.messages: list[tuple[str, str]] | None = None

    def invoke(
        self,
        messages: list[tuple[str, str]],
    ) -> SimpleNamespace:
        self.messages = messages
        return SimpleNamespace(content=self.response_content)

class FailingChatModel:
    """Simulate an unavailable Ollama service."""

    def invoke(
        self,
        messages: list[tuple[str, str]],
    ) -> SimpleNamespace:
        raise ConnectionError("Ollama is unavailable")


def test_ollama_provider_sends_system_and_user_prompts() -> None:
    client = FakeChatModel()
    provider = OllamaLLMProvider(client=client)

    answer = provider.generate(
        "Answer only from the evidence.",
        "How should the joint be supported?",
    )

    assert answer == "Grounded answer."
    assert client.messages == [
        ("system", "Answer only from the evidence."),
        ("human", "How should the joint be supported?"),
    ]

def test_ollama_service_failure_is_wrapped() -> None:
    provider = OllamaLLMProvider(client=FailingChatModel())

    with pytest.raises(
        LLMServiceError,
        match="language-model service could not generate a response",
    ) as error:
        provider.generate("System prompt", "User question")

    assert isinstance(error.value.__cause__, ConnectionError)


@pytest.mark.parametrize(
    ("system_prompt", "user_prompt", "message"),
    [
        ("", "Valid question", "System prompt cannot be empty"),
        ("Valid system prompt", "   ", "User prompt cannot be empty"),
    ],
)
def test_empty_prompts_are_rejected(
    system_prompt: str,
    user_prompt: str,
    message: str,
) -> None:
    provider = OllamaLLMProvider(client=FakeChatModel())

    with pytest.raises(ValueError, match=message):
        provider.generate(system_prompt, user_prompt)


def test_non_text_llm_response_is_rejected() -> None:
    provider = OllamaLLMProvider(client=FakeChatModel(response_content=[]))

    with pytest.raises(TypeError, match="response content must be text"):
        provider.generate("System prompt", "User question")


def test_default_ollama_client_disables_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_arguments: dict[str, Any] = {}

    class FakeChatOllama:
        def __init__(self, **arguments: Any) -> None:
            captured_arguments.update(arguments)

    monkeypatch.setattr(
        "src.llm_provider.ChatOllama",
        FakeChatOllama,
    )

    provider = OllamaLLMProvider()

    assert provider.reasoning is False
    assert provider.num_predict == 256
    assert captured_arguments["reasoning"] is False
    assert captured_arguments["num_predict"] == 256
