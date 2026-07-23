"""Configurable language-model providers for grounded answer generation."""

from __future__ import annotations

from typing import Any, Protocol

from langchain_ollama import ChatOllama


class LLMProvider(Protocol):
    """Common interface used by the RAG pipeline."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate one text response from system and user prompts."""

class LLMServiceError(RuntimeError):
    """Raised when the configured language-model service cannot generate a response."""

class OllamaLLMProvider:
    """Generate responses using a local Ollama chat model."""

    def __init__(
        self,
        model: str = "llama3.2",
        *,
        temperature: float = 0.0,
        base_url: str = "http://localhost:11434",
        reasoning: bool = False,
        num_predict: int = 256,
        client: Any | None = None,
    ) -> None:
        if num_predict <= 0:
            raise ValueError("num_predict must be greater than zero")

        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.reasoning = reasoning
        self.num_predict = num_predict
        self._client = client or ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            reasoning=reasoning,
            num_predict=num_predict,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a plain-text response using the configured model."""
        if not system_prompt.strip():
            raise ValueError("System prompt cannot be empty")
        if not user_prompt.strip():
            raise ValueError("User prompt cannot be empty")

        try:
            response = self._client.invoke(
                [
                    ("system", system_prompt),
                    ("human", user_prompt),
                ]
            )
        except Exception as exc:
            raise LLMServiceError(
                "The language-model service could not generate a response"
            ) from exc
        if not isinstance(response.content, str):
            raise TypeError("LLM response content must be text")

        return response.content.strip()
