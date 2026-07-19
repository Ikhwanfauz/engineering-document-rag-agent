"""Tests for grounded question answering."""

from __future__ import annotations

import pytest

from src.rag_pipeline import (
    GROUNDING_SYSTEM_PROMPT,
    GroundingValidationError,
    NoRetrievedEvidenceError,
    RAGPipeline,
)
from src.retriever import RetrievedChunk


def make_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-50-0",
        document_id="document-123",
        source_name="service-manual.pdf",
        page_number=50,
        page_label="50",
        chunk_index=0,
        text="Support the joint while removing the second side of the clamp.",
        distance=0.12,
        similarity_score=0.88,
    )


class FakeRetriever:
    def __init__(
        self,
        results: tuple[RetrievedChunk, ...],
    ) -> None:
        self.results = results
        self.last_query: str | None = None
        self.last_top_k: int | None = None
        self.last_document_id: str | None = None

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        self.last_query = query
        self.last_top_k = top_k
        self.last_document_id = document_id
        return self.results


class FakeLLMProvider:
    def __init__(self) -> None:
        self.system_prompt: str | None = None
        self.user_prompt: str | None = None

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return "Support the joint while removing the clamp [Evidence 1]."


def test_pipeline_returns_grounded_answer_citations_and_evidence() -> None:
    retriever = FakeRetriever((make_chunk(),))
    llm_provider = FakeLLMProvider()
    pipeline = RAGPipeline(retriever, llm_provider)

    result = pipeline.answer(
        "How should the joint be supported?",
        top_k=3,
        document_id="document-123",
    )

    assert retriever.last_query == "How should the joint be supported?"
    assert retriever.last_top_k == 3
    assert retriever.last_document_id == "document-123"

    assert result.question == "How should the joint be supported?"
    assert result.answer.endswith("[Evidence 1].")
    assert result.evidence == (make_chunk(),)

    assert len(result.citations) == 1
    assert result.citations[0].label == "service-manual.pdf, page 50"

    assert llm_provider.system_prompt == GROUNDING_SYSTEM_PROMPT
    assert "QUESTION:\nHow should the joint be supported?" in (
        llm_provider.user_prompt or ""
    )
    assert "[Evidence 1]" in (llm_provider.user_prompt or "")
    assert "Physical page: 50" in (llm_provider.user_prompt or "")
    assert "Support the joint while removing" in (llm_provider.user_prompt or "")


def test_pipeline_does_not_call_llm_without_evidence() -> None:
    llm_provider = FakeLLMProvider()
    pipeline = RAGPipeline(FakeRetriever(()), llm_provider)

    with pytest.raises(
        NoRetrievedEvidenceError,
        match="No document evidence",
    ):
        pipeline.answer("How should the joint be supported?")

    assert llm_provider.system_prompt is None
    assert llm_provider.user_prompt is None


@pytest.mark.parametrize("question", ["", "   "])
def test_empty_question_is_rejected(question: str) -> None:
    pipeline = RAGPipeline(
        FakeRetriever((make_chunk(),)),
        FakeLLMProvider(),
    )

    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline.answer(question)


class SequencedFakeLLMProvider:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.user_prompts: list[str] = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.user_prompts.append(user_prompt)
        return self.responses.pop(0)


def test_softened_mandatory_answer_is_automatically_corrected() -> None:
    mandatory_chunk = make_chunk()
    mandatory_chunk = RetrievedChunk(
        chunk_id=mandatory_chunk.chunk_id,
        document_id=mandatory_chunk.document_id,
        source_name=mandatory_chunk.source_name,
        page_number=mandatory_chunk.page_number,
        page_label=mandatory_chunk.page_label,
        chunk_index=mandatory_chunk.chunk_index,
        text=(
            "MANDATORY ACTION The joint can fall off if not supported. "
            "Support the joint while removing the clamp."
        ),
        distance=mandatory_chunk.distance,
        similarity_score=mandatory_chunk.similarity_score,
    )
    llm_provider = SequencedFakeLLMProvider(
        [
            "The joint should be supported [Evidence 1].",
            "The joint must be supported [Evidence 1].",
        ]
    )
    pipeline = RAGPipeline(
        FakeRetriever((mandatory_chunk,)),
        llm_provider,
    )

    result = pipeline.answer("How should the joint be supported?")

    assert result.answer == "The joint must be supported [Evidence 1]."
    assert len(llm_provider.user_prompts) == 2
    assert "MANDATORY LANGUAGE REQUIREMENT" in llm_provider.user_prompts[0]
    assert "CORRECTION REQUIRED" in llm_provider.user_prompts[1]


def test_persistently_softened_mandatory_answer_is_rejected() -> None:
    mandatory_chunk = make_chunk()
    mandatory_chunk = RetrievedChunk(
        chunk_id=mandatory_chunk.chunk_id,
        document_id=mandatory_chunk.document_id,
        source_name=mandatory_chunk.source_name,
        page_number=mandatory_chunk.page_number,
        page_label=mandatory_chunk.page_label,
        chunk_index=mandatory_chunk.chunk_index,
        text=(
            "MANDATORY ACTION The joint can fall off if not supported. "
            "Support the joint while removing the clamp."
        ),
        distance=mandatory_chunk.distance,
        similarity_score=mandatory_chunk.similarity_score,
    )
    llm_provider = SequencedFakeLLMProvider(
        [
            "The joint should be supported [Evidence 1].",
            "Supporting the joint is recommended [Evidence 1].",
        ]
    )
    pipeline = RAGPipeline(
        FakeRetriever((mandatory_chunk,)),
        llm_provider,
    )

    with pytest.raises(
        GroundingValidationError,
        match="softened a mandatory",
    ):
        pipeline.answer("How should the joint be supported?")
