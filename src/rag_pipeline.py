"""Grounded question answering over retrieved engineering evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from src.citation_manager import Citation, build_citations
from src.llm_provider import LLMProvider
from src.retriever import RetrievedChunk


GROUNDING_SYSTEM_PROMPT = """
You are an engineering document question-answering assistant.

Follow these rules:
1. Answer using only the supplied evidence.
2. Do not add facts from memory or outside knowledge.
3. Keep technical instructions precise and concise.
4. Support statements using evidence markers such as [Evidence 1].
5. Do not invent document names, page numbers, procedures, or safety warnings.
6. Preserve mandatory requirements and safety warnings without softening them.
7. Never rewrite "must" or "mandatory" as "recommended", "suggested", or "should".
8. When evidence is labeled "MANDATORY ACTION", state the action using "must".
9. Use only evidence that directly answers the question and ignore unrelated chunks.
10. Cite an evidence block only when it directly supports the associated statement.
11. Answer in no more than three concise sentences without repeating the conclusion.
12. Answer only the information explicitly requested by the question.
13. Do not add explanations, purposes, consequences, or inferred details unless the question asks for them.
14. Never describe an action as mandatory unless the evidence explicitly labels it MANDATORY ACTION.
15. For a simple procedural question, answer in one concise sentence using wording close to the evidence.
16. When the evidence gives an explicit sequence, preserve every relevant step in the original order; conciseness must not omit required actions.
17. For broad questions, combine all distinct directly relevant requirements across the evidence; do not collapse or omit them merely to make the answer shorter.
""".strip()

DEFAULT_MINIMUM_SIMILARITY = 0.60
INSUFFICIENT_EVIDENCE_ANSWER = "I don't know based on the uploaded documents."


class EvidenceRetriever(Protocol):
    """Retrieval interface required by the RAG pipeline."""

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> tuple[RetrievedChunk, ...]:
        """Retrieve evidence for one question."""


class NoRetrievedEvidenceError(RuntimeError):
    """Raised when no document evidence is available for generation."""


class GroundingValidationError(RuntimeError):
    """Raised when an answer fails deterministic grounding validation."""


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    """Generated answer together with its supporting evidence."""

    question: str
    answer: str
    citations: tuple[Citation, ...]
    evidence: tuple[RetrievedChunk, ...]
    abstained: bool = False


class RAGPipeline:
    """Retrieve evidence and generate a grounded answer."""

    def __init__(
        self,
        retriever: EvidenceRetriever,
        llm_provider: LLMProvider,
        *,
        minimum_similarity: float = DEFAULT_MINIMUM_SIMILARITY,
    ) -> None:
        if not 0.0 <= minimum_similarity <= 1.0:
            raise ValueError("minimum_similarity must be between 0.0 and 1.0")

        self.retriever = retriever
        self.llm_provider = llm_provider
        self.minimum_similarity = minimum_similarity

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> GroundedAnswer:
        """Answer one question using retrieved document evidence only."""
        if not question.strip():
            raise ValueError("Question cannot be empty")

        evidence = self.retriever.retrieve(
            question,
            top_k=top_k,
            document_id=document_id,
        )

        evidence = tuple(
            chunk
            for chunk in evidence
            if chunk.similarity_score >= self.minimum_similarity
        )

        if not evidence:
            return GroundedAnswer(
                question=question,
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                citations=(),
                evidence=(),
                abstained=True,
            )

        user_prompt = _build_user_prompt(question, evidence)
        answer_text = self.llm_provider.generate(
            GROUNDING_SYSTEM_PROMPT,
            user_prompt,
        )

        if _contains_mandatory_action(evidence):
            answer_text = self._validate_mandatory_language(
                user_prompt=user_prompt,
                answer_text=answer_text,
            )

        return GroundedAnswer(
            question=question,
            answer=answer_text,
            citations=build_citations(evidence),
            evidence=evidence,
        )

    def _validate_mandatory_language(
        self,
        *,
        user_prompt: str,
        answer_text: str,
    ) -> str:
        """Retry once when mandatory evidence is softened."""
        if _uses_mandatory_language(answer_text):
            return answer_text

        correction_prompt = (
            f"{user_prompt}\n\n"
            f"PREVIOUS ANSWER:\n{answer_text}\n\n"
            "CORRECTION REQUIRED:\n"
            'The evidence is labeled "MANDATORY ACTION". Rewrite the answer '
            'using the word "must" for the required action. Do not use '
            '"should", "recommended", or "suggested".'
        )

        corrected_answer = self.llm_provider.generate(
            GROUNDING_SYSTEM_PROMPT,
            correction_prompt,
        )

        if not _uses_mandatory_language(corrected_answer):
            raise GroundingValidationError(
                "Generated answer softened a mandatory document instruction"
            )

        return corrected_answer


def _build_user_prompt(
    question: str,
    evidence: tuple[RetrievedChunk, ...],
) -> str:
    """Create a clearly separated question-and-evidence prompt."""
    evidence_blocks: list[str] = []

    for number, chunk in enumerate(evidence, start=1):
        evidence_blocks.append(
            "\n".join(
                [
                    f"[Evidence {number}]",
                    f"Source: {chunk.source_name}",
                    f"Physical page: {chunk.page_number}",
                    f"PDF page label: {chunk.page_label}",
                    f"Similarity score: {chunk.similarity_score:.4f}",
                    "Text:",
                    chunk.text,
                ]
            )
        )

    joined_evidence = "\n\n".join(evidence_blocks)

    mandatory_requirement = ""
    if _contains_mandatory_action(evidence):
        mandatory_requirement = (
            "\n\nMANDATORY LANGUAGE REQUIREMENT:\n"
            'The evidence contains "MANDATORY ACTION". State the required '
            'action using the word "must". Do not use "should", '
            '"recommended", or "suggested".'
        )

    return (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED EVIDENCE:\n{joined_evidence}"
        f"{mandatory_requirement}\n\n"
        "Before answering, inspect every evidence block separately and identify "
        "each instruction that directly answers the question. Include every "
        "distinct relevant instruction, even when it appears in a different "
        "evidence block. Use one concise sentence for one action or a short "
        "numbered list for multiple actions. Preserve procedural steps in their "
        "original order. Do not add information that is not stated in the evidence."
    )


def _contains_mandatory_action(
    evidence: tuple[RetrievedChunk, ...],
) -> bool:
    """Return whether any retrieved evidence contains a mandatory action."""
    return any("mandatory action" in chunk.text.casefold() for chunk in evidence)


def _uses_mandatory_language(answer_text: str) -> bool:
    """Return whether the answer uses the word 'must'."""
    return re.search(r"\bmust\b", answer_text, flags=re.IGNORECASE) is not None
