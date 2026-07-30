"""Controlled maintenance-checklist workflow over existing RAG evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from src.citation_manager import Citation, build_citations
from src.prompt_guardrails import contains_prompt_injection
from src.rag_pipeline import DEFAULT_MINIMUM_SIMILARITY, EvidenceRetriever
from src.retriever import RetrievedChunk

class EvidenceCategory(StrEnum):
    """Evidence categories required for a maintenance checklist."""

    PROCEDURES = "procedures"
    WARNINGS = "warnings"
    TOOLS = "tools"
    PARTS = "parts"
    PREREQUISITES = "prerequisites"

@dataclass(frozen=True, slots=True)
class CategorizedEvidence:
    """Retrieved chunks and citations for one evidence category."""

    category: EvidenceCategory
    chunks: tuple[RetrievedChunk, ...]
    citations: tuple[Citation, ...]

EVIDENCE_CATEGORY_QUERIES: dict[EvidenceCategory, str] = {
    EvidenceCategory.PROCEDURES: (
        "maintenance procedure steps and required sequence"
    ),
    EvidenceCategory.WARNINGS: (
        "safety warnings hazards cautions and mandatory actions"
    ),
    EvidenceCategory.TOOLS: (
        "tools and equipment required for maintenance"
    ),
    EvidenceCategory.PARTS: (
        "parts components seals rings and replacement items required"
    ),
    EvidenceCategory.PREREQUISITES: (
        "prerequisites preparation conditions and actions required before maintenance"
    ),
}

class ChecklistStage(StrEnum):
    """Permitted stages of the maintenance-checklist workflow."""

    REQUEST_RECEIVED = "request_received"
    RETRIEVING_EVIDENCE = "retrieving_evidence"
    VALIDATING_EVIDENCE = "validating_evidence"
    GENERATING_CHECKLIST = "generating_checklist"
    AWAITING_HUMAN_REVIEW = "awaiting_human_review"
    ABSTAINED = "abstained"

PERMITTED_TRANSITIONS: dict[ChecklistStage, frozenset[ChecklistStage]] = {
    ChecklistStage.REQUEST_RECEIVED: frozenset(
        {ChecklistStage.RETRIEVING_EVIDENCE}
    ),
    ChecklistStage.RETRIEVING_EVIDENCE: frozenset(
        {ChecklistStage.VALIDATING_EVIDENCE}
    ),
    ChecklistStage.VALIDATING_EVIDENCE: frozenset(
        {
            ChecklistStage.GENERATING_CHECKLIST,
            ChecklistStage.ABSTAINED,
        }
    ),
    ChecklistStage.GENERATING_CHECKLIST: frozenset(
        {ChecklistStage.AWAITING_HUMAN_REVIEW}
    ),
    ChecklistStage.AWAITING_HUMAN_REVIEW: frozenset(),
    ChecklistStage.ABSTAINED: frozenset(),
}


class ChecklistWorkflowState(TypedDict):
    """Shared state passed between checklist-workflow stages."""

    request: str
    document_id: str | None
    stage: ChecklistStage
    evidence_sufficient: NotRequired[bool]
    categorized_evidence: NotRequired[tuple[CategorizedEvidence, ...]]
    missing_evidence_categories: NotRequired[tuple[EvidenceCategory, ...]]

def retrieve_categorized_evidence(
    retriever: EvidenceRetriever,
    request: str,
    *,
    document_id: str | None = None,
    top_k: int = 5,
    minimum_similarity: float = DEFAULT_MINIMUM_SIMILARITY,
) -> tuple[CategorizedEvidence, ...]:
    """Retrieve safe, relevant evidence separately for every category."""
    if not request.strip():
        raise ValueError("Checklist request cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not 0.0 <= minimum_similarity <= 1.0:
        raise ValueError("minimum_similarity must be between 0.0 and 1.0")

    categorized_evidence: list[CategorizedEvidence] = []

    for category, category_query in EVIDENCE_CATEGORY_QUERIES.items():
        query = f"{request}\nEvidence needed: {category_query}"
        retrieved_chunks = retriever.retrieve(
            query,
            top_k=top_k,
            document_id=document_id,
        )
        accepted_chunks = tuple(
            chunk
            for chunk in retrieved_chunks
            if chunk.similarity_score >= minimum_similarity
            and not contains_prompt_injection(chunk.text)
        )

        categorized_evidence.append(
            CategorizedEvidence(
                category=category,
                chunks=accepted_chunks,
                citations=build_citations(accepted_chunks),
            )
        )

    return tuple(categorized_evidence)

def validate_categorized_evidence(
    categorized_evidence: tuple[CategorizedEvidence, ...],
) -> tuple[bool, tuple[EvidenceCategory, ...]]:
    """Determine whether every required evidence category has evidence."""
    missing_categories = tuple(
        category
        for category in EvidenceCategory
        if not any(
            evidence.category is category and evidence.chunks
            for evidence in categorized_evidence
        )
    )

    return not missing_categories, missing_categories

def transition_stage(
    state: ChecklistWorkflowState,
    next_stage: ChecklistStage,
) -> ChecklistWorkflowState:
    """Return updated state only when the requested transition is permitted."""
    current_stage = state["stage"]

    if next_stage not in PERMITTED_TRANSITIONS[current_stage]:
        raise ValueError(
            f"Transition from {current_stage.value} to "
            f"{next_stage.value} is not permitted"
        )

    return {
        **state,
        "stage": next_stage,
    }

def start_evidence_retrieval(
    state: ChecklistWorkflowState,
) -> ChecklistWorkflowState:
    """Move a received checklist request into evidence retrieval."""
    return transition_stage(
        state,
        ChecklistStage.RETRIEVING_EVIDENCE,
    )

def retrieve_workflow_evidence(
    state: ChecklistWorkflowState,
    retriever: EvidenceRetriever,
    *,
    top_k: int = 5,
    minimum_similarity: float = DEFAULT_MINIMUM_SIMILARITY,
) -> ChecklistWorkflowState:
    """Retrieve categorized evidence and store it in workflow state."""
    retrieving_state = start_evidence_retrieval(state)
    categorized_evidence = retrieve_categorized_evidence(
        retriever,
        state["request"],
        document_id=state["document_id"],
        top_k=top_k,
        minimum_similarity=minimum_similarity,
    )

    return {
        **retrieving_state,
        "categorized_evidence": categorized_evidence,
    }

def start_evidence_validation(
    state: ChecklistWorkflowState,
) -> ChecklistWorkflowState:
    """Move retrieved checklist evidence into validation."""
    return transition_stage(
        state,
        ChecklistStage.VALIDATING_EVIDENCE,
    )

def validate_workflow_evidence(
    state: ChecklistWorkflowState,
) -> ChecklistWorkflowState:
    """Validate categorized evidence and store the decision in workflow state."""
    validating_state = start_evidence_validation(state)
    evidence_sufficient, missing_categories = validate_categorized_evidence(
        state.get("categorized_evidence", ())
    )

    return {
        **validating_state,
        "evidence_sufficient": evidence_sufficient,
        "missing_evidence_categories": missing_categories,
    }

def route_after_evidence_validation(
    state: ChecklistWorkflowState,
) -> Literal["generate_checklist", "abstain"]:
    """Choose the permitted route after evidence validation."""
    if state.get("evidence_sufficient", False):
        return "generate_checklist"

    return "abstain"

def start_checklist_generation(
    state: ChecklistWorkflowState,
) -> ChecklistWorkflowState:
    """Move sufficient validated evidence into checklist generation."""
    return transition_stage(
        state,
        ChecklistStage.GENERATING_CHECKLIST,
    )

def start_human_review(
    state: ChecklistWorkflowState,
) -> ChecklistWorkflowState:
    """Move the generated checklist into mandatory human review."""
    return transition_stage(
        state,
        ChecklistStage.AWAITING_HUMAN_REVIEW,
    )

def abstain_from_checklist_generation(
    state: ChecklistWorkflowState,
) -> ChecklistWorkflowState:
    """End the workflow safely when validated evidence is insufficient."""
    return transition_stage(
        state,
        ChecklistStage.ABSTAINED,
    )

def build_checklist_workflow(
    retriever: EvidenceRetriever | None = None,
    *,
    top_k: int = 5,
    minimum_similarity: float = DEFAULT_MINIMUM_SIMILARITY,
):
    """Build and compile the controlled checklist workflow."""
    workflow = StateGraph(ChecklistWorkflowState)

    retrieve_evidence_node = (
        start_evidence_retrieval
        if retriever is None
        else partial(
            retrieve_workflow_evidence,
            retriever=retriever,
            top_k=top_k,
            minimum_similarity=minimum_similarity,
        )
    )
    validate_evidence_node = (
        start_evidence_validation
        if retriever is None
        else validate_workflow_evidence
    )

    workflow.add_node("retrieve_evidence", retrieve_evidence_node)
    workflow.add_node("validate_evidence", validate_evidence_node)
    workflow.add_node("generate_checklist", start_checklist_generation)
    workflow.add_node("human_review", start_human_review)
    workflow.add_node("abstain", abstain_from_checklist_generation)

    workflow.add_edge(START, "retrieve_evidence")
    workflow.add_edge("retrieve_evidence", "validate_evidence")
    workflow.add_conditional_edges(
        "validate_evidence",
        route_after_evidence_validation,
        {
            "generate_checklist": "generate_checklist",
            "abstain": "abstain",
        },
    )
    workflow.add_edge("generate_checklist", "human_review")
    workflow.add_edge("human_review", END)
    workflow.add_edge("abstain", END)

    return workflow.compile()