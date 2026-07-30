"""Controlled maintenance-checklist workflow over existing RAG evidence."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, NotRequired, TypedDict
from langgraph.graph import END, START, StateGraph


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

def start_evidence_validation(
    state: ChecklistWorkflowState,
) -> ChecklistWorkflowState:
    """Move retrieved checklist evidence into validation."""
    return transition_stage(
        state,
        ChecklistStage.VALIDATING_EVIDENCE,
    )

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

def build_checklist_workflow():
    """Build and compile the controlled checklist workflow."""
    workflow = StateGraph(ChecklistWorkflowState)

    workflow.add_node("retrieve_evidence", start_evidence_retrieval)
    workflow.add_node("validate_evidence", start_evidence_validation)
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