"""Tests for the controlled maintenance-checklist workflow."""

import pytest

from src.checklist_agent import (
    ChecklistStage,
    build_checklist_workflow,
    transition_stage,
)


def test_transition_stage_allows_permitted_transition() -> None:
    state = {
        "request": "Create a maintenance checklist",
        "document_id": "manual-1",
        "stage": ChecklistStage.REQUEST_RECEIVED,
    }

    updated_state = transition_stage(
        state,
        ChecklistStage.RETRIEVING_EVIDENCE,
    )

    assert updated_state["stage"] is ChecklistStage.RETRIEVING_EVIDENCE
    assert state["stage"] is ChecklistStage.REQUEST_RECEIVED


def test_transition_stage_rejects_invalid_transition() -> None:
    state = {
        "request": "Create a maintenance checklist",
        "document_id": None,
        "stage": ChecklistStage.REQUEST_RECEIVED,
    }

    with pytest.raises(
        ValueError,
        match=(
            "Transition from request_received to "
            "generating_checklist is not permitted"
        ),
    ):
        transition_stage(
            state,
            ChecklistStage.GENERATING_CHECKLIST,
        )

def test_workflow_routes_sufficient_evidence_to_human_review() -> None:
    workflow = build_checklist_workflow()

    result = workflow.invoke(
        {
            "request": "Create a maintenance checklist",
            "document_id": "manual-1",
            "stage": ChecklistStage.REQUEST_RECEIVED,
            "evidence_sufficient": True,
        }
    )

    assert result["stage"] is ChecklistStage.AWAITING_HUMAN_REVIEW


def test_workflow_abstains_when_evidence_is_insufficient() -> None:
    workflow = build_checklist_workflow()

    result = workflow.invoke(
        {
            "request": "Create a maintenance checklist",
            "document_id": "manual-1",
            "stage": ChecklistStage.REQUEST_RECEIVED,
            "evidence_sufficient": False,
        }
    )

    assert result["stage"] is ChecklistStage.ABSTAINED