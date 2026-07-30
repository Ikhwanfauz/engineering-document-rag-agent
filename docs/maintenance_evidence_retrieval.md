# Maintenance Evidence Retrieval

## Version

Version 9B adds categorized evidence retrieval to the controlled maintenance-checklist workflow.

## Evidence Categories

The workflow retrieves evidence separately for:

- Procedures
- Safety warnings
- Tools
- Parts
- Prerequisites

Each category uses a targeted retrieval query based on the maintenance request.

## Evidence Controls

Retrieved chunks must:

- Meet the configured minimum similarity score.
- Pass the prompt-injection guardrail.
- Preserve document, page, and excerpt citations.
- Belong to the selected document when a document ID is provided.

Unsafe or insufficient chunks are excluded from both the evidence and its citations.

## Evidence Validation

All five evidence categories are required.

A category is treated as missing when:

- It contains no accepted chunks.
- It is entirely absent from the categorized evidence.

If any required category is missing, the workflow routes safely to `ABSTAINED`.

## Workflow Integration

When an evidence retriever is provided, the LangGraph workflow:

1. Enters `RETRIEVING_EVIDENCE`.
2. Retrieves and stores categorized evidence.
3. Enters `VALIDATING_EVIDENCE`.
4. Records evidence sufficiency and missing categories.
5. Routes sufficient evidence toward checklist generation.
6. Routes insufficient evidence to `ABSTAINED`.

A checklist with sufficient evidence still ends at `AWAITING_HUMAN_REVIEW`. It is not automatically approved.

## Verification

Version 9B tests cover:

- Five-category evidence retrieval.
- Page-level citation preservation.
- Missing and absent evidence categories.
- Similarity filtering.
- Prompt-injection filtering.
- End-to-end LangGraph retrieval and validation.
- Compatibility with the Version 9A controlled workflow.

Verification result:

- `212 passed, 1 skipped`
- Ruff: all checks passed