# Structured Checklist Generation

## Version

Version 9C adds grounded, structured maintenance-checklist generation to the controlled workflow.

## Checklist structure

Each generated checklist separates:

- Prerequisites
- Tools
- Parts
- Safety warnings
- Ordered procedure steps
- Human-review notes

Each checklist item contains concise text and one or more page-level citations.

## Grounded generation

Retrieved maintenance evidence is assigned stable evidence identifiers before being sent to the language model.

The model must return JSON containing only the required checklist sections and evidence identifiers. The parser resolves those identifiers back to the original citation objects.

The generated checklist is rejected when:

- The response is not valid JSON.
- Required or unexpected fields are present.
- A checklist item has empty text.
- A checklist item does not reference evidence.
- An unknown or invented evidence identifier is used.
- Review notes contain invalid values.

This prevents the model from attaching unsupported citations or adding ungrounded checklist content.

## Controlled workflow

When every required evidence category is available, the workflow follows:

1. Retrieve categorized evidence.
2. Validate evidence sufficiency.
3. Generate the structured checklist.
4. Resolve and validate citations.
5. Move the checklist to mandatory human review.

If evidence is incomplete, the workflow abstains without generating a checklist.

A generated checklist is not treated as approved while its stage is `AWAITING_HUMAN_REVIEW`.

## Verification

Version 9C verification completed successfully:

- Focused checklist tests: 13 passed.
- Complete project tests: 216 passed, 1 skipped.
- Complete Ruff check: passed.

The tests cover citation resolution, malformed JSON rejection, invented evidence-ID rejection, and the complete generation-to-human-review workflow.