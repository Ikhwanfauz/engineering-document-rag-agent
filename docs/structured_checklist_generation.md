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

## API access

Version 9D exposes the controlled checklist workflow through:

```text
POST /checklists/generate
```

The request accepts:

- `request`: The maintenance task to convert into a checklist.
- `document_id`: An optional document filter.
- `top_k`: The retrieval result limit.
- `minimum_similarity`: The minimum accepted evidence similarity.

When all required evidence categories are available, the endpoint returns:

- A structured checklist with page-level citations.
- The `awaiting_human_review` workflow stage.
- `evidence_sufficient` set to `true`.
- `human_review_required` set to `true`.

When required evidence is incomplete, the endpoint returns:

- The `abstained` workflow stage.
- The missing evidence categories.
- No generated checklist.
- `human_review_required` set to `false`.

Invalid checklist requests return HTTP `400` with the error code
`invalid_checklist_request`. Language-model service failures return HTTP `503`.

## API verification

The checklist API tests verify:

- Grounded checklist serialization with page-level citations.
- Mandatory human-review status for generated checklists.
- Safe abstention when required evidence is missing.
- HTTP `400` handling for invalid checklist requests.

The complete API test file passed with 36 tests. The complete project verification
passed with 219 tests and 1 skipped test. Ruff checks also passed.
