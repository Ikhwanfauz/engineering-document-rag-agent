# Evaluation Dataset

## Version

Version 7A introduces a labeled and version-controlled evaluation dataset for the Engineering Document RAG Agent.

## Purpose

The dataset provides reproducible test cases for future retrieval, abstention, citation, and answer-quality evaluation.

## Dataset Location

The dataset is stored at `evaluation/evaluation_dataset.json`.

## Source Document

- Document: Universal Robots e-Series Service Manual
- Filename: `e-Series_Service_Manual_en.pdf`
- Document ID: `e-series-service-manual`

## Dataset Coverage

The initial Version 7A dataset contains five evaluation examples:

- Four answerable engineering questions
- One unanswerable out-of-domain question

The answerable questions cover:

- Supporting a joint during clamp removal
- Handling ESD-sensitive parts
- Replacing seals and rings in a clamp connection
- Removing the blue lid before clamp disassembly

The unanswerable question verifies that the system abstains when the uploaded documents do not contain sufficient evidence.

## Dataset Fields

Each example contains:

- `id`: Unique and stable identifier
- `question`: Evaluation question
- `answerable`: Whether the source document contains sufficient evidence
- `expected_documents`: Expected source filenames
- `expected_pages`: Expected physical PDF page numbers
- `expected_page_labels`: Expected user-facing page labels
- `required_answer_points`: Information that a grounded answer should contain
- `expected_response`: Exact required response for unanswerable examples

The dataset also records its schema version, dataset version, description, and source-document metadata.

## Abstention Behaviour

Unanswerable examples must not contain expected evidence or required answer points.

The required abstention response is `I don't know based on the uploaded documents.`

## Automated Validation

Dataset validation is implemented in `tests/test_evaluation_dataset.py`.

The tests verify:

- Required dataset metadata
- Unique example IDs
- Answerable and unanswerable coverage
- Required fields and data types
- Matching physical-page and page-label counts
- Expected evidence for answerable examples
- Empty evidence and the exact abstention response for unanswerable examples

## Verification Results

Version 7A verification completed successfully:

- 6 focused evaluation-dataset tests passed
- 112 full-project tests passed
- Ruff reported `All checks passed!`

The full test suite must be run with `python -m pytest` so the project packages are resolved through the active Python environment.

## Reproducibility

The evaluation dataset is stored as JSON and committed with the repository. Its stable IDs, explicit evidence labels, and schema version allow future evaluation runs to use the same labeled cases consistently.

Version 7B will use this dataset to measure retrieval hit rate, expected-page recall at different `top_k` values, and similarity-threshold behaviour.