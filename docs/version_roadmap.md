# Engineering Document RAG Agent — Official Version Roadmap

## Project status

- Current completed version: Version 5E
- Next checkpoint: Version 6A
- Current branch: `main`
- Latest completed commit: `c700a64`
- Version 5 completion date: 25 July 2026

This document is the official source of truth for the project. Future development should follow the checkpoints below to avoid repeating completed work or adding features outside the current scope.

---

## Version 0 — Project planning and setup [COMPLETE]

### Version 0A — Foundation

- Define the engineering-document question-answering problem.
- Define the project scope and limitations.
- Create the Conda development environment.
- Create the initial project structure.
- Record architecture decisions.
- Add the initial validation script and starter test.

Completion result:

- Project foundation established.
- Initial documentation and tests created.
- Development environment verified.

---

## Version 1 — PDF ingestion and chunking [COMPLETE]

### Version 1A — Page-aware PDF ingestion

- Load digital-text PDF documents.
- Extract text page by page.
- Preserve the filename, page number, and page label.
- Detect and report empty or unreadable pages.
- Add focused ingestion tests.

### Version 1B — Citation-safe chunking

- Clean extracted document text.
- Split text into searchable chunks.
- Compare chunk size and overlap settings.
- Preserve page-aware metadata in every chunk.
- Generate document hashes.
- Save normalized processing output.
- Add chunking and regression tests.

Completion result:

- The 126-page engineering manual can be extracted successfully.
- Every searchable chunk retains its original document and page metadata.
- Cleaning and chunking produce stable, testable output.

---

## Version 2 — Embeddings and semantic retrieval [COMPLETE]

### Version 2A — Embedding and vector indexing

- Configure the local `all-MiniLM-L6-v2` embedding model.
- Create a persistent ChromaDB collection.
- Convert document chunks into embeddings.
- Store chunks and citation metadata in the vector database.
- Prevent duplicate indexing.
- Add embedding and indexing tests.

### Version 2B — Citation-aware retrieval

- Implement semantic similarity search.
- Return the top matching evidence chunks.
- Display similarity scores.
- Preserve filename, page number, page label, and excerpt.
- Create retrieval validation questions.
- Verify that expected manual pages are retrieved.

Completion result:

- Document chunks can be indexed persistently.
- Re-indexing does not create duplicates.
- Relevant evidence can be retrieved with page-level metadata.

---

## Version 3 — Grounded question answering [COMPLETE]

### Version 3A — Grounded answer generation

- Create a configurable Ollama language-model provider.
- Build prompts using retrieved evidence only.
- Generate answers grounded in the engineering manual.
- Return document name, page number, page label, and excerpt.
- Validate citation formatting.
- Add grounded-answer tests.

### Version 3B — Controlled abstention

- Define insufficient-evidence behaviour.
- Tune the minimum retrieval similarity threshold.
- Prevent unsupported answers.
- Return a controlled “I don’t know” response when evidence is insufficient.
- Test answerable and unanswerable questions.
- Improve document cleaning and rebuild the index where necessary.

Completion result:

- The system answers supported questions using retrieved evidence.
- Unsupported questions produce a controlled abstention.
- Answers include traceable page-level citations.

---

## Version 4 — FastAPI backend [COMPLETE]

### Version 4A — PDF upload endpoint

- Add validated PDF upload.
- Validate file extension, content, and file size.
- Prevent unsafe filenames and duplicate uploads.
- Return structured upload results and errors.

### Version 4B — Document indexing endpoint

- Process uploaded PDF documents.
- Extract, clean, chunk, embed, and index documents.
- Return page count, chunk count, new chunk count, and processing time.
- Preserve duplicate-indexing protection.

### Version 4C — Question-answering endpoint

- Accept questions through FastAPI.
- Run retrieval and grounded answer generation.
- Return the answer, status, abstention state, citations, evidence, and latency.
- Validate request content and retrieval settings.

### Version 4D — Document management endpoints

- List uploaded and indexed documents.
- Return document information.
- Support controlled document deletion.
- Return structured not-found and validation errors.

### Version 4E — Service failure handling

- Detect Ollama or language-model service failures.
- Convert internal failures into structured API errors.
- Prevent raw internal exceptions from reaching users.
- Add API failure-handling tests.

Completion result:

- FastAPI exposes the complete document and question-answering workflow.
- Upload, indexing, questioning, and document management are validated.
- API and Ollama failures are returned safely and consistently.

---

## Version 5 — Streamlit user interface [COMPLETE]

### Version 5A — Dashboard foundation

- Create the Streamlit application.
- Connect the dashboard to FastAPI.
- Display backend health and version status.
- Create the main dashboard structure and layout.

### Version 5B — PDF upload and indexing interface

- Upload engineering PDFs through the dashboard.
- Display the selected filename and file size.
- Show upload and indexing loading states.
- Display indexing success, metrics, and structured errors.
- Handle documents that already exist safely.

### Version 5C — Question-answering interface

- Submit questions from the dashboard.
- Display grounded answers.
- Display answer status, accepted evidence count, and response time.
- Display page-level citations and evidence excerpts.
- Show controlled abstention responses.

### Version 5D — Error handling and usability

- Display friendly FastAPI-unavailable messages.
- Display structured Ollama service errors.
- Display missing-document and invalid-file errors.
- Display insufficient-evidence warnings.
- Provide loading states for upload, indexing, and answering.
- Use clear headings, dividers, metrics, and citation expanders.

Note:

- These behaviours were implemented across Versions 5A–5C.
- The complete Version 5D requirement list was audited before final verification.
- No separate Version 5D code commit was required.

### Version 5E — Verification and completion

- Add focused automated dashboard tests.
- Verify backend-unavailable handling.
- Verify controlled abstention display.
- Verify structured API and Ollama error display.
- Run the complete automated test suite.
- Run Ruff checks.
- Perform a manual end-to-end dashboard test.
- Confirm the repository is clean and synchronized with GitHub.

Completion result:

- 81 automated tests passed.
- Ruff checks passed.
- Manual dashboard flow passed:
  - Upload PDF
  - Index document
  - Ask a question
  - Generate a grounded answer
  - Display page-level citations
- Verified manual result:
  - 126 pages
  - 154 total chunks
  - 0 duplicate chunks added
  - Answer status: `ANSWERED`
  - Citations returned from pages 16, 13, and 10
- Version 5 completed and pushed in commit `c700a64`.

---

## Version 6 — Logging and user feedback [PLANNED]

### Version 6A — SQLite logging foundation ✅

- Design the SQLite database schema.
- Create safe database initialization.
- Store question, answer, status, latency, and timestamp.
- Store citation references without unnecessarily duplicating document content.
- Avoid logging secrets or sensitive configuration.
- Add database tests.

Completion condition:

- A local SQLite database is created automatically.
- A complete question-answering interaction can be stored and retrieved.

### Version 6B — Pipeline and API logging

- Connect logging to the FastAPI question-answering workflow.
- Log answered and abstained requests.
- Record the model name and retrieval configuration.
- Record accepted citation references and response latency.
- Add an API endpoint for retrieving interaction history.
- Add focused API logging tests.

Completion condition:

- Every successful or abstained API answer creates one valid database record.
- Logged records can be retrieved through FastAPI.

### Version 6C — Dashboard history and feedback

- Display recent question-answering history.
- Add simple positive and negative user feedback.
- Save feedback against the correct interaction.
- Prevent duplicate or invalid feedback submissions.
- Add dashboard and API feedback tests.

Completion condition:

- Users can review recent interactions and submit feedback through Streamlit.
- Feedback is stored and linked to the correct logged interaction.

### Version 6D — Logging verification and documentation

- Test database failure handling.
- Verify that unnecessary document text and secrets are not logged.
- Run the complete tests and Ruff.
- Perform a manual logging and feedback test.
- Document the schema and logging behaviour.
- Commit and push Version 6.

---

## Version 7 — Evaluation and benchmarking [PLANNED]

### Version 7A — Evaluation dataset

## Version 7 — Evaluation and benchmarking [IN PROGRESS]

### Version 7A — Evaluation dataset [COMPLETED]

- Created a labeled evaluation dataset in `evaluation/evaluation_dataset.json`.
- Added four answerable engineering questions and one unanswerable question.
- Recorded expected source documents, physical pages, and user-facing page labels.
- Recorded required answer points for grounded-answer evaluation.
- Recorded the exact expected abstention response for unanswerable questions.
- Added schema, dataset-version, and source-document metadata for reproducibility.
- Added automated dataset validation in `tests/test_evaluation_dataset.py`.
- Added Version 7A documentation in `docs/evaluation_dataset.md`.
- Verified successfully with 6 focused tests, 112 full-project tests, and Ruff.

### Version 7B — Retrieval evaluation ✅

- Added reusable retrieval metrics in `evaluation/metrics.py`.
- Measured retrieval hit rate and expected-page recall.
- Evaluated top-k values of `1`, `3`, and `5`.
- Compared similarity thresholds of none, `0.50`, `0.60`, and `0.70`.
- Identified common retrieval failures.
- Saved machine-readable results in `evaluation/results/retrieval_evaluation.json`.
- Added focused automated tests.
- Added documentation in `docs/retrieval_evaluation.md`.
- Recommended `top_k=3` with similarity threshold `0.60`.
- Achieved `100%` hit rate and `100%` expected-page recall with the recommended configuration.

### Version 7C — Answer and abstention evaluation ✅

- Added answer-quality evaluation in `evaluation/evaluate_answers.py`.
- Measured answer-point coverage, evidence grounding, citation correctness, abstention accuracy, and response latency.
- Saved the `qwen3:8b` baseline in `evaluation/results/answer_evaluation.json`.
- Achieved `100%` abstention accuracy, `75%` citation correctness, `56.95%` answer-point coverage, and `86.38%` evidence grounding.
- Verified successfully with 138 full-project tests and Ruff.
- Committed and pushed as `d2ace89`.

### Version 7D — Configuration comparison and report ✅

- Compared the selected retrieval configurations using measured Version 7B results.
- Compared `qwen3:8b` and `llama3.2` using the same evaluation dataset and retrieval settings.
- Saved the Llama comparison in `evaluation/results/answer_evaluation_llama3_2.json`.
- Selected `top_k=3` with similarity threshold `0.60`.
- Selected `qwen3:8b` as the final supported language model because it produced more focused and grounded answers.
- Documented results, limitations, reproducibility steps, and improvement opportunities in `docs/configuration_comparison.md`.
- Ran complete tests and Ruff.
- Completed and pushed Version 7.

Completion condition:

- Evaluation can be run reproducibly.
- Final metrics and configuration decisions are documented.

---

## Version 8 — Guardrails and reliability [PLANNED]

### Version 8A — Prompt-injection guardrails [COMPLETED]

- Added deterministic prompt-injection detection in `src/prompt_guardrails.py`.
- Added patterns for instruction override, security bypass, prompt extraction, role reassignment, instruction replacement, and forged role markers.
- Reject suspicious user questions before retrieval.
- Remove suspicious document chunks before language-model generation.
- Clearly separate trusted system instructions from untrusted questions and retrieved document content.
- Prevent retrieved document content from overriding system behaviour.
- Added focused guardrail and RAG pipeline integration tests.
- Verified successfully with 29 focused tests, 156 full-project tests, and Ruff.

### Version 8B — File and path reliability [COMPLETED]

- Added centralized filename and path validation in `src/file_safety.py`.
- Reject empty filenames, path traversal, directory components, drive prefixes, control characters, trailing dots or spaces, and reserved Windows device names.
- Ensure resolved document paths remain inside the permitted upload directory.
- Protected document upload, lookup, indexing, and deletion operations.
- Handle empty, oversized, non-PDF, corrupted, and unreadable uploads safely.
- Added focused filename, path-security, symlink-escape, and API integration tests.
- Verified successfully with 67 focused tests, 198 full-project tests, one expected Windows symlink skip, and Ruff.

### Version 8C — Service and recovery reliability

- Verify unavailable Ollama, embedding model, vector database, and API behaviour.
- Return consistent structured errors.
- Prevent partial operations from leaving invalid state.
- Add recovery and error-handling tests.

### Version 8D — Reliability verification

- Run guardrail and failure scenarios manually.
- Run the complete tests and Ruff.
- Document supported protections and known limitations.
- Commit and push Version 8.

Completion condition:

- Common injection, file, path, and service-failure cases are handled safely.
- Reliability behaviour is covered by automated tests.

Verification result:

- Focused reliability tests: 107 passed, 1 skipped.
- Complete automated suite: 203 passed, 1 skipped.
- Ruff: all checks passed.
- Representative manual prompt-injection, invalid-PDF, unsafe-path, and Ollama failure/recovery checks passed.

Supported protections:

- Common prompt-injection attempts are detected and handled safely.
- Uploaded files, filenames, sizes, and resolved paths are validated.
- LLM, embedding, and vector-database failures return controlled errors.
- Existing indexed data remains available when embedding or writing fails.
- Partial new writes are removed before an indexing failure is reported.
- Stale chunks are deleted only after all replacement chunks are written.

Known limitations:

- Prompt-injection detection is rule-based and cannot guarantee detection of every possible attack.
- Answer generation still depends on the configured Ollama service being available.
- Index recovery covers tested application-level failures, but not sudden operating-system or hardware interruption.
- Uploaded documents are validated for supported application use; this is not a general malware-scanning system.

---

## Version 9 — Maintenance-checklist agent [PLANNED]

### Version 9A — Controlled workflow foundation [COMPLETED]

- Added LangGraph for the maintenance-checklist workflow.
- Defined explicit workflow stages and permitted transitions.
- Added controlled branching based on evidence sufficiency.
- Route sufficient evidence toward mandatory human review.
- Route insufficient evidence toward safe abstention.
- Kept the existing RAG pipeline unchanged as the future evidence source.
- Added focused workflow-transition and routing tests.
- Verified successfully with 4 focused tests, 207 full-project tests, one expected skip, and Ruff.

Completion result:

- The controlled LangGraph workflow is operational.
- Invalid workflow-stage jumps are rejected.
- Sufficient evidence reaches mandatory human review.
- Insufficient evidence ends safely as `ABSTAINED`.

### Version 9B — Maintenance evidence retrieval ✅

- Retrieve procedures, warnings, tools, parts, and prerequisites.
- Preserve page-level citations for every evidence category.
- Filter evidence below the minimum similarity threshold.
- Remove prompt-injected evidence and its citations.
- Detect categories that are empty or entirely absent.
- Store categorized evidence and validation results in workflow state.
- Route missing required information safely to `ABSTAINED`.

Completion result:

- Five-category maintenance evidence retrieval is operational.
- Page-level citations are preserved for accepted evidence.
- Unsafe and insufficient evidence is excluded.
- LangGraph performs real retrieval and validation when a retriever is provided.
- Sufficient evidence reaches mandatory human review.
- Full verification passed with `212 passed, 1 skipped`; Ruff passed.

### Version 9C — Structured checklist generation [COMPLETED]

- Generate ordered maintenance steps.
- Attach citations to steps and warnings.
- Separate prerequisites, tools, safety warnings, procedure, and review notes.
- Require human review before the checklist is treated as approved.

Completion result:

- Added strict JSON-based structured checklist generation.
- Added stable evidence identifiers and page-level citation resolution.
- Added rejection of malformed output and unknown evidence identifiers.
- Integrated real LLM generation into the controlled LangGraph workflow.
- Preserved mandatory human review before approval.
- Added focused tests for grounded generation and workflow integration.
- Verified 13 focused tests, 216 complete-project tests with 1 skipped, and full Ruff checks.
- Documented the implementation in `docs/structured_checklist_generation.md`.

### Version 9D — Agent testing and verification ✅

Completed:

- Added `POST /checklists/generate` API access to the controlled checklist workflow.
- Added API tests for successful checklist generation, safe abstention, and invalid requests.
- Verified structured checklist responses with page-level citations.
- Verified mandatory human review for every generated checklist.
- Verified safe abstention when required evidence categories are missing.
- Manually demonstrated checklist generation for replacing clamp seals and rings.
- Increased checklist generation capacity to prevent truncated JSON responses.
- Verified 13 focused checklist tests.
- Verified 219 complete-project tests with 1 skipped.
- Verified all Ruff checks passed.
- Documented the API workflow and verification in `docs/structured_checklist_generation.md`.

Completion result:

- The system generates a structured, cited maintenance checklist only when sufficient evidence exists.
- Unsupported or incomplete requests safely abstain.
- Human review remains mandatory.
- Version 9 is complete.

---

## Version 10 — Scanned PDF and OCR support [IN PROGRESS]

### Version 10A — Scanned-page detection ✅

- Detect pages with little or no extractable digital text.
- Distinguish digital-text pages from image-only pages.
- Record extraction-method metadata.

Completion result:

- Added page-level scanned-content detection.
- Distinguished digital-text, image-only scanned, and blank pages.
- Detected image-based pages containing fewer than 20 digital-text characters.
- Recorded `is_scanned` and `extraction_method` metadata for every page.
- Added focused tests for digital-text, blank, image-only, and low-text scanned pages.
- Verified 6 focused loader tests and 221 complete-project tests with 1 skipped.
- Verified all Ruff and Git whitespace checks passed.

### Version 10B — OCR fallback ✅

- Add OCR for image-only pages.
- Normalize OCR output for chunking.
- Preserve document and page metadata.
- Prevent duplicate digital-text and OCR content.

Completion result:

- Added EasyOCR fallback for scanned and image-only pages during indexing.
- Rendered scanned pages into image arrays and normalized OCR output into clean text lines for chunking.
- Preserved document identity, physical page numbers, page labels, and scanned-page metadata.
- Kept valid digital text as the preferred extraction method to prevent duplicate digital and OCR content.
- Added a cached GPU-enabled OCR reader that is reused across indexing requests.
- Added focused loader and API tests using lightweight fake OCR readers.
- Recorded NumPy and EasyOCR as direct project dependencies.
- Verified 44 focused loader and API tests.
- Verified 223 complete-project tests passed with 1 skipped.
- Verified all Ruff checks passed.

### Version 10C — OCR quality and warnings

- Compare OCR output with digital extraction where possible.
- Define OCR quality indicators.
- Expose low-quality OCR warnings.
- Prevent uncertain OCR text from being presented as high-confidence evidence.

### Version 10D — OCR integration and verification

- Integrate OCR documents into indexing and retrieval.
- Display OCR status in the API and dashboard.
- Add scanned-document fixtures and tests.
- Run complete tests and Ruff.
- Perform a manual scanned-PDF test.
- Document OCR limitations.
- Commit and push Version 10.

Completion condition:

- The system can index and search supported scanned PDFs.
- Users are warned when OCR quality may affect answers.

---

## Version 11 — Docker, documentation, and final release [PLANNED]

### Version 11A — Containerization

- Containerize FastAPI.
- Containerize Streamlit.
- Define persistent storage for documents, ChromaDB, and SQLite.
- Configure service communication using Docker Compose.
- Document Ollama connection requirements.

### Version 11B — Deployment verification

- Build all containers from a clean environment.
- Start the complete system with Docker Compose.
- Verify health, upload, indexing, questioning, citations, logging, and feedback.
- Verify persistent data after container restart.

### Version 11C — Final documentation and demonstration

- Update the README.
- Finalize architecture and setup documentation.
- Add a complete demo walkthrough.
- Document evaluation results.
- Document security, OCR, model, and deployment limitations.
- Add screenshots or demo media where useful.

### Version 11D — Final release

- Run the complete automated test suite.
- Run Ruff and all final quality checks.
- Perform the final end-to-end acceptance test.
- Confirm the repository is clean.
- Create the final release commit and tag.
- Push the completed project to GitHub.

Completion condition:

- A new user can set up and run the project using the documentation.
- The complete system works through Docker Compose.
- Final metrics, limitations, and demonstration steps are published.

---

## Development workflow for every checkpoint

For each checkpoint:

1. Confirm the checkpoint scope from this roadmap.
2. Inspect the existing code before changing anything.
3. Implement only the current checkpoint.
4. Run focused tests.
5. Run the complete test suite.
6. Run Ruff.
7. Perform a manual test when applicable.
8. Update the relevant documentation.
9. Review the Git diff.
10. Commit and push only after verification.

Do not begin the next checkpoint until the current checkpoint is verified and completed.