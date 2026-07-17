# Project Scope

## Problem

Engineers and technicians often search long manuals and SOPs manually. A normal general-purpose chatbot may answer fluently without using the correct document evidence. This project will provide a document-grounded assistant that retrieves relevant technical passages before answering and exposes the evidence used.

## Primary users

- maintenance technicians
- manufacturing and process engineers
- technical support staff
- engineering students working with approved manuals

## Core user workflow

1. Upload an engineering manual or SOP in PDF format.
2. Process and index the document while preserving page metadata.
3. Ask a technical question.
4. Retrieve relevant passages.
5. Generate an answer grounded only in the retrieved evidence.
6. Display the answer, document name, page citation, and source excerpt.
7. Abstain when the available evidence is insufficient.
8. Optionally generate a cited maintenance checklist for human review.

## Required capabilities

- digital-text PDF ingestion
- page-aware chunking and metadata
- local embeddings and vector search
- grounded question answering
- source-document and page citations
- visible supporting evidence
- explicit insufficient-evidence response
- simple, repeatable evaluation set
- FastAPI backend and Streamlit interface
- local logging and feedback
- basic file validation and prompt-injection defense
- Docker deployment

## Planned advanced capabilities

- OCR fallback for scanned documents
- duplicate-document detection using file hashes
- embedding and response caching
- multi-document filtering
- cited maintenance-checklist agent
- latency and failure monitoring

## Out of scope for the first release

- fine-tuning an LLM
- autonomous execution of maintenance actions
- replacing a qualified engineer or approved safety procedure
- interpreting complex engineering drawings with guaranteed accuracy
- unrestricted web search during technical answering

## Success criteria

The final demonstration must show:

1. one successful PDF upload and index operation;
2. one correctly answered technical question with a valid page citation;
3. one unanswerable question that produces a controlled abstention;
4. one evaluation run over a labeled question set;
5. one cited maintenance checklist generated for human approval;
6. one-command local deployment using Docker Compose.

