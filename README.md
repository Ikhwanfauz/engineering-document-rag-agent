# Engineering Document RAG Agent

An AI assistant for engineering manuals and standard operating procedures (SOPs). Users will upload technical PDFs, ask questions, receive evidence-grounded answers with page citations, and generate cited maintenance checklists.

## Project status

**Version 0 complete:** project planning, clean environment creation, dependency installation, folder structure, architecture decisions, roadmap, and local Windows validation all passed.

The retrieval and generation pipeline has not been implemented yet. Version 1A will begin PDF ingestion and page-level metadata extraction.

## Portfolio objective

This project is designed to demonstrate more than a basic "chat with PDF" application. The completed system will include:

- PDF manual and SOP upload
- page-aware document ingestion
- semantic retrieval with a vector database
- grounded technical answers with document and page citations
- visible retrieved evidence
- explicit "I don't know" handling when evidence is insufficient
- a repeatable evaluation dataset and metrics
- basic prompt-injection and file-upload guardrails
- query, source, latency, and user-feedback logging
- an AI agent that creates cited maintenance checklists
- OCR fallback for scanned PDFs
- FastAPI, Streamlit, SQLite, and Docker deployment

## Planned stack

- Python 3.11
- LangChain for RAG orchestration
- LangGraph for the later agent workflow
- ChromaDB for local vector storage
- Sentence Transformers for local embeddings
- FastAPI for the backend API
- Streamlit for the user interface
- SQLite for logs and feedback
- PyPDF and PyMuPDF for document processing
- Pytest for testing

The LLM and embedding model will be selected and benchmarked in a later checkpoint. Keeping the provider configurable prevents the architecture from being tied to one model.

## Quick start

Create a clean environment:

```bash
conda env create -f environment.yml
conda activate EngDocAI
python -m pip install -r requirements.txt
```

Copy the environment template:

```bash
copy .env.example .env
```

On macOS or Linux, use `cp .env.example .env` instead.

Verify Version 0:

```bash
python scripts/check_setup.py
python -m pytest
```

## Project structure

```text
engineering-document-rag-agent/
|-- api/                 # FastAPI application
|-- dashboard/           # Streamlit interface
|-- src/                 # ingestion, retrieval, RAG, guardrails, and agent logic
|-- evaluation/          # evaluation dataset, runner, and metrics
|-- database/            # SQLite schema and access layer
|-- data/
|   |-- manuals/         # local source PDFs (ignored by Git)
|   |-- processed/       # extracted and normalized content
|   `-- vector_store/    # persistent ChromaDB data
|-- results/             # evaluation outputs and runtime logs
|-- tests/               # automated tests
|-- docs/                # scope, architecture, roadmap, decisions, and tracking
|-- scripts/             # setup and utility scripts
|-- environment.yml
|-- requirements.txt
`-- README.md
```

## Documentation

- [Project scope](docs/project_scope.md)
- [Architecture](docs/architecture.md)
- [Version roadmap](docs/version_roadmap.md)
- [Decision log](docs/decision_log.md)
- [Personal tracking](docs/personal_tracking.md)

## Current boundary

Version 0 intentionally contains no fake RAG implementation. Each pipeline component will be added, tested, and documented checkpoint by checkpoint.
