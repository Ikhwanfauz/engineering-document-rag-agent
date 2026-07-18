# Engineering Document RAG Agent

An AI assistant for engineering manuals and standard operating procedures (SOPs). Users will upload technical PDFs, ask questions, receive evidence-grounded answers with page citations, and generate cited maintenance checklists.

## Project status

**Version 2B complete:** page-aware PDF ingestion, conservative layout cleaning, citation-safe chunking, local MiniLM embeddings, persistent ChromaDB indexing, stale-chunk synchronization, and citation-aware semantic retrieval are implemented and validated on the 126-page Universal Robots e-Series Service Manual.

The initial retrieval baseline returned the correct physical page at rank 1 for all four answerable technical questions. An unrelated question produced a much lower similarity score, providing an initial baseline for later insufficient-evidence handling.

The current automated test suite contains 34 passing tests.

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

The initial embedding baseline uses `sentence-transformers/all-MiniLM-L6-v2` with normalized vectors and cosine similarity. The LLM provider remains configurable and will be selected during grounded answer-generation development.

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

Verify the current project:

```bash
python -m scripts.check_setup
python -m pytest -q
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
- [PDF ingestion](docs/pdf_ingestion.md)
- [Personal tracking](docs/personal_tracking.md)
- [Cleaning and chunking](docs/chunking.md)
- [Embeddings and ChromaDB indexing](docs/embedding_indexing.md)
- [Semantic retrieval validation](docs/retrieval_validation.md)

## Current boundary

Version 2B intentionally stops before LLM answer generation. The current system retrieves ranked evidence with source filenames, physical-page citations, and similarity scores, but it does not yet generate answers or decide when evidence is insufficient.
