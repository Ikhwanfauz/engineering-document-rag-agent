# Engineering Document RAG Agent

An AI assistant for engineering manuals and standard operating procedures (SOPs). Users will upload technical PDFs, ask questions, receive evidence-grounded answers with page citations, and generate cited maintenance checklists.

## Project status

**Version 3B complete:** page-aware PDF ingestion, conservative layout cleaning, citation-safe chunking, local MiniLM embeddings, persistent ChromaDB indexing, citation-aware semantic retrieval, grounded local-LLM generation, and controlled insufficient-evidence abstention are implemented.

Retrieved chunks must meet the calibrated minimum similarity of `0.60`. When no evidence passes the threshold, the system returns `I don't know based on the uploaded documents.`, skips LLM generation, and produces no citations.

The system has been validated on the 126-page Universal Robots e-Series Service Manual. It retrieves ranked evidence, sends only that evidence to a configurable Ollama model, generates a grounded answer, and returns document, physical-page, PDF-label, and excerpt citations.

The initial retrieval baseline returned the correct physical page at rank 1 for all four answerable technical questions. Real answer-generation tests covered clamp disassembly, mandatory joint support, ESD-sensitive parts, and conditional seal and ring replacement.

`llama3.2` established the initial local baseline. `qwen3:8b` followed procedural evidence more reliably and is the preferred tested model. Broad questions requiring exhaustive synthesis across multiple pages remain a documented small-model limitation.

The current automated test suite contains 56 passing tests.

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

## Technology stack

- Python 3.11
- LangChain for RAG orchestration
- LangGraph for the later maintenance-checklist workflow
- Ollama for configurable local LLM generation
- ChromaDB for persistent local vector storage
- Sentence Transformers for local embeddings
- FastAPI for the backend API
- Streamlit for the user interface
- SQLite for logs and feedback
- PyPDF and PyMuPDF for document processing
- Pytest for automated testing

The embedding baseline uses `sentence-transformers/all-MiniLM-L6-v2` with normalized vectors and cosine similarity.

The LLM provider remains configurable. `qwen3:8b` is the preferred tested model for grounded engineering answers, while `llama3.2` remains a smaller local baseline.

## Quick start

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate EngDocAI
python -m pip install -r requirements.txt
```

Copy the environment template:

```bash
copy .env.example .env
```

On macOS or Linux, use:

```bash
cp .env.example .env
```

Install Ollama separately, then download the preferred model:

```bash
ollama pull qwen3:8b
```

Confirm that Ollama can run the model:

```bash
ollama run qwen3:8b "Reply with exactly: Qwen ready."
```

Verify the project:

```bash
python -m scripts.check_setup
python -m pytest -q
```

After indexing a manual, ask a grounded question:

```bash
python -m scripts.ask_manual "How should the joint be supported when removing the clamp?" --top-k 3 --llm-model qwen3:8b
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
- [Grounded question answering](docs/grounded_qa.md)
- [Don't-know handling](docs/dont_know_handling.md)

## Current boundary

Version 3B prevents unsupported generation when retrieved evidence does not meet the calibrated minimum similarity of `0.60`. During abstention, the pipeline skips the LLM and returns no citations.

Threshold calibration accepted 6 of 7 tested answerable questions and rejected all 6 tested unanswerable questions. The conservative threshold prioritizes preventing unsupported engineering answers, although a valid low-scoring paraphrase may receive an unnecessary abstention.

Version 3B does not solve the existing small-model limitation involving exhaustive synthesis across broad multi-page evidence.

The next checkpoint is **Version 4A — FastAPI backend**, which will expose health, upload, indexing, question-answering, and document-management endpoints.