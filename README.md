# Engineering Document RAG Agent

An AI assistant for engineering manuals and standard operating procedures (SOPs). Users will upload technical PDFs, ask questions, receive evidence-grounded answers with page citations, and generate cited maintenance checklists.

## Project status

**Version 11C is in progress:** the core Engineering Document RAG Agent is complete, including page-aware PDF ingestion, OCR fallback, persistent vector indexing, grounded question answering, controlled abstention, safety guardrails, interaction logging, user feedback, retrieval evaluation, cited maintenance-checklist generation, FastAPI, Streamlit, SQLite, and Docker deployment.

The system has been validated using the 126-page Universal Robots e-Series Service Manual. Retrieved evidence retains document, physical-page, PDF-label, similarity-score, and excerpt metadata.

Technical answers and maintenance checklists are generated only from retrieved evidence. When the available evidence is insufficient, the system abstains instead of producing unsupported engineering guidance. Generated maintenance checklists remain subject to human review.

`qwen3:8b` is the preferred tested Ollama model. The embedding baseline remains `sentence-transformers/all-MiniLM-L6-v2` with normalized vectors and cosine similarity.

The current automated test suite contains **227 passing tests and 1 skipped test**. Ruff static checks also pass.

## Portfolio objective

This project demonstrates more than a basic "chat with PDF" application. The completed system includes:

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
- A controlled state-machine workflow for maintenance-checklist generation
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

Start the FastAPI backend:

```bash
uvicorn api.main:app --reload
```

In a second terminal, start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

Open the dashboard at `http://localhost:8501`. Interactive API documentation is available at `http://localhost:8000/docs`.
Alternatively, run both application services with Docker Compose:

```bash
docker compose up --build
```

Keep Ollama running on the host so the containerized API can connect through `host.docker.internal:11434`.



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
- [Evaluation dataset](docs/evaluation_dataset.md)
- [Retrieval evaluation](docs/retrieval_evaluation.md)
- [Maintenance evidence retrieval](docs/maintenance_evidence_retrieval.md)
- [Structured checklist generation](docs/structured_checklist_generation.md)
- [Interaction logging](docs/interaction_logging.md)
- [Configuration comparison](docs/configuration_comparison.md)

## OCR limitations

- OCR accuracy depends on scan resolution, image clarity, page orientation, fonts, and document layout.
- Complex tables, diagrams, handwritten text, and damaged pages may not be extracted reliably.
- OCR processing is slower than extracting native digital PDF text.
- Low-confidence OCR evidence is rejected from answer and checklist generation.
- OCR quality scores and warnings help identify uncertain text, but important engineering instructions must still be verified against the original PDF.


## Current limitations

- Retrieval quality depends on the uploaded document, chunking, embedding model, similarity threshold, and selected `top_k`.
- OCR may misread complex tables, diagrams, handwriting, damaged scans, or low-resolution pages.
- Prompt and file guardrails reduce common risks but do not replace secure production infrastructure or human review.
- Local language-model quality depends on the selected Ollama model and available hardware. Broad multi-page synthesis may remain incomplete.
- Docker deployment expects Ollama to remain available on the host through `host.docker.internal:11434`.
- Generated answers and maintenance checklists must be verified against the cited source pages before engineering use.