# ENGINEERING DOCUMENT RAG AGENT - PERSONAL TRACKING

## VERSION 0 - PROJECT PLANNING, ENVIRONMENT SETUP AND FOLDER STRUCTURE ✅

### What we did

- Confirmed Project 2 as a RAG and AI-agent system for engineering documents.
- Defined the main workflow: upload PDF, ask a question, retrieve evidence, answer with page citations, and refuse unsupported questions.
- Confirmed that the maintenance-checklist AI agent remains part of the project.
- Analysed the uploaded seven-page GenAI roadmap.
- Selected useful additions from the roadmap: evaluation, guardrails, prompt-injection protection, logging, caching, OCR, failure handling, and system design.
- Created the initial project folders and placeholder modules.
- Added the initial environment, dependency, configuration, documentation, validation, and test files.

### Why we did it

- We needed a clear boundary before installing packages or writing the RAG pipeline.
- Page citations and "don't know" handling affect the design from the first ingestion step.
- A clean structure makes the project easier to understand, debug, deploy, and present in a portfolio.
- The project files now preserve our architecture and decisions even if we continue in a new chat.

### Environment decision

- New Conda environment: `EngDocAI`
- Python version: `3.11`
- We decided not to reuse `SFAI` because it contains unrelated YOLO, vision, and OCR dependencies.
- Conda and pip can reuse downloaded package caches, so a new environment does not always mean downloading every package again.

### Important files and folders

- `README.md` - project introduction, features, stack, setup, and current status
- `environment.yml` - creates the clean `EngDocAI` environment
- `requirements.txt` - initial direct dependencies
- `.env.example` - safe configuration template without secrets
- `.gitignore` - excludes manuals, generated vector data, logs, databases, and secrets
- `src/` - future ingestion, retrieval, RAG, citation, guardrail, and agent logic
- `api/` - future FastAPI backend
- `dashboard/` - future Streamlit interface
- `evaluation/` - evaluation dataset, runner, and metrics
- `database/` - future SQLite logging layer
- `data/manuals/` - local engineering manuals and SOPs
- `docs/project_scope.md` - project requirements and success criteria
- `docs/architecture.md` - planned pipeline and agent workflow
- `docs/version_roadmap.md` - checkpoint-by-checkpoint plan
- `docs/decision_log.md` - important technical decisions and reasons
- `scripts/check_setup.py` - verifies the Version 0 skeleton
- `tests/test_project_structure.py` - automated structure test

### Important commands

```bash
conda env create -f environment.yml
conda activate EngDocAI
python -m pip install -r requirements.txt
python scripts/check_setup.py
python -m pytest
```

### Problem faced

- Reusing `SFAI` looked easier because many Python packages were already installed.
- However, it would mix computer-vision packages with the new RAG stack and make dependency management and Docker deployment unnecessarily complicated.
- During Work Mode validation, `python -m pytest` could not run because this temporary runtime does not contain `pytest` yet.

### Solution

- Use the separate `EngDocAI` environment.
- Keep only Project 2 dependencies in this environment.
- Create a tested dependency lock file after the first successful full installation.
- Keep `pytest` declared in `requirements.txt` and run the same structure assertion directly with Python for the Version 0 verification. The normal Pytest command will work after the `EngDocAI` dependencies are installed.

### Important lesson

- A new project environment is not wasted work. It documents the real dependencies and prevents an old project from silently making the new project appear reproducible.
- Citations must be planned during PDF ingestion. If page metadata is discarded early, reliable page citations cannot simply be added later.
- An AI agent should be built on top of tested retrieval instead of being added before the evidence pipeline works.

### Current status


- The project was extracted successfully to `C:\Users\ikhwa\engineering-document-rag-agent-v0\engineering-document-rag-agent`.
- The `EngDocAI` Conda environment was created and activated successfully on Windows.
- All packages from `requirements.txt` were installed successfully.
- `.env.example` was copied to the local `.env` configuration file.
- Local Pytest result: `1 passed in 0.02s`.
- The local Git repository was initialized and connected to `Ikhwanfauz/engineering-document-rag-agent`.
- The first project commit was pushed successfully to the `main` branch.
- Version 0 is officially complete ✅.
- No RAG or agent code has been implemented yet.
- The next checkpoint is **Version 1A - PDF ingestion and page-level metadata extraction**.
- Before Version 1A evaluation, we still need to select a suitable public engineering manual or SOP.

## VERSION 1A - PDF INGESTION AND PAGE-LEVEL METADATA ✅

### What we did

- Selected the official Universal Robots e-Series Service Manual as the first engineering document.
- Kept the copyrighted manual local inside `data/manuals/` and confirmed that Git ignores it.
- Inspected the manual before implementation instead of assuming that text extraction would work.
- Replaced the placeholder in `src/document_loader.py` with a page-aware PDF loader.
- Added `PDFPage` and `LoadedPDF` data models for predictable extraction output.
- Preserved the source filename, one-based physical page number, and embedded PDF page label for later citations.
- Added document metadata, text-page counts, empty-page detection, and total-character statistics.
- Added clear errors for missing files, wrong extensions, corrupt PDFs, empty PDFs, and password-protected PDFs.
- Added `scripts/inspect_pdf.py` for repeatable command-line inspection.
- Added `tests/test_document_loader.py` with valid, empty-page, wrong-extension, missing-file, and corrupt-file cases.
- Tested the loader against the complete 126-page UR service manual.

### Why we did it

- RAG answers cannot provide trustworthy page citations if source and page metadata are lost during ingestion.
- We needed to know whether the development manual was digital text or an image-only scan before designing OCR behavior.
- A separate inspection command gives us measurable evidence before chunking, embeddings, or LLM generation are introduced.

### Important files

- `src/document_loader.py` - validates PDFs and returns ordered page-aware text
- `scripts/inspect_pdf.py` - reports extraction coverage, metadata, statistics, and page samples
- `tests/test_document_loader.py` - automated loader and failure tests
- `docs/pdf_ingestion.md` - Version 1A design, results, validation, and limitations
- `data/manuals/e-Series_Service_Manual_en.pdf` - local ignored development document

### Important commands and results

```bash
python -m pytest -q
# 5 passed in 0.34s

python -m ruff check src scripts tests
# All checks passed!

python -m scripts.inspect_pdf "data/manuals/e-Series_Service_Manual_en.pdf" --sample-page 10
```

- File size: `23,243,459` bytes
- PDF format: `PDF 1.4`
- Pages: `126`
- Pages with text: `126`
- Empty or image-only pages: `0`
- Total extracted characters: `349,749`
- Average characters per page: `2,775.8`
- Median characters per page: `3,300.0`
- Embedded author: `UR`
- Embedded title: not provided
- Encrypted: no

### Problem faced

- The first Git patch failed its safety check even though `src/document_loader.py` looked unchanged.
- Byte inspection showed that the real placeholder ended with two newline bytes (`\n\n`), while the first patch was built against one newline.
- The full-project Ruff format check reported 11 older placeholder files that would be reformatted.
- Page 10 contained duplicated headings and vertically split copyright/footer text after extraction.

### Solution

- Did not force the failed patch or ignore the mismatch.
- Inspected the exact file bytes, rebuilt the patch against the correct 83-byte placeholder, and validated it before applying.
- Scoped the format check to the three Version 1A Python files so unrelated placeholders were not changed.
- Recorded the layout noise as a Version 1B cleaning and chunking problem instead of hiding it inside the ingestion layer.

### Important lesson

- `git apply --check` is valuable because it prevents a patch from changing files when the expected base does not match exactly.
- A PDF can be fully text-extractable while still containing headers, footers, rotated text, and reading-order noise.
- Ingestion should preserve trustworthy page evidence; cleaning and chunking should be separate, testable steps.
- Copyrighted development manuals can be used locally without publishing their bytes in the repository.

### Current status

- Version 1A page-aware PDF ingestion is implemented and locally validated.
- All 126 pages of the UR service manual produced extractable text.
- No OCR was needed for the primary development manual.
- Source filename, physical page number, PDF page label, metadata, and empty-page status are preserved.
- Version 1A has not added chunking, embeddings, retrieval, an LLM, or an agent yet.
- The next checkpoint is **Version 1B - text cleaning, chunking, and document metadata**.


## VERSION 1B - CLEANING AND CITATION-SAFE CHUNKING ✅

### What we did

- Changed back to the simple solo Git workflow: edit, test, commit, and push directly to `main`.
- Merged the completed Version 1A feature branch into `main` using a local fast-forward merge.
- Replaced the placeholder in `src/text_chunker.py` with the document-cleaning and chunking pipeline.
- Created stable SHA-256 document IDs from extracted document content.
- Added configurable chunk size, chunk overlap, margin inspection, and repeated-line thresholds.
- Used a default chunk size of `1,000` characters with `150` characters of overlap.
- Kept every chunk inside one physical PDF page.
- Preserved the document ID, chunk ID, filename, physical page number, PDF page label, page-local chunk index, and chunk text.
- Added repeated header/footer detection.
- Protected safety words such as `WARNING`, `DANGER`, `CAUTION`, `SAFETY`, `IMPORTANT`, and `NOTE`.
- Removed duplicated headings caused by PDF reading order.
- Added conservative copyright/footer cleaning.
- Added `scripts/process_pdf.py` to process the complete manual and save normalized JSON.
- Saved generated processing output inside `data/processed/`, which remains ignored by Git.
- Added automated cleaning, hashing, chunking, metadata, persistence, and error tests.
- Added `docs/chunking.md`.

### Why we did it

- Embeddings work better with meaningful technical text than with page numbers, repeated headers, copyright words, and layout spacing.
- Every retrieved chunk needs one reliable page citation.
- Crossing a page boundary would make the source page ambiguous.
- Stable document and chunk IDs are needed later for duplicate detection and persistent vector indexing.
- Raw extraction and cleaned output needed to remain separate so cleaning behavior could be measured and debugged.

### Important files

- `src/text_chunker.py` - document hashing, cleaning, chunking, and JSON persistence
- `scripts/process_pdf.py` - real-PDF processing and inspection command
- `tests/test_text_chunker.py` - cleaning and chunking regression tests
- `docs/chunking.md` - Version 1B design, metrics, and limitations
- `data/processed/` - ignored generated JSON output

### Important commands and results

```bash
python -m ruff format src/text_chunker.py tests/test_text_chunker.py scripts/process_pdf.py
python -m ruff check src scripts tests
python -m pytest -q
python -m scripts.process_pdf "data/manuals/e-Series_Service_Manual_en.pdf" --sample-page 10


```

## VERSION 2A - EMBEDDINGS AND VECTOR STORAGE ✅

- Added MiniLM embeddings and persistent ChromaDB storage.
- Preserved source and page metadata in indexed chunks.
- Added stable indexing, duplicate detection, and stale-chunk replacement.
- Indexed the complete engineering service manual locally.

## VERSION 2B - SEMANTIC RETRIEVAL VALIDATION ✅

- Added top-k semantic retrieval with similarity scores.
- Returned source filenames, physical pages, PDF labels, excerpts, and chunk IDs.
- Validated answerable engineering questions against expected pages.
- Kept retrieval visible before introducing LLM generation.

## VERSION 3A - GROUNDED QUESTION ANSWERING ✅

- Added configurable local Ollama generation.
- Added the grounded RAG pipeline, citation manager, and `ask_manual.py` CLI.
- Added mandatory-action validation and one automatic correction attempt.
- Cleaned wide-layout PDF footer contamination and reindexed the manual.
- Compared `llama3.2` with `qwen3:8b`; Qwen was the stronger tested model.
- Validated real questions covering clamp removal, mandatory joint support, ESD handling, and seal replacement.
- Reached 51 passing automated tests.
- Documented that small local models may still omit or add details in broad multi-page answers.
- Next checkpoint: Version 3B insufficient-evidence thresholds and controlled abstention.