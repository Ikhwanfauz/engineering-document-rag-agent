# ENGINEERING DOCUMENT RAG AGENT - PERSONAL TRACKING

## VERSION 0 - PROJECT PLANNING, ENVIRONMENT SETUP AND FOLDER STRUCTURE 🟡

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

- Version 0 planning, files, folder structure, and Work Mode validation are complete.
- Installation and validation inside the Windows `EngDocAI` environment are still pending.
- Version 0 will receive the final ✅ only after the local commands run successfully.
- No RAG or agent code has been implemented yet.
- After local setup, the next checkpoint is **Version 1A - PDF ingestion and page-level metadata extraction**.
- Before Version 1A evaluation, we still need to select a suitable public engineering manual or SOP.
