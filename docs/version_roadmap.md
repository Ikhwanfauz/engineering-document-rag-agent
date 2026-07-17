# Version Roadmap

## Version 0 - Planning and setup

- define the business problem and project boundary
- choose a clean Conda environment
- create the folder structure
- record architecture decisions
- create a validation script and starter test

## Version 1A - PDF ingestion

- load digital-text PDFs
- extract text per page
- preserve filename and page number
- report empty or unreadable pages

## Version 1B - Chunking and document metadata

- compare chunk size and overlap
- preserve page-aware citations
- generate document hashes
- save normalized processing output

## Version 2A - Embeddings and vector database

- select and load an embedding model
- create a persistent ChromaDB collection
- index document chunks
- avoid duplicate indexing

## Version 2B - Retrieval validation

- implement semantic retrieval
- display top-k chunks and relevance scores
- create initial retrieval questions
- test whether expected pages are found

## Version 3A - Grounded question answering

- select a configurable LLM provider
- generate answers from retrieved evidence only
- return document name, page number, and excerpt
- validate citation formatting

## Version 3B - Don't-know handling

- define insufficient-evidence behavior
- tune retrieval thresholds using examples
- prevent unsupported answers
- test answerable and unanswerable questions

## Version 4A - FastAPI backend

- add health, upload, index, question, and document endpoints
- validate file type and size
- return structured errors
- expose generated answer and evidence

## Version 4B - Streamlit dashboard

- upload and manage documents
- ask questions
- display citations and retrieved evidence
- show controlled error and abstention states

## Version 5 - Logging and feedback

- add SQLite schema
- record questions, answers, sources, latency, and status
- add simple user feedback
- avoid logging secrets and unnecessary document content

## Version 6 - Evaluation

- create a labeled evaluation dataset
- measure retrieval hit rate and citation correctness
- measure answer correctness, groundedness, and abstention accuracy
- record latency and compare configurations

## Version 7 - Guardrails and reliability

- detect basic prompt-injection patterns
- separate document content from system instructions
- handle corrupted PDFs and unavailable services
- add upload, path, and error-handling tests

## Version 8 - Maintenance-checklist agent

- implement a controlled LangGraph workflow
- retrieve procedures, warnings, tools, and prerequisites
- generate a structured checklist with citations
- require human review and abstain when evidence is missing

## Version 9 - Scanned PDFs and OCR

- detect image-only pages
- add OCR fallback
- compare OCR text with digital extraction
- expose OCR quality warnings

## Version 10 - Docker and finalization

- containerize API and dashboard
- run services with Docker Compose
- complete documentation and demo walkthrough
- publish final evaluation results and limitations

