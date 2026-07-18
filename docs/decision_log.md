# Decision Log

## D001 - Use a separate Conda environment

**Decision:** Create `EngDocAI` with Python 3.11 instead of reusing the `SFAI` environment.

**Reason:** The first environment contains computer-vision and OCR dependencies that are unrelated to the core RAG system. Separation makes installation, debugging, locking dependencies, and Docker deployment cleaner.

## D002 - Use one primary orchestration stack

**Decision:** Begin with LangChain for RAG components and add LangGraph only for the maintenance-checklist workflow.

**Reason:** Using LangChain and LlamaIndex together would duplicate abstractions without proving additional engineering skill. One clear stack will be easier to understand and evaluate.

## D003 - Use ChromaDB initially

**Decision:** Use persistent local ChromaDB for the first release.

**Reason:** It supports local development, metadata filtering, and reproducible demonstrations without requiring a hosted database. The vector-store interface will remain replaceable.

## D004 - Preserve page metadata from the beginning

**Decision:** Every extracted chunk must retain its source document and page number.

**Reason:** Page citations cannot be added reliably at the end if page boundaries were lost during ingestion.

## D005 - Treat abstention as a core feature

**Decision:** "I don't know based on the uploaded documents" behavior is required, not optional polish.

**Reason:** Engineering answers must distinguish supported evidence from plausible model output.

## D006 - Add a controlled agent after core RAG evaluation

**Decision:** Build the maintenance-checklist agent only after retrieval, citations, and abstention have been evaluated.

**Reason:** The agent depends on trustworthy retrieval. It will create a cited proposal for human review and will not perform autonomous maintenance actions.

## D007 - Do not use the roadmap scan as the demonstration manual

**Decision:** Use the uploaded AI-learning roadmap only as planning input.

**Reason:** It is an image-based learning roadmap, not an engineering manual or SOP. A proper public technical document will provide better questions, citations, safety warnings, and evaluation examples.

## D010 - Validate retrieval before adding an LLM

**Decision:** Expose retrieved evidence, page citations, cosine distance, and similarity scores before implementing answer generation.

**Reason:** Retrieval errors must be visible and measurable. Adding an LLM before validating evidence would hide retrieval weaknesses behind fluent generated answers.

