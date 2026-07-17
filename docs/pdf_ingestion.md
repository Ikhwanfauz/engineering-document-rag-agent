# PDF Ingestion

## Purpose

Version 1A creates a reliable boundary between source PDFs and the later RAG pipeline. It extracts ordered page text while preserving the source filename, one-based physical page number, and embedded PDF page label needed for citations.

This checkpoint intentionally does not perform chunking, embeddings, OCR, retrieval, or answer generation.

## Development document

The local development document is the official Universal Robots e-Series Service Manual:

- local filename: `e-Series_Service_Manual_en.pdf`
- local size: 23,243,459 bytes
- physical pages: 126
- PDF format: 1.4
- embedded author: `UR`
- embedded title: not provided
- encrypted: no

The manual covers UR3e, UR5e, UR10e, and UR16e robots and e-Series control-box variants. It stays inside `data/manuals/` and is ignored by Git because it is a copyrighted source document.

## Implementation

`src/document_loader.py` provides:

- `PDFPage` for text and citation metadata from one physical page;
- `LoadedPDF` for ordered pages, document metadata, and extraction totals;
- `load_pdf()` for validation, opening, extraction, and clear failures;
- one-based physical page numbers;
- embedded page labels with physical-number fallback;
- empty or image-only page detection;
- missing-file, wrong-extension, corrupt-file, and password-protection errors.

Text normalization is deliberately conservative. It normalizes newline characters and removes trailing whitespace without rewriting the document's wording.

## Inspection command

Run the page-level inspection from the project root:

```bash
python -m scripts.inspect_pdf "data/manuals/e-Series_Service_Manual_en.pdf" --sample-page 10
```

The command reports document metadata, text coverage, character statistics, pages that may require OCR, and an optional page sample.

## UR manual extraction result

| Metric | Result |
| --- | ---: |
| Physical pages | 126 |
| Pages with extractable text | 126 |
| Empty or image-only pages | 0 |
| Total extracted characters | 349,749 |
| Average characters per page | 2,775.8 |
| Median characters per page | 3,300.0 |

The result confirms that the primary development manual can use direct text extraction. OCR remains necessary as a later fallback for other scanned documents.

## Validation

```bash
python -m pytest -q
python -m ruff check src scripts tests
python -m ruff format --check src/document_loader.py scripts/inspect_pdf.py tests/test_document_loader.py
git diff --check
```

Validated result:

- `5 passed in 0.34s`
- Ruff lint passed
- the three Version 1A Python files passed the scoped format check
- no whitespace errors were found

## Known limitations

The extracted page-10 sample contains repeated section headings, page furniture, and vertically arranged copyright/footer words. This is a layout artifact rather than an ingestion failure.

Version 1B will evaluate cleaning and chunking rules while protecting page citations. The raw page-aware output remains the evidence source so cleaning decisions can be measured instead of hidden.
