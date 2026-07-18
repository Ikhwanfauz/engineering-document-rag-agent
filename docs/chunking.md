# Cleaning and Citation-Safe Chunking

## Purpose

Version 1B converts page-aware PDF extraction into cleaned text chunks for later embedding and retrieval.

Every chunk belongs to exactly one physical PDF page. Chunks are never allowed to cross page boundaries because each retrieved result must have one reliable source filename and page citation.

## Processing pipeline

1. Load the PDF as ordered physical pages.
2. Generate a stable SHA-256 document ID from extracted content.
3. Detect repeated page-margin lines.
4. remove copyright and layout noise;
5. preserve safety-related labels such as WARNING, DANGER, and CAUTION;
6. remove adjacent duplicated headings;
7. split each page independently;
8. attach document, source, page, and chunk metadata;
9. save the processed document as JSON.

## Default configuration

| Setting | Value |
| --- | ---: |
| Maximum chunk size | 1,000 characters |
| Chunk overlap | 150 characters |
| Page-margin inspection | 3 non-empty lines |
| Repeated-line threshold | 25% of text pages, minimum 3 pages |

The initial configuration is suitable for local embedding models, but final tuning will be based on retrieval evaluation rather than chunk statistics alone.

## Citation metadata

Every chunk preserves:

- stable document ID;
- stable chunk ID;
- source filename;
- one-based physical page number;
- embedded PDF page label;
- page-local chunk index;
- cleaned chunk text.

## UR service manual result

| Metric | Result |
| --- | ---: |
| Physical pages | 126 |
| Original extracted characters | 349,749 |
| Cleaned characters | 80,996 |
| Non-whitespace content retained | 88.92% |
| Total chunks | 155 |
| Average chunk size | 533.6 characters |
| Median chunk size | 536 characters |
| Largest chunk | 998 characters |

Page 10 was reduced to its clean section heading after duplicated heading and copyright-footer removal.

Page 50 preserved a maintenance procedure and its mandatory joint-support warning in one page-aware chunk.

## Output

Processed files are written to `data/processed/` with the document ID and chunk configuration in the filename:

```text
e-Series_Service_Manual_en_223204344883_c1000_o150.json