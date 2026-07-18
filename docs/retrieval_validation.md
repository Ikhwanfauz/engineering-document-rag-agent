# Semantic Retrieval Validation

## Purpose

Version 2B validates whether technical questions retrieve the correct manual evidence before an LLM is added.

The retriever embeds the question using the same MiniLM model used during indexing, searches ChromaDB with cosine distance, and returns ranked chunks with visible evidence and physical-page citations.

## Retrieval output

Each result includes:

- rank;
- source filename;
- physical page number;
- PDF page label;
- chunk ID;
- cosine distance;
- similarity score;
- retrieved evidence text.

The similarity score is `1 - cosine distance`. It is a ranking signal, not a calibrated probability or confidence percentage.

## Initial baseline

| Question | Expected page | Rank-1 page | Similarity | Result |
| --- | ---: | ---: | ---: | --- |
| How should the joint be supported when removing the clamp? | 50 | 50 | 0.8433 | Correct |
| How should ESD-sensitive parts be handled? | 13 | 13 | 0.7878 | Correct |
| Which seals and rings should be replaced when assembling a clamp connection? | 51 | 51 | 0.6732 | Correct |
| How is the blue lid removed before disassembling a clamp connection? | 49 | 49 | 0.6959 | Correct |
| What is the capital of Malaysia? | Not answerable | 126 | 0.1476 | Correct low-score behavior |

The four answerable questions achieved a rank-1 page hit rate of 100% in this small initial baseline.

The unrelated question still returned a nearest chunk because vector search always finds the closest available vector. Its much lower similarity demonstrates why the final RAG system needs explicit insufficient-evidence handling.

## Cleaning discovered during retrieval

Real retrieval exposed vertically arranged copyright text mixed with body instructions. The cleaning pipeline was improved to:

- remove exact repeated footer sequences;
- strip footer fragments attached to real body lines;
- preserve the genuine beginning of mixed lines;
- retain safety labels such as NOTICE;
- replace stale ChromaDB chunks after processing changes.

The indexed collection now synchronizes one document by adding new chunks and deleting obsolete chunk IDs.

## Current interpretation

The observed relevant scores range from 0.6732 to 0.8433, while the initial unrelated example scored 0.1476.

No final acceptance threshold is selected from only one negative example. Threshold tuning and abstention testing will use a larger labeled question set in later checkpoints.

## Validation

Version 2B completed with 34 passing automated tests.

The tests cover retrieval ranking, citations, document filtering, empty collections, invalid queries, footer-cleaning regressions, and stale-vector replacement.