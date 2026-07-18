# Embeddings and ChromaDB Indexing

## Purpose

Version 2A converts cleaned, citation-safe document chunks into semantic vectors and stores them in a persistent local ChromaDB collection.

## Embedding model

The initial model is:

```text
sentence-transformers/all-MiniLM-L6-v2