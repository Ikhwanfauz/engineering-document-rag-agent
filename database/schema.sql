PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS qa_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_seconds REAL NOT NULL CHECK (latency_seconds >= 0),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    )
);

CREATE TABLE IF NOT EXISTS interaction_citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL,
    citation_order INTEGER NOT NULL CHECK (citation_order >= 1),
    document_id TEXT NOT NULL,
    source_name TEXT NOT NULL,
    page_number INTEGER NOT NULL CHECK (page_number >= 1),
    page_label TEXT NOT NULL,
    FOREIGN KEY (interaction_id)
        REFERENCES qa_interactions (id)
        ON DELETE CASCADE,
    UNIQUE (interaction_id, citation_order)
);

CREATE INDEX IF NOT EXISTS idx_interaction_citations_interaction_id
ON interaction_citations (interaction_id);

CREATE TABLE IF NOT EXISTS interaction_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id INTEGER NOT NULL UNIQUE,
    feedback TEXT NOT NULL CHECK (
        feedback IN ('POSITIVE', 'NEGATIVE')
    ),
    created_at TEXT NOT NULL DEFAULT (
        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    ),
    FOREIGN KEY (interaction_id)
        REFERENCES qa_interactions (id)
        ON DELETE CASCADE
);

