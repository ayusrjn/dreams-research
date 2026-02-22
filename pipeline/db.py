"""
Central database module for the DREAMS Research pipeline.

Provides SQLite (relational data) and ChromaDB (vector embeddings) access.
"""

import sqlite3
import sys
from pathlib import Path

import chromadb

sys.path.insert(0, str(Path(__file__).parent))
from config import DREAMS_DB_PATH, CHROMA_DB_DIR, PROCESSED_DIR


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY,
    user_id     TEXT NOT NULL,
    caption     TEXT,
    timestamp   TEXT,
    lat         REAL,
    lon         REAL,
    image_url   TEXT,
    local_image TEXT,
    snapshot_id TEXT,
    created_at  TEXT
);

CREATE TABLE IF NOT EXISTS emotion_scores (
    id       INTEGER PRIMARY KEY REFERENCES memories(id),
    user_id  TEXT NOT NULL,
    valence  REAL,
    arousal  REAL,
    anger    REAL DEFAULT 0,
    disgust  REAL DEFAULT 0,
    fear     REAL DEFAULT 0,
    joy      REAL DEFAULT 0,
    neutral  REAL DEFAULT 0,
    sadness  REAL DEFAULT 0,
    surprise REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS temporal_features (
    id           INTEGER PRIMARY KEY REFERENCES memories(id),
    user_id      TEXT NOT NULL,
    absolute_utc TEXT,
    relative_day INTEGER,
    sin_hour     REAL,
    cos_hour     REAL
);


CREATE VIEW IF NOT EXISTS master_manifest AS
SELECT
    m.*,
    e.valence, e.arousal,
    e.anger, e.disgust, e.fear, e.joy, e.neutral, e.sadness, e.surprise,
    t.absolute_utc, t.relative_day, t.sin_hour, t.cos_hour
FROM memories m
LEFT JOIN emotion_scores e ON m.id = e.id
LEFT JOIN temporal_features t ON m.id = t.id;
"""


def get_db() -> sqlite3.Connection:
    """Return a connection to the DREAMS SQLite database."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DREAMS_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> sqlite3.Connection:
    """Create all tables and views if they don't exist. Returns the connection."""
    conn = get_db()
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

_chroma_client: chromadb.ClientAPI | None = None


def get_chroma_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client (singleton)."""
    global _chroma_client
    if _chroma_client is None:
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    return _chroma_client


def get_collection(name: str) -> chromadb.Collection:
    """Get or create a ChromaDB collection with cosine distance."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )
