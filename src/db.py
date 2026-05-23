"""Database connection and schema management."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from src.config import Settings


def connect(settings: Settings) -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def init_schema(conn: psycopg.Connection, dimension: int) -> None:
    # pgvector column size must be a literal in DDL; parameters are not allowed.
    dim = int(dimension)
    if dim <= 0 or dim > 16_000:
        raise ValueError(f"invalid embedding dimension: {dimension}")

    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                source TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                doc_type TEXT NOT NULL DEFAULT 'txt',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding vector({dim}) NOT NULL,
                UNIQUE (document_id, chunk_index)
            )
            """
        )
        cur.execute(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS category TEXT NOT NULL DEFAULT 'general'"
        )
        cur.execute(
            "ALTER TABLE documents ADD COLUMN IF NOT EXISTS doc_type TEXT NOT NULL DEFAULT 'txt'"
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS documents_source_key
            ON documents (source)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS documents_category_idx
            ON documents (category)
            """
        )
    conn.commit()


def ensure_vector_index(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks")
        count = cur.fetchone()[0]
        if count == 0:
            return

        cur.execute(
            """
            SELECT 1
            FROM pg_indexes
            WHERE indexname = 'chunks_embedding_idx'
            """
        )
        if cur.fetchone():
            return

        lists = max(min(count, 100), 1)
        cur.execute(
            f"""
            CREATE INDEX chunks_embedding_idx
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = {lists})
            """
        )
    conn.commit()


def list_categories(conn: psycopg.Connection) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT category
            FROM documents
            ORDER BY category
            """
        )
        return [row[0] for row in cur.fetchall()]


def get_stats(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        documents = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks")
        chunks = cur.fetchone()[0]
    return {"documents": documents, "chunks": chunks}
