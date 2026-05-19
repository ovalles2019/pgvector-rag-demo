"""Document ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

from src.chunker import chunk_text
from src.config import Settings
from src.db import connect, ensure_vector_index, init_schema
from src.embedder import Embedder
from src.loaders import SUPPORTED_EXTENSIONS, load_document


def ingest_path(
    settings: Settings,
    path: str | Path,
    category: str = "general",
) -> int:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    embedder = Embedder(settings)
    total_chunks = 0

    with connect(settings) as conn:
        init_schema(conn, settings.embedding_dimension)

        if path.is_file():
            total_chunks += _ingest_file(conn, embedder, settings, path, category)
        else:
            files = sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
            if not files:
                supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                raise ValueError(f"No supported files ({supported}) found under {path}")
            for file_path in files:
                file_category = _category_from_path(path, file_path, category)
                total_chunks += _ingest_file(
                    conn, embedder, settings, file_path, file_category
                )

        ensure_vector_index(conn)

    return total_chunks


def ingest_bytes(
    settings: Settings,
    filename: str,
    content: bytes,
    category: str = "general",
    source_root: str | Path = "uploads",
) -> int:
    source_root = Path(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    file_path = source_root / filename
    file_path.write_bytes(content)

    with connect(settings) as conn:
        init_schema(conn, settings.embedding_dimension)
        embedder = Embedder(settings)
        count = _ingest_file(conn, embedder, settings, file_path, category)
        ensure_vector_index(conn)
    return count


def _category_from_path(root: Path, file_path: Path, default_category: str) -> str:
    relative = file_path.relative_to(root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    return default_category


def _ingest_file(
    conn,
    embedder: Embedder,
    settings: Settings,
    file_path: Path,
    category: str,
) -> int:
    content, doc_type = load_document(file_path)
    title = file_path.stem.replace("_", " ").replace("-", " ").title()
    chunks = chunk_text(content, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return 0

    embeddings = embedder.embed(chunks)
    source = str(file_path.resolve())

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (source, title, category, doc_type)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (source)
            DO UPDATE SET
                title = EXCLUDED.title,
                category = EXCLUDED.category,
                doc_type = EXCLUDED.doc_type
            RETURNING id
            """,
            (source, title, category, doc_type),
        )
        document_id = cur.fetchone()[0]
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))

        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO chunks (document_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s)
                """,
                (document_id, index, chunk, embedding),
            )

    conn.commit()
    return len(chunks)
