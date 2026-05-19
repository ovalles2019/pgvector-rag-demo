"""Retrieval and optional RAG answer generation."""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from src.config import Settings
from src.db import connect, init_schema, list_categories
from src.embedder import Embedder


@dataclass
class RetrievedChunk:
    content: str
    source: str
    title: str
    category: str
    doc_type: str
    similarity: float


def retrieve(
    settings: Settings,
    question: str,
    top_k: int | None = None,
    category: str | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or settings.top_k
    embedder = Embedder(settings)
    query_embedding = embedder.embed_query(question)

    filters = []
    params: list[object] = [query_embedding, query_embedding]
    if category:
        filters.append("d.category = %s")
        params.append(category)
    params.append(top_k)

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""

    with connect(settings) as conn:
        init_schema(conn, settings.embedding_dimension)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    c.content,
                    d.source,
                    d.title,
                    d.category,
                    d.doc_type,
                    1 - (c.embedding <=> %s::vector) AS similarity
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                {where_clause}
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                params,
            )
            rows = cur.fetchall()

    return [
        RetrievedChunk(
            content=row[0],
            source=row[1],
            title=row[2],
            category=row[3],
            doc_type=row[4],
            similarity=float(row[5]),
        )
        for row in rows
    ]


def get_available_categories(settings: Settings) -> list[str]:
    with connect(settings) as conn:
        init_schema(conn, settings.embedding_dimension)
        return list_categories(conn)


def build_context(chunks: list[RetrievedChunk]) -> str:
    sections = []
    for index, chunk in enumerate(chunks, start=1):
        sections.append(
            f"[Source {index}: {chunk.title} ({chunk.category}) | score={chunk.similarity:.3f}]\n"
            f"{chunk.content}"
        )
    return "\n\n".join(sections)


def generate_answer(settings: Settings, question: str, chunks: list[RetrievedChunk]) -> str:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Run in retrieval-only mode or add your API key to .env."
        )

    client = OpenAI(api_key=settings.openai_api_key)
    context = build_context(chunks)

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer using only the provided context. "
                    "If the context does not contain enough information, say you do not know."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""


def ask(
    settings: Settings,
    question: str,
    generate: bool = True,
    category: str | None = None,
    top_k: int | None = None,
) -> dict:
    chunks = retrieve(settings, question, top_k=top_k, category=category)
    result = {
        "question": question,
        "chunks": chunks,
        "answer": None,
    }

    if generate and settings.openai_api_key:
        result["answer"] = generate_answer(settings, question, chunks)

    return result
