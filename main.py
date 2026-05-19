#!/usr/bin/env python3
"""CLI for the pgvector RAG demo."""

from __future__ import annotations

import click

from src.config import get_settings
from src.db import connect, init_schema


@click.group()
def cli() -> None:
    """pgvector RAG demo: ingest documents and ask questions."""


@cli.command("init-db")
def init_db() -> None:
    """Create pgvector tables and indexes."""
    settings = get_settings()
    with connect(settings) as conn:
        init_schema(conn, settings.embedding_dimension)
    click.echo("Database schema initialized.")


@cli.command("ingest")
@click.argument("path", type=click.Path(exists=True, path_type=str))
@click.option("--category", default="general", show_default=True, help="Metadata category tag.")
def ingest_cmd(path: str, category: str) -> None:
    """Ingest a file or directory of .txt, .md, or .pdf documents."""
    from src.ingest import ingest_path

    settings = get_settings()
    count = ingest_path(settings, path, category=category)
    click.echo(f"Ingested {count} chunks from {path} (category={category})")


@cli.command("search")
@click.argument("question")
@click.option("--top-k", default=None, type=int, help="Number of chunks to retrieve.")
@click.option("--category", default=None, help="Filter results to a document category.")
def search_cmd(question: str, top_k: int | None, category: str | None) -> None:
    """Retrieve similar chunks without LLM generation."""
    from src.query import retrieve

    settings = get_settings()
    chunks = retrieve(settings, question, top_k=top_k, category=category)

    if not chunks:
        click.echo("No chunks found. Run `python main.py ingest data/sample_docs` first.")
        return

    for index, chunk in enumerate(chunks, start=1):
        click.echo(f"\n--- Result {index} ({chunk.similarity:.3f}) ---")
        click.echo(f"Title: {chunk.title}")
        click.echo(f"Category: {chunk.category}")
        click.echo(f"Type: {chunk.doc_type}")
        click.echo(f"Source: {chunk.source}")
        click.echo(chunk.content)


@cli.command("ask")
@click.argument("question")
@click.option(
    "--retrieve-only",
    is_flag=True,
    help="Skip OpenAI answer generation even if OPENAI_API_KEY is set.",
)
@click.option("--category", default=None, help="Filter results to a document category.")
@click.option("--top-k", default=None, type=int, help="Number of chunks to retrieve.")
def ask_cmd(
    question: str,
    retrieve_only: bool,
    category: str | None,
    top_k: int | None,
) -> None:
    """Ask a question using retrieval and optional OpenAI generation."""
    from src.query import ask

    settings = get_settings()
    result = ask(
        settings,
        question,
        generate=not retrieve_only,
        category=category,
        top_k=top_k,
    )

    chunks = result["chunks"]
    if not chunks:
        click.echo("No chunks found. Run `python main.py ingest data/sample_docs` first.")
        return

    click.echo("\nRetrieved context:")
    for index, chunk in enumerate(chunks, start=1):
        click.echo(f"\n[{index}] {chunk.title} ({chunk.category}, {chunk.similarity:.3f})")
        preview = chunk.content[:220] + ("..." if len(chunk.content) > 220 else "")
        click.echo(preview)

    if result["answer"]:
        click.echo("\nAnswer:")
        click.echo(result["answer"])
    elif not settings.openai_api_key:
        click.echo(
            "\nRetrieval-only mode. Set OPENAI_API_KEY in .env to enable generated answers."
        )


@cli.command("ui")
def ui_cmd() -> None:
    """Launch the Streamlit web UI."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)


if __name__ == "__main__":
    cli()
