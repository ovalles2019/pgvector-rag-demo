"""Streamlit UI for the pgvector RAG demo."""

from __future__ import annotations

import os

import streamlit as st

from src.config import get_settings
from src.db import connect, get_stats, init_schema
from src.ingest import ingest_bytes, ingest_path
from src.loaders import SUPPORTED_EXTENSIONS
from src.query import ask, get_available_categories, retrieve

IS_DEMO = os.environ.get("PGVECTOR_RAG_DEMO", "").strip().lower() in ("1", "true", "yes", "on")

DEMO_QUESTIONS = [
    "What is pgvector used for?",
    "How does RAG reduce hallucination?",
    "What embedding model does this demo use?",
    "Explain cosine similarity search",
]

st.set_page_config(
    page_title="pgvector RAG Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed" if IS_DEMO else "expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 920px; }
    h1 { font-weight: 700; letter-spacing: -0.03em; }
    .demo-pill {
        display: inline-block; margin-bottom: 1rem; padding: 0.35rem 0.75rem;
        border-radius: 999px; font-size: 0.8rem; font-weight: 600;
        background: #ede9fe; color: #5b21b6; border: 1px solid #c4b5fd;
    }
    .hint { color: #64748b; font-size: 0.92rem; margin-bottom: 1.25rem; }
    div[data-testid="stChatMessage"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

SUPPORTED_UPLOAD_TYPES = sorted(SUPPORTED_EXTENSIONS)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar(settings) -> tuple[str | None, int, bool]:
    with st.sidebar:
        st.markdown("### Settings")

        try:
            with connect(settings) as conn:
                init_schema(conn, settings.embedding_dimension)
                stats = get_stats(conn)
                categories = get_available_categories(settings)
        except Exception as exc:
            st.error(f"Database unavailable: {exc}")
            return None, settings.top_k, False

        c1, c2 = st.columns(2)
        c1.metric("Docs", stats["documents"])
        c2.metric("Chunks", stats["chunks"])

        category_options = ["All categories", *categories]
        selected = st.selectbox("Category filter", category_options)
        category_filter = None if selected == "All categories" else selected

        top_k = st.slider("Chunks to retrieve", 1, 8, min(settings.top_k, 4))
        generate_answers = st.toggle(
            "Generate answers (OpenAI)",
            value=False,
            disabled=not settings.openai_api_key,
            help="Optional — retrieval works without an API key.",
        )

        if not IS_DEMO:
            st.divider()
            st.caption("Ingest more documents")
            upload_category = st.text_input("Category", value="general")
            uploaded = st.file_uploader(
                "Upload files",
                type=[ext.lstrip(".") for ext in SUPPORTED_UPLOAD_TYPES],
                accept_multiple_files=True,
            )
            if st.button("Ingest uploads", use_container_width=True) and uploaded:
                with st.spinner("Ingesting…"):
                    total = 0
                    for f in uploaded:
                        total += ingest_bytes(
                            settings,
                            filename=f.name,
                            content=f.getvalue(),
                            category=upload_category.strip() or "general",
                        )
                st.success(f"Ingested {total} chunks.")
                st.rerun()

            local_path = st.text_input("Local path", value="data/sample_docs")
            if st.button("Ingest path", use_container_width=True):
                with st.spinner("Ingesting…"):
                    count = ingest_path(settings, local_path, category="general")
                st.success(f"Ingested {count} chunks.")
                st.rerun()

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return category_filter, top_k, generate_answers


def render_sources(chunks) -> None:
    st.markdown("**Sources**")
    for index, chunk in enumerate(chunks, start=1):
        label = f"{index}. {chunk.title} · {chunk.category} · score {chunk.similarity:.2f}"
        with st.expander(label):
            st.caption(chunk.source)
            st.write(chunk.content)


def run_query(
    settings,
    prompt: str,
    category_filter: str | None,
    top_k: int,
    generate_answers: bool,
) -> tuple[str, list]:
    if generate_answers and settings.openai_api_key:
        result = ask(
            settings,
            prompt,
            generate=True,
            category=category_filter,
            top_k=top_k,
        )
        return result["answer"] or "_No answer generated._", result["chunks"]

    chunks = retrieve(settings, prompt, top_k=top_k, category=category_filter)
    if not chunks:
        return "No matching passages found. Try another question.", []

    answer = (
        "Here are the most relevant passages from the knowledge base "
        "(retrieval-only — add `OPENAI_API_KEY` on the server for generated answers)."
    )
    return answer, chunks


def main() -> None:
    init_session_state()
    settings = get_settings()

    if IS_DEMO:
        st.markdown('<span class="demo-pill">Portfolio demo · PostgreSQL + pgvector</span>', unsafe_allow_html=True)

    st.title("Ask your documents")
    st.markdown(
        '<p class="hint">RAG over sample docs with <strong>pgvector</strong> semantic search '
        "and local <strong>sentence-transformers</strong> embeddings.</p>",
        unsafe_allow_html=True,
    )

    category_filter, top_k, generate_answers = render_sidebar(settings)

    if IS_DEMO and not st.session_state.messages:
        st.caption("Try a sample question:")
        cols = st.columns(2)
        for i, q in enumerate(DEMO_QUESTIONS):
            if cols[i % 2].button(q, key=f"demo-q-{i}", use_container_width=True):
                st.session_state["_pending_prompt"] = q
                st.rerun()

    pending = st.session_state.pop("_pending_prompt", None)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("chunks"):
                render_sources(message["chunks"])

    prompt = pending or st.chat_input("Ask about pgvector, RAG, or embeddings…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            try:
                answer, chunks = run_query(
                    settings, prompt, category_filter, top_k, generate_answers
                )
            except Exception as exc:
                answer, chunks = f"Something went wrong: {exc}", []

        st.markdown(answer)
        if chunks:
            render_sources(chunks)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "chunks": chunks}
    )


if __name__ == "__main__":
    main()
