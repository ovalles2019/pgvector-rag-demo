"""Streamlit UI for the pgvector RAG demo."""

from __future__ import annotations

import streamlit as st

from src.config import get_settings
from src.db import connect, get_stats, init_schema
from src.ingest import ingest_bytes, ingest_path
from src.loaders import SUPPORTED_EXTENSIONS
from src.query import ask, get_available_categories, retrieve

st.set_page_config(
    page_title="pgvector RAG Demo",
    page_icon="🔍",
    layout="wide",
)

SUPPORTED_UPLOAD_TYPES = sorted(SUPPORTED_EXTENSIONS)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def render_sidebar(settings) -> tuple[str | None, int, bool]:
    st.sidebar.title("Knowledge Base")

    try:
        with connect(settings) as conn:
            init_schema(conn, settings.embedding_dimension)
            stats = get_stats(conn)
            categories = get_available_categories(settings)
    except Exception as exc:
        st.sidebar.error(f"Database unavailable: {exc}")
        st.sidebar.info("Run `docker compose up -d` and `python main.py init-db` first.")
        return None, settings.top_k, True

    st.sidebar.metric("Documents", stats["documents"])
    st.sidebar.metric("Chunks", stats["chunks"])

    category_options = ["All categories", *categories]
    selected_category = st.sidebar.selectbox("Filter by category", category_options)
    category_filter = None if selected_category == "All categories" else selected_category

    top_k = st.sidebar.slider("Results to retrieve", min_value=1, max_value=10, value=settings.top_k)
    generate_answers = st.sidebar.toggle(
        "Generate answers with OpenAI",
        value=bool(settings.openai_api_key),
        disabled=not settings.openai_api_key,
        help="Set OPENAI_API_KEY in .env to enable answer generation.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Ingest documents")

    upload_category = st.sidebar.text_input("Upload category", value="general")
    uploaded_files = st.sidebar.file_uploader(
        "Upload .txt, .md, or .pdf files",
        type=[ext.lstrip(".") for ext in SUPPORTED_UPLOAD_TYPES],
        accept_multiple_files=True,
    )

    if st.sidebar.button("Ingest uploaded files", use_container_width=True):
        if not uploaded_files:
            st.sidebar.warning("Choose at least one file to ingest.")
        else:
            with st.spinner("Ingesting uploaded files..."):
                total_chunks = 0
                for uploaded in uploaded_files:
                    total_chunks += ingest_bytes(
                        settings,
                        filename=uploaded.name,
                        content=uploaded.getvalue(),
                        category=upload_category.strip() or "general",
                    )
            st.sidebar.success(f"Ingested {total_chunks} chunks from {len(uploaded_files)} file(s).")
            st.rerun()

    local_path = st.sidebar.text_input("Or ingest local path", value="data/sample_docs")
    local_category = st.sidebar.text_input("Local ingest category", value="general")
    if st.sidebar.button("Ingest local path", use_container_width=True):
        with st.spinner(f"Ingesting {local_path}..."):
            count = ingest_path(
                settings,
                local_path,
                category=local_category.strip() or "general",
            )
        st.sidebar.success(f"Ingested {count} chunks from {local_path}.")
        st.rerun()

    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    return category_filter, top_k, generate_answers


def render_sources(chunks) -> None:
    st.markdown("**Sources**")
    for index, chunk in enumerate(chunks, start=1):
        label = (
            f"{index}. {chunk.title} · {chunk.category} · "
            f"{chunk.doc_type.upper()} · score {chunk.similarity:.3f}"
        )
        with st.expander(label):
            st.caption(chunk.source)
            st.write(chunk.content)


def main() -> None:
    init_session_state()
    settings = get_settings()

    st.title("pgvector RAG Demo")
    st.caption("Retrieve grounded answers from PostgreSQL + pgvector")

    category_filter, top_k, generate_answers = render_sidebar(settings)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("chunks"):
                render_sources(message["chunks"])

    prompt = st.chat_input("Ask a question about your documents...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                if generate_answers and settings.openai_api_key:
                    result = ask(
                        settings,
                        prompt,
                        generate=True,
                        category=category_filter,
                        top_k=top_k,
                    )
                    chunks = result["chunks"]
                    answer = result["answer"] or "_No answer generated._"
                else:
                    chunks = retrieve(
                        settings,
                        prompt,
                        top_k=top_k,
                        category=category_filter,
                    )
                    if chunks:
                        answer = (
                            "_Retrieval-only mode._ Top matching passages are shown below. "
                            "Set `OPENAI_API_KEY` to enable generated answers."
                        )
                    else:
                        answer = (
                            "No matching chunks found. Ingest documents from the sidebar first."
                        )
            except Exception as exc:
                chunks = []
                answer = f"Something went wrong: {exc}"

        st.markdown(answer)
        if chunks:
            render_sources(chunks)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "chunks": chunks}
    )


if __name__ == "__main__":
    main()
