"""Streamlit UI for the pgvector RAG demo."""

from __future__ import annotations

import os

import streamlit as st

from src.config import Settings, get_settings
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

SUPPORTED_UPLOAD_TYPES = sorted(SUPPORTED_EXTENSIONS)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


@st.cache_data(ttl=30, show_spinner=False)
def load_db_snapshot(database_url: str, dimension: int) -> dict:
    settings = get_settings()
    with connect(settings) as conn:
        init_schema(conn, settings.embedding_dimension)
        stats = get_stats(conn)
        categories = get_available_categories(settings)
    return {"stats": stats, "categories": categories}


def apply_demo_theme() -> None:
    st.markdown(
        """
        <style>
        /* Hide empty collapsed sidebar on the public demo */
        section[data-testid="stSidebar"] { display: none !important; }
        button[data-testid="stSidebarCollapsedControl"] { display: none !important; }
        header[data-testid="stHeader"] { background: rgba(255,255,255,0.92); }
        footer { visibility: hidden; }
        #MainMenu { visibility: hidden; }
        section.main > div.block-container {
            max-width: 44rem;
            padding-top: 1.25rem;
            padding-bottom: 7rem;
        }
        .demo-pill {
            display: inline-block;
            margin: 0 0 0.75rem 0;
            padding: 0.3rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: #ede9fe;
            color: #5b21b6;
            border: 1px solid #c4b5fd;
        }
        .demo-subtitle { color: #64748b; font-size: 1rem; line-height: 1.5; margin-bottom: 0; }
        div[data-testid="stChatMessage"] { border-radius: 12px; }
        [data-testid="stBottomBlockContainer"] {
            padding-bottom: 0.75rem;
            background: linear-gradient(transparent, #ffffff 28%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_local_theme() -> None:
    st.markdown(
        """
        <style>
        section.main > div.block-container { max-width: 52rem; padding-bottom: 6rem; }
        div[data-testid="stChatMessage"] { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(settings: Settings) -> tuple[str | None, int, bool]:
    with st.sidebar:
        st.markdown("### Settings")

        try:
            snapshot = load_db_snapshot(settings.database_url, settings.embedding_dimension)
            stats = snapshot["stats"]
            categories = snapshot["categories"]
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
            load_db_snapshot.clear()
            st.rerun()

        local_path = st.text_input("Local path", value="data/sample_docs")
        if st.button("Ingest path", use_container_width=True):
            with st.spinner("Ingesting…"):
                count = ingest_path(settings, local_path, category="general")
            st.success(f"Ingested {count} chunks.")
            load_db_snapshot.clear()
            st.rerun()

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    return category_filter, top_k, generate_answers


def render_demo_controls(
    settings: Settings,
) -> tuple[str | None, int, bool, str | None]:
    try:
        snapshot = load_db_snapshot(settings.database_url, settings.embedding_dimension)
        stats = snapshot["stats"]
    except Exception as exc:
        st.error(f"Database unavailable: {exc}")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", stats["documents"])
    m2.metric("Chunks", stats["chunks"])
    m3.metric("Mode", "Retrieval")

    st.caption("Sample questions — click one or type below.")
    picked: str | None = None
    row1 = st.columns(2)
    row2 = st.columns(2)
    for i, question in enumerate(DEMO_QUESTIONS):
        col = row1[i % 2] if i < 2 else row2[i % 2]
        if col.button(question, key=f"demo-q-{i}", use_container_width=True):
            picked = question

    return None, min(settings.top_k, 4), False, picked


def render_sources(chunks) -> None:
    st.markdown("**Sources**")
    for index, chunk in enumerate(chunks, start=1):
        label = f"{index}. {chunk.title} · {chunk.category} · score {chunk.similarity:.2f}"
        with st.expander(label):
            st.caption(chunk.source)
            st.write(chunk.content)


def run_query(
    settings: Settings,
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


def render_chat(
    settings: Settings,
    category_filter: str | None,
    top_k: int,
    generate_answers: bool,
    pending: str | None = None,
) -> None:
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


def main() -> None:
    st.set_page_config(
        page_title="pgvector RAG Demo",
        page_icon="🔍",
        layout="centered" if IS_DEMO else "wide",
        initial_sidebar_state="collapsed" if IS_DEMO else "expanded",
    )
    init_session_state()
    settings = get_settings()

    if IS_DEMO:
        apply_demo_theme()
    else:
        apply_local_theme()

    pending_from_button: str | None = None

    if IS_DEMO:
        st.markdown(
            '<span class="demo-pill">Portfolio demo · PostgreSQL + pgvector</span>',
            unsafe_allow_html=True,
        )
        st.title("Ask your documents")
        st.markdown(
            '<p class="demo-subtitle">Semantic search over sample docs using '
            "<strong>pgvector</strong> and local <strong>sentence-transformers</strong> "
            "embeddings.</p>",
            unsafe_allow_html=True,
        )
        st.divider()
        category_filter, top_k, generate_answers, pending_from_button = render_demo_controls(
            settings
        )
    else:
        st.title("Ask your documents")
        st.caption(
            "RAG over your documents with pgvector semantic search and sentence-transformers."
        )
        category_filter, top_k, generate_answers = render_sidebar(settings)

    pending = pending_from_button or st.session_state.pop("_pending_prompt", None)
    render_chat(settings, category_filter, top_k, generate_answers, pending=pending)


if __name__ == "__main__":
    main()
