#!/usr/bin/env python3
"""Streamlit chat UI for the Medical RAG Agent."""
import sys
import os
import tempfile
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import requests
import streamlit as st
from llama_index.core import SimpleDirectoryReader, Document
from src.rag import add_documents, query_stream, get_collection_count

DOCS_DIR = ROOT / "documents"
DOCS_DIR.mkdir(exist_ok=True)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@st.cache_resource(show_spinner="Warming up model...")
def warmup_model(model: str):
    """Load the model into Ollama's memory at startup so first query is fast."""
    try:
        requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": "", "keep_alive": "60m"},
            timeout=60,
        )
    except Exception:
        pass


def check_ollama() -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

SYSTEM_PREAMBLE = (
    "You are a specialized medical research assistant. "
    "Answer based on the provided context from medical documents and research papers. "
    "Always cite when information comes from the research context. "
    "If you are unsure, say so. This is for informational purposes only, not medical advice."
)

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; }
    .source-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #7eb8f7;
        border-radius: 6px;
        padding: 2px 10px;
        margin: 2px 4px 2px 0;
        font-size: 0.78em;
    }
    .stat-card {
        background: #0e1117;
        border: 1px solid #262730;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🩺 Medical RAG")
    st.caption("Local · Private · On-premise")
    st.divider()

    ollama_ok = check_ollama()
    if ollama_ok:
        st.success("Ollama connected", icon="✅")
    else:
        st.error("Ollama not running — start with `ollama serve`", icon="🔴")

    doc_count = get_collection_count()
    st.markdown(f"""
    <div class="stat-card">
        <b>Knowledge Base</b><br>
        <span style="font-size:1.4em;font-weight:700;color:#7eb8f7">{doc_count}</span>
        <span style="color:#888"> chunks indexed</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.subheader("Settings")
    top_k = st.slider("Chunks retrieved per query", min_value=1, max_value=15, value=3)
    llm_model = st.text_input("Ollama model", value=os.getenv("LLM_MODEL", "llama3.2:3b"))

    if ollama_ok:
        warmup_model(llm_model)

    st.divider()
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDFs, DOCX, or TXT files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Ingest uploaded files", type="primary", use_container_width=True):
        with st.spinner("Ingesting documents..."):
            documents = []
            for uf in uploaded_files:
                suffix = Path(uf.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.read())
                    tmp_path = tmp.name

                dest = DOCS_DIR / uf.name
                Path(tmp_path).rename(dest)

            reader = SimpleDirectoryReader(
                input_dir=str(DOCS_DIR),
                recursive=True,
                required_exts=[".pdf", ".docx", ".txt", ".md"],
            )
            documents = reader.load_data()
            add_documents(documents)
        st.success(f"Ingested {len(uploaded_files)} file(s) — {len(documents)} chunks added.")
        st.rerun()

    st.divider()
    st.subheader("Research Agent")
    pubmed_query = st.text_input(
        "PubMed search query",
        value=os.getenv("PUBMED_SEARCH_QUERY", ""),
        placeholder="e.g. type 2 diabetes insulin resistance",
    )
    col1, col2 = st.columns(2)
    with col1:
        pub_limit = st.number_input("Max papers", value=10, min_value=1, max_value=50)
    with col2:
        pub_days = st.number_input("Days back", value=90, min_value=7, max_value=365)

    if st.button("Fetch new research", use_container_width=True):
        if not pubmed_query:
            st.warning("Enter a PubMed search query first.")
        else:
            with st.spinner("Fetching from PubMed..."):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts" / "research_agent.py"),
                        "--query", pubmed_query,
                        "--limit", str(pub_limit),
                        "--days", str(pub_days),
                    ],
                    capture_output=True,
                    text=True,
                )
            if result.returncode == 0:
                st.success("Research fetched and ingested!")
            else:
                st.error(f"Error: {result.stderr[:300]}")
            st.rerun()

    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Chat ─────────────────────────────────────────────────────────────────

st.title("Medical Research Assistant")
st.caption("Ask questions about your medical documents and research papers.")

if "messages" not in st.session_state:
    st.session_state.messages = []

def _render_sources(sources: list):
    if not sources:
        return
    seen = set()
    badges = []
    for s in sources:
        label = s.get("title") or s.get("name", "source")
        label = label[:50] + "…" if len(label) > 50 else label
        key = s.get("name", label)
        if key in seen:
            continue
        seen.add(key)
        url = s.get("url", "")
        if url:
            badges.append(f'<a href="{url}" target="_blank" class="source-badge">📄 {label}</a>')
        else:
            badges.append(f'<span class="source-badge">📄 {label}</span>')
    if badges:
        st.markdown("**Sources:** " + " ".join(badges), unsafe_allow_html=True)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🩺"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"])


if doc_count == 0:
    st.info("Your knowledge base is empty. Upload documents or fetch research using the sidebar to get started.")

prompt = st.chat_input("Ask about your medical condition, treatments, research findings...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🩺"):
        placeholder = st.empty()
        full_response = ""
        final_sources = []

        if doc_count == 0:
            full_response = "The knowledge base is empty. Please upload your medical documents or fetch research from PubMed using the sidebar first."
            placeholder.markdown(full_response)
        else:
            try:
                full_question = f"{SYSTEM_PREAMBLE}\n\nQuestion: {prompt}"
                for token, sources in query_stream(full_question, top_k=top_k, llm_model=llm_model):
                    if sources is not None:
                        final_sources = sources
                    else:
                        full_response += token
                        placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
                if final_sources:
                    _render_sources(final_sources)
            except Exception as e:
                err = str(e)
                full_response = f"Error: {err}"
                if "connection" in err.lower():
                    full_response += "\n\n> Is Ollama running? Start it with: `ollama serve`"
                placeholder.markdown(full_response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": final_sources,
    })
