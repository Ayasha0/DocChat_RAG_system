"""
Streamlit frontend — full RAG app with ingestion UI
Run: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline
from ingestion import run_ingestion
from config import DATA_DIR

load_dotenv()

# ─── Page Config ─────────────────────────────
st.set_page_config(
    page_title="DocChat — RAG System",
    page_icon="📚",
    layout="centered",
)

st.title("📚 DocChat — Ask Your Documents")
st.caption("Gemini 2.5 flash + LangChain + FAISS")

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─── Pipeline ────────────────────────────────
@st.cache_resource
def get_pipeline():
    return RAGPipeline()

def reload_pipeline():
    st.cache_resource.clear()
    return get_pipeline()

try:
    pipeline = get_pipeline()
except FileNotFoundError:
    pipeline = None

# ─── Session State ───────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── SIDEBAR ────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        if pipeline:
            pipeline.clear_history()
        st.rerun()

    st.divider()
    st.subheader("📥 Add Data")

    source_type = st.selectbox(
        "Choose source",
        ["PDF Upload", "Text Upload", "Web URL"]
    )

    if source_type == "PDF Upload":
        pdf_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Ingest PDFs") and pdf_files:
            for file in pdf_files:
                with open(DATA_DIR / file.name, "wb") as f:
                    f.write(file.read())

            with st.spinner("Processing PDFs..."):
                run_ingestion(source="directory")

            pipeline = reload_pipeline()
            st.success("PDFs ingested!")
            st.rerun()

    elif source_type == "Text Upload":
        txt_files = st.file_uploader(
            "Upload text files",
            type=["txt"],
            accept_multiple_files=True
        )

        if st.button("Ingest Text Files") and txt_files:
            for file in txt_files:
                with open(DATA_DIR / file.name, "wb") as f:
                    f.write(file.read())

            with st.spinner("Processing text..."):
                run_ingestion(source="text")

            pipeline = reload_pipeline()
            st.success("Text ingested!")
            st.rerun()

    elif source_type == "Web URL":
        url = st.text_input("Enter URL")

        if st.button("Ingest URL") and url:
            with st.spinner("Fetching web content..."):
                run_ingestion(source="web", url=url)

            pipeline = reload_pipeline()
            st.success("Web content ingested!")
            st.rerun()

    st.divider()

    if st.button("🔄 Rebuild Knowledge Base"):
        with st.spinner("Rebuilding..."):
            run_ingestion(source="directory")

        pipeline = reload_pipeline()
        st.success("Rebuilt!")
        st.rerun()

    st.divider()

    show_sources = st.toggle("Show sources", value=True)

# ─── MAIN CHECK ─────────────────────────────
if not pipeline:
    st.warning("⚠️ No data ingested yet.")
    st.stop()

# ─── CHAT HISTORY ───────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("sources") and show_sources:
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# ─── CHAT INPUT ─────────────────────────────
if question := st.chat_input("Ask a question..."):

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching + generating..."):
            result = pipeline.ask(question)

        answer = result["answer"]
        sources = result["sources"]

        st.markdown(answer)

        if sources and show_sources:
            with st.expander("📎 Sources"):
                for src in sources:
                    st.markdown(f"- {src}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })