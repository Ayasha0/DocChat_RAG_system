# app.py
"""
Streamlit frontend — the user-facing chat interface.
This is the entry point: streamlit run app.py
"""
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

# Load .env file (GOOGLE_API_KEY etc.)
load_dotenv()

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — RAG System",
    page_icon="📚",
    layout="centered",
)

st.title("📚 DocChat — Ask Your Documents")
st.caption("Powered by Gemini 2.5 Flash + LangChain + FAISS")

# ─── Initialize Pipeline (once per session) ────────────────────────────────────
# st.session_state persists across reruns of the app
# so the pipeline is only built once, not on every message
@st.cache_resource  # Cache across sessions too — loads model once
def get_pipeline():
    return RAGPipeline()

try:
    pipeline = get_pipeline()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `python ingestion.py` first to ingest your documents.")
    st.stop()

# ─── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # List of {"role": ..., "content": ...}

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        pipeline.clear_history()
        st.rerun()
    
    st.divider()
    st.markdown("**How to use:**")
    st.markdown("1. Drop PDFs in `data/docs/`")
    st.markdown("2. Run `python ingestion.py`")
    st.markdown("3. Ask anything!")
    
    st.divider()
    show_sources = st.toggle("Show source citations", value=True)

# ─── Chat Display ─────────────────────────────────────────────────────────────
# Replay all previous messages on rerun
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources") and show_sources:
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# ─── Chat Input ───────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about your documents..."):
    
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents and generating answer..."):
            result = pipeline.ask(question)
        
        answer = result["answer"]
        sources = result["sources"]
        
        st.markdown(answer)
        
        if sources and show_sources:
            with st.expander("📎 Sources"):
                for src in sources:
                    st.markdown(f"- {src}")
    
    # Save assistant response to session
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })