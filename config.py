# config.py
import os
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "docs"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"
# VECTOR_STORE_DIR = "./vector_store"

# ─── Embedding Model ─────────────────────────────────────────────────────────
# Free HuggingFace model — no API key needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─── LLM ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"  # Free tier available

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE = 1000      # ~200 words per chunk
CHUNK_OVERLAP = 200    # Overlap to not lose context at boundaries

# ─── Retrieval ────────────────────────────────────────────────────────────────
TOP_K = 4              # Return 4 most relevant chunks

# ─── Memory ───────────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 5  # Keep last 5 conversation turns