# ingestion.py
"""
OFFLINE PIPELINE — Run this ONCE to ingest documents.
Think of this as "teaching" the system about your documents.
"""
import os
# from config import VECTOR_STORE_DIR
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import (
    DATA_DIR, VECTOR_STORE_DIR,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
)

def load_documents(source: str = "directory", url: str = None):
    """
    Load documents from various sources.
    
    Analogy: Think of this as the research assistant
    picking up all the files from the desk.
    """
    if source == "directory":
        # Load all PDFs from the data/docs/ folder
        loader = DirectoryLoader(
            str(DATA_DIR),
            glob="**/*.pdf", #Find every PDF file inside this folder and all subfolders
            loader_cls=PyPDFLoader,
            show_progress=True,
        )
        documents = loader.load()
        
    elif source == "web":
        # Load from a URL (scrapes the page text)
        loader = WebBaseLoader(url)
        documents = loader.load()
        
    elif source == "text":
        # Load plain .txt files
        loader = DirectoryLoader(
            str(DATA_DIR),
            glob="**/*.txt",
            loader_cls=TextLoader,
        )
        documents = loader.load()
    
    print(f"✅ Loaded {len(documents)} document(s)")
    return documents


def chunk_documents(documents):
    """
    Split documents into smaller overlapping chunks.
    
    Analogy: The assistant writes sticky notes for each section,
    making sure adjacent notes share a sentence so nothing is lost.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # This tries to split at paragraphs first, then sentences,
        # then words — never breaking mid-sentence if avoidable
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    chunks = splitter.split_documents(documents)
    
    print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
    print(f"   Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    
    return chunks


def create_embeddings():
    """
    Load the embedding model.
    
    Analogy: This is the "fingerprinting machine" that converts
    any piece of text into a unique numeric signature.
    Free — runs locally, no API key needed.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},    # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity needs normalized vectors
    )
    print(f"✅ Loaded embedding model: {EMBEDDING_MODEL}")
    return embeddings


def build_vector_store(chunks, embeddings):
    """
    Embed all chunks and store in FAISS.
    
    Analogy: Fingerprint every sticky note and file it in
    a cabinet sorted by fingerprint similarity.
    This is the slow step — be patient.
    """
    print(f"⏳ Embedding {len(chunks)} chunks... (this may take a minute)")
    
    # FAISS.from_documents embeds every chunk and builds the index
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    
    # Save to disk so we don't re-embed on every app start
    VECTOR_STORE_DIR = Path("vector_store")

    # ✅ IMPORTANT FIX
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
      # 🔥 FIX: ensure directory exists
    # os.makedirs(str(VECTOR_STORE_DIR), exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    
    print(f"✅ Vector store saved to: {VECTOR_STORE_DIR}")
    return vector_store


def run_ingestion(source: str = "directory", url: str = None):
    """Master function — runs the full offline pipeline."""
    print("\n🚀 Starting ingestion pipeline...\n")
    
    documents = load_documents(source, url)
    if not documents:
        raise ValueError(f"No documents found in {DATA_DIR}. Add some PDFs!")
    
    chunks = chunk_documents(documents)
    embeddings = create_embeddings()
    vector_store = build_vector_store(chunks, embeddings)
    
    print("\n✅ Ingestion complete! You can now run app.py\n")
    return vector_store


if __name__ == "__main__":
    run_ingestion()