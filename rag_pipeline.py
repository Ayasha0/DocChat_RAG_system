# rag_pipeline.py
"""
ONLINE PIPELINE — Runs on every user query.
Connects retriever, memory, prompt, and LLM into one chain.
"""
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# from langchain.memory import ConversationBufferWindowMemory
from langchain_classic.memory import ConversationBufferMemory

# from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import HumanMessage, AIMessage
from config import (
    VECTOR_STORE_DIR, EMBEDDING_MODEL,
    GEMINI_MODEL, TOP_K, MAX_HISTORY_TURNS
)
from prompts import RAG_PROMPT, REFORMULATION_PROMPT
from langchain_core.runnables import RunnableLambda

# new version 
# create_history_aware_retriever
# Create a chain that takes conversation history and returns documents.
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

def load_vector_store():
    """
    Load the pre-built FAISS index from disk.
    This is fast — the heavy work was done during ingestion.
    """
    if not VECTOR_STORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {VECTOR_STORE_DIR}.\n"
            "Run: python ingestion.py"
        )
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,  # Safe for local files you created
    )
    
    print("✅ Vector store loaded")
    return vector_store


def build_retriever(vector_store):
    """
    Turn the vector store into a retriever.
    
    Analogy: The retriever is the librarian who takes your
    question and comes back with the 4 most relevant books (chunks).
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",   # cosine similarity search
        search_kwargs={"k": TOP_K}, # return top 4 chunks
    )

    # retriever = vector_store.as_retriever(
    # search_type="similarity_score_threshold",
    # search_kwargs={
    #     "k": TOP_K,
    #     "score_threshold": 0.5
    # },
    # )
    return retriever


def build_llm():
    """Load the Gemini LLM."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY not found. Check your .env file.")
    
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.2,  # Low temperature = more factual, less creative
        max_tokens=1024,
    )
    return llm



def build_rag_chain(retriever, llm):
    """
    Modern LCEL-based RAG chain (replacement for ConversationalRetrievalChain)
    """

    # ─── 1. History-aware question rewriter ─────────────────────────────
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "reformulate it into a standalone question. Do NOT answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    """
    WHAT THE BELOW PIPELINE DOES: 
    It creates a pipeline:

    (input + chat_history)
        ↓
    rephrase_prompt
        ↓
    LLM → standalone question
        ↓
    retriever.search()

    OUTPUT will be: top 4 relevant chunks
    """

    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        rephrase_prompt #(input + chat history)
    )

    # ─── 2. Answer generation chain ─────────────────────────────────────
    """
    It creates a chain:

    (docs + question)
        ↓
    Insert into prompt
        ↓
    LLM generates answer

    OUTPUT : is the answer for the prompt
    """
    qa_chain = create_stuff_documents_chain(
        llm,
        RAG_PROMPT,   # your existing prompt (KEEP THIS unchanged)
    )

    # ─── 3. Full retrieval + generation pipeline ────────────────────────
    """
    User Input
       ↓
    Rewriter (LLM)       -> reformualte
       ↓
    Retriever (FAISS)    -> top 4 chunks
       ↓
    QA Chain (LLM)       -> answer for prompt


    for example: 
    {
      "answer": "...",
      "context": [Document1, Document2, ...]
    }
    """
    
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain
    )

    # ─── 4. 🔥 FIX: map question → input ─────────────────────
    final_chain = RunnableLambda(
        lambda x: {
            "input": x["question"],       # for retriever
            "question": x["question"],    # for prompt
            "chat_history": x["chat_history"],
        }
    ) | rag_chain

    return final_chain

"""
When you call:

self.chain.invoke({
    "question": "What are eligibility criteria?",
    "chat_history": [...]
})

Actual execution:
1. RunnableLambda runs
Transforms input
↓

2. History-aware retriever
rewrites question
retrieves docs
↓

3. QA chain
injects docs into prompt
LLM answers
↓

4. Output returned
{
  "answer": "...",
  "context": [docs]
}
"""


class RAGPipeline:
    """
    Clean wrapper class that manages the RAG chain + conversation history.
    This is what app.py instantiates and calls.
    """
    
    def __init__(self):
        print("⚙️  Initializing RAG pipeline...")
        vector_store = load_vector_store()
        retriever = build_retriever(vector_store)
        llm = build_llm()
        self.chain = build_rag_chain(retriever, llm)
        self.chat_history = []  # List of (HumanMessage, AIMessage) tuples
        print("✅ RAG pipeline ready!\n")
    
    def ask(self, question: str) -> dict:
        """
        Ask a question and get an answer with sources.
        
        Returns:
            {
                "answer": "The answer text...",
                "sources": ["Page 3 of doc.pdf", "Page 7 of doc.pdf"],
            }
        """
        # Run the chain
        result = self.chain.invoke({
            "question": question,
            # "input": question,
            "chat_history": self.chat_history,
        })
        
        answer = result["answer"]
        # source_docs = result.get("source_documents", [])
        source_docs = result.get("context", [])
        
        # Extract unique source citations
        sources = list({
            f"📄 {doc.metadata.get('source', 'Unknown')} — page {doc.metadata.get('page', '?')}"
            for doc in source_docs
        })

        # sources = []

        # for doc in source_docs:
        #     source = doc.metadata.get("source", "Unknown")
        #     page = doc.metadata.get("page", "?")
        #     snippet = doc.page_content[:200].replace("\n", " ")

        #     sources.append(f"""
        # 📄 {source} (Page {page})
        # 🧩 {snippet}...
        # """)

        sources = list(set(sources))
        
        # Update chat history (keep last N turns to avoid token overflow)
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))
        
        # Trim history to MAX_HISTORY_TURNS * 2 messages
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.chat_history) > max_messages:
            self.chat_history = self.chat_history[-max_messages:]
        
        return {"answer": answer, "sources": sources}
    
    def clear_history(self):
        """Reset conversation — start fresh."""
        self.chat_history = []
        print("🗑️  Conversation history cleared.")
