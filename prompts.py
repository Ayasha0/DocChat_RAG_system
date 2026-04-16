# prompts.py
"""
Prompt templates are the INSTRUCTIONS given to the LLM.
Bad prompts = hallucinations. Good prompts = grounded, honest answers.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ─── Main RAG Prompt ─────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a helpful, precise AI assistant.
Your job is to answer questions STRICTLY based on the provided context.

Rules you must follow:
1. Answer ONLY from the context below. Never invent information.
2. If the answer is not in the context, say: "I don't have enough information in the provided documents to answer this."
3. Always be concise and direct.
4. When relevant, cite which part of the document your answer comes from.
5. If the user asks a follow-up, use the chat history to understand what they're referring to.

Context from documents:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),  # Injects conversation memory
    ("human", "{question}"),
])


# ─── Standalone Question Reformulation Prompt ─────────────────────────────────
# When user asks "Tell me more about that", the LLM needs to know what "that" is.
# This prompt reformulates follow-up questions into self-contained questions.
REFORMULATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """Given the conversation history and the follow-up question,
reformulate the question to be fully standalone and self-contained.
Do NOT answer it — just rewrite it clearly.
If it's already standalone, return it unchanged."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Follow-up question: {question}\n\nReformulated standalone question:"),
])