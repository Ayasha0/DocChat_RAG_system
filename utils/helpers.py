# utils/helpers.py
"""Utility functions for formatting and evaluation."""

def format_docs(docs) -> str:
    """
    Convert a list of LangChain Document objects into a single context string.
    Used when building prompts manually.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        formatted.append(
            f"[Source {i} — {source}, page {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


def evaluate_answer(question: str, answer: str, contexts: list[str]) -> dict:
    """
    Simple heuristic evaluation of answer quality.
    
    In production you'd use RAGAS or a judge LLM,
    but this gives a quick sanity check.
    
    Returns:
        {
            "has_answer": bool,      # Did LLM refuse to answer?
            "uses_context": bool,    # Does answer reference context?
            "answer_length": int,    # Word count of answer
        }
    """
    has_answer = "don't have enough information" not in answer.lower()
    
    # Check if any context words appear in the answer
    context_text = " ".join(contexts).lower()
    answer_words = set(answer.lower().split())
    context_words = set(context_text.split())
    overlap = answer_words & context_words
    uses_context = len(overlap) > 10  # At least 10 words in common
    
    return {
        "has_answer": has_answer,
        "uses_context": uses_context,
        "answer_length": len(answer.split()),
    }