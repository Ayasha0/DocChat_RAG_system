# Quick test in Python REPL
from rag_pipeline import RAGPipeline
p = RAGPipeline()
result = p.ask("What is the main topic of the document?")
print(result["answer"])
print(result["sources"])