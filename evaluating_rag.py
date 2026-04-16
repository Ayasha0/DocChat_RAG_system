from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

# RAGAS scores your RAG system automatically:
# faithfulness: is the answer grounded in the retrieved context?
# answer_relevancy: does the answer actually address the question?
# context_recall: did retrieval find the right chunks?