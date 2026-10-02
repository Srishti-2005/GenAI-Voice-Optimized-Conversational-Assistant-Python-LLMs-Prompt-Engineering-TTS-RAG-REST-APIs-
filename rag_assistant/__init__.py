"""
GenAI-Powered Document RAG Assistant
A production-grade Retrieval-Augmented Generation pipeline combining:
- Python multi-format document ingestion & semantic chunking
- Dual SQL metadata catalog and Dense Vector Store
- Hybrid Retrieval (Dense Vector + Lexical BM25 with Reciprocal Rank Fusion)
- Multi-provider LLM orchestration with citation grounding
- Comprehensive MLflow experiment tracking & retrieval metrics evaluation
"""

from rag_assistant.pipeline import RAGPipeline
from rag_assistant.config import RAGConfig

__version__ = "1.0.0"
__all__ = ["RAGPipeline", "RAGConfig"]
