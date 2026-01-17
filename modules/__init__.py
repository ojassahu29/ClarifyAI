"""
RAG Policy Assistant Modules

This package contains modular components for the RAG-based policy Q&A system:
- document_loader: Document ingestion and chunking
- vector_store: Embeddings and vector database management
- retriever: Semantic search and document retrieval
- llm_interface: Grok LLM wrapper
- rag_chain: Complete RAG pipeline
- memory: Conversation memory management
- conflict_detector: Ambiguity and conflict detection
- sensitive_handler: Sensitive topic handling
"""

from .document_loader import DocumentLoader
from .vector_store import VectorStoreManager
from .retriever import PolicyRetriever
from .llm_interface import GrokLLM
from .rag_chain import RAGChain
from .memory import ConversationMemory
from .conflict_detector import ConflictDetector
from .sensitive_handler import SensitiveTopicHandler
from .hf_dataset_loader import HuggingFaceDatasetLoader, download_hr_dataset

__all__ = [
    "DocumentLoader",
    "VectorStoreManager", 
    "PolicyRetriever",
    "GrokLLM",
    "RAGChain",
    "ConversationMemory",
    "ConflictDetector",
    "SensitiveTopicHandler",
    "HuggingFaceDatasetLoader",
    "download_hr_dataset"
]
