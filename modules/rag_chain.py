"""
RAG Chain Module

Complete Retrieval-Augmented Generation pipeline integrating
document retrieval, context formatting, and LLM generation.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from langchain_core.documents import Document

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import TOP_K_RETRIEVAL

from .document_loader import DocumentLoader
from .vector_store import VectorStoreManager
from .retriever import PolicyRetriever
from .llm_interface import GrokLLM


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    context_used: str
    is_sensitive: bool
    confidence: float
    conflict_analysis: Optional[Dict[str, Any]] = None


class RAGChain:
    """
    Complete RAG pipeline for policy question answering.
    
    Integrates:
    - Document retrieval via semantic search
    - Context-aware answer generation with Grok
    - Source citation tracking
    - Conflict detection (optional)
    """
    
    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        llm: Optional[GrokLLM] = None,
        retriever: Optional[PolicyRetriever] = None,
        top_k: int = TOP_K_RETRIEVAL,
        check_conflicts: bool = True
    ):
        """
        Initialize the RAG chain.
        
        Args:
            vector_store_manager: Initialized vector store
            llm: Grok LLM interface
            retriever: Policy retriever
            top_k: Number of documents to retrieve
            check_conflicts: Whether to run conflict detection
        """
        self.vector_manager = vector_store_manager or VectorStoreManager()
        self.retriever = retriever or PolicyRetriever(self.vector_manager)
        self.top_k = top_k
        self.check_conflicts = check_conflicts
        
        # Initialize LLM (may raise error if no API key)
        try:
            self.llm = llm or GrokLLM()
            self.llm_available = True
        except ValueError as e:
            print(f"Warning: LLM not initialized - {e}")
            self.llm = None
            self.llm_available = False
    
    def ingest_documents(self, source_path: Optional[str] = None) -> int:
        """
        Ingest documents into the vector store.
        
        Args:
            source_path: Path to documents (directory or file)
            
        Returns:
            Number of chunks ingested
        """
        loader = DocumentLoader()
        
        if source_path and Path(source_path).is_file():
            chunks = loader.load_and_split(source_path, is_directory=False)
        else:
            chunks = loader.load_and_split(source_path, is_directory=True)
        
        if not chunks:
            print("No documents found to ingest.")
            return 0
        
        self.vector_manager.create_store(chunks)
        return len(chunks)
    
    def query(
        self,
        question: str,
        chat_history: Optional[List[dict]] = None,
        is_sensitive: bool = False,
        k: Optional[int] = None
    ) -> RAGResponse:
        """
        Process a user question through the RAG pipeline.
        
        Args:
            question: User's question
            chat_history: Previous conversation messages
            is_sensitive: Whether this is a sensitive topic
            k: Number of documents to retrieve
            
        Returns:
            RAGResponse with answer and metadata
        """
        k = k or self.top_k
        
        # Step 1: Retrieve relevant documents
        documents = self.retriever.retrieve(question, k=k)
        
        if not documents:
            return RAGResponse(
                answer="I couldn't find any relevant information in the policy documents for your question.",
                sources=[],
                context_used="",
                is_sensitive=is_sensitive,
                confidence=0.0
            )
        
        # Step 2: Format context
        context = self.retriever.format_context(documents)
        sources = self.retriever.get_source_citations(documents)
        
        # Step 3: Generate answer
        if not self.llm_available:
            return RAGResponse(
                answer="LLM not configured. Please set XAI_API_KEY in your .env file.",
                sources=sources,
                context_used=context,
                is_sensitive=is_sensitive,
                confidence=0.0
            )
        
        answer = self.llm.generate(
            query=question,
            context=context,
            chat_history=chat_history,
            sensitive=is_sensitive
        )
        
        # Step 4: Optional conflict detection
        conflict_analysis = None
        confidence = 0.8
        
        if self.check_conflicts and len(documents) > 1:
            conflict_analysis = self.llm.analyze_conflicts(context, answer)
            
            # Always use confidence from conflict analysis when available
            confidence = conflict_analysis.get("confidence", 0.8)
            
            if conflict_analysis.get("has_conflict"):
                # Add conflict warning to answer
                issues = conflict_analysis.get("issues", [])
                if issues:
                    answer += "\n\n⚠️ **Note:** " + "; ".join(issues)
        else:
            # Single document or conflict checking disabled - high confidence
            # since there's no possibility of conflicting information
            confidence = 0.9
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            context_used=context,
            is_sensitive=is_sensitive,
            confidence=confidence,
            conflict_analysis=conflict_analysis
        )
    
    def is_ready(self) -> bool:
        """Check if the RAG chain is ready for queries."""
        return (
            self.vector_manager.vector_store is not None and
            self.llm_available
        )
    
    def get_status(self) -> dict:
        """Get status of RAG chain components."""
        return {
            "vector_store_initialized": self.vector_manager.vector_store is not None,
            "llm_available": self.llm_available,
            "vector_store_stats": self.vector_manager.get_store_stats(),
            "top_k": self.top_k,
            "conflict_checking_enabled": self.check_conflicts
        }


# Example usage
if __name__ == "__main__":
    # Initialize the RAG chain
    rag = RAGChain()
    
    # Check status
    status = rag.get_status()
    print("RAG Chain Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    if rag.is_ready():
        # Test query
        response = rag.query("What is the vacation policy?")
        print(f"\nAnswer: {response.answer}")
        print(f"Sources: {response.sources}")
        print(f"Confidence: {response.confidence}")
    else:
        print("\nRAG chain not ready. Please:")
        print("1. Add policy documents to data/policies/")
        print("2. Run ingest_documents()")
        print("3. Set XAI_API_KEY in .env file")
