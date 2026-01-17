"""
Retriever Module

Provides semantic search capabilities over the policy document store.
Wraps the vector store with additional filtering and ranking logic.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from langchain.schema import Document

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import TOP_K_RETRIEVAL, SIMILARITY_THRESHOLD
from .vector_store import VectorStoreManager


class PolicyRetriever:
    """
    Semantic search retriever for policy documents.
    
    Provides:
    - Top-K retrieval with configurable count
    - Similarity score filtering
    - Source citation tracking
    - Result deduplication
    """
    
    def __init__(
        self,
        vector_store_manager: Optional[VectorStoreManager] = None,
        top_k: int = TOP_K_RETRIEVAL,
        similarity_threshold: float = SIMILARITY_THRESHOLD
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store_manager: Initialized VectorStoreManager instance
            top_k: Default number of results to retrieve
            similarity_threshold: Minimum similarity score for results
        """
        self.vector_manager = vector_store_manager or VectorStoreManager()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        # Try to load existing store
        if self.vector_manager.vector_store is None:
            self.vector_manager.load_store()
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_sources: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Retrieve relevant policy documents for a query.
        
        Args:
            query: User's question or search query
            k: Number of results (overrides default)
            filter_sources: Optional list of source filenames to filter by
            
        Returns:
            List of relevant Document objects
        """
        k = k or self.top_k
        
        if self.vector_manager.vector_store is None:
            raise ValueError("Vector store not initialized. Please ingest documents first.")
        
        # Perform similarity search
        results = self.vector_manager.similarity_search(
            query=query,
            k=k * 2  # Fetch more for filtering
        )
        
        # Filter by source if specified
        if filter_sources:
            results = [
                doc for doc in results
                if doc.metadata.get("filename") in filter_sources
            ]
        
        # Deduplicate and limit results
        seen_content = set()
        unique_results = []
        
        for doc in results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)
            
            if len(unique_results) >= k:
                break
        
        return unique_results
    
    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with their similarity scores.
        
        Args:
            query: User's question
            k: Number of results
            
        Returns:
            List of (Document, score) tuples
        """
        k = k or self.top_k
        
        if self.vector_manager.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        results_with_scores = self.vector_manager.vector_store.similarity_search_with_score(
            query, k=k
        )
        
        return results_with_scores
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string for the LLM.
        
        Args:
            documents: List of retrieved Document objects
            
        Returns:
            Formatted context string with source citations
        """
        if not documents:
            return "No relevant policy documents found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("filename", "Unknown Source")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            
            context_parts.append(
                f"[Source {i}: {source}, Section {chunk_idx}]\n"
                f"{doc.page_content}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def get_source_citations(self, documents: List[Document]) -> List[dict]:
        """
        Extract source citations from documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        seen_sources = set()
        
        for doc in documents:
            source = doc.metadata.get("filename", "Unknown")
            if source not in seen_sources:
                seen_sources.add(source)
                citations.append({
                    "source": source,
                    "full_path": doc.metadata.get("source", ""),
                    "chunk_index": doc.metadata.get("chunk_index", 0)
                })
        
        return citations
    
    def get_langchain_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a LangChain-compatible retriever for use in chains.
        
        Args:
            search_kwargs: Search configuration
            
        Returns:
            LangChain Retriever object
        """
        search_kwargs = search_kwargs or {"k": self.top_k}
        return self.vector_manager.get_retriever(search_kwargs)


# Example usage
if __name__ == "__main__":
    retriever = PolicyRetriever()
    
    if retriever.vector_manager.vector_store:
        test_query = "What are the working hours?"
        results = retriever.retrieve(test_query)
        
        print(f"Query: {test_query}")
        print(f"Found {len(results)} relevant documents\n")
        
        context = retriever.format_context(results)
        print("Formatted Context:")
        print(context)
    else:
        print("No vector store found. Please ingest documents first.")
