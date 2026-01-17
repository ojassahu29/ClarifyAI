"""
Vector Store Module

Manages embeddings and vector database for semantic search.
Uses HuggingFace Sentence-Transformers for free, local embeddings.
Supports ChromaDB and FAISS as vector stores.
"""

from pathlib import Path
from typing import List, Optional, Literal

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import EMBEDDING_MODEL, VECTOR_STORE_DIR


class VectorStoreManager:
    """
    Manages vector embeddings and storage for policy documents.
    
    Uses HuggingFace Sentence-Transformers (all-MiniLM-L6-v2) for embeddings.
    Supports ChromaDB (default) and FAISS for vector storage.
    """
    
    def __init__(
        self,
        embedding_model: str = EMBEDDING_MODEL,
        store_type: Literal["chroma", "faiss"] = "chroma",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the vector store manager.
        
        Args:
            embedding_model: HuggingFace model name for embeddings
            store_type: Type of vector store ("chroma" or "faiss")
            persist_directory: Directory to persist the vector store
        """
        self.embedding_model = embedding_model
        self.store_type = store_type
        self.persist_directory = Path(persist_directory or VECTOR_STORE_DIR)
        
        # Initialize embeddings
        print(f"Loading embedding model: {embedding_model}...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("✓ Embedding model loaded")
        
        self.vector_store = None
    
    def create_store(self, documents: List[Document]) -> None:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to embed and store
        """
        if not documents:
            raise ValueError("No documents provided to create vector store")
        
        print(f"Creating {self.store_type} vector store with {len(documents)} documents...")
        
        if self.store_type == "chroma":
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory / "chroma")
            )
        elif self.store_type == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # Persist FAISS index
            faiss_path = self.persist_directory / "faiss"
            faiss_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(faiss_path))
        else:
            raise ValueError(f"Unsupported store type: {self.store_type}")
        
        print(f"✓ Vector store created and persisted to {self.persist_directory}")
    
    def load_store(self) -> bool:
        """
        Load an existing vector store from disk.
        
        Returns:
            True if store was loaded successfully, False otherwise
        """
        try:
            if self.store_type == "chroma":
                chroma_path = self.persist_directory / "chroma"
                if chroma_path.exists():
                    self.vector_store = Chroma(
                        persist_directory=str(chroma_path),
                        embedding_function=self.embeddings
                    )
                    print(f"✓ Loaded Chroma store from {chroma_path}")
                    return True
                    
            elif self.store_type == "faiss":
                faiss_path = self.persist_directory / "faiss"
                if faiss_path.exists():
                    self.vector_store = FAISS.load_local(
                        str(faiss_path),
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    print(f"✓ Loaded FAISS store from {faiss_path}")
                    return True
            
            print(f"No existing {self.store_type} store found")
            return False
            
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add new documents to an existing vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Create or load a store first.")
        
        self.vector_store.add_documents(documents)
        
        # Persist changes
        if self.store_type == "chroma":
            pass  # Chroma auto-persists
        elif self.store_type == "faiss":
            faiss_path = self.persist_directory / "faiss"
            self.vector_store.save_local(str(faiss_path))
        
        print(f"✓ Added {len(documents)} documents to vector store")
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query string
            k: Number of results to return
            score_threshold: Minimum similarity score (optional)
            
        Returns:
            List of most similar Document objects
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Create or load a store first.")
        
        if score_threshold is not None:
            # Use similarity search with score filtering
            results_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            results = [
                doc for doc, score in results_with_scores
                if score >= score_threshold
            ]
        else:
            results = self.vector_store.similarity_search(query, k=k)
        
        return results
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever interface for the vector store.
        
        Args:
            search_kwargs: Additional search parameters
            
        Returns:
            Retriever object compatible with LangChain chains
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Create or load a store first.")
        
        search_kwargs = search_kwargs or {"k": 4}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def delete_store(self) -> None:
        """Delete the persisted vector store."""
        import shutil
        
        if self.store_type == "chroma":
            chroma_path = self.persist_directory / "chroma"
            if chroma_path.exists():
                shutil.rmtree(chroma_path)
                print(f"✓ Deleted Chroma store at {chroma_path}")
                
        elif self.store_type == "faiss":
            faiss_path = self.persist_directory / "faiss"
            if faiss_path.exists():
                shutil.rmtree(faiss_path)
                print(f"✓ Deleted FAISS store at {faiss_path}")
        
        self.vector_store = None
    
    def get_store_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with store statistics
        """
        if self.vector_store is None:
            return {"status": "not_initialized"}
        
        # Get collection count for Chroma
        if self.store_type == "chroma":
            try:
                collection = self.vector_store._collection
                count = collection.count()
                return {
                    "store_type": "chroma",
                    "document_count": count,
                    "embedding_model": self.embedding_model,
                    "persist_directory": str(self.persist_directory)
                }
            except Exception:
                pass
        
        return {
            "store_type": self.store_type,
            "embedding_model": self.embedding_model,
            "persist_directory": str(self.persist_directory),
            "status": "initialized"
        }


# Example usage
if __name__ == "__main__":
    from document_loader import DocumentLoader
    
    # Initialize components
    loader = DocumentLoader()
    vector_manager = VectorStoreManager()
    
    # Try to load existing store
    if not vector_manager.load_store():
        print("No existing store found. Please add documents first.")
    else:
        # Test similarity search
        test_query = "What is the remote work policy?"
        results = vector_manager.similarity_search(test_query, k=3)
        print(f"\nSearch results for: '{test_query}'")
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(doc.page_content[:200] + "...")
