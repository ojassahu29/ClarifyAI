"""
Document Loader Module

Handles loading and preprocessing of policy documents from various formats.
Splits documents into manageable chunks for embedding and retrieval.
"""

import os
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader
)
from langchain_core.documents import Document

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CHUNK_SIZE, CHUNK_OVERLAP, POLICIES_DIR


class DocumentLoader:
    """
    Loads and processes policy documents from various file formats.
    
    Supports:
    - PDF files (.pdf)
    - Markdown files (.md)
    - Text files (.txt)
    
    Documents are split into chunks suitable for embedding and retrieval.
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        Initialize the document loader.
        
        Args:
            chunk_size: Target size of each text chunk in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter with settings optimized for policy documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_single_document(self, file_path: str) -> List[Document]:
        """
        Load a single document from file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Select appropriate loader based on file extension
        extension = file_path.suffix.lower()
        
        if extension == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif extension == ".md":
            # Use TextLoader for markdown to avoid dependency issues
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif extension == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            # Default to text loader for unknown extensions
            loader = TextLoader(str(file_path), encoding="utf-8")
        
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata["source"] = str(file_path)
            doc.metadata["filename"] = file_path.name
        
        return documents
    
    def load_directory(
        self,
        directory_path: Optional[str] = None,
        glob_pattern: str = "**/*.*"
    ) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory (defaults to POLICIES_DIR)
            glob_pattern: Pattern to match files
            
        Returns:
            List of Document objects from all files
        """
        if directory_path is None:
            directory_path = str(POLICIES_DIR)
        
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        all_documents = []
        
        # Load different file types
        for extension, loader_cls in [
            (".pdf", PyPDFLoader),
            (".md", TextLoader),
            (".txt", TextLoader)
        ]:
            pattern = f"**/*{extension}"
            files = list(directory.glob(pattern))
            
            for file_path in files:
                try:
                    docs = self.load_single_document(str(file_path))
                    all_documents.extend(docs)
                    print(f"✓ Loaded: {file_path.name}")
                except Exception as e:
                    print(f"✗ Error loading {file_path.name}: {e}")
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for embedding.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects
        """
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def load_and_split(
        self,
        source: Optional[str] = None,
        is_directory: bool = True
    ) -> List[Document]:
        """
        Convenience method to load and split documents in one step.
        
        Args:
            source: File or directory path (defaults to POLICIES_DIR)
            is_directory: Whether source is a directory
            
        Returns:
            List of chunked Document objects ready for embedding
        """
        if is_directory:
            documents = self.load_directory(source)
        else:
            documents = self.load_single_document(source)
        
        return self.split_documents(documents)
    
    def get_document_stats(self, documents: List[Document]) -> dict:
        """
        Get statistics about loaded documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Dictionary with document statistics
        """
        total_chars = sum(len(doc.page_content) for doc in documents)
        sources = set(doc.metadata.get("source", "unknown") for doc in documents)
        
        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "average_chunk_size": total_chars // len(documents) if documents else 0,
            "unique_sources": len(sources),
            "sources": list(sources)
        }


# Example usage
if __name__ == "__main__":
    loader = DocumentLoader()
    
    # Check if policies directory has documents
    policies_dir = POLICIES_DIR
    
    if any(policies_dir.glob("*.*")):
        chunks = loader.load_and_split()
        stats = loader.get_document_stats(chunks)
        print(f"\nDocument Statistics:")
        print(f"  Total chunks: {stats['total_documents']}")
        print(f"  Total characters: {stats['total_characters']}")
        print(f"  Sources: {stats['sources']}")
    else:
        print(f"No documents found in {policies_dir}")
        print("Please add policy documents (PDF, MD, or TXT) to this directory.")
