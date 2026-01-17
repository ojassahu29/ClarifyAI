"""
Basic Integration Tests for RAG Policy Assistant

Tests core functionality of the RAG pipeline.
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDocumentLoader:
    """Tests for document loading functionality."""
    
    def test_import(self):
        """Test that DocumentLoader can be imported."""
        from modules.document_loader import DocumentLoader
        loader = DocumentLoader()
        assert loader is not None
    
    def test_chunk_settings(self):
        """Test that chunk settings are applied."""
        from modules.document_loader import DocumentLoader
        loader = DocumentLoader(chunk_size=200, chunk_overlap=30)
        assert loader.chunk_size == 200
        assert loader.chunk_overlap == 30


class TestVectorStore:
    """Tests for vector store functionality."""
    
    def test_import(self):
        """Test that VectorStoreManager can be imported."""
        from modules.vector_store import VectorStoreManager
        # Note: This will load the embedding model, which takes time
        # In CI, you might want to skip or mock this
        pass
    
    def test_embedding_model_name(self):
        """Test that correct embedding model is configured."""
        from config import EMBEDDING_MODEL
        assert "MiniLM" in EMBEDDING_MODEL or "minilm" in EMBEDDING_MODEL.lower()


class TestRetriever:
    """Tests for retriever functionality."""
    
    def test_import(self):
        """Test that PolicyRetriever can be imported."""
        from modules.retriever import PolicyRetriever
        assert PolicyRetriever is not None


class TestSensitiveHandler:
    """Tests for sensitive topic handling."""
    
    def test_import(self):
        """Test that SensitiveTopicHandler can be imported."""
        from modules.sensitive_handler import SensitiveTopicHandler
        handler = SensitiveTopicHandler()
        assert handler is not None
    
    def test_detect_legal_topic(self):
        """Test detection of legal-related queries."""
        from modules.sensitive_handler import SensitiveTopicHandler
        handler = SensitiveTopicHandler()
        
        is_sensitive, categories, _ = handler.detect_sensitivity(
            "Can I sue the company?"
        )
        
        assert is_sensitive is True
        assert "legal" in categories
    
    def test_detect_normal_topic(self):
        """Test that normal queries are not flagged as sensitive."""
        from modules.sensitive_handler import SensitiveTopicHandler
        handler = SensitiveTopicHandler()
        
        is_sensitive, categories, _ = handler.detect_sensitivity(
            "What time is lunch?"
        )
        
        assert is_sensitive is False
        assert len(categories) == 0
    
    def test_disclaimer_legal(self):
        """Test that legal disclaimer is returned for legal topics."""
        from modules.sensitive_handler import SensitiveTopicHandler
        handler = SensitiveTopicHandler()
        
        disclaimer = handler.get_disclaimer(["legal"])
        assert "legal advice" in disclaimer.lower() or "attorney" in disclaimer.lower()


class TestConflictDetector:
    """Tests for conflict detection."""
    
    def test_import(self):
        """Test that ConflictDetector can be imported."""
        from modules.conflict_detector import ConflictDetector
        detector = ConflictDetector()
        assert detector is not None
    
    def test_detect_uncertainties(self):
        """Test uncertainty marker detection."""
        from modules.conflict_detector import ConflictDetector
        detector = ConflictDetector()
        
        text = "Employees may work from home depending on approval."
        uncertainties = detector.detect_uncertainties(text)
        
        assert len(uncertainties) > 0


class TestMemory:
    """Tests for conversation memory."""
    
    def test_import(self):
        """Test that ConversationMemory can be imported."""
        from modules.memory import ConversationMemory
        memory = ConversationMemory()
        assert memory is not None
    
    def test_add_messages(self):
        """Test adding messages to memory."""
        from modules.memory import ConversationMemory
        memory = ConversationMemory()
        
        memory.add_user_message("Test question")
        memory.add_assistant_message("Test answer")
        
        history = memory.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    def test_memory_limit(self):
        """Test that memory respects turn limit."""
        from modules.memory import ConversationMemory
        memory = ConversationMemory(max_turns=2)
        
        # Add 4 turns
        for i in range(4):
            memory.add_user_message(f"Question {i}")
            memory.add_assistant_message(f"Answer {i}")
        
        history = memory.get_history()
        # Should only have last 2 turns = 4 messages
        assert len(history) == 4


class TestConfig:
    """Tests for configuration."""
    
    def test_paths_exist(self):
        """Test that configured paths can be created."""
        from config import POLICIES_DIR, VECTOR_STORE_DIR
        
        assert POLICIES_DIR.exists() or POLICIES_DIR.parent.exists()
    
    def test_prompts_defined(self):
        """Test that system prompts are defined."""
        from config import SYSTEM_PROMPT, SENSITIVE_SYSTEM_PROMPT
        
        assert len(SYSTEM_PROMPT) > 50
        assert len(SENSITIVE_SYSTEM_PROMPT) > 50


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
