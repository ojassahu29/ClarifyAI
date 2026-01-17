"""
Conversation Memory Module

Manages short-term and long-term conversation memory.
Enables context-aware follow-up questions and user preference tracking.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MAX_MEMORY_TURNS, DATA_DIR


class ConversationMemory:
    """
    Manages conversation history for context-aware responses.
    
    Features:
    - Short-term memory: Current session history
    - Long-term memory: Persistent storage across sessions
    - User preference tracking
    - Memory window management to avoid context overflow
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        max_turns: int = MAX_MEMORY_TURNS,
        persist_path: Optional[str] = None
    ):
        """
        Initialize conversation memory.
        
        Args:
            session_id: Unique identifier for this conversation
            max_turns: Maximum turns to keep in memory
            persist_path: Path for persistent storage
        """
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.max_turns = max_turns
        self.persist_dir = Path(persist_path or DATA_DIR / "memory")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Short-term memory (current session)
        self.messages: List[Dict[str, Any]] = []
        
        # User preferences and metadata
        self.preferences: Dict[str, Any] = {
            "verbosity": "normal",  # "brief", "normal", "detailed"
            "expertise_level": "general",  # "general", "technical", "expert"
            "topics_of_interest": []
        }
        
        # Feedback tracking for learning
        self.feedback: List[Dict[str, Any]] = []
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a message to conversation history.
        
        Args:
            role: "user" or "assistant"
            content: Message content
            metadata: Additional metadata (sources, confidence, etc.)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        
        # Trim to max turns (2 messages per turn)
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to history."""
        self.add_message("user", content)
    
    def add_assistant_message(
        self,
        content: str,
        sources: Optional[List[Dict]] = None,
        confidence: Optional[float] = None
    ) -> None:
        """Add an assistant message with optional metadata."""
        metadata = {}
        if sources:
            metadata["sources"] = sources
        if confidence is not None:
            metadata["confidence"] = confidence
        
        self.add_message("assistant", content, metadata)
    
    def get_history(self, n_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Get conversation history formatted for LLM context.
        
        Args:
            n_turns: Number of turns to return (default: all)
            
        Returns:
            List of message dictionaries with role and content
        """
        n_messages = (n_turns * 2) if n_turns else len(self.messages)
        
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.messages[-n_messages:]
        ]
    
    def get_context_summary(self) -> str:
        """
        Get a brief summary of conversation context.
        
        Returns:
            String summary of recent topics discussed
        """
        if not self.messages:
            return "This is the start of our conversation."
        
        user_messages = [
            msg["content"][:100] for msg in self.messages
            if msg["role"] == "user"
        ][-3:]
        
        if not user_messages:
            return "No recent questions."
        
        return "Recent topics: " + " | ".join(user_messages)
    
    def add_feedback(
        self,
        message_index: int,
        is_helpful: bool,
        correction: Optional[str] = None
    ) -> None:
        """
        Record user feedback on a response.
        
        Args:
            message_index: Index of the message being rated
            is_helpful: Whether the response was helpful
            correction: Optional correction text
        """
        feedback_entry = {
            "message_index": message_index,
            "is_helpful": is_helpful,
            "correction": correction,
            "timestamp": datetime.now().isoformat()
        }
        
        self.feedback.append(feedback_entry)
        
        # Update message metadata
        if 0 <= message_index < len(self.messages):
            self.messages[message_index]["metadata"]["feedback"] = is_helpful
    
    def update_preference(self, key: str, value: Any) -> None:
        """Update a user preference setting."""
        self.preferences[key] = value
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference value."""
        return self.preferences.get(key, default)
    
    def save(self) -> None:
        """Persist memory to disk."""
        data = {
            "session_id": self.session_id,
            "messages": self.messages,
            "preferences": self.preferences,
            "feedback": self.feedback
        }
        
        file_path = self.persist_dir / f"{self.session_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, session_id: str) -> bool:
        """
        Load memory from a previous session.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            True if loaded successfully
        """
        file_path = self.persist_dir / f"{session_id}.json"
        
        if not file_path.exists():
            return False
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self.session_id = data.get("session_id", session_id)
            self.messages = data.get("messages", [])
            self.preferences = data.get("preferences", self.preferences)
            self.feedback = data.get("feedback", [])
            return True
            
        except Exception as e:
            print(f"Error loading memory: {e}")
            return False
    
    def clear(self) -> None:
        """Clear current session memory."""
        self.messages = []
        self.feedback = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "session_id": self.session_id,
            "total_messages": len(self.messages),
            "total_turns": len(self.messages) // 2,
            "positive_feedback": sum(1 for f in self.feedback if f.get("is_helpful")),
            "negative_feedback": sum(1 for f in self.feedback if not f.get("is_helpful")),
            "preferences": self.preferences
        }


# Example usage
if __name__ == "__main__":
    memory = ConversationMemory()
    
    # Simulate a conversation
    memory.add_user_message("What is the vacation policy?")
    memory.add_assistant_message(
        "Employees receive 15 days of paid vacation per year.",
        sources=[{"source": "hr_policy.pdf"}],
        confidence=0.9
    )
    
    memory.add_user_message("Can I carry over unused days?")
    memory.add_assistant_message(
        "Yes, you can carry over up to 5 unused days to the next year.",
        sources=[{"source": "hr_policy.pdf"}],
        confidence=0.85
    )
    
    # Get history for LLM
    history = memory.get_history()
    print("Conversation History:")
    for msg in history:
        print(f"  {msg['role']}: {msg['content'][:50]}...")
    
    # Save session
    memory.save()
    print(f"\nSaved to session: {memory.session_id}")
    
    # Show stats
    print(f"Stats: {memory.get_stats()}")
