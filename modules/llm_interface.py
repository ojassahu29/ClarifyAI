"""
LLM Interface Module

Wrapper for Groq's LLM using LangChain's ChatGroq.
Provides configurable temperature and prompting for different contexts.
"""

from pathlib import Path
from typing import Optional, List

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    DEFAULT_TEMPERATURE,
    SENSITIVE_TEMPERATURE,
    MAX_TOKENS,
    SYSTEM_PROMPT,
    SENSITIVE_SYSTEM_PROMPT
)


class GroqLLM:
    """
    Interface to Groq's LLM via LangChain.
    
    Provides:
    - Configurable temperature for different query types
    - System prompt management
    - Response generation with context
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = GROQ_MODEL,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        """
        Initialize the Groq LLM interface.
        
        Args:
            api_key: Groq API key (defaults to env variable)
            model: Groq model name
            temperature: Default temperature for generation
        """
        self.api_key = api_key or GROQ_API_KEY
        self.model = model
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Please set it in your .env file or pass it directly."
            )
        
        self.llm = self._create_llm(temperature)
        self.sensitive_llm = self._create_llm(SENSITIVE_TEMPERATURE)
    
    def _create_llm(self, temperature: float) -> ChatGroq:
        """Create a ChatGroq instance with specified temperature."""
        return ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model,
            temperature=temperature,
            max_tokens=MAX_TOKENS
        )
    
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        chat_history: Optional[List[dict]] = None,
        sensitive: bool = False
    ) -> str:
        """
        Generate a response using the Groq model.
        
        Args:
            query: User's question
            context: Retrieved policy context
            system_prompt: Custom system prompt (optional)
            chat_history: Previous conversation messages
            sensitive: Use low temperature for sensitive topics
            
        Returns:
            Generated response string
        """
        # Select appropriate LLM and system prompt
        llm = self.sensitive_llm if sensitive else self.llm
        default_prompt = SENSITIVE_SYSTEM_PROMPT if sensitive else SYSTEM_PROMPT
        system_prompt = system_prompt or default_prompt
        
        # Build messages
        messages = [SystemMessage(content=system_prompt)]
        
        # Add chat history if provided
        if chat_history:
            for msg in chat_history[-6:]:  # Keep last 6 messages
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
        
        # Build the main prompt with context
        user_prompt = f"""Based on the following policy documents, please answer the user's question.

POLICY CONTEXT:
{context}

USER QUESTION:
{query}

Please provide a clear, accurate answer based only on the information in the policy context above. If the answer is not found in the provided context, clearly state that."""

        messages.append(HumanMessage(content=user_prompt))
        
        # Generate response
        try:
            response = llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def analyze_conflicts(self, excerpts: str, answer: str) -> dict:
        """
        Analyze generated answer for conflicts with source excerpts.
        
        Args:
            excerpts: Source policy excerpts
            answer: Generated answer to verify
            
        Returns:
            Dictionary with conflict analysis
        """
        from config import CONFLICT_CHECK_PROMPT
        import json
        
        prompt = CONFLICT_CHECK_PROMPT.format(
            excerpts=excerpts,
            answer=answer
        )
        
        messages = [
            SystemMessage(content="You are a careful fact-checker. Respond only with valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.sensitive_llm.invoke(messages)
            # Try to parse JSON response
            result = json.loads(response.content)
            return result
        except json.JSONDecodeError:
            return {
                "has_conflict": False,
                "confidence": 0.5,
                "issues": ["Could not parse conflict analysis"],
                "raw_response": response.content
            }
        except Exception as e:
            return {
                "has_conflict": False,
                "confidence": 0.0,
                "issues": [str(e)]
            }
    
    def get_llm(self, sensitive: bool = False) -> ChatGroq:
        """
        Get the underlying LangChain LLM for use in chains.
        
        Args:
            sensitive: Whether to return low-temperature LLM
            
        Returns:
            ChatGroq instance
        """
        return self.sensitive_llm if sensitive else self.llm


# Alias for backward compatibility
GrokLLM = GroqLLM
GeminiLLM = GroqLLM


# Example usage
if __name__ == "__main__":
    try:
        llm = GroqLLM()
        
        test_context = """
        Remote Work Policy:
        Employees may work remotely up to 3 days per week with manager approval.
        Core hours are 10 AM to 3 PM when all employees should be available.
        """
        
        test_query = "How many days can I work from home?"
        
        response = llm.generate(test_query, test_context)
        print(f"Query: {test_query}")
        print(f"Response: {response}")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set your GROQ_API_KEY in the .env file")
