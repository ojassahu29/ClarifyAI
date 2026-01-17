"""
Configuration module for RAG Policy Assistant.
Loads environment variables and provides centralized settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
POLICIES_DIR = DATA_DIR / "policies"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Ensure directories exist
POLICIES_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# API KEYS
# =============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Groq Model Settings (Llama 3.3 70B is very capable and fast)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Embedding Model (Free HuggingFace SentenceBERT)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# =============================================================================
# RAG SETTINGS
# =============================================================================

# Document chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Retrieval settings
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))

# =============================================================================
# LLM PARAMETERS
# =============================================================================

# Default temperature for general queries
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.3"))

# Low temperature for sensitive/compliance topics
SENSITIVE_TEMPERATURE = float(os.getenv("SENSITIVE_TEMPERATURE", "0.0"))

# Top-p for nucleus sampling
TOP_P = float(os.getenv("TOP_P", "0.9"))

# Maximum tokens for response
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# =============================================================================
# MEMORY SETTINGS
# =============================================================================

# Maximum conversation turns to keep in memory
MAX_MEMORY_TURNS = int(os.getenv("MAX_MEMORY_TURNS", "10"))

# =============================================================================
# SENSITIVE TOPICS CONFIGURATION
# =============================================================================

SENSITIVE_KEYWORDS = [
    "legal", "lawsuit", "litigation", "attorney", "lawyer",
    "termination", "firing", "dismissal", "disciplinary",
    "harassment", "discrimination", "complaint",
    "salary", "compensation", "confidential",
    "safety", "accident", "injury", "violation",
    "compliance", "audit", "investigation"
]

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SYSTEM_PROMPT = """You are a helpful Policy Assistant that answers questions based on company policies and documents.

IMPORTANT GUIDELINES:
1. Answer ONLY based on the provided policy context
2. If the information is not in the context, say "I couldn't find this information in the available policies"
3. Always cite which policy section your answer comes from
4. Be clear and concise in your responses
5. If there's ambiguity in the policies, acknowledge it"""

SENSITIVE_SYSTEM_PROMPT = """You are a Policy Assistant providing information on sensitive topics.

CRITICAL GUIDELINES:
1. You are NOT a lawyer, HR manager, or compliance officer
2. Provide general guidance ONLY, not definitive advice
3. Always recommend consulting with appropriate professionals
4. Use cautious language: "typically", "generally", "as stated in the policy"
5. Include disclaimers where appropriate
6. Do NOT make promises or guarantees about outcomes
7. If unsure, acknowledge uncertainty clearly"""

CONFLICT_CHECK_PROMPT = """Analyze the following policy excerpts and the generated answer for any conflicts or ambiguities:

Policy Excerpts:
{excerpts}

Generated Answer:
{answer}

Questions to check:
1. Do any excerpts contradict each other?
2. Is the answer fully supported by the excerpts?
3. Are there any ambiguous terms that could be interpreted differently?

Respond with a JSON object:
{{"has_conflict": true/false, "confidence": 0.0-1.0, "issues": ["list of issues if any"]}}"""
