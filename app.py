"""
RAG Policy Assistant - Streamlit Application

Interactive chat interface for querying company policies
using retrieval-augmented generation with Grok LLM.
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import POLICIES_DIR, GROQ_API_KEY
from modules import (
    DocumentLoader,
    VectorStoreManager,
    PolicyRetriever,
    RAGChain,
    ConversationMemory,
    ConflictDetector,
    SensitiveTopicHandler
)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="ClarifyAI",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 3px solid #1E88E5;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()
    
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    if "sensitive_handler" not in st.session_state:
        st.session_state.sensitive_handler = SensitiveTopicHandler()
    
    if "conflict_detector" not in st.session_state:
        st.session_state.conflict_detector = ConflictDetector()
    
    if "initialized" not in st.session_state:
        st.session_state.initialized = False


def initialize_rag():
    """Initialize RAG chain components."""
    try:
        # Check for API key
        if not GROQ_API_KEY:
            st.sidebar.error("⚠️ GROQ_API_KEY not set in .env file")
            return False
        
        # Initialize vector store
        vector_manager = VectorStoreManager()
        
        # Try to load existing store
        if not vector_manager.load_store():
            st.sidebar.warning("No vector store found. Please ingest documents first.")
            return False
        
        # Initialize RAG chain
        st.session_state.rag_chain = RAGChain(
            vector_store_manager=vector_manager,
            check_conflicts=True
        )
        
        st.session_state.initialized = True
        return True
        
    except Exception as e:
        st.sidebar.error(f"Initialization error: {str(e)}")
        return False


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar with controls and info."""
    st.sidebar.markdown("## 📋 ClarifyAI")
    st.sidebar.markdown("---")
    
    # Status section
    st.sidebar.markdown("### Status")
    
    if st.session_state.initialized:
        st.sidebar.success("✅ System Ready")
    else:
        st.sidebar.warning("⚠️ Not initialized")
    
    # Document management section
    st.sidebar.markdown("### 📁 Document Management")
    
    # Show policies directory
    policy_files = list(POLICIES_DIR.glob("**/*.*"))
    st.sidebar.markdown(f"**Documents found:** {len(policy_files)}")
    
    if policy_files:
        with st.sidebar.expander("View documents"):
            for f in policy_files[:10]:
                st.markdown(f"- {f.name}")
    
    # Ingest local documents button
    if st.sidebar.button("🔄 Ingest Local Documents", type="primary"):
        with st.spinner("Ingesting documents..."):
            try:
                loader = DocumentLoader()
                chunks = loader.load_and_split()
                
                if chunks:
                    vector_manager = VectorStoreManager()
                    vector_manager.create_store(chunks)
                    st.sidebar.success(f"✅ Ingested {len(chunks)} chunks")
                    
                    # Reinitialize RAG
                    initialize_rag()
                else:
                    st.sidebar.warning("No documents found to ingest")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    # HuggingFace Dataset Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤗 HuggingFace Dataset")
    
    hf_dataset = st.sidebar.selectbox(
        "Select HR Dataset",
        options=["syncora/hr-policies-qa", "mawared-hr"],
        index=0
    )
    
    hf_samples = st.sidebar.slider("Max samples", 20, 200, 100, 20)
    
    if st.sidebar.button("📥 Load HuggingFace Dataset"):
        with st.spinner(f"Loading {hf_dataset}..."):
            try:
                from modules import download_hr_dataset
                
                documents = download_hr_dataset(
                    dataset_key=hf_dataset,
                    max_samples=hf_samples,
                    save_locally=True
                )
                
                if documents:
                    # Create vector store from HF documents
                    vector_manager = VectorStoreManager()
                    vector_manager.create_store(documents)
                    st.sidebar.success(f"✅ Loaded {len(documents)} HR policy entries")
                    
                    # Reinitialize RAG
                    initialize_rag()
                else:
                    st.sidebar.warning("No documents loaded from HuggingFace")
            except ImportError:
                st.sidebar.error("Install datasets: pip install datasets")
            except Exception as e:
                st.sidebar.error(f"Error: {str(e)}")
    
    # Clear conversation
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()
    
    # Settings
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Settings")
    
    show_sources = st.sidebar.checkbox("Show source documents", value=True)
    show_confidence = st.sidebar.checkbox("Show confidence score", value=True)
    
    return show_sources, show_confidence



# =============================================================================
# CHAT INTERFACE
# =============================================================================

def display_message(role: str, content: str, sources: list = None, 
                   confidence: float = None, is_sensitive: bool = False,
                   show_sources: bool = True, show_confidence: bool = True):
    """Display a chat message with optional metadata."""
    
    with st.chat_message(role):
        st.markdown(content)
        
        # Show metadata for assistant messages
        if role == "assistant":
            cols = st.columns([2, 1])
            
            # Show sources
            if show_sources and sources:
                with cols[0]:
                    with st.expander(f"📄 Sources ({len(sources)})"):
                        for source in sources:
                            st.markdown(f"- {source.get('source', 'Unknown')}")
            
            # Show confidence
            if show_confidence and confidence is not None:
                with cols[1]:
                    if confidence >= 0.8:
                        color = "confidence-high"
                        icon = "✅"
                    elif confidence >= 0.5:
                        color = "confidence-medium"
                        icon = "⚠️"
                    else:
                        color = "confidence-low"
                        icon = "❌"
                    
                    st.markdown(
                        f"<span class='{color}'>{icon} Confidence: {confidence:.0%}</span>",
                        unsafe_allow_html=True
                    )


def process_query(query: str) -> dict:
    """Process a user query through the RAG pipeline."""
    
    if not st.session_state.rag_chain:
        return {
            "answer": "❌ System not initialized. Please ingest documents first.",
            "sources": [],
            "confidence": 0.0,
            "is_sensitive": False
        }
    
    # Check for sensitive topics
    sensitive_handler = st.session_state.sensitive_handler
    is_sensitive, categories, _ = sensitive_handler.detect_sensitivity(query)
    
    # Get chat history
    chat_history = st.session_state.memory.get_history(n_turns=5)
    
    # Query RAG chain
    response = st.session_state.rag_chain.query(
        question=query,
        chat_history=chat_history,
        is_sensitive=is_sensitive
    )
    
    # Format response for sensitive topics
    answer = response.answer
    if is_sensitive:
        answer = sensitive_handler.format_safe_response(
            answer,
            categories,
            add_disclaimer=True
        )
    
    return {
        "answer": answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "is_sensitive": is_sensitive,
        "categories": categories if is_sensitive else []
    }


def main():
    """Main application entry point."""
    
    # Initialize session state
    init_session_state()
    
    # Render sidebar and get settings
    show_sources, show_confidence = render_sidebar()
    
    # Initialize RAG if not done
    if not st.session_state.initialized:
        initialize_rag()
    
    # Header
    st.markdown('<p class="main-header">📋 ClarifyAI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Ask questions about company policies and procedures</p>',
        unsafe_allow_html=True
    )
    
    # Display existing messages
    for msg in st.session_state.messages:
        display_message(
            role=msg["role"],
            content=msg["content"],
            sources=msg.get("sources"),
            confidence=msg.get("confidence"),
            is_sensitive=msg.get("is_sensitive", False),
            show_sources=show_sources,
            show_confidence=show_confidence
        )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about company policies..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        st.session_state.memory.add_user_message(prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process and display response
        with st.chat_message("assistant"):
            with st.spinner("Searching policies..."):
                result = process_query(prompt)
            
            st.markdown(result["answer"])
            
            # Show metadata
            if result["sources"] and show_sources:
                with st.expander(f"📄 Sources ({len(result['sources'])})"):
                    for source in result["sources"]:
                        st.markdown(f"- {source.get('source', 'Unknown')}")
            
            if show_confidence and result["confidence"]:
                confidence = result["confidence"]
                if confidence >= 0.8:
                    st.success(f"✅ Confidence: {confidence:.0%}")
                elif confidence >= 0.5:
                    st.warning(f"⚠️ Confidence: {confidence:.0%}")
                else:
                    st.error(f"❌ Confidence: {confidence:.0%}")
        
        # Save to session
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "is_sensitive": result["is_sensitive"]
        })
        st.session_state.memory.add_assistant_message(
            result["answer"],
            sources=result["sources"],
            confidence=result["confidence"]
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<center><small>Powered by LangChain + Grok | "
        "For official guidance, please consult HR</small></center>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
