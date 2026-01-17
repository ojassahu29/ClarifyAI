# 📋 RAG Policy Assistant

A modular Retrieval-Augmented Generation (RAG) system for answering policy-related questions using LangChain with xAI's Grok model.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## ✨ Features

- **🔍 Semantic Search**: Finds relevant policy excerpts using vector embeddings
- **🤖 Grok LLM Integration**: Generates contextual answers via xAI's Grok model
- **💬 Conversation Memory**: Maintains context across multi-turn conversations
- **⚠️ Conflict Detection**: Flags contradictions and ambiguities in policies
- **🛡️ Sensitive Topic Handling**: Adds disclaimers for legal/HR/safety topics
- **📊 Source Citations**: Shows which documents were used for each answer
- **🎨 Streamlit UI**: Clean, interactive chat interface

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (app.py)                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────┐  │
│  │   Memory    │   │  Sensitive  │   │     Conflict     │  │
│  │   Module    │   │   Handler   │   │     Detector     │  │
│  └─────────────┘   └─────────────┘   └──────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    RAG Chain                          │  │
│  │  ┌───────────┐  ┌───────────┐  ┌────────────────┐   │  │
│  │  │ Retriever │──│ LLM (Grok)│──│ Response Gen   │   │  │
│  │  └───────────┘  └───────────┘  └────────────────┘   │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                   Vector Store                        │  │
│  │  ┌──────────────┐  ┌────────────────────────────────┐ │  │
│  │  │ HuggingFace  │  │ ChromaDB / FAISS               │ │  │
│  │  │ Embeddings   │  │ (Persistent Vector Storage)     │ │  │
│  │  └──────────────┘  └────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Document Loader                          │  │
│  │  [PDF] [Markdown] [Text Files]                        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
AhaanKiTooli/
├── app.py                    # Streamlit UI entry point
├── config.py                 # Configuration and settings
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variable template
│
├── modules/
│   ├── __init__.py
│   ├── document_loader.py    # Document ingestion & chunking
│   ├── vector_store.py       # Embeddings & ChromaDB
│   ├── retriever.py          # Semantic search
│   ├── llm_interface.py      # Grok LLM wrapper
│   ├── rag_chain.py          # Complete RAG pipeline
│   ├── memory.py             # Conversation memory
│   ├── conflict_detector.py  # Ambiguity detection
│   └── sensitive_handler.py  # Sensitive topic handling
│
├── data/
│   ├── policies/             # Your policy documents (PDF, MD, TXT)
│   │   └── sample_hr_policy.md
│   ├── vector_store/         # Persisted embeddings
│   └── memory/               # Conversation history
│
└── tests/
    └── test_rag.py           # Integration tests
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd AhaanKiTooli

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file from the template:

```bash
copy .env.example .env
```

Edit `.env` and add your xAI API key:

```env
XAI_API_KEY=your_xai_api_key_here
```

### 3. Add Policy Documents

Place your policy documents (PDF, Markdown, or TXT) in the `data/policies/` folder:

```bash
# Example: copy your HR handbook
copy "path\to\hr_handbook.pdf" data\policies\
```

### 4. Run the Application

```bash
python -m streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### 5. Ingest Documents

Click the **🔄 Ingest Documents** button in the sidebar to:
1. Load all documents from `data/policies/`
2. Split them into chunks
3. Generate embeddings
4. Store in ChromaDB

### 6. Start Asking Questions!

Example queries:
- "How many vacation days do I get?"
- "What is the remote work policy?"
- "Can I carry over unused PTO days?"
- "What is the process for expense reimbursement?"

## 📚 Module Reference

### Document Loader
```python
from modules import DocumentLoader

loader = DocumentLoader(chunk_size=500, chunk_overlap=50)
chunks = loader.load_and_split("data/policies/")
```

### Vector Store
```python
from modules import VectorStoreManager

manager = VectorStoreManager(store_type="chroma")
manager.create_store(chunks)

# Later, load existing store
manager.load_store()
results = manager.similarity_search("vacation policy", k=4)
```

### RAG Chain
```python
from modules import RAGChain

rag = RAGChain()
rag.ingest_documents()  # First time only

response = rag.query("How many vacation days do I get?")
print(response.answer)
print(response.sources)
```

### Sensitive Topic Detection
```python
from modules import SensitiveTopicHandler

handler = SensitiveTopicHandler()
is_sensitive, categories, score = handler.detect_sensitivity(
    "Can I sue for discrimination?"
)
# is_sensitive: True
# categories: ['legal', 'harassment_discrimination']
```

## ⚙️ Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `GROK_MODEL` | `grok-2-latest` | Grok model version |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `4` | Documents to retrieve |
| `DEFAULT_TEMPERATURE` | `0.3` | LLM temperature |
| `SENSITIVE_TEMPERATURE` | `0.0` | Temperature for sensitive topics |

## 🧪 Running Tests

```bash
# Install pytest
pip install pytest

# Run all tests
python -m pytest tests/ -v
```

## 🔒 Security Considerations

1. **API Keys**: Never commit `.env` files. The `.env.example` is provided as a template.
2. **Sensitive Data**: Policy documents may contain confidential information. Ensure appropriate access controls.
3. **Disclaimers**: The system adds disclaimers for legal/HR topics but should not replace professional advice.

## 🛠️ Customization

### Adding Custom Sensitive Keywords

```python
# In config.py, modify SENSITIVE_KEYWORDS
SENSITIVE_KEYWORDS = [
    "legal", "lawsuit", "termination",
    "your_custom_keyword",  # Add custom terms
]
```

### Using FAISS Instead of ChromaDB

```python
vector_manager = VectorStoreManager(store_type="faiss")
```

### Adjusting Chunk Size for Your Documents

For detailed policies, use smaller chunks:
```env
CHUNK_SIZE=300
CHUNK_OVERLAP=75
```

For high-level summaries, use larger chunks:
```env
CHUNK_SIZE=1000
CHUNK_OVERLAP=100
```

## 📋 Task Mapping

| Task | Module |
|------|--------|
| **Task 1**: RAG Pipeline | `document_loader.py`, `vector_store.py`, `retriever.py`, `llm_interface.py`, `rag_chain.py` |
| **Task 2**: Memory & Conflicts | `memory.py`, `conflict_detector.py` |
| **Task 3**: Sensitive Topics | `sensitive_handler.py` |
| **Task 4**: Deployment | `app.py` |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is for educational and internal use. Ensure compliance with your organization's policies before deployment.

---

**Built with ❤️ using LangChain, Grok, and Streamlit**
