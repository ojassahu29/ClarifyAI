"""
HuggingFace Dataset Loader Module

Loads HR policy datasets from HuggingFace for use as ground truth.
Supports multiple HR/policy datasets.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import json

from langchain_core.documents import Document

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import POLICIES_DIR


# Available HR datasets on HuggingFace
AVAILABLE_DATASETS = {
    "syncora/hr-policies-qa": {
        "name": "syncora/hr-policies-qa-dataset",
        "description": "Synthetic HR policy Q&A conversations for chatbot training",
        "content_field": "messages",  # Field containing the content
        "type": "conversation"
    },
    "mawared-hr": {
        "name": "MawaredHR/Fullredv5",
        "description": "HR management system Q&A (leave, performance, etc.)",
        "content_field": "question",
        "answer_field": "answer",
        "type": "qa_pairs"
    }
}


class HuggingFaceDatasetLoader:
    """
    Loads HR policy datasets from HuggingFace Hub.
    
    Converts datasets into LangChain Documents for the RAG pipeline.
    """
    
    def __init__(self):
        """Initialize the HuggingFace dataset loader."""
        if not HF_AVAILABLE:
            raise ImportError(
                "Please install the datasets library: pip install datasets"
            )
    
    def list_available_datasets(self) -> Dict[str, str]:
        """List available pre-configured HR datasets."""
        return {
            key: info["description"] 
            for key, info in AVAILABLE_DATASETS.items()
        }
    
    def load_dataset(
        self,
        dataset_key: str = "syncora/hr-policies-qa",
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> List[Document]:
        """
        Load a HuggingFace dataset and convert to Documents.
        
        Args:
            dataset_key: Key from AVAILABLE_DATASETS or full HF dataset name
            split: Dataset split to load
            max_samples: Maximum number of samples to load
            
        Returns:
            List of LangChain Document objects
        """
        # Get dataset config
        if dataset_key in AVAILABLE_DATASETS:
            config = AVAILABLE_DATASETS[dataset_key]
            dataset_name = config["name"]
        else:
            # Assume it's a direct dataset name
            dataset_name = dataset_key
            config = {"type": "generic", "content_field": "text"}
        
        print(f"Loading dataset: {dataset_name}...")
        
        try:
            dataset = load_dataset(dataset_name, split=split)
            print(f"✓ Loaded {len(dataset)} samples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
        
        # Limit samples if specified
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            print(f"Limited to {max_samples} samples")
        
        # Convert to Documents based on dataset type
        documents = []
        
        if config.get("type") == "conversation":
            documents = self._convert_conversation_dataset(dataset, config)
        elif config.get("type") == "qa_pairs":
            documents = self._convert_qa_dataset(dataset, config)
        else:
            documents = self._convert_generic_dataset(dataset, config)
        
        print(f"✓ Created {len(documents)} documents")
        return documents
    
    def _convert_conversation_dataset(
        self,
        dataset,
        config: Dict
    ) -> List[Document]:
        """Convert conversation-style dataset to Documents."""
        documents = []
        content_field = config.get("content_field", "messages")
        
        for idx, sample in enumerate(dataset):
            try:
                messages = sample.get(content_field, [])
                
                # Extract text from messages
                if isinstance(messages, list):
                    text_parts = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")
                            text_parts.append(f"{role.upper()}: {content}")
                        elif isinstance(msg, str):
                            text_parts.append(msg)
                    
                    text = "\n\n".join(text_parts)
                else:
                    text = str(messages)
                
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": f"hf:{config.get('name', 'unknown')}",
                            "sample_idx": idx,
                            "type": "hr_conversation"
                        }
                    ))
            except Exception as e:
                print(f"  Warning: Skipping sample {idx}: {e}")
        
        return documents
    
    def _convert_qa_dataset(
        self,
        dataset,
        config: Dict
    ) -> List[Document]:
        """Convert Q&A pair dataset to Documents."""
        documents = []
        q_field = config.get("content_field", "question")
        a_field = config.get("answer_field", "answer")
        
        for idx, sample in enumerate(dataset):
            try:
                question = sample.get(q_field, "")
                answer = sample.get(a_field, "")
                
                # Combine Q&A into a document
                text = f"Question: {question}\n\nAnswer: {answer}"
                
                if text.strip() and len(text) > 20:
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": f"hf:{config.get('name', 'unknown')}",
                            "sample_idx": idx,
                            "type": "hr_qa_pair",
                            "question": question[:100]
                        }
                    ))
            except Exception as e:
                print(f"  Warning: Skipping sample {idx}: {e}")
        
        return documents
    
    def _convert_generic_dataset(
        self,
        dataset,
        config: Dict
    ) -> List[Document]:
        """Convert generic text dataset to Documents."""
        documents = []
        content_field = config.get("content_field", "text")
        
        for idx, sample in enumerate(dataset):
            try:
                # Try to get content from the specified field
                if content_field in sample:
                    text = str(sample[content_field])
                else:
                    # Fallback: combine all text fields
                    text_parts = []
                    for key, value in sample.items():
                        if isinstance(value, str) and len(value) > 10:
                            text_parts.append(f"{key}: {value}")
                    text = "\n".join(text_parts)
                
                if text.strip() and len(text) > 20:
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": f"hf:{config.get('name', 'unknown')}",
                            "sample_idx": idx,
                            "type": "hr_policy"
                        }
                    ))
            except Exception as e:
                print(f"  Warning: Skipping sample {idx}: {e}")
        
        return documents
    
    def save_as_markdown(
        self,
        documents: List[Document],
        output_path: Optional[str] = None
    ) -> str:
        """
        Save documents as a Markdown file for local use.
        
        Args:
            documents: List of Document objects
            output_path: Path to save the file
            
        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_path = POLICIES_DIR / "huggingface_hr_policies.md"
        else:
            output_path = Path(output_path)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# HR Policies Dataset (HuggingFace)\n\n")
            f.write("This document contains HR policy information loaded from HuggingFace.\n\n")
            f.write("---\n\n")
            
            for i, doc in enumerate(documents):
                f.write(f"## Entry {i+1}\n\n")
                f.write(doc.page_content)
                f.write("\n\n---\n\n")
        
        print(f"✓ Saved {len(documents)} entries to {output_path}")
        return str(output_path)


def download_hr_dataset(
    dataset_key: str = "syncora/hr-policies-qa",
    max_samples: int = 100,
    save_locally: bool = True
) -> List[Document]:
    """
    Convenience function to download and prepare HR dataset.
    
    Args:
        dataset_key: HuggingFace dataset identifier
        max_samples: Maximum samples to download
        save_locally: Whether to save as local Markdown file
        
    Returns:
        List of Document objects
    """
    loader = HuggingFaceDatasetLoader()
    
    documents = loader.load_dataset(
        dataset_key=dataset_key,
        max_samples=max_samples
    )
    
    if save_locally and documents:
        loader.save_as_markdown(documents)
    
    return documents


# Example usage
if __name__ == "__main__":
    print("HuggingFace HR Dataset Loader")
    print("=" * 40)
    
    try:
        loader = HuggingFaceDatasetLoader()
        
        print("\nAvailable datasets:")
        for key, desc in loader.list_available_datasets().items():
            print(f"  - {key}: {desc}")
        
        print("\nLoading syncora/hr-policies-qa dataset...")
        documents = download_hr_dataset(
            dataset_key="syncora/hr-policies-qa",
            max_samples=50,
            save_locally=True
        )
        
        if documents:
            print(f"\nSample document content:")
            print("-" * 40)
            print(documents[0].page_content[:500])
            print("...")
            
    except ImportError as e:
        print(f"\n⚠️ {e}")
        print("Run: pip install datasets")
