"""
Conflict Detector Module

Identifies contradictions, ambiguities, and inconsistencies
in policy documents and generated responses.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

from langchain.schema import Document

import sys
sys.path.append(str(Path(__file__).parent.parent))


class ConflictDetector:
    """
    Detects conflicts and ambiguities in policy documents.
    
    Features:
    - Cross-document contradiction detection
    - Ambiguous term identification
    - Confidence scoring for responses
    - Uncertainty flagging
    """
    
    # Words/phrases indicating uncertainty or conflict
    UNCERTAINTY_MARKERS = [
        "may", "might", "could", "possibly", "potentially",
        "sometimes", "occasionally", "depending on", "subject to",
        "at the discretion of", "case by case", "varies"
    ]
    
    # Contradictory phrase patterns
    CONTRADICTION_PATTERNS = [
        (r"must\s+(?!not)", r"optional|may|can choose"),
        (r"always", r"never|sometimes|rarely"),
        (r"required", r"optional|voluntary"),
        (r"prohibited", r"allowed|permitted"),
        (r"all employees", r"some employees|certain employees|only"),
    ]
    
    def __init__(self, llm=None):
        """
        Initialize conflict detector.
        
        Args:
            llm: Optional LLM for advanced conflict analysis
        """
        self.llm = llm
    
    def detect_uncertainties(self, text: str) -> List[str]:
        """
        Find uncertainty markers in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of found uncertainty phrases
        """
        text_lower = text.lower()
        found = []
        
        for marker in self.UNCERTAINTY_MARKERS:
            if marker in text_lower:
                # Extract surrounding context
                idx = text_lower.find(marker)
                start = max(0, idx - 20)
                end = min(len(text), idx + len(marker) + 30)
                found.append(text[start:end].strip())
        
        return found
    
    def check_document_conflict(
        self,
        documents: List[Document]
    ) -> Dict[str, any]:
        """
        Check for conflicts between multiple document chunks.
        
        Args:
            documents: List of Document objects to compare
            
        Returns:
            Dictionary with conflict analysis results
        """
        if len(documents) < 2:
            return {"has_conflict": False, "conflicts": [], "confidence": 1.0}
        
        conflicts = []
        
        # Compare each pair of documents
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents[i+1:], i+1):
                text1 = doc1.page_content.lower()
                text2 = doc2.page_content.lower()
                
                # Check for contradictory patterns
                for pattern1, pattern2 in self.CONTRADICTION_PATTERNS:
                    if re.search(pattern1, text1) and re.search(pattern2, text2):
                        conflicts.append({
                            "type": "potential_contradiction",
                            "source1": doc1.metadata.get("filename", f"doc_{i}"),
                            "source2": doc2.metadata.get("filename", f"doc_{j}"),
                            "pattern": f"{pattern1} vs {pattern2}"
                        })
                    elif re.search(pattern2, text1) and re.search(pattern1, text2):
                        conflicts.append({
                            "type": "potential_contradiction",
                            "source1": doc1.metadata.get("filename", f"doc_{i}"),
                            "source2": doc2.metadata.get("filename", f"doc_{j}"),
                            "pattern": f"{pattern2} vs {pattern1}"
                        })
        
        # Calculate confidence (lower if conflicts found)
        confidence = max(0.3, 1.0 - (len(conflicts) * 0.2))
        
        return {
            "has_conflict": len(conflicts) > 0,
            "conflicts": conflicts,
            "confidence": confidence
        }
    
    def analyze_answer_consistency(
        self,
        answer: str,
        source_documents: List[Document]
    ) -> Dict[str, any]:
        """
        Check if an answer is consistent with source documents.
        
        Args:
            answer: Generated answer to verify
            source_documents: Documents used to generate the answer
            
        Returns:
            Consistency analysis results
        """
        # Combine source text
        source_text = " ".join(doc.page_content for doc in source_documents)
        source_lower = source_text.lower()
        answer_lower = answer.lower()
        
        issues = []
        
        # Check for absolute statements not supported by sources
        absolute_patterns = [
            r"\balways\b", r"\bnever\b", r"\ball employees\b",
            r"\beveryone\b", r"\bno one\b", r"\bmust\b"
        ]
        
        for pattern in absolute_patterns:
            if re.search(pattern, answer_lower):
                # Check if same pattern exists in sources
                if not re.search(pattern, source_lower):
                    match = re.search(pattern, answer_lower)
                    issues.append({
                        "type": "unsupported_absolute",
                        "phrase": match.group() if match else pattern,
                        "severity": "medium"
                    })
        
        # Check for uncertainty markers in answer
        uncertainties = self.detect_uncertainties(answer)
        
        return {
            "is_consistent": len(issues) == 0,
            "issues": issues,
            "uncertainties": uncertainties,
            "confidence": max(0.5, 1.0 - (len(issues) * 0.15))
        }
    
    def get_conflict_warning(
        self,
        conflict_result: Dict
    ) -> Optional[str]:
        """
        Generate a user-friendly warning message for conflicts.
        
        Args:
            conflict_result: Result from conflict detection
            
        Returns:
            Warning message or None if no conflicts
        """
        if not conflict_result.get("has_conflict"):
            return None
        
        conflicts = conflict_result.get("conflicts", [])
        
        if not conflicts:
            return None
        
        warnings = ["⚠️ **Policy Conflict Detected**"]
        
        for conflict in conflicts[:3]:  # Limit to 3 warnings
            source1 = conflict.get("source1", "Document 1")
            source2 = conflict.get("source2", "Document 2")
            warnings.append(
                f"- Possible inconsistency between {source1} and {source2}"
            )
        
        warnings.append(
            "\n*Please verify with your HR department for clarification.*"
        )
        
        return "\n".join(warnings)
    
    def calculate_response_confidence(
        self,
        answer: str,
        documents: List[Document],
        document_conflict: Optional[Dict] = None
    ) -> Tuple[float, List[str]]:
        """
        Calculate overall confidence score for a response.
        
        Args:
            answer: Generated answer
            documents: Source documents
            document_conflict: Pre-computed conflict analysis
            
        Returns:
            Tuple of (confidence score, list of factors)
        """
        confidence = 1.0
        factors = []
        
        # Factor 1: Number of supporting documents
        if len(documents) == 0:
            confidence -= 0.5
            factors.append("No supporting documents found")
        elif len(documents) == 1:
            confidence -= 0.1
            factors.append("Only one source document")
        
        # Factor 2: Document conflicts
        if document_conflict is None:
            document_conflict = self.check_document_conflict(documents)
        
        if document_conflict.get("has_conflict"):
            confidence -= 0.2
            factors.append("Potential conflicts between sources")
        
        # Factor 3: Answer consistency
        consistency = self.analyze_answer_consistency(answer, documents)
        if not consistency.get("is_consistent"):
            confidence -= 0.15
            factors.append("Answer may contain unsupported claims")
        
        # Factor 4: Uncertainty in answer
        if consistency.get("uncertainties"):
            factors.append("Answer contains uncertain language (which may be appropriate)")
        
        return max(0.0, min(1.0, confidence)), factors


# Example usage
if __name__ == "__main__":
    detector = ConflictDetector()
    
    # Test uncertainty detection
    test_text = "Employees may work from home depending on manager approval."
    uncertainties = detector.detect_uncertainties(test_text)
    print("Uncertainties found:", uncertainties)
    
    # Test with mock documents
    from langchain.schema import Document
    
    docs = [
        Document(
            page_content="All employees must attend weekly meetings.",
            metadata={"filename": "policy_a.md"}
        ),
        Document(
            page_content="Team meetings are optional for remote workers.",
            metadata={"filename": "policy_b.md"}
        )
    ]
    
    conflicts = detector.check_document_conflict(docs)
    print("\nConflict analysis:", conflicts)
    
    warning = detector.get_conflict_warning(conflicts)
    if warning:
        print("\n" + warning)
