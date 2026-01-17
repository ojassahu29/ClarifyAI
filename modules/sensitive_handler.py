"""
Sensitive Topic Handler Module

Handles queries about sensitive topics (legal, HR, safety)
with appropriate caution, disclaimers, and conservative responses.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import re

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SENSITIVE_KEYWORDS, SENSITIVE_SYSTEM_PROMPT


class SensitiveTopicHandler:
    """
    Handles sensitive topic detection and safe response formatting.
    
    Features:
    - Topic classification (legal, HR, safety, financial)
    - Automatic sensitivity detection
    - Disclaimer injection
    - Conservative response enforcement
    """
    
    # Topic categories with associated keywords
    TOPIC_CATEGORIES = {
        "legal": [
            "legal", "lawsuit", "litigation", "attorney", "lawyer",
            "sue", "court", "law", "regulation", "statute", "liability"
        ],
        "hr_disciplinary": [
            "termination", "firing", "dismissal", "disciplinary",
            "warning", "probation", "suspension", "performance review",
            "misconduct", "violation", "grievance"
        ],
        "harassment_discrimination": [
            "harassment", "discrimination", "hostile work environment",
            "complaint", "title vii", "eeoc", "retaliation", "whistleblower"
        ],
        "compensation": [
            "salary", "wage", "compensation", "bonus", "raise",
            "pay", "overtime", "benefits", "severance", "stock options"
        ],
        "safety_compliance": [
            "safety", "accident", "injury", "osha", "hazard",
            "compliance", "audit", "investigation", "violation"
        ],
        "confidential": [
            "confidential", "private", "secret", "proprietary",
            "nda", "non-disclosure", "trade secret"
        ],
        "medical": [
            "medical", "health", "disability", "fmla", "leave",
            "accommodation", "ada", "illness", "mental health"
        ]
    }
    
    # Standard disclaimers by category
    DISCLAIMERS = {
        "legal": "⚖️ **Disclaimer:** This information is for general guidance only and does not constitute legal advice. Please consult with a qualified attorney for specific legal questions.",
        "hr_disciplinary": "📋 **Note:** For specific questions about disciplinary procedures, please contact your HR representative directly.",
        "harassment_discrimination": "🛡️ **Important:** If you believe you are experiencing harassment or discrimination, please report it to HR or your company's designated reporting channel immediately.",
        "compensation": "💰 **Note:** Compensation details may vary by role, location, and individual circumstances. Please verify with HR for your specific situation.",
        "safety_compliance": "🔒 **Safety Notice:** If this is an emergency situation, please contact emergency services immediately. Report safety hazards through proper channels.",
        "medical": "🏥 **Health Notice:** This information is for reference only. Please consult with HR and appropriate medical professionals for specific accommodation requests.",
        "default": "ℹ️ **Note:** This is general policy guidance. For specific situations, please consult with the appropriate department."
    }
    
    def __init__(self, custom_keywords: Optional[List[str]] = None):
        """
        Initialize the sensitive topic handler.
        
        Args:
            custom_keywords: Additional keywords to consider sensitive
        """
        self.sensitive_keywords = set(SENSITIVE_KEYWORDS)
        if custom_keywords:
            self.sensitive_keywords.update(custom_keywords)
    
    def detect_sensitivity(self, query: str) -> Tuple[bool, List[str], float]:
        """
        Detect if a query involves sensitive topics.
        
        Args:
            query: User's question
            
        Returns:
            Tuple of (is_sensitive, list of categories, sensitivity_score)
        """
        query_lower = query.lower()
        detected_categories = []
        keyword_matches = 0
        
        # Check each category
        for category, keywords in self.TOPIC_CATEGORIES.items():
            for keyword in keywords:
                if keyword in query_lower:
                    if category not in detected_categories:
                        detected_categories.append(category)
                    keyword_matches += 1
        
        # Check against general sensitive keywords
        for keyword in self.sensitive_keywords:
            if keyword.lower() in query_lower:
                keyword_matches += 1
        
        is_sensitive = len(detected_categories) > 0 or keyword_matches >= 2
        
        # Calculate sensitivity score (0.0 to 1.0)
        score = min(1.0, keyword_matches * 0.2 + len(detected_categories) * 0.3)
        
        return is_sensitive, detected_categories, score
    
    def get_disclaimer(self, categories: List[str]) -> str:
        """
        Get appropriate disclaimer for detected categories.
        
        Args:
            categories: List of sensitive topic categories
            
        Returns:
            Formatted disclaimer text
        """
        if not categories:
            return self.DISCLAIMERS["default"]
        
        # Use highest priority category disclaimer
        priority_order = [
            "harassment_discrimination", "legal", "safety_compliance",
            "medical", "hr_disciplinary", "compensation", "confidential"
        ]
        
        for category in priority_order:
            if category in categories:
                return self.DISCLAIMERS.get(category, self.DISCLAIMERS["default"])
        
        return self.DISCLAIMERS["default"]
    
    def format_safe_response(
        self,
        answer: str,
        categories: List[str],
        add_disclaimer: bool = True,
        add_consultation_advice: bool = True
    ) -> str:
        """
        Format a response with appropriate safety measures.
        
        Args:
            answer: Original generated answer
            categories: Detected sensitive categories
            add_disclaimer: Whether to add a disclaimer
            add_consultation_advice: Whether to add advice to consult experts
            
        Returns:
            Safely formatted response
        """
        parts = [answer]
        
        # Add consultation advice
        if add_consultation_advice and categories:
            consultation_text = self._get_consultation_advice(categories)
            if consultation_text:
                parts.append(consultation_text)
        
        # Add disclaimer at the end
        if add_disclaimer and categories:
            disclaimer = self.get_disclaimer(categories)
            parts.append(disclaimer)
        
        return "\n\n".join(parts)
    
    def _get_consultation_advice(self, categories: List[str]) -> str:
        """Generate consultation advice based on categories."""
        experts = []
        
        if "legal" in categories:
            experts.append("a legal professional")
        if any(cat in categories for cat in ["hr_disciplinary", "harassment_discrimination", "compensation"]):
            experts.append("your HR representative")
        if "safety_compliance" in categories:
            experts.append("your safety officer")
        if "medical" in categories:
            experts.append("HR and appropriate medical professionals")
        
        if experts:
            return f"**For your specific situation, we recommend consulting with {', or '.join(experts)}.**"
        
        return ""
    
    def sanitize_query(self, query: str) -> str:
        """
        Remove or flag potentially harmful content from queries.
        
        Args:
            query: Original user query
            
        Returns:
            Sanitized query
        """
        # Remove potential prompt injection attempts
        injection_patterns = [
            r"ignore previous instructions",
            r"disregard.*rules",
            r"pretend you are",
            r"act as if",
            r"forget.*told"
        ]
        
        sanitized = query
        for pattern in injection_patterns:
            sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def get_safe_system_prompt(
        self,
        categories: List[str],
        base_prompt: Optional[str] = None
    ) -> str:
        """
        Get an appropriate system prompt for sensitive topics.
        
        Args:
            categories: Detected sensitive categories
            base_prompt: Optional base prompt to extend
            
        Returns:
            Enhanced system prompt for safe responses
        """
        base = base_prompt or SENSITIVE_SYSTEM_PROMPT
        
        category_specific = []
        
        if "legal" in categories:
            category_specific.append(
                "- Never provide specific legal advice or interpretations of law"
            )
        if "harassment_discrimination" in categories:
            category_specific.append(
                "- Be supportive but do not make judgments about specific situations"
            )
        if "compensation" in categories:
            category_specific.append(
                "- Do not make promises about compensation or benefits entitlements"
            )
        
        if category_specific:
            additions = "\n\nADDITIONAL GUIDELINES FOR THIS QUERY:\n" + "\n".join(category_specific)
            return base + additions
        
        return base


# Example usage
if __name__ == "__main__":
    handler = SensitiveTopicHandler()
    
    # Test queries
    test_queries = [
        "What is the remote work policy?",
        "Can I sue the company for discrimination?",
        "What is the process for filing a harassment complaint?",
        "How do I request a salary review?",
        "What should I do if I'm injured at work?"
    ]
    
    for query in test_queries:
        is_sensitive, categories, score = handler.detect_sensitivity(query)
        print(f"\nQuery: {query}")
        print(f"  Sensitive: {is_sensitive}")
        print(f"  Categories: {categories}")
        print(f"  Score: {score:.2f}")
        
        if is_sensitive:
            disclaimer = handler.get_disclaimer(categories)
            print(f"  Disclaimer: {disclaimer[:50]}...")
