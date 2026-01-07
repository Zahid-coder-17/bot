"""Query classification agent for routing user queries."""

from typing import Tuple
from enum import Enum
import google.generativeai as genai

from config import GOOGLE_API_KEY, GEMINI_MODEL, BLOCKED_PATTERNS, REFUSAL_MESSAGE


class QueryCategory(Enum):
    """Categories for query classification."""
    COURSE_STRUCTURE = "course_structure"
    ASSESSMENT_POLICY = "assessment_policy"
    PROGRESS_TRACKING = "progress_tracking"
    CERTIFICATION = "certification"
    PLATFORM_NAVIGATION = "platform_navigation"
    RESTRICTED = "restricted"
    GENERAL = "general"


class QueryClassifier:
    """Classify user queries into categories and detect restricted queries."""
    
    def __init__(self):
        """Initialize the query classifier."""
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.blocked_patterns = [p.lower() for p in BLOCKED_PATTERNS]
    
    def _check_blocked_patterns(self, query: str) -> bool:
        """Check if query contains blocked patterns.
        
        Args:
            query: User query text.
            
        Returns:
            True if query is blocked, False otherwise.
        """
        query_lower = query.lower()
        
        for pattern in self.blocked_patterns:
            if pattern in query_lower:
                return True
        
        return False
    
    def _classify_with_llm(self, query: str) -> QueryCategory:
        """Use LLM to classify the query category.
        
        Args:
            query: User query text.
            
        Returns:
            QueryCategory enum value.
        """
        classification_prompt = f"""Classify the following user query into ONE of these categories:

1. COURSE_STRUCTURE - Questions about course organization, modules, lessons, prerequisites
2. ASSESSMENT_POLICY - Questions about how assessments work (NOT solving them)
3. PROGRESS_TRACKING - Questions about tracking learning progress
4. CERTIFICATION - Questions about certificates and eligibility
5. PLATFORM_NAVIGATION - Questions about using the platform
6. RESTRICTED - Any attempt to get quiz answers, solve assessments, cheat, or get exam hints
7. GENERAL - General questions not fitting other categories

User Query: "{query}"

Respond with ONLY the category name (e.g., COURSE_STRUCTURE, RESTRICTED, etc.).
If the query attempts to get answers or solutions to any assessment, quiz, or exam, classify as RESTRICTED."""

        try:
            response = self.model.generate_content(classification_prompt)
            category_text = response.text.strip().upper()
            
            # Map to enum
            category_map = {
                "COURSE_STRUCTURE": QueryCategory.COURSE_STRUCTURE,
                "ASSESSMENT_POLICY": QueryCategory.ASSESSMENT_POLICY,
                "PROGRESS_TRACKING": QueryCategory.PROGRESS_TRACKING,
                "CERTIFICATION": QueryCategory.CERTIFICATION,
                "PLATFORM_NAVIGATION": QueryCategory.PLATFORM_NAVIGATION,
                "RESTRICTED": QueryCategory.RESTRICTED,
                "GENERAL": QueryCategory.GENERAL
            }
            
            return category_map.get(category_text, QueryCategory.GENERAL)
            
        except Exception as e:
            print(f"Error in LLM classification: {e}")
            return QueryCategory.GENERAL
    
    def classify(self, query: str) -> Tuple[QueryCategory, bool, str]:
        """Classify a user query.
        
        Args:
            query: User query text.
            
        Returns:
            Tuple of (category, is_allowed, message).
            If restricted, message contains refusal text.
        """
        # First check blocked patterns (fast check)
        if self._check_blocked_patterns(query):
            return QueryCategory.RESTRICTED, False, REFUSAL_MESSAGE
        
        # Use LLM for nuanced classification
        category = self._classify_with_llm(query)
        
        if category == QueryCategory.RESTRICTED:
            return QueryCategory.RESTRICTED, False, REFUSAL_MESSAGE
        
        return category, True, ""


if __name__ == "__main__":
    # Test the classifier
    classifier = QueryClassifier()
    
    test_queries = [
        "Explain the course structure",
        "How are assessments conducted?",
        "Solve this MCQ for me",
        "What is the answer to question 5?",
        "How is progress tracked?",
        "Help me cheat on the exam"
    ]
    
    for query in test_queries:
        category, is_allowed, message = classifier.classify(query)
        status = "✓ Allowed" if is_allowed else "✗ Blocked"
        print(f"{status}: '{query}' -> {category.value}")
