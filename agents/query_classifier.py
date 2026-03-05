"""Query classification agent using Ollama local LLM."""

import re
from typing import Tuple
from enum import Enum
import requests

from config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, BLOCKED_PATTERNS, REFUSAL_MESSAGE


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

    CATEGORY_MAP = {
        "COURSE_STRUCTURE": QueryCategory.COURSE_STRUCTURE,
        "ASSESSMENT_POLICY": QueryCategory.ASSESSMENT_POLICY,
        "PROGRESS_TRACKING": QueryCategory.PROGRESS_TRACKING,
        "CERTIFICATION": QueryCategory.CERTIFICATION,
        "PLATFORM_NAVIGATION": QueryCategory.PLATFORM_NAVIGATION,
        "RESTRICTED": QueryCategory.RESTRICTED,
        "GENERAL": QueryCategory.GENERAL,
    }

    def __init__(self):
        """Initialize the query classifier."""
        self.base_url = OLLAMA_BASE_URL
        self.model = OLLAMA_CHAT_MODEL
        self.blocked_patterns = [p.lower() for p in BLOCKED_PATTERNS]

    def _check_blocked_patterns(self, query: str) -> bool:
        """Check if query contains blocked patterns."""
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in self.blocked_patterns)

    def _classify_with_llm(self, query: str) -> QueryCategory:
        """Use local Ollama LLM to classify the query category."""
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
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": classification_prompt,
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 20},
                },
                timeout=60,
            )
            response.raise_for_status()
            raw = response.json().get("response", "").strip().upper()

            # Extract first matching category keyword from the response
            for key in self.CATEGORY_MAP:
                if key in raw:
                    return self.CATEGORY_MAP[key]

            return QueryCategory.GENERAL

        except Exception as e:
            print(f"Error in LLM classification: {e}")
            return QueryCategory.GENERAL

    def classify(self, query: str) -> Tuple[QueryCategory, bool, str]:
        """Classify a user query.

        Returns:
            Tuple of (category, is_allowed, message).
            If restricted, message contains refusal text.
        """
        # Fast pattern check first
        if self._check_blocked_patterns(query):
            return QueryCategory.RESTRICTED, False, REFUSAL_MESSAGE

        # LLM-based nuanced classification
        category = self._classify_with_llm(query)

        if category == QueryCategory.RESTRICTED:
            return QueryCategory.RESTRICTED, False, REFUSAL_MESSAGE

        return category, True, ""


if __name__ == "__main__":
    classifier = QueryClassifier()

    test_queries = [
        "Explain the course structure",
        "How are assessments conducted?",
        "Solve this MCQ for me",
        "What is the answer to question 5?",
        "How is progress tracked?",
        "Help me cheat on the exam",
    ]

    for query in test_queries:
        category, is_allowed, message = classifier.classify(query)
        status = "✓ Allowed" if is_allowed else "✗ Blocked"
        print(f"{status}: '{query}' -> {category.value}")
