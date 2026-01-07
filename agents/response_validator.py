"""Response validation agent for final safety checks."""

from typing import Tuple, List
import re


class ResponseValidator:
    """Validate LLM responses for prohibited content."""
    
    def __init__(self):
        """Initialize the response validator."""
        # Patterns that indicate answer-giving
        self.prohibited_patterns = [
            r"\bthe answer is\b",
            r"\bcorrect answer\b",
            r"\bthe solution is\b",
            r"\boption [A-D] is correct\b",
            r"\bchoose option\b",
            r"\bselect answer\b",
            r"\bthe right answer\b",
            r"\banswer: [A-D]\b",
            r"\b[A-D]\) is correct\b",
            r"\bstep 1.*step 2.*step 3\b",  # Step-by-step solutions
            r"\bhere's how to solve\b",
            r"\bto solve this problem\b",
            r"\bthe formula is\b.*=.*\b",  # Mathematical solutions
            r"\byour grade is\b",
            r"\byour score would be\b",
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.prohibited_patterns
        ]
        
        # Replacement message for sanitized responses
        self.sanitized_suffix = (
            "\n\n*Note: I can only provide general explanations about how the "
            "learning platform works. I cannot provide specific answers or "
            "solutions to assessments.*"
        )
    
    def _find_violations(self, response: str) -> List[str]:
        """Find all pattern violations in a response.
        
        Args:
            response: LLM response text.
            
        Returns:
            List of matched violation patterns.
        """
        violations = []
        
        for pattern in self.compiled_patterns:
            if pattern.search(response):
                violations.append(pattern.pattern)
        
        return violations
    
    def validate(self, response: str) -> Tuple[bool, str, List[str]]:
        """Validate an LLM response.
        
        Args:
            response: LLM response text.
            
        Returns:
            Tuple of (is_safe, validated_response, violations).
        """
        violations = self._find_violations(response)
        
        if not violations:
            return True, response, []
        
        # Response has violations - sanitize it
        sanitized = self._sanitize_response(response, violations)
        
        return False, sanitized, violations
    
    def _sanitize_response(self, response: str, violations: List[str]) -> str:
        """Sanitize a response with violations.
        
        Args:
            response: Original response.
            violations: List of violation patterns found.
            
        Returns:
            Sanitized response or replacement.
        """
        # For severe violations, replace entirely
        severe_indicators = ["answer is", "correct answer", "solution is", "option"]
        
        has_severe = any(
            indicator in v.lower() 
            for v in violations 
            for indicator in severe_indicators
        )
        
        if has_severe:
            return (
                "I understand you're looking for information, but I can only "
                "explain how the learning platform works. I cannot provide "
                "answers or solutions to assessments.\n\n"
                "Would you like me to explain how assessments are structured "
                "or how the grading policy works instead?"
            )
        
        # For minor violations, add disclaimer
        return response + self.sanitized_suffix
    
    def get_safe_response(self, response: str) -> str:
        """Get a safe version of a response (convenience method).
        
        Args:
            response: LLM response text.
            
        Returns:
            Safe response string.
        """
        _, validated_response, _ = self.validate(response)
        return validated_response


if __name__ == "__main__":
    # Test the validator
    validator = ResponseValidator()
    
    test_responses = [
        "Courses are structured in modules and lessons.",
        "The answer is A. You should select option A.",
        "Progress is tracked by completed lessons and quiz scores.",
        "Here's how to solve the problem: Step 1, Step 2, Step 3"
    ]
    
    for response in test_responses:
        is_safe, validated, violations = validator.validate(response)
        status = "✓ Safe" if is_safe else "✗ Violated"
        print(f"{status}: {response[:50]}...")
        if violations:
            print(f"  Violations: {violations}")
