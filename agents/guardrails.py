"""Prompt guardrails agent for ensuring safe prompt construction."""

from typing import Optional

from config import SYSTEM_PROMPT


class PromptGuardrails:
    """Apply guardrails to prompts before sending to LLM."""
    
    def __init__(self):
        """Initialize the prompt guardrails."""
        self.system_prompt = SYSTEM_PROMPT
        
        # Additional safety instructions
        self.safety_suffix = """

CRITICAL SAFETY REMINDERS:
- Do NOT provide any answers to quizzes, MCQs, or assessments
- Do NOT give step-by-step solutions to problems
- Do NOT provide personalized grading feedback
- Do NOT reveal or hint at assessment content
- If unsure whether a response might help with cheating, err on the side of caution and decline
- Only explain HOW things work, never WHAT the answers are

Respond based ONLY on the provided context. If the information is not available, say so."""

    def build_safe_prompt(
        self, 
        user_query: str, 
        context: str,
        conversation_history: Optional[str] = None
    ) -> str:
        """Build a safe prompt with all guardrails applied.
        
        Args:
            user_query: The user's query.
            context: Retrieved knowledge context.
            conversation_history: Optional previous conversation.
            
        Returns:
            Safe prompt string for the LLM.
        """
        prompt_parts = [
            self.system_prompt,
            self.safety_suffix,
            "\n---\nKNOWLEDGE BASE CONTEXT:",
            context,
            "\n---"
        ]
        
        if conversation_history:
            prompt_parts.extend([
                "\nPREVIOUS CONVERSATION:",
                conversation_history,
                "\n---"
            ])
        
        prompt_parts.extend([
            "\nUSER QUERY:",
            user_query,
            "\n\nProvide a helpful, informational response based on the context. Remember: explain concepts and processes, never provide assessment answers."
        ])
        
        return "\n".join(prompt_parts)
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for chat initialization.
        
        Returns:
            System prompt string.
        """
        return self.system_prompt + self.safety_suffix


if __name__ == "__main__":
    # Test the guardrails
    guardrails = PromptGuardrails()
    
    test_query = "How are module quizzes structured?"
    test_context = "Module quizzes contain 10-20 questions and require a 70% passing score."
    
    safe_prompt = guardrails.build_safe_prompt(test_query, test_context)
    
    print("Safe Prompt Preview (first 1000 chars):")
    print(safe_prompt[:1000])
