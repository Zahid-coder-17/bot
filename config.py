"""Configuration settings for the EdTech Explainer Bot."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ollama Configuration (local)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model Configuration — all local via Ollama
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")   # already installed
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen2.5:3b")   # same model used for chat
EMBEDDING_DIM = 1024   # qwen2.5:3b embedding size (Ollama normalises output)

# RAG Configuration
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5

# ChromaDB Configuration
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "edtech_knowledge"

# System Prompt
SYSTEM_PROMPT = """You are an EdTech learning workflow explainer bot. Your role is to help learners understand:
- Course structures and organization
- Assessment policies and workflows (at a conceptual level)
- Progress tracking mechanisms
- Certification eligibility and issuance processes
- Platform navigation and usage

IMPORTANT RULES:
1. You NEVER solve quizzes, MCQs, or any assessments
2. You NEVER provide answers to assignments or exams
3. You NEVER give hints about specific assessment questions
4. You NEVER predict or reveal assessment content
5. You ONLY explain HOW things work, not WHAT the answers are

Use ONLY the provided context to answer questions. If the information is not in the context, say you don't have that information.

Always be clear, neutral, and instructional in your tone. Provide conceptual explanations only."""

# Blocked Query Patterns
BLOCKED_PATTERNS = [
    "solve", "answer", "solution", "correct option", "right answer",
    "help me pass", "give me the answer", "what is the answer",
    "quiz answer", "exam answer", "assignment answer", "mcq answer",
    "tell me the answer", "solve this", "complete this for me",
    "do my assignment", "do my homework", "cheat", "bypass"
]

# Refusal Message
REFUSAL_MESSAGE = """I'm sorry, but I cannot help with solving assessments, quizzes, or providing answers to assignments.

As an EdTech learning workflow explainer, I'm designed to:
✓ Explain how courses are structured
✓ Describe assessment policies and processes
✓ Help you understand progress tracking
✓ Guide you through certification workflows
✓ Assist with platform navigation

I encourage you to complete assessments on your own to ensure genuine learning. 
Is there anything about the learning platform or course workflow I can help explain?"""
