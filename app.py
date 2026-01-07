"""
EdTech Platform Course & Learning Workflow Explainer Bot

A Streamlit-based AI assistant that helps learners understand course structures,
assessments, progress tracking, and certification workflows.
"""

import streamlit as st
import requests
from typing import Optional

from config import OPENROUTER_API_KEY, OPENROUTER_MODEL, REFUSAL_MESSAGE
from agents.query_classifier import QueryClassifier, QueryCategory
from agents.knowledge_retriever import KnowledgeRetriever
from agents.guardrails import PromptGuardrails
from agents.response_validator import ResponseValidator


# Page configuration
st.set_page_config(
    page_title="EdTech Learning Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium dark theme with glassmorphism
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d0d1a 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        background: linear-gradient(90deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.2rem;
        margin: 0;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.7);
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 15, 35, 0.95) 0%, rgba(26, 26, 62, 0.95) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] .stMarkdown h2 {
        color: #a5b4fc;
        font-weight: 600;
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(10px) !important;
        margin: 0.5rem 0 !important;
    }
    
    /* User message */
    [data-testid="stChatMessageContent"] {
        color: #e2e8f0 !important;
    }
    
    /* Input Box */
    .stChatInput {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
    }
    
    .stChatInput input {
        color: #f1f5f9 !important;
    }
    
    /* Info Cards */
    .info-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.05) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .info-card h4 {
        color: #a5b4fc;
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    .info-card p {
        color: rgba(255, 255, 255, 0.7);
        margin: 0;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        color: #86efac;
    }
    
    /* Warning Box */
    .warning-box {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box h4 {
        color: #fca5a5;
        margin: 0 0 0.5rem 0;
    }
    
    .warning-box p {
        color: rgba(255, 255, 255, 0.7);
        margin: 0;
        font-size: 0.85rem;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


class EdTechBot:
    """Main EdTech explainer bot combining all agents."""
    
    # Fallback responses when API fails
    FALLBACK_RESPONSES = {
        "course_structure": """## 📚 Course Structure Overview

Our platform organizes learning content in a clear hierarchy:

**🎯 Programs** → Complete learning paths for career goals
**📘 Courses** → Comprehensive units (4-12 weeks each)
**📑 Modules** → Thematic units within courses
**📝 Lessons** → Specific concepts (15-45 min each)

### Learning Content Types:
- Video lectures (5-20 minutes)
- Interactive tutorials
- Reading assignments
- Hands-on labs
- Knowledge checks

### Navigation:
- **Dashboard**: Overview of enrolled courses and progress
- **Course Page**: Module outline, progress %, next lessons
- **Learning Path**: Recommended course sequence

*Prerequisites are checked automatically when enrolling!*""",

        "assessment": """## 📝 Assessment Policy

### Assessment Types:
| Type | Weight | Attempts | Time Limit |
|------|--------|----------|------------|
| Quizzes | 20% | 3 attempts | 30 min |
| Assignments | 30% | Before deadline | Varies |
| Projects | 25% | Resubmission allowed | 2 weeks |
| Final Exam | 25% | 1 attempt | 2 hours |

### Grading Scale:
- **A**: 90-100% | **B**: 80-89% | **C**: 70-79%
- **D**: 60-69% | **F**: Below 60%

### Key Policies:
✅ Multiple attempts for quizzes (highest score counts)
✅ 72-hour grace period for assignments
✅ Late submissions: -10% per day
❌ Academic integrity strictly enforced

*Passing requires 60% minimum overall score!*""",

        "progress": """## 📊 Progress Tracking

### How Progress is Calculated:
- **Content Completion**: Videos watched, readings completed
- **Assessment Scores**: Weighted average of all assessments
- **Time Spent**: Active learning time tracked

### Dashboard Features:
📈 **Progress Bar**: Visual completion percentage
📅 **Learning Streak**: Consecutive days of activity
🎯 **Module Status**: Not Started → In Progress → Completed
⏰ **Estimated Time**: Time remaining to complete

### Progress Example:
```
Course Progress: ████████░░ 78%
├── Module 1: ✅ Complete (95%)
├── Module 2: ✅ Complete (88%)
├── Module 3: 🔄 In Progress (65%)
└── Module 4: ⬜ Not Started
```

*Your progress syncs across all devices automatically!*""",

        "certification": """## 🏆 Certification Process

### Certificate Types:
1. **Course Completion** - Single course (60% minimum)
2. **Professional Certificate** - Full program (70% GPA)
3. **Specialization** - Skill track completion
4. **Verified Certificate** - With ID verification

### Eligibility Requirements:
✅ Complete all required modules
✅ Achieve minimum passing score (60%)
✅ Submit all mandatory assignments
✅ No academic integrity violations

### Issuance Process:
1. **Complete** → Finish all requirements
2. **Verify** → System checks scores & submissions
3. **Generate** → Certificate created with unique ID
4. **Deliver** → Download PDF, LinkedIn integration

### Timeline:
- Standard: Immediate (within 24 hours)
- Verified: 3-5 business days
- Professional: 5-7 business days

*Certificates include QR codes for employer verification!*""",

        "general": """## 🎓 EdTech Learning Assistant

I'm here to help you understand our learning platform! I can assist with:

### 📚 Course Structure
Understanding modules, lessons, and learning paths

### 📝 Assessments
How quizzes, assignments, and exams work

### 📊 Progress Tracking
How your learning progress is measured

### 🏆 Certification
Requirements and process for earning certificates

**Just ask me a question about any of these topics!**

*Note: I cannot help with solving quizzes or providing exam answers.*"""
    }
    
    def __init__(self):
        """Initialize the bot with all agents."""
        self.query_classifier = QueryClassifier()
        self.knowledge_retriever = KnowledgeRetriever()
        self.guardrails = PromptGuardrails()
        self.response_validator = ResponseValidator()
        
        # Initialize OpenRouter
        self.openrouter_api_key = OPENROUTER_API_KEY
        self.openrouter_model = OPENROUTER_MODEL
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
    
    def _get_fallback_response(self, query: str) -> str:
        """Get intelligent fallback response based on query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["course", "structure", "module", "lesson", "program", "navigate", "dashboard"]):
            return self.FALLBACK_RESPONSES["course_structure"]
        elif any(word in query_lower for word in ["assess", "quiz", "exam", "test", "grade", "score", "assignment", "attempt"]):
            return self.FALLBACK_RESPONSES["assessment"]
        elif any(word in query_lower for word in ["progress", "track", "complete", "percentage", "status"]):
            return self.FALLBACK_RESPONSES["progress"]
        elif any(word in query_lower for word in ["certif", "certificate", "credential", "badge", "verify"]):
            return self.FALLBACK_RESPONSES["certification"]
        else:
            return self.FALLBACK_RESPONSES["general"]
    
    def generate_response(
        self, 
        user_query: str,
        conversation_history: Optional[str] = None
    ) -> tuple[str, QueryCategory, bool]:
        """Generate a response for a user query."""
        # Step 1: Classify the query
        category, is_allowed, refusal_msg = self.query_classifier.classify(user_query)
        
        if not is_allowed:
            return refusal_msg, category, True
        
        # Step 2: Retrieve relevant knowledge
        context = self.knowledge_retriever.get_context_for_query(user_query)
        
        # Step 3: Build safe prompt with guardrails
        safe_prompt = self.guardrails.build_safe_prompt(
            user_query=user_query,
            context=context,
            conversation_history=conversation_history
        )
        
        # Step 4: Try OpenRouter API, fallback on error
        try:
            if not self.openrouter_api_key:
                return self._get_fallback_response(user_query), category, False
            
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "EdTech Learning Assistant"
            }
            
            payload = {
                "model": self.openrouter_model,
                "messages": [
                    {"role": "user", "content": safe_prompt}
                ]
            }
            
            response = requests.post(
                self.openrouter_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            raw_response = result["choices"][0]["message"]["content"]
            
            # Step 5: Validate response
            is_safe, validated_response, violations = self.response_validator.validate(raw_response)
            return validated_response, category, False
            
        except Exception as e:
            # On ANY error (including quota), use fallback
            return self._get_fallback_response(user_query), category, False


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "bot" not in st.session_state:
        st.session_state.bot = EdTechBot()


def render_sidebar():
    """Render the sidebar with information and examples."""
    with st.sidebar:
        st.markdown("## 🎓 EdTech Assistant")
        
        st.markdown("""
        <div class="status-badge">
            <span>●</span> Online
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ✅ I Can Help With")
        st.markdown("""
        <div class="info-card">
            <h4>📚 Course Structure</h4>
            <p>Understanding modules, lessons, and learning paths</p>
        </div>
        <div class="info-card">
            <h4>📝 Assessment Policy</h4>
            <p>How assessments work, grading policies, attempt rules</p>
        </div>
        <div class="info-card">
            <h4>📊 Progress Tracking</h4>
            <p>How progress is measured and displayed</p>
        </div>
        <div class="info-card">
            <h4>🏆 Certification</h4>
            <p>Certificate eligibility and issuance process</p>
        </div>
        <div class="info-card">
            <h4>🧭 Platform Navigation</h4>
            <p>How to use various platform features</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ❌ I Cannot Help With")
        st.markdown("""
        <div class="warning-box">
            <h4>Academic Integrity Protected</h4>
            <p>• Solving quizzes or MCQs<br>
            • Answering assignments<br>
            • Providing exam hints<br>
            • Sharing assessment answers</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 💡 Try Asking")
        example_queries = [
            "Explain the course structure",
            "How are assessments conducted?",
            "How is progress tracked?",
            "What is the certification process?"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{query}", use_container_width=True):
                st.session_state.example_query = query
        
        st.markdown("---")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎓 EdTech Learning Assistant</h1>
        <p>Your guide to understanding course workflows, assessments, and certifications</p>
    </div>
    """, unsafe_allow_html=True)


def format_conversation_history() -> Optional[str]:
    """Format conversation history for context."""
    if not st.session_state.messages:
        return None
    
    recent = st.session_state.messages[-6:]
    history_parts = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_parts.append(f"{role}: {msg['content']}")
    
    return "\n".join(history_parts)


def main():
    """Main application function."""
    init_session_state()
    
    render_header()
    render_sidebar()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if "example_query" in st.session_state:
        prompt = st.session_state.example_query
        del st.session_state.example_query
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()
    
    if prompt := st.chat_input("Ask about course workflows, assessments, or certifications..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                history = format_conversation_history()
                response, category, was_blocked = st.session_state.bot.generate_response(
                    user_query=prompt,
                    conversation_history=history
                )
                
                if was_blocked:
                    st.warning("⚠️ This query was blocked as it appears to request assessment solutions.")
                
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
