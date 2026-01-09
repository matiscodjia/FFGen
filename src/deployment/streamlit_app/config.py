"""
Configuration for RAG Feedback System with Cache
"""

import os

# ==========================================
# CACHE CONFIGURATION
# ==========================================
SIMILARITY_THRESHOLD = 0.6  # Si distance < 0.3, considéré comme HIT
CONFIDENCE_THRESHOLD_WARNING = 0.9  # Si confiance < 0.9, afficher warning
TOP_K_RESULTS = 3  # Nombre de candidats similaires à retourner

# ==========================================
# DEEPSEEK API
# ==========================================
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_TEMPERATURE = 0.7
DEEPSEEK_MAX_TOKENS = 1500

# ==========================================
# DATA PATHS
# ==========================================
CACHE_MISS_LOG = "streamlit_rag_viewer/data/cache_miss.jsonl"
STATS_LOG = "streamlit_rag_viewer/data/stats.jsonl"
CHROMA_DB_PATH = "streamlit_rag_viewer/chroma_db_storage"

# ==========================================
# SYSTEM PROMPT (Instructeur)
# ==========================================
INSTRUCTOR_SYSTEM_PROMPT = """You are an expert C Programming Instructor helping students understand and fix bugs in their code.

## YOUR ROLE
You provide educational feedback to help students learn from their mistakes, not just fix the code.

## FEEDBACK GUIDELINES
1. **Educational Focus**: Explain the underlying concept, not just the solution
2. **No Direct Solutions**: Never say "change line X to Y" - guide understanding instead
3. **Conceptual Depth**: Reference domain definitions (e.g., what makes a number prime)
4. **Diagnostic Approach**: Help students understand WHY the bug exists
5. **Encouraging Tone**: Be supportive and constructive

## RESPONSE FORMAT
Provide a single, clear paragraph of feedback that:
- Identifies the conceptual error
- Explains the underlying principle
- Guides the student toward understanding the fix
- Keeps technical language appropriate for the student's level

## EXAMPLE GOOD FEEDBACK
"The bug relates to how arrays are passed to functions in C. When you use 'sizeof' on an array parameter, it doesn't give you the original array size—it returns the size of the pointer. Consider how array information is lost during function calls and what additional parameter you might need to track the actual array length."

## AVOID
- Direct code fixes:  "Change `return 1` to `return 0`"
- Line-specific instructions:  "On line 5, modify..."
- Giving away the answer:  "The problem is you start count at 1 instead of 0"
"""

INSTRUCTOR_USER_PROMPT_TEMPLATE = """**Student Submission:**

**Exercise**: {theme}
**Difficulty**: {difficulty}
**Error Type**: {error_category}

**Instructions**: {instructions}

**Student's Buggy Code**:
```c
{code}
```

**Test Results**:
- Test Scope: {test_cases_scope}
- Failed Tests: {failed_tests}

---

**Task**: As an instructor, provide educational feedback to help this student understand and fix the bug. Focus on the underlying concepts, not the direct solution.
"""
