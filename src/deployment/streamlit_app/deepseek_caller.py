"""
DeepSeek API Wrapper for Feedback Generation
"""

from openai import OpenAI
from typing import Dict, Any
import time
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_MAX_TOKENS,
    INSTRUCTOR_SYSTEM_PROMPT,
    INSTRUCTOR_USER_PROMPT_TEMPLATE
)

class DeepSeekCaller:
    def __init__(self):
        if not DEEPSEEK_API_KEY:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")

        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )

    def generate_feedback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Génère un feedback pédagogique pour un code bugué.

        Args:
            context: {
                "theme": str,
                "difficulty": str,
                "error_category": str,
                "instructions": str,
                "code": str,
                "test_cases_scope": list,
                "failed_tests": list
            }

        Returns:
            {
                "feedback": str,
                "tokens_prompt": int,
                "tokens_completion": int,
                "tokens_total": int,
                "generation_time_ms": float
            }
        """
        start_time = time.time()

        # Formater le prompt
        user_prompt = INSTRUCTOR_USER_PROMPT_TEMPLATE.format(
            theme=context.get('theme', 'N/A'),
            difficulty=context.get('difficulty', 'intermediate'),
            error_category=context.get('error_category', 'Unknown'),
            instructions=context.get('instructions', 'No instructions provided'),
            code=context.get('code', ''),
            test_cases_scope=str(context.get('test_cases_scope', [])),
            failed_tests=str(context.get('failed_tests', []))
        )

        try:
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": INSTRUCTOR_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=DEEPSEEK_TEMPERATURE,
                max_tokens=DEEPSEEK_MAX_TOKENS
            )

            feedback = response.choices[0].message.content
            usage = response.usage

            generation_time = (time.time() - start_time) * 1000  # en ms

            return {
                "feedback": feedback,
                "tokens_prompt": usage.prompt_tokens,
                "tokens_completion": usage.completion_tokens,
                "tokens_total": usage.total_tokens,
                "generation_time_ms": generation_time
            }

        except Exception as e:
            return {
                "feedback": None,
                "error": str(e),
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "tokens_total": 0,
                "generation_time_ms": (time.time() - start_time) * 1000
            }

    def test_connection(self) -> bool:
        """Test si l'API est accessible"""
        try:
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            return True
        except Exception as e:
            print(f"API Test Failed: {e}")
            return False
