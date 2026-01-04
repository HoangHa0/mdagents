"""
LLM API Interface for EMERGE - OpenAI Integration

This module mirrors the Ollama-based interface (call_llm.py) but uses OpenAI
models behind the same helpers for:
- Named Entity Recognition (NER) from clinical notes
- Summary generation from EHR + notes + KG context
"""

from dotenv import load_dotenv
import os
import time
from openai import OpenAI
from .utils import build_messages, base_ask, normalize_message

# Load environment variables and configure OpenAI client
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client_kwargs = {"api_key": OPENAI_API_KEY}
openai_client = OpenAI(**openai_client_kwargs)

def ask(
    user_prompt: str,
    sys_prompt: str = "",
    model_name: str = "gpt-5-nano",
    max_tokens: int = 32768,
    temperature: float = 0.1,
    infinite_retry: bool = False,
) -> str:
    """
    Wrapper for backward compatibility with original EMERGE code
    
    Args:
        user_prompt: User prompt
        sys_prompt: System prompt (optional)
        model_name: Model name
        max_tokens: Context window size
        
    Returns:
        Generated text
    """
    messages = build_messages(user_prompt, sys_prompt)

    def _call_once() -> str:
        completion = openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort="minimal",
        )
        return completion.choices[0].message.content or ""

    return base_ask(_call_once, infinite_retry)

class KGSumLLM_openai:
    def __init__(self, model_name: str = "gpt-5-nano"):
        self.model_name = model_name

    def chat(
        self,
        messages,
        model_name: str = "gpt-5-nano",
        max_tokens: int = 32768,
        infinite_retry: bool = False,
    ):
        """Compatibility shim for callers expecting llm.chat(messages)."""
        normalized_messages = [normalize_message(msg) for msg in (messages or [])]

        def _call_once() -> str:
            completion = openai_client.chat.completions.create(
                model=model_name or self.model_name,
                messages=normalized_messages,
                max_completion_tokens=max_tokens,
            )
            return completion.choices[0].message.content or ""

        return base_ask(_call_once, infinite_retry)

# Test Functions

if __name__ == "__main__":
    """Test LLM API functionality"""
    
    print("Testing OpenAI Integration")
    
    try:
        start = time.time()
        response = ask(
            user_prompt="Why is the sky blue?",
            model_name="gpt-5-nano",
            max_tokens=32768,
        )
        elapsed = time.time() - start
        print(f"Success: '{response}' ({elapsed:.2f}s)")
    except Exception as e:
        print(f"Failed: {e}")
