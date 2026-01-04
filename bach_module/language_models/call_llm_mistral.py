
"""
LLM API Interface for EMERGE - Mistral Integration

This module mirrors the Ollama-based interface (call_llm.py) but uses Mistral
API behind the same helpers for:
- Named Entity Recognition (NER) from clinical notes
- Summary generation from EHR + notes + KG context
"""

from dotenv import load_dotenv
import os
import time
import requests
from threading import Lock
from .utils import build_messages, base_ask, normalize_message

# Load environment variables and configure Mistral API
load_dotenv()
MISTRAL_API_KEYS = [
    os.getenv("MISTRAL_API_KEY"),
    os.getenv("MISTRAL_API_KEY_1"),
    os.getenv("MISTRAL_API_KEY_2"),
    os.getenv("MISTRAL_API_KEY_3"),
    os.getenv("MISTRAL_API_KEY_4"),
    os.getenv("MISTRAL_API_KEY_5"),
    os.getenv("MISTRAL_API_KEY_6"),
    os.getenv("MISTRAL_API_KEY_7"),
    # os.getenv("MISTRAL_API_KEY_8"),
    # os.getenv("MISTRAL_API_KEY_9"),
    # os.getenv("MISTRAL_API_KEY_10"),
]

MISTRAL_API_KEYS = [k for k in MISTRAL_API_KEYS if k]  # Remove None values
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL_NAME = "mistral-small-2506"

# Round-robin API key selector
_api_rr_counter = [0]
_api_rr_lock = Lock()
def get_next_api_key():
    with _api_rr_lock:
        idx = _api_rr_counter[0] % len(MISTRAL_API_KEYS)
        _api_rr_counter[0] += 1
        return idx

def ask(
    user_prompt: str,
    sys_prompt: str = "",
    model_name: str = MISTRAL_MODEL_NAME,
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
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        last_exception = None
        n = len(MISTRAL_API_KEYS)
        start_idx = get_next_api_key()
        for i in range(n):
            api_idx = (start_idx + i) % n
            api_key = MISTRAL_API_KEYS[api_idx]
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            try:
                response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:
                last_exception = e
                continue
        if last_exception:
            raise last_exception

    return base_ask(_call_once, infinite_retry)


class KGSumLLM_mistral:
    def __init__(self, model_name: str = MISTRAL_MODEL_NAME):
        self.model_name = model_name

    def chat(
        self,
        messages,
        model_name: str = MISTRAL_MODEL_NAME,
        max_tokens: int = 32768,
        infinite_retry: bool = False,
    ):
        """Compatibility shim for callers expecting llm.chat(messages)."""
        normalized_messages = [normalize_message(msg) for msg in (messages or [])]

        def _call_once() -> str:
            payload = {
                "model": model_name or self.model_name,
                "messages": normalized_messages,
                "max_tokens": max_tokens
            }
            last_exception = None
            n = len(MISTRAL_API_KEYS)
            start_idx = get_next_api_key()
            for i in range(n):
                api_idx = (start_idx + i) % n
                api_key = MISTRAL_API_KEYS[api_idx]
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                try:
                    response = requests.post(MISTRAL_API_URL, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()
                    return data["choices"][0]["message"]["content"] or ""
                except Exception as e:
                    last_exception = e
                    continue
            if last_exception:
                raise last_exception

        return base_ask(_call_once, infinite_retry)

# Test Functions

if __name__ == "__main__":
    """Test LLM API functionality"""
    print("Testing Mistral Integration")
    try:
        start = time.time()
        response = ask(
            user_prompt="Why is the sky blue?",
            model_name=MISTRAL_MODEL_NAME,
            max_tokens=1024,
        )
        elapsed = time.time() - start
        print(f"Success: '{response}' ({elapsed:.2f}s)")
    except Exception as e:
        print(f"Failed: {e}")
