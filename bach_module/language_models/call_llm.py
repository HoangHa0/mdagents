"""
LLM API Interface for EMERGE - Ollama Integration

This module provides a clean interface to Ollama API for:
- Named Entity Recognition (NER) from clinical notes
- Summary generation from EHR + notes + KG context
"""

from dotenv import load_dotenv
import os
import json
import re
import requests
import time
from typing import List, Optional
from .utils import build_messages, base_ask, normalize_message, remove_reasoning
from ..utils.logging import log_to_file

# Load environment variables
load_dotenv()
OLLAMA_LAB_URL = os.getenv("BAILAB_HTTP")
OLLAMA_LOCAL_URL = os.getenv("LOCAL_HTTP")
OLLAMA_TUN_URL = os.getenv("TUN_HTTP")
OLLAMA_CHAMP_URL = os.getenv("CHAMP_HTTP")

# Normalize URLs
if OLLAMA_LAB_URL:
    OLLAMA_LAB_URL = OLLAMA_LAB_URL.rstrip("/")
if OLLAMA_LOCAL_URL:
    OLLAMA_LOCAL_URL = OLLAMA_LOCAL_URL.rstrip("/")
if OLLAMA_TUN_URL:
    OLLAMA_TUN_URL = OLLAMA_TUN_URL.rstrip("/")
if OLLAMA_CHAMP_URL:
    OLLAMA_CHAMP_URL = OLLAMA_CHAMP_URL.rstrip("/")

class OllamaClient:
    """Clean Ollama API client with proper error handling"""
    
    def __init__(self, base_url: str = OLLAMA_LAB_URL, timeout: int = 60):
        self.base_url = base_url
        self.generate_url = f"{base_url}"
        self.timeout = timeout
    
    def generate(
        self,
        prompt: str,
        model: str = "qwen2.5:7b-instruct",
        thinking: bool = False,
        temperature: float = 0.3,
        num_ctx: int = 8192,
        system: Optional[str] = None,
    ) -> str:
        """
        Generate text using Ollama API
        
        Args:
            prompt: User prompt
            model: Model name (e.g., "qwen:7b", "deepseek-v2:16b")
            thinking: Whether to enable "thinking" mode
            temperature: Sampling temperature (0.0 = deterministic)
            num_ctx: Context window size
            system: Optional system prompt
            
        Returns:
            Generated text
            
        Raises:
            requests.exceptions.RequestException: On API failure
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": thinking,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "main_gpu": 0,
                "num_gpu": -1,
                "num_batch": 1,
                "num_thread": 24
            }
        }
        
        if system:
            payload["system"] = system
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                # timeout=(self.timeout, None)
                timeout=None
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "").strip()
            
        except requests.exceptions.Timeout:
            raise Exception(f"Ollama API timeout after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama at {self.base_url}")
        except requests.exceptions.HTTPError as e:
            raise Exception(f"Ollama API error: {e}")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON response from Ollama")

# Global client instance
ollama_client = OllamaClient()

def ask(
    user_prompt: str,
    sys_prompt: str = "",
    model_name: str = "qwen2.5:7b-instruct",
    thinking: bool = False,
    max_tokens: int = 8192,
    temperature: float = 0.3,
    infinite_retry: bool = False,
) -> str:
    """
    Wrapper for backward compatibility with original EMERGE code
    
    Args:
        user_prompt: User prompt
        sys_prompt: System prompt (optional)
        model_name: Model name
        thinking: Whether to enable "thinking" mode
        max_tokens: Context window size
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    
    def try_generate_with_fallback():
        """Try primary URL first, then fallback to local URL if available"""
        urls_to_try = []

        def add_url(url: Optional[str]):
            if url and url not in urls_to_try:
                urls_to_try.append(url)

        # Preferred -> fallback order
        add_url(OLLAMA_LAB_URL)
        # add_url(OLLAMA_LOCAL_URL)
        
        last_error = None
        for url in urls_to_try:
            try:
                client = OllamaClient(base_url=url)
                response = client.generate(
                    prompt=user_prompt,
                    model=model_name,
                    thinking=thinking,
                    temperature=temperature,
                    num_ctx=max_tokens,
                    system=sys_prompt if sys_prompt else None,
                )
                # log_to_file("llm_calls.txt", response)
                return response
            except Exception as e:
                continue  # Try next URL
        
        # If all URLs failed, raise the last error
        if last_error:
            raise last_error
        else:
            raise Exception("No valid Ollama URLs configured")
    
    return base_ask(try_generate_with_fallback, infinite_retry)

class KGSumLLM:
    def __init__(self, model_name: str = "qwen2.5:7b-instruct"):
        self.model_name = model_name

    def chat(
        self,
        messages,
        model_name: str = "qwen2.5:7b-instruct-q4_K_M",
        max_tokens: int = 32768,
        thinking = False,
        infinite_retry: bool = True,
    ):
        """Compatibility shim for callers expecting llm.chat(messages)."""
        normalized_messages = [normalize_message(msg) for msg in (messages or [])]

        response = ask(
            # sys_prompt="Answer concisely. Do not show reasoning. Provide only the final result.",
            user_prompt="\n\n".join([msg["content"] for msg in normalized_messages]),
            model_name=self.model_name or model_name,
            max_tokens=max_tokens,
            thinking=thinking,
            infinite_retry=infinite_retry,
        )
        
        return response

# Test Functions

if __name__ == "__main__":
    """Test LLM API functionality"""
    
    print("Testing Ollama Integration")
    
    try:
        start = time.time()
        response = ask(
            user_prompt="Why is the sky blue?",
            model_name="gpt-oss:20b-cloud",
            thinking=False,
            max_tokens=1024,
            temperature=0.3
        )
        elapsed = time.time() - start
        print(f"Success: '{response}' ({elapsed:.2f}s)")
    except Exception as e:
        print(f"Failed: {e}")