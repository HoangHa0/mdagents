"""
Common utility functions for LLM API interfaces.
"""

import re
from typing import Callable, TypeVar, List, Dict, Any, Optional

T = TypeVar('T')


def build_messages(user_prompt: str, sys_prompt: str = "") -> List[Dict[str, str]]:
    """
    Build a standard messages list for chat completions.
    
    Args:
        user_prompt: The user's prompt
        sys_prompt: Optional system prompt
        
    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def base_ask(
    call_fn: Callable[[], str],
    infinite_retry: bool = False,
) -> str:
    """
    Base ask function that wraps a provider-specific call with retry logic.
    
    Args:
        call_fn: Provider-specific function that makes the API call and returns response
        infinite_retry: If True, retry forever on exceptions
        
    Returns:
        Generated text with reasoning removed
    """
    return call_with_retry(call_fn, infinite_retry, remove_reasoning)


def call_with_retry(
    call_fn: Callable[[], T],
    infinite_retry: bool = False,
    post_process: Callable[[T], T] = None,
) -> T:
    """
    Execute a callable with optional infinite retry logic.
    
    Args:
        call_fn: The function to call (e.g., API call)
        infinite_retry: If True, retry forever on exceptions
        post_process: Optional function to apply to the result (e.g., remove_reasoning)
        
    Returns:
        The result of call_fn, optionally post-processed
    """
    if infinite_retry:
        while True:
            try:
                result = call_fn()
                return post_process(result) if post_process else result
            except Exception:
                continue
    
    result = call_fn()
    return post_process(result) if post_process else result


def remove_reasoning(response_content: str) -> str:
    """Remove reasoning part if present (e.g., content wrapped in <think> tags)"""
    match = re.search(r"</think>\s*(.*)", response_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response_content.strip()


def extract_content(msg) -> str:
    """
    Extract text content from various message formats.
    
    Handles:
    - Objects with .content attribute
    - llama_index ChatMessage with .blocks
    - Dict-based messages
    """
    # Prefer direct content if available and non-empty
    if hasattr(msg, "content") and msg.content:
        return str(msg.content)

    # llama_index ChatMessage may carry text blocks
    if hasattr(msg, "blocks") and msg.blocks:
        texts = []
        for block in msg.blocks:
            if hasattr(block, "text") and block.text:
                texts.append(str(block.text))
            elif isinstance(block, dict) and block.get("text"):
                texts.append(str(block.get("text")))
        if texts:
            return "\n".join(texts)

    # Dict-based messages
    if isinstance(msg, dict):
        content_val = msg.get("content")
        if content_val:
            return str(content_val)
        blocks_val = msg.get("blocks")
        if blocks_val:
            texts = []
            for block in blocks_val:
                if isinstance(block, dict) and block.get("text"):
                    texts.append(str(block.get("text")))
            if texts:
                return "\n".join(texts)

    return ""


def normalize_message(message) -> dict:
    """
    Normalize a message to a standard dict format with 'role' and 'content' keys.
    
    Handles various message formats including objects with .role attribute,
    dicts, and enums for role values.
    """
    def _role_to_str(role_obj) -> str:
        # Handle enums (e.g., MessageRole.SYSTEM) by using .value when present
        if hasattr(role_obj, "value"):
            return str(role_obj.value)
        return str(role_obj)

    if hasattr(message, "role"):
        role_val = _role_to_str(message.role)
    elif isinstance(message, dict):
        role_val = _role_to_str(message.get("role", "user"))
    else:
        role_val = "user"

    content_val = extract_content(message)
    if not content_val:
        content_val = str(message)

    return {"role": role_val, "content": content_val}
