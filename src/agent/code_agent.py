"""
code_agent.py - RAG Orchestrator using the Google GenAI SDK (google-genai).

This module initialises the Gemini client, builds context-augmented prompts
from retrieved AST chunks, and exposes the main execution pipeline for
answering code-related questions.
"""

import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables from .env before accessing any env var.
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from environment variables.")

import google.genai as genai
from google.genai.errors import ClientError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
    before_sleep_log,
)
from src.db.vector_store import CodeVectorStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini client — authenticates with the key loaded from the environment.
# ---------------------------------------------------------------------------
_client = genai.Client(api_key=api_key)

_MODEL = "gemini-2.5-flash"

_SYSTEM_INSTRUCTION = (
    "You are an expert software architect. "
    "Use the provided code context to answer the user's questions accurately. "
    "Always reference the file paths and line numbers when discussing the code."
)

# In-memory vector store instance (to be populated during app startup)
VECTOR_STORE = CodeVectorStore()

__all__ = ["VECTOR_STORE", "get_agent_response"]

# ---------------------------------------------------------------------------
# Retry helper — catches 429 ClientError, waits 15 s, retries up to 3 times.
# ---------------------------------------------------------------------------
_RETRY_WAIT_SECONDS = 15
_MAX_ATTEMPTS = 3


def _is_rate_limit(exc: BaseException) -> bool:
    """Return True only for HTTP 429 responses."""
    return isinstance(exc, ClientError) and exc.code == 429


@retry(
    reraise=True,
    retry=retry_if_exception_type(ClientError),
    wait=wait_fixed(_RETRY_WAIT_SECONDS),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _generate_with_retry(prompt: str) -> str:
    """
    Calls the Gemini generate_content API.
    Tenacity wraps this synchronous call and retries on ClientError (429).
    """
    response = _client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )
    return response.text


async def get_agent_response(prompt: str, repo_context: List[Dict[str, Any]]) -> str:
    """
    Builds a context-augmented prompt from retrieved AST chunks and calls
    the Gemini API via the google-genai SDK.
    """
    # 1. Build the context block from retrieved code chunks.
    context_parts = []
    for r in repo_context:
        path = r.get("file_path", "unknown")
        start = r.get("start_line", "?")
        end = r.get("end_line", "?")
        func = r.get("function_name", "unknown")
        snippet = r.get("code_snippet", "")

        context_parts.append(
            f"--- File: {path} (Lines {start}-{end}) | Name: {func} ---\n"
            f"```python\n{snippet}\n```"
        )

    context_str = (
        "\n\n".join(context_parts) if context_parts else "No relevant code context found."
    )

    # 2. Compose the final prompt.
    final_prompt = (
        f"Here is some relevant context from the codebase:\n\n"
        f"{context_str}\n\n"
        f"User Question: {prompt}\n\n"
        f"Please provide your answer based on the context above."
    )

    # 3. Call the API — _generate_with_retry backs off automatically on 429s.
    return _generate_with_retry(final_prompt)
