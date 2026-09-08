"""
code_agent.py - RAG Orchestrator using the google-antigravity SDK.

This module sets up the Agent with system instructions and exposes
the main execution pipeline for answering code-related questions.
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

from google.antigravity import Agent, LocalAgentConfig
from google.api_core.exceptions import ResourceExhausted
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
    before_sleep_log,
)
from src.db.vector_store import CodeVectorStore

logger = logging.getLogger(__name__)

# In-memory vector store instance (to be populated during app startup)
VECTOR_STORE = CodeVectorStore()

__all__ = ["VECTOR_STORE", "get_agent_response"]

# Use gemini-2.5-flash: higher free-tier RPM ceiling than 1.5-flash / 3.8-flash.
_CONFIG = LocalAgentConfig(
    model="gemini-2.5-flash",
    system_instruction=(
        "You are an expert software architect. "
        "Use the provided code context to answer the user's questions accurately. "
        "Always reference the file paths and line numbers when discussing the code."
    ),
)

# ---------------------------------------------------------------------------
# Retry helper — catches 429 ResourceExhausted, waits 15 s, retries up to 3x.
# ---------------------------------------------------------------------------
_RETRY_WAIT_SECONDS = 15
_MAX_ATTEMPTS = 3

@retry(
    reraise=True,
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_fixed(_RETRY_WAIT_SECONDS),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
async def _call_agent_with_retry(agent: Agent, prompt: str) -> str:
    """Inner call isolated so tenacity can wrap only the network hop."""
    response = await agent.chat(prompt)
    return await response.text()

async def get_agent_response(prompt: str, repo_context: List[Dict[str, Any]]) -> str:
    """
    Executes a chat turn with the google-antigravity SDK.
    Uses an async context manager and the `chat()` API.
    """
    # Construct a context-augmented prompt
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

    context_str = "\n\n".join(context_parts) if context_parts else "No relevant code context found."

    final_prompt = (
        f"Here is some relevant context from the codebase:\n\n"
        f"{context_str}\n\n"
        f"User Question: {prompt}\n\n"
        f"Please provide your answer based on the context above."
    )

    # Pass the prompt to the google.antigravity.Agent.
    # _call_agent_with_retry will automatically back off on 429s.
    async with Agent(config=_CONFIG) as agent:
        return await _call_agent_with_retry(agent, final_prompt)
