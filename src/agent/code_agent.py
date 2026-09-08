"""
code_agent.py - RAG Orchestrator using the Google GenAI SDK (google-genai).

Builds context-augmented prompts from retrieved AST chunks and calls the
Gemini API to answer code-related questions about the indexed repository.
"""

import os
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# 1. Load .env FIRST — must happen before any os.getenv() call.
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# 2. Read and sanitise GEMINI_API_KEY (strips stray quotes / whitespace).
# ---------------------------------------------------------------------------
api_key = os.getenv("GEMINI_API_KEY", "").strip("'\" ")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing from environment variables. "
        "Add it to your .env file:\n  GEMINI_API_KEY=<your-key>"
    )

# ---------------------------------------------------------------------------
# 3. Third-party imports (after key is confirmed present).
# ---------------------------------------------------------------------------
import google.genai as genai
from google.genai.errors import APIError
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
# 4. Initialise the Gemini client bypassing default Auth methods for AQ keys
# ---------------------------------------------------------------------------
client = genai.Client(
    api_key=api_key,
    http_options={'headers': {'x-goog-api-key': api_key}}
)

_MODEL = "gemini-2.5-flash"

_SYSTEM_INSTRUCTION = (
    "You are an expert software architect. "
    "Use the provided code context to answer the user's questions accurately. "
    "Always reference the file paths and line numbers when discussing the code."
)

# ---------------------------------------------------------------------------
# 5. In-memory vector store (populated during app startup).
# ---------------------------------------------------------------------------
VECTOR_STORE = CodeVectorStore()

__all__ = ["VECTOR_STORE", "get_agent_response"]

# ---------------------------------------------------------------------------
# 6. Retry-wrapped Gemini call — backs off 15 s on any APIError (rate limit,
#    transient server errors, etc.) and retries up to 3 times.
# ---------------------------------------------------------------------------
_RETRY_WAIT_SECONDS = 15
_MAX_ATTEMPTS = 3


@retry(
    reraise=True,
    retry=retry_if_exception_type(APIError),
    wait=wait_fixed(_RETRY_WAIT_SECONDS),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _generate_with_retry(prompt: str) -> str:
    """
    Calls client.models.generate_content() with the configured model.
    Tenacity retries the call on any APIError (429, 500, 503, etc.).
    """
    response = client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.2,
        ),
    )
    return response.text


# ---------------------------------------------------------------------------
# 7. Public entry point — called by src/ui/app.py via asyncio.run().
# ---------------------------------------------------------------------------
async def get_agent_response(prompt: str, repo_context: List[Dict[str, Any]]) -> str:
    """
    Builds a context-augmented prompt from retrieved AST chunks and calls
    the Gemini API via _generate_with_retry.
    """
    # Build the context block from retrieved code chunks.
    context_parts = []
    for r in repo_context:
        path    = r.get("file_path",     "unknown")
        start   = r.get("start_line",   "?")
        end     = r.get("end_line",     "?")
        func    = r.get("function_name","unknown")
        snippet = r.get("code_snippet", "")

        context_parts.append(
            f"--- File: {path} (Lines {start}-{end}) | Name: {func} ---\n"
            f"```python\n{snippet}\n```"
        )

    context_str = (
        "\n\n".join(context_parts) if context_parts
        else "No relevant code context found."
    )

    final_prompt = (
        f"Here is some relevant context from the codebase:\n\n"
        f"{context_str}\n\n"
        f"User Question: {prompt}\n\n"
        f"Please provide your answer based on the context above."
    )

    return _generate_with_retry(final_prompt)
