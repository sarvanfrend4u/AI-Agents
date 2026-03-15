"""
LLM client for Atlas Realty agent system.
Wraps the Anthropic SDK directly — no LangChain model dependency.
Uses litellm for provider-agnostic fallback if needed.
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Model defaults — overridable via .env
# ---------------------------------------------------------------------------
MODEL_DEFAULTS = {
    "pm":          os.getenv("PM_MODEL",          "claude-sonnet-4-6"),
    "arch":        os.getenv("ARCH_MODEL",         "claude-sonnet-4-6"),
    "design":      os.getenv("DESIGN_MODEL",       "claude-sonnet-4-6"),
    "security":    os.getenv("SECURITY_MODEL",     "claude-sonnet-4-6"),
    "dev_backend": os.getenv("DEV_BACKEND_MODEL",  "claude-sonnet-4-6"),
    "dev_frontend":os.getenv("DEV_FRONTEND_MODEL", "claude-sonnet-4-6"),
    "code_review": os.getenv("CODE_REVIEW_MODEL",  "claude-sonnet-4-6"),
    "test":        os.getenv("TEST_MODEL",         "claude-sonnet-4-6"),
    "performance": os.getenv("PERFORMANCE_MODEL",  "claude-haiku-4-5-20251001"),
    "docs":        os.getenv("DOCS_MODEL",         "claude-haiku-4-5-20251001"),
}

from typing import Optional
_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    """Return a shared Anthropic client (lazy init)."""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Copy agent-system/.env.example to agent-system/.env and fill it in."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def call_agent(
    agent_name: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 8192,
) -> str:
    """
    Send a message to the LLM and return the text response.

    Args:
        agent_name: Key from MODEL_DEFAULTS (e.g. "pm", "arch").
        system_prompt: Full system prompt for this agent.
        user_message: The user turn (contains state context).
        max_tokens: Maximum tokens to generate.

    Returns:
        The assistant text response as a plain string.
    """
    model = MODEL_DEFAULTS.get(agent_name, "claude-sonnet-4-6")
    client = get_client()

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text
