"""
agent/llm.py — Lazy LLM client.

Supports Anthropic (default) and OpenAI via a single call() interface.
Client is instantiated on first use (lazy) — no import-time cost.

Design:
- One module-level singleton per provider, created on first call()
- Raises clear errors if LLM_API_KEY is missing
- All prompts go through call(prompt, system) → str
- Temperature + max_tokens from settings; overridable per-call for tests
"""

from __future__ import annotations

import logging
from typing import Any

from ira.settings import settings

logger = logging.getLogger(__name__)

# ── Singletons (None until first use) ────────────────────────────────────────
_anthropic_client: Any = None
_openai_client: Any = None


def _get_anthropic_client() -> Any:
    global _anthropic_client
    if _anthropic_client is None:
        if not settings.LLM_API_KEY:
            raise RuntimeError(
                "LLM_API_KEY is not set. "
                "Add LLM_API_KEY=your_key to your .env file."
            )
        import anthropic  # lazy import — only if provider == anthropic
        _anthropic_client = anthropic.Anthropic(api_key=settings.LLM_API_KEY)
        logger.debug("Anthropic client initialised", extra={"model": settings.LLM_MODEL})
    return _anthropic_client


def _get_openai_client() -> Any:
    global _openai_client
    if _openai_client is None:
        if not settings.LLM_API_KEY:
            raise RuntimeError(
                "LLM_API_KEY is not set. "
                "Add LLM_API_KEY=your_key to your .env file."
            )
        import openai  # lazy import — only if provider == openai
        _openai_client = openai.OpenAI(api_key=settings.LLM_API_KEY)
        logger.debug("OpenAI client initialised", extra={"model": settings.LLM_MODEL})
    return _openai_client


def call(
    prompt: str,
    system: str = "You are a precise research assistant specialising in LLM efficiency and optimisation.",
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """
    Send a prompt to the configured LLM and return the text response.

    Args:
        prompt:      User-turn content (the main instruction + context).
        system:      System prompt. Override per-call for specialised tasks.
        max_tokens:  Override settings.LLM_MAX_TOKENS for this call.
        temperature: Override settings.LLM_TEMPERATURE for this call.

    Returns:
        Raw text string from the model.

    Raises:
        RuntimeError: If LLM_API_KEY is missing.
        Exception:    Any API error propagates up — caller handles retries.
    """
    _max_tokens = max_tokens if max_tokens is not None else settings.LLM_MAX_TOKENS
    _temperature = temperature if temperature is not None else settings.LLM_TEMPERATURE

    if settings.LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt, system, _max_tokens, _temperature)
    elif settings.LLM_PROVIDER == "openai":
        return _call_openai(prompt, system, _max_tokens, _temperature)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.LLM_PROVIDER!r}. Use 'anthropic' or 'openai'.")


def _call_anthropic(
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
) -> str:
    client = _get_anthropic_client()
    response = client.messages.create(
        model=settings.LLM_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    # response.content is a list of ContentBlock; grab first text block
    return response.content[0].text


def _call_openai(
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
) -> str:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=settings.LLM_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def reset_clients() -> None:
    """
    Reset cached singletons. Used in tests to inject mock clients.

    Usage in tests:
        import ira.agent.llm as llm_module
        llm_module._anthropic_client = MockClient()
    """
    global _anthropic_client, _openai_client
    _anthropic_client = None
    _openai_client = None