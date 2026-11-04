"""LLM abstraction and a single validated JSON-parsing layer.

The legacy agents call ``qwen_client.chat_json(...)`` directly and then do
``data = raw.get("json") or {}`` with no validation, so a missing/mistyped key
silently degrades to a fallback. This package funnels every model call through:

    client: LLMClient  ->  raw = client.chat_json(prompt, images)
                        ->  parsed = parse_llm_json(raw, schema)

``parsed`` is an :class:`LLMJson` that records the source (model JSON, a
reasoning-block fallback, or a hardcoded default) and any validation errors, so
fallbacks are never silently indistinguishable from real model output.
"""

from __future__ import annotations

from .base import CaseSessionLLMClient, LLMClient, LLMResponse, LegacyQwenAdapter
from .parsing import LLMJson, parse_llm_json
from .schema import Field, Schema, ValidationError

__all__ = [
    "LLMClient",
    "LLMResponse",
    "LegacyQwenAdapter",
    "CaseSessionLLMClient",
    "LLMJson",
    "parse_llm_json",
    "Schema",
    "Field",
    "ValidationError",
]
