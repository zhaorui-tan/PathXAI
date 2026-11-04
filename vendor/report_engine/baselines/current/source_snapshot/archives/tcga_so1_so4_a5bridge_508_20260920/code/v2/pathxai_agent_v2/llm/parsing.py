"""The single entry point for turning a raw LLM response into validated data.

Replaces the scattered ``data = raw.get("json") or {}`` pattern. Every stage
should call :func:`parse_llm_json` and inspect the returned :class:`LLMJson`
(especially ``.source`` and ``.errors``) so a fallback is never mistaken for a
genuine model answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .base import LLMResponse
from .schema import Schema


class JsonSource(str, Enum):
    """Where the parsed object actually came from."""

    MODEL = "model_json"             # parsed from the model's content
    REASONING_FALLBACK = "reasoning_fallback"  # recovered from a think block
    DEFAULT = "default"              # no JSON at all; schema defaults used


@dataclass
class LLMJson:
    """Result of parsing + validating one LLM response."""

    data: dict[str, Any]
    source: JsonSource
    errors: list[str] = field(default_factory=list)
    raw_content: str = ""
    reasoning: str = ""
    elapsed_sec: float = 0.0

    @property
    def ok(self) -> bool:
        """True when JSON came from the model and passed validation."""
        return self.source is JsonSource.MODEL and not self.errors

    @property
    def is_fallback(self) -> bool:
        return self.source is JsonSource.DEFAULT

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data[key]


def parse_llm_json(
    raw: Optional[LLMResponse],
    schema: Optional[Schema] = None,
    *,
    strict: bool = False,
) -> LLMJson:
    """Normalize + validate a raw ``chat_json`` response.

    Parameters
    ----------
    raw:
        The dict returned by :meth:`LLMClient.chat_json`. May be ``None`` (e.g.
        dry-run or a failed call), in which case a DEFAULT result is returned.
    schema:
        Optional :class:`Schema` to validate/coerce against. Without one, the
        parsed object is passed through untouched.
    strict:
        Forwarded to :meth:`Schema.validate`; raises on any validation error.
    """
    if not raw:
        data, errors = _apply_schema(schema, {}, strict=strict)
        return LLMJson(data=data, source=JsonSource.DEFAULT, errors=errors)

    content = str(raw.get("content") or "")
    reasoning = str(raw.get("reasoning") or "")
    elapsed = float(raw.get("elapsed_sec") or 0.0)

    parsed = raw.get("json")
    source = JsonSource.MODEL
    if parsed is None:
        source = JsonSource.DEFAULT
    elif raw.get("json_source") == "reasoning_fallback":
        source = JsonSource.REASONING_FALLBACK

    base_obj = parsed if isinstance(parsed, dict) else {}
    data, errors = _apply_schema(schema, base_obj, strict=strict)

    if parsed is not None and not isinstance(parsed, dict):
        errors.insert(
            0,
            f"model json was {type(parsed).__name__}, expected object",
        )
        source = JsonSource.DEFAULT

    return LLMJson(
        data=data,
        source=source,
        errors=errors,
        raw_content=content,
        reasoning=reasoning,
        elapsed_sec=elapsed,
    )


def _apply_schema(
    schema: Optional[Schema],
    obj: dict[str, Any],
    *,
    strict: bool,
) -> tuple[dict[str, Any], list[str]]:
    if schema is None:
        return dict(obj), []
    return schema.validate(obj, strict=strict)
