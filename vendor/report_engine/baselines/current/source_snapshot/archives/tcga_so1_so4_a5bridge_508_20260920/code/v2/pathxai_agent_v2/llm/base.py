"""LLM client abstraction.

The pipeline depends on this ``LLMClient`` protocol, not on the concrete
``QwenVisionClient``.  This decouples stages from the transport/model so a
different backend (another local vLLM model, a hosted API, or a test double)
can be dropped in without touching stage logic or prompts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

# Shape returned by a raw chat_json call. Matches the legacy QwenVisionClient
# contract so an adapter is trivial:
#   {"content": str, "reasoning": str, "json": dict | None,
#    "json_source": str | None, "elapsed_sec": float, ...}
LLMResponse = dict[str, Any]


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface every model backend must provide."""

    def chat_json(
        self,
        prompt: str,
        image_paths: Optional[Iterable[str | Path]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Return a raw response dict that may contain a parsed ``json`` field."""
        ...

    def chat(
        self,
        prompt: str,
        image_paths: Optional[Iterable[str | Path]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Return a raw chat response dict."""
        ...


class LegacyQwenAdapter:
    """Adapts the legacy ``pathxai_agent.llm.qwen_client.QwenVisionClient``.

    Lets v2 stages run against the already-deployed Qwen/MedGemma endpoints
    without importing the concrete class at module load time.
    """

    def __init__(self, inner: Any) -> None:
        if not hasattr(inner, "chat_json"):
            raise TypeError("inner client must expose chat_json(...)")
        self._inner = inner

    def chat_json(
        self,
        prompt: str,
        image_paths: Optional[Iterable[str | Path]] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[list[dict]] = None,
    ) -> LLMResponse:
        kwargs: dict = {"image_paths": image_paths, "system_prompt": system_prompt}
        if messages is not None:
            kwargs["messages"] = messages
        return self._inner.chat_json(prompt, **kwargs)

    def chat(
        self,
        prompt: str,
        image_paths: Optional[Iterable[str | Path]] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[list[dict]] = None,
    ) -> LLMResponse:
        kwargs: dict = {"image_paths": image_paths, "system_prompt": system_prompt}
        if messages is not None:
            kwargs["messages"] = messages
        return self._inner.chat(prompt, **kwargs)


class CaseSessionLLMClient:
    """Client-side case-level chat memory for Qwen calls.

    llama-server's OpenAI-compatible endpoint does not expose a stable
    server-side slot/session id.  This wrapper implements the reproducible
    equivalent by resending compact text-only prior turns with each new request.
    Current images are sent only for the current call; previous image pixels are
    deliberately not stored in history.
    """

    def __init__(
        self,
        inner: Any,
        *,
        out_dir: str | Path | None = None,
        max_turns: int | None = None,
        max_chars: int | None = None,
    ) -> None:
        if not hasattr(inner, "chat_json") or not hasattr(inner, "chat"):
            raise TypeError("inner client must expose chat_json(...) and chat(...)")
        self.inner = inner
        self.out_dir = Path(out_dir) if out_dir else None
        self.max_turns = int(max_turns or os.environ.get("PATHXAI_CASE_SESSION_MAX_TURNS", 16))
        self.max_chars = int(max_chars or os.environ.get("PATHXAI_CASE_SESSION_MAX_CHARS", 16000))
        self.messages: list[dict[str, str]] = []
        self.calls: list[dict[str, Any]] = []

    def chat_json(
        self,
        prompt: str,
        image_paths: Optional[Iterable[str | Path]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        result = self._call("chat_json", prompt, image_paths=image_paths, system_prompt=system_prompt)
        result["case_session_enabled"] = True
        return result

    def chat(
        self,
        prompt: str,
        image_paths: Optional[Iterable[str | Path]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        result = self._call("chat", prompt, image_paths=image_paths, system_prompt=system_prompt)
        result["case_session_enabled"] = True
        return result

    def _call(
        self,
        method: str,
        prompt: str,
        *,
        image_paths: Optional[Iterable[str | Path]] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        image_list = [str(p) for p in (image_paths or [])]
        history = self._trimmed_messages()
        fn = getattr(self.inner, method)
        try:
            result = fn(prompt, image_paths=image_list, system_prompt=system_prompt, messages=history)
            used_messages = True
        except TypeError:
            # Fallback for any non-Qwen test double/client that does not accept
            # the messages kwarg.  This still preserves case context as text.
            result = fn(
                self._prompt_with_history(prompt, history),
                image_paths=image_list,
                system_prompt=system_prompt,
            )
            used_messages = False
        backend_messages = result.pop("messages", None)
        if backend_messages is not None:
            try:
                result["backend_message_count"] = len(backend_messages)
            except Exception:
                result["backend_message_count"] = "unknown"
        self._record_turn(method, prompt, image_list, result, used_messages)
        self._save()
        return result

    def _record_turn(
        self,
        method: str,
        prompt: str,
        image_paths: list[str],
        result: LLMResponse,
        used_messages: bool,
    ) -> None:
        user_text = self._compact(
            prompt
            + f"\n\n[Current call attached {len(image_paths)} image(s). "
              "Prior images are not retained in chat history.]",
            limit=2500,
        )
        assistant_text = self._assistant_text(result)
        self.messages.append({"role": "user", "content": user_text})
        self.messages.append({"role": "assistant", "content": self._compact(assistant_text, limit=2500)})
        self.messages = self._trimmed_messages()
        self.calls.append({
            "call_index": len(self.calls) + 1,
            "method": method,
            "image_count": len(image_paths),
            "used_messages_kwarg": used_messages,
            "elapsed_sec": result.get("elapsed_sec"),
            "assistant_chars": len(assistant_text),
        })

    def _trimmed_messages(self) -> list[dict[str, str]]:
        msgs = list(self.messages[-2 * max(1, self.max_turns):])
        while msgs and sum(len(str(m.get("content", ""))) for m in msgs) > self.max_chars:
            msgs = msgs[2:]
        return msgs

    def _prompt_with_history(self, prompt: str, history: list[dict[str, str]]) -> str:
        if not history:
            return prompt
        parts = [
            "Prior case-level Qwen dialogue history follows. It is text-only; "
            "previous images are not available unless reattached in the current request."
        ]
        for msg in history:
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))
            parts.append(f"[{role}]\n{content}")
        parts.append("[current_user_request]\n" + prompt)
        return "\n\n".join(parts)

    def _assistant_text(self, result: LLMResponse) -> str:
        if isinstance(result.get("json"), (dict, list)):
            try:
                return json.dumps(result["json"], ensure_ascii=False)
            except Exception:
                pass
        text = str(result.get("content") or "")
        if text.strip():
            return text
        return str(result.get("reasoning") or "")

    def _compact(self, text: str, *, limit: int) -> str:
        text = str(text or "").strip()
        if len(text) <= limit:
            return text
        head = max(1, limit // 2)
        tail = max(1, limit - head - 80)
        return text[:head] + "\n...[truncated for case-session memory]...\n" + text[-tail:]

    def _save(self) -> None:
        if not self.out_dir:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "llm_case_session_history.json"
        path.write_text(
            json.dumps(
                {
                    "enabled": True,
                    "max_turns": self.max_turns,
                    "max_chars": self.max_chars,
                    "calls": self.calls,
                    "messages": self.messages,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
