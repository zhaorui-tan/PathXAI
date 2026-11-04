from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def encode_image_b64(
    image_path: str | Path,
    max_edge: Optional[int] = None,
    jpeg_quality: int = 85,
) -> str:
    path = Path(image_path)
    if max_edge:
        from PIL import Image

        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size
            scale = min(1.0, float(max_edge) / float(max(width, height)))
            if scale < 1.0:
                new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def extract_json_object(text: str) -> Optional[dict]:
    """Extract the first complete JSON object from model text."""
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                block = text[start: idx + 1]
                try:
                    return json.loads(block)
                except Exception:
                    return None
    return None


def strip_think_blocks(text: str) -> tuple[str, str]:
    reasoning = ""
    if "<think>" in text:
        match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text, reasoning


class QwenVisionClient:
    """Small OpenAI-compatible Qwen/VLM client.

    The local PathXAI scripts use a vLLM-style endpoint at
    `http://HOST:PORT/v1/chat/completions`. This client keeps that contract but
    supports multiple patch images per call.

    Args:
        api_url: vLLM/OpenAI-compatible endpoint (host:port or full URL).
        model: Model name as registered by the serving endpoint.
        temperature: Sampling temperature. Use 0.0 for deterministic output
            (recommended for MedGemma and structured JSON stages).
        max_tokens: Maximum tokens to generate.
        timeout: (connect_timeout, read_timeout) in seconds.
        image_max_edge: Resize images so that the longest edge does not exceed
            this pixel count before encoding. None disables resizing.
        image_jpeg_quality: JPEG quality for image encoding.
        use_system_role: When True, the ``system_prompt`` argument of :meth:`chat`
            is sent as a proper ``role: system`` message rather than prepended as
            a text block in the user message.  Set this for Qwen3 and other models
            that honour the system role; leave False for MedGemma which does not
            use a separate system role.
        max_images: Hard limit on the number of images sent per call.  Excess
            images are silently dropped.  None means unlimited.
        enable_thinking: Controls Qwen3 extended thinking via vLLM's
            ``chat_template_kwargs``.  ``False`` suppresses the thinking chain
            (equivalent to a model-level ``/no_think``), which is strongly
            recommended for JSON-output stages to prevent the model from exhausting
            its token budget on CoT and producing empty ``content``.  ``True``
            enables thinking.  ``None`` (default) does not pass the kwarg and
            relies on the server default / the ``/no_think`` prompt prefix.
    """

    def __init__(
        self,
        api_url: str,
        model: str = "qwen",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: tuple[int, Optional[int]] = (10, 600),
        image_max_edge: Optional[int] = 2048,
        image_jpeg_quality: int = 85,
        use_system_role: bool = False,
        max_images: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
        max_retries: Optional[int] = None,
        retry_backoff: Optional[float] = None,
    ):
        self.api_url = api_url.replace("http://", "").replace("https://", "").rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.image_max_edge = image_max_edge
        self.image_jpeg_quality = image_jpeg_quality
        self.use_system_role = use_system_role
        self.max_images = max_images
        self.enable_thinking = enable_thinking
        self.max_retries = int(max_retries if max_retries is not None else os.environ.get("PATHXAI_LLM_MAX_RETRIES", 2))
        self.retry_backoff = float(retry_backoff if retry_backoff is not None else os.environ.get("PATHXAI_LLM_RETRY_BACKOFF", 3.0))

    def chat(
        self,
        prompt: str,
        image_paths: Optional[Iterable[str | Path]] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        """Send a chat request. If ``messages`` is provided, it is used as
        conversation history and the new prompt+images are appended to it."""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        retries = Retry(total=self.max_retries, backoff_factor=self.retry_backoff, status_forcelist=[500, 502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retries))

        # ── Build messages ──
        if messages is not None:
            msg_list: List[dict] = list(messages)
        else:
            msg_list = []

        # System role: send as a proper system message when use_system_role=True,
        # otherwise prepend to user content (MedGemma / models without system role).
        if system_prompt and self.use_system_role:
            msg_list.append({"role": "system", "content": system_prompt})

        content: List[dict] = []
        if system_prompt and not self.use_system_role:
            content.append({"type": "text", "text": system_prompt})
        content.append({"type": "text", "text": prompt})

        image_list = list(image_paths or [])
        if self.max_images is not None:
            image_list = image_list[: self.max_images]
        for image_path in image_list:
            b64 = encode_image_b64(
                image_path,
                max_edge=self.image_max_edge,
                jpeg_quality=self.image_jpeg_quality,
            )
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

        msg_list.append({"role": "user", "content": content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": msg_list,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        # Qwen3 thinking control: pass chat_template_kwargs so vLLM honours it
        # even when the model ignores the /no_think prompt prefix.
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        start = time.time()
        last_exc: Exception | None = None
        attempts = max(1, self.max_retries + 1)
        for attempt in range(attempts):
            try:
                resp = session.post(
                    f"http://{self.api_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                break
            except Exception as exc:
                last_exc = exc
                import requests as _req
                is_conn = isinstance(exc, (_req.exceptions.ConnectionError,
                                           _req.exceptions.ChunkedEncodingError,
                                           _req.exceptions.ReadTimeout,
                                           _req.exceptions.Timeout))
                if not is_conn or attempt >= attempts - 1:
                    raise
                wait = self.retry_backoff * (2 ** attempt)
                time.sleep(wait)
        else:
            raise last_exc  # type: ignore[misc]
        data = resp.json()
        choice = data["choices"][0]
        message = choice.get("message") or {}
        text = message.get("content", "") or ""
        reasoning = message.get("reasoning_content", "") or ""
        if not reasoning:
            text, reasoning = strip_think_blocks(text)
        # Append assistant response to history for session continuation
        msg_list.append({"role": "assistant", "content": text})
        return {
            "content": text,
            "reasoning": reasoning,
            "elapsed_sec": round(time.time() - start, 3),
            "raw": data,
            "messages": msg_list,  # for session reuse
        }

    def chat_json(
        self,
        prompt: str,
        image_paths: Optional[Iterable[str | Path]] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[List[dict]] = None,
    ) -> Dict[str, Any]:
        result = self.chat(prompt, image_paths=image_paths, system_prompt=system_prompt, messages=messages)
        parsed = extract_json_object(result["content"])
        # Fallback: Qwen3 thinking models sometimes produce non-empty reasoning_content
        # but empty content when the entire chain-of-thought consumes the generation budget.
        # Try to recover the final JSON answer from the reasoning block.
        if parsed is None and result.get("reasoning"):
            parsed = extract_json_object(result["reasoning"])
            if parsed is not None:
                result["json_source"] = "reasoning_fallback"
        result["json"] = parsed
        return result


class QwenEmbeddingClient:
    """OpenAI-compatible embedding client for local Qwen embedding endpoints."""

    def __init__(
        self,
        api_url: str,
        model: str = "qwen-embedding",
        timeout: tuple[int, Optional[int]] = (10, 120),
    ):
        self.api_url = api_url.replace("http://", "").replace("https://", "").rstrip("/")
        self.model = model
        self.timeout = timeout

    def embed(self, texts: Iterable[str]) -> Dict[str, Any]:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        inputs = [str(text or "") for text in texts]
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        session.mount("http://", HTTPAdapter(max_retries=retries))
        payload = {
            "model": self.model,
            "input": inputs,
        }
        start = time.time()
        resp = session.post(
            f"http://{self.api_url}/v1/embeddings",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        rows = sorted(data.get("data", []), key=lambda row: row.get("index", 0))
        return {
            "embeddings": [row.get("embedding", []) for row in rows],
            "elapsed_sec": round(time.time() - start, 3),
            "raw_model": data.get("model"),
            "usage": data.get("usage"),
        }
