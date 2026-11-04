"""Explicit, bounded compatibility smoke test for the configured vision API.

This helper is only for a user-triggered connectivity check. It is not used by
the scientific pipeline and does not modify its model parameters.
"""

from __future__ import annotations

import base64
from io import BytesIO
import json
import socket
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .transport import rewrite_request


_MAX_RESPONSE_BYTES = 256 * 1024
_TIMEOUT_SECONDS = 15


class _ProbeConfigError(ValueError):
    pass


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _result(ok: bool, message: str, *, model: str = "", elapsed_ms: int = 0) -> dict[str, Any]:
    return {
        "ok": ok,
        "model": model[:256],
        "elapsed_ms": max(0, int(elapsed_ms)),
        "message": message[:500],
        "scope": "A successful probe does not establish full scientific API compatibility.",
    }


def _read_bounded(response, *, deadline: float) -> bytes:
    chunks = bytearray()
    while len(chunks) <= _MAX_RESPONSE_BYTES:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        if sock is not None:
            sock.settimeout(remaining)
        chunk = response.read(min(8192, _MAX_RESPONSE_BYTES + 1 - len(chunks)))
        if not chunk:
            break
        chunks.extend(chunk)
    return bytes(chunks)


def _synthetic_jpeg() -> bytes:
    """Create a deterministic 32x32 image with no case or user data."""
    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (32, 32), color=(127, 127, 127)).save(output, format="JPEG", quality=85)
    return output.getvalue()


def probe_api(config: dict[str, Any], *, timeout_seconds: float = _TIMEOUT_SECONDS) -> dict[str, Any]:
    """Send one synthetic image + JSON smoke-test request; never log credentials."""
    started = time.monotonic()
    try:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds <= 300:
            raise _ProbeConfigError("Probe timeout must be greater than zero and at most 300 seconds.")
        if not isinstance(config, dict):
            raise _ProbeConfigError("Configuration must be an object.")
        endpoint = config.get("qwen_api")
        requested_model = config.get("qwen_model")
        api_key = config.get("api_key") or ""
        if not isinstance(endpoint, str) or not endpoint.strip() or len(endpoint) > 2048:
            raise _ProbeConfigError("Enter a valid API endpoint before testing.")
        if not isinstance(requested_model, str) or not requested_model.strip() or len(requested_model) > 512:
            raise _ProbeConfigError("Enter a model identifier before testing.")
        if not isinstance(api_key, str) or len(api_key) > 8192:
            raise _ProbeConfigError("The API key value is invalid.")
        if "\r" in api_key or "\n" in api_key:
            raise _ProbeConfigError("The API key value is invalid.")
        parsed_endpoint = urlsplit(endpoint.strip() if "://" in endpoint else "http://" + endpoint.strip())
        if (parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.hostname
                or parsed_endpoint.username or parsed_endpoint.password
                or parsed_endpoint.query or parsed_endpoint.fragment):
            raise _ProbeConfigError("Endpoint must be an http(s) URL without embedded credentials or query parameters.")

        url, headers = rewrite_request(
            "http://frozen-client.invalid/v1/chat/completions",
            {"Accept": "application/json", "Content-Type": "application/json"},
            base_url=endpoint.strip(), api_key=api_key,
        )
        prompt = (
            "This is a connectivity test, not a clinical task. Inspect the attached synthetic "
            "32x32 test image and reply with exactly this JSON object and no other text: "
            '{"pathxai_api_probe":"ok"}'
        )
        payload = {
            "model": requested_model.strip(),
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(_synthetic_jpeg()).decode("ascii")
                }},
            ]}],
            # These are probe-only values. No scientific run settings are read or changed.
            "temperature": 0.2,
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request = Request(url, data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                          headers=headers, method="POST")
        opener = build_opener(_NoRedirect())
        with opener.open(request, timeout=timeout_seconds) as response:
            raw = _read_bounded(response, deadline=started + timeout_seconds)
        if len(raw) > _MAX_RESPONSE_BYTES:
            return _result(False, "Endpoint response exceeded the 256 KiB limit.",
                           elapsed_ms=(time.monotonic() - started) * 1000)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return _result(False, "Endpoint did not return valid UTF-8 JSON.",
                           elapsed_ms=(time.monotonic() - started) * 1000)
        choices = body.get("choices") if isinstance(body, dict) else None
        message = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        actual_model = body.get("model", "") if isinstance(body, dict) else ""
        if not isinstance(actual_model, str) or not actual_model.strip():
            return _result(False, "Endpoint response omitted the actual model identifier.",
                           elapsed_ms=(time.monotonic() - started) * 1000)
        if not isinstance(content, str):
            return _result(False, "Endpoint response did not contain a text completion.", model=actual_model,
                           elapsed_ms=(time.monotonic() - started) * 1000)
        try:
            probe_reply = json.loads(content.strip())
        except json.JSONDecodeError:
            return _result(False, "Image request completed, but the response was not a JSON object.", model=actual_model,
                           elapsed_ms=(time.monotonic() - started) * 1000)
        if not isinstance(probe_reply, dict) or probe_reply.get("pathxai_api_probe") != "ok":
            return _result(False, "Image request completed, but the JSON probe response did not match.", model=actual_model,
                           elapsed_ms=(time.monotonic() - started) * 1000)
        return _result(True, "Synthetic image request and JSON response verified.", model=actual_model,
                       elapsed_ms=(time.monotonic() - started) * 1000)
    except HTTPError as exc:
        return _result(False, f"Endpoint returned HTTP {exc.code}.",
                       elapsed_ms=(time.monotonic() - started) * 1000)
    except (TimeoutError, socket.timeout):
        return _result(False, f"Request timed out after {timeout_seconds} seconds.",
                       elapsed_ms=(time.monotonic() - started) * 1000)
    except URLError as exc:
        if isinstance(exc.reason, (TimeoutError, socket.timeout)):
            message = f"Request timed out after {timeout_seconds} seconds."
        else:
            message = "Could not reach the configured API endpoint."
        return _result(False, message, elapsed_ms=(time.monotonic() - started) * 1000)
    except _ProbeConfigError as exc:
        return _result(False, str(exc), elapsed_ms=(time.monotonic() - started) * 1000)
    except (ValueError, TypeError):
        return _result(False, "API probe request could not be constructed from the supplied configuration.",
                       elapsed_ms=(time.monotonic() - started) * 1000)
    except Exception:
        # Never expose exception strings: some HTTP libraries include request
        # URLs or headers in them.
        return _result(False, "API probe failed due to a transport or response error.",
                       elapsed_ms=(time.monotonic() - started) * 1000)
