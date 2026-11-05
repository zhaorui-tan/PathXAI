"""Strict local LLM response capture/replay and scientific output comparison.

Trace files contain prompts, messages, and model responses. Keep them in a
private run directory. Image bytes are never embedded; only SHA-256 digests of
source and encoded request images are stored.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib
import importlib.util
import json
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

TRACE_SCHEMA = "pathxai-llm-trace-v1"
_GENERATION_FIELDS = (
    "model", "temperature", "max_tokens", "top_p", "top_k", "seed",
    "enable_thinking", "chat_template_kwargs", "image_max_edge",
    "image_jpeg_quality", "max_images", "use_system_role", "timeout",
    "max_retries", "retry_backoff",
)
_TIMESTAMP_FIELD_NAMES = {"created_at", "updated_at", "timestamp", "run_timestamp"}
_PATH_FIELD_NAMES = {"path", "image_path", "slide_path", "resource_path", "output_path", "uri"}


class ReplayError(RuntimeError):
    """A recorded response cannot safely satisfy the current call."""


class MissingRecordedResponse(ReplayError):
    pass


class RequestMismatch(ReplayError):
    pass


class _ReplayProgressJournal:
    """Private, append-only fingerprint evidence surviving abrupt cancellation."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        try:
            info = os.fstat(fd)
            self.identity = (info.st_dev, info.st_ino)
        finally:
            os.close(fd)
        self.events_written = 0

    def append(self, event: dict[str, Any]) -> None:
        if self.path.is_symlink():
            raise ReplayError("Replay progress journal was replaced; refusing to append")
        flags = os.O_APPEND | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(self.path, flags)
        with os.fdopen(fd, "wb") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or (info.st_dev, info.st_ino) != self.identity):
                raise ReplayError("Replay progress journal identity changed; refusing to append")
            row = {"schema": "pathxai-replay-progress-v1", "sequence": self.events_written, **event}
            stream.write(_ordered_json(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events_written += 1

    def verify(self, *, trace_sha256: str, entries: list[dict], consumed: int,
               violation_count: int) -> dict[str, Any]:
        if self.path.is_symlink():
            raise ReplayError("Replay progress journal was replaced")
        fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if ((before.st_dev, before.st_ino) != self.identity or before.st_nlink != 1
                    or not stat.S_ISREG(before.st_mode)):
                raise ReplayError("Replay progress journal identity changed")
            raw = stream.read()
            after = os.fstat(stream.fileno())
        if ((before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ReplayError("Replay progress journal changed during verification")
        if not raw.endswith(b"\n"):
            raise ReplayError("Replay progress journal has an incomplete final line")
        try:
            rows = [json.loads(line) for line in raw.split(b"\n")[:-1]]
        except (UnicodeError, ValueError) as exc:
            raise ReplayError("Replay progress journal has an invalid row") from exc
        allowed = {"started", "request_matched", "request_mismatch", "trace_exhausted",
                   "request_build_error", "network_blocked", "finished"}
        if (len(rows) != self.events_written or not rows or rows[0].get("event") != "started"
                or rows[-1].get("event") != "finished" or rows[0].get("trace_sha256") != trace_sha256
                or rows[0].get("recorded_requests") != len(entries)):
            raise ReplayError("Replay progress journal header, length, or finish does not match")
        matched = 0
        violations = 0
        for index, row in enumerate(rows):
            event = row.get("event")
            if (row.get("schema") != "pathxai-replay-progress-v1" or row.get("sequence") != index
                    or event not in allowed or (event == "started" and index != 0)
                    or (event == "finished" and index != len(rows) - 1)):
                raise ReplayError("Replay progress journal sequence or event is invalid")
            if event == "request_matched":
                if matched >= len(entries):
                    raise ReplayError("Replay progress journal matched beyond its trace")
                wanted = entries[matched]["request"]
                if (row.get("trace_index") != matched or row.get("expected_fingerprint") != wanted["fingerprint"]
                        or row.get("actual_fingerprint") != wanted["fingerprint"]
                        or row.get("method") != wanted["method"]
                        or row.get("image_count") != len(wanted.get("images", []))):
                    raise ReplayError("Replay progress journal request fingerprint differs from loaded trace")
                matched += 1
            elif event not in {"started", "finished"}:
                violations += 1
            if (row.get("consumed_requests") != matched
                    or row.get("remaining_requests") != len(entries) - matched):
                raise ReplayError("Replay progress journal cursor is inconsistent")
        if matched != consumed or violations != violation_count:
            raise ReplayError("Replay progress journal differs from observed in-memory counters")
        return {"status": "verified", "sha256": _sha256_bytes(raw), "size_bytes": len(raw),
                "complete_lines": True, "events": len(rows), "matched_requests": matched,
                "violation_count": violations}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _ordered_json(value: Any) -> bytes:
    """Serialize JSON without sorting mappings, preserving model response order."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _image_encoder(inner: Any) -> tuple[Any | None, Any | None]:
    """Return the backend's actual image encoder and its leaf client, if found."""
    pending, visited = [inner], set()
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        direct_encoder = getattr(current, "encode_image_b64", None)
        if callable(direct_encoder):
            return direct_encoder, current
        module_name = getattr(type(current), "__module__", "")
        if module_name:
            try:
                encoder = getattr(importlib.import_module(module_name), "encode_image_b64", None)
                if callable(encoder):
                    return encoder, current
            except Exception:
                pass
        pending.extend(getattr(current, name, None) for name in ("inner", "_inner"))
    return None, None


def _client_generation_params(inner: Any, explicit: Mapping[str, Any] | None) -> dict[str, Any]:
    params = {name: None for name in _GENERATION_FIELDS}
    pending, visited = [inner], set()
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        for name in _GENERATION_FIELDS:
            value = getattr(current, name, None)
            if value is not None:
                params[name] = value
        pending.extend(getattr(current, name, None) for name in ("inner", "_inner"))
    if explicit:
        params.update(dict(explicit))
    return _json_safe(params)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    raise TypeError(f"trace values must be JSON-safe; got {type(value).__name__}")


def _hash_images(image_paths: Iterable[str | Path] | None, inner: Any,
                 generation_params: Mapping[str, Any]) -> list[dict[str, Any]]:
    paths = list(image_paths or [])
    encoder, leaf = _image_encoder(inner)
    rows = []
    for index, item in enumerate(paths):
        path = Path(item)
        if not path.is_file():
            raise FileNotFoundError(f"request image {index} is unavailable")
        source_digest = _sha256_bytes(path.read_bytes())
        row: dict[str, Any] = {"index": index, "source_sha256": source_digest}
        if encoder is not None and leaf is not None:
            max_edge = generation_params.get("image_max_edge", getattr(leaf, "image_max_edge", None))
            jpeg_quality = generation_params.get("image_jpeg_quality", getattr(leaf, "image_jpeg_quality", 85))
            encoded = encoder(path, max_edge=max_edge, jpeg_quality=jpeg_quality)
            payload = base64.b64decode(encoded, validate=True)
            row["payload_sha256"] = _sha256_bytes(payload)
            row["encoding"] = {"function": "backend.encode_image_b64", "max_edge": max_edge,
                                "jpeg_quality": jpeg_quality}
        else:
            # The original bytes are the backend payload only when no encoder
            # is configured. Record that fact so audits can distinguish it.
            row["payload_sha256"] = source_digest
            row["encoding"] = {"function": "identity", "max_edge": None, "jpeg_quality": None}
        rows.append(row)
    return rows


def request_record(
    method: str,
    prompt: str,
    *,
    system_prompt: str | None,
    messages: Any,
    image_paths: Iterable[str | Path] | None,
    generation_params: Mapping[str, Any],
    extra_parameters: Mapping[str, Any] | None = None,
    image_backend: Any = None,
) -> dict[str, Any]:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    normalized_messages = _json_safe(messages)
    submitted_paths = list(image_paths or [])
    max_images = generation_params.get("max_images")
    if max_images is not None:
        submitted_paths = submitted_paths[:max_images]
    image_rows = _hash_images(submitted_paths, image_backend, generation_params)
    core = {
        "method": method,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "messages": normalized_messages,
        "images": image_rows,
        "generation_params": _json_safe(generation_params),
        "extra_parameters": _json_safe(extra_parameters or {}),
    }
    return {**core, "fingerprint": _sha256_bytes(_canonical_json(core))}


class TraceLLMClient:
    """LLMClient decorator that records exact ordered requests and raw responses.

    Put this around the concrete client (or its adapter), then put any case
    session/memory wrapper outside it so the exact messages passed to the
    backend are captured. ``generation_params`` can supplement backend fields
    that are not exposed as attributes. This decorator never performs a call
    during replay.
    """

    def __init__(self, inner: Any, trace_path: str | Path, *,
                 generation_params: Mapping[str, Any] | None = None) -> None:
        if not hasattr(inner, "chat") or not hasattr(inner, "chat_json"):
            raise TypeError("inner client must expose chat(...) and chat_json(...)")
        self.inner = inner
        self.trace_path = Path(trace_path)
        self.generation_params = dict(generation_params or {})
        self._lock = threading.Lock()
        self._next_sequence = 0
        # A record belongs to one run. Reserving it exclusively also prevents
        # concurrent workers or an explicit reused path from mixing histories.
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.trace_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise FileExistsError("Recording requires a new trace path; existing evidence will not be appended or overwritten.") from exc
        os.close(descriptor)
        from .request_audit import RequestJournal, install_http_audit_hooks
        self.events_path = self.trace_path.with_name(self.trace_path.name + ".events.jsonl")
        self.journal = RequestJournal(self.events_path)
        install_http_audit_hooks()

    def chat(self, prompt: str, image_paths=None, system_prompt=None, messages=None, **kwargs):
        return self._call("chat", prompt, image_paths, system_prompt, messages, kwargs)

    def chat_json(self, prompt: str, image_paths=None, system_prompt=None, messages=None, **kwargs):
        return self._call("chat_json", prompt, image_paths, system_prompt, messages, kwargs)

    def _call(self, method, prompt, image_paths, system_prompt, messages, kwargs):
        images = list(image_paths or [])
        extra = dict(kwargs)
        call_context = extra.pop("trace_generation_params", {})
        params = _client_generation_params(self.inner, {**self.generation_params, **call_context})
        with self._lock:
            sequence = self._next_sequence
            self._next_sequence += 1
        from .request_audit import audit_call, _without_image_bytes
        self.journal.append("call_started", call_id=sequence, method=method,
                            prompt=prompt, system_prompt=system_prompt,
                            messages=_without_image_bytes(_json_safe(messages)),
                            generation_params=params, image_count=len(images))
        start = time.monotonic()
        try:
            request = request_record(method, prompt, system_prompt=system_prompt, messages=messages,
                                     image_paths=images, generation_params=params,
                                     extra_parameters=extra, image_backend=self.inner)
            self.journal.append("request_prepared", call_id=sequence, request=_without_image_bytes(request))
            with audit_call(self.journal, sequence):
                response = getattr(self.inner, method)(prompt, image_paths=images,
                                                      system_prompt=system_prompt,
                                                      messages=messages, **extra)
            entry = self._save_response(sequence, request, response)
            self.journal.append("call_succeeded", call_id=sequence,
                                response_sha256=entry["response_sha256"],
                                elapsed_seconds=time.monotonic() - start)
        except BaseException as exc:
            self.journal.append("call_failed", call_id=sequence, error_type=type(exc).__name__,
                                elapsed_seconds=time.monotonic() - start)
            raise
        return response

    def _save_response(self, sequence, request, response):
        # Some backends return a message history containing base64-encoded
        # images. It is deliberately omitted; replay does not need it below
        # CaseSessionLLMClient, which consumes and removes this field.
        saved_response = dict(response)
        omitted = []
        if "messages" in saved_response:
            saved_response.pop("messages")
            omitted.append("messages")
        safe_response = _json_safe(saved_response)
        entry = {"schema": TRACE_SCHEMA, "sequence": None, "request": request,
                 "response": safe_response,
                 "response_sha256": _sha256_bytes(_canonical_json(safe_response)),
                 "ordered_response_sha256": _sha256_bytes(_ordered_json(safe_response)),
                 "omitted_response_fields": omitted}
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        entry["sequence"] = sequence
        with self._lock:
            with self.trace_path.open("a", encoding="utf-8") as stream:
                # Keep nested response key order intact: downstream frozen
                # prompts may serialize parsed JSON directly.
                stream.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        return entry


class ReplayLLMClient:
    """Strict ordered offline response replayer; it has no network fallback."""

    def __init__(self, trace_path: str | Path, *,
                 generation_params: Mapping[str, Any] | None = None,
                 image_backend: Any = None,
                 legacy_response_order_recovery: bool = False,
                 v1_root: Path | None = None,
                 progress_path: str | Path | None = None,
                 progress_context: Callable[[], Mapping[str, Any]] | None = None) -> None:
        self.trace_path = Path(trace_path)
        self.generation_params = dict(generation_params or {})
        self.image_backend = image_backend
        self._entries, self.recovery_receipt = _read_trace_with_receipt(
            self.trace_path,
            legacy_response_order_recovery=legacy_response_order_recovery,
            v1_root=v1_root,
        )
        from .request_audit import verify_call_completion
        try:
            self.recording_events_receipt = verify_call_completion(
                self.trace_path.with_name(self.trace_path.name + ".events.jsonl"), self._entries)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ReplayError("Request lifecycle evidence is incomplete or inconsistent") from exc
        self._index = 0
        self._lock = threading.Lock()
        self._progress = _ReplayProgressJournal(Path(progress_path)) if progress_path is not None else None
        self._progress_context = progress_context
        self._progress_violations = 0
        self._progress_write_error = False
        self._progress_final = None
        if self._progress is not None:
            self._progress_event("started", trace_sha256=self.recovery_receipt["trace_sha256"],
                                 recorded_requests=len(self._entries),
                                 legacy_response_order_recovery=legacy_response_order_recovery is True)

    def _progress_event(self, event: str, **values: Any) -> None:
        if self._progress is None:
            return
        self._progress_final = None
        if "method" in values and values["method"] not in {"chat", "chat_json"}:
            values["method"] = "UNKNOWN"
        row = {"event": event, "consumed_requests": self._index,
               "remaining_requests": len(self._entries) - self._index, **values}
        try:
            if self._progress_context is not None:
                context = self._progress_context()
                # Only fixed counter fields may cross into the journal.
                # Never persist URLs, request bodies, credentials, or errors.
                guard = {key: context.get(key) for key in
                         ("enabled", "blocked_attempt_count", "attempted")}
                if (not isinstance(guard["enabled"], bool) or not isinstance(guard["attempted"], bool)
                        or type(guard["blocked_attempt_count"]) is not int or guard["blocked_attempt_count"] < 0):
                    raise ValueError("Replay journal network context must contain only valid counters")
                row["network_guard"] = guard
            self._progress.append(row)
        except Exception as exc:
            self._progress_write_error = True
            raise ReplayError("Replay progress evidence could not be persisted") from exc

    @property
    def progress_receipt(self) -> dict[str, Any]:
        return {"enabled": self._progress is not None,
                "path": str(self._progress.path) if self._progress is not None else None,
                "events_written": self._progress.events_written if self._progress is not None else 0,
                "violation_count": self._progress_violations,
                "write_error": self._progress_write_error,
                "final_verification": self._progress_final,
                "scope": "Matched request prefix; no delivery, completion, or final network-counter claim."}

    def record_network_block(self) -> None:
        """Observe denied HTTP attempts even between ReplayClient calls."""
        with self._lock:
            self._progress_violations += 1
            self._progress_event("network_blocked")

    def finish_progress(self) -> dict[str, Any]:
        """Seal and re-read normal-end evidence; interruption leaves a prefix."""
        with self._lock:
            if self._progress is not None:
                self._progress_event("finished")
                try:
                    self._progress_final = self._progress.verify(
                        trace_sha256=self.recovery_receipt["trace_sha256"], entries=self._entries,
                        consumed=self._index, violation_count=self._progress_violations)
                except Exception as exc:
                    self._progress_write_error = True
                    raise ReplayError("Replay progress final verification failed") from exc
        return self.progress_receipt

    def chat(self, prompt, image_paths=None, system_prompt=None, messages=None, **kwargs):
        return self._call("chat", prompt, image_paths, system_prompt, messages, kwargs)

    def chat_json(self, prompt, image_paths=None, system_prompt=None, messages=None, **kwargs):
        return self._call("chat_json", prompt, image_paths, system_prompt, messages, kwargs)

    def _call(self, method, prompt, image_paths, system_prompt, messages, kwargs):
        images = list(image_paths or [])
        extra = dict(kwargs)
        call_context = extra.pop("trace_generation_params", {})
        try:
            params = _client_generation_params(self.image_backend, {**self.generation_params, **call_context})
            request = request_record(method, prompt, system_prompt=system_prompt, messages=messages,
                                     image_paths=images, generation_params=params,
                                     extra_parameters=extra, image_backend=self.image_backend)
        except Exception as exc:
            with self._lock:
                self._progress_violations += 1
                self._progress_event("request_build_error", method=method, error_type=type(exc).__name__)
            raise
        with self._lock:
            if self._index >= len(self._entries):
                self._progress_violations += 1
                self._progress_event("trace_exhausted", method=method,
                                     actual_fingerprint=request["fingerprint"])
                raise MissingRecordedResponse(f"trace exhausted at call {self._index}; network fallback is disabled")
            expected = self._entries[self._index]
            wanted = expected.get("request", {})
            if wanted.get("fingerprint") != request["fingerprint"]:
                self._progress_violations += 1
                self._progress_event("request_mismatch", method=method, trace_index=self._index,
                    expected_fingerprint=wanted.get("fingerprint"), actual_fingerprint=request["fingerprint"])
                raise RequestMismatch(
                    f"call {self._index} does not match the next recorded request "
                    f"(expected {wanted.get('fingerprint')}, got {request['fingerprint']}); trace position unchanged"
                )
            self._index += 1
            self._progress_event("request_matched", method=method, trace_index=self._index - 1,
                expected_fingerprint=wanted.get("fingerprint"), actual_fingerprint=request["fingerprint"],
                image_count=len(request.get("images", [])))
        response = copy.deepcopy(expected["response"])
        if "messages" in expected.get("omitted_response_fields", []):
            response["messages"] = _rebuild_backend_messages(request, response, images, self.image_backend)
        return response

    @property
    def consumed(self) -> int:
        return self._index

    @property
    def remaining(self) -> int:
        return len(self._entries) - self._index


def recording_client(inner: Any, trace_path: str | Path,
                     parameters: Mapping[str, Any] | None = None) -> TraceLLMClient:
    """Factory used by the demo worker's optional trace mode."""
    return TraceLLMClient(inner, trace_path, generation_params=parameters)


class ReplayClient(ReplayLLMClient):
    """Worker-facing alias with the same positional parameters contract."""

    def __init__(self, trace_path: str | Path, parameters: Mapping[str, Any] | None = None,
                 *, image_backend: Any = None, v1_root: Path | None = None,
                 legacy_response_order_recovery: bool = False,
                 progress_path: str | Path | None = None,
                 progress_context: Callable[[], Mapping[str, Any]] | None = None) -> None:
        if image_backend is None:
            frozen_v1 = v1_root or Path(__file__).resolve().parents[1] / "baselines/current/source_snapshot/v1"
            image_backend = load_frozen_qwen_image_backend(frozen_v1, parameters)
        super().__init__(trace_path, generation_params=parameters, image_backend=image_backend,
                         legacy_response_order_recovery=legacy_response_order_recovery,
                         v1_root=v1_root, progress_path=progress_path, progress_context=progress_context)


class _ImageEncoderBackend:
    """Backend-shaped carrier for the frozen client image encoder."""

    def __init__(self, encoder, params: Mapping[str, Any] | None = None):
        self.encode_image_b64 = encoder
        for key, value in (params or {}).items():
            setattr(self, key, value)


def load_frozen_qwen_image_backend(v1_root: Path, params: Mapping[str, Any] | None = None) -> Any:
    """Load only the image encoder from the frozen v1 client module; no API client is created."""
    module, _ = _load_frozen_qwen_module(v1_root)
    return _ImageEncoderBackend(module.encode_image_b64, params)


def _load_frozen_qwen_module(v1_root: Path):
    module_path = Path(v1_root) / "src/pathxai_agent/llm/qwen_client.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"frozen qwen_client.py not found under {v1_root}")
    spec = importlib.util.spec_from_file_location("pathxai_frozen_qwen_client", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load frozen Qwen client parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module_path


def _read_trace(path: Path) -> list[dict[str, Any]]:
    """Compatibility reader; legacy recovery is deliberately opt-in elsewhere."""
    entries, _ = _read_trace_with_receipt(path)
    return entries


def _read_trace_with_receipt(
    path: Path,
    *,
    legacy_response_order_recovery: bool = False,
    v1_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parser = None
    parser_sha256 = None
    if legacy_response_order_recovery:
        frozen_root = v1_root or Path(__file__).resolve().parents[1] / "baselines/current/source_snapshot/v1"
        parser, parser_path = _load_frozen_qwen_module(frozen_root)
        parser_sha256 = _sha256_bytes(parser_path.read_bytes())
    entries = []
    receipt_rows = []
    recovered_count = 0
    raw_trace = path.read_bytes()
    # Parse and hash the same buffer. Split only LF, preserving literal Unicode
    # line separators inside JSON strings and handling CRLF as JSON whitespace.
    for lineno, line in enumerate(raw_trace.decode("utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"invalid JSONL trace at line {lineno}") from exc
        if entry.get("schema") != TRACE_SCHEMA or "request" not in entry or "response" not in entry:
            raise ReplayError(f"unsupported or incomplete trace entry at line {lineno}")
        response = entry["response"]
        semantic_sha = _sha256_bytes(_canonical_json(response))
        recorded_semantic_sha = entry.get("response_sha256")
        if recorded_semantic_sha != semantic_sha:
            raise ReplayError(f"recorded response checksum mismatch at line {lineno}")
        recorded_ordered_sha = entry.get("ordered_response_sha256")
        raw_ordered_sha = _sha256_bytes(_ordered_json(response))
        if recorded_ordered_sha is not None and recorded_ordered_sha != raw_ordered_sha:
            raise ReplayError(f"recorded ordered response checksum mismatch at line {lineno}")
        recovered = False
        recovery_source = None
        # Historical JSONL used recursive key sorting. Restore JSON object
        # insertion order only with explicit opt-in and only when the
        # frozen client parser recreates the exact recorded semantics.
        if (legacy_response_order_recovery and recorded_ordered_sha is None
                and entry.get("request", {}).get("method") == "chat_json"
                and isinstance(response.get("json"), dict)):
            content = response.get("content")
            if "content" not in response or not isinstance(content, str):
                raise ReplayError(f"legacy order recovery requires a raw content string at line {lineno}")
            parsed = parser.extract_json_object(content)
            recovery_source = "content"
            if parsed is None:
                reasoning = response.get("reasoning")
                if not isinstance(reasoning, str) or not reasoning:
                    raise ReplayError(f"legacy order recovery has no parseable content or reasoning at line {lineno}")
                parsed = parser.extract_json_object(reasoning)
                recovery_source = "reasoning"
            if not isinstance(parsed, dict):
                raise ReplayError(f"legacy order recovery could not parse a JSON object at line {lineno}")
            if _canonical_json(parsed) != _canonical_json(response["json"]):
                raise ReplayError(f"legacy recovered JSON semantics differ from recorded response at line {lineno}")
            response["json"] = parsed
            if _sha256_bytes(_canonical_json(response)) != recorded_semantic_sha:
                raise ReplayError(f"legacy recovered response semantics differ at line {lineno}")
            recovered = True
            recovered_count += 1
        receipt_rows.append({
            "sequence": entry.get("sequence"),
            "semantic_response_sha256": recorded_semantic_sha,
            "ordered_response_sha256": raw_ordered_sha,
            "recovered_ordered_response_sha256": _sha256_bytes(_ordered_json(response)) if recovered else None,
            "recovered": recovered,
            "recovery_source": recovery_source,
        })
        request = entry["request"]
        fingerprint = request.get("fingerprint")
        core = {key: value for key, value in request.items() if key != "fingerprint"}
        if fingerprint != _sha256_bytes(_canonical_json(core)):
            raise ReplayError(f"recorded request checksum mismatch at line {lineno}")
        entries.append(entry)
    entries.sort(key=lambda item: item.get("sequence", -1))
    if [entry.get("sequence") for entry in entries] != list(range(len(entries))):
        raise ReplayError("trace sequence discontinuity or duplicate call sequence")
    receipt = {
        "enabled": bool(legacy_response_order_recovery),
        "parser_sha256": parser_sha256,
        "recovered_count": recovered_count,
        "entries": receipt_rows,
        "trace_sha256": _sha256_bytes(raw_trace),
        "trace_size_bytes": len(raw_trace),
    }
    return entries, receipt


def _rebuild_backend_messages(request: Mapping[str, Any], response: Mapping[str, Any],
                              image_paths: list[str | Path], image_backend: Any) -> list[dict[str, Any]]:
    """Rebuild QwenVisionClient's return-only message history without a network call."""
    params = request["generation_params"]
    messages = copy.deepcopy(request.get("messages") or [])
    system = request.get("system_prompt")
    if system and params.get("use_system_role"):
        messages.append({"role": "system", "content": system})
    content: list[dict[str, Any]] = []
    if system and not params.get("use_system_role"):
        content.append({"type": "text", "text": system})
    content.append({"type": "text", "text": request["prompt"]})
    if image_paths:
        encoder, leaf = _image_encoder(image_backend)
        if encoder is None:
            raise ReplayError("cannot rebuild returned message history without the frozen image encoder")
        max_images = params.get("max_images")
        used_paths = image_paths[:max_images] if max_images is not None else image_paths
        for image_path in used_paths:
            encoded = encoder(image_path,
                              max_edge=params.get("image_max_edge", getattr(leaf, "image_max_edge", None)),
                              jpeg_quality=params.get("image_jpeg_quality", getattr(leaf, "image_jpeg_quality", 85)))
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    messages.append({"role": "user", "content": content})
    messages.append({"role": "assistant", "content": response.get("content", "") or ""})
    return messages


def compare_values(expected: Any, actual: Any, *, ignore_pointers: Iterable[str] = (),
                   prompt_texts: Iterable[str] = ()) -> list[str]:
    """Return structural mismatch paths; ignored values must be proven prompt-independent."""
    ignored = set(ignore_pointers)
    prompts = tuple(p for p in prompt_texts if p)
    for pointer in ignored:
        if not pointer.startswith("/"):
            raise ValueError(f"ignore paths must be JSON Pointers: {pointer}")
        field = pointer.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~").lower()
        if field not in _TIMESTAMP_FIELD_NAMES and field not in _PATH_FIELD_NAMES:
            raise ValueError(f"only timestamp/resource path fields may be ignored: {pointer}")
        left = _json_pointer_get(expected, pointer)
        right = _json_pointer_get(actual, pointer)
        if not prompts and (left is not _MISSING or right is not _MISSING):
            raise ValueError(f"prompt text is required to establish ignored field is prompt-independent: {pointer}")
        for value in (left, right):
            if isinstance(value, str) and any(value and value in text for text in prompts):
                raise ValueError(f"ignored field value appears in a prompt: {pointer}")
    mismatches: list[str] = []

    def walk(left: Any, right: Any, pointer: str) -> None:
        if pointer in ignored:
            return
        if type(left) is not type(right):
            mismatches.append(pointer or "/")
            return
        if isinstance(left, dict):
            keys = sorted(set(left) | set(right))
            for key in keys:
                child = pointer + "/" + str(key).replace("~", "~0").replace("/", "~1")
                if child in ignored:
                    continue
                if key not in left or key not in right:
                    mismatches.append(child)
                else:
                    walk(left[key], right[key], child)
        elif isinstance(left, list):
            if len(left) != len(right):
                mismatches.append(pointer + "/length")
            for index, (a, b) in enumerate(zip(left, right)):
                walk(a, b, f"{pointer}/{index}")
        elif left != right:
            mismatches.append(pointer or "/")

    walk(expected, actual, "")
    return mismatches


_MISSING = object()


def _json_pointer_get(value: Any, pointer: str) -> Any:
    current = value
    for raw in pointer.lstrip("/").split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        try:
            current = current[int(key)] if isinstance(current, list) else current[key]
        except (KeyError, IndexError, ValueError, TypeError):
            return _MISSING
    return current


def compare_files(expected_path: Path, actual_path: Path, *, ignore_pointers: Iterable[str] = (),
                  prompt_texts: Iterable[str] = ()) -> list[str]:
    expected, actual = _load_value(expected_path), _load_value(actual_path)
    return compare_values(expected, actual, ignore_pointers=ignore_pointers,
                          prompt_texts=prompt_texts)


def _load_value(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _key_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            found.append(path)
            found.extend(_key_paths(child, path))
    elif isinstance(value, list):
        for child in value[:1]:  # structural key scan only; no case text is emitted
            found.extend(_key_paths(child, prefix + "[]"))
    return found


def audit_case_artifacts(case_dir: Path) -> dict[str, Any]:
    """Inspect only top-level files in one case output; never return values or case id."""
    root = case_dir
    files = sorted(p for p in root.iterdir() if p.is_file())
    candidate_names = [p.name for p in files if re.search(r"(prompt|request|response|trace|llm.?log|cache)", p.name, re.I)]
    trace_key_paths: list[str] = []
    json_count = 0
    for path in files:
        if path.suffix.lower() != ".json":
            continue
        json_count += 1
        try:
            keys = _key_paths(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
        for key_path in keys:
            if re.search(r"(^|\.)(prompt|request|response|messages|raw_request|raw_response)(\.|\[|$)", key_path, re.I):
                trace_key_paths.append(f"{path.name}:{key_path}")
    return {
        "artifact_file_count": len(files),
        "json_file_count": json_count,
        "request_response_named_files": candidate_names,
        "request_response_key_paths": sorted(set(trace_key_paths)),
        "full_historical_replay_supported": False,
        "reason": "no complete ordered request/response records with exact message, image, and generation metadata were found",
    }


def audit_run_sample(run_root: Path, variant: str) -> dict[str, Any]:
    """Choose one deterministic sample by directory ordering, without exposing its identifier."""
    root = run_root / variant
    if not root.is_dir():
        raise FileNotFoundError(f"variant output not found: {variant}")
    sample = None
    for cohort_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        children = sorted(p for p in cohort_dir.iterdir() if p.is_dir())
        if children:
            sample = children[0]
            break
    if sample is None:
        return {"variant": variant, "sample_found": False, "full_historical_replay_supported": False}
    return {"variant": variant, "sample_found": True, **audit_case_artifacts(sample)}


def replay_request_manifest(trace_path: Path, requests_path: Path,
                            image_backend: Any = None) -> int:
    requests = json.loads(requests_path.read_text(encoding="utf-8"))
    if not isinstance(requests, list):
        raise ValueError("request manifest must be a JSON array")
    replay = ReplayLLMClient(trace_path, image_backend=image_backend)
    for row in requests:
        method = row.get("method")
        if method not in {"chat", "chat_json"}:
            raise ValueError("each request method must be chat or chat_json")
        params = row.get("generation_params", {})
        replay._call(method, row["prompt"], row.get("image_paths"), row.get("system_prompt"),
                     row.get("messages"), {"trace_generation_params": params,
                                            **row.get("extra_parameters", {})})
    if replay.remaining:
        raise ReplayError(f"request manifest ended with {replay.remaining} recorded response(s) unused")
    return replay.consumed


def _read_prompt_files(paths: list[str]) -> list[str]:
    return [Path(p).read_text(encoding="utf-8") for p in paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    audit = subs.add_parser("audit-case", help="audit one existing case directory's file/field inventory")
    audit.add_argument("case_dir", type=Path)
    run_audit = subs.add_parser("audit-run", help="audit one deterministic sample from a run and variant")
    run_audit.add_argument("run_root", type=Path)
    run_audit.add_argument("variant")
    compare = subs.add_parser("compare", help="strictly compare JSON or text output files")
    compare.add_argument("expected", type=Path)
    compare.add_argument("actual", type=Path)
    compare.add_argument("--ignore", action="append", default=[], help="exact JSON Pointer to a timestamp/path field")
    compare.add_argument("--prompt-file", action="append", default=[], help="prompt text used to prove ignored values are not prompt inputs")
    replay = subs.add_parser("replay", help="strictly replay request manifest without network access")
    replay.add_argument("trace", type=Path)
    replay.add_argument("requests", type=Path)
    replay.add_argument("--v1-root", type=Path,
                        default=Path(__file__).resolve().parents[1] / "baselines/current/source_snapshot/v1",
                        help="frozen v1 root used only to reproduce request image encoding")
    args = parser.parse_args(argv)
    if args.command == "audit-case":
        result = audit_case_artifacts(args.case_dir)
    elif args.command == "audit-run":
        result = audit_run_sample(args.run_root, args.variant)
    elif args.command == "compare":
        mismatches = compare_files(args.expected, args.actual, ignore_pointers=args.ignore,
                                   prompt_texts=_read_prompt_files(args.prompt_file))
        result = {"equal": not mismatches, "mismatches": mismatches}
    else:
        request_rows = json.loads(args.requests.read_text(encoding="utf-8"))
        first_params = request_rows[0].get("generation_params", {}) if request_rows else {}
        backend = load_frozen_qwen_image_backend(args.v1_root, first_params) if request_rows else None
        count = replay_request_manifest(args.trace, args.requests, backend)
        result = {"matched_requests": count, "network_used": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if args.command == "compare" and not result["equal"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
