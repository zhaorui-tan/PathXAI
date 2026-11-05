"""Private request lifecycle evidence for future runs; no network behavior changes.

Events contain clinical text. They are not anonymized or intended for publication.
Credentials, URLs and exception messages are omitted. Image bodies are hashed.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
import time

_active = ContextVar("pathxai_request_audit", default=None)
_install_lock = threading.Lock()
_installed = False


class AuditWriteError(RuntimeError):
    """A request could not be recorded durably; do not silently continue."""


class RequestJournal:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     getattr(os, "O_NOFOLLOW", 0), 0o600)
        info = os.fstat(fd)
        self.identity = info.st_dev, info.st_ino
        os.close(fd)
        self._lock = threading.Lock()
        self._sequence = 0

    def append(self, event, **fields):
        with self._lock:
            try:
                fd = os.open(self.path, os.O_APPEND | os.O_WRONLY |
                             getattr(os, "O_NOFOLLOW", 0))
                with os.fdopen(fd, "wb") as stream:
                    info = os.fstat(stream.fileno())
                    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                            or (info.st_dev, info.st_ino) != self.identity):
                        raise AuditWriteError("Request journal identity changed")
                    row = {"schema": "pathxai-request-events-v1", "sequence": self._sequence,
                           "utc": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
                    stream.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode())
                    stream.flush()
                    os.fsync(stream.fileno())
                self._sequence += 1
            except (OSError, TypeError, ValueError) as exc:
                raise AuditWriteError("Unable to persist request audit event") from exc


@contextmanager
def audit_call(journal, call_id):
    token = _active.set({"journal": journal, "call_id": call_id, "attempt": 0})
    try:
        yield
    finally:
        _active.reset(token)


def _emit(event, **fields):
    context = _active.get()
    if context is not None:
        context["journal"].append(event, call_id=context["call_id"], **fields)


def _without_image_bytes(value):
    if isinstance(value, dict):
        return {k: _without_image_bytes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_without_image_bytes(v) for v in value]
    if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
        header, encoded = value.split(",", 1)
        try:
            raw = base64.b64decode(encoded, validate=True)
        except ValueError:
            return {"image_payload": "invalid_base64", "encoded_sha256": hashlib.sha256(encoded.encode()).hexdigest()}
        return {"image_media_type": header[5:].split(";")[0],
                "image_sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return value


def _body_record(body):
    # Never consume request streams/generators: the real client sends JSON bytes.
    if isinstance(body, str):
        body = body.encode()
    if not isinstance(body, bytes):
        return {"body_capture": "unavailable_streaming_body"}
    record = {"body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes": len(body)}
    try:
        record["body_json"] = _without_image_bytes(json.loads(body))
    except (ValueError, UnicodeDecodeError):
        record["body_capture"] = "non_json_hash_only"
    return record


def verify_call_completion(path, entries):
    """Reject incomplete/failed calls when replaying a new journalled trace.

    Older traces have no sidecar and retain their existing replay contract.
    This detects accidental truncation, not deliberate removal or alteration.
    """
    path = Path(path)
    if not path.exists():
        return {"status": "legacy_no_event_journal"}
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("Request event journal is empty or truncated")
    calls = {}
    rows = [json.loads(line) for line in raw.splitlines()]
    for index, row in enumerate(rows):
        if row.get("schema") != "pathxai-request-events-v1" or row.get("sequence") != index:
            raise ValueError("Request event journal sequence/schema differs")
        call_id, event = row.get("call_id"), row.get("event")
        if type(call_id) is not int or call_id < 0:
            raise ValueError("Request event journal has an invalid call ID")
        if event == "call_started":
            if call_id in calls:
                raise ValueError("Request event journal repeats a call")
            calls[call_id] = None
        elif call_id not in calls or calls[call_id] is not None:
            raise ValueError("Request event journal has an invalid call lifecycle")
        elif event == "call_failed":
            raise ValueError("Request journal contains a failed call; failure replay is not supported")
        elif event == "call_succeeded":
            calls[call_id] = row.get("response_sha256")
    if sorted(calls) != list(range(len(entries))):
        raise ValueError("Request journal and successful trace contain different calls")
    for entry in entries:
        if not calls[entry["sequence"]] or calls[entry["sequence"]] != entry["response_sha256"]:
            raise ValueError("Request journal contains an unfinished call or mismatched response")
    return {"status": "complete_calls_verified", "calls": len(calls),
            "events": len(rows), "sha256": hashlib.sha256(raw).hexdigest()}


def install_http_audit_hooks():
    """Observe requests/urllib3 calls only inside an audit_call context.

    A pool attempt records body/settings before send, then status or exception
    class. Retry decisions include exhausted retries. Session responses include
    already-loaded JSON; intermediate retry response bodies are not consumed.
    """
    global _installed
    with _install_lock:
        if _installed:
            return
        import requests
        from urllib3.connectionpool import HTTPConnectionPool
        from urllib3.util.retry import Retry

        make_request = HTTPConnectionPool._make_request
        increment = Retry.increment
        send = requests.Session.send

        def observed_request(pool, conn, method, url, *args, **kwargs):
            context = _active.get()
            if context is None:
                return make_request(pool, conn, method, url, *args, **kwargs)
            context["attempt"] += 1
            attempt = context["attempt"]
            body = kwargs.get("body", args[0] if args else None)
            _emit("http_attempt_started", attempt=attempt, method=str(method), **_body_record(body))
            start = time.monotonic()
            try:
                response = make_request(pool, conn, method, url, *args, **kwargs)
            except AuditWriteError:
                raise
            except BaseException as exc:
                _emit("http_attempt_failed", attempt=attempt, error_type=type(exc).__name__,
                      elapsed_seconds=time.monotonic() - start)
                raise
            _emit("http_attempt_returned", attempt=attempt, status=response.status,
                  elapsed_seconds=time.monotonic() - start)
            return response

        def observed_increment(retry, method=None, url=None, response=None, error=None, *args, **kwargs):
            if _active.get() is None:
                return increment(retry, method, url, response, error, *args, **kwargs)
            fields = {"status": getattr(response, "status", None),
                      "error_type": type(error).__name__ if error is not None else None}
            try:
                updated = increment(retry, method, url, response, error, *args, **kwargs)
            except BaseException as exc:
                _emit("http_retry_exhausted", **fields, raised_type=type(exc).__name__)
                raise
            _emit("http_retry_scheduled", **fields)
            return updated

        def observed_send(session, request, **kwargs):
            if _active.get() is None:
                return send(session, request, **kwargs)
            _emit("http_request_started", method=request.method, **_body_record(request.body))
            try:
                response = send(session, request, **kwargs)
            except AuditWriteError:
                raise
            except BaseException as exc:
                _emit("http_request_failed", error_type=type(exc).__name__)
                raise
            content = getattr(response, "_content", False)
            _emit("http_response", status=response.status_code, **_body_record(content))
            return response

        HTTPConnectionPool._make_request = observed_request
        Retry.increment = observed_increment
        requests.Session.send = observed_send
        _installed = True
