"""Transport-only URL and authentication adaptation for the frozen Qwen client."""

from __future__ import annotations

import inspect
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def rewrite_request(
    request_url: str,
    headers: dict | None,
    *,
    base_url: str,
    api_key: str = "",
) -> tuple[str, dict]:
    """Rewrite only the transport destination and auth header.

    The request method, JSON/body, prompt, messages, and all model parameters
    remain owned by the original client. ``base_url`` may be an origin, API
    prefix, or a complete chat-completions endpoint.
    """
    base_text = base_url.strip()
    if "://" not in base_text:
        base_text = "http://" + base_text
    base = urlsplit(base_text)
    if base.scheme not in {"http", "https"} or not base.netloc:
        raise ValueError("qwen_api must be an http(s) URL with a host")
    req = urlsplit(request_url)
    route = "/v1/chat/completions"
    base_path = base.path.rstrip("/")
    if base_path.endswith(route):
        path = base.path
    elif base_path.endswith("/v1"):
        path = base_path + "/chat/completions"
    else:
        path = base_path + route
    rewritten = urlunsplit((base.scheme, base.netloc, path or route, req.query, ""))
    clean_headers = dict(headers or {})
    if api_key:
        # Avoid leaving a caller-supplied stale credential when a configured
        # session key is supplied; never include it in logs or results.
        for name in tuple(clean_headers):
            if name.lower() == "authorization":
                del clean_headers[name]
        clean_headers["Authorization"] = f"Bearer {api_key}"
    return rewritten, clean_headers


def install_requests_transport(base_url: str, api_key: str = "") -> None:
    """Patch requests.Session.post inside the worker, preserving the payload."""
    import requests

    original_post = requests.Session.post

    def adapted_post(session, url, *args, **kwargs):
        request_path = urlsplit(str(url)).path.rstrip("/")
        if not request_path.endswith("/v1/chat/completions"):
            return original_post(session, url, *args, **kwargs)
        headers = kwargs.pop("headers", None)
        new_url, new_headers = rewrite_request(
            str(url), headers, base_url=base_url, api_key=api_key
        )
        if headers is not None or api_key:
            kwargs["headers"] = new_headers
        return original_post(session, new_url, *args, **kwargs)

    requests.Session.post = adapted_post


def install_segmentation_cache_adapter(seg_visual_module, cache_root: str | Path) -> dict[str, str]:
    """Redirect frozen seg_visual defaults to a per-run cache directory.

    The original module binds ``DEFAULT_CACHE_ROOT`` into several function
    defaults at definition time and also reads it directly in one helper.
    Update both the module global and existing function objects so this also
    covers aliases imported before installation, without changing cache keys
    or image-generation code.
    """
    target = Path(cache_root).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    if not hasattr(seg_visual_module, "DEFAULT_CACHE_ROOT"):
        raise AttributeError("frozen seg_visual module has no DEFAULT_CACHE_ROOT")
    seg_visual_module.DEFAULT_CACHE_ROOT = target
    updated = []
    for name in (
        "build_roi_seg_composite",
        "build_global_seg_legend",
        "attach_composites_to_seg_context",
    ):
        function = getattr(seg_visual_module, name, None)
        if function is None:
            raise AttributeError(f"frozen seg_visual module has no {name}")
        signature = inspect.signature(function)
        parameters = list(signature.parameters.values())
        positional = [p for p in parameters if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD
        )]
        defaults = list(function.__defaults__ or ())
        first_default = len(positional) - len(defaults)
        for offset, parameter in enumerate(positional[first_default:]):
            if parameter.name == "cache_root":
                defaults[offset] = target
                updated.append(f"{name}.cache_root")
        if function.__defaults__ is not None:
            function.__defaults__ = tuple(defaults)
        keyword_defaults = dict(function.__kwdefaults__ or {})
        for parameter in parameters:
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.name == "cache_root":
                keyword_defaults[parameter.name] = target
                updated.append(f"{name}.cache_root")
        if keyword_defaults:
            function.__kwdefaults__ = keyword_defaults
    return {"cache_root": str(target), "patched_defaults": ",".join(updated)}


class ReplayNetworkGuard:
    """Block all requests-library HTTP entry points and count attempted calls."""

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self._blocked_attempt_count = 0
        self._observer = None

    def set_observer(self, callback) -> None:
        """Attach an optional counter-only observer; never supply HTTP data."""
        if callback is not None and not callable(callback):
            raise TypeError("Replay network observer must be callable")
        with self._lock:
            self._observer = callback

    def _block(self, method: object = "UNKNOWN") -> None:
        with self._lock:
            self._blocked_attempt_count += 1
            observer = self._observer
        if observer is not None:
            observer()
        safe_method = str(method).upper()
        if not safe_method.isalpha() or len(safe_method) > 16:
            safe_method = "UNKNOWN"
        raise RuntimeError(f"network disabled during strict replay ({safe_method})")

    def receipt(self) -> dict[str, object]:
        with self._lock:
            count = self._blocked_attempt_count
        return {"enabled": True, "blocked_attempt_count": count, "attempted": count > 0}


def install_requests_replay_guard() -> ReplayNetworkGuard:
    """Install a requests-level deny guard for strict replay only.

    Both the normal Session.request route and lower Session.send/adapter
    routes are blocked, so a client cannot bypass ReplayClient and reach HTTP.
    Recording mode does not call this function.
    """
    import requests

    guard = ReplayNetworkGuard()

    def blocked_request(session, method, url, *args, **kwargs):
        guard._block(method)

    def blocked_send(session, request, *args, **kwargs):
        guard._block(getattr(request, "method", "UNKNOWN"))

    def blocked_adapter_send(adapter, request, *args, **kwargs):
        guard._block(getattr(request, "method", "UNKNOWN"))

    requests.Session.request = blocked_request
    requests.Session.send = blocked_send
    requests.adapters.HTTPAdapter.send = blocked_adapter_send
    return guard


def install_segmentation_subprocess_adapter(
    medsam_module,
    *,
    bootstrap_script,
    v1_source,
    segmentation_runtime,
    medsam3_source,
    sam3_bpe,
    sam3_checkpoint,
    shared_artifacts,
) -> None:
    """Relocate only the frozen wrapper's segmentation-worker subprocess.

    The archived wrapper's scientific code and CLI arguments remain intact.
    Its hard-coded cwd/PYTHONPATH are replaced at the process boundary with
    the configured, hash-verified source roots.
    """
    import os
    import subprocess
    from types import SimpleNamespace

    original_run = subprocess.run
    source_paths = [str(v1_source), str(segmentation_runtime), str(medsam3_source)]

    def adapted_run(command, *args, **kwargs):
        if (isinstance(command, (list, tuple)) and len(command) >= 3
                and command[1] == "-m"
                and command[2] == "pathxai_agent.segmentation.medsam_roi"):
            command = [command[0], "-B", str(bootstrap_script), *command[3:]]
            child_env = dict(kwargs.get("env") or os.environ)
            child_env.update({
                "PATHXAI_V1_SOURCE": str(v1_source),
                "PATHXAI_SEG_RUNTIME": str(segmentation_runtime),
                "PATHXAI_MEDSAM3_SOURCE": str(medsam3_source),
                "PATHXAI_SAM3_BPE": str(sam3_bpe),
                "PATHXAI_SAM3_CHECKPOINT": str(sam3_checkpoint),
                "PATHXAI_SHARED_ARTIFACTS": str(shared_artifacts),
                "PYTHONDONTWRITEBYTECODE": "1",
            })
            existing = [item for item in child_env.get("PYTHONPATH", "").split(os.pathsep) if item]
            child_env["PYTHONPATH"] = os.pathsep.join(source_paths + existing)
            kwargs["env"] = child_env
            kwargs["cwd"] = str(segmentation_runtime)
        return original_run(command, *args, **kwargs)

    medsam_module.subprocess = SimpleNamespace(run=adapted_run)

