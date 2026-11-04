"""Subprocess adapter for the original SO1/SO4 pipeline."""

from __future__ import annotations

import json
import hashlib
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .assets import AssetError, _catalog_map
from .profiles import config_defaults, get_profile, profile_names, validate_config
from .process_control import (WorkerCleanupError, capture_linux_group,
                              linux_worker_exited, terminate_linux_group)

def _checkpoint_provenance(path_value: str, asset_id: str) -> dict[str, Any]:
    """Verify selected bytes, including replacements preserving size/mtime."""
    path = Path(path_value).expanduser().resolve()
    stat = path.stat()
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    demo_root = Path(__file__).resolve().parents[1]
    manifest_path = demo_root / "baselines" / "current" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Archived runs identify the PathXAI segmentation checkpoint only.
        # Never let those hashes authenticate a differently-role-tagged asset
        # such as SAM3 weights or the tokenizer vocabulary.
        references = []
        if asset_id in {"pathxai_seg", "seg_d1"}:
            references = [
                {"name": item.get("run_id"), "sha256": (item.get("checkpoint") or {}).get("sha256")}
                for item in manifest.get("archived_runs", [])
                if (item.get("checkpoint") or {}).get("sha256")
            ]
    except (OSError, ValueError, TypeError, AttributeError):
        references = []
    try:
        local_manifest = json.loads((demo_root / "assets" / "model_references.json").read_text(encoding="utf-8"))
        local_asset = next((item for item in local_manifest.get("assets", []) if item.get("id") == asset_id), None)
        if local_asset and local_asset.get("sha256"):
            references.append({"name": asset_id, "sha256": local_asset["sha256"]})
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    if asset_id == "sam3_bpe":
        try:
            local_manifest = json.loads((demo_root / "assets" / "manifest.json").read_text(encoding="utf-8"))
            local_asset = next((item for item in local_manifest.get("assets", []) if item.get("id") == asset_id), None)
            if local_asset and local_asset.get("sha256"):
                references.append({"name": asset_id, "sha256": local_asset["sha256"]})
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    matches = [item["name"] for item in references if item["sha256"] == digest]
    return {
        "path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "sha256": digest, "known_baseline_match": matches or None,
        "known_references_available": bool(references),
    }


def _asset_provenance(config: dict[str, Any]) -> dict[str, Any]:
    """Capture model assets by role; knowledge inputs are recorded separately."""
    return {
        "pathxai_seg": _checkpoint_provenance(str(config["seg_ckpt"]), "seg_d1"),
        "sam3": _checkpoint_provenance(str(config["sam3_checkpoint"]), "sam3"),
        "sam3_bpe": _checkpoint_provenance(str(config["sam3_bpe"]), "sam3_bpe"),
    }


def _pinned_asset_sizes_and_hashes() -> dict[str, tuple[int, str]]:
    """Load one exact role-specific byte count and digest per required asset."""
    try:
        catalogue = _catalog_map()
        expected = {}
        for provenance_key, asset_id in (("pathxai_seg", "seg_d1"),
                                         ("sam3", "sam3"),
                                         ("sam3_bpe", "sam3_bpe")):
            row = catalogue.get(asset_id)
            if not isinstance(row, dict):
                raise ValueError("missing role pin")
            size = row.get("bytes")
            digest = str(row.get("sha256") or "").lower()
            if (not isinstance(size, int) or isinstance(size, bool) or size <= 0
                    or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError("malformed role pin")
            expected[provenance_key] = (size, digest)
        return expected
    except (AssetError, OSError, TypeError, ValueError, KeyError) as exc:
        raise ValueError("Pinned asset references are unavailable") from exc


def _asset_identity_error(asset_provenance: dict[str, Any]) -> str | None:
    """Fail closed if any required role does not match its single catalog pin."""
    try:
        expected = _pinned_asset_sizes_and_hashes()
    except ValueError:
        return "Pinned model asset references are unavailable; the worker was not started."
    labels = {"pathxai_seg": "PathXAI-Seg", "sam3": "SAM3", "sam3_bpe": "SAM3 tokenizer"}
    mismatches = [labels[key] for key, (size, digest) in expected.items()
                  if not isinstance(asset_provenance.get(key), dict)
                  or asset_provenance[key].get("size_bytes") != size
                  or str(asset_provenance[key].get("sha256") or "").lower() != digest]
    if mismatches:
        return ("Selected asset does not match its pinned identity: " + ", ".join(mismatches)
                + ". The worker was not started.")
    return None


_KEEP_RESOURCE_FILES = ("disease_knowledge_graph.json", "pathology_vl_semantic_groups.json")


def _keep_resource_provenance(keep_root_value: str, *, required: bool) -> dict[str, Any]:
    """Record the two original KEEP JSON inputs without copying their contents."""
    if not required:
        return {"schema": "pathxai-keep-resources-v1", "required": False,
                "status": "not_required", "known_baseline_match": None,
                "baseline_match_status": "unknown", "files": {}}
    root = Path(keep_root_value).expanduser().resolve() if keep_root_value else None
    files: dict[str, Any] = {}
    for name in _KEEP_RESOURCE_FILES:
        path = (root / name) if root is not None else None
        record: dict[str, Any] = {"path": str(path) if path else None,
                                  "available": bool(path and path.is_file()),
                                  "size_bytes": None, "sha256": None}
        if path is not None and path.is_file():
            stat = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            record.update(size_bytes=stat.st_size, sha256=digest.hexdigest())
        files[name] = record
    complete = all(item["available"] and item["sha256"] for item in files.values())
    return {
        "schema": "pathxai-keep-resources-v1",
        "required": bool(required),
        "status": "verified" if complete else "incomplete",
        "known_baseline_match": None,
        "baseline_match_status": "unknown",
        "files": files,
    }


class EngineCancelled(RuntimeError):
    """Raised when the active scientific worker is cancelled."""


def build_command(config: dict[str, Any]) -> list[str]:
    """Build a shell-free command; the request and credentials travel on stdin."""
    python = str(config.get("pipeline_python") or "").strip()
    if not python:
        raise ValueError("pipeline_python must point to the configured pipeline interpreter")
    worker = Path(__file__).with_name("engine_worker.py")
    return [python, str(worker)]


def _pipeline_child_environment() -> dict[str, str]:
    """Build the isolated adapter environment with a reproducible hash seed."""
    env = os.environ.copy()
    for name in ("API_KEY", "HF_TOKEN", "OPENAI_API_KEY", "QWEN_API_KEY"):
        env.pop(name, None)
    for name in tuple(env):
        if name.startswith("PATHXAI_"):
            env.pop(name, None)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # Notebook kernels export their inline backend. Workers render to files and
    # may use a different Python environment without matplotlib-inline.
    env["MPLBACKEND"] = "Agg"
    return env


def _redact(value: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[redacted]")
    return value


def _redact_data(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, dict):
        return {key: _redact_data(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_data(item, secrets) for item in value]
    if isinstance(value, str):
        return _redact(value, secrets)
    return value


def _terminate_worker_tree(proc: subprocess.Popen, group_identity: dict | None = None) -> dict | None:
    """Stop only this task's owned process tree on the current platform."""
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=5, check=False)
            proc.wait(timeout=5)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        return
    return terminate_linux_group(proc, group_identity)


def run_engine(
    config: dict[str, Any],
    case: dict[str, Any],
    output_dir: Path,
    event: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    """Run one original-engine case and return a structured worker outcome.

    Core and Full call the original `run_single_case` with their frozen SO1/SO4
    ablation names. Heavy inference stays inside `pipeline_python`; WSI prep is
    performed there via the original v1 foundation ROI implementation.
    """
    config = {**config_defaults(), **(config or {})}
    mode = str(case.get("mode") or "core").lower()
    profile = get_profile(mode)
    output_dir = Path(output_dir).expanduser().resolve()
    source_roots = [Path(str(config.get(key))).expanduser().resolve() for key in
                    ("engine_root", "v1_root", "pathxai_root", "medsam3_root", "segmentation_source_root")
                    if config.get(key)]
    if any(output_dir == source or source in output_dir.parents for source in source_roots):
        return {"status": "failed", "case_id": str(case.get("case_id") or case.get("id") or ""),
                "mode": mode, "errors": ["output_dir must be outside all frozen source roots"]}
    case_id = str(case.get("case_id") or case.get("id") or "").strip()
    if (not case_id or Path(case_id).name != case_id or case_id in {".", ".."}
            or "/" in case_id or "\\" in case_id or ":" in case_id):
        return {"status": "failed", "case_id": case_id, "mode": mode, "errors": ["case_id must be a non-empty path-safe identifier"]}
    output_dir = output_dir / "cases" / case_id

    def cancelled_outcome() -> dict[str, Any]:
        return {"status": "cancelled", "case_id": case_id, "mode": mode,
                "ablation": profile["ablation"], "output_dir": str(output_dir), "errors": []}

    # A cancel can arrive while the app-level preflight is still running. In
    # that case avoid model hashing, cache setup, and scientific process start.
    if cancel_event is not None and cancel_event.is_set():
        return cancelled_outcome()
    output_dir.mkdir(parents=True, exist_ok=True)
    # LLM QC caches affect which external calls occur. Keep the initial cache
    # state empty for each new run so recording and strict replay have the same
    # call sequence, while keeping original cache algorithms untouched.
    config["shared_artifacts"] = str(output_dir / "runtime_artifacts")
    config["llm_cache_dir"] = str(output_dir / "runtime_artifacts" / "llm_cache")
    if event:
        event({"stage": "preflight", "message": "Checking engine configuration"})
    validation = validate_config({**config, "mode": mode})
    if cancel_event is not None and cancel_event.is_set():
        return cancelled_outcome()
    if not validation["ok"]:
        return {
            "status": "failed",
            "case_id": case.get("case_id"),
            "mode": mode,
            "ablation": profile["ablation"],
            "output_dir": str(output_dir),
            "errors": [row["message"] for row in validation["checks"] if not row["ok"]],
            "error": "; ".join(row["message"] for row in validation["checks"] if not row["ok"]),
            "validation": validation,
        }

    secrets = [str(config.get(name) or "") for name in ("api_key", "hf_token")]
    case = {**case, "mode": mode}
    if event:
        event({"stage": "asset_preflight", "message": "Verifying pinned model asset identities"})
    if cancel_event is not None and cancel_event.is_set():
        return cancelled_outcome()
    try:
        asset_provenance = _asset_provenance(config)
    except (OSError, ValueError, TypeError, KeyError):
        if cancel_event is not None and cancel_event.is_set():
            return cancelled_outcome()
        message = "Selected model assets or their role-specific references are invalid; the worker was not started."
        return {
            "status": "failed", "case_id": case_id, "mode": mode,
            "ablation": profile["ablation"], "output_dir": str(output_dir),
            "errors": [message], "error": message,
        }
    # Asset identity checks hash large local files. If cancellation arrived
    # during that read, honor it before checking identities or preparing a
    # subprocess request.
    if cancel_event is not None and cancel_event.is_set():
        return cancelled_outcome()
    identity_error = _asset_identity_error(asset_provenance)
    if cancel_event is not None and cancel_event.is_set():
        return cancelled_outcome()
    if identity_error:
        return {
            "status": "failed", "case_id": case.get("case_id"), "mode": mode,
            "ablation": profile["ablation"], "output_dir": str(output_dir),
            "errors": [identity_error], "error": identity_error,
        }
    request = {
        "config": config, "case": case, "mode": mode, "output_dir": str(output_dir),
        "asset_provenance": asset_provenance,
        "knowledge_assets": _keep_resource_provenance(
            str(config.get("keep_root") or ""), required=mode == "full"),
    }
    try:
        command = build_command(config)
        # These roots are also inserted explicitly by the worker. Do not put
        # credential material in environment variables or command arguments.
        env = _pipeline_child_environment()
        runtime_cache = Path(str(config.get("runtime_cache_dir") or output_dir.parent / "runtime_cache")).expanduser().resolve()
        runtime_dirs = {
            "NUMBA_CACHE_DIR": runtime_cache / "numba",
            "MPLCONFIGDIR": runtime_cache / "matplotlib",
            "TMPDIR": runtime_cache / "tmp",
            "TEMP": runtime_cache / "tmp",
            "TMP": runtime_cache / "tmp",
        }
        for name, directory in runtime_dirs.items():
            directory.mkdir(parents=True, exist_ok=True)
            env[name] = str(directory)
        for name in tuple(env):
            if name.startswith("PATHXAI_"):
                env.pop(name, None)
        # Cancellation can also arrive during request/runtime preparation.
        # Keep the owned-process termination path below for cancellations
        # that race with or follow Popen.
        if cancel_event is not None and cancel_event.is_set():
            return cancelled_outcome()
        popen_options: dict[str, Any] = {
            "cwd": config["engine_root"],
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "bufsize": 1,
            "env": env,
        }
        if os.name == "nt":
            popen_options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_options["start_new_session"] = True
        proc = subprocess.Popen(command, **popen_options)
        group_identity = capture_linux_group(proc) if os.name != "nt" else None
        try:
            proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        lines: queue.Queue[str | None] = queue.Queue()
        stderr_parts: list[str] = []

        def read_stream(stream, sink=None):
            try:
                for line in iter(stream.readline, ""):
                    if sink is None:
                        lines.put(line)
                    else:
                        sink.append(line)
            finally:
                stream.close()
                if sink is None:
                    lines.put(None)

        readers = [threading.Thread(target=read_stream, args=(proc.stdout,), daemon=True),
                   threading.Thread(target=read_stream, args=(proc.stderr, stderr_parts), daemon=True)]
        for reader in readers:
            reader.start()

        def finish_readers():
            for reader in readers:
                reader.join(timeout=2)
            if any(reader.is_alive() for reader in readers):
                raise WorkerCleanupError("Worker log pipes remain open; shutdown is unconfirmed.")
        stdout_done = False
        group_closed = False
        cleanup = None
        stderr_seen = 0
        last_event_at = time.monotonic()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                if not group_closed:
                    cleanup = _terminate_worker_tree(proc, group_identity)
                    group_closed = True
                finish_readers()
                return {"status": "cancelled", "case_id": case.get("case_id"), "mode": mode, "ablation": profile["ablation"], "output_dir": str(output_dir), "errors": [],
                        "worker_cleanup": cleanup}
            if os.name != "nt":
                # A crashed leader can leave a child holding its stdout pipe.
                # Observe the unreaped leader independently of pipe EOF, stop
                # remaining group members, then drain the already-read lines.
                if not group_closed and linux_worker_exited(group_identity):
                    cleanup = terminate_linux_group(proc, group_identity)
                    group_closed = True
                    finish_readers()
                if group_closed and stdout_done:
                    break
            elif stdout_done and proc.poll() is not None:
                break
            try:
                line = lines.get(timeout=0.2)
            except queue.Empty:
                if event and stderr_seen < len(stderr_parts):
                    for raw_log in stderr_parts[stderr_seen:]:
                        log_line = _redact(raw_log.rstrip(), secrets)
                        if log_line:
                            event({"stage": "engine_log", "message": log_line, "log": log_line})
                    stderr_seen = len(stderr_parts)
                if event and time.monotonic() - last_event_at > 1:
                    event({"stage": "running", "message": "Original engine is running"})
                    last_event_at = time.monotonic()
                continue
            if line is None:
                stdout_done = True
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event and "stage" in message:
                safe_message = {key: value for key, value in message.items() if key != "status"}
                safe_message["message"] = _redact(str(safe_message.get("message", "")), secrets)
                event(safe_message)
            if event and stderr_seen < len(stderr_parts):
                for raw_log in stderr_parts[stderr_seen:]:
                    log_line = _redact(raw_log.rstrip(), secrets)
                    if log_line:
                        event({"stage": "engine_log", "message": log_line, "log": log_line})
                stderr_seen = len(stderr_parts)
            if message.get("status") in {"succeeded", "failed"}:
                worker_outcome = message
        # Keep the unreaped Linux leader as the identity anchor until all
        # inherited workers have stopped, including after a worker failure.
        returncode = proc.wait()
        finish_readers()
        outcome = locals().get("worker_outcome")
        if outcome is None:
            stderr = _redact("".join(stderr_parts)[-12000:], secrets)
            return {
                "status": "failed",
                "case_id": case.get("case_id"),
                "mode": mode,
                "ablation": profile["ablation"],
                "output_dir": str(output_dir),
                "errors": [f"Engine worker exited with code {returncode}" + (f": {stderr}" if stderr else "")],
                "error": f"Engine worker exited with code {returncode}" + (f": {stderr}" if stderr else ""),
                "worker_cleanup": cleanup,
            }
        if outcome.get("error"):
            outcome["error"] = _redact(str(outcome["error"]), secrets)
        if outcome.get("traceback"):
            outcome["traceback"] = _redact(str(outcome["traceback"]), secrets)
        if returncode and outcome.get("status") == "succeeded":
            outcome["status"] = "failed"
            outcome["errors"] = [f"Engine worker exited with code {returncode}"]
        if cleanup is not None:
            outcome["worker_cleanup"] = cleanup
        return _redact_data(outcome, secrets)
    except EngineCancelled:
        return {"status": "cancelled", "case_id": case.get("case_id"), "mode": mode, "ablation": profile["ablation"], "output_dir": str(output_dir), "errors": []}

