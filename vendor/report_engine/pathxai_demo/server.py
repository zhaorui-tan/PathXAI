"""Loopback-only local application server; no cloud component is involved."""

from __future__ import annotations

import argparse
import hmac
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import secrets
import signal
import threading
import time
from urllib.parse import parse_qs, quote, unquote, urlsplit
import webbrowser

from . import __version__
from .config import ConfigStore, atomic_json, redact, SECRET_KEYS
from .exporting import make_archive, prepare_report
from .jobs import JobManager, TERMINAL, validate_case
from .asset_tasks import AssetTasks, ASSET_CONFIG_KEYS

APP_ROOT = Path(__file__).resolve().parents[1]
MAX_BODY = 2 * 1024 * 1024


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    SHUTDOWN_TIMEOUT_SECONDS = 90.0

    def __init__(self, address, workspace: Path, *, runner=None, defaults=None, validator=None):
        if address[0] != "127.0.0.1":
            raise ValueError("PathXAI must bind only to 127.0.0.1.")
        from .engine import config_defaults, run_engine, validate_config
        self.workspace = workspace.resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.report_secret = secrets.token_bytes(32)
        self.config = ConfigStore(self.workspace, defaults if defaults is not None else config_defaults(workspace=self.workspace))
        self.validator = validator or validate_config

        def execute(config, case, out, event, cancel):
            outcome = run_engine(config, case, out, event, cancel)
            atomic_json(out / "scientific_outcome.json", outcome)
            if cancel.is_set() or outcome.get("status") != "succeeded":
                return outcome
            return prepare_report(config, case, out, outcome, event, cancel)

        self.jobs = JobManager(self.workspace, runner or execute)
        self.asset_tasks = AssetTasks(self.workspace, self.config)
        self.dialog_lock = threading.Lock()
        self.export_lock = threading.Lock()
        self.lifecycle_lock = threading.RLock()
        self.closing = False
        self.shutdown_deadline: float | None = None
        self.shutdown_reason: str | None = None
        self.shutdown_clean: bool | None = None
        self._shutdown_thread_started = False
        self._server_closed = False
        super().__init__(address, Handler)
        self.cookie_name = "PathXAI_" + str(self.server_port)

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.server_port}"

    @property
    def launch_url(self):
        return self.origin + "/?token=" + self.token

    def report_token(self, job_id: str):
        return hmac.new(self.report_secret, job_id.encode("utf-8"), "sha256").hexdigest()

    def request_exit(self):
        """Close an idle desktop session without abandoning a scientific worker."""
        with self.lifecycle_lock:
            if self.closing:
                raise ValueError("PathXAI is already closing.")
            jobs = self.jobs.list()
            if any(job.get("status") not in TERMINAL or job.get("cleanup_status") == "unconfirmed"
                   for job in jobs):
                raise ValueError("Finish or cancel active runs and verify their shutdown before quitting.")
            if any(item.get("status") in {"queued", "running"} for item in self.asset_tasks.list()):
                raise ValueError("Finish or cancel model file operations before quitting.")
            self.closing = True
            self.shutdown_reason = "user_quit"
            self.shutdown_deadline = time.monotonic() + self.SHUTDOWN_TIMEOUT_SECONDS
            # Starts use the same lifecycle lock, so no new work can enter
            # between the idle check and closing the two queues.
            self.jobs.close()
            self.asset_tasks.close()

    def begin_signal_shutdown(self, signum: int) -> None:
        """Stop admitting work, cancel owned tasks, then stop the HTTP loop."""
        with self.lifecycle_lock:
            if not self.closing:
                self.closing = True
                self.shutdown_reason = signal.Signals(signum).name
                self.shutdown_deadline = time.monotonic() + self.SHUTDOWN_TIMEOUT_SECONDS
                self.jobs.close()
                self.asset_tasks.close()
        self.schedule_shutdown("pathxai-signal-exit")

    def schedule_shutdown(self, name: str) -> None:
        with self.lifecycle_lock:
            if self._shutdown_thread_started:
                return
            self._shutdown_thread_started = True
        # BaseServer.shutdown() must run off the serve_forever thread.
        threading.Thread(target=self.shutdown, name=name, daemon=True).start()

    def server_close(self):
        if self._server_closed:
            return
        self._server_closed = True
        if self.shutdown_deadline is None:
            self.shutdown_deadline = time.monotonic() + self.SHUTDOWN_TIMEOUT_SECONDS
        self.jobs.close()
        self.asset_tasks.close()
        jobs_stopped = self.jobs.wait_until(self.shutdown_deadline)
        assets_stopped = self.asset_tasks.wait_until(self.shutdown_deadline)
        jobs_state = self.jobs.shutdown_diagnostics()
        assets_state = self.asset_tasks.shutdown_diagnostics()
        self.shutdown_clean = (jobs_stopped and assets_stopped
                               and not jobs_state["unconfirmed_job_ids"]
                               and not assets_state["unconfirmed_operation_ids"])
        state = {
            "status": "confirmed" if self.shutdown_clean else "unconfirmed",
            "reason": self.shutdown_reason or "server_close",
            "timeout_seconds": self.SHUTDOWN_TIMEOUT_SECONDS,
            "jobs_worker_stopped": jobs_stopped,
            "asset_worker_stopped": assets_stopped,
            "jobs": jobs_state,
            "asset_operations": assets_state,
        }
        try:
            atomic_json(self.workspace / "shutdown_status.json", state)
        except OSError:
            # Keep the process shutdown bounded even when its workspace is no
            # longer writable; active job records were already marked uncertain.
            self.shutdown_clean = False
        finally:
            super().server_close()


class Handler(BaseHTTPRequestHandler):
    server: LocalServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):
        # Request URLs may carry the initial authentication token or local paths.
        pass

    def _json(self, status: int, value):
        credentials = tuple(str(self.server.config.snapshot().get(key) or "") for key in SECRET_KEYS)
        content = json.dumps(redact(value, credentials), ensure_ascii=False).encode("utf-8")
        self._send(status, content, "application/json; charset=utf-8")

    def _send(self, status, data: bytes, content_type, *, headers=None, report=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        if report:
            self.send_header("Content-Security-Policy", "sandbox allow-scripts; default-src 'none'; "
                             "img-src 'self' data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
                             "font-src 'self' data:; connect-src 'none'; form-action 'none'; "
                             "base-uri 'none'; frame-ancestors 'none'")
        else:
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; "
                             "script-src 'self'; style-src 'self'; connect-src 'self'; "
                             "object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _authenticated(self):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            return False
        try:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            item = cookie.get(self.server.cookie_name)
            return bool(item and hmac.compare_digest(item.value, self.server.token))
        except (ValueError, TypeError):
            return False

    def _check(self, path, *, mutation=False):
        if not self._authenticated():
            self._json(401, {"error": "Open the app using the local launch URL."})
            return False
        if path.startswith("/api/") and not path.endswith("/export"):
            if self.headers.get("X-PathXAI-Client") != "1":
                self._json(403, {"error": "A local application request header is required."})
                return False
        origin = self.headers.get("Origin")
        if origin and origin != self.server.origin:
            self._json(403, {"error": "Cross-origin application requests are not permitted."})
            return False
        if mutation and self.headers.get("Sec-Fetch-Site") == "cross-site":
            self._json(403, {"error": "Cross-site application requests are not permitted."})
            return False
        return True

    def _body(self):
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
            raise ValueError("Send application/json.")
        size = int(self.headers.get("Content-Length", "0"))
        if size < 1 or size > MAX_BODY:
            raise ValueError("Request body is missing or too large.")
        value = json.loads(self.rfile.read(size))
        if not isinstance(value, dict):
            raise ValueError("Send a JSON object.")
        return value

    def _file(self, root: Path, relative: str, *, report=False, download=None):
        path = (root / unquote(relative)).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            self._json(404, {"error": "File not found."})
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        headers = {"Content-Disposition": 'attachment; filename="' + download + '"'} if download else {}
        self._send(200, path.read_bytes(), content_type, headers=headers, report=report)

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        split = urlsplit(self.path)
        path = split.path
        if path.startswith("/report-view/"):
            self._report_view(path)
            return
        supplied = parse_qs(split.query).get("token", [""])[0]
        if path == "/" and supplied:
            if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}" or not hmac.compare_digest(supplied, self.server.token):
                self._json(401, {"error": "Invalid launch URL."})
                return
            self._send(303, b"", "text/plain", headers={"Location": "/", "Set-Cookie":
                       f"{self.server.cookie_name}={self.server.token}; Path=/; HttpOnly; SameSite=Strict"})
            return
        if not self._check(path):
            return
        try:
            if path == "/":
                self._file(APP_ROOT / "static", "index.html")
            elif path.startswith("/static/"):
                self._file(APP_ROOT / "static", path[len("/static/"):])
            elif path == "/api/bootstrap":
                from .profiles import get_profile
                config = self.server.config.public()
                runtime_ready = all(Path(str(config.get(key) or "__missing__")).is_file()
                                    for key in ("pipeline_python", "seg_python"))
                checkpoint_ready = Path(str(config.get("seg_ckpt") or "__missing__")).is_file()
                keep = Path(str(config.get("keep_root") or "__missing__"))
                keep_ready = all((keep / name).is_file() for name in (
                    "disease_knowledge_graph.json", "pathology_vl_semantic_groups.json"))
                capabilities = [
                    {"name": "runtime", "label": "Local runtime", "status": "ready" if runtime_ready else "missing",
                     "message": "Available" if runtime_ready else "Configure runtime paths"},
                    {"name": "segmentation", "label": "PathXAI-Seg", "status": "ready" if checkpoint_ready else "missing",
                     "message": "Checkpoint available" if checkpoint_ready else "Checkpoint required"},
                    {"name": "keep", "label": "KEEP", "status": "ready" if keep_ready else "missing",
                     "message": "Available for Full" if keep_ready else "Required for Full"},
                ]
                self._json(200, {"version": __version__, "config": self.server.config.public(),
                                 "profiles": [get_profile("core"), get_profile("full")],
                                 "jobs": self.server.jobs.list(),
                                 "capabilities": capabilities})
            elif path == "/api/jobs":
                self._json(200, {"jobs": self.server.jobs.list()})
            elif path == "/api/assets":
                from .assets import asset_catalog
                config = self.server.config.public()
                catalog = []
                for row in asset_catalog():
                    if row["id"] not in ASSET_CONFIG_KEYS:
                        continue
                    selected_path = str(config.get(ASSET_CONFIG_KEYS[row["id"]]) or "")
                    catalog.append(dict(row, configured_path=selected_path,
                                        exists=bool(selected_path and Path(selected_path).is_file())))
                self._json(200, {"assets": catalog,
                                 "operations": self.server.asset_tasks.list()})
            elif path.startswith("/api/jobs/"):
                parts = path.strip("/").split("/")
                if len(parts) == 3:
                    self._json(200, self.server.jobs.get(parts[2]))
                elif len(parts) == 4 and parts[3] == "export":
                    job = self.server.jobs.get(parts[2])
                    if job["status"] != "succeeded":
                        raise ValueError("Only a completed task can be exported.")
                    with self.server.export_lock:
                        archive = make_archive(self.server.jobs.directory(parts[2]), job)
                    self._file(archive.parent, archive.name, download=job["case_id"] + "_PathXAI.zip")
                else:
                    self._json(404, {"error": "Route not found."})
            elif path.startswith("/reports/"):
                parts = path.split("/", 3)
                if len(parts) != 4:
                    raise ValueError("Invalid report path.")
                job = self.server.jobs.get(parts[2])
                if job["status"] != "succeeded":
                    raise ValueError("This task has no completed report.")
                relative = unquote(parts[3])
                # Only the portable viewing copy is served, never logs or raw manifests.
                if not relative.startswith("portable/report/"):
                    self._json(404, {"error": "File not found."})
                    return
                # The sandboxed report has an opaque origin, so browsers omit
                # the app's Strict cookie on image requests. A read-only,
                # per-report capability in the path lets relative resources
                # load without giving report scripts the app session.
                target = ("/report-view/" + quote(parts[2], safe="") + "/"
                          + self.server.report_token(parts[2]) + "/" + quote(relative, safe="/"))
                self._send(303, b"", "text/plain", headers={"Location": target})
            else:
                self._json(404, {"error": "Route not found."})
        except KeyError:
            self._json(404, {"error": "Task not found."})
        except (OSError, ValueError, RuntimeError) as exc:
            self._json(400, {"error": str(exc)})

    def _report_view(self, path):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self._json(401, {"error": "Invalid local report address."})
            return
        parts = path.split("/", 4)
        if len(parts) != 5 or not hmac.compare_digest(parts[3], self.server.report_token(parts[2])):
            self._json(401, {"error": "Reopen this report from the local application."})
            return
        try:
            job = self.server.jobs.get(parts[2])
            relative = unquote(parts[4])
            prefix = "portable/report/"
            if job["status"] != "succeeded" or not relative.startswith(prefix):
                self._json(404, {"error": "Report resource not found."})
                return
            self._file(self.server.jobs.directory(parts[2]) / "portable" / "report",
                       relative[len(prefix):], report=True)
        except (KeyError, OSError, ValueError):
            self._json(404, {"error": "Report resource not found."})

    def do_POST(self):
        path = urlsplit(self.path).path
        if not self._check(path, mutation=True):
            return
        try:
            body = self._body()
            if path == "/api/config":
                values = body["config"] if set(body) == {"config"} else body
                self._json(200, {"config": self.server.config.update(values)})
            elif path == "/api/preflight":
                config = self.server.config.snapshot()
                # Preview only recognized settings. Credentials never come back in a response.
                if "config" in body:
                    if not isinstance(body["config"], dict) or set(body["config"]) - set(config):
                        raise ValueError("Preflight configuration contains unknown fields.")
                    config.update(body["config"])
                result = self.server.validator(config)
                if "case" in body:
                    try:
                        case = validate_case(body["case"])
                        result.setdefault("checks", []).append({"name": "Case inputs", "ok": True,
                                                                 "message": "Local inputs are readable."})
                        if case["mode"] == "full":
                            root = Path(str(config.get("keep_root") or ""))
                            ready = all((root / item).is_file() for item in (
                                "disease_knowledge_graph.json", "pathology_vl_semantic_groups.json"))
                            result["checks"].append({"name": "KEEP", "ok": ready,
                                                     "message": "Full requires both fixed KEEP resources."})
                            result["ok"] = bool(result.get("ok") and ready)
                    except ValueError as exc:
                        result.setdefault("checks", []).append({"name": "Case inputs", "ok": False, "message": str(exc)})
                        result["ok"] = False
                self._json(200, result)
            elif path == "/api/api-probe":
                from .api_probe import probe_api
                values = body.get("config", {})
                allowed = {"qwen_api", "qwen_model", "api_key"}
                if set(body) - {"config"} or not isinstance(values, dict) or set(values) - allowed:
                    raise ValueError("API test accepts only endpoint, model, and session API key.")
                if any(not isinstance(value, str) for value in values.values()):
                    raise ValueError("API test values must be strings.")
                config = self.server.config.snapshot()
                config.update({key: value for key, value in values.items() if key != "api_key" or value})
                self._json(200, probe_api(config))
            elif path == "/api/jobs":
                with self.server.lifecycle_lock:
                    if self.server.closing:
                        raise ValueError("PathXAI is closing.")
                    job = self.server.jobs.submit(body, self.server.config.snapshot())
                self._json(201, job)
            elif path == "/api/shutdown":
                if body:
                    raise ValueError("Quit app does not accept configuration or task fields.")
                self.server.request_exit()
                try:
                    self._json(200, {"status": "closing", "message": "PathXAI is closing. You can close this tab."})
                finally:
                    # A tab closing before reading the response must not leave
                    # an unusable service holding the native instance lock.
                    self.server.schedule_shutdown("pathxai-desktop-exit")
            elif path.startswith("/api/jobs/") and path.endswith("/cancel"):
                parts = path.strip("/").split("/")
                if len(parts) != 4:
                    raise ValueError("Invalid task path.")
                self._json(200, self.server.jobs.cancel(parts[2]))
            elif path == "/api/dialog":
                self._json(200, self._dialog(body.get("kind")))
            elif path.startswith("/api/assets/"):
                parts = path.strip("/").split("/")
                if len(parts) == 4 and parts[3] in {"import", "download"}:
                    with self.server.lifecycle_lock:
                        if self.server.closing:
                            raise ValueError("PathXAI is closing.")
                        operation = self.server.asset_tasks.start(parts[2], parts[3], body.get("path"))
                    self._json(202, operation)
                elif len(parts) == 5 and parts[2] == "operations" and parts[4] == "cancel":
                    self._json(200, self.server.asset_tasks.cancel(parts[3]))
                else:
                    self._json(404, {"error": "Route not found."})
            else:
                self._json(404, {"error": "Route not found."})
        except KeyError:
            self._json(404, {"error": "Task not found."})
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            self._json(400, {"error": str(exc)})

    def _dialog(self, kind):
        if kind not in {"slides", "directory", "file"}:
            raise ValueError("Unknown file dialog type.")
        if not self.server.dialog_lock.acquire(blocking=False):
            return {"available": False, "paths": [], "error": "Another file picker is already open."}
        try:
            # Tk must own its own native event thread. The request thread owns this dialog.
            import tkinter as tk
            from tkinter import filedialog
            window = tk.Tk()
            window.withdraw()
            try:
                window.attributes("-topmost", True)
                if kind == "slides":
                    value = filedialog.askopenfilenames(parent=window, title="Select whole-slide images",
                                                         filetypes=[("Whole-slide images", "*.svs *.tif *.tiff *.ndpi *.mrxs *.scn"),
                                                                    ("All files", "*")])
                elif kind == "directory":
                    item = filedialog.askdirectory(parent=window, title="Select directory")
                    value = [item] if item else []
                else:
                    item = filedialog.askopenfilename(parent=window, title="Select file")
                    value = [item] if item else []
                return {"available": True, "paths": list(value)}
            finally:
                window.destroy()
        except Exception as exc:
            return {"available": False, "paths": [], "error": "Native picker unavailable. Paste local paths instead. " + str(exc)}
        finally:
            self.server.dialog_lock.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description="PathXAI local report application")
    parser.add_argument("--port", type=int, default=0, help="Local port (default: choose a free port)")
    parser.add_argument("--workspace", type=Path, default=APP_ROOT / "workspace")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    server = LocalServer(("127.0.0.1", args.port), args.workspace)
    old_handlers = install_shutdown_signal_handlers(server)
    try:
        print("PathXAI " + __version__, flush=True)
        print("Local workspace: " + str(server.workspace), flush=True)
        print("Open this private local URL: " + server.launch_url, flush=True)
        if not args.no_browser:
            threading.Timer(0.4, lambda: webbrowser.open(server.launch_url)).start()
        server.serve_forever(poll_interval=0.3)
    finally:
        try:
            server.server_close()
        finally:
            restore_shutdown_signal_handlers(old_handlers)
    return 0 if server.shutdown_clean else 2


def install_shutdown_signal_handlers(server: LocalServer) -> dict[int, object]:
    """Install process-local handlers; callers must restore them after close."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Shutdown signal handlers must be installed from the main thread.")
    previous = {}
    signals = [signal.SIGTERM, signal.SIGINT]
    if hasattr(signal, "SIGHUP"):
        signals.append(signal.SIGHUP)

    def request_shutdown(signum, _frame):
        try:
            print(f"Received {signal.Signals(signum).name}; cancelling active work and shutting down.", flush=True)
        except OSError:
            pass
        server.begin_signal_shutdown(signum)

    for signum in signals:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_shutdown)
    return previous


def restore_shutdown_signal_handlers(previous: dict[int, object]) -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Shutdown signal handlers must be restored from the main thread.")
    for signum, handler in previous.items():
        signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
