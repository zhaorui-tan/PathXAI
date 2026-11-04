"""Observable model file imports/downloads, separate from scientific tasks."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import time
import uuid

from .assets import (asset_catalog, download_asset, download_keep_dataset,
                     import_asset, import_keep_dataset)
from .config import atomic_json, redact
from .jobs import now

ASSET_CONFIG_KEYS = {"seg_d1": "seg_ckpt", "sam3": "sam3_checkpoint",
                     "conch": "foundation_model_path", "keep_dataset": "keep_root"}


class AssetTasks:
    def __init__(self, workspace: Path, config):
        self.root = workspace / "asset_operations"
        self.root.mkdir(parents=True, exist_ok=True)
        self.models = workspace / "models"
        self.config = config
        self._lock = threading.RLock()
        self._items = {}
        self._cancel = {}
        self._queue = queue.Queue()
        self._closed = False
        self._close_enqueued = False
        self._shutdown_timed_out = False
        self._active_operation_id: str | None = None
        for path in self.root.glob("*.json"):
            try:
                item = json.loads(path.read_text())
                if item.get("status") in {"queued", "running"}:
                    item.update(status="interrupted", message="Previous file operation was interrupted. Start a new operation.")
                    atomic_json(path, item)
                self._items[item["id"]] = item
            except (OSError, ValueError, KeyError):
                continue
        self._worker = threading.Thread(target=self._work, name="pathxai-model-files", daemon=True)
        self._worker.start()

    def list(self):
        with self._lock:
            return json.loads(json.dumps(sorted(self._items.values(), key=lambda row: row["created_at"], reverse=True)))

    def get(self, operation_id):
        with self._lock:
            if operation_id not in self._items:
                raise KeyError("Model operation not found.")
            return json.loads(json.dumps(self._items[operation_id]))

    def start(self, asset_id, action, path=None):
        catalog = {row["id"]: row for row in asset_catalog()}
        if asset_id not in catalog or asset_id not in ASSET_CONFIG_KEYS:
            raise ValueError("Choose a supported model asset.")
        if action not in {"import", "download"}:
            raise ValueError("Unsupported model action.")
        if action == "import":
            supplied = Path(path).expanduser() if isinstance(path, str) else None
            is_valid_source = (supplied is not None and
                               (supplied.is_dir() if asset_id == "keep_dataset" else supplied.is_file()))
            if not is_valid_source:
                raise ValueError("Choose the folder containing both KEEP files." if asset_id == "keep_dataset"
                                 else "Choose an existing local model file.")
        if action == "download" and not catalog[asset_id]["can_download"]:
            raise ValueError("This model has no configured public download. Import an authorized local copy.")
        with self._lock:
            if self._closed:
                raise ValueError("Application is shutting down.")
            if any(row["asset_id"] == asset_id and row["status"] in {"queued", "running"}
                   for row in self._items.values()):
                raise ValueError("A file operation for this model is already in progress.")
            operation_id = uuid.uuid4().hex[:16]
            item = {"id": operation_id, "asset_id": asset_id, "action": action, "status": "queued",
                    "created_at": now(), "updated_at": now(), "message": "Waiting for model file worker.",
                    "downloaded": 0, "total": catalog[asset_id]["bytes"]}
            self._items[operation_id] = item
            self._cancel[operation_id] = threading.Event()
            atomic_json(self.root / (operation_id + ".json"), item)
            self._queue.put((operation_id, path, str(self.config.snapshot().get("hf_token") or "")))
            return self.get(operation_id)

    def cancel(self, operation_id):
        with self._lock:
            item = self.get(operation_id)
            flag = self._cancel.get(operation_id)
            if flag and item["status"] in {"queued", "running"}:
                flag.set()
            return item

    def _update(self, operation_id, value, token):
        with self._lock:
            item = self._items[operation_id]
            if self._shutdown_timed_out and item.get("cleanup_status") == "unconfirmed":
                return
            item.update(redact(value, (token,)))
            item["updated_at"] = now()
            atomic_json(self.root / (operation_id + ".json"), item)

    def _work(self):
        while True:
            request = self._queue.get()
            if request is None:
                self._queue.task_done()
                return
            operation_id, path, token = request
            try:
                item = self.get(operation_id)
                flag = self._cancel[operation_id]
                if flag.is_set():
                    self._update(operation_id, {"status": "cancelled", "message": "Cancelled before file transfer."}, token)
                    continue
                with self._lock:
                    if flag.is_set() or self._closed:
                        self._update(operation_id, {"status": "cancelled",
                                                    "message": "Cancelled before file transfer."}, token)
                        continue
                    self._active_operation_id = operation_id
                self._update(operation_id, {"status": "running", "message": "Preparing model file."}, token)
                callback = lambda value: self._update(operation_id, value, token)
                if item["asset_id"] == "keep_dataset":
                    keep_target = self.root.parent / "keep"
                    if item["action"] == "import":
                        result = import_keep_dataset(path, keep_target, progress=callback, cancel_event=flag)
                    else:
                        result = download_keep_dataset(keep_target, token=token,
                                                       progress=callback, cancel_event=flag)
                    if not result.get("ok") or len(result.get("files") or []) != 2:
                        raise ValueError("Both KEEP files must pass pinned verification before configuration.")
                    configured_path = str(result["path"])
                elif item["action"] == "import":
                    result = import_asset(item["asset_id"], path, self.models, progress=callback, cancel_event=flag)
                    configured_path = str(result["path"])
                else:
                    result = download_asset(item["asset_id"], self.models, token=token, progress=callback, cancel_event=flag)
                    configured_path = str(result["path"])
                if flag.is_set():
                    self._update(operation_id, {"status": "cancelled", "message": "File operation cancelled."}, token)
                    continue
                # Serialize the final cancellation check with config commit.
                # A cancel request arriving during the final KEEP hash must
                # either win before configuration changes or observe a
                # completed operation; it cannot leave a cancelled task with
                # its resource configured.
                with self._lock:
                    if flag.is_set():
                        self._update(operation_id, {"status": "cancelled",
                                                    "message": "File operation cancelled."}, token)
                        continue
                    self.config.update({ASSET_CONFIG_KEYS[item["asset_id"]]: configured_path})
                    self._update(operation_id, {"status": "succeeded",
                                                "message": "Model file verified and configured.",
                                                "result": result}, token)
            except Exception as exc:
                cancelled = self._cancel[operation_id].is_set()
                self._update(operation_id, {"status": "cancelled" if cancelled else "failed", "message": str(exc)}, token)
            finally:
                with self._lock:
                    if self._active_operation_id == operation_id:
                        self._active_operation_id = None
                self._queue.task_done()

    def close(self):
        with self._lock:
            self._closed = True
            for operation_id, flag in self._cancel.items():
                flag.set()
                item = self._items.get(operation_id)
                if item and item.get("status") == "queued":
                    item.update(status="cancelled", message="Cancelled before file transfer during application shutdown.",
                                updated_at=now())
                    atomic_json(self.root / (operation_id + ".json"), item)
            enqueue = not self._close_enqueued
            self._close_enqueued = True
        if enqueue:
            self._queue.put(None)

    def wait_until(self, deadline: float) -> bool:
        self._worker.join(timeout=max(0.0, deadline - time.monotonic()))
        if not self._worker.is_alive():
            return True
        with self._lock:
            self._shutdown_timed_out = True
            active = self._active_operation_id
            for operation_id, item in self._items.items():
                if operation_id == active or item.get("status") in {"queued", "running"}:
                    message = "Application shutdown deadline expired; file-operation cleanup could not be confirmed."
                    item.update(status="interrupted", message=message, error=message,
                                cleanup_status="unconfirmed", updated_at=now())
                    atomic_json(self.root / (operation_id + ".json"), item)
        return False

    def shutdown_diagnostics(self) -> dict:
        with self._lock:
            return {"worker_alive": self._worker.is_alive(),
                    "active_operation_id": self._active_operation_id,
                    "unconfirmed_operation_ids": [item["id"] for item in self._items.values()
                                                  if item.get("cleanup_status") == "unconfirmed"]}
