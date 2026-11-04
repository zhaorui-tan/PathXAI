"""Serial local task queue, with observable state independent of engine logic."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any, Callable
import uuid

from .config import atomic_json, redact, SECRET_KEYS
from .process_control import WorkerCleanupError

TERMINAL = frozenset({"succeeded", "failed", "cancelled", "interrupted"})
CHECKPOINT_STAGES = {
    "ckpt_01_roi_selection.json": ("roi_selection", "ROI selection saved."),
    "ckpt_02_site_prediction.json": ("site_context", "Sampling-site context saved."),
    "ckpt_03_segmentation.json": ("segmentation", "PathXAI-Seg evidence saved."),
    "ckpt_05_knowledge.json": ("knowledge", "KEEP context saved."),
    "ckpt_06_diagnosis.json": ("candidate_diagnoses", "Candidate diagnoses saved."),
    "ckpt_07_evidence_sufficiency.json": ("evidence_sufficiency", "Evidence-sufficiency assessment saved."),
    "ckpt_08_reinspection.json": ("reinspection", "Reinspection records saved."),
    "ckpt_09_evidence_graph.json": ("claim_verification", "Evidence graph and claim verification saved."),
    "ckpt_10_report.json": ("report_writing", "Report writing saved."),
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_case(case: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(case, dict):
        raise ValueError("Case must be a JSON object.")
    output = {key: case[key] for key in ("case_id", "mode", "slide_paths", "known_site", "roi_json")
              if key in case}
    case_id = str(output.get("case_id", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", case_id) or case_id in {".", ".."}:
        raise ValueError("Case ID must use 1–128 letters, numbers, dots, underscores or hyphens.")
    if output.get("mode") not in {"core", "full"}:
        raise ValueError("Choose Core or Full.")
    slides = output.get("slide_paths", [])
    if not isinstance(slides, list) or not all(isinstance(path, str) and path.strip() for path in slides):
        raise ValueError("slide_paths must be a list of local WSI paths.")
    if len(slides) > 100:
        raise ValueError("At most 100 WSI files may be associated with a case.")
    roi_json = output.get("roi_json")
    if not slides and not roi_json:
        raise ValueError("Choose at least one local WSI or an existing candidate ROI JSON.")
    for path in slides:
        if not Path(path).is_file():
            raise ValueError("WSI file not found: " + path)
    if roi_json and (not isinstance(roi_json, str) or not Path(roi_json).is_file()):
        raise ValueError("Candidate ROI JSON file not found.")
    output.update(case_id=case_id, slide_paths=slides, known_site=str(output.get("known_site", "")).strip())
    return output


class JobManager:
    def __init__(self, workspace: Path, runner: Callable, *, start_worker: bool = True):
        self.root = workspace / "runs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._close_enqueued = False
        self._shutdown_timed_out = False
        self._active_job_id: str | None = None
        self._load()
        self._worker = None
        if start_worker:
            self._worker = threading.Thread(target=self._work, name="pathxai-task-queue", daemon=True)
            self._worker.start()

    def _load(self) -> None:
        for path in self.root.glob("*/job.json"):
            try:
                job = json.loads(path.read_text(encoding="utf-8"))
                if not re.fullmatch(r"[a-f0-9]{16}", str(job.get("id", ""))):
                    continue
                if job["id"] != path.parent.name:
                    continue
                if job.get("status") not in TERMINAL:
                    job.update(status="interrupted", stage="interrupted",
                               message="The previous app session ended. This run was preserved; it was not auto-resumed.",
                               updated_at=now())
                    atomic_json(path, job)
                self._jobs[job["id"]] = job
                self._cancel[job["id"]] = threading.Event()
            except (OSError, ValueError, KeyError, TypeError):
                continue

    def _save(self, job: dict) -> None:
        atomic_json(self.root / job["id"] / "job.json", job)

    def list(self) -> list[dict]:
        with self._lock:
            return [self._public(job) for job in sorted(self._jobs.values(),
                                                       key=lambda item: item["created_at"], reverse=True)]

    def _public(self, job: dict) -> dict:
        return json.loads(json.dumps(job))

    def get(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError("Task not found.")
            return self._public(self._jobs[job_id])

    def directory(self, job_id: str) -> Path:
        self.get(job_id)
        return self.root / job_id

    def submit(self, case: dict, config: dict) -> dict:
        case = validate_case(case)
        with self._lock:
            if self._closed:
                raise ValueError("The application is shutting down.")
            job_id = uuid.uuid4().hex[:16]
            job = {"id": job_id, "case_id": case["case_id"], "mode": case["mode"],
                   "status": "queued", "stage": "queued", "message": "Waiting for the local worker.",
                   "created_at": now(), "updated_at": now(), "events": [], "log_tail": [],
                   "slide_count": len(case["slide_paths"]),
                   "case": case, "report_url": None, "error": None}
            self._jobs[job_id] = job
            self._cancel[job_id] = threading.Event()
            self._save(job)
            secrets = tuple(str(config.get(key) or "") for key in SECRET_KEYS)
            atomic_json(self.root / job_id / "resolved_config.json", redact(config, secrets))
            # The queue owns a copy so later settings edits cannot alter this run.
            self._queue.put((job_id, dict(config), secrets))
            return self._public(job)

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError("Task not found.")
            if job["status"] not in TERMINAL:
                self._cancel[job_id].set()
                job.update(message="Cancellation requested; waiting for worker shutdown.", updated_at=now())
                if job["status"] == "queued":
                    job.update(status="cancelled", stage="cancelled", message="Cancelled before execution.")
                self._save(job)
            return self._public(job)

    def _event(self, job_id: str, event: dict, secrets: tuple[str, ...]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if job["status"] in TERMINAL:
                return
            event = redact(dict(event), secrets)
            event["time"] = now()
            if event.get("stage") and event["stage"] not in {"running", "engine_log"}:
                job["stage"] = str(event["stage"])
            if event.get("message"):
                job["message"] = str(event["message"])
            log = event.get("log")
            if log is not None:
                job["log_tail"] = (job["log_tail"] + [str(log)])[-120:]
            job["events"] = (job["events"] + [event])[-200:]
            for name in ("roi_count", "candidate_count", "reinspection_rounds"):
                if isinstance(event.get(name), int):
                    job[name] = event[name]
            job["updated_at"] = event["time"]
            with (self.root / job_id / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._save(job)

    def _observe_checkpoints(self, job_id: str, stop: threading.Event, secrets: tuple[str, ...]):
        """Read completed stage artifacts without touching the engine's control flow."""
        case_dir = self.root / job_id / "cases" / self._jobs[job_id]["case_id"]
        seen: dict[str, str] = {}
        while not stop.wait(0.4):
            if not case_dir.is_dir():
                continue
            available = []
            for name, (stage, message) in CHECKPOINT_STAGES.items():
                path = case_dir / name
                if not path.is_file():
                    continue
                try:
                    raw = path.read_bytes()
                    fingerprint = hashlib.sha256(raw).hexdigest()
                    if seen.get(name) == fingerprint:
                        continue
                    data = json.loads(raw)
                    available.append((path.stat().st_mtime_ns, name, stage, message, data, fingerprint))
                except (OSError, ValueError):
                    # A partially-written checkpoint is not a completed stage.
                    continue
            for _, name, stage, message, data, fingerprint in sorted(available):
                seen[name] = fingerprint
                event = {"stage": stage, "message": message, "source": "checkpoint", "artifact": name}
                if isinstance(data, dict):
                    for key in ("selected_rois", "roi_results"):
                        if isinstance(data.get(key), list):
                            event["roi_count"] = len(data[key])
                    if isinstance(data.get("selected_count"), int):
                        event["roi_count"] = data["selected_count"]
                    if name == "ckpt_08_reinspection.json" and isinstance(data.get("round"), int):
                        event["reinspection_rounds"] = data["round"]
                        event["message"] = f"Reinspection round {data['round']} saved."
                self._event(job_id, event, secrets)
            for checkpoint in sorted(case_dir.glob("reinspect_*/ckpt_reinspect.json")):
                relative = checkpoint.relative_to(case_dir).as_posix()
                try:
                    raw = checkpoint.read_bytes()
                    fingerprint = hashlib.sha256(raw).hexdigest()
                    if seen.get(relative) == fingerprint:
                        continue
                    json.loads(raw)
                    round_number = int(checkpoint.parent.name.rsplit("_", 1)[1])
                except (OSError, ValueError, IndexError):
                    continue
                seen[relative] = fingerprint
                self._event(job_id, {"stage": "reinspection", "message": f"Reinspection round {round_number} saved.",
                                     "reinspection_rounds": round_number, "source": "checkpoint", "artifact": relative}, secrets)

    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            job_id, config, secrets = item
            observer_stop = threading.Event()
            observer = None
            try:
                with self._lock:
                    job = self._jobs[job_id]
                    if self._cancel[job_id].is_set():
                        if job["status"] not in TERMINAL:
                            job.update(status="cancelled", stage="cancelled",
                                       message="Cancelled before execution.", updated_at=now())
                            self._save(job)
                        continue
                    self._active_job_id = job_id
                    job.update(status="running", stage="preflight", message="Checking runtime and case inputs.",
                               updated_at=now())
                    self._save(job)
                    case = dict(job["case"])
                observer = threading.Thread(target=self._observe_checkpoints,
                                            args=(job_id, observer_stop, secrets), daemon=True)
                observer.start()
                outcome = self.runner(config, case, self.root / job_id,
                                      lambda event: self._event(job_id, event, secrets), self._cancel[job_id])
                with self._lock:
                    if self._shutdown_timed_out and job.get("cleanup_status") == "unconfirmed":
                        outcome = {"status": "interrupted", "worker_cleanup": {"status": "unconfirmed"},
                                   "error": "Shutdown wait expired before worker cleanup was confirmed."}
                        job.update(status="interrupted", stage="interrupted",
                                   message=outcome["error"], error=outcome["error"], report_url=None,
                                   updated_at=now())
                        job.pop("report_path", None)
                    elif self._cancel[job_id].is_set() or outcome.get("status") == "cancelled":
                        job.update(status="cancelled", stage="cancelled", message="Task cancelled; partial artifacts preserved.")
                    else:
                        if outcome.get("status") == "failed":
                            raise RuntimeError(outcome.get("error") or "Engine task failed.")
                        report = Path(outcome.get("report_path") or "").resolve()
                        if not report.is_file() or not report.is_relative_to((self.root / job_id).resolve()):
                            raise RuntimeError("The worker did not produce a report inside this task directory.")
                        job.update(status="succeeded", stage="complete", message="Report generated.",
                                   report_path=str(report),
                                   report_url=f"/reports/{job_id}/{report.relative_to(self.root / job_id).as_posix()}")
                        result = outcome.get("result") or {}
                        if isinstance(result, dict):
                            for name in ("selected_count", "candidate_count", "reinspection_rounds"):
                                if isinstance(result.get(name), int):
                                    job["roi_count" if name == "selected_count" else name] = result[name]
                    job["updated_at"] = now()
                    self._save(job)
                    atomic_json(self.root / job_id / "worker_outcome.json", redact(outcome, secrets))
            except WorkerCleanupError as exc:
                with self._lock:
                    job = self._jobs[job_id]
                    message = str(redact(str(exc), secrets))
                    job.update(status="interrupted", stage="interrupted", message=message,
                               error=message, cleanup_status="unconfirmed", report_url=None,
                               updated_at=now())
                    job.pop("report_path", None)
                    self._save(job)
                    atomic_json(self.root / job_id / "worker_outcome.json",
                                {"status": "interrupted", "worker_cleanup": {"status": "unconfirmed"},
                                 "error": message})
                    # Do not start the next task while a worker may still own
                    # GPU/CPU resources. Queued tasks have no worker to stop.
                    self._closed = True
                    for pending_id, pending in self._jobs.items():
                        if pending["status"] == "queued":
                            self._cancel[pending_id].set()
                            pending.update(status="cancelled", stage="cancelled",
                                           message="Not started: previous worker shutdown is unconfirmed.",
                                           updated_at=now())
                            self._save(pending)
                    self._queue.put(None)
            except Exception as exc:
                with self._lock:
                    job = self._jobs[job_id]
                    if self._shutdown_timed_out and job.get("cleanup_status") == "unconfirmed":
                        message = "Application shutdown deadline expired; worker cleanup could not be confirmed."
                        job.update(status="interrupted", stage="interrupted", message=message,
                                   error=message, report_url=None, updated_at=now())
                        job.pop("report_path", None)
                        atomic_json(self.root / job_id / "worker_outcome.json",
                                    {"status": "interrupted", "worker_cleanup": {"status": "unconfirmed"},
                                     "error": message})
                        self._save(job)
                        continue
                    cancelled = self._cancel[job_id].is_set()
                    message = str(redact(str(exc), secrets))
                    job.update(status="cancelled" if cancelled else "failed", stage="cancelled" if cancelled else job["stage"],
                               message="Task cancelled." if cancelled else message,
                               error=None if cancelled else message, updated_at=now())
                    self._save(job)
            finally:
                observer_stop.set()
                if observer:
                    observer.join(timeout=2)
                with self._lock:
                    if self._active_job_id == job_id:
                        self._active_job_id = None
                self._queue.task_done()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for job_id, job in self._jobs.items():
                if job["status"] not in TERMINAL:
                    self._cancel[job_id].set()
                    if job["status"] == "queued":
                        job.update(status="cancelled", stage="cancelled",
                                   message="Cancelled before execution during application shutdown.", updated_at=now())
                        self._save(job)
            enqueue = not self._close_enqueued
            self._close_enqueued = True
        if enqueue:
            self._queue.put(None)

    def wait_until(self, deadline: float) -> bool:
        worker = self._worker
        if worker is None:
            return True
        worker.join(timeout=max(0.0, deadline - time.monotonic()))
        if not worker.is_alive():
            return True
        with self._lock:
            self._shutdown_timed_out = True
            active = self._active_job_id
            for job_id, job in self._jobs.items():
                if job_id == active or job.get("status") not in TERMINAL:
                    message = "Application shutdown deadline expired; worker cleanup could not be confirmed."
                    job.update(status="interrupted", stage="interrupted", message=message,
                               error=message, cleanup_status="unconfirmed", report_url=None, updated_at=now())
                    job.pop("report_path", None)
                    self._save(job)
                    atomic_json(self.root / job_id / "worker_outcome.json",
                                {"status": "interrupted", "worker_cleanup": {"status": "unconfirmed"},
                                 "error": message})
        return False

    def shutdown_diagnostics(self) -> dict:
        with self._lock:
            return {"worker_alive": bool(self._worker and self._worker.is_alive()),
                    "active_job_id": self._active_job_id,
                    "unconfirmed_job_ids": [job["id"] for job in self._jobs.values()
                                            if job.get("cleanup_status") == "unconfirmed"]}
