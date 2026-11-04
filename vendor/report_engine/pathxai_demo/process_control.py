"""Bounded shutdown of a task's private Linux process group.

The engine creates this group with start_new_session=True. Scientific children
inherit it. Processes that deliberately escape that session are outside this
contract; this helper is not a general-purpose process-name killer.
"""
from __future__ import annotations

import os
import math
from pathlib import Path
import signal
import subprocess
import time


class WorkerCleanupError(RuntimeError):
    """The task's workers could not be confirmed stopped."""


def _identity(pid: int) -> dict | None:
    try:
        root = Path(f"/proc/{pid}")
        raw = (root / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        return {"pid": pid, "state": fields[0], "pgrp": int(fields[2]),
                "session": int(fields[3]), "start_ticks": int(fields[19]),
                "uid": root.stat().st_uid}
    except (FileNotFoundError, ProcessLookupError):
        return None
    except (OSError, ValueError, IndexError) as exc:
        raise WorkerCleanupError("Cannot verify worker process identity; shutdown is unconfirmed.") from exc


def _assert_leader(expected: dict) -> dict:
    current = _identity(expected["pid"])
    keys = ("pid", "start_ticks", "uid", "pgrp", "session")
    if current is None or any(current[key] != expected[key] for key in keys):
        raise WorkerCleanupError("Worker leader identity is unavailable or changed; shutdown is unconfirmed.")
    return current


def capture_linux_group(proc: subprocess.Popen) -> dict:
    """Pin the direct child before any poll/wait can release its PID."""
    if proc.returncode is not None:
        raise WorkerCleanupError("Worker leader was already reaped; group ownership is unconfirmed.")
    current = _identity(proc.pid)
    if (current is None or current["pgrp"] != proc.pid or current["session"] != proc.pid
            or current["uid"] != os.geteuid()):
        raise WorkerCleanupError("Cannot capture the private worker session identity.")
    return current


def linux_worker_exited(leader: dict) -> bool:
    """Observe exit without reaping the ownership anchor while children exist."""
    return _assert_leader(leader)["state"] in {"Z", "X"}


def _group_members(group: int, leader: dict) -> list[dict]:
    _assert_leader(leader)
    members = []
    try:
        with os.scandir("/proc") as entries:
            for entry in entries:
                if not entry.name.isdigit():
                    continue
                pid = int(entry.name)
                try:
                    if os.getpgid(pid) != group:
                        continue
                except ProcessLookupError:
                    continue
                current = _identity(pid)
                if current is None or current["pgrp"] != group:
                    continue
                if current["session"] != group or current["uid"] != os.geteuid():
                    raise WorkerCleanupError("Worker group ownership changed; shutdown is unconfirmed.")
                if current["state"] not in {"Z", "X"}:
                    members.append(current)
    except OSError as exc:
        raise WorkerCleanupError("Cannot inspect worker process group; shutdown is unconfirmed.") from exc
    _assert_leader(leader)
    return members


def _signal_member(expected: dict, sig: int) -> None:
    current = _identity(expected["pid"])
    if current is None:
        return
    keys = ("pid", "start_ticks", "uid", "pgrp", "session")
    if any(current[key] != expected[key] for key in keys):
        raise WorkerCleanupError("Worker process identity changed; no signal was sent to its replacement.")
    if current["state"] in {"Z", "X"}:
        return
    try:
        os.kill(expected["pid"], sig)
    except ProcessLookupError:
        pass
    except OSError as exc:
        raise WorkerCleanupError("Cannot stop an owned worker; shutdown is unconfirmed.") from exc


def terminate_linux_group(proc: subprocess.Popen, leader: dict | None = None, *, grace_seconds: float = 5,
                          kill_seconds: float = 2) -> dict:
    """Stop all active members, even when the direct child exits first.

    Only members of the private SID/PGID created for this Popen are eligible.
    Recheck each numeric identity immediately before signalling it; new members
    are observed until the group is empty or the bounded deadline expires.
    Zombies cannot compute or retain pipes and are left for their parent to
    reap; the direct Popen child is always reaped here.
    """
    if (not math.isfinite(grace_seconds) or not math.isfinite(kill_seconds)
            or grace_seconds < 0 or kill_seconds <= 0):
        raise ValueError("Shutdown deadlines must be nonnegative/positive.")
    leader = capture_linux_group(proc) if leader is None else leader
    if leader["pid"] != proc.pid:
        raise WorkerCleanupError("Worker handle and pinned leader identity differ.")
    started = time.monotonic()
    term_sent = set()
    kill_sent = set()
    for sig, seconds, sent in ((signal.SIGTERM, grace_seconds, term_sent),
                               (signal.SIGKILL, kill_seconds, kill_sent)):
        deadline = time.monotonic() + seconds
        while True:
            members = _group_members(proc.pid, leader)
            if not members and linux_worker_exited(leader):
                proc.wait()
                return {"status": "verified_no_active_group_members",
                        "scope": "linux_private_session_process_group",
                        "term_identities": len(term_sent), "kill_identities": len(kill_sent),
                        "elapsed_seconds": round(time.monotonic() - started, 6)}
            # Signal the leader last, retaining ancestry until its children
            # have received their own signal. Do not signal an entire group.
            for member in sorted(members, key=lambda row: row["pid"] == proc.pid):
                key = (member["pid"], member["start_ticks"])
                if key not in sent:
                    _signal_member(member, sig)
                    sent.add(key)
            if time.monotonic() >= deadline:
                break
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    raise WorkerCleanupError("Worker shutdown could not be confirmed within the deadline; partial artifacts are preserved.")
