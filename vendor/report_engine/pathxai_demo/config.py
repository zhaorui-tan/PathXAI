"""Persist public settings; keep access credentials in this process only."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Any

SECRET_KEYS = frozenset({"api_key", "hf_token"})


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


class ConfigStore:
    def __init__(self, workspace: Path, defaults: dict[str, Any]):
        self.path = workspace / "settings.json"
        self._lock = threading.RLock()
        self._defaults = dict(defaults)
        self._defaults.update({key: "" for key in SECRET_KEYS})
        self._settings = dict(self._defaults)
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError("Settings file is invalid; preserve it and correct its JSON.") from exc
            if not isinstance(loaded, dict):
                raise ValueError("Settings must be a JSON object.")
            self._settings.update({key: value for key, value in loaded.items()
                                   if key in self._defaults and key not in SECRET_KEYS})
        # Older settings may contain empty path fields from before models and
        # interpreters were discovered relative to the application package.
        # Fill only blanks and only from currently discovered defaults; keep
        # every non-empty user-selected path untouched.
        for key in ("seg_ckpt", "sam3_checkpoint", "foundation_model_path",
                    "pipeline_python", "seg_python", "keep_root"):
            if not self._settings.get(key) and self._defaults.get(key):
                self._settings[key] = self._defaults[key]
        # KEEP is a two-file gated resource. Never expose a saved directory as
        # configured unless both catalogued files still match their full hashes.
        if self._settings.get("keep_root"):
            try:
                from .assets import verify_keep_dataset
                verify_keep_dataset(self._settings["keep_root"])
            except (OSError, RuntimeError, TypeError, ValueError):
                self._settings["keep_root"] = ""

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._settings)

    def public(self) -> dict[str, Any]:
        with self._lock:
            output = {key: value for key, value in self._settings.items() if key not in SECRET_KEYS}
            output.update({key + "_configured": bool(self._settings.get(key)) for key in SECRET_KEYS})
            return output

    def update(self, values: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object.")
        unknown = set(values) - set(self._defaults)
        if unknown:
            raise ValueError("Unsupported configuration fields: " + ", ".join(sorted(unknown)))
        if any(not isinstance(value, (str, int, float, bool, type(None))) for value in values.values()):
            raise ValueError("Configuration values must be strings or scalar values.")
        if "keep_root" in values and values["keep_root"]:
            from .assets import verify_keep_dataset
            try:
                verify_keep_dataset(str(values["keep_root"]))
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError("KEEP configuration requires both files to pass pinned verification.") from exc
        with self._lock:
            self._settings.update(values)
            atomic_json(self.path, {key: value for key, value in self._settings.items()
                                    if key not in SECRET_KEYS})
            return self.public()


def redact(value: Any, secrets: tuple[str, ...] | list[str] = ()) -> Any:
    """Remove known secrets from persisted records and user-visible errors."""
    if isinstance(value, dict):
        return {key: ("[redacted]" if key.lower() in SECRET_KEYS or key.lower() in {
            "authorization", "access_token", "token", "password"} else redact(item, secrets))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in sorted((item for item in secrets if item), key=len, reverse=True):
            value = value.replace(secret, "[redacted]")
        return value
    return value
