"""Immutable byte snapshots for reviewed demo adapter source modules.

This module uses only the Python standard library and never imports scientific
runtime modules, reads settings, or inspects the process environment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SOURCE_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIRNAME = "adapter_source_snapshot"
MANIFEST_NAME = "manifest.json"
SCHEMA = "pathxai-demo-adapter-source-v1"
SOURCE_FILES = (
    "pathxai_demo/engine.py",
    "pathxai_demo/engine_worker.py",
    "pathxai_demo/profiles.py",
    "pathxai_demo/preparation.py",
    "pathxai_demo/transport.py",
    "pathxai_demo/replay.py",
    "pathxai_demo/request_audit.py",
    "pathxai_demo/identifiers.py",
    "pathxai_demo/segmentation_worker.py",
    "pathxai_demo/process_control.py",
)


class AdapterProvenanceError(RuntimeError):
    """The immutable source snapshot could not be safely created or verified."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_source_files(source_root: Path) -> list[tuple[str, Path]]:
    root = source_root.resolve()
    if not root.is_dir():
        raise AdapterProvenanceError(f"adapter source root is missing: {root}")
    files: list[tuple[str, Path]] = []
    for rel in SOURCE_FILES:
        path = root / rel
        if path.is_symlink():
            raise AdapterProvenanceError(f"allowlisted adapter source is a symlink: {rel}")
        if not path.is_file():
            raise AdapterProvenanceError(f"allowlisted adapter source is missing: {rel}")
        resolved = path.resolve()
        if root not in resolved.parents:
            raise AdapterProvenanceError(f"allowlisted adapter source escapes source root: {rel}")
        files.append((rel, path))
    return files


def _snapshot_root(run_dir: Path, source_root: Path) -> Path:
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise AdapterProvenanceError("run directory must be an existing, non-symlink directory")
    run = run_dir.resolve()
    source = source_root.resolve()
    module_tree = source / "pathxai_demo"
    if run == module_tree or module_tree in run.parents:
        raise AdapterProvenanceError("run directory cannot be inside the adapter source module tree")
    dest = run / SNAPSHOT_DIRNAME
    if dest.exists() or dest.is_symlink():
        raise AdapterProvenanceError(f"adapter source snapshot already exists; refusing overwrite: {dest}")
    if dest.parent != run:
        raise AdapterProvenanceError("adapter snapshot destination escapes run directory")
    return dest


def capture_adapter_sources(run_dir: str | Path, *, source_root: str | Path | None = None) -> Path:
    """Copy the reviewed adapter allowlist as original bytes and write its hash manifest.

    The optional source_root exists for isolated unit tests; production calls
    should use the default root derived from this module.
    """
    root = Path(source_root) if source_root is not None else SOURCE_ROOT
    files = _safe_source_files(root)
    destination = _snapshot_root(Path(run_dir), root)
    destination.mkdir(mode=0o700)
    inventory: dict[str, dict[str, Any]] = {}
    for rel, source_path in files:
        raw = source_path.read_bytes()
        target_rel = f"files/{rel}"
        target = destination / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as stream:
                stream.write(raw)
                stream.flush()
        except FileExistsError as exc:
            raise AdapterProvenanceError(f"snapshot target already exists; refusing overwrite: {target_rel}") from exc
        inventory[rel] = {
            "snapshot_path": target_rel,
            "size_bytes": len(raw),
            "sha256": _sha256(raw),
        }
    manifest = {"schema": SCHEMA, "files": inventory}
    manifest_path = destination / MANIFEST_NAME
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + chr(10)).encode("utf-8")
    try:
        with manifest_path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
        digest = _sha256(encoded)
        with (destination / "manifest.sha256").open("xb") as stream:
            stream.write(f"{digest}  {MANIFEST_NAME}{chr(10)}".encode("ascii"))
    except FileExistsError as exc:
        raise AdapterProvenanceError("manifest target already exists; refusing overwrite") from exc
    errors = verify_adapter_sources(destination, source_root=root)
    if errors:
        raise AdapterProvenanceError("source changed during snapshot capture: " + "; ".join(errors))
    return manifest_path


def verify_adapter_sources(run_or_manifest: str | Path, *, source_root: str | Path | None = None) -> list[str]:
    """Verify manifest, frozen byte copies, and current allowlisted source bytes.

    Returns every explicit verification failure. A caller must treat a
    non-empty list as a fail-closed source-integrity failure.
    """
    root = Path(source_root) if source_root is not None else SOURCE_ROOT
    base = Path(run_or_manifest)
    if base.is_symlink():
        return ["source snapshot evidence path is a symlink"]
    if base.is_file():
        base = base.parent
    elif base.name != SNAPSHOT_DIRNAME:
        base = base / SNAPSHOT_DIRNAME
    if base.is_symlink():
        return ["source snapshot directory is a symlink"]
    base = base.resolve()
    errors: list[str] = []
    manifest_path = base / MANIFEST_NAME
    sidecar = base / "manifest.sha256"
    try:
        encoded = manifest_path.read_bytes()
    except OSError as exc:
        return [f"source snapshot manifest is unavailable: {exc.strerror or type(exc).__name__}"]
    try:
        sidecar_fields = sidecar.read_text(encoding="ascii").split()
        if len(sidecar_fields) < 1 or sidecar_fields[0].lower() != _sha256(encoded):
            errors.append("source snapshot manifest SHA-256 sidecar mismatch")
    except OSError as exc:
        errors.append(f"source snapshot manifest sidecar is unavailable: {exc.strerror or type(exc).__name__}")
    try:
        manifest = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return errors + [f"source snapshot manifest is invalid: {type(exc).__name__}"]
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        return errors + ["source snapshot manifest schema is unsupported"]
    inventory = manifest.get("files")
    if not isinstance(inventory, dict):
        return errors + ["source snapshot manifest files inventory is invalid"]
    expected_names = set(SOURCE_FILES)
    if set(inventory) != expected_names:
        missing = sorted(expected_names - set(inventory))
        extra = sorted(set(inventory) - expected_names)
        if missing:
            errors.append("source snapshot manifest is missing allowlisted files: " + ", ".join(missing))
        if extra:
            errors.append("source snapshot manifest contains non-allowlisted files: " + ", ".join(extra))
    root_resolved = root.resolve()
    base_resolved = base.resolve()
    for rel in SOURCE_FILES:
        item = inventory.get(rel)
        if not isinstance(item, dict):
            continue
        expected_path = f"files/{rel}"
        if item.get("snapshot_path") != expected_path:
            errors.append(f"invalid snapshot path for allowlisted source: {rel}")
            continue
        frozen_path = (base / expected_path).resolve()
        if base_resolved not in frozen_path.parents:
            errors.append(f"snapshot path escapes evidence directory: {rel}")
            continue
        try:
            frozen_bytes = frozen_path.read_bytes()
        except OSError as exc:
            errors.append(f"frozen source copy is missing for {rel}: {exc.strerror or type(exc).__name__}")
            frozen_bytes = None
        expected_sha = item.get("sha256")
        if frozen_bytes is not None:
            actual_sha = _sha256(frozen_bytes)
            if actual_sha != expected_sha:
                errors.append(f"frozen source copy hash mismatch: {rel}")
            if item.get("size_bytes") != len(frozen_bytes):
                errors.append(f"frozen source copy size mismatch: {rel}")
        source_path = root_resolved / rel
        if source_path.is_symlink() or not source_path.is_file():
            errors.append(f"current allowlisted source is missing or unsafe: {rel}")
            continue
        try:
            source_bytes = source_path.read_bytes()
        except OSError as exc:
            errors.append(f"current allowlisted source cannot be read for {rel}: {exc.strerror or type(exc).__name__}")
            continue
        if _sha256(source_bytes) != expected_sha:
            errors.append(f"current allowlisted source changed since capture: {rel}")
    return errors


__all__ = [
    "AdapterProvenanceError",
    "SOURCE_FILES",
    "capture_adapter_sources",
    "verify_adapter_sources",
]
