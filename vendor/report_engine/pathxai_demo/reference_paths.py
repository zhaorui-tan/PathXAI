"""Narrow runtime path binding for strict Full reference replay.

The frozen selector continues to choose and rank candidates.  This module only
lets the exact aux-grade JSON cache report the producer's logical path while
reading and writing under a new, isolated reference run directory.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


class ReferencePathError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path_value: Any, *, recorded: dict[str, Any] | None = None,
                   cache: dict[str, tuple[int, str]] | None = None) -> dict[str, Any]:
    path = Path(str(path_value or "")).expanduser()
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise OSError("not a regular file")
        size = resolved.stat().st_size
        key = str(resolved)
        if cache is not None and key in cache:
            actual_size, digest = cache[key]
        else:
            actual_size, digest = size, _sha(resolved)
            if cache is not None:
                cache[key] = (actual_size, digest)
    except (OSError, RuntimeError) as exc:
        raise ReferencePathError("a referenced runtime asset is unavailable or unsafe") from exc
    if recorded is not None:
        if recorded.get("sha256") != digest or recorded.get("size_bytes") != actual_size:
            raise ReferencePathError("recorded runtime asset identity does not match its file bytes")
    return {"resolved_path": key, "size_bytes": actual_size, "sha256": digest}


def _compare_asset_maps(producer: Any, reference: Any, *, label: str,
                        cache: dict[str, tuple[int, str]]) -> None:
    if not isinstance(producer, dict) or not isinstance(reference, dict) or set(producer) != set(reference):
        raise ReferencePathError(f"producer {label} roles differ from this reference")
    for role in sorted(producer):
        left, right = producer[role], reference[role]
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise ReferencePathError(f"producer {label} record is invalid: {role}")
        left_id = _file_identity(left.get("path"), recorded=left, cache=cache)
        right_id = _file_identity(right.get("path"), recorded=right, cache=cache)
        if left_id != right_id:
            raise ReferencePathError(f"producer {label} asset differs from this reference: {role}")


def _compare_keep_assets(producer: Any, reference: Any, *, cache: dict[str, tuple[int, str]]) -> None:
    if not isinstance(producer, dict) or not isinstance(reference, dict):
        raise ReferencePathError("producer KEEP asset receipts are missing")
    if producer.get("required") is not True or reference.get("required") is not True:
        raise ReferencePathError("Full producer/reference do not require the same KEEP resources")
    left_files, right_files = producer.get("files"), reference.get("files")
    if not isinstance(left_files, dict) or not isinstance(right_files, dict) or set(left_files) != set(right_files):
        raise ReferencePathError("producer KEEP resource roles differ from this reference")
    for name in sorted(left_files):
        left, right = left_files[name], right_files[name]
        if not isinstance(left, dict) or not isinstance(right, dict) or not left.get("available") or not right.get("available"):
            raise ReferencePathError(f"producer KEEP resource is unavailable: {name}")
        left_id = _file_identity(left.get("path"), recorded=left, cache=cache)
        right_id = _file_identity(right.get("path"), recorded=right, cache=cache)
        if left_id != right_id:
            raise ReferencePathError(f"producer KEEP resource differs from this reference: {name}")


def _safe_component(value: Any, label: str) -> str:
    text = str(value or "")
    if not text or text in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ReferencePathError(f"unsafe aux-cache {label}")
    return text


def _assert_no_symlink_path(path: Path, *, allow_missing_leaf: bool = True) -> None:
    absolute = Path(path.anchor)
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for index, part in enumerate(parts):
        absolute = absolute / part
        if absolute.is_symlink():
            raise ReferencePathError("aux-cache path contains a symlink")
        if not absolute.exists() and (not allow_missing_leaf or index < len(parts) - 1):
            continue


def validate_replay_runtime_profile(profile_path: str | Path, *, case_id: str,
                                    trace_path: str | Path, mode: str,
                                    expected_profile: dict[str, Any],
                                    expected_candidates: list[dict[str, Any]],
                                    source_manifest: str | Path) -> dict[str, Any]:
    """Validate and identify the actual producer profile and trace for Full replay."""
    if mode != "full":
        raise ReferencePathError("replay runtime profile is supported only for Full mode")
    profile = Path(profile_path).expanduser()
    _assert_no_symlink_path(profile, allow_missing_leaf=False)
    if profile.is_symlink() or not profile.is_file():
        raise ReferencePathError("producer resolved profile is missing or unsafe")
    profile = profile.resolve()
    if profile.parent.name != case_id or profile.parent.parent.name != "cases":
        raise ReferencePathError("producer profile does not identify the requested case")
    try:
        value = json.loads(profile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReferencePathError("producer resolved profile is invalid") from exc
    if value.get("mode") != "full" or value.get("ablation") != "SO4_grade_aux_keep":
        raise ReferencePathError("producer profile is not the verified Full/SO4 run")
    source = value.get("source_integrity") or {}
    manifest_input = Path(str(source.get("manifest_path") or ""))
    _assert_no_symlink_path(manifest_input, allow_missing_leaf=False)
    manifest = manifest_input.resolve()
    expected_manifest = Path(source_manifest).resolve()
    if manifest != expected_manifest or source.get("manifest_verified") is not True:
        raise ReferencePathError("producer source manifest identity is not verified")
    if source.get("selected_roots_status") != "frozen_default" or source.get("custom_roots_are_not_baseline_verified") is True:
        raise ReferencePathError("producer did not select the verified frozen source roots")
    from .provenance import verify_manifest
    errors = verify_manifest(manifest)
    if errors:
        raise ReferencePathError("frozen source manifest failed verification")

    settings = value.get("effective_settings") or value.get("settings") or {}
    expected_settings = expected_profile.get("effective_settings") or expected_profile.get("settings") or {}
    identity_cache: dict[str, tuple[int, str]] = {}
    def scientific_settings(data: dict[str, Any]) -> dict[str, Any]:
        normalized = json.loads(json.dumps(data))
        normalized.pop("paths", None)
        llm = normalized.get("llm")
        if isinstance(llm, dict):
            for key in ("qwen_api", "report_api", "medgemma_api", "api_url", "base_url"):
                llm.pop(key, None)
        return normalized
    if scientific_settings(settings) != scientific_settings(expected_settings):
        raise ReferencePathError("producer effective settings differ from this Full reference")
    producer_paths = settings.get("paths") or {}
    reference_paths = expected_settings.get("paths") or {}
    for key in ("project_root", "v1_src_root", "seg_ckpt", "keep_root"):
        if producer_paths.get(key) != reference_paths.get(key):
            raise ReferencePathError(f"producer source or asset path differs from this reference: {key}")

    interpreter_identities: dict[str, dict[str, Any]] = {}
    for role in ("pipeline_python", "seg_python"):
        producer_path = producer_paths.get(role)
        reference_path = reference_paths.get(role)
        if not producer_path and not reference_path:
            if role == "seg_python":
                raise ReferencePathError("producer segmentation interpreter path is missing")
            # Older profiles may omit pipeline_python on both sides. Preserve
            # that exact unset/unset behavior; one-sided omission is unsafe.
            interpreter_identities[role] = {
                "configured": False, "producer_path": None, "reference_path": None,
                "path_relocated": False, "size_bytes": None, "sha256": None,
            }
            continue
        if not producer_path or not reference_path:
            raise ReferencePathError(f"producer and reference {role} must both be configured")
        producer_identity = _file_identity(producer_path, cache=identity_cache)
        reference_identity = _file_identity(reference_path, cache=identity_cache)
        if (producer_identity["size_bytes"] != reference_identity["size_bytes"] or
                producer_identity["sha256"] != reference_identity["sha256"]):
            raise ReferencePathError(f"producer {role} bytes differ from this reference")
        interpreter_identities[role] = {
            "configured": True,
            "producer_path": str(producer_path),
            "reference_path": str(reference_path),
            "producer_resolved_path": producer_identity["resolved_path"],
            "reference_resolved_path": reference_identity["resolved_path"],
            "path_relocated": producer_identity["resolved_path"] != reference_identity["resolved_path"],
            "size_bytes": producer_identity["size_bytes"],
            "sha256": producer_identity["sha256"],
        }
    _compare_asset_maps(value.get("segmentation_checkpoint"),
                        expected_profile.get("segmentation_checkpoint"),
                        label="model", cache=identity_cache)
    producer_foundation = (value.get("user_configuration") or {}).get("foundation_model_path")
    reference_foundation = (expected_profile.get("user_configuration") or {}).get("foundation_model_path")
    if not producer_foundation or not reference_foundation:
        raise ReferencePathError("producer foundation model path is missing")
    producer_foundation_id = _file_identity(producer_foundation, cache=identity_cache)
    reference_foundation_id = _file_identity(reference_foundation,
                                              recorded=expected_profile.get("foundation_model"),
                                              cache=identity_cache)
    if producer_foundation_id != reference_foundation_id:
        raise ReferencePathError("producer foundation model asset differs from this reference")
    _compare_keep_assets(value.get("knowledge_assets"), expected_profile.get("knowledge_assets"),
                         cache=identity_cache)

    candidate_path = profile.parent / "candidate_rois_input.json"
    _assert_no_symlink_path(candidate_path, allow_missing_leaf=False)
    if candidate_path.is_symlink() or not candidate_path.is_file():
        raise ReferencePathError("producer candidate pool is missing or unsafe")
    producer_candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    if producer_candidates != expected_candidates:
        raise ReferencePathError("producer ordered candidate ROI pool differs from this reference")

    metadata_path = profile.parent / "llm_trace_metadata.json"
    _assert_no_symlink_path(metadata_path, allow_missing_leaf=False)
    if metadata_path.is_symlink() or not metadata_path.is_file():
        raise ReferencePathError("producer trace metadata is missing or unsafe")
    trace_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_trace = Path(trace_path).expanduser()
    _assert_no_symlink_path(actual_trace, allow_missing_leaf=False)
    if actual_trace.is_symlink() or not actual_trace.is_file():
        raise ReferencePathError("replay trace is missing or unsafe")
    actual_trace = actual_trace.resolve()
    actual_sha = _sha(actual_trace)
    if trace_meta.get("mode") != "record" or trace_meta.get("sha256") != actual_sha:
        raise ReferencePathError("replay trace bytes do not match the producer trace receipt")
    producer_trace_path = Path(str(trace_meta.get("path") or "")).expanduser()
    _assert_no_symlink_path(producer_trace_path, allow_missing_leaf=False)
    if (producer_trace_path.resolve() != profile.parent / "llm_trace.jsonl"
            or not producer_trace_path.is_file() or producer_trace_path.is_symlink()
            or _sha(producer_trace_path) != actual_sha):
        raise ReferencePathError("producer trace path does not contain the verified trace bytes")
    recorded = trace_meta.get("recorded_requests")
    if not isinstance(recorded, int) or recorded <= 0:
        raise ReferencePathError("producer trace receipt has no recorded requests")
    result_path = profile.parent / "pipeline_result.json"
    _assert_no_symlink_path(result_path, allow_missing_leaf=False)
    if result_path.is_symlink() or not result_path.is_file():
        raise ReferencePathError("producer scientific outcome is missing or unsafe")
    result_value = json.loads(result_path.read_text(encoding="utf-8"))
    if result_value.get("errors"):
        raise ReferencePathError("producer scientific outcome contains errors")

    producer_settings = value.get("effective_settings") or value.get("settings") or {}
    logical_root = Path(str((producer_settings.get("paths") or {}).get("shared_artifacts") or "")).expanduser()
    if not logical_root.is_absolute():
        raise ReferencePathError("producer shared_artifacts path is missing or unsafe")
    _assert_no_symlink_path(logical_root)
    logical_root = logical_root.resolve()
    if logical_root != profile.parent / "runtime_artifacts":
        raise ReferencePathError("producer shared_artifacts path is not its isolated case runtime")
    return {
        "producer_profile_path": str(profile), "producer_profile_sha256": _sha(profile),
        "producer_trace_path": str(producer_trace_path.resolve()), "producer_trace_sha256": actual_sha,
        "producer_trace_requests": recorded, "producer_case_id": case_id,
        "producer_mode": "full", "producer_ablation": "SO4_grade_aux_keep",
        "producer_source_manifest": str(manifest), "producer_shared_artifacts": str(logical_root),
        "candidate_rois_sha256": _sha(candidate_path),
        "source_manifest_verified": True,
        "interpreter_path_binding": {
            "schema": "pathxai-reference-interpreter-path-binding-v1",
            "identity_basis": "current file size and SHA-256 only",
            "proves_full_environment_or_historical_python_identity": False,
            "roles": interpreter_identities,
        },
    }


class _LogicalParent:
    def __init__(self, logical: Path, physical: Path, events: list[dict[str, Any]]):
        self._logical, self._physical, self._events = logical, physical, events

    def __str__(self) -> str:
        return str(self._logical)

    def mkdir(self, *args: Any, **kwargs: Any) -> None:
        self._physical.mkdir(*args, **kwargs)
        self._events.append({"operation": "parent_mkdir", "logical_path": str(self._logical),
                            "physical_path": str(self._physical), "exists_after": self._physical.is_dir()})


class LogicalPhysicalAuxPath:
    """Duck-typed Path surface used by frozen `_select_aux_grade_rois`."""
    def __init__(self, logical: Path, physical: Path, events: list[dict[str, Any]]):
        self._logical, self._physical, self._events = logical, physical, events

    def __str__(self) -> str:
        return str(self._logical)

    @property
    def parent(self) -> _LogicalParent:
        return _LogicalParent(self._logical.parent, self._physical.parent, self._events)

    def exists(self) -> bool:
        result = self._physical.exists()
        self._events.append({"operation": "exists", "logical_path": str(self._logical),
                            "physical_path": str(self._physical), "result": result})
        return result

    def read_text(self, *args: Any, **kwargs: Any) -> str:
        value = self._physical.read_text(*args, **kwargs)
        self._events.append({"operation": "read_text", "logical_path": str(self._logical),
                            "physical_path": str(self._physical),
                            "size_bytes": len(value.encode(kwargs.get("encoding") or "utf-8")),
                            "sha256": hashlib.sha256(value.encode(kwargs.get("encoding") or "utf-8")).hexdigest()})
        return value

    def write_text(self, *args: Any, **kwargs: Any) -> int:
        count = self._physical.write_text(*args, **kwargs)
        self._events.append({"operation": "write_text", "logical_path": str(self._logical),
                            "physical_path": str(self._physical), "characters_written": count,
                            "size_bytes": self._physical.stat().st_size,
                            "sha256": _sha(self._physical)})
        return count


class AuxCacheBinding:
    def __init__(self, *, orchestrator: Any, physical_root: Path, logical_root: Path,
                 case_id: str, receipt: dict[str, Any]):
        self.orchestrator = orchestrator
        self.original = orchestrator._aux_roi_cache_path
        self.physical_root = physical_root.resolve()
        self.logical_root = logical_root.resolve()
        self.case_id = case_id
        self.receipt_data = receipt
        self.paths: list[tuple[Path, Path]] = []
        self.events: list[dict[str, Any]] = []
        self.calls = 0

        def mapped(settings: Any, *, purpose: str, foundation: str, queryset_id: str,
                   cancer_type: str, case_id: str) -> LogicalPhysicalAuxPath:
            if purpose != "grade" or case_id != self.case_id:
                raise ReferencePathError("unexpected frozen aux-cache request")
            physical = Path(self.original(settings, purpose=purpose, foundation=foundation,
                                          queryset_id=queryset_id, cancer_type=cancer_type,
                                          case_id=case_id))
            base = self.physical_root / "aux_roi_cache"
            try:
                relative = physical.relative_to(base)
            except ValueError as exc:
                raise ReferencePathError("frozen aux-cache path escaped reference runtime root") from exc
            parts = relative.parts
            if len(parts) != 6 or parts[0] != "grade" or parts[-1] != "grade_roi_candidates.json":
                raise ReferencePathError("frozen aux-cache path does not match exact grade JSON subtree")
            _safe_component(parts[1], "foundation")
            _safe_component(parts[2], "queryset")
            _safe_component(parts[3], "cancer type")
            if parts[4] != case_id:
                raise ReferencePathError("frozen aux-cache case identity changed")
            _assert_no_symlink_path(physical)
            resolved = physical.resolve(strict=False)
            if resolved != self.physical_root and self.physical_root not in resolved.parents:
                raise ReferencePathError("physical aux-cache target escapes isolated runtime")
            if not self.calls and physical.exists():
                raise ReferencePathError("physical Full aux cache already exists before replay selection")
            logical = self.logical_root / "aux_roi_cache" / relative
            _assert_no_symlink_path(logical)
            self.calls += 1
            self.paths.append((logical, physical))
            self.events.append({"operation": "path_resolution", "logical_path": str(logical),
                                "physical_path": str(physical), "selector_call": self.calls})
            return LogicalPhysicalAuxPath(logical, physical, self.events)

        orchestrator._aux_roi_cache_path = mapped
        receipt["binding_status"] = "installed"
        receipt["selector_calls"] = 0

    def receipt(self) -> dict[str, Any]:
        rows = []
        for logical, physical in self.paths:
            item = {"logical_path": str(logical), "physical_path": str(physical),
                    "exists": physical.is_file()}
            if physical.is_file():
                item.update(size_bytes=physical.stat().st_size, sha256=_sha(physical))
            rows.append(item)
        data = dict(self.receipt_data)
        data.update(selector_calls=self.calls, cache_files=rows, operations=list(self.events))
        if self.calls == 0:
            data["binding_status"] = data.get("binding_status", "installed_no_calls")
        elif rows and all(row["exists"] for row in rows):
            data["binding_status"] = "completed"
        else:
            data["binding_status"] = "incomplete"
        return data


def install_full_replay_aux_cache_binding(orchestrator: Any, *, settings: Any,
                                          output_dir: str | Path, case_id: str,
                                          producer_profile: str | Path,
                                          trace_path: str | Path,
                                          expected_profile: dict[str, Any],
                                          expected_candidates: list[dict[str, Any]],
                                          source_manifest: str | Path) -> AuxCacheBinding:
    identity = validate_replay_runtime_profile(
        producer_profile, case_id=case_id, trace_path=trace_path, mode="full",
        expected_profile=expected_profile, expected_candidates=expected_candidates,
        source_manifest=source_manifest)
    physical_root = Path(settings.paths.shared_artifacts).expanduser()
    output = Path(output_dir).resolve()
    if physical_root.is_symlink() or physical_root.resolve() != output / "runtime_artifacts":
        raise ReferencePathError("reference shared_artifacts is not its isolated runtime directory")
    physical_root.mkdir(parents=True, exist_ok=True)
    aux_root = physical_root / "aux_roi_cache"
    if aux_root.exists() or aux_root.is_symlink():
        raise ReferencePathError("reference aux-cache subtree must be absent before replay")
    logical_root = Path(identity["producer_shared_artifacts"])
    receipt = {
        "schema": "pathxai-reference-aux-cache-binding-v1",
        **identity,
        "logical_shared_artifacts": str(logical_root),
        "physical_shared_artifacts": str(physical_root.resolve()),
        "logical_aux_cache_root": str(logical_root / "aux_roi_cache"),
        "physical_aux_cache_root": str(aux_root),
        "physical_aux_cache_absent_before_selection": True,
        "source_manifest_path": str(Path(source_manifest).resolve()),
        "source_manifest_sha256": _sha(Path(source_manifest).resolve()),
    }
    return AuxCacheBinding(orchestrator=orchestrator, physical_root=physical_root,
                           logical_root=logical_root, case_id=case_id, receipt=receipt)

