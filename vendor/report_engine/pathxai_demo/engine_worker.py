"""Child-process bridge. All model/source imports happen in pipeline_python."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:  # direct-script invocation from engine.build_command
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pathxai_demo.preparation import prepare_candidates
    from pathxai_demo.profiles import config_defaults, resolve_profile
    from pathxai_demo.transport import (
        install_requests_transport, install_segmentation_cache_adapter,
        install_segmentation_subprocess_adapter, install_requests_replay_guard,
    )
else:
    from .preparation import prepare_candidates
    from .profiles import config_defaults, resolve_profile
    from .transport import (
        install_requests_transport, install_segmentation_cache_adapter,
        install_segmentation_subprocess_adapter, install_requests_replay_guard,
    )


_CAPTURED_ADAPTER_RUNS: set[str] = set()


def _emit(event: dict[str, Any] | None = None, **values: Any) -> None:
    payload = dict(event or {})
    payload.update(values)
    sys.__stdout__.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    sys.__stdout__.flush()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.is_dir():
        return "missing"
    for source in sorted(root.rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        digest.update(source.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Normalize dataclass settings to the same JSON value form stored on disk."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str, allow_nan=False))


def _adapter_capture_record(manifest_path: Path, *, source_root: Path) -> dict[str, Any]:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    sidecar = (manifest_path.parent / "manifest.sha256").read_text(encoding="ascii").split()
    return {
        "status": "captured",
        "capture_phase": "before_scientific_imports",
        "source_root": str(source_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_sidecar_sha256": sidecar[0] if sidecar else None,
        "files": manifest.get("files", {}),
    }


def _finish_adapter_provenance(out_dir: Path, resolved: dict[str, Any] | None = None) -> list[str]:
    """Record final source verification without changing scientific outputs."""
    from pathxai_demo.adapter_provenance import verify_adapter_sources

    profile_path = out_dir / "resolved_profile.json"
    if resolved is None:
        try:
            value = json.loads(profile_path.read_text(encoding="utf-8"))
            resolved = value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            resolved = {}
    receipt = resolved.get("adapter_source_integrity")
    if not isinstance(receipt, dict):
        receipt = {"status": "missing_capture"}
    errors = verify_adapter_sources(out_dir)
    receipt["end_verification"] = {"status": "verified" if not errors else "failed", "errors": errors}
    receipt["status"] = "verified" if not errors else "failed"
    resolved["adapter_source_integrity"] = receipt
    if profile_path.exists() or resolved:
        profile_path.write_text(json.dumps(resolved, ensure_ascii=False, indent=2, default=str) + "\n",
                                encoding="utf-8")
    _CAPTURED_ADAPTER_RUNS.discard(str(out_dir.resolve()))
    return errors


def _segmentation_cache_namespace(
    *, config: dict[str, Any], candidates: list[dict], settings: Any,
    v1_src: Path, request_asset_provenance: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    candidate_images = []
    for row in candidates:
        image = Path(str(row.get("image_path") or "")).expanduser()
        if not image.is_file():
            candidate_images.append({"path": str(image), "sha256": None})
            continue
        digest = hashlib.sha256()
        with image.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        candidate_images.append({"path": str(image.resolve()), "sha256": digest.hexdigest()})
    roots = {
        "v2": Path(str(config["engine_root"])) / "src" / "pathxai_agent_v2",
        "v1": v1_src / "pathxai_agent",
        "segmentation_wrapper": Path(str(config["segmentation_source_root"])) / "pathxai_agent" / "segmentation",
        "segmentation_runtime": Path(str(config["pathxai_root"])),
        "medsam3": Path(str(config["medsam3_root"])),
    }
    identity = {
        "source_sha256": {name: _tree_sha256(path) for name, path in roots.items()},
        "assets": request_asset_provenance,
        "candidate_images": candidate_images,
        "segmentation_parameters": vars(settings.segmentation).copy(),
    }
    canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), identity


def _run(request: dict[str, Any]) -> dict[str, Any]:
    config = request["config"]
    case = request["case"]
    out_dir = Path(request["output_dir"]).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = out_dir / "resolved_profile.json"
    demo_root = Path(__file__).resolve().parents[1]
    from pathxai_demo.adapter_provenance import capture_adapter_sources
    try:
        adapter_manifest = capture_adapter_sources(out_dir, source_root=demo_root)
        adapter_provenance = _adapter_capture_record(adapter_manifest, source_root=demo_root)
        _CAPTURED_ADAPTER_RUNS.add(str(out_dir))
    except Exception as exc:
        # No scientific module is imported after a failed capture. Preserve a
        # small failure receipt so the task output explains the fail-closed stop.
        profile_path.write_text(json.dumps({
            "mode": str(case.get("mode") or request.get("mode") or "core").lower(),
            "adapter_source_integrity": {
                "status": "capture_failed",
                "capture_phase": "before_scientific_imports",
                "error_type": type(exc).__name__,
            },
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Adapter source capture failed before scientific imports ({type(exc).__name__})") from exc
    profile_path.write_text(json.dumps({
        "mode": str(case.get("mode") or request.get("mode") or "core").lower(),
        "adapter_source_integrity": adapter_provenance,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    mode = str(case.get("mode") or request.get("mode") or "core").lower()
    trace_mode = str(config.get("trace_mode") or "record")
    replay_network_guard = install_requests_replay_guard() if trace_mode == "replay" else None
    profile = resolve_profile(config, mode)
    from pathxai_demo.provenance import verify_manifest
    baseline_manifest = demo_root / "baselines" / "current" / "manifest.json"
    source_integrity_errors = verify_manifest(baseline_manifest)
    if source_integrity_errors:
        raise RuntimeError("Frozen source integrity check failed: " + "; ".join(source_integrity_errors[:8]))
    defaults = config_defaults()
    source_roots = ("engine_root", "v1_root", "pathxai_root", "medsam3_root", "segmentation_source_root")
    source_mode = "frozen_default" if all(
        str(Path(str(config.get(key, ""))).expanduser().resolve())
        == str(Path(str(defaults.get(key, ""))).expanduser().resolve())
        for key in source_roots
    ) else "custom_unverified"
    # The scientific source reads a small number of legacy PATHXAI_* knobs
    # directly. Remove inherited values and establish the audited SO4 defaults.
    for name in tuple(os.environ):
        if name.startswith("PATHXAI_"):
            os.environ.pop(name, None)
    # Storage constants in v1 are read at import time, so establish the
    # selected artifact root before importing any v1 module.
    shared_artifacts = Path(str(config.get("shared_artifacts") or (
        Path(__file__).resolve().parents[1] / "workspace" / "shared_artifacts"
    ))).expanduser().resolve()
    shared_artifacts.mkdir(parents=True, exist_ok=True)
    os.environ["PATHXAI_SHARED_ARTIFACTS"] = str(shared_artifacts)
    engine_root = Path(str(config["engine_root"])).expanduser()
    engine_src = engine_root / "src"
    if not (engine_src / "pathxai_agent_v2").is_dir():
        raise FileNotFoundError(f"PathXAI v2 source not found under engine_root: {engine_root}")
    sys.path.insert(0, str(engine_src))
    v1_root = Path(str(config.get("v1_root") or ""))
    v1_candidates = [v1_root, v1_root / "src", Path(str(config.get("pathxai_root") or "")) / "src"]
    v1_src = next((path for path in v1_candidates if (path / "pathxai_agent").is_dir()), None)
    if v1_src is not None:
        sys.path.insert(0, str(v1_src))
    if v1_src is None:
        raise FileNotFoundError("Could not locate configured frozen v1 source root")
    archived_v1 = Path(str(config.get("segmentation_source_root") or ""))
    archived_segmentation = archived_v1 / "pathxai_agent" / "segmentation"
    import pathxai_agent
    import pathxai_agent.segmentation as segmentation_package
    if archived_segmentation.is_dir():
        segmentation_package.__path__.insert(0, str(archived_segmentation))
    medsam_module = importlib.import_module("pathxai_agent.segmentation.medsam_roi")
    seg_visual_module = importlib.import_module("pathxai_agent.segmentation.seg_visual")
    seg_visual_cache_root = out_dir / "runtime_artifacts" / "seg_roi_pairs"
    seg_visual_cache = install_segmentation_cache_adapter(seg_visual_module, seg_visual_cache_root)

    # Transport configuration is kept out of settings, result manifests, and logs.
    endpoint = str(config.get("qwen_api") or "http://127.0.0.1:8000")
    install_requests_transport(endpoint, str(config.get("api_key") or ""))
    if config.get("hf_token"):
        os.environ["HF_TOKEN"] = str(config["hf_token"])
    os.environ["PATHXAI_AUX_ROI_FOUNDATION"] = "conch"
    os.environ["PATHXAI_AUX_GRADE_ROIS"] = "2"

    from pathxai_agent_v2.config import load_settings
    settings_module = importlib.import_module("pathxai_agent_v2.config.settings")
    from pathxai_agent_v2.leakage_guard import DiagnosisPriors
    from pathxai_agent_v2.run import _attach_slide_paths_to_candidates, _create_client, run_single_case
    from pathxai_demo.replay import ReplayClient, recording_client

    settings = load_settings(use_env=False, overrides=profile["settings"])
    os.environ["V1_SRC_ROOT"] = settings.paths.v1_src_root
    os.environ["PATHXAI_SHARED_ARTIFACTS"] = settings.paths.shared_artifacts
    install_segmentation_subprocess_adapter(
        medsam_module,
        bootstrap_script=Path(__file__).with_name("segmentation_worker.py"),
        v1_source=archived_v1,
        segmentation_runtime=Path(str(config["pathxai_root"])),
        medsam3_source=Path(str(config["medsam3_root"])),
        sam3_bpe=Path(str(config["sam3_bpe"])),
        sam3_checkpoint=Path(str(config["sam3_checkpoint"])),
        shared_artifacts=Path(settings.paths.shared_artifacts),
    )
    out_dir = Path(request["output_dir"]).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved = {
        **profile,
        "adapter_source_integrity": adapter_provenance,
        "user_configuration": {
            key: config.get(key)
            for key in (
                "engine_root", "v1_root", "segmentation_source_root", "pathxai_root", "medsam3_root",
                "sam3_bpe", "pipeline_python", "seg_python", "seg_ckpt", "keep_root", "roi_cache_dir", "seg_cache_dir", "llm_cache_dir",
                "sam3_checkpoint", "runtime_cache_dir", "trace_mode", "trace_path",
                "shared_artifacts", "device", "qwen_api", "qwen_model", "foundation_model_path",
            )
        },
        "session_credentials_present": {
            "api_key": bool(config.get("api_key")),
            "hf_token": bool(config.get("hf_token")),
        },
        "knowledge_assets": request.get("knowledge_assets") or {
            "schema": "pathxai-keep-resources-v1", "required": False,
            "status": "not_required", "known_baseline_match": None,
            "baseline_match_status": "unknown", "files": {},
        },
        "legacy_environment": {
            "PATHXAI_AUX_ROI_FOUNDATION": "conch",
            "PATHXAI_AUX_GRADE_ROIS": "2",
            "PATHXAI_SHARED_ARTIFACTS": settings.paths.shared_artifacts,
        },
        "environment": {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "hash_info": {"algorithm": sys.hash_info.algorithm, "width": sys.hash_info.width},
        },
        "segmentation_runtime_adapter": {
            "wrapper_source_root": str(archived_v1),
            "runtime_source_root": str(config["pathxai_root"]),
            "medsam3_source_root": str(config["medsam3_root"]),
            "sam3_bpe_path": str(config["sam3_bpe"]),
            "sam3_checkpoint_path": str(config["sam3_checkpoint"]),
            "bootstrap": "pathxai_demo/segmentation_worker.py",
            "cuda_memory_fraction": 0.55,
            "source_difference": "Archived SO1/SO4 wrapper is used; all v1 Python files match the current frozen v1 snapshot except this wrapper.",
        },
        "segmentation_visual_cache": {
            **seg_visual_cache,
            "install_phase": "before_original_v2_imports",
            "frozen_defaults_rebound": True,
        },
        "source_integrity": {
            "manifest_path": str(baseline_manifest),
            "manifest_verified": True,
            "selected_roots_status": source_mode,
            "custom_roots_are_not_baseline_verified": source_mode == "custom_unverified",
        },
        "segmentation_checkpoint": request.get("asset_provenance"),
    }
    profile_path.write_text(json.dumps(resolved, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _emit(stage="wsi_preparation", message="Resolving candidate ROIs")
    candidates = prepare_candidates(config, case, out_dir, _emit)
    if not candidates:
        raise ValueError(f"No candidate ROIs resolved for case {case.get('case_id')}")
    candidate_path = out_dir / "candidate_rois_input.json"
    candidate_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _emit(stage="engine", message=f"Starting original {mode.upper()} pipeline ({profile['ablation']})", candidate_count=len(candidates))
    known_site = case.get("known_site")
    known_site_source = str(case.get("known_site_source") or "user") if known_site else None
    if known_site_source not in {None, "user", "tcga"}:
        raise ValueError("known_site_source must be 'user' or 'tcga' when known_site is provided")
    priors = DiagnosisPriors(known_site=known_site, known_site_source=known_site_source)
    candidates = _attach_slide_paths_to_candidates(candidates, settings)
    requested_seg_cache = Path(str(config["seg_cache_dir"])).expanduser().resolve()
    cache_key, cache_identity = _segmentation_cache_namespace(
        config=config,
        candidates=candidates,
        settings=settings,
        v1_src=v1_src,
        request_asset_provenance=request.get("asset_provenance") or {},
    )
    effective_seg_cache = requested_seg_cache / "pathxai-demo-cache-v1" / cache_key
    settings.paths.seg_cache_dir = str(effective_seg_cache)
    resolved["settings"]["paths"]["seg_cache_dir"] = str(effective_seg_cache)
    resolved["effective_settings"] = _json_safe(settings_module.to_dict(settings))
    loader_path = Path(settings_module.__file__).resolve()
    defaults_path = Path(settings_module._DEFAULTS_PATH).resolve()
    resolved["effective_settings_provenance"] = {
        "loader_path": str(loader_path),
        "loader_sha256": _sha256_file(loader_path),
        "defaults_path": str(defaults_path),
        "defaults_sha256": _sha256_file(defaults_path),
        "use_env": False,
        "overrides_source": "resolved_profile.settings",
        "effective_settings_sha256": hashlib.sha256(json.dumps(
            resolved["effective_settings"], ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
    }
    resolved["segmentation_cache"] = {
        "requested_root": str(requested_seg_cache),
        "effective_root": str(effective_seg_cache),
        "namespace_sha256": cache_key,
        "identity": cache_identity,
        "frozen_cache_key_semantics_preserved": True,
    }
    profile_path.write_text(json.dumps(resolved, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    trace_path = Path(str(config.get("trace_path") or out_dir / "llm_trace.jsonl")).expanduser()
    if trace_mode == "record":
        stages_llm = recording_client(_create_client(settings), trace_path)
    elif trace_mode == "replay":
        v1_replay_root = Path(str(config.get("v1_root") or "")).expanduser()
        if (v1_replay_root / "src" / "pathxai_agent").is_dir():
            pass
        elif v1_replay_root.name == "src":
            v1_replay_root = v1_replay_root.parent
        else:
            v1_replay_root = Path(str(v1_src)).parent
        # Constructor-only use of the original client makes no network call;
        # its actual encoder/settings detect configuration changes on replay.
        stages_llm = ReplayClient(str(config["trace_path"]),
                                  image_backend=_create_client(settings), v1_root=v1_replay_root,
                                  legacy_response_order_recovery=(
                                      config.get("legacy_response_order_recovery") is True),
                                  progress_path=out_dir / "llm_replay_progress.jsonl",
                                  progress_context=replay_network_guard.receipt)
        replay_network_guard.set_observer(stages_llm.record_network_block)
    else:
        stages_llm = None
    result = run_single_case(
        settings,
        str(case.get("case_id") or case.get("id")),
        candidates,
        out_dir,
        diagnosis_priors=priors,
        stages_llm=stages_llm,
        dry_run=False,
        resume=False,
        ablation=profile["ablation"],
    )
    result_dict = result.to_dict() if hasattr(result, "to_dict") else dict(result)
    if trace_mode == "replay" and stages_llm.remaining:
        raise RuntimeError(f"Replay ended with {stages_llm.remaining} unused recorded responses; control flow differs.")
    if trace_mode == "replay":
        stages_llm.finish_progress()
    result_path = out_dir / "pipeline_result.json"
    result_path.write_text(json.dumps(result_dict, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    from pathxai_agent_v2.html_viz import render_html
    report_path = out_dir / "report.html"
    render_html(out_dir, report_path)
    if not report_path.is_file():
        raise RuntimeError("Original HTML renderer did not create report.html")
    errors = list(result_dict.get("errors") or [])
    if trace_mode == "replay" and replay_network_guard.receipt()["blocked_attempt_count"]:
        errors.append("Replay blocked an attempted HTTP request; strict replay cannot be considered successful")
    if trace_mode == "replay" and (stages_llm.progress_receipt["violation_count"]
                                   or stages_llm.progress_receipt["write_error"]):
        errors.append("Replay request or progress-evidence violation occurred; strict replay cannot be considered successful")
    trace_metadata = {"mode": trace_mode, "path": str(trace_path),
                      "strict_request_verification": trace_mode == "replay",
                      "network_fallback": False if trace_mode == "replay" else None}
    if trace_mode == "replay":
        trace_metadata["legacy_response_order_recovery"] = stages_llm.recovery_receipt
        trace_metadata["replay_network_guard"] = replay_network_guard.receipt()
        trace_metadata["replay_progress"] = stages_llm.progress_receipt
    if trace_path.is_file():
        trace_bytes = trace_path.read_bytes()
        trace_metadata.update(sha256=hashlib.sha256(trace_bytes).hexdigest(),
                              recorded_requests=sum(bool(line.strip()) for line in trace_bytes.splitlines()))
    if trace_mode == "replay":
        trace_metadata.update(consumed_requests=stages_llm.consumed, remaining_requests=stages_llm.remaining)
    (out_dir / "llm_trace_metadata.json").write_text(json.dumps(trace_metadata, indent=2) + "\n", encoding="utf-8")
    adapter_errors = _finish_adapter_provenance(out_dir, resolved)
    outcome_errors = list(errors)
    if adapter_errors:
        outcome_errors.append("Adapter source integrity verification failed: " + "; ".join(adapter_errors))
    return {
        "status": "failed" if outcome_errors else "succeeded",
        "case_id": str(case.get("case_id") or case.get("id")),
        "mode": mode,
        "ablation": profile["ablation"],
        "candidate_count": len(candidates),
        "output_dir": str(out_dir),
        "report_path": str(report_path) if report_path.is_file() else None,
        "pipeline_result_path": str(result_path),
        "resolved_profile_path": str(profile_path),
        "resolved_profile": resolved,
        "trace_path": str(trace_path) if trace_mode == "record" else None,
        "result": result_dict,
        "errors": outcome_errors,
        "llm_trace": trace_metadata,
        "error": "; ".join(str(item) for item in outcome_errors) if outcome_errors else None,
    }


def main() -> int:
    request: dict[str, Any] = {}
    try:
        request = json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):
            outcome = _run(request)
        _emit(**outcome)
        return 0 if outcome["status"] == "succeeded" else 1
    except Exception as exc:
        try:
            output = request.get("output_dir") if isinstance(request, dict) else None
            output_path = Path(str(output)).expanduser().resolve() if output else None
            if output_path is not None and str(output_path) in _CAPTURED_ADAPTER_RUNS:
                _finish_adapter_provenance(output_path)
        except Exception:
            # Keep the original worker exception as the task's primary failure.
            pass
        _emit(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(limit=12))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

