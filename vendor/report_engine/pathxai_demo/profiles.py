"""Demo-facing Core/Full profiles and configuration validation.

Profiles deliberately expose the complete frozen engine settings. They do not
introduce run-time algorithm switches beyond the Core/Full mapping.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit

from .assets import discover_asset_path, discover_keep_root

CORE_ABLATION = "SO1_ss21_seg_overlay_composite"
FULL_ABLATION = "SO4_grade_aux_keep"
ARCHIVED_QWEN_MODEL = "E:/tanja1/qwen3.6/Qwen3.6-27B-Q6_K.gguf"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1] / "baselines" / "current" / "source_snapshot" / "v2"


def _v1_root() -> Path:
    return Path(__file__).resolve().parents[1] / "baselines" / "current" / "source_snapshot" / "v1"


def _archive_code_root() -> Path:
    return (Path(__file__).resolve().parents[1] / "baselines" / "current" / "source_snapshot" / "archives"
            / "tcga_so1_so4_a5bridge_508_20260920" / "code")


def discover_runtime_python(role: str, *, app_root: str | Path | None = None,
                           environ: dict[str, str] | None = None) -> str:
    """Locate a bundled interpreter without falling back to the host Python.

    An explicit PATHXAI_* override is preserved even when it is invalid so
    normal preflight reports the bad user configuration instead of silently
    selecting another interpreter.
    """
    role = str(role).strip().lower()
    if role not in {"pipeline", "seg"}:
        raise ValueError("role must be 'pipeline' or 'seg'")
    env = os.environ if environ is None else environ
    variable = "PATHXAI_PIPELINE_PYTHON" if role == "pipeline" else "PATHXAI_SEG_PYTHON"
    override = str(env.get(variable) or "").strip()
    if override:
        return override
    root = Path(app_root) if app_root is not None else Path(__file__).resolve().parents[1]
    bin_dir = root / "runtime" / role / "bin"
    for name in ("python3", "python"):
        candidate = bin_dir / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return ""


def config_defaults(*, workspace: str | Path | None = None) -> dict[str, Any]:
    """Return user-editable adapter defaults; secrets are session-only inputs."""
    app_root = Path(__file__).resolve().parents[1]
    workspace = Path(workspace).expanduser().resolve() if workspace is not None else app_root / "workspace"
    return {
        "engine_root": str(_project_root()),
        "v1_root": str(_v1_root()),
        "pathxai_root": str(_archive_code_root() / "seg_runtime"),
        "medsam3_root": str(_archive_code_root() / "medsam3"),
        "sam3_bpe": str(app_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"),
        "sam3_checkpoint": discover_asset_path("sam3", app_root=app_root),
        "segmentation_source_root": str(_archive_code_root() / "v1"),
        "pipeline_python": discover_runtime_python("pipeline", app_root=app_root),
        "seg_python": discover_runtime_python("seg", app_root=app_root),
        "seg_ckpt": discover_asset_path("seg_d1", app_root=app_root),
        "keep_root": discover_keep_root(app_root=app_root),
        "roi_cache_dir": str(workspace / "roi_cache"),
        "seg_cache_dir": str(workspace / "seg_cache" / "medsam3_d1"),
        "shared_artifacts": str(workspace / "shared_artifacts"),
        "llm_cache_dir": str(workspace / "llm_cache"),
        "runtime_cache_dir": str(workspace / "runtime_cache"),
        "qwen_api": "http://127.0.0.1:8000",
        "qwen_model": ARCHIVED_QWEN_MODEL,
        "device": "cuda:0",
        # Foundation model location is an asset path; preparation behavior is
        # fixed by the selected Core/Full profile below.
        "foundation_model_path": discover_asset_path("conch", app_root=app_root),
        "trace_mode": "record",
        "trace_path": "",
        "legacy_response_order_recovery": False,
        "output_root": str(workspace / "runs"),
    }


def profile_names() -> list[dict[str, str]]:
    return [
        {"id": "core", "label": "PathXAI-Report (core)", "variant": CORE_ABLATION},
        {"id": "full", "label": "PathXAI-Report", "variant": FULL_ABLATION},
    ]


def get_profile(mode: str) -> dict[str, Any]:
    mode = str(mode).strip().lower()
    if mode not in {"core", "full"}:
        raise ValueError("mode must be 'core' or 'full'")
    return {
        "mode": mode,
        "ablation": CORE_ABLATION if mode == "core" else FULL_ABLATION,
        "policy_notes": [
            "For new WSI cases, only an explicitly entered known_site is supplied with source=user; no site is inferred from filenames.",
            "Archived TCGA runs used known_site_from=tcga. The app does not derive site priors from case IDs or hidden cancer-type labels.",
        ],
        "preparation": {
            "foundation_model": "conch",
            "ranking_mode": "diagnostic",
            "tile_size": 1024,
            "final_roi_size": 4096,
            "mpp": 0.5,
            "batch_size": 16,
            "candidate_rois_per_slide": 20,
            "max_candidate_rois_per_case": 20,
            "max_tiles": None,
            "num_workers": 0,
            "amp": True,
            "diversity_factor": 0.75,
        },
        "settings": {
            "segmentation": {
                "enabled": True,
                "device": "cuda:0",
                "patch_size": 4096,
                "stride": 2048,
                "max_rois": 11,
                "use_subprocess": True,
                "qc_max_images": 11,
                "qc_missing_retries": 2,
                "qc_cache_version": "seg_qc_pair_v2",
            },
            "roi": {
                "candidate_rois": 20,
                "selected_rois": 3,
                "max_site_rois": 3,
                "llm_triage_candidates": 8,
                "foundation_model": "conch",
                "foundation_ranking_mode": "diagnostic",
                "foundation_tile_size": 1024,
                "foundation_mpp": 0.5,
                "foundation_batch_size": 16,
            },
            "reinspection": {
                "max_reinspection_rois": 2,
                "max_reinspection_rounds": 4,
                "max_zoom_depth": 1,
                "max_final_rois": 11,
            },
            "context_budget": {
                "medgemma_max_chars": 6000,
                "segmentation_max_chars": 12000,
                "knowledge_max_chars": 4000,
            },
            "knowledge": {
                "enabled": True,
                "max_disease": 5,
                "max_semantic_groups": 5,
                "max_captions_per_hypothesis": 3,
                "max_children": 5,
            },
            "session": {"memory_version": "diagnostic_session_memory_v1"},
            "leakage": {"known_site_from": "tcga", "min_hypotheses": 3},
            "report": {"enforce_staged_diagnosis": False},
            "llm": {
                "qwen_model": ARCHIVED_QWEN_MODEL,
                "read_timeout": 600,
                "connect_timeout": 10,
                "max_retries": 2,
                "retry_backoff": 3.0,
                "image_max_edge": 768,
                "image_jpeg_quality": 75,
                "temperature": 0.2,
                "max_tokens": 4096,
                "report_api": "",
                "report_model": "",
                "report_max_tokens": 4096,
            },
        },
    }


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Perform local, non-network preflight checks without exposing secrets."""
    defaults = config_defaults()
    cfg = {**defaults, **(config or {})}
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, message: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "message": message})

    for key in ("engine_root", "v1_root", "pathxai_root"):
        value = str(cfg.get(key, "") or "")
        check(key, bool(value and Path(value).is_dir()), f"Directory {'available' if value and Path(value).is_dir() else 'not found'}: {value or '(empty)'}")
    frozen_roots = [Path(str(cfg.get(key))).expanduser().resolve() for key in
                    ("engine_root", "v1_root", "pathxai_root", "medsam3_root", "segmentation_source_root")
                    if cfg.get(key)]
    for key in ("roi_cache_dir", "seg_cache_dir", "shared_artifacts", "llm_cache_dir", "runtime_cache_dir", "output_root"):
        value = str(cfg.get(key) or "")
        target = Path(value).expanduser().resolve() if value else Path("/")
        under_source = any(target == source or source in target.parents for source in frozen_roots)
        check(f"{key}_writable_location", bool(value) and not under_source,
              f"{key} must be configured outside frozen source roots" if under_source else f"{key} configured" if value else f"{key} is empty")
    v1_root = Path(str(cfg.get("v1_root") or "")).expanduser()
    v1_import_root = v1_root / "src" if (v1_root / "src" / "pathxai_agent").is_dir() else v1_root
    v1_ok = (v1_import_root / "pathxai_agent").is_dir()
    check("v1_source", v1_ok, "Frozen v1 Python package is available" if v1_ok else "Frozen v1 package missing")
    segmentation_source = Path(str(cfg.get("segmentation_source_root") or "")).expanduser()
    wrapper_ok = (segmentation_source / "pathxai_agent" / "segmentation" / "medsam_roi.py").is_file()
    check("segmentation_source", wrapper_ok, "Archived MedSAM ROI wrapper is available" if wrapper_ok else "Archived MedSAM ROI wrapper missing")
    runtime_root = Path(str(cfg.get("pathxai_root") or "")).expanduser()
    runtime_ok = (runtime_root / "train_medsam_FINAL.py").is_file()
    check("segmentation_runtime", runtime_ok, "Frozen segmentation runtime entry is available" if runtime_ok else "Frozen train_medsam_FINAL.py missing")
    medsam3_root = Path(str(cfg.get("medsam3_root") or "")).expanduser()
    medsam3_ok = (medsam3_root / "sam3" / "model_builder.py").is_file()
    check("medsam3_source", medsam3_ok, "Frozen MedSAM3 source is available" if medsam3_ok else "Frozen MedSAM3 model_builder.py missing")
    bpe_path = Path(str(cfg.get("sam3_bpe") or "")).expanduser()
    bpe_ok = bpe_path.is_file()
    check("medsam3_tokenizer", bpe_ok, "MedSAM3 BPE asset is available" if bpe_ok else f"MedSAM3 BPE tokenizer asset missing: {bpe_path or '(empty)'}")
    sam3_checkpoint = Path(str(cfg.get("sam3_checkpoint") or "")).expanduser()
    sam3_checkpoint_ok = sam3_checkpoint.is_file()
    check("sam3_checkpoint", sam3_checkpoint_ok, "Pinned SAM3 checkpoint is available" if sam3_checkpoint_ok else f"Pinned SAM3 checkpoint missing: {sam3_checkpoint or '(empty)'}")
    for key in ("pipeline_python", "seg_python"):
        value = str(cfg.get(key, "") or "")
        interpreter_ok = bool(value and Path(value).is_file() and os.access(Path(value), os.X_OK))
        check(key, interpreter_ok, f"Interpreter {'available' if interpreter_ok else 'not found or not executable'}: {value or '(empty)'}")
    for key in ("seg_ckpt", "keep_root"):
        value = str(cfg.get(key, "") or "")
        required = key == "seg_ckpt" or bool(config.get("mode") == "full")
        ok = bool(value and (Path(value).is_file() if key == "seg_ckpt" else Path(value).is_dir()))
        check(key, ok if required else True, f"{'Available' if ok else 'Required asset missing' if required else 'Optional asset not configured'}: {value or '(empty)'}")
    api = str(cfg.get("qwen_api", "") or "")
    parsed_api = urlsplit(api if "://" in api else f"http://{api}") if api else None
    userinfo_present = bool(parsed_api and "@" in parsed_api.netloc)
    api_ok = bool(parsed_api and parsed_api.scheme in {"http", "https"} and parsed_api.netloc
                  and not parsed_api.username and not parsed_api.password
                  and not userinfo_present and not parsed_api.query and not parsed_api.fragment)
    check("qwen_api", api_ok, "Endpoint configured" if api_ok else "Endpoint must be an http(s) URL with a host and no embedded credentials/query")
    device = str(cfg.get("device") or "")
    device_ok = bool(re.fullmatch(r"cuda:[0-9]+", device))
    check("device", device_ok, "CUDA device configured" if device_ok else "Use the supported CUDA device form, for example cuda:0")
    trace_mode_ok = str(cfg.get("trace_mode") or "record") in {"record", "replay", "off"}
    trace_path_ok = str(cfg.get("trace_mode") or "record") != "replay" or bool(cfg.get("trace_path") and Path(str(cfg["trace_path"])).is_file())
    check("trace_mode", trace_mode_ok and trace_path_ok, "Trace mode configured" if trace_mode_ok and trace_path_ok else "Use record/off, or replay with an existing trace file")
    recovery_flag_ok = isinstance(cfg.get("legacy_response_order_recovery"), bool)
    check("legacy_response_order_recovery", recovery_flag_ok,
          "Legacy response-order recovery is an explicit boolean" if recovery_flag_ok
          else "legacy_response_order_recovery must be true or false")
    warnings = []
    if not cfg.get("api_key"):
        warnings.append("No API key configured in this session; local endpoints may not require one.")
    return {"ok": all(row["ok"] for row in checks), "checks": checks, "warnings": warnings}


def resolve_profile(config: dict[str, Any], mode: str) -> dict[str, Any]:
    """Merge frozen profile settings with only path/service/device overrides."""
    profile = get_profile(mode)
    result = copy.deepcopy(profile["settings"])
    cfg = config or {}
    result.setdefault("paths", {}).update({
        "project_root": cfg.get("engine_root", _project_root()),
        "shared_artifacts": cfg.get("shared_artifacts", str(Path(cfg.get("output_root", _project_root())).parent / "shared_artifacts")),
        "v1_src_root": str(Path(cfg.get("v1_root", "")).expanduser() / "src")
        if (Path(cfg.get("v1_root", "")).expanduser() / "src" / "pathxai_agent").is_dir()
        else cfg.get("v1_root", ""),
        "outputs_root": cfg.get("output_root", ""),
        "roi_cache_dir": cfg.get("roi_cache_dir", ""),
        "seg_cache_dir": cfg.get("seg_cache_dir", ""),
        "llm_cache_dir": cfg.get("llm_cache_dir", ""),
        "seg_python": cfg.get("seg_python", ""),
        "pipeline_python": cfg.get("pipeline_python", ""),
        "seg_ckpt": cfg.get("seg_ckpt", ""),
        "keep_root": cfg.get("keep_root", ""),
    })
    result.setdefault("llm", {}).update({
        "qwen_api": cfg.get("qwen_api", "http://127.0.0.1:8000"),
        "qwen_model": cfg.get("qwen_model", ARCHIVED_QWEN_MODEL),
    })
    result["segmentation"]["device"] = cfg.get("device", "cuda:0")
    return {"mode": mode, "ablation": profile["ablation"], "preparation": profile["preparation"],
            "policy_notes": profile["policy_notes"], "settings": result}

