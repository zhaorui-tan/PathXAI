"""Configuration dataclasses and loader.

The dataclasses mirror the sections of ``defaults.yaml``.  Loading merges, in
order: bundled defaults -> optional user YAML -> environment overrides ->
explicit keyword overrides.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

_DEFAULTS_PATH = Path(__file__).with_name("defaults.yaml")


# --------------------------------------------------------------------------- #
# Section dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class PathsConfig:
    project_root: str = "/local/PathXAI_Agent_v2"
    shared_artifacts: str = "/local/PathXAI_Agent_v2/shared_artifacts"
    outputs_root: str = "/local/PathXAI_Agent_v2/outputs"
    v1_src_root: str = "/local/PathXAI_Agent/src"  # for reusing v1 modules
    seg_python: str = "/local/conda_staffs/conda_envs/medsam/bin/python"
    pipeline_python: str = "/local/conda_staffs/conda_envs/conch/bin/python"
    seg_ckpt: str = ""
    seg_cache_dir: str = "/local/PathXAI_Agent_v2/shared_artifacts/seg_cache/medsam3_d1"
    roi_cache_dir: str = "/local/PathXAI_Agent_v2/shared_artifacts/roi_cache"
    llm_cache_dir: str = "/local/PathXAI_Agent_v2/shared_artifacts/llm_cache"
    keep_root: str = "/local/datasets/public/KEEP_dataset"


@dataclass
class LLMConfig:
    qwen_api: str = "127.0.0.1:8000"
    qwen_model: str = "qwen"
    medgemma_api: str = "127.0.0.1:8001"
    medgemma_model: str = "google/medgemma-4b-it"
    read_timeout: int = 600
    connect_timeout: int = 10
    max_retries: int = 2
    retry_backoff: float = 3.0
    image_max_edge: int = 768
    image_jpeg_quality: int = 75
    temperature: float = 0.2
    max_tokens: int = 4096
    report_api: str = ""
    report_model: str = ""
    report_max_tokens: int = 4096

    @property
    def timeout(self) -> tuple[int, int]:
        """(connect, read) tuple in the shape the requests client expects."""
        return (self.connect_timeout, self.read_timeout)


@dataclass
class SegmentationConfig:
    enabled: bool = True
    device: str = "cuda:0"
    patch_size: int = 4096
    stride: int = 2048
    max_rois: int = 11
    use_subprocess: bool = True
    qc_max_images: int = 11
    qc_missing_retries: int = 2
    qc_cache_version: str = "seg_qc_pair_v2"


@dataclass
class RoiConfig:
    candidate_rois: int = 20
    selected_rois: int = 3
    max_site_rois: int = 3
    llm_triage_candidates: int = 8
    foundation_model: str = "conch"
    foundation_ranking_mode: str = "diagnostic"
    foundation_tile_size: int = 1024
    foundation_mpp: float = 0.5
    foundation_batch_size: int = 16


@dataclass
class ReinspectionConfig:
    max_reinspection_rois: int = 2
    max_reinspection_rounds: int = 4
    max_zoom_depth: int = 1
    max_final_rois: int = 11


@dataclass
class ContextBudgetConfig:
    medgemma_max_chars: int = 6000
    segmentation_max_chars: int = 12000
    knowledge_max_chars: int = 4000


@dataclass
class KnowledgeConfig:
    enabled: bool = True
    max_disease: int = 5
    max_semantic_groups: int = 5
    max_captions_per_hypothesis: int = 3
    max_children: int = 5


@dataclass
class SessionConfig:
    memory_version: str = "diagnostic_session_memory_v1"


@dataclass
class LeakageConfig:
    """Controls what label-derived information may reach the agent.

    ``known_cancer_type`` deliberately has no field here — the ground-truth
    cancer type must never be supplied to the agent. See leakage_guard.py.
    """

    known_site_from: str = "tcga"  # "none" | "tcga"
    min_hypotheses: int = 3

    def __post_init__(self) -> None:
        allowed = {"none", "tcga"}
        if self.known_site_from not in allowed:
            raise ValueError(
                f"known_site_from must be one of {sorted(allowed)}, "
                f"got {self.known_site_from!r}"
            )


@dataclass
class ReportConfig:
    enforce_staged_diagnosis: bool = False
    long_report_format: str = (
        "Specimen/Site:\nDiagnosis:\nTumor/Lesion Description:\n"
        "Key Microscopic Findings:\nInvasion/Necrosis/Margins:\n"
        "Immunohistochemistry/Ancillary Studies:\nStage/Grade/Prognostic Features:\n"
        "Final Summary:"
    )


@dataclass
class EvalConfig:
    """Evaluation configuration — uses a separate LLM from generation."""
    llm_api: str = ""  # empty = use rule-based extraction only
    llm_model: str = "qwen"
    synonym_map_path: str = ""
    report_sections: tuple[str, ...] = (
        "Specimen/Site", "Diagnosis", "Tumor/Lesion Description",
        "Key Microscopic Findings", "Invasion/Necrosis/Margins",
        "Immunohistochemistry/Ancillary Studies",
        "Stage/Grade/Prognostic Features", "Final Summary",
    )
    field_weights: dict[str, float] = field(default_factory=lambda: {
        "site": 0.15, "cancer_type": 0.20, "histologic_subtype": 0.25,
        "malignancy_status": 0.15, "invasive_status": 0.15,
        "grade": 0.05, "necrosis": 0.05,
    })
    # Independent tiers for the hierarchical diagnosis score. Site is a known
    # input and is NOT scored; broad/subtype/malignancy/grade are scored
    # separately so a correct broad lineage still earns credit when the subtype
    # is wrong. Grade is case-specific (GT from the reference report); when a
    # case has no reference grade its weight is renormalized away.
    tier_weights: dict[str, float] = field(default_factory=lambda: {
        "broad": 0.35, "subtype": 0.35, "malignancy": 0.15, "grade": 0.15,
    })


@dataclass
class Settings:
    paths: PathsConfig = field(default_factory=PathsConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    reinspection: ReinspectionConfig = field(default_factory=ReinspectionConfig)
    context_budget: ContextBudgetConfig = field(default_factory=ContextBudgetConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    leakage: LeakageConfig = field(default_factory=LeakageConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# Section key -> dataclass type. Used to rebuild Settings from a merged dict.
_SECTION_TYPES: dict[str, type] = {
    "paths": PathsConfig,
    "llm": LLMConfig,
    "segmentation": SegmentationConfig,
    "roi": RoiConfig,
    "reinspection": ReinspectionConfig,
    "context_budget": ContextBudgetConfig,
    "knowledge": KnowledgeConfig,
    "session": SessionConfig,
    "leakage": LeakageConfig,
    "report": ReportConfig,
    "eval": EvalConfig,
}

# Environment overrides: ENV_VAR -> (section, field). Mirrors the env vars the
# legacy shell scripts exported, so existing run configs keep working.
ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "SHARED_ROOT": ("paths", "shared_artifacts"),
    "V1_SRC_ROOT": ("paths", "v1_src_root"),
    "SEG_PYTHON": ("paths", "seg_python"),
    "SEG_CKPT": ("paths", "seg_ckpt"),
    "SEG_CACHE_DIR": ("paths", "seg_cache_dir"),
    "ROI_CACHE_DIR": ("paths", "roi_cache_dir"),
    "QWEN_API": ("llm", "qwen_api"),
    "REPORT_LLM_API": ("llm", "report_api"),
    "REPORT_LLM_MODEL": ("llm", "report_model"),
    "REPORT_LLM_MAX_TOKENS": ("llm", "report_max_tokens"),
    "MEDGEMMA_API": ("llm", "medgemma_api"),
    "MEDGEMMA_MODEL": ("llm", "medgemma_model"),
    "LLM_READ_TIMEOUT": ("llm", "read_timeout"),
    "QWEN_IMAGE_MAX_EDGE": ("llm", "image_max_edge"),
    "QWEN_IMAGE_JPEG_QUALITY": ("llm", "image_jpeg_quality"),
    "PATHXAI_LLM_MAX_RETRIES": ("llm", "max_retries"),
    "PATHXAI_LLM_RETRY_BACKOFF": ("llm", "retry_backoff"),
    "SEG_DEVICE": ("segmentation", "device"),
    "SEG_MAX_ROIS": ("segmentation", "max_rois"),
    "PATHXAI_SEG_QC_MAX_IMAGES": ("segmentation", "qc_max_images"),
    "PATHXAI_SEG_QC_MISSING_RETRIES": ("segmentation", "qc_missing_retries"),
    "PATHXAI_MEDGEMMA_CONTEXT_MAX_CHARS": ("context_budget", "medgemma_max_chars"),
    "PATHXAI_SEGMENTATION_CONTEXT_MAX_CHARS": ("context_budget", "segmentation_max_chars"),
    "PATHXAI_KNOWLEDGE_CONTEXT_MAX_CHARS": ("context_budget", "knowledge_max_chars"),
    "INITIAL_SELECTED_ROIS": ("roi", "selected_rois"),
    "MAX_REINSPECTION_ROIS": ("reinspection", "max_reinspection_rois"),
    "MAX_REINSPECTION_ROUNDS": ("reinspection", "max_reinspection_rounds"),
    "MAX_ZOOM_DEPTH": ("reinspection", "max_zoom_depth"),
    "MAX_FINAL_ROIS": ("reinspection", "max_final_rois"),
    "KNOWN_SITE_FROM": ("leakage", "known_site_from"),
    "PATHXAI_KNOWN_SITE_FROM": ("leakage", "known_site_from"),
}


# --------------------------------------------------------------------------- #
# Loading / merging
# --------------------------------------------------------------------------- #
def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError(
            "PyYAML is required to load pathxai_agent_v2 configuration"
        ) from exc
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML {path} must map sections to values")
    return data


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, Mapping)
        ):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(field_type: Any, raw: Any) -> Any:
    """Coerce a string (env value) to the target field's type."""
    if field_type is bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    if field_type is int:
        return int(raw)
    if field_type is float:
        return float(raw)
    return raw


def _env_layer() -> dict[str, dict[str, Any]]:
    layer: dict[str, dict[str, Any]] = {}
    field_types = {
        section: {f.name: f.type for f in fields(cls)}
        for section, cls in _SECTION_TYPES.items()
    }
    for env_var, (section, key) in ENV_OVERRIDES.items():
        if env_var not in os.environ:
            continue
        target_type = field_types[section].get(key, str)
        # dataclass field `.type` may be a string under `from __future__ import
        # annotations`; map the common ones explicitly.
        type_map = {"int": int, "float": float, "bool": bool, "str": str}
        if isinstance(target_type, str):
            target_type = type_map.get(target_type, str)
        layer.setdefault(section, {})[key] = _coerce(target_type, os.environ[env_var])
    return layer


def _build_section(cls: type, data: Mapping[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in known})


def load_settings(
    path: str | Path | None = None,
    *,
    use_env: bool = True,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> Settings:
    """Build :class:`Settings` from the layered sources.

    Parameters
    ----------
    path:
        Optional user YAML whose sections override the bundled defaults.
    use_env:
        When True (default), apply :data:`ENV_OVERRIDES` from ``os.environ``.
    overrides:
        Final ``{section: {field: value}}`` overrides applied last (e.g. CLI).
    """
    merged = _read_yaml(_DEFAULTS_PATH)
    if path is not None:
        merged = _deep_merge(merged, _read_yaml(Path(path)))
    if use_env:
        merged = _deep_merge(merged, _env_layer())
    if overrides:
        merged = _deep_merge(merged, overrides)

    sections: dict[str, Any] = {}
    for section, cls in _SECTION_TYPES.items():
        sections[section] = _build_section(cls, merged.get(section, {}) or {})
    return Settings(**sections)


def to_dict(settings: Any) -> dict[str, Any]:
    """Recursively convert a Settings/dataclass tree to a plain dict."""
    if is_dataclass(settings) and not isinstance(settings, type):
        return {f.name: to_dict(getattr(settings, f.name)) for f in fields(settings)}
    return settings
