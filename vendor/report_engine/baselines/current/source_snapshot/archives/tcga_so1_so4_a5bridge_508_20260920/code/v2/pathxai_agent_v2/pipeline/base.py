"""Base protocol and data classes for pipeline stages.

Every stage in the pipeline conforms to the same contract:

    stage_input  →  Stage.run(stage_input)  →  StageOutput

This makes stages independently testable and composable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Optional, TypeVar

from pathxai_agent_v2.config import Settings
from pathxai_agent_v2.llm import LLMClient

# ── Type variable for stage-specific input/output ──────────────────────────────

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


@dataclass
class StageResult(Generic[TOutput]):
    """Standardized output from every pipeline stage.

    Attributes
    ----------
    data:
        The stage's primary output (type varies by stage).
    stage_name:
        Human-readable name for checkpointing / logging.
    metadata:
        Extra information (timing, model used, etc.).
    errors:
        Non-fatal warnings or issues encountered.
    checkpoint_data:
        Dict suitable for JSON serialization in checkpoints.
    """

    data: TOutput
    stage_name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    checkpoint_data: dict[str, Any] = field(default_factory=dict)


class PipelineStage(ABC, Generic[TInput, TOutput]):
    """Abstract base for a single pipeline stage.

    Subclasses implement :meth:`run`, which takes typed input and returns
    a :class:`StageResult`.

    The base class provides:
    - ``self.settings`` — the global config
    - ``self.llm`` — the LLM client for this stage
    - ``self.out_dir`` — stage-specific output directory
    """

    stage_name: str = "base"

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        out_dir: str | Path,
    ):
        self.settings = settings
        self.llm = llm
        self.out_dir = Path(out_dir)

    def _ensure_out_dir(self) -> Path:
        """Create the stage output dir on demand (only when a stage writes)."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self.out_dir

    @abstractmethod
    def run(self, stage_input: TInput, dry_run: bool = False) -> StageResult[TOutput]:
        """Execute the stage. Must be implemented by subclasses."""
        ...


# ── Shared input/output types used across stages ──────────────────────────────

@dataclass
class RoiInfo:
    """Normalized ROI representation used throughout the pipeline."""
    roi_id: str
    image_path: str
    x: int | None = None
    y: int | None = None
    width: int | None = None
    height: int | None = None
    level: int = 0
    source: str = "roi"
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_candidate(cls, roi: dict) -> "RoiInfo":
        return cls(
            roi_id=str(roi.get("roi_id", "")),
            image_path=str(roi.get("image_path", "")),
            x=roi.get("x"),
            y=roi.get("y"),
            width=roi.get("width"),
            height=roi.get("height"),
            level=int(roi.get("level", 0)),
            source=str(roi.get("source", "roi")),
            score=float(roi.get("score", 0.0)),
            metadata=dict(roi.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "roi_id": self.roi_id,
            "image_path": self.image_path,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "level": self.level,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class SitePrediction:
    """Output of the site prediction stage."""
    predicted_site: str
    confidence: str = "low"  # high / medium / low
    top_alternatives: list[str] = field(default_factory=list)
    visual_evidence: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    foundation_site_hint: dict[str, Any] = field(default_factory=dict)
    source: str = "predicted"  # "predicted" or "known_site:<src>"

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_site": self.predicted_site,
            "confidence": self.confidence,
            "top_alternatives": self.top_alternatives,
            "visual_evidence": self.visual_evidence,
            "limitations": self.limitations,
            "foundation_site_hint": self.foundation_site_hint,
            "source": self.source,
        }


@dataclass
class DiagnosisPrediction:
    """Output of the diagnosis stage.

    The diagnosis is expressed on three independent hierarchical tiers so the
    downstream report and eval can score them separately (partial credit):

    * ``broad_type``  — lineage / broad category (carcinoma, sarcoma, lymphoma,
      melanoma, glioma, germ cell tumor, neuroendocrine neoplasm, ...).
    * ``subtype``     — the specific entity (e.g. adrenocortical carcinoma).
    * ``malignancy``  — benign / borderline / malignant.
    * ``grade``       — histologic grade when assessable from H&E.
    """
    primary_diagnosis: str
    confidence: str = "low"
    broad_type: str = ""
    subtype: str = ""
    malignancy: str = ""
    grade: str = ""
    differential: list[str] = field(default_factory=list)
    tumor_architecture: str = ""
    cytology: str = ""
    necrosis: str = ""
    invasion: str = ""
    stroma_inflammation: str = ""
    ihc_molecular: str = ""
    missing_evidence: list[str] = field(default_factory=list)
    recommended_ancillary_studies: list[str] = field(default_factory=list)
    hypothesis_set: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_diagnosis": self.primary_diagnosis,
            "confidence": self.confidence,
            "broad_type": self.broad_type,
            "subtype": self.subtype,
            "malignancy": self.malignancy,
            "grade": self.grade,
            "differential": self.differential,
            "tumor_architecture": self.tumor_architecture,
            "cytology": self.cytology,
            "necrosis": self.necrosis,
            "invasion": self.invasion,
            "stroma_inflammation": self.stroma_inflammation,
            "ihc_molecular": self.ihc_molecular,
            "missing_evidence": self.missing_evidence,
            "recommended_ancillary_studies": self.recommended_ancillary_studies,
            "hypothesis_set": self.hypothesis_set,
        }


@dataclass
class EvidenceSufficiency:
    """Output of the evidence sufficiency assessment."""
    sufficient_for_diagnosis: str = "No"  # Yes / Partially / No
    diagnostic_confidence: str = "low"
    missing_image_evidence: list[str] = field(default_factory=list)
    missing_nonimage_metadata: list[str] = field(default_factory=list)
    recommended_additional_rois: list[str] = field(default_factory=list)
    can_confidently_report: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sufficient_for_diagnosis": self.sufficient_for_diagnosis,
            "diagnostic_confidence": self.diagnostic_confidence,
            "missing_image_evidence": self.missing_image_evidence,
            "missing_nonimage_metadata": self.missing_nonimage_metadata,
            "recommended_additional_rois": self.recommended_additional_rois,
            "can_confidently_report": self.can_confidently_report,
        }


# ── Helper for stage result construction ──────────────────────────────────────

def ok(data: TOutput, stage_name: str, **meta: Any) -> StageResult[TOutput]:
    """Shorthand for creating a successful StageResult."""
    return StageResult(
        data=data,
        stage_name=stage_name,
        metadata=meta,
        checkpoint_data=data.to_dict() if hasattr(data, "to_dict") else {},
    )


def build_llm_image_paths(
    rois: list["RoiInfo"],
    segmentation_context: dict[str, Any] | None,
    image_mode: str = "he",
) -> list[str]:
    """Build the image list to send to Qwen/VLM.

    Defaults to pure H&E ROI images.  The explicit ``seg_overlay_composite``
    mode is reserved for SO0-style controls that test whether directly
    showing segmentation overlays to the VLM helps or hurts, without changing
    the default SR/SS behavior.
    """
    mode = str(image_mode or "he").strip().lower()
    if mode in {"seg_overlay_composite", "composite", "overlay"} and segmentation_context:
        by_roi: dict[str, dict[str, Any]] = {}
        for row in segmentation_context.get("roi_results") or []:
            roi_id = str(row.get("roi_id") or row.get("id") or "")
            if roi_id:
                by_roi[roi_id] = row
        paths: list[str] = []
        for roi in rois:
            seg_row = by_roi.get(str(roi.roi_id), {})
            path = ""
            if mode in {"seg_overlay_composite", "composite"}:
                path = str(seg_row.get("composite_path") or "")
            if not path:
                path = str(seg_row.get("overlay_path") or "")
            if not path:
                path = roi.image_path
            paths.append(path)
        return paths
    return [r.image_path for r in rois]
