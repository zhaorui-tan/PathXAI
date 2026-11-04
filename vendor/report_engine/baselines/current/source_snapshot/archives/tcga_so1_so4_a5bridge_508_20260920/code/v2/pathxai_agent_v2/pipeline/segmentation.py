"""Segmentation stage — wraps MedSAM ROI segmentation via subprocess.

Reuses v1's medsam_roi.py and seg_visual.py transparently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathxai_agent_v2.config import Settings
from pathxai_agent_v2.pipeline.base import PipelineStage, RoiInfo, StageResult, ok


@dataclass
class SegmentationOutput:
    """Output of the MedSAM segmentation stage."""
    available: bool = False
    roi_results: list[dict] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    class_names: dict[int, str] = field(default_factory=dict)
    legend_path: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "roi_results": self.roi_results,
            "summary": self.summary,
            "class_names": self.class_names,
            "legend_path": self.legend_path,
            "error": self.error,
        }


class SegmentationStage(PipelineStage[list[RoiInfo], SegmentationOutput]):
    """Stage 3: Run MedSAM segmentation on selected ROIs."""

    stage_name = "segmentation"

    def run(
        self,
        stage_input: list[RoiInfo],
        dry_run: bool = False,
    ) -> StageResult[SegmentationOutput]:
        cfg = self.settings.segmentation
        if not cfg.enabled or dry_run:
            return ok(
                SegmentationOutput(
                    available=False,
                    error="disabled" if not cfg.enabled else "dry_run",
                ),
                self.stage_name,
            )

        roi_dicts = [r.to_dict() for r in stage_input[:cfg.max_rois]]
        if not roi_dicts:
            return ok(SegmentationOutput(available=False, error="no ROIs"), self.stage_name)

        try:
            import sys
            v1_src = str(Path(self.settings.paths.v1_src_root))
            if v1_src not in sys.path:
                sys.path.insert(0, v1_src)
            from pathxai_agent.segmentation.medsam_roi import MedSAMRoiSegmenter

            segmenter = MedSAMRoiSegmenter(
                ckpt_path=self.settings.paths.seg_ckpt,
                device=cfg.device,
                max_rois=cfg.max_rois,
                patch_size=cfg.patch_size,
                stride=cfg.stride,
                cache_dir=self.settings.paths.seg_cache_dir,
                python_path=self.settings.paths.seg_python,
                use_subprocess=cfg.use_subprocess,
            )
            result = segmenter.segment_rois(roi_dicts, self._ensure_out_dir().resolve())

            # Attach composite overlays
            try:
                from pathxai_agent.segmentation.seg_visual import attach_composites_to_seg_context
                result = attach_composites_to_seg_context(result)
            except Exception:
                pass

            return ok(
                SegmentationOutput(
                    available=result.get("available", False),
                    roi_results=result.get("roi_results", []),
                    summary=result.get("summary", {}),
                    class_names=result.get("class_names", {}),
                    legend_path=result.get("legend_path", ""),
                ),
                self.stage_name,
            )
        except Exception as exc:
            err_path = self._ensure_out_dir() / "segmentation_error.json"
            err_path.write_text(json.dumps({"error": str(exc)}, indent=2))
            return ok(
                SegmentationOutput(available=False, error=str(exc)),
                self.stage_name,
            )
