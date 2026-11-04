"""ROI selection stage — triages candidate ROIs to find the most diagnostic ones."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pathxai_agent_v2.config import Settings
from pathxai_agent_v2.llm import LLMClient, LLMJson, parse_llm_json
from pathxai_agent_v2.pipeline.base import (
    PipelineStage,
    RoiInfo,
    StageResult,
    ok,
)
from pathxai_agent_v2.prompts import candidate_text, roi_selection_prompt


@dataclass
class RoiSelectionInput:
    """Input for the ROI selection stage."""
    candidates: list[RoiInfo]
    max_selected: int = 5
    max_triage: int | None = None  # limit candidates shown to LLM


@dataclass
class RoiSelectionOutput:
    """Output of the ROI selection stage."""
    selected_rois: list[RoiInfo]
    reason: str
    roi_scores: list[dict] = field(default_factory=list)
    triage_candidate_count: int = 0
    total_candidate_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_rois": [r.to_dict() for r in self.selected_rois],
            "reason": self.reason,
            "roi_scores": self.roi_scores,
            "triage_candidate_count": self.triage_candidate_count,
            "total_candidate_count": self.total_candidate_count,
        }


class RoiSelectionStage(PipelineStage[RoiSelectionInput, RoiSelectionOutput]):
    """Stage 1: Select the most diagnostically valuable ROIs from candidates."""

    stage_name = "roi_selection"

    def run(
        self,
        stage_input: RoiSelectionInput,
        dry_run: bool = False,
    ) -> StageResult[RoiSelectionOutput]:
        candidates = stage_input.candidates
        max_sel = stage_input.max_selected
        triage_n = min(len(candidates), stage_input.max_triage or len(candidates))
        triage_n = max(max_sel, triage_n)
        top = candidates[:triage_n]

        if dry_run:
            selected = top[:min(max_sel, len(top))]
            return ok(
                RoiSelectionOutput(
                    selected_rois=selected,
                    reason="DRY_RUN: selected top foundation-ranked ROIs.",
                    triage_candidate_count=len(top),
                    total_candidate_count=len(candidates),
                ),
                self.stage_name,
            )

        # Build candidate dicts for the prompt
        cand_dicts = [r.to_dict() for r in top]
        prompt = roi_selection_prompt(candidate_text(cand_dicts), max_sel)
        image_paths = [r.image_path for r in top]

        raw = self.llm.chat_json(prompt, image_paths=image_paths)
        parsed: LLMJson = parse_llm_json(raw)

        indices = _int_list(parsed.get("selected_roi_indices", []))
        indices = [i for i in indices if 1 <= i <= len(top)]

        selected = [top[i - 1] for i in indices[:max_sel]]
        if not selected:
            selected = top[:min(max_sel, len(top))]

        return ok(
            RoiSelectionOutput(
                selected_rois=selected,
                reason=str(parsed.get("reason", "")),
                roi_scores=parsed.get("roi_scores", []),
                triage_candidate_count=len(top),
                total_candidate_count=len(candidates),
            ),
            self.stage_name,
            parsed=parsed,
        )


def _int_list(val: Any) -> list[int]:
    if isinstance(val, list):
        return [int(x) for x in val if isinstance(x, (int, float))]
    return []
