"""Site prediction stage — determines the specimen organ/site from ROI images."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pathxai_agent_v2.config import Settings
from pathxai_agent_v2.leakage_guard import DiagnosisPriors
from pathxai_agent_v2.llm import LLMClient, LLMJson, parse_llm_json
from pathxai_agent_v2.pipeline.base import (
    PipelineStage,
    RoiInfo,
    SitePrediction,
    StageResult,
    ok,
)
from pathxai_agent_v2.prompts import candidate_text, site_prediction_prompt


@dataclass
class SitePredictionInput:
    """Input for the site prediction stage."""
    selected_rois: list[RoiInfo]
    site_evidence_rois: list[RoiInfo] = field(default_factory=list)
    diagnosis_priors: DiagnosisPriors | None = None
    foundation_hint: dict[str, Any] | None = None


class SitePredictionStage(PipelineStage[SitePredictionInput, SitePrediction]):
    """Stage 2: Predict the specimen site from ROI morphology."""

    stage_name = "site_prediction"

    def run(
        self,
        stage_input: SitePredictionInput,
        dry_run: bool = False,
    ) -> StageResult[SitePrediction]:
        priors = stage_input.diagnosis_priors

        # If known_site is provided via priors, skip LLM prediction
        if priors and priors.has_known_site:
            return ok(
                SitePrediction(
                    predicted_site=priors.known_site or "",
                    confidence="high",
                    visual_evidence=[
                        "Specimen/site was provided as known clinical prior."
                    ],
                    limitations=[
                        "Known site is metadata context, not inferred from ROI morphology."
                    ],
                    source=priors.site_prediction_source(),
                ),
                self.stage_name,
            )

        if dry_run:
            return ok(
                SitePrediction(
                    predicted_site="DRY_RUN",
                    confidence="low",
                    source="predicted",
                    foundation_site_hint=stage_input.foundation_hint or {},
                ),
                self.stage_name,
            )

        # Merge site evidence ROIs with selected ROIs
        all_rois = _dedupe_rois(
            stage_input.site_evidence_rois + stage_input.selected_rois
        )
        cand_dicts = [r.to_dict() for r in all_rois]

        prompt = site_prediction_prompt(
            candidate_text(cand_dicts),
            foundation_hint=stage_input.foundation_hint,
        )
        image_paths = [r.image_path for r in all_rois]

        raw = self.llm.chat_json(prompt, image_paths=image_paths)
        parsed: LLMJson = parse_llm_json(raw)

        return ok(
            SitePrediction(
                predicted_site=str(parsed.get("predicted_site", "")),
                confidence=str(parsed.get("confidence", "low")),
                top_alternatives=_str_list(parsed.get("top_alternatives", [])),
                visual_evidence=_str_list(parsed.get("visual_evidence", [])),
                limitations=_str_list(parsed.get("limitations", [])),
                foundation_site_hint=stage_input.foundation_hint or {},
                source="predicted",
            ),
            self.stage_name,
            parsed=parsed,
        )


def _dedupe_rois(rois: list[RoiInfo]) -> list[RoiInfo]:
    seen: set[str] = set()
    out: list[RoiInfo] = []
    for r in rois:
        if r.roi_id not in seen:
            seen.add(r.roi_id)
            out.append(r)
    return out


def _str_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    return []
