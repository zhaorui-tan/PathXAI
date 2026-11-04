"""Evidence verification stage — assesses sufficiency and identifies gaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pathxai_agent_v2.llm import LLMClient, LLMJson, parse_llm_json
from pathxai_agent_v2.pipeline.base import (
    DiagnosisPrediction,
    EvidenceSufficiency,
    PipelineStage,
    RoiInfo,
    SitePrediction,
    StageResult,
    build_llm_image_paths,
    ok,
)
from pathxai_agent_v2.prompts import candidate_text, evidence_sufficiency_prompt


@dataclass
class EvidenceInput:
    """Input for the evidence sufficiency assessment."""
    selected_rois: list[RoiInfo]
    site_prediction: SitePrediction
    diagnosis_prediction: DiagnosisPrediction
    segmentation_context: dict[str, Any] | None = None
    knowledge_context: dict[str, Any] | None = None
    morphology_context: dict[str, Any] | None = None


class EvidenceVerificationStage(PipelineStage[EvidenceInput, EvidenceSufficiency]):
    """Stage 5: Assess whether evidence is sufficient for a confident diagnosis."""

    stage_name = "evidence_sufficiency"

    def run(
        self,
        stage_input: EvidenceInput,
        dry_run: bool = False,
    ) -> StageResult[EvidenceSufficiency]:
        if dry_run:
            return ok(
                EvidenceSufficiency(
                    sufficient_for_diagnosis="DRY_RUN",
                    diagnostic_confidence="low",
                ),
                self.stage_name,
            )

        rois = stage_input.selected_rois
        cand_dicts = [r.to_dict() for r in rois]

        prompt = evidence_sufficiency_prompt(
            candidate_text(cand_dicts),
            site_prediction=stage_input.site_prediction.to_dict(),
            diagnosis_prediction=stage_input.diagnosis_prediction.to_dict(),
            segmentation_context=stage_input.segmentation_context,
            knowledge_context=stage_input.knowledge_context,
            morphology_context=stage_input.morphology_context,
        )
        image_paths = build_llm_image_paths(rois, stage_input.segmentation_context)

        raw = self.llm.chat_json(prompt, image_paths=image_paths)
        parsed: LLMJson = parse_llm_json(raw)

        suff = EvidenceSufficiency(
            sufficient_for_diagnosis=str(
                parsed.get("sufficient_for_diagnosis", "No")
            ),
            diagnostic_confidence=str(
                parsed.get("diagnostic_confidence", "low")
            ),
            missing_image_evidence=_str_list(
                parsed.get("missing_image_evidence", [])
            ),
            missing_nonimage_metadata=_str_list(
                parsed.get("missing_nonimage_metadata", [])
            ),
            recommended_additional_rois=_str_list(
                parsed.get("recommended_additional_rois", [])
            ),
            can_confidently_report=bool(
                parsed.get("can_confidently_report", False)
            ),
        )

        return ok(suff, self.stage_name, parsed=parsed)


def _str_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    return []
