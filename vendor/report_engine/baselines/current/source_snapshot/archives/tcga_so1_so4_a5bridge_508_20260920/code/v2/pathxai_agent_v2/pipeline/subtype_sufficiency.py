"""Broad-hypothesis and subtype-sufficiency stages.

These stages restore the strongest part of the v1 diagnostic flow in a v2
modular form: first reason at broad lineage level, then decide whether current
ROI evidence supports a specific site-relevant subtype.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pathxai_agent_v2.llm import LLMClient, parse_llm_json
from pathxai_agent_v2.pipeline.base import (
    DiagnosisPrediction,
    PipelineStage,
    RoiInfo,
    SitePrediction,
    StageResult,
    build_llm_image_paths,
    ok,
)
from pathxai_agent_v2.prompts import (
    broad_hypotheses_prompt,
    broad_hypotheses_prompt_sr1_frozen_20260722,
    candidate_text,
    subtype_sufficiency_prompt,
    subtype_sufficiency_prompt_sr1_frozen_20260722,
)


@dataclass
class BroadHypothesesOutput:
    available: bool = False
    hypotheses: list[dict] = field(default_factory=list)
    best_broad_type: str = ""
    confidence: str = "low"
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "hypotheses": self.hypotheses,
            "best_broad_type": self.best_broad_type,
            "confidence": self.confidence,
            "reason": self.reason,
        }


@dataclass
class SubtypeSufficiencyOutput:
    available: bool = False
    best_supported_level: str = "not_assessable"
    recommended_primary_diagnosis: str = ""
    confidence: str = "low"
    subtype_assessments: list[dict] = field(default_factory=list)
    image_actionable_gaps: list[str] = field(default_factory=list)
    nonimage_metadata_gaps: list[str] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "best_supported_level": self.best_supported_level,
            "recommended_primary_diagnosis": self.recommended_primary_diagnosis,
            "confidence": self.confidence,
            "subtype_assessments": self.subtype_assessments,
            "image_actionable_gaps": self.image_actionable_gaps,
            "nonimage_metadata_gaps": self.nonimage_metadata_gaps,
            "reason": self.reason,
        }


@dataclass
class SubtypeReasoningInput:
    selected_rois: list[RoiInfo]
    site_prediction: SitePrediction
    segmentation_context: dict[str, Any] | None = None
    morphology_context: dict[str, Any] | None = None
    knowledge_context: dict[str, Any] | None = None
    broad_hypotheses: BroadHypothesesOutput | None = None
    diagnosis_prediction: DiagnosisPrediction | None = None
    prompt_profile: str = ""


class BroadHypothesesStage(PipelineStage[SubtypeReasoningInput, BroadHypothesesOutput]):
    """Generate broad lineage hypotheses before subtype commitment."""

    stage_name = "broad_hypotheses"

    def run(
        self,
        stage_input: SubtypeReasoningInput,
        dry_run: bool = False,
    ) -> StageResult[BroadHypothesesOutput]:
        if dry_run:
            return ok(BroadHypothesesOutput(available=False, reason="dry run"), self.stage_name)

        rois = stage_input.selected_rois
        prompt_fn = (
            broad_hypotheses_prompt_sr1_frozen_20260722
            if stage_input.prompt_profile == "sr1_frozen_20260722"
            else broad_hypotheses_prompt
        )
        prompt = prompt_fn(
            candidate_text([r.to_dict() for r in rois]),
            site_prediction=stage_input.site_prediction.to_dict(),
            segmentation_context=stage_input.segmentation_context,
            morphology_context=stage_input.morphology_context,
            knowledge_context=stage_input.knowledge_context,
        )
        raw = self.llm.chat_json(prompt, image_paths=build_llm_image_paths(rois, stage_input.segmentation_context))
        parsed = parse_llm_json(raw)
        out = BroadHypothesesOutput(
            available=True,
            hypotheses=parsed.get("hypotheses", []) if isinstance(parsed.get("hypotheses"), list) else [],
            best_broad_type=str(parsed.get("best_broad_type", "")).strip(),
            confidence=str(parsed.get("confidence", "low")).strip() or "low",
            reason=str(parsed.get("reason", "")).strip(),
        )
        return ok(out, self.stage_name, parsed=parsed)


class SubtypeSufficiencyStage(PipelineStage[SubtypeReasoningInput, SubtypeSufficiencyOutput]):
    """Judge if subtype/malignancy can be supported by current evidence."""

    stage_name = "subtype_sufficiency"

    def run(
        self,
        stage_input: SubtypeReasoningInput,
        dry_run: bool = False,
    ) -> StageResult[SubtypeSufficiencyOutput]:
        if dry_run:
            return ok(SubtypeSufficiencyOutput(available=False, reason="dry run"), self.stage_name)

        rois = stage_input.selected_rois
        prompt_fn = (
            subtype_sufficiency_prompt_sr1_frozen_20260722
            if stage_input.prompt_profile == "sr1_frozen_20260722"
            else subtype_sufficiency_prompt
        )
        prompt = prompt_fn(
            candidate_text([r.to_dict() for r in rois]),
            site_prediction=stage_input.site_prediction.to_dict(),
            broad_hypotheses=stage_input.broad_hypotheses.to_dict() if stage_input.broad_hypotheses else {},
            diagnosis_prediction=stage_input.diagnosis_prediction.to_dict() if stage_input.diagnosis_prediction else {},
            segmentation_context=stage_input.segmentation_context,
            morphology_context=stage_input.morphology_context,
            knowledge_context=stage_input.knowledge_context,
        )
        raw = self.llm.chat_json(prompt, image_paths=build_llm_image_paths(rois, stage_input.segmentation_context))
        parsed = parse_llm_json(raw)
        out = SubtypeSufficiencyOutput(
            available=True,
            best_supported_level=str(parsed.get("best_supported_level", "not_assessable")).strip() or "not_assessable",
            recommended_primary_diagnosis=_clean_entity(str(parsed.get("recommended_primary_diagnosis", ""))),
            confidence=str(parsed.get("confidence", "low")).strip() or "low",
            subtype_assessments=parsed.get("subtype_assessments", []) if isinstance(parsed.get("subtype_assessments"), list) else [],
            image_actionable_gaps=_str_list(parsed.get("image_actionable_gaps", [])),
            nonimage_metadata_gaps=_str_list(parsed.get("nonimage_metadata_gaps", [])),
            reason=str(parsed.get("reason", "")).strip(),
        )
        return ok(out, self.stage_name, parsed=parsed)


def apply_subtype_sufficiency_guard(
    diagnosis: DiagnosisPrediction,
    subtype_sufficiency: SubtypeSufficiencyOutput | None,
) -> DiagnosisPrediction:
    """Demote unsupported subtype/malignancy commitments into a broad diagnosis.

    The guard is intentionally conservative about *removing* the final
    diagnosis. It should prevent unsupported subtype overclaiming, but it
    should not erase a high-confidence final diagnostic entity merely because
    ancillary confirmation, grade, stage, or complete scoring criteria are
    unavailable. Those limitations belong in missing evidence / ancillary
    recommendations.
    """
    if not subtype_sufficiency or not subtype_sufficiency.available:
        return diagnosis

    level = subtype_sufficiency.best_supported_level.lower().strip()
    recommended = subtype_sufficiency.recommended_primary_diagnosis.strip()
    if level == "subtype":
        return diagnosis

    matched = _matching_subtype_assessment(diagnosis, subtype_sufficiency)
    matched_status = str((matched or {}).get("status") or "").lower()
    has_visual_support = bool((matched or {}).get("supporting_features"))
    unresolved_malignant = _has_unresolved_malignant_candidate(subtype_sufficiency)

    primary = recommended or diagnosis.broad_type or diagnosis.primary_diagnosis
    primary_low = primary.lower()
    diff = list(diagnosis.differential)
    if diagnosis.primary_diagnosis and diagnosis.primary_diagnosis != primary and diagnosis.primary_diagnosis not in diff:
        diff.insert(0, diagnosis.primary_diagnosis)

    missing = _dedupe([
        *diagnosis.missing_evidence,
        subtype_sufficiency.reason,
        *subtype_sufficiency.image_actionable_gaps,
        *subtype_sufficiency.nonimage_metadata_gaps,
    ])
    malignancy = diagnosis.malignancy
    missing_text = " ".join(missing).lower()
    needs_malignancy_downgrade = any(
        term in missing_text
        for term in (
            "malignant potential",
            "weiss",
            "mitotic",
            "atypical mitoses",
            "capsular invasion",
            "vascular invasion",
            "sinusoidal invasion",
            "ki-67",
        )
    )

    benign_claim = (
        diagnosis.malignancy.lower().strip() == "benign"
        or any(term in diagnosis.primary_diagnosis.lower() for term in ("benign", "adenoma"))
    )
    if benign_claim and unresolved_malignant and needs_malignancy_downgrade:
        return DiagnosisPrediction(
            primary_diagnosis=primary,
            confidence="medium" if diagnosis.confidence.lower() == "high" else diagnosis.confidence,
            broad_type=primary if level in {"broad_type", "not_assessable"} else diagnosis.broad_type,
            subtype="",
            malignancy="not assessable",
            grade="",
            differential=diff[:6],
            tumor_architecture=diagnosis.tumor_architecture,
            cytology=diagnosis.cytology,
            necrosis=diagnosis.necrosis,
            invasion=diagnosis.invasion,
            stroma_inflammation=diagnosis.stroma_inflammation,
            ihc_molecular=diagnosis.ihc_molecular,
            missing_evidence=missing,
            recommended_ancillary_studies=diagnosis.recommended_ancillary_studies,
            hypothesis_set=diagnosis.hypothesis_set,
        )

    # If the final diagnosis is supported or plausible-but-incomplete, keep it
    # as the best working diagnosis and put uncertainty in missing evidence.
    # Only force a broad diagnosis when the candidate is unsupported,
    # contradicted, or truly not assessable.
    keep_final_entity = (
        diagnosis.primary_diagnosis
        and diagnosis.confidence.lower() in {"high", "medium"}
        and matched_status in {"supported", "possible_but_insufficient"}
        and matched_status not in {"unsupported", "contradicted", "not_assessable"}
        and has_visual_support
    )

    if keep_final_entity:
        kept_missing = _dedupe(missing)
        return DiagnosisPrediction(
            primary_diagnosis=diagnosis.primary_diagnosis,
            confidence="medium" if matched_status == "possible_but_insufficient" else diagnosis.confidence,
            broad_type=diagnosis.broad_type,
            subtype=diagnosis.subtype,
            malignancy=diagnosis.malignancy,
            grade=diagnosis.grade,
            differential=diff[:6],
            tumor_architecture=diagnosis.tumor_architecture,
            cytology=diagnosis.cytology,
            necrosis=diagnosis.necrosis,
            invasion=diagnosis.invasion,
            stroma_inflammation=diagnosis.stroma_inflammation,
            ihc_molecular=diagnosis.ihc_molecular,
            missing_evidence=kept_missing,
            recommended_ancillary_studies=diagnosis.recommended_ancillary_studies,
            hypothesis_set=diagnosis.hypothesis_set,
        )

    if level == "not_assessable" or (needs_malignancy_downgrade and not malignancy):
        malignancy = "not assessable"
    elif not malignancy and any(term in primary_low for term in ("carcinoma", "sarcoma", "lymphoma", "melanoma")):
        malignancy = "malignant"

    return DiagnosisPrediction(
        primary_diagnosis=primary,
        confidence="medium" if diagnosis.confidence.lower() == "high" else diagnosis.confidence,
        broad_type=primary if level in {"broad_type", "not_assessable"} else diagnosis.broad_type,
        subtype="" if level != "subtype" else diagnosis.subtype,
        malignancy=malignancy,
        grade="" if level != "subtype" else diagnosis.grade,
        differential=diff[:6],
        tumor_architecture=diagnosis.tumor_architecture,
        cytology=diagnosis.cytology,
        necrosis=diagnosis.necrosis,
        invasion=diagnosis.invasion,
        stroma_inflammation=diagnosis.stroma_inflammation,
        ihc_molecular=diagnosis.ihc_molecular,
        missing_evidence=missing,
        recommended_ancillary_studies=diagnosis.recommended_ancillary_studies,
        hypothesis_set=diagnosis.hypothesis_set,
    )


def _clean_entity(text: str) -> str:
    text = " ".join(str(text or "").split()).strip(" ,;")
    lowered = text.lower()
    for marker in (
        "not further classifiable from roi evidence",
        "not further classifiable",
        "from roi evidence",
    ):
        lowered = lowered.replace(marker, "")
    # Apply replacement on original case-insensitively by falling back to lower
    # only when the model returned a qualifier-heavy phrase.
    if lowered != text.lower():
        text = lowered
    return " ".join(text.split()).strip(" ,;")


def _matching_subtype_assessment(
    diagnosis: DiagnosisPrediction,
    subtype_sufficiency: SubtypeSufficiencyOutput,
) -> dict:
    targets = [
        diagnosis.subtype,
        diagnosis.primary_diagnosis,
        diagnosis.broad_type,
    ]
    target_keys = {_norm_entity(x) for x in targets if x}
    if not target_keys:
        return {}
    for assessment in subtype_sufficiency.subtype_assessments:
        if _norm_entity(assessment.get("subtype")) in target_keys:
            return assessment
    # Fall back to loose containment for entities with modifiers.
    for assessment in subtype_sufficiency.subtype_assessments:
        cand = _norm_entity(assessment.get("subtype"))
        if any(cand and (cand in target or target in cand) for target in target_keys):
            return assessment
    return {}


def _has_unresolved_malignant_candidate(subtype_sufficiency: SubtypeSufficiencyOutput) -> bool:
    """Return true when a malignant candidate remains plausible.

    This is a generic safety guard: absence of malignant features in a small ROI
    sample should not be converted into a benign diagnosis if a malignant
    differential remains possible and the missing discriminators are exactly
    those needed to decide benign vs malignant.
    """
    malignant_terms = (
        "malignant",
        "malignancy",
        "carcinoma",
        "sarcoma",
        "lymphoma",
        "melanoma",
        "blastoma",
        "high grade",
        "invasive",
    )
    unresolved_statuses = {"supported", "possible_but_insufficient"}
    for assessment in subtype_sufficiency.subtype_assessments:
        status = str(assessment.get("status") or "").lower()
        subtype = str(assessment.get("subtype") or "").lower()
        if status in unresolved_statuses and any(term in subtype for term in malignant_terms):
            return True
    return False


def _norm_entity(text: Any) -> str:
    return " ".join(str(text or "").lower().replace("-", " ").split())


def _str_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(x) for x in val if str(x).strip()]
    return []


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in items:
        text = " ".join(str(item or "").split()).strip()
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            out.append(text)
    return out
