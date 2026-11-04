"""Diagnosis stage — predicts the primary diagnosis and differential from ROI evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Optional

from pathxai_agent_v2.llm import LLMClient, LLMJson, parse_llm_json
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
    candidate_text,
    unified_diagnosis_prompt,
    unified_diagnosis_prompt_sr1_frozen_20260722,
)


@dataclass
class DiagnosisInput:
    """Input for the diagnosis stage."""
    selected_rois: list[RoiInfo]
    site_prediction: SitePrediction
    segmentation_context: dict[str, Any] | None = None
    morphology_context: dict[str, Any] | None = None
    knowledge_context: dict[str, Any] | None = None
    allow_chat_override: bool = False
    prompt_profile: str = ""
    prompt_tag: str = ""
    image_mode: str = "he"
    image_segmentation_context: dict[str, Any] | None = None


class DiagnosisStage(PipelineStage[DiagnosisInput, DiagnosisPrediction]):
    """Stage 4: Predict primary diagnosis and differential from ROI evidence."""

    stage_name = "diagnosis"

    def run(
        self,
        stage_input: DiagnosisInput,
        dry_run: bool = False,
    ) -> StageResult[DiagnosisPrediction]:
        if dry_run:
            return ok(
                DiagnosisPrediction(
                    primary_diagnosis="DRY_RUN",
                    confidence="low",
                ),
                self.stage_name,
            )

        rois = stage_input.selected_rois
        cand_dicts = [r.to_dict() for r in rois]
        image_paths = build_llm_image_paths(
            rois,
            stage_input.image_segmentation_context or stage_input.segmentation_context,
            image_mode=stage_input.image_mode,
        )
        site_dict = stage_input.site_prediction.to_dict()

        # Single-call two-step: broad from image → subtype from context
        prompt_fn = (
            unified_diagnosis_prompt_sr1_frozen_20260722
            if stage_input.prompt_profile == "sr1_frozen_20260722"
            else unified_diagnosis_prompt
        )
        prompt = prompt_fn(
            candidate_text(cand_dicts),
            site_prediction=site_dict,
            segmentation_context=stage_input.segmentation_context,
            morphology_context=stage_input.morphology_context,
            knowledge_context=stage_input.knowledge_context,
            allow_chat_override=stage_input.allow_chat_override,
        )
        audit_dir = self._ensure_out_dir()
        (audit_dir / "prompt_diagnosis.txt").write_text(prompt, encoding="utf-8")
        (audit_dir / "prompt_inputs.json").write_text(
            json.dumps(
                {
                    "image_paths": image_paths,
                    "site_prediction": site_dict,
                    "selected_rois": cand_dicts,
                    "has_segmentation_context": bool(stage_input.segmentation_context),
                    "has_image_segmentation_context": bool(stage_input.image_segmentation_context),
                    "has_morphology_context": bool(stage_input.morphology_context),
                    "has_knowledge_context": bool(stage_input.knowledge_context),
                    "allow_chat_override": bool(stage_input.allow_chat_override),
                    "prompt_profile": stage_input.prompt_profile,
                    "prompt_tag": stage_input.prompt_tag,
                    "image_mode": stage_input.image_mode,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        if stage_input.prompt_tag:
            safe_tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage_input.prompt_tag).strip("_")
            if safe_tag:
                (audit_dir / f"prompt_diagnosis_{safe_tag}.txt").write_text(prompt, encoding="utf-8")
                (audit_dir / f"prompt_inputs_{safe_tag}.json").write_text(
                    json.dumps(
                        {
                            "image_paths": image_paths,
                            "site_prediction": site_dict,
                            "selected_rois": cand_dicts,
                            "has_segmentation_context": bool(stage_input.segmentation_context),
                            "has_image_segmentation_context": bool(stage_input.image_segmentation_context),
                            "has_morphology_context": bool(stage_input.morphology_context),
                            "has_knowledge_context": bool(stage_input.knowledge_context),
                            "allow_chat_override": bool(stage_input.allow_chat_override),
                            "prompt_profile": stage_input.prompt_profile,
                            "prompt_tag": stage_input.prompt_tag,
                            "image_mode": stage_input.image_mode,
                        },
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    ),
                    encoding="utf-8",
                )
        raw = self.llm.chat_json(prompt, image_paths=image_paths)
        parsed: LLMJson = parse_llm_json(raw)

        ihc_observed, ancillary = _sanitize_ihc_fields(parsed)

        # Merge: broad from Turn 1, subtype/malignancy/grade from Turn 2
        diagnosis = DiagnosisPrediction(
            primary_diagnosis=_clean_diagnosis(str(parsed.get("primary_diagnosis", ""))),
            confidence=str(parsed.get("confidence", "low")),
            broad_type=str(parsed.get("broad_type", "")).strip(),
            subtype=_clean_diagnosis(str(parsed.get("subtype", ""))) if str(parsed.get("subtype", "")).strip() else "",
            malignancy=str(parsed.get("malignancy", "")).strip().lower(),
            grade=str(parsed.get("grade", "")).strip(),
            differential=_str_list(parsed.get("differential", [])),
            tumor_architecture=str(parsed.get("tumor_architecture", "")),
            cytology=str(parsed.get("cytology", "")),
            necrosis=str(parsed.get("necrosis", "")),
            invasion=str(parsed.get("invasion", "")),
            stroma_inflammation=str(parsed.get("stroma_inflammation", "")),
            ihc_molecular=ihc_observed,
            missing_evidence=_str_list(parsed.get("missing_evidence", [])),
            recommended_ancillary_studies=ancillary,
            hypothesis_set=parsed.get("hypothesis_set", []),
        )

        return ok(diagnosis, self.stage_name, parsed=parsed)


def _clean_diagnosis(dx: str) -> str:
    """Remove 'not further classifiable' qualifier from the diagnosis field.

    The qualifier belongs in uncertainty fields, not primary_diagnosis.
    """
    dx = re.sub(r"\s+", " ", dx).strip()
    dx = re.split(
        r"\b(?:differential(?: diagnosis| diagnoses)?|differentials?|alternatives?|"
        r"consider(?:ed)?|versus|vs\.?)\b\s*:?",
        dx,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,;")
    # If a model returns a list-like diagnosis, keep the first asserted primary.
    if ";" in dx:
        dx = dx.split(";", 1)[0].strip(" ,")
    patterns = [
        r",?\s*not further classifiable from ROI evidence(?: without molecular markers)?",
        r",?\s*not further classifiable",
    ]
    for pat in patterns:
        if re.search(pat, dx, re.IGNORECASE):
            dx = re.sub(pat, "", dx, flags=re.IGNORECASE).strip(" ,;")
            break
    return dx or "not specified"


_IHC_RESULT_RE = re.compile(
    r"\b(positive|negative|pos\.?|neg\.?|diffuse|focal|strong|weak|lost|retained|"
    r"express(?:ed|ion)?|immunoreactive|staining|marker|markers)\b",
    re.IGNORECASE,
)

_MARKER_RE = re.compile(
    r"\b(SF-?1|inhibin|melan-?A|synaptophysin|chromogranin(?:\s*A)?|S100|SOX10|"
    r"keratin|AE1/AE3|CAM5\.2|EMA|CD45|CD20|CD3|PAX8|TTF-?1|Napsin\s*A|"
    r"GATA3|ER|PR|HER2|Ki-?67|p53|p40|p63|CK7|CK20)\b",
    re.IGNORECASE,
)


def _sanitize_ihc_fields(parsed: LLMJson) -> tuple[str, list[str]]:
    """Keep observed IHC/molecular results out of H&E-only diagnosis.

    The pipeline does not pass IHC slides or IHC metadata into this stage. A model
    may still phrase recommended markers as if they were observed positive/negative
    results, so observed IHC is always reset to not assessable and marker names are
    preserved only as recommended ancillary studies.
    """
    raw_ihc = " ".join(
        str(parsed.get(k, "") or "")
        for k in ("ihc_observed", "ihc_molecular", "immunohistochemistry", "molecular")
    ).strip()
    recommended = _str_list(parsed.get("recommended_ancillary_studies", []))
    if raw_ihc:
        markers = []
        for match in _MARKER_RE.finditer(raw_ihc):
            marker = re.sub(r"\s+", " ", match.group(0)).strip()
            if marker.lower() not in {m.lower() for m in markers}:
                markers.append(marker)
        if markers:
            recommended.extend(f"IHC: {m}" for m in markers)
        elif not _IHC_RESULT_RE.search(raw_ihc) and "not assessable" not in raw_ihc.lower():
            recommended.append(raw_ihc)

    # Stable order, case-insensitive de-duplication.
    seen = set()
    ancillary: list[str] = []
    for item in recommended:
        text = re.sub(r"\s+", " ", str(item)).strip(" .;")
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            ancillary.append(text)
    return "not assessable from H&E images", ancillary


def _str_list(val: Any) -> list[str]:
    if isinstance(val, list):
        return [str(x) for x in val]
    return []
