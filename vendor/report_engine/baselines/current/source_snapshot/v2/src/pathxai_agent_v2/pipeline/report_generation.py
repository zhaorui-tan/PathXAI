"""Report generation stage — produces the final structured pathology report."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import Any

from pathxai_agent_v2.config import Settings
from pathxai_agent_v2.llm import LLMClient
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
from pathxai_agent_v2.prompts import candidate_text, long_report_prompt


EXPECTED_SECTIONS = [
    "Specimen/Site",
    "Diagnosis",
    "Tumor/Lesion Description",
    "Key Microscopic Findings",
    "Invasion/Necrosis/Margins",
    "Immunohistochemistry/Ancillary Studies",
    "Stage/Grade/Prognostic Features",
    "Final Summary",
]


@dataclass
class ReportInput:
    """Input for the report generation stage."""
    selected_rois: list[RoiInfo]
    site_prediction: SitePrediction
    diagnosis_prediction: DiagnosisPrediction
    evidence_sufficiency: EvidenceSufficiency
    segmentation_context: dict[str, Any] | None = None
    knowledge_context: dict[str, Any] | None = None
    medgemma_context: dict[str, Any] | None = None
    evidence_graph_context: dict[str, Any] | None = None
    enable_claim_filter: bool = True
    use_natural_language_stats: bool = False
    seg_stats_style: str = "compact_json"
    lock_diagnosis_fields: bool = False
    lock_report_grade_field: bool = False
    image_mode: str = "he"
    image_segmentation_context: dict[str, Any] | None = None
    auxiliary_evidence_context: dict[str, Any] | None = None


@dataclass
class ReportOutput:
    """Output of the report generation stage."""
    long_report: str
    sections: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "long_report": self.long_report,
            "sections": self.sections,
        }


class ReportGenerationStage(PipelineStage[ReportInput, ReportOutput]):
    """Stage 6: Generate the final structured pathology report."""

    stage_name = "report_generation"

    def run(
        self,
        stage_input: ReportInput,
        dry_run: bool = False,
    ) -> StageResult[ReportOutput]:
        if dry_run:
            return ok(
                ReportOutput(
                    long_report=_dry_run_report(),
                    sections={},
                ),
                self.stage_name,
            )

        rois = stage_input.selected_rois
        cand_dicts = [r.to_dict() for r in rois]

        diagnosis_payload = _report_diagnosis_payload(
            stage_input.diagnosis_prediction,
            stage_input.evidence_graph_context,
            stage_input.evidence_sufficiency,
        )
        if stage_input.lock_report_grade_field or stage_input.lock_diagnosis_fields:
            staged_grade = _clean_payload_text(stage_input.diagnosis_prediction.grade)
            if staged_grade:
                diagnosis_payload["grade"] = staged_grade

        prompt = long_report_prompt(
            candidate_text(cand_dicts),
            site_prediction=stage_input.site_prediction.to_dict(),
            diagnosis_prediction=diagnosis_payload,
            sufficiency=stage_input.evidence_sufficiency.to_dict(),
            segmentation_context=stage_input.segmentation_context,
            knowledge_context=stage_input.knowledge_context,
            medgemma_context=stage_input.medgemma_context,
            auxiliary_evidence_context=stage_input.auxiliary_evidence_context,
            enforce_staged_diagnosis=self.settings.report.enforce_staged_diagnosis,
            use_natural_language_stats=stage_input.use_natural_language_stats,
            seg_stats_style=stage_input.seg_stats_style,
            lock_diagnosis_fields=stage_input.lock_diagnosis_fields,
            lock_report_grade_field=stage_input.lock_report_grade_field,
        )
        image_paths = build_llm_image_paths(
            rois,
            stage_input.image_segmentation_context or stage_input.segmentation_context,
            image_mode=stage_input.image_mode,
        )

        try:
            raw = self.llm.chat(prompt, image_paths=image_paths)
            report_text = _normalize_report_sections(_sanitize_report_text(str(raw.get("content", ""))))
        except Exception as exc:
            self._write_report_attempt("report_attempt_1_exception.txt", repr(exc))
            report_text = ""
        fallback_reason = ""

        # Validate report structure — retry if missing required sections
        if not _is_valid_report(report_text):
            self._write_report_attempt("report_attempt_1_invalid.txt", report_text)
            try:
                retry_raw = self.llm.chat(prompt, image_paths=image_paths)
                report_text = _normalize_report_sections(_sanitize_report_text(str(retry_raw.get("content", ""))))
            except Exception as exc:
                self._write_report_attempt("report_attempt_2_exception.txt", repr(exc))
                report_text = ""
            if not _is_valid_report(report_text):
                self._write_report_attempt("report_attempt_2_invalid.txt", report_text)
                fallback_reason = "llm_report_validation_failed"
                report_text = _diagnosis_preserving_fallback_report(
                    diagnosis_payload=diagnosis_payload,
                    site_prediction=stage_input.site_prediction.to_dict(),
                    sufficiency=stage_input.evidence_sufficiency.to_dict(),
                    segmentation_context=stage_input.segmentation_context,
                )

        if stage_input.enable_claim_filter:
            report_text = _claim_filter_rewrite(
                self.llm,
                report_text,
                stage_input.evidence_graph_context,
                diagnosis_payload,
                stage_input.evidence_sufficiency,
            )

        report_text = _enforce_report_payload_consistency(
            report_text,
            diagnosis_payload,
            lock_diagnosis_fields=stage_input.lock_diagnosis_fields,
            lock_report_grade_field=stage_input.lock_report_grade_field,
        )
        sections = _extract_sections(report_text)
        return ok(
            ReportOutput(
                long_report=report_text,
                sections=sections,
            ),
            self.stage_name,
            fallback_reason=fallback_reason,
        )

    def _write_report_attempt(self, filename: str, text: str) -> None:
        try:
            out_dir = self._ensure_out_dir()
            (out_dir / filename).write_text(text or "", encoding="utf-8")
        except Exception:
            pass


def _has_required_sections(text: str) -> bool:
    """Check that the report contains all required section headers."""
    return all(
        re.search(re.escape(sec), text, re.IGNORECASE)
        for sec in EXPECTED_SECTIONS
    )


def _is_valid_report(text: str) -> bool:
    if _is_dry_run_report_text(text):
        return False
    if not _has_required_sections(text):
        return False
    sections = _extract_sections(text)
    min_chars = {
        "Diagnosis": 8,
        "Key Microscopic Findings": 30,
        "Final Summary": 30,
    }
    for sec, min_len in min_chars.items():
        if len(sections.get(sec, "").strip()) < min_len:
            return False
    return len(text.strip()) >= 900


def _is_dry_run_report_text(text: str) -> bool:
    lowered = (text or "").lower()
    return "dry_run" in lowered or "no diagnosis generated" in lowered


def _diagnosis_preserving_fallback_report(
    *,
    diagnosis_payload: dict[str, Any],
    site_prediction: dict[str, Any],
    sufficiency: dict[str, Any],
    segmentation_context: dict[str, Any] | None,
) -> str:
    """Build a valid report from the final diagnosis state if report LLM fails.

    This is intentionally not a DRY_RUN placeholder. The upstream diagnosis
    stage may already contain a usable case-level diagnosis; using a dry-run
    string would erase that state and corrupt subtype/grade evaluation.
    """
    primary = _clean_payload_text(diagnosis_payload.get("primary_diagnosis"))
    subtype = _clean_payload_text(diagnosis_payload.get("subtype"))
    broad = _clean_payload_text(diagnosis_payload.get("broad_type"))
    confidence = _clean_payload_text(diagnosis_payload.get("confidence"))
    grade = _clean_payload_text(diagnosis_payload.get("grade"))
    architecture = _clean_payload_text(diagnosis_payload.get("tumor_architecture"))
    cytology = _clean_payload_text(diagnosis_payload.get("cytology"))
    necrosis = _clean_payload_text(diagnosis_payload.get("necrosis"))
    invasion = _clean_payload_text(diagnosis_payload.get("invasion"))
    stroma = _clean_payload_text(diagnosis_payload.get("stroma_inflammation"))
    malignancy = _clean_payload_text(diagnosis_payload.get("malignancy"))
    site = _clean_payload_text(site_prediction.get("predicted_site")) or "not specified from ROI evidence"
    missing_image = sufficiency.get("missing_image_evidence") or []
    missing_nonimage = sufficiency.get("missing_nonimage_metadata") or []
    recommended = diagnosis_payload.get("recommended_ancillary_studies") or []
    differential = diagnosis_payload.get("differential") or []

    if not primary:
        primary = subtype or broad or "neoplasm, not further specified from current ROI evidence"

    diagnosis_lines = [primary]
    if confidence:
        diagnosis_lines.append(f"Diagnostic confidence: {confidence}.")

    lesion_lines = []
    if broad:
        lesion_lines.append(f"The broad diagnostic family is {broad}.")
    if subtype:
        lesion_lines.append(f"The favored histologic subtype is {subtype}.")
    if architecture:
        lesion_lines.append(f"Architecture: {architecture}.")
    if not lesion_lines:
        lesion_lines.append("The lesion is described from the selected H&E ROI evidence; detailed subtype-level features are limited by available fields.")

    findings = []
    if cytology:
        findings.append(f"Cytology: {cytology}.")
    if stroma:
        findings.append(f"Stroma/inflammation: {stroma}.")
    seg_summary = _compact_segmentation_summary(segmentation_context)
    if seg_summary:
        findings.append(seg_summary)
    if not findings:
        findings.append("Microscopic interpretation is based on the selected H&E ROI images and the final diagnosis state; no unsupported ancillary findings are added.")

    inv_lines = []
    if invasion:
        inv_lines.append(f"Invasion: {invasion}.")
    else:
        inv_lines.append("Invasion is not definitively assessable from the selected ROI evidence.")
    if necrosis:
        inv_lines.append(f"Necrosis: {necrosis}.")
    else:
        inv_lines.append("Necrosis is not definitively assessable from the selected ROI evidence.")
    inv_lines.append("Margins, lymph nodes, gross size, and stage are not assessable from H&E ROI evidence.")

    ancillary_lines = ["IHC, molecular testing, stage, margins, and gross findings are not assessable from H&E ROI images."]
    if recommended:
        ancillary_lines.append("Recommended ancillary studies for confirmation/differential diagnosis: " + "; ".join(map(str, recommended[:8])) + ".")

    stage_lines = []
    if malignancy.lower() in {"benign", "borderline", "malignant"}:
        stage_lines.append(f"Malignancy: {malignancy}.")
    else:
        stage_lines.append("Malignancy status follows the final diagnosis state when explicitly supported; otherwise it is not assessable from ROI evidence alone.")
    if grade:
        stage_lines.append(f"Grade: {grade}.")
    else:
        stage_lines.append("Grade is not specified unless supported by the final diagnosis state.")
    if missing_image or missing_nonimage:
        limits = [str(x) for x in list(missing_image)[:3] + list(missing_nonimage)[:3] if x]
        if limits:
            stage_lines.append("Key limitations: " + "; ".join(limits) + ".")

    summary_lines = [f"Final diagnosis favored by the staged diagnostic state: {primary}."]
    if subtype and subtype.lower() not in primary.lower():
        summary_lines.append(f"Subtype context: {subtype}.")
    if differential:
        summary_lines.append("Differential diagnoses considered: " + "; ".join(map(str, differential[:5])) + ".")
    summary_lines.append("This fallback report preserves the validated diagnosis state after report-writer validation failure and does not invent ancillary data.")

    text = "\n\n".join([
        "Specimen/Site:\n" + site,
        "Diagnosis:\n" + "\n".join(diagnosis_lines),
        "Tumor/Lesion Description:\n" + "\n".join(lesion_lines),
        "Key Microscopic Findings:\n" + "\n".join(findings),
        "Invasion/Necrosis/Margins:\n" + "\n".join(inv_lines),
        "Immunohistochemistry/Ancillary Studies:\n" + "\n".join(ancillary_lines),
        "Stage/Grade/Prognostic Features:\n" + "\n".join(stage_lines),
        "Final Summary:\n" + " ".join(summary_lines),
    ])
    return _normalize_report_sections(_sanitize_report_text(text))


def _compact_segmentation_summary(segmentation_context: dict[str, Any] | None) -> str:
    if not segmentation_context:
        return ""
    summaries = segmentation_context.get("compact_stats") or segmentation_context.get("summary") or {}
    if isinstance(summaries, str):
        return f"Segmentation summary: {summaries}."
    if not isinstance(summaries, dict):
        return ""
    parts = []
    for key in ("neoplastic_fraction", "connective_fraction", "inflammatory_fraction", "necrosis_fraction", "gland_fraction"):
        value = summaries.get(key)
        if isinstance(value, (int, float)):
            parts.append(f"{key.replace('_', ' ')} {value:.2f}")
    if parts:
        return "Segmentation-derived compact evidence: " + "; ".join(parts) + "."
    return ""


def _report_diagnosis_payload(
    diagnosis: DiagnosisPrediction,
    evidence_graph_context: dict[str, Any] | None,
    sufficiency: EvidenceSufficiency,
) -> dict[str, Any]:
    payload = diagnosis.to_dict()
    filtered = (evidence_graph_context or {}).get("filtered_slots") or {}
    if filtered:
        for key in (
            "grade",
            "tumor_architecture",
            "cytology",
            "necrosis",
            "invasion",
            "stroma_inflammation",
            "ihc_molecular",
            "missing_evidence",
            "recommended_ancillary_studies",
        ):
            if key in filtered and filtered[key] not in ("", [], None):
                payload[key] = filtered[key]
    payload["ihc_molecular"] = "not assessable from H&E images"
    if not sufficiency.can_confidently_report and payload.get("subtype"):
        payload["missing_evidence"] = _dedupe([
            *[str(x) for x in payload.get("missing_evidence", []) if x],
            "current ROI evidence is insufficient for a definitive subtype-level diagnosis",
        ])
    return payload


def _claim_filter_rewrite(
    llm: LLMClient,
    report_text: str,
    evidence_graph_context: dict[str, Any] | None,
    diagnosis_payload: dict[str, Any],
    sufficiency: EvidenceSufficiency,
) -> str:
    """Rewrite final report so unsupported claims are softened/removed."""
    graph = evidence_graph_context or {}
    claims = graph.get("verified_claims") or graph.get("claims") or []
    if not claims:
        return report_text
    prompt = f"""/no_think
Rewrite this structured pathology report so every statement obeys claim verification.

Diagnosis payload:
{json.dumps(diagnosis_payload, indent=2, ensure_ascii=False)}

Evidence sufficiency:
{json.dumps(sufficiency.to_dict(), indent=2, ensure_ascii=False)}

Claim verification:
{json.dumps(claims, indent=2, ensure_ascii=False)}

Draft report:
---
{report_text}
---

Rules:
- Supported claims may be written affirmatively.
- Weak claims must be hedged as possible/suggestive.
- Unsupported or contradicted claims must be removed.
- Not-assessable claims must be written as "not assessable from H&E ROI evidence".
- Do not downgrade malignancy to not-assessable when the diagnosis payload
  explicitly states benign, borderline, or malignant. In that case only grade,
  Weiss score, IHC, molecular, stage, margins, and other unavailable fields are
  not assessable.
- Do not invent IHC, molecular, stage, gross size, margin distance, or lymph-node count.
- The Diagnosis section must contain only the main diagnosis entity, not limitations,
  differential diagnoses, or qualifiers.
- Keep exactly these sections:
{chr(10).join(EXPECTED_SECTIONS)}

Return only the rewritten report text."""
    try:
        raw = llm.chat(prompt, image_paths=[])
        rewritten = _normalize_report_sections(_sanitize_report_text(str(raw.get("content", ""))))
    except Exception:
        return report_text
    if _is_valid_report(rewritten):
        return rewritten
    return report_text


def _enforce_report_payload_consistency(
    text: str,
    diagnosis_payload: dict[str, Any],
    *,
    lock_diagnosis_fields: bool = False,
    lock_report_grade_field: bool = False,
) -> str:
    """Deterministically keep final report fields aligned with final diagnosis.

    LLM claim-rewrite can overgeneralize "not assessable" limitations. For
    example, when subtype/grade or Weiss-score evidence is insufficient it may
    rewrite the whole malignancy field as not assessable, even though the final
    diagnosis payload explicitly says malignant. Keep the primary diagnosis
    hierarchy authoritative, while leaving unavailable prognostic details as
    not assessable.
    """
    sections = _extract_sections(text)
    if not sections:
        return text

    primary = _clean_payload_text(diagnosis_payload.get("primary_diagnosis"))
    subtype = _clean_payload_text(diagnosis_payload.get("subtype"))
    grade = _clean_payload_text(diagnosis_payload.get("grade"))
    invasion = _clean_payload_text(diagnosis_payload.get("invasion"))
    necrosis = _clean_payload_text(diagnosis_payload.get("necrosis"))
    if lock_diagnosis_fields:
        if primary:
            sections["Diagnosis"] = primary

    malignancy = str(diagnosis_payload.get("malignancy") or "").strip().lower()
    if malignancy not in {"benign", "borderline", "malignant"} and not lock_diagnosis_fields:
        return text

    stage = sections.get("Stage/Grade/Prognostic Features", "").strip()
    malig_line = ""
    if malignancy == "malignant":
        if (lock_diagnosis_fields or lock_report_grade_field) and grade:
            malig_line = f"Malignancy: malignant based on the locked final diagnosis; grade: {grade}."
        else:
            malig_line = (
                "Malignancy: malignant based on the final diagnosis; exact grade, "
                "Weiss score, stage, and prognostic risk are not assessable from "
                "current H&E ROI evidence."
            )
    elif malignancy == "borderline":
        malig_line = (
            "Malignancy: borderline/indeterminate malignant potential based on "
            "the final diagnosis; exact grade, stage, and prognostic risk are "
            "not assessable from current H&E ROI evidence."
        )
    elif malignancy == "benign":
        malig_line = (
            "Malignancy: benign based on the final diagnosis; stage and "
            "prognostic risk are not applicable or not assessable from current "
            "H&E ROI evidence."
        )

    lines = [ln.strip() for ln in stage.splitlines() if ln.strip()]
    replaced = not malig_line
    new_lines: list[str] = []
    for line in lines:
        if re.match(r"^[-*]?\s*malignancy\s*:", line, flags=re.IGNORECASE):
            if malig_line:
                new_lines.append(malig_line)
            replaced = True
        else:
            new_lines.append(line)
    if not replaced and malig_line:
        new_lines.insert(0, malig_line)
    if lock_diagnosis_fields:
        locked_lines = []
        if subtype:
            locked_lines.append(f"Subtype: {subtype}.")
        if grade:
            locked_lines.append(f"Grade: {grade}.")
        if invasion:
            locked_lines.append(f"Invasion: {invasion}.")
        if necrosis:
            locked_lines.append(f"Necrosis: {necrosis}.")
        existing = "\n".join(new_lines).lower()
        for line in locked_lines:
            key = line.split(":", 1)[0].lower()
            if key not in existing:
                new_lines.append(line)
    elif lock_report_grade_field and grade:
        new_lines = _replace_or_append_grade_line(new_lines, grade)
    sections["Stage/Grade/Prognostic Features"] = "\n".join(new_lines)

    summary = sections.get("Final Summary", "")
    summary = re.sub(
        r"\bDefinitive diagnosis and grading require\b",
        "Further grading and ancillary confirmation require",
        summary,
        flags=re.IGNORECASE,
    )
    sections["Final Summary"] = summary

    return "\n\n".join(f"{sec}:\n{sections.get(sec, '').strip()}" for sec in EXPECTED_SECTIONS)


def _replace_or_append_grade_line(lines: list[str], grade: str) -> list[str]:
    """Ensure the final report verbalizes the diagnosis payload grade.

    Some smaller report writers copy the diagnosis correctly but then say the
    exact grade is not assessable in the Stage/Grade section.  When the staged
    diagnosis payload already contains a grade, keep that field explicit while
    still allowing stage, margins, and other prognostic fields to remain
    not-assessable.
    """
    grade_line = f"Grade: {grade}."
    out: list[str] = []
    replaced = False
    for line in lines:
        if re.match(r"^[-*]?\s*grade\s*:", line, flags=re.IGNORECASE):
            out.append(grade_line)
            replaced = True
            continue
        cleaned = re.sub(
            r"\bexact grade,\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"\bgrade,\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        if cleaned.strip():
            out.append(cleaned.strip())
    if not replaced:
        out.append(grade_line)
    return out


def _clean_payload_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text or text.lower() in {"not specified", "unknown", "none", "null", "not assessable"}:
        return ""
    return text


def _sanitize_report_text(text: str) -> str:
    """Remove observed IHC claims that leak through the final report."""
    if not text:
        return text
    marker = (
        r"(synaptophysin|chromogranin(?:\s*A)?|S100|SOX10|SF-?1|inhibin|melan-?A|"
        r"keratin|AE1/AE3|CAM5\.2|EMA|CD45|CD20|CD3|PAX8|TTF-?1|Napsin\s*A|"
        r"GATA3|ER|PR|HER2|Ki-?67|p53|p40|p63|CK7|CK20)"
    )
    result_re = re.compile(
        rf"\b({marker})\b[^.\n;]*\b(positive|negative|diffuse|focal|strong|weak|lost|retained|expressed|immunoreactive)\b[^.\n]*",
        re.IGNORECASE,
    )
    text = result_re.sub("IHC/molecular results are not assessable from H&E images", text)
    return text


def _normalize_report_sections(text: str) -> str:
    """Keep the primary diagnosis clean and move qualifiers to proper sections."""
    sections = _extract_sections(text)
    if not sections:
        return text

    diagnosis = sections.get("Diagnosis", "")
    diag_lines = [ln.strip() for ln in diagnosis.splitlines() if ln.strip()]
    moved_lines: list[str] = []
    clean_diag: list[str] = []
    for i, line in enumerate(diag_lines):
        low = line.lower()
        is_qualifier = (
            i > 0
            or low.startswith(("malignancy", "grade", "subtype", "differential", "limitation"))
            or "not assessable" in low
            or "differential" in low
        )
        if is_qualifier:
            moved_lines.append(line)
        else:
            clean_diag.append(line)
    if clean_diag:
        sections["Diagnosis"] = clean_diag[0]
    if moved_lines:
        stage = sections.get("Stage/Grade/Prognostic Features", "").strip()
        sections["Stage/Grade/Prognostic Features"] = "\n".join(
            [x for x in [stage, *moved_lines] if x]
        )

    return "\n\n".join(f"{sec}:\n{sections.get(sec, '').strip()}" for sec in EXPECTED_SECTIONS)


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen = set()
    for item in items:
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out


def _extract_sections(text: str) -> dict[str, str]:
    """Parse a structured report into a section dict."""
    sections: dict[str, str] = {}
    current = ""
    current_lines: list[str] = []

    for line in text.split("\n"):
        stripped = line.strip()
        matched = False
        for sec in EXPECTED_SECTIONS:
            if re.match(re.escape(sec) + r"\s*:?", stripped, re.IGNORECASE):
                if current:
                    sections[current] = "\n".join(current_lines).strip()
                current = sec
                # Extract content after the heading
                after = re.sub(
                    r"^" + re.escape(sec) + r"\s*:?\s*",
                    "",
                    stripped,
                    flags=re.IGNORECASE,
                )
                current_lines = [after] if after else []
                matched = True
                break
        if not matched:
            if current:
                current_lines.append(line)

    if current and current_lines:
        sections[current] = "\n".join(current_lines).strip()

    return sections


def _dry_run_report() -> str:
    return """Specimen/Site: DRY_RUN — Not specified from ROI evidence.
Diagnosis: DRY_RUN — No diagnosis generated.
Tumor/Lesion Description: DRY_RUN.
Key Microscopic Findings: DRY_RUN.
Invasion/Necrosis/Margins: Not assessable.
Immunohistochemistry/Ancillary Studies: Not performed.
Stage/Grade/Prognostic Features: Not assessable.
Final Summary: DRY_RUN report generation. No real LLM was called."""
