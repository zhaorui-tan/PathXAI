"""Enhanced pipeline orchestrator — all stages including reinspection, segmentation,
MedGemma morphology, KEEP knowledge, evidence graph, session memory, and lineage guard.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pathxai_agent_v2.config import Settings
from pathxai_agent_v2.leakage_guard import DiagnosisPriors, assert_no_cancer_type_leak
from pathxai_agent_v2.llm import CaseSessionLLMClient, LLMClient
from pathxai_agent_v2.pipeline.base import (
    DiagnosisPrediction, EvidenceSufficiency, RoiInfo, SitePrediction, StageResult,
    ok,
)
from pathxai_agent_v2.pipeline.roi_selection import RoiSelectionInput, RoiSelectionOutput, RoiSelectionStage
from pathxai_agent_v2.pipeline.site_prediction import SitePredictionInput, SitePredictionStage
from pathxai_agent_v2.pipeline.segmentation import SegmentationStage, SegmentationOutput
from pathxai_agent_v2.pipeline.morphology import (
    MorphologyStage,
    MorphologyOutput,
    SegReaderMorphologyStage,
    apply_seg_reader_guard,
)
from pathxai_agent_v2.pipeline.knowledge import KnowledgeStage, KnowledgeOutput
from pathxai_agent_v2.pipeline.diagnosis import DiagnosisInput, DiagnosisStage
from pathxai_agent_v2.pipeline.subtype_sufficiency import (
    BroadHypothesesOutput,
    BroadHypothesesStage,
    SubtypeReasoningInput,
    SubtypeSufficiencyOutput,
    SubtypeSufficiencyStage,
    apply_subtype_sufficiency_guard,
)
from pathxai_agent_v2.pipeline.lineage_guard import apply_evidence_sufficiency_guard, apply_site_lineage_guard
from pathxai_agent_v2.pipeline.evidence_verification import EvidenceInput, EvidenceVerificationStage
from pathxai_agent_v2.pipeline.reinspection import ReinspectionStage, ReinspectionPlan
from pathxai_agent_v2.pipeline.seg_policy import seg_driven_action, seg_policy_summary

# ── Valid broad types (must match _BROAD_TYPE_KEYWORDS in eval/metrics.py) ───
_VALID_BROAD_TYPES = frozenset({
    "carcinoma", "sarcoma", "lymphoma", "melanoma", "glioma",
    "neuroendocrine neoplasm", "germ cell tumor", "mesothelioma",
    "carcinosarcoma", "thymic epithelial neoplasm", "neoplasm",
})
from pathxai_agent_v2.pipeline.evidence_graph import EvidenceGraphStage, EvidenceGraphOutput
from pathxai_agent_v2.pipeline.report_generation import ReportGenerationStage, ReportInput, ReportOutput
from pathxai_agent_v2.pipeline.session_memory import SessionMemory

CKPT_FILES = {
    "roi": "ckpt_01_roi_selection.json", "site": "ckpt_02_site_prediction.json",
    "seg": "ckpt_03_segmentation.json", "morph": "ckpt_04_morphology.json",
    "knowledge": "ckpt_05_knowledge.json", "diagnosis": "ckpt_06_diagnosis.json",
    "broad": "ckpt_06a_broad_hypotheses.json", "subtype": "ckpt_06b_subtype_sufficiency.json",
    "knowledge_hypothesis": "ckpt_06c_knowledge_hypothesis.json",
    "evidence": "ckpt_07_evidence_sufficiency.json", "reinspect": "ckpt_08_reinspection.json",
    "final_broad": "ckpt_08a_final_broad_hypotheses.json",
    "final_subtype": "ckpt_08b_final_subtype_sufficiency.json",
    "final_diagnosis": "ckpt_08c_final_diagnosis.json",
    "final_knowledge": "ckpt_08d_final_knowledge.json",
    "graph": "ckpt_09_evidence_graph.json", "report": "ckpt_10_report.json",
    "final": "pipeline_result.json", "session": "session_memory.json", "error": "error.json",
}

ABSOLUTE_MAX_VISUAL_EVIDENCE = 11


def _diagnosis_signature(diag: DiagnosisPrediction | None) -> tuple[str, str, str, str]:
    """Compact signature for deciding whether KEEP hypothesis context is stale."""
    if diag is None:
        return ("", "", "", "")

    def norm(value: str) -> str:
        value = str(value or "").lower()
        value = re.sub(r"[^a-z0-9]+", " ", value)
        return " ".join(value.split())

    return (
        norm(diag.broad_type),
        norm(diag.subtype),
        norm(diag.primary_diagnosis),
        norm(diag.malignancy),
    )


def _hypothesis_changed(prev: tuple[str, str, str, str], cur: tuple[str, str, str, str]) -> bool:
    """Return True for meaningful diagnostic drift requiring KEEP refresh."""
    if not prev:
        return True
    # Broad lineage or explicit subtype changes are always material.
    if prev[0] != cur[0] or prev[1] != cur[1]:
        return True
    # If subtype is still blank, a primary-diagnosis change is material.
    if not cur[1] and prev[2] != cur[2]:
        return True
    # Malignancy changes affect differential framing and reporting constraints.
    if prev[3] != cur[3]:
        return True
    return False


def _positive_only_seg_context(seg_ctx: dict[str, Any]) -> dict[str, Any]:
    """Keep only positive, low-risk segmentation support for SS5.

    SS5 tests whether segmentation helps when exposed as minimal evidence:
    tumor burden, usable tissue, and meaningful necrosis.  It deliberately
    removes epithelial/stromal fractions and all low/absent class fractions
    because those were the main triggers for subtype drift.
    """
    ctx = json.loads(json.dumps(seg_ctx, default=str))
    for row in ctx.get("roi_results") or []:
        stats = row.get("stats") or {}
        named = stats.get("named_class_fractions") or {}
        filtered = {}
        for name, info in named.items():
            frac = 0.0
            if isinstance(info, dict):
                try:
                    frac = float(info.get("fraction") or 0.0)
                except Exception:
                    frac = 0.0
            lname = str(name).lower()
            keep = False
            if "neoplastic" in lname and frac >= 0.05:
                keep = True
            elif ("necrotic" in lname or "dead" in lname) and frac >= 0.10:
                keep = True
            if keep:
                filtered[name] = info
        stats["named_class_fractions"] = filtered
        stats["positive_only_note"] = (
            "Only positive neoplastic and substantial necrotic/dead fractions "
            "are retained. Omitted classes and low fractions are not negative "
            "evidence and must not be used for subtype, grade, or invasion."
        )
    ctx["seg_context_mode"] = "positive_only"
    return ctx


def _norm_diag_text(value: str) -> str:
    value = str(value or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _diag_is_invalid(diag: DiagnosisPrediction | None) -> bool:
    if diag is None:
        return True
    text = _norm_diag_text(f"{diag.primary_diagnosis} {diag.subtype}")
    return not text or "dry run" in text or "no diagnosis generated" in text


def _diag_specificity(diag: DiagnosisPrediction | None) -> int:
    if _diag_is_invalid(diag):
        return -1
    subtype = _norm_diag_text(diag.subtype)
    primary = _norm_diag_text(diag.primary_diagnosis)
    broad = _sanitize_broad_type(diag.broad_type)
    generic = {
        "", "neoplasm", "carcinoma", "sarcoma", "lymphoma", "melanoma",
        "glioma", "neuroendocrine neoplasm", "germ cell tumor",
        "mesothelioma", "other", "not specified",
    }
    if subtype and subtype not in generic:
        return 3
    if primary and primary not in generic and not primary.endswith(" nos"):
        return 2
    if broad and broad != "neoplasm":
        return 1
    return 0


def _subtype_richness(subtype: str) -> int:
    """Rate how information-rich a subtype name is.

    Longer, more specific entity names score higher than short generic ones.
    Penalises NOS/low-information suffixes.
    """
    text = _norm_diag_text(subtype)
    if not text:
        return -1
    score = len(text)
    generic_markers = (
        "nos", "not otherwise specified", "not further classifiable",
        "uncertain", "unclassified", "undifferentiated", "not specified",
        "carcinoma", "adenocarcinoma", "sarcoma", "neoplasm", "malignancy",
        "tumor", "tumour", "other",
    )
    for marker in generic_markers:
        if marker in text:
            score -= 20
    score -= text.count(" ") * 2
    return score


def _has_invasion_conflict(a: DiagnosisPrediction, b: DiagnosisPrediction) -> bool:
    ta = _norm_diag_text(f"{a.primary_diagnosis} {a.subtype} {a.invasion}")
    tb = _norm_diag_text(f"{b.primary_diagnosis} {b.subtype} {b.invasion}")

    def flags(t: str) -> tuple[bool, bool]:
        in_situ = bool(re.search(r"\b(?:in situ|dcis|cis)\b", t))
        invasive = "invasive" in t or "invasion" in t
        return invasive and not in_situ, in_situ

    a_inv, a_is = flags(ta)
    b_inv, b_is = flags(tb)
    return (a_inv and b_is) or (a_is and b_inv)


def _choose_dual_path_diagnosis(
    image_only: DiagnosisPrediction,
    seg_stats: DiagnosisPrediction,
) -> tuple[DiagnosisPrediction, dict[str, Any]]:
    """Guard SS7: accept seg-stats diagnosis only when it is plausibly helpful.

    This is intentionally conservative.  Seg stats should help recover a more
    specific diagnosis, but they should not change broad lineage, replace a
    specific image-supported entity with NOS, or flip invasive/in-situ state.
    """
    if _diag_is_invalid(image_only) and not _diag_is_invalid(seg_stats):
        return seg_stats, {"selected": "seg_stats", "reason": "image_only_invalid"}
    if _diag_is_invalid(seg_stats):
        return image_only, {"selected": "image_only", "reason": "seg_stats_invalid"}

    ib = _sanitize_broad_type(image_only.broad_type)
    sb = _sanitize_broad_type(seg_stats.broad_type)
    if ib != sb and ib != "neoplasm" and sb != "neoplasm":
        return image_only, {
            "selected": "image_only",
            "reason": "seg_stats_changed_broad_lineage",
            "image_broad": ib,
            "seg_broad": sb,
        }

    if _has_invasion_conflict(image_only, seg_stats):
        return image_only, {
            "selected": "image_only",
            "reason": "seg_stats_changed_invasive_vs_insitu",
        }

    i_spec = _diag_specificity(image_only)
    s_spec = _diag_specificity(seg_stats)
    if i_spec >= 2 and s_spec < i_spec:
        return image_only, {
            "selected": "image_only",
            "reason": "seg_stats_less_specific_than_image_only",
            "image_specificity": i_spec,
            "seg_specificity": s_spec,
        }

    if s_spec > i_spec:
        return seg_stats, {
            "selected": "seg_stats",
            "reason": "seg_stats_more_specific_without_guard_violation",
            "image_specificity": i_spec,
            "seg_specificity": s_spec,
        }

    return seg_stats, {
        "selected": "seg_stats",
        "reason": "same_specificity_no_guard_violation",
        "image_specificity": i_spec,
        "seg_specificity": s_spec,
    }


def _choose_dual_path_diagnosis_ss12(
    image_only: DiagnosisPrediction,
    seg_stats: DiagnosisPrediction,
) -> tuple[DiagnosisPrediction, dict[str, Any]]:
    """SS12: SS7 guard + subtype-richness tiebreaker for same-specificity.

    Unlike SS7 which defaults to seg_stats when specificity is equal, SS12
    selects the path whose subtype carries more diagnostic information.
    This prevents seg-stats from replacing a specific entity (e.g.
    \"clear cell renal cell carcinoma\") with a generic equivalent
    (\"renal cell carcinoma\") when both score the same specificity level.
    """
    if _diag_is_invalid(image_only) and not _diag_is_invalid(seg_stats):
        return seg_stats, {"selected": "seg_stats", "reason": "image_only_invalid"}
    if _diag_is_invalid(seg_stats):
        return image_only, {"selected": "image_only", "reason": "seg_stats_invalid"}

    ib = _sanitize_broad_type(image_only.broad_type)
    sb = _sanitize_broad_type(seg_stats.broad_type)
    if ib != sb and ib != "neoplasm" and sb != "neoplasm":
        return image_only, {
            "selected": "image_only",
            "reason": "seg_stats_changed_broad_lineage",
            "image_broad": ib,
            "seg_broad": sb,
        }

    if _has_invasion_conflict(image_only, seg_stats):
        return image_only, {
            "selected": "image_only",
            "reason": "seg_stats_changed_invasive_vs_insitu",
        }

    i_spec = _diag_specificity(image_only)
    s_spec = _diag_specificity(seg_stats)
    if i_spec >= 2 and s_spec < i_spec:
        return image_only, {
            "selected": "image_only",
            "reason": "seg_stats_less_specific_than_image_only",
            "image_specificity": i_spec,
            "seg_specificity": s_spec,
        }

    if s_spec > i_spec:
        return seg_stats, {
            "selected": "seg_stats",
            "reason": "seg_stats_more_specific_without_guard_violation",
            "image_specificity": i_spec,
            "seg_specificity": s_spec,
        }

    # ── SS12 tiebreaker: same specificity → prefer richer subtype ──
    i_rich = _subtype_richness(image_only.subtype)
    s_rich = _subtype_richness(seg_stats.subtype)
    if s_rich > i_rich:
        return seg_stats, {
            "selected": "seg_stats",
            "reason": "same_specificity_seg_subtype_richer",
            "image_specificity": i_spec,
            "seg_specificity": s_spec,
            "image_richness": i_rich,
            "seg_richness": s_rich,
        }
    else:
        return image_only, {
            "selected": "image_only",
            "reason": "same_specificity_image_subtype_richer",
            "image_specificity": i_spec,
            "seg_specificity": s_spec,
            "image_richness": i_rich,
            "seg_richness": s_rich,
        }


def _seg_necrosis_supported(seg_ctx: dict[str, Any] | None, *, threshold: float = 0.10) -> bool:
    """Return True when masks provide positive necrosis/dead-tissue support."""
    if not seg_ctx:
        return False
    for row in seg_ctx.get("roi_results") or []:
        stats = row.get("stats") or {}
        named = stats.get("named_class_fractions") or {}
        for name, info in named.items():
            lname = str(name).lower()
            if "necrotic" not in lname and "dead" not in lname:
                continue
            if not isinstance(info, dict):
                continue
            try:
                if float(info.get("fraction") or 0.0) >= threshold:
                    return True
            except Exception:
                continue
    return False


def _is_assessable_text(value: str) -> bool:
    text = _norm_diag_text(value)
    if not text:
        return False
    return not any(
        marker in text
        for marker in (
            "not assessable", "not assessed", "cannot assess", "cannot determine",
            "indeterminate", "unknown", "unspecified", "not specified", "none",
        )
    )


def _is_positive_necrosis_text(value: str) -> bool:
    text = _norm_diag_text(value)
    if not text:
        return False
    negative = (
        "absent", "no necrosis", "not identified", "not seen", "not evident",
        "without necrosis", "none", "negative", "not assessable", "unknown",
    )
    if any(marker in text for marker in negative):
        return False
    positive = (
        "present", "focal", "identified", "seen", "extensive", "geographic",
        "comedonecrosis", "necrotic", "dead tissue", "tumor necrosis",
    )
    return any(marker in text for marker in positive)


def _grade_tier(value: str) -> str:
    """Normalize noisy grade wording to low/intermediate/high."""
    text = _norm_diag_text(value)
    if not text:
        return ""
    if re.search(r"\b(?:not assessable|not applicable|not specified|unknown|cannot assess|insufficient)\b", text):
        return ""
    if text in {"high", "grade high"}:
        return "high"
    if text in {"intermediate", "moderate", "medium", "grade intermediate", "grade moderate"}:
        return "intermediate"
    if text in {"low", "grade low"}:
        return "low"
    if re.search(
        r"\b(?:high grade|poorly differentiated|undifferentiated|dedifferentiated|"
        r"anaplastic|grade 3|grade iii|grade 4|grade iv|g3|g4|fuhrman 3|"
        r"fuhrman iii|fuhrman 4|fuhrman iv|nuclear grade 3|nuclear grade 4)\b",
        text,
    ):
        return "high"
    if re.search(
        r"\b(?:intermediate grade|moderately differentiated|moderate differentiation|"
        r"grade 2|grade ii|g2|fuhrman 2|fuhrman ii|nuclear grade 2)\b",
        text,
    ):
        return "intermediate"
    if re.search(
        r"\b(?:low grade|well differentiated|well formed|grade 1|grade i|g1|"
        r"fuhrman 1|fuhrman i|nuclear grade 1)\b",
        text,
    ):
        return "low"
    return ""


def _grade_evidence_text(diag: DiagnosisPrediction) -> str:
    """Evidence-bearing text for grade adjudication.

    This deliberately excludes raw segmentation fractions.  Grade can be
    supported by diagnosis/morphology language, but not by mask proportions
    alone.
    """
    parts = [
        diag.primary_diagnosis,
        diag.subtype,
        diag.tumor_architecture,
        diag.cytology,
        diag.necrosis,
        diag.stroma_inflammation,
    ]
    return _norm_diag_text(" ".join(str(p or "") for p in parts))


def _grade_support_score(tier: str, diag: DiagnosisPrediction) -> tuple[float, list[str]]:
    """Score whether a diagnosis payload gives morphology support for grade."""
    if not tier:
        return 0.0, []
    text = _grade_evidence_text(diag)
    evidence: list[str] = []
    score = 0.0

    def add(pattern: str, label: str, weight: float) -> None:
        nonlocal score
        if re.search(pattern, text):
            score += weight
            evidence.append(label)

    if tier == "high":
        add(r"\b(?:high grade|poorly differentiated|undifferentiated|dedifferentiated|anaplastic)\b", "explicit high-grade/dedifferentiated wording", 2.0)
        add(r"\b(?:marked|severe|high)\s+(?:nuclear\s+)?(?:pleomorphism|atypia|anaplasia)\b", "marked cytologic atypia/pleomorphism", 1.5)
        add(r"\b(?:brisk|frequent|numerous|high)\s+(?:mitotic|mitoses|mitotic activity)\b", "high mitotic activity", 1.25)
        add(r"\b(?:necrosis|comedonecrosis|geographic necrosis|tumor necrosis)\b", "necrosis supports high-grade behavior", 0.75)
        add(r"\b(?:solid sheets|sheet like|diffuse sheets|small blue cells|high cellularity)\b", "high-cellularity aggressive architecture", 0.75)
    elif tier == "intermediate":
        add(r"\b(?:intermediate grade|moderately differentiated|moderate differentiation)\b", "explicit intermediate/moderate differentiation wording", 2.0)
        add(r"\bmoderate\s+(?:nuclear\s+)?(?:pleomorphism|atypia)\b", "moderate cytologic atypia", 1.25)
        add(r"\b(?:gland forming|glandular|tubular|papillary|nested)\b", "partly differentiated architecture", 0.5)
    elif tier == "low":
        add(r"\b(?:low grade|well differentiated|well formed|bland)\b", "explicit low-grade/well-differentiated wording", 2.0)
        add(r"\b(?:minimal|mild|low)\s+(?:nuclear\s+)?(?:pleomorphism|atypia)\b", "mild cytologic atypia", 1.25)
        add(r"\b(?:low mitotic|rare mitoses|infrequent mitoses)\b", "low mitotic activity", 1.0)
        add(r"\b(?:orderly|uniform|monomorphic|well circumscribed)\b", "orderly/bland morphology", 0.5)
    return score, evidence


def _adjudicate_grade(
    image_only: DiagnosisPrediction,
    seg_stats: DiagnosisPrediction,
    selected_path: str,
) -> tuple[str, dict[str, Any]]:
    """Choose grade by morphology/diagnostic consistency, not fixed source trust."""
    candidates: dict[str, dict[str, Any]] = {
        "image_only": {
            "value": image_only.grade,
            "tier": _grade_tier(image_only.grade),
            "support": 0.0,
            "evidence": [],
        },
        "seg_stats": {
            "value": seg_stats.grade,
            "tier": _grade_tier(seg_stats.grade),
            "support": 0.0,
            "evidence": [],
        },
    }
    for name, diag in (("image_only", image_only), ("seg_stats", seg_stats)):
        score, evidence = _grade_support_score(str(candidates[name]["tier"]), diag)
        candidates[name]["support"] = score
        candidates[name]["evidence"] = evidence

    assessable = {k: v for k, v in candidates.items() if v["tier"]}
    if not assessable:
        return "", {"selected": "not_assessable", "reason": "no_assessable_grade", "candidates": candidates}

    tiers = {str(v["tier"]) for v in assessable.values()}
    if len(tiers) == 1:
        tier = next(iter(tiers))
        preferred = selected_path if selected_path in assessable else next(iter(assessable))
        return str(assessable[preferred]["value"] or tier), {
            "selected": preferred,
            "reason": "both_paths_same_grade_tier" if len(assessable) > 1 else "single_assessable_grade",
            "tier": tier,
            "candidates": candidates,
        }

    ranked = sorted(
        assessable.items(),
        key=lambda kv: (float(kv[1]["support"]), 1 if kv[0] == selected_path else 0),
        reverse=True,
    )
    best_name, best = ranked[0]
    second_support = float(ranked[1][1]["support"]) if len(ranked) > 1 else 0.0
    best_support = float(best["support"])
    if best_support >= 1.5 and best_support >= second_support + 0.5:
        return str(best["value"] or best["tier"]), {
            "selected": best_name,
            "reason": "stronger_grade_morphology_support",
            "tier": best["tier"],
            "candidates": candidates,
        }
    if best_name == selected_path and best_support >= 1.5:
        return str(best["value"] or best["tier"]), {
            "selected": best_name,
            "reason": "selected_path_grade_has_sufficient_support",
            "tier": best["tier"],
            "candidates": candidates,
        }
    return "", {
        "selected": "not_assessable",
        "reason": "conflicting_grade_without_clear_support",
        "candidates": candidates,
    }


def _conservative_confidence(*values: str) -> str:
    order = {"low": 0, "medium": 1, "moderate": 1, "high": 2}
    labels = [str(v or "").strip().lower() for v in values if str(v or "").strip()]
    if not labels:
        return "low"
    best = min(labels, key=lambda v: order.get(v, 0))
    return "medium" if best == "moderate" else best


def _choose_dual_path_diagnosis_with_field_guard(
    image_only: DiagnosisPrediction,
    seg_stats: DiagnosisPrediction,
    seg_ctx: dict[str, Any] | None,
) -> tuple[DiagnosisPrediction, dict[str, Any]]:
    """SS8: SS7 diagnosis-path guard plus field-level protection.

    Seg stats can help choose the broad/subtype diagnosis, but fields such as
    grade and invasive/in-situ status are noisy when inferred from compact mask
    fractions.  Merge them conservatively rather than accepting a whole
    diagnosis object from one path.
    """
    chosen, guard = _choose_dual_path_diagnosis(image_only, seg_stats)
    selected_path = guard.get("selected", "seg_stats")
    merged = DiagnosisPrediction(**chosen.to_dict())
    field_merge: dict[str, dict[str, str]] = {}

    def source_diag(path: str) -> DiagnosisPrediction:
        return seg_stats if path == "seg_stats" else image_only

    # Grade: preserve image-only unless it is unavailable.
    if _is_assessable_text(image_only.grade):
        if merged.grade != image_only.grade:
            field_merge["grade"] = {
                "selected": "image_only",
                "reason": "grade_protected_from_seg_stats",
                "previous": str(merged.grade or ""),
                "value": str(image_only.grade or ""),
            }
        merged.grade = image_only.grade
    elif _is_assessable_text(seg_stats.grade):
        merged.grade = seg_stats.grade
        field_merge["grade"] = {
            "selected": "seg_stats",
            "reason": "image_only_grade_unavailable",
            "value": str(seg_stats.grade or ""),
        }

    # Invasion: never let seg stats alone flip the image-only assessment.
    if _is_assessable_text(image_only.invasion):
        if merged.invasion != image_only.invasion:
            field_merge["invasion"] = {
                "selected": "image_only",
                "reason": "invasion_protected_from_seg_stats",
                "previous": str(merged.invasion or ""),
                "value": str(image_only.invasion or ""),
            }
        merged.invasion = image_only.invasion
    elif _is_assessable_text(seg_stats.invasion) and not _has_invasion_conflict(image_only, seg_stats):
        merged.invasion = seg_stats.invasion
        field_merge["invasion"] = {
            "selected": "seg_stats",
            "reason": "image_only_invasion_unavailable_no_conflict",
            "value": str(seg_stats.invasion or ""),
        }

    # Malignancy: if paths conflict, preserve image-only; otherwise keep chosen.
    if (
        _is_assessable_text(image_only.malignancy)
        and _is_assessable_text(seg_stats.malignancy)
        and _norm_diag_text(image_only.malignancy) != _norm_diag_text(seg_stats.malignancy)
    ):
        merged.malignancy = image_only.malignancy
        field_merge["malignancy"] = {
            "selected": "image_only",
            "reason": "malignancy_conflict_protected",
            "seg_stats_value": str(seg_stats.malignancy or ""),
            "value": str(image_only.malignancy or ""),
        }

    # Necrosis: accept seg-stats necrosis only with positive mask support.
    necrosis_supported = _seg_necrosis_supported(seg_ctx)
    if _is_assessable_text(seg_stats.necrosis) and necrosis_supported:
        merged.necrosis = seg_stats.necrosis
        field_merge["necrosis"] = {
            "selected": "seg_stats",
            "reason": "positive_necrosis_fraction_detected",
            "value": str(seg_stats.necrosis or ""),
        }
    elif _is_assessable_text(image_only.necrosis):
        if merged.necrosis != image_only.necrosis:
            field_merge["necrosis"] = {
                "selected": "image_only",
                "reason": "seg_stats_necrosis_without_mask_support",
                "previous": str(merged.necrosis or ""),
                "value": str(image_only.necrosis or ""),
            }
        merged.necrosis = image_only.necrosis

    if field_merge:
        merged.confidence = _conservative_confidence(image_only.confidence, seg_stats.confidence)

    audit = {
        **guard,
        "selected_diagnostic_path": selected_path,
        "field_merge": field_merge,
        "seg_necrosis_supported": necrosis_supported,
    }
    return merged, audit


def _choose_dual_path_diagnosis_with_grade_necrosis_guard(
    image_only: DiagnosisPrediction,
    seg_stats: DiagnosisPrediction,
    seg_ctx: dict[str, Any] | None,
) -> tuple[DiagnosisPrediction, dict[str, Any]]:
    """SS9: SS7 path guard with minimal field merge.

    SS8 protected too many fields and sometimes converted a useful SS7 report
    into a weaker mixed state. SS9 keeps the selected diagnostic path intact
    except for two bounded fields:
      - grade: preserve image-only grade when available;
      - necrosis: accept seg-stats necrosis only when both the mask and the
        seg-stats text are positive for necrosis.
    """
    chosen, guard = _choose_dual_path_diagnosis(image_only, seg_stats)
    selected_path = guard.get("selected", "seg_stats")
    merged = DiagnosisPrediction(**chosen.to_dict())
    field_merge: dict[str, dict[str, str]] = {}

    if _is_assessable_text(image_only.grade):
        if merged.grade != image_only.grade:
            field_merge["grade"] = {
                "selected": "image_only",
                "reason": "grade_protected_from_seg_stats",
                "previous": str(merged.grade or ""),
                "value": str(image_only.grade or ""),
            }
        merged.grade = image_only.grade
    elif _is_assessable_text(seg_stats.grade):
        merged.grade = seg_stats.grade
        field_merge["grade"] = {
            "selected": "seg_stats",
            "reason": "image_only_grade_unavailable",
            "value": str(seg_stats.grade or ""),
        }

    necrosis_supported = _seg_necrosis_supported(seg_ctx)
    seg_necrosis_positive = _is_positive_necrosis_text(seg_stats.necrosis)
    if necrosis_supported and seg_necrosis_positive:
        merged.necrosis = seg_stats.necrosis
        field_merge["necrosis"] = {
            "selected": "seg_stats",
            "reason": "mask_and_text_positive_for_necrosis",
            "value": str(seg_stats.necrosis or ""),
        }
    elif _is_assessable_text(image_only.necrosis):
        if merged.necrosis != image_only.necrosis:
            reason = (
                "mask_positive_but_seg_text_not_positive"
                if necrosis_supported and not seg_necrosis_positive
                else "seg_stats_necrosis_without_mask_support"
            )
            field_merge["necrosis"] = {
                "selected": "image_only",
                "reason": reason,
                "previous": str(merged.necrosis or ""),
                "value": str(image_only.necrosis or ""),
            }
        merged.necrosis = image_only.necrosis

    audit = {
        **guard,
        "selected_diagnostic_path": selected_path,
        "field_merge": field_merge,
        "seg_necrosis_supported": necrosis_supported,
        "seg_necrosis_text_positive": seg_necrosis_positive,
    }
    return merged, audit


def _choose_dual_path_diagnosis_with_grade_adjudication(
    image_only: DiagnosisPrediction,
    seg_stats: DiagnosisPrediction,
    seg_ctx: dict[str, Any] | None,
) -> tuple[DiagnosisPrediction, dict[str, Any]]:
    """SS10: SS7 path guard plus evidence-consistency grade adjudication.

    Unlike SS9, this does not assume image-only grade is always better.  The
    primary diagnosis/subtype path is selected by SS7's lineage/specificity
    guard, then grade is selected independently by morphology support in the
    two diagnosis payloads.
    """
    chosen, guard = _choose_dual_path_diagnosis(image_only, seg_stats)
    selected_path = guard.get("selected", "seg_stats")
    merged = DiagnosisPrediction(**chosen.to_dict())
    field_merge: dict[str, dict[str, Any]] = {}

    grade_value, grade_audit = _adjudicate_grade(image_only, seg_stats, selected_path)
    if grade_value != merged.grade:
        field_merge["grade"] = {
            "selected": grade_audit.get("selected", ""),
            "reason": grade_audit.get("reason", ""),
            "previous": str(merged.grade or ""),
            "value": str(grade_value or ""),
        }
    merged.grade = grade_value

    necrosis_supported = _seg_necrosis_supported(seg_ctx)
    seg_necrosis_positive = _is_positive_necrosis_text(seg_stats.necrosis)
    if necrosis_supported and seg_necrosis_positive:
        if merged.necrosis != seg_stats.necrosis:
            field_merge["necrosis"] = {
                "selected": "seg_stats",
                "reason": "mask_and_text_positive_for_necrosis",
                "previous": str(merged.necrosis or ""),
                "value": str(seg_stats.necrosis or ""),
            }
        merged.necrosis = seg_stats.necrosis
    elif _is_assessable_text(image_only.necrosis):
        if merged.necrosis != image_only.necrosis:
            reason = (
                "mask_positive_but_seg_text_not_positive"
                if necrosis_supported and not seg_necrosis_positive
                else "seg_stats_necrosis_without_mask_support"
            )
            field_merge["necrosis"] = {
                "selected": "image_only",
                "reason": reason,
                "previous": str(merged.necrosis or ""),
                "value": str(image_only.necrosis or ""),
            }
        merged.necrosis = image_only.necrosis

    audit = {
        **guard,
        "selected_diagnostic_path": selected_path,
        "field_merge": field_merge,
        "grade_adjudication": grade_audit,
        "seg_necrosis_supported": necrosis_supported,
        "seg_necrosis_text_positive": seg_necrosis_positive,
    }
    return merged, audit


def _sanitize_broad_type(raw: str) -> str:
    """Map a free-form broad type string to a valid canonical label."""
    cleaned = str(raw or "").strip().lower()
    if cleaned in _VALID_BROAD_TYPES:
        return cleaned
    if "carcinoma" in cleaned or "adenocarcinoma" in cleaned:
        return "carcinoma"
    if "sarcoma" in cleaned:
        return "sarcoma"
    if "lymphoma" in cleaned:
        return "lymphoma"
    if "melanoma" in cleaned:
        return "melanoma"
    if "glioma" in cleaned or "glioblastoma" in cleaned or "astrocytoma" in cleaned:
        return "glioma"
    if "neuroendocrine" in cleaned or "carcinoid" in cleaned or "pheochromocytoma" in cleaned:
        return "neuroendocrine neoplasm"
    if "germ cell" in cleaned or "seminoma" in cleaned or "teratoma" in cleaned:
        return "germ cell tumor"
    if "mesothelioma" in cleaned:
        return "mesothelioma"
    return "neoplasm" if cleaned else "neoplasm"


def _infer_broad_from_subtype(subtype: str) -> str:
    """Reverse-infer broad_type from a specific subtype (bottom-up path).

    When the LLM confidently identifies a subtype, the broad_type is
    deterministic from it.  E.g. "adrenocortical carcinoma" → "carcinoma",
    "pheochromocytoma" → "neuroendocrine neoplasm".
    """
    s = str(subtype or "").strip().lower()
    if not s:
        return ""
    if any(t in s for t in ("neuroendocrine", "carcinoid", "pheochromocytoma",
                             "paraganglioma", "medullary carcinoma", "small cell carcinoma", "merkel")):
        return "neuroendocrine neoplasm"
    if any(t in s for t in ("sarcoma", "gist", "leiomyosarcoma", "liposarcoma",
                             "angiosarcoma", "chondrosarcoma", "osteosarcoma", "rhabdomyosarcoma", "ewing")):
        return "sarcoma"
    if any(t in s for t in ("lymphoma", "lymphoid", "hodgkin", "myeloma", "plasmacytoma", "leukemia")):
        return "lymphoma"
    if any(t in s for t in ("melanoma", "melanocytic")):
        return "melanoma"
    if any(t in s for t in ("glioma", "glioblastoma", "astrocytoma", "oligodendroglioma", "ependymoma")):
        return "glioma"
    if any(t in s for t in ("germ cell", "seminoma", "teratoma", "yolk sac", "embryonal carcinoma", "choriocarcinoma")):
        return "germ cell tumor"
    if "mesothelioma" in s:
        return "mesothelioma"
    if "carcinosarcoma" in s or "mmmt" in s:
        return "carcinosarcoma"
    if "thymic" in s or "thymoma" in s:
        return "thymic epithelial neoplasm"
    if any(t in s for t in ("carcinoma", "adenoma", "papilloma",
                             "adenocarcinoma", "squamous", "neoplasm of", "tumor of")):
        return "carcinoma"
    return ""


def _reconcile_tiers(diag, broad_hypotheses):
    """Post-process diagnosis: subtype-first, broad-from-subtype when possible.

    1. If subtype is specific → infer broad_type from it (bottom-up wins).
    2. If subtype is empty → use broad_hypotheses, sanitized.
    3. Always ensure broad_type is a canonical label.
    """
    subtype = (diag.subtype or "").strip()
    inferred = _infer_broad_from_subtype(subtype)

    if inferred:
        diag.broad_type = inferred
        return diag

    # No subtype — sanitize existing broad_type
    sanitized = _sanitize_broad_type(diag.broad_type)
    if sanitized:
        diag.broad_type = sanitized
    return diag


def _disabled_components(ablation: str) -> set[str]:
    """Return disabled logical components for an ablation label.

    The labels are grouped by paper question:
    R*: reasoning / claim pipeline, E*: evidence inputs, Q*: acquisition policy.
    R0/E0/Q0 are aliases for the same full model and should not be run twice in
    the same study.
    """
    full_aliases = {"full", "A0_full", "R0_full", "E0_full_inputs", "Q0_active", "V0_full"}
    if ablation in full_aliases:
        return set()
    mapping: dict[str, list[str]] = {
        # Legacy labels kept runnable.
        "A1_no_seg": ["segmentation"],
        "A2_no_medgemma": ["morphology"],
        "A3_no_knowledge": ["knowledge"],
        "A4_no_reinspect": ["reinspection"],
        "A5_no_claims": ["evidence_graph", "claim_filter_rewrite"],
        "A6_no_hybrid": ["morphology"],
        # Reasoning / claim ablations.
        "R1_no_claim_filter": ["claim_filter_rewrite"],
        "R2_no_llm_claim_verifier": ["llm_claim_verifier"],
        "R3_no_claim_verification": ["claim_verification", "claim_filter_rewrite"],
        "R4_no_evidence_graph": ["evidence_graph", "claim_filter_rewrite"],
        "R5_no_subtype_sufficiency": ["subtype_sufficiency"],
        "R6_no_broad_hypotheses": ["broad_hypotheses"],
        "R7_no_reinspection": ["reinspection"],
        # Evidence input ablations.  Visual ROI images remain available.
        "E1_visual_only": ["segmentation", "morphology", "knowledge"],
        "E2_visual_seg": ["morphology", "knowledge"],
        "E3_visual_morph": ["segmentation", "knowledge"],
        "E4_visual_keep": ["segmentation", "morphology"],
        "E5_visual_seg_morph": ["knowledge"],
        "E6_visual_seg_keep": ["morphology"],
        "E7_visual_morph_keep": ["segmentation"],
        "E8_no_known_site": [],
        # ── Segmentation-supported report ablation suite ──
        # Fixed-ROI, no active reinspection.  This suite is scoped for the
        # segmentation/report paper: visual ROIs are always available, while
        # segmentation, morphology text, and KEEP are added/removed as evidence
        # inputs.  Reinspection and seg-policy are disabled to avoid mixing in
        # the later policy contribution.
        "S0_image_only": ["reinspection", "segmentation", "morphology", "knowledge", "seg_policy"],
        "S1_image_seg": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "S2_image_morph": ["reinspection", "segmentation", "knowledge", "seg_policy"],
        "S3_image_seg_morph": ["reinspection", "knowledge", "seg_policy"],
        "S4_image_seg_keep": ["reinspection", "morphology", "seg_policy"],
        "S5_image_seg_morph_keep": ["reinspection", "seg_policy"],
        "S6_seg_morph_no_claim_filter": ["reinspection", "knowledge", "seg_policy", "claim_filter_rewrite"],
        "S7_image_seg_ft_reader": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "S8_image_seg_ft_reader_keep": ["reinspection", "morphology", "seg_policy"],
        # ── Seg-reader isolation suite ─────────────────────────────────────
        # Final reasoning/report always receives raw H&E ROI images only.
        # Segmentation is either exposed as compact text/stat context, converted
        # through the fine-tuned seg-reader, or both.  Direct overlay/composite
        # images are not sent to the final report LLM in these variants.
        "SR0_image_only": ["reinspection", "segmentation", "morphology", "knowledge", "seg_policy"],
        "B0_visual_reinspect_baseline": ["segmentation", "morphology", "knowledge", "seg_policy"],
        "SR1_seg_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SR2_seg_reader": ["reinspection", "morphology", "knowledge", "seg_policy", "seg_stats_context"],
        "SR3_seg_reader_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SR4_seg_reader_v2": ["reinspection", "morphology", "knowledge", "seg_policy", "seg_stats_context"],
        "SR5_seg_reader_v2_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SR6_seg_reader_v3": ["reinspection", "morphology", "knowledge", "seg_policy", "seg_stats_context"],
        "SR7_seg_reader_v3_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SR8_seg_reader_v3_gated_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SR9_seg_reader_v3_stats_case_session": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SR10_seg_reader_v3_gated_session": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SR11_seg_reader_v3_stats_nl_session": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SR12_seg_reader_v3_stats_case_session_override": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SR13_seg_reader_v3_stats_nl_session_override": ["reinspection", "morphology", "knowledge", "seg_policy"],
        # ── Seg-stats wording/structure suite ─────────────────────────────
        # No reader, no morphology/KEEP/reinspection. These variants keep the
        # same segmentation masks as SR1 and vary how stats are verbalized for
        # final report generation.
        "SS2_seg_claim_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SS3_seg_hier_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SS4_seg_quality_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SS5_positive_seg_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SS6_report_only_seg_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SS7_dual_path_guarded_seg_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SS8_dual_path_field_guarded_seg_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SS9_dual_path_grade_necrosis_guarded_seg_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SS10_dual_path_grade_adjudicated_seg_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        "SS12_dual_path_richness_tiebreaker_seg_stats": ["reinspection", "morphology", "knowledge", "seg_policy"],
        # Acquisition ablations.
        "Q1_no_reinspect": ["reinspection"],
        "Q2_independent_only": ["zoom"],
        "Q3_no_seg_policy": ["seg_policy"],
        # ── v2 subtractive ablation suite ──
        "V0_full": [],                                    # Full v2 design
        "V1_no_seg_policy": ["seg_policy"],               # − seg-driven acquisition
        "V2_no_reinspection": ["reinspection"],            # − adaptive reinspection
        "V3_no_morph": ["morphology"],                    # − MedGemma morphology
        "V4_no_keep": ["knowledge"],                      # − KEEP knowledge base
        "V5_no_seg": ["segmentation"],                    # − MedSAM segmentation
        "V6_no_subtype_guard": ["subtype_sufficiency"],   # − subtype sufficiency guard
        "V7_vlm_only": ["segmentation", "morphology", "knowledge",
                        "subtype_sufficiency", "broad_hypotheses",
                        "seg_policy", "reinspection"],    # Pure VLM baseline
        # ── v3 two-stage diagnosis ablation suite ──
        "W0_full": ["reinspection"],                   # Base: no reinsp (3 ROI)
        "W2_no_morph": ["reinspection", "morphology"],        # − MedGemma
        "W3_no_keep": ["reinspection", "knowledge"],          # − KEEP
        "W4_no_seg": ["reinspection", "segmentation"],        # − MedSAM
        "W5_no_subtype_guard": ["reinspection", "subtype_sufficiency"],  # − guard
        "W6_vlm_only": ["reinspection", "segmentation", "morphology",
                        "knowledge", "subtype_sufficiency",
                        "broad_hypotheses", "seg_policy"],   # Pure VLM
        "W7_with_reinsp": [],                                 # +Reins (all ON)
        "W8_best_with_reinsp": ["morphology"],               # W2 + Reins
        "W9_best_no_seg": ["reinspection", "morphology", "segmentation", "seg_policy"],  # W2 − seg
    }
    return set(mapping.get(ablation, []))


def _enabled_components(ablation: str) -> set[str]:
    mapping: dict[str, list[str]] = {
        "S7_image_seg_ft_reader": ["seg_reader"],
        "S8_image_seg_ft_reader_keep": ["seg_reader"],
        "SR2_seg_reader": ["seg_reader"],
        "SR3_seg_reader_stats": ["seg_reader"],
        "SR4_seg_reader_v2": ["seg_reader"],
        "SR5_seg_reader_v2_stats": ["seg_reader"],
        "SR6_seg_reader_v3": ["seg_reader"],
        "SR7_seg_reader_v3_stats": ["seg_reader"],
        "SR8_seg_reader_v3_gated_stats": ["seg_reader", "seg_reader_guard"],
        "SR11_seg_reader_v3_stats_nl_session": ["seg_reader", "case_chat_session"],
        "SR10_seg_reader_v3_gated_session": ["seg_reader", "seg_reader_guard", "case_chat_session"],
        "SR9_seg_reader_v3_stats_case_session": ["seg_reader", "case_chat_session"],
        "SR12_seg_reader_v3_stats_case_session_override": ["seg_reader", "case_chat_session"],
        "SR13_seg_reader_v3_stats_nl_session_override": ["seg_reader", "case_chat_session"],
    }
    return set(mapping.get(ablation, []))

@dataclass
class EnhancedPipelineResult:
    case_id: str
    mode: str = "pathxai_v2_enhanced"
    roi_selection: Optional[RoiSelectionOutput] = None
    site_prediction: Optional[SitePrediction] = None
    segmentation: Optional[SegmentationOutput] = None
    morphology: Optional[MorphologyOutput] = None
    knowledge: Optional[KnowledgeOutput] = None
    broad_hypotheses: Optional[BroadHypothesesOutput] = None
    subtype_sufficiency: Optional[SubtypeSufficiencyOutput] = None
    diagnosis_prediction: Optional[DiagnosisPrediction] = None
    evidence_sufficiency: Optional[EvidenceSufficiency] = None
    reinspection_history: list = field(default_factory=list)
    seg_policy_decisions: list[dict] = field(default_factory=list)  # trace of seg-driven actions
    final_selected_rois: list[RoiInfo] = field(default_factory=list)
    evidence_graph: Optional[EvidenceGraphOutput] = None
    report: Optional[ReportOutput] = None
    candidate_count: int = 0
    selected_count: int = 0
    reinspection_rounds: int = 0
    errors: list = field(default_factory=list)
    elapsed_sec: float = 0.0
    session: Optional[SessionMemory] = None
    ablation: str = "full"

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "mode": self.mode, "ablation": self.ablation,
            "candidate_count": self.candidate_count, "selected_count": self.selected_count,
            "reinspection_rounds": self.reinspection_rounds,
            "roi_selection": self.roi_selection.to_dict() if self.roi_selection else None,
            "site_prediction": self.site_prediction.to_dict() if self.site_prediction else None,
            "segmentation_context": self.segmentation.to_dict() if self.segmentation else None,
            "medgemma_morphology_context": self.morphology.to_dict() if self.morphology else None,
            "knowledge_context": self.knowledge.to_dict() if self.knowledge else None,
            "broad_hypotheses": self.broad_hypotheses.to_dict() if self.broad_hypotheses else None,
            "subtype_sufficiency": self.subtype_sufficiency.to_dict() if self.subtype_sufficiency else None,
            "diagnosis_prediction": self.diagnosis_prediction.to_dict() if self.diagnosis_prediction else None,
            "evidence_sufficiency": self.evidence_sufficiency.to_dict() if self.evidence_sufficiency else None,
            "reinspection_plans": [r.to_dict() for r in self.reinspection_history],
            "seg_policy_decisions": self.seg_policy_decisions,
            "final_selected_rois": [r.to_dict() for r in self.final_selected_rois],
            "evidence_graph": self.evidence_graph.to_dict() if self.evidence_graph else None,
            "generated_long_report": self.report.long_report if self.report else "",
            "report_sections": self.report.sections if self.report else {},
            "errors": self.errors, "elapsed_sec": self.elapsed_sec,
            "session_memory": self.session.to_dict() if self.session else None,
        }


class EnhancedOrchestrator:
    """Full pipeline with reinspection, seg, MedGemma, KEEP, evidence graph, session memory."""

    def __init__(self, settings: Settings, stages_llm: LLMClient, out_dir: str | Path,
                 *, medgemma_client: LLMClient | None = None,
                 seg_reader_client: LLMClient | None = None,
                 report_llm_client: LLMClient | None = None,
                 ablation: str = "full"):
        self.settings = settings
        self.llm = stages_llm
        self.medgemma_client = medgemma_client or stages_llm
        self.seg_reader_client = seg_reader_client
        self.report_llm_client = report_llm_client or stages_llm
        self.out_dir = Path(out_dir)
        self.ablation = ablation
        self._step = 0          # current step counter for logging
        self._total_steps = 10  # total logical steps in the pipeline

    def _log(self, step: int | None, msg: str, *, arrow: bool = False) -> None:
        """Print a formatted pipeline step message to stdout."""
        prefix = f"    [{step}/{self._total_steps}]" if step else "   "
        leader = " →" if arrow else "  "
        print(f"{prefix}{leader} {msg}", flush=True)

    def run(self, case_id: str, candidate_rois: list[RoiInfo],
            diagnosis_priors: DiagnosisPriors | None = None,
            *, dry_run: bool = False, resume: bool = True) -> EnhancedPipelineResult:
        t0 = time.monotonic()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._prev_diag_sigs: list[tuple] = []  # reset per-case diagnosis stability tracker

        # ── Case-level resume: skip if already completed ──
        final_path = self.out_dir / CKPT_FILES["final"]
        if resume and final_path.exists():
            try:
                data = json.loads(final_path.read_text(encoding="utf-8"))
                report_text = (self.out_dir / "generated_long_report.txt").read_text(encoding="utf-8", errors="ignore")
                # Quick sanity — report must be non-trivial
                if len(report_text.strip().split()) >= 30:
                    return EnhancedPipelineResult(
                        case_id=case_id, mode="pathxai_v2_enhanced",
                        ablation=self.ablation,
                        candidate_count=data.get("candidate_count", 0),
                        selected_count=data.get("selected_count", 0),
                        elapsed_sec=0.0,
                        errors=["resumed_from_checkpoint"],
                    )
            except Exception:
                pass  # Corrupted checkpoint — re-run
        assert_no_cancer_type_leak({"priors": str(diagnosis_priors) if diagnosis_priors else None},
                                   where=f"orchestrator input for {case_id}")
        candidates = candidate_rois
        if not candidates:
            raise ValueError(f"No candidate ROIs for case {case_id}")

        original_llm = self.llm
        original_report_llm = self.report_llm_client
        case_session_client: CaseSessionLLMClient | None = None
        if self._on("case_chat_session"):
            case_session_client = CaseSessionLLMClient(
                self.llm,
                out_dir=self.out_dir,
            )
            self.llm = case_session_client
            # SR9 is explicitly testing one case-level Qwen dialogue. Keep the
            # final report in the same client-side history as the reasoning
            # stages, even if settings.llm.report_api points elsewhere.
            self.report_llm_client = case_session_client
            self._log(None, "Qwen case chat session enabled for reasoning + report", arrow=True)

        session = self._load_session(resume)
        session.case_id = case_id
        if diagnosis_priors and diagnosis_priors.has_known_site:
            session.known_site = diagnosis_priors.known_site or ""

        result = EnhancedPipelineResult(case_id=case_id, candidate_count=len(candidates),
                                         session=session, ablation=self.ablation)

        # ── Step 1: ROI selection ─────────────────────────────────────────────
        self._log(1, f"ROI evidence seeding ({len(candidates)} candidates) ...")
        roi_sel = self._ckpt("roi", resume, lambda: self._s1(candidates, dry_run))
        result.roi_selection = roi_sel.data
        result.selected_count = len(roi_sel.data.selected_rois)
        max_visual_evidence = min(
            ABSOLUTE_MAX_VISUAL_EVIDENCE,
            max(1, int(self.settings.reinspection.max_final_rois)),
        )
        selected = list(roi_sel.data.selected_rois)[:max_visual_evidence]
        _roi_ids = ", ".join(r.roi_id for r in selected[:5])
        self._log(1, f"{len(selected)} seed ROIs selected ({_roi_ids})", arrow=True)
        session.record("roi_selection", {"selected": len(selected)}, len(selected))

        # ── Step 2: Site prediction ───────────────────────────────────────────
        _has_prior_site = bool(diagnosis_priors and diagnosis_priors.has_known_site)
        if _has_prior_site:
            self._log(2, f"Site: known {diagnosis_priors.known_site} (no LLM)")
        else:
            self._log(2, "Site prediction (LLM) ...")
        site_pred = self._ckpt("site", resume, lambda: self._s2(selected, diagnosis_priors, dry_run))
        result.site_prediction = site_pred.data
        self._log(2, f"{site_pred.data.predicted_site}  [source={site_pred.data.source}, conf={site_pred.data.confidence}]", arrow=True)
        session.record("site", {"site": site_pred.data.predicted_site}, len(selected))

        # ── Step 3: Segmentation ──────────────────────────────────────────────
        if self._on("segmentation"):
            self._log(3, f"Segmentation (MedSAM, {len(selected)} ROIs) ...")
            seg_out = self._ckpt("seg", resume, lambda: self._s3(selected, dry_run))
            result.segmentation = seg_out.data
            _n_masks = len(seg_out.data.roi_results) if seg_out.data.available else 0
            self._log(3, f"available={seg_out.data.available}, {_n_masks} masks", arrow=True)
        else:
            self._log(3, "Segmentation — ABLATED (skipped)")
            result.segmentation = SegmentationOutput(available=False, error="ablated")

        # ── Step 4: Morphology / seg-reader evidence ─────────────────────────
        if self._on("morphology") or self._on("seg_reader"):
            source = "fine-tuned seg-reader" if self._on("seg_reader") and not self._on("morphology") else "MedGemma morphology"
            self._log(4, f"{source} (initial {len(selected)} ROIs) ...")
            morph_out = self._ckpt("morph", resume, lambda: self._s4(selected, site_pred.data, result.segmentation, dry_run))
            result.morphology = morph_out.data
            _morph_chars = sum(len(str(r.get("description", ""))) for r in (morph_out.data.rounds or []))
            self._log(4, f"available={morph_out.data.available}, chars≈{_morph_chars}", arrow=True)
        else:
            self._log(4, "Morphology / seg-reader evidence — ABLATED (skipped)")
            result.morphology = MorphologyOutput(available=False)

        # ── Step 5: Knowledge (broad/site) ────────────────────────────────────
        if self._on("knowledge"):
            self._log(5, f"KEEP knowledge retrieval (broad, site={site_pred.data.predicted_site}) ...")
            know_out = self._ckpt("knowledge", resume, lambda: self._s5(site_pred.data, None, dry_run))
            result.knowledge = know_out.data
            _sc = "; ".join(f"{h.get('name', h.get('entity','?'))}({h.get('count', h.get('score',0))})" for h in (know_out.data.site_subtype_candidates or [])[:3])
            _dh = "; ".join(f"{h.get('name', h.get('entity','?'))}({h.get('score',0)})" for h in (know_out.data.disease_hits or [])[:3])
            self._log(5, f"site_subtypes=[{_sc}]; disease=[{_dh}]", arrow=True)
        else:
            self._log(5, "KEEP knowledge — ABLATED (skipped)")
            result.knowledge = KnowledgeOutput(available=False, error="ablated")

        # ── Step 6: Preliminary broad/subtype reasoning + diagnosis ───────────
        self._log(6, f"Preliminary broad/subtype reasoning ({len(selected)} ROIs) ...")
        if self._on("broad_hypotheses"):
            broad_out = self._ckpt("broad", resume, lambda: self._s6a(
                selected, site_pred.data, result.segmentation, result.morphology, result.knowledge, dry_run))
            result.broad_hypotheses = broad_out.data
            self._log(6, f"broad={broad_out.data.best_broad_type or '?'} ({broad_out.data.confidence})", arrow=True)
        else:
            broad_out = StageResult(BroadHypothesesOutput(available=False, reason="ablated"), "broad_hypotheses")
            result.broad_hypotheses = broad_out.data
            self._log(6, "broad hypotheses — ABLATED", arrow=True)
        if self._on("subtype_sufficiency"):
            subtype_out = self._ckpt("subtype", resume, lambda: self._s6b(
                selected, site_pred.data, result.segmentation, result.morphology, result.knowledge,
                broad_out.data, None, dry_run))
            result.subtype_sufficiency = subtype_out.data
            self._log(6, f"subtype_level={subtype_out.data.best_supported_level}, recommended={subtype_out.data.recommended_primary_diagnosis or '?'}", arrow=True)
        else:
            subtype_out = StageResult(SubtypeSufficiencyOutput(available=False, reason="ablated"), "subtype_sufficiency")
            result.subtype_sufficiency = subtype_out.data
            self._log(6, "subtype sufficiency — ABLATED", arrow=True)

        self._log(6, f"Preliminary diagnosis ({len(selected)} ROIs) ...")
        diag_out = self._ckpt("diagnosis", resume, lambda: self._s6(
            selected, site_pred.data, result.segmentation, result.morphology, result.knowledge,
            broad_out.data, subtype_out.data, dry_run))
        diag = diag_out.data
        
        # ── Broad type locking: VLM's broad from ROI images is authoritative ──
        # Morph/keep pipelines can refine subtype but MUST NOT change broad_type.
        vlm_broad = diag.broad_type
        vlm_primary = diag.primary_diagnosis
        
        if self._on("subtype_sufficiency"):
            diag = apply_subtype_sufficiency_guard(diag, subtype_out.data)
        if self._on("lineage_guard"):
            diag = apply_site_lineage_guard(diag, site_pred.data)
        diag = _reconcile_tiers(diag, broad_out.data)
        
        # Restore VLM's broad_type — it cannot be overridden by downstream guards
        if vlm_broad and vlm_broad not in ("", "unspecified", "neoplasm"):
            diag.broad_type = vlm_broad
        # If subtype guard erased primary_diagnosis, keep VLM's
        if not diag.primary_diagnosis and vlm_primary:
            diag.primary_diagnosis = vlm_primary
            
        result.diagnosis_prediction = diag
        session.record("diagnosis", {"primary": diag.primary_diagnosis}, len(selected))
        self._log(6, f"broad={diag.broad_type} (locked), subtype={diag.subtype or '(tbd)'}, conf={diag.confidence}", arrow=True)

        # Step 6b: hypothesis-anchored knowledge (now that broad_type is known)
        knowledge_diag_signature = _diagnosis_signature(diag)
        if self._on("knowledge") and not dry_run:
            self._log(6, f"KEEP knowledge hypothesis phase (subtype={diag.subtype}) ...")
            know2 = self._ckpt("knowledge_hypothesis", resume,
                               lambda: self._s5(site_pred.data, diag, dry_run))
            if know2.data and (know2.data.hypothesis_hits or know2.data.available):
                result.knowledge = know2.data
                _hh = "; ".join(f"{h.get('diagnosis', h.get('name','?'))}" for h in (know2.data.hypothesis_hits or [])[:3])
                self._log(6, f"KEEP hypothesis hits: [{_hh}]", arrow=True)
                knowledge_diag_signature = _diagnosis_signature(diag)

        # Step 6c: preliminary evidence sufficiency
        self._log(6, "Evidence sufficiency check ...")
        ev_out = self._ckpt("evidence", resume, lambda: self._s7(
            selected, site_pred.data, diag, result.segmentation, result.morphology, result.knowledge, dry_run))
        result.evidence_sufficiency = ev_out.data
        diag = apply_evidence_sufficiency_guard(diag, site_pred.data, ev_out.data)
        result.diagnosis_prediction = diag
        _img_gaps = len(ev_out.data.missing_image_evidence)
        _nonimg_gaps = len(ev_out.data.missing_nonimage_metadata)
        self._log(6, f"sufficient={ev_out.data.sufficient_for_diagnosis}, image_gaps={_img_gaps}, nonimage_gaps={_nonimg_gaps}, can_report={ev_out.data.can_confidently_report}", arrow=True)

        # ── Step 7: Reinspection loop ─────────────────────────────────────────
        all_selected = list(selected)
        used_ids = {r.roi_id for r in all_selected}
        ri = 0
        if self._on("reinspection"):
            mr = self.settings.reinspection.max_reinspection_rounds
            me = self.settings.reinspection.max_reinspection_rois
            mz = self.settings.reinspection.max_zoom_depth if self._on("zoom") else 0
            mf = max_visual_evidence
            _use_seg_policy = self._on("seg_policy")
            self._log(7, f"Reinspection loop (max {mr} rounds, max +{me} ROIs, seg_policy={_use_seg_policy}) ...")
            while ri < mr and me > 0 and not ev_out.data.can_confidently_report:
                remaining_budget = max(0, mf - len(all_selected))
                if remaining_budget <= 0:
                    self._log(7, f"max visual evidence reached ({len(all_selected)}/{mf}) — stopping", arrow=True)
                    break
                remaining = [c for c in candidates if c.roi_id not in used_ids]
                if not remaining:
                    self._log(7, "no remaining candidates — stopping", arrow=True)
                    break
                max_extra = min(me, remaining_budget)

                # ── Seg-driven policy (before keyword planner) ────────────
                seg_action = None
                if _use_seg_policy and result.segmentation and result.segmentation.available:
                    seg_action = seg_driven_action(
                        result.segmentation.to_dict(), diag, ev_out.data,
                    )
                    if seg_action and seg_action.action:
                        _summary = seg_policy_summary(
                            result.segmentation.to_dict(), diag, ev_out.data,
                        )
                        result.seg_policy_decisions.append({
                            "round": ri,
                            **seg_action.to_dict(),
                            "seg_summary": _summary,
                        })
                        self._log(7,
                            f"seg-policy → {seg_action.action} "
                            f"(tumor={_summary.get('avg_tumor_pct')}pct "
                            f"dead={_summary.get('max_dead_pct')}pct "
                            f"interface={_summary.get('interface_rois')}/{_summary.get('roi_count')})",
                            arrow=True,
                        )
                        if seg_action.action == "stop":
                            self._log(7, "seg-policy suggests stop — breaking", arrow=True)
                            result.reinspection_history.append(ReinspectionPlan(
                                action="none",
                                stop_reason=f"seg_policy: {seg_action.reason}",
                            ))
                            break
                        # For non-stop actions, let keyword planner handle ROI selection
                        # but inject the seg recommendation as context
                        self._log(7, f"  → delegating ROI selection to keyword planner (seg hint: {seg_action.action})")

                ps = ReinspectionStage(self.settings, self.llm, self.out_dir / f"reinspect_{ri+1}")
                plan_input = {
                    "candidates": remaining, "selected": all_selected,
                    "sufficiency": ev_out.data, "diagnosis": diag,
                    "segmentation_context": result.segmentation.to_dict() if result.segmentation else {},
                    "max_extra_rois": max_extra, "max_zoom_depth": mz,
                }
                if seg_action and seg_action.action and seg_action.action != "stop":
                    plan_input["seg_policy_hint"] = seg_action.to_dict()
                plan = ps.run(plan_input, dry_run=dry_run)
                pd = plan.data
                result.reinspection_history.append(pd)
                if pd.action == "none" or not pd.new_rois:
                    self._log(7, f"round {ri+1}: action={pd.action}, no new ROIs — stopping", arrow=True)
                    break
                fresh_rois = []
                duplicate_ids = []
                for roi in pd.new_rois:
                    if roi.roi_id in used_ids:
                        duplicate_ids.append(roi.roi_id)
                        continue
                    fresh_rois.append(roi)
                if duplicate_ids:
                    self._log(7, f"round {ri+1}: ignored {len(duplicate_ids)} duplicate reinspection ROI(s)", arrow=True)
                new_rois = fresh_rois[:max_extra]
                if not new_rois:
                    self._log(7, f"round {ri+1}: action={pd.action}, no fresh ROIs after de-dup — stopping", arrow=True)
                    break
                all_selected.extend(new_rois)
                used_ids.update(r.roi_id for r in new_rois)
                ri += 1
                self._log(7, f"round {ri}: action={pd.action}, adding {len(new_rois)} ROIs (total={len(all_selected)}), remaining_candidates={len(remaining)-len(new_rois)}", arrow=True)
                if self._on("segmentation") and new_rois:
                    inherited_seg = [
                        (r.metadata or {}).get("inherited_segmentation")
                        for r in new_rois
                        if (r.metadata or {}).get("inherited_segmentation")
                    ]
                    needs_seg = [
                        r for r in new_rois
                        if not (r.metadata or {}).get("inherited_segmentation")
                    ]
                    new_seg_results: list[dict] = list(inherited_seg)
                    if inherited_seg:
                        self._log(7, f"  inherited seg for {len(inherited_seg)} zoom ROI(s) from parent masks")
                    if needs_seg:
                        self._log(7, f"  MedSAM seg for {len(needs_seg)} independent new ROI(s) ...")
                        ns = self._s3(needs_seg, dry_run)
                        if ns.data.available:
                            new_seg_results.extend(list(ns.data.roi_results))
                    if new_seg_results and result.segmentation:
                        m = dict(result.segmentation.to_dict())
                        merged = list(result.segmentation.roi_results) + new_seg_results
                        result.segmentation = SegmentationOutput(
                            available=True,
                            roi_results=merged,
                            summary=m.get("summary", {}),
                            class_names=m.get("class_names", {}),
                            legend_path=m.get("legend_path", ""),
                        )
                if (self._on("morphology") or self._on("seg_reader")) and new_rois:
                    src = "seg-reader" if self._on("seg_reader") and not self._on("morphology") else "MedGemma"
                    self._log(7, f"  {src} for {len(new_rois)} new ROIs (round {ri+1}) ...")
                    nm = self._s4(new_rois, site_pred.data, result.segmentation, dry_run,
                                  previous_rounds=result.morphology.rounds if result.morphology else None,
                                  round_index=ri+1, input_role="reinspection")
                    if nm.data.available:
                        result.morphology = nm.data
                        _new_chars = sum(len(str(r.get("description", ""))) for r in (nm.data.rounds or [])[-1:])
                        self._log(7, f"  → {src} chars +{_new_chars}", arrow=True)
                if self._on("broad_hypotheses"):
                    broad_out2 = self._s6a(all_selected, site_pred.data, result.segmentation, result.morphology, result.knowledge, dry_run)
                    result.broad_hypotheses = broad_out2.data
                else:
                    broad_out2 = StageResult(BroadHypothesesOutput(available=False, reason="ablated"), "broad_hypotheses")
                if self._on("subtype_sufficiency"):
                    subtype_out2 = self._s6b(
                        all_selected, site_pred.data, result.segmentation, result.morphology, result.knowledge,
                        broad_out2.data, diag, dry_run)
                    result.subtype_sufficiency = subtype_out2.data
                else:
                    subtype_out2 = StageResult(SubtypeSufficiencyOutput(available=False, reason="ablated"), "subtype_sufficiency")
                diag_out2 = self._s6(
                    all_selected, site_pred.data, result.segmentation, result.morphology, result.knowledge,
                    broad_out2.data, subtype_out2.data, dry_run)
                diag = diag_out2.data
                if self._on("subtype_sufficiency"):
                    diag = apply_subtype_sufficiency_guard(diag, subtype_out2.data)
                if self._on("lineage_guard"):
                    diag = apply_site_lineage_guard(diag, site_pred.data)
                diag = _reconcile_tiers(diag, broad_out2.data)
                result.diagnosis_prediction = diag

                # ── Diagnosis stability check: stop if diagnosis unchanged ─
                _cur_sig = _diagnosis_signature(diag)
                self._prev_diag_sigs.append(_cur_sig)
                STABILITY_WINDOW = 2  # stop after N rounds with same diagnosis
                if len(self._prev_diag_sigs) >= STABILITY_WINDOW:
                    recent = self._prev_diag_sigs[-STABILITY_WINDOW:]
                    if all(s == recent[0] for s in recent):
                        self._log(7,
                            f"diagnosis stable for {STABILITY_WINDOW} rounds "
                            f"({diag.primary_diagnosis[:60]}) — stopping",
                            arrow=True,
                        )
                        result.reinspection_history.append(ReinspectionPlan(
                            action="none",
                            stop_reason=f"diagnosis_stable: {diag.primary_diagnosis[:80]}",
                        ))
                        break

                current_sig = _diagnosis_signature(diag)
                if (
                    self._on("knowledge")
                    and not dry_run
                    and _hypothesis_changed(knowledge_diag_signature, current_sig)
                ):
                    self._log(7, "  KEEP refresh: diagnosis hypothesis changed; retrieving subtype context ...")
                    kr = self._s5(site_pred.data, diag, dry_run)
                    refresh_path = self.out_dir / f"ckpt_08e_keep_refresh_round_{ri}.json"
                    refresh_path.write_text(
                        json.dumps(
                            {
                                "round": ri,
                                "trigger": {
                                    "previous_signature": knowledge_diag_signature,
                                    "current_signature": current_sig,
                                },
                                "data": kr.data.to_dict() if hasattr(kr.data, "to_dict") else str(kr.data),
                            },
                            indent=2,
                            ensure_ascii=False,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
                    if kr.data and (kr.data.hypothesis_hits or kr.data.available):
                        result.knowledge = kr.data
                        _hh = "; ".join(
                            f"{h.get('diagnosis', h.get('name','?'))}"
                            for h in (kr.data.hypothesis_hits or [])[:3]
                        )
                        self._log(7, f"  → KEEP refreshed: [{_hh}]", arrow=True)
                    else:
                        self._log(7, f"  → KEEP refresh unavailable: {getattr(kr.data, 'error', '')}", arrow=True)
                    knowledge_diag_signature = current_sig
                ev_out2 = self._s7(all_selected, site_pred.data, diag, result.segmentation, result.morphology, result.knowledge, dry_run)
                ev_out = ev_out2
                result.evidence_sufficiency = ev_out2.data
                diag = apply_evidence_sufficiency_guard(diag, site_pred.data, ev_out2.data)
                result.diagnosis_prediction = diag
                _img2 = len(ev_out2.data.missing_image_evidence)
                _non2 = len(ev_out2.data.missing_nonimage_metadata)
                self._log(7, f"  after round {ri}: sufficient={ev_out2.data.sufficient_for_diagnosis}, image_gaps={_img2}, nonimage_gaps={_non2}, diag={diag.primary_diagnosis}", arrow=True)
                self._save("reinspect", {"round": ri, "plan": pd.to_dict(), "selected_count": len(all_selected)})
                session.record(f"reinspect_{ri}", {"action": pd.action, "added": len(new_rois)}, len(all_selected))
            if not self._on("reinspection") or ri == 0:
                pass  # message already printed or ablated
            else:
                self._log(7, f"{ri} round(s) completed, {len(all_selected)} ROIs total, can_report={ev_out.data.can_confidently_report}", arrow=True)
        else:
            self._log(7, "Reinspection — ABLATED (skipped)")
        result.reinspection_rounds = ri
        result.selected_count = len(all_selected)
        result.final_selected_rois = list(all_selected)

        # ── Step 8: Final diagnosis ───────────────────────────────────────────
        self._log(8, f"Final diagnosis ({len(all_selected)} ROIs) ...")
        if self._on("broad_hypotheses"):
            fb = self._ckpt("final_broad", resume, lambda: self._s6a(
                all_selected, site_pred.data, result.segmentation, result.morphology, result.knowledge, dry_run))
            result.broad_hypotheses = fb.data
        else:
            fb = StageResult(BroadHypothesesOutput(available=False, reason="ablated"), "broad_hypotheses")
        if self._on("subtype_sufficiency"):
            fs = self._ckpt("final_subtype", resume, lambda: self._s6b(
                all_selected, site_pred.data, result.segmentation, result.morphology, result.knowledge,
                fb.data, diag, dry_run))
            result.subtype_sufficiency = fs.data
        else:
            fs = StageResult(SubtypeSufficiencyOutput(available=False, reason="ablated"), "subtype_sufficiency")
        fd = self._ckpt("final_diagnosis", resume, lambda: self._s6(
            all_selected, site_pred.data, result.segmentation, result.morphology, result.knowledge,
            fb.data, fs.data, dry_run))
        diag = fd.data
        
        # ── Broad type locking: VLM's broad from ROI images is authoritative ──
        vlm_broad_final = diag.broad_type
        vlm_primary_final = diag.primary_diagnosis
        
        if self._on("subtype_sufficiency"):
            diag = apply_subtype_sufficiency_guard(diag, fs.data)
        if self._on("lineage_guard"):
            diag = apply_site_lineage_guard(diag, site_pred.data)
        diag = _reconcile_tiers(diag, fb.data)
        
        # Restore VLM's broad_type — it cannot be overridden by downstream guards
        if vlm_broad_final and vlm_broad_final not in ("", "unspecified", "neoplasm"):
            diag.broad_type = vlm_broad_final
        if not diag.primary_diagnosis and vlm_primary_final:
            diag.primary_diagnosis = vlm_primary_final
            
        result.diagnosis_prediction = diag
        if self._on("knowledge") and not dry_run:
            fk = self._ckpt("final_knowledge", resume, lambda: self._s5(site_pred.data, diag, dry_run))
            if fk.data and (fk.data.hypothesis_hits or fk.data.available):
                result.knowledge = fk.data
        # Stage 9: Evidence Graph
        if self._on("evidence_graph"):
            go = self._ckpt("graph", resume, lambda: self._s9(case_id, all_selected, site_pred.data, diag,
                result.evidence_sufficiency, result.segmentation, result.morphology, result.knowledge,
                result.broad_hypotheses, result.subtype_sufficiency, result.reinspection_history, dry_run))
            result.evidence_graph = go.data
        else:
            result.evidence_graph = EvidenceGraphOutput()
        _n_claims = len(result.evidence_graph.verified_claims or [])
        _slot_obj = result.evidence_graph.filtered_slots or result.evidence_graph.diagnosis_slots or {}
        _n_slots = len([k for k, v in _slot_obj.items() if v not in ("", [], None)])
        self._log(8, f"primary={diag.primary_diagnosis} ({diag.confidence}); broad={diag.broad_type}, subtype={diag.subtype or '—'}, malignancy={diag.malignancy or '—'}, grade={diag.grade or '—'}", arrow=True)

        # ── Step 9: Evidence graph / claim verification ───────────────────────
        if self._on("evidence_graph"):
            self._log(9, f"Evidence graph: {_n_claims} claims verified, {_n_slots} diagnosis slots", arrow=True)
        else:
            self._log(9, "Evidence graph — ABLATED (skipped)")

        # ── Step 10: Report generation ────────────────────────────────────────
        self._log(10, "Generating structured report ...")
        ro = self._ckpt("report", resume, lambda: self._s10(all_selected, site_pred.data, diag,
            result.evidence_sufficiency, result.segmentation, result.morphology, result.knowledge,
            result.evidence_graph, dry_run))
        result.report = ro.data
        result.elapsed_sec = time.monotonic() - t0
        session.record("report", {"elapsed_sec": result.elapsed_sec}, len(all_selected))
        _words = len(result.report.long_report.split()) if result.report else 0
        self._log(10, f"report={_words} words, elapsed={result.elapsed_sec:.1f}s", arrow=True)
        self._save("final", result.to_dict())
        self._save("session", session.to_dict())
        if result.report:
            (self.out_dir / "generated_long_report.txt").write_text(result.report.long_report, encoding="utf-8")
        if result.errors:
            (self.out_dir / CKPT_FILES["error"]).write_text(json.dumps({"errors": result.errors}, indent=2))
        if case_session_client is not None:
            self.llm = original_llm
            self.report_llm_client = original_report_llm
        return result

    def _s1(self, c, d): return RoiSelectionStage(self.settings, self.llm, self.out_dir/"01_roi").run(RoiSelectionInput(candidates=c, max_selected=self.settings.roi.selected_rois, max_triage=self.settings.roi.llm_triage_candidates), dry_run=d)
    def _s2(self, s, p, d): return SitePredictionStage(self.settings, self.llm, self.out_dir/"02_site").run(SitePredictionInput(selected_rois=s, diagnosis_priors=p), dry_run=d)
    def _s3(self, s, d): return SegmentationStage(self.settings, self.llm, self.out_dir/"03_seg").run(s, dry_run=d)
    def _s4(self, s, sp, seg, d, **kw):
        if self._on("seg_reader"):
            if not self.seg_reader_client:
                return ok(MorphologyOutput(available=False), "seg_reader_morphology")
            out = SegReaderMorphologyStage(
                self.settings, self.llm, self.out_dir/"04_morph",
                seg_reader_client=self.seg_reader_client,
            ).run(s, dry_run=d, site_prediction=sp, segmentation_context=seg.to_dict() if seg else {}, **kw)
            if self._on("seg_reader_guard") and out.data:
                out.data = apply_seg_reader_guard(out.data)
            return out
        return MorphologyStage(self.settings, self.llm, self.out_dir/"04_morph", medgemma_client=self.medgemma_client).run(s, dry_run=d, site_prediction=sp, **kw)
    def _s5(self, sp, di, d): return KnowledgeStage(self.settings, self.llm, self.out_dir/"05_knowledge").run({"site_prediction": sp, "diagnosis_prediction": di}, dry_run=d)
    def _ctx(self, know=None, broad=None, subtype=None):
        ctx = know.to_dict() if know and getattr(know, "available", False) else {}
        if broad:
            ctx["broad_hypotheses"] = broad.to_dict()
        if subtype:
            ctx["subtype_sufficiency"] = subtype.to_dict()
        return ctx or None

    def _seg_ctx(self, seg, *, role: str = "reasoning"):
        """Segmentation context visible to reasoning/report stages.

        Segmentation may still run for visualization and for the fine-tuned
        seg-reader.  The separate ``seg_stats_context`` switch controls whether
        compact segmentation stats are injected into Qwen reasoning/report
        prompts.  This lets SR2 isolate the reader signal from raw stats.
        """
        if self.ablation == "SS6_report_only_seg_stats" and role != "report":
            return None
        if not self._on("seg_stats_context"):
            return None
        if not seg or not seg.available:
            return None
        ctx = seg.to_dict()
        if self.ablation == "SS5_positive_seg_stats":
            return _positive_only_seg_context(ctx)
        return ctx

    def _s6a(self, s, sp, seg, morph, know, d):
        out = BroadHypothesesStage(self.settings, self.llm, self.out_dir/"06a_broad").run(
            SubtypeReasoningInput(selected_rois=s, site_prediction=sp,
                                  segmentation_context=self._seg_ctx(seg, role="reasoning"),
                                  morphology_context=morph.to_dict() if morph and morph.available else None,
                                  knowledge_context=know.to_dict() if know and know.available else None),
            dry_run=d)
        # Sanitize: ensure best_broad_type is a valid canonical label
        raw = out.data.best_broad_type
        sanitized = _sanitize_broad_type(raw)
        if sanitized != str(raw or "").lower():
            out.data.best_broad_type = sanitized
            out.data.reason += f" [sanitized: {raw} -> {sanitized}]"
        return out

    def _s6b(self, s, sp, seg, morph, know, broad, diag, d):
        return SubtypeSufficiencyStage(self.settings, self.llm, self.out_dir/"06b_subtype").run(
            SubtypeReasoningInput(selected_rois=s, site_prediction=sp,
                                  segmentation_context=self._seg_ctx(seg, role="reasoning"),
                                  morphology_context=morph.to_dict() if morph and morph.available else None,
                                  knowledge_context=know.to_dict() if know and know.available else None,
                                  broad_hypotheses=broad, diagnosis_prediction=diag),
            dry_run=d)

    def _s6(self, s, sp, seg, morph, know, broad, subtype, d):
        diag_stage = DiagnosisStage(self.settings, self.llm, self.out_dir/"06_diag")
        morph_ctx = morph.to_dict() if morph and morph.available else None
        know_ctx = self._ctx(know, broad, subtype)
        seg_ctx = self._seg_ctx(seg, role="reasoning")
        out = diag_stage.run(
            DiagnosisInput(selected_rois=s, site_prediction=sp,
                           segmentation_context=seg_ctx,
                           morphology_context=morph_ctx,
                           knowledge_context=know_ctx,
                           allow_chat_override="override" in self.ablation),
            dry_run=d)
        if self.ablation in {
            "SS7_dual_path_guarded_seg_stats",
            "SS8_dual_path_field_guarded_seg_stats",
            "SS9_dual_path_grade_necrosis_guarded_seg_stats",
            "SS10_dual_path_grade_adjudicated_seg_stats",
            "SS12_dual_path_richness_tiebreaker_seg_stats",
        } and not d:
            image_only = diag_stage.run(
                DiagnosisInput(selected_rois=s, site_prediction=sp,
                               segmentation_context=None,
                               morphology_context=morph_ctx,
                               knowledge_context=know_ctx,
                               allow_chat_override="override" in self.ablation),
                dry_run=d)
            if self.ablation == "SS8_dual_path_field_guarded_seg_stats":
                chosen, guard = _choose_dual_path_diagnosis_with_field_guard(
                    image_only.data, out.data, seg_ctx
                )
            elif self.ablation == "SS9_dual_path_grade_necrosis_guarded_seg_stats":
                chosen, guard = _choose_dual_path_diagnosis_with_grade_necrosis_guard(
                    image_only.data, out.data, seg_ctx
                )
            elif self.ablation == "SS10_dual_path_grade_adjudicated_seg_stats":
                chosen, guard = _choose_dual_path_diagnosis_with_grade_adjudication(
                    image_only.data, out.data, seg_ctx
                )
            elif self.ablation == "SS12_dual_path_richness_tiebreaker_seg_stats":
                chosen, guard = _choose_dual_path_diagnosis_ss12(image_only.data, out.data)
            else:
                chosen, guard = _choose_dual_path_diagnosis(image_only.data, out.data)
            sidecar = self.out_dir / f"ckpt_dual_path_diagnosis_{len(s)}roi.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "selected_roi_count": len(s),
                        "image_only": image_only.data.to_dict(),
                        "seg_stats": out.data.to_dict(),
                        "guard": guard,
                        "chosen": guard,
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )
            out.data = chosen
            out.metadata["dual_path_guard"] = guard
        raw = out.data.broad_type
        sanitized = _sanitize_broad_type(raw)
        if sanitized != str(raw or "").lower():
            out.data.broad_type = sanitized
        return out
    def _s7(self, s, sp, di, seg, morph, know, d): return EvidenceVerificationStage(self.settings, self.llm, self.out_dir/"07_evidence").run(EvidenceInput(selected_rois=s, site_prediction=sp, diagnosis_prediction=di, segmentation_context=self._seg_ctx(seg, role="reasoning"), knowledge_context=know.to_dict() if know and know.available else None, morphology_context=morph.to_dict() if morph and morph.available else None), dry_run=d)
    def _s9(self, cid, s, sp, di, suf, seg, morph, know, broad, subtype, reinspect, d): return EvidenceGraphStage(self.settings, self.llm, self.out_dir/"09_graph").run({"case_id": cid, "selected_rois": s, "site_prediction": sp, "diagnosis_prediction": di, "evidence_sufficiency": suf, "segmentation_context": self._seg_ctx(seg, role="reasoning") or {}, "morphology_context": morph.to_dict() if morph else {}, "knowledge_context": self._ctx(know, broad, subtype) or {}, "broad_hypotheses": broad, "subtype_sufficiency": subtype, "reinspection_history": reinspect, "enable_claim_verification": self._on("claim_verification"), "enable_llm_claim_verifier": self._on("llm_claim_verifier")}, dry_run=d)
    def _seg_stats_style(self) -> str:
        if self.ablation == "SS2_seg_claim_stats":
            return "claim_style"
        if self.ablation == "SS3_seg_hier_stats":
            return "hierarchical"
        if self.ablation == "SS4_seg_quality_stats":
            return "quality_aware"
        if self.ablation == "SS5_positive_seg_stats":
            return "positive_only"
        if "nl_session" in self.ablation:
            return "natural_language"
        return "compact_json"

    def _s10(self, s, sp, di, suf, seg, morph, know, graph, d): return ReportGenerationStage(self.settings, self.report_llm_client, self.out_dir/"10_report").run(ReportInput(selected_rois=s, site_prediction=sp, diagnosis_prediction=di, evidence_sufficiency=suf, segmentation_context=self._seg_ctx(seg, role="report"), knowledge_context=know.to_dict() if know and know.available else None, medgemma_context=morph.to_dict() if morph and morph.available else None, evidence_graph_context=graph.to_dict() if graph else None, enable_claim_filter=self._on("claim_filter_rewrite"), use_natural_language_stats="nl_session" in self.ablation, seg_stats_style=self._seg_stats_style()), dry_run=d)

    def _on(self, c: str) -> bool:
        if c in {"seg_reader", "seg_reader_guard", "case_chat_session"}:
            return c in _enabled_components(self.ablation)
        return c not in _disabled_components(self.ablation)

    def _ckpt(self, stage: str, resume: bool, runner):
        """Run a stage, with checkpoint skip. Always re-runs logic (stages have
        their own internal caches), but skips the LLM/work if checkpoint exists
        and we're resuming — by passing a flag. Actually, checkpoints here serve
        only as a *case-level* skip marker: if pipeline_result.json exists and
        resume=True, the caller skips the entire case. Per-stage checkpoints are
        saved for debugging/audit but do NOT short-circuit execution because
        reconstructing typed dataclasses from JSON dicts is fragile.

        If resume=True and checkpoint exists, we still re-run: individual stages
        (segmentation, morphology) have their own disk caches that make re-runs
        near-instant.
        """
        sr = runner()
        self._save(stage, {"data": sr.data.to_dict() if hasattr(sr.data, "to_dict") else str(sr.data)})
        return sr

    def _save(self, stage: str, data: dict) -> None:
        (self.out_dir / CKPT_FILES[stage]).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))

    def _load_session(self, resume: bool) -> SessionMemory:
        path = self.out_dir / CKPT_FILES["session"]
        return SessionMemory.load(path) if resume and path.exists() else SessionMemory()
