from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from pathxai_agent.agents.staged_report_agent import (
    LONG_REPORT_FORMAT,
    _dedupe_selected,
    _ensure_site_visual_roi,
    _raw_summary,
    _save_figures,
    _selected_candidates,
    _split_site_tumor_candidates,
    _strip_heavy,
    _annotate_wsi_thumbnail_with_rois,
    _generate_wsi_thumbnail,
    _get_wsi_thumbnail_from_rois,
    _img_data_uri as _staged_img_data_uri,
    _slide_artifact_key,
    assess_evidence_sufficiency,
    build_fallback_long_report,
    candidate_text,
    generate_medgemma_visual_description,
    generate_long_report_hybrid,
    known_site_prediction,
    predict_diagnosis,
    predict_site,
    retrieve_knowledge_context,
    retrieve_hypothesis_knowledge_context,
    run_roi_segmentation,
    select_diagnostic_rois,
)
from pathxai_agent.llm.qwen_client import QwenVisionClient
try:
    from pathxai_agent.wsi.slide_reader import SlideReader
except ImportError:
    SlideReader = None  # type: ignore[assignment]


DIAGNOSIS_SLOT_KEYS = [
    "site",
    "primary_diagnosis",
    "differential_diagnosis",
    "tumor_architecture",
    "cytology",
    "necrosis",
    "invasion",
    "stroma_inflammation",
    "ihc_molecular",
    "confidence",
    "missing_evidence",
    "recommended_ancillary_studies",
]


NFC_DIAGNOSIS_PATTERNS = [
    r",?\s*not further classifiable from ROI evidence(?: without molecular markers)?",
    r",?\s*not further classifiable",
]


def normalize_primary_diagnosis_payload(payload: dict | None) -> dict:
    """Keep primary diagnosis as the disease entity, not a prose limitation.

    "not further classifiable from ROI evidence" is an uncertainty qualifier.
    It belongs in not_further_classifiable_reason / histologic_subtype, not in
    primary_diagnosis or the Diagnosis section used for metrics.
    """
    if not isinstance(payload, dict):
        return {}
    out = dict(payload)
    primary = re.sub(r"\s+", " ", str(out.get("primary_diagnosis") or "").strip())
    if primary:
        cleaned = primary
        removed = False
        for pat in NFC_DIAGNOSIS_PATTERNS:
            if re.search(pat, cleaned, flags=re.I):
                cleaned = re.sub(pat, "", cleaned, flags=re.I).strip(" ,;")
                removed = True
                break
        if removed and cleaned:
            out["primary_diagnosis"] = cleaned
            out.setdefault("histologic_subtype", "not further classifiable")
            out.setdefault(
                "not_further_classifiable_reason",
                "Specific subtype cannot be further classified from current ROI evidence.",
            )
    # Differential belongs in its own field and must not be concatenated into
    # primary_diagnosis by downstream code.
    if isinstance(out.get("differential_diagnosis"), list) and not out.get("differential"):
        out["differential"] = out.get("differential_diagnosis")
    return out


def normalize_subtype_sufficiency_payload(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    out = dict(payload)
    rec = re.sub(r"\s+", " ", str(out.get("recommended_primary_diagnosis") or "").strip())
    if rec:
        cleaned = rec
        removed = False
        for pat in NFC_DIAGNOSIS_PATTERNS:
            if re.search(pat, cleaned, flags=re.I):
                cleaned = re.sub(pat, "", cleaned, flags=re.I).strip(" ,;")
                removed = True
                break
        if removed and cleaned:
            out["recommended_primary_diagnosis"] = cleaned
            out.setdefault(
                "not_further_classifiable_reason",
                "Specific subtype cannot be further classified from current ROI evidence.",
            )
    return out


def apply_indeterminate_potential_if_supported(
    diagnosis_prediction: dict | None,
    subtype_sufficiency: dict | None,
) -> dict:
    """Map broad neoplasm + unresolved benign/malignant subtype to UMP wording.

    This is intentionally morphology/logic-driven, not TCGA-type driven.  When
    the model supports a broad neoplasm but explicitly cannot distinguish
    adenoma/borderline/carcinoma because key discriminators are missing, the
    clinically faithful diagnosis is often "neoplasm of uncertain/indeterminate
    malignant potential" rather than either carcinoma or plain NOS neoplasm.
    """
    if not isinstance(diagnosis_prediction, dict):
        return {}
    out = dict(diagnosis_prediction)
    sub = subtype_sufficiency or {}
    primary = str(out.get("primary_diagnosis") or "").strip()
    primary_low = primary.lower()
    if not primary or "neoplasm" not in primary_low:
        return out
    if "indeterminate malignant potential" in primary_low or "uncertain malignant potential" in primary_low:
        return out
    level = str(sub.get("best_supported_level") or "").lower()
    assessments = sub.get("subtype_assessments") or []
    names = " ".join(str(row.get("subtype") or "") for row in assessments if isinstance(row, dict)).lower()
    statuses = " ".join(str(row.get("status") or "") for row in assessments if isinstance(row, dict)).lower()
    missing = " ".join(
        str(x)
        for row in assessments
        if isinstance(row, dict)
        for x in (row.get("missing_discriminators") or []) + (row.get("needed_image_evidence") or [])
    ).lower()
    unresolved_benign_malignant = (
        "broad" in level
        and any(term in names for term in ["carcinoma", "malignant"])
        and any(term in names for term in ["adenoma", "benign"])
        and any(term in statuses for term in ["insufficient", "not_assessable", "possible"])
    )
    missing_behavior_criteria = any(
        term in missing
        for term in ["weiss", "mitotic", "necrosis", "capsular", "vascular", "invasion"]
    )
    if not (unresolved_benign_malignant or missing_behavior_criteria):
        return out
    if "adreno" in primary_low or "adrenal cortical" in primary_low:
        out["primary_diagnosis"] = "Adrenal cortical neoplasm of indeterminate malignant potential"
    else:
        out["primary_diagnosis"] = f"{primary} of uncertain malignant potential"
    out["malignancy_status"] = "borderline/uncertain"
    out["histologic_subtype"] = primary
    out.setdefault(
        "behavior_qualifier_reason",
        "Subtype/behavior is indeterminate because benign versus malignant discriminators are insufficient in the current ROI evidence.",
    )
    return out


SITE_LINEAGE_PRIORS = [
    (("adrenal", "adrenocortical"), "adrenal cortical neoplasm", ("adrenal", "adreno", "cortical")),
    (("bladder", "urothelial"), "urothelial neoplasm", ("urothelial", "bladder")),
    (("breast", "mammary"), "breast epithelial neoplasm", ("breast", "mammary", "ductal", "lobular")),
    (("cervix", "cervical"), "cervical epithelial neoplasm", ("cervical", "cervix", "squamous", "endocervical")),
    (("colon", "rectum", "colorectal"), "colorectal epithelial neoplasm", ("colorectal", "colon", "rectal")),
    (("kidney", "renal"), "renal epithelial neoplasm", ("renal", "kidney")),
    (("bile duct", "biliary", "cholangi"), "biliary epithelial neoplasm", ("biliary", "cholangi", "bile duct")),
    (("liver", "hepatic"), "hepatic neoplasm", ("hepatic", "hepatocellular", "liver")),
    (("lung", "pulmonary"), "pulmonary epithelial neoplasm", ("lung", "pulmonary", "adenocarcinoma", "squamous")),
    (("lymphoid", "lymph node"), "lymphoid neoplasm", ("lymphoid", "lymphoma", "b-cell", "t-cell")),
    (("brain", "glial", "cns"), "glial neoplasm", ("glial", "glioma", "astrocytoma", "oligodendroglioma")),
    (("eye", "uveal", "choroid"), "uveal melanocytic neoplasm", ("uveal", "melanocytic", "melanoma")),
    (("uterus", "endomet"), "endometrial epithelial neoplasm", ("uter", "endometr")),
    (("ovary", "ovarian"), "ovarian epithelial neoplasm", ("ovarian", "ovary")),
    (("prostate",), "prostatic epithelial neoplasm", ("prostatic", "prostate")),
    (("thyroid",), "thyroid neoplasm", ("thyroid",)),
    (("skin", "melanocytic"), "melanocytic neoplasm", ("melanocytic", "melanoma", "skin")),
]


NON_SITE_LINEAGE_TERMS = (
    "lymphoma",
    "lymphoid",
    "reactive lymphoid",
    "metastatic",
    "breast",
    "renal",
    "urothelial",
    "colorectal",
    "pulmonary",
    "lung",
    "biliary",
    "glioma",
    "melanoma",
)


def _known_site_lineage_prior(site_prediction: dict | None) -> dict:
    site = str((site_prediction or {}).get("predicted_site") or "").lower()
    for site_terms, broad_type, match_terms in SITE_LINEAGE_PRIORS:
        if any(term in site for term in site_terms):
            return {
                "broad_type": broad_type,
                "match_terms": list(match_terms),
                "site": str((site_prediction or {}).get("predicted_site") or ""),
            }
    return {}


def _diagnosis_matches_site_lineage(diagnosis: str, site_prior: dict | None) -> bool:
    low = str(diagnosis or "").lower()
    if not low or not site_prior:
        return False
    broad = str(site_prior.get("broad_type") or "").lower()
    if broad and broad in low:
        return True
    return any(term and term in low for term in (site_prior.get("match_terms") or []))


def _has_non_site_lineage_signal(diagnosis: str, site_prior: dict | None) -> bool:
    low = str(diagnosis or "").lower()
    if not low:
        return False
    if _diagnosis_matches_site_lineage(low, site_prior):
        return False
    return any(term in low for term in NON_SITE_LINEAGE_TERMS)


def apply_known_site_lineage_guard(
    diagnosis_prediction: dict | None,
    subtype_sufficiency: dict | None,
    site_prediction: dict | None,
) -> dict:
    """Keep site-incompatible lineages as differential unless strongly proven.

    Known site is legal metadata in the known-site ablation.  It should not
    force a specific subtype, but it should prevent MedGemma's descriptive
    suspected diagnoses or D1 inflammatory fractions from becoming the primary
    diagnosis for a different lineage without independent strong support.
    """
    if not isinstance(diagnosis_prediction, dict):
        return {}
    out = dict(diagnosis_prediction)
    if not str((site_prediction or {}).get("source") or "").startswith("known_site"):
        return out
    site_prior = _known_site_lineage_prior(site_prediction)
    if not site_prior:
        return out
    primary = str(out.get("primary_diagnosis") or "").strip()
    recommended = str((subtype_sufficiency or {}).get("recommended_primary_diagnosis") or "").strip()
    if not _has_non_site_lineage_signal(primary, site_prior) and not _has_non_site_lineage_signal(recommended, site_prior):
        return out

    guarded_primary = site_prior["broad_type"]
    if recommended and _diagnosis_matches_site_lineage(recommended, site_prior):
        guarded_primary = recommended

    differential = [str(x).strip() for x in (out.get("differential") or []) if str(x).strip()]
    if primary and primary.lower() != guarded_primary.lower() and primary not in differential:
        differential.insert(0, primary)
    out["primary_diagnosis"] = guarded_primary
    out["confidence"] = "medium" if str(out.get("confidence") or "").lower() == "high" else (out.get("confidence") or "medium")
    out["differential"] = differential[:6]
    out.setdefault("limitations", [])
    if isinstance(out["limitations"], list):
        out["limitations"].append(
            "Known-site lineage guard applied: a non-site-lineage diagnosis was kept as differential because it was driven mainly by morphology description or inflammatory segmentation rather than definitive lineage evidence."
        )
    out["site_lineage_guard"] = {
        "applied": True,
        "known_site": site_prior.get("site"),
        "site_lineage_primary": guarded_primary,
        "demoted_primary": primary,
        "reason": "Non-site lineage requires strong independent evidence/IHC before becoming primary in known-site ablation.",
    }
    return out


def apply_known_site_lineage_guard_to_subtype_sufficiency(
    subtype_sufficiency: dict | None,
    site_prediction: dict | None,
) -> dict:
    if not isinstance(subtype_sufficiency, dict):
        return {}
    out = dict(subtype_sufficiency)
    if not str((site_prediction or {}).get("source") or "").startswith("known_site"):
        return out
    site_prior = _known_site_lineage_prior(site_prediction)
    rec = str(out.get("recommended_primary_diagnosis") or "").strip()
    if not site_prior or not _has_non_site_lineage_signal(rec, site_prior):
        return out
    out["site_lineage_guard"] = {
        "applied": True,
        "known_site": site_prior.get("site"),
        "demoted_recommendation": rec,
        "site_lineage_primary": site_prior["broad_type"],
        "reason": "Subtype sufficiency cannot promote a different lineage using MedGemma description or D1 inflammatory fraction alone.",
    }
    out["recommended_primary_diagnosis"] = site_prior["broad_type"]
    out["best_supported_level"] = "broad_type"
    out["confidence"] = "medium" if str(out.get("confidence") or "").lower() == "high" else (out.get("confidence") or "medium")
    out.setdefault("non_site_lineage_differential", [])
    if rec and rec not in out["non_site_lineage_differential"]:
        out["non_site_lineage_differential"].insert(0, rec)
    reason = str(out.get("reason") or "").strip()
    guard_reason = (
        "Known-site lineage guard: non-site-lineage recommendation is retained as a differential, "
        "not as primary, unless confirmed by stronger lineage evidence."
    )
    out["reason"] = f"{reason} {guard_reason}".strip()
    return out


IMAGE_ACTIONABLE_TERMS = {
    "necrosis",
    "invasion",
    "vascular",
    "capsular",
    "interface",
    "tumor-normal",
    "tumour-normal",
    "tumor stroma",
    "tumour stroma",
    "mitotic",
    "mitoses",
    "atypia",
    "cytology",
    "architecture",
    "diffuse",
    "gland",
    "trabecular",
    "solid",
    "inflammation",
    "hemorrhage",
    "haemorrhage",
    "quality",
    "high power",
}

STRONG_IMAGE_ACTIONABLE_TERMS = {
    "necrosis",
    "tumor-normal",
    "tumour-normal",
    "interface",
    "tumor stroma",
    "tumour stroma",
    "architecture",
    "cytology",
    "gland",
    "trabecular",
    "solid",
    "inflammation",
    "hemorrhage",
    "haemorrhage",
    "tissue quality",
    "representative",
    "high power region",
    "high-power region",
}

NONIMAGE_METADATA_TERMS = {
    "ihc",
    "immunohistochemistry",
    "molecular",
    "genetic",
    "mutation",
    "ki-67",
    "sf-1",
    "inhibin",
    "melan-a",
    "gross",
    "size",
    "measurement",
    "margin",
    "tnm",
    "stage",
    "clinical",
    "history",
    "procedure",
    "specimen",
    "mitotic count",
    "weiss score",
}

NONIMAGE_DOMINANT_TERMS = {
    "ihc",
    "immunohistochemistry",
    "molecular",
    "genetic",
    "mutation",
    "ki-67",
    "gross",
    "size",
    "measurement",
    "margin",
    "tnm",
    "stage",
    "clinical",
    "history",
    "procedure",
    "mitotic count",
    "per 50 hpf",
    "weiss score",
    "capsular invasion",
    "vascular invasion",
    "lymph node",
}

ZOOM_ACTIONABLE_TERMS = {
    "high power",
    "high-power",
    "higher power",
    "cytology",
    "nuclear",
    "atypia",
    "mitotic",
    "mitoses",
    "cellular",
    "detail",
    "small focus",
    "subtle",
}

ADD_ROI_ACTIONABLE_TERMS = {
    "necrosis",
    "interface",
    "tumor-normal",
    "tumour-normal",
    "tumor stroma",
    "tumour stroma",
    "invasive front",
    "alternative morphology",
    "heterogeneity",
    "representative",
    "quality",
    "hemorrhage",
    "haemorrhage",
}


def run_evidence_graph_agent_hybrid(
    case_id: str,
    candidate_rois: list[dict],
    slides_used: list[dict],
    stages_client: QwenVisionClient | None,
    short_report_client: QwenVisionClient | None,
    long_report_client: QwenVisionClient | None,
    out_dir: str | Path,
    max_selected_rois: int = 5,
    max_triage_candidates: int | None = None,
    max_site_rois: int = 3,
    dry_run: bool = False,
    enforce_staged_diagnosis: bool = False,
    skip_claim_verification: bool = False,
    known_site: str | None = None,
    known_cancer_type: str | None = None,
    segmentation_config: dict | None = None,
    knowledge_config: dict | None = None,
    max_reinspection_rois: int = 2,
    max_zoom_depth: int = 1,
    max_reinspection_rounds: int = 3,
    seg_visual_mode: str = "text_only",
) -> dict:
    """Case-level evidence graph agent.

    This is intentionally added beside the older staged pipeline.  The older
    staged/hybrid reports remain reproducible; this variant changes the core
    output to evidence graph -> diagnosis slots -> verified claims -> report.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_rows(candidate_rois)
    if not candidates:
        raise ValueError("No candidate ROIs were provided to evidence graph agent")

    site_candidates, tumor_candidates, has_separate_evidence = _split_site_tumor_candidates(candidates)
    selection_pool = tumor_candidates if has_separate_evidence and tumor_candidates else candidates
    site_evidence = site_candidates[:max_site_rois] if has_separate_evidence and site_candidates else []
    diagnostic_session_context: dict = {
        "version": "diagnostic_session_memory_v1",
        "events": [],
        "known_site": known_site,
    }

    ready = load_report_ready_checkpoint(out_dir)
    if ready:
        print("    [resume] report-ready checkpoint found; skipping to final report generation", flush=True)
        diagnostic_session_context = _context_from_checkpoint(ready, diagnostic_session_context, "resume_report_ready")
        selected_tumor = ready.get("selected_rois") or []
        site_prediction = ready.get("site_prediction") or known_site_prediction(known_site)
        broad_hypotheses = ready.get("broad_hypotheses") or {}
        subtype_sufficiency = ready.get("subtype_sufficiency") or {}
        diagnosis_prediction = ready.get("diagnosis_prediction") or {}
        sufficiency = ready.get("evidence_sufficiency") or {}
        segmentation_context = ready.get("segmentation_context") or {"available": False}
        knowledge_context = ready.get("knowledge_context") or {"available": False}
        medgemma_morphology_context = ready.get("medgemma_morphology_context") or {"available": False}
        reinspection = ready.get("reinspection_plan") or {}
        graph = ready.get("evidence_graph") or {}
        diagnosis_slots = ready.get("diagnosis_slots") or {}
        filtered_slots = ready.get("filtered_diagnosis_slots") or diagnosis_slots
        claim_verification = ready.get("claim_verification") or {"verification_version": "checkpoint_missing", "claims": []}
        roi_selection = ready.get("roi_selection") or {"reason": "loaded from report-ready checkpoint"}
        visual_rois = _ensure_site_visual_roi(_dedupe_selected(site_evidence + selected_tumor), site_evidence)
        report_payload = generate_claim_verified_report(
            selected_rois=selected_tumor,
            visual_rois=visual_rois,
            site_prediction=site_prediction,
            diagnosis_prediction=diagnosis_prediction,
            sufficiency=sufficiency,
            diagnosis_slots=filtered_slots,
            claim_verification=claim_verification,
            segmentation_context=segmentation_context,
            knowledge_context=knowledge_context,
            short_report_client=short_report_client,
            long_report_client=long_report_client,
            dry_run=dry_run,
            enforce_staged_diagnosis=enforce_staged_diagnosis,
            medgemma_context=medgemma_morphology_context,
            diagnostic_session_context=diagnostic_session_context,
        )
        payload = {
            "case_id": case_id,
            "mode": "case_evidence_graph_agent_v1",
            "resume_from_progress_checkpoint": True,
            "candidate_count": len(candidates),
            "selected_count": len(selected_tumor),
            "max_zoom_depth": max_zoom_depth,
            "max_reinspection_rounds": max_reinspection_rounds,
            "slides_used": slides_used,
            "candidate_rois": candidates,
            "site_rois": site_evidence,
            "tumor_rois": selected_tumor,
            "selected_rois": selected_tumor,
            "visual_rois": visual_rois,
            "roi_selection": roi_selection,
            "site_prediction": site_prediction,
            "broad_hypotheses": broad_hypotheses,
            "subtype_sufficiency": subtype_sufficiency,
            "diagnosis_prediction": diagnosis_prediction,
            "segmentation_context": segmentation_context,
            "knowledge_context": knowledge_context,
            "medgemma_morphology_context": medgemma_morphology_context,
            "evidence_sufficiency": sufficiency,
            "reinspection_plan": reinspection,
            "evidence_graph": graph,
            "diagnosis_slots": diagnosis_slots,
            "filtered_diagnosis_slots": filtered_slots,
            "claim_verification": claim_verification,
            "generated_long_report": report_payload["long_report"],
            "generated_long_report_rich": report_payload.get("long_report_rich", report_payload["long_report"]),
            "medgemma_short_report": report_payload.get("short_report"),
            "figures": report_payload.get("figures") or [],
            "raw_generation": report_payload.get("raw"),
            "diagnostic_session_context": diagnostic_session_context,
        }
        _save_figures(payload["figures"], out_dir)
        save_evidence_graph_outputs(payload, out_dir)
        return payload

    stage1 = load_progress_checkpoint(out_dir, "01_roi_selection")
    if stage1.get("selected_rois"):
        diagnostic_session_context = _context_from_checkpoint(stage1, diagnostic_session_context, "01_roi_selection")
        selected_tumor = stage1.get("selected_rois") or []
        roi_selection = stage1.get("roi_selection") or {"reason": "loaded from checkpoint"}
        print(f"    [1/10] ROI evidence seeding → loaded {len(selected_tumor)} ROI(s) from checkpoint", flush=True)
    else:
        print(f"    [1/10] ROI evidence seeding ({len(selection_pool)} candidates) ...", flush=True)
        # Do not let one early diagnostic hypothesis control the evidence search.
        # Seed the case with a small, diverse evidence set: high tumor-score ROIs,
        # optional site/support ROIs, and spatially distinct representative ROIs.
        selected_tumor = select_initial_evidence_rois(
            selection_pool,
            site_evidence=site_evidence,
            max_selected_rois=max_selected_rois,
            max_per_bucket=2,
        )
        roi_selection = {
            "reason": (
                f"seeded {len(selected_tumor)} ROIs using limited per-bucket evidence "
                "selection to reduce preliminary-diagnosis confirmation bias"
            ),
            "selection_strategy": "bucketed_evidence_seed_v1",
            "selected_roi_indices": [int(row.get("roi_index") or i + 1) for i, row in enumerate(selected_tumor)],
            "roi_scores": [],
            "raw": None,
        }
        buckets = ", ".join(
            f"{row.get('evidence_bucket', 'evidence')}#{row.get('roi_index')}"
            for row in selected_tumor
        )
        print(f"    [1/10] → {len(selected_tumor)} seed ROIs selected ({buckets})", flush=True)
        write_progress_checkpoint(out_dir, "01_roi_selection", {
            "roi_selection": roi_selection,
            "selected_rois": selected_tumor,
            "candidate_count": len(candidates),
            "diagnostic_session_context": update_diagnostic_session_context(
                diagnostic_session_context,
                "01_roi_selection",
                {"roi_selection": roi_selection, "selected_rois": selected_tumor, "candidate_count": len(candidates)},
                len(selected_tumor),
            ),
        })
        diagnostic_session_context = load_progress_checkpoint(out_dir, "01_roi_selection").get("diagnostic_session_context") or diagnostic_session_context

    stage2 = load_progress_checkpoint(out_dir, "02_site_prediction")
    if stage2.get("site_prediction"):
        diagnostic_session_context = _context_from_checkpoint(stage2, diagnostic_session_context, "02_site_prediction")
        site_prediction = stage2.get("site_prediction") or {}
        print(f"    [2/10] Site → loaded {site_prediction.get('predicted_site', '?')} from checkpoint", flush=True)
    else:
        print(f"    [2/10] Site: {'known ' + str(known_site) + ' (no LLM)' if known_site else 'predicting ...'}", flush=True)
        site_prediction = known_site_prediction(known_site) if known_site else predict_site(
            _dedupe_selected(site_evidence + selected_tumor),
            qwen_client=stages_client,
            dry_run=dry_run,
        )
        print(f"    [2/10] → {site_prediction.get('predicted_site', '?')}", flush=True)
        diagnostic_session_context = update_diagnostic_session_context(
            diagnostic_session_context,
            "02_site_prediction",
            {"site_prediction": site_prediction},
            len(selected_tumor),
        )
        write_progress_checkpoint(out_dir, "02_site_prediction", {
            "site_prediction": site_prediction,
            "diagnostic_session_context": diagnostic_session_context,
        })

    stage4 = load_progress_checkpoint(out_dir, "04_segmentation_qc")
    if stage4.get("segmentation_context"):
        diagnostic_session_context = _context_from_checkpoint(stage4, diagnostic_session_context, "04_segmentation_qc")
        segmentation_context = stage4.get("segmentation_context") or {"available": False}
        print(f"    [3/10] Segmentation + [4/10] Seg QC → loaded from checkpoint", flush=True)
    else:
        print(f"    [3/10] Segmentation (MedSAM) ...", flush=True)
        segmentation_context = run_roi_segmentation(
            selected_tumor,
            segmentation_config=segmentation_config,
            out_dir=out_dir / "segmentation",
        )
        print(f"    [4/10] Seg QC (LLM visual audit) ...", flush=True)
        seg_qc: dict = {"available": False, "note": "skipped", "per_roi": {}}
        if seg_visual_mode == "composite":
            seg_qc = safe_assess_segmentation_quality(
                segmentation_context,
                qwen_client=stages_client,
                dry_run=dry_run,
                stage_label="initial_seg_qc",
            )
        else:
            print(f"    [4/10] → skipped (seg_visual_mode={seg_visual_mode})", flush=True)
        segmentation_context["seg_qc"] = seg_qc
        if seg_qc.get("available"):
            qc_counts = _seg_qc_counts(seg_qc, expected_count=len(segmentation_context.get("roi_results") or []))
            print(
                f"    [4/10] → {qc_counts['good']} good, {qc_counts['flagged']} flagged, "
                f"{qc_counts['missing']} missing/unchecked "
                f"({qc_counts['returned']}/{qc_counts['submitted']} QC returned)",
                flush=True,
            )
            if qc_counts["missing"]:
                missing_ids = ", ".join(str(x) for x in (seg_qc.get("missing_roi_indices") or [])[:8])
                print(f"    [4/10] → seg QC missing ROI indices: {missing_ids or 'unknown'}", flush=True)
        elif seg_qc.get("error"):
            print(f"    [4/10] → Seg QC skipped after error: {seg_qc.get('error')}", flush=True)
        diagnostic_session_context = update_diagnostic_session_context(
            diagnostic_session_context,
            "04_segmentation_qc",
            {"segmentation_context": segmentation_context},
            len(selected_tumor),
        )
        write_progress_checkpoint(out_dir, "04_segmentation_qc", {
            "segmentation_context": segmentation_context,
            "diagnostic_session_context": diagnostic_session_context,
        })

    medgemma_morphology_rounds: list[dict] = []
    stage5 = load_progress_checkpoint(out_dir, "05_medgemma_morphology")
    if stage5.get("medgemma_morphology_context"):
        diagnostic_session_context = _context_from_checkpoint(stage5, diagnostic_session_context, "05_medgemma_morphology")
        medgemma_morphology_context = stage5.get("medgemma_morphology_context") or {"available": False}
        medgemma_morphology_rounds = list(medgemma_morphology_context.get("rounds") or [])
        print(f"    [5/10] MedGemma morphology evidence → loaded from checkpoint", flush=True)
    else:
        print(f"    [5/10] MedGemma morphology evidence (initial {len(selected_tumor)} ROIs) ...", flush=True)
        medgemma_initial = cached_medgemma_visual_description(
            out_dir=out_dir,
            selected_rois=selected_tumor,
            site_prediction=site_prediction,
            diagnosis_prediction=None,
            short_report_client=short_report_client,
            dry_run=dry_run,
            previous_context=None,
            round_index=1,
            input_role="initial",
        )
        medgemma_morphology_rounds.append(medgemma_initial)
        medgemma_morphology_context = accumulate_medgemma_morphology(medgemma_morphology_rounds)
        print(
            f"    [5/10] → available={medgemma_morphology_context.get('available')}, "
            f"chars={len(str(medgemma_morphology_context.get('description') or ''))}",
            flush=True,
        )
        diagnostic_session_context = update_diagnostic_session_context(
            diagnostic_session_context,
            "05_medgemma_morphology",
            {"medgemma_morphology_context": medgemma_morphology_context},
            len(selected_tumor),
        )
        write_progress_checkpoint(out_dir, "05_medgemma_morphology", {
            "medgemma_morphology_context": medgemma_morphology_context,
            "diagnostic_session_context": diagnostic_session_context,
        })

    stage6 = load_progress_checkpoint(out_dir, "06_preliminary_diagnosis_sufficiency")
    if stage6.get("diagnosis_prediction") and stage6.get("evidence_sufficiency"):
        diagnostic_session_context = _context_from_checkpoint(stage6, diagnostic_session_context, "06_preliminary_diagnosis_sufficiency")
        diagnosis_prediction = stage6.get("diagnosis_prediction") or {}
        broad_hypotheses = stage6.get("broad_hypotheses") or {}
        subtype_sufficiency = stage6.get("subtype_sufficiency") or {}
        knowledge_context = stage6.get("knowledge_context") or {"available": False}
        subtype_sufficiency = apply_known_site_lineage_guard_to_subtype_sufficiency(
            normalize_subtype_sufficiency_payload(subtype_sufficiency),
            site_prediction,
        )
        diagnosis_prediction = apply_known_site_lineage_guard(
            apply_indeterminate_potential_if_supported(
                normalize_primary_diagnosis_payload(diagnosis_prediction),
                subtype_sufficiency,
            ),
            subtype_sufficiency,
            site_prediction,
        )
        diagnosis_prediction = apply_indeterminate_potential_if_supported(diagnosis_prediction, subtype_sufficiency)
        sufficiency = normalize_evidence_gaps(stage6.get("evidence_sufficiency") or {}, diagnosis_prediction)
        print(f"    [6/10] Preliminary diagnosis + sufficiency → loaded from checkpoint", flush=True)
        print(f"    [6/10] → KEEP loaded: {_keep_context_summary(knowledge_context)}", flush=True)
    else:
        print(f"    [6/10] Preliminary diagnosis + sufficiency (initial {len(selected_tumor)} ROIs) ...", flush=True)
        prior = site_prediction.get("diagnosis_prior") or {}
        prior_knowledge_context = retrieve_knowledge_context(
            site_prediction=site_prediction,
            diagnosis_prediction={
                "primary_diagnosis": prior.get("broad_type") or "",
                "differential": prior.get("subtype_candidates") or [],
                "visual_evidence": [],
            },
            knowledge_config=knowledge_config,
        )
        print(f"    [6/10] → KEEP prior candidates: {_keep_context_summary(prior_knowledge_context)}", flush=True)
        broad_hypotheses = predict_broad_hypotheses(
            selected_rois=selected_tumor,
            site_prediction=site_prediction,
            segmentation_context=segmentation_context,
            morphology_context=medgemma_morphology_context,
            knowledge_context=prior_knowledge_context,
            qwen_client=stages_client,
            dry_run=dry_run,
            seg_visual_mode=seg_visual_mode,
            diagnostic_session_context=diagnostic_session_context,
        )
        subtype_sufficiency = judge_subtype_sufficiency(
            selected_rois=selected_tumor,
            site_prediction=site_prediction,
            broad_hypotheses=broad_hypotheses,
            knowledge_context=prior_knowledge_context,
            segmentation_context=segmentation_context,
            morphology_context=medgemma_morphology_context,
            qwen_client=stages_client,
            dry_run=dry_run,
            seg_visual_mode=seg_visual_mode,
            diagnostic_session_context=diagnostic_session_context,
        )
        subtype_sufficiency = normalize_subtype_sufficiency_payload(subtype_sufficiency)
        subtype_sufficiency = apply_known_site_lineage_guard_to_subtype_sufficiency(
            subtype_sufficiency,
            site_prediction,
        )
        prior_knowledge_context = _enrich_knowledge_context(
            prior_knowledge_context,
            broad_hypotheses=broad_hypotheses,
            subtype_sufficiency=subtype_sufficiency,
        )
        print(
            f"    [6/10] → broad={broad_hypotheses.get('best_broad_type') or '?'}, "
            f"subtype_level={subtype_sufficiency.get('best_supported_level') or '?'}",
            flush=True,
        )
        diagnosis_prediction = predict_diagnosis(
            selected_tumor,
            site_prediction=site_prediction,
            segmentation_context=segmentation_context,
            morphology_context=medgemma_morphology_context,
            qwen_client=stages_client,
            dry_run=dry_run,
            seg_visual_mode=seg_visual_mode,
            knowledge_context=prior_knowledge_context,
            diagnostic_session_context=diagnostic_session_context,
        )
        diagnosis_prediction = normalize_primary_diagnosis_payload(diagnosis_prediction)
        diagnosis_prediction = apply_indeterminate_potential_if_supported(diagnosis_prediction, subtype_sufficiency)
        diagnosis_prediction = apply_known_site_lineage_guard(
            diagnosis_prediction,
            subtype_sufficiency,
            site_prediction,
        )
        diagnosis_prediction = apply_indeterminate_potential_if_supported(diagnosis_prediction, subtype_sufficiency)
        knowledge_context = retrieve_knowledge_context(
            site_prediction=site_prediction,
            diagnosis_prediction=diagnosis_prediction,
            knowledge_config=knowledge_config,
        )
        knowledge_context = _enrich_knowledge_context(
            knowledge_context,
            broad_hypotheses=broad_hypotheses,
            subtype_sufficiency=subtype_sufficiency,
        )
        print(f"    [6/10] → KEEP diagnosis context: {_keep_context_summary(knowledge_context)}", flush=True)
        sufficiency = assess_evidence_sufficiency(
            _dedupe_selected(site_evidence + selected_tumor),
            site_prediction=site_prediction,
            diagnosis_prediction=diagnosis_prediction,
            segmentation_context=segmentation_context,
            knowledge_context=knowledge_context,
            morphology_context=medgemma_morphology_context,
            qwen_client=stages_client,
            dry_run=dry_run,
            diagnostic_session_context=diagnostic_session_context,
        )
        sufficiency = normalize_evidence_gaps(sufficiency, diagnosis_prediction)
        print(
            f"    [6/10] → preliminary_hypotheses: {_diagnosis_hypothesis_summary(diagnosis_prediction)}, "
            f"sufficient={sufficiency.get('sufficient_for_diagnosis')}, "
            f"image_gaps={len(sufficiency.get('missing_image_evidence') or [])}, "
            f"nonimage_gaps={len(sufficiency.get('missing_nonimage_metadata') or [])}",
            flush=True,
        )
        diagnostic_session_context = update_diagnostic_session_context(
            diagnostic_session_context,
            "06_preliminary_diagnosis_sufficiency",
            {
                "broad_hypotheses": broad_hypotheses,
                "subtype_sufficiency": subtype_sufficiency,
                "diagnosis_prediction": diagnosis_prediction,
                "knowledge_context": knowledge_context,
                "evidence_sufficiency": sufficiency,
            },
            len(selected_tumor),
        )
        write_progress_checkpoint(out_dir, "06_preliminary_diagnosis_sufficiency", {
            "diagnosis_prediction": diagnosis_prediction,
            "broad_hypotheses": broad_hypotheses,
            "subtype_sufficiency": subtype_sufficiency,
            "knowledge_context": knowledge_context,
            "evidence_sufficiency": sufficiency,
            "diagnostic_session_context": diagnostic_session_context,
        })

    # ── Reinspection loop ──
    print(f"    [7/10] Reinspection loop ...", flush=True)
    reinspection_rounds = []
    reinspection = {"extra_rois": [], "plan": "", "enabled": False}
    max_reinspect_iter = max(0, int(max_reinspection_rounds))
    reinspect_iter = 0
    used_candidate_ids = {r.get("roi_id") for r in selected_tumor}
    reinspection_stop_reason = ""
    last_remaining_count = 0
    last_planned_reinspection: dict | None = None
    nonimage_dominant_rounds = 0
    restored_rounds = restore_reinspection_round_checkpoints(out_dir, max_reinspect_iter)
    if restored_rounds:
        latest = restored_rounds[-1]
        diagnostic_session_context = _context_from_checkpoint(latest, diagnostic_session_context, f"07_reinspection_round_{len(restored_rounds)}")
        print(
            f"    [7/10] → resume: restored {len(restored_rounds)} completed reinspection round(s)",
            flush=True,
        )
        selected_tumor = latest.get("selected_rois") or selected_tumor
        segmentation_context = latest.get("segmentation_context") or segmentation_context
        medgemma_morphology_context = latest.get("medgemma_morphology_context") or medgemma_morphology_context
        broad_hypotheses = latest.get("broad_hypotheses") or broad_hypotheses
        subtype_sufficiency = latest.get("subtype_sufficiency") or subtype_sufficiency
        diagnosis_prediction = latest.get("diagnosis_prediction") or diagnosis_prediction
        knowledge_context = latest.get("knowledge_context") or knowledge_context
        sufficiency = latest.get("evidence_sufficiency") or sufficiency
        sufficiency = normalize_evidence_gaps(sufficiency, diagnosis_prediction)
        medgemma_morphology_rounds = list(medgemma_morphology_context.get("rounds") or medgemma_morphology_rounds)
        reinspection_rounds = [
            row.get("reinspection_round") or {}
            for row in restored_rounds
            if isinstance(row.get("reinspection_round"), dict)
        ]
        reinspection = reinspection_rounds[-1] if reinspection_rounds else reinspection
        reinspect_iter = len(restored_rounds)
        used_candidate_ids = {r.get("roi_id") for r in selected_tumor}
    while (
        reinspect_iter < max_reinspect_iter
        and not sufficiency.get("sufficient_for_diagnosis")
        and sufficiency.get("has_image_actionable_gap")
        and max_reinspection_rois > 0
    ):
        remaining = [c for c in candidates if c.get("roi_id") not in used_candidate_ids]
        last_remaining_count = len(remaining)
        if not remaining:
            reinspection_stop_reason = (
                f"candidate pool exhausted: {len(used_candidate_ids)} used ROI id(s), "
                f"{len(candidates)} total candidate ROI(s)"
            )
            break
        ri = plan_reinspection(
            candidates=remaining,
            selected_rois=selected_tumor,
            sufficiency=sufficiency,
            diagnosis_prediction=diagnosis_prediction,
            max_extra_rois=max_reinspection_rois,
            out_dir=out_dir,
            max_zoom_depth=max_zoom_depth,
        )
        last_planned_reinspection = ri
        extra = ri.get("new_rois") or ri.get("zoom_rois") or ri.get("extra_rois") or []
        print(f"    [7/10] → plan: action={ri.get('action')}, new_rois={len(extra)}, remaining_candidates={len(remaining)}, image_gaps={len(ri.get('missing_image_evidence') or [])}", flush=True)
        if not extra:
            is_actionable = sufficiency.get("has_image_actionable_gap")
            reinspection_stop_reason = (
                f"planner produced no new ROI for action={ri.get('action')}; "
                f"remaining_candidates={len(remaining)}, "
                f"actionable_gap={is_actionable}, "
                f"missing_image={ri.get('missing_image_evidence')[:3]}, "
                f"zoom_error={ri.get('zoom_error') or 'none'}"
            )
            print(f"    [7/10] → STOP: {reinspection_stop_reason}", flush=True)
            break
        reinspect_iter += 1
        action = ri.get("action") or ri.get("strategy") or "add_independent_roi"
        print(f"    [7/10] → round {reinspect_iter}: {action}, adding {len(extra)} ROIs, rerunning ...", flush=True)
        selected_tumor = _dedupe_selected(selected_tumor + extra)[:max_selected_rois + max_reinspection_rois * reinspect_iter]
        used_candidate_ids |= {r.get("roi_id") for r in extra}
        # Zoom ROIs: crop parent seg mask instead of re-running MedSAM
        _patch_zoom_seg_from_parents(selected_tumor, segmentation_context, out_dir)
        # Only run MedSAM on non-zoom ROIs
        non_zoom = [r for r in selected_tumor if not r.get("zoom_parent_roi_id")]
        if non_zoom:
            zoom_seg_ctx = run_roi_segmentation(
                non_zoom,
                segmentation_config=segmentation_config,
                out_dir=out_dir / "segmentation",
            )
            # Merge zoom seg results back
            _merge_zoom_seg_into_context(zoom_seg_ctx, segmentation_context, selected_tumor)
            segmentation_context = zoom_seg_ctx
        seg_qc2: dict = {"available": False, "note": "skipped", "per_roi": {}}
        if seg_visual_mode == "composite":
            seg_qc2 = safe_assess_segmentation_quality(
                segmentation_context,
                qwen_client=stages_client,
                dry_run=dry_run,
                stage_label=f"reinspection_round_{reinspect_iter}_seg_qc",
            )
        segmentation_context["seg_qc"] = seg_qc2
        if seg_qc2.get("available"):
            qc_counts = _seg_qc_counts(seg_qc2, expected_count=len(segmentation_context.get("roi_results") or []))
            print(
                f"    [7/10] → seg QC after reinspection: {qc_counts['good']} good, "
                f"{qc_counts['flagged']} flagged, {qc_counts['missing']} missing/unchecked "
                f"({qc_counts['returned']}/{qc_counts['submitted']} QC returned)",
                flush=True,
            )
            if qc_counts["missing"]:
                missing_ids = ", ".join(str(x) for x in (seg_qc2.get("missing_roi_indices") or [])[:8])
                print(f"    [7/10] → seg QC missing ROI indices: {missing_ids or 'unknown'}", flush=True)
        elif seg_qc2.get("error"):
            print(f"    [7/10] → seg QC skipped after error: {seg_qc2.get('error')}", flush=True)
        elif not seg_qc2.get("available"):
            pass  # silencely skipped for non-composite modes
        medgemma_round = cached_medgemma_visual_description(
            out_dir=out_dir,
            selected_rois=extra,
            site_prediction=site_prediction,
            diagnosis_prediction=diagnosis_prediction,
            short_report_client=short_report_client,
            dry_run=dry_run,
            previous_context=None,
            round_index=reinspect_iter + 1,
            input_role="reinspection_added_rois",
        )
        medgemma_morphology_rounds.append(medgemma_round)
        medgemma_morphology_context = accumulate_medgemma_morphology(medgemma_morphology_rounds)
        print(
            f"    [7/10] → MedGemma morphology added round {reinspect_iter + 1}: "
            f"available={medgemma_morphology_context.get('available')}, "
            f"chars={len(str(medgemma_morphology_context.get('description') or ''))}",
            flush=True,
        )
        broad_hypotheses = predict_broad_hypotheses(
            selected_rois=selected_tumor,
            site_prediction=site_prediction,
            segmentation_context=segmentation_context,
            morphology_context=medgemma_morphology_context,
            knowledge_context=knowledge_context,
            qwen_client=stages_client,
            dry_run=dry_run,
            seg_visual_mode=seg_visual_mode,
            diagnostic_session_context=diagnostic_session_context,
        )
        subtype_sufficiency = judge_subtype_sufficiency(
            selected_rois=selected_tumor,
            site_prediction=site_prediction,
            broad_hypotheses=broad_hypotheses,
            knowledge_context=knowledge_context,
            segmentation_context=segmentation_context,
            morphology_context=medgemma_morphology_context,
            qwen_client=stages_client,
            dry_run=dry_run,
            seg_visual_mode=seg_visual_mode,
            diagnostic_session_context=diagnostic_session_context,
        )
        subtype_sufficiency = normalize_subtype_sufficiency_payload(subtype_sufficiency)
        subtype_sufficiency = apply_known_site_lineage_guard_to_subtype_sufficiency(
            subtype_sufficiency,
            site_prediction,
        )
        knowledge_context = _enrich_knowledge_context(
            knowledge_context,
            broad_hypotheses=broad_hypotheses,
            subtype_sufficiency=subtype_sufficiency,
        )
        diagnosis_prediction = predict_diagnosis(
            selected_tumor,
            site_prediction=site_prediction,
            segmentation_context=segmentation_context,
            morphology_context=medgemma_morphology_context,
            qwen_client=stages_client,
            dry_run=dry_run,
            seg_visual_mode=seg_visual_mode,
            knowledge_context=knowledge_context,
            diagnostic_session_context=diagnostic_session_context,
        )
        diagnosis_prediction = normalize_primary_diagnosis_payload(diagnosis_prediction)
        diagnosis_prediction = apply_indeterminate_potential_if_supported(diagnosis_prediction, subtype_sufficiency)
        diagnosis_prediction = apply_known_site_lineage_guard(
            diagnosis_prediction,
            subtype_sufficiency,
            site_prediction,
        )
        diagnosis_prediction = apply_indeterminate_potential_if_supported(diagnosis_prediction, subtype_sufficiency)
        # Phase 3: only re-search KEEP if hypothesis_set changed
        prev_hypotheses = (knowledge_context.get("per_hypothesis") if isinstance(knowledge_context, dict) else None) or []
        new_hypothesis_set = diagnosis_prediction.get("hypothesis_set") or []
        if _hypothesis_set_changed(new_hypothesis_set, prev_hypotheses):
            knowledge_context = retrieve_hypothesis_knowledge_context(
                hypothesis_set=new_hypothesis_set,
                site_prediction=site_prediction,
                knowledge_config=knowledge_config,
            )
            print(f"    [7/10] → KEEP refreshed (hypotheses changed)", flush=True)
        else:
            print(f"    [7/10] → KEEP unchanged (hypotheses stable)", flush=True)
        knowledge_context = _enrich_knowledge_context(
            knowledge_context,
            broad_hypotheses=broad_hypotheses,
            subtype_sufficiency=subtype_sufficiency,
        )
        print(f"    [7/10] → KEEP updated context: {_keep_context_summary(knowledge_context)}", flush=True)
        sufficiency = assess_evidence_sufficiency(
            _dedupe_selected(site_evidence + selected_tumor),
            site_prediction=site_prediction,
            diagnosis_prediction=diagnosis_prediction,
            segmentation_context=segmentation_context,
            knowledge_context=knowledge_context,
            morphology_context=medgemma_morphology_context,
            qwen_client=stages_client,
            dry_run=dry_run,
            diagnostic_session_context=diagnostic_session_context,
        )
        sufficiency = normalize_evidence_gaps(sufficiency, diagnosis_prediction)
        ri["post_reinspection"] = {
            "selected_roi_count": len(selected_tumor),
            "medgemma_round": _strip_heavy(medgemma_round),
            "medgemma_round_count": medgemma_morphology_context.get("round_count"),
            "broad_hypotheses": _strip_heavy(broad_hypotheses),
            "subtype_sufficiency": _strip_heavy(subtype_sufficiency),
            "diagnosis_prediction": _strip_heavy(diagnosis_prediction),
            "evidence_sufficiency": _strip_heavy(sufficiency),
        }
        reinspection_rounds.append(ri)
        reinspection = ri
        print(
            f"    [7/10] → round {reinspect_iter}: "
            f"sufficient={sufficiency.get('sufficient_for_diagnosis')}, "
            f"image_gaps={len(sufficiency.get('missing_image_evidence') or [])}, "
            f"nonimage_gaps={len(sufficiency.get('missing_nonimage_metadata') or [])}, "
            f"interim_hypotheses={_diagnosis_hypothesis_summary(diagnosis_prediction)}",
            flush=True,
        )
        if (not sufficiency.get("sufficient_for_diagnosis")) and _nonimage_dominant_gaps(sufficiency):
            nonimage_dominant_rounds += 1
        else:
            nonimage_dominant_rounds = 0
        diagnostic_session_context = update_diagnostic_session_context(
            diagnostic_session_context,
            f"07_reinspection_round_{reinspect_iter}",
            {
                "reinspection_round": ri,
                "medgemma_morphology_context": medgemma_morphology_context,
                "broad_hypotheses": broad_hypotheses,
                "subtype_sufficiency": subtype_sufficiency,
                "diagnosis_prediction": diagnosis_prediction,
                "knowledge_context": knowledge_context,
                "evidence_sufficiency": sufficiency,
            },
            len(selected_tumor),
        )
        write_progress_checkpoint(out_dir, f"07_reinspection_round_{reinspect_iter}", {
            "reinspection_round": ri,
            "selected_rois": selected_tumor,
            "segmentation_context": segmentation_context,
            "medgemma_morphology_context": medgemma_morphology_context,
            "broad_hypotheses": broad_hypotheses,
            "subtype_sufficiency": subtype_sufficiency,
            "diagnosis_prediction": diagnosis_prediction,
            "knowledge_context": knowledge_context,
            "evidence_sufficiency": sufficiency,
            "nonimage_dominant_rounds": nonimage_dominant_rounds,
            "diagnostic_session_context": diagnostic_session_context,
        })
        if nonimage_dominant_rounds >= 2:
            reinspection_stop_reason = (
                "stopped after two consecutive reinspection rounds where remaining gaps "
                "were dominated by non-image metadata (IHC, margins, stage, gross or molecular data)"
            )
            sufficiency["has_image_actionable_gap"] = False
            sufficiency["recommended_next_step"] = "generate_report_with_limitations"
            sufficiency["reinspection_stop_reason"] = reinspection_stop_reason
            print(f"    [7/10] → stop: {reinspection_stop_reason}", flush=True)
            break

    if reinspect_iter == 0:
        if sufficiency.get("sufficient_for_diagnosis"):
            reason = "diagnosis evidence already sufficient"
        elif not sufficiency.get("has_image_actionable_gap"):
            reason = "missing evidence is nonimage metadata or not actionable by ROI search"
        elif max_reinspection_rois <= 0:
            reason = "reinspection disabled"
        elif reinspection_stop_reason:
            reason = reinspection_stop_reason
        else:
            reason = "no extra ROI candidates available"
        reinspection = {
            "extra_rois": [],
            "plan": reason,
            "enabled": False,
            "stop_reason": reason,
            "candidate_count": len(candidates),
            "selected_roi_count": len(selected_tumor),
            "used_candidate_id_count": len(used_candidate_ids),
            "remaining_candidate_count": last_remaining_count,
            "last_planned_reinspection": _strip_heavy(last_planned_reinspection or {}),
            "missing_image_evidence": sufficiency.get("missing_image_evidence") or [],
            "missing_nonimage_metadata": sufficiency.get("missing_nonimage_metadata") or [],
        }
        print(f"    [7/10] → no extra ROIs added: {reason}", flush=True)
    else:
        last_reinspection = dict(reinspection)
        reinspection = {
            **last_reinspection,
            "total_rounds": reinspect_iter,
            "rounds": [dict(row) for row in reinspection_rounds],
        }
        print(f"    [7/10] → {reinspect_iter} round(s), {len(selected_tumor)} ROIs total, sufficient={sufficiency.get('sufficient_for_diagnosis')}", flush=True)

    # ── Final diagnosis with ALL collected ROIs ──
    print(f"    [8/10] Final diagnosis (all {len(selected_tumor)} ROIs) ...", flush=True)
    broad_hypotheses = predict_broad_hypotheses(
        selected_rois=selected_tumor,
        site_prediction=site_prediction,
        segmentation_context=segmentation_context,
        morphology_context=medgemma_morphology_context,
        knowledge_context=knowledge_context,
        qwen_client=stages_client,
        dry_run=dry_run,
        seg_visual_mode=seg_visual_mode,
        diagnostic_session_context=diagnostic_session_context,
    )
    subtype_sufficiency = judge_subtype_sufficiency(
        selected_rois=selected_tumor,
        site_prediction=site_prediction,
        broad_hypotheses=broad_hypotheses,
        knowledge_context=knowledge_context,
        segmentation_context=segmentation_context,
        morphology_context=medgemma_morphology_context,
        qwen_client=stages_client,
        dry_run=dry_run,
        seg_visual_mode=seg_visual_mode,
        diagnostic_session_context=diagnostic_session_context,
    )
    subtype_sufficiency = normalize_subtype_sufficiency_payload(subtype_sufficiency)
    subtype_sufficiency = apply_known_site_lineage_guard_to_subtype_sufficiency(
        subtype_sufficiency,
        site_prediction,
    )
    knowledge_context = _enrich_knowledge_context(
        knowledge_context,
        broad_hypotheses=broad_hypotheses,
        subtype_sufficiency=subtype_sufficiency,
    )
    diagnosis_prediction = predict_diagnosis(
        selected_tumor,
        site_prediction=site_prediction,
        segmentation_context=segmentation_context,
        morphology_context=medgemma_morphology_context,
        qwen_client=stages_client,
        dry_run=dry_run,
        seg_visual_mode=seg_visual_mode,
        knowledge_context=knowledge_context,
        diagnostic_session_context=diagnostic_session_context,
    )
    diagnosis_prediction = normalize_primary_diagnosis_payload(diagnosis_prediction)
    diagnosis_prediction = apply_indeterminate_potential_if_supported(diagnosis_prediction, subtype_sufficiency)
    diagnosis_prediction = apply_known_site_lineage_guard(
        diagnosis_prediction,
        subtype_sufficiency,
        site_prediction,
    )
    diagnosis_prediction = apply_indeterminate_potential_if_supported(diagnosis_prediction, subtype_sufficiency)
    # Phase 3: only re-search KEEP if hypothesis_set changed
    prev_hypotheses = (knowledge_context.get("per_hypothesis") if isinstance(knowledge_context, dict) else None) or []
    new_hypothesis_set = diagnosis_prediction.get("hypothesis_set") or []
    if _hypothesis_set_changed(new_hypothesis_set, prev_hypotheses):
        knowledge_context = retrieve_hypothesis_knowledge_context(
            hypothesis_set=new_hypothesis_set,
            site_prediction=site_prediction,
            knowledge_config=knowledge_config,
        )
        print(f"    [8/10] → KEEP refreshed (hypotheses changed)", flush=True)
    else:
        print(f"    [8/10] → KEEP unchanged (hypotheses stable)", flush=True)
    knowledge_context = _enrich_knowledge_context(
        knowledge_context,
        broad_hypotheses=broad_hypotheses,
        subtype_sufficiency=subtype_sufficiency,
    )
    print(f"    [8/10] → KEEP final context: {_keep_context_summary(knowledge_context)}", flush=True)
    print(
        f"    [8/10] → broad={broad_hypotheses.get('best_broad_type') or '?'}, "
        f"subtype_level={subtype_sufficiency.get('best_supported_level') or '?'}, "
        f"final_hypotheses: {_diagnosis_hypothesis_summary(diagnosis_prediction)}",
        flush=True,
    )
    diagnostic_session_context = update_diagnostic_session_context(
        diagnostic_session_context,
        "08_final_diagnosis",
        {
            "broad_hypotheses": broad_hypotheses,
            "subtype_sufficiency": subtype_sufficiency,
            "diagnosis_prediction": diagnosis_prediction,
            "knowledge_context": knowledge_context,
            "evidence_sufficiency": sufficiency,
            "reinspection_plan": reinspection,
        },
        len(selected_tumor),
    )
    write_progress_checkpoint(out_dir, "08_final_diagnosis", {
        "selected_rois": selected_tumor,
        "roi_selection": roi_selection,
        "site_prediction": site_prediction,
        "broad_hypotheses": broad_hypotheses,
        "subtype_sufficiency": subtype_sufficiency,
        "diagnosis_prediction": diagnosis_prediction,
        "segmentation_context": segmentation_context,
        "knowledge_context": knowledge_context,
        "medgemma_morphology_context": medgemma_morphology_context,
        "evidence_sufficiency": sufficiency,
        "reinspection_plan": reinspection,
        "diagnostic_session_context": diagnostic_session_context,
    })

    visual_rois = _ensure_site_visual_roi(_dedupe_selected(site_evidence + selected_tumor), site_evidence)
    print(f"    [9/10] Diagnosis slots + claim verification ...", flush=True)
    graph = build_case_evidence_graph(
        case_id=case_id,
        slides_used=slides_used,
        candidate_rois=candidates,
        selected_rois=selected_tumor,
        visual_rois=visual_rois,
        segmentation_context=segmentation_context,
        site_prediction=site_prediction,
        broad_hypotheses=broad_hypotheses,
        subtype_sufficiency=subtype_sufficiency,
        diagnosis_prediction=diagnosis_prediction,
        sufficiency=sufficiency,
        knowledge_context=knowledge_context,
    )
    diagnosis_slots = plan_diagnosis_slots(
        selected_rois=selected_tumor,
        evidence_graph=graph,
        site_prediction=site_prediction,
        diagnosis_prediction=diagnosis_prediction,
        segmentation_context=segmentation_context,
        qwen_client=stages_client,
        dry_run=dry_run,
        diagnostic_session_context=diagnostic_session_context,
    )
    if skip_claim_verification:
        claim_verification = {"verification_version": "skipped", "claims": []}
        filtered_slots = diagnosis_slots
        print(f"    [9/10] → claim verification SKIPPED", flush=True)
    else:
        claims = build_claims_from_slots(diagnosis_slots)
        claim_verification = verify_claims(
            claims=claims,
            evidence_graph=graph,
            segmentation_context=segmentation_context,
            selected_rois=selected_tumor,
            qwen_client=stages_client,
            dry_run=dry_run,
            diagnostic_session_context=diagnostic_session_context,
        )
        filtered_slots = filter_slots_by_verified_claims(diagnosis_slots, claim_verification)
        n_supported = sum(1 for c in claim_verification.get("claims", []) if c.get("status") == "supported")
        print(f"    [9/10] → {n_supported} claims supported", flush=True)
    diagnostic_session_context = update_diagnostic_session_context(
        diagnostic_session_context,
        "09_slots_claim_verification",
        {
            "diagnosis_slots": diagnosis_slots,
            "filtered_diagnosis_slots": filtered_slots,
            "claim_verification": claim_verification,
        },
        len(selected_tumor),
    )
    write_progress_checkpoint(out_dir, "09_slots_claim_verification", {
        "evidence_graph": graph,
        "diagnosis_slots": diagnosis_slots,
        "filtered_diagnosis_slots": filtered_slots,
        "claim_verification": claim_verification,
        "selected_rois": selected_tumor,
        "site_prediction": site_prediction,
        "broad_hypotheses": broad_hypotheses,
        "subtype_sufficiency": subtype_sufficiency,
        "diagnosis_prediction": diagnosis_prediction,
        "segmentation_context": segmentation_context,
        "knowledge_context": knowledge_context,
        "medgemma_morphology_context": medgemma_morphology_context,
        "evidence_sufficiency": sufficiency,
        "reinspection_plan": reinspection,
        "roi_selection": roi_selection,
        "diagnostic_session_context": diagnostic_session_context,
    })

    print(f"    [10/10] Generating structured report ...", flush=True)
    report_payload = generate_claim_verified_report(
        selected_rois=selected_tumor,
        visual_rois=visual_rois,
        site_prediction=site_prediction,
        diagnosis_prediction=diagnosis_prediction,
        sufficiency=sufficiency,
        diagnosis_slots=filtered_slots,
        claim_verification=claim_verification,
        segmentation_context=segmentation_context,
        knowledge_context=knowledge_context,
        short_report_client=short_report_client,
        long_report_client=long_report_client,
        dry_run=dry_run,
        enforce_staged_diagnosis=enforce_staged_diagnosis,
        medgemma_context=medgemma_morphology_context,
        diagnostic_session_context=diagnostic_session_context,
    )

    payload = {
        "case_id": case_id,
        "mode": "case_evidence_graph_agent_v1",
        "candidate_count": len(candidates),
        "selected_count": len(selected_tumor),
        "max_zoom_depth": max_zoom_depth,
        "max_reinspection_rounds": max_reinspection_rounds,
        "slides_used": slides_used,
        "candidate_rois": candidates,
        "site_rois": site_evidence,
        "tumor_rois": selected_tumor,
        "selected_rois": selected_tumor,
        "visual_rois": visual_rois,
        "roi_selection": roi_selection,
        "site_prediction": site_prediction,
        "broad_hypotheses": broad_hypotheses,
        "subtype_sufficiency": subtype_sufficiency,
        "diagnosis_prediction": diagnosis_prediction,
        "segmentation_context": segmentation_context,
        "knowledge_context": knowledge_context,
        "medgemma_morphology_context": medgemma_morphology_context,
        "evidence_sufficiency": sufficiency,
        "reinspection_plan": reinspection,
        "evidence_graph": graph,
        "diagnosis_slots": diagnosis_slots,
        "filtered_diagnosis_slots": filtered_slots,
        "claim_verification": claim_verification,
        "generated_long_report": report_payload["long_report"],
        "generated_long_report_rich": report_payload.get("long_report_rich", report_payload["long_report"]),
        "medgemma_short_report": report_payload.get("short_report"),
        "figures": report_payload.get("figures") or [],
        "raw_generation": report_payload.get("raw"),
        "diagnostic_session_context": diagnostic_session_context,
    }
    _save_figures(payload["figures"], out_dir)
    save_evidence_graph_outputs(payload, out_dir)
    return payload


def _patch_zoom_seg_from_parents(
    selected_rois: list[dict],
    prev_seg_ctx: dict,
    out_dir: Path | None = None,
) -> None:
    """Crop parent seg masks for zoom ROIs, attaching overlay/composite paths.

    Cache is per-run under out_dir/zoom_seg_cache (not shared_artifacts)
    because zoom ROI coordinates differ between runs and the operation is
    a fast crop with no model inference.
    """
    zoom_rois = [r for r in selected_rois if r.get("zoom_parent_roi_id")]
    if not zoom_rois:
        return
    prev_results = prev_seg_ctx.get("roi_results") or []
    parent_by_id: dict[str, dict] = {}
    for pr in prev_results:
        parent_by_id[str(pr.get("roi_id") or "")] = pr
        if pr.get("image_path"):
            parent_by_id[str(pr.get("image_path"))] = pr
        if pr.get("roi_index") is not None:
            parent_by_id[str(pr.get("roi_index"))] = pr
    selected_parent_by_id: dict[str, dict] = {}
    for row in selected_rois:
        if row.get("zoom_parent_roi_id"):
            continue
        selected_parent_by_id[str(row.get("roi_id") or "")] = row
        if row.get("image_path"):
            selected_parent_by_id[str(row.get("image_path"))] = row
        if row.get("roi_index") is not None:
            selected_parent_by_id[str(row.get("roi_index"))] = row
    try:
        from pathxai_agent.segmentation.seg_visual import crop_seg_for_zoom
    except Exception:
        return
    for zr in zoom_rois:
        pid = str(zr.get("zoom_parent_roi_id") or "")
        parent = parent_by_id.get(pid)
        if not parent:
            zr["zoom_seg_error"] = f"parent segmentation not found for parent id {pid}"
            continue
        parent_roi = selected_parent_by_id.get(pid) or {}
        mask = parent.get("mask_path_file") or parent.get("mask_path")
        he = parent.get("image_path")
        if not mask or not he:
            zr["zoom_seg_error"] = "parent segmentation is missing mask_path or image_path"
            continue
        try:
            parent_x = int(parent_roi.get("x") if parent_roi.get("x") is not None else parent.get("x") or 0)
            parent_y = int(parent_roi.get("y") if parent_roi.get("y") is not None else parent.get("y") or 0)
            local_x = max(0, int(zr.get("x") or 0) - parent_x)
            local_y = max(0, int(zr.get("y") or 0) - parent_y)
            result = crop_seg_for_zoom(
                he, mask,
                zoom_x=local_x,
                zoom_y=local_y,
                zoom_w=int(zr.get("width") or 512),
                zoom_h=int(zr.get("height") or 512),
                cache_root=(out_dir / "zoom_seg_cache") if out_dir else Path("/tmp/pathxai_zoom_seg"),
            )
            zr["overlay_path"] = result.get("overlay_path")
            zr["composite_path"] = result.get("composite_path")
            zr["mask_path"] = mask  # reuse parent mask
            zr["zoom_seg_local_xy"] = [local_x, local_y]
            zr.pop("zoom_seg_error", None)
        except Exception as exc:
            zr["zoom_seg_error"] = f"{type(exc).__name__}: {exc}"


def _merge_zoom_seg_into_context(
    seg_ctx: dict,
    prev_seg_ctx: dict,
    selected_rois: list[dict],
) -> None:
    """Append zoom ROI seg entries to seg_ctx.roi_results."""
    zoom_rois = [r for r in selected_rois if r.get("zoom_parent_roi_id")]
    if not zoom_rois:
        return
    roi_results = seg_ctx.get("roi_results") or []
    max_idx = max((int(r.get("roi_index") or 0) for r in roi_results), default=0)
    for zr in zoom_rois:
        max_idx += 1
        ov = zr.get("overlay_path")
        roi_results.append({
            "roi_index": max_idx,
            "roi_id": zr.get("roi_id"),
            "image_path": zr.get("image_path"),
            "overlay_path": ov,
            "overlay_path_file": ov,
            "mask_path": zr.get("mask_path"),
            "composite_path": zr.get("composite_path"),
            "width": zr.get("width"),
            "height": zr.get("height"),
            "stats": {},  # zoom inherits parent stats conceptually
            "cache_key": f"zoom_{zr.get('roi_id')}",
            "cache_hit": False,
            "zoom_seg_local_xy": zr.get("zoom_seg_local_xy"),
            "zoom_seg_error": zr.get("zoom_seg_error"),
        })
    seg_ctx["roi_results"] = roi_results


def _candidate_rows(rois: list[dict]) -> list[dict]:
    rows = []
    for idx, row in enumerate(rois, start=1):
        out = dict(row)
        out.setdefault("roi_index", idx)
        out.setdefault("roi_id", row.get("roi_id") or f"roi_{idx:03d}")
        out.setdefault("evidence_role", row.get("evidence_role") or row.get("role") or "tumor_candidate")
        rows.append(out)
    return rows


def select_initial_evidence_rois(
    candidates: list[dict],
    site_evidence: list[dict] | None = None,
    max_selected_rois: int = 5,
    max_per_bucket: int = 2,
) -> list[dict]:
    """Seed ROI evidence without committing to a single preliminary diagnosis."""
    selected: list[dict] = []
    seen: set[str] = set()
    bucket_counts: dict[str, int] = {}

    def add(row: dict, bucket: str) -> None:
        if len(selected) >= max_selected_rois:
            return
        if bucket_counts.get(bucket, 0) >= max_per_bucket:
            return
        key = _roi_key(row)
        if key in seen:
            return
        out = dict(row)
        out["evidence_bucket"] = bucket
        selected.append(out)
        seen.add(key)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    # Highest tumor/diagnostic score, capped to avoid confirmation bias.
    for row in candidates[: max_per_bucket * 2]:
        add(row, "tumor_core")

    # Optional site/support evidence, also capped.
    for row in site_evidence or []:
        add(row, "site_support")

    # Spatially distinct representative evidence.
    for row in _spatially_diverse_rows(candidates, existing=selected):
        add(row, "representative_diverse")
        if len(selected) >= max_selected_rois:
            break

    # A lower-ranked alternative morphology ROI can prevent overfitting to the
    # most malignant-looking top tiles.
    tail_start = max(0, len(candidates) // 2)
    for row in candidates[tail_start:]:
        add(row, "alternative_morphology")
        if len(selected) >= max_selected_rois:
            break

    for row in candidates:
        add(row, "fallback_ranked")
        if len(selected) >= max_selected_rois:
            break
    return selected[:max_selected_rois]


def select_precompute_evidence_rois(
    candidates: list[dict],
    site_evidence: list[dict] | None = None,
    selected_rois: int = 5,
    max_reinspection_rois: int = 2,
) -> list[dict]:
    """Select ROIs whose segmentation cache should be precomputed."""
    seed = select_initial_evidence_rois(
        candidates,
        site_evidence=site_evidence,
        max_selected_rois=selected_rois,
        max_per_bucket=2,
    )
    out = list(seed)
    seen = {_roi_key(row) for row in out}
    target = selected_rois + max_reinspection_rois
    for row in _spatially_diverse_rows(candidates, existing=out):
        if len(out) >= target:
            break
        key = _roi_key(row)
        if key in seen:
            continue
        extra = dict(row)
        extra["evidence_bucket"] = "precompute_reinspection_pool"
        out.append(extra)
        seen.add(key)
    for row in candidates:
        if len(out) >= target:
            break
        key = _roi_key(row)
        if key in seen:
            continue
        extra = dict(row)
        extra["evidence_bucket"] = "precompute_ranked_pool"
        out.append(extra)
        seen.add(key)
    return out


def _roi_key(row: dict) -> str:
    return str(row.get("roi_id") or row.get("image_path") or row.get("roi_index") or id(row))


def _roi_center(row: dict) -> tuple[str, float, float] | None:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    slide = str(meta.get("slide_path") or row.get("slide_path") or row.get("slide_id") or "")
    x = row.get("x")
    y = row.get("y")
    w = row.get("width")
    h = row.get("height")
    if x is None or y is None:
        m = re.search(r"_x(\d+)_y(\d+)_l\d+_w(\d+)_h(\d+)", str(row.get("roi_id") or row.get("image_path") or ""))
        if m:
            x, y, w, h = [int(v) for v in m.groups()]
    try:
        return slide, float(x) + float(w or 0) / 2.0, float(y) + float(h or 0) / 2.0
    except Exception:
        return None


def _spatially_diverse_rows(candidates: list[dict], existing: list[dict]) -> list[dict]:
    existing_centers = [c for c in (_roi_center(row) for row in existing) if c is not None]
    scored = []
    for rank, row in enumerate(candidates):
        center = _roi_center(row)
        if center is None:
            distance = float(rank)
        else:
            slide, cx, cy = center
            same_slide = [(ex, ey) for s, ex, ey in existing_centers if s == slide]
            distance = min(((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5 for ex, ey in same_slide) if same_slide else 1e9
        score = float(row.get("score") or row.get("tumor_rank_score") or 0.0)
        scored.append((distance, score, -rank, row))
    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [row for _, _, _, row in scored]


def build_case_evidence_graph(
    case_id: str,
    slides_used: list[dict],
    candidate_rois: list[dict],
    selected_rois: list[dict],
    visual_rois: list[dict],
    segmentation_context: dict,
    site_prediction: dict,
    broad_hypotheses: dict | None,
    subtype_sufficiency: dict | None,
    diagnosis_prediction: dict,
    sufficiency: dict,
    knowledge_context: dict,
) -> dict:
    seg_by_roi = {
        str(row.get("roi_id") or row.get("image_path")): row
        for row in (segmentation_context or {}).get("results", [])
        if isinstance(row, dict)
    }
    slide_nodes = []
    for slide in slides_used or []:
        slide_nodes.append({
            "node_id": f"slide::{slide.get('slide_id')}",
            "type": "slide",
            "slide_index": slide.get("slide_index"),
            "slide_id": slide.get("slide_id"),
            "slide_path": slide.get("slide_path"),
            "roi_count": slide.get("roi_count"),
            "foundation_model": slide.get("foundation_model"),
        })

    roi_nodes = []
    for idx, row in enumerate(selected_rois, start=1):
        meta = row.get("metadata") or {}
        roi_id = str(row.get("roi_id") or f"roi_{idx:03d}")
        seg = seg_by_roi.get(roi_id) or seg_by_roi.get(str(row.get("image_path"))) or {}
        roi_nodes.append({
            "node_id": f"roi::{roi_id}",
            "type": "roi",
            "roi_id": roi_id,
            "role": row.get("evidence_role") or "tumor_roi",
            "image_path": row.get("image_path"),
            "score": row.get("score"),
            "slide_id": Path(str(meta.get("slide_path") or "")).stem,
            "slide_path": meta.get("slide_path"),
            "segmentation": _compact_segmentation(seg),
        })

    evidence_nodes = [
        {"node_id": "evidence::site", "type": "site_prediction", "payload": _strip_heavy(site_prediction)},
        {"node_id": "evidence::broad_hypotheses", "type": "broad_hypotheses", "payload": _strip_heavy(broad_hypotheses or {})},
        {"node_id": "evidence::subtype_sufficiency", "type": "subtype_sufficiency", "payload": _strip_heavy(subtype_sufficiency or {})},
        {"node_id": "evidence::diagnosis", "type": "diagnosis_prediction", "payload": _strip_heavy(diagnosis_prediction)},
        {"node_id": "evidence::sufficiency", "type": "evidence_sufficiency", "payload": _strip_heavy(sufficiency)},
        {"node_id": "evidence::knowledge", "type": "knowledge_context", "payload": _strip_heavy(knowledge_context)},
    ]
    edges = []
    for roi in roi_nodes:
        if roi.get("slide_id"):
            edges.append({"source": f"slide::{roi['slide_id']}", "target": roi["node_id"], "relation": "contains_roi"})
        edges.append({"source": roi["node_id"], "target": "evidence::diagnosis", "relation": "supports"})
    return {
        "graph_version": "case_evidence_graph_v1",
        "case_node": {"node_id": f"case::{case_id}", "type": "case", "case_id": case_id},
        "slide_nodes": slide_nodes,
        "roi_nodes": roi_nodes,
        "evidence_nodes": evidence_nodes,
        "edges": edges,
        "candidate_count": len(candidate_rois),
        "visual_roi_count": len(visual_rois),
    }


def predict_broad_hypotheses(
    selected_rois: list[dict],
    site_prediction: dict,
    segmentation_context: dict | None,
    morphology_context: dict | None,
    knowledge_context: dict | None,
    qwen_client: QwenVisionClient | None,
    dry_run: bool,
    seg_visual_mode: str = "text_only",
    diagnostic_session_context: dict | None = None,
) -> dict:
    if dry_run or qwen_client is None:
        return {"available": False, "hypotheses": [], "reason": "dry run"}
    prompt = f"""/no_think
Generate BROAD cancer-type / lineage hypotheses from the current case evidence.

This is not the final subtype diagnosis. Stay at broad level first:
- examples: renal epithelial neoplasm, adrenal cortical neoplasm, urothelial neoplasm,
  breast epithelial neoplasm, lymphoma/lymphoid neoplasm, glioma, squamous carcinoma.
- Do not choose a specific subtype unless it is only used as a low-level clue.
- Use known site metadata as context.
- Use KEEP only as background candidate knowledge, not patient-specific evidence.
- Treat MedGemma morphology suspected diagnoses as non-binding descriptive clues;
  do not copy them unless supported by ROI images, segmentation evidence, and site context.
- D1 segmentation classes are tissue-pattern evidence, not lineage labels:
  a high Inflammatory fraction can reflect lymphoid-rich stroma/inflammation and
  is not sufficient to diagnose lymphoma; a high Neoplastic fraction does not
  determine tumor lineage by itself.
- If a proposed broad lineage conflicts with a high-confidence known site,
  keep it as a differential unless multiple independent ROIs show direct
  lineage-defining morphology and the site-specific primary lineage is excluded.
- Do not use filenames, case IDs, TCGA codes, directory names, or coordinates.

Known site / prior:
{json.dumps(_strip_heavy(site_prediction), indent=2, ensure_ascii=False)}

KEEP site subtype candidate context:
{json.dumps(_strip_heavy(knowledge_context or {"available": False}), indent=2, ensure_ascii=False)}

Segmentation evidence:
{json.dumps(_strip_heavy(segmentation_context or {"available": False}), indent=2, ensure_ascii=False)}

MedGemma morphology evidence:
{json.dumps(_strip_heavy(morphology_context or {"available": False}), indent=2, ensure_ascii=False)}

Case diagnostic session memory:
{json.dumps(_strip_heavy(diagnostic_session_context or {"available": False}), indent=2, ensure_ascii=False)}

Selected ROI evidence:
{candidate_text(selected_rois)}

Return ONLY JSON:
{{
  "hypotheses": [
    {{"broad_type": "...", "confidence": "high/medium/low", "supporting_features": ["..."], "main_mimics": ["..."], "missing_discriminators": ["..."]}}
  ],
  "best_broad_type": "...",
  "confidence": "high/medium/low",
  "reason": "..."
}}"""
    try:
        from pathxai_agent.segmentation.seg_visual import build_image_paths_for_rois
        image_paths = build_image_paths_for_rois(selected_rois, segmentation_context or {}, seg_visual_mode)
    except Exception:
        image_paths = [row["image_path"] for row in selected_rois]
    raw = qwen_client.chat_json(prompt, image_paths=image_paths)
    data = raw.get("json") or {}
    if not isinstance(data.get("hypotheses"), list):
        data["hypotheses"] = []
    data["available"] = True
    data["raw"] = _raw_summary(raw)
    return data


def judge_subtype_sufficiency(
    selected_rois: list[dict],
    site_prediction: dict,
    broad_hypotheses: dict | None,
    knowledge_context: dict | None,
    segmentation_context: dict | None,
    morphology_context: dict | None,
    qwen_client: QwenVisionClient | None,
    dry_run: bool,
    seg_visual_mode: str = "text_only",
    diagnostic_session_context: dict | None = None,
) -> dict:
    if dry_run or qwen_client is None:
        return {"available": False, "subtype_assessments": [], "reason": "dry run"}
    subtype_candidates = _subtype_candidates_from_context(site_prediction, knowledge_context)
    prompt = f"""/no_think
Judge whether the current ROI evidence is sufficient to distinguish site-specific subtype candidates.

Rules:
- Evaluate the subtype candidates one by one.
- Use KEEP candidates as a candidate list only; do not treat them as patient-specific evidence.
- Treat MedGemma morphology suspected diagnoses as non-binding descriptive clues;
  evaluate subtype candidates using ROI morphology, segmentation evidence, and site context.
- D1 segmentation classes are tissue-pattern evidence, not lineage labels:
  Inflammatory does not equal lymphoma, and Neoplastic does not determine lineage.
- If the known site is high confidence, site-specific primary lineage candidates
  remain the default primary space. Non-site lineages may be listed as
  differential/secondary involvement but should not become recommended primary
  unless directly supported by strong multi-ROI lineage evidence.
- If broad lineage/site is supported but subtype discriminators are missing, recommend a broad/NOS/not-further-classifiable diagnosis.
- Distinguish image-actionable missing evidence from nonimage metadata.
- Do not use filenames, case IDs, TCGA codes, directory names, or coordinates.
- recommended_primary_diagnosis must contain only the diagnostic entity. Do not append phrases such as "not further classifiable from ROI evidence"; put that limitation in reason and subtype assessment fields.

Known site / prior:
{json.dumps(_strip_heavy(site_prediction), indent=2, ensure_ascii=False)}

Broad hypotheses:
{json.dumps(_strip_heavy(broad_hypotheses or {"available": False}), indent=2, ensure_ascii=False)}

Site-specific subtype candidates:
{json.dumps(subtype_candidates[:12], indent=2, ensure_ascii=False)}

Segmentation evidence:
{json.dumps(_strip_heavy(segmentation_context or {"available": False}), indent=2, ensure_ascii=False)}

MedGemma morphology evidence:
{json.dumps(_strip_heavy(morphology_context or {"available": False}), indent=2, ensure_ascii=False)}

Case diagnostic session memory:
{json.dumps(_strip_heavy(diagnostic_session_context or {"available": False}), indent=2, ensure_ascii=False)}

Selected ROI evidence:
{candidate_text(selected_rois)}

Return ONLY JSON:
{{
  "best_supported_level": "broad_type/subtype/not_assessable",
  "recommended_primary_diagnosis": "...",
  "confidence": "high/medium/low",
  "subtype_assessments": [
    {{
      "subtype": "...",
      "status": "supported/possible_but_insufficient/unsupported/not_assessable",
      "supporting_features": ["..."],
      "missing_discriminators": ["..."],
      "needed_image_evidence": ["..."],
      "needed_nonimage_metadata": ["..."]
    }}
  ],
  "image_actionable_gaps": ["..."],
  "nonimage_metadata_gaps": ["..."],
  "reason": "..."
}}"""
    try:
        from pathxai_agent.segmentation.seg_visual import build_image_paths_for_rois
        image_paths = build_image_paths_for_rois(selected_rois, segmentation_context or {}, seg_visual_mode)
    except Exception:
        image_paths = [row["image_path"] for row in selected_rois]
    raw = qwen_client.chat_json(prompt, image_paths=image_paths)
    data = raw.get("json") or {}
    data["available"] = True
    data["candidate_count"] = len(subtype_candidates)
    data["candidate_source"] = "known-site lineage candidates + KEEP site_subtype_candidates"
    data["raw"] = _raw_summary(raw)
    return data


def _subtype_candidates_from_context(site_prediction: dict, knowledge_context: dict | None) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()

    def add(name: str, source: str, **extra: Any) -> None:
        name = str(name or "").strip()
        if not name:
            return
        key = name.lower()
        if key in seen:
            return
        seen.add(key)
        row = {"name": name, "source": source}
        row.update({k: v for k, v in extra.items() if v is not None})
        candidates.append(row)

    prior = site_prediction.get("diagnosis_prior") or {}
    for name in prior.get("subtype_candidates") or []:
        add(str(name), "tcga_prior")
    if prior.get("not_further_classifiable"):
        add(str(prior.get("not_further_classifiable")), "tcga_prior_broad_fallback")

    if isinstance(knowledge_context, dict):
        for row in knowledge_context.get("site_subtype_candidates") or []:
            if isinstance(row, dict):
                add(
                    row.get("name") or "",
                    "keep_site_subtype",
                    count=row.get("count"),
                    label_id=row.get("label_id"),
                    tumor_type=row.get("tumor_type"),
                    cancer_type=row.get("cancer_type"),
                )
        for row in knowledge_context.get("disease_hits") or []:
            if isinstance(row, dict) and row.get("site_matched_terms"):
                add(row.get("name") or "", "keep_site_matched_disease", score=row.get("score"))
    return candidates


def _enrich_knowledge_context(
    knowledge_context: dict | None,
    broad_hypotheses: dict | None = None,
    subtype_sufficiency: dict | None = None,
) -> dict:
    out = dict(knowledge_context or {"available": False})
    if broad_hypotheses is not None:
        out["broad_hypotheses"] = _strip_heavy(broad_hypotheses)
    if subtype_sufficiency is not None:
        out["subtype_sufficiency"] = _strip_heavy(subtype_sufficiency)
    return out


def _hypothesis_set_changed(
    new_hypotheses: list[dict],
    prev_per_hypothesis: list[dict],
) -> bool:
    """Detect if the diagnostic hypothesis set has substantively changed.

    Returns True if a new entity appears (by diagnosis name) that wasn't
    in the previous per-hypothesis list.  This gates Phase 3 KEEP re-search.
    """
    if not new_hypotheses:
        return False
    if not prev_per_hypothesis:
        return True  # first time, always search
    prev_names = {str(h.get("diagnosis") or h.get("name") or "").lower().strip()
                  for h in prev_per_hypothesis}
    prev_names.discard("")
    new_names = {str(h.get("diagnosis") or h.get("name") or "").lower().strip()
                 for h in new_hypotheses}
    new_names.discard("")
    return bool(new_names - prev_names)


def plan_diagnosis_slots(
    selected_rois: list[dict],
    evidence_graph: dict,
    site_prediction: dict,
    diagnosis_prediction: dict,
    segmentation_context: dict,
    qwen_client: QwenVisionClient | None,
    dry_run: bool,
    diagnostic_session_context: dict | None = None,
) -> dict:
    if dry_run or qwen_client is None:
        return _fallback_slots(site_prediction, diagnosis_prediction)
    prompt = f"""/no_think
Create structured diagnosis slots from ROI evidence. Do not write a final report.

Rules:
- Use only visible ROI morphology, segmentation statistics, and provided known site if present.
- If information is not visible or unavailable, write "not assessable".
- Do not invent IHC/molecular results, stage, gross size, margin distance, or lymph-node counts. You may recommend ancillary studies (IHC panels, molecular tests) in recommended_ancillary_studies when confidence is not high.
- Do not use filenames, TCGA codes, coordinates, or directory names.
- primary_diagnosis must contain only the main diagnostic entity.
- Do not append uncertainty phrases such as "not further classifiable from ROI evidence" to primary_diagnosis; put that limitation in missing_evidence or recommended_ancillary_studies.
- Keep differential diagnoses only in differential_diagnosis; never concatenate them into primary_diagnosis.

Site prediction:
{json.dumps(_strip_heavy(site_prediction), indent=2, ensure_ascii=False)}

Diagnosis prediction:
{json.dumps(_strip_heavy(diagnosis_prediction), indent=2, ensure_ascii=False)}

Segmentation evidence:
{json.dumps(_strip_heavy(segmentation_context or {"available": False}), indent=2, ensure_ascii=False)}

Case diagnostic session memory:
{json.dumps(_strip_heavy(diagnostic_session_context or {"available": False}), indent=2, ensure_ascii=False)}

Selected ROI evidence:
{candidate_text(selected_rois)}

Return ONLY JSON with these keys:
{{
  "site": "...",
  "primary_diagnosis": "...",
  "differential_diagnosis": ["...", "..."],
  "tumor_architecture": ["...", "..."],
  "cytology": ["...", "..."],
  "necrosis": "present/absent/not assessable + evidence",
  "invasion": "present/absent/not assessable + evidence",
  "stroma_inflammation": ["...", "..."],
  "ihc_molecular": "not provided / not assessable",
  "confidence": "high/medium/low",
  "missing_evidence": ["...", "..."],
  "recommended_ancillary_studies": ["Order site- and differential-appropriate IHC/molecular tests to confirm lineage", "Consider proliferation marker if relevant", "..."]
}}"""
    raw = qwen_client.chat_json(prompt, image_paths=[row["image_path"] for row in selected_rois])
    data = raw.get("json") or {}
    for key in DIAGNOSIS_SLOT_KEYS:
        data.setdefault(key, [] if key in {"differential_diagnosis", "tumor_architecture", "cytology", "stroma_inflammation", "missing_evidence", "recommended_ancillary_studies"} else "not assessable")
    data["raw"] = _raw_summary(raw)
    return data


def build_claims_from_slots(slots: dict) -> list[dict]:
    claims = []
    claim_id = 1
    for field in ["site", "primary_diagnosis", "necrosis", "invasion", "ihc_molecular"]:
        text = str(slots.get(field) or "").strip()
        if text:
            claims.append({"claim_id": f"C{claim_id:03d}", "field": field, "claim": text})
            claim_id += 1
    for field in ["tumor_architecture", "cytology", "stroma_inflammation"]:
        values = slots.get(field) or []
        if isinstance(values, str):
            values = [values]
        for value in values:
            text = str(value or "").strip()
            if text:
                claims.append({"claim_id": f"C{claim_id:03d}", "field": field, "claim": text})
                claim_id += 1
    return claims


def verify_claims(
    claims: list[dict],
    evidence_graph: dict,
    segmentation_context: dict,
    selected_rois: list[dict],
    qwen_client: QwenVisionClient | None,
    dry_run: bool,
    diagnostic_session_context: dict | None = None,
) -> dict:
    heuristic = [_heuristic_verify_claim(claim, segmentation_context, selected_rois) for claim in claims]
    if dry_run or qwen_client is None:
        return {"verification_version": "claim_verifier_v1_heuristic", "claims": heuristic}

    prompt = f"""/no_think
Verify pathology claims against the provided evidence graph.

For each claim, assign one status:
- supported: directly supported by ROI visual evidence or segmentation stats.
- weak: plausible but not directly proven by the available evidence.
- unsupported: no evidence supports the claim.
- contradicted: evidence suggests the claim is false.
- not_assessable: cannot be assessed from H&E ROI evidence.

Do not reward claims based on filenames, case IDs, TCGA codes, or external prior knowledge.

Claims:
{json.dumps(claims, indent=2, ensure_ascii=False)}

Evidence graph:
{json.dumps(_scrub_llm_context(_strip_heavy(evidence_graph)), indent=2, ensure_ascii=False)}

Segmentation evidence:
{json.dumps(_strip_heavy(segmentation_context or {"available": False}), indent=2, ensure_ascii=False)}

Case diagnostic session memory:
{json.dumps(_strip_heavy(diagnostic_session_context or {"available": False}), indent=2, ensure_ascii=False)}

Return ONLY JSON:
{{
  "claims": [
    {{"claim_id": "C001", "status": "supported/weak/unsupported/contradicted/not_assessable", "evidence": ["roi_id or figure"], "reason": "..."}}
  ],
  "unsupported_summary": ["..."],
  "missing_evidence": ["..."]
}}"""
    raw = qwen_client.chat_json(prompt, image_paths=[row["image_path"] for row in selected_rois])
    data = raw.get("json") or {}
    llm_claims = data.get("claims") if isinstance(data.get("claims"), list) else []
    merged = _merge_verification(heuristic, llm_claims)
    return {
        "verification_version": "claim_verifier_v1_llm_plus_heuristic",
        "claims": merged,
        "unsupported_summary": data.get("unsupported_summary") or [],
        "missing_evidence": data.get("missing_evidence") or [],
        "raw": _raw_summary(raw),
    }


def safe_assess_segmentation_quality(
    segmentation_context: dict | None,
    qwen_client: QwenVisionClient | None,
    dry_run: bool = False,
    stage_label: str = "seg_qc",
) -> dict:
    """Run LLM segmentation QC without letting endpoint failures kill the case."""
    try:
        return assess_segmentation_quality(
            segmentation_context or {},
            qwen_client=qwen_client,
            dry_run=dry_run,
        )
    except Exception as exc:
        roi_results = (segmentation_context or {}).get("roi_results") or []
        indices = [str(r.get("roi_index", i + 1)) for i, r in enumerate(roi_results)]
        return {
            "available": False,
            "nonfatal": True,
            "stage": stage_label,
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "submitted_count": len(roi_results),
            "returned_count": 0,
            "missing_count": len(roi_results),
            "submitted_roi_indices": indices,
            "returned_roi_indices": [],
            "missing_roi_indices": indices,
            "note": "Segmentation QC is advisory; pipeline continued without LLM seg-quality audit.",
        }


def shared_artifacts_dir() -> Path:
    return Path(os.environ.get("PATHXAI_SHARED_ARTIFACTS", "/local/PathXAI_Agent/shared_artifacts"))


def _file_fingerprint(path: str | Path | None) -> dict:
    if not path:
        return {"path": ""}
    p = Path(path)
    info = {"path": str(p)}
    try:
        st = p.stat()
        info.update({"size": st.st_size, "mtime_ns": st.st_mtime_ns})
    except Exception:
        pass
    return info


def _stable_hash(data: Any) -> str:
    blob = json.dumps(_json_safe(data), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_llm_cache(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def write_llm_cache(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def seg_qc_cache_path(roi_result: dict, class_legend: str) -> Path:
    image_path = roi_result.get("image_path")
    overlay_path = roi_result.get("overlay_path_file") or roi_result.get("overlay_path")
    mask_path = roi_result.get("mask_path_file") or roi_result.get("mask_path")
    key = _stable_hash({
        "task": "segmentation_qc",
        "version": "seg_qc_pair_v2",
        "roi_id": roi_result.get("roi_id"),
        "roi_index": roi_result.get("roi_index"),
        "image": _file_fingerprint(image_path),
        "overlay": _file_fingerprint(overlay_path),
        "mask": _file_fingerprint(mask_path),
        "class_legend": class_legend,
    })
    return shared_artifacts_dir() / "llm_cache" / "seg_qc" / f"{key}.json"


def build_seg_qc_pair_image(roi_result: dict) -> str:
    image_path = roi_result.get("image_path")
    overlay_path = roi_result.get("overlay_path_file") or roi_result.get("overlay_path")
    primary = image_path if image_path and Path(image_path).is_file() else overlay_path
    overlay = overlay_path if overlay_path and Path(overlay_path).is_file() else image_path
    if not primary or not overlay:
        return ""
    key = _stable_hash({
        "task": "seg_qc_pair_image",
        "version": "seg_qc_pair_v2",
        "image": _file_fingerprint(primary),
        "overlay": _file_fingerprint(overlay),
    })
    out = shared_artifacts_dir() / "llm_cache" / "seg_qc_pairs" / f"{key}.jpg"
    if out.exists():
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image, ImageDraw, ImageFont

        max_side = int(os.environ.get("PATHXAI_SEG_QC_PAIR_MAX_EDGE", "768") or 768)
        left = Image.open(str(primary)).convert("RGB")
        right = Image.open(str(overlay)).convert("RGB")
        for img in (left, right):
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        h = max(left.height, right.height)
        label_h = 32
        legend_h = 28
        total_h = h + label_h + legend_h
        total_w = left.width + right.width
        canvas = Image.new("RGB", (total_w, total_h), "white")
        canvas.paste(left, (0, label_h))
        canvas.paste(right, (left.width, label_h))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        # Top labels
        draw.rectangle([0, 0, left.width, label_h], fill=(248, 250, 252))
        draw.rectangle([left.width, 0, total_w, label_h], fill=(239, 246, 255))
        draw.text((10, 9), "Original H&E ROI", fill=(15, 23, 42), font=font)
        draw.text((left.width + 10, 9), "D1 segmentation overlay", fill=(15, 23, 42), font=font)
        # Bottom legend
        _TISSUE_COLORS = {
            1: (228, 26, 28),   # Neoplastic
            2: (55, 126, 184),  # Inflammatory
            3: (255, 127, 0),   # Connective
            4: (77, 175, 74),   # Epithelial
            5: (152, 78, 163),  # Dead
            6: (166, 86, 40),   # Gland
        }
        _TISSUE_NAMES = {1: "Neoplastic", 2: "Inflammatory", 3: "Connective",
                         4: "Epithelial", 5: "Dead", 6: "Gland"}
        draw.rectangle([0, total_h - legend_h, total_w, total_h], fill=(248, 250, 252))
        draw.line([(0, total_h - legend_h), (total_w, total_h - legend_h)], fill=(220, 220, 220))
        x = 8
        for cid in [1, 2, 3, 4, 5, 6]:
            color = _TISSUE_COLORS.get(cid, (128, 128, 128))
            name = _TISSUE_NAMES.get(cid, f"C{cid}")
            draw.rectangle([x, total_h - legend_h + 8, x + 10, total_h - legend_h + 18], fill=color, outline=(180, 180, 180))
            draw.text((x + 13, total_h - legend_h + 6), name, fill=(51, 65, 85))
            x += 13 + len(name) * 6 + 10
        canvas.save(str(out), format="JPEG", quality=int(os.environ.get("PATHXAI_SEG_QC_PAIR_JPEG_QUALITY", "82") or 82), optimize=True)
        return str(out)
    except Exception:
        return str(overlay or primary)


def assess_segmentation_quality(
    segmentation_context: dict,
    qwen_client: QwenVisionClient | None,
    dry_run: bool = False,
) -> dict:
    """LLM visual QC: inspect original ROI + overlay pairs to flag poor segmentations.

    Returns a dict mapping roi_index -> {quality, reason}.
    quality is one of: good, questionable, reject.
    """
    if dry_run or qwen_client is None:
        return {"available": False, "note": "dry_run or no client", "per_roi": {}}

    roi_results = segmentation_context.get("roi_results") or []
    if not roi_results:
        return {"available": False, "note": "no roi results", "per_roi": {}}

    class_names = segmentation_context.get("class_names") or {}
    class_legend = ", ".join(f"{cid}={name}" for cid, name in sorted(class_names.items(), key=lambda x: int(x[0])))

    per_roi: dict[str, dict] = {}
    uncached: list[dict] = []
    qc_errors: dict[str, list[str]] = {}
    max_qc_images = int(os.environ.get("PATHXAI_SEG_QC_MAX_IMAGES", "6") or 6)
    for r in roi_results[:max_qc_images]:
        idx = r.get("roi_index", "?")
        idx_s = str(idx)
        cache_path = seg_qc_cache_path(r, class_legend)
        cached = read_llm_cache(cache_path)
        if cached and (cached.get("quality") in {"good", "questionable", "reject"}):
            row = dict(cached)
            row["cache_hit"] = True
            per_roi[idx_s] = row
            continue
        stats = r.get("stats") or {}
        named = stats.get("named_class_fractions") or {}
        top_classes = sorted(named.items(), key=lambda x: x[1].get("fraction", 0), reverse=True)[:3]
        top_desc = "; ".join(f"{name}: {info.get('fraction', 0)*100:.1f}%" for name, info in top_classes)
        uncached.append({
            "idx": idx_s,
            "row": r,
            "cache_path": cache_path,
            "top_desc": top_desc,
        })

    submitted_indices = [str(r.get("roi_index", i + 1)) for i, r in enumerate(roi_results[:max_qc_images])]
    # Check if we have images to send
    has_images = any(
        (item["row"].get("image_path") or item["row"].get("overlay_path_file") or item["row"].get("overlay_path"))
        for item in uncached
    )
    if uncached and not has_images:
        return {"available": False, "note": "no original/overlay images", "per_roi": per_roi}

    raw_summary = None
    if uncached:
        # One composite pair per ROI: [legend, pair_1, pair_2, ...]
        # Use same composite as diagnosis for consistency
        for item in uncached:
            idx_s = item["idx"]
            r = item["row"]
            he = r.get("image_path")
            ov = r.get("overlay_path_file") or r.get("overlay_path")
            if not he or not ov or not Path(he).exists() or not Path(ov).exists():
                qc_errors.setdefault(idx_s, []).append(
                    "missing original ROI image or segmentation overlay path"
                )
                continue
            try:
                from pathxai_agent.segmentation.seg_visual import build_roi_seg_composite, build_global_seg_legend
                legend = build_global_seg_legend()
                pair = build_roi_seg_composite(he, ov)
                images = []
                if legend and Path(legend).exists():
                    images.append(legend)
                images.append(pair["composite_path"])
            except Exception as exc:
                qc_errors.setdefault(idx_s, []).append(f"failed to build ROI/seg composite: {type(exc).__name__}: {exc}")
                continue

            prompt = f"""/no_think
You are auditing one ROI's tissue segmentation.

Image order: FIRST is a global tissue color legend (Neoplastic=red, Inflammatory=blue, Connective=orange, Epithelial=green, Dead=purple, Gland=brown). SECOND is a composite pair: left = original H&E ROI labeled "H&E ROI", right = segmentation overlay labeled "Segmentation".

Tissue class legend (text): {class_legend}
ROI {idx_s} stats: {item['top_desc']}

Assess whether the segmentation overlay matches the H&E morphology:
- good: boundaries align, class assignments make sense.
- questionable: some misclassifications but overall recognizable.
- reject: clearly wrong (wrong tissue type, random noise, empty/blank).

Return ONLY JSON:
{{"quality": "good/questionable/reject", "reason": "brief explanation"}}"""
            try:
                raw = qwen_client.chat_json(prompt, image_paths=images)
                data = raw.get("json") or {}
                qc = {
                    "quality": str(data.get("quality") or "questionable").lower(),
                    "reason": str(data.get("reason") or ""),
                    "cache_hit": False,
                    "cache_version": "seg_qc_pair_v2",
                }
                if qc["quality"] not in {"good", "questionable", "reject"}:
                    qc["quality"] = "questionable"
                per_roi[idx_s] = qc
                write_llm_cache(item["cache_path"], qc)
            except Exception as exc:
                qc_errors.setdefault(idx_s, []).append(f"initial QC LLM call failed: {type(exc).__name__}: {exc}")

        # Retry: same composite pair format, max PATHXAI_SEG_QC_MISSING_RETRIES attempts
        missing_retries = int(os.environ.get("PATHXAI_SEG_QC_MISSING_RETRIES", "2") or 2)
        for attempt in range(1, max(0, missing_retries) + 1):
            still_missing = [item for item in uncached if item["idx"] not in per_roi]
            if not still_missing:
                break
            for item in still_missing:
                idx_s = item["idx"]
                r = item["row"]
                he = r.get("image_path")
                ov = r.get("overlay_path_file") or r.get("overlay_path")
                if not he or not ov:
                    qc_errors.setdefault(idx_s, []).append(f"retry {attempt}: missing original ROI image or overlay path")
                    continue
                try:
                    from pathxai_agent.segmentation.seg_visual import build_roi_seg_composite, build_global_seg_legend
                    legend = build_global_seg_legend()
                    pair = build_roi_seg_composite(he, ov)
                    images = []
                    if legend and Path(legend).exists():
                        images.append(legend)
                    images.append(pair["composite_path"])
                except Exception as exc:
                    qc_errors.setdefault(idx_s, []).append(f"retry {attempt}: failed to build composite: {type(exc).__name__}: {exc}")
                    continue
                prompt = f"""/no_think
You are auditing one ROI's tissue segmentation (retry {attempt}).

Image order: FIRST = global tissue color legend, SECOND = composite pair (left "H&E ROI", right "Segmentation").
Tissue class legend: {class_legend}
ROI {idx_s} stats: {item['top_desc']}

Return ONLY JSON:
{{"quality": "good/questionable/reject", "reason": "brief explanation"}}"""
                try:
                    retry = qwen_client.chat_json(prompt, image_paths=images)
                    data = retry.get("json") or {}
                    qc = {
                        "quality": str(data.get("quality") or "questionable").lower(),
                        "reason": str(data.get("reason") or ""),
                        "cache_hit": False,
                        "cache_version": "seg_qc_pair_v2",
                        "retry_attempt": attempt,
                    }
                    if qc["quality"] not in {"good", "questionable", "reject"}:
                        qc["quality"] = "questionable"
                    per_roi[idx_s] = qc
                    write_llm_cache(item["cache_path"], qc)
                except Exception as exc:
                    qc_errors.setdefault(idx_s, []).append(f"retry {attempt}: QC LLM call failed: {type(exc).__name__}: {exc}")
        raw_summary = None

    returned_indices = {str(k) for k in per_roi.keys()}
    missing_indices = [idx for idx in submitted_indices if idx not in returned_indices]
    # Fallback: mark LLM-unreturned ROIs as "not_audited" instead of leaving gaps
    for idx in missing_indices:
        per_roi[idx] = {
            "quality": "not_audited",
            "reason": "; ".join(qc_errors.get(idx) or ["LLM did not return a quality assessment for this ROI after retries"]),
            "cache_hit": False,
            "cache_version": "seg_qc_pair_v2",
            "errors": qc_errors.get(idx) or [],
        }
    return {
        "available": True,
        "cache_version": "seg_qc_pair_v2",
        "submitted_roi_indices": submitted_indices,
        "returned_roi_indices": sorted(returned_indices),
        "missing_roi_indices": missing_indices,
        "missing_roi_errors": {idx: qc_errors.get(idx, []) for idx in missing_indices},
        "submitted_count": len(submitted_indices),
        "returned_count": len(returned_indices),
        "missing_count": len(missing_indices),
        "per_roi": per_roi,
        "overall_assessment": (
            "loaded from shared per-ROI QC cache"
            if not uncached else "newly audited uncached ROI pairs; cached entries reused where available"
        ),
        "raw": raw_summary,
    }


def _seg_qc_counts(seg_qc: dict, expected_count: int | None = None) -> dict:
    per_roi = seg_qc.get("per_roi") or {}
    submitted = int(seg_qc.get("submitted_count") or 0)
    if not submitted:
        submitted = len(seg_qc.get("submitted_roi_indices") or []) or int(expected_count or 0) or len(per_roi)
    returned = int(seg_qc.get("returned_count") or len(per_roi))
    missing_recorded = int(seg_qc.get("missing_count") or max(0, submitted - returned))
    good = sum(1 for v in per_roi.values() if v.get("quality") == "good")
    flagged = sum(1 for v in per_roi.values() if v.get("quality") in ("questionable", "reject"))
    unchecked = sum(1 for v in per_roi.values() if v.get("quality") in ("missing", "not_returned", "not_audited"))
    # Missing ROIs are represented twice in newer QC payloads: as
    # `missing_count` metadata and as per-ROI `not_audited` placeholders.
    # Report the actual missing count, not the sum of both views.
    missing = max(missing_recorded, unchecked, max(0, submitted - returned))
    return {
        "submitted": submitted,
        "returned": returned,
        "missing": missing,
        "good": good,
        "flagged": flagged,
    }


def normalize_evidence_gaps(sufficiency: dict, diagnosis_prediction: dict | None = None) -> dict:
    """Split missing evidence into WSI-actionable gaps and nonimage metadata gaps."""
    out = dict(sufficiency or {})
    image = [str(x) for x in (out.get("missing_image_evidence") or []) if str(x).strip()]
    nonimage = [str(x) for x in (out.get("missing_nonimage_metadata") or []) if str(x).strip()]
    all_gaps = []
    for source in (
        out.get("missing_evidence"),
        out.get("discriminative_questions"),
        (diagnosis_prediction or {}).get("discriminative_evidence_needed"),
        (diagnosis_prediction or {}).get("missing_information"),
    ):
        if isinstance(source, list):
            all_gaps.extend(str(x) for x in source if str(x).strip())
        elif source:
            all_gaps.append(str(source))

    seen_image = {x.lower() for x in image}
    seen_nonimage = {x.lower() for x in nonimage}
    ambiguous = []
    for gap in all_gaps:
        low = gap.lower()
        is_nonimage = any(term in low for term in NONIMAGE_METADATA_TERMS)
        is_image = any(term in low for term in IMAGE_ACTIONABLE_TERMS)
        is_nonimage_dominant = any(term in low for term in NONIMAGE_DOMINANT_TERMS)
        is_strong_image = any(term in low for term in STRONG_IMAGE_ACTIONABLE_TERMS)
        if is_nonimage_dominant:
            if low not in seen_nonimage:
                nonimage.append(gap)
                seen_nonimage.add(low)
        elif is_nonimage and not is_image:
            if low not in seen_nonimage:
                nonimage.append(gap)
                seen_nonimage.add(low)
        elif is_strong_image or is_image:
            if low not in seen_image:
                image.append(gap)
                seen_image.add(low)
        else:
            ambiguous.append(gap)

    out["missing_image_evidence"] = image
    out["missing_nonimage_metadata"] = nonimage
    out["ambiguous_missing_evidence"] = ambiguous
    out["has_image_actionable_gap"] = bool(image)
    if not image and out.get("recommended_next_step") == "select_more_rois":
        out["recommended_next_step"] = "require_nonimage_metadata" if nonimage else "generate_report_with_limitations"
    if image and out.get("recommended_next_step") == "require_nonimage_metadata":
        out["recommended_next_step"] = "select_more_rois"
    out["gap_normalization_version"] = "image_actionable_v2_nonimage_dominant"
    return out


def _nonimage_dominant_gaps(sufficiency: dict) -> bool:
    """Return True when more ROI search is unlikely to address remaining gaps."""
    image = [str(x) for x in (sufficiency.get("missing_image_evidence") or []) if str(x).strip()]
    nonimage = [str(x) for x in (sufficiency.get("missing_nonimage_metadata") or []) if str(x).strip()]
    if not nonimage:
        return False
    if len(nonimage) >= max(4, len(image) * 2):
        return True
    nonimage_text = " ".join(x.lower() for x in nonimage)
    strong_nonimage = sum(1 for term in NONIMAGE_DOMINANT_TERMS if term in nonimage_text)
    return strong_nonimage >= 3 and len(nonimage) >= len(image)


def _diagnosis_hypothesis_summary(diagnosis_prediction: dict, max_items: int = 3) -> str:
    primary = str(diagnosis_prediction.get("primary_diagnosis") or "?")
    conf = str(diagnosis_prediction.get("confidence") or "?")
    hypotheses = diagnosis_prediction.get("hypothesis_set") or []
    labels: list[str] = []
    if isinstance(hypotheses, list):
        for row in hypotheses[:max_items]:
            if isinstance(row, dict):
                label = row.get("diagnosis") or row.get("name") or row.get("label")
                row_conf = row.get("confidence")
                if label:
                    labels.append(f"{label} ({row_conf})" if row_conf else str(label))
            elif row:
                labels.append(str(row))
    if not labels:
        diff = diagnosis_prediction.get("differential") or []
        if isinstance(diff, list):
            labels = [str(x) for x in diff[:max_items] if str(x).strip()]
    if labels:
        return f"primary={primary} ({conf}); hypotheses=[{'; '.join(labels)}]"
    return f"primary={primary} ({conf})"


def _keep_context_summary(knowledge_context: dict | None, max_items: int = 3) -> str:
    if not isinstance(knowledge_context, dict):
        return "unavailable"
    if not knowledge_context.get("available"):
        reason = knowledge_context.get("reason") or knowledge_context.get("error") or "no hits"
        return f"available=False ({reason})"

    # Phase 2 format: per-hypothesis anchored knowledge
    per_hyp = knowledge_context.get("per_hypothesis")
    if isinstance(per_hyp, list) and per_hyp:
        hyp_summaries: list[str] = []
        for h in per_hyp[:max_items]:
            if not isinstance(h, dict):
                continue
            name = h.get("diagnosis", "?")
            dg = h.get("disease_graph") or {}
            defn = str(dg.get("definition") or "")[:80]
            children = dg.get("children") or []
            child_names = [c.get("name", "") for c in children[:3] if c.get("name")]
            caps = h.get("morphology_captions") or []
            parts: list[str] = [name]
            if defn:
                parts.append(f"def={defn}")
            if child_names:
                parts.append(f"children=[{'; '.join(child_names)}]")
            if caps:
                parts.append(f"captions={len(caps)}")
            hyp_summaries.append(" | ".join(parts))
        return f"Phase2: {len(per_hyp)} hypotheses → {'; '.join(hyp_summaries)}"

    # Phase 1 format (legacy): site-driven broad search
    disease_hits = knowledge_context.get("disease_hits") or []
    semantic_hits = knowledge_context.get("semantic_group_hits") or []
    subtype_hits = knowledge_context.get("site_subtype_candidates") or []
    parts: list[str] = []
    subtypes: list[str] = []
    if isinstance(subtype_hits, list):
        for row in subtype_hits[:max_items]:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or row.get("label_id") or "").strip()
            count = row.get("count")
            if name:
                subtypes.append(f"{name}({count})" if count is not None else name)
    if subtypes:
        parts.append(f"site_subtypes=[{'; '.join(subtypes)}]")
    diseases: list[str] = []
    if isinstance(disease_hits, list):
        for row in disease_hits[:max_items]:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or row.get("id") or "").strip()
            score = row.get("score")
            if name:
                diseases.append(f"{name}({score})" if score is not None else name)
    groups: list[str] = []
    if isinstance(semantic_hits, list):
        for row in semantic_hits[:max_items]:
            if not isinstance(row, dict):
                continue
            anatomy = str(row.get("anatomy") or row.get("id") or "semantic_group").strip()
            score = row.get("score")
            groups.append(f"{anatomy}({score})" if score is not None else anatomy)
    if diseases:
        parts.append(f"disease=[{'; '.join(diseases)}]")
    if groups:
        parts.append(f"semantic=[{'; '.join(groups)}]")
    if not parts:
        return "available=True but no printable hits"
    return "; ".join(parts)


def _compact_session_payload(value: Any) -> Any:
    """Compact a stage payload for resumable diagnostic memory.

    Persist reasoning state, not large visual payloads or raw model transcripts.
    This makes resume behavior deterministic without writing base64 chat history.
    """
    value = _compact_selected_rois_for_memory(value)
    compact = _scrub_llm_context(_strip_heavy(value))
    if isinstance(compact, dict):
        compact.pop("raw", None)
    return compact


def _compact_selected_rois_for_memory(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            if key in {"selected_rois", "tumor_rois", "visual_rois"} and isinstance(val, list):
                out[key] = [
                    {
                        "roi_index": row.get("roi_index", idx),
                        "score": row.get("score"),
                        "width": row.get("width"),
                        "height": row.get("height"),
                        "evidence_role": row.get("evidence_role"),
                        "evidence_bucket": row.get("evidence_bucket"),
                        "zoom": bool(row.get("zoom_parent_roi_id")),
                    }
                    for idx, row in enumerate(val, start=1)
                    if isinstance(row, dict)
                ]
            else:
                out[key] = _compact_selected_rois_for_memory(val)
        return out
    if isinstance(value, list):
        return [_compact_selected_rois_for_memory(item) for item in value]
    return value


LEAKY_CONTEXT_KEYS = {
    "case_id",
    "cancer_type",
    "cancer_type_tcga",
    "dataset_name",
    "slide_path",
    "path",
    "foundation_roi_ranking_path",
    "reference_text_path",
    "prediction_text_path",
}


def _scrub_llm_context(value: Any) -> Any:
    """Remove identifiers and dataset labels before sending/saving LLM memory."""
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            k = str(key)
            if k in LEAKY_CONTEXT_KEYS:
                continue
            if k in {"source"} and isinstance(val, str) and "TCGA-" in val:
                continue
            out[k] = _scrub_llm_context(val)
        return out
    if isinstance(value, list):
        return [_scrub_llm_context(item) for item in value[:12]]
    if isinstance(value, str):
        text = re.sub(r"/local/\S+", "[path removed]", value)
        text = re.sub(r"\bTCGA-[A-Z0-9]+(?:-[A-Z0-9]+)+\b", "[case-id removed]", text)
        text = re.sub(r"\bTCGA-[A-Z]{2,5}\b", "[cohort removed]", text)
        return text
    return value


def update_diagnostic_session_context(
    context: dict | None,
    stage: str,
    payload: dict | None = None,
    selected_count: int | None = None,
) -> dict:
    ctx = dict(context or {})
    ctx.setdefault("version", "diagnostic_session_memory_v1")
    events = list(ctx.get("events") or [])
    event = {
        "stage": stage,
        "selected_roi_count": selected_count,
        "payload": _compact_session_payload(payload or {}),
    }
    events = [row for row in events if row.get("stage") != stage]
    events.append(event)
    ctx["events"] = events[-20:]
    if selected_count is not None:
        ctx["selected_roi_count"] = selected_count

    payload = payload or {}
    if "site_prediction" in payload:
        ctx["site_prediction"] = _compact_session_payload(payload.get("site_prediction"))
    if "medgemma_morphology_context" in payload:
        mg = payload.get("medgemma_morphology_context") or {}
        ctx["medgemma_summary"] = {
            "available": mg.get("available"),
            "round_count": mg.get("round_count") or len(mg.get("rounds") or []),
            "description_chars": len(str(mg.get("description") or "")),
        }
    if "broad_hypotheses" in payload:
        broad = payload.get("broad_hypotheses") or {}
        ctx["broad_hypotheses"] = {
            "best_broad_type": broad.get("best_broad_type"),
            "confidence": broad.get("confidence"),
            "hypotheses": _compact_session_payload((broad.get("hypotheses") or [])[:5]),
        }
    if "subtype_sufficiency" in payload:
        sub = payload.get("subtype_sufficiency") or {}
        ctx["subtype_sufficiency"] = {
            "best_supported_level": sub.get("best_supported_level"),
            "recommended_primary_diagnosis": sub.get("recommended_primary_diagnosis"),
            "confidence": sub.get("confidence"),
            "reason": sub.get("reason"),
        }
    if "diagnosis_prediction" in payload:
        dx = payload.get("diagnosis_prediction") or {}
        ctx["diagnosis_prediction"] = {
            "summary": _diagnosis_hypothesis_summary(dx),
            "primary_diagnosis": dx.get("primary_diagnosis"),
            "confidence": dx.get("confidence"),
            "hypothesis_set": _compact_session_payload((dx.get("hypothesis_set") or [])[:5]),
            "differential": _compact_session_payload((dx.get("differential") or [])[:5]),
        }
    if "knowledge_context" in payload:
        ctx["knowledge_context_summary"] = _keep_context_summary(payload.get("knowledge_context"))
    if "evidence_sufficiency" in payload:
        suff = payload.get("evidence_sufficiency") or {}
        ctx["evidence_sufficiency"] = {
            "sufficient_for_diagnosis": suff.get("sufficient_for_diagnosis"),
            "sufficient_for_report": suff.get("sufficient_for_report"),
            "recommended_next_step": suff.get("recommended_next_step"),
            "reinspection_action": suff.get("reinspection_action"),
            "missing_image_evidence": _compact_session_payload((suff.get("missing_image_evidence") or [])[:8]),
            "missing_nonimage_metadata": _compact_session_payload((suff.get("missing_nonimage_metadata") or [])[:8]),
        }
    if "claim_verification" in payload:
        claims = (payload.get("claim_verification") or {}).get("claims") or []
        ctx["claim_verification_summary"] = {
            "supported": sum(1 for row in claims if row.get("status") == "supported"),
            "weak": sum(1 for row in claims if row.get("status") == "weak"),
            "unsupported": sum(1 for row in claims if row.get("status") == "unsupported"),
            "not_assessable": sum(1 for row in claims if row.get("status") == "not_assessable"),
        }
    return ctx


def _context_from_checkpoint(data: dict | None, fallback: dict | None = None, stage: str = "loaded_checkpoint") -> dict:
    if isinstance(data, dict) and isinstance(data.get("diagnostic_session_context"), dict):
        return data.get("diagnostic_session_context") or {}
    selected = None
    if isinstance(data, dict):
        selected_rois = data.get("selected_rois") or data.get("tumor_rois") or []
        selected = len(selected_rois) if isinstance(selected_rois, list) else data.get("selected_count")
    return update_diagnostic_session_context(fallback, stage, data or {}, selected)


def accumulate_medgemma_morphology(rounds: list[dict]) -> dict:
    """Build a compact case-level morphology memory from per-round MedGemma outputs."""
    valid_rounds = [row for row in rounds if isinstance(row, dict) and row.get("description")]
    parts = []
    roi_ids: list[str] = []
    for row in valid_rounds:
        ridx = row.get("round") or len(parts) + 1
        role = row.get("input_role") or "roi_evidence"
        ids = [str(x) for x in (row.get("input_roi_ids") or []) if str(x)]
        roi_ids.extend(ids)
        label = f"Round {ridx} ({role})"
        if ids:
            label += f" ROI(s): {', '.join(ids[:8])}"
        parts.append(f"{label}\n{str(row.get('description') or '').strip()}")
    return {
        "available": bool(parts),
        "description": "\n\n".join(parts),
        "round_count": len(valid_rounds),
        "input_roi_ids": list(dict.fromkeys(roi_ids)),
        "rounds": valid_rounds,
    }


def medgemma_round_cache_path(out_dir: str | Path, round_index: int, input_role: str) -> Path:
    safe_role = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(input_role or "round")).strip("_") or "round"
    return Path(out_dir) / "medgemma_morphology" / f"round_{int(round_index):02d}_{safe_role}.json"


def cached_medgemma_visual_description(
    out_dir: str | Path,
    selected_rois: list[dict],
    site_prediction: dict,
    diagnosis_prediction: dict | None,
    short_report_client: QwenVisionClient | None,
    dry_run: bool = False,
    previous_context: dict | None = None,
    round_index: int = 1,
    input_role: str = "initial",
) -> dict:
    """Load/save per-round MedGemma morphology so failed report retries are cheaper."""
    path = medgemma_round_cache_path(out_dir, round_index, input_role)
    input_ids = [f"ROI {idx}" for idx, _ in enumerate(selected_rois, start=1)]
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            cached_ids = [str(x) for x in (cached.get("input_roi_ids") or [])]
            if cached_ids == input_ids and cached.get("description"):
                cached["cache_hit"] = True
                print(f"    [cache] MedGemma round {round_index} loaded from {path}", flush=True)
                return cached
        except Exception:
            pass
    result = generate_medgemma_visual_description(
        selected_rois,
        site_prediction=site_prediction,
        diagnosis_prediction=diagnosis_prediction,
        short_report_client=short_report_client,
        dry_run=dry_run,
        previous_context=previous_context,
        round_index=round_index,
        input_role=input_role,
    )
    result["cache_hit"] = False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8")
    if result.get("description"):
        path.with_suffix(".txt").write_text(str(result.get("description") or ""), encoding="utf-8")
    return result


def write_progress_checkpoint(out_dir: str | Path, stage: str, data: dict) -> None:
    """Persist lightweight stage checkpoints before final report generation."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "progress_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "data": data,
    }
    (ckpt_dir / f"{stage}.json").write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    trace_path = out / "progress_trace.json"
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {"stages": []}
    except Exception:
        trace = {"stages": []}
    stages = [row for row in (trace.get("stages") or []) if row.get("stage") != stage]
    stages.append({"stage": stage, "path": str(ckpt_dir / f"{stage}.json")})
    trace["stages"] = stages
    trace_path.write_text(json.dumps(_json_safe(trace), indent=2, ensure_ascii=False), encoding="utf-8")


def load_progress_checkpoint(out_dir: str | Path, stage: str) -> dict:
    path = Path(out_dir) / "progress_checkpoints" / f"{stage}.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("data") if isinstance(payload.get("data"), dict) else {}
    except Exception:
        return {}


def load_report_ready_checkpoint(out_dir: str | Path) -> dict:
    """Load stage-9 checkpoint so retry can skip directly to final report."""
    stage9 = load_progress_checkpoint(out_dir, "09_slots_claim_verification")
    if not stage9:
        return {}
    stage8 = load_progress_checkpoint(out_dir, "08_final_diagnosis")
    merged = {}
    merged.update(stage8)
    merged.update(stage9)
    guard_changed_primary = False
    if merged.get("diagnosis_prediction") and merged.get("site_prediction"):
        original_primary = str((merged.get("diagnosis_prediction") or {}).get("primary_diagnosis") or "")
        subtype_suff = apply_known_site_lineage_guard_to_subtype_sufficiency(
            normalize_subtype_sufficiency_payload(merged.get("subtype_sufficiency") or {}),
            merged.get("site_prediction") or {},
        )
        dx = apply_known_site_lineage_guard(
            apply_indeterminate_potential_if_supported(
                normalize_primary_diagnosis_payload(merged.get("diagnosis_prediction") or {}),
                subtype_suff,
            ),
            subtype_suff,
            merged.get("site_prediction") or {},
        )
        dx = apply_indeterminate_potential_if_supported(dx, subtype_suff)
        merged["subtype_sufficiency"] = subtype_suff
        merged["diagnosis_prediction"] = dx
        guard_changed_primary = bool(original_primary and original_primary != str(dx.get("primary_diagnosis") or ""))
    if guard_changed_primary:
        return {}
    required = ["selected_rois", "site_prediction", "diagnosis_prediction", "diagnosis_slots", "claim_verification"]
    if all(merged.get(key) is not None for key in required):
        return merged
    return {}


def restore_reinspection_round_checkpoints(out_dir: str | Path, max_rounds: int) -> list[dict]:
    """Return consecutively completed reinspection checkpoints from round 1."""
    rounds: list[dict] = []
    for idx in range(1, max(0, int(max_rounds)) + 1):
        data = load_progress_checkpoint(out_dir, f"07_reinspection_round_{idx}")
        if not data:
            break
        required = ["selected_rois", "segmentation_context", "diagnosis_prediction", "evidence_sufficiency"]
        if not all(key in data for key in required):
            break
        rounds.append(data)
    return rounds


def plan_reinspection(
    candidates: list[dict],
    selected_rois: list[dict],
    sufficiency: dict,
    diagnosis_prediction: dict,
    max_extra_rois: int,
    out_dir: str | Path | None = None,
    max_zoom_depth: int = 1,
) -> dict:
    norm = normalize_evidence_gaps(sufficiency, diagnosis_prediction)
    missing = list(norm.get("missing_image_evidence") or [])
    blocked_by_nonimage = list(norm.get("missing_nonimage_metadata") or [])
    action = _infer_reinspection_action(norm, max_zoom_depth=max_zoom_depth)
    zoom_rois: list[dict] = []
    zoom_error: str | None = None
    if action == "zoom_current_roi" and out_dir is not None and max_zoom_depth > 0:
        try:
            zoom_rois = make_zoom_rois_from_sufficiency(
                selected_rois,
                sufficiency=norm,
                out_dir=Path(out_dir) / "zoom_rois",
                max_zoom_depth=max_zoom_depth,
                max_count=max_extra_rois,
            )
        except Exception as exc:
            zoom_error = str(exc)
            zoom_rois = []
        if not zoom_rois:
            action = "add_independent_roi"

    selected_paths = {str(row.get("image_path")) for row in selected_rois}
    extra = []
    if missing and action == "add_independent_roi":
        for row in _spatially_diverse_rows(candidates, existing=selected_rois):
            if str(row.get("image_path")) in selected_paths:
                continue
            out = dict(row)
            out["evidence_bucket"] = "discriminative_reinspection"
            extra.append(out)
            if len(extra) >= max_extra_rois:
                break
    new_rois = zoom_rois if zoom_rois else extra
    return {
        "enabled": bool(missing),
        "loop_limit": 1,
        "strategy": "v3_action_based_zoom_or_add_roi_reinspection",
        "action": action,
        "llm_reinspection_action": sufficiency.get("reinspection_action") or sufficiency.get("recommended_next_step"),
        "missing_image_evidence": missing,
        "missing_nonimage_metadata": blocked_by_nonimage,
        "missing_evidence": missing + blocked_by_nonimage,
        "zoom_rois": zoom_rois,
        "zoom_error": zoom_error,
        "extra_rois": extra,
        "new_rois": new_rois,
        "note": (
            "For image-actionable gaps, zooms into existing ROIs when higher-power detail is needed; "
            "otherwise adds small numbers of spatially diverse independent ROIs. Nonimage metadata gaps "
            "such as IHC, gross size, margins, or molecular data do not trigger ROI search."
        ),
    }


def _infer_reinspection_action(sufficiency: dict, max_zoom_depth: int = 1) -> str:
    # Always trust actual gap classification over LLM's recommended_next_step text.
    if sufficiency.get("has_image_actionable_gap"):
        missing_text = " ".join(str(x).lower() for x in (sufficiency.get("missing_image_evidence") or []))
        if max_zoom_depth > 0 and any(term in missing_text for term in ZOOM_ACTIONABLE_TERMS):
            return "zoom_current_roi"
        return "add_independent_roi"
    if sufficiency.get("missing_nonimage_metadata"):
        return "require_nonimage_metadata"
    # Fallback: check explicit text
    explicit = str(sufficiency.get("reinspection_action") or sufficiency.get("recommended_next_step") or "").lower()
    if "zoom" in explicit and max_zoom_depth > 0:
        return "zoom_current_roi"
    if "select_more" in explicit or "add" in explicit:
        return "add_independent_roi"
    if "require_nonimage" in explicit or "metadata" in explicit:
        return "require_nonimage_metadata"
    return "none"


def make_zoom_rois_from_sufficiency(
    selected_rois: list[dict],
    sufficiency: dict,
    out_dir: Path,
    max_zoom_depth: int = 1,
    max_count: int = 2,
) -> list[dict]:
    if not selected_rois or max_count <= 0 or max_zoom_depth <= 0:
        return []
    target_ids = [str(x) for x in (sufficiency.get("target_roi_ids") or []) if str(x)]
    regions = [str(x).strip().lower() for x in (sufficiency.get("zoom_regions") or []) if str(x)]
    valid_regions = {"center", "top-left", "top-right", "bottom-left", "bottom-right"}
    regions = [r for r in regions if r in valid_regions] or ["center"]
    targets = []
    if target_ids:
        wanted = set(target_ids)
        targets.extend([
            row for row in selected_rois
            if str(row.get("roi_id") or row.get("roi_index") or row.get("image_path")) in wanted
        ])
    if not targets:
        targets = sorted(selected_rois, key=lambda row: float(row.get("score") or 0.0), reverse=True)

    out: list[dict] = []
    existing_ids = {str(row.get("roi_id") or "") for row in selected_rois}
    existing_paths = {str(row.get("image_path") or "") for row in selected_rois}
    out_dir.mkdir(parents=True, exist_ok=True)
    for row in targets:
        for region in regions:
            zoom = _make_zoom_roi(row, region=region, out_dir=out_dir, max_zoom_depth=max_zoom_depth, reason=sufficiency.get("reinspection_reason") or sufficiency.get("reason"))
            if zoom:
                if str(zoom.get("roi_id") or "") in existing_ids or str(zoom.get("image_path") or "") in existing_paths:
                    continue
                out.append(zoom)
                existing_ids.add(str(zoom.get("roi_id") or ""))
                existing_paths.add(str(zoom.get("image_path") or ""))
            if len(out) >= max_count:
                return out
    return out


def _make_zoom_roi(row: dict, region: str, out_dir: Path, max_zoom_depth: int, reason: str | None = None) -> dict | None:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    slide_path = str(meta.get("slide_path") or row.get("slide_path") or "")
    if not slide_path or SlideReader is None:
        return None
    try:
        x = int(row.get("x"))
        y = int(row.get("y"))
        width = int(row.get("width"))
        height = int(row.get("height"))
    except Exception:
        m = re.search(r"_x(\d+)_y(\d+)_l(\d+)_w(\d+)_h(\d+)", str(row.get("roi_id") or row.get("image_path") or ""))
        if not m:
            return None
        x, y, _level, width, height = [int(v) for v in m.groups()]
    try:
        level = int(row.get("level") if row.get("level") is not None else 0)
    except Exception:
        level = 0
    depth = int(row.get("zoom_depth") or meta.get("zoom_depth") or 0)
    if depth >= max_zoom_depth or width <= 1 or height <= 1:
        return None
    rx, ry = {
        "center": (0.50, 0.50),
        "top-left": (0.25, 0.25),
        "top-right": (0.75, 0.25),
        "bottom-left": (0.25, 0.75),
        "bottom-right": (0.75, 0.75),
    }.get(region, (0.50, 0.50))
    new_w = max(512, int(width * 0.5))
    new_h = max(512, int(height * 0.5))
    new_w = min(new_w, width)
    new_h = min(new_h, height)
    cx = x + int(width * rx)
    cy = y + int(height * ry)
    new_x = max(0, cx - new_w // 2)
    new_y = max(0, cy - new_h // 2)
    parent_id = str(row.get("roi_id") or row.get("roi_index") or Path(str(row.get("image_path") or "roi")).stem)
    clean_region = region.replace("-", "")
    zoom_id = f"{parent_id}_zoom{depth + 1}_{clean_region}_x{new_x}_y{new_y}_l{level}_w{new_w}_h{new_h}"
    out_path = out_dir / f"{zoom_id}.png"
    if not out_path.exists():
        reader = SlideReader(slide_path)
        try:
            image = reader.read_region(new_x, new_y, new_w, new_h, level)
        finally:
            reader.close()
        image.save(out_path)
    zoom = dict(row)
    zoom.update({
        "roi_id": zoom_id,
        "image_path": str(out_path),
        "x": new_x,
        "y": new_y,
        "width": new_w,
        "height": new_h,
        "level": level,
        "score": float(row.get("score") or 0.0) + 0.0001 * (depth + 1),
        "source": f"{row.get('source') or 'evidence_graph'}:llm_zoom",
        "evidence_bucket": "llm_zoom_reinspection",
        "evidence_role": row.get("evidence_role") or "tumor_candidate",
        "zoom_depth": depth + 1,
        "zoom_parent_roi_id": parent_id,
        "zoom_region": region,
        "zoom_reason": str(reason or ""),
    })
    zmeta = dict(meta)
    zmeta.update({
        "slide_path": slide_path,
        "zoom_depth": depth + 1,
        "zoom_parent_roi_id": parent_id,
        "zoom_region": region,
        "zoom_reason": str(reason or ""),
    })
    zoom["metadata"] = zmeta
    return zoom


def filter_slots_by_verified_claims(slots: dict, claim_verification: dict) -> dict:
    filtered = dict(slots)
    bad = {"unsupported", "contradicted"}
    not_assess = {"not_assessable"}
    by_field: dict[str, list[dict]] = {}
    for row in claim_verification.get("claims") or []:
        by_field.setdefault(str(row.get("field") or ""), []).append(row)
    for field, rows in by_field.items():
        statuses = {str(row.get("status") or "").lower() for row in rows}
        if statuses and statuses <= bad:
            filtered[field] = "not supported by available ROI evidence"
        elif statuses and statuses <= not_assess:
            filtered[field] = "not assessable from provided ROI evidence"
    return filtered


def generate_claim_verified_report(
    selected_rois: list[dict],
    visual_rois: list[dict],
    site_prediction: dict,
    diagnosis_prediction: dict,
    sufficiency: dict,
    diagnosis_slots: dict,
    claim_verification: dict,
    segmentation_context: dict,
    knowledge_context: dict,
    short_report_client: QwenVisionClient | None,
    long_report_client: QwenVisionClient | None,
    dry_run: bool,
    enforce_staged_diagnosis: bool,
    medgemma_context: dict | None = None,
    diagnostic_session_context: dict | None = None,
) -> dict:
    if dry_run or long_report_client is None:
        report = build_fallback_long_report(site_prediction, diagnosis_prediction, sufficiency)
        return {"long_report": report, "long_report_rich": report, "short_report": "", "figures": [], "raw": None}

    base = generate_long_report_hybrid(
        selected_rois=selected_rois,
        site_prediction=site_prediction,
        diagnosis_prediction=diagnosis_prediction,
        sufficiency=sufficiency,
        segmentation_context=segmentation_context,
        knowledge_context=knowledge_context,
        short_report_client=short_report_client or long_report_client,
        long_report_client=long_report_client,
        dry_run=False,
        enforce_staged_diagnosis=enforce_staged_diagnosis,
        visual_rois=visual_rois,
        medgemma_context=medgemma_context,
        diagnostic_session_context=diagnostic_session_context,
    )
    prompt = f"""/no_think
Rewrite the draft pathology report so that every statement obeys claim verification.

Diagnosis slots:
{json.dumps(_strip_heavy(diagnosis_slots), indent=2, ensure_ascii=False)}

Claim verification:
{json.dumps(_strip_heavy(claim_verification), indent=2, ensure_ascii=False)}

Case diagnostic session memory:
{json.dumps(_strip_heavy(diagnostic_session_context or {"available": False}), indent=2, ensure_ascii=False)}

Draft report:
---
{base.get("long_report_rich") or base.get("long_report")}
---

Rules:
- Supported claims may be written affirmatively.
- Weak claims must be hedged as possible/suggestive.
- Unsupported or contradicted claims must be removed.
- Not-assessable claims must be written as "not assessable from provided ROI evidence".
- Do not invent IHC, molecular, stage, gross size, margin distance, or lymph-node count
  unless explicitly listed in the diagnosis slots' recommended_ancillary_studies.
  When recommended_ancillary_studies are present, include them in the
  "Immunohistochemistry/Ancillary Studies" section prefixed with
  "Recommended for confirmation: ...".
- Follow exactly this section order:
{LONG_REPORT_FORMAT}

Return only the report text."""
    raw = long_report_client.chat(prompt, image_paths=[])
    report = (raw.get("content") or "").strip()
    if not report:
        report = base.get("long_report_rich") or base.get("long_report")
    base["long_report_rich"] = report
    base["long_report"] = _strip_figure_refs(report)
    base.setdefault("raw", {})
    base["raw"]["claim_filter_rewrite"] = _raw_summary(raw)
    return base


def save_evidence_graph_outputs(payload: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Add reusable visual artifact paths before serializing the trace so HTML,
    # staged_trace.json, and case_artifact_index.json can all point to the same
    # per-slide WSI overview artifacts.
    if not payload.get("wsi_overviews"):
        _attach_wsi_thumbnails(payload, out_dir)

    files = {
        "staged_trace.json": payload,
        "case_evidence_graph.json": payload.get("evidence_graph"),
        "diagnosis_slots.json": payload.get("diagnosis_slots"),
        "filtered_diagnosis_slots.json": payload.get("filtered_diagnosis_slots"),
        "claim_verification.json": payload.get("claim_verification"),
        "reinspection_plan.json": payload.get("reinspection_plan"),
        "roi_selection_scores.json": payload.get("roi_selection"),
        "site_prediction.json": payload.get("site_prediction"),
        "broad_hypotheses.json": payload.get("broad_hypotheses"),
        "subtype_sufficiency.json": payload.get("subtype_sufficiency"),
        "diagnosis_prediction.json": payload.get("diagnosis_prediction"),
        "segmentation_context.json": payload.get("segmentation_context"),
        "knowledge_context.json": payload.get("knowledge_context"),
        "medgemma_morphology_context.json": payload.get("medgemma_morphology_context"),
        "evidence_sufficiency.json": payload.get("evidence_sufficiency"),
        "diagnostic_session_context.json": payload.get("diagnostic_session_context"),
    }
    for name, data in files.items():
        (out_dir / name).write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "generated_long_report.txt").write_text(payload.get("generated_long_report") or "", encoding="utf-8")
    (out_dir / "generated_long_report_rich.txt").write_text(payload.get("generated_long_report_rich") or "", encoding="utf-8")
    if payload.get("medgemma_short_report"):
        (out_dir / "medgemma_short_report.txt").write_text(payload["medgemma_short_report"], encoding="utf-8")
    medgemma_ctx = payload.get("medgemma_morphology_context") or {}

    if medgemma_ctx.get("description"):
        (out_dir / "medgemma_morphology_context.txt").write_text(
            str(medgemma_ctx.get("description") or ""),
            encoding="utf-8",
        )
    medgemma_rounds = medgemma_ctx.get("rounds") or []
    if medgemma_rounds:
        mg_dir = out_dir / "medgemma_morphology"
        mg_dir.mkdir(parents=True, exist_ok=True)
        (mg_dir / "accumulated_morphology_context.json").write_text(
            json.dumps(_json_safe(medgemma_ctx), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (mg_dir / "accumulated_morphology_context.txt").write_text(
            str(medgemma_ctx.get("description") or ""),
            encoding="utf-8",
        )
        for row in medgemma_rounds:
            ridx = row.get("round") or len(list(mg_dir.glob("round_*.json"))) + 1
            try:
                ridx_int = int(ridx)
            except Exception:
                ridx_int = len(list(mg_dir.glob("round_*.json"))) + 1
            role = str(row.get("input_role") or "roi_evidence").replace("/", "_")
            (mg_dir / f"round_{ridx_int:02d}_{role}.json").write_text(
                json.dumps(_json_safe(row), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    (out_dir / "report.md").write_text(render_evidence_graph_markdown(payload), encoding="utf-8")

    # Use external review workspace renderer for HTML (scripts/render_review_workspace_html.py)
    import sys
    _scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    from render_review_workspace_html import render_html, read_case

    case_payload = read_case(out_dir)
    html_out = render_html(case_payload, out_dir / "report.html")
    (out_dir / "report.html").write_text(html_out, encoding="utf-8")


def render_evidence_graph_markdown(payload: dict) -> str:
    lines = [
        f"# Evidence Graph Agent: {payload.get('case_id')}",
        "",
        "## Diagnosis Slots",
        "```json",
        json.dumps(_json_safe(payload.get("diagnosis_slots")), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Claim Verification",
        "```json",
        json.dumps(_json_safe(payload.get("claim_verification")), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Report",
        payload.get("generated_long_report_rich") or payload.get("generated_long_report") or "",
    ]
    return "\n".join(lines)


def _json_safe(value: Any, seen: set[int] | None = None) -> Any:
    """Return a JSON-serializable copy while breaking accidental cycles."""
    if seen is None:
        seen = set()
    if isinstance(value, dict):
        obj_id = id(value)
        if obj_id in seen:
            return "[Circular reference removed]"
        seen.add(obj_id)
        out = {str(key): _json_safe(val, seen) for key, val in value.items()}
        seen.remove(obj_id)
        return out
    if isinstance(value, list):
        obj_id = id(value)
        if obj_id in seen:
            return ["[Circular reference removed]"]
        seen.add(obj_id)
        out = [_json_safe(item, seen) for item in value]
        seen.remove(obj_id)
        return out
    if isinstance(value, tuple):
        return [_json_safe(item, seen) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


# ── WSI thumbnail helpers ──────────────────────────────────────────────────────

_ROI_PALETTE_HEX = [
    "#3b82f6", "#ef4444", "#22c55e", "#f59e0b",
    "#a855f7", "#14b8a6", "#ec4899", "#fb923c",
]
_ROI_PALETTE_RGB = [
    (59, 130, 246), (239, 68, 68), (34, 197, 94), (245, 158, 11),
    (168, 85, 247), (20, 184, 166), (236, 72, 153), (251, 146, 60),
]


def _attach_wsi_thumbnails(payload: dict, out_dir: Path) -> None:
    """Generate clean + ROI-annotated WSI thumbnails for every slide with evidence.

    Earlier HTML used the first slide thumbnail for all selected ROIs.  In a
    multi-slide case that silently mixed coordinate systems and produced boxes
    on the wrong tissue.  This stores per-slide overview artifacts instead.
    """
    records = _figure_roi_records(payload, payload.get("figures") or [])
    if not records:
        return

    grouped: dict[str, list[dict]] = {}
    for rec in records:
        slide_id = str(rec.get("slide_id") or "")
        if slide_id:
            grouped.setdefault(slide_id, []).append(rec)

    overviews = []
    for slide_id, rows in grouped.items():
        slide_path = rows[0].get("slide_path")
        if not slide_path:
            continue
        try:
            clean = _generate_wsi_thumbnail(slide_path)
            if not clean:
                continue
            slide_w = int(rows[0].get("slide_width") or 0)
            slide_h = int(rows[0].get("slide_height") or 0)
            if (not slide_w or not slide_h) and SlideReader is not None:
                rd = SlideReader(str(slide_path))
                try:
                    slide_w, slide_h = rd.size
                finally:
                    rd.close()
            annotated = None
            if slide_w and slide_h:
                annotated = _annotate_wsi_thumbnail_with_labeled_rois(
                    clean,
                    rows,
                    slide_w,
                    slide_h,
                    out_dir / f"wsi_thumbnail_{_safe_id(slide_id)}_annotated.jpg",
                )
            overviews.append({
                "slide_id": slide_id,
                "slide_path": str(slide_path),
                "thumbnail_path": str(Path(clean).resolve()),
                "annotated_path": str(Path(annotated).resolve()) if annotated else str(Path(clean).resolve()),
                "figure_numbers": [int(r["figure_number"]) for r in rows if r.get("figure_number")],
                "roi_count": len(rows),
                "slide_width": slide_w,
                "slide_height": slide_h,
            })
        except Exception:
            continue

    if overviews:
        payload["wsi_overviews"] = overviews
        payload["wsi_thumbnail_path"] = overviews[0]["thumbnail_path"]
        payload["wsi_thumbnail_annotated_path"] = overviews[0]["annotated_path"]


def _annotate_wsi_thumbnail_with_labeled_rois(
    thumb_path: str | Path,
    rois: list[dict],
    slide_w: int,
    slide_h: int,
    out_path: Path,
) -> Path | None:
    if out_path.exists():
        return out_path
    if not Path(str(thumb_path)).is_file() or not rois or not slide_w or not slide_h:
        return None
    try:
        from PIL import Image, ImageDraw  # noqa: PLC0415

        thumb = Image.open(str(thumb_path)).convert("RGB")
        tw, th = thumb.size
        sx, sy = tw / slide_w, th / slide_h
        draw = ImageDraw.Draw(thumb)
        lw = max(2, tw // 320)
        for rec in rois:
            fig_num = int(rec.get("figure_number") or 0)
            color = _ROI_PALETTE_RGB[(fig_num - 1) % len(_ROI_PALETTE_RGB)]
            x0 = int(float(rec.get("x") or 0) * sx)
            y0 = int(float(rec.get("y") or 0) * sy)
            bw = max(lw + 2, int(float(rec.get("width") or 0) * sx))
            bh = max(lw + 2, int(float(rec.get("height") or 0) * sy))
            x1 = min(tw - 1, x0 + bw)
            y1 = min(th - 1, y0 + bh)
            x0 = max(0, min(tw - 1, x0))
            y0 = max(0, min(th - 1, y0))
            for t in range(lw):
                draw.rectangle([x0 + t, y0 + t, x1 - t, y1 - t], outline=color)
            label = str(fig_num)
            badge = max(14, lw * 6)
            bx0, by0 = x0 + lw, y0 + lw
            draw.rectangle([bx0, by0, bx0 + badge, by0 + badge], fill=color)
            tx = bx0 + max(1, (badge - len(label) * 6) // 2)
            ty = by0 + max(1, (badge - 8) // 2)
            draw.text((tx, ty), label, fill=(255, 255, 255))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(str(out_path), format="JPEG", quality=88, optimize=True)
        return out_path
    except Exception:
        return None


def _figure_roi_records(payload: dict, figures: list[dict]) -> list[dict]:
    """Join figure entries back to selected ROI metadata.

    Figures intentionally avoid heavy fields, but the HTML needs slide-level
    coordinates for correct multi-slide WSI overlays.
    """
    selected = payload.get("visual_rois") or payload.get("selected_rois") or []
    by_roi_id = {str(r.get("roi_id") or ""): r for r in selected if r.get("roi_id")}
    by_image = {str(r.get("image_path") or ""): r for r in selected if r.get("image_path")}
    # Fallback: use evidence_graph roi_nodes if selected_rois is empty
    if not selected:
        roi_nodes = (payload.get("evidence_graph") or {}).get("roi_nodes") or []
        by_roi_id = {str(r.get("roi_id") or ""): r for r in roi_nodes if r.get("roi_id")}
        by_image = {str(r.get("image_path") or ""): r for r in roi_nodes if r.get("image_path")}
    records = []
    for idx, fig in enumerate(figures, start=1):
        roi = by_roi_id.get(str(fig.get("roi_id") or "")) or by_image.get(str(fig.get("image_path") or "")) or {}
        meta = roi.get("metadata") or {}
        slide_info = meta.get("slide_info") or {}
        slide_path = meta.get("slide_path") or fig.get("slide_path") or roi.get("slide_path")
        slide_id = roi.get("slide_id") or fig.get("slide_id") or (Path(str(slide_path)).stem if slide_path else "")
        x, y, w, h = roi.get("x"), roi.get("y"), roi.get("width"), roi.get("height")
        if x is None or y is None or w is None or h is None:
            m = re.search(r"_x(\d+)_y(\d+)_l\d+_w(\d+)_h(\d+)", str(fig.get("roi_id") or ""))
            if m:
                x, y, w, h = [int(v) for v in m.groups()]
        rec = dict(fig)
        rec.update({
            "figure_number": idx,
            "roi_index": fig.get("roi_index") or roi.get("roi_index") or idx,
            "roi_id": fig.get("roi_id") or roi.get("roi_id"),
            "image_path": fig.get("image_path") or roi.get("image_path"),
            "overlay_path": fig.get("overlay_path") or roi.get("overlay_path"),
            "overlay_path_file": fig.get("overlay_path_file") or roi.get("overlay_path_file"),
            "composite_path": fig.get("composite_path") or roi.get("composite_path"),
            "composite_path_file": fig.get("composite_path_file") or roi.get("composite_path_file"),
            "mask_path": fig.get("mask_path") or roi.get("mask_path"),
            "mask_path_file": fig.get("mask_path_file") or roi.get("mask_path_file"),
            "is_zoom": bool(fig.get("is_zoom") or roi.get("is_zoom") or roi.get("zoom_depth") or roi.get("zoom_parent_roi_id")),
            "zoom_depth": fig.get("zoom_depth") or roi.get("zoom_depth"),
            "zoom_parent_roi_id": fig.get("zoom_parent_roi_id") or roi.get("zoom_parent_roi_id"),
            "zoom_reason": fig.get("zoom_reason") or roi.get("zoom_reason"),
            "slide_id": slide_id,
            "slide_path": slide_path,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "slide_width": slide_info.get("width"),
            "slide_height": slide_info.get("height"),
            "metadata": meta,
        })
        records.append(rec)
    return records


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:180]


def _figure_image_path(fig: dict) -> str:
    return str(
        fig.get("figure_file")
        or fig.get("copied_image_path")
        or fig.get("image_path")
        or ""
    )


def _figure_overlay_path(fig: dict) -> str:
    seg = fig.get("segmentation") if isinstance(fig.get("segmentation"), dict) else {}
    return str(
        fig.get("overlay_path_file")
        or fig.get("overlay_path")
        or seg.get("overlay_path_file")
        or seg.get("overlay_path")
        or ""
    )


def _figure_composite_path(fig: dict) -> str:
    seg = fig.get("segmentation") if isinstance(fig.get("segmentation"), dict) else {}
    return str(
        fig.get("composite_path_file")
        or fig.get("composite_path")
        or seg.get("composite_path_file")
        or seg.get("composite_path")
        or ""
    )


def _ref_button(ref: str, label: str, cls: str = "") -> str:
    if not ref:
        return _esc(label)
    cls_attr = f"ref-chip {cls}".strip()
    return (
        f'<button type="button" class="{_esc(cls_attr)}" '
        f'data-ref="{_esc(ref)}">{_esc(label)}</button>'
    )


def _build_inspector_index(payload: dict, figures: list[dict], claims: list[dict]) -> dict:
    """Build right-side inspector records for report/evidence cross references."""
    index: dict[str, dict] = {}

    def add(
        ref: str,
        type_: str,
        title: str,
        *,
        meta: str = "",
        summary: str = "",
        details: Any = None,
        image: str = "",
        supports: list[str] | None = None,
        related: list[dict] | None = None,
    ) -> None:
        if not ref:
            return
        index[ref] = {
            "type": type_,
            "title": title,
            "meta": meta,
            "summary": summary,
            "details": _strip_heavy(details) if details is not None else None,
            "image": image,
            "supports": supports or [],
            "related": related or [],
        }

    fig_ref_by_roi: dict[str, str] = {}
    fig_refs: list[dict] = []
    for i, fig in enumerate(figures):
        if not isinstance(fig, dict):
            continue
        fig_n = int(fig.get("figure_number") or i + 1)
        ref = f"fig-{fig_n}"
        roi_id = str(fig.get("roi_id") or "")
        if roi_id:
            fig_ref_by_roi[roi_id] = ref
            fig_ref_by_roi[f"roi::{roi_id}"] = ref
        fig_refs.append({"ref": ref, "label": f"Fig {fig_n}"})

    claim_refs: list[dict] = []
    claims_by_fig: dict[str, list[str]] = {}
    for i, claim in enumerate(claims or [], start=1):
        if not isinstance(claim, dict):
            continue
        raw_id = str(claim.get("claim_id") or claim.get("id") or f"{i}")
        claim_ref = raw_id if raw_id.startswith("claim-") else f"claim-{raw_id}"
        claim_refs.append({"ref": claim_ref, "label": claim.get("field") or f"Claim {i}"})
        claim_label = str(claim.get("claim") or claim.get("field") or claim_ref)
        for ev in claim.get("evidence") or []:
            ev_ref = fig_ref_by_roi.get(str(ev))
            if ev_ref:
                claims_by_fig.setdefault(ev_ref, []).append(claim_label)

    for i, fig in enumerate(figures):
        if not isinstance(fig, dict):
            continue
        fig_n = int(fig.get("figure_number") or i + 1)
        ref = f"fig-{fig_n}"
        seg = fig.get("segmentation") if isinstance(fig.get("segmentation"), dict) else {}
        qc = ((payload.get("segmentation_context") or {}).get("seg_qc") or {}).get("per_roi", {}).get(str(fig_n), {})
        add(
            ref,
            "Visual Evidence",
            f"Figure {fig_n}",
            meta=str(fig.get("evidence_bucket") or fig.get("evidence_role") or "selected ROI"),
            summary=str(fig.get("caption") or fig.get("roi_id") or ""),
            details={
                "roi_id": fig.get("roi_id"),
                "slide_id": fig.get("slide_id"),
                "evidence_bucket": fig.get("evidence_bucket") or fig.get("evidence_role"),
                "score": fig.get("score"),
                "caption": fig.get("caption"),
                "is_zoom": fig.get("is_zoom"),
                "segmentation": _compact_segmentation(seg) if seg else {},
                "segmentation_qc": qc,
            },
            image=_img_data_uri(_figure_composite_path(fig) or _figure_overlay_path(fig) or _figure_image_path(fig), max_edge=760),
            supports=claims_by_fig.get(ref, []),
            related=[{"ref": "decision-segmentation", "label": "Segmentation"}] if seg else [],
        )

    for i, claim in enumerate(claims or [], start=1):
        if not isinstance(claim, dict):
            continue
        raw_id = str(claim.get("claim_id") or claim.get("id") or f"{i}")
        ref = raw_id if raw_id.startswith("claim-") else f"claim-{raw_id}"
        related = []
        for ev in claim.get("evidence") or []:
            evs = str(ev)
            if evs in fig_ref_by_roi:
                ev_ref = fig_ref_by_roi[evs]
                related.append({"ref": ev_ref, "label": ev_ref.replace("-", " ").title()})
            elif evs.startswith("evidence::knowledge"):
                related.append({"ref": "keep-main", "label": "KEEP"})
            elif evs.startswith("evidence::morphology") or evs.startswith("evidence::visual_description"):
                related.append({"ref": "morphology-main", "label": "Morphology"})
            elif evs.startswith("evidence::segmentation"):
                related.append({"ref": "decision-segmentation", "label": "Segmentation"})
        add(
            ref,
            "Claim",
            str(claim.get("field") or f"Claim {i}"),
            meta=f"status={claim.get('status') or 'unknown'}",
            summary=str(claim.get("claim") or ""),
            details={"status": claim.get("status"), "reason": claim.get("reason"), "evidence": claim.get("evidence")},
            related=related,
        )

    medgemma_ctx = payload.get("medgemma_morphology_context") or {}
    medgemma_text = medgemma_ctx.get("description") or payload.get("medgemma_short_report") or ""
    add(
        "morphology-main",
        "Morphology Evidence",
        "MedGemma morphology summary",
        meta=f"rounds={medgemma_ctx.get('round_count') or 0}",
        summary=str(medgemma_text)[:2500],
        details=medgemma_ctx,
        related=fig_refs[:8],
    )

    kg = payload.get("knowledge_context") or {}
    add(
        "keep-main",
        "Knowledge Evidence",
        "KEEP retrieval context",
        meta=f"available={bool(kg.get('available') or kg.get('used'))}",
        summary=str(kg.get("query") or kg.get("summary") or "Auxiliary KEEP context"),
        details={
            "query": kg.get("query"),
            "site_subtype_candidates": kg.get("site_subtype_candidates"),
            "disease_context": kg.get("disease_context"),
            "semantic_context": kg.get("semantic_context"),
            "final_context_summary": kg.get("final_context_summary"),
        },
    )

    subtype = payload.get("subtype_sufficiency") or {}
    add(
        "keep-subtypes",
        "Knowledge Decision",
        "Site-specific subtype candidates",
        meta=f"candidate_count={subtype.get('candidate_count') or 0}",
        summary=str(subtype.get("recommended_primary_diagnosis") or ""),
        details=subtype,
        related=[{"ref": "keep-main", "label": "KEEP context"}],
    )

    decision_payloads = [
        ("decision-roi-selection", "ROI selection", payload.get("roi_selection") or {}),
        ("decision-site", "Site input / prediction", payload.get("site_prediction") or {}),
        ("decision-segmentation", "Segmentation + QC", payload.get("segmentation_context") or {}),
        ("decision-morphology", "MedGemma morphology", medgemma_ctx),
        ("decision-broad", "Broad hypothesis planning", payload.get("broad_hypotheses") or {}),
        ("decision-subtype", "Subtype sufficiency", subtype),
        ("decision-diagnosis", "Diagnostic hypothesis", payload.get("diagnosis_prediction") or {}),
        ("decision-sufficiency", "Evidence sufficiency", payload.get("evidence_sufficiency") or {}),
        ("decision-reinspection", "Reinspection loop", payload.get("reinspection_plan") or {}),
        ("decision-claims", "Claim verification", payload.get("claim_verification") or {}),
        ("decision-report", "Report generation", {
            "generated_long_report_len": len(payload.get("generated_long_report") or ""),
            "rich_report_len": len(payload.get("generated_long_report_rich") or ""),
        }),
    ]
    for ref, title, details in decision_payloads:
        summary = ""
        if isinstance(details, dict):
            summary = str(
                details.get("primary_diagnosis")
                or details.get("predicted_site")
                or details.get("recommended_primary_diagnosis")
                or details.get("recommended_next_step")
                or ""
            )
        related = []
        if ref in {"decision-broad", "decision-diagnosis", "decision-sufficiency"}:
            related.append({"ref": "morphology-main", "label": "Morphology"})
        if ref in {"decision-subtype", "decision-diagnosis"}:
            related.append({"ref": "keep-subtypes", "label": "Subtype candidates"})
        if ref in {"decision-roi-selection", "decision-segmentation", "decision-diagnosis"}:
            related.extend(fig_refs[:6])
        add(ref, "Decision", title, summary=summary, details=details, related=related)

    return index


def render_evidence_graph_html(payload: dict) -> str:
    """Render a comprehensive, beautiful HTML case report with all evidence."""
    figures = payload.get("figures") or []
    claims = (payload.get("claim_verification") or {}).get("claims") or []
    graph = payload.get("evidence_graph") or {}
    case_id = _esc(payload.get("case_id") or "unknown")
    cancer_type = _esc((payload.get("metadata") or {}).get("cancer_type") or "")
    site_pred = payload.get("site_prediction") or {}
    broad_hyp = payload.get("broad_hypotheses") or {}
    subtype_suff = payload.get("subtype_sufficiency") or {}
    diag_pred = payload.get("diagnosis_prediction") or {}
    eval_data = payload.get("evaluation") or {}
    seg_ctx = payload.get("segmentation_context") or {}
    diag_slots = payload.get("filtered_diagnosis_slots") or payload.get("diagnosis_slots") or {}
    roi_scores = payload.get("roi_selection") or {}
    evidence_suff = payload.get("evidence_sufficiency") or {}
    report = payload.get("generated_long_report_rich") or payload.get("generated_long_report") or ""

    # ── WSI Overview + Visual Evidence (merged) ──────────────────────────────
    figure_records = _figure_roi_records(payload, figures)
    combined_wsi_html = _build_combined_wsi_visual(payload, figure_records)

    # ── Interactive Report (with embedded evidence) ─────────────────────────
    report_sections = _parse_report_sections(report)
    interactive_report_html = _render_interactive_report(report_sections, claims, figure_records)
    interactive_report_html = _render_report_visual_strip(figure_records) + interactive_report_html

    # ── Diagnosis slots / concise report summary ────────────────────────────
    diag_html = _render_diagnosis_slots(diag_slots, site_pred, diag_pred)

    # ── Segmentation stats ──────────────────────────────────────────────────
    seg_html = _render_segmentation_summary(seg_ctx, figures)

    # ── Knowledge context ───────────────────────────────────────────────────
    knowledge_html = _render_knowledge_context(payload.get("knowledge_context") or {})

    # ── Decision process timeline ───────────────────────────────────────────
    timeline_html = _render_decision_timeline(payload, diag_html)

    # ── Evidence graph visualization ────────────────────────────────────────
    graph_viz_html = _render_evidence_graph_viz(payload, graph)

    # ── Site / Diagnosis / Evidence Sufficiency cards ───────────────────────
    pred_cards = _render_prediction_cards(site_pred, diag_pred, evidence_suff, payload)
    medgemma_ctx = payload.get("medgemma_morphology_context") or {}
    medgemma_text = medgemma_ctx.get("description") or payload.get("medgemma_short_report") or ""
    medgemma_html = _render_medgemma_context(medgemma_text)
    claim_audit_html = _render_claim_audit(payload.get("claim_verification") or {}, figure_records)

    # ── Graph summary ───────────────────────────────────────────────────────
    n_slides = len(graph.get("slide_nodes") or [])
    n_rois = len(graph.get("roi_nodes") or [])
    n_edges = len(graph.get("edges") or [])
    graph_summary = f"Slides: {n_slides} &nbsp;|&nbsp; ROIs: {n_rois} &nbsp;|&nbsp; Edges: {n_edges}"
    if n_slides:
        slide_names = [n.get("slide_id","") for n in (graph.get("slide_nodes") or [])[:3]]
        graph_summary += f"<br><small>{', '.join(slide_names[:2])}{'...' if n_slides>2 else ''}</small>"
    inspector_index = _build_inspector_index(payload, figure_records, claims)
    inspector_json = json.dumps(_json_safe(inspector_index), ensure_ascii=False).replace("</", "<\\/")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Case Report: {case_id}</title>
<style>
:root {{
  --bg: #f1f5f9; --card: #ffffff; --border: #e2e8f0;
  --text: #1e293b; --muted: #64748b; --accent: #3b82f6; --accent2: #8b5cf6;
  --green: #16a34a; --red: #dc2626; --amber: #d97706; --teal: #0d9488;
  --green-bg: #dcfce7; --red-bg: #fee2e2; --amber-bg: #fef3c7; --teal-bg: #ccfbf1;
  --gray-bg: #f8fafc; --radius: 8px;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ scroll-behavior: smooth; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); line-height: 1.5; }}
a {{ color: var(--accent); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
/* ── Header ── */
.header {{ background: linear-gradient(135deg, #1e293b 0%, #334155 100%); color: white; padding: 20px 28px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }}
.header h1 {{ font-size: 1.5rem; font-weight: 700; }}
.header .badge {{ padding: 4px 12px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }}
.badge-case {{ background: #3b82f6; }} .badge-cancer {{ background: #8b5cf6; }}
/* ── Content ── */
.content {{ padding: 24px 28px; max-width: 1680px; margin: 0 auto; }}
.page-shell {{ display:grid; grid-template-columns:minmax(0,1fr) 360px; gap:18px; align-items:start; max-width:1860px; }}
.main-reader {{ min-width:0; }}
.inspector {{ position:sticky; top:14px; max-height:calc(100vh - 28px); overflow:auto; background:#fff; border:1px solid var(--border); border-radius:10px; box-shadow:0 10px 28px rgba(15,23,42,.08); }}
.inspector-head {{ padding:14px 16px; border-bottom:1px solid var(--border); background:#f8fafc; }}
.inspector-head h3 {{ font-size:.98rem; color:#0f172a; margin-bottom:3px; }}
.inspector-head p {{ font-size:.74rem; color:var(--muted); }}
.inspector-body {{ padding:14px 16px; }}
.inspector-type {{ display:inline-block; border-radius:999px; background:#e0f2fe; color:#075985; padding:2px 8px; font-size:.68rem; font-weight:700; margin-bottom:8px; }}
.inspector-title {{ font-size:1rem; font-weight:800; color:#0f172a; margin-bottom:7px; }}
.inspector-meta {{ font-size:.76rem; color:var(--muted); margin-bottom:10px; }}
.inspector-img {{ width:100%; border:1px solid var(--border); border-radius:8px; margin:8px 0; cursor:pointer; }}
.inspector-section {{ border-top:1px solid var(--border); padding-top:10px; margin-top:10px; }}
.inspector-section strong {{ display:block; font-size:.78rem; color:#334155; margin-bottom:4px; }}
.ref-chip {{ border:1px solid var(--border); background:white; color:#1e40af; border-radius:999px; padding:3px 8px; font-size:.68rem; font-weight:700; cursor:pointer; margin:2px; display:inline-flex; align-items:center; gap:4px; }}
.ref-chip:hover,.ref-chip.active {{ background:#eff6ff; border-color:#93c5fd; text-decoration:none; }}
.ref-chip.morph {{ color:#6d28d9; }}
.ref-chip.keep {{ color:#0f766e; }}
.ref-chip.claim {{ color:#b45309; }}
.ref-chip.decision {{ color:#334155; }}
.ref-row {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:7px; }}
.quicknav {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }}
.quicknav a {{ background:white; border:1px solid var(--border); border-radius:999px; padding:6px 12px; font-size:.78rem; color:var(--text); }}
.section-panel {{ margin-bottom:22px; }}
.section-head {{ margin:6px 0 14px; }}
.section-head h2 {{ font-size:1.15rem; color:#0f172a; margin-bottom:4px; }}
.section-head p {{ color:var(--muted); font-size:.84rem; max-width:980px; }}
.section-grid {{ display:grid; grid-template-columns:minmax(0,1fr); gap:16px; }}
.section-grid.two {{ grid-template-columns:minmax(0,.95fr) minmax(0,1.05fr); align-items:start; }}
.section-grid.report {{ grid-template-columns:minmax(0,1.2fr) minmax(320px,.8fr); align-items:start; }}
.section-title {{ font-size: 1.1rem; font-weight: 700; margin: 24px 0 14px; color: var(--text); }}
.section-title .count {{ font-size: 0.8rem; color: var(--muted); font-weight: 400; }}
/* ── Cards ── */
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); padding: 18px; margin-bottom: 16px; min-width:0; overflow:hidden; }}
.card h3 {{ font-size: 1rem; margin-bottom: 12px; color: #0f172a; font-weight: 700; cursor: pointer; user-select: none; display: flex; align-items: center; gap: 6px; border-bottom:1px solid var(--border); padding-bottom:9px; }}
.card h3::before {{ content: "▼"; font-size: 0.65rem; transition: transform .2s; }}
.card.collapsed h3::before {{ transform: rotate(-90deg); }}
.card.collapsed .card-body {{ display: none; }}
.card-body {{ }}
.card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom:16px; }}
.main-layout {{ display:grid; grid-template-columns: minmax(0,1.15fr) minmax(0,1fr); gap:18px; align-items:start; }}
.left-stack,.right-stack {{ display:flex; flex-direction:column; gap:16px; min-width:0; }}
/* ── WSI + Visual merged ── */
.wsi-viz {{ display:grid; grid-template-columns: minmax(0, .95fr) minmax(0, 1.35fr); gap:16px; align-items:start; min-width:0; }}
.wsi-slide-card {{ border:1px solid var(--border); border-radius:8px; background:#fff; padding:10px; margin-bottom:12px; scroll-margin-top:80px; }}
.wsi-slide-title {{ font-size:.76rem; font-weight:700; color:#334155; margin-bottom:6px; word-break:break-all; }}
.wsi-main img {{ border:1px solid var(--border); border-radius:6px; }}
.wsi-gallery {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; max-height:860px; overflow-y:auto; overflow-x:hidden; padding-right:4px; min-width:0; }}
.roi-card {{ background:white; border:1px solid var(--border); border-radius:8px; padding:8px; scroll-margin-top:90px; transition:box-shadow .15s, transform .15s; min-width:0; overflow:hidden; }}
.roi-card:hover,.roi-card:target {{ box-shadow:0 0 0 2px rgba(59,130,246,.25), 0 6px 20px rgba(15,23,42,.10); transform:translateY(-1px); }}
.roi-card summary {{ cursor:pointer; list-style:none; }}
.roi-card summary::-webkit-details-marker {{ display:none; }}
.roi-card img {{ border-radius:4px; max-width:100%; height:auto; display:block; }}
.roi-mini {{ width:100%; aspect-ratio:1 / 1; object-fit:cover; cursor:pointer; border:1px solid var(--border); }}
.roi-expanded {{ margin-top:8px; display:grid; grid-template-columns:1fr; gap:8px; min-width:0; overflow:hidden; }}
.roi-card:not([open]) .roi-expanded {{ display:none; }}
.qc-badge {{ font-size:0.65rem; padding:1px 6px; border-radius:3px; font-weight:600; white-space:nowrap; }}
.qc-good {{ background:#dcfce7; color:#166534; }}
.qc-warn {{ background:#fef3c7; color:#92400e; }}
.qc-rej {{ background:#fee2e2; color:#991b1b; }}
.roi-hint {{ font-size:.68rem; color:var(--muted); margin-top:5px; }}
.roi-img-label {{ font-size:.68rem; color:var(--muted); font-weight:700; margin:4px 0 3px; }}
/* ── Interactive Report ── */
.report-block {{ border-left: 4px solid var(--border); padding: 10px 14px; margin-bottom: 12px; background: var(--gray-bg); border-radius: 0 6px 6px 0; scroll-margin-top:90px; }}
.report-block:target {{ border-left-color: var(--accent); background: #eff6ff; }}
.report-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }}
.report-header strong {{ font-size: 0.88rem; color: #1e40af; }}
.report-body {{ font-size: 0.88rem; white-space: pre-wrap; color:#1f2937; }}
.report-reason {{ font-size: 0.72rem; color: var(--muted); margin-top: 4px; font-style: italic; }}
.report-visual-strip {{ border:1px solid var(--border); background:#f8fafc; border-radius:8px; padding:10px; margin-bottom:14px; }}
.report-visual-strip-title {{ font-size:.78rem; font-weight:700; color:#334155; margin-bottom:8px; }}
.report-thumb-row {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
.report-thumb {{ display:inline-flex; flex-direction:column; align-items:center; gap:3px; text-decoration:none; color:#475569; font-size:.66rem; background:transparent; border:0; cursor:pointer; }}
.report-thumb img {{ width:52px; height:52px; object-fit:cover; border-radius:5px; border:1px solid var(--border); background:white; }}
.report-thumb:hover img {{ box-shadow:0 0 0 2px rgba(59,130,246,.35); }}
.report-thumb-more {{ font-size:.72rem; color:var(--muted); padding:4px 6px; }}
/* ── Two-col ── */
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
@media(max-width: 900px) {{ .two-col,.wsi-viz {{ grid-template-columns: 1fr; }} }}
/* ── Tables ── */
table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
th {{ background: var(--gray-bg); text-align: left; padding: 8px 10px; font-weight: 600; color: var(--muted); text-transform: uppercase; font-size: 0.75rem; }}
td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
tr:hover td {{ background: #f8fafc; }}
.status {{ display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; }}
.status.supported {{ background: var(--green-bg); color: var(--green); }}
.status.weak {{ background: var(--amber-bg); color: var(--amber); }}
.status.unsupported,.status.contradicted {{ background: var(--red-bg); color: var(--red); }}
.status.not_assessable {{ background: var(--teal-bg); color: var(--teal); }}
/* ── Figure Gallery ── */
.gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; }}
.figure-card {{ background: var(--card); border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; }}
.figure-card img {{ width: 100%; display: block; cursor: pointer; }}
.figure-card .caption {{ padding: 10px 14px; font-size: 0.8rem; color: var(--muted); }}
.figure-card .fig-header {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; border-bottom: 1px solid var(--border); }}
.figure-card .fig-header strong {{ font-size: 0.85rem; }}
.figure-card .dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 4px; }}
/* ── Diagnosis Slots ── */
.slot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
@media(max-width: 768px) {{ .slot-grid {{ grid-template-columns: 1fr; }} }}
.slot-item {{ background: var(--gray-bg); padding: 10px 14px; border-radius: 8px; border:1px solid var(--border); }}
.slot-item .slot-label {{ font-size: 0.82rem; font-weight: 700; color:#1e40af; }}
.slot-item .slot-value {{ font-size: 0.9rem; margin-top: 4px; color:#0f172a; }}
.slot-section {{ border:1px solid var(--border); border-radius:8px; overflow:hidden; margin-bottom:12px; background:#fff; }}
.slot-section h5 {{ margin:0; padding:9px 12px; font-size:.86rem; color:#0f172a; background:#f8fafc; border-bottom:1px solid var(--border); }}
.slot-section .slot-section-note {{ padding:8px 12px 0; font-size:.72rem; color:var(--muted); }}
.slot-section.judgement {{ border-left:4px solid var(--amber); background:#fffbeb; }}
.slot-section.judgement h5 {{ background:#fef3c7; }}
.slot-card {{ padding:10px 12px; border-top:1px solid #eef2f7; background:#fff; }}
.slot-card:first-of-type {{ border-top:none; }}
.slot-title {{ font-size:.84rem; font-weight:700; color:#334155; margin-bottom:4px; }}
.slot-body {{ font-size:.88rem; color:#1f2937; white-space:pre-wrap; }}
.slot-note {{ font-size:.68rem; color:var(--muted); margin-top:4px; }}
.audit-subblock {{ margin-top:16px; padding-top:14px; border-top:1px solid var(--border); }}
.audit-subblock h4 {{ font-size:.95rem; color:#0f172a; margin-bottom:8px; }}
.medgemma-grid {{ display:grid; grid-template-columns:1fr; gap:10px; }}
.medgemma-part {{ border:1px solid var(--border); border-left:4px solid var(--accent2); border-radius:8px; padding:10px 12px; background:white; }}
.medgemma-part strong {{ display:block; color:#4c1d95; font-size:.84rem; margin-bottom:4px; }}
.medgemma-part div {{ white-space:pre-wrap; color:#1f2937; font-size:.86rem; line-height:1.55; }}
/* ── Seg Stats ── */
.class-bar {{ display: flex; align-items: center; gap: 10px; margin: 4px 0; }}
.class-bar .label {{ width: 120px; font-size: 0.8rem; text-align: right; color: var(--muted); }}
.class-bar .bar-wrap {{ flex: 1; height: 20px; border-radius: 4px; background:#e2e8f0; overflow:hidden; }}
.class-bar .bar-fill {{ display:block; height:100%; min-width:2px; border-radius:4px; }}
.class-bar .pct {{ width: 50px; font-size: 0.75rem; text-align: right; font-weight: 600; }}
.seg-gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 10px; }}
/* ── Score bar ── */
.score-bar {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; }}
.score-bar .name {{ width: 140px; font-size: 0.78rem; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); }}
.score-bar .bar-bg {{ flex: 1; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }}
.score-bar .bar-fill {{ height: 100%; border-radius: 3px; background: var(--accent); }}
.score-bar .val {{ width: 48px; font-size: 0.72rem; font-weight: 600; }}
/* ── Lightbox ── */
.lightbox {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,.9); z-index: 9999; justify-content: center; align-items: center; cursor: pointer; }}
.lightbox.active {{ display: flex; }}
.lightbox img {{ max-width: 95vw; max-height: 95vh; border-radius: 6px; }}
.lightbox .close {{ position: absolute; top: 20px; right: 30px; color: white; font-size: 2rem; cursor: pointer; }}
/* ── Report sections ── */
.report-section {{ margin-bottom: 14px; }}
.report-section h4 {{ font-size: 0.85rem; color: var(--accent); margin-bottom: 4px; }}
.report-section p {{ font-size: 0.85rem; white-space: pre-wrap; }}
/* ── Prediction cards ── */
.pred-card {{ background: var(--card); border-left: 4px solid var(--accent); padding: 14px 18px; border-radius: 0 var(--radius) var(--radius) 0; margin-bottom: 12px; }}
.pred-card h4 {{ font-size: 0.92rem; color: #0f172a; font-weight: 700; margin-bottom: 6px; }}
.pred-card .value {{ font-size: 1.05rem; font-weight: 600; }}
.pred-card .meta {{ font-size: 0.75rem; color: var(--muted); margin-top: 4px; }}
.conf-high {{ border-left-color: var(--green); }} .conf-medium {{ border-left-color: var(--amber); }} .conf-low {{ border-left-color: var(--red); }}
/* ── Metric cards ── */
.metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
.metric {{ text-align: center; padding: 14px; background: var(--gray-bg); border-radius: 8px; }}
.metric .num {{ font-size: 1.5rem; font-weight: 700; color: var(--accent); }}
.metric .lbl {{ font-size: 0.7rem; color: var(--muted); text-transform: uppercase; margin-top: 2px; }}
/* ── Timeline ── */
.timeline {{ position: relative; padding-left: 4px; }}
.tl-item {{ display: flex; gap: 12px; align-items: flex-start; margin-bottom: 0; }}
.tl-dot {{ width: 14px; height: 14px; border-radius: 50%; margin-top: 4px; flex-shrink: 0; border: 3px solid white; box-shadow: 0 0 0 2px var(--border); }}
.tl-card {{ background: var(--gray-bg); border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; flex: 1; }}
.tl-card summary {{ cursor:pointer; list-style:none; }}
.tl-card summary::-webkit-details-marker {{ display:none; }}
.tl-card summary strong {{ font-size: 0.82rem; display: block; margin-bottom: 2px; }}
.tl-card summary strong::before {{ content:"▾"; color:var(--muted); margin-right:6px; font-size:.72rem; }}
.tl-card:not([open]) summary strong::before {{ content:"▸"; }}
.tl-detail {{ font-size:.82rem; color:#1e293b; margin-top:2px; }}
.tl-meta {{ font-size:0.72rem; color:var(--muted); margin-top:4px; }}
.tl-steps {{ margin:8px 0 0 0; padding:0; list-style:none; display:grid; gap:5px; }}
.tl-steps li {{ display:flex; gap:7px; align-items:flex-start; font-size:.76rem; color:#334155; background:white; border:1px solid var(--border); border-radius:6px; padding:6px 8px; }}
.tl-step-tag {{ flex:0 0 auto; font-weight:700; color:#475569; min-width:58px; }}
.tl-chips {{ display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; }}
.tl-chip {{ border:1px solid var(--border); background:#fff; border-radius:999px; padding:3px 8px; font-size:.7rem; color:#475569; }}
.tl-connector {{ width: 2px; background: var(--border); height: 12px; margin-left: 6px; }}
.tl-node-title {{ display:flex; align-items:center; justify-content:space-between; gap:10px; }}
.tl-node-kind {{ font-size:.66rem; color:var(--muted); border:1px solid var(--border); background:#fff; border-radius:999px; padding:2px 7px; white-space:nowrap; }}
/* ── DAG ── */
.dag-container {{ display: flex; flex-direction: column; gap: 10px; }}
.dag-layer-label {{ font-size: 0.7rem; font-weight: 700; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }}
.dag-chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.dag-node {{ background: var(--card); border: 2px solid var(--border); border-radius: 8px; padding: 8px 12px; max-width: 230px; color:var(--text); }}
.dag-node strong {{ font-size: 0.78rem; display: block; }}
.ev-dag {{ border-style: dashed; }}
.dag-arrows {{ display: flex; flex-wrap: wrap; gap: 6px; padding: 8px 16px; background: var(--gray-bg); border-radius: 6px; font-size: 0.72rem; }}
.dag-edge {{ margin: 2px 8px; }}
.graph-svg-wrap {{ width:100%; overflow-x:auto; border:1px solid var(--border); border-radius:8px; background:linear-gradient(180deg,#ffffff,#f8fafc); }}
.graph-svg {{ min-width:760px; width:100%; height:auto; display:block; }}
.flow-summary {{ display:flex; flex-wrap:wrap; gap:8px; margin:8px 0 12px; }}
.flow-summary span {{ background:#f8fafc; border:1px solid var(--border); border-radius:999px; padding:4px 9px; font-size:.74rem; color:#475569; }}
.graph-node rect {{ fill:#fff; stroke:#cbd5e1; stroke-width:2; rx:8; }}
.graph-node.roi rect {{ stroke:#3b82f6; }}
.graph-node.inf rect {{ stroke:#8b5cf6; stroke-dasharray:5 3; }}
.graph-node.input rect {{ stroke:#0d9488; fill:#f0fdfa; }}
.graph-node.derived rect {{ stroke:#f59e0b; fill:#fffbeb; }}
.graph-node.context rect {{ stroke:#7c3aed; fill:#faf5ff; }}
.graph-node.report rect {{ stroke:#16a34a; fill:#f0fdf4; }}
.graph-node text {{ font-size:13px; fill:#0f172a; font-weight:700; }}
.graph-node .sub {{ font-size:10px; fill:#64748b; font-weight:400; }}
.graph-edge {{ stroke:#94a3b8; stroke-width:1.6; fill:none; marker-end:url(#arrow); opacity:.85; }}
.kg-svg .query rect {{ stroke:#8b5cf6; fill:#faf5ff; }}
.kg-svg .disease rect {{ stroke:#7c3aed; fill:#fff; }}
.kg-svg .semantic rect {{ stroke:#0d9488; fill:#f0fdfa; }}
/* ── ROI ranking ── */
.ranking-table td:nth-child(2) {{ font-family: monospace; font-size: 0.78rem; }}
.kg-query {{ border:1px solid #c4b5fd; background:#faf5ff; border-radius:8px; padding:10px 12px; margin-bottom:12px; }}
.kg-query strong {{ display:block; color:#312e81; font-size:.82rem; margin-bottom:3px; }}
.knowledge-card-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; }}
.kg-hit {{ border:1px solid var(--border); border-left:4px solid var(--accent2); border-radius:8px; background:white; padding:10px 12px; min-width:0; }}
.kg-hit.semantic {{ border-left-color:var(--teal); background:#f0fdfa; }}
.kg-hit strong {{ display:block; font-size:.84rem; color:#312e81; margin-bottom:4px; }}
.kg-hit.semantic strong {{ color:#115e59; }}
.kg-hit .meta {{ font-size:.72rem; color:var(--muted); margin-top:4px; }}
.kg-hit .detail {{ font-size:.78rem; color:#334155; margin-top:6px; }}
.decision-state-flow {{ display:flex; flex-direction:column; gap:16px; }}
.state-after-timeline {{ border-top:1px solid var(--border); padding-top:14px; }}
@media(max-width: 1120px) {{ .main-layout,.wsi-viz,.two-col,.section-grid.two,.section-grid.report,.page-shell {{ grid-template-columns:1fr; }} .inspector {{ position:relative; top:auto; max-height:none; order:-1; }} }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>🧬 Case Report: {case_id}</h1>
    <div style="margin-top:6px;display:flex;gap:8px;">
      <span class="badge badge-case">{case_id}</span>
      {f'<span class="badge badge-cancer">{cancer_type}</span>' if cancer_type else ''}
    </div>
  </div>
  <div style="text-align:right;font-size:0.8rem;opacity:0.8;">
    {graph_summary}
  </div>
</div>

<div class="content page-shell">
<main class="main-reader">
  <div class="quicknav">
    <a href="#section-report">Report</a>
    <a href="#section-visual">Visual Evidence</a>
    <a href="#section-knowledge">Additional Knowledge</a>
    <a href="#section-audit">Decision and Audit</a>
  </div>

  <section class="section-panel" id="section-report">
    <div class="section-head">
      <h2>1. Generated Report</h2>
      <p>Final case-level output. The report is claim-filtered where possible; the right column keeps the structured diagnosis state and headline model predictions.</p>
    </div>
    <div class="section-grid report">
      <div class="card" id="report">
        <h3>Claim-Verified Report</h3>
        <div class="card-body">{interactive_report_html}</div>
      </div>
      <div>{pred_cards}</div>
    </div>
  </section>

  <section class="section-panel" id="section-visual">
    <div class="section-head">
      <h2>2. Visual Evidence</h2>
      <p>WSI-level localization, selected ROI patches, segmentation overlays, and D1 tissue statistics used as visual evidence.</p>
    </div>
    <div class="section-grid">
      {combined_wsi_html}
      {seg_html}
    </div>
  </section>

  <section class="section-panel" id="section-knowledge">
    <div class="section-head">
      <h2>3. Additional Knowledge</h2>
      <p>Auxiliary model description and retrieved KEEP context. These are supporting context, not ground truth and not direct visual evidence.</p>
    </div>
    <div class="section-grid two">
      {medgemma_html}
      {knowledge_html}
    </div>
  </section>

  <section class="section-panel" id="section-audit">
    <div class="section-head">
      <h2>4. Model Decision and Audit</h2>
      <p>How the agent selected evidence, where evidence was insufficient, and which claims are supported, weak, unsupported, or not assessable.</p>
    </div>
    <div class="section-grid two">
      {graph_viz_html}
      {timeline_html}
    </div>
    {claim_audit_html}
  </section>
</main>
<aside class="inspector" id="inspector">
  <div class="inspector-head">
    <h3>Evidence Inspector</h3>
    <p>Click any Fig, Claim, Decision, Morphology, or KEEP chip to inspect details without leaving your reading position.</p>
  </div>
  <div class="inspector-body" id="inspector-body">
    <span class="inspector-type">Case</span>
    <div class="inspector-title">{case_id}</div>
    <div class="inspector-meta">Select a reference chip in the report, support map, decision process, or evidence library.</div>
  </div>
</aside>
</div>

<!-- Lightbox -->
<div class="lightbox" id="lightbox" onclick="closeLightbox()">
  <span class="close">&times;</span>
  <img id="lightbox-img" src="" alt="">
</div>

<script>
const REF_INDEX = {inspector_json};

function escHtml(value) {{
  return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}
function renderList(items) {{
  if (!Array.isArray(items) || !items.length) return '<span style="color:var(--muted)">none</span>';
  return '<ul style="padding-left:18px;font-size:.82rem;">' + items.slice(0, 12).map(x => '<li>' + escHtml(x) + '</li>').join('') + '</ul>';
}}
function renderObject(obj) {{
  if (!obj || typeof obj !== 'object') return escHtml(obj || '');
  return '<pre style="white-space:pre-wrap;font-size:.72rem;background:#f8fafc;border:1px solid var(--border);border-radius:8px;padding:8px;max-height:260px;overflow:auto;">' + escHtml(JSON.stringify(obj, null, 2)) + '</pre>';
}}
function cssAttrEscape(value) {{
  if (window.CSS && CSS.escape) return CSS.escape(value);
  return String(value).replace(/["\\\\]/g, '\\\\$&');
}}
function showRef(refId) {{
  const item = REF_INDEX[refId];
  const body = document.getElementById('inspector-body');
  document.querySelectorAll('.ref-chip.active').forEach(el => el.classList.remove('active'));
  document.querySelectorAll(`[data-ref="${{cssAttrEscape(refId)}}"]`).forEach(el => el.classList.add('active'));
  if (!item) {{
    body.innerHTML = '<span class="inspector-type">Missing</span><div class="inspector-title">' + escHtml(refId) + '</div><div class="inspector-meta">No inspector record was saved for this reference.</div>';
    return;
  }}
  let html = '<span class="inspector-type">' + escHtml(item.type || 'Evidence') + '</span>';
  html += '<div class="inspector-title">' + escHtml(item.title || refId) + '</div>';
  if (item.meta) html += '<div class="inspector-meta">' + escHtml(item.meta) + '</div>';
  if (item.image) html += '<img class="inspector-img" src="' + item.image + '" onclick="openLightbox(this.src)" alt="">';
  if (item.summary) html += '<div class="inspector-section"><strong>Summary</strong><div style="font-size:.84rem;white-space:pre-wrap;">' + escHtml(item.summary) + '</div></div>';
  if (item.details) html += '<div class="inspector-section"><strong>Details</strong>' + renderObject(item.details) + '</div>';
  if (item.supports) html += '<div class="inspector-section"><strong>Supports / used by</strong>' + renderList(item.supports) + '</div>';
  if (item.related && item.related.length) {{
    html += '<div class="inspector-section"><strong>Related</strong><div class="ref-row">' + item.related.map(r => '<button class="ref-chip" data-ref="' + escHtml(r.ref) + '">' + escHtml(r.label || r.ref) + '</button>').join('') + '</div></div>';
  }}
  body.innerHTML = html;
}}

document.addEventListener('click', function(ev) {{
  const chip = ev.target.closest('[data-ref]');
  if (!chip) return;
  ev.preventDefault();
  ev.stopPropagation();
  showRef(chip.getAttribute('data-ref'));
}});

// Auto-wrap card content for collapsible sections
document.querySelectorAll('.card').forEach(card => {{
  const h3 = card.querySelector('h3');
  if (!h3) return;
  // Wrap everything after h3 in card-body if not already wrapped
  let body = card.querySelector('.card-body');
  if (!body) {{
    body = document.createElement('div');
    body.className = 'card-body';
    let el = h3.nextSibling;
    while (el) {{
      const next = el.nextSibling;
      body.appendChild(el);
      el = next;
    }}
    card.appendChild(body);
  }}
  h3.onclick = function() {{ card.classList.toggle('collapsed'); }};
}});

function openLightbox(src) {{
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('active');
}}
function closeLightbox() {{
  document.getElementById('lightbox').classList.remove('active');
}}
</script>
</body></html>"""


# ── Helper builders ────────────────────────────────────────────────────────────

def _build_combined_wsi_visual(payload: dict, figures: list[dict]) -> str:
    """WSI annotated thumbnail + interactive ROI gallery, merged into one section."""
    overviews = payload.get("wsi_overviews") or []
    seg_ctx = payload.get("segmentation_context") or {}
    seg_qc = seg_ctx.get("seg_qc") or {}
    per_roi_qc = seg_qc.get("per_roi") or {}

    # ── Tissue class color legend ──
    class_names = seg_ctx.get("class_names") or {}
    _TISSUE_COLORS = {
        0: "#000000", 1: "#E41A1C", 2: "#377EB8", 3: "#FF7F00",
        4: "#4DAF4A", 5: "#984EA3", 6: "#A65628",
    }
    legend_items = []
    for cid in sorted(class_names.keys(), key=int):
        cls_name = class_names.get(cid, f"Class {cid}")
        color = _TISSUE_COLORS.get(int(cid), "#94a3b8")
        legend_items.append(
            f'<span style="display:inline-flex;align-items:center;gap:3px;margin-right:10px;font-size:0.72rem;">'
            f'<span style="width:10px;height:10px;border-radius:2px;background:{color};display:inline-block;"></span>'
            f'{_esc(cls_name)}'
            f'</span>'
        )
    legend_html = '<div style="margin-bottom:10px;padding:6px 10px;background:#f8fafc;border-radius:4px;">'
    legend_html += '<span style="font-size:0.72rem;color:var(--muted);margin-right:8px;">Seg classes:</span>'
    legend_html += "".join(legend_items) + "</div>"

    # ── QC badge helper ──
    def _qc_badge(roi_index: int) -> str:
        qc = per_roi_qc.get(str(roi_index))
        if not qc:
            missing = {str(x) for x in seg_qc.get("missing_roi_indices") or []}
            if str(roi_index) in missing or (seg_qc.get("available") and per_roi_qc):
                return ' <span title="Qwen QC did not return an assessment for this ROI" style="background:#f1f5f9;color:#475569;font-size:0.65rem;padding:1px 5px;border-radius:3px;">not checked</span>'
            return ""
        quality = qc.get("quality", "")
        reason = _esc(str(qc.get("reason", ""))[:80])
        if quality == "reject":
            return f' <span title="{reason}" style="background:#fee2e2;color:#991b1b;font-size:0.65rem;padding:1px 5px;border-radius:3px;font-weight:700;">REJECT</span>'
        elif quality == "questionable":
            return f' <span title="{reason}" style="background:#fef3c7;color:#92400e;font-size:0.65rem;padding:1px 5px;border-radius:3px;font-weight:700;">⚠ QUESTIONABLE</span>'
        elif quality == "good":
            return f' <span title="{reason}" style="background:#dcfce7;color:#166534;font-size:0.65rem;padding:1px 5px;border-radius:3px;">✓ good</span>'
        return ""

    # Build ROI figure cards (compact, clickable to highlight on WSI)
    fig_cards = []
    for i, fig in enumerate(figures):
        img = _img_data_uri(_figure_image_path(fig), max_edge=480)
        if not img:
            continue
        seg = fig.get("segmentation") if isinstance(fig.get("segmentation"), dict) else {}
        overlay = _img_data_uri(_figure_overlay_path(fig), max_edge=480)
        caption = _esc((fig.get("caption") or "")[:120])
        color = _ROI_PALETTE_HEX[i % len(_ROI_PALETTE_HEX)]
        score = fig.get("score", "")
        score_str = f"{float(score):.3f}" if score else ""
        slide_id = _esc(fig.get("slide_id") or "")
        fig_n = int(fig.get("figure_number") or i + 1)

        if overlay:
            pair = (
                f'<div class="roi-expanded">'
                f'<div class="roi-img-label">H&E ROI</div>'
                f'<img src="{img}" onclick="openLightbox(&quot;{img}&quot;)" style="cursor:pointer;width:100%;">'
                f'<div class="roi-img-label">Segmentation overlay{_qc_badge(fig_n)}</div>'
                f'<img src="{overlay}" onclick="openLightbox(&quot;{overlay}&quot;)" style="cursor:pointer;width:100%;">'
                f'</div>'
            )
        else:
            pair = (
                f'<div class="roi-expanded">'
                f'<div class="roi-img-label">H&E ROI</div>'
                f'<img src="{img}" onclick="openLightbox(&quot;{img}&quot;)" style="cursor:pointer;width:100%;">'
                f'</div>'
            )

        is_zoom = bool(fig.get("is_zoom"))
        zoom_badge = ' <span style="background:#ede9fe;color:#6b21a8;font-size:0.6rem;padding:1px 4px;border-radius:3px;font-weight:700;">ZOOM</span>' if is_zoom else ""

        inspect_btn = _ref_button(f"fig-{fig_n}", "Inspect Figure")
        fig_cards.append(f"""
<details class="roi-card" id="roi-{fig_n}" style="border-left:4px solid {color};" open>
  <summary>
    <div style="font-weight:700;margin-bottom:6px;display:flex;align-items:center;justify-content:space-between;gap:8px;">
      <span><span class="dot" style="background:{color};"></span> Figure {fig_n}{_qc_badge(fig_n)}{zoom_badge}</span>
      {f'<span style="font-size:0.72rem;color:var(--muted);font-weight:400;">score={score_str}</span>' if score_str else ''}
    </div>
    <div class="roi-hint">Click header to collapse/expand, or inspect this evidence in the right panel. {inspect_btn}</div>
  </summary>
  {pair}
  <div style="font-size:0.68rem;color:var(--muted);margin-top:5px;word-break:break-all;">Slide: <a href="#slide-{_safe_id(slide_id)}">{slide_id}</a></div>
  {f'<div style="font-size:0.72rem;color:var(--muted);margin-top:4px;">{caption}</div>' if caption else ''}
</details>""")

    overview_cards = []
    if overviews:
        fig_by_num = {int(f.get("figure_number") or i + 1): f for i, f in enumerate(figures)}
        for ov in overviews:
            path = ov.get("annotated_path") or ov.get("thumbnail_path")
            img = _img_data_uri(path, max_edge=800) if path else ""
            if not img:
                continue
            slide_id = str(ov.get("slide_id") or "")
            chips = []
            for num in ov.get("figure_numbers") or []:
                color = _ROI_PALETTE_HEX[(int(num) - 1) % len(_ROI_PALETTE_HEX)]
                fig = fig_by_num.get(int(num), {})
                cap = _esc((fig.get("caption") or "")[:42])
                chips.append(
                    f'<button type="button" data-ref="fig-{int(num)}" style="display:inline-flex;align-items:center;gap:4px;margin:2px 5px 2px 0;border:0;background:transparent;cursor:pointer;">'
                    f'<span style="background:{color};color:white;border-radius:999px;padding:1px 7px;font-size:.68rem;font-weight:700;">Fig {int(num)}</span>'
                    f'<span style="font-size:.68rem;color:var(--muted);">{cap}</span></button>'
                )
            overview_cards.append(f"""
<div class="wsi-slide-card" id="slide-{_safe_id(slide_id)}">
  <div class="wsi-slide-title">{_esc(slide_id)}</div>
  <img src="{img}" alt="WSI overview {slide_id}" onclick="openLightbox(&quot;{img}&quot;)" style="cursor:pointer;width:100%;">
  <div style="margin-top:7px;">{''.join(chips)}</div>
</div>""")

    if not overview_cards:
        thumb = payload.get("wsi_thumbnail_annotated_path") or payload.get("wsi_thumbnail_path")
        thumb_img = _img_data_uri(thumb, max_edge=800) if thumb else ""
        if thumb_img:
            fallback_inner = (
                f'<img src="{thumb_img}" onclick="openLightbox(&quot;{thumb_img}&quot;)" '
                f'style="cursor:pointer;width:100%;">'
            )
        else:
            fallback_inner = (
                '<div style="color:var(--muted);padding:40px;text-align:center;">'
                'WSI thumbnail not available</div>'
            )
        overview_cards.append(f'<div class="wsi-slide-card">{fallback_inner}</div>')

    return f"""<div class="card" id="visual-evidence">
<h3>WSI Overview and Visual Evidence ({len(figures)} ROIs)</h3>
<div class="wsi-viz">
  <div class="wsi-main">{"".join(overview_cards)}</div>
  {legend_html}
  <div class="wsi-gallery">{"".join(fig_cards)}</div>
</div>
</div>"""


def _render_interactive_report(sections: dict[str, str], claims: list[dict], figures: list[dict]) -> str:
    """Report sections with inline evidence — clicking shows supporting ROIs."""
    claim_by_field = {}
    for c in claims:
        claim_by_field[str(c.get("field") or "").lower()] = c

    field_to_label = {
        "site": ("Specimen/Site", "📍"),
        "primary_diagnosis": ("Diagnosis", "🩺"),
        "tumor_architecture": ("Tumor/Lesion Description", "🔬"),
        "cytology": ("Key Microscopic Findings", "🔬"),
        "necrosis": ("Invasion/Necrosis/Margins", "⚠️"),
        "ihc_molecular": ("Immunohistochemistry/Ancillary Studies", "🧪"),
        "grade": ("Stage/Grade/Prognostic Features", "📊"),
        "summary": ("Final Summary", "📝"),
    }
    # Reverse lookup
    label_to_field = {v[0]: k for k, v in field_to_label.items()}

    fig_by_id = {str(f.get("roi_id")): f for f in figures}
    fig_by_id.update({f"roi::{str(f.get('roi_id'))}": f for f in figures if f.get("roi_id")})
    parts = []

    for sec_name, body in sections.items():
        field_key = label_to_field.get(sec_name, "")
        claim = claim_by_field.get(field_key)
        status = str(claim.get("status") or "") if claim else ""
        evidence = claim.get("evidence") or [] if claim else []

        # Build evidence inline: small thumbnails
        ev_html = ""
        if evidence:
            ev_items = []
            for ev in evidence:
                ev_str = str(ev)
                fig = fig_by_id.get(ev_str)
                if fig:
                    idx = int(fig.get("figure_number") or (figures.index(fig) + 1 if fig in figures else 0) or 0)
                    thumb = _img_data_uri(_figure_image_path(fig), max_edge=120)
                    if thumb:
                        ev_items.append(
                            f'<button type="button" class="report-thumb" data-ref="fig-{idx}" style="margin:2px;">'
                            f'<img src="{thumb}" style="width:40px;height:40px;object-fit:cover;border-radius:3px;border:1px solid var(--border);" title="Fig {idx}">'
                            f'<span style="font-size:0.6rem;display:block;text-align:center;">Fig{idx}</span></button>'
                        )
                elif ev_str.startswith("evidence::"):
                    ev_ref = "decision-claims"
                    if ev_str.endswith("knowledge"):
                        ev_ref = "keep-main"
                    elif "morphology" in ev_str or "visual_description" in ev_str:
                        ev_ref = "morphology-main"
                    elif "segmentation" in ev_str:
                        ev_ref = "decision-segmentation"
                    ev_items.append(
                        _ref_button(ev_ref, ev_str, "keep" if ev_ref == "keep-main" else "")
                    )
            if ev_items:
                ev_html = f'<div style="margin-top:6px;display:flex;flex-wrap:wrap;align-items:center;gap:2px;">{"".join(ev_items)}</div>'

        status_badge = f'<span class="status {status}" style="font-size:0.65rem;">{status}</span>' if status else ""
        reason = _esc(str(claim.get("reason") or "")[:200]) if claim else ""

        parts.append(f"""
<div class="report-block" id="report-section-{_safe_id(sec_name)}">
  <div class="report-header">
    <strong>{_esc(sec_name)}</strong>{status_badge}
  </div>
  <div class="report-body">{_esc(body)}</div>
  {f'<div class="report-reason">{reason}</div>' if reason else ''}
  {ev_html}
</div>""")

    return "".join(parts) if parts else f'<pre style="white-space:pre-wrap;font-size:0.85rem;">{_esc(sections.get("_full",""))}</pre>'


def _render_report_visual_strip(figures: list[dict], max_items: int = 12) -> str:
    """Always show visual evidence thumbnails in the generated-report panel.

    The generated text may or may not contain explicit [Figure n] citations
    depending on the LLM response.  This strip keeps the report visually linked
    to the selected ROI evidence regardless of citation style.
    """
    if not figures:
        return ""
    items = []
    for i, fig in enumerate(figures[:max_items], start=1):
        fig_n = int(fig.get("figure_number") or i)
        thumb = _img_data_uri(_figure_image_path(fig), max_edge=140)
        if not thumb:
            continue
        bucket = _esc(fig.get("evidence_bucket") or fig.get("evidence_role") or "visual evidence")
        items.append(
            f'<button type="button" class="report-thumb" data-ref="fig-{fig_n}" title="Inspect Figure {fig_n}: {bucket}">'
            f'<img src="{thumb}" alt="Figure {fig_n} thumbnail">'
            f'<span>Fig {fig_n}</span></button>'
        )
    if not items:
        return ""
    more = ""
    if len(figures) > max_items:
        more = f'<span class="report-thumb-more">+{len(figures) - max_items} more</span>'
    return f"""
<div class="report-visual-strip">
  <div class="report-visual-strip-title">Visual evidence used by this report</div>
  <div class="report-thumb-row">{"".join(items)}{more}</div>
</div>"""


def _parse_report_sections(report: str) -> dict[str, str]:
    """Parse a section-headered report into a dict."""
    sections = {}
    pattern = re.compile(
        r"(?im)^\s*(Specimen/Site|Diagnosis|Tumor/Lesion Description|"
        r"Key Microscopic Findings|Invasion/Necrosis/Margins|"
        r"Immunohistochemistry/Ancillary Studies|Stage/Grade/Prognostic Features|"
        r"Final Summary)\s*:\s*"
    )
    parts = pattern.split(report)
    if len(parts) <= 1:
        return {"_full": report}
    # First part is before any header
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        # Strip trailing section header from body
        body = pattern.sub("", body).strip()
        sections[header] = body
    return sections


def _render_diagnosis_slots(diag_slots: dict, site_pred: dict, diag_pred: dict) -> str:
    """Render intermediate diagnosis slots as grouped audit state."""
    slots = diag_slots if isinstance(diag_slots, dict) else {}
    if not slots:
        # Fallback from site/diag predictions
        slots = {
            "site": site_pred.get("predicted_site") or "—",
            "primary_diagnosis": diag_pred.get("primary_diagnosis") or "—",
            "differential_diagnosis": diag_pred.get("differential") or [],
            "confidence": diag_pred.get("confidence") or "—",
        }

    def fmt(v):
        if isinstance(v, list):
            return ", ".join(str(x) for x in v)
        return str(v) if v else "—"

    groups = [
        {
            "class": "key",
            "title": "Key Results",
            "note": "Case-level site input and primary diagnostic hypothesis used to control the final report.",
            "fields": [("site", "Site"), ("primary_diagnosis", "Primary Diagnosis")],
        },
        {
            "class": "findings",
            "title": "Morphologic Findings",
            "note": "Image-derived or evidence-derived morphology slots. These should be supported by ROI or segmentation evidence.",
            "fields": [
                ("tumor_architecture", "Architecture"),
                ("cytology", "Cytology"),
                ("necrosis", "Necrosis"),
                ("invasion", "Invasion"),
                ("stroma_inflammation", "Stroma / Inflammation"),
                ("ihc_molecular", "IHC / Molecular"),
                ("grade", "Grade / Stage"),
            ],
        },
        {
            "class": "judgement",
            "title": "Judgement and Limits",
            "note": "Interpretive uncertainty, alternatives considered, and evidence that remains missing.",
            "fields": [
                ("differential_diagnosis", "Differential"),
                ("confidence", "Slot Confidence"),
                ("missing_evidence", "Missing Evidence"),
                ("recommended_ancillary_studies", "Recommended Ancillary Studies"),
            ],
        },
    ]

    sections = []
    for group in groups:
        items = []
        for key, label in group["fields"]:
            val = slots.get(key)
            if val is None or val == "":
                continue
            cls = "slot-card"
            items.append(
                f'<div class="{cls}" id="slot-{_safe_id(key)}">'
                f'<div class="slot-title">{_esc(label)}</div>'
                f'<div class="slot-body">{_esc(fmt(val))}</div>'
                f'</div>'
            )
        if items:
            sections.append(
                f'<div class="slot-section {group["class"]}">'
                f'<h5>{_esc(group["title"])}</h5>'
                f'<div class="slot-section-note">{_esc(group["note"])}</div>'
                f'{"".join(items)}'
                f'</div>'
            )

    return f"""<div class="audit-subblock" id="diagnosis-slots-audit">
<h4>Structured Diagnosis State</h4>
<p style="font-size:.78rem;color:var(--muted);margin-bottom:12px;">
  Intermediate state before claim verification, grouped by role so it is easier to audit what is a result, what is morphology, and what is uncertainty.
</p>
<div>{''.join(sections)}</div>
</div>"""


def _conf_bg(hyp: dict) -> str:
    c = (hyp.get("confidence") or "").lower()
    return "#dcfce7" if c == "high" else "#fef3c7" if c == "medium" else "#f1f5f9"

def _conf_fg(hyp: dict) -> str:
    c = (hyp.get("confidence") or "").lower()
    return "#166534" if c == "high" else "#92400e" if c == "medium" else "#475569"

def _esc_defn(definition: str) -> str:
    if not definition:
        return ""
    return '<div class="detail">' + _esc(definition) + "</div>"

def _esc_children(child_names: str) -> str:
    if not child_names:
        return ""
    return '<div class="detail"><strong>Subtypes:</strong> ' + _esc(child_names) + "</div>"


def _render_knowledge_context(knowledge: dict) -> str:
    """Render KEEP knowledge base context as direct retrieval cards.

    Supports both Phase 1 (site-driven broad: disease_hits, semantic_group_hits)
    and Phase 2 (hypothesis-anchored: per_hypothesis) formats.
    """
    if not knowledge or not knowledge.get("available"):
        return ""

    # ── Phase 2: per-hypothesis anchored knowledge ──
    per_hyp = knowledge.get("per_hypothesis")
    if isinstance(per_hyp, list) and per_hyp:
        cards = []
        for hyp in per_hyp:
            if not isinstance(hyp, dict):
                continue
            dg = hyp.get("disease_graph") or {}
            caps = hyp.get("morphology_captions") or []
            doid = dg.get("doid", "")
            definition = str(dg.get("definition") or "")[:300]
            children = dg.get("children") or []
            child_names = ", ".join(c.get("name", "") for c in children[:5] if c.get("name"))
            cap_html = ""
            if caps:
                cap_items = []
                for c in caps[:3]:
                    tag = " [cross-site]" if c.get("cross_site") else ""
                    cap_items.append(
                        f'<div class="detail" style="margin-bottom:4px;">'
                        f'<span style="color:var(--muted);font-size:.68rem;">[{_esc(c.get("anatomy") or "?")}]{_esc(tag)}</span> '
                        f'{_esc(str(c.get("snippet") or "")[:300])}</div>'
                    )
                cap_html = "".join(cap_items)
            cards.append(f"""<div class="kg-hit" style="border-left:3px solid var(--accent2);">
<strong>{_esc(hyp.get("diagnosis", "?"))}</strong>
<span class="badge" style="margin-left:8px;background:{_conf_bg(hyp)};color:{_conf_fg(hyp)};">{_esc(hyp.get("confidence", "?"))}</span>
<div class="meta">{_esc(doid) + " · " if doid else ""}found: {dg.get("found")}</div>
{_esc_defn(definition)}
{_esc_children(child_names)}
{cap_html}
</div>""")
        return f"""<div class="card">
<h3 id="knowledge">Knowledge Context (Phase 2 — Hypothesis-anchored)</h3>
<p style="font-size:.78rem;color:var(--muted);margin-bottom:10px;">
  Per-hypothesis KEEP retrieval: disease definitions, subtypes, and morphology-relevant captions
  from the KEEP knowledge base. Cross-referenced with visual evidence during diagnosis.
</p>
<div class="knowledge-card-grid">{"".join(cards)}</div>
</div>"""

    # ── Phase 1: site-driven broad search (legacy) ──
    hits = knowledge.get("disease_hits") or []
    group_hits = knowledge.get("semantic_group_hits") or []
    if not hits and not group_hits:
        return ""
    query = str(knowledge.get("query", ""))[:400] or "KEEP query"
    cards = []
    for h in hits[:5]:
        matched = ", ".join(str(x) for x in (h.get("matched_terms") or [])[:6])
        definition = str(h.get("definition") or h.get("caption") or "")[:260]
        matched_html = f'<div class="detail">Matched: {_esc(matched)}</div>' if matched else ""
        definition_html = f'<div class="detail">{_esc(definition)}</div>' if definition else ""
        cards.append(
            '<div class="kg-hit">'
            f'<strong>{_esc(str(h.get("name") or "Disease hit"))}</strong>'
            f'<div class="meta">{_esc(str(h.get("id") or ""))}'
            f'{(" · score %.0f" % _safe_float(h.get("score"), 0.0)) if h.get("score") is not None else ""}</div>'
            f'{matched_html}'
            f'{definition_html}'
            '</div>'
        )
    for h in group_hits[:3]:
        caption = str(h.get("caption") or "")[:260]
        caption_html = f'<div class="detail">{_esc(caption)}</div>' if caption else ""
        cards.append(
            '<div class="kg-hit semantic">'
            f'<strong>{_esc(str(h.get("anatomy") or "Semantic group"))}</strong>'
            f'<div class="meta">semantic group · score {_safe_float(h.get("score"), 0.0):.0f}</div>'
            f'{caption_html}'
            '</div>'
        )
    cards_html = "".join(cards) if cards else '<p style="color:var(--muted);font-size:.84rem;">No KEEP hits saved.</p>'
    legend = ""
    if hits and group_hits:
        legend = (
            '<div style="display:flex;gap:16px;margin-bottom:10px;font-size:.72rem;color:var(--muted);">'
            '<span>🟣 Disease matches</span>'
            '<span>🟢 Semantic / anatomy groups</span>'
            '</div>'
        )
    return f"""<div class="card">
<h3 id="knowledge">Knowledge Context (Phase 1 — Site-driven)</h3>
<p style="font-size:.78rem;color:var(--muted);margin-bottom:10px;">
  Retrieved KEEP context is shown as reference material only. It is not visual evidence and should not be treated as a verified diagnosis.
</p>
{legend}
<div class="kg-query"><strong>Retrieval query</strong><div>{_esc(query)}</div></div>
<div class="knowledge-card-grid">{cards_html}</div>
</div>"""


def _render_medgemma_context(short_report: str) -> str:
    if not short_report:
        return (
            '<div class="card"><h3>MedGemma Visual Description</h3>'
            '<p style="color:var(--muted);font-size:.84rem;">No auxiliary visual description was saved for this case.</p></div>'
        )
    parsed = _parse_medgemma_sections(short_report)
    if parsed:
        body = '<div class="medgemma-grid">' + "".join(
            f'<div class="medgemma-part"><strong>{_esc(k)}</strong><div>{_esc(v)}</div></div>'
            for k, v in parsed.items()
            if str(v).strip()
        ) + '</div>'
    else:
        body = (
            '<div class="report-block" style="border-left-color:var(--accent2);background:#faf5ff;">'
            f'<div class="report-body">{_esc(short_report)}</div>'
            '</div>'
        )
    return f"""<div class="card" id="medgemma-context">
<h3>MedGemma Visual Description</h3>
<p style="font-size:.78rem;color:var(--muted);margin-bottom:10px;">
  Auxiliary image-only pathology description used as context for report generation.
</p>
{body}
</div>"""


def _parse_medgemma_sections(text: str) -> dict[str, str]:
    text = _collapse_repeated_medgemma_lines(text or "")
    pattern = re.compile(
        r"(?im)(?:^|\n|\.\s+)\s*"
        r"(SITE(?:\s+CONTEXT)?|FINDINGS|IMPRESSION|FINAL DIAGNOSIS|DIAGNOSIS|"
        r"ARCHITECTURE/CYTOLOGY|ARCHITECTURE|CYTOLOGY|LIMITATIONS?)\s*:\s*"
    )
    parts = pattern.split(text)
    if len(parts) <= 1:
        cleaned = _collapse_repeated_medgemma_lines(text, max_repeats=1)
        return {"Findings": cleaned.strip()} if cleaned.strip() else {}
    sections: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        header = _canonical_medgemma_header(parts[i].strip())
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        body = pattern.sub("", body).strip()
        if body:
            if header in sections:
                sections[header] = f"{sections[header]}\n{body}"
            else:
                sections[header] = body
    return sections


def _canonical_medgemma_header(header: str) -> str:
    key = re.sub(r"\s+", " ", header.strip().lower())
    mapping = {
        "site": "Site Context",
        "site context": "Site Context",
        "findings": "Findings",
        "architecture/cytology": "Architecture and Cytology",
        "architecture": "Architecture and Cytology",
        "cytology": "Architecture and Cytology",
        "impression": "Impression",
        "diagnosis": "Impression",
        "final diagnosis": "Impression",
        "limitation": "Limitations",
        "limitations": "Limitations",
    }
    return mapping.get(key, header.strip().title())


def _collapse_repeated_medgemma_lines(text: str, max_repeats: int = 1) -> str:
    """Collapse pathological MedGemma repetition while preserving content."""
    raw_lines = []
    for line in (text or "").replace("\r", "\n").split("\n"):
        # Split accidental run-on bullet lists: "-a. -b." remains readable.
        line = line.strip()
        line = re.sub(r"^-\s*", "- ", line)
        if line:
            raw_lines.append(line)
    if len(raw_lines) <= 1:
        return text or ""
    counts: dict[str, int] = {}
    out = []
    collapsed = 0
    for line in raw_lines:
        key = re.sub(r"\s+", " ", line.strip().lower())
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= max_repeats:
            out.append(line)
        else:
            collapsed += 1
    if collapsed:
        out.append(f"[Collapsed {collapsed} repeated MedGemma line(s).]")
    return "\n".join(out)


def _render_claim_audit(claim_verification: dict, figures: list[dict]) -> str:
    claims = claim_verification.get("claims") if isinstance(claim_verification, dict) else []
    if not claims:
        return ""
    fig_by_roi = {str(f.get("roi_id") or ""): i + 1 for i, f in enumerate(figures)}
    fig_by_roi.update({f"roi::{str(f.get('roi_id'))}": i + 1 for i, f in enumerate(figures) if f.get("roi_id")})
    rows = []
    counts: dict[str, int] = {}
    for claim in claims:
        status = str(claim.get("status") or "unknown").lower()
        counts[status] = counts.get(status, 0) + 1
        ev_links = []
        for ev in claim.get("evidence") or []:
            evs = str(ev)
            if evs in fig_by_roi:
                n = fig_by_roi[evs]
                ev_links.append(_ref_button(f"fig-{n}", f"Fig {n}"))
            elif evs.startswith("evidence::knowledge"):
                ev_links.append(_ref_button("keep-main", "KEEP", "keep"))
            elif evs.startswith("evidence::morphology") or evs.startswith("evidence::visual_description"):
                ev_links.append(_ref_button("morphology-main", "Morphology", "morph"))
            elif evs.startswith("evidence::segmentation"):
                ev_links.append(_ref_button("decision-segmentation", "Segmentation", "decision"))
            elif evs.startswith("evidence::"):
                ev_links.append(_ref_button("decision-claims", "Evidence node", "decision"))
            else:
                ev_links.append(_esc(evs[:40]))
        evidence_html = (
            ", ".join(ev_links)
            if ev_links
            else '<span style="color:var(--muted)">none</span>'
        )
        rows.append(
            f'<tr>'
            f'<td><span class="status {status}">{_esc(status)}</span></td>'
            f'<td>{_esc(claim.get("field") or "")}</td>'
            f'<td>{_esc(claim.get("claim") or "")}</td>'
            f'<td>{evidence_html}</td>'
            f'<td style="font-size:.78rem;color:var(--muted);">{_esc(claim.get("reason") or "")}</td>'
            f'</tr>'
        )
    summary = " · ".join(f"{_esc(k)}: {v}" for k, v in sorted(counts.items()))
    return f"""<div class="card" id="claim-audit">
<h3>Claim Verification Audit</h3>
<p style="font-size:.78rem;color:var(--muted);margin-bottom:10px;">
  {summary}. Unsupported or not-assessable claims should not be presented as definitive report findings.
</p>
<table>
  <thead><tr><th>Status</th><th>Field</th><th>Claim</th><th>Evidence</th><th>Reason</th></tr></thead>
  <tbody>{"".join(rows)}</tbody>
</table>
</div>"""


def _render_decision_timeline(payload: dict, diagnosis_slots_html: str = "") -> str:
    """Render the pipeline execution timeline matching the actual agent stages."""
    stages = []
    site_pred = payload.get("site_prediction") or {}
    broad_hyp = payload.get("broad_hypotheses") or {}
    subtype_suff = payload.get("subtype_sufficiency") or {}
    diag_pred = payload.get("diagnosis_prediction") or {}
    seg_ctx = payload.get("segmentation_context") or {}
    ev_suff = payload.get("evidence_sufficiency") or {}
    claim_ver = payload.get("claim_verification") or {}
    reinspect = payload.get("reinspection_plan") or {}
    medgemma_ctx = payload.get("medgemma_morphology_context") or {}

    # Stage 1: ROI selection
    n_candidates = payload.get("candidate_count", len(payload.get("candidate_rois") or []))
    n_selected = payload.get("selected_count", len(payload.get("selected_rois") or payload.get("figures") or []))
    stages.append({
        "icon": "1", "label": "ROI Selection",
        "detail": f"{n_selected} ROIs from {n_candidates} candidates",
        "meta": "CONCH foundation score ranking (no LLM)",
        "steps": [
            ("Input", f"{n_candidates} cached/foundation-ranked candidate ROI(s) across {len(payload.get('slides_used') or [])} slide(s)"),
            ("Seed", f"bucketed evidence seed selected {n_selected} initial ROI(s)"),
            ("Purpose", "avoid single-hypothesis confirmation bias by mixing tumor-core, representative, and alternative morphology regions"),
        ],
        "chips": [f"selected={n_selected}", f"candidates={n_candidates}"],
        "color": "blue",
    })

    # Stage 2: Site prediction
    site_text = site_pred.get("predicted_site") or "—"
    site_conf = site_pred.get("confidence") or "medium"
    site_src = site_pred.get("source") or ""
    site_method = "known metadata" if site_src.startswith("known_site") else "LLM image prediction"
    stages.append({
        "icon": "2", "label": "Site Prediction",
        "detail": site_text,
        "meta": f"{site_method} · confidence: {site_conf}",
        "steps": [
            ("Input", "TCGA/case metadata" if site_src.startswith("known_site") else "selected ROI images"),
            ("Output", f"{site_text} ({site_conf})"),
        ],
        "chips": [site_method, f"confidence={site_conf}"],
        "color": "blue",
    })

    # Stage 3: Segmentation
    seg_summary = seg_ctx.get("summary") or {}
    seg_valid = seg_summary.get("valid_roi_count", "?")
    seg_total = len(seg_ctx.get("roi_results") or [])
    seg_status = f"{seg_valid}/{seg_total} ROIs segmented" if seg_ctx.get("available") else "unavailable"
    stages.append({
        "icon": "3", "label": "Segmentation (MedSAM)",
        "detail": seg_status,
        "meta": "tissue classification" if seg_ctx.get("available") else "skipped",
        "steps": [
            ("Input", f"{seg_total} selected ROI image(s)"),
            ("Output", seg_status),
            ("Cache", "shared segmentation cache is reused when available"),
        ],
        "chips": [f"available={bool(seg_ctx.get('available'))}", f"valid={seg_valid}"],
        "color": "purple",
    })

    # Stage 4: Seg QC
    seg_qc = seg_ctx.get("seg_qc") or {}
    if seg_qc.get("available"):
        qc_counts = _seg_qc_counts(seg_qc, expected_count=len(seg_ctx.get("roi_results") or []))
        stages.append({
            "icon": "4", "label": "Seg QC (LLM Audit)",
            "detail": (
                f"{qc_counts['good']} good, {qc_counts['flagged']} flagged, "
                f"{qc_counts['missing']} missing/unchecked"
            ),
            "meta": f"{qc_counts['returned']}/{qc_counts['submitted']} QC entries returned from visual overlay inspection",
            "steps": [
                ("Input", "original ROI + segmentation overlay pairs"),
                ("Audit", f"{qc_counts['returned']}/{qc_counts['submitted']} QC entries returned"),
                ("Result", f"{qc_counts['good']} good, {qc_counts['flagged']} flagged, {qc_counts['missing']} missing/unchecked"),
            ],
            "chips": [f"good={qc_counts['good']}", f"flagged={qc_counts['flagged']}", f"missing={qc_counts['missing']}"],
            "color": "teal",
        })
    else:
        stages.append({
            "icon": "4", "label": "Seg QC",
            "detail": "skipped / unavailable",
            "meta": seg_qc.get("note", ""),
            "steps": [("Reason", seg_qc.get("note", "") or "QC was not available")],
            "color": "red",
        })

    # Stage 5: MedGemma morphology description from image-only ROI evidence.
    medgemma_available = bool(medgemma_ctx.get("available") or medgemma_ctx.get("description"))
    medgemma_len = len(str(medgemma_ctx.get("description") or payload.get("medgemma_short_report") or ""))
    stages.append({
        "icon": "5", "label": "MedGemma Morphology Evidence",
        "detail": f"{'available' if medgemma_available else 'unavailable'} ({medgemma_len} chars)",
        "meta": "image-only ROI morphology context passed into Qwen hypothesis and sufficiency steps",
        "steps": [
            ("Input", "H&E ROI images only"),
            ("Memory", f"{medgemma_ctx.get('round_count') or 0} morphology round(s) accumulated"),
            ("Use", "text morphology summary is passed to Qwen diagnosis and sufficiency prompts"),
        ],
        "chips": [f"available={medgemma_available}", f"rounds={medgemma_ctx.get('round_count') or 0}"],
        "color": "purple" if medgemma_available else "red",
    })

    # Stage 6: Broad cancer-type hypotheses before subtype matching.
    broad_best = broad_hyp.get("best_broad_type") or "—"
    broad_conf = broad_hyp.get("confidence") or "unknown"
    broad_rows = broad_hyp.get("hypotheses") if isinstance(broad_hyp.get("hypotheses"), list) else []
    broad_preview = []
    for row in broad_rows[:4]:
        if isinstance(row, dict):
            broad_preview.append(f"{row.get('broad_type') or row.get('diagnosis') or '?'} ({row.get('confidence') or '?'})")
        elif row:
            broad_preview.append(str(row))
    stages.append({
        "icon": "6", "label": "Broad Hypothesis Planning",
        "kind": "Decision",
        "detail": f"{broad_best} ({broad_conf})",
        "meta": "broad lineage/type first; subtype is not selected at this stage",
        "steps": [
            ("Input", "ROI evidence + MedGemma morphology + known site + KEEP site candidates"),
            ("Output", broad_best),
            ("Alternatives", "; ".join(broad_preview) if broad_preview else "not saved"),
            ("Rule", "avoid anchoring on a specific subtype before discriminators are checked"),
        ],
        "chips": [f"available={bool(broad_hyp.get('available'))}", f"hypotheses={len(broad_rows)}", f"confidence={broad_conf}"],
        "color": "amber",
    })

    # Stage 7: KEEP site-specific subtype candidates + sufficiency.
    subtype_level = subtype_suff.get("best_supported_level") or "unknown"
    subtype_primary = subtype_suff.get("recommended_primary_diagnosis") or "—"
    subtype_rows = subtype_suff.get("subtype_assessments") if isinstance(subtype_suff.get("subtype_assessments"), list) else []
    subtype_preview = []
    for row in subtype_rows[:4]:
        if isinstance(row, dict):
            subtype_preview.append(f"{row.get('subtype') or '?'}: {row.get('status') or '?'}")
    stages.append({
        "icon": "7", "label": "KEEP Subtype Sufficiency",
        "kind": "Decision",
        "detail": f"{subtype_level}: {subtype_primary}",
        "meta": subtype_suff.get("reason") or "site-specific subtype candidates are judged against current evidence",
        "steps": [
            ("Input", f"{subtype_suff.get('candidate_count', 'unknown')} KEEP/TCGA site-specific subtype candidate(s)"),
            ("Decision", f"best_supported_level={subtype_level}; recommended={subtype_primary}"),
            ("Assessments", "; ".join(subtype_preview) if subtype_preview else "not saved"),
            ("Image gaps", _list_preview(subtype_suff.get("image_actionable_gaps") or [])),
            ("Nonimage gaps", _list_preview(subtype_suff.get("nonimage_metadata_gaps") or [])),
        ],
        "chips": [f"level={subtype_level}", f"candidates={subtype_suff.get('candidate_count', 0)}", f"confidence={subtype_suff.get('confidence') or 'unknown'}"],
        "color": "teal",
    })

    # Stage 8: Diagnostic hypothesis from the current visual set.  In older
    # traces this is the final stored hypothesis; newer traces store per-round
    # snapshots inside reinspection_plan.rounds.
    dx = diag_pred.get("primary_diagnosis") or "—"
    dx_conf = diag_pred.get("confidence") or "low"
    hypotheses = diag_pred.get("hypothesis_set") or []
    hyp_text = ""
    if isinstance(hypotheses, list) and hypotheses:
        names = []
        for h in hypotheses[:3]:
            if isinstance(h, dict):
                names.append(str(h.get("diagnosis") or h.get("name") or h.get("label") or ""))
            else:
                names.append(str(h))
        hyp_text = "top hypotheses: " + "; ".join([n for n in names if n])
    if not hyp_text:
        diff = diag_pred.get("differential") or []
        if isinstance(diff, list) and diff:
            hyp_text = "differential: " + "; ".join(str(x) for x in diff[:3])
    missing_dx = diag_pred.get("discriminative_evidence_needed") or diag_pred.get("missing_information") or []
    if isinstance(missing_dx, list) and missing_dx:
        miss = "; ".join(str(x) for x in missing_dx[:3])
        hyp_text = f"{hyp_text}; discriminators: {miss}" if hyp_text else f"discriminators: {miss}"
    stages.append({
        "icon": "8", "label": "Diagnostic Hypothesis",
        "kind": "Decision",
        "detail": f"{dx} ({dx_conf})",
        "meta": hyp_text or "generated from ROI + segmentation + MedGemma morphology evidence before final report writing",
        "steps": [
            ("Input", "ROI images + segmentation stats + MedGemma morphology + known site"),
            ("Output", f"{dx} ({dx_conf})"),
            ("Differential", hyp_text or "not provided"),
        ],
        "chips": [f"confidence={dx_conf}", f"hypotheses={len(hypotheses) if isinstance(hypotheses, list) else 0}"],
        "color": "amber",
    })

    # Stage 9: Evidence sufficiency check after the preliminary hypothesis.
    suff = ev_suff.get("sufficient_for_diagnosis")
    report_suff = ev_suff.get("sufficient_for_report")
    next_step = ev_suff.get("recommended_next_step") or ""
    missing = ev_suff.get("missing_evidence") or []
    image_gaps = ev_suff.get("missing_image_evidence") or []
    nonimage_gaps = ev_suff.get("missing_nonimage_metadata") or []
    discr_q = ev_suff.get("discriminative_questions") or []
    suff_bits = []
    suff_bits.append(f"diagnosis={'sufficient' if suff else 'insufficient'}")
    suff_bits.append(f"report={'sufficient' if report_suff else 'insufficient'}")
    if next_step:
        suff_bits.append(f"next={next_step}")
    if isinstance(image_gaps, list) and image_gaps:
        suff_bits.append("image-actionable: " + "; ".join(str(x) for x in image_gaps[:3]))
    if isinstance(nonimage_gaps, list) and nonimage_gaps:
        suff_bits.append("nonimage: " + "; ".join(str(x) for x in nonimage_gaps[:3]))
    if isinstance(missing, list) and missing:
        suff_bits.append("missing: " + "; ".join(str(x) for x in missing[:3]))
    if isinstance(discr_q, list) and discr_q:
        suff_bits.append("questions: " + "; ".join(str(x) for x in discr_q[:2]))
    stages.append({
        "icon": "9", "label": "Evidence Sufficiency Check",
        "kind": "Decision",
        "detail": f"initial judgement with {n_selected} ROI(s)",
        "meta": " · ".join(suff_bits),
        "steps": [
            ("Decision", f"diagnosis={'sufficient' if suff else 'insufficient'}, report={'sufficient' if report_suff else 'insufficient'}"),
            ("Next", str(next_step or ev_suff.get("reinspection_action") or "none")),
            ("Image gaps", _list_preview(image_gaps)),
            ("Nonimage gaps", _list_preview(nonimage_gaps)),
            ("Questions", _list_preview(discr_q, limit=2)),
        ],
        "chips": [f"image_gaps={len(image_gaps) if isinstance(image_gaps, list) else 0}", f"nonimage_gaps={len(nonimage_gaps) if isinstance(nonimage_gaps, list) else 0}"],
        "color": "green" if suff else "amber",
    })

    # Stage 8: Reinspection loop.  Show each recorded round when available.
    total_rounds = reinspect.get("total_rounds") or 0
    rounds = reinspect.get("rounds") or []
    if total_rounds > 0:
        stages.append({
            "icon": "10", "label": f"Reinspection Loop ({total_rounds} round(s))",
            "kind": "Loop",
            "detail": f"added targeted ROI evidence and re-ran seg + hypothesis + sufficiency",
            "meta": f"final: {'sufficient' if suff else 'insufficient'}, {n_selected} ROIs total",
            "steps": [
                ("Trigger", "initial sufficiency found image-actionable gaps"),
                ("Limit", f"up to {payload.get('max_zoom_depth', 1)} zoom depth and {total_rounds} executed reinspection round(s)"),
                ("Final", f"{n_selected} ROI(s) used after reinspection"),
            ],
            "chips": [f"rounds={total_rounds}", f"final_rois={n_selected}"],
            "color": "blue",
        })
        for ridx, rr in enumerate(rounds, start=1):
            if not isinstance(rr, dict):
                continue
            added_rois = rr.get("new_rois") or rr.get("zoom_rois") or rr.get("extra_rois") or []
            added = len(added_rois)
            post = rr.get("post_reinspection") or {}
            rr_suff_payload = post.get("evidence_sufficiency") or rr.get("evidence_sufficiency") or {}
            rr_dx_payload = post.get("diagnosis_prediction") or rr.get("diagnosis_prediction") or {}
            rr_suff = rr_suff_payload.get("sufficient_for_diagnosis")
            rr_dx = rr_dx_payload.get("primary_diagnosis") or ""
            action = rr.get("action") or rr.get("llm_reinspection_action") or "add_independent_roi"
            why = rr.get("reinspection_reason") or rr.get("reason") or rr.get("strategy") or rr.get("recommended_next_step") or ""
            zoom_count = len(rr.get("zoom_rois") or [])
            independent_count = len(rr.get("extra_rois") or [])
            image_gaps_round = rr_suff_payload.get("missing_image_evidence") or []
            nonimage_gaps_round = rr_suff_payload.get("missing_nonimage_metadata") or []
            selected_after = post.get("selected_roi_count")
            medgemma_round = post.get("medgemma_round") or {}
            detail = f"round {ridx}: {action}, added {added} ROI(s)"
            if rr_dx:
                detail += f", interim dx: {rr_dx}"
            stages.append({
                "icon": f"10.{ridx}", "label": "Reinspection Round Detail",
                "kind": "Loop Round",
                "detail": detail,
                "meta": str(why),
                "steps": [
                    ("Action", f"{action}; zoom_rois={zoom_count}, independent_rois={independent_count}"),
                    ("New ROI", _list_preview([r.get("roi_id") or r.get("image_path") for r in added_rois], limit=3)),
                    ("Seg/QC", "segmentation and QC re-run on the updated ROI set"),
                    ("MedGemma", f"new morphology round available={bool(medgemma_round.get('available') or medgemma_round.get('description'))}"),
                    ("Post dx", rr_dx or "not available"),
                    ("Post sufficiency", f"{'sufficient' if rr_suff else 'insufficient' if rr_suff is not None else 'unknown'}; selected_rois={selected_after or 'unknown'}"),
                    ("Remaining gaps", f"image={len(image_gaps_round) if isinstance(image_gaps_round, list) else 0}, nonimage={len(nonimage_gaps_round) if isinstance(nonimage_gaps_round, list) else 0}"),
                ],
                "chips": [f"action={action}", f"added={added}", f"zoom={zoom_count}", f"independent={independent_count}"],
                "color": "blue",
            })
    else:
        why = reinspect.get("plan") or reinspect.get("note") or ""
        stages.append({
            "icon": "10", "label": "Reinspection Loop",
            "kind": "Loop",
            "detail": "not triggered",
            "meta": why,
            "steps": [
                ("Reason", why or "not needed"),
                ("Candidates", f"{reinspect.get('remaining_candidate_count', 'unknown')} remaining, {reinspect.get('candidate_count', 'unknown')} total"),
                ("Selected", f"{reinspect.get('selected_roi_count', n_selected)} ROI(s)"),
            ],
            "chips": [str(reinspect.get("stop_reason") or why or "not triggered")[:80]],
            "color": "green",
        })

    # Stage 9: Final diagnosis after any reinspection updates.
    stages.append({
        "icon": "11", "label": "Final Diagnosis",
        "kind": "Decision",
        "detail": f"{dx} ({dx_conf})",
        "meta": f"based on all {n_selected} ROIs",
        "steps": [
            ("Input", f"all selected/zoomed ROI evidence ({n_selected} ROI(s))"),
            ("Output", f"{dx} ({dx_conf})"),
        ],
        "chips": [f"final_rois={n_selected}", f"confidence={dx_conf}"],
        "color": "amber",
    })

    # Stage 10: Evidence graph and structured diagnostic slots.
    graph = payload.get("evidence_graph") or {}
    roi_nodes = len(graph.get("roi_nodes") or [])
    ev_nodes = len(graph.get("evidence_nodes") or [])
    slot_count = len((payload.get("diagnosis_slots") or {}).keys())
    stages.append({
        "icon": "12", "label": "Evidence Graph + Diagnosis Slots",
        "kind": "State",
        "detail": f"{roi_nodes} ROI nodes, {ev_nodes} evidence/context nodes, {slot_count} slot fields",
        "meta": "case evidence is converted into structured diagnostic state before claim audit",
        "steps": [
            ("Graph", f"{roi_nodes} ROI node(s), {ev_nodes} evidence/context node(s)"),
            ("Slots", f"{slot_count} structured diagnostic field(s)"),
            ("Purpose", "prepare claim-level audit before final report text"),
        ],
        "chips": [f"roi_nodes={roi_nodes}", f"evidence_nodes={ev_nodes}", f"slots={slot_count}"],
        "color": "purple",
    })

    # Stage 11: Claim extraction and verification.
    claims = claim_ver.get("claims") or []
    if claims:
        n_sup = sum(1 for c in claims if c.get("status") == "supported")
        n_weak = sum(1 for c in claims if c.get("status") == "weak")
        n_unsup = sum(1 for c in claims if c.get("status") in ("unsupported", "contradicted"))
        stages.append({
            "icon": "13", "label": "Claim Extraction + Verification",
            "kind": "Audit",
            "detail": f"{n_sup} supported, {n_weak} weak, {n_unsup} unsupported / {len(claims)} claims",
            "meta": "claims are audited before report generation; unsupported claims are removed or softened",
            "steps": [
                ("Extract", f"{len(claims)} claim(s) from diagnosis slots"),
                ("Verify", f"{n_sup} supported, {n_weak} weak, {n_unsup} unsupported/contradicted"),
                ("Effect", "unsupported claims are removed or changed to not-assessable in the final report"),
            ],
            "chips": [f"supported={n_sup}", f"weak={n_weak}", f"unsupported={n_unsup}"],
            "color": "teal",
        })
    else:
        stages.append({
            "icon": "13", "label": "Claim Extraction + Verification",
            "kind": "Audit",
            "detail": "No claims to verify",
            "meta": "",
            "steps": [("Status", "claim verification disabled or unavailable")],
            "color": "red",
        })

    # Stage 12: Additional knowledge context.
    kg = payload.get("knowledge_context") or {}
    if kg.get("available") or kg.get("used"):
        n_kg = len(kg.get("entries") or kg.get("knowledge_items") or kg.get("disease_context") or [])
        stages.append({
            "icon": "14", "label": "Additional Knowledge Context",
            "kind": "Context",
            "detail": f"KEEP/context retrieval available ({n_kg} item group(s))",
            "meta": "used as auxiliary context, not direct visual evidence",
            "steps": [
                ("Retrieve", f"{n_kg} knowledge item group(s)"),
                ("Use", "context only; visual claims still require ROI/seg/MedGemma support"),
            ],
            "chips": [f"knowledge_groups={n_kg}"],
            "color": "blue",
        })
    elif payload.get("knowledge_context") is not None:
        stages.append({
            "icon": "14", "label": "Additional Knowledge Context",
            "kind": "Context",
            "detail": "not available or not used",
            "meta": "",
            "steps": [("Status", "knowledge retrieval disabled or returned no usable context")],
            "color": "red",
        })

    # Stage 13: Report generation and claim-filter rewrite.
    rep = payload.get("generated_long_report") or ""
    rep_len = len(rep)
    stages.append({
        "icon": "15", "label": "Report Generation",
        "kind": "Output",
        "detail": f"Qwen long report → claim-filtered rewrite ({rep_len} chars)",
        "meta": "final output",
        "steps": [
            ("Draft", "Qwen generates a structured long report from evidence, morphology, segmentation, and context"),
            ("Filter", "claim verification rewrites or removes unsupported statements when enabled"),
            ("Save", "generated_long_report.txt, generated_long_report_rich.txt, report.md/html"),
        ],
        "chips": [f"chars={rep_len}"],
        "color": "amber",
    })

    if not stages:
        return ""

    decision_ref_by_label = {
        "ROI Selection": "decision-roi-selection",
        "Site Prediction": "decision-site",
        "Segmentation (MedSAM)": "decision-segmentation",
        "Seg QC (LLM Audit)": "decision-segmentation",
        "Seg QC": "decision-segmentation",
        "MedGemma Morphology Evidence": "morphology-main",
        "Broad Hypothesis Planning": "decision-broad",
        "KEEP Subtype Sufficiency": "decision-subtype",
        "Diagnostic Hypothesis": "decision-diagnosis",
        "Evidence Sufficiency Check": "decision-sufficiency",
        "Reinspection Loop": "decision-reinspection",
        "Final Diagnosis": "decision-diagnosis",
        "Evidence Graph + Diagnosis Slots": "decision-claims",
        "Claim Extraction + Verification": "decision-claims",
        "Additional Knowledge Context": "keep-main",
        "Report Generation": "decision-report",
    }
    items = []
    for i, s in enumerate(stages):
        connector = '<div class="tl-connector"></div>' if i < len(stages) - 1 else ""
        sc = s["color"]
        dot_color = {"green": "var(--green)", "red": "var(--red)", "blue": "var(--accent)",
                      "purple": "var(--accent2)", "teal": "var(--teal)", "amber": "var(--amber)"}.get(sc, "var(--accent)")
        step_html = ""
        steps = s.get("steps") or []
        if steps:
            step_rows = []
            for step in steps:
                if isinstance(step, (list, tuple)) and len(step) >= 2:
                    tag, text = step[0], step[1]
                else:
                    tag, text = "Step", step
                if text is None or str(text).strip() == "":
                    continue
                step_rows.append(
                    f'<li><span class="tl-step-tag">{_esc(str(tag))}</span><span>{_esc(str(text))}</span></li>'
                )
            if step_rows:
                step_html = f'<ul class="tl-steps">{"".join(step_rows)}</ul>'
        chips = [str(chip) for chip in (s.get("chips") or []) if str(chip)]
        ref = decision_ref_by_label.get(str(s.get("label") or ""))
        if str(s.get("label") or "").startswith("Reinspection Round"):
            ref = "decision-reinspection"
        chip_html = ""
        chip_bits = []
        if ref:
            chip_bits.append(_ref_button(ref, "Inspect", "decision"))
        chip_bits.extend(f'<span class="tl-chip">{_esc(chip)}</span>' for chip in chips[:8])
        if chip_bits:
            chip_html = '<div class="tl-chips">' + "".join(chip_bits) + '</div>'
        kind = str(s.get("kind") or "Stage")
        items.append(f"""
<div class="tl-item">
  <div class="tl-dot" style="background:{dot_color};"></div>
  <details class="tl-card" open>
    <summary>
      <div class="tl-node-title">
        <strong>{s['icon']} {s['label']}</strong>
        <span class="tl-node-kind">{_esc(kind)}</span>
      </div>
      <div class="tl-detail">{_esc(str(s['detail']))}</div>
      {f'<div class="tl-meta">{_esc(str(s["meta"]))}</div>' if s.get("meta") else ""}
    </summary>
    {step_html}
    {chip_html}
  </details>
</div>
{connector}""")

    timeline_block = f'<div class="timeline">{"".join(items)}</div>'
    if diagnosis_slots_html:
        content = f"""
<div class="decision-state-flow">
  <div>{timeline_block}</div>
  <div class="state-after-timeline">{diagnosis_slots_html}</div>
</div>"""
    else:
        content = timeline_block
    return f"""<div class="card" id="decision">
<h3>Decision Process and Structured State</h3>
<p style="font-size:.78rem;color:var(--muted);margin-bottom:10px;">
  The execution flow is shown first. The structured diagnosis state below is the intermediate state audited before the final report.
</p>
{content}
</div>"""


def _render_evidence_graph_viz(payload: dict, graph: dict) -> str:
    """Render the current evidence flow as a visual SVG DAG.

    The underlying v1 graph is still coarse: ROI nodes mostly support the
    diagnosis node, while known site and KEEP are context inputs.  The HTML
    should show that honestly instead of implying a fully claim-level graph.
    """
    selected_rois = payload.get("selected_rois") or []
    roi_nodes = graph.get("roi_nodes") or []
    if not roi_nodes and selected_rois:
        roi_nodes = selected_rois
    evidence_nodes = graph.get("evidence_nodes") or []
    if not roi_nodes and not evidence_nodes:
        return ""
    max_display = 10
    roi_display = roi_nodes[:max_display]
    hidden_roi_count = max(0, len(roi_nodes) - len(roi_display))
    bucket_counts: dict[str, int] = {}
    for row in selected_rois or roi_nodes:
        bucket = str(row.get("evidence_bucket") or row.get("role") or row.get("evidence_role") or "roi")
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    bucket_summary = ", ".join(f"{k}: {v}" for k, v in bucket_counts.items())
    seg_ctx = payload.get("segmentation_context") or {}
    seg_count = len(seg_ctx.get("roi_results") or [])
    reinspect = payload.get("reinspection_plan") or {}
    reinspect_rounds = int(reinspect.get("total_rounds") or len(reinspect.get("rounds") or []) or 0)
    site_node = next((e for e in evidence_nodes if e.get("type") == "site_prediction"), {})
    diag_node = next((e for e in evidence_nodes if e.get("type") == "diagnosis_prediction"), {})
    suff_node = next((e for e in evidence_nodes if e.get("type") == "evidence_sufficiency"), {})
    kg_node = next((e for e in evidence_nodes if e.get("type") == "knowledge_context"), {})
    site_payload = site_node.get("payload") or {}
    site_is_input = str(site_payload.get("source") or "").startswith("known_site")
    diag_payload = diag_node.get("payload") or {}
    rec_studies_dag = diag_payload.get("recommended_ancillary_studies")
    has_rec = isinstance(rec_studies_dag, list) and rec_studies_dag
    rec_y = 504 if has_rec else 0
    height = max(470, 68 * len(roi_display) + 150)
    if has_rec:
        height = max(height, rec_y + 80)
    svg_parts = [
        '<defs><marker id="eg-arrow" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#94a3b8"/></marker></defs>',
        '<text x="36" y="24" class="sub">Inputs</text>',
        '<text x="268" y="24" class="sub">Visual evidence</text>',
        '<text x="540" y="24" class="sub">Derived evidence and context</text>',
        '<text x="780" y="24" class="sub">Report use</text>',
    ]
    # Input: known/provided site, or inferred site when not known.
    site_label = "Known Site Input" if site_is_input else "Site Prediction"
    site_sub = str(site_payload.get("predicted_site") or "")[:34]
    svg_parts.append(
        f'<g class="graph-node {"input" if site_is_input else "inf"}">'
        f'<rect x="34" y="54" width="190" height="66"/>'
        f'{_svg_multiline(site_label, 50, 78, max_chars=22, max_lines=1)}'
        f'{_svg_multiline(site_sub, 50, 100, max_chars=24, max_lines=1, cls="sub")}</g>'
    )
    roi_center_ys = []
    for i, r in enumerate(roi_display, start=1):
        y = 48 + (i - 1) * 68
        roi_center_ys.append(y + 28)
        label = f"Figure {i}"
        sub = f"{r.get('role') or 'roi'} score={_safe_float(r.get('score'), 0.0):.3f}"
        svg_parts.append(
            f'<a href="#roi-{i}"><g class="graph-node roi">'
            f'<rect x="268" y="{y}" width="210" height="56"/>'
            f'{_svg_multiline(label, 284, y + 22, max_chars=22, max_lines=1)}'
            f'{_svg_multiline(sub, 284, y + 43, max_chars=28, max_lines=1, cls="sub")}'
            f'</g></a>'
        )
    diagnosis_y = 84
    seg_y = 174
    suff_y = 264
    reinspect_y = 344
    kg_y = 424
    diag_payload = diag_node.get("payload") or {}
    suff_payload = suff_node.get("payload") or {}
    diag_sub = str(diag_payload.get("primary_diagnosis") or "")[:36]
    suff_sub = (
        "sufficient" if suff_payload.get("sufficient_for_report")
        else "insufficient / needs evidence"
    )
    kg_sub = "KEEP retrieved context" if kg_node else "not used"
    derived_nodes = [
        ("Diagnosis Inference", diag_sub, diagnosis_y, "#decision", "inf"),
        ("D1 Segmentation Stats", f"{seg_count}/{len(selected_rois) or len(roi_nodes)} ROIs segmented", seg_y, "#segmentation", "derived"),
        ("Evidence Sufficiency", suff_sub, suff_y, "#decision", "derived"),
        ("Reinspection", f"{reinspect_rounds} round(s), {hidden_roi_count} hidden ROI nodes" if hidden_roi_count else f"{reinspect_rounds} round(s)", reinspect_y, "#decision", "derived"),
        ("External Knowledge Context", kg_sub, kg_y, "#knowledge", "context"),
    ]
    if has_rec:
        rec_sub = "; ".join(str(s) for s in rec_studies_dag[:2])
        derived_nodes.append(
            ("Recommended Ancillary Studies", rec_sub, rec_y, "#decision", "context"),
        )
    for title, sub, y, href, cls in derived_nodes:
        svg_parts.append(
            f'<a href="{href}"><g class="graph-node {cls}">'
            f'<rect x="540" y="{y}" width="210" height="62"/>'
            f'{_svg_multiline(title, 556, y + 23, max_chars=26, max_lines=1)}'
            f'{_svg_multiline(sub, 556, y + 45, max_chars=27, max_lines=1, cls="sub")}</g></a>'
        )
    report_y = 174
    svg_parts.append(
        f'<a href="#report"><g class="graph-node report">'
        f'<rect x="780" y="{report_y}" width="170" height="70"/>'
        f'{_svg_multiline("Claim-verified report", 796, report_y + 28, max_chars=22, max_lines=1)}'
        f'{_svg_multiline("final text output", 796, report_y + 50, max_chars=20, max_lines=1, cls="sub")}</g></a>'
    )
    # Edges: current v1 conceptual flow, not a fully claim-level proof graph.
    edge_paths = []
    edge_paths.append((224, 87, 540, diagnosis_y + 31))
    for cy in roi_center_ys:
        edge_paths.append((478, cy, 540, diagnosis_y + 31))
        edge_paths.append((478, cy, 540, seg_y + 31))
    edge_paths.extend([
        (750, diagnosis_y + 31, 780, report_y + 35),
        (750, seg_y + 31, 780, report_y + 35),
        (750, suff_y + 31, 780, report_y + 35),
        (750, reinspect_y + 31, 780, report_y + 35),
        (750, kg_y + 31, 780, report_y + 35),
        (224, 87, 540, suff_y + 31),
    ])
    if has_rec:
        edge_paths.extend([
            (750, diagnosis_y + 31, 540, rec_y + 31),
            (750, rec_y + 31, 780, report_y + 35),
        ])
    for x1, y1, x2, y2 in edge_paths:
        svg_parts.insert(
            1,
            f'<path class="graph-edge" marker-end="url(#eg-arrow)" '
            f'd="M {x1} {y1} C {(x1 + x2) / 2:.1f} {y1}, {(x1 + x2) / 2:.1f} {y2}, {x2} {y2}"/>'
        )
    svg = (
        f'<div class="graph-svg-wrap"><svg class="graph-svg" viewBox="0 0 980 {height}" '
        f'role="img" aria-label="Evidence graph">{"".join(svg_parts)}</svg></div>'
    )

    flow_summary = f"""
<div class="flow-summary">
  <span><b>{len(roi_nodes)}</b> selected visual ROI node(s)</span>
  <span><b>{seg_count}</b> segmented ROI(s)</span>
  <span><b>{reinspect_rounds}</b> reinspection round(s)</span>
  {f'<span>{_esc(bucket_summary)}</span>' if bucket_summary else ''}
</div>"""
    return f"""<div class="card" id="evidence-graph">
<h3>Evidence Flow</h3>
<p style="font-size:.78rem;color:var(--muted);margin-bottom:10px;">
  This is the current v1 flow, not a full claim-level proof graph. Known site is shown as metadata input when used. The diagram now reflects actual ROI count, segmentation count, and reinspection rounds; large ROI sets are summarized after the first {max_display} nodes.
</p>
{flow_summary}
{svg}
</div>"""


def _render_segmentation_summary(seg_ctx: dict, figures: list[dict]) -> str:
    """Render segmentation results summary."""
    if not seg_ctx or not seg_ctx.get("available"):
        return '<div class="card"><h3>Segmentation</h3><p style="color:var(--muted);">Segmentation unavailable for this case.</p></div>'

    class_names = seg_ctx.get("class_names") or {}
    summary = seg_ctx.get("summary") or {}
    roi_results = seg_ctx.get("roi_results") or []

    # Aggregate class stats.  Current MedSAM ROI output stores per-ROI values
    # under stats.named_class_fractions and run-level means under
    # summary.mean_class_fractions, not at top-level class_counts.
    mean_fracs = summary.get("mean_class_fractions") or {}
    agg_frac: dict[str, float] = {}
    if mean_fracs:
        for cls_name, row in mean_fracs.items():
            if isinstance(row, dict):
                agg_frac[str(cls_name)] = _safe_float(row.get("mean_fraction"), 0.0)
    else:
        accum: dict[str, list[float]] = {}
        for r in roi_results:
            named = ((r.get("stats") or {}).get("named_class_fractions") or {})
            for cls_name, row in named.items():
                if isinstance(row, dict):
                    accum.setdefault(str(cls_name), []).append(_safe_float(row.get("fraction"), 0.0))
        for cls_name, vals in accum.items():
            if vals:
                agg_frac[cls_name] = sum(vals) / len(vals)
    foreground_mean = _safe_float(summary.get("mean_foreground_fraction"), 0.0)

    # ── Color mapping MUST match _save_overlay() palette in medsam_roi.py ──
    _TISSUE_COLORS = {
        0: "#000000", 1: "#DC2626", 2: "#2563EB", 3: "#16A34A",
        4: "#EAB308", 5: "#9333EA", 6: "#0EA5E9",
    }

    # Build bars with correct colors
    bars = []
    legend_items = []
    for cid in sorted(class_names.keys(), key=int):
        cls_name = class_names.get(cid, f"Class {cid}")
        frac = agg_frac.get(str(cls_name), 0.0)
        pct = frac * 100.0
        color = _TISSUE_COLORS.get(int(cid), "#94a3b8")
        bars.append(
            f'<div class="class-bar">'
            f'<span class="label">{_esc(cls_name)}</span>'
            f'<span class="bar-wrap"><span class="bar-fill" style="width:{max(pct, 0.6)}%;background:{color};"></span></span>'
            f'<span class="pct">{pct:.1f}%</span>'
            f'</div>'
        )
        legend_items.append(
            f'<span style="display:inline-flex;align-items:center;gap:4px;margin-right:14px;font-size:0.78rem;">'
            f'<span style="width:12px;height:12px;border-radius:2px;background:{color};display:inline-block;"></span>'
            f'{_esc(cls_name)}'
            f'</span>'
        )
    legend_html = " &nbsp; ".join(legend_items)

    valid_count = summary.get("valid_roi_count", len([r for r in roi_results if not r.get("error")]))
    total_count = len(roi_results)

    bars_html = "".join(bars)

    return f"""<div class="card" id="segmentation">
<h3>Segmentation Results</h3>
<p style="font-size:0.85rem;margin-bottom:10px;">
  Valid: {valid_count}/{total_count} ROIs &nbsp;|&nbsp;
  Mean foreground: {foreground_mean * 100:.1f}%
</p>
<p style="font-size:0.75rem;color:var(--muted);margin-bottom:8px;">Color legend: {legend_html}</p>
<div style="max-width:600px;">{bars_html}</div>
<p style="font-size:.76rem;color:var(--muted);margin-top:10px;">
  Per-ROI segmentation quality is annotated on each overlay in the WSI Overview section above.
</p>
</div>"""


def _render_eval_metrics(eval_data: dict) -> str:
    """Render evaluation metrics summary."""
    if not eval_data or not eval_data.get("available"):
        return '<div class="card"><h3>📊 Metrics</h3><p style="color:var(--muted);">Evaluation not available.</p></div>'

    v2 = eval_data.get("v2_metrics") or {}
    v1 = eval_data.get("v1_metrics") or {}
    core = eval_data.get("image_only_core") or {}
    pq = eval_data.get("pathology_metrics") or {}
    primary = eval_data.get("primary_metrics") or {}

    metrics = []
    if primary.get("primary_score") is not None:
        metrics.append(("Primary Image-only", f"{primary['primary_score']:.3f}"))
    if core.get("core_score"):
        metrics.append(("Image-only Core", f"{core['core_score']:.3f}"))
    if v2.get("report_quality_score"):
        metrics.append(("Image-only Structured", f"{v2['report_quality_score']:.3f}"))
    if v2.get("site_similarity"):
        metrics.append(("Site Sim", f"{v2['site_similarity']:.3f}"))
    if v2.get("diagnosis_similarity"):
        metrics.append(("Dx Sim", f"{v2['diagnosis_similarity']:.3f}"))
    if pq.get("pathology_report_score"):
        metrics.append(("Path Score", f"{pq['pathology_report_score']:.3f}"))
    if v1.get("report_quality_score"):
        metrics.append(("Full Ref Secondary", f"{v1['report_quality_score']:.3f}"))

    metric_cards = "".join(
        f'<div class="metric"><div class="num">{val}</div><div class="lbl">{label}</div></div>'
        for label, val in metrics
    )

    # V2 section details
    v2_secs = v2.get("section_similarities") or {}
    sec_rows = "".join(
        f'<tr><td>{_esc(k)}</td><td>{v:.3f}</td></tr>'
        for k, v in v2_secs.items()
    )

    diag_hit = (pq.get("detail") or {}).get("diagnosis_hit") or {}
    hit_level = diag_hit.get("hit_level", "—")

    return f"""<div class="card">
<h3>📊 Evaluation Metrics</h3>
<div class="metric-grid">{metric_cards}</div>
<div style="margin-top:14px;display:grid;grid-template-columns:1fr 1fr;gap:12px;">
  <div>
    <h4 style="font-size:0.78rem;color:var(--muted);margin-bottom:6px;">V2 Section Similarities</h4>
    <table style="font-size:0.78rem;"><thead><tr><th>Section</th><th>Similarity</th></tr></thead><tbody>{sec_rows}</tbody></table>
  </div>
  <div>
    <h4 style="font-size:0.78rem;color:var(--muted);margin-bottom:6px;">Pathology</h4>
    <div class="slot-item"><span class="slot-label">Diagnosis Hit</span> <span class="slot-value">{hit_level}</span></div>
    <div class="slot-item"><span class="slot-label">Morph Term F1</span> <span class="slot-value">{pq.get("components",{}).get("morph_term_f1",0):.3f}</span></div>
    <div class="slot-item"><span class="slot-label">Critical Recall</span> <span class="slot-value">{pq.get("components",{}).get("critical_finding_recall",0):.3f}</span></div>
  </div>
</div>
</div>"""


def _render_prediction_cards(site_pred: dict, diag_pred: dict, evidence_suff: dict, payload: dict) -> str:
    """Render concise report-side summary cards."""
    cards = []

    # Site context card. In known-site ablations this is metadata input, not a
    # visual prediction, so label it honestly.
    site = site_pred.get("predicted_site") or "—"
    site_conf = site_pred.get("confidence") or "low"
    site_source = str(site_pred.get("source") or "")
    site_title = "Known Site Input" if site_source.startswith("known_site") else "Site Prediction"
    site_meta = (
        "provided metadata; not inferred from ROI morphology"
        if site_source.startswith("known_site")
        else f"confidence: {site_conf}"
    )
    top_score = site_pred.get("top_score")
    alts = site_pred.get("top_alternatives") or site_pred.get("scores") or []
    alt_html = ""
    if alts:
        alt_items = []
        for a in alts[:5]:
            if isinstance(a, dict):
                alt_items.append(f'{_esc(a.get("site",""))} ({_safe_float(a.get("score"), 0.0):.3f})')
            else:
                alt_items.append(str(a))
        alt_html = f'<div class="meta">Alternatives: {", ".join(alt_items)}</div>'

    cards.append(f"""<div class="pred-card conf-{site_conf}">
<h4>{site_title}</h4>
<div class="value">{_esc(site)}</div>
<div class="meta">{_esc(site_meta)}{f' · score: {_safe_float(top_score, 0.0):.4f}' if top_score is not None and not site_source.startswith("known_site") else ''}</div>
{alt_html}
</div>""")

    # Diagnosis card
    dx = diag_pred.get("primary_diagnosis") or "—"
    dx_conf = diag_pred.get("confidence") or "low"
    diffs = diag_pred.get("differential") or []
    diff_html = f'<div class="meta">Differential: {", ".join(_esc(d) for d in diffs[:5])}</div>' if diffs else ""

    cards.append(f"""<div class="pred-card conf-{dx_conf}">
<h4>Diagnosis Prediction</h4>
<div class="value">{_esc(dx)}</div>
<div class="meta">confidence: {dx_conf}</div>
{diff_html}
</div>""")

    # Evidence sufficiency card
    if evidence_suff:
        diag_ok = evidence_suff.get("sufficient_for_diagnosis")
        report_ok = evidence_suff.get("sufficient_for_report")
        if diag_ok is not None or report_ok is not None:
            suff_val = (
                f"Diagnosis: {'sufficient' if diag_ok else 'insufficient'}; "
                f"Report: {'sufficient' if report_ok else 'insufficient'}"
            )
        else:
            suff_val = evidence_suff.get("sufficiency") or evidence_suff.get("level") or "Not specified"
        suff_note = evidence_suff.get("note") or evidence_suff.get("reason") or ""
        cards.append(f"""<div class="pred-card">
<h4>Evidence Sufficiency</h4>
<div class="value">{_esc(str(suff_val))}</div>
{f'<div class="meta">{_esc(suff_note)}</div>' if suff_note else ''}
</div>""")

    # Recommended Ancillary Studies card
    rec_studies = diag_pred.get("recommended_ancillary_studies")
    if isinstance(rec_studies, list) and rec_studies:
        rec_items = "".join(f"<li>{_esc(str(s))}</li>" for s in rec_studies)
        cards.append(f"""<div class="pred-card">
<h4>Recommended Ancillary Studies</h4>
<ul style="margin:4px 0 0 18px;font-size:.82rem;color:#334155;">{rec_items}</ul>
</div>""")

    return '<div class="card-grid">' + "".join(cards) + "</div>"


def _build_roi_ranking_table(roi_scores: dict, figures: list[dict]) -> str:
    """Build ROI selection scores ranking table."""
    scores_list = roi_scores.get("scores") or roi_scores.get("rois") or []
    if not scores_list:
        return '<p style="color:var(--muted);">No ROI ranking data available.</p>'

    rows = []
    for i, item in enumerate(scores_list[:20]):
        if isinstance(item, dict):
            roi_id = _esc(str(item.get("roi_id") or "")[:60])
            score = item.get("score") or item.get("roi_score") or 0
            role = _esc(item.get("role") or item.get("evidence_role") or "")
            site = _esc(item.get("top_site") or "")
        else:
            roi_id = str(item)[:60]
            score = 0
            role = site = ""
        score_str = f"{float(score):.4f}" if score else "—"
        color = _ROI_PALETTE_HEX[i % len(_ROI_PALETTE_HEX)]
        rows.append(
            f'<tr>'
            f'<td><span class="dot" style="background:{color};"></span> {i+1}</td>'
            f'<td>{roi_id}</td>'
            f'<td>{score_str}</td>'
            f'<td>{role}</td>'
            f'<td>{site}</td>'
            f'</tr>'
        )

    return f"""<table class="ranking-table">
<thead><tr><th>Rank</th><th>ROI ID</th><th>Score</th><th>Role</th><th>Top Site</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>"""


def _fallback_slots(site_prediction: dict, diagnosis_prediction: dict) -> dict:
    return {
        "site": site_prediction.get("predicted_site") or "not assessable",
        "primary_diagnosis": diagnosis_prediction.get("primary_diagnosis") or "not assessable",
        "differential_diagnosis": diagnosis_prediction.get("differential") or [],
        "tumor_architecture": diagnosis_prediction.get("visual_evidence") or [],
        "cytology": [],
        "necrosis": "not assessable",
        "invasion": "not assessable",
        "stroma_inflammation": [],
        "ihc_molecular": "not provided",
        "confidence": diagnosis_prediction.get("confidence") or "low",
        "missing_evidence": diagnosis_prediction.get("missing_information") or [],
        "recommended_ancillary_studies": diagnosis_prediction.get("recommended_ancillary_studies") or [],
    }


def _heuristic_verify_claim(claim: dict, segmentation_context: dict, selected_rois: list[dict]) -> dict:
    text = str(claim.get("claim") or "").lower()
    field = str(claim.get("field") or "")
    status = "weak"
    reason = "Claim is plausible from diagnosis slots but needs visual confirmation."
    if any(term in text for term in ["not assessable", "not provided", "unavailable"]):
        status = "not_assessable"
        reason = "Claim explicitly states unavailable/not assessable information."
    elif field in {"site", "primary_diagnosis", "tumor_architecture", "cytology", "stroma_inflammation"}:
        status = "supported" if selected_rois else "unsupported"
        reason = "Supported by selected ROI visual evidence stream." if selected_rois else "No selected ROI evidence."
    elif field in {"ihc_molecular"}:
        status = "not_assessable"
        reason = "IHC/molecular information is not inferable from H&E ROI evidence."
    elif field in {"necrosis", "invasion"}:
        status = "weak"
        reason = "Requires explicit ROI visual support; kept weak unless verifier upgrades it."
    out = dict(claim)
    out.update({"status": status, "reason": reason, "evidence": [row.get("roi_id") for row in selected_rois[:3]]})
    return out


def _merge_verification(heuristic: list[dict], llm_claims: list[dict]) -> list[dict]:
    by_id = {str(row.get("claim_id")): row for row in heuristic}
    for row in llm_claims:
        cid = str(row.get("claim_id") or "")
        if not cid or cid not in by_id:
            continue
        merged = dict(by_id[cid])
        for key in ["status", "evidence", "reason"]:
            if row.get(key):
                merged[key] = row[key]
        by_id[cid] = merged
    return [by_id[str(row.get("claim_id"))] for row in heuristic]


def _compact_segmentation(seg: dict) -> dict:
    if not seg:
        return {}
    return {
        "available": seg.get("available", True),
        "foreground_fraction": (seg.get("stats") or {}).get("foreground_fraction"),
        "class_fractions": (seg.get("stats") or {}).get("class_fractions"),
        "mask_path": seg.get("mask_path"),
        "overlay_path": seg.get("overlay_path"),
        "cache_hit": seg.get("cache_hit"),
    }


def _strip_figure_refs(text: str) -> str:
    return re.sub(r"\s*\[Figure\s+\d+\]", "", text or "").strip()


def _img_data_uri(path: str | None, max_edge: int = 480) -> str:
    """Return base64 data URI for an image, scaled to max_edge and JPEG compressed.

    Images live in shared_artifacts/ and are embedded directly so the HTML
    remains self-contained and loads regardless of browser file:// restrictions.
    max_edge=480 keeps individual images ~60KB and total HTML ~1MB per case.
    """
    if not path:
        return ""
    p = Path(str(path))
    if not p.exists():
        return ""
    try:
        from PIL import Image
        import io
        img = Image.open(p).convert("RGB")
        img.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        data = p.read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


def _esc(value: Any) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _list_preview(values: Any, limit: int = 3) -> str:
    if not isinstance(values, list) or not values:
        return "none"
    return "; ".join(str(v) for v in values[:limit])


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _svg_multiline(
    text: Any,
    x: float,
    y: float,
    max_chars: int = 34,
    max_lines: int = 2,
    cls: str | None = None,
) -> str:
    words = str(text or "").split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else f"{cur} {word}"
        if len(trial) <= max_chars:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
        if len(lines) >= max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if not lines:
        lines = [""]
    if len(lines) == max_lines and len(" ".join(words)) > len(" ".join(lines)):
        lines[-1] = lines[-1][: max(0, max_chars - 1)].rstrip() + "..."
    class_attr = f' class="{_esc(cls)}"' if cls else ""
    return "".join(
        f'<text{class_attr} x="{x:.1f}" y="{y + i * 14:.1f}">{_esc(line)}</text>'
        for i, line in enumerate(lines)
    )
