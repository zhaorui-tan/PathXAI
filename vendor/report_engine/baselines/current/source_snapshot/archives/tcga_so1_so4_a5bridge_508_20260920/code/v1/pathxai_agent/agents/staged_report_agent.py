from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from pathxai_agent.llm.qwen_client import QwenVisionClient
from pathxai_agent.storage import SEG_CACHE_DIR, WSI_THUMBNAIL_CACHE_DIR


LONG_REPORT_FORMAT = """Specimen/Site:
Diagnosis:
Tumor/Lesion Description:
Key Microscopic Findings:
Invasion/Necrosis/Margins:
Immunohistochemistry/Ancillary Studies:
Stage/Grade/Prognostic Features:
Final Summary:"""

LONG_REPORT_SECTIONS = [
    "Specimen/Site",
    "Diagnosis",
    "Tumor/Lesion Description",
    "Key Microscopic Findings",
    "Invasion/Necrosis/Margins",
    "Immunohistochemistry/Ancillary Studies",
    "Stage/Grade/Prognostic Features",
    "Final Summary",
]


def run_staged_report_agent(
    case_id: str,
    candidate_rois: list[dict],
    qwen_client: QwenVisionClient | None,
    out_dir: str | Path,
    max_selected_rois: int = 5,
    max_triage_candidates: int | None = None,
    max_site_rois: int = 3,
    dry_run: bool = False,
) -> dict:
    """Run staged ROI -> site -> diagnosis -> report generation.

    This is intentionally separate from the direct PathAgent-Qwen baseline so
    the older early-ablation experiment remains reproducible.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_rows(candidate_rois)
    if not candidates:
        raise ValueError("No candidate ROIs were provided to staged report agent")

    site_candidates, tumor_candidates, has_separate_evidence = _split_site_tumor_candidates(candidates)
    selection_pool = tumor_candidates if has_separate_evidence and tumor_candidates else candidates
    site_evidence = site_candidates[:max_site_rois] if has_separate_evidence and site_candidates else None

    roi_selection = select_diagnostic_rois(
        selection_pool,
        qwen_client=qwen_client,
        max_selected_rois=max_selected_rois,
        max_triage_candidates=max_triage_candidates,
        dry_run=dry_run,
    )
    selected_tumor = _selected_candidates(selection_pool, roi_selection, max_selected_rois)
    selected = _dedupe_selected(selected_tumor + (site_evidence or []))
    site_input = _dedupe_selected((site_evidence or []) + selected_tumor)
    site_prediction = predict_site(
        site_input,
        qwen_client=qwen_client,
        dry_run=dry_run,
    )
    diagnosis_prediction = predict_diagnosis(
        selected_tumor,
        site_prediction=site_prediction,
        segmentation_context=None,
        morphology_context=None,
        qwen_client=qwen_client,
        dry_run=dry_run,
    )
    sufficiency = assess_evidence_sufficiency(
        selected,
        site_prediction=site_prediction,
        diagnosis_prediction=diagnosis_prediction,
        segmentation_context=None,
        knowledge_context=None,
        morphology_context=None,
        qwen_client=qwen_client,
        dry_run=dry_run,
    )
    report_payload = generate_long_report(
        selected,
        site_prediction=site_prediction,
        diagnosis_prediction=diagnosis_prediction,
        sufficiency=sufficiency,
        qwen_client=qwen_client,
        dry_run=dry_run,
    )
    payload = {
        "case_id": case_id,
        "mode": "staged_report_pipeline_v1",
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "candidate_rois": candidates,
        "site_rois": site_evidence or [],
        "tumor_rois": selected_tumor,
        "selected_rois": selected,
        "roi_selection": roi_selection,
        "site_prediction": site_prediction,
        "diagnosis_prediction": diagnosis_prediction,
        "evidence_sufficiency": sufficiency,
        "generated_long_report": report_payload["long_report"],
        "raw_generation": report_payload.get("raw"),
    }
    _attach_wsi_thumbnails(payload, out_dir)
    save_staged_outputs(payload, out_dir)
    return payload


def run_staged_report_agent_hybrid(
    case_id: str,
    candidate_rois: list[dict],
    stages_client: QwenVisionClient | None,
    short_report_client: QwenVisionClient | None,
    long_report_client: QwenVisionClient | None,
    out_dir: str | Path,
    max_selected_rois: int = 5,
    max_triage_candidates: int | None = None,
    max_site_rois: int = 3,
    dry_run: bool = False,
    enforce_staged_diagnosis: bool = False,
    known_site: str | None = None,
    segmentation_config: dict | None = None,
    knowledge_config: dict | None = None,
) -> dict:
    """Run staged pipeline with hybrid MedGemma→Qwen3 report generation.

    Stages 1-4 (ROI selection, site, diagnosis, sufficiency) use `stages_client`.
    Stage 5a (short pathological description) uses `short_report_client` (MedGemma).
    Stage 5b (full long report expansion) uses `long_report_client` (Qwen3).

    This preserves all previous ablations; results format is identical to
    ``run_staged_report_agent`` so they can be compared directly.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_rows(candidate_rois)
    if not candidates:
        raise ValueError("No candidate ROIs were provided to staged report agent")

    site_candidates, tumor_candidates, has_separate_evidence = _split_site_tumor_candidates(candidates)
    selection_pool = tumor_candidates if has_separate_evidence and tumor_candidates else candidates
    site_evidence = site_candidates[:max_site_rois] if has_separate_evidence and site_candidates else None

    print(f"  [{case_id}] Stage 1 ROI triage: candidates={len(selection_pool)}", flush=True)
    roi_selection = select_diagnostic_rois(
        selection_pool,
        qwen_client=stages_client,
        max_selected_rois=max_selected_rois,
        max_triage_candidates=max_triage_candidates,
        dry_run=dry_run,
    )
    selected_tumor = _selected_candidates(selection_pool, roi_selection, max_selected_rois)
    selected = _dedupe_selected(selected_tumor + (site_evidence or []))
    site_input = _dedupe_selected((site_evidence or []) + selected_tumor)
    print(f"  [{case_id}] Stage 2 site: {'known metadata' if known_site else 'LLM image inference'}", flush=True)
    site_prediction = known_site_prediction(known_site) if known_site else predict_site(
        site_input,
        qwen_client=stages_client,
        dry_run=dry_run,
    )
    print(f"  [{case_id}] Stage 2b segmentation: {'enabled' if (segmentation_config or {}).get('enabled') else 'disabled'}", flush=True)
    segmentation_context = run_roi_segmentation(
        selected_tumor,
        segmentation_config=segmentation_config,
        out_dir=out_dir / "segmentation",
    )
    print(f"  [{case_id}] Stage 2c MedGemma morphology", flush=True)
    morphology_context = generate_medgemma_visual_description(
        selected_tumor,
        site_prediction=site_prediction,
        diagnosis_prediction=None,
        short_report_client=short_report_client,
        dry_run=dry_run,
    )
    print(f"  [{case_id}] Stage 3 diagnosis", flush=True)
    diagnosis_prediction = predict_diagnosis(
        selected_tumor,
        site_prediction=site_prediction,
        segmentation_context=segmentation_context,
        morphology_context=morphology_context,
        qwen_client=stages_client,
        dry_run=dry_run,
    )
    print(f"  [{case_id}] Stage 3b KEEP knowledge: {'enabled' if (knowledge_config or {}).get('enabled') else 'disabled'}", flush=True)
    knowledge_context = retrieve_knowledge_context(
        site_prediction=site_prediction,
        diagnosis_prediction=diagnosis_prediction,
        knowledge_config=knowledge_config,
    )
    print(f"  [{case_id}] Stage 4 evidence sufficiency", flush=True)
    sufficiency = assess_evidence_sufficiency(
        selected,
        site_prediction=site_prediction,
        diagnosis_prediction=diagnosis_prediction,
        segmentation_context=segmentation_context,
        knowledge_context=knowledge_context,
        morphology_context=morphology_context,
        qwen_client=stages_client,
        dry_run=dry_run,
    )
    visual_rois = _ensure_site_visual_roi(selected, site_evidence or [])
    print(f"  [{case_id}] Stage 5 report generation", flush=True)
    report_payload = generate_long_report_hybrid(
        selected,
        site_prediction=site_prediction,
        diagnosis_prediction=diagnosis_prediction,
        sufficiency=sufficiency,
        segmentation_context=segmentation_context,
        knowledge_context=knowledge_context,
        short_report_client=short_report_client,
        long_report_client=long_report_client,
        dry_run=dry_run,
        enforce_staged_diagnosis=enforce_staged_diagnosis,
        visual_rois=visual_rois,
        medgemma_context=morphology_context,
    )
    payload = {
        "case_id": case_id,
        "mode": "staged_report_pipeline_hybrid_v1",
        "ablation_flags": {"enforce_staged_diagnosis": enforce_staged_diagnosis},
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "candidate_rois": candidates,
        "site_rois": site_evidence or [],
        "tumor_rois": selected_tumor,
        "selected_rois": selected,
        "visual_rois": visual_rois,
        "roi_selection": roi_selection,
        "site_prediction": site_prediction,
        "diagnosis_prediction": diagnosis_prediction,
        "segmentation_context": segmentation_context,
        "medgemma_morphology_context": morphology_context,
        "knowledge_context": knowledge_context,
        "evidence_sufficiency": sufficiency,
        "generated_long_report": report_payload["long_report"],          # clean (no figure refs)
        "generated_long_report_rich": report_payload.get("long_report_rich", report_payload["long_report"]),
        "medgemma_short_report": report_payload.get("short_report"),
        "figures": report_payload.get("figures") or [],
        "raw_generation": report_payload.get("raw"),
    }
    _attach_wsi_thumbnails(payload, out_dir)
    # Copy ROI images to figures/ directory before saving trace
    _save_figures(payload["figures"], out_dir)
    save_staged_outputs(payload, out_dir)
    # Also save the MedGemma short report for ablation analysis
    if report_payload.get("short_report"):
        (out_dir / "medgemma_short_report.txt").write_text(
            report_payload["short_report"], encoding="utf-8"
        )
    return payload


def select_diagnostic_rois(
    candidates: list[dict],
    qwen_client: QwenVisionClient | None,
    max_selected_rois: int,
    max_triage_candidates: int | None = None,
    dry_run: bool = False,
) -> dict:
    triage_count = len(candidates) if max_triage_candidates is None else max_triage_candidates
    triage_count = max(max_selected_rois, min(len(candidates), triage_count))
    top = candidates[:max(1, triage_count)]
    if dry_run:
        return {
            "selected_roi_indices": list(range(1, min(max_selected_rois, len(top)) + 1)),
            "reason": "DRY_RUN selected top foundation-ranked ROIs.",
            "roi_scores": [],
            "raw": None,
        }
    if qwen_client is None:
        raise ValueError("qwen_client is required unless dry_run=True")
    prompt = f"""/no_think
You are selecting WSI ROI evidence for pathology report generation.

You will see candidate ROI images in the same order as the list below.
Choose up to {max_selected_rois} ROIs that best support diagnosis and report generation.

Selection criteria:
- Must contain viable lesional/tumor tissue when possible.
- Prefer ROIs with diagnostic architecture, cytology, necrosis, invasion interface, or representative tumor.
- Avoid mostly blank/background/artifact/normal tissue.
- Select fewer than {max_selected_rois} if fewer are truly useful.
- Do not infer site or diagnosis from filenames, case IDs, directory names, or coordinates.

Candidate list:
{candidate_text(top)}

Return ONLY JSON:
{{
  "selected_roi_indices": [1, 2],
  "reason": "why these ROIs are enough or why more would be needed",
  "roi_scores": [
    {{"roi_index": 1, "diagnostic_value": 0-5, "tumor_likelihood": 0-5, "tissue_quality": 0-5, "reason": "..."}}
  ]
}}"""
    raw = qwen_client.chat_json(prompt, image_paths=[row["image_path"] for row in top])
    data = raw.get("json") or {}
    return {
        "selected_roi_indices": _int_list(data.get("selected_roi_indices")),
        "reason": str(data.get("reason") or ""),
        "roi_scores": data.get("roi_scores") if isinstance(data.get("roi_scores"), list) else [],
        "triage_candidate_count": len(top),
        "total_candidate_count": len(candidates),
        "raw": _raw_summary(raw),
    }


def predict_site(
    selected_rois: list[dict],
    qwen_client: QwenVisionClient | None,
    dry_run: bool = False,
) -> dict:
    foundation_hint = _foundation_site_hint(selected_rois)
    if dry_run:
        return {
            "predicted_site": "DRY_RUN",
            "confidence": "low",
            "top_alternatives": [],
            "visual_evidence": [],
            "foundation_site_hint": foundation_hint,
            "raw": None,
        }
    if qwen_client is None:
        raise ValueError("qwen_client is required unless dry_run=True")
    prompt = f"""/no_think
You are predicting specimen/site from selected pathology ROI images.

Use only morphology visible in the ROI images plus the foundation site candidates.
The foundation site candidates are computed from all tissue tiles in the WSI by comparing
image embeddings to all site prompt embeddings, then aggregating the highest-confidence
site types at slide level.
If the foundation site candidates are high confidence with a clear margin, treat them as strong site evidence.
Do not overturn a high-confidence foundation site hint just because tumor morphology mimics another organ
such as liver, kidney, or adrenal; describe the mimicry in limitations instead.
If the foundation site candidates are low confidence, use them only as ranked hypotheses and
decide from visual morphology whether any site is supportable.
If the organ/site is not visually identifiable, say "not specified from ROI evidence".
Do not use filenames, case IDs, TCGA codes, directory names, or coordinates.

Foundation site candidates:
{json.dumps(foundation_hint, indent=2, ensure_ascii=False)}

Selected ROI evidence:
{candidate_text(selected_rois)}

Return ONLY JSON:
{{
  "predicted_site": "...",
  "confidence": "high/medium/low",
  "top_alternatives": ["...", "..."],
  "visual_evidence": ["visible morphology supporting the site"],
  "limitations": ["why site may be uncertain"]
}}"""
    raw = qwen_client.chat_json(prompt, image_paths=[row["image_path"] for row in selected_rois])
    data = raw.get("json") or {}
    data["foundation_site_hint"] = foundation_hint
    data["raw"] = _raw_summary(raw)
    return data


def known_site_prediction(site: str | None) -> dict:
    site = str(site or "").strip()
    return {
        "predicted_site": site or "not specified from ROI evidence",
        "confidence": "high" if site else "low",
        "top_alternatives": [],
        "visual_evidence": [
            "Specimen/site was provided as known dataset metadata for this ablation."
        ] if site else [],
        "limitations": [
            "Known site is metadata context, not inferred from ROI morphology."
        ],
        "source": "known_site_metadata" if site else "unknown",
    }


def run_roi_segmentation(
    selected_rois: list[dict],
    segmentation_config: dict | None,
    out_dir: Path,
) -> dict:
    cfg = segmentation_config or {}
    if not cfg.get("enabled"):
        return {"available": False, "reason": "segmentation disabled"}
    try:
        from pathxai_agent.segmentation.medsam_roi import MedSAMRoiSegmenter

        segmenter = MedSAMRoiSegmenter(
            ckpt_path=cfg.get("ckpt_path"),
            device=cfg.get("device", "cuda:0"),
            max_rois=int(cfg.get("max_rois") or len(selected_rois) or 5),
            patch_size=int(cfg.get("patch_size") or 4096),
            stride=int(cfg.get("stride") or 2048),
            cache_dir=cfg.get("cache_dir", str(SEG_CACHE_DIR / "medsam3_d1")),
            python_path=cfg.get("python_path", "/local/conda_staffs/conda_envs/medsam/bin/python"),
            use_subprocess=bool(cfg.get("use_subprocess", True)),
        )
        result = segmenter.segment_rois(selected_rois, out_dir)
        try:
            from pathxai_agent.segmentation.seg_visual import attach_composites_to_seg_context

            result = attach_composites_to_seg_context(result)
        except Exception:
            pass
        return result
    except Exception as exc:  # noqa: BLE001
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {"available": False, "error": str(exc), "config": cfg}
        (out_dir / "segmentation_error.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return payload


def retrieve_knowledge_context(
    site_prediction: dict,
    diagnosis_prediction: dict,
    knowledge_config: dict | None,
) -> dict:
    cfg = knowledge_config or {}
    if not cfg.get("enabled"):
        return {"available": False, "reason": "knowledge retrieval disabled"}
    site = str(site_prediction.get("predicted_site") or "")
    diagnosis = str(diagnosis_prediction.get("primary_diagnosis") or "")
    query = " ".join([
        site,
        diagnosis,
        " ".join(str(x) for x in diagnosis_prediction.get("differential", []) if x),
        " ".join(str(x) for x in diagnosis_prediction.get("visual_evidence", []) if x),
    ])
    try:
        from pathxai_agent.knowledge.keep_retriever import retrieve_keep_knowledge

        return retrieve_keep_knowledge(
            query=query,
            site=site,
            diagnosis=diagnosis,
            keep_root=cfg.get("keep_root", "/local/datasets/public/KEEP_dataset"),
            max_disease=int(cfg.get("max_disease") or 5),
            max_semantic_groups=int(cfg.get("max_semantic_groups") or 5),
        )
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc), "config": cfg}


def retrieve_hypothesis_knowledge_context(
    hypothesis_set: list[dict],
    site_prediction: dict,
    knowledge_config: dict | None,
) -> dict:
    """Phase 2 KEEP: retrieve fine-grained knowledge anchored to each hypothesis."""
    cfg = knowledge_config or {}
    if not cfg.get("enabled"):
        return {"available": False, "reason": "knowledge retrieval disabled"}
    site = str(site_prediction.get("predicted_site") or "")
    try:
        from pathxai_agent.knowledge.keep_retriever import retrieve_hypothesis_knowledge

        return retrieve_hypothesis_knowledge(
            hypothesis_set=hypothesis_set,
            site=site,
            keep_root=cfg.get("keep_root", "/local/datasets/public/KEEP_dataset"),
            max_captions_per_hypothesis=int(cfg.get("max_captions_per_hypothesis") or 3),
            max_children=int(cfg.get("max_children") or 5),
        )
    except Exception as exc:
        return {"available": False, "error": str(exc), "config": cfg}


def normalize_diagnosis_hypotheses(data: dict, min_hypotheses: int = 3) -> dict:
    """Ensure diagnosis output carries a usable differential hypothesis set."""
    out = dict(data or {})
    primary = str(out.get("primary_diagnosis") or "").strip()
    primary_conf = str(out.get("confidence") or "medium").strip() or "medium"
    raw_hypotheses = out.get("hypothesis_set") or []
    hypotheses: list[dict] = []
    seen: set[str] = set()

    def add(label: str, confidence: str = "low", source: str = "normalized") -> None:
        label = str(label or "").strip()
        if not label:
            return
        key = label.lower()
        if key in seen:
            return
        seen.add(key)
        hypotheses.append({
            "diagnosis": label,
            "supporting_evidence": [],
            "missing_discriminators": list(out.get("discriminative_evidence_needed") or []),
            "confidence": confidence,
            "source": source,
        })

    if isinstance(raw_hypotheses, list):
        for row in raw_hypotheses:
            if isinstance(row, dict):
                label = str(row.get("diagnosis") or row.get("name") or row.get("label") or "").strip()
                if not label:
                    continue
                key = label.lower()
                if key in seen:
                    continue
                seen.add(key)
                hyp = dict(row)
                hyp["diagnosis"] = label
                hyp.setdefault("confidence", hyp.get("confidence") or "low")
                hyp.setdefault("supporting_evidence", [])
                hyp.setdefault("missing_discriminators", list(out.get("discriminative_evidence_needed") or []))
                hypotheses.append(hyp)
            elif row:
                add(str(row), "low", "normalized_from_hypothesis_text")

    if primary:
        add(primary, primary_conf, "primary_diagnosis")

    diff = out.get("differential") or []
    if isinstance(diff, list):
        for label in diff:
            add(str(label), "low", "differential")

    if hypotheses and len(hypotheses) < min_hypotheses:
        out["hypothesis_set_warning"] = (
            f"Only {len(hypotheses)} diagnostic hypothesis/hypotheses were available after normalization; "
            "the model did not provide a full differential set."
        )
    out["hypothesis_set"] = hypotheses
    out["hypothesis_set_normalized"] = True
    return out


def predict_diagnosis(
    selected_rois: list[dict],
    site_prediction: dict,
    segmentation_context: dict | None,
    morphology_context: dict | None,
    qwen_client: QwenVisionClient | None,
    dry_run: bool = False,
    seg_visual_mode: str = "text_only",
    knowledge_context: dict | None = None,
    diagnostic_session_context: dict | None = None,
) -> dict:
    if dry_run:
        return {
            "primary_diagnosis": "DRY_RUN",
            "confidence": "low",
            "differential": [],
            "visual_evidence": [],
            "missing_information": [],
            "limitations": [],
            "recommended_ancillary_studies": [],
            "raw": None,
        }
    if qwen_client is None:
        raise ValueError("qwen_client is required unless dry_run=True")

    # Build seg evidence text
    seg_text = json.dumps(_strip_heavy(segmentation_context or {"available": False}), indent=2, ensure_ascii=False)

    # Build seg visual note
    overlay_roi_indices: list[int] = []
    if seg_visual_mode == "composite":
        seg_visual_note = (
            "\nImage order: first image is a global tissue class color legend "
            "(Neoplastic=red, Inflammatory=blue, Connective=orange, Epithelial=green, Dead=purple, Gland=brown). "
            "Each subsequent image is a pair: left side = original H&E ROI, right side = its D1 segmentation overlay. "
            "Use the legend to interpret overlay colors."
        )
    elif seg_visual_mode == "none":
        seg_text = '{"available": false, "note": "segmentation intentionally disabled for this ablation"}'
        seg_visual_note = ""
    else:
        seg_visual_note = ""

    prompt = f"""/no_think
You are building a morphology-based diagnostic hypothesis set from selected WSI ROI images.
{seg_visual_note}
Use the site prediction only as context. Do not force a diagnosis if morphology is insufficient.
If the site prediction is low confidence or "not specified from ROI evidence", keep the diagnosis broad and morphology-based.
Do not anchor on a single early malignant diagnosis. First consider a differential set, then choose the most supportable primary diagnosis.
If the key discriminating evidence is missing, keep the primary diagnosis conservative/broad rather than upgrading to a more malignant entity.
Do not infer an organ-specific diagnosis from morphology alone when it conflicts with the provided high-confidence site context. If morphology resembles a tumor from another organ, keep that entity as a low-confidence differential and request confirmatory ancillary studies instead of making it the primary diagnosis.
For every cancer/site type, apply the same standard: do not upgrade to a more specific or more malignant diagnosis unless the selected ROI evidence directly supports the required discriminating features. If key features are not assessable, use a cautious broad category for that site.
Return 3 to 5 diagnostic hypotheses whenever possible, even if one is most likely. Include broad/cautious categories and important mimics when visual evidence is insufficient. Do not return a single-item hypothesis_set unless the images unequivocally exclude the main differentials.
MedGemma morphology evidence is descriptive support only. Do not copy its suspected diagnoses unless they are independently supported by the ROI images, segmentation evidence, and site context.
D1 segmentation classes are tissue-pattern evidence, not lineage labels: Inflammatory fraction can reflect lymphoid-rich stroma/inflammation and is not sufficient to diagnose lymphoma; Neoplastic fraction does not determine tumor lineage by itself.
If the site is high-confidence known metadata, keep non-site-lineage entities as differential/secondary involvement unless multiple independent ROI images show direct lineage-defining morphology and the site-specific primary lineage is excluded.
Do not use filenames, case IDs, TCGA codes, directory names, or coordinates.
Be explicit about uncertainty.

Site prediction:
{json.dumps(site_prediction, indent=2, ensure_ascii=False)}

Segmentation evidence:
{seg_text}

MedGemma morphology evidence:
{json.dumps(_strip_heavy(morphology_context or {"available": False}), indent=2, ensure_ascii=False)}

External pathology knowledge:
{json.dumps(_strip_heavy(knowledge_context or {"available": False}), indent=2, ensure_ascii=False)}

Case diagnostic session memory:
{json.dumps(_strip_heavy(diagnostic_session_context or {"available": False}), indent=2, ensure_ascii=False)}

Selected ROI evidence:
{candidate_text(selected_rois)}

Return ONLY JSON:
{{
  "primary_diagnosis": "...",
  "confidence": "high/medium/low",
  "hypothesis_set": [
    {{"diagnosis": "...", "supporting_evidence": ["..."], "missing_discriminators": ["..."], "confidence": "high/medium/low"}},
    {{"diagnosis": "...", "supporting_evidence": ["..."], "missing_discriminators": ["..."], "confidence": "high/medium/low"}},
    {{"diagnosis": "...", "supporting_evidence": ["..."], "missing_discriminators": ["..."], "confidence": "high/medium/low"}}
  ],
  "differential": ["...", "..."],
  "discriminative_evidence_needed": ["vascular invasion", "mitotic activity", "necrosis", "..."],
  "visual_evidence": ["architecture/cytology/necrosis/invasion evidence"],
  "missing_information": ["IHC not available", "margin not assessable"],
  "limitations": ["..."],
  "recommended_ancillary_studies": ["Order site- and differential-appropriate IHC/molecular tests to confirm lineage", "Consider proliferation marker if relevant", "..."]
}}"""
    # Build image paths based on seg_visual_mode
    try:
        from pathxai_agent.segmentation.seg_visual import build_image_paths_for_rois
        image_paths = build_image_paths_for_rois(selected_rois, segmentation_context or {}, seg_visual_mode)
    except Exception:
        image_paths = [row["image_path"] for row in selected_rois]
    raw = qwen_client.chat_json(prompt, image_paths=image_paths)
    data = raw.get("json") or {}
    data = normalize_diagnosis_hypotheses(data)
    data["raw"] = _raw_summary(raw)
    return data


def assess_evidence_sufficiency(
    selected_rois: list[dict],
    site_prediction: dict,
    diagnosis_prediction: dict,
    segmentation_context: dict | None,
    knowledge_context: dict | None,
    morphology_context: dict | None,
    qwen_client: QwenVisionClient | None,
    dry_run: bool = False,
    diagnostic_session_context: dict | None = None,
) -> dict:
    if dry_run:
        return {
            "sufficient_for_report": False,
            "sufficient_for_diagnosis": False,
            "reason": "DRY_RUN",
            "missing_evidence": [],
            "raw": None,
        }
    if qwen_client is None:
        raise ValueError("qwen_client is required unless dry_run=True")
    prompt = f"""/no_think
Assess whether the selected ROI evidence is sufficient for a pathology report.

Judge sufficiency for the differential diagnosis, not just for confirming the current primary diagnosis.
If evidence supports a broad neoplasm category but cannot distinguish borderline/uncertain potential from carcinoma,
mark diagnosis sufficiency as false and request discriminative visual evidence instead of forcing a malignant diagnosis.
Separate missing evidence into:
- image-actionable evidence that could plausibly be obtained by inspecting more WSI ROIs, such as necrosis, tumor-normal interface, capsular/vascular invasion-like areas, mitotic activity-like high power regions, architecture, cytology, gland formation, inflammation, or tissue quality.
- nonimage metadata that cannot be reliably obtained from WSI ROI images alone, such as gross tumor size, margins, TNM stage, IHC, molecular data, clinical history, procedure details, or formal mitotic count when the field-of-view/magnification is insufficient.
Only recommend select_more_rois when there is important missing_image_evidence.
For image-actionable gaps, choose the reinspection action carefully:
- use zoom_current_roi when the missing evidence needs higher-power detail inside an already suspicious ROI
  (cytology, nuclear atypia, mitotic activity-like foci, subtle architecture).
- use add_independent_roi when the missing evidence needs another tissue region
  (necrosis, tumor-normal interface, invasive front, alternative morphology, tissue quality).
- use require_nonimage_metadata when the missing evidence is IHC, molecular, gross size, margin,
  TNM/stage, clinical history, or formal quantified mitotic count.

Inputs:
Site prediction:
{json.dumps(site_prediction, indent=2, ensure_ascii=False)}

Diagnosis prediction:
{json.dumps(diagnosis_prediction, indent=2, ensure_ascii=False)}

Segmentation evidence:
{json.dumps(_strip_heavy(segmentation_context or {"available": False}), indent=2, ensure_ascii=False)}

MedGemma morphology evidence:
{json.dumps(_strip_heavy(morphology_context or {"available": False}), indent=2, ensure_ascii=False)}

External knowledge context:
{json.dumps(_strip_heavy(knowledge_context or {"available": False}), indent=2, ensure_ascii=False)}

Case diagnostic session memory:
{json.dumps(_strip_heavy(diagnostic_session_context or {"available": False}), indent=2, ensure_ascii=False)}

Return ONLY JSON:
{{
  "sufficient_for_diagnosis": true/false,
  "sufficient_for_report": true/false,
  "reason": "...",
  "missing_evidence": ["vascular/capsular invasion", "mitotic activity", "tumor-normal interface", "necrosis", "IHC", "..."],
  "missing_image_evidence": ["necrosis-rich ROI", "tumor-normal interface", "vascular/capsular invasion-like area", "..."],
  "missing_nonimage_metadata": ["gross tumor size", "IHC", "molecular data", "margin status", "..."],
  "discriminative_questions": ["What evidence would distinguish the top hypotheses?"],
  "recommended_next_step": "generate_report_with_limitations or select_more_rois or zoom_current_roi or require_nonimage_metadata",
  "reinspection_action": "none or zoom_current_roi or add_independent_roi or require_nonimage_metadata",
  "target_roi_ids": ["roi id to zoom if applicable"],
  "zoom_regions": ["center", "top-left", "top-right", "bottom-left", "bottom-right"],
  "reinspection_reason": "brief reason for the action"
}}"""
    raw = qwen_client.chat_json(prompt, image_paths=[row["image_path"] for row in selected_rois])
    data = raw.get("json") or {}
    data["raw"] = _raw_summary(raw)
    return data


def generate_long_report(
    selected_rois: list[dict],
    site_prediction: dict,
    diagnosis_prediction: dict,
    sufficiency: dict,
    qwen_client: QwenVisionClient | None,
    dry_run: bool = False,
) -> dict:
    if dry_run:
        return {
            "long_report": """Specimen/Site: DRY_RUN.
Diagnosis: DRY_RUN.
Tumor/Lesion Description: DRY_RUN.
Key Microscopic Findings: DRY_RUN.
Invasion/Necrosis/Margins: Not assessable from provided ROI patches.
Immunohistochemistry/Ancillary Studies: Not provided.
Stage/Grade/Prognostic Features: Not assessable from provided ROI patches.
Final Summary: DRY_RUN.""",
            "raw": None,
        }
    if qwen_client is None:
        raise ValueError("qwen_client is required unless dry_run=True")
    prompt = f"""/no_think
Generate one structured long pathology report from staged evidence.

You must follow exactly this section order:
{LONG_REPORT_FORMAT}

Staged site prediction:
{json.dumps(site_prediction, indent=2, ensure_ascii=False)}

Staged diagnosis prediction:
{json.dumps(diagnosis_prediction, indent=2, ensure_ascii=False)}

Evidence sufficiency:
{json.dumps(sufficiency, indent=2, ensure_ascii=False)}

Rules:
- Base visual claims on selected ROI images and staged evidence.
- Do not mention ROI, patch, crop, selected evidence, filenames, coordinates, model, or agent steps in the final report.
- Do not invent IHC, molecular, stage, grade, gross size, margin, or lymph-node status
  unless explicitly listed in the staged diagnosis's recommended_ancillary_studies.
  When recommended_ancillary_studies are present (especially with medium/low confidence),
  include them in the "Immunohistochemistry/Ancillary Studies" section prefixed with
  "Recommended for confirmation: ...".
- For items not listed in recommended_ancillary_studies, write "Not provided"
  or "Not assessable from provided ROI patches."
- If site or diagnosis is uncertain, state uncertainty clearly.
- Keep the report concise and clinically styled.

Return only the report text, no markdown fences."""
    attempts = []
    raw = qwen_client.chat(prompt, image_paths=[row["image_path"] for row in selected_rois])
    report = raw.get("content", "").strip()
    attempts.append({"name": "image_report", "valid": is_valid_long_report(report), "raw": _raw_summary(raw)})
    if is_valid_long_report(report):
        return {
            "long_report": report,
            "raw": {"attempts": attempts, "fallback_used": False},
        }

    retry_prompt = f"""/no_think
The previous pathology report response was empty or did not follow the required report format.

Generate the final long pathology report using ONLY the staged evidence below.
Do not mention ROI, patch, crop, evidence, model, or agent steps.
Do not invent unavailable gross size, margin, lymph-node, IHC, molecular, stage, or grade information
unless explicitly listed in the staged diagnosis's recommended_ancillary_studies.
When recommended_ancillary_studies are present, include them in the
"Immunohistochemistry/Ancillary Studies" section.
Use exactly these sections and keep each section concise:
{LONG_REPORT_FORMAT}

Staged site prediction:
{json.dumps(_strip_raw(site_prediction), indent=2, ensure_ascii=False)}

Staged diagnosis prediction:
{json.dumps(_strip_raw(diagnosis_prediction), indent=2, ensure_ascii=False)}

Evidence sufficiency:
{json.dumps(_strip_raw(sufficiency), indent=2, ensure_ascii=False)}

Return only the report text."""
    raw_retry = qwen_client.chat(retry_prompt, image_paths=[])
    retry_report = raw_retry.get("content", "").strip()
    attempts.append({"name": "text_retry", "valid": is_valid_long_report(retry_report), "raw": _raw_summary(raw_retry)})
    if is_valid_long_report(retry_report):
        return {
            "long_report": retry_report,
            "raw": {"attempts": attempts, "fallback_used": False},
        }

    fallback = build_fallback_long_report(site_prediction, diagnosis_prediction, sufficiency)
    attempts.append({"name": "deterministic_fallback", "valid": True, "raw": None})
    return {
        "long_report": fallback,
        "raw": {"attempts": attempts, "fallback_used": True},
    }


def generate_medgemma_visual_description(
    selected_rois: list[dict],
    site_prediction: dict,
    diagnosis_prediction: dict | None,
    short_report_client: QwenVisionClient | None,
    dry_run: bool = False,
    previous_context: dict | None = None,
    round_index: int = 1,
    input_role: str = "initial",
) -> dict:
    """Generate ROI morphology evidence with MedGemma before diagnosis/reporting."""
    input_labels = [f"ROI {idx}" for idx, _ in enumerate(selected_rois, start=1)]
    if dry_run:
        return {
            "available": False,
            "description": "DRY_RUN_MEDGEMMA",
            "round": round_index,
            "input_role": input_role,
            "input_roi_ids": input_labels,
            "previous_summary_used": bool(previous_context),
            "raw": None,
        }
    if short_report_client is None:
        return {
            "available": False,
            "description": "",
            "round": round_index,
            "input_role": input_role,
            "input_roi_ids": input_labels,
            "previous_summary_used": bool(previous_context),
            "error": "short_report_client is not available",
            "raw": None,
        }

    attempts = []
    site_hint = str(site_prediction.get("predicted_site") or "unknown tissue").strip()
    diag_hint = str((diagnosis_prediction or {}).get("primary_diagnosis") or "").strip()
    known_site_text = ""
    if site_prediction.get("source") == "known_site_metadata" and site_hint:
        known_site_text = f"Known specimen/site: {site_hint}. "
    previous_text = str((previous_context or {}).get("description") or "").strip()
    previous_block = ""
    if previous_text and round_index == 1:
        # Only use previous context for the very first round
        previous_block = (
            f"\nPrior impression from earlier ROI rounds (for context only, do NOT repeat): {previous_text[:200]}\n"
        )
    medgemma_prompt = (
        "You are an instructor teaching medical students. "
        "For education purposes, provide a brief descriptive text for the set of pathology "
        "patches extracted from a pathology slide. "
        f"{known_site_text}"
        f"This is morphology evidence round {round_index} ({input_role}). "
        "Assume the procedure and diagnosis are unknown. "
        "Base the morphology description only on visible morphology in the images. "
        "Include visible architecture, cytology, tumor/lesion presence, "
        "necrosis or stromal reaction if present, and uncertainty if the images are not definitive. "
        "Do not make a definitive final diagnosis from limited ROI patches; "
        "provide a ranked differential of up to 3 suspected diagnoses with brief reasoning for each. "
        "Do not mention filenames, coordinates, models, or selection steps.\n"
        f"{previous_block}"
        "Use exactly this concise format:\n"
        "Site context: ...\n"
        "Findings: ...\n"
        "Impression: ...\n"
        "Top 3 suspected diagnoses:\n"
        "  1. [diagnosis] — [brief morphologic reasoning]\n"
        "  2. [diagnosis] — [brief morphologic reasoning]\n"
        "  3. [diagnosis] — [brief morphologic reasoning]\n"
        "Limitations: ...\n"
        "Do not repeat bullet points or phrases.\n"
        "pathology slide patches:"
    )
    image_paths = [row["image_path"] for row in selected_rois if row.get("image_path")]
    try:
        raw_short = short_report_client.chat(medgemma_prompt, image_paths=image_paths)
        short_report = _strip_medgemma_collapse(raw_short.get("content", "").strip())
        attempts.append({
            "name": "medgemma_visual_description",
            "valid": is_valid_medgemma_description(short_report),
            "raw": _raw_summary(raw_short),
        })
    except Exception as exc:
        short_report = ""
        attempts.append({"name": "medgemma_visual_description", "valid": False, "error": str(exc), "raw": None})

    if not is_valid_medgemma_description(short_report):
        retry_prompt = medgemma_prompt + (
            "\n\nThe previous response was too short or poorly formatted. Please provide a richer "
            "educational description while keeping exactly the four requested fields and avoiding repeated lines."
        )
        try:
            raw_retry = short_report_client.chat(retry_prompt, image_paths=image_paths)
            retry_report = _strip_medgemma_collapse(raw_retry.get("content", "").strip())
            attempts.append({
                "name": "medgemma_visual_description_retry",
                "valid": is_valid_medgemma_description(retry_report),
                "raw": _raw_summary(raw_retry),
            })
            if is_valid_medgemma_description(retry_report):
                short_report = retry_report
        except Exception as exc:
            attempts.append({"name": "medgemma_visual_description_retry", "valid": False, "error": str(exc), "raw": None})

    if not is_valid_medgemma_description(short_report):
        short_report = (
            f"Site context: {site_hint}. "
            f"Findings: {'; '.join(str(x) for x in (diagnosis_prediction or {}).get('visual_evidence', [])[:4])}. "
            f"Impression: {diag_hint or 'not determined from ROI morphology'}. "
            f"Limitations: {'; '.join(str(x) for x in (diagnosis_prediction or {}).get('limitations', [])[:3])}."
        )
        attempts.append({"name": "medgemma_fallback_to_staged", "valid": True, "raw": None})

    return {
        "available": bool(short_report),
        "description": short_report,
        "round": round_index,
        "input_role": input_role,
        "input_roi_ids": input_labels,
        "previous_summary_used": bool(previous_text),
        "raw": {"attempts": attempts},
    }


def generate_long_report_hybrid(
    selected_rois: list[dict],
    site_prediction: dict,
    diagnosis_prediction: dict,
    sufficiency: dict,
    segmentation_context: dict | None,
    knowledge_context: dict | None,
    short_report_client: QwenVisionClient | None,
    long_report_client: QwenVisionClient | None,
    dry_run: bool = False,
    enforce_staged_diagnosis: bool = False,
    visual_rois: list[dict] | None = None,
    medgemma_context: dict | None = None,
    diagnostic_session_context: dict | None = None,
) -> dict:
    """Ablation flag ``enforce_staged_diagnosis``:
    When True, the Stage 5b prompt includes a hard constraint requiring the
    Diagnosis section to reproduce the staged ``primary_diagnosis`` exactly.
    This prevents the LLM from downgrading / overriding the diagnosis determined
    in Stage 3, which is the main failure mode observed in early ACC results.
    """
    """Generate long report via MedGemma short report → Qwen3 expansion.

    Stage 5a: MedGemma generates a concise pathological description from ROI images,
               leveraging its strong pathological knowledge.
    Stage 5b: Qwen3 takes MedGemma's short report + staged evidence and expands it
              into the full structured long report, leveraging its long-text ability.

    The prompt style for MedGemma follows the official MedGemma example:
    https://github.com/Google-Health/medgemma/blob/main/notebooks/high_dimensional_pathology_hugging_face.ipynb
    """
    if dry_run:
        return {
            "long_report": """Specimen/Site: DRY_RUN_HYBRID.
Diagnosis: DRY_RUN_HYBRID.
Tumor/Lesion Description: DRY_RUN_HYBRID.
Key Microscopic Findings: DRY_RUN_HYBRID.
Invasion/Necrosis/Margins: Not assessable from provided ROI patches.
Immunohistochemistry/Ancillary Studies: Not provided.
Stage/Grade/Prognostic Features: Not assessable from provided ROI patches.
Final Summary: DRY_RUN_HYBRID (MedGemma→Qwen3 hybrid).""",
            "short_report": "DRY_RUN_HYBRID: short pathological description.",
            "raw": None,
        }
    if short_report_client is None or long_report_client is None:
        raise ValueError("Both short_report_client and long_report_client are required unless dry_run=True")

    # ── Stage 5a: MedGemma short pathological description ──
    print("    Stage 5a MedGemma visual description", flush=True)
    report_rois = visual_rois or selected_rois
    medgemma_context = medgemma_context or generate_medgemma_visual_description(
        report_rois,
        site_prediction=site_prediction,
        diagnosis_prediction=diagnosis_prediction,
        short_report_client=short_report_client,
        dry_run=dry_run,
    )
    short_report = str((medgemma_context or {}).get("description") or "")
    attempts = list(((medgemma_context or {}).get("raw") or {}).get("attempts") or [])

    # ── Stage 5b: Qwen3 expands short report → full long report ──
    print("    Stage 5b Qwen long report", flush=True)
    figures_legend, figures = _build_figures(
        report_rois,
        segmentation_context=segmentation_context,
    )
    qwen_image_paths, qwen_image_legend = _build_qwen_report_image_inputs(figures, fallback_rois=report_rois)

    # ── Build optional enforce_staged_diagnosis constraint block ───────────
    _primary_diag = str(diagnosis_prediction.get("primary_diagnosis") or "").strip()
    _diag_conf = str(diagnosis_prediction.get("confidence") or "low").lower()
    if enforce_staged_diagnosis and _primary_diag:
        _diag_constraint = (
            f"\nHARD CONSTRAINT — enforce_staged_diagnosis is ACTIVE:\n"
            f"  The Diagnosis section MUST state '{_primary_diag}' as the primary diagnosis.\n"
            f"  Do NOT soften, downgrade, or contradict this diagnosis in any section.\n"
            f"  You may note diagnostic uncertainty or list a differential ONLY IF "
            f"the staged confidence is '{_diag_conf}' (i.e., low/medium),\n"
            f"  but the primary diagnosis line must still be '{_primary_diag}'.\n"
        )
    else:
        _diag_constraint = ""

    long_prompt = f"""/no_think
Generate one structured long pathology report using the expert pathological findings below and the staged evidence.

You must follow exactly this section order:
{LONG_REPORT_FORMAT}

Figure reference legend (the images are provided in this exact order):
{figures_legend}

Qwen visual input order:
{qwen_image_legend}

Expert pathological findings (from a specialist review of the ROI images):
---
{short_report}
---

Staged site prediction:
{json.dumps(site_prediction, indent=2, ensure_ascii=False)}

Staged diagnosis prediction:
{json.dumps(diagnosis_prediction, indent=2, ensure_ascii=False)}

Evidence sufficiency:
{json.dumps(sufficiency, indent=2, ensure_ascii=False)}

Segmentation evidence and statistics:
{json.dumps(_strip_heavy(segmentation_context or {"available": False}), indent=2, ensure_ascii=False)}

External pathology knowledge context:
{json.dumps(_strip_heavy(knowledge_context or {"available": False}), indent=2, ensure_ascii=False)}

Case diagnostic session memory:
{json.dumps(_strip_heavy(diagnostic_session_context or {"available": False}), indent=2, ensure_ascii=False)}
{_diag_constraint}
Rules:
- Use the expert pathological findings primarily for visible morphology details.
- Treat MedGemma/short-report suspected diagnoses as non-binding morphology notes; do not let them override staged diagnosis, claim verification, site context, or segmentation-supported evidence.
- D1 segmentation classes are tissue-pattern evidence, not lineage labels; do not use Inflammatory fraction alone to diagnose lymphoma or Neoplastic fraction alone to determine tumor lineage.
- If staged diagnosis used a known-site lineage guard, write the guarded primary diagnosis as the main diagnosis and list the demoted non-site lineage only as a differential or limitation.
- Use staged site/diagnosis as the controlling structured interpretation, especially when
  the expert findings conflict with high-confidence staged evidence.
	- Most provided images are paired composites: left = original H&E ROI, right = segmentation
	  overlay for the same ROI. Some fallback images may be original H&E only.
	- Compare segmentation statistics/overlays against the visible H&E morphology and ignore
	  segmentation-derived statements that look inconsistent with the original ROI.
- Supplement with staged evidence where the findings are brief.
- Use segmentation statistics only as supporting quantitative evidence; do not treat
  segmentation as ground truth if it conflicts with visible morphology.
- Use external knowledge only for general disease/site background and differential
  reasoning. Do not invent patient-specific IHC, molecular, stage, size, or margins
  from the knowledge graph.
	- Cite [Figure N] inline when stating a specific finding you can visually confirm in the
	  original H&E part of that figure's paired composite. Use overlays only to audit
	  segmentation boundaries and class regions.
- Do not mention filenames, patch coordinates, selection process, or model/agent steps.
- Do not invent IHC, molecular, stage, grade, gross size, margin, or lymph-node status
  unless explicitly listed in the staged diagnosis's recommended_ancillary_studies.
  When recommended_ancillary_studies are present (especially with medium/low confidence),
  include them in the "Immunohistochemistry/Ancillary Studies" section prefixed with
  "Recommended for confirmation: ...".
- For items not listed in recommended_ancillary_studies, write "Not provided"
  or "Not assessable from provided ROI patches."
- If site or diagnosis is uncertain, state uncertainty clearly.
- Do not declare benign tissue if staged diagnosis predicts a neoplasm unless the staged
  diagnosis is low confidence and the images clearly show no lesion.

Return only the report text, no markdown fences."""
    roi_only_image_paths = [
        str(fig.get("image_path"))
        for fig in figures
        if fig.get("image_path") and Path(str(fig.get("image_path"))).is_file()
    ]

    try:
        raw_long = long_report_client.chat(
            long_prompt,
            image_paths=qwen_image_paths,
        )
        report = raw_long.get("content", "").strip()
        attempts.append({"name": "qwen3_long_report_full_images", "valid": is_valid_long_report(report), "raw": _raw_summary(raw_long)})
    except Exception as exc:
        report = ""
        attempts.append({"name": "qwen3_long_report_full_images", "valid": False, "error": str(exc), "raw": None})

    used_report_mode = "full_paired_composite_images"
    if not is_valid_long_report(report) and roi_only_image_paths and roi_only_image_paths != qwen_image_paths:
        try:
            raw_roi_only = long_report_client.chat(
                long_prompt + "\n\nIf segmentation overlay images were too expensive to process, use the original H&E ROI images plus the segmentation statistics text above.",
                image_paths=roi_only_image_paths,
            )
            report = raw_roi_only.get("content", "").strip()
            attempts.append({"name": "qwen3_long_report_roi_only_images", "valid": is_valid_long_report(report), "raw": _raw_summary(raw_roi_only)})
            if is_valid_long_report(report):
                used_report_mode = "roi_only_images_fallback"
        except Exception as exc:
            attempts.append({"name": "qwen3_long_report_roi_only_images", "valid": False, "error": str(exc), "raw": None})

    if is_valid_long_report(report):
        return {
            "long_report": _strip_figure_refs(report),
            "long_report_rich": report,
            "short_report": short_report,
            "figures": figures,
            "raw": {"attempts": attempts, "fallback_used": used_report_mode != "full_paired_composite_images", "report_mode": used_report_mode},
        }

    # Retry with text only, so API timeouts do not fail the whole case.
    retry_prompt = f"""/no_think
The previous pathology report response was empty or did not follow the required report format.

Generate the final long pathology report using the expert findings and staged evidence below.
Do not mention filenames, patch coordinates, or model/agent steps.
Cite [Figure N] inline when stating a specific finding visible in that image.
Do not invent unavailable gross size, margin, lymph-node, IHC, molecular, stage, or grade information
unless explicitly listed in the staged diagnosis's recommended_ancillary_studies.
When recommended_ancillary_studies are present, include them in the
"Immunohistochemistry/Ancillary Studies" section.
Use exactly these sections and keep each section concise:
{LONG_REPORT_FORMAT}

Figure reference legend:
{figures_legend}

Qwen visual input order:
{qwen_image_legend}

Expert pathological findings:
---
{short_report}
---

Staged site prediction:
{json.dumps(_strip_raw(site_prediction), indent=2, ensure_ascii=False)}

Staged diagnosis prediction:
{json.dumps(_strip_raw(diagnosis_prediction), indent=2, ensure_ascii=False)}

Evidence sufficiency:
{json.dumps(_strip_raw(sufficiency), indent=2, ensure_ascii=False)}

Segmentation evidence and statistics:
{json.dumps(_strip_heavy(segmentation_context or {"available": False}), indent=2, ensure_ascii=False)}

External pathology knowledge context:
{json.dumps(_strip_heavy(knowledge_context or {"available": False}), indent=2, ensure_ascii=False)}

Case diagnostic session memory:
{json.dumps(_strip_heavy(diagnostic_session_context or {"available": False}), indent=2, ensure_ascii=False)}

Return only the report text."""
    try:
        raw_retry = long_report_client.chat(retry_prompt, image_paths=[])
        retry_report = raw_retry.get("content", "").strip()
        attempts.append({"name": "qwen3_text_retry", "valid": is_valid_long_report(retry_report), "raw": _raw_summary(raw_retry)})
    except Exception as exc:
        retry_report = ""
        attempts.append({"name": "qwen3_text_retry", "valid": False, "error": str(exc), "raw": None})
    if is_valid_long_report(retry_report):
        return {
            "long_report": _strip_figure_refs(retry_report),
            "long_report_rich": retry_report,
            "short_report": short_report,
            "figures": figures,
            "raw": {"attempts": attempts, "fallback_used": True, "report_mode": "text_only_retry"},
        }

    fallback = build_fallback_long_report(site_prediction, diagnosis_prediction, sufficiency)
    attempts.append({"name": "deterministic_fallback", "valid": True, "raw": None})
    return {
        "long_report": fallback,
        "long_report_rich": fallback,
        "short_report": short_report,
        "figures": figures,
        "raw": {"attempts": attempts, "fallback_used": True},
    }


def save_staged_outputs(payload: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "staged_trace.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "roi_selection_scores.json").write_text(
        json.dumps(payload["roi_selection"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "site_prediction.json").write_text(
        json.dumps(payload["site_prediction"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "diagnosis_prediction.json").write_text(
        json.dumps(payload["diagnosis_prediction"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if "segmentation_context" in payload:
        (out_dir / "segmentation_context.json").write_text(
            json.dumps(payload["segmentation_context"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if "knowledge_context" in payload:
        (out_dir / "knowledge_context.json").write_text(
            json.dumps(payload["knowledge_context"], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    (out_dir / "evidence_sufficiency.json").write_text(
        json.dumps(payload["evidence_sufficiency"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "generated_long_report.txt").write_text(payload["generated_long_report"], encoding="utf-8")
    if payload.get("generated_long_report_rich"):
        (out_dir / "generated_long_report_rich.txt").write_text(
            payload["generated_long_report_rich"], encoding="utf-8"
        )
    if payload.get("medgemma_short_report"):
        (out_dir / "medgemma_short_report.txt").write_text(
            payload["medgemma_short_report"], encoding="utf-8"
        )
    (out_dir / "report.md").write_text(render_staged_markdown(payload), encoding="utf-8")
    if payload.get("figures"):
        try:
            html = render_report_html(payload)
            (out_dir / "report.html").write_text(html, encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            pass  # HTML is best-effort; don't fail the pipeline


def render_staged_markdown(payload: dict) -> str:
    lines = [
        f"# Staged Report Pipeline: {payload.get('case_id')}",
        f"Mode: {payload.get('mode', 'unknown')}",
        "",
        "## Selected ROIs",
        "",
    ]
    for idx, row in enumerate(payload.get("selected_rois") or [], start=1):
        lines.append(f"- {idx}. `{row.get('roi_id')}` score={row.get('score')} path={row.get('image_path')}")
    if payload.get("visual_rois"):
        lines.extend(["", "## Visualized ROIs", ""])
        for idx, row in enumerate(payload.get("visual_rois") or [], start=1):
            role = row.get("evidence_role") or "unknown"
            lines.append(f"- {idx}. `{row.get('roi_id')}` role={role} score={row.get('score')} path={row.get('image_path')}")
    lines.extend([
        "",
        "## ROI Selection",
        "",
        json.dumps(payload.get("roi_selection"), indent=2, ensure_ascii=False),
        "",
        "## Site Prediction",
        "",
        json.dumps(payload.get("site_prediction"), indent=2, ensure_ascii=False),
        "",
        "## Diagnosis Prediction",
        "",
        json.dumps(payload.get("diagnosis_prediction"), indent=2, ensure_ascii=False),
        "",
        "## Evidence Sufficiency",
        "",
        json.dumps(payload.get("evidence_sufficiency"), indent=2, ensure_ascii=False),
        "",
    ])
    if payload.get("medgemma_short_report"):
        lines.extend([
            "## MedGemma Short Report (Hybrid Pipeline)",
            "",
            payload["medgemma_short_report"],
            "",
        ])
    if payload.get("figures"):
        lines.extend(["## Visual Evidence Figures", ""])
        for entry in payload["figures"]:
            fig_file = entry.get("figure_file") or entry.get("image_path", "")
            lines.append(
                f"- **{entry['figure']}**: ROI #{entry.get('roi_index')}  "
                f"score={entry.get('score', 0):.3f}  `{fig_file}`"
            )
        lines.append("")
    lines.extend([
        "## Generated Long Report",
        "",
        payload.get("generated_long_report") or "",
        "",
    ])
    return "\n".join(lines)


def _candidate_rows(candidate_rois: list[dict]) -> list[dict]:
    rows = []
    for idx, row in enumerate(candidate_rois, start=1):
        metadata = dict(row.get("metadata") or {})
        rows.append({
            "roi_index": idx,
            "roi_id": str(row.get("roi_id") or Path(str(row.get("image_path"))).stem),
            "image_path": str(row.get("image_path")),
            "x": row.get("x"),
            "y": row.get("y"),
            "width": row.get("width"),
            "height": row.get("height"),
            "score": float(row.get("score") or 0.0),
            "site_rank_score": _optional_float(row.get("site_rank_score")),
            "tumor_rank_score": _optional_float(row.get("tumor_rank_score")),
            "combined_rank_score": _optional_float(row.get("combined_rank_score")),
            "foundation_tile_index": row.get("foundation_tile_index"),
            "evidence_role": row.get("evidence_role"),
            "top_site": row.get("top_site"),
            "top_site_score": _optional_float(row.get("top_site_score")),
            "site_margin_to_second": _optional_float(row.get("site_margin_to_second")),
            "top_k_sites": row.get("top_k_sites") if isinstance(row.get("top_k_sites"), list) else [],
            "source": row.get("source"),
            "metadata": metadata,
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["roi_index"] = idx
    return rows


def _split_site_tumor_candidates(candidates: list[dict]) -> tuple[list[dict], list[dict], bool]:
    site = [row for row in candidates if row.get("evidence_role") in {"site", "site_support"}]
    tumor = [row for row in candidates if row.get("evidence_role") == "tumor"]
    return site, tumor, bool(site or tumor)


def _dedupe_selected(rows: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for row in rows:
        key = (
            str(row.get("image_path") or ""),
            row.get("x"),
            row.get("y"),
            row.get("width"),
            row.get("height"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _ensure_site_visual_roi(selected: list[dict], site_evidence: list[dict]) -> list[dict]:
    """Ensure HTML/figure exports include at least one site-support ROI."""
    visual = list(selected)
    has_site = any(str(row.get("evidence_role") or "") in {"site", "site_support"} for row in visual)
    if has_site or not site_evidence:
        return _dedupe_selected(visual)
    return _dedupe_selected([site_evidence[0]] + visual)


def _selected_candidates(candidates: list[dict], roi_selection: dict, max_selected_rois: int) -> list[dict]:
    indices = [idx for idx in _int_list(roi_selection.get("selected_roi_indices")) if 1 <= idx <= len(candidates)]
    selected = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            selected.append(candidates[idx - 1])
            seen.add(idx)
    if not selected:
        selected = candidates[:max_selected_rois]
    return selected[:max_selected_rois]


def _foundation_site_hint(selected_rois: list[dict]) -> dict:
    slide_candidates = []
    for row in selected_rois:
        candidates = ((row.get("metadata") or {}).get("foundation_slide_site_candidates"))
        if isinstance(candidates, list) and candidates:
            slide_candidates = candidates
            break
    if slide_candidates:
        top = slide_candidates[0] if isinstance(slide_candidates[0], dict) else {}
        return {
            "available": True,
            "method": "slide_site_type_aggregation_from_all_tissue_tiles",
            "top_site": top.get("site"),
            "confidence": top.get("confidence") or "low",
            "top_score": top.get("aggregate_score"),
            "margin_to_second": top.get("margin_to_second"),
            "top_alternatives": [
                {
                    "site": item.get("site"),
                    "score": item.get("aggregate_score"),
                    "margin_to_second": item.get("margin_to_second"),
                    "supporting_tile_indices": item.get("supporting_tile_indices"),
                }
                for item in slide_candidates[1:5]
                if isinstance(item, dict)
            ],
            "slide_site_candidates": [
                {
                    "rank": item.get("rank"),
                    "site": item.get("site"),
                    "score": item.get("aggregate_score"),
                    "confidence": item.get("confidence"),
                    "margin_to_second": item.get("margin_to_second"),
                    "supporting_tile_indices": item.get("supporting_tile_indices"),
                }
                for item in slide_candidates[:5]
                if isinstance(item, dict)
            ],
            "support_roi_site_scores": [
                {
                    "roi_index": row.get("roi_index"),
                    "role": row.get("evidence_role"),
                    "top_site": row.get("top_site"),
                    "top_site_score": row.get("top_site_score"),
                    "margin_to_second": row.get("site_margin_to_second"),
                    "top_k_sites": row.get("top_k_sites"),
                }
                for row in selected_rois
                if row.get("top_site") or row.get("top_k_sites")
            ][:8],
        }
    for row in selected_rois:
        hint = ((row.get("metadata") or {}).get("foundation_site_prediction"))
        if isinstance(hint, dict) and hint.get("available"):
            confidence = str(hint.get("confidence") or "").lower()
            margin = hint.get("margin_to_second")
            try:
                margin_value = float(margin) if margin is not None else None
            except Exception:
                margin_value = None
            if confidence == "low" or (margin_value is not None and margin_value < 0.02):
                return {
                    "available": False,
                    "reason": "foundation site hint was low confidence and should not constrain site/diagnosis",
                    "low_confidence_top_site": hint.get("top_site"),
                    "confidence": hint.get("confidence"),
                    "top_score": hint.get("top_score"),
                    "margin_to_second": hint.get("margin_to_second"),
                    "top_alternatives": [
                        {"site": item.get("site"), "score": item.get("score")}
                        for item in (hint.get("scores") or [])[1:5]
                    ],
                }
            return {
                "top_site": hint.get("top_site"),
                "confidence": hint.get("confidence"),
                "top_score": hint.get("top_score"),
                "margin_to_second": hint.get("margin_to_second"),
                "top_alternatives": [
                    {"site": item.get("site"), "score": item.get("score")}
                    for item in (hint.get("scores") or [])[1:5]
                ],
            }
    return {"available": False}


def _strip_figure_refs(text: str) -> str:
    """Remove [Figure N] citations for metric computation (clean version)."""
    return re.sub(r"\s*\[Figure\s+\d+\]", "", text or "").strip()


def _env_unlimited_int(name: str, default: int) -> int | None:
    """Return None for <=0, meaning unlimited."""
    try:
        value = int(os.environ.get(name, str(default)))
    except Exception:
        value = default
    return None if value <= 0 else value


def _under_cap(paths: list[str], cap: int | None) -> bool:
    return cap is None or len(paths) < cap


def _build_qwen_report_image_inputs(figures: list[dict], fallback_rois: list[dict] | None = None) -> tuple[list[str], str]:
    """Build final-report visual inputs for Qwen.

    Preferred format is one global segmentation legend followed by one paired
    H&E|segmentation composite per selected/zoomed Figure.  If a paired
    composite cannot be resolved, fall back to that Figure's original H&E ROI.
    Environment caps use <=0 as "unlimited".
    """
    max_images = _env_unlimited_int("PATHXAI_QWEN_REPORT_MAX_IMAGES", 8)
    max_pairs = _env_unlimited_int("PATHXAI_QWEN_REPORT_MAX_OVERLAY_PAIRS", 3)
    image_paths: list[str] = []
    legend: list[str] = []
    paired_count = 0

    def maybe_append(path: str, label: str) -> bool:
        if not path or not Path(path).is_file() or not _under_cap(image_paths, max_images):
            return False
        image_paths.append(path)
        legend.append(f"Image {len(image_paths)}: {label}")
        return True

    has_any_pair = False
    for fig in figures or []:
        seg = fig.get("segmentation") if isinstance(fig.get("segmentation"), dict) else {}
        composite = str(
            seg.get("composite_path_file")
            or seg.get("composite_path")
            or fig.get("composite_path_file")
            or fig.get("composite_path")
            or ""
        )
        if composite and Path(composite).is_file():
            has_any_pair = True
            break
        roi_path = str(fig.get("image_path") or "")
        overlay_path = str(seg.get("overlay_path_file") or seg.get("overlay_path") or "")
        if roi_path and overlay_path and Path(roi_path).is_file() and Path(overlay_path).is_file():
            has_any_pair = True
            break

    if has_any_pair:
        try:
            from pathxai_agent.segmentation.seg_visual import build_global_seg_legend

            legend_path = build_global_seg_legend()
            if max_images is None or max_images > 1:
                maybe_append(str(legend_path), "global segmentation color legend")
        except Exception:
            pass

    for idx, fig in enumerate(figures or [], start=1):
        if not _under_cap(image_paths, max_images):
            break
        fig_label = fig.get("figure") or f"Figure {idx}"
        seg = fig.get("segmentation") if isinstance(fig.get("segmentation"), dict) else {}
        composite_path = str(
            seg.get("composite_path_file")
            or seg.get("composite_path")
            or fig.get("composite_path_file")
            or fig.get("composite_path")
            or ""
        )
        if composite_path and Path(composite_path).is_file() and _under_cap([], max_pairs if max_pairs is not None else None):
            if max_pairs is None or paired_count < max_pairs:
                if maybe_append(composite_path, f"{fig_label} paired H&E ROI + segmentation overlay"):
                    paired_count += 1
                    continue
        overlay_path = str(seg.get("overlay_path_file") or seg.get("overlay_path") or "")
        roi_path = str(fig.get("image_path") or "")
        if roi_path and overlay_path and Path(roi_path).is_file() and Path(overlay_path).is_file():
            if max_pairs is None or paired_count < max_pairs:
                try:
                    from pathxai_agent.segmentation.seg_visual import build_roi_seg_composite

                    result = build_roi_seg_composite(roi_path, overlay_path)
                    if maybe_append(str(result.get("composite_path") or ""), f"{fig_label} paired H&E ROI + segmentation overlay"):
                        paired_count += 1
                        continue
                except Exception:
                    pass
        maybe_append(roi_path, f"{fig_label} original H&E ROI")

    if not image_paths:
        for idx, row in enumerate(fallback_rois or [], start=1):
            if not _under_cap(image_paths, max_images):
                break
            roi_path = str(row.get("image_path") or "")
            maybe_append(roi_path, f"fallback ROI {idx} original H&E")
    return image_paths, "\n".join(legend) if legend else "No visual inputs available."


def _build_figures(
    selected_rois: list[dict],
    segmentation_context: dict | None = None,
) -> tuple[str, list[dict]]:
    """Build a numbered figure legend for report citations.

    Returns
    -------
    legend_text : str
        Multi-line string to embed in the LLM prompt.
    figures : list[dict]
        Mapping of figure number → ROI metadata, included in the output payload.
    """
    legend_lines: list[str] = []
    figures: list[dict] = []
    seg_by_roi_id, seg_by_image = _segmentation_lookup(segmentation_context)
    for n, row in enumerate(selected_rois, start=1):
        role = str(row.get("evidence_role") or "evidence")
        top_site = str(row.get("top_site") or "").strip()
        site_text = f"  top_site={top_site}" if top_site else ""
        legend_lines.append(
            f"Figure {n}: ROI patch #{row.get('roi_index')}  "
            f"role={role}  foundation_score={row.get('score', 0):.3f}  "
            f"size={row.get('width')}x{row.get('height')}{site_text}"
        )
        seg_row = seg_by_roi_id.get(str(row.get("roi_id") or "")) or seg_by_image.get(str(row.get("image_path") or ""))
        seg_payload = _strip_heavy(seg_row) if isinstance(seg_row, dict) else seg_row
        if isinstance(seg_payload, dict) and isinstance(seg_row, dict):
            for key in ("composite_path", "composite_path_file", "overlay_path_file", "mask_path_file"):
                if seg_row.get(key) and not seg_payload.get(key):
                    seg_payload[key] = seg_row.get(key)
        figures.append({
            "figure": f"Figure {n}",
            "roi_index": row.get("roi_index"),
            "roi_id": row.get("roi_id"),
            "image_path": row.get("image_path"),
            "score": row.get("score"),
            "evidence_role": row.get("evidence_role"),
            "top_site": row.get("top_site"),
            "top_site_score": row.get("top_site_score"),
            "site_margin_to_second": row.get("site_margin_to_second"),
            "top_k_sites": row.get("top_k_sites") if isinstance(row.get("top_k_sites"), list) else [],
            "site_rank_score": row.get("site_rank_score"),
            "tumor_rank_score": row.get("tumor_rank_score"),
            "segmentation": seg_payload,
        })
    return "\n".join(legend_lines), figures


def _segmentation_lookup(segmentation_context: dict | None) -> tuple[dict[str, dict], dict[str, dict]]:
    by_roi_id = {}
    by_image = {}
    if not isinstance(segmentation_context, dict):
        return by_roi_id, by_image
    for row in segmentation_context.get("roi_results") or []:
        if not isinstance(row, dict):
            continue
        if row.get("roi_id"):
            by_roi_id[str(row["roi_id"])] = row
        if row.get("image_path"):
            by_image[str(row["image_path"])] = row
    return by_roi_id, by_image


def _save_figures(figures: list[dict], out_dir: Path) -> None:
    """Register figure artifacts without duplicating large ROI/segmentation files.

    By default reports reference the shared ROI and segmentation caches.  This
    keeps multi-experiment ablations small: the same foundation-selected ROI and
    the same MedSAM mask/overlay are stored once and reused by many runs.

    Set PATHXAI_SELF_CONTAINED_FIGURES=1 only when you need a portable report
    directory that can be moved without the cache.
    """
    if os.environ.get("PATHXAI_SELF_CONTAINED_FIGURES", "0") != "1":
        for entry in figures:
            src = Path(str(entry.get("image_path") or ""))
            if src.is_file():
                entry["figure_file"] = str(src)
                entry["artifact_storage"] = "shared_reference"
            seg = entry.get("segmentation") if isinstance(entry.get("segmentation"), dict) else {}
            for key in ["overlay_path", "mask_path", "composite_path"]:
                seg_src = Path(str(seg.get(key) or ""))
                if seg_src.is_file():
                    seg[f"{key}_file"] = str(seg_src)
            if seg:
                seg["artifact_storage"] = "shared_reference"
        return

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    for entry in figures:
        src = Path(str(entry.get("image_path") or ""))
        n = entry["figure"].replace("Figure ", "")  # e.g. "1"
        if src.is_file():
            dst = fig_dir / f"figure_{n}{src.suffix or '.jpg'}"
            shutil.copy2(src, dst)
            entry["figure_file"] = str(dst.relative_to(out_dir))
        seg = entry.get("segmentation") if isinstance(entry.get("segmentation"), dict) else {}
        for key, suffix in [("overlay_path", "overlay.jpg"), ("mask_path", "mask.png"), ("composite_path", "composite.jpg")]:
            seg_src = Path(str(seg.get(key) or ""))
            if seg_src.is_file():
                seg_dst = fig_dir / f"figure_{n}_{suffix}"
                shutil.copy2(seg_src, seg_dst)
                seg[f"{key}_file"] = str(seg_dst.relative_to(out_dir))


def candidate_text(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        extra = []
        if row.get("evidence_role"):
            extra.append(f"role={row.get('evidence_role')}")
        if row.get("site_rank_score") is not None:
            extra.append(f"site_score={row.get('site_rank_score'):.3f}")
        if row.get("tumor_rank_score") is not None:
            extra.append(f"tumor_score={row.get('tumor_rank_score'):.3f}")
        score_text = "; " + "; ".join(extra) if extra else ""
        lines.append(
            f"[ROI {row['roi_index']}] foundation_score={row.get('score')}; "
            f"size={row.get('width')}x{row.get('height')}{score_text}"
        )
    return "\n".join(lines)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def is_valid_long_report(text: str, min_tokens: int = 60) -> bool:
    if len((text or "").split()) < min_tokens:
        return False
    section_count = 0
    for section in LONG_REPORT_SECTIONS:
        pattern = rf"(?im)^\s*{re.escape(section)}\s*:"
        if re.search(pattern, text or ""):
            section_count += 1
    return section_count >= 6


def is_valid_medgemma_description(text: str, min_tokens: int = 35) -> bool:
    text = text or ""
    if len(text.split()) < min_tokens:
        return False
    lower = text.lower()
    # Keep this lightweight: the official prompt asks for brief descriptive
    # text, not a strict report template.
    concepts = [
        ("architecture", "arranged", "pattern", "nested", "trabecular", "gland"),
        ("cell", "cytology", "nuclei", "cytoplasm", "atypia"),
        ("tumor", "lesion", "neoplasm", "carcinoma", "benign"),
    ]
    hits = 0
    for group in concepts:
        if any(term in lower for term in group):
            hits += 1
    return hits >= 2


def _strip_medgemma_collapse(text: str) -> str:
    """Truncate MedGemma output at the first sign of LLM collapse (repeated lines).

    MedGemma sometimes enters a loop generating "FINAL DIAGNOSIS: X" repeatedly
    after completing the requested format.  Detect and strip this suffix.
    """
    lines = text.splitlines()
    # Find the first "FINAL DIAGNOSIS:" or "FINAL PATHOLOGY DIAGNOSIS:" line
    # and keep only content before it (the requested format sections).
    collapse_markers = ("final diagnosis:", "final pathology diagnosis:", "morphology:")
    cut_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if any(stripped.startswith(m) for m in collapse_markers):
            cut_idx = i
            break
    if cut_idx is not None and cut_idx > 0:
        return "\n".join(lines[:cut_idx]).strip()
    # Fallback: detect line repetition (same line >= 3 times in a row)
    repeat_count = 0
    last_line = ""
    for i, line in enumerate(lines):
        s = line.strip()
        if s and s == last_line:
            repeat_count += 1
            if repeat_count >= 2:
                return "\n".join(lines[: i - repeat_count + 1]).strip()
        else:
            repeat_count = 0
            last_line = s
    return text


def build_fallback_long_report(site_prediction: dict, diagnosis_prediction: dict, sufficiency: dict) -> str:
    site = str(site_prediction.get("predicted_site") or "Not specified from ROI evidence").strip()
    diagnosis = str(diagnosis_prediction.get("primary_diagnosis") or "Not specified from ROI evidence").strip()
    site_conf = str(site_prediction.get("confidence") or "not specified").strip()
    diag_conf = str(diagnosis_prediction.get("confidence") or "not specified").strip()
    visual_evidence = diagnosis_prediction.get("visual_evidence")
    if isinstance(visual_evidence, list) and visual_evidence:
        microscopic = "; ".join(str(item).strip() for item in visual_evidence if str(item).strip())
    else:
        microscopic = "Diagnostic microscopic details are limited by the available ROI evidence."
    limitations = []
    for source in (diagnosis_prediction.get("missing_information"), sufficiency.get("missing_evidence")):
        if isinstance(source, list):
            limitations.extend(str(item).strip() for item in source if str(item).strip())
    limitation_text = "; ".join(dict.fromkeys(limitations)) or "Margins, gross findings, IHC, molecular data, and staging are not provided."
    # Include recommended ancillary studies in the IHC section when available
    rec_studies = diagnosis_prediction.get("recommended_ancillary_studies")
    if isinstance(rec_studies, list) and rec_studies:
        ihc_line = "Immunohistochemistry/Ancillary Studies: " + "; ".join(str(s) for s in rec_studies)
    else:
        ihc_line = "Immunohistochemistry/Ancillary Studies: Not provided."
    return "\n".join([
        f"Specimen/Site: {site} (confidence: {site_conf}).",
        f"Diagnosis: {diagnosis} (confidence: {diag_conf}).",
        "Tumor/Lesion Description: Tumor size and gross lesion description are not provided in the image evidence.",
        f"Key Microscopic Findings: {microscopic}",
        "Invasion/Necrosis/Margins: Not assessable from provided ROI patches unless explicitly noted in the microscopic findings.",
        ihc_line,
        "Stage/Grade/Prognostic Features: Not assessable from provided ROI patches.",
        f"Final Summary: Image-based staged assessment favors {diagnosis}. Important limitations: {limitation_text}",
    ])


def _strip_raw(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_raw(val) for key, val in value.items() if key != "raw"}
    if isinstance(value, list):
        return [_strip_raw(item) for item in value]
    return value


def _strip_heavy(value: Any) -> Any:
    value = _strip_raw(value)
    if isinstance(value, dict):
        out = {}
        for key, val in value.items():
            if key in {"raw", "all_roi_site_scores", "image_path"}:
                continue
            if key == "roi_results" and isinstance(val, list):
                out[key] = [
                    {
                        "roi_index": row.get("roi_index"),
                        "stats": row.get("stats"),
                        "error": row.get("error"),
                    }
                    for row in val[:8]
                    if isinstance(row, dict)
                ]
            elif key in {"disease_hits", "semantic_group_hits"} and isinstance(val, list):
                out[key] = val[:5]
            else:
                out[key] = _strip_heavy(val)
        return out
    if isinstance(value, list):
        return [_strip_heavy(item) for item in value[:12]]
    return value


def _raw_summary(raw: dict | None) -> dict | None:
    if not raw:
        return None
    return {
        "content": raw.get("content"),
        "reasoning": raw.get("reasoning"),
        "elapsed_sec": raw.get("elapsed_sec"),
    }


# ── HTML report renderer ─────────────────────────────────────────────────────

def _slide_artifact_key(slide_path: str | Path) -> str:
    p = Path(str(slide_path))
    stat_text = ""
    try:
        st = p.stat()
        stat_text = f"|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        pass
    return hashlib.sha1((str(p.resolve() if p.exists() else p) + stat_text).encode("utf-8")).hexdigest()[:20]


def _wsi_thumbnail_cache_dir(slide_path: str | Path) -> Path:
    return WSI_THUMBNAIL_CACHE_DIR / _slide_artifact_key(slide_path)


def _generate_wsi_thumbnail(slide_path: str | Path, cache_dir: Path | None = None, max_size: int = 1024) -> Path | None:
    """Generate a WSI overview thumbnail and cache it as wsi_thumbnail.jpg.

    Returns the path to the cached thumbnail, or None on failure.
    max_size=1024 gives enough detail for the lightbox while staying small (~100-300 KB).
    """
    cache_dir = cache_dir or _wsi_thumbnail_cache_dir(slide_path)
    thumb_path = cache_dir / f"thumbnail_max{max_size}.jpg"
    if thumb_path.exists():
        return thumb_path
    if not Path(str(slide_path)).is_file():
        return None
    try:
        from pathxai_agent.wsi.slide_reader import SlideReader  # noqa: PLC0415
        reader = SlideReader(str(slide_path))
        try:
            thumb = reader.thumbnail(max_size=max_size)
        finally:
            reader.close()
        cache_dir.mkdir(parents=True, exist_ok=True)
        thumb.save(str(thumb_path), format="JPEG", quality=88, optimize=True)
        return thumb_path
    except Exception:  # noqa: BLE001
        return None


def _get_wsi_thumbnail_from_rois(candidates: list[dict], fallback_dir: Path) -> Path | None:
    """Find the first valid slide_path and return a globally cached thumbnail."""
    for roi in candidates:
        meta = roi.get("metadata") or {}
        slide_path = meta.get("slide_path")
        if not slide_path:
            continue
        return _generate_wsi_thumbnail(slide_path, _wsi_thumbnail_cache_dir(slide_path))
    return None


# Color palette used for ROI bounding-box overlays (thumbnail) and figure card dots.
# Index 0 = Figure 1 / ROI 1, etc.
_ROI_PALETTE_HEX = [
    "#3b82f6",  # blue
    "#ef4444",  # red
    "#22c55e",  # green
    "#f59e0b",  # amber
    "#a855f7",  # purple
    "#14b8a6",  # teal
    "#ec4899",  # pink
    "#fb923c",  # orange
]
_ROI_PALETTE_RGB = [
    (59, 130, 246), (239, 68, 68), (34, 197, 94), (245, 158, 11),
    (168, 85, 247), (20, 184, 166), (236, 72, 153), (251, 146, 60),
]


def _annotate_wsi_thumbnail_with_rois(
    thumb_path: str | Path,
    rois: list[dict],
    slide_w: int,
    slide_h: int,
    out_path: Path,
) -> Path | None:
    """Draw selected-ROI bounding boxes on a copy of the clean WSI thumbnail.

    Each ROI gets a distinct colored border + a small numbered badge matching
    the order of figures in the report.  The annotated image is saved to
    out_path (typically the case output dir, not the shared ROI cache).
    Cached: returns immediately if out_path already exists.
    """
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
        lw = max(2, tw // 300)  # border width proportional to thumbnail size
        for i, roi in enumerate(rois):
            x0 = int((roi.get("x") or 0) * sx)
            y0 = int((roi.get("y") or 0) * sy)
            bw = max(lw + 2, int((roi.get("width") or 0) * sx))
            bh = max(lw + 2, int((roi.get("height") or 0) * sy))
            x1, y1 = x0 + bw, y0 + bh
            color = _ROI_PALETTE_RGB[i % len(_ROI_PALETTE_RGB)]
            # Colored border
            for t in range(lw):
                draw.rectangle([x0 + t, y0 + t, x1 - t, y1 - t], outline=color)
            # Numbered badge (filled square at top-left corner of box)
            badge = max(10, lw * 5)
            bx0, by0 = x0 + lw, y0 + lw
            draw.rectangle([bx0, by0, bx0 + badge, by0 + badge], fill=color)
            label = str(i + 1)
            # PIL default font is ~6 px wide × 8 px tall per char
            tx = bx0 + max(1, (badge - len(label) * 6) // 2)
            ty = by0 + max(1, (badge - 8) // 2)
            draw.text((tx, ty), label, fill=(255, 255, 255))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(str(out_path), format="JPEG", quality=88, optimize=True)
        return out_path
    except Exception:  # noqa: BLE001
        return None


def _attach_wsi_thumbnails(payload: dict, out_dir: Path) -> None:
    """Generate clean + ROI-annotated WSI thumbnails; write paths into payload."""
    candidates = payload.get("candidate_rois") or []
    selected = payload.get("visual_rois") or payload.get("selected_rois") or []
    # Clean thumbnail (shared ROI cache dir — reused across runs)
    clean = _get_wsi_thumbnail_from_rois(candidates, out_dir)
    if not clean:
        return
    payload["wsi_thumbnail_path"] = str(clean)
    # Annotated thumbnail (per-run case dir — specific to this ROI selection)
    if not selected or not candidates:
        return
    si = ((candidates[0].get("metadata") or {}).get("slide_info") or {})
    slide_w = int(si.get("width") or 0)
    slide_h = int(si.get("height") or 0)
    ann = _annotate_wsi_thumbnail_with_rois(
        clean, selected, slide_w, slide_h,
        out_dir / "wsi_thumbnail_annotated.jpg",
    )
    if ann:
        payload["wsi_thumbnail_annotated_path"] = str(ann)


def _img_data_uri(path: str, max_edge: int = 1024) -> str:
    """Return a base64 data URI, resizing to max_edge px (JPEG) to keep HTML small."""
    p = Path(str(path or ""))
    if not p.is_file():
        return ""
    try:
        from PIL import Image  # type: ignore  # noqa: PLC0415
        import io
        img = Image.open(p).convert("RGB")
        img.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82, optimize=True)
        data = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{data}"
    except Exception:  # noqa: BLE001
        try:
            ext = p.suffix.lower().lstrip(".")
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
            data = base64.b64encode(p.read_bytes()).decode()
            return f"data:{mime};base64,{data}"
        except Exception:  # noqa: BLE001
            return ""


def _parse_report_sections(text: str) -> list[tuple[str, str]]:
    """Split a report into (section_name, content) pairs."""
    pattern = (
        r"^(" + "|".join(re.escape(s) for s in LONG_REPORT_SECTIONS) + r")\s*:"
        r"(.*?)(?=^(?:" + "|".join(re.escape(s) for s in LONG_REPORT_SECTIONS) + r")\s*:|\Z)"
    )
    sections = []
    for m in re.finditer(pattern, text, re.DOTALL | re.MULTILINE):
        sections.append((m.group(1).strip(), m.group(2).strip()))
    if not sections:
        sections = [("Report", text.strip())]
    return sections


def _htmlesc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _inject_figure_links(text: str, figures: list[dict]) -> str:
    """Replace [Figure N] with a clickable inline badge."""
    def repl(m: re.Match) -> str:
        n = m.group(1)
        return (
            f'<a class="fig-ref" href="#figure-{n}" '
            f'title="Jump to Figure {n}">[Figure\u00a0{n}]</a>'
        )
    return re.sub(r"\[Figure\s+(\d+)\]", repl, _htmlesc(text))


# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
     background:#f0f2f5;color:#0f172a;font-size:14px;line-height:1.65;
     -webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration:none}
.page{max-width:1500px;margin:0 auto;padding:22px 18px}

/* \u2500\u2500 Header \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
.hdr{background:linear-gradient(120deg,#0e2a4f 0%,#1b4690 55%,#2563eb 100%);
     color:#fff;padding:24px 32px;border-radius:14px;margin-bottom:20px;
     display:flex;align-items:flex-start;gap:20px;
     box-shadow:0 6px 24px rgba(37,99,235,.22)}
.hdr-icon{font-size:2.4em;line-height:1;margin-top:3px;flex-shrink:0;opacity:.9}
.hdr h1{font-size:1.3em;font-weight:700;letter-spacing:-.01em;line-height:1.25}
.hdr-case{font-size:.83em;font-weight:600;opacity:.8;margin-top:4px;
          overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hdr-diag{font-size:.95em;font-weight:500;font-style:italic;opacity:.88;
          margin-top:4px;line-height:1.4}
.hdr-chips{display:flex;flex-wrap:wrap;gap:5px;margin-top:12px}
.hdr-chip{display:inline-flex;align-items:center;background:rgba(255,255,255,.14);
          border:1px solid rgba(255,255,255,.22);border-radius:20px;
          padding:3px 11px;font-size:.71em;font-weight:500}

/* \u2500\u2500 Badges / chips \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
.badge{display:inline-block;border-radius:20px;padding:2px 10px;font-size:.72em;
       font-weight:600;letter-spacing:.3px;white-space:nowrap}
.badge-high,.badge-true{background:#dcfce7;color:#15803d}
.badge-medium{background:#fef9c3;color:#92400e}
.badge-low,.badge-false{background:#fee2e2;color:#b91c1c}
.badge-neutral{background:#f1f5f9;color:#475569;border:1px solid #e2e8f0}
.chip{display:inline-block;background:#eff6ff;color:#2563eb;border-radius:20px;
      padding:2px 9px;font-size:.72em;margin:2px;border:1px solid #dbeafe}

/* \u2500\u2500 Panel card \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
.panel{background:#fff;border-radius:12px;padding:22px 24px;
       box-shadow:0 1px 4px rgba(0,0,0,.06),0 6px 18px rgba(0,0,0,.04);
       border:1px solid rgba(0,0,0,.05)}
.panel-title{font-size:.67em;font-weight:700;text-transform:uppercase;
             letter-spacing:1.3px;color:#94a3b8;margin-bottom:16px;
             display:flex;align-items:center;gap:8px}
.panel-title::after{content:'';flex:1;height:1px;background:#f1f5f9}
.panel-title .dot{width:7px;height:7px;border-radius:50%;background:#2563eb;flex-shrink:0}

/* \u2500\u2500 WSI Overview panel \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
.wsi-panel{position:sticky;top:20px;align-self:start}
.wsi-overview{width:100%;height:auto;display:block;border-radius:8px;
              cursor:zoom-in;border:1px solid #e2e8f0;
              box-shadow:0 1px 8px rgba(0,0,0,.08);transition:box-shadow .2s}
.wsi-overview:hover{box-shadow:0 3px 16px rgba(0,0,0,.15)}
.wsi-legend{display:flex;flex-wrap:wrap;gap:5px;margin-top:10px}
.wsi-fig-chip{display:inline-flex;align-items:center;gap:3px;color:#fff;
              font-size:.7em;font-weight:700;padding:3px 9px;
              border-radius:20px;letter-spacing:.2px}

/* \u2500\u2500 Main 3-col grid \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
.top-grid{display:grid;grid-template-columns:320px 1fr 278px;
          gap:18px;margin-bottom:20px;align-items:start}
.top-grid.no-wsi{grid-template-columns:1fr 278px}
@media(max-width:1100px){.top-grid{grid-template-columns:1fr 278px}}
@media(max-width:860px){.top-grid,.top-grid.no-wsi{grid-template-columns:1fr}}

/* \u2500\u2500 Report content \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
.medgemma-box{background:#eff6ff;border-left:3px solid #2563eb;
              border-radius:0 10px 10px 0;padding:12px 18px;margin-bottom:16px}
.medgemma-box .lbl{font-size:.67em;font-weight:700;text-transform:uppercase;
                   letter-spacing:1.1px;color:#1d4ed8;margin-bottom:6px}
.section{padding:10px 0;border-bottom:1px solid #f4f6f9}
.section:last-child{border-bottom:none;padding-bottom:0}
.section-name{font-size:.75em;font-weight:600;color:#2563eb;margin-bottom:4px;
              letter-spacing:.2px}
.section-content{line-height:1.78;color:#1e293b;font-size:.93em}
.fig-ref{display:inline-block;background:#eff6ff;color:#2563eb;border-radius:4px;
         border:1px solid #bfdbfe;padding:0 5px;font-size:.78em;
         font-family:monospace;cursor:pointer;transition:background .15s}
.fig-ref:hover{background:#dbeafe}
.clean-note{display:flex;align-items:flex-start;gap:8px;background:#f8fafc;
            border:1px solid #e2e8f0;border-radius:8px;padding:8px 12px;
            font-size:.74em;color:#64748b;margin-bottom:14px;line-height:1.5}

/* \u2500\u2500 Figure sidebar \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
.fig-card{background:#fff;border-radius:10px;margin-bottom:12px;
          box-shadow:0 1px 4px rgba(0,0,0,.07);overflow:hidden;
          scroll-margin-top:80px;border:1px solid rgba(0,0,0,.05);
          transition:box-shadow .2s,transform .1s}
.fig-card:hover{box-shadow:0 4px 16px rgba(0,0,0,.1);transform:translateY(-1px)}
.fig-card:target{box-shadow:0 0 0 2px #2563eb}
.fig-card img{width:100%;display:block;cursor:zoom-in}
.fig-card .cap{padding:9px 11px;background:#fafbfc;border-top:1px solid #f1f5f9;
               font-size:.76em;color:#475569}
.fig-card .cap strong{color:#0f172a;display:flex;align-items:center;
                      gap:5px;margin-bottom:3px}
.score-row{display:flex;align-items:center;gap:6px;margin-top:4px}
.score-label{font-size:.69em;color:#94a3b8;width:94px;flex-shrink:0}
.score-bar{flex:1;height:5px;background:#f1f5f9;border-radius:3px;overflow:hidden}
.score-fill{height:100%;border-radius:3px}

/* \u2500\u2500 Pipeline reasoning \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
details.pipeline-section>summary{list-style:none;cursor:pointer}
details.pipeline-section>summary::-webkit-details-marker,
details.pipeline-section>summary::marker{display:none}
.pipeline-hdr{font-size:.7em;font-weight:700;text-transform:uppercase;
              letter-spacing:1.3px;color:#94a3b8;
              margin:18px 0 12px;display:flex;align-items:center;gap:10px;user-select:none}
.pipeline-hdr::before{content:'';width:3px;height:14px;
                      border-radius:2px;background:#2563eb;flex-shrink:0}
.pipeline-hdr::after{content:'';flex:1;height:1px;background:#e2e8f0}
.stages-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));
             gap:12px;margin-bottom:16px}
.stage-card{background:#fff;border-radius:12px;padding:15px 17px;
            box-shadow:0 1px 4px rgba(0,0,0,.06);
            border:1px solid rgba(0,0,0,.05);border-left:3px solid #2563eb}
.stage-num{font-size:.63em;font-weight:700;text-transform:uppercase;
           letter-spacing:1.2px;color:#94a3b8;margin-bottom:3px}
.stage-title{font-size:.9em;font-weight:700;color:#0f172a;
             margin-bottom:10px;line-height:1.4}
.stage-value{font-size:.95em;font-weight:600;color:#1e293b;margin-bottom:8px}
.ev-list{list-style:none;padding:0}
.ev-list li{padding:4px 0 4px 14px;position:relative;font-size:.8em;
            color:#334155;border-bottom:1px solid #f8fafc;line-height:1.5}
.ev-list li:last-child{border-bottom:none}
.ev-list li::before{content:"\25b8";position:absolute;left:0;color:#2563eb;
                    font-size:.75em;top:5px}
.missing-list{list-style:none;padding:0}
.missing-list li{padding:3px 0 3px 18px;position:relative;
                 font-size:.78em;color:#b45309}
.missing-list li::before{content:"\26a0";position:absolute;left:0;font-size:.75em}
.alt-chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.reason-text{font-size:.79em;color:#475569;line-height:1.65;
             background:#f8fafc;border-radius:8px;padding:10px 12px;
             margin-top:8px;border:1px solid #f1f5f9}
.roi-score-card{background:#f8fafc;border-radius:8px;padding:9px 11px;
                margin-bottom:7px;font-size:.79em;border:1px solid #f1f5f9}
.roi-score-card .roi-hdr{font-weight:600;color:#0f172a;margin-bottom:5px}
.roi-score-card .roi-reason{font-size:.85em;color:#64748b;margin-top:4px;font-style:italic}

/* \u2500\u2500 Lightbox \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
.lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.88);
          z-index:1000;align-items:center;justify-content:center;cursor:zoom-out}
.lightbox.active{display:flex}
.lightbox img{max-width:92vw;max-height:92vh;border-radius:8px;
              box-shadow:0 8px 48px rgba(0,0,0,.6)}
"""

# ── JS ────────────────────────────────────────────────────────────────────────
_JS = """
function openLightbox(src){
  var lb=document.getElementById('lightbox');
  lb.querySelector('img').src=src;
  lb.classList.add('active');
}
document.addEventListener('DOMContentLoaded',function(){
  document.getElementById('lightbox').addEventListener('click',function(){
    this.classList.remove('active');
  });
  document.querySelectorAll('.fig-card img').forEach(function(img){
    img.addEventListener('click',function(){openLightbox(this.src);});
  });
});
"""


def _score_bar(value: float, max_val: float = 5.0) -> str:
    pct = min(100, max(0, value / max_val * 100))
    if pct >= 70:
        color = "#22c55e"
    elif pct >= 40:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    return (
        f'<div class="score-bar"><div class="score-fill" '
        f'style="width:{pct:.0f}%;background:{color}"></div></div>'
    )


def _confidence_badge(conf: str) -> str:
    c = str(conf or "").lower()
    cls = "badge-high" if c == "high" else ("badge-medium" if c == "medium" else "badge-low")
    return f'<span class="badge {cls}">{_htmlesc(str(conf or ""))}</span>'


def _bool_badge(val: bool | None, true_label: str = "Sufficient", false_label: str = "Insufficient") -> str:
    if val is True:
        return f'<span class="badge badge-true">✓ {true_label}</span>'
    elif val is False:
        return f'<span class="badge badge-false">✗ {false_label}</span>'
    return f'<span class="badge badge-neutral">Unknown</span>'


def _role_badge(role: str | None) -> str:
    role_text = str(role or "evidence")
    if role_text in {"site", "site_support"}:
        return '<span class="badge badge-medium">Site support</span>'
    if role_text == "tumor":
        return '<span class="badge badge-high">Tumor evidence</span>'
    return f'<span class="badge badge-neutral">{_htmlesc(role_text)}</span>'


def _ev_list(items: list) -> str:
    if not items:
        return ""
    lis = "".join(f"<li>{_htmlesc(str(i))}</li>" for i in items)
    return f'<ul class="ev-list">{lis}</ul>'


def _stage_roi_selection(roi_selection: dict, selected_rois: list[dict], figures: list[dict]) -> str:
    n_sel = len(selected_rois)
    n_cand = roi_selection.get("total_candidate_count") or roi_selection.get("triage_candidate_count") or "?"
    reason = roi_selection.get("reason") or ""
    roi_scores: list[dict] = roi_selection.get("roi_scores") or []

    # Build a lookup from roi_index → figure number
    fig_by_roi: dict[int, int] = {}
    for entry in figures:
        idx = entry.get("roi_index")
        fig_n = entry["figure"].replace("Figure ", "")
        if idx:
            fig_by_roi[int(idx)] = int(fig_n)

    score_cards = ""
    for rs in roi_scores:
        ridx = rs.get("roi_index", "?")
        fn = fig_by_roi.get(int(ridx), None) if str(ridx).isdigit() else None
        fig_label = f"→ Figure {fn}" if fn else ""
        dv = float(rs.get("diagnostic_value") or 0)
        tl = float(rs.get("tumor_likelihood") or 0)
        tq = float(rs.get("tissue_quality") or 0)
        rs_reason = rs.get("reason") or ""
        score_cards += (
            f'<div class="roi-score-card">'
            f'<div class="roi-hdr">ROI {ridx} {fig_label}</div>'
            f'<div class="score-row"><span class="score-label">Diagnostic value</span>{_score_bar(dv)} <b>{dv:.0f}</b>/5</div>'
            f'<div class="score-row"><span class="score-label">Tumor likelihood</span>{_score_bar(tl)} <b>{tl:.0f}</b>/5</div>'
            f'<div class="score-row"><span class="score-label">Tissue quality</span>{_score_bar(tq)} <b>{tq:.0f}</b>/5</div>'
            f'<div class="roi-reason">{_htmlesc(rs_reason)}</div>'
            f'</div>'
        )

    return (
        f'<div class="stage-card">'
        f'<div class="stage-num">Stage 1</div>'
        f'<div class="stage-title">ROI Selection</div>'
        f'<div class="stage-value">{n_sel} selected from {n_cand} candidates</div>'
        f'<div class="reason-text">{_htmlesc(reason)}</div>'
        f'{score_cards}'
        f'</div>'
    )


def _stage_site(site_prediction: dict) -> str:
    site = site_prediction.get("predicted_site") or "Unknown"
    conf = site_prediction.get("confidence") or ""
    alts = site_prediction.get("top_alternatives") or []
    ev = site_prediction.get("visual_evidence") or []

    alts_html = ""
    if alts:
        chips = "".join(f'<span class="chip">{_htmlesc(str(a))}</span>' for a in alts)
        alts_html = f'<div style="margin-top:8px;font-size:.75em;color:#64748b">Alternatives:</div><div class="alt-chips">{chips}</div>'

    return (
        f'<div class="stage-card">'
        f'<div class="stage-num">Stage 2</div>'
        f'<div class="stage-title">Site Prediction</div>'
        f'<div class="stage-value">{_htmlesc(site)} &nbsp;{_confidence_badge(conf)}</div>'
        f'{alts_html}'
        f'<div style="margin-top:10px;font-size:.75em;color:#64748b;font-weight:600">Visual evidence:</div>'
        f'{_ev_list(ev)}'
        f'</div>'
    )


def _stage_diagnosis(diagnosis_prediction: dict) -> str:
    diag = diagnosis_prediction.get("primary_diagnosis") or "Unknown"
    conf = diagnosis_prediction.get("confidence") or ""
    diff = diagnosis_prediction.get("differential") or []
    ev = diagnosis_prediction.get("visual_evidence") or []
    limits = diagnosis_prediction.get("limitations") or []

    diff_html = ""
    if diff:
        chips = "".join(f'<span class="chip">{_htmlesc(str(d))}</span>' for d in diff)
        diff_html = f'<div style="margin-top:8px;font-size:.75em;color:#64748b">Differential:</div><div class="alt-chips">{chips}</div>'

    limits_html = ""
    if limits:
        lis = "".join(f"<li>{_htmlesc(str(l))}</li>" for l in limits)
        limits_html = (
            f'<div style="margin-top:8px;font-size:.75em;color:#64748b;font-weight:600">Limitations:</div>'
            f'<ul class="ev-list">{lis}</ul>'
        )

    return (
        f'<div class="stage-card">'
        f'<div class="stage-num">Stage 3</div>'
        f'<div class="stage-title">Diagnosis Prediction</div>'
        f'<div class="stage-value">{_htmlesc(diag)} &nbsp;{_confidence_badge(conf)}</div>'
        f'{diff_html}'
        f'<div style="margin-top:10px;font-size:.75em;color:#64748b;font-weight:600">Visual evidence:</div>'
        f'{_ev_list(ev)}'
        f'{limits_html}'
        f'</div>'
    )


def _stage_sufficiency(sufficiency: dict) -> str:
    suf_diag = sufficiency.get("sufficient_for_diagnosis")
    suf_rep = sufficiency.get("sufficient_for_report")
    reason = sufficiency.get("reason") or ""
    missing = sufficiency.get("missing_evidence") or []
    next_step = sufficiency.get("recommended_next_step") or ""

    missing_html = ""
    if missing:
        lis = "".join(f"<li>{_htmlesc(str(m))}</li>" for m in missing)
        missing_html = (
            f'<div style="margin-top:8px;font-size:.75em;color:#b45309;font-weight:600">Missing evidence:</div>'
            f'<ul class="missing-list">{lis}</ul>'
        )

    next_html = ""
    if next_step:
        next_html = f'<div style="margin-top:8px;font-size:.75em;color:#64748b">Recommended: <b>{_htmlesc(next_step)}</b></div>'

    return (
        f'<div class="stage-card">'
        f'<div class="stage-num">Stage 4</div>'
        f'<div class="stage-title">Evidence Sufficiency</div>'
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">'
        f'<div>Diagnosis: {_bool_badge(suf_diag)}</div>'
        f'<div>Report: {_bool_badge(suf_rep)}</div>'
        f'</div>'
        f'<div class="reason-text">{_htmlesc(reason)}</div>'
        f'{missing_html}'
        f'{next_html}'
        f'</div>'
    )


def _stage_medgemma(short_report: str) -> str:
    return (
        f'<div class="stage-card" style="border-top-color:#9333ea">'
        f'<div class="stage-num">Stage 5a</div>'
        f'<div class="stage-title">MedGemma Visual Description</div>'
        f'<div style="font-size:.88em;line-height:1.7;color:#1e2a38">{_htmlesc(short_report)}</div>'
        f'</div>'
    )


def _segmentation_html(seg: dict) -> str:
    if not isinstance(seg, dict) or not seg.get("stats"):
        return ""
    stats = seg.get("stats") or {}
    named = stats.get("named_class_fractions") or {}
    rows = []
    for name, payload in named.items():
        if not isinstance(payload, dict):
            continue
        frac = float(payload.get("fraction") or 0.0)
        cid = int(payload.get("class_id") or 0)
        if cid == 0 or frac < 0.01:
            continue
        rows.append((frac, str(name)))
    rows.sort(reverse=True)
    if not rows:
        return ""
    chips = "".join(
        f'<span class="chip">{_htmlesc(name)} {frac * 100:.1f}%</span>'
        for frac, name in rows[:4]
    )
    fg = stats.get("foreground_fraction")
    fg_text = ""
    if fg is not None:
        try:
            fg_text = f'<div style="font-size:.72em;color:#64748b;margin-top:4px">Foreground: <b>{float(fg) * 100:.1f}%</b></div>'
        except Exception:
            pass
    cache_text = "cache hit" if seg.get("cache_hit") else "computed"
    return (
        f'<div style="margin:5px 0 4px;font-size:.72em;color:#64748b">'
        f'Segmentation ({_htmlesc(cache_text)}):</div>'
        f'<div class="alt-chips">{chips}</div>'
        f'{fg_text}'
    )


def render_report_html(payload: dict) -> str:
    """Render a fully visualized self-contained HTML pathology report."""
    case_id = payload.get("case_id", "unknown")
    mode = payload.get("mode", "")
    n_selected = payload.get("selected_count", 0)
    n_candidates = payload.get("candidate_count", 0)
    figures: list[dict] = payload.get("figures") or []
    selected_rois: list[dict] = payload.get("selected_rois") or []
    roi_selection: dict = payload.get("roi_selection") or {}
    site_prediction: dict = payload.get("site_prediction") or {}
    diagnosis_prediction: dict = payload.get("diagnosis_prediction") or {}
    sufficiency: dict = payload.get("evidence_sufficiency") or {}
    short_report: str = payload.get("medgemma_short_report") or ""
    rich_report: str = payload.get("generated_long_report_rich") or payload.get("generated_long_report") or ""

    # ── MedGemma reasoning box ────────────────────────────────────────────────
    reasoning_html = ""
    if short_report:
        reasoning_html = (
            f'<div class="medgemma-box">'
            f'<div class="lbl">MedGemma Expert Visual Findings</div>'
            f'{_htmlesc(short_report)}'
            f'</div>'
        )

    # ── Report sections ───────────────────────────────────────────────────────
    sections = _parse_report_sections(rich_report)
    section_parts: list[str] = []
    for name, content in sections:
        content_html = _inject_figure_links(content, figures)
        # Preserve newlines
        content_html = content_html.replace("\n", "<br>")
        section_parts.append(
            f'<div class="section">'
            f'<div class="section-name">{_htmlesc(name)}</div>'
            f'<div class="section-content">{content_html}</div>'
            f'</div>'
        )
    sections_html = "\n".join(section_parts)

    # ── Figure sidebar ────────────────────────────────────────────────────────
    # Build roi_scores lookup for the sidebar
    roi_score_by_idx: dict[int, dict] = {}
    for rs in (roi_selection.get("roi_scores") or []):
        idx = rs.get("roi_index")
        if idx is not None:
            roi_score_by_idx[int(idx)] = rs

    card_parts: list[str] = []
    for fig_idx, entry in enumerate(figures):
        n = entry["figure"].replace("Figure ", "")
        dot_color = _ROI_PALETTE_HEX[fig_idx % len(_ROI_PALETTE_HEX)]
        img_src = _img_data_uri(entry.get("image_path", ""))
        score = entry.get("score") or 0
        roi_idx = entry.get("roi_index")
        role = entry.get("evidence_role")
        top_site = str(entry.get("top_site") or "").strip()
        top_site_score = entry.get("top_site_score")
        site_margin = entry.get("site_margin_to_second")
        seg = entry.get("segmentation") if isinstance(entry.get("segmentation"), dict) else {}
        overlay_src = _img_data_uri(seg.get("overlay_path", "")) if seg else ""
        mask_src = _img_data_uri(seg.get("mask_path", ""), max_edge=512) if seg else ""
        rs = roi_score_by_idx.get(int(roi_idx)) if roi_idx else None

        if img_src and overlay_src:
            mask_tag = (
                f'<img src="{mask_src}" alt="{entry["figure"]} mask" '
                f'style="border-top:1px solid #e2e8f0">'
                if mask_src else ""
            )
            img_tag = (
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#e2e8f0;'
                f'border-top:3px solid {dot_color}">'
                f'<div><div style="font-size:.62em;color:#64748b;background:#f8fafc;padding:2px 5px">ROI</div>'
                f'<img src="{img_src}" alt="{entry["figure"]} ROI"></div>'
                f'<div><div style="font-size:.62em;color:#64748b;background:#f8fafc;padding:2px 5px">Seg overlay</div>'
                f'<img src="{overlay_src}" alt="{entry["figure"]} segmentation overlay"></div>'
                f'</div>{mask_tag}'
            )
        else:
            img_tag = (
                f'<img src="{img_src}" alt="{entry["figure"]}" style="border-top:3px solid {dot_color}">'
                if img_src else
                '<div style="height:160px;background:#f0f0f0;display:flex;align-items:center;'
                'justify-content:center;font-size:.75em;color:#aaa">image not found</div>'
            )

        scores_html = ""
        if rs:
            dv = float(rs.get("diagnostic_value") or 0)
            tl = float(rs.get("tumor_likelihood") or 0)
            tq = float(rs.get("tissue_quality") or 0)
            scores_html = (
                f'<div class="score-row"><span class="score-label">Diagnostic</span>{_score_bar(dv)}</div>'
                f'<div class="score-row"><span class="score-label">Tumor likeli.</span>{_score_bar(tl)}</div>'
                f'<div class="score-row"><span class="score-label">Tissue qual.</span>{_score_bar(tq)}</div>'
            )
            if rs.get("reason"):
                scores_html += f'<div style="font-size:.72em;color:#64748b;margin-top:5px;font-style:italic">{_htmlesc(rs["reason"])}</div>'

        site_html = ""
        if top_site:
            site_score_text = ""
            if top_site_score is not None:
                try:
                    site_score_text += f" score={float(top_site_score):.3f}"
                except Exception:
                    pass
            if site_margin is not None:
                try:
                    site_score_text += f" margin={float(site_margin):.3f}"
                except Exception:
                    pass
            site_html = (
                f'<div style="color:#475569;margin:3px 0">Top site: '
                f'<b>{_htmlesc(top_site)}</b>{_htmlesc(site_score_text)}</div>'
            )

        seg_html = _segmentation_html(seg)

        card_parts.append(
            f'<div class="fig-card" id="figure-{n}">'
            f'{img_tag}'
            f'<div class="cap">'
            f'<strong><span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
            f'background:{dot_color};margin-right:5px;vertical-align:middle"></span>'
            f'{entry["figure"]} &nbsp;<span style="font-weight:400;color:#94a3b8">ROI #{roi_idx}</span></strong>'
            f'<div style="margin:4px 0">{_role_badge(role)}</div>'
            f'<div style="color:#64748b;margin:3px 0">Foundation score: <b>{score:.3f}</b></div>'
            f'{site_html}'
            f'{seg_html}'
            f'{scores_html}'
            f'</div>'
            f'</div>'
        )
    figure_cards_html = (
        "\n".join(card_parts) or
        '<p style="font-size:.8em;color:#aaa">No figures available.</p>'
    )

    # ── Pipeline stages ───────────────────────────────────────────────────────
    stages_html = ""
    stages_html += _stage_roi_selection(roi_selection, selected_rois, figures)
    if site_prediction:
        stages_html += _stage_site(site_prediction)
    if diagnosis_prediction:
        stages_html += _stage_diagnosis(diagnosis_prediction)
    if sufficiency:
        stages_html += _stage_sufficiency(sufficiency)
    if short_report:
        stages_html += _stage_medgemma(short_report)

    # ── WSI panel (prefer annotated with ROI boxes, fall back to clean) ──────
    wsi_thumb_path = (
        payload.get("wsi_thumbnail_annotated_path")
        or payload.get("wsi_thumbnail_path")
        or ""
    )
    wsi_thumb_uri = _img_data_uri(wsi_thumb_path, max_edge=1024) if wsi_thumb_path else ""
    if wsi_thumb_uri:
        has_rois = bool(payload.get("wsi_thumbnail_annotated_path"))
        thumb_title = (
            "WSI Overview — colored boxes show selected ROI locations (click to zoom)"
            if has_rois else
            "WSI Overview (click to zoom)"
        )
        legend_chips = "".join(
            f'<span class="wsi-fig-chip" style="background:{_ROI_PALETTE_HEX[i % len(_ROI_PALETTE_HEX)]}">'
            f'{"S" if str((figures[i].get("evidence_role") or "")) in {"site", "site_support"} else "T"}{i + 1}</span>'
            for i in range(len(figures))
        ) if has_rois else ""
        legend_html = f'<div class="wsi-legend">{legend_chips}</div>' if legend_chips else ""
        wsi_panel_html = (
            f'<div class="wsi-panel panel">'
            f'<div class="wsi-hdr"><span class="dot"></span>WSI Overview</div>'
            f'<img class="wsi-overview" src="{wsi_thumb_uri}" title="{thumb_title}" '
            f'alt="WSI Overview" onclick="openLightbox(this.src)">'
            f'{legend_html}'
            f'</div>'
        )
        top_grid_class = "top-grid"
    else:
        wsi_panel_html = ""
        top_grid_class = "top-grid no-wsi"

    # ── Cancer type from case_id ──────────────────────────────────────────────
    site_label = site_prediction.get("predicted_site") or ""
    diag_label = diagnosis_prediction.get("primary_diagnosis") or ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PathXAI Report \u2013 {_htmlesc(case_id)}</title>
<style>{_CSS}</style>
</head>
<body>
<div id="lightbox" class="lightbox"><img src="" alt=""></div>
<div class="page">

<!-- Header -->
<div class="hdr">
  <div class="hdr-icon">\U0001f52c</div>
  <div style="flex:1;min-width:0">
    <h1>PathXAI Pathology Report</h1>
    <div class="hdr-case">{_htmlesc(case_id)}</div>
    {f'<div class="hdr-diag">{_htmlesc(diag_label)}</div>' if diag_label else ''}
    <div class="hdr-chips">
      <span class="hdr-chip">{n_selected} ROIs from {n_candidates} candidates</span>
      <span class="hdr-chip">{_htmlesc(mode)}</span>
    </div>
  </div>
</div>

<!-- WSI + Report + Figures -->
<div class="{top_grid_class}">
  {wsi_panel_html}
  <div class="panel">
    <div class="panel-title"><span class="dot"></span>Generated Report (Stage 5b)</div>
    <div class="clean-note">
      <span>\u2139\ufe0f</span>
      <span><b>generated_long_report.txt</b> — clean version without figure citations, used for metric computation. This view shows the rich version with visual evidence references.</span>
    </div>
    {reasoning_html}
    {sections_html}
  </div>
  <div>
    <div class="panel-title" style="margin-bottom:10px"><span class="dot"></span>Visual Evidence</div>
    {figure_cards_html}
  </div>
</div>

<!-- Pipeline Reasoning (collapsible) -->
<details class="pipeline-section" open>
  <summary class="pipeline-hdr">Pipeline Reasoning Chain</summary>
  <div class="stages-grid">
    {stages_html}
  </div>
</details>

</div><!-- /page -->
<script>{_JS}</script>
</body>
</html>"""

    return html
