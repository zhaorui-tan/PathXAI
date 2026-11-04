"""Prompt templates for the PathXAI v2 pipeline.

All prompts are separated from code so they can be versioned, audited,
and edited without touching pipeline logic.
"""

from __future__ import annotations

import json
from typing import Any


# ── ROI Selection ─────────────────────────────────────────────────────────────

def roi_selection_prompt(
    candidate_text: str,
    max_selected_rois: int,
) -> str:
    """Prompt for the initial ROI triage stage."""
    return f"""/no_think
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
{candidate_text}

Return ONLY JSON:
{{
  "selected_roi_indices": [1, 2],
  "reason": "why these ROIs are enough or why more would be needed",
  "roi_scores": [
    {{"roi_index": 1, "diagnostic_value": 0-5, "tumor_likelihood": 0-5, "tissue_quality": 0-5, "reason": "..."}}
  ]
}}"""


# ── Site Prediction ───────────────────────────────────────────────────────────

def site_prediction_prompt(
    candidate_text: str,
    foundation_hint: dict | None = None,
) -> str:
    """Prompt for specimen/site prediction from ROI images."""
    hint_block = ""
    if foundation_hint:
        hint_block = f"""Foundation site candidates (computed from WSI tile embeddings):
{json.dumps(foundation_hint, indent=2, ensure_ascii=False)}

If the foundation site candidates are high confidence with a clear margin, treat them as strong site evidence.
Do not overturn a high-confidence foundation site hint just because tumor morphology mimics another organ.
If the foundation site candidates are low confidence, use them only as ranked hypotheses."""

    return f"""/no_think
You are predicting specimen/site from selected pathology ROI images.

Use only morphology visible in the ROI images plus the foundation site candidates.
The foundation site candidates are computed from all tissue tiles in the WSI by comparing
image embeddings to all site prompt embeddings, then aggregating the highest-confidence
site types at slide level.
{hint_block}
If the organ/site is not visually identifiable, say "not specified from ROI evidence".
Do not use filenames, case IDs, TCGA codes, directory names, or coordinates.

Selected ROI evidence:
{candidate_text}

Return ONLY JSON:
{{
  "predicted_site": "...",
  "confidence": "high/medium/low",
  "top_alternatives": ["...", "..."],
  "visual_evidence": ["visible morphology supporting the site"],
  "limitations": ["why site may be uncertain"]
}}"""


# ── Diagnosis ─────────────────────────────────────────────────────────────────

def broad_hypotheses_prompt(
    selected_rois_text: str,
    site_prediction: dict,
    segmentation_context: dict | None = None,
    morphology_context: dict | None = None,
    knowledge_context: dict | None = None,
) -> str:
    """Prompt for broad lineage hypotheses before committing to a diagnosis."""
    morph_label, morph_text, morph_rules = _morphology_prompt_block(morphology_context)

    return f"""/no_think
Classify the histogenetic BROAD LINEAGE from the ROI morphology.

**CRITICAL: best_broad_type MUST be exactly ONE of these labels:**
  carcinoma | sarcoma | lymphoma | melanoma | glioma |
  neuroendocrine neoplasm | germ cell tumor | mesothelioma |
  carcinosarcoma | thymic epithelial neoplasm | neoplasm

**DO NOT** put a specific diagnosis here (e.g. "Adrenocortical Carcinoma" is NOT
a broad type — its broad type is "carcinoma"). If you cannot narrow below
"neoplasm", say "neoplasm".

Rules:
- Use only visible ROI morphology, segmentation evidence, morphology evidence text,
  known/predicted site context, and KEEP background candidate knowledge.
- KEEP is background knowledge, not patient-specific evidence.
- Morphology evidence text is a non-binding descriptive clue, not ground truth.
- Segmentation fractions are localized tissue-composition / ROI-quality evidence,
  not lineage labels. They may support tumor amount, necrosis, stromal or
  epithelial-labeled tissue presence, but must not override broad lineage from
  H&E morphology.
- Low or absent segmentation fractions are not negative exclusion evidence for a
  lineage or subtype.
- Do not use filenames, case IDs, TCGA codes, directory names, or coordinates.
- If known site is high confidence, site-specific primary lineages remain the
  default primary space unless multiple ROIs directly support another lineage.
{morph_rules}

Site:
{json.dumps(site_prediction, indent=2, ensure_ascii=False)}

Segmentation:
{json.dumps(_safe_seg_context(segmentation_context or {}), indent=2, ensure_ascii=False)}

{morph_label}:
{morph_text}

KEEP context:
{json.dumps(_safe_knowledge_context(knowledge_context or {}), indent=2, ensure_ascii=False)}

Selected ROI evidence:
{selected_rois_text}

Return ONLY JSON:
{{
  "hypotheses": [
    {{"broad_type": "one of the allowed labels above", "confidence": "high/medium/low", "supporting_features": ["..."], "main_mimics": ["..."], "missing_discriminators": ["..."]}}
  ],
  "best_broad_type": "one of: carcinoma | sarcoma | lymphoma | melanoma | glioma | neuroendocrine neoplasm | germ cell tumor | mesothelioma | carcinosarcoma | thymic epithelial neoplasm | neoplasm",
  "confidence": "high/medium/low",
  "reason": "..."
}}"""


def subtype_sufficiency_prompt(
    selected_rois_text: str,
    site_prediction: dict,
    broad_hypotheses: dict | None = None,
    diagnosis_prediction: dict | None = None,
    segmentation_context: dict | None = None,
    morphology_context: dict | None = None,
    knowledge_context: dict | None = None,
) -> str:
    """Prompt for deciding whether a site-specific subtype is supported."""
    site = site_prediction.get("predicted_site", "unknown")
    subtype_candidates = _subtype_candidates_from_knowledge(knowledge_context or {})
    morph_label, morph_text, morph_rules = _morphology_prompt_block(morphology_context)
    return f"""/no_think
Judge whether current ROI evidence is sufficient to distinguish site-specific subtype candidates.

Rules:
- Evaluate subtype candidates one by one.
- Use KEEP candidates as a candidate list only; do not treat them as patient-specific evidence.
- Treat morphology evidence suspected diagnoses as non-binding clues.
- D1 segmentation classes are tissue-pattern evidence, not lineage labels.
- Segmentation stats may support tumor richness, necrosis amount, tissue-class
  composition, and ROI quality only.
- Do NOT use segmentation stats to decide exact subtype, invasive vs in-situ
  status, histologic grade, or malignancy unless direct H&E morphology also
  supports that statement.
- Do NOT infer DCIS/in-situ disease, invasion, or subtype from epithelial/glandular
  fractions, central necrosis, low stromal fraction, or absent segmentation
  classes alone.
- Low or absent segmentation fractions are not negative exclusion evidence.
- If broad lineage/site is supported but subtype discriminators are missing,
  recommend a broad/NOS/not-further-classifiable diagnosis.
- Do NOT demote a visually supported likely subtype only because ancillary
  confirmation, molecular tests, grade, stage, margins, or complete scoring
  systems are unavailable. Those are limitations/prognostic gaps, not always
  reasons to erase the best-supported diagnostic entity.
- Use "possible_but_insufficient" when a candidate has some supporting visual
  features but unresolved differential diagnosis remains; use "supported" when
  current visual+morphology evidence supports it as the best working diagnosis,
  even if ancillary confirmation is still recommended.
- Distinguish image-actionable missing evidence from nonimage metadata.
- Do not use filenames, case IDs, TCGA codes, directory names, or coordinates.
- `recommended_primary_diagnosis` must contain only the diagnostic entity, not
  phrases like "not further classifiable from ROI evidence".
{morph_rules}

Site: {site}

Broad hypotheses:
{json.dumps(broad_hypotheses or {}, indent=2, ensure_ascii=False)}

Current diagnosis, if any:
{json.dumps(diagnosis_prediction or {}, indent=2, ensure_ascii=False)}

Site-specific subtype candidates:
{json.dumps(subtype_candidates[:15], indent=2, ensure_ascii=False)}

Segmentation:
{json.dumps(_safe_seg_context(segmentation_context or {}), indent=2, ensure_ascii=False)}

{morph_label}:
{morph_text}

Selected ROI evidence:
{selected_rois_text}

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


def broad_hypotheses_prompt_sr1_frozen_20260722(
    selected_rois_text: str,
    site_prediction: dict,
    segmentation_context: dict | None = None,
    morphology_context: dict | None = None,
    knowledge_context: dict | None = None,
) -> str:
    """Frozen SR1 broad-hypothesis prompt recovered from VSCode history."""
    return f"""/no_think
Classify the histogenetic BROAD LINEAGE from the ROI morphology.

**CRITICAL: best_broad_type MUST be exactly ONE of these labels:**
  carcinoma | sarcoma | lymphoma | melanoma | glioma |
  neuroendocrine neoplasm | germ cell tumor | mesothelioma |
  carcinosarcoma | thymic epithelial neoplasm | neoplasm

**DO NOT** put a specific diagnosis here (e.g. "Adrenocortical Carcinoma" is NOT
a broad type — its broad type is "carcinoma"). If you cannot narrow below
"neoplasm", say "neoplasm".

Rules:
- Use only visible ROI morphology, segmentation evidence, MedGemma morphology text,
  known/predicted site context, and KEEP background candidate knowledge.
- KEEP is background knowledge, not patient-specific evidence.
- MedGemma suspected diagnoses are non-binding descriptive clues.
- Do not use filenames, case IDs, TCGA codes, directory names, or coordinates.
- If known site is high confidence, site-specific primary lineages remain the
  default primary space unless multiple ROIs directly support another lineage.

Site:
{json.dumps(site_prediction, indent=2, ensure_ascii=False)}

Segmentation:
{json.dumps(_safe_seg_context(segmentation_context or {}), indent=2, ensure_ascii=False)}

MedGemma morphology:
{(morphology_context or {}).get('description', '')[:4000] if morphology_context else ''}

KEEP context:
{json.dumps(_safe_knowledge_context(knowledge_context or {}), indent=2, ensure_ascii=False)}

Selected ROI evidence:
{selected_rois_text}

Return ONLY JSON:
{{
  "hypotheses": [
    {{"broad_type": "one of the allowed labels above", "confidence": "high/medium/low", "supporting_features": ["..."], "main_mimics": ["..."], "missing_discriminators": ["..."]}}
  ],
  "best_broad_type": "one of: carcinoma | sarcoma | lymphoma | melanoma | glioma | neuroendocrine neoplasm | germ cell tumor | mesothelioma | carcinosarcoma | thymic epithelial neoplasm | neoplasm",
  "confidence": "high/medium/low",
  "reason": "..."
}}"""


def subtype_sufficiency_prompt_sr1_frozen_20260722(
    selected_rois_text: str,
    site_prediction: dict,
    broad_hypotheses: dict | None = None,
    diagnosis_prediction: dict | None = None,
    segmentation_context: dict | None = None,
    morphology_context: dict | None = None,
    knowledge_context: dict | None = None,
) -> str:
    """Frozen SR1 subtype-sufficiency prompt recovered from VSCode history."""
    site = site_prediction.get("predicted_site", "unknown")
    subtype_candidates = _subtype_candidates_from_knowledge(knowledge_context or {})
    return f"""/no_think
Judge whether current ROI evidence is sufficient to distinguish site-specific subtype candidates.

Rules:
- Evaluate subtype candidates one by one.
- Use KEEP candidates as a candidate list only; do not treat them as patient-specific evidence.
- Treat MedGemma morphology suspected diagnoses as non-binding clues.
- D1 segmentation classes are tissue-pattern evidence, not lineage labels.
- If broad lineage/site is supported but subtype discriminators are missing,
  recommend a broad/NOS/not-further-classifiable diagnosis.
- Do NOT demote a visually supported likely subtype only because ancillary
  confirmation, molecular tests, grade, stage, margins, or complete scoring
  systems are unavailable. Those are limitations/prognostic gaps, not always
  reasons to erase the best-supported diagnostic entity.
- Use "possible_but_insufficient" when a candidate has some supporting visual
  features but unresolved differential diagnosis remains; use "supported" when
  current visual+morphology evidence supports it as the best working diagnosis,
  even if ancillary confirmation is still recommended.
- Distinguish image-actionable missing evidence from nonimage metadata.
- Do not use filenames, case IDs, TCGA codes, directory names, or coordinates.
- `recommended_primary_diagnosis` must contain only the diagnostic entity, not
  phrases like "not further classifiable from ROI evidence".

Site: {site}

Broad hypotheses:
{json.dumps(broad_hypotheses or {}, indent=2, ensure_ascii=False)}

Current diagnosis, if any:
{json.dumps(diagnosis_prediction or {}, indent=2, ensure_ascii=False)}

Site-specific subtype candidates:
{json.dumps(subtype_candidates[:15], indent=2, ensure_ascii=False)}

Segmentation:
{json.dumps(_safe_seg_context(segmentation_context or {}), indent=2, ensure_ascii=False)}

MedGemma morphology:
{(morphology_context or {}).get('description', '')[:4000] if morphology_context else ''}

Selected ROI evidence:
{selected_rois_text}

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


def broad_only_prompt(
    candidate_text: str,
    site_prediction: dict,
) -> str:
    """Prompt for Turn 1 broad-type-only diagnosis — pure VLM from ROI images."""
    site = site_prediction.get("predicted_site", "unknown")
    
    return f"""/no_think
You are a pathologist making a broad lineage determination from H&E WSI ROI images.

Specimen site: {site}

You have ROI images showing representative tumor tissue.

Determine the BROAD histogenetic lineage from the ROI morphology:
carcinoma / sarcoma / lymphoma / melanoma / glioma / germ cell tumor / 
neuroendocrine neoplasm / mesothelioma / other neoplasm

Do NOT attempt to specify the exact subtype yet — only the broad category.

Critical rules:
- Base your decision on the ROI images alone.
- The site provides clinical context but the morphology in the images has the final say.
- These are H&E ROI images unless explicitly stated otherwise.

Candidate ROIs:
{candidate_text}

Return ONLY JSON:
{{
  "broad_type": "one of: carcinoma, sarcoma, lymphoma, melanoma, glioma, germ cell tumor, neuroendocrine neoplasm, mesothelioma, other",
  "confidence": "high/medium/low",
  "rationale": "brief morphologic basis for broad classification (1-2 sentences)"
}}"""


def unified_diagnosis_prompt(
    candidate_text: str,
    site_prediction: dict,
    segmentation_context: dict | None = None,
    morphology_context: dict | None = None,
    knowledge_context: dict | None = None,
    allow_chat_override: bool = False,
) -> str:
    """Single-call two-step diagnosis: broad from images, subtype from context.
    
    Unlike the old two-call design, this keeps everything in ONE chat session
    so the VLM can cross-reference visual and text evidence in a single pass.
    """
    site = site_prediction.get("predicted_site", "unknown")
    
    morph_block = ""
    if morphology_context and morphology_context.get("available"):
        morph_label, morph_text, morph_rules = _morphology_prompt_block(morphology_context)
        morph_block = f"""
{morph_label}:
{morph_text}
{morph_rules}"""
    
    knowledge_block = ""
    candidate_block = ""
    if knowledge_context and knowledge_context.get("available"):
        hits = knowledge_context.get("disease_hits", [])[:3]
        if hits:
            knowledge_block = "\nRelevant pathology knowledge:\n" + "\n".join(
                f"- {h.get('name', '')}: {h.get('definition', '')[:300]}"
                for h in hits
            )
        cands = knowledge_context.get("site_subtype_candidates", [])[:8]
        if cands:
            candidate_block = (
                f"\nEntities reported at the {site} site (KEEP knowledge, for orientation only):\n"
                + "\n".join(
                    f"- {c.get('name', '')}"
                    + (f" (syn: {', '.join(str(s) for s in c.get('synonyms', [])[:3])})"
                       if c.get('synonyms') else "")
                    for c in cands
                )
            )

    chat_override_block = ""
    if allow_chat_override:
        chat_override_block = (
            "Chat-history rule (applies when prior case-level reasoning is in the conversation):\n"
            "- You may have expressed a tentative or limited diagnosis earlier. That is normal.\n"
            "- You are NOT required to stay consistent with your prior reasoning.\n"
            "- If additional evidence or re-examination of the ROI images supports a MORE SPECIFIC diagnosis\n"
            "  than what you proposed earlier, COMMIT to the most specific defensible diagnosis NOW.\n"
            "- Conversely, if you were over-confident earlier and the evidence is weaker than you thought,\n"
            "  you may downgrade the confidence or broaden the diagnosis.\n"
            "- The final output must reflect your BEST assessment at THIS moment, not a compromise\n"
            "  with what you said before.\n"
        )

    return f"""/no_think
You are a pathologist making a diagnosis from H&E WSI ROI images.

Specimen site: {site}

You have ROI images showing representative tumor tissue.

REASON IN TWO STEPS:

STEP 1 — BROAD LINEAGE (from ROI images ONLY):
Look at the ROI images. Determine the broad histogenetic lineage:
carcinoma / sarcoma / lymphoma / melanoma / glioma / germ cell tumor / 
neuroendocrine neoplasm / mesothelioma / other neoplasm

Base this decision on the ROI MORPHOLOGY alone. Do not let any text context
below influence this step. The images have the final say.

STEP 2 — SUBTYPE REFINEMENT (from text context + broad from Step 1):
With the broad lineage locked from Step 1, use the context below to refine
the specific subtype, malignancy, and grade. The context is supplemental —
only commit to a subtype when the morphology supports it.

{morph_block}
{knowledge_block}
{candidate_block}

Critical evidence-boundary rules:
- These are H&E ROI images unless explicitly stated otherwise.
- Do NOT report actual IHC/molecular positivity or negativity from H&E.
- Morphology evidence and KEEP are supporting context, not ground truth.
- Segmentation context, when provided, is support for localized tissue
  composition / tumor burden / necrosis / ROI quality only. It cannot determine
  the primary diagnosis, subtype, invasive-vs-in-situ status, grade, or
  malignancy by itself.
- Do not change an invasive-vs-in-situ call based only on segmentation stats.
  If the invasion interface or in-situ architecture is not directly visible in
  H&E morphology, keep that field uncertain and do not let it replace the
  primary diagnosis.
- Low or absent segmentation fractions are not negative exclusion evidence
  against a cancer type or subtype.
- If ROI evidence does not support a specific subtype, leave subtype empty.

Candidate ROIs:
{candidate_text}

{chat_override_block}
Return ONLY JSON:
{{
  "broad_type": "from Step 1: carcinoma/sarcoma/lymphoma/melanoma/glioma/germ cell tumor/neuroendocrine neoplasm/mesothelioma/other",
  "subtype": "from Step 2: specific entity, or '' if not classifiable",
  "malignancy": "benign/borderline/malignant",
  "grade": "histologic grade if assessable, else ''",
  "primary_diagnosis": "the most specific defensible diagnosis",
  "confidence": "high/medium/low",
  "differential": ["alternative 1", "alternative 2"],
  "tumor_architecture": "growth pattern description",
  "cytology": "nuclear and cellular features",
  "necrosis": "present/absent with description",
  "invasion": "invasive features if visible",
  "stroma_inflammation": "stromal and inflammatory features",
  "ihc_observed": "actual IHC/molecular if provided; else 'not assessable from H&E images'",
  "ihc_molecular": "same as ihc_observed; do not place recommended markers here",
  "missing_evidence": ["what image evidence is missing"],
  "recommended_ancillary_studies": ["recommended IHC/molecular/clinical studies"]
}}"""


def unified_diagnosis_prompt_sr1_frozen_20260722(
    candidate_text: str,
    site_prediction: dict,
    segmentation_context: dict | None = None,
    morphology_context: dict | None = None,
    knowledge_context: dict | None = None,
    allow_chat_override: bool = False,
) -> str:
    """Frozen SR1 diagnosis prompt recovered from VSCode history.

    This preserves the July 2026 SR1 template for reproducibility.  The
    signature matches ``unified_diagnosis_prompt`` so pipeline code can switch
    by prompt profile without touching callers.
    """
    site = site_prediction.get("predicted_site", "unknown")

    morph_block = ""
    if morphology_context and morphology_context.get("available"):
        morph_block = f"""
MedGemma cellular morphology description:
{morphology_context.get('description', '')[:4000]}"""

    knowledge_block = ""
    candidate_block = ""
    if knowledge_context and knowledge_context.get("available"):
        hits = knowledge_context.get("disease_hits", [])[:3]
        if hits:
            knowledge_block = "\nRelevant pathology knowledge:\n" + "\n".join(
                f"- {h.get('name', '')}: {h.get('definition', '')[:300]}"
                for h in hits
            )
        cands = knowledge_context.get("site_subtype_candidates", [])[:8]
        if cands:
            candidate_block = (
                f"\nEntities reported at the {site} site (KEEP knowledge, for orientation only):\n"
                + "\n".join(
                    f"- {c.get('name', '')}"
                    + (f" (syn: {', '.join(str(s) for s in c.get('synonyms', [])[:3])})"
                       if c.get('synonyms') else "")
                    for c in cands
                )
            )

    return f"""/no_think
You are a pathologist making a diagnosis from H&E WSI ROI images.

Specimen site: {site}

You have ROI images showing representative tumor tissue.

REASON IN TWO STEPS:

STEP 1 — BROAD LINEAGE (from ROI images ONLY):
Look at the ROI images. Determine the broad histogenetic lineage:
carcinoma / sarcoma / lymphoma / melanoma / glioma / germ cell tumor / 
neuroendocrine neoplasm / mesothelioma / other neoplasm

Base this decision on the ROI MORPHOLOGY alone. Do not let any text context
below influence this step. The images have the final say.

STEP 2 — SUBTYPE REFINEMENT (from text context + broad from Step 1):
With the broad lineage locked from Step 1, use the context below to refine
the specific subtype, malignancy, and grade. The context is supplemental —
only commit to a subtype when the morphology supports it.

{morph_block}
{knowledge_block}
{candidate_block}

Critical evidence-boundary rules:
- These are H&E ROI images unless explicitly stated otherwise.
- Do NOT report actual IHC/molecular positivity or negativity from H&E.
- MedGemma and KEEP are supporting context, not ground truth.
- If ROI evidence does not support a specific subtype, leave subtype empty.

Candidate ROIs:
{candidate_text}

Return ONLY JSON:
{{
  "broad_type": "from Step 1: carcinoma/sarcoma/lymphoma/melanoma/glioma/germ cell tumor/neuroendocrine neoplasm/mesothelioma/other",
  "subtype": "from Step 2: specific entity, or '' if not classifiable",
  "malignancy": "benign/borderline/malignant",
  "grade": "histologic grade if assessable, else ''",
  "primary_diagnosis": "the most specific defensible diagnosis",
  "confidence": "high/medium/low",
  "differential": ["alternative 1", "alternative 2"],
  "tumor_architecture": "growth pattern description",
  "cytology": "nuclear and cellular features",
  "necrosis": "present/absent with description",
  "invasion": "invasive features if visible",
  "stroma_inflammation": "stromal and inflammatory features",
  "ihc_observed": "actual IHC/molecular if provided; else 'not assessable from H&E images'",
  "ihc_molecular": "same as ihc_observed; do not place recommended markers here",
  "missing_evidence": ["what image evidence is missing"],
  "recommended_ancillary_studies": ["recommended IHC/molecular/clinical studies"]
}}"""


def subtype_prompt(
    candidate_text: str,
    site_prediction: dict,
    locked_broad_type: str,
    segmentation_context: dict | None = None,
    morphology_context: dict | None = None,
    knowledge_context: dict | None = None,
) -> str:
    """Prompt for subtype refinement — broad is already locked, now use morph + KEEP for specificity.
    
    NOTE: segmentation_context is accepted but NOT injected into the prompt.
    Seg data has been shown to provide zero diagnostic signal (VLM cannot 
    distinguish cancer types from class fractions alone). Seg is only used 
    for SegPolicy (acquisition) and claim verification.
    """
    site = site_prediction.get("predicted_site", "unknown")
    
    morph_block = ""
    if morphology_context and morphology_context.get("available"):
        morph_label, morph_text, morph_rules = _morphology_prompt_block(morphology_context)
        morph_block = f"""
{morph_label}:
{morph_text}
{morph_rules}"""
    
    knowledge_block = ""
    candidate_block = ""
    if knowledge_context and knowledge_context.get("available"):
        hits = knowledge_context.get("disease_hits", [])[:3]
        if hits:
            knowledge_block = "\nRelevant pathology knowledge:\n" + "\n".join(
                f"- {h.get('name', '')}: {h.get('definition', '')[:300]}"
                for h in hits
            )
        cands = knowledge_context.get("site_subtype_candidates", [])[:8]
        if cands:
            candidate_block = (
                f"\nEntities reported at the {site} site (KEEP knowledge, for orientation only):\n"
                + "\n".join(
                    f"- {c.get('name', '')}"
                    + (f" (syn: {', '.join(str(s) for s in c.get('synonyms', [])[:3])})"
                       if c.get('synonyms') else "")
                    for c in cands
                )
            )
    
    return f"""/no_think
You are a pathologist refining a diagnosis. The broad lineage is already determined: {locked_broad_type}

Specimen site: {site}

You have ROI images showing representative tumor tissue.

Now determine the SPECIFIC subtype within {locked_broad_type}, plus malignancy and grade.
Use the morphology evidence, KEEP knowledge, and site context below to help narrow down the specific entity.

{morph_block}
{knowledge_block}
{candidate_block}

Critical evidence-boundary rules:
- These are H&E ROI images unless explicitly stated otherwise.
- Do NOT report actual IHC/molecular positivity or negativity from H&E morphology.
- Put actual IHC/molecular results in `ihc_observed` ONLY when an IHC/molecular
  result is explicitly provided in the input metadata. Otherwise write
  "not assessable from H&E images".
- Morphology evidence and KEEP entities are supporting context, not ground truth.
- Commit to a subtype only when the morphology supports it, otherwise leave it empty.

Candidate ROIs:
{candidate_text}

Return ONLY JSON:
{{
  "subtype": "specific entity within {locked_broad_type}, or '' if not classifiable from ROI evidence",
  "malignancy": "benign/borderline/malignant",
  "grade": "histologic grade if assessable, else ''",
  "primary_diagnosis": "the most specific defensible diagnosis (subtype if committed, else broad lineage)",
  "confidence": "high/medium/low",
  "differential": ["alternative diagnosis 1", "alternative diagnosis 2"],
  "tumor_architecture": "growth pattern description",
  "cytology": "nuclear and cellular features",
  "necrosis": "present/absent with description",
  "invasion": "invasive features if visible",
  "stroma_inflammation": "stromal and inflammatory features",
  "ihc_observed": "actual IHC/molecular results if explicitly provided; otherwise 'not assessable from H&E images'",
  "ihc_molecular": "same as ihc_observed; do not place recommended markers here",
  "missing_evidence": ["what image evidence is missing"],
  "recommended_ancillary_studies": ["recommended IHC/molecular/clinical studies to confirm or classify"]
}}"""


def evidence_sufficiency_prompt(
    selected_rois_text: str,
    site_prediction: dict,
    diagnosis_prediction: dict,
    segmentation_context: dict | None = None,
    knowledge_context: dict | None = None,
    morphology_context: dict | None = None,
) -> str:
    """Prompt for assessing whether evidence is sufficient for a confident diagnosis."""
    site = site_prediction.get("predicted_site", "unknown")
    diagnosis = diagnosis_prediction.get("primary_diagnosis", "unknown")
    
    return f"""/no_think
Assess whether the current ROI evidence is sufficient for a confident pathology diagnosis.

Current diagnosis: {diagnosis} (site: {site})

Evidence ROIs:
{selected_rois_text}

Evaluate:
1. Is there enough tumor tissue to characterize the lesion?
2. Are key diagnostic features (architecture, cytology, invasion, necrosis) visible?
3. What image-observable evidence is missing?
4. What non-image information (IHC, molecular, clinical) is needed?

Return ONLY JSON:
{{
  "sufficient_for_diagnosis": "Yes/Partially/No",
  "diagnostic_confidence": "high/medium/low",
  "missing_image_evidence": ["..."],
  "missing_nonimage_metadata": ["..."],
  "recommended_additional_rois": ["type of ROI that would help"],
  "can_confidently_report": true/false
}}"""


# ── Report Generation ─────────────────────────────────────────────────────────

def long_report_prompt(
    selected_rois_text: str,
    site_prediction: dict,
    diagnosis_prediction: dict,
    sufficiency: dict,
    segmentation_context: dict | None = None,
    knowledge_context: dict | None = None,
    medgemma_context: dict | None = None,
    auxiliary_evidence_context: dict | None = None,
    enforce_staged_diagnosis: bool = False,
    use_natural_language_stats: bool = False,
    seg_stats_style: str = "compact_json",
    lock_diagnosis_fields: bool = False,
    lock_report_grade_field: bool = False,
) -> str:
    """Prompt for generating the final structured pathology report."""
    site = site_prediction.get("predicted_site", "not specified")
    diagnosis = diagnosis_prediction.get("primary_diagnosis", "not specified")
    differential = diagnosis_prediction.get("differential", [])

    broad_type = diagnosis_prediction.get("broad_type", "")
    subtype = diagnosis_prediction.get("subtype", "")
    malignancy = diagnosis_prediction.get("malignancy", "")
    grade = diagnosis_prediction.get("grade", "")
    tier_lines = []
    if broad_type:
        tier_lines.append(f"- Broad lineage: {broad_type}")
    if subtype:
        tier_lines.append(f"- Subtype: {subtype}")
    if malignancy:
        tier_lines.append(f"- Malignancy: {malignancy}" + (f" (grade: {grade})" if grade else ""))
    tier_block = ""
    if tier_lines:
        tier_block = (
            "\nDiagnosis tiers (state these explicitly in the Diagnosis section):\n"
            + "\n".join(tier_lines)
        )

    diag_instruction = ""
    if enforce_staged_diagnosis:
        diag_instruction = f"\nIMPORTANT: The Diagnosis section MUST state: {diagnosis}"
    lock_block = ""
    if lock_diagnosis_fields:
        locked_payload = {
            "primary_diagnosis": diagnosis,
            "broad_type": broad_type,
            "subtype": subtype,
            "malignancy": malignancy,
            "grade": grade,
            "invasion": diagnosis_prediction.get("invasion", ""),
            "necrosis": diagnosis_prediction.get("necrosis", ""),
        }
        lock_block = (
            "\n\nLOCKED FINAL DIAGNOSIS STATE — do not revise these fields. "
            "The report writer must verbalize this state, not perform a new diagnosis:\n"
            + json.dumps(locked_payload, ensure_ascii=False, indent=2)
        )
    elif lock_report_grade_field and grade:
        lock_block = (
            "\n\nLOCKED REPORT GRADE FIELD:\n"
            f"- The staged diagnosis payload already states grade = {grade}.\n"
            "- The Stage/Grade/Prognostic Features section must include exactly "
            f"this grade as an explicit line: Grade: {grade}.\n"
            "- Stage, margins, molecular data, and other unprovided prognostic "
            "fields may remain not assessable. Do not write that the grade "
            "itself is not assessable."
        )

    diff_block = ""
    if differential:
        diff_block = "\nDifferential diagnoses to consider: " + ", ".join(str(d) for d in differential[:5])

    morph_block = ""
    if medgemma_context and medgemma_context.get("available"):
        morph_label, morph_text, morph_rules = _morphology_prompt_block(medgemma_context, max_chars=3000)
        morph_block = f"\n\n{morph_label} (use as reference, not ground truth):\n{morph_text}\n{morph_rules}"

    seg_block = ""
    if segmentation_context and segmentation_context.get("available"):
        style = str(seg_stats_style or "").strip().lower()
        if use_natural_language_stats and style in ("", "compact_json"):
            style = "natural_language"
        if style in {"natural_language", "natural", "nl"}:
            seg_nl = _seg_natural_language(segmentation_context)
            seg_block = f"\n\n{seg_nl}"
        elif style in {"claim_style", "claims", "ss2"}:
            seg_block = f"\n\n{_seg_claim_style(segmentation_context)}"
        elif style in {"hierarchical", "hierarchy", "ss3"}:
            seg_block = f"\n\n{_seg_hierarchical_stats(segmentation_context)}"
        elif style in {"quality_aware", "quality", "ss4"}:
            seg_block = f"\n\n{_seg_quality_aware_stats(segmentation_context)}"
        elif style in {"positive_only", "positive", "ss5"}:
            seg_block = f"\n\n{_seg_positive_only_stats(segmentation_context)}"
        else:
            safe_seg = _safe_seg_context(segmentation_context, max_chars=2500)
            seg_block = (
                "\n\nCompact segmentation evidence (localized ROI tissue support; do not use this alone as a cancer-type label):\n"
                + json.dumps(safe_seg, ensure_ascii=False)[:2500]
            )

    aux_block = ""
    if auxiliary_evidence_context:
        safe_aux = dict(auxiliary_evidence_context)
        aux_block = (
            "\n\nAuxiliary evidence role policy:\n"
            + json.dumps(safe_aux, ensure_ascii=False, indent=2)[:1800]
            + "\nUse auxiliary grade-focused ROIs only for grade, necrosis, cytologic atypia, "
            "mitotic activity, and ROI-level quality support. Do not use these auxiliary ROIs "
            "to change the primary diagnosis, broad lineage, histologic subtype, specimen site, "
            "or invasive-vs-in-situ status unless the main diagnostic ROI set and staged "
            "diagnosis already support that change."
        )

    return f"""/no_think
Generate a structured pathology report from the provided ROI evidence.

Specimen site: {site}
Primary diagnosis: {diagnosis}{diag_instruction}{tier_block}{diff_block}
{lock_block}

Evidence gaps identified:
- Missing image evidence: {json.dumps(sufficiency.get('missing_image_evidence', []))}
- Missing non-image metadata: {json.dumps(sufficiency.get('missing_nonimage_metadata', []))}
{morph_block}{seg_block}{aux_block}

Rules:
1. Only report what is visible in the ROI images or stated above.
2. Do NOT fabricate IHC results, stage, margin status, or gross measurements.
3. If IHC/molecular data were not explicitly provided, write actual IHC results as
   "Not assessable from H&E images"; recommended stains may be listed separately.
4. If the evidence sufficiency says the case cannot be confidently reported,
   do not automatically erase the best-supported primary diagnosis. Use the
   staged primary diagnosis above when it is supported by morphology,
   segmentation, subtype-sufficiency, or claim verification. Put uncertainty
   about subtype, malignancy, IHC, stage, margins, or grade in the appropriate
   sections rather than appending it to the Diagnosis section.
	5. The Diagnosis section must contain ONLY the primary diagnosis name. Put
	   malignancy, grade, subtype uncertainty, and limitations in the Stage/Grade or
	   Final Summary sections. Do not merge differential diagnoses into Diagnosis.
	6. Be specific about what IS visible — use precise histomorphological terminology.
	7. Use segmentation evidence only as localized support for tissue composition,
	   tumor-rich regions, necrosis, epithelial-labeled/stromal tissue classes, or
	   ROI quality. Do not infer IHC, molecular status, margins, stage, clinical
	   outcomes, exact subtype, grade, malignancy, or invasive-vs-in-situ status
	   from segmentation.
    8. Do not amplify segmentation/reader evidence. If necrosis is described as
       focal, trace, limited, uncertain, or low-fraction, do NOT call it extensive,
       geographic, widespread, or high-grade necrosis.
    9. The Diagnosis section must stay aligned with the staged Primary diagnosis
       above. Do not replace it with a different entity because segmentation
       stats mention epithelial-labeled tissue, necrosis, tumor burden, or
       low/absent tissue classes.
    10. If a LOCKED FINAL DIAGNOSIS STATE is provided, copy its diagnosis,
        subtype, malignancy, grade, invasion, and necrosis conclusions into the
        appropriate report sections. Segmentation or ROI text may support or
        limit those fields but may not overwrite them.
    11. If auxiliary grade-focused ROIs are listed, treat them as secondary
        grading evidence only. They are not subtype-selection evidence.
    12. If a LOCKED REPORT GRADE FIELD is provided, the Stage/Grade section must
        verbalize that exact grade. Do not contradict it with "grade not
        assessable"; reserve not-assessable language for stage, margins,
        molecular/IHC, or other unavailable prognostic fields.

ROI evidence:
{selected_rois_text}

Return the report in this EXACT format (8 sections, each on a new line after the heading):

Specimen/Site:
Diagnosis:
Tumor/Lesion Description:
Key Microscopic Findings:
Invasion/Necrosis/Margins:
Immunohistochemistry/Ancillary Studies:
Stage/Grade/Prognostic Features:
Final Summary:"""


# ── MedGemma Short Report ─────────────────────────────────────────────────────

def medgemma_short_description_prompt(
    site_prediction: dict,
    diagnosis_prediction: dict | None = None,
) -> str:
    """System prompt prefix for MedGemma visual descriptions."""
    site = site_prediction.get("predicted_site", "unknown")
    diagnosis = ""
    if diagnosis_prediction:
        diagnosis = f" Suspected diagnosis: {diagnosis_prediction.get('primary_diagnosis', 'unknown')}."
    
    return (
        f"Describe the key histopathological findings in this H&E image from {site}.{diagnosis} "
        "Describe: tissue architecture, cellular features, nuclear atypia, mitotic activity, "
        "necrosis, invasion pattern, stromal reaction, and inflammatory infiltrate. "
        "Be specific and use precise histopathological terminology. "
        "Do NOT mention that you are an AI or that this is a description task."
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _morphology_prompt_block(
    morphology_context: dict | None,
    *,
    max_chars: int = 4000,
) -> tuple[str, str, str]:
    """Return a safe prompt label/text/rules for morphology evidence.

    Full PathXAI still uses MedGemma morphology, so the default label remains
    MedGemma.  Seg-supported report variants can instead provide a fine-tuned
    seg-reader source; those prompts need stricter wording because the reader is
    a compact translator of paired ROI+seg inputs, not an independent diagnosis.
    """
    ctx = morphology_context or {}
    source = str(ctx.get("source") or "medgemma").lower()
    text = str(ctx.get("description", ""))[:max_chars]
    if "seg_reader" in source:
        guard_text = ""
        if "guard" in source:
            guard_text = (
                " This context has already been filtered by a reader evidence "
                "guard: absent classes and weak negative labels were removed "
                "from diagnostic consideration. Do not infer a subtype from "
                "what the reader did not see, and do not let reader text "
                "override the ROI images or locked broad lineage."
            )
        return (
            "Fine-tuned seg-reader morphology evidence",
            text,
            (
                "Seg-reader rules: use this only as localized morphology/tissue "
                "support translated from paired ROI+seg inputs. Do not treat it "
                "as ground truth or as a cancer-type label. In the final report, "
                "convert this into natural pathology language; do not write "
                "\"seg-reader\", \"reader suggests\", or raw fraction-dump prose. "
                "Keep necrosis, tumor richness, glandular/epithelial evidence, "
                "and ROI quality proportional to the stated fractions."
                + guard_text
            ),
        )
    return (
        "MedGemma morphology",
        text,
        "MedGemma rules: use this only as non-binding descriptive morphology support.",
    )


def candidate_text(rois: list[dict]) -> str:
    """Format ROI candidates for inclusion in prompts.

    Deliberately anonymises file/case/slide identifiers. TCGA case IDs and slide
    UUIDs may leak dataset priors into a VLM, so prompts only expose the
    one-based image order, coarse crop geometry, score, and non-identifying role
    metadata. The real ``roi_id`` stays in checkpoints/HTML for traceability.
    """
    lines = []
    for idx, roi in enumerate(rois, start=1):
        meta = roi.get("metadata") or {}
        fnd = meta.get("foundation_site_prediction") or {}
        score = roi.get("score", 0.0)
        w, h = roi.get("width", "?"), roi.get("height", "?")
        source = str(roi.get("source") or "")
        role = str(meta.get("evidence_role") or roi.get("evidence_role") or roi.get("role") or "")
        source_hint = ""
        if "zoom" in source.lower() or "zoom" in str(roi.get("roi_id", "")).lower():
            source_hint = ", crop_type=zoomed_detail"
        elif role:
            source_hint = f", role={role[:40]}"
        site_hint = f", foundation_site={fnd.get('top_site', '?')}" if fnd.get("available") else ""
        lines.append(
            f"  [{idx}] ROI {idx}: size={w}x{h}, score={score:.3f}{source_hint}{site_hint}"
        )
    return "\n".join(lines) if lines else "(no ROIs)"


def _safe_seg_context(seg: dict, max_chars: int = 3000) -> dict:
    """Truncate segmentation context to fit token budget, using actual field names."""
    out = {"available": seg.get("available", False)}
    class_names = seg.get("class_names") or {}  # e.g. {"1": "Neoplastic", ...}
    roi_results = seg.get("roi_results", [])
    if roi_results:
        summary = []
        for r in roi_results[:8]:
            stats = r.get("stats") or {}
            named = stats.get("named_class_fractions") or {}
            # Build readable fraction dict: {ClassName: pct%}
            fractions = {
                name: f"{info.get('fraction', 0) * 100:.1f}%"
                for name, info in named.items()
                if name != "Background" and isinstance(info, dict) and info.get("fraction", 0) > 0.005
            }
            summary.append({
                "roi_index": r.get("roi_index"),
                "foreground_fraction": f"{stats.get('foreground_fraction', 0) * 100:.1f}%",
                "tissue_class_fractions": fractions,
            })
        out["roi_summary"] = summary
        out["total_rois"] = len(roi_results)
        out["class_legend"] = {v: k for k, v in class_names.items() if v != "Background"}
    return out


def _seg_natural_language(seg: dict) -> str:
    """Convert compact seg stats into a natural-language paragraph for the VLM.
    
    Instead of raw JSON like {"Neoplastic": "95.0%"}, produces:
    "ROI 1: 95% Neoplastic, 3% Glandular/Epithelial, 0% Necrosis."
    """
    safe = _safe_seg_context(seg)
    if not safe.get("available") or not safe.get("roi_summary"):
        return "No segmentation evidence available."
    
    lines = []
    for r in safe["roi_summary"]:
        fracs = r.get("tissue_class_fractions", {})
        if not fracs:
            continue
        parts = []
        for cls_name, pct in sorted(fracs.items(), key=lambda x: float(x[1].rstrip('%')), reverse=True):
            parts.append(f"{pct} {cls_name}")
        roi_idx = r.get("roi_index", "?")
        lines.append(f"ROI {roi_idx}: {', '.join(parts)}.")
    
    if not lines:
        return "No segmentation evidence available."
    
    return (
        "Segmentation tissue composition (per ROI):\n"
        + "\n".join(lines)
        + "\n\nUse these fractions as localized evidence of tissue types and tumor cellularity. "
        "They describe WHAT tissue is present, not the diagnosis."
    )


def _percent_to_float(value: Any) -> float:
    text = str(value or "0").strip()
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except Exception:
        return 0.0


def _seg_summary_rows(seg: dict) -> list[dict[str, Any]]:
    safe = _safe_seg_context(seg)
    rows: list[dict[str, Any]] = []
    for row in safe.get("roi_summary") or []:
        fracs = row.get("tissue_class_fractions") or {}
        rows.append({
            "roi_index": row.get("roi_index", "?"),
            "foreground_pct": _percent_to_float(row.get("foreground_fraction")),
            "fractions": {str(k): _percent_to_float(v) for k, v in fracs.items()},
        })
    return rows


def _seg_case_aggregates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def vals_for_any(label_parts: list[str]) -> list[float]:
        label_parts = [p.lower() for p in label_parts]
        out = []
        for row in rows:
            vals = [
                float(v)
                for k, v in (row.get("fractions") or {}).items()
                if any(part in k.lower() for part in label_parts)
            ]
            out.append(max(vals) if vals else 0.0)
        return out

    tumor = vals_for_any(["neoplastic"])
    necrosis = vals_for_any(["necrotic", "dead"])
    gland = vals_for_any(["glandular", "epithelial"])
    stroma = vals_for_any(["connective", "stroma"])
    foreground = [float(r.get("foreground_pct") or 0.0) for r in rows]
    return {
        "roi_count": len(rows),
        "usable_roi_count": sum(1 for v in foreground if v >= 30.0),
        "tumor_rich_roi_count": sum(1 for v in tumor if v >= 50.0),
        "max_tumor_pct": max(tumor) if tumor else 0.0,
        "mean_tumor_pct": sum(tumor) / len(tumor) if tumor else 0.0,
        "max_necrosis_pct": max(necrosis) if necrosis else 0.0,
        "glandular_or_epithelial_roi_count": sum(1 for v in gland if v >= 5.0),
        "stroma_rich_roi_count": sum(1 for v in stroma if v >= 20.0),
        "low_foreground_roi_count": sum(1 for v in foreground if v < 30.0),
    }


def _seg_claim_style(seg: dict) -> str:
    """SS2: translate mask fractions into bounded support claims."""
    rows = _seg_summary_rows(seg)
    if not rows:
        return "Segmentation-derived support claims: no usable segmentation statistics available."
    agg = _seg_case_aggregates(rows)
    claims = [
        f"- Tumor-rich evidence: {agg['tumor_rich_roi_count']}/{agg['roi_count']} ROI(s) have >=50% neoplastic tissue; max neoplastic fraction {agg['max_tumor_pct']:.1f}%.",
        f"- Necrosis evidence: max necrotic/dead fraction {agg['max_necrosis_pct']:.1f}% across ROIs.",
        f"- Epithelial-labeled tissue class: detected above 5% in {agg['glandular_or_epithelial_roi_count']}/{agg['roi_count']} ROI(s); this is not an in-situ/invasive discriminator.",
        f"- Stroma/connective component: stroma-rich pattern in {agg['stroma_rich_roi_count']}/{agg['roi_count']} ROI(s).",
        "- Invasion/in-situ status: not determined by segmentation masks.",
        "- Exact subtype and grade: not determined by segmentation masks.",
    ]
    roi_lines = []
    for row in rows[:8]:
        top = sorted((row.get("fractions") or {}).items(), key=lambda kv: kv[1], reverse=True)[:4]
        if top:
            roi_lines.append(
                f"  ROI {row['roi_index']}: "
                + ", ".join(f"{name} {pct:.1f}%" for name, pct in top)
                + f"; foreground {row['foreground_pct']:.1f}%."
            )
    return (
        "Segmentation-derived support claims (mask fractions; bounded evidence, not diagnosis):\n"
        + "\n".join(claims)
        + "\nROI-level support:\n"
        + "\n".join(roi_lines)
        + "\nUse these claims only to support visible tissue composition, tumor amount, necrosis, and ROI quality. Do not use absent classes as exclusion evidence."
    )


def _seg_hierarchical_stats(seg: dict) -> str:
    """SS3: case-level summary followed by ROI-level detail."""
    rows = _seg_summary_rows(seg)
    if not rows:
        return "Hierarchical segmentation summary: no usable segmentation statistics available."
    agg = _seg_case_aggregates(rows)
    case_lines = [
        f"- Usable segmented ROIs: {agg['usable_roi_count']}/{agg['roi_count']}.",
        f"- Neoplastic burden: mean {agg['mean_tumor_pct']:.1f}%, maximum {agg['max_tumor_pct']:.1f}%, tumor-rich ROIs {agg['tumor_rich_roi_count']}/{agg['roi_count']}.",
        f"- Necrotic/dead component: maximum {agg['max_necrosis_pct']:.1f}%.",
        f"- Tissue-class composition: epithelial-labeled class in {agg['glandular_or_epithelial_roi_count']} ROI(s); stroma/connective-rich in {agg['stroma_rich_roi_count']} ROI(s). These are not subtype or in-situ/invasive labels.",
    ]
    roi_lines = []
    for row in rows[:8]:
        fracs = row.get("fractions") or {}
        top = sorted(fracs.items(), key=lambda kv: kv[1], reverse=True)[:5]
        roi_lines.append(
            f"  ROI {row['roi_index']}: foreground {row['foreground_pct']:.1f}%; "
            + (", ".join(f"{name} {pct:.1f}%" for name, pct in top) if top else "no non-background class above threshold")
            + "."
        )
    return (
        "Hierarchical segmentation evidence:\n"
        "Case-level summary:\n"
        + "\n".join(case_lines)
        + "\nROI-level detail:\n"
        + "\n".join(roi_lines)
        + "\nInterpretation rule: use case-level stats for overall support and ROI-level stats for localized evidence; do not infer subtype, grade, malignancy, or invasive-vs-in-situ status from tissue-class fractions or absent classes alone."
    )


def _seg_quality_aware_stats(seg: dict) -> str:
    """SS4: hierarchical stats plus reliability/quality gating."""
    rows = _seg_summary_rows(seg)
    if not rows:
        return "Quality-aware segmentation evidence: no usable segmentation statistics available."
    agg = _seg_case_aggregates(rows)
    quality_lines = []
    for row in rows[:8]:
        fg = float(row.get("foreground_pct") or 0.0)
        fracs = row.get("fractions") or {}
        tumor = max([v for k, v in fracs.items() if "neoplastic" in k.lower()] or [0.0])
        necrosis = max([v for k, v in fracs.items() if "necrotic" in k.lower() or "dead" in k.lower()] or [0.0])
        if fg < 30.0:
            quality = "low reliability for diagnosis because foreground tissue is limited"
        elif tumor >= 50.0:
            quality = "usable high-neoplastic-fraction ROI"
        elif tumor >= 15.0:
            quality = "usable but not tumor-rich ROI"
        else:
            quality = "limited tumor evidence; use cautiously"
        if necrosis >= 15.0:
            quality += "; supports necrotic/dead component"
        quality_lines.append(f"  ROI {row['roi_index']}: {quality} (foreground {fg:.1f}%, neoplastic {tumor:.1f}%, necrotic/dead {necrosis:.1f}%).")
    return (
        "Quality-aware segmentation evidence:\n"
        f"- Reliable/usable ROIs: {agg['usable_roi_count']}/{agg['roi_count']}; low-foreground ROIs: {agg['low_foreground_roi_count']}.\n"
        f"- Tumor-rich ROIs: {agg['tumor_rich_roi_count']}/{agg['roi_count']}; max neoplastic fraction {agg['max_tumor_pct']:.1f}%.\n"
        f"- Necrosis support: max necrotic/dead fraction {agg['max_necrosis_pct']:.1f}%.\n"
        + "\n".join(quality_lines)
        + "\nUse only usable ROIs for confident morphology statements; downweight low-foreground or low-tumor ROIs. These stats do not establish exact subtype, grade, malignancy, or invasive-vs-in-situ status."
    )


def _seg_positive_only_stats(seg: dict) -> str:
    """SS5: minimal positive evidence only, avoiding absent-class drift."""
    rows = _seg_summary_rows(seg)
    if not rows:
        return "Positive-only segmentation evidence: no usable segmentation statistics available."
    lines = []
    for row in rows[:8]:
        fracs = row.get("fractions") or {}
        tumor = max([v for k, v in fracs.items() if "neoplastic" in k.lower()] or [0.0])
        necrosis = max([v for k, v in fracs.items() if "necrotic" in k.lower() or "dead" in k.lower()] or [0.0])
        parts = [f"foreground {row['foreground_pct']:.1f}%"]
        if tumor >= 5.0:
            parts.append(f"neoplastic {tumor:.1f}%")
        if necrosis >= 10.0:
            parts.append(f"necrotic/dead {necrosis:.1f}%")
        lines.append(f"  ROI {row['roi_index']}: " + ", ".join(parts) + ".")
    return (
        "Positive-only segmentation evidence (minimal mask support):\n"
        + "\n".join(lines)
        + "\nRules: use only retained positive fractions as support for tumor amount, substantial necrosis, and ROI usability. Omitted classes and low fractions are not negative evidence. Do not infer exact subtype, grade, malignancy, or invasive-vs-in-situ status from this block."
    )


def _safe_knowledge_context(knowledge: dict, max_items: int = 8) -> dict:
    """Keep the KEEP context small but diagnostic enough for prompting."""
    if not knowledge:
        return {"available": False}
    return {
        "available": knowledge.get("available", False),
        "phase": knowledge.get("phase", ""),
        "query": knowledge.get("query", ""),
        "site_subtype_candidates": [
            {
                "name": r.get("name") or r.get("entity") or "",
                "synonyms": (r.get("synonyms") or [])[:4],
                "count": r.get("count"),
                "score": r.get("score"),
            }
            for r in (knowledge.get("site_subtype_candidates") or [])[:max_items]
            if isinstance(r, dict)
        ],
        "disease_hits": [
            {
                "name": r.get("name") or r.get("entity") or "",
                "definition": str(r.get("definition") or r.get("description") or "")[:240],
                "score": r.get("score"),
            }
            for r in (knowledge.get("disease_hits") or [])[:max_items]
            if isinstance(r, dict)
        ],
        "hypothesis_hits": [
            {
                "hypothesis": r.get("hypothesis") or r.get("diagnosis") or r.get("name") or "",
                "hits": (r.get("hits") or r.get("disease_hits") or [])[:3],
            }
            for r in (knowledge.get("hypothesis_hits") or knowledge.get("per_hypothesis") or [])[:max_items]
            if isinstance(r, dict)
        ],
    }


def _subtype_candidates_from_knowledge(knowledge: dict) -> list[dict]:
    """Extract candidate subtype rows from KEEP context for subtype judging."""
    out: list[dict] = []
    seen: set[str] = set()

    def add(name: str, source: str, **extra: Any) -> None:
        name = str(name or "").strip()
        key = name.lower()
        if not key or key in seen:
            return
        seen.add(key)
        row = {"name": name, "source": source}
        row.update({k: v for k, v in extra.items() if v not in (None, "", [])})
        out.append(row)

    for row in knowledge.get("site_subtype_candidates") or []:
        if isinstance(row, dict):
            add(
                row.get("name") or row.get("entity") or "",
                "keep_site_subtype",
                synonyms=(row.get("synonyms") or [])[:4],
                count=row.get("count"),
                score=row.get("score"),
            )
    for row in knowledge.get("disease_hits") or []:
        if isinstance(row, dict):
            add(row.get("name") or row.get("entity") or "", "keep_disease", score=row.get("score"))
    for group in knowledge.get("hypothesis_hits") or knowledge.get("per_hypothesis") or []:
        if not isinstance(group, dict):
            continue
        add(group.get("hypothesis") or group.get("diagnosis") or group.get("name") or "", "keep_hypothesis")
    return out
