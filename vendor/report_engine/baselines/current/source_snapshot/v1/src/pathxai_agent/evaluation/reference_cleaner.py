"""Image-observable-only reference report cleaning.

Problem
-------
TCGA (and most surgical pathology) reports mix information from two very
different sources:
  * Gross examination  — size, weight, colour, orientation, ink description
  * Microscopic slides — architecture, cytology, invasion, grade

A model that only sees H&E WSI patches can produce detailed microscopic
findings but *cannot* reproduce gross dimensions or exact IHC titres.
Comparing generated reports directly against full clinical reports therefore
systematically underscores models on "Tumor/Lesion Description" (GT = gross)
and "Key Microscopic Findings" (GT = minimal admin note) while over-penalising
hallucination when differentials are listed in the Final Summary.

This module provides two cleaning strategies:

1. **Rule-based** (`rule_based_image_only`) — deterministic, zero-cost,
   runs during every eval call.  Strips obvious non-image content
   (measurements, gross terms, TNM, exact IHC results) and marks sections
   where the GT is entirely gross-only as "not comparable from images".

2. **LLM-based** (`llm_image_only`) — higher quality, run once offline as
   a preprocessing step.  Uses Qwen to rewrite the GT into the same
   structured format as the generated report, retaining only microscopic
   content.  Results are cached to ``report_image_only_ref.txt``.

Usage
-----
``staged_metrics.py`` calls ``rule_based_image_only()`` automatically for
every eval, so no preprocessing is required.

To run LLM-based cleaning in batch, use:
    scripts/generate_image_only_references.py
"""
from __future__ import annotations

import re
from typing import Dict, Optional

from pathxai_agent.evaluation.report_text_metrics import (
    EXPECTED_LONG_REPORT_SECTIONS,
    extract_sections,
)

# ── Pattern banks ─────────────────────────────────────────────────────────────

# Sentences containing ANY of these are dropped from Tumor/Lesion Description
# and Specimen/Site because they describe gross/procedural content.
_GROSS_TERMS = [
    "grossly", "gross examination", "cut surface", "tan", "pink-gray", "gray-tan",
    "pink-tan", "cream", "yellow", "brown", "white", "nodular", "cystic",
    "received", "submitted as", "formalin", "container", "previously inked",
    "serially sectioned", "specimen is", "orientation", "excisional",
    "orientation is", "remainder of specimen", "representative section",
    "additional section", "tissue procurement",
]

# Measurement patterns (e.g. "6.5 x 6.5 cm", "105 grams", "0.3 mm")
_MEASUREMENT_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:x\s*\d+(?:\.\d+)?\s*)?(?:cm|mm|g\b|grams?|kg)\b",
    flags=re.IGNORECASE,
)

# Administrative microscopy notes, not real findings
_ADMIN_MICROSCOPY_RE = re.compile(
    r"\b\d+\s+slides?\s+reviewed\b|slides?\s+reviewed",
    flags=re.IGNORECASE,
)

# TNM / stage notation
_TNM_RE = re.compile(
    r"\b(?:stage\s+[ivxlcIVXLC]+|stage\s+\d|pt\d[a-z]?|pn\d[a-z]?|pm\d[a-z]?|tnm)\b",
    flags=re.IGNORECASE,
)

# Specific IHC result (keep "not specified" / "not performed", drop positive/negative results)
_IHC_RESULT_RE = re.compile(
    r"\b(?:ki-?67|sf-?1|inhibin|melan-?a|ae1[/\-]ae3|synaptophysin|chromogranin"
    r"|ttf-?1|p40|p63|ck7|ck20|s-?100|cd\d+|her-?2)\b"
    r"(?:\s*:\s*|\s+is\s+|\s+was\s+|\s+)(?:positive|negative|\d+%)",
    flags=re.IGNORECASE,
)

# Exact margin-distance sentences (keep "positive" / "negative" qualitatives)
_MARGIN_DISTANCE_RE = re.compile(
    r"margin[s]?\s+(?:are\s+)?(?:close|positive|negative|clear)[,\s]*\d+(?:\.\d+)?\s*mm",
    flags=re.IGNORECASE,
)

# Sentences to always drop from Specimen/Site (procedural details)
_PROCEDURE_TERMS = [
    "open adrenalectomy", "radical nephrectomy", "modified radical",
    "dissection", "lumpectomy", "mastectomy", "cystoprostatectomy",
    "pneumonectomy", "lobectomy", "colectomy", "hemicolectomy",
    "procedure:", "submitted for",
]

# Patterns that mark differential-diagnosis sentences in Final Summary
_DIFFERENTIAL_RE = re.compile(
    r"\b(such as|including|vs\.?|rule out|cannot exclude|cannot rule out|"
    r"differential\s*(?:diagnosis)?|differentials|overlap with|"
    r"may represent|could represent|other\s+(?:entity|entities)|"
    r"correlation\s+(?:with|is)|clinical\s+correlation)\b",
    flags=re.IGNORECASE,
)


# ── Rule-based cleaner ────────────────────────────────────────────────────────

def rule_based_image_only(report_text: str) -> Dict[str, str]:
    """Return a dict of section_name → cleaned text for image-observable content.

    Also returns two extra keys:
        ``__full_text__``   — reassembled full cleaned report
        ``__comparable__``  — comma-separated list of sections with real content

    Sections that have zero image-comparable content after cleaning are set to
    ``"Not assessable from H&E images."``  This prevents them from penalising
    the model for providing legitimate microscopic information that the GT
    happens not to contain.
    """
    sections = extract_sections(report_text)
    cleaned: Dict[str, str] = {}

    for section in EXPECTED_LONG_REPORT_SECTIONS:
        raw = sections.get(section, "")
        if section == "Specimen/Site":
            cleaned[section] = _clean_site(raw)
        elif section == "Tumor/Lesion Description":
            cleaned[section] = _clean_tumor_description(raw)
        elif section == "Key Microscopic Findings":
            cleaned[section] = _clean_microscopic(raw)
        elif section == "Invasion/Necrosis/Margins":
            cleaned[section] = _clean_invasion(raw)
        elif section == "Immunohistochemistry/Ancillary Studies":
            cleaned[section] = _clean_ihc(raw)
        elif section == "Stage/Grade/Prognostic Features":
            cleaned[section] = _clean_stage_grade(raw)
        elif section == "Final Summary":
            cleaned[section] = _clean_final_summary(raw)
        else:
            cleaned[section] = raw  # Diagnosis — keep as-is

    comparable = [s for s in EXPECTED_LONG_REPORT_SECTIONS
                  if cleaned[s] and cleaned[s] != "Not assessable from H&E images."]
    lines = []
    for s in EXPECTED_LONG_REPORT_SECTIONS:
        lines.append(f"{s}:\n{cleaned[s]}")
    cleaned["__full_text__"] = "\n\n".join(lines)
    cleaned["__comparable__"] = ", ".join(comparable)
    return cleaned


# ── LLM-based cleaner ─────────────────────────────────────────────────────────

IMAGE_ONLY_PROMPT = """\
Convert the pathology report below into an *image-observable-only* format.

Rules:
1. Keep ONLY information a pathologist can observe directly from H&E histology slides.
2. REMOVE: exact specimen measurements (cm / mm / grams), gross macroscopic descriptions \
(colour, weight, texture, orientation, cut-surface appearance), procedural details \
(how the specimen was received / inked / submitted), exact IHC titres or marker \
results (Ki-67 %, SF-1+, etc.), TNM or clinical stage (Stage I/II/III, pT1/pN0 etc.), \
exact mitotic counts or lymph node counts, surgical margin distances in mm, \
clinical history (symptoms, prior procedures).
3. KEEP: organ / site name, cancer type / diagnosis, microscopic growth pattern \
(tubular, papillary, solid…), cytological features (pleomorphism, nuclear grade, \
mitotic activity level), qualitative invasion or necrosis (present/absent), \
histologic grade level (high grade / low grade / poorly differentiated).
4. For sections where no image-observable information exists: write exactly \
"Not assessable from H&E images."
5. Preserve exactly this 8-section structure with the same headings:

Specimen/Site:
Diagnosis:
Tumor/Lesion Description:
Key Microscopic Findings:
Invasion/Necrosis/Margins:
Immunohistochemistry/Ancillary Studies:
Stage/Grade/Prognostic Features:
Final Summary:

Reference report:
---
{report_text}
---

Return ONLY the reformatted report, no explanations."""


def llm_image_only(
    report_text: str,
    llm_client,  # QwenVisionClient
    site_hint: str = "",
) -> str:
    """Use an LLM to rewrite ``report_text`` into image-observable-only format.

    Returns the cleaned report as a plain string.  Call this offline
    (see ``scripts/generate_image_only_references.py``) and cache the result.
    """
    prompt = IMAGE_ONLY_PROMPT.format(report_text=report_text.strip())
    result = llm_client.chat(prompt, image_paths=[])
    return result.get("content", "").strip()


# ── Section-level helpers ─────────────────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|;\s*", text) if s.strip()]


def _drop_if(sentence: str, patterns: list[str]) -> bool:
    low = sentence.lower()
    return any(p in low for p in patterns)


_NON_ORGAN_WORDS = {"procedure", "site", "specimen", "tissue", "material", "submitted", "received"}

def _clean_site(text: str) -> str:
    """Keep organ/tissue; strip procedure and laterality details."""
    if not text:
        return ""
    kept = []
    for sent in _sentences(text):
        low = sent.lower()
        # Drop detailed procedure info, keep short site descriptions
        if _drop_if(sent, _PROCEDURE_TERMS):
            # Try to salvage the organ part before procedure details
            organ_match = re.match(r"^([^;,]+?)(?:\s*[;,]|\s+procedure\b|\s+(?:left|right)\s+open)", sent, re.IGNORECASE)
            if organ_match:
                organ = organ_match.group(1).strip().rstrip(":")
                # Reject if the salvaged part is itself a non-organ/admin word
                if organ and organ.lower() not in _NON_ORGAN_WORDS:
                    kept.append(organ)
                    kept.append(organ)
        elif _MEASUREMENT_RE.search(sent):
            continue
        else:
            # Strip trailing "procedure:" artifacts
            sent_clean = re.sub(r"\s+procedure\s*:?\s*$", "", sent, flags=re.IGNORECASE).strip()
            kept.append(sent_clean or sent)
    return " ".join(kept) if kept else text


def _clean_tumor_description(text: str) -> str:
    """Remove gross description; keep microscopic features.

    If nothing remains, returns the "not assessable" sentinel so the section
    is skipped during metric calculation rather than penalising the model.
    """
    if not text:
        return "Not assessable from H&E images."
    kept = []
    for sent in _sentences(text):
        low = sent.lower()
        if _drop_if(sent, _GROSS_TERMS):
            continue
        if _MEASUREMENT_RE.search(sent):
            continue
        if re.search(r"\b(ink|inked|margin|orientation)\b", low):
            continue
        kept.append(sent)
    if not kept:
        return "Not assessable from H&E images."
    return " ".join(kept)


def _clean_microscopic(text: str) -> str:
    """Remove administrative notes; keep actual microscopic findings."""
    if not text:
        return ""
    kept = []
    for sent in _sentences(text):
        if _ADMIN_MICROSCOPY_RE.search(sent):
            continue
        # Drop pure number-only mitotic count lines ("Mitotic figures: 8.")
        if re.match(r"^mitotic\s+(?:figures?|count)\s*:\s*\d+\.?\s*$", sent, re.IGNORECASE):
            continue
        kept.append(sent)
    return " ".join(kept) if kept else ""


def _clean_invasion(text: str) -> str:
    """Keep qualitative invasion/necrosis; remove mm margin distances, lymph node counts."""
    if not text:
        return ""
    kept = []
    for sent in _sentences(text):
        low = sent.lower()
        # Remove lymph node count sentences ("0/15 nodes")
        if re.search(r"\b\d+/\d+\s*(?:lymph\s*)?nodes?\b", low):
            continue
        # Remove pure TNM lines
        if _TNM_RE.search(sent) and not re.search(r"\b(invasion|necrosis|margin)\b", low):
            continue
        # Remove exact margin distances (keep qualitative positive/negative)
        sent_clean = _MARGIN_DISTANCE_RE.sub(
            lambda m: re.sub(r"\d+(?:\.\d+)?\s*mm", "", m.group(0)).rstrip(", "),
            sent,
        )
        # Strip measurement numbers from the sentence
        sent_clean = _MEASUREMENT_RE.sub("", sent_clean).strip()
        # Clean up empty parentheses or brackets left after number removal
        sent_clean = re.sub(r"\(\s*\)", "", sent_clean).strip()
        sent_clean = re.sub(r"\s{2,}", " ", sent_clean).strip()
        if sent_clean:
            kept.append(sent_clean)
    return " ".join(kept) if kept else ""


def _clean_ihc(text: str) -> str:
    """Drop specific IHC marker results; keep 'not performed' / 'not specified'."""
    if not text:
        return ""
    # If the section is empty or just "not specified/not performed", normalise it
    norm = text.lower().strip()
    if not norm or re.match(r"^(not\s+(?:specified|performed|provided|available|done|applicable)\.?)$", norm):
        return "Not provided."
    kept = []
    for sent in _sentences(text):
        if _IHC_RESULT_RE.search(sent):
            continue
        kept.append(sent)
    return " ".join(kept) if kept else "Not provided."


def _clean_stage_grade(text: str) -> str:
    """Keep histologic grade level; remove TNM staging and exact counts."""
    if not text:
        return ""
    kept = []
    for sent in _sentences(text):
        low = sent.lower()
        # Remove pure TNM/stage lines
        if _TNM_RE.search(sent) and not re.search(
            r"\b(grade|differentiat|mitotic\s+activit|pleomorphi)\b", low
        ):
            continue
        # Remove exact mitotic counts (keep "high mitotic activity")
        sent_clean = re.sub(r"\bmitotic\s+(?:figures?|count)\s*:\s*\d+\b", "mitotic activity present", sent, flags=re.IGNORECASE)
        kept.append(sent_clean)
    return " ".join(kept) if kept else ""


def _clean_final_summary(text: str) -> str:
    """Keep the core diagnosis summary; remove procedure/gross details."""
    if not text:
        return ""
    kept = []
    for sent in _sentences(text):
        if _drop_if(sent, _GROSS_TERMS):
            continue
        if _MEASUREMENT_RE.search(sent):
            continue
        kept.append(sent)
    return " ".join(kept) if kept else text


# ── Differential-sentence extractor (used by overclaim metric) ────────────────

def extract_differential_terms_from_summary(report_text: str) -> set[str]:
    """Return the set of specific clinical terms that appear in differential-diagnosis
    sentences in the Final Summary section.  These should be excluded from the
    overclaim calculation because mentioning differentials is clinically appropriate.
    """
    from pathxai_agent.evaluation.pathology_metrics import (
        DIAGNOSIS_SYNONYMS,
        MORPH_VOCAB,
        normalize_text,
        _contains,
    )
    sections = extract_sections(report_text)
    summary = sections.get("Final Summary", "")
    if not summary:
        return set()

    diff_terms: set[str] = set()
    for sent in _sentences(summary):
        if _DIFFERENTIAL_RE.search(sent):
            sent_norm = normalize_text(sent)
            all_specific = [t for terms in MORPH_VOCAB.values() for t in terms]
            all_specific += [a for _, aliases in DIAGNOSIS_SYNONYMS for a in aliases]
            for term in all_specific:
                if _contains(sent_norm, term):
                    diff_terms.add(term)
    return diff_terms
