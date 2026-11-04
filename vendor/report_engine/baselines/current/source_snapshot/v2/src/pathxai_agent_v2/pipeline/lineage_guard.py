"""Site lineage guard — prevents wrong-site diagnoses when known_site is used.

When known_site is provided (explicit opt-in), the guard ensures that a non-site-lineage
diagnosis (e.g., diagnosing "lung carcinoma" when site is "adrenal") is kept as a
differential, not as primary, unless strong evidence supports it.
"""

from __future__ import annotations

from typing import Any

from pathxai_agent_v2.pipeline.base import DiagnosisPrediction, EvidenceSufficiency, SitePrediction

# Site → expected lineage / broad type mapping
SITE_LINEAGE: dict[str, dict[str, Any]] = {
    "adrenal": {
        "site_terms": ["adrenal", "adrenocortical", "suprarenal"],
        "broad_type": "adrenal neoplasm",
        "allowed_lineages": [
            "adrenal cortical neoplasm",
            "adrenocortical neoplasm",
            "adrenal neoplasm",
            "pheochromocytoma",
            "paraganglioma",
            "neuroendocrine neoplasm",
        ],
        "forbidden_lineages": ["lung", "breast", "colorectal", "gastric", "prostate"],
    },
    "bladder": {
        "site_terms": ["bladder", "urothelial", "urinary"],
        "broad_type": "urothelial neoplasm",
        "allowed_lineages": ["carcinoma", "urothelial carcinoma"],
    },
    "breast": {
        "site_terms": ["breast", "mammary"],
        "broad_type": "breast epithelial neoplasm",
        "allowed_lineages": ["carcinoma", "ductal carcinoma", "lobular carcinoma"],
    },
    "kidney": {
        "site_terms": ["kidney", "renal"],
        "broad_type": "renal cell neoplasm",
        "allowed_lineages": ["carcinoma", "renal cell carcinoma"],
    },
    "liver": {
        "site_terms": ["liver", "hepatic", "hepatocellular"],
        "broad_type": "hepatic neoplasm",
        "allowed_lineages": ["carcinoma", "hepatocellular carcinoma", "cholangiocarcinoma"],
    },
    "lung": {
        "site_terms": ["lung", "pulmonary", "pleural"],
        "broad_type": "lung neoplasm",
        "allowed_lineages": ["carcinoma", "adenocarcinoma", "squamous cell carcinoma", "mesothelioma"],
    },
    "colon": {
        "site_terms": ["colon", "colorectal", "rectal"],
        "broad_type": "colorectal neoplasm",
        "allowed_lineages": ["carcinoma", "adenocarcinoma"],
    },
    "stomach": {
        "site_terms": ["stomach", "gastric"],
        "broad_type": "gastric neoplasm",
        "allowed_lineages": ["carcinoma", "adenocarcinoma"],
    },
    "brain": {
        "site_terms": ["brain", "glial", "cerebral", "cns"],
        "broad_type": "glial neoplasm",
        "allowed_lineages": ["glioma", "glioblastoma", "astrocytoma", "oligodendroglioma"],
    },
    "thyroid": {
        "site_terms": ["thyroid"],
        "broad_type": "thyroid neoplasm",
        "allowed_lineages": ["carcinoma", "papillary carcinoma", "follicular carcinoma"],
    },
    "prostate": {
        "site_terms": ["prostate", "prostatic"],
        "broad_type": "prostate neoplasm",
        "allowed_lineages": ["carcinoma", "adenocarcinoma"],
    },
    "cervix": {
        "site_terms": ["cervix", "cervical", "endocervical"],
        "broad_type": "cervical neoplasm",
        "allowed_lineages": ["carcinoma", "squamous cell carcinoma", "adenocarcinoma"],
    },
    "uterus": {
        "site_terms": ["uterus", "uterine", "endometrium", "endometrial"],
        "broad_type": "endometrial neoplasm",
        "allowed_lineages": ["carcinoma", "adenocarcinoma", "sarcoma"],
    },
    "ovary": {
        "site_terms": ["ovary", "ovarian"],
        "broad_type": "ovarian neoplasm",
        "allowed_lineages": ["carcinoma", "serous carcinoma", "adenocarcinoma"],
    },
    "skin": {
        "site_terms": ["skin", "cutaneous", "melanocytic"],
        "broad_type": "skin/melanocytic neoplasm",
        "allowed_lineages": ["melanoma", "carcinoma", "squamous cell carcinoma"],
    },
}


def find_site_lineage(site: str) -> dict | None:
    """Find the lineage config for a site string (fuzzy match)."""
    site_lower = site.lower()
    for key, config in SITE_LINEAGE.items():
        if key in site_lower:
            return config
        for term in config.get("site_terms", []):
            if term in site_lower:
                return config
    return None


def apply_site_lineage_guard(
    diagnosis: DiagnosisPrediction,
    site_prediction: SitePrediction,
) -> DiagnosisPrediction:
    """If known_site is used, guard against wrong-site primary diagnosis.

    A non-site-lineage diagnosis is kept as differential, not primary,
    unless confirmed by strong independent evidence.
    """
    # Only apply when known_site is explicitly provided (not predicted)
    if not site_prediction.source.startswith("known_site"):
        return diagnosis

    lineage = find_site_lineage(site_prediction.predicted_site)
    if not lineage:
        return diagnosis

    primary = diagnosis.primary_diagnosis.lower()
    subtype = diagnosis.subtype.lower()
    broad = diagnosis.broad_type.lower()
    forbidden = lineage.get("forbidden_lineages", [])

    # Check if primary diagnosis belongs to wrong site lineage
    has_forbidden = any(f in primary for f in forbidden)
    needs_guard = has_forbidden
    if not needs_guard:
        return diagnosis

    # Guard: demote primary to differential, use broad_type as primary
    guarded_primary = lineage["broad_type"]
    diff = list(diagnosis.differential)
    if diagnosis.primary_diagnosis not in diff:
        diff.insert(0, diagnosis.primary_diagnosis)

    return DiagnosisPrediction(
        primary_diagnosis=guarded_primary,
        confidence="medium" if diagnosis.confidence == "high" else diagnosis.confidence,
        broad_type=guarded_primary,
        subtype="",
        malignancy=diagnosis.malignancy,
        grade=diagnosis.grade,
        differential=diff[:6],
        tumor_architecture=diagnosis.tumor_architecture,
        cytology=diagnosis.cytology,
        necrosis=diagnosis.necrosis,
        invasion=diagnosis.invasion,
        stroma_inflammation=diagnosis.stroma_inflammation,
        ihc_molecular=diagnosis.ihc_molecular,
        missing_evidence=_dedupe([
            *diagnosis.missing_evidence,
            "current evidence does not support assigning an out-of-site-lineage diagnosis as primary",
        ]),
        recommended_ancillary_studies=diagnosis.recommended_ancillary_studies,
        hypothesis_set=diagnosis.hypothesis_set,
    )


def apply_evidence_sufficiency_guard(
    diagnosis: DiagnosisPrediction,
    site_prediction: SitePrediction,
    sufficiency: EvidenceSufficiency | None,
) -> DiagnosisPrediction:
    """Downgrade over-specific diagnoses when current ROI evidence is insufficient.

    This is not a dataset-label prior. It only uses the known/predicted anatomy
    and the model's own evidence-sufficiency assessment to avoid writing a
    definitive subtype/malignancy/grade when the required image evidence is
    missing.
    """
    if sufficiency is None or sufficiency.can_confidently_report:
        return diagnosis

    missing = " ".join(
        [*sufficiency.missing_image_evidence, *sufficiency.missing_nonimage_metadata]
    ).lower()
    needs_downgrade = (
        diagnosis.confidence.lower() == "high"
        or bool(diagnosis.subtype)
        or any(t in missing for t in ["invasion", "mitotic", "weiss", "capsular", "vascular", "ki-67", "ihc"])
    )
    if not needs_downgrade:
        return diagnosis

    primary = diagnosis.broad_type or diagnosis.primary_diagnosis
    broad = diagnosis.broad_type

    diff = list(diagnosis.differential)
    if diagnosis.primary_diagnosis and diagnosis.primary_diagnosis not in diff and diagnosis.primary_diagnosis != primary:
        diff.insert(0, diagnosis.primary_diagnosis)

    missing_evidence = _dedupe([
        *diagnosis.missing_evidence,
        "current ROI evidence is insufficient for a definitive subtype, malignant potential, or grade",
    ])

    malignancy = diagnosis.malignancy
    if not malignancy and any(term in primary.lower() for term in ("carcinoma", "sarcoma", "lymphoma", "melanoma")):
        malignancy = "malignant"
    elif any(term in missing for term in ["malignant potential", "weiss", "mitotic", "capsular", "vascular", "ki-67"]):
        malignancy = "not assessable"

    return DiagnosisPrediction(
        primary_diagnosis=primary,
        confidence="low" if sufficiency.diagnostic_confidence.lower() == "low" else "medium",
        broad_type=broad,
        subtype="",
        malignancy=malignancy,
        grade="",
        differential=diff[:6],
        tumor_architecture=diagnosis.tumor_architecture,
        cytology=diagnosis.cytology,
        necrosis=diagnosis.necrosis,
        invasion=diagnosis.invasion,
        stroma_inflammation=diagnosis.stroma_inflammation,
        ihc_molecular=diagnosis.ihc_molecular,
        missing_evidence=missing_evidence,
        recommended_ancillary_studies=diagnosis.recommended_ancillary_studies,
        hypothesis_set=diagnosis.hypothesis_set,
    )


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
