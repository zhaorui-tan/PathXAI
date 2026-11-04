"""Pathology-specific evaluation metrics for H&E WSI report generation.

Design rationale
----------------
Standard NLP metrics (BLEU, ROUGE) treat all tokens equally and are insensitive
to the clinical semantics of pathology language.  These metrics are designed to
capture domain-specific quality axes that standard metrics miss.

Metric overview
---------------
1. **MorphTermF1** (`morph_term_f1`)
   Measures recall and precision of histomorphological descriptors
   (tissue architecture, cytological features, pathological events).
   Inspired by "clinical concept F1" in CheXBert (Smit et al. 2020) and
   radiology report evaluation work (Zhang et al. 2020 ClinicalNLP), adapted
   to H&E pathology vocabulary.

2. **DiagnosisHitScore** (`diagnosis_hit_score`)
   Cancer-type recognition with synonym normalisation: exact hit (same canonical
   cancer type), family hit (same cancer super-category), or miss.
   Motivated by PathVQA evaluation (He et al. 2020) and TCGA-based benchmarks.

3. **ClinicalOverclaimRate** (`clinical_overclaim_rate`)
   Fraction of specific clinical entity terms in the prediction that are NOT
   supported by the reference ("hallucination rate").  Inspired by overclaiming
   analysis in clinical NLP (Wallace et al. 2019; Smit et al. 2020).

4. **CriticalFindingRecall** (`critical_finding_recall`)
   Recall of key pathological findings (necrosis, invasion, grade markers, etc.)
   that a clinically complete report should not miss.

5. **PathologyReportScore** (`pathology_report_score`)
   Composite score combining the four metrics above, designed for ranking
   ablation runs on image-only WSI evaluation.

References
----------
- Smit et al. (2020). CheXBert: Combining Automatic Labelers and Expert
  Annotations for Accurate Radiology Report Labeling Using BERT.
  EMNLP 2020.
- He et al. (2020). PathVQA: 30000+ Questions for Medical Visual Question
  Answering. ArXiv:2003.10286.
- Zhang et al. (2020). Optimizing the Factual Correctness of a Summary:
  A Study of Summarizing Radiology Reports. ACL 2020.
- Chen et al. (2024). Automated Pathology Report Generation from WSI.
  (concept F1 / clinical accuracy evaluation pattern).
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Tuple

from pathxai_agent.evaluation.report_text_metrics import normalize_text, tokenize

# ── Morphological term vocabulary ─────────────────────────────────────────────
# Curated H&E histopathology terms organised by semantic category.
# Each category is weighted separately in the composite score.

MORPH_VOCAB: Dict[str, List[str]] = {
    # Growth pattern / tissue architecture
    "architecture": [
        "tubular", "glandular", "papillary", "micropapillary", "cribriform",
        "solid", "nested", "cord", "sheet", "trabecular", "alveolar", "lobular",
        "organoid", "acinar", "follicular", "fascicular", "storiform", "zellballen",
        "mucinous", "signet ring", "micropapillary", "infiltrative", "diffuse",
    ],
    # Cellular / nuclear features
    "cytology": [
        "pleomorphism", "pleomorphic", "hyperchromasia", "mitosis", "mitotic",
        "atypia", "atypical", "enlarged nuclei", "prominent nucleolus",
        "prominent nucleoli", "nuclear grooves", "eosinophilic cytoplasm",
        "clear cytoplasm", "clear cell", "spindle cell", "giant cell",
        "anisocytosis", "anisokaryosis", "high nuclear grade", "low nuclear grade",
    ],
    # Stroma and microenvironment
    "stroma": [
        "desmoplasia", "desmoplastic", "fibrosis", "fibrous", "sclerosis",
        "myxoid", "vascular", "hyalinized", "inflammatory infiltrate",
        "lymphocytic infiltrate", "tumor infiltrating lymphocytes",
    ],
    # Key pathological events (high clinical weight)
    "findings": [
        "necrosis", "necrotic", "coagulative necrosis", "hemorrhage",
        "calcification", "invasion", "invasive", "perineural invasion",
        "lymphovascular invasion", "vascular invasion", "capsular invasion",
        "angiolymphatic invasion", "tumor emboli",
    ],
}

# ── Diagnosis synonym groups ───────────────────────────────────────────────────
# Each tuple: (canonical_name, [aliases]).
# Used to normalise diagnosis strings before comparison.

DIAGNOSIS_SYNONYMS: List[Tuple[str, List[str]]] = [
    ("adrenocortical carcinoma", [
        "adrenocortical carcinoma", "acc", "adrenal carcinoma", "adrenal cortical carcinoma",
        "adrenal cortex carcinoma",
    ]),
    ("adrenocortical adenoma", [
        "adrenocortical adenoma", "adrenal adenoma", "adrenal cortical adenoma",
    ]),
    ("hepatocellular carcinoma", [
        "hepatocellular carcinoma", "hcc", "hepatic carcinoma", "liver carcinoma",
    ]),
    ("cholangiocarcinoma", [
        "cholangiocarcinoma", "bile duct carcinoma", "biliary carcinoma",
    ]),
    ("clear cell renal cell carcinoma", [
        "clear cell renal cell carcinoma", "ccRCC", "clear cell rcc",
        "renal cell carcinoma clear cell", "kidney clear cell carcinoma",
    ]),
    ("papillary renal cell carcinoma", [
        "papillary renal cell carcinoma", "prcc", "papillary rcc",
    ]),
    ("chromophobe renal cell carcinoma", [
        "chromophobe renal cell carcinoma", "chromophobe rcc", "ccrcc",
    ]),
    ("urothelial carcinoma", [
        "urothelial carcinoma", "transitional cell carcinoma", "bladder carcinoma",
    ]),
    ("invasive ductal carcinoma", [
        "invasive ductal carcinoma", "idc", "ductal carcinoma nst",
        "invasive carcinoma of no special type", "breast carcinoma",
    ]),
    ("invasive lobular carcinoma", [
        "invasive lobular carcinoma", "ilc", "lobular breast carcinoma",
    ]),
    ("lung adenocarcinoma", [
        "lung adenocarcinoma", "pulmonary adenocarcinoma", "luad",
    ]),
    ("lung squamous cell carcinoma", [
        "lung squamous cell carcinoma", "pulmonary squamous cell carcinoma", "lusc",
    ]),
    ("colorectal adenocarcinoma", [
        "colorectal adenocarcinoma", "colon adenocarcinoma", "rectal adenocarcinoma",
        "colorectal carcinoma", "coad", "read",
    ]),
    ("gastric adenocarcinoma", [
        "gastric adenocarcinoma", "stomach adenocarcinoma", "gastric carcinoma",
    ]),
    ("prostate adenocarcinoma", [
        "prostate adenocarcinoma", "prostatic adenocarcinoma",
        "prostate carcinoma", "prostate cancer",
    ]),
    ("endometrial adenocarcinoma", [
        "endometrial adenocarcinoma", "uterine adenocarcinoma",
        "endometrial carcinoma", "ucec",
    ]),
    ("ovarian serous carcinoma", [
        "ovarian serous carcinoma", "high grade serous carcinoma",
        "serous carcinoma ovary", "ovarian carcinoma",
    ]),
    ("cervical squamous cell carcinoma", [
        "cervical squamous cell carcinoma", "cervical carcinoma",
    ]),
    ("glioblastoma", [
        "glioblastoma", "gbm", "glioblastoma multiforme",
        "grade iv glioma", "glioma grade 4",
    ]),
    ("low grade glioma", [
        "low grade glioma", "lgg", "oligodendroglioma", "astrocytoma",
        "diffuse astrocytoma", "grade ii glioma", "grade iii glioma",
    ]),
    ("melanoma", [
        "melanoma", "cutaneous melanoma", "malignant melanoma",
    ]),
    ("sarcoma", [
        "sarcoma", "leiomyosarcoma", "liposarcoma", "synovial sarcoma",
        "rhabdomyosarcoma", "undifferentiated pleomorphic sarcoma",
    ]),
    ("mesothelioma", [
        "mesothelioma", "pleural mesothelioma",
    ]),
    ("paraganglioma", [
        "paraganglioma", "pheochromocytoma", "pheo", "chromaffin tumor",
        "extra-adrenal paraganglioma", "carotid body tumor",
    ]),
    ("seminoma", [
        "seminoma", "testicular seminoma",
    ]),
    ("thyroid papillary carcinoma", [
        "papillary thyroid carcinoma", "ptc", "thyroid papillary carcinoma",
    ]),
    ("thyroid follicular carcinoma", [
        "follicular thyroid carcinoma", "ftc",
    ]),
]

# Cancer super-family: used for partial ("family") hit
CANCER_FAMILY: Dict[str, str] = {
    "carcinoma": "carcinoma",
    "adenocarcinoma": "carcinoma",
    "squamous cell carcinoma": "carcinoma",
    "sarcoma": "sarcoma",
    "melanoma": "melanoma",
    "glioma": "glioma",
    "glioblastoma": "glioma",
    "lymphoma": "lymphoma",
    "seminoma": "germ cell tumor",
    "paraganglioma": "neuroendocrine tumor",
    "pheochromocytoma": "neuroendocrine tumor",
    "neuroendocrine tumor": "neuroendocrine tumor",
    "carcinoid": "neuroendocrine tumor",
}

# Critical pathological findings whose mention/omission has direct clinical impact
CRITICAL_FINDINGS: List[str] = [
    "necrosis",
    "invasion",
    "perineural invasion",
    "lymphovascular invasion",
    "vascular invasion",
    "capsular invasion",
    "high grade",
    "poorly differentiated",
    "pleomorphism",
    "mitosis",
    "metastasis",
    "metastatic",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _contains(text_norm: str, phrase: str) -> bool:
    phrase_norm = normalize_text(phrase)
    return bool(re.search(r"\b" + re.escape(phrase_norm) + r"\b", text_norm))


def _canonical_diagnosis(text: str) -> str | None:
    """Return the canonical cancer-type name for the first match found in text."""
    text_norm = normalize_text(text)
    for canonical, aliases in DIAGNOSIS_SYNONYMS:
        for alias in aliases:
            if _contains(text_norm, alias):
                return canonical
    return None


def _cancer_family(canonical: str | None) -> str | None:
    if canonical is None:
        return None
    norm = normalize_text(canonical)
    for key, family in CANCER_FAMILY.items():
        if key in norm:
            return family
    return None


def _morph_term_hits(text_norm: str) -> Dict[str, List[str]]:
    hits: Dict[str, List[str]] = {}
    for category, terms in MORPH_VOCAB.items():
        matched = [t for t in terms if _contains(text_norm, t)]
        if matched:
            hits[category] = matched
    return hits


# ── Public metric functions ───────────────────────────────────────────────────

def morph_term_f1(prediction: str, reference: str) -> Dict[str, Any]:
    """F1 of histomorphological descriptor terms.

    Precision = fraction of predicted morph terms supported by reference.
    Recall    = fraction of reference morph terms mentioned in prediction.
    Terms are matched by substring (with word-boundary check) after normalisation.

    Categories:
        architecture  — tissue growth pattern
        cytology      — nuclear/cellular features
        stroma        — stromal / microenvironment
        findings      — key pathological events (necrosis, invasion, …)

    The *findings* category is given 2× weight in the composite F1 because
    missing/wrong invasion or necrosis statements have higher clinical impact.
    """
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)

    all_terms = [t for terms in MORPH_VOCAB.values() for t in terms]
    pred_terms = set(t for t in all_terms if _contains(pred_norm, t))
    ref_terms = set(t for t in all_terms if _contains(ref_norm, t))

    category_detail: Dict[str, Any] = {}
    weighted_parts: List[Tuple[float, float]] = []
    for category, terms in MORPH_VOCAB.items():
        pred_cat = set(t for t in terms if t in pred_terms)
        ref_cat = set(t for t in terms if t in ref_terms)
        if not pred_cat and not ref_cat:
            continue
        tp = len(pred_cat & ref_cat)
        prec = tp / len(pred_cat) if pred_cat else 0.0
        rec = tp / len(ref_cat) if ref_cat else 0.0
        f1 = _f1(prec, rec)
        weight = 2.0 if category == "findings" else 1.0
        category_detail[category] = {
            "precision": prec, "recall": rec, "f1": f1,
            "prediction_terms": sorted(pred_cat),
            "reference_terms": sorted(ref_cat),
            "missing_terms": sorted(ref_cat - pred_cat),
        }
        weighted_parts.append((weight, f1))

    overall_f1 = (
        sum(w * s for w, s in weighted_parts) / sum(w for w, _ in weighted_parts)
        if weighted_parts else 0.0
    )
    return {
        "morph_term_f1": overall_f1,
        "prediction_morph_terms": sorted(pred_terms),
        "reference_morph_terms": sorted(ref_terms),
        "missing_morph_terms": sorted(ref_terms - pred_terms),
        "unsupported_morph_terms": sorted(pred_terms - ref_terms),
        "category_detail": category_detail,
    }


def diagnosis_hit_score(prediction: str, reference: str) -> Dict[str, Any]:
    """Cancer-type recognition score with synonym normalisation.

    Score:
        1.0  — exact canonical match (same cancer type or alias).
        0.5  — family match (same super-category: carcinoma vs carcinoma, etc.)
               but different specific type.
        0.0  — no recognisable match.

    Motivation: a model that says "carcinoma" when the reference is
    "adrenocortical carcinoma" deserves partial credit.  This mirrors how
    pathologists evaluate diagnostic accuracy at different levels of specificity.
    """
    pred_canon = _canonical_diagnosis(prediction)
    ref_canon = _canonical_diagnosis(reference)

    if ref_canon is None:
        # Reference has no parseable canonical type; fall back to simple token overlap
        pred_tokens = set(tokenize(prediction))
        ref_tokens = set(tokenize(reference))
        diag_tokens = {
            normalize_text(t)
            for group in [a for _, aliases in DIAGNOSIS_SYNONYMS for a in aliases]
            for t in [group]
            if normalize_text(group) in ref_tokens
        }
        if not diag_tokens:
            score = 0.0
            hit_level = "no_reference_canonical"
        else:
            score = len(pred_tokens & diag_tokens) / len(diag_tokens)
            hit_level = "token_overlap_fallback"
    elif pred_canon is None:
        score = 0.0
        hit_level = "no_prediction_canonical"
    elif pred_canon == ref_canon:
        score = 1.0
        hit_level = "exact"
    elif _cancer_family(pred_canon) is not None and _cancer_family(pred_canon) == _cancer_family(ref_canon):
        score = 0.5
        hit_level = "family"
    else:
        score = 0.0
        hit_level = "miss"

    return {
        "diagnosis_hit_score": score,
        "hit_level": hit_level,
        "prediction_canonical": pred_canon,
        "reference_canonical": ref_canon,
        "prediction_family": _cancer_family(pred_canon),
        "reference_family": _cancer_family(ref_canon),
    }


def clinical_overclaim_rate(prediction: str, reference: str) -> Dict[str, Any]:
    """Fraction of specific clinical terms predicted but unsupported by the reference.

    An "overclaim" is a specific pathological term that appears in the prediction
    but has no support in the reference.  A high overclaim rate suggests the model
    is hallucinating clinical findings.

    Differential-diagnosis terms mentioned in the Final Summary are *excluded*
    from the overclaim count — listing appropriate differentials is clinically
    correct and should not be penalised as hallucination.

    Only *specific* vocabulary terms are evaluated (general words like "tissue"
    are excluded).  Computed over the combined MORPH_VOCAB + DIAGNOSIS terms.

    Overclaim rate = |unsupported ∩ specific ∩ non-differential| / max(|predicted ∩ specific ∩ non-differential|, 1)
    """
    from pathxai_agent.evaluation.reference_cleaner import extract_differential_terms_from_summary
    # Terms from differential sentences in Final Summary are exempt
    exempt_terms = extract_differential_terms_from_summary(prediction)

    all_specific = [t for terms in MORPH_VOCAB.values() for t in terms]
    all_specific += [a for _, aliases in DIAGNOSIS_SYNONYMS for a in aliases]

    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)

    predicted = {t for t in all_specific if _contains(pred_norm, t)}
    # Exclude differential-exempt terms from both sides
    predicted_accountable = predicted - exempt_terms
    supported = {t for t in predicted_accountable if _contains(ref_norm, t)}
    unsupported = predicted_accountable - supported

    rate = len(unsupported) / max(len(predicted_accountable), 1)
    return {
        "overclaim_rate": rate,
        "predicted_specific_terms": sorted(predicted),
        "exempt_differential_terms": sorted(exempt_terms),
        "accountable_terms": sorted(predicted_accountable),
        "supported_terms": sorted(supported),
        "unsupported_terms": sorted(unsupported),
    }


def critical_finding_recall(prediction: str, reference: str) -> Dict[str, Any]:
    """Recall of critical pathological findings.

    Measures whether findings with direct clinical impact (necrosis, invasion,
    high grade, etc.) that appear in the reference are also mentioned in the
    prediction.  Low recall on this metric means the model is missing important
    prognostic/staging findings.
    """
    pred_norm = normalize_text(prediction)
    ref_norm = normalize_text(reference)

    ref_findings = [f for f in CRITICAL_FINDINGS if _contains(ref_norm, f)]
    if not ref_findings:
        return {
            "critical_finding_recall": 1.0,
            "reference_findings": [],
            "recalled_findings": [],
            "missed_findings": [],
            "note": "no critical findings in reference",
        }
    recalled = [f for f in ref_findings if _contains(pred_norm, f)]
    missed = [f for f in ref_findings if f not in recalled]
    recall = len(recalled) / len(ref_findings)
    return {
        "critical_finding_recall": recall,
        "reference_findings": ref_findings,
        "recalled_findings": recalled,
        "missed_findings": missed,
    }


def pathology_report_score(prediction: str, reference: str) -> Dict[str, Any]:
    """Composite pathology-specific quality score.

    Combines the four domain-specific metrics into a single scalar.
    Weights reflect clinical importance:
        morph_term_f1          × 0.30  (morphological accuracy)
        diagnosis_hit_score    × 0.40  (diagnostic accuracy — most important)
        1 - overclaim_rate     × 0.15  (hallucination penalty)
        critical_finding_recall × 0.15 (recall of high-impact findings)

    Score range: [0, 1].  Higher is better.
    """
    morph = morph_term_f1(prediction, reference)
    diag = diagnosis_hit_score(prediction, reference)
    overclaim = clinical_overclaim_rate(prediction, reference)
    cfr = critical_finding_recall(prediction, reference)

    score = (
        0.30 * morph["morph_term_f1"]
        + 0.40 * diag["diagnosis_hit_score"]
        + 0.15 * (1.0 - overclaim["overclaim_rate"])
        + 0.15 * cfr["critical_finding_recall"]
    )
    return {
        "pathology_report_score": score,
        "components": {
            "morph_term_f1": morph["morph_term_f1"],
            "diagnosis_hit_score": diag["diagnosis_hit_score"],
            "hallucination_penalty": 1.0 - overclaim["overclaim_rate"],
            "critical_finding_recall": cfr["critical_finding_recall"],
        },
        "detail": {
            "morph_term_f1": morph,
            "diagnosis_hit": diag,
            "clinical_overclaim": overclaim,
            "critical_finding_recall": cfr,
        },
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
