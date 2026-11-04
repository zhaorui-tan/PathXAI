"""Evaluation framework for the PathXAI v2 pipeline.

Key improvements over v1:
1. Separate eval LLM from generation LLM — no self-preference bias.
2. Rule-based fact extraction as primary, LLM as opt-in secondary.
3. Confidence calibration metrics.
4. Extended diagnosis synonym coverage.
5. Cross-case consistency checks.

Architecture:
    EvalRunner
        ├── TextMetrics        (BLEU, ROUGE, semantic similarity)
        ├── PathologyMetrics   (MorphTermF1, DiagnosisHitScore, etc.)
        ├── HierarchyMetrics   (weighted diagnostic field scoring)
        └── ConsistencyMetrics (cross-case, calibration)
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pathxai_agent_v2.config import Settings
from pathxai_agent_v2.llm import LLMClient
from pathxai_agent_v2.pipeline.report_generation import EXPECTED_SECTIONS


# ── Text metrics ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.split()


def _ngrams(tokens: list[str], n: int) -> list[tuple]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu(prediction: str, reference: str, max_n: int = 4) -> dict[str, float]:
    """Sentence-level BLEU with add-1 smoothing."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)} | {"bleu_avg": 0.0}
    bp = min(1.0, math.exp(1.0 - len(ref_tokens) / max(len(pred_tokens), 1)))
    scores = {}
    log_sum = 0.0
    for n in range(1, max_n + 1):
        p_ng = Counter(_ngrams(pred_tokens, n))
        r_ng = Counter(_ngrams(ref_tokens, n))
        clipped = sum(min(c, r_ng[g]) for g, c in p_ng.items())
        total = max(len(pred_tokens) - n + 1, 1)
        precision = (clipped + 1) / (total + 1)
        scores[f"bleu_{n}"] = precision * bp
        log_sum += math.log(precision)
    scores["bleu_avg"] = bp * math.exp(log_sum / max_n)
    return scores


def rouge_n(prediction: str, reference: str, n: int = 1) -> dict[str, float]:
    """ROUGE-N recall-oriented overlap."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    p_ng = Counter(_ngrams(pred_tokens, n))
    r_ng = Counter(_ngrams(ref_tokens, n))
    overlap = sum((p_ng & r_ng).values())
    precision = overlap / max(1, sum(p_ng.values()))
    recall = overlap / max(1, sum(r_ng.values()))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def section_similarity(pred_sections: dict[str, str], ref_sections: dict[str, str]) -> dict[str, float]:
    """Compute token-F1 similarity per section."""
    sims = {}
    for section in EXPECTED_SECTIONS:
        pred = pred_sections.get(section, "")
        ref = ref_sections.get(section, "")
        if not ref or ref in {"Not assessable from H&E images.", "Not specified.", ""}:
            continue
        p_tokens = set(_tokenize(pred))
        r_tokens = set(_tokenize(ref))
        if not r_tokens:
            continue
        overlap = p_tokens & r_tokens
        precision = len(overlap) / max(1, len(p_tokens))
        recall = len(overlap) / max(1, len(r_tokens))
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        sims[section] = f1
    return sims


# ── Pathology-specific metrics ─────────────────────────────────────────────────

# Extended diagnosis synonym groups (covers all TCGA types)
DIAGNOSIS_SYNONYMS: dict[str, list[str]] = {
    "adrenocortical carcinoma": [
        "adrenocortical carcinoma", "acc", "adrenal carcinoma",
        "adrenal cortical carcinoma",
    ],
    "adrenocortical neoplasm": [
        "adrenocortical neoplasm", "adrenal cortical neoplasm",
        "adrenocortical tumor", "adrenal cortical tumor",
        "adrenal cortical neoplasm of indeterminate malignant potential",
        "adrenocortical neoplasm of indeterminate malignant potential",
        "neoplasm of indeterminate malignant potential",
        "uncertain malignant potential adrenocortical neoplasm",
        "indeterminate malignant potential adrenocortical neoplasm",
    ],
    "adrenocortical adenoma": [
        "adrenocortical adenoma", "adrenal adenoma", "adrenal cortical adenoma",
    ],
    "hepatocellular carcinoma": [
        "hepatocellular carcinoma", "hcc", "hepatic carcinoma", "liver carcinoma",
        "liver cancer",
    ],
    "cholangiocarcinoma": [
        "cholangiocarcinoma", "bile duct carcinoma", "biliary carcinoma",
        "intrahepatic cholangiocarcinoma",
    ],
    "clear cell renal cell carcinoma": [
        "clear cell renal cell carcinoma", "ccrcc", "clear cell rcc",
        "renal cell carcinoma clear cell",
    ],
    "papillary renal cell carcinoma": [
        "papillary renal cell carcinoma", "prcc", "papillary rcc",
    ],
    "chromophobe renal cell carcinoma": [
        "chromophobe renal cell carcinoma", "chromophobe rcc",
    ],
    "urothelial carcinoma": [
        "urothelial carcinoma", "transitional cell carcinoma", "bladder carcinoma",
        "bladder cancer",
    ],
    "invasive ductal carcinoma": [
        "invasive ductal carcinoma", "idc", "ductal carcinoma nst",
        "invasive carcinoma of no special type", "breast carcinoma",
        "invasive breast carcinoma",
    ],
    "invasive lobular carcinoma": [
        "invasive lobular carcinoma", "ilc", "lobular breast carcinoma",
    ],
    "lung adenocarcinoma": [
        "lung adenocarcinoma", "pulmonary adenocarcinoma", "luad",
    ],
    "lung squamous cell carcinoma": [
        "lung squamous cell carcinoma", "pulmonary squamous cell carcinoma", "lusc",
    ],
    "colorectal adenocarcinoma": [
        "colorectal adenocarcinoma", "colon adenocarcinoma", "rectal adenocarcinoma",
        "colorectal carcinoma", "coad", "read",
    ],
    "gastric adenocarcinoma": [
        "gastric adenocarcinoma", "stomach adenocarcinoma", "gastric carcinoma",
    ],
    "esophageal carcinoma": [
        "esophageal carcinoma", "oesophageal carcinoma", "esophageal adenocarcinoma",
        "esophageal squamous cell carcinoma",
    ],
    "prostate adenocarcinoma": [
        "prostate adenocarcinoma", "prostatic adenocarcinoma",
        "prostatic acinar adenocarcinoma", "acinar adenocarcinoma",
        "prostate carcinoma", "prostate cancer",
    ],
    "cervical squamous cell carcinoma": [
        "cervical squamous cell carcinoma", "cervical carcinoma",
        "cervical cancer",
    ],
    "endometrial adenocarcinoma": [
        "endometrial adenocarcinoma", "uterine adenocarcinoma",
        "endometrial carcinoma", "ucec",
    ],
    "ovarian serous carcinoma": [
        "ovarian serous carcinoma", "high grade serous carcinoma",
        "serous carcinoma ovary", "ovarian carcinoma",
    ],
    "glioblastoma": [
        "glioblastoma", "gbm", "glioblastoma multiforme",
        "grade iv glioma",
    ],
    "low grade glioma": [
        "low grade glioma", "lgg", "oligodendroglioma", "astrocytoma",
        "diffuse astrocytoma", "grade ii glioma", "grade iii glioma",
    ],
    "head and neck squamous cell carcinoma": [
        "head and neck squamous cell carcinoma", "hnscc",
        "oral squamous cell carcinoma", "laryngeal squamous cell carcinoma",
    ],
    "melanoma": [
        "melanoma", "cutaneous melanoma", "malignant melanoma",
    ],
    "sarcoma": [
        "sarcoma", "leiomyosarcoma", "liposarcoma", "synovial sarcoma",
        "undifferentiated pleomorphic sarcoma",
    ],
    "mesothelioma": [
        "mesothelioma", "pleural mesothelioma",
    ],
    "paraganglioma": [
        "paraganglioma", "pheochromocytoma", "chromaffin tumor",
    ],
    "seminoma": [
        "seminoma", "testicular seminoma", "germ cell tumor",
    ],
    "thyroid papillary carcinoma": [
        "papillary thyroid carcinoma", "ptc", "thyroid papillary carcinoma",
    ],
    "thyroid follicular carcinoma": [
        "follicular thyroid carcinoma", "ftc",
    ],
    "thymoma": [
        "thymoma", "thymic carcinoma", "thymic tumor",
    ],
    "uveal melanoma": [
        "uveal melanoma", "ocular melanoma", "choroidal melanoma",
    ],
    "diffuse large b-cell lymphoma": [
        "diffuse large b-cell lymphoma", "dlbcl", "large b-cell lymphoma",
    ],
}

# Morphological term categories
MORPH_VOCAB: dict[str, list[str]] = {
    "architecture": [
        "tubular", "glandular", "papillary", "micropapillary", "cribriform",
        "solid", "nested", "cord", "sheet", "trabecular", "alveolar", "lobular",
        "organoid", "acinar", "follicular", "fascicular", "storiform",
        "mucinous", "signet ring", "infiltrative", "diffuse",
    ],
    "cytology": [
        "pleomorphism", "pleomorphic", "hyperchromasia", "mitosis", "mitotic",
        "atypia", "atypical", "enlarged nuclei", "prominent nucleolus",
        "prominent nucleoli", "nuclear grooves", "eosinophilic cytoplasm",
        "clear cytoplasm", "clear cell", "spindle cell", "giant cell",
        "anisocytosis", "anisokaryosis", "high nuclear grade", "low nuclear grade",
    ],
    "stroma": [
        "desmoplasia", "desmoplastic", "fibrosis", "fibrous", "sclerosis",
        "myxoid", "vascular", "hyalinized", "inflammatory infiltrate",
        "lymphocytic infiltrate", "tumor infiltrating lymphocytes",
    ],
    "findings": [
        "necrosis", "necrotic", "coagulative necrosis", "hemorrhage",
        "calcification", "invasion", "invasive", "perineural invasion",
        "lymphovascular invasion", "vascular invasion", "capsular invasion",
        "angiolymphatic invasion", "tumor emboli",
    ],
}

CRITICAL_FINDINGS = [
    "necrosis", "invasion", "perineural invasion",
    "lymphovascular invasion", "vascular invasion", "capsular invasion",
    "high grade", "poorly differentiated", "pleomorphism", "mitosis",
    "metastasis", "metastatic",
]


def _contains_word(text: str, phrase: str) -> bool:
    """Check if phrase appears as a word/subphrase in text."""
    text_norm = " ".join(_tokenize(text))
    phrase_norm = " ".join(_tokenize(phrase))
    return phrase_norm in text_norm


def canonical_diagnosis(text: str) -> str | None:
    """Return the canonical diagnosis name for the most-specific match found."""
    text_norm = " ".join(_tokenize(text))
    matches: list[tuple[int, int, str]] = []
    for canonical, aliases in DIAGNOSIS_SYNONYMS.items():
        for alias in aliases:
            if _contains_word(text_norm, alias):
                toks = _tokenize(alias)
                matches.append((len(toks), len(alias), canonical))
    if not matches:
        return None
    matches.sort(reverse=True)
    return matches[0][2]


def _diagnosis_aliases(canonical: str | None) -> list[str]:
    if not canonical:
        return []
    c = canonical.lower().strip()
    for key, aliases in DIAGNOSIS_SYNONYMS.items():
        if key == c:
            return [key] + [str(a).lower() for a in aliases]
    return [c]


def morph_term_f1(prediction: str, reference: str) -> dict[str, Any]:
    """F1 of histomorphological descriptor terms."""
    pred_norm = " ".join(_tokenize(prediction))
    ref_norm = " ".join(_tokenize(reference))

    all_terms = [t for terms in MORPH_VOCAB.values() for t in terms]
    pred_terms = {t for t in all_terms if _contains_word(pred_norm, t)}
    ref_terms = {t for t in all_terms if _contains_word(ref_norm, t)}

    category_scores = {}
    for cat, terms in MORPH_VOCAB.items():
        p_cat = {t for t in terms if t in pred_terms}
        r_cat = {t for t in terms if t in ref_terms}
        if not p_cat and not r_cat:
            continue
        tp = len(p_cat & r_cat)
        prec = tp / len(p_cat) if p_cat else 0.0
        rec = tp / len(r_cat) if r_cat else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        category_scores[cat] = {"precision": prec, "recall": rec, "f1": f1}

    if not category_scores:
        return {"overall_f1": 0.0, "categories": {}}

    # findings category gets 2x weight
    total_weight = sum(2.0 if c == "findings" else 1.0 for c in category_scores)
    weighted = sum(
        s["f1"] * (2.0 if c == "findings" else 1.0)
        for c, s in category_scores.items()
    )
    return {
        "overall_f1": weighted / total_weight if total_weight > 0 else 0.0,
        "categories": category_scores,
    }


def diagnosis_hit_score(prediction: str, reference: str) -> dict[str, Any]:
    """Cancer-type recognition with synonym normalization."""
    pred_dx = canonical_diagnosis(prediction)
    ref_dx = canonical_diagnosis(reference)
    if pred_dx and ref_dx:
        match_type = "exact" if pred_dx == ref_dx else "miss"
        return {"score": 1.0 if match_type == "exact" else 0.0, "match_type": match_type,
                "predicted": pred_dx, "reference": ref_dx}
    if not pred_dx and not ref_dx:
        return {"score": 0.0, "match_type": "no_diagnosis", "predicted": None, "reference": None}
    return {"score": 0.0, "match_type": "mismatch", "predicted": pred_dx, "reference": ref_dx}


def critical_finding_recall(prediction: str, reference: str) -> dict[str, Any]:
    """Recall of critical pathological findings."""
    pred_norm = " ".join(_tokenize(prediction))
    ref_norm = " ".join(_tokenize(reference))
    ref_findings = {f for f in CRITICAL_FINDINGS if _contains_word(ref_norm, f)}
    if not ref_findings:
        return {"recall": 1.0, "matched": [], "missing": [], "note": "no critical findings in reference"}
    matched = [f for f in ref_findings if _contains_word(pred_norm, f)]
    missing = sorted(ref_findings - set(matched))
    return {"recall": len(matched) / len(ref_findings), "matched": matched, "missing": missing}


def clinical_overclaim_rate(prediction: str, reference: str) -> dict[str, Any]:
    """Fraction of clinical entity terms in prediction NOT in reference."""
    pred_norm = " ".join(_tokenize(prediction))
    ref_norm = " ".join(_tokenize(reference))
    all_terms = [t for terms in MORPH_VOCAB.values() for t in terms] + CRITICAL_FINDINGS
    pred_terms = {t for t in all_terms if _contains_word(pred_norm, t)}
    ref_terms = {t for t in all_terms if _contains_word(ref_norm, t)}
    if not pred_terms:
        return {"rate": 0.0, "overclaimed": [], "total": 0}
    overclaimed = sorted(pred_terms - ref_terms)
    return {"rate": len(overclaimed) / len(pred_terms), "overclaimed": overclaimed,
            "total": len(pred_terms)}


def pathology_report_score(prediction: str, reference: str) -> dict[str, Any]:
    """Composite pathology report score."""
    morph = morph_term_f1(prediction, reference)
    diag = diagnosis_hit_score(prediction, reference)
    crit = critical_finding_recall(prediction, reference)
    overclaim = clinical_overclaim_rate(prediction, reference)

    composite = (
        0.30 * morph["overall_f1"]
        + 0.35 * diag["score"]
        + 0.25 * crit["recall"]
        + 0.10 * (1.0 - overclaim["rate"])
    )
    return {
        "composite_score": composite,
        "morph_term_f1": morph["overall_f1"],
        "diagnosis_hit": diag["score"],
        "critical_finding_recall": crit["recall"],
        "overclaim_rate": overclaim["rate"],
        "details": {"morph": morph, "diagnosis": diag, "critical": crit, "overclaim": overclaim},
    }


# ── Diagnostic hierarchy scoring ───────────────────────────────────────────────

def extract_diagnostic_facts_rule(report_text: str) -> dict[str, str]:
    """Rule-based extraction of diagnostic facts from a structured report."""
    sections = _extract_report_sections(report_text)
    diagnosis = sections.get("Diagnosis", "")
    site = sections.get("Specimen/Site", "")
    # Score factual fields from the final report sections that state the
    # primary diagnosis and observable findings.  Do not let differential
    # diagnoses or recommended ancillary stains turn a guarded report into a
    # false malignant/subtype claim.
    finding_text = _primary_evaluable_text(sections, report_text)

    facts = {
        "site": site or "not specified",
        "primary_diagnosis": diagnosis or "not specified",
        "cancer_type": _infer_cancer_type(diagnosis),
        "histologic_subtype": diagnosis,
        "malignancy_status": _infer_malignancy(finding_text),
        "invasive_status": _infer_invasive(finding_text),
        "grade": _infer_grade(finding_text),
        "necrosis": _infer_necrosis(finding_text),
        "lymphovascular_invasion": _infer_lvi(finding_text),
        "margin_status": "not specified",
        "stage": "not specified",
        "ihc_molecular": "not specified",
    }
    return facts


def evaluate_diagnostic_hierarchy(
    pred_facts: dict[str, str],
    ref_facts: dict[str, str],
    field_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Weighted hierarchical scoring of diagnostic fields."""
    weights = field_weights or {
        "site": 0.15, "cancer_type": 0.20, "histologic_subtype": 0.25,
        "malignancy_status": 0.15, "invasive_status": 0.15,
        "grade": 0.05, "necrosis": 0.05,
    }
    field_scores = {}
    weighted = 0.0
    weight_sum = 0.0

    for field, weight in weights.items():
        pred = str(pred_facts.get(field, "not specified")).strip().lower()
        ref = str(ref_facts.get(field, "not specified")).strip().lower()
        score, status = _compare_field(field, pred, ref)
        # Fields the reference cannot evaluate (no ground truth) are excluded
        # from the weighted average so an uninformative reference does not
        # penalise the prediction. Mirrors the tier scorer's renormalization.
        applicable = status not in ("not_evaluated", "not_applicable")
        field_scores[field] = {"score": score, "status": status,
                                "prediction": pred, "reference": ref,
                                "weight": weight, "applicable": applicable}
        if applicable:
            weighted += score * weight
            weight_sum += weight

    return {
        "overall_score": weighted / weight_sum if weight_sum > 0 else 0.0,
        "field_scores": field_scores,
    }


# ── Hierarchical diagnosis scoring (broad → subtype → malignancy) ──────────────

# Ordered most-specific-first so compound entities win over their substrings
# (e.g. "carcinosarcoma" before "carcinoma", "embryonal carcinoma" before it).
_BROAD_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("carcinosarcoma", ["carcinosarcoma", "malignant mixed mullerian", "mmmt"]),
    ("germ cell tumor", ["germ cell", "seminoma", "teratoma", "yolk sac", "embryonal carcinoma"]),
    ("thymic epithelial neoplasm", ["thymoma", "thymic carcinoma"]),
    ("neuroendocrine neoplasm", ["neuroendocrine", "pheochromocytoma", "paraganglioma", "carcinoid"]),
    ("glioma", ["glioblastoma", "glioma", "astrocytoma", "oligodendroglioma"]),
    ("melanoma", ["melanoma", "melanocytic"]),
    ("lymphoma", ["lymphoma", "lymphoid neoplasm"]),
    ("mesothelioma", ["mesothelioma"]),
    ("sarcoma", ["sarcoma"]),
    ("carcinoma", ["carcinoma", "adenocarcinoma"]),
    ("neoplasm", ["neoplasm", "tumor", "tumour"]),
]


def infer_broad_type(text: str) -> str:
    """Infer the histogenetic broad lineage from free text.

    Returns one of the canonical broad-type labels, or "unspecified".
    """
    t = " ".join(_tokenize(text))
    for label, keywords in _BROAD_TYPE_KEYWORDS:
        if any(_contains_word(t, kw) for kw in keywords):
            return label
    return "unspecified"


# ── Hyponym mapping: specific broad types that are subsumed by broader labels ─
_BROAD_HYPONYMS: dict[str, list[str]] = {
    "neoplasm": [
        "carcinoma", "sarcoma", "lymphoma", "melanoma", "glioma",
        "neuroendocrine neoplasm", "germ cell tumor", "mesothelioma",
        "carcinosarcoma", "thymic epithelial neoplasm",
    ],
    "carcinoma": ["adenocarcinoma"],
}


def _is_broad_hyponym(pred_broad: str, ref_broad: str) -> bool:
    """Return True if pred_broad is a specific subtype of ref_broad.

    e.g. "carcinoma" IS a "neoplasm" → 0.75 partial credit.
    """
    return pred_broad in _BROAD_HYPONYMS.get(ref_broad, [])


def _are_broad_siblings(a: str, b: str) -> bool:
    """Return True if both broad types share a common parent in the taxonomy.

    e.g. "neuroendocrine neoplasm" and "carcinoma" are both under "neoplasm" → True.
    """
    if a == b or not a or not b:
        return False
    for parent, children in _BROAD_HYPONYMS.items():
        if a in children and b in children:
            return True
    return False


def _subtype_contained(pred_norm: str, ref_norm: str, pred_entity: str, ref_entity: str) -> bool:
    """Return True if pred is a less-detailed version of ref (token subset).
    
    e.g. "urothelial carcinoma" ⊂ "urothelial cell carcinoma + cis" → 0.75.
    """
    if not pred_entity or not ref_entity:
        return False
    pred_toks = set(_tokenize(pred_entity))
    ref_toks = set(_tokenize(ref_entity))
    return bool(pred_toks) and pred_toks.issubset(ref_toks)


def _infer_broad_from_entity(entity: str, gt: dict[str, Any]) -> str:
    """Infer canonical broad_type from a specific diagnostic entity.

    Mirrors orchestrator._infer_broad_from_subtype but uses GT lineage as
    tie-breaker when the entity is ambiguous.
    """
    s = str(entity or "").strip().lower()
    if not s:
        return "neoplasm"
    if any(t in s for t in ("neuroendocrine", "carcinoid", "pheochromocytoma", "paraganglioma")):
        return "neuroendocrine neoplasm"
    if any(t in s for t in ("sarcoma", "gist", "leiomyosarcoma", "liposarcoma", "angiosarcoma", "chondrosarcoma", "osteosarcoma")):
        return "sarcoma"
    if any(t in s for t in ("lymphoma", "lymphoid", "hodgkin", "myeloma", "leukemia")):
        return "lymphoma"
    if any(t in s for t in ("melanoma", "melanocytic")):
        return "melanoma"
    if any(t in s for t in ("glioma", "glioblastoma", "astrocytoma", "oligodendroglioma")):
        return "glioma"
    if any(t in s for t in ("germ cell", "seminoma", "teratoma", "yolk sac")):
        return "germ cell tumor"
    if "mesothelioma" in s:
        return "mesothelioma"
    if "carcinosarcoma" in s or "mmmt" in s:
        return "carcinosarcoma"
    if "thymic" in s or "thymoma" in s:
        return "thymic epithelial neoplasm"
    if any(t in s for t in ("carcinoma", "adenoma", "papilloma", "adenocarcinoma", "squamous")):
        return "carcinoma"
    return gt.get("broad_type", "neoplasm").lower()


def _infer_malignancy_tier(text: str) -> str:
    """Coarse malignancy tier: malignant / borderline / benign / unspecified."""
    t = " ".join(_tokenize(text))
    raw = text.lower()
    if _malignancy_not_assessable(raw):
        return "unspecified"
    if any(_contains_word(t, w) for w in
           ["carcinoma", "sarcoma", "lymphoma", "melanoma", "glioblastoma",
            "malignant", "metastatic", "high grade", "poorly differentiated"]):
        return "malignant"
    if any(_contains_word(t, w) for w in
           ["borderline", "uncertain malignant potential", "indeterminate",
            "low malignant potential", "atypical"]):
        return "borderline"
    if any(_contains_word(t, w) for w in ["benign", "adenoma", "hyperplasia"]):
        return "benign"
    return "unspecified"


# Histologic grade is a CASE-SPECIFIC axis (it varies case to case and cannot be
# anchored on the cohort), so ground truth is taken from the reference report.
_GRADE_ORDINAL = {"low": 1, "intermediate": 2, "high": 3}

_GRADE_HIGH_RE = re.compile(
    r"\b(high[\s-]*grade|poorly[\s-]*differentiated|undifferentiated|dedifferentiated"
    r"|grade\s*(?:3|4|iii|iv)|g[34]\b|fuhrman\s*(?:3|4|iii|iv)|nuclear\s*grade\s*[34])\b",
    re.IGNORECASE,
)
_GRADE_LOW_RE = re.compile(
    r"\b(low[\s-]*grade|well[\s-]*differentiated"
    r"|grade\s*(?:1|i)\b|g1\b|fuhrman\s*(?:1|i)\b|nuclear\s*grade\s*1)\b",
    re.IGNORECASE,
)
_GRADE_INT_RE = re.compile(
    r"\b(intermediate[\s-]*grade|moderately[\s-]*differentiated|moderate[\s-]*differentiation"
    r"|grade\s*(?:2|ii)\b|g2\b|fuhrman\s*(?:2|ii)\b|nuclear\s*grade\s*2)\b",
    re.IGNORECASE,
)


def _infer_grade_tier(text: str) -> str:
    """Normalize histologic grade to low / intermediate / high, else 'unspecified'.

    Ordering matters: high (grade 3/4) is checked before low (grade 1) so that
    roman/numeric overlaps (``grade ii`` vs ``grade i``) resolve correctly.
    """
    if not text:
        return "unspecified"
    t = text.lower()
    if _GRADE_HIGH_RE.search(t):
        return "high"
    if _GRADE_LOW_RE.search(t):
        return "low"
    if _GRADE_INT_RE.search(t):
        return "intermediate"
    return "unspecified"



def evaluate_diagnosis_tiers(
    prediction: str,
    cancer_type: str,
    reference: str = "",
    tier_weights: dict[str, float] | None = None,
    pred_facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score a diagnosis on four INDEPENDENT tiers with partial credit.

    Site is a known input (not scored). The four answer tiers are:
        broad      — histogenetic lineage (carcinoma / sarcoma / glioma / ...)
        subtype    — the specific entity
        malignancy — benign / borderline / malignant
        grade      — histologic grade (low / intermediate / high)

    broad/subtype/malignancy are anchored on the known TCGA cohort
    (``cancer_type``). Grade is CASE-SPECIFIC (it varies per case and cannot be
    read off the cohort), so its ground truth comes from the reference report.
    When the reference states no grade, the grade tier is marked not-applicable
    and its weight is renormalized away so the case is not penalised.

    Getting the broad lineage right but the subtype wrong still earns the broad
    tier's credit — it is never zero.

    ``pred_facts`` — optional structured tiers emitted by the generation side
    (``DiagnosisPrediction.broad_type/subtype/malignancy/grade``). When present
    these are used directly instead of re-extracting them from the report text,
    which avoids the "rules can't be exhaustive" failure mode. Text inference
    remains the fallback for external / v1 reports that lack structured fields.
    """
    from pathxai_agent_v2.eval.tcga_ground_truth import ground_truth_for

    weights = tier_weights or {"broad": 0.35, "subtype": 0.35,
                               "malignancy": 0.15, "grade": 0.15}
    gt = ground_truth_for(cancer_type)

    # Prefer the Diagnosis section for broad/subtype inference so differential
    # mentions elsewhere in the report don't pollute the primary lineage.
    dx_section = _extract_report_sections(prediction).get("Diagnosis", "")
    dx_text = dx_section or prediction
    tier_text = _primary_evaluable_text(_extract_report_sections(prediction), prediction)
    dx_norm = " ".join(_tokenize(dx_text))
    pred_norm = " ".join(_tokenize(prediction))

    # Structured tiers from the generation side take precedence over text.
    facts = pred_facts or {}
    fact_broad = str(facts.get("broad_type") or "").strip()
    fact_subtype = str(facts.get("subtype") or "").strip()
    fact_mal = str(facts.get("malignancy") or "").strip().lower()
    fact_grade = str(facts.get("grade") or "").strip().lower()
    # A structured subtype string is also a searchable haystack for aliases.
    if fact_subtype:
        dx_norm = (dx_norm + " " + " ".join(_tokenize(fact_subtype))).strip()

    def _pred_broad(text: str) -> str:
        if fact_broad:
            b = infer_broad_type(fact_broad)
            return b if b != "unspecified" else fact_broad.lower()
        return infer_broad_type(text)

    def _pred_mal(text: str) -> str:
        if fact_mal in ("benign", "borderline", "malignant"):
            return fact_mal
        return _infer_malignancy_tier(text)

    def _pred_grade(text: str) -> str:
        g = _infer_grade_tier(fact_grade) if fact_grade else "unspecified"
        return g if g != "unspecified" else _infer_grade_tier(text)

    # Grade tier (case-specific; ground truth taken from the reference report).
    ref_grade = _infer_grade_tier(reference)
    pred_grade = _pred_grade(dx_text if dx_section else prediction)
    if ref_grade == "unspecified":
        # No gradable ground truth for this case — exclude from the score.
        grade_score, grade_status, grade_applicable = 0.0, "not_applicable", False
    elif pred_grade == "unspecified":
        grade_score, grade_status, grade_applicable = 0.0, "missing", True
    else:
        dist = abs(_GRADE_ORDINAL[pred_grade] - _GRADE_ORDINAL[ref_grade])
        if dist == 0:
            grade_score, grade_status, grade_applicable = 1.0, "exact", True
        elif dist == 1:
            grade_score, grade_status, grade_applicable = 0.5, "adjacent", True
        else:
            grade_score, grade_status, grade_applicable = 0.0, "miss", True

    if not gt:
        # No cohort anchor — fall back to text-vs-text lineage comparison.
        pred_broad = _pred_broad(dx_text)
        ref_broad = infer_broad_type(reference)
        broad_score = 1.0 if pred_broad != "unspecified" and pred_broad == ref_broad else 0.0
        pred_sub = fact_subtype.lower() if fact_subtype else canonical_diagnosis(dx_text)
        ref_sub = canonical_diagnosis(reference)
        subtype_score = 1.0 if pred_sub and pred_sub == ref_sub else 0.0
        pred_mal = _pred_mal(tier_text)
        ref_mal = _infer_malignancy_tier(reference)
        mal_score = 1.0 if pred_mal != "unspecified" and pred_mal == ref_mal else 0.0
        return _tier_result(
            weights, anchored=False,
            broad=(broad_score, pred_broad, ref_broad, ""),
            subtype=(subtype_score, pred_sub or "", ref_sub or "", ""),
            malignancy=(mal_score, pred_mal, ref_mal, ""),
            grade=(grade_score, pred_grade, ref_grade, grade_status, grade_applicable),
        )

    # ── Subtype tier (scored FIRST — if correct, broad is implied correct) ──
    aliases = [str(a).lower() for a in gt.get("subtype_aliases", [])]
    ref_subtype = str(gt["subtype"])
    # The reference subtype is cohort-anchored. Raw report text may contain
    # broad/noisy wording, so it is only used for grade above, not to replace the
    # cohort-level subtype target.
    ref_canon = canonical_diagnosis(ref_subtype)
    ref_aliases = list(dict.fromkeys(aliases + _diagnosis_aliases(ref_canon or ref_subtype)))
    matched_alias = next((a for a in ref_aliases if _contains_word(dx_norm, a)), "")
    pred_canon = canonical_diagnosis(dx_text)
    fact_canon = canonical_diagnosis(fact_subtype) if fact_subtype else None
    pred_entity = fact_canon or pred_canon
    if matched_alias:
        subtype_score, subtype_status = 1.0, "exact"
    else:
        pred_subtype_text = fact_subtype or pred_entity or pred_canon or dx_text
        subtype_score, subtype_status = _compare_histologic_subtype(
            str(pred_subtype_text).lower(),
            str(ref_subtype).lower(),
        )
        if subtype_status == "exact_match":
            subtype_status = "exact"
        elif subtype_status == "synonym_match":
            subtype_status = "synonym_exact"
        elif subtype_status == "mismatch":
            subtype_status = "miss"

    # ── Broad tier (implied correct if subtype is correct) ─────────────────
    ref_broad = str(gt["broad_type"]).lower()
    pred_broad = _pred_broad(dx_text)
    lineage = str(gt.get("lineage", "")).lower()
    lineage_terms = [w for w in _tokenize(lineage) if len(w) > 3]

    if subtype_score >= 1.0:
        # Subtype correct → broad_type is implied (any specific entity
        # belongs to exactly one broad lineage).  Infer the broad from
        # the subtype and give full credit.
        inferred = _infer_broad_from_entity(matched_alias or pred_entity or "", gt)
        broad_score, broad_status = 1.0, "implied_by_subtype"
    elif pred_broad == ref_broad:
        broad_score, broad_status = 1.0, "exact"
    elif _is_broad_hyponym(pred_broad, ref_broad):
        broad_score, broad_status = 0.75, "hyponym"
    elif lineage_terms and any(_contains_word(dx_norm, w) for w in lineage_terms):
        broad_score, broad_status = 0.75, "lineage_partial"
    elif _is_broad_hyponym(ref_broad, pred_broad):
        # Pred is a hypernym (parent) of ref → conservative but correct.
        # e.g. pred="neoplasm", ref="neuroendocrine neoplasm" → 0.5.
        broad_score, broad_status = 0.5, "hypernym"
    elif _are_broad_siblings(pred_broad, ref_broad):
        # Both are hyponyms of the same parent (both under "neoplasm").
        # e.g. pred="neuroendocrine neoplasm", ref="carcinoma" → 0.5.
        broad_score, broad_status = 0.5, "sibling_lineage"
    elif subtype_score >= 0.75:
        # Subtype partially correct → broad at least partially right
        broad_score, broad_status = 0.5, "subtype_partial"
    else:
        broad_score, broad_status = 0.0, "miss"

    # ── Malignancy tier (independent) ─────────────────────────────────────
    ref_mal = str(gt["malignancy"]).lower()
    pred_mal = _pred_mal(tier_text)
    if pred_mal == ref_mal:
        mal_score, mal_status = 1.0, "exact"
    elif {pred_mal, ref_mal} == {"borderline", "malignant"}:
        mal_score, mal_status = 0.5, "adjacent"
    elif pred_mal == "unspecified":
        mal_score, mal_status = 0.0, "missing"
    else:
        mal_score, mal_status = 0.0, "miss"

    return _tier_result(
        weights, anchored=True,
        broad=(broad_score, pred_broad, ref_broad, broad_status),
        subtype=(subtype_score,
                 matched_alias or (pred_entity or fact_subtype.lower()
                                   or canonical_diagnosis(dx_text) or "(none)"),
                 ref_subtype, subtype_status),
        malignancy=(mal_score, pred_mal, ref_mal, mal_status),
        grade=(grade_score, pred_grade, ref_grade, grade_status, grade_applicable),
    )


def _tier_result(
    weights: dict[str, float],
    *,
    anchored: bool,
    broad: tuple[float, str, str, str],
    subtype: tuple[float, str, str, str],
    malignancy: tuple[float, str, str, str],
    grade: tuple[float, str, str, str, bool],
) -> dict[str, Any]:
    """Assemble the tier result, renormalizing weights over applicable tiers.

    Each tier tuple is ``(score, prediction, reference, status)``; the grade
    tuple carries an extra ``applicable`` flag. Non-applicable tiers (grade with
    no reference ground truth) are dropped from the weighted average so the case
    is scored only on the axes that actually have a ground truth.
    """
    g_score, g_pred, g_ref, g_status, g_applicable = grade
    tiers = {
        "broad": {"score": broad[0], "prediction": broad[1], "reference": broad[2],
                  "status": broad[3], "weight": weights.get("broad", 0.0), "applicable": True},
        "subtype": {"score": subtype[0], "prediction": subtype[1], "reference": subtype[2],
                    "status": subtype[3], "weight": weights.get("subtype", 0.0), "applicable": True},
        "malignancy": {"score": malignancy[0], "prediction": malignancy[1], "reference": malignancy[2],
                       "status": malignancy[3], "weight": weights.get("malignancy", 0.0), "applicable": True},
        "grade": {"score": g_score, "prediction": g_pred, "reference": g_ref,
                  "status": g_status, "weight": weights.get("grade", 0.0), "applicable": g_applicable},
    }
    num = sum(t["weight"] * t["score"] for t in tiers.values() if t["applicable"])
    den = sum(t["weight"] for t in tiers.values() if t["applicable"]) or 1.0
    return {
        "diagnosis_score": num / den,
        "anchored": anchored,
        "tiers": tiers,
    }


# ── Calibration metrics ───────────────────────────────────────────────────────

def confidence_calibration(results: list[dict]) -> dict[str, Any]:
    """Compute calibration: is confidence correlated with accuracy?"""
    conf_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
    bins: dict[str, list[float]] = defaultdict(list)
    for r in results:
        conf = str(r.get("confidence", "low")).lower()
        if conf in conf_map:
            bins[conf].append(float(r.get("accuracy", 0)))

    calibration = {}
    for conf, accs in bins.items():
        calibration[conf] = {
            "expected": conf_map[conf],
            "actual": sum(accs) / len(accs) if accs else 0.0,
            "count": len(accs),
        }
    return {"calibration": calibration, "ece": _expected_calibration_error(bins, conf_map)}


# ── Cross-case consistency ────────────────────────────────────────────────────

def cross_case_consistency(results: list[dict], group_by: str = "cancer_type") -> dict[str, Any]:
    """Check if the same cancer type produces consistent descriptions."""
    groups: dict[str, list[str]] = defaultdict(list)
    for r in results:
        key = str(r.get(group_by, "unknown"))
        diag = str(r.get("primary_diagnosis", ""))
        if diag:
            groups[key].append(diag)

    consistency = {}
    for key, diagnoses in groups.items():
        if len(diagnoses) < 2:
            continue
        canon = [canonical_diagnosis(d) for d in diagnoses]
        canon = [c for c in canon if c]
        if not canon:
            continue
        most_common = Counter(canon).most_common(1)[0]
        consistency[key] = {
            "count": len(diagnoses),
            "most_common_diagnosis": most_common[0],
            "agreement_rate": most_common[1] / len(canon),
        }
    return {"by_group": consistency,
            "overall_agreement": sum(c["agreement_rate"] for c in consistency.values()) / max(1, len(consistency))}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_report_sections(text: str) -> dict[str, str]:
    sections = {}
    current = ""
    current_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        matched = False
        for sec in EXPECTED_SECTIONS:
            if re.match(re.escape(sec) + r"\s*:?", stripped, re.IGNORECASE):
                if current:
                    sections[current] = "\n".join(current_lines).strip()
                current = sec
                after = re.sub(r"^" + re.escape(sec) + r"\s*:?\s*", "", stripped, flags=re.IGNORECASE)
                current_lines = [after] if after else []
                matched = True
                break
        if not matched and current:
            current_lines.append(line)
    if current and current_lines:
        sections[current] = "\n".join(current_lines).strip()
    return sections


def _infer_cancer_type(diagnosis: str) -> str:
    d = diagnosis.lower()
    if "carcinoma" in d:
        return "carcinoma"
    if "sarcoma" in d:
        return "sarcoma"
    if "lymphoma" in d:
        return "lymphoma"
    if "melanoma" in d:
        return "melanoma"
    if "glioma" in d or "glioblastoma" in d:
        return "glioma"
    if "adenoma" in d:
        return "adenoma"
    return "neoplasm"


def _infer_malignancy(text: str) -> str:
    t = text.lower()
    if _malignancy_not_assessable(t):
        return "not assessable"
    if any(w in t for w in ["carcinoma", "sarcoma", "malignant", "metastatic",
                              "high grade", "poorly differentiated"]):
        return "malignant"
    if any(w in t for w in ["borderline", "uncertain malignant", "indeterminate"]):
        return "borderline"
    if any(w in t for w in ["benign", "adenoma"]):
        return "benign"
    return "not specified"


def _infer_invasive(text: str) -> str:
    t = text.lower()
    if _field_not_assessable(t, "invasion"):
        return "not assessable"
    if re.search(r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?\s*invasion\s*:?\s*(?:\*\*)?\s*:?\s*(?:absent|not identified|not seen|no clear evidence)\b", t, flags=re.IGNORECASE):
        return "not assessable"
    if any(w in t for w in ["invasion", "invasive", "infiltrative", "infiltrating"]):
        return "invasive"
    if any(w in t for w in ["in situ", "non-invasive", "noninvasive"]):
        return "in situ"
    return "not assessable"


def _primary_evaluable_text(sections: dict[str, str], fallback: str = "") -> str:
    """Text used for final-report evaluation.

    It intentionally excludes differential diagnoses and recommended stains.
    Those are useful clinical context, but they are not primary diagnostic
    claims and should not drive malignancy/subtype scoring.
    """
    names = [
        "Diagnosis",
        "Tumor/Lesion Description",
        "Key Microscopic Findings",
        "Invasion/Necrosis/Margins",
        "Stage/Grade/Prognostic Features",
        "Final Summary",
    ]
    parts = [sections.get(name, "") for name in names if sections.get(name)]
    text = "\n".join(parts) if parts else fallback
    cleaned = []
    for line in text.splitlines():
        low = line.lower()
        label = re.sub(r"^[\s\-\*\d\.\)\:]+", "", low).strip()
        label = label.replace("**", "").replace("__", "").strip()
        if label.startswith(("differential diagnosis", "differential diagnoses")):
            continue
        if "recommended stains" in low or "recommended ancillary" in low:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _field_not_assessable(text: str, field: str) -> bool:
    field_re = re.escape(field)
    return bool(re.search(
        rf"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?\s*{field_re}\s*:?\s*(?:\*\*)?\s*:?\s*not assessable\b",
        text,
        flags=re.IGNORECASE,
    ))


def _malignancy_not_assessable(text: str) -> bool:
    """Detect explicit uncertainty about malignant potential before keyword hits.

    Reports often say "benign versus malignant is not assessable" or mention a
    malignant differential in a limitation sentence. Those should not be parsed
    as a positive malignant claim.
    """
    t = text.lower()
    if _field_not_assessable(t, "malignancy") or _field_not_assessable(t, "malignant potential"):
        return True
    patterns = [
        r"\bmalignanc(?:y|ies)\s+(?:status\s+)?(?:is\s+)?(?:not\s+assessable|not\s+determined|indeterminate|uncertain|cannot\s+be\s+determined|can\s+not\s+be\s+determined)\b",
        r"\bmalignant\s+potential\s+(?:is\s+)?(?:not\s+assessable|not\s+determined|indeterminate|uncertain|cannot\s+be\s+determined|can\s+not\s+be\s+determined)\b",
        r"\b(?:benign|adenoma)\s+(?:versus|vs\.?|or)\s+(?:malignant|carcinoma)\b.{0,120}\b(?:not\s+assessable|cannot\s+be\s+determined|indeterminate|uncertain)\b",
        r"\b(?:malignant|carcinoma)\s+(?:versus|vs\.?|or)\s+(?:benign|adenoma)\b.{0,120}\b(?:not\s+assessable|cannot\s+be\s+determined|indeterminate|uncertain)\b",
        r"\bnot\s+possible\s+to\s+(?:definitively\s+)?(?:classify|determine)\b.{0,120}\b(?:benign|malignant|malignancy|malignant\s+potential)\b",
    ]
    return any(re.search(p, t, flags=re.IGNORECASE | re.DOTALL) for p in patterns)


def _lineage_neoplasm_partial(pred_norm: str, ref_subtype: str, gt: dict[str, Any]) -> bool:
    """Partial subtype credit when pred and ref share organ-lineage but differ in specificity.

    Handles cases like:
    - pred "adrenocortical carcinoma" vs ref "adrenal cortical neoplasm"
    - pred "lung adenocarcinoma" vs ref "pulmonary neoplasm"

    Uses loose substring matching (not strict word boundary) for organ-site
    overlap, since compound terms like "adrenocortical" won't match "adrenal"
    or "cortical" as separate words.
    """
    pred_lower = pred_norm.lower()
    ref_norm = " ".join(_tokenize(ref_subtype))
    ref_lower = ref_norm.lower()

    # Organ/site overlap — longest common substrings (>3 chars)
    pred_parts = set(re.findall(r"[a-z]{4,}", pred_lower))
    ref_parts = set(re.findall(r"[a-z]{4,}", ref_lower))
    site_overlap = bool(pred_parts & ref_parts)

    # Lineage words from GT (e.g. "adrenal cortical" → ["adrenal", "cortical"])
    lineage = str(gt.get("lineage", "")).lower()
    lineage_words = [w for w in _tokenize(lineage) if len(w) > 4 and w not in {"neoplasm", "tumor", "tumour"}]
    has_lineage = lineage_words and any(w in pred_lower for w in lineage_words)

    # Both are neoplasms
    pred_is_neoplasm = any(t in pred_lower for t in ("carcinoma", "neoplasm", "tumor", "tumour", "adenoma", "sarcoma", "lymphoma", "melanoma", "glioma"))
    ref_is_neoplasm = any(t in ref_lower for t in ("neoplasm", "tumor", "tumour", "carcinoma", "adenoma"))

    # Reference is broader/indeterminate
    ref_is_broader = any(t in ref_lower for t in
                         ("indeterminate", "uncertain malignant potential",
                          "not otherwise specified", "nos"))

    return (site_overlap or has_lineage) and pred_is_neoplasm and (ref_is_neoplasm or ref_is_broader)


def _infer_grade(text: str) -> str:
    t = text.lower()
    if "high grade" in t or "poorly differentiated" in t or "grade 3" in t or "grade iii" in t:
        return "high grade"
    if "low grade" in t or "well differentiated" in t or "grade 1" in t or "grade i" in t:
        return "low grade"
    if "moderately differentiated" in t or "grade 2" in t or "grade ii" in t:
        return "intermediate grade"
    return "not specified"


def _infer_necrosis(text: str) -> str:
    t = text.lower()
    if re.search(r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?\s*necrosis\s*:?\s*(?:\*\*)?\s*:?\s*(?:absent|not identified|not seen|no necrosis)\b", t, flags=re.IGNORECASE):
        return "absent"
    if any(w in t for w in ["necrosis", "necrotic", "comedo necrosis"]):
        return "present"
    if "no necrosis" in t or "necrosis absent" in t:
        return "absent"
    return "not specified"


def _infer_lvi(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["lymphovascular invasion", "vascular invasion", "angiolymphatic"]):
        return "present"
    if "no lymphovascular" in t:
        return "absent"
    return "not assessable"


def _compare_field(field: str, pred: str, ref: str) -> tuple[float, str]:
    """Compare a diagnostic field. Returns (score, status)."""
    ref_lower = ref.lower().strip()
    pred_lower = pred.lower().strip()

    # GT doesn't have this field → not applicable, weight should be zeroed
    if not ref or ref_lower in ("not specified", "unspecified"):
        return 0.0, "not_applicable"
    if not pred or pred_lower in ("not specified", "unspecified"):
        return 0.0, "missing"
    if ref_lower in ("not assessable",):
        # GT explicitly says "cannot assess from H&E" → match if pred agrees, otherwise not_applicable
        if pred_lower in ("not assessable", "not specified", "unspecified"):
            return 1.0, "exact_match"
        return 0.0, "not_applicable"
    if field == "histologic_subtype":
        return _compare_histologic_subtype(pred_lower, ref_lower)
    if field == "grade":
        return _compare_grade_field(pred_lower, ref_lower)
    if field == "invasive_status":
        return _compare_categorical_field(pred_lower, ref_lower, {
            "invasive": ["invasive", "present"],
            "in_situ": ["in situ", "in_situ", "carcinoma in situ", "non invasive", "non-invasive", "noninvasive"],
            "absent": ["absent", "not identified", "not seen", "no invasion"],
        })
    if field == "malignancy_status":
        return _compare_malignancy_field(pred_lower, ref_lower)
    if field in ("necrosis", "lymphovascular_invasion"):
        return _compare_categorical_field(pred_lower, ref_lower, {
            "present": ["present", "identified", "seen"],
            "absent": ["absent", "not identified", "not seen", "no"],
        })
    if pred_lower == ref_lower:
        return 1.0, "exact_match"
    # Canonical synonym match
    pred_canon = canonical_diagnosis(pred)
    ref_canon = canonical_diagnosis(ref)
    if pred_canon and ref_canon and pred_canon == ref_canon:
        return 1.0, "synonym_match"
    if field == "cancer_type" and _is_broad_hyponym(pred_canon or pred_lower, ref_canon or ref_lower):
        return 0.75, "hyponym"
    return 0.0, "mismatch"


def _compare_histologic_subtype(pred: str, ref: str) -> tuple[float, str]:
    """Strict subtype comparison; broad shared words like carcinoma give no credit."""
    if pred == ref:
        return 1.0, "exact_match"
    pred_canon = canonical_diagnosis(pred)
    ref_canon = canonical_diagnosis(ref)
    if pred_canon and ref_canon and pred_canon == ref_canon:
        return 1.0, "synonym_match"

    # Less-detailed version of the same entity, e.g. "urothelial carcinoma" vs
    # "high grade muscle invasive urothelial cell carcinoma + CIS".
    stop = {"high", "low", "grade", "invasive", "in", "situ", "cell", "with", "and", "cis"}
    pred_toks = {t for t in _tokenize(pred) if t not in stop}
    ref_toks = {t for t in _tokenize(ref) if t not in stop}
    pred_raw_toks = set(_tokenize(pred))
    ref_raw_toks = set(_tokenize(ref))
    invasive_conflict = (
        ("invasive" in pred_raw_toks and {"situ", "in_situ"} & ref_raw_toks)
        or ("invasive" in ref_raw_toks and {"situ", "in_situ"} & pred_raw_toks)
    )
    if invasive_conflict:
        return 0.0, "mismatch"
    diagnostic_toks = pred_toks - {"carcinoma", "neoplasm", "tumor", "tumour", "adenoma"}
    if diagnostic_toks and diagnostic_toks.issubset(ref_toks) and ("carcinoma" in pred_toks or "neoplasm" in pred_toks):
        return 0.75, "entity_contained"

    # Correct organ-lineage neoplasm but different specificity. This is partial,
    # not a synonym, and should not be triggered by "carcinoma" alone.
    organ_terms = {"adrenal", "adrenocortical", "cortical", "urothelial", "colorectal", "colon", "rectal",
                   "renal", "hepatocellular", "biliary", "gastric", "pulmonary", "lung", "breast",
                   "endometrial", "ovarian", "thyroid", "prostatic", "melanocytic"}
    if (pred_toks & ref_toks & organ_terms) and (
        {"neoplasm", "tumor", "tumour"} & (pred_toks | ref_toks)
    ):
        return 0.5, "lineage_neoplasm_partial"

    return 0.0, "mismatch"


def _compare_grade_field(pred: str, ref: str) -> tuple[float, str]:
    pred_grade = _infer_grade_tier(pred)
    ref_grade = _infer_grade_tier(ref)
    if ref_grade == "unspecified":
        return 0.0, "not_applicable"
    if pred_grade == "unspecified":
        return 0.0, "missing"
    if pred_grade == ref_grade:
        return 1.0, "exact_match"
    dist = abs(_GRADE_ORDINAL[pred_grade] - _GRADE_ORDINAL[ref_grade])
    if dist == 1:
        return 0.5, "adjacent"
    return 0.0, "mismatch"


def _normalise_category(value: str, mapping: dict[str, list[str]]) -> str:
    v = " ".join(_tokenize(value))
    for label, aliases in mapping.items():
        if any(_contains_word(v, a) for a in aliases):
            return label
    return value.strip().lower()


def _compare_categorical_field(pred: str, ref: str, mapping: dict[str, list[str]]) -> tuple[float, str]:
    pred_cat = _normalise_category(pred, mapping)
    ref_cat = _normalise_category(ref, mapping)
    if not ref_cat or ref_cat in ("not specified", "unspecified", "not assessable"):
        return 0.0, "not_applicable"
    if not pred_cat or pred_cat in ("not specified", "unspecified"):
        return 0.0, "missing"
    if pred_cat == ref_cat:
        return 1.0, "exact_match"
    return 0.0, "mismatch"


def _compare_malignancy_field(pred: str, ref: str) -> tuple[float, str]:
    pred_mal = _infer_malignancy_tier(pred)
    ref_mal = _infer_malignancy_tier(ref)
    if ref_mal == "unspecified":
        return 0.0, "not_applicable"
    if pred_mal == "unspecified":
        return 0.0, "missing"
    if pred_mal == ref_mal:
        return 1.0, "exact_match"
    if {pred_mal, ref_mal} == {"borderline", "malignant"}:
        return 0.5, "adjacent"
    return 0.0, "mismatch"


def _expected_calibration_error(
    bins: dict[str, list[float]], conf_map: dict[str, float]
) -> float:
    total = sum(len(accs) for accs in bins.values())
    if total == 0:
        return 0.0
    ece = 0.0
    for conf, accs in bins.items():
        if not accs:
            continue
        expected = conf_map[conf]
        actual = sum(accs) / len(accs)
        ece += len(accs) / total * abs(expected - actual)
    return ece
