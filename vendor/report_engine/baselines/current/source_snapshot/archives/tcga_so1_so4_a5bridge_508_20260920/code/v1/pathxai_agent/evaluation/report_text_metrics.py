from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List


METRIC_VERSION = "report_eval_v3_section_semantic"

EXPECTED_LONG_REPORT_SECTIONS = [
    "Specimen/Site",
    "Diagnosis",
    "Tumor/Lesion Description",
    "Key Microscopic Findings",
    "Invasion/Necrosis/Margins",
    "Immunohistochemistry/Ancillary Studies",
    "Stage/Grade/Prognostic Features",
    "Final Summary",
]

EMBEDDING_SECTION_MAP = {
    "Specimen/Site": "Specimen/Site",
    "Diagnosis": "Diagnosis",
    "Microscopic Findings": "Key Microscopic Findings",
    "Invasion/Necrosis/Margins": "Invasion/Necrosis/Margins",
    "IHC": "Immunohistochemistry/Ancillary Studies",
    "Stage/Grade": "Stage/Grade/Prognostic Features",
    "Final Summary": "Final Summary",
}

SITE_TERMS = [
    "adrenal", "adrenal cortical", "adrenocortical", "bladder", "breast", "cervix",
    "colon", "colorectal", "rectum", "esophagus", "stomach", "gastric", "head and neck",
    "kidney", "renal", "liver", "hepatic", "bile duct", "cholangiocarcinoma", "lung",
    "pleura", "brain", "glioma", "ovary", "uterus", "endometrium", "prostate", "skin",
    "melanoma", "sarcoma", "testis", "thyroid", "thymus", "uveal", "eye", "lymph node",
]

DIAGNOSIS_TERMS = [
    "carcinoma", "adenocarcinoma", "squamous cell carcinoma", "adrenocortical carcinoma",
    "adrenocortical adenoma", "hepatocellular carcinoma", "cholangiocarcinoma",
    "renal cell carcinoma", "clear cell carcinoma", "papillary carcinoma",
    "urothelial carcinoma", "ductal carcinoma", "lobular carcinoma", "melanoma",
    "glioblastoma", "astrocytoma", "oligodendroglioma", "sarcoma", "lymphoma",
    "seminoma", "germ cell tumor", "mesothelioma", "neoplasm", "tumor", "mass",
    "adenoma", "benign", "malignant", "metastatic", "metastasis",
]

BEHAVIOR_TERMS = [
    "benign", "borderline", "uncertain malignant potential", "malignant", "carcinoma",
    "sarcoma", "metastatic", "invasive", "in situ",
]

FIELD_PATTERNS = {
    "specimen_site": SITE_TERMS,
    "diagnosis": DIAGNOSIS_TERMS,
    "malignant_behavior": BEHAVIOR_TERMS,
    "necrosis": ["necrosis", "necrotic", "coagulative necrosis"],
    "invasion": ["invasion", "invasive", "vascular invasion", "capsular invasion", "lymphovascular invasion"],
    "margin": ["margin", "margins", "positive margin", "negative margin", "free margin"],
    "lymph_node": ["lymph node", "lymph nodes", "node positive", "node negative"],
    "grade": ["grade", "well differentiated", "moderately differentiated", "poorly differentiated", "high grade", "low grade"],
    "stage": ["stage", "pt", "pn", "pm", "tnm"],
    "size": ["cm", "mm", "size", "measures", "measuring"],
    "ihc": [
        "immunohistochemistry", "ihc", "positive", "negative", "ki-67", "p53", "ae1/ae3",
        "synaptophysin", "chromogranin", "inhibin", "melan-a", "sf-1", "s100", "ttf-1",
    ],
    "mitosis": ["mitosis", "mitotic", "mitoses"],
    "atypia": ["atypia", "atypical", "pleomorphism", "pleomorphic", "hyperchromasia"],
    "uncertainty_limitations": [
        "not specified", "not provided", "not assessable", "cannot be assessed",
        "limited", "uncertain", "suggests", "compatible with", "consistent with",
    ],
}

NEGATION_RE = re.compile(
    r"\b(no|not|without|absent|negative for|free of|lack of|cannot identify|not identified)\b",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9%./+-]+", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> List[str]:
    return normalize_text(text).split()


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def bleu(prediction: str, reference: str, max_n: int = 4) -> Dict[str, float]:
    """Sentence-level BLEU with add-1 (Laplace) smoothing for zero-count n-grams.

    Smoothing follows Chen & Cherry (2014) method 4 / SacreBLEU epsilon smoothing
    to avoid -inf log-prob when a higher-order n-gram has 0 matches.
    Brevity penalty is applied so short outputs are penalised.

    Standard reference: Papineni et al. (2002).
    Commonly reported in path/radiology report generation papers alongside ROUGE.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)} | {"bleu_avg": 0.0}

    brevity_penalty = min(1.0, math.exp(1.0 - len(ref_tokens) / max(len(pred_tokens), 1)))

    scores: Dict[str, float] = {}
    log_sum = 0.0
    for n in range(1, max_n + 1):
        pred_ng = Counter(_ngrams(pred_tokens, n))
        ref_ng = Counter(_ngrams(ref_tokens, n))
        # Clipped count: each n-gram only credited up to its ref frequency
        clipped = sum(min(cnt, ref_ng[gram]) for gram, cnt in pred_ng.items())
        total = max(len(pred_tokens) - n + 1, 1)
        # Add-1 smoothing so ln(0) never occurs
        precision = (clipped + 1) / (total + 1)
        scores[f"bleu_{n}"] = precision * brevity_penalty
        log_sum += math.log(precision)
    scores["bleu_avg"] = brevity_penalty * math.exp(log_sum / max_n)
    return scores


def rouge_n_score(prediction: str, reference: str, n: int = 1) -> Dict[str, float]:
    """ROUGE-N (unigram or bigram recall-oriented overlap).

    ROUGE-1 and ROUGE-2 are the most commonly reported variants.
    Reference: Lin (2004). Standard in summarisation and report-generation papers.
    """
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    pred_ng = Counter(_ngrams(pred_tokens, n))
    ref_ng = Counter(_ngrams(ref_tokens, n))
    overlap = sum((pred_ng & ref_ng).values())
    precision = overlap / max(1, sum(pred_ng.values()))
    recall = overlap / max(1, sum(ref_ng.values()))
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def token_f1(prediction: str, reference: str) -> Dict[str, float]:
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    f1 = _f1(precision, recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l_f1(prediction: str, reference: str) -> Dict[str, float]:
    pred = tokenize(prediction)
    ref = tokenize(reference)
    if not pred or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_len(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    f1 = _f1(precision, recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def keyword_recall(prediction: str, reference: str) -> Dict[str, object]:
    pred_terms = set(_content_terms(prediction))
    ref_terms = set(_content_terms(reference))
    if not ref_terms:
        return {"recall": 0.0, "matched": [], "missing": []}
    matched = sorted(pred_terms & ref_terms)
    missing = sorted(ref_terms - pred_terms)
    return {
        "recall": len(matched) / len(ref_terms),
        "matched": matched[:100],
        "missing": missing[:100],
    }


def report_similarity(prediction: str, reference: str, report_style: str = "long") -> Dict[str, object]:
    """Report-quality metrics focused on section-level semantic agreement.

    v3 intentionally keeps brittle keyword matching out of the main score.
    Keyword/term buckets are retained only under `legacy_keyword_debug`.
    """
    tf1 = token_f1(prediction, reference)
    rouge = rouge_l_f1(prediction, reference)
    kw = keyword_recall(prediction, reference)
    pred_len = max(1, len(tokenize(prediction)))
    ref_len = max(1, len(tokenize(reference)))
    length_ratio = pred_len / ref_len
    length_penalty = math.exp(-abs(math.log(max(length_ratio, 1e-6))))
    legacy_aggregate = (0.45 * tf1["f1"] + 0.35 * rouge["f1"] + 0.20 * float(kw["recall"]))
    legacy_aggregate *= length_penalty

    sections = section_metrics(prediction) if report_style == "long" else short_report_style_metrics(prediction)
    section_alignment_payload = section_alignment(prediction, reference, report_style)
    clinical = legacy_keyword_alignment(prediction, reference)
    numeric = numeric_alignment(prediction, reference)
    factual = conservative_factuality(prediction, reference, section_alignment_payload, numeric)
    completeness = report_completeness(prediction) if report_style == "long" else short_report_completeness(prediction)
    critical = semantic_critical_error_penalty(section_alignment_payload, clinical)
    if report_style == "short":
        raw_report_quality = (
            0.45 * section_alignment_payload["overall_semantic_score"]
            + 0.30 * section_alignment_payload["diagnosis_accuracy_proxy"]
            + 0.15 * factual["factual_consistency_score"]
            + 0.10 * sections["structure_score"]
        )
    else:
        raw_report_quality = (
            0.40 * section_alignment_payload["overall_semantic_score"]
            + 0.30 * section_alignment_payload["diagnosis_accuracy_proxy"]
            + 0.10 * factual["factual_consistency_score"]
            + 0.10 * sections["structure_score"]
            + 0.05 * completeness["completeness_score"]
            + 0.05 * numeric["f1"]
        )
    report_quality = raw_report_quality * critical["penalty_multiplier"]

    return {
        "metric_version": METRIC_VERSION,
        "report_style": report_style,
        "aggregate": section_alignment_payload["overall_semantic_score"],
        "report_quality_score": report_quality,
        "key_scores": {
            "report_quality_score": report_quality,
            "diagnosis_accuracy_proxy": section_alignment_payload["diagnosis_accuracy_proxy"],
            "diagnosis_section_similarity": section_alignment_payload["diagnosis_section_similarity"],
            "site_section_similarity": section_alignment_payload["site_section_similarity"],
            "overall_semantic_score": section_alignment_payload["overall_semantic_score"],
            "factual_consistency_score": factual["factual_consistency_score"],
            "structure_score": sections["structure_score"],
            "critical_penalty_multiplier": critical["penalty_multiplier"],
        },
        "prediction_tokens": pred_len,
        "reference_tokens": ref_len,
        "length_ratio": length_ratio,
        "length_penalty": length_penalty,
        "structure": sections,
        "section_alignment": section_alignment_payload,
        "numeric_alignment": numeric,
        "factual_precision": factual,
        "completeness": completeness,
        "critical_error_penalty": critical,
        "legacy_lexical_debug": {
            "legacy_aggregate": legacy_aggregate,
            "token_f1": tf1,
            "rouge_l": rouge,
            "keyword_recall": {
                "recall": kw["recall"],
                "matched_count": len(kw["matched"]),
                "missing_count": len(kw["missing"]),
            },
        },
        "legacy_keyword_debug": clinical,
        # ── Standard NLP metrics ──
        # These are the metrics conventionally reported in text generation papers
        # (Papineni 2002, Lin 2004). Added to allow direct comparison with published work.
        "standard_nlp": {
            **bleu(prediction, reference, max_n=4),
            "rouge_1": rouge_n_score(prediction, reference, n=1)["f1"],
            "rouge_2": rouge_n_score(prediction, reference, n=2)["f1"],
            "rouge_l": rouge["f1"],
            "rouge_1_detail": rouge_n_score(prediction, reference, n=1),
            "rouge_2_detail": rouge_n_score(prediction, reference, n=2),
        },
    }


def report_embedding_similarity(
    long_prediction: str,
    long_reference: str | None,
    short_prediction: str | None,
    short_reference: str | None,
    embedding_client,
) -> Dict[str, object]:
    """Embedding similarity with Qwen/OpenAI-compatible embedding clients."""
    tasks = []
    section_scores = {}
    if long_reference:
        pred_sections = extract_sections(long_prediction)
        ref_sections = extract_sections(long_reference)
        for public_name, section_name in EMBEDDING_SECTION_MAP.items():
            pred_text = pred_sections.get(section_name, "")
            ref_text = ref_sections.get(section_name, "")
            if pred_text.strip() or ref_text.strip():
                tasks.append(("long_section", public_name, pred_text, ref_text))
    if short_prediction is not None and short_reference is not None:
        tasks.append(("short_report", "whole_short_report", short_prediction, short_reference))
    if not tasks:
        return {
            "enabled": True,
            "available": False,
            "reason": "no embedding comparison targets",
        }

    embed_inputs = []
    for _kind, _name, pred_text, ref_text in tasks:
        embed_inputs.extend([pred_text or "", ref_text or ""])
    result = embedding_client.embed(embed_inputs)
    embeddings = result["embeddings"]
    if len(embeddings) != len(embed_inputs):
        raise ValueError(f"Embedding endpoint returned {len(embeddings)} embeddings for {len(embed_inputs)} inputs")

    scores = {}
    for idx, (kind, name, pred_text, ref_text) in enumerate(tasks):
        pred_emb = embeddings[2 * idx]
        ref_emb = embeddings[2 * idx + 1]
        score = cosine_similarity(pred_emb, ref_emb)
        payload = {
            "cosine": score,
            "prediction_chars": len(pred_text or ""),
            "reference_chars": len(ref_text or ""),
        }
        if kind == "long_section":
            section_scores[name] = payload
        else:
            scores[name] = payload

    section_values = [payload["cosine"] for payload in section_scores.values()]
    scores["long_report_section_mean"] = sum(section_values) / len(section_values) if section_values else None
    return {
        "enabled": True,
        "available": True,
        "model": result.get("raw_model"),
        "usage": result.get("usage"),
        "elapsed_sec": result.get("elapsed_sec"),
        "long_report": {
            "section_mean_cosine": scores["long_report_section_mean"],
            "section_scores": section_scores,
        },
        "short_report": {
            "available": short_prediction is not None and short_reference is not None,
            "whole_short_report_cosine": (scores.get("whole_short_report") or {}).get("cosine"),
        },
    }


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    av = [float(x) for x in a]
    bv = [float(x) for x in b]
    if not av or not bv or len(av) != len(bv):
        return 0.0
    dot = sum(x * y for x, y in zip(av, bv))
    an = math.sqrt(sum(x * x for x in av))
    bn = math.sqrt(sum(y * y for y in bv))
    if an == 0 or bn == 0:
        return 0.0
    return dot / (an * bn)


def compact_metric_summary(metric: dict) -> Dict[str, object]:
    section_alignment_payload = metric.get("section_alignment") or {}
    section_scores = section_alignment_payload.get("section_scores") or {}
    structure = metric.get("structure") or {}
    factual = metric.get("factual_precision") or {}
    critical = metric.get("critical_error_penalty") or {}
    legacy = metric.get("legacy_lexical_debug") or {}
    return {
        "metric_version": metric.get("metric_version"),
        "report_quality_score": metric.get("report_quality_score"),
        "diagnosis_accuracy_proxy": section_alignment_payload.get("diagnosis_accuracy_proxy"),
        "diagnosis_section_similarity": section_alignment_payload.get("diagnosis_section_similarity"),
        "site_section_similarity": section_alignment_payload.get("site_section_similarity"),
        "overall_semantic_score": section_alignment_payload.get("overall_semantic_score"),
        "section_similarities": {
            name: payload.get("semantic_similarity")
            for name, payload in section_scores.items()
        },
        "factual_consistency_score": factual.get("factual_consistency_score"),
        "structure_score": structure.get("structure_score"),
        # Standard NLP metrics for publication-ready comparison
        "bleu_1": (metric.get("standard_nlp") or {}).get("bleu_1"),
        "bleu_2": (metric.get("standard_nlp") or {}).get("bleu_2"),
        "bleu_4": (metric.get("standard_nlp") or {}).get("bleu_4"),
        "rouge_1": (metric.get("standard_nlp") or {}).get("rouge_1"),
        "rouge_2": (metric.get("standard_nlp") or {}).get("rouge_2"),
        "rouge_l": (metric.get("standard_nlp") or {}).get("rouge_l"),
        "legacy_token_f1": ((legacy.get("token_f1") or {}).get("f1")),
        "legacy_rouge_l": ((legacy.get("rouge_l") or {}).get("f1")),
        "length_ratio": metric.get("length_ratio"),
        "critical_penalty_multiplier": critical.get("penalty_multiplier"),
        "critical_error_reasons": critical.get("reasons", []),
    }


def section_metrics(text: str) -> Dict[str, object]:
    sections = extract_sections(text)
    found = [section for section in EXPECTED_LONG_REPORT_SECTIONS if section in sections]
    non_empty = [section for section in found if _meaningful_section_text(sections[section])]
    order_score = _section_order_score(text)
    label_score = len(found) / len(EXPECTED_LONG_REPORT_SECTIONS)
    non_empty_score = len(non_empty) / len(EXPECTED_LONG_REPORT_SECTIONS)
    structure_score = 0.45 * label_score + 0.45 * non_empty_score + 0.10 * order_score
    return {
        "expected_sections": EXPECTED_LONG_REPORT_SECTIONS,
        "found_sections": found,
        "missing_sections": [s for s in EXPECTED_LONG_REPORT_SECTIONS if s not in sections],
        "empty_sections": [s for s in found if not _meaningful_section_text(sections[s])],
        "label_score": label_score,
        "non_empty_score": non_empty_score,
        "order_score": order_score,
        "structure_score": structure_score,
    }


def short_report_style_metrics(text: str) -> Dict[str, object]:
    tokens = tokenize(text)
    has_long_section_headers = any(
        re.search(rf"(?im)^\s*{re.escape(section)}\s*:", text)
        for section in EXPECTED_LONG_REPORT_SECTIONS
    )
    sentence_count = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
    length_ok = 15 <= len(tokens) <= 120
    sentence_ok = 1 <= sentence_count <= 4
    no_long_headers_score = 0.0 if has_long_section_headers else 1.0
    length_score = 1.0 if length_ok else math.exp(-abs(math.log(max(len(tokens), 1) / 50.0)))
    sentence_score = 1.0 if sentence_ok else 0.5
    structure_score = 0.50 * no_long_headers_score + 0.30 * length_score + 0.20 * sentence_score
    return {
        "style": "short_annotation_paragraph",
        "has_long_section_headers": has_long_section_headers,
        "token_count": len(tokens),
        "sentence_count": sentence_count,
        "no_long_headers_score": no_long_headers_score,
        "length_score": length_score,
        "sentence_score": sentence_score,
        "structure_score": structure_score,
    }


def section_alignment(prediction: str, reference: str, report_style: str = "long") -> Dict[str, object]:
    if report_style == "long":
        pred_sections = extract_sections(prediction)
        ref_sections = extract_sections(reference)
        section_scores = {}
        for section in EXPECTED_LONG_REPORT_SECTIONS:
            pred_text = pred_sections.get(section, "")
            ref_text = ref_sections.get(section, "")
            section_scores[section] = text_semantic_similarity(pred_text, ref_text)
        diagnosis_similarity = section_scores.get("Diagnosis", {}).get("semantic_similarity", 0.0)
        site_similarity = section_scores.get("Specimen/Site", {}).get("semantic_similarity", 0.0)
        microscopic = section_scores.get("Key Microscopic Findings", {}).get("semantic_similarity", 0.0)
        final_summary = section_scores.get("Final Summary", {}).get("semantic_similarity", 0.0)
        available = [
            payload["semantic_similarity"]
            for payload in section_scores.values()
            if payload["reference_chars"] > 0 or payload["prediction_chars"] > 0
        ]
        overall = sum(available) / len(available) if available else 0.0
        diagnosis_proxy = (
            0.55 * diagnosis_similarity
            + 0.25 * site_similarity
            + 0.10 * microscopic
            + 0.10 * final_summary
        )
        return {
            "mode": "long_section_alignment",
            "diagnosis_accuracy_proxy": diagnosis_proxy,
            "diagnosis_section_similarity": diagnosis_similarity,
            "site_section_similarity": site_similarity,
            "overall_semantic_score": overall,
            "section_scores": section_scores,
        }
    whole = text_semantic_similarity(prediction, reference)
    return {
        "mode": "short_whole_report_alignment",
        "diagnosis_accuracy_proxy": whole["semantic_similarity"],
        "diagnosis_section_similarity": whole["semantic_similarity"],
        "site_section_similarity": None,
        "overall_semantic_score": whole["semantic_similarity"],
        "section_scores": {
            "whole_short_report": whole,
        },
    }


def text_semantic_similarity(prediction: str, reference: str) -> Dict[str, object]:
    pred_norm = normalize_text(prediction or "")
    ref_norm = normalize_text(reference or "")
    token_set = jaccard_similarity(set(tokenize(pred_norm)), set(tokenize(ref_norm)))
    char_ngrams = jaccard_similarity(char_ngrams_for_text(pred_norm), char_ngrams_for_text(ref_norm))
    containment = token_containment(pred_norm, ref_norm)
    lcs = rouge_l_f1(pred_norm, ref_norm)["f1"] if pred_norm and ref_norm else 0.0
    if not pred_norm and not ref_norm:
        semantic = 1.0
    elif not pred_norm or not ref_norm:
        semantic = 0.0
    else:
        semantic = 0.35 * token_set + 0.30 * char_ngrams + 0.20 * containment + 0.15 * lcs
    return {
        "semantic_similarity": semantic,
        "token_set_jaccard": token_set,
        "char_4gram_jaccard": char_ngrams,
        "token_containment": containment,
        "rouge_l_f1": lcs,
        "prediction_text": prediction or "",
        "reference_text": reference or "",
        "prediction_chars": len(prediction or ""),
        "reference_chars": len(reference or ""),
    }


def jaccard_similarity(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def char_ngrams_for_text(text: str, n: int = 4) -> set[str]:
    compact = re.sub(r"\s+", " ", text.strip())
    if not compact:
        return set()
    if len(compact) <= n:
        return {compact}
    return {compact[i:i + n] for i in range(len(compact) - n + 1)}


def token_containment(prediction: str, reference: str) -> float:
    pred = set(tokenize(prediction))
    ref = set(tokenize(reference))
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    return len(pred & ref) / len(ref)


def legacy_keyword_alignment(prediction: str, reference: str) -> Dict[str, object]:
    pred_fields = extract_clinical_fields(prediction)
    ref_fields = extract_clinical_fields(reference)
    category_scores = {}
    weighted = []
    for category in FIELD_PATTERNS:
        pred_terms = set(pred_fields[category]["terms"])
        ref_terms = set(ref_fields[category]["terms"])
        score = _set_prf(pred_terms, ref_terms)
        status_match = assertion_status(prediction, FIELD_PATTERNS[category]) == assertion_status(reference, FIELD_PATTERNS[category])
        if ref_terms or pred_terms:
            category_scores[category] = {
                **score,
                "prediction_terms": sorted(pred_terms),
                "reference_terms": sorted(ref_terms),
                "unsupported_terms": sorted(pred_terms - ref_terms),
                "missing_terms": sorted(ref_terms - pred_terms),
                "assertion_status_match": status_match,
            }
        if ref_terms:
            weight = 2.0 if category in {"diagnosis", "specimen_site", "malignant_behavior"} else 1.0
            weighted.append((weight, score["f1"]))
    clinical_f1 = _weighted_mean(weighted)
    diagnosis_site_score = _weighted_mean([
        (2.0, category_scores.get("diagnosis", {}).get("f1", 0.0)),
        (1.5, category_scores.get("specimen_site", {}).get("f1", 0.0)),
        (1.0, category_scores.get("malignant_behavior", {}).get("f1", 0.0)),
    ])
    overclaims = clinical_overclaims(category_scores)
    return {
        "clinical_f1": clinical_f1,
        "diagnosis_site_score": diagnosis_site_score,
        "category_scores": category_scores,
        "overclaims": overclaims,
    }


def extract_clinical_fields(text: str) -> Dict[str, dict]:
    return {
        category: {
            "terms": sorted(_matched_phrases(text, patterns)),
            "assertion_status": assertion_status(text, patterns),
        }
        for category, patterns in FIELD_PATTERNS.items()
    }


def numeric_alignment(prediction: str, reference: str) -> Dict[str, object]:
    pred_numbers = extract_numbers(prediction)
    ref_numbers = extract_numbers(reference)
    score = _set_prf(set(pred_numbers), set(ref_numbers))
    return {
        **score,
        "prediction_numbers": pred_numbers,
        "reference_numbers": ref_numbers,
        "unsupported_numbers": sorted(set(pred_numbers) - set(ref_numbers)),
        "missing_numbers": sorted(set(ref_numbers) - set(pred_numbers)),
    }


def conservative_factuality(prediction: str, reference: str, sections: dict, numeric: dict) -> Dict[str, object]:
    section_scores = sections.get("section_scores") or {}
    values = []
    for name, payload in section_scores.items():
        if payload.get("prediction_chars", 0) > 0:
            values.append(float(payload.get("semantic_similarity", 0.0)))
    if numeric.get("prediction_numbers") or numeric.get("reference_numbers"):
        values.append(float(numeric.get("precision", 0.0)))
    if not values:
        score = 0.0
    else:
        score = sum(values) / len(values)
    return {
        "factual_consistency_score": score,
        "basis": "section semantic similarity plus numeric precision",
        "numeric_precision": numeric.get("precision"),
        "low_similarity_sections": [
            name
            for name, payload in section_scores.items()
            if payload.get("reference_chars", 0) > 0
            and payload.get("prediction_chars", 0) > 0
            and payload.get("semantic_similarity", 0.0) < 0.20
        ],
    }


def factual_precision(prediction: str, reference: str, clinical: dict, numeric: dict) -> Dict[str, object]:
    category_scores = clinical.get("category_scores") or {}
    precisions = []
    unsupported = {}
    for category, score in category_scores.items():
        if score.get("prediction_terms"):
            precisions.append(float(score.get("precision", 0.0)))
        if score.get("unsupported_terms"):
            unsupported[category] = score["unsupported_terms"]
    if numeric.get("prediction_numbers"):
        precisions.append(float(numeric.get("precision", 0.0)))
    unsupported_content = unsupported_content_terms(prediction, reference)
    content_precision = 1.0 - min(1.0, len(unsupported_content) / max(1, len(set(_content_terms(prediction)))))
    precisions.append(content_precision)
    return {
        "factual_precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "unsupported_clinical_terms": unsupported,
        "unsupported_content_terms": unsupported_content[:100],
        "unsupported_numeric_values": numeric.get("unsupported_numbers", []),
    }


def semantic_critical_error_penalty(sections: dict, legacy_keyword_debug: dict | None = None) -> Dict[str, object]:
    reasons = []
    multiplier = 1.0
    diagnosis_score = float(sections.get("diagnosis_section_similarity") or 0.0)
    site_score_raw = sections.get("site_section_similarity")
    site_score = None if site_score_raw is None else float(site_score_raw)
    if site_score is not None and site_score < 0.15:
        multiplier *= 0.50
        reasons.append("low_specimen_site_section_similarity")
    if diagnosis_score < 0.18:
        multiplier *= 0.55
        reasons.append("low_diagnosis_section_similarity")
    elif diagnosis_score < 0.35:
        multiplier *= 0.80
        reasons.append("partial_diagnosis_section_similarity")
    return {
        "penalty_multiplier": multiplier,
        "reasons": reasons,
        "diagnosis_section_similarity": diagnosis_score,
        "site_section_similarity": site_score,
        "basis": "section semantic similarity thresholds",
    }


def critical_error_penalty(clinical: dict) -> Dict[str, object]:
    category_scores = clinical.get("category_scores") or {}
    reasons = []
    multiplier = 1.0

    site = category_scores.get("specimen_site") or {}
    if site.get("prediction_terms") and site.get("reference_terms") and site.get("f1", 0.0) == 0.0:
        multiplier *= 0.45
        reasons.append("wrong_or_unsupported_specimen_site")

    diagnosis = category_scores.get("diagnosis") or {}
    unsupported_diag = set(diagnosis.get("unsupported_terms") or [])
    missing_diag = set(diagnosis.get("missing_terms") or [])
    if unsupported_diag and missing_diag:
        multiplier *= 0.70
        reasons.append("unsupported_or_missing_diagnosis_terms")
    if any(term for term in unsupported_diag if "carcinoma" in term or "sarcoma" in term or "melanoma" in term):
        multiplier *= 0.75
        reasons.append("unsupported_specific_malignant_diagnosis")

    behavior = category_scores.get("malignant_behavior") or {}
    if behavior.get("prediction_terms") and behavior.get("reference_terms") and behavior.get("f1", 0.0) == 0.0:
        multiplier *= 0.75
        reasons.append("wrong_malignant_behavior")

    return {
        "penalty_multiplier": multiplier,
        "reasons": reasons,
    }


def report_completeness(text: str) -> Dict[str, object]:
    sections = extract_sections(text)
    present = [section for section in EXPECTED_LONG_REPORT_SECTIONS if _meaningful_section_text(sections.get(section, ""))]
    missing = [section for section in EXPECTED_LONG_REPORT_SECTIONS if section not in present]
    completeness_score = len(present) / len(EXPECTED_LONG_REPORT_SECTIONS)
    return {
        "basis": "expected long-report section coverage",
        "present_sections": present,
        "missing_or_empty_sections": missing,
        "core_category_score": completeness_score,
        "completeness_score": completeness_score,
    }


def short_report_completeness(text: str) -> Dict[str, object]:
    tokens = tokenize(text)
    sentence_count = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
    enough_content = len(tokens) >= 15
    concise = len(tokens) <= 120
    sentence_ok = 1 <= sentence_count <= 4
    score = (float(enough_content) + float(concise) + float(sentence_ok)) / 3.0
    return {
        "basis": "short-report length and sentence-count coverage",
        "token_count": len(tokens),
        "sentence_count": sentence_count,
        "enough_content": enough_content,
        "concise": concise,
        "sentence_count_ok": sentence_ok,
        "core_category_score": score,
        "completeness_score": score,
    }


def extract_sections(text: str) -> Dict[str, str]:
    escaped = "|".join(re.escape(s) for s in EXPECTED_LONG_REPORT_SECTIONS)
    pattern = re.compile(rf"(?im)^\s*({escaped})\s*:\s*")
    matches = list(pattern.finditer(text))
    sections = {}
    for idx, match in enumerate(matches):
        label = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections[label] = text[start:end].strip()
    return sections


def extract_numbers(text: str) -> List[str]:
    norm = text.lower()
    values = set()
    for match in re.finditer(r"\b\d+(?:\.\d+)?\s*(?:cm|mm|%)\b", norm):
        values.add(re.sub(r"\s+", "", match.group(0)))
    for match in re.finditer(r"\b(?:pt|pn|pm|stage)\s*[0-9ivx]+[a-z]?\b", norm):
        values.add(re.sub(r"\s+", "", match.group(0)))
    return sorted(values)


def assertion_status(text: str, patterns: Iterable[str]) -> str:
    norm = normalize_text(text)
    for phrase in patterns:
        phrase_norm = normalize_text(phrase)
        idx = norm.find(phrase_norm)
        if idx < 0:
            continue
        window = norm[max(0, idx - 80): idx + len(phrase_norm) + 30]
        if NEGATION_RE.search(window):
            return "negated"
        return "affirmed"
    return "not_mentioned"


def clinical_overclaims(category_scores: dict) -> Dict[str, object]:
    critical = {}
    for category in ["specimen_site", "diagnosis", "malignant_behavior", "invasion", "necrosis", "margin", "grade", "stage"]:
        score = category_scores.get(category) or {}
        unsupported = score.get("unsupported_terms") or []
        if unsupported:
            critical[category] = unsupported
    return {
        "critical_unsupported_categories": critical,
        "count": sum(len(v) for v in critical.values()),
    }


def unsupported_content_terms(prediction: str, reference: str) -> List[str]:
    pred_terms = set(_content_terms(prediction))
    ref_terms = set(_content_terms(reference))
    return sorted(pred_terms - ref_terms)


def _matched_phrases(text: str, phrases: Iterable[str]) -> List[str]:
    norm = normalize_text(text)
    matched = []
    for phrase in phrases:
        phrase_norm = normalize_text(phrase)
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase_norm)}(?![a-z0-9])", norm):
            matched.append(phrase)
    return matched


def _content_terms(text: str) -> List[str]:
    stop = {
        "the", "and", "with", "for", "from", "that", "this", "are", "was", "were",
        "has", "have", "had", "into", "not", "but", "can", "may", "show", "shows",
        "showing", "present", "seen", "section", "sections", "tumor", "tumour",
        "specimen", "report", "pathology", "diagnosis", "provided", "current",
        "context", "evidence", "patch", "patches", "field", "fields",
    }
    return [tok for tok in tokenize(text) if len(tok) > 3 and tok not in stop]


def _lcs_len(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _set_prf(pred: set, ref: set) -> Dict[str, float]:
    if not pred and not ref:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred or not ref:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    overlap = len(pred & ref)
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _weighted_mean(values: List[tuple[float, float]]) -> float:
    total_weight = sum(weight for weight, _value in values)
    if total_weight <= 0:
        return 0.0
    return sum(weight * value for weight, value in values) / total_weight


def _section_order_score(text: str) -> float:
    positions = []
    for section in EXPECTED_LONG_REPORT_SECTIONS:
        match = re.search(rf"(?im)^\s*{re.escape(section)}\s*:", text)
        if match:
            positions.append(match.start())
    if not positions:
        return 0.0
    return 1.0 if positions == sorted(positions) else 0.0


def _meaningful_section_text(text: str) -> bool:
    stripped = normalize_text(text)
    return bool(stripped) and stripped not in {"not specified", "none", "n/a", "na"}
