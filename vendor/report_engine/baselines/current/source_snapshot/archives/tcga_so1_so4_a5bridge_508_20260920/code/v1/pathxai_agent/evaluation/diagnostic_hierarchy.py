from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pathxai_agent.datasets.tcga_sites import known_site_for_tcga
from pathxai_agent.evaluation.report_text_metrics import extract_sections, text_semantic_similarity
from pathxai_agent.llm.qwen_client import QwenVisionClient
from pathxai_agent.storage import SHARED_ROOT


FACT_SCHEMA_KEYS = [
    "site",
    "primary_diagnosis",
    "cancer_type",
    "histologic_subtype",
    "malignancy_status",
    "invasive_status",
    "grade",
    "necrosis",
    "lymphovascular_invasion",
    "margin_status",
    "stage",
    "ihc_molecular",
    "image_assessable_fields",
    "nonimage_only_fields",
    "not_assessable_fields",
]

DEFAULT_SYNONYM_PATH = SHARED_ROOT / "synonyms" / "diagnostic_synonyms.json"


DEFAULT_SYNONYMS = {
    "site": {
        "adrenal": [
            "adrenal gland",
            "adrenocortical tissue",
            "adrenal cortical tissue",
            "left adrenal gland",
            "right adrenal gland",
            "adrenal mass",
            "suprarenal",
        ],
        "breast": ["breast tissue", "right breast", "left breast", "mammary tissue"],
        "bladder": ["bladder", "urothelial tissue", "urinary bladder", "bladder wall", "urothelium"],
        "lung": ["lung", "pulmonary tissue"],
        "kidney": ["kidney", "renal tissue", "renal parenchyma", "renal mass"],
        "liver": ["liver", "hepatic tissue"],
        "colon": ["colon", "colorectal", "large bowel", "colonic mucosa"],
        "stomach": ["stomach", "gastric mucosa"],
        "cervix": ["cervix", "cervical tissue", "uterine cervix", "endocervix", "ectocervix"],
        "bile duct": ["bile duct", "biliary", "cholangiocarcinoma", "intrahepatic bile duct"],
        "esophagus": ["esophagus", "oesophagus", "esophageal", "oesophageal"],
        "head and neck": ["head and neck", "oral cavity", "oropharynx", "larynx", "hypopharynx", "squamous mucosa"],
        "lymphoid": ["lymphoid tissue", "lymph node", "lymphatic tissue"],
        "brain": ["brain", "glial tissue", "cerebral", "central nervous system", "cns"],
        "eye": ["eye", "uveal", "choroid", "ocular", "uveal melanocytic tissue"],
        "thyroid": ["thyroid"],
        "prostate": ["prostate"],
        "uterus": ["uterus", "endometrium", "endometrial"],
        "ovary": ["ovary", "ovarian"],
    },
    "cancer_type": {
        "carcinoma": ["carcinoma", "adenocarcinoma", "urothelial carcinoma", "ductal carcinoma"],
        "neoplasm": ["neoplasm", "tumor", "tumour"],
        "lymphoma": ["lymphoma"],
        "melanoma": ["melanoma"],
        "sarcoma": ["sarcoma"],
        "adenoma": ["adenoma"],
    },
    "histologic_subtype": {
        "adrenocortical carcinoma": ["adrenocortical carcinoma", "adrenal cortical carcinoma"],
        "adrenocortical neoplasm": [
            "adrenocortical neoplasm",
            "adrenal cortical neoplasm",
            "cortical adrenal tumor",
            "adrenal cortical neoplasm of indeterminate malignant potential",
            "adrenocortical neoplasm of uncertain malignant potential",
        ],
        "urothelial carcinoma": ["urothelial carcinoma", "transitional cell carcinoma"],
        "cholangiocarcinoma": ["cholangiocarcinoma", "bile duct adenocarcinoma", "biliary adenocarcinoma"],
        "colorectal adenocarcinoma": ["colorectal adenocarcinoma", "colon adenocarcinoma", "colonic adenocarcinoma"],
        "squamous cell carcinoma": ["squamous cell carcinoma", "scc", "keratinizing squamous carcinoma", "non keratinizing squamous carcinoma"],
        "diffuse large b-cell lymphoma": ["diffuse large b cell lymphoma", "dlbcl", "large b cell lymphoma"],
        "clear cell renal cell carcinoma": ["clear cell renal cell carcinoma", "clear cell rcc", "renal cell carcinoma clear cell type"],
        "papillary renal cell carcinoma": ["papillary renal cell carcinoma", "papillary rcc"],
        "chromophobe renal cell carcinoma": ["chromophobe renal cell carcinoma", "chromophobe rcc"],
        "glioma": ["diffuse glioma", "glioma", "astrocytoma", "oligodendroglioma", "oligoastrocytoma"],
        "small cell carcinoma": ["small cell carcinoma", "neuroendocrine carcinoma"],
        "invasive ductal carcinoma": ["invasive ductal carcinoma", "invasive breast carcinoma", "ductal carcinoma, invasive", "carcinoma of no special type", "nst"],
        "ductal carcinoma in situ": ["ductal carcinoma in situ", "dcis", "intraductal carcinoma"],
        "invasive lobular carcinoma": ["invasive lobular carcinoma", "lobular carcinoma"],
    },
    "malignancy_status": {
        "malignant": ["malignant", "carcinoma", "sarcoma", "lymphoma", "melanoma", "poorly differentiated"],
        "borderline": ["borderline", "uncertain malignant potential", "indeterminate malignant potential", "uncertain biological behavior"],
        "benign": ["benign", "adenoma", "hyperplasia"],
    },
    "invasive_status": {
        "invasive": ["invasive", "invasion present", "infiltrative", "invading", "stromal invasion"],
        "in_situ": ["in situ", "dcis", "carcinoma in situ", "non-invasive"],
        "not_assessable": ["not assessable", "cannot be assessed", "not determined"],
    },
    "necrosis": {
        "present": ["necrosis present", "necrosis is present", "comedo necrosis", "geographic necrosis", "dead tissue"],
        "absent": ["necrosis absent", "no necrosis", "not identified"],
        "not_assessable": ["not assessable", "cannot be assessed"],
    },
}


FIELD_WEIGHTS = {
    "site": 0.15,
    "cancer_type": 0.20,
    "histologic_subtype": 0.25,
    "malignancy_status": 0.15,
    "invasive_status": 0.15,
    "grade": 0.05,
    "necrosis": 0.05,
}

NONIMAGE_FIELDS = {"margin_status", "stage", "ihc_molecular"}


def ensure_default_synonym_map(path: str | Path = DEFAULT_SYNONYM_PATH) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_SYNONYMS, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_synonym_map(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ensure_default_synonym_map()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _merge_synonym_maps(DEFAULT_SYNONYMS, data) if isinstance(data, dict) else DEFAULT_SYNONYMS
    except Exception:
        return DEFAULT_SYNONYMS


def _merge_synonym_maps(defaults: dict, loaded: dict) -> dict:
    merged = json.loads(json.dumps(defaults))
    for field, groups in (loaded or {}).items():
        if not isinstance(groups, dict):
            continue
        merged.setdefault(field, {})
        for canonical, terms in groups.items():
            if not isinstance(terms, list):
                continue
            existing = list(merged[field].get(canonical, []))
            seen = {_norm(x) for x in existing}
            for term in terms:
                if _norm(term) not in seen:
                    existing.append(term)
                    seen.add(_norm(term))
            merged[field][canonical] = existing
    return merged


def extract_diagnostic_facts(
    report_text: str,
    client: QwenVisionClient | None = None,
    source: str = "prediction",
    dry_run: bool = False,
) -> dict:
    if client is not None and not dry_run:
        try:
            return _extract_facts_llm(report_text, client=client, source=source)
        except Exception as exc:
            facts = _extract_facts_rule(report_text, source=source)
            facts["llm_error"] = str(exc)
            return facts
    return _extract_facts_rule(report_text, source=source)


def _extract_facts_llm(report_text: str, client: QwenVisionClient, source: str) -> dict:
    prompt = f"""/no_think
Extract diagnostic facts from this pathology report text.

Rules:
- Only extract what is explicitly stated in the report.
- Do not infer missing IHC, stage, margin, molecular, lymph-node, or gross-size results.
- If a field is absent, use "not specified".
- If a field is explicitly not visible/not assessable from images, use "not assessable".
- cancer_type is broad lineage/type, e.g. carcinoma, lymphoma, melanoma, sarcoma, adenoma, neoplasm.
- histologic_subtype is specific diagnosis, e.g. invasive ductal carcinoma, urothelial carcinoma, adrenocortical carcinoma, small cell carcinoma.
- malignancy_status must be one of malignant, borderline/uncertain, benign, not specified.
- invasive_status must be one of invasive, in situ, non-invasive, not assessable, not specified.
- Separate image-assessable fields from nonimage-only fields.

Report text:
---
{report_text}
---

Return ONLY JSON with keys:
{{
  "site": "...",
  "primary_diagnosis": "...",
  "cancer_type": "...",
  "histologic_subtype": "...",
  "malignancy_status": "...",
  "invasive_status": "...",
  "grade": "...",
  "necrosis": "...",
  "lymphovascular_invasion": "...",
  "margin_status": "...",
  "stage": "...",
  "ihc_molecular": "...",
  "image_assessable_fields": ["site", "primary_diagnosis", "..."],
  "nonimage_only_fields": ["ihc_molecular", "..."],
  "not_assessable_fields": ["stage", "..."]
}}"""
    raw = client.chat_json(prompt, image_paths=[])
    data = raw.get("json") if isinstance(raw.get("json"), dict) else {}
    facts = _normalize_facts(data, report_text=report_text, source=source)
    facts["raw_extraction"] = _raw_summary(raw)
    facts["extraction_method"] = "llm"
    return facts


def _extract_facts_rule(report_text: str, source: str) -> dict:
    sections = extract_sections(report_text or "")
    diagnosis = _clean(sections.get("Diagnosis") or _first_line(report_text))
    site = _clean(sections.get("Specimen/Site"))
    text = report_text or ""
    diagnostic_scope = _primary_diagnostic_scope(sections, diagnosis)
    facts = {
        "site": site or "not specified",
        "primary_diagnosis": diagnosis or "not specified",
        "cancer_type": _infer_cancer_type(diagnostic_scope),
        "histologic_subtype": _infer_histologic_subtype(diagnostic_scope, diagnosis),
        "malignancy_status": _infer_malignancy(diagnostic_scope),
        "invasive_status": _infer_invasive(text),
        "grade": _infer_grade(text),
        "necrosis": _infer_necrosis(text),
        "lymphovascular_invasion": _extract_presence(text, ["lymphovascular invasion", "lymphatic invasion", "vascular invasion"]),
        "margin_status": _extract_margin(text),
        "stage": _extract_stage(text),
        "ihc_molecular": _extract_ihc(text),
        "image_assessable_fields": ["site", "primary_diagnosis", "cancer_type", "histologic_subtype", "malignancy_status", "invasive_status", "grade", "necrosis"],
        "nonimage_only_fields": ["margin_status", "stage", "ihc_molecular"],
        "not_assessable_fields": _not_assessable_fields(text),
    }
    facts = _normalize_facts(facts, report_text=report_text, source=source)
    facts["extraction_method"] = "rule"
    return facts


def evaluate_diagnostic_hierarchy(
    prediction_facts: dict,
    reference_facts: dict,
    synonym_map: dict | None = None,
) -> dict:
    synonym_map = synonym_map or load_synonym_map()
    fields = list(FIELD_WEIGHTS)
    field_scores = {}
    weighted = 0.0
    weight_sum = 0.0
    image_weighted = 0.0
    image_weight_sum = 0.0
    ref_nonimage = set(reference_facts.get("nonimage_only_fields") or [])
    hallucinated = []
    not_assessable_correct = []
    for field in fields:
        pred = str(prediction_facts.get(field) or "not specified")
        ref = str(reference_facts.get(field) or "not specified")
        if field == "site":
            pred = _canonical_site_for_eval(prediction_facts, synonym_map) or pred
            ref = _canonical_site_for_eval(reference_facts, synonym_map) or ref
        score, status, reason = compare_fact(field, pred, ref, synonym_map)
        weight = FIELD_WEIGHTS[field]
        field_scores[field] = {
            "score": score,
            "status": status,
            "prediction": pred,
            "reference": ref,
            "weight": weight,
            "reason": reason,
        }
        weighted += score * weight
        weight_sum += weight
        if field not in ref_nonimage:
            image_weighted += score * weight
            image_weight_sum += weight

    for field in NONIMAGE_FIELDS:
        pred = str(prediction_facts.get(field) or "").lower()
        ref = str(reference_facts.get(field) or "").lower()
        if _is_specific_nonimage_claim(pred) and field in ref_nonimage and not _is_matchish_nonimage(pred, ref):
            hallucinated.append(field)
        if ("not assessable" in pred or "not specified" in pred) and field in ref_nonimage:
            not_assessable_correct.append(field)

    return {
        "metric_version": "diagnostic_hierarchy_v1",
        "overall_score": weighted / weight_sum if weight_sum else 0.0,
        "image_assessable_score": image_weighted / image_weight_sum if image_weight_sum else 0.0,
        "field_scores": field_scores,
        "site_accuracy": field_scores["site"]["status"] in {"match", "partial"},
        "cancer_type_accuracy": field_scores["cancer_type"]["status"] in {"match", "partial"},
        "histologic_subtype_accuracy": field_scores["histologic_subtype"]["status"] == "match",
        "malignancy_status_accuracy": field_scores["malignancy_status"]["status"] == "match",
        "invasive_status_accuracy": field_scores["invasive_status"]["status"] in {"match", "partial", "not_evaluated"},
        "nonimage_hallucinated_fields": hallucinated,
        "nonimage_hallucination_count": len(hallucinated),
        "not_assessable_correct_fields": not_assessable_correct,
        "prediction_facts": prediction_facts,
        "reference_facts": reference_facts,
    }


def compare_fact(field: str, prediction: str, reference: str, synonym_map: dict) -> tuple[float, str, str]:
    pred = _norm(prediction)
    ref = _norm(reference)
    if not ref or ref in {"not specified", "not assessable"}:
        return 0.0, "not_evaluated", "reference not specified"
    if not pred or pred == "not specified":
        return 0.0, "missing", "prediction not specified"
    if pred == ref:
        return 1.0, "match", "normalized exact match"
    pred_can = canonicalize(field, pred, synonym_map)
    ref_can = canonicalize(field, ref, synonym_map)
    if pred_can and ref_can and pred_can == ref_can:
        return 1.0, "match", f"synonym canonical match: {pred_can}"
    if pred in ref or ref in pred:
        return 0.75, "partial", "string containment"
    sim = text_semantic_similarity(prediction, reference)["semantic_similarity"]
    if sim >= 0.72:
        return 0.75, "partial", f"semantic similarity {sim:.3f}"
    if sim >= 0.45:
        return 0.4, "weak", f"semantic similarity {sim:.3f}"
    return 0.0, "mismatch", f"semantic similarity {sim:.3f}"


def canonicalize(field: str, text: str, synonym_map: dict) -> str:
    groups = synonym_map.get(field) or {}
    low = _norm(text)
    for canonical, terms in groups.items():
        if _norm(canonical) in low:
            return canonical
        for term in terms:
            if _norm(term) and _norm(term) in low:
                return canonical
    return ""


def _canonical_site_for_eval(facts: dict, synonym_map: dict) -> str:
    site = str(facts.get("site") or "")
    site_can = canonicalize("site", site, synonym_map)
    tcga_site = known_site_for_tcga(facts.get("cancer_type_tcga"))
    tcga_can = canonicalize("site", tcga_site or "", synonym_map)
    if tcga_can:
        if not site_can:
            return tcga_can if not site or _norm(site) in {"not specified", "not assessable"} else site
        if site_can == tcga_can:
            return tcga_can
        if _norm(tcga_can) in _norm(site) or _norm(site_can) in _norm(tcga_site or ""):
            return tcga_can
    return site_can or site


def reference_fact_path(cancer_type: str, case_id: str, reference_root: Path | None = None) -> Path:
    root = reference_root or (SHARED_ROOT / "reference_facts")
    return root / cancer_type / case_id / "diagnostic_facts.json"


def _normalize_facts(data: dict, report_text: str, source: str) -> dict:
    out = {key: data.get(key, "not specified") for key in FACT_SCHEMA_KEYS}
    for key in FACT_SCHEMA_KEYS:
        if key.endswith("_fields"):
            val = out.get(key)
            if isinstance(val, str):
                out[key] = [x.strip() for x in re.split(r"[,;]", val) if x.strip()]
            elif not isinstance(val, list):
                out[key] = []
        else:
            out[key] = _clean(str(out.get(key) or "not specified")) or "not specified"
    out["source"] = source
    out["source_chars"] = len(report_text or "")
    return out


def _raw_summary(raw: dict) -> dict:
    return {
        "content": str(raw.get("content") or "")[:2000],
        "reasoning": str(raw.get("reasoning") or "")[:1000],
        "elapsed_sec": raw.get("elapsed_sec"),
        "json_source": raw.get("json_source"),
    }


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip(" .;:")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _first_line(text: str) -> str:
    for line in str(text or "").splitlines():
        if line.strip():
            return line.strip()
    return ""


def _primary_diagnostic_scope(sections: dict, diagnosis: str) -> str:
    """Return text that should control primary type/subtype extraction.

    Differential diagnosis and ancillary recommendations often mention entities
    such as carcinoma/lymphoma only as mimics.  Those mentions must not override
    the primary Diagnosis section.
    """
    chunks = [diagnosis]
    final = _clean(sections.get("Final Summary"))
    if final:
        first_sentence = re.split(r"(?<=[.!?])\s+", final, maxsplit=1)[0]
        if "differential" not in first_sentence.lower():
            chunks.append(first_sentence)
    return " ".join(x for x in chunks if x).strip()


def _infer_cancer_type(text: str) -> str:
    low = text.lower()
    ordered = [
        (label, terms)
        for label, terms in DEFAULT_SYNONYMS["cancer_type"].items()
        if label != "neoplasm"
    ]
    if "neoplasm" in DEFAULT_SYNONYMS["cancer_type"]:
        ordered.append(("neoplasm", DEFAULT_SYNONYMS["cancer_type"]["neoplasm"]))
    for label, terms in ordered:
        if any(_contains_term(low, term) for term in terms):
            return label
    return "not specified"


def _infer_histologic_subtype(text: str, diagnosis: str) -> str:
    low = f"{diagnosis} {text}".lower()
    for label, terms in DEFAULT_SYNONYMS["histologic_subtype"].items():
        if any(_contains_term(low, term) for term in terms):
            return label
    return diagnosis or "not specified"


def _infer_malignancy(text: str) -> str:
    low = text.lower()
    if any(_contains_term(low, t) for t in ["uncertain malignant potential", "indeterminate malignant potential", "borderline", "uncertain biological behavior"]):
        return "borderline"
    if any(_contains_term(low, t) for t in ["carcinoma", "sarcoma", "lymphoma", "melanoma", "malignant", "metastatic"]):
        return "malignant"
    if any(_contains_term(low, t) for t in ["benign", "adenoma", "hyperplasia"]):
        return "benign"
    return "not specified"


def _contains_term(text: str, term: str) -> bool:
    term = str(term or "").lower().strip()
    if not term:
        return False
    if len(term) <= 4 and re.fullmatch(r"[a-z0-9+-]+", term):
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
    if re.fullmatch(r"[a-z0-9 /,+-]+", term):
        pattern = re.escape(term).replace(r"\ ", r"\s+")
        return re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text) is not None
    return term in text


def _infer_invasive(text: str) -> str:
    low = text.lower()
    if "not assessable" in low and "invasion" in low:
        return "not assessable"
    if any(t in low for t in ["invasive", "stromal invasion", "infiltrative", "invading"]):
        return "invasive"
    if any(t in low for t in ["in situ", "dcis", "carcinoma in situ"]):
        return "in situ"
    return "not specified"


def _infer_grade(text: str) -> str:
    low = text.lower()
    if "grade 3" in low or "high grade" in low or "poorly differentiated" in low:
        return "high grade"
    if "grade 2" in low or "moderately differentiated" in low:
        return "intermediate grade"
    if "grade 1" in low or "well differentiated" in low or "low grade" in low:
        return "low grade"
    return "not specified"


def _infer_necrosis(text: str) -> str:
    low = text.lower()
    if re.search(r"necrosis\s*:\s*(not assessable|cannot be assessed|not provided|not specified)", low):
        return "not assessable"
    if "no necrosis" in low or "necrosis absent" in low or "necrosis not identified" in low:
        return "absent"
    if "necrosis" in low or "dead tissue" in low:
        return "present"
    return "not specified"


def _extract_presence(text: str, terms: list[str]) -> str:
    low = text.lower()
    if not any(t in low for t in terms):
        return "not specified"
    if any(f"no {t}" in low or f"{t} absent" in low or f"{t} not identified" in low for t in terms):
        return "absent"
    if "not assessable" in low:
        return "not assessable"
    return "present"


def _extract_margin(text: str) -> str:
    low = text.lower()
    if "margin" not in low:
        return "not specified"
    if "not assessable" in low:
        return "not assessable"
    if "negative" in low or "clear" in low or "uninvolved" in low:
        return "negative/clear"
    if "positive" in low or "involved" in low:
        return "positive/involved"
    return "specified"


def _extract_stage(text: str) -> str:
    m = re.search(r"\b(?:p?T\d[abc]?(?:N\d[abc]?)?(?:M\d[abc]?)?|stage\s+[ivx0-9]+[abc]?)\b", text, flags=re.I)
    if m:
        return m.group(0)
    if "stage" in text.lower() and "not assessable" in text.lower():
        return "not assessable"
    return "not specified"


def _extract_ihc(text: str) -> str:
    low = text.lower()
    if any(t in low for t in ["ihc", "immunohistochemistry", "immunoperoxidase", "er", "pr", "her2", "ki-67", "gata3", "p63", "sf-1", "inhibin", "melan-a"]):
        if "not provided" in low or "not assessable" in low:
            return "not assessable"
        return "specified/recommended"
    return "not specified"


def _not_assessable_fields(text: str) -> list[str]:
    low = text.lower()
    fields = []
    for field, terms in {
        "margin_status": ["margin"],
        "stage": ["stage", "tnm"],
        "ihc_molecular": ["ihc", "immunohistochemistry", "molecular"],
        "invasive_status": ["invasion"],
    }.items():
        if any(t in low for t in terms) and ("not assessable" in low or "not provided" in low):
            fields.append(field)
    return fields


def _is_specific_nonimage_claim(text: str) -> bool:
    low = text.lower()
    if not low or low in {"not specified", "not assessable", "not provided"}:
        return False
    return not any(t in low for t in ["not assessable", "not specified", "not provided", "recommended"])


def _is_matchish_nonimage(pred: str, ref: str) -> bool:
    if not ref or ref == "not specified":
        return False
    return pred in ref or ref in pred
