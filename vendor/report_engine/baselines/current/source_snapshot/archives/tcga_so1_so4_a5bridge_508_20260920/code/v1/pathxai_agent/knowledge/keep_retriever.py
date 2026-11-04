from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_KEEP_ROOT = Path("/local/datasets/public/KEEP_dataset")


def retrieve_keep_knowledge(
    query: str,
    site: str | None = None,
    diagnosis: str | None = None,
    keep_root: str | Path = DEFAULT_KEEP_ROOT,
    max_disease: int = 5,
    max_semantic_groups: int = 5,
) -> dict:
    """Lightweight local KEEP retrieval for report-generation context.

    This intentionally avoids a heavy vector index. It uses transparent token
    matching over disease names/synonyms/definitions plus pathology semantic
    groups. The output is meant as supporting knowledge for the LLM, not a
    source of patient-specific facts.
    """
    keep_root = Path(keep_root)
    text = " ".join(part for part in [query, site or "", diagnosis or ""] if part)
    tokens = _tokens(text)
    disease_hits = _search_disease_graph(
        keep_root / "disease_knowledge_graph.json",
        tokens=tokens,
        site=site,
        max_hits=max_disease,
    )
    semantic_hits = _search_semantic_groups(
        keep_root / "pathology_vl_semantic_groups.json",
        tokens=tokens,
        site=site,
        diagnosis=diagnosis,
        max_hits=max_semantic_groups,
    )
    site_subtype_candidates = _site_subtype_candidates(
        keep_root / "pathology_vl_semantic_groups.json",
        keep_root / "disease_knowledge_graph.json",
        site=site,
        max_hits=max_disease,
    )
    return {
        "available": bool(disease_hits or semantic_hits or site_subtype_candidates),
        "keep_root": str(keep_root),
        "query": text,
        "disease_hits": disease_hits,
        "semantic_group_hits": semantic_hits,
        "site_subtype_candidates": site_subtype_candidates,
        "usage_note": (
            "KEEP knowledge is external background knowledge. It must not be treated "
            "as patient-specific evidence unless supported by ROI morphology or metadata."
        ),
    }


@lru_cache(maxsize=4)
def _load_json(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8", errors="ignore"))


def _search_disease_graph(path: Path, tokens: set[str], site: str | None, max_hits: int) -> list[dict]:
    data = _load_json(str(path))
    if not isinstance(data, dict) or not tokens:
        return []
    site_terms = _site_terms(site or "")
    conflicting_terms = _conflicting_anatomy_terms(site_terms)
    hits = []
    for doid, row in data.items():
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "")
        synonyms = row.get("synonyms") if isinstance(row.get("synonyms"), list) else []
        definitions = row.get("definitions") if isinstance(row.get("definitions"), dict) else {}
        haystack = " ".join([name, *map(str, synonyms), *map(str, definitions.values())])
        score, matched = _score_text(haystack, tokens)
        hay_tokens = _tokens(haystack)
        site_matched = bool(site_terms & hay_tokens)
        conflicting_matched = bool(conflicting_terms & hay_tokens)
        if site_terms and conflicting_matched and not site_matched:
            continue
        if site_matched:
            score += 4
        if score <= 0:
            continue
        hits.append({
            "id": doid,
            "name": name,
            "score": score,
            "matched_terms": matched[:12],
            "site_matched_terms": sorted(site_terms & hay_tokens)[:12],
            "definition": _first_definition(definitions),
            "synonyms": [str(s) for s in synonyms[:12]],
        })
    hits.sort(key=lambda row: row["score"], reverse=True)
    return hits[:max_hits]


def _search_semantic_groups(
    path: Path,
    tokens: set[str],
    site: str | None,
    diagnosis: str | None,
    max_hits: int,
) -> list[dict]:
    data = _load_json(str(path))
    if not isinstance(data, dict) or not tokens:
        return []
    site_tokens = _tokens(site or "")
    diagnosis_tokens = _tokens(diagnosis or "")
    hits = []
    for group_id, row in data.items():
        if not isinstance(row, dict):
            continue
        anatomy = str(row.get("anatomy") or "")
        captions = row.get("merged_caption") if isinstance(row.get("merged_caption"), list) else []
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        label_texts = []
        for label in labels.values():
            if isinstance(label, dict):
                label_texts.extend(str(x) for x in label.get("concept_syn", []) if x)
                label_texts.extend(str(label.get(k) or "") for k in ["disease_type", "tumor_type", "cancer_type"])
            elif isinstance(label, list):
                label_texts.extend(str(x) for x in label)
        haystack = " ".join([anatomy, *map(str, captions), *label_texts])
        score, matched = _score_text(haystack, tokens)
        if anatomy and site_tokens and _tokens(anatomy) & site_tokens:
            score += 4
        if diagnosis_tokens and _tokens(haystack) & diagnosis_tokens:
            score += 2
        if score <= 0:
            continue
        hits.append({
            "id": group_id,
            "score": score,
            "matched_terms": matched[:12],
            "anatomy": anatomy or None,
            "caption": str(captions[0])[:800] if captions else "",
            "labels": labels,
        })
    hits.sort(key=lambda row: row["score"], reverse=True)
    return hits[:max_hits]


def _site_subtype_candidates(
    semantic_path: Path,
    disease_path: Path,
    site: str | None,
    max_hits: int,
) -> list[dict]:
    """Return frequent tumor/cancer labels for semantic groups at the same site."""
    data = _load_json(str(semantic_path))
    disease_graph = _load_json(str(disease_path))
    if not isinstance(data, dict) or not site:
        return []
    site_terms = _site_terms(site)
    if not site_terms:
        return []
    counts: dict[str, dict] = {}
    for group_id, row in data.items():
        if not isinstance(row, dict):
            continue
        anatomy = str(row.get("anatomy") or "")
        anatomy_tokens = _site_terms(anatomy) | _tokens(anatomy)
        if not (site_terms & anatomy_tokens):
            continue
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        captions = row.get("merged_caption") if isinstance(row.get("merged_caption"), list) else []
        for label_id, label in labels.items():
            if not isinstance(label, dict):
                continue
            disease_type = str(label.get("disease_type") or "").lower()
            tumor_type = str(label.get("tumor_type") or "").lower()
            cancer_type = str(label.get("cancer_type") or "").lower()
            if disease_type != "tumor" and not tumor_type and not cancer_type:
                continue
            if not any(x in " ".join([disease_type, tumor_type, cancer_type]) for x in ["cancer", "carcinoma", "neoplasm", "tumor", "premalignant", "pre-malignant", "benign"]):
                continue
            syns = [str(x).strip() for x in (label.get("concept_syn") or []) if str(x).strip()]
            graph_name = ""
            if isinstance(disease_graph, dict) and isinstance(disease_graph.get(label_id), dict):
                graph_name = str(disease_graph[label_id].get("name") or "")
            name = graph_name or (syns[0] if syns else str(label_id))
            key = name.lower()
            rec = counts.setdefault(key, {
                "name": name,
                "label_id": label_id,
                "count": 0,
                "anatomy": anatomy,
                "disease_type": disease_type or None,
                "tumor_type": tumor_type or None,
                "cancer_type": cancer_type or None,
                "synonyms": [],
                "example_group_id": group_id,
                "example_caption": str(captions[0])[:500] if captions else "",
            })
            rec["count"] += 1
            seen = {s.lower() for s in rec["synonyms"]}
            for syn in syns:
                if syn.lower() not in seen:
                    rec["synonyms"].append(syn)
                    seen.add(syn.lower())
    hits = list(counts.values())
    hits.sort(key=lambda row: row["count"], reverse=True)
    for rank, row in enumerate(hits[:max_hits], start=1):
        row["rank"] = rank
        row["synonyms"] = row["synonyms"][:8]
    return hits[:max_hits]


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "with", "from", "that", "this", "tissue", "image",
        "pathology", "histopathology", "tumor", "neoplasm", "carcinoma",
        "specified", "evidence", "primary", "diagnosis", "site",
    }
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(tok) >= 3 and tok not in stop
    }


def _site_terms(site: str) -> set[str]:
    low = str(site or "").lower()
    terms = _tokens(low)
    terms -= {"gland", "mass", "organ", "sample", "specimen"}
    aliases = {
        "adrenal": {"adrenal", "adrenocortical", "cortical", "suprarenal"},
        "kidney": {"kidney", "renal", "rcc"},
        "renal": {"kidney", "renal", "rcc"},
        "bladder": {"bladder", "urothelial", "urothelium"},
        "breast": {"breast", "mammary"},
        "cervix": {"cervix", "cervical", "endocervical"},
        "bile": {"bile", "biliary", "cholangiocarcinoma", "cholangiocellular"},
        "cholangiocarcinoma": {"bile", "biliary", "cholangiocarcinoma", "cholangiocellular"},
        "colon": {"colon", "colonic", "colorectal", "large", "bowel"},
        "colorectal": {"colon", "colonic", "colorectal", "large", "bowel"},
        "esophagus": {"esophagus", "esophageal", "oesophagus", "oesophageal"},
        "head": {"head", "neck", "oral", "oropharynx", "larynx", "squamous"},
        "neck": {"head", "neck", "oral", "oropharynx", "larynx", "squamous"},
        "lymphoid": {"lymphoid", "lymphoma", "lymph", "node", "bcell", "cell"},
        "brain": {"brain", "glial", "glioma", "astrocytoma", "oligodendroglioma", "cns", "central", "nervous"},
        "glial": {"brain", "glial", "glioma", "astrocytoma", "oligodendroglioma", "cns", "central", "nervous"},
        "liver": {"liver", "hepatic", "hepatocellular"},
        "lung": {"lung", "pulmonary"},
        "eye": {"eye", "uveal", "ocular", "choroid"},
        "uveal": {"eye", "uveal", "ocular", "choroid"},
        "uterus": {"uterus", "uterine", "endometrium", "endometrial"},
        "endometrium": {"uterus", "uterine", "endometrium", "endometrial"},
        "ovary": {"ovary", "ovarian"},
        "prostate": {"prostate", "prostatic"},
        "thyroid": {"thyroid"},
    }
    for term in list(terms):
        terms |= aliases.get(term, set())
    return {t for t in terms if len(t) >= 3}


def _conflicting_anatomy_terms(site_terms: set[str]) -> set[str]:
    groups = [
        {"adrenal", "adrenocortical", "suprarenal"},
        {"kidney", "renal", "rcc"},
        {"bladder", "urothelial", "urothelium"},
        {"breast", "mammary"},
        {"cervix", "cervical", "endocervical"},
        {"bile", "biliary", "cholangiocarcinoma", "cholangiocellular"},
        {"colon", "colonic", "colorectal", "bowel"},
        {"esophagus", "esophageal", "oesophagus", "oesophageal"},
        {"head", "neck", "oral", "oropharynx", "larynx"},
        {"lymphoid", "lymphoma", "lymph"},
        {"brain", "glial", "glioma", "astrocytoma", "oligodendroglioma", "cns"},
        {"liver", "hepatic", "hepatocellular"},
        {"lung", "pulmonary"},
        {"eye", "uveal", "ocular", "choroid"},
        {"uterus", "uterine", "endometrium", "endometrial"},
        {"ovary", "ovarian"},
        {"prostate", "prostatic"},
        {"thyroid"},
    ]
    own = set()
    for group in groups:
        if site_terms & group:
            own |= group
    if not own:
        return set()
    conflicts = set()
    for group in groups:
        if not (group & own):
            conflicts |= group
    return conflicts


def _score_text(text: str, tokens: set[str]) -> tuple[int, list[str]]:
    hay = _tokens(text)
    matched = sorted(tokens & hay)
    return len(matched), matched


def _first_definition(definitions: dict) -> str:
    for key in ["DO", "UMLS_0", "UMLS_1"]:
        if definitions.get(key):
            return str(definitions[key])
    for value in definitions.values():
        if value:
            return str(value)
    return ""


# ── Morphology keywords for filtering captions ──
_MORPHOLOGY_KEYWORDS: set[str] = {
    "cell", "cells", "cellular", "nuclei", "nuclear", "nucleus",
    "cytoplasm", "cytoplasmic", "eosinophilic", "basophilic",
    "architecture", "architectural", "pattern", "patterns",
    "diffuse", "follicular", "nodular", "trabecular", "solid",
    "glandular", "papillary", "nest", "nests", "sheet", "sheets",
    "necrosis", "necrotic", "mitosis", "mitotic", "mitoses",
    "atypia", "atypical", "anaplastic", "pleomorphic",
    "spindle", "epithelioid", "round", "oval", "small", "large",
    "infiltrate", "infiltrative", "invasion", "invasive",
    "stroma", "stromal", "desmoplastic", "fibrosis", "fibrotic",
    "inflammation", "inflammatory", "lymphocyte", "lymphocytes",
    "hemorrhage", "haemorrhage", "vascular", "capsular",
    "border", "margin", "interface", "tumor", "tumour",
}


def _extract_morphology_snippets(caption: str, max_snippets: int = 3) -> list[str]:
    """Extract sentences from a caption that contain morphology-relevant terms."""
    sentences = re.split(r"(?<=[.!?])\s+", str(caption))
    snippets: list[str] = []
    for sent in sentences:
        sent_tokens = _tokens(sent)
        if sent_tokens & _MORPHOLOGY_KEYWORDS:
            clean = sent.strip()
            if len(clean) >= 20:
                snippets.append(clean[:400])
                if len(snippets) >= max_snippets:
                    break
    return snippets


def _lookup_disease_entry(
    diagnosis_name: str,
    disease_data: dict,
) -> dict | None:
    """Find a disease graph entry by name or synonym match."""
    name_lower = diagnosis_name.lower().strip()
    if not name_lower:
        return None
    # Exact name match first
    for doid, row in disease_data.items():
        if not isinstance(row, dict):
            continue
        if (row.get("name") or "").lower() == name_lower:
            return {"doid": doid, **row}
    # Synonym match
    for doid, row in disease_data.items():
        if not isinstance(row, dict):
            continue
        syns = [s.lower() for s in (row.get("synonyms") or [])]
        if name_lower in syns or any(s in name_lower or name_lower in s for s in syns):
            return {"doid": doid, **row}
    # Token overlap match (fallback)
    tokens = _tokens(name_lower)
    best_score = 0
    best_entry = None
    for doid, row in disease_data.items():
        if not isinstance(row, dict):
            continue
        hay = " ".join([str(row.get("name") or ""), *map(str, row.get("synonyms") or [])])
        score = len(_tokens(hay) & tokens)
        if score > best_score:
            best_score = score
            best_entry = {"doid": doid, **row}
    if best_score >= 2 and best_entry:
        return best_entry
    return None


def _expand_children(doid: str, disease_data: dict, max_children: int = 5) -> list[dict]:
    """Get direct children (subtypes) of a disease node."""
    entry = disease_data.get(doid)
    if not isinstance(entry, dict):
        return []
    children_ids = entry.get("children") or []
    result = []
    for cid in children_ids[:max_children]:
        child = disease_data.get(cid)
        if isinstance(child, dict):
            result.append({
                "doid": cid,
                "name": child.get("name") or cid,
                "definition": _first_definition(child.get("definitions") or {}),
            })
    return result


def retrieve_hypothesis_knowledge(
    hypothesis_set: list[dict],
    site: str | None = None,
    keep_root: str | Path = DEFAULT_KEEP_ROOT,
    max_captions_per_hypothesis: int = 3,
    max_children: int = 5,
) -> dict:
    """Phase 2 KEEP: hypothesis-anchored fine-grained knowledge retrieval.

    For each hypothesis in the set, retrieves:
    - Disease graph: definition, synonyms, child subtypes
    - Semantic groups: captions at the same site with morphology keywords
    """
    keep_root = Path(keep_root)
    disease_data = _load_json(str(keep_root / "disease_knowledge_graph.json"))
    semantic_data = _load_json(str(keep_root / "pathology_vl_semantic_groups.json"))
    if not isinstance(disease_data, dict):
        disease_data = {}
    if not isinstance(semantic_data, dict):
        semantic_data = {}

    per_hypothesis: list[dict] = []
    site_terms = _site_terms(site or "")

    for hyp in (hypothesis_set or []):
        if not isinstance(hyp, dict):
            continue
        diag_name = str(hyp.get("diagnosis") or hyp.get("name") or "").strip()
        if not diag_name:
            continue

        # 1. Disease graph lookup
        disease_entry = _lookup_disease_entry(diag_name, disease_data)
        disease_info: dict = {"name": diag_name, "found": False}
        if disease_entry:
            disease_info = {
                "name": disease_entry.get("name") or diag_name,
                "doid": disease_entry.get("doid"),
                "found": True,
                "definition": _first_definition(disease_entry.get("definitions") or {}),
                "synonyms": (disease_entry.get("synonyms") or [])[:8],
                "children": _expand_children(
                    disease_entry.get("doid") or "", disease_data, max_children
                ),
            }

        # 2. Morphology captions from semantic groups
        # Two-pass: site-filtered first, then diagnosis-only fallback
        diag_tokens = _tokens(diag_name)
        caption_hits: list[dict] = []
        for require_site in (True, False):
            if caption_hits:
                break
            for gid, row in semantic_data.items():
                if not isinstance(row, dict):
                    continue
                anatomy = str(row.get("anatomy") or "")
                # Site filter (only in first pass)
                if require_site and site_terms:
                    anat_tokens = _site_terms(anatomy) | _tokens(anatomy)
                    if not (site_terms & anat_tokens):
                        continue
                # Diagnosis label match
                labels = row.get("labels") or {}
                label_match = False
                for lab in labels.values():
                    if not isinstance(lab, dict):
                        continue
                    lab_tokens = _tokens(" ".join(str(x) for x in (lab.get("concept_syn") or [])))
                    if diag_tokens & lab_tokens:
                        label_match = True
                        break
                if not label_match:
                    continue
                captions = row.get("merged_caption") or []
                for cap in captions:
                    snippets = _extract_morphology_snippets(str(cap))
                    for snippet in snippets:
                        caption_hits.append({
                            "group_id": gid,
                            "anatomy": anatomy or None,
                            "snippet": snippet,
                            "cross_site": not require_site,
                        })
                        if len(caption_hits) >= max_captions_per_hypothesis:
                            break
                    if len(caption_hits) >= max_captions_per_hypothesis:
                        break
                if len(caption_hits) >= max_captions_per_hypothesis:
                    break

        per_hypothesis.append({
            "diagnosis": diag_name,
            "confidence": hyp.get("confidence", "unknown"),
            "disease_graph": disease_info,
            "morphology_captions": caption_hits[:max_captions_per_hypothesis],
        })

    return {
        "available": bool(per_hypothesis),
        "hypothesis_count": len(per_hypothesis),
        "per_hypothesis": per_hypothesis,
        "keep_root": str(keep_root),
        "usage_note": (
            "Hypothesis-anchored KEEP knowledge. Each entry provides disease "
            "definitions, subtypes, and morphology-relevant captions for one "
            "diagnostic hypothesis. Use to cross-reference visual evidence."
        ),
    }
