"""Auditable, report-only evaluation. No generation objects or legacy scores.

This is a deterministic *reference-agreement* evaluator, not a clinical reader.
Every primary denominator is reference-defined. Uncertain/contradictory
predictions are abstentions; undocumented reference facts are not negatives.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import re
from typing import Any

VERSION = "eval_v3_report_only_20260915_r3"
DIAGNOSIS_WEIGHTS = {"broad": .35, "subtype": .35, "malignancy": .15}
GRADE_RANGES = {"histologic3": (1, 3), "nuclear3": (1, 3),
                "who4": (1, 4), "fuhrman4": (1, 4), "isup5": (1, 5),
                "gleason10": (2, 10)}

HEADINGS = {
    "specimen/site": "site", "site": "site", "specimen": "site",
    "diagnosis": "diagnosis", "primary diagnosis": "diagnosis",
    "final diagnosis": "diagnosis", "diagnostic impression": "diagnosis",
    "key findings": "morphology", "key microscopic findings": "morphology",
    "microscopic findings": "morphology", "microscopic description": "morphology",
    "tumor/lesion description": "morphology",
    "invasion/necrosis/margins": "findings",
    "stage/grade/prognostic features": "grade", "histologic grade": "grade",
    "final summary": "summary", "conclusion": "summary",
    "differential diagnoses": "excluded", "differential diagnosis": "excluded",
    "limitations": "excluded", "evidence rationale": "excluded",
    "immunohistochemistry/ancillary studies": "excluded",
    "immunohistochemistry": "excluded", "recommendations": "excluded",
    "ancillary studies": "excluded", "clinical correlation": "excluded",
}
UNKNOWN = re.compile(r"\b(?:not assessable|not specified|unspecified|unknown|indeterminate|"
                     r"cannot (?:be )?(?:assess|determin|evaluat)|unable to|insufficient|"
                     r"not evaluable|not provided|not reported|not applicable)\b", re.I)
UNCERTAIN = re.compile(r"\b(?:uncertain|equivocal|possible|possibly|suspicious|suspected|"
                       r"suggestive|suggests?|may|might|could|cannot exclude|cannot rule out|"
                       r"not (?:definite|definitive)|requires? confirmation|pending|vs)\b|\?", re.I)
NEGATIVE = re.compile(r"\b(?:no|not|without|absent|negative for|lack(?:s|ing)?(?: of)?|"
                      r"free of|no evidence of|not identified|not seen)\b", re.I)
HYPOTHETICAL = re.compile(r"\b(?:differential|rule out|exclude|recommended|recommend|"
                          r"consider|if confirmed|would support|e\.g\.|such as)\b", re.I)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def contains(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(normalize(phrase)) + r"(?!\w)", normalize(text)))


def sections(text: str) -> list[tuple[str, str]]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"<think>.*$", "", text, flags=re.S | re.I)
    result, current = [], "unsectioned"
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:#{1,6}\s*|\d+[.)]\s*|[-*]\s+)", "", line).strip()
        line = line.replace("**", "").replace("__", "")
        if not line:
            continue
        title, sep, body = line.partition(":")
        heading = HEADINGS.get(title.strip().lower())
        if heading is not None:
            current = heading
            if sep and body.strip():
                result.append((current, body.strip()))
        else:
            result.append((current, line))
    return result


def clauses(text: str) -> list[str]:
    # Decimal measurements do not create sentence boundaries.
    return [s.strip() for s in re.split(r"(?<!\d)\.|\.(?!\d)|;|\n|\bbut\b|\bhowever\b", text) if s.strip()]


def polarity(clause: str, start: int = 0, end: int | None = None) -> str:
    if UNKNOWN.search(clause) or UNCERTAIN.search(clause):
        return "uncertain"
    before = clause[max(0, start-90):start]
    after = clause[end if end is not None else start:]
    # Restrict negation to its clause; "no special type" is not negation.
    before = re.sub(r"no special type|non.small cell", "", before, flags=re.I)
    if NEGATIVE.search(before) or re.match(r"\s*(?:is |was |: )?(?:absent|not (?:seen|identified|present)|negative)\b", after, re.I):
        return "absent"
    return "present"


def resolve(hits: list[dict], *, binary: bool = False) -> dict:
    certain = [h for h in hits if h["status"] == "asserted"]
    vals = {str(h["value"]) for h in certain}
    if len(vals) > 1:
        return dict(value=None, status="conflict", evidence=hits)
    if vals:
        return dict(value=certain[0]["value"], status="asserted", evidence=hits)
    return dict(value=None, status="uncertain" if hits else "missing", evidence=hits)


def fact(value=None, text="", status="asserted") -> dict:
    return dict(value=value, status=status if value is not None else "missing",
                evidence=[dict(value=value, status=status, text=text)] if text else [])


# Conservative entity groups. Parents are not synonyms of their children.
ENTITIES = {
    "invasive ductal carcinoma": ["invasive ductal carcinoma", "ductal carcinoma nst", "idc",
                                   "invasive carcinoma of no special type", "invasive breast carcinoma no special type"],
    "invasive lobular carcinoma": ["invasive lobular carcinoma", "ilc"],
    "invasive breast carcinoma": ["invasive breast carcinoma", "breast invasive carcinoma", "invasive breast cancer"],
    "invasive carcinoma": ["invasive carcinoma"],
    "ductal carcinoma in situ": ["ductal carcinoma in situ", "dcis"],
    "lobular carcinoma in situ": ["lobular carcinoma in situ", "lcis"],
    "adrenocortical carcinoma": ["adrenocortical carcinoma", "adrenal cortical carcinoma"],
    "adrenocortical neoplasm": ["adrenocortical neoplasm", "adrenal cortical neoplasm", "cortical adrenal tumor"],
    "clear cell renal cell carcinoma": ["clear cell renal cell carcinoma", "clear cell rcc"],
    "papillary renal cell carcinoma": ["papillary renal cell carcinoma", "papillary rcc"],
    "chromophobe renal cell carcinoma": ["chromophobe renal cell carcinoma", "chromophobe rcc"],
    "urothelial carcinoma": ["urothelial carcinoma", "transitional cell carcinoma"],
    "prostate adenocarcinoma": ["prostate adenocarcinoma", "prostatic adenocarcinoma",
                                "prostatic acinar adenocarcinoma", "acinar adenocarcinoma of the prostate", "acinar carcinoma"],
    "benign prostatic hyperplasia": ["benign prostatic hyperplasia", "bph", "prostatic glandular hyperplasia", "active hyperplasia of prostatic glands"],
    "benign prostate tissue": ["benign prostate tissue", "benign prostatic tissue"],
    "breast carcinoma": ["breast carcinoma", "breast cancer"],
    "tubular adenoma": ["tubular adenoma"],
    "tubulovillous adenoma": ["tubulovillous adenoma", "villous tubular adenoma", "villous-tubular adenoma"],
    "sessile serrated lesion": ["sessile serrated lesion", "sessile serrated polyp", "ssl"],
    "hyperplastic polyp": ["hyperplastic polyp"],
    "colorectal adenocarcinoma": ["colorectal adenocarcinoma", "colonic adenocarcinoma",
                                  "colon adenocarcinoma", "rectal adenocarcinoma"],
    "lung adenocarcinoma": ["lung adenocarcinoma", "pulmonary adenocarcinoma"],
    "gastric adenocarcinoma": ["gastric adenocarcinoma", "stomach adenocarcinoma"],
    "hepatocellular carcinoma": ["hepatocellular carcinoma", "hcc"],
    "cholangiocarcinoma": ["cholangiocarcinoma", "intrahepatic cholangiocarcinoma"],
    "squamous cell carcinoma": ["squamous cell carcinoma"],
    "adenocarcinoma": ["adenocarcinoma"], "carcinoma": ["carcinoma"],
    "melanoma": ["melanoma", "malignant melanoma"],
    "nodular melanoma": ["nodular melanoma", "nodular malignant melanoma"],
    "superficial spreading melanoma": ["superficial spreading melanoma", "superficial spreading malignant melanoma"],
    "spitzoid melanoma": ["spitzoid melanoma", "spitz like melanoma"],
    "diffuse large b cell lymphoma": ["diffuse large b cell lymphoma", "dlbcl"],
    "glioblastoma": ["glioblastoma", "glioblastoma multiforme"],
    "astrocytoma": ["astrocytoma"], "oligodendroglioma": ["oligodendroglioma"],
    "mesothelioma": ["mesothelioma"], "sarcoma": ["sarcoma"],
    "pheochromocytoma": ["pheochromocytoma"], "paraganglioma": ["paraganglioma"],
    "papillary thyroid carcinoma": ["papillary thyroid carcinoma", "thyroid papillary carcinoma"],
    "follicular thyroid carcinoma": ["follicular thyroid carcinoma", "thyroid follicular carcinoma"],
    "serous carcinoma": ["serous carcinoma", "high grade serous carcinoma"],
    "endometrioid carcinoma": ["endometrioid carcinoma", "endometrioid adenocarcinoma"],
    "seminoma": ["seminoma"], "thymoma": ["thymoma"],
    "carcinosarcoma": ["carcinosarcoma", "malignant mixed mullerian tumor"],
}
BROAD = [
    ("carcinosarcoma", r"\bcarcinosarcoma\b|malignant mixed mullerian"),
    ("sarcoma", r"\b(?:\w*?sarcoma|gastrointestinal stromal tumor)\b"),
    ("lymphoma", r"\b(?:lymphoma|leukemia|myeloma|lymphoid neoplasm)\b"),
    ("melanoma", r"\bmelanoma\b"), ("glioma", r"\b(?:glioma|glioblastoma|astrocytoma|oligodendroglioma)\b"),
    ("germ cell tumor", r"\b(?:germ cell|seminoma|teratoma|yolk sac tumor)\b"),
    ("mesothelioma", r"\bmesothelioma\b"),
    ("thymic epithelial neoplasm", r"\b(?:thymoma|thymic epithelial neoplasm)\b"),
    ("carcinoma", r"\b(?:\w*carcinoma|carcinoma)\b"),
    ("neuroendocrine neoplasm", r"\b(?:neuroendocrine neoplasm|neuroendocrine tumor|pheochromocytoma|paraganglioma|carcinoid)\b"),
]


def diagnosis_scope(lines: list[tuple[str, str]]) -> str:
    chosen = [t for s, t in lines if s == "diagnosis"]
    if not chosen:
        chosen = [t for s, t in lines if s in ("unsectioned", "summary")]
    # Inline structured labels are retained, but never auxiliary IHC/differentials.
    return "\n".join(re.split(r"\bwith associated\b|\band associated\b", c, flags=re.I)[0]
                     for t in chosen for c in clauses(t) if not HYPOTHETICAL.search(c))


def entity(text: str) -> dict:
    hits = []
    for c in clauses(text):
        if re.match(r'(?:broad(?: type| lineage)?|malignancy(?: status)?|grade|confidence)\s*:', c, re.I):
            continue
        t = normalize(c)
        matches = [(len(normalize(a)), key, a) for key, aliases in ENTITIES.items()
                   for a in aliases if contains(t, a)]
        if matches and not HYPOTHETICAL.search(c):
            _, key, alias = max(matches)
            start = t.find(normalize(alias))
            p = polarity(t, start, start+len(normalize(alias)))
            hits.append(dict(value=key, text=c, status="asserted" if p == "present" else p))
    # A repeated parent label does not contradict a more specific primary entity.
    certain = [h for h in hits if h['status'] == 'asserted']
    parents = {'invasive ductal carcinoma': {'invasive breast carcinoma', 'invasive carcinoma'},
               'invasive lobular carcinoma': {'invasive breast carcinoma', 'invasive carcinoma'},
               'invasive breast carcinoma': {'invasive carcinoma', 'breast carcinoma'}}
    for h in certain:
        if any(h is not other and h['value'] != other['value'] and
               (contains(other['value'], h['value']) or h['value'] in parents.get(other['value'], set()) or
                h['value'] == 'carcinoma' and other['value'].endswith('carcinoma')) for other in certain):
            h['status'] = 'parent_label'
    return resolve(hits)


def broad_type(text: str) -> dict:
    hits = []
    for c in clauses(text):
        for label, pattern in BROAD:
            m = re.search(pattern, c, re.I)
            if m:
                p = polarity(c, m.start(), m.end())
                hits.append(dict(value=label, text=c, status="asserted" if p == "present" else p))
                break
    return resolve(hits)


def malignancy(text: str) -> dict:
    explicit = re.findall(r"(?:^|\n)\s*malignancy(?: status)?\s*:\s*([^\n]+)", text, re.I)
    source = explicit or clauses(text)
    hits = []
    for c in source:
        if UNKNOWN.search(c) and not re.search(r"(?:indeterminate|uncertain|low) malignant potential", c, re.I):
            continue
        if re.search(r"borderline|(?:indeterminate|uncertain|low) malignant potential", c, re.I):
            value = "borderline"
        elif re.search(r"\bbenign\b|\badenoma\b|\bhyperplasia\b", c, re.I):
            value = "benign"
        elif re.search(r"\bmalignant\b|\w*carcinoma\b|\w*sarcoma\b|lymphoma|melanoma|glioblastoma|leukemia", c, re.I):
            value = "malignant"
        else:
            continue
        # Malignancy and invasion are separate: in-situ carcinoma is not benign.
        target = re.search(r"\bmalignant\b|\w*carcinoma\b|\w*sarcoma\b|lymphoma|melanoma|\bbenign\b|\badenoma\b", c, re.I)
        p = polarity(c, target.start(), target.end()) if target else "present"
        if value == "borderline":
            p = "present"
        hits.append(dict(value=value, text=c, status="asserted" if p == "present" else p))
    return resolve(hits)


FINDINGS = {
    "invasion": r"\b(?:invas(?:ion|ive)|infiltrat(?:es?|ing|ion|ive))\b",
    "necrosis": r"\bnecro(?:sis|tic)\b",
    "lymphovascular_invasion": r"\b(?:lymphovascular|angiolymphatic|lymphatic|vascular) invasion\b",
    "perineural_invasion": r"\bperineural invasion\b",
    "metastasis": r"\bmetasta(?:sis|ses|tic)\b",
    "pleomorphism": r"\bpleomorph(?:ism|ic)\b",
    "mitoses": r"\bmito(?:ses|sis|tic figures|tic activity)\b",
    "glandular": r"\b(?:glandular|gland formation|gland forming)\b",
    "papillary": r"\bpapillary\b", "solid": r"\bsolid (?:sheets|nests|growth|architecture|pattern)\b",
    "nested": r"\b(?:nested|nests of|tumor nests)\b",
    "trabecular": r"\btrabecular\b", "cribriform": r"\bcribriform\b",
    "spindle": r"\bspindle(?:d)? cells?\b", "clear_cytoplasm": r"\bclear cytoplasm\b",
    "hyperchromasia": r"\bhyperchrom(?:asia|atic)\b",
    "nucleoli": r"\bprominent nucleol(?:i|us)\b",
    "desmoplasia": r"\bdesmoplas(?:ia|tic)\b",
    "inflammation": r"\b(?:inflammatory|lymphocytic) infiltrate\b",
    "calcification": r"\b(?:micro)?calcifications?\b",
}
CRITICAL = {"invasion", "necrosis", "lymphovascular_invasion", "perineural_invasion", "metastasis"}


def finding(text: str, key: str) -> dict:
    hits = []
    for c in clauses(text):
        if HYPOTHETICAL.search(c):
            continue
        if key == "invasion":
            # General stromal invasion is not the same field as LVI/PNI.
            c = re.sub(r"(?:lymphovascular|angiolymphatic|vascular|lymphatic|perineural|capsular) invasion", "", c, flags=re.I)
        for m in re.finditer(FINDINGS[key], c, re.I):
            p = polarity(c, m.start(), m.end())
            if key == 'invasion' and re.search(r"non[ -]?invasive", c, re.I):
                p = 'absent'
            hits.append(dict(value=p if p != "uncertain" else None, text=c,
                             status="asserted" if p != "uncertain" else "uncertain"))
    return resolve(hits, binary=True)


def structured_binary(text: str, key: str) -> dict:
    if not text or UNKNOWN.search(text) or UNCERTAIN.search(text):
        return fact(None, text)
    if re.search(r"\b(?:absent|negative|not identified|not seen|noninvasive|non invasive|non-invasive|in situ)\b", text, re.I):
        return fact("absent", text)
    if re.search(r"\b(?:present|positive|invasive|invasion|focal)\b", text, re.I):
        return fact("present", text)
    return finding(text, key)


def grade(lines: list[tuple[str, str]], explicit: str | None = None) -> dict:
    """Keep grading systems separate; morphology alone is not overall grade."""
    hits = []
    primary_invasive = bool(re.search(r'\binvasive\b', diagnosis_scope(lines), re.I))
    selected = [("grade", explicit)] if explicit is not None else lines
    for section, line in selected:
        if section in ("excluded", "site"):
            continue
        for c in clauses(line):
            if HYPOTHETICAL.search(c) or re.search(r"dysplasia|intraepithelial lesion|\bincidental\b", c, re.I):
                continue
            if primary_invasive and re.search(r'\bin situ\b|\b[dl]cis\b', c, re.I) and not re.search(r'\binvasive\b', c, re.I):
                continue
            # Enumerations of required grading components are not a grade claim.
            if re.search(r"nuclear (?:features|atypia)|high.power|mitotic count|tubule formation", c, re.I) and not re.search(r"grade\s*(?:[1-4]|i{1,3}v?)\b|nottingham", c, re.I):
                continue
            system = ("isup5" if re.search(r"isup|grade group", c, re.I) else
                      "fuhrman4" if re.search(r"fuhrman", c, re.I) else
                      "who4" if re.search(r"\bwho\b", c, re.I) else
                      "gleason10" if re.search(r"gleason", c, re.I) else
                      "histologic3" if re.search(r"\bsbr\b|nottingham|elston|fnclcc|overall grade|histolog(?:ic|ical) grade", c, re.I) else
                      "nuclear3" if re.search(r"nuclear grade", c, re.I) else "histologic3")
            candidates = []
            for m in re.finditer(r"\b(?:grade(?: group)?|fuhrman|gleason(?: score)?)\s*[:=]?\s*(10|[1-9]|iv|iii|ii|i)\b", c, re.I):
                if system == 'isup5' and m.group().lower().startswith('gleason'):
                    continue
                raw = m.group(1).lower()
                value = {"i": 1, "ii": 2, "iii": 3, "iv": 4}.get(raw, int(raw) if raw.isdigit() else 0)
                if system == "gleason10" and re.match(r"\s*\+", c[m.end():]):
                    pair = re.match(r"\s*\+\s*([1-5])", c[m.end():])
                    if pair:
                        value += int(pair.group(1))
                candidates.append((value, m))
            if not candidates and system in ("histologic3", "nuclear3"):
                if section == 'grade':
                    candidates.extend((int(m.group(1)), m) for m in re.finditer(r'\bG([1-3])\b', c, re.I))
                for value, pattern in [(1, r"\blow[ -]grade\b|\bwell[ -]differentiated\b"),
                                       (2, r"\bintermediate[ -]grade\b|\bmoderately[ -]differentiated\b"),
                                       (3, r"\bhigh[ -]grade\b|\bpoorly[ -]differentiated\b")]:
                    candidates.extend((value, m) for m in re.finditer(pattern, c, re.I))
            for value, m in candidates:
                # Do not infer overall grade from isolated nuclear morphology.
                if section == "morphology" and not re.search(r"(?:histologic|histological|nottingham|fuhrman|who|gleason|isup|nuclear) grade|grade\s*:", c, re.I):
                    continue
                p = polarity(c, m.start(), m.end())
                lo, hi = GRADE_RANGES[system]
                status = "asserted" if p == "present" and lo <= value <= hi else "uncertain"
                hits.append(dict(value=f"{system}:{value}", ordinal=value, system=system,
                                 status=status, text=c, priority=int(section == 'grade' or bool(re.match(r"grade\s*:", c, re.I)))))
            if re.match(r"grade\s*:\s*(?:not assessable|not specified|cannot|unknown)", c, re.I):
                hits.append(dict(value=None, status="uncertain", text=c, priority=1))
    # Explicit overall grade takes precedence over adjectives in diagnosis.
    overall = [h for h in hits if h.get('system', 'histologic3') != 'nuclear3']
    if overall:
        hits_for_score = overall
    else:
        hits_for_score = hits
    priority = [h for h in hits_for_score if h.get('priority')]
    if priority:
        hits_for_score = priority
    result = resolve(hits_for_score)
    if any(h['value'] is None for h in hits_for_score):
        result.update(value=None, status='uncertain')
    result['evidence'] = hits
    if result['value']:
        system, value = result['value'].split(':')
        result.update(system=system, ordinal=int(value), bounds=list(GRADE_RANGES[system]))
    else:
        result.update(system=None, ordinal=None, bounds=None)
    return result


def extract(text: str) -> dict:
    lines = sections(text)
    dx = diagnosis_scope(lines)
    evidence = "\n".join(t for s, t in lines if s not in ("site", "excluded"))
    findings = {key: finding(evidence, key) for key in FINDINGS}
    return dict(diagnosis_text=dx, subtype=entity(dx), broad=broad_type(dx),
                malignancy=malignancy(dx), grade=grade(lines), findings=findings,
                sections=[dict(section=s, text=t) for s, t in lines])


def reference(text: str, *, structured: dict | None = None, cohort: dict | None = None) -> dict:
    result = extract(text)
    result['source'] = 'case_report'
    if structured is not None:
        primary = str(structured.get('primary_diagnosis') or '')
        sub = str(structured.get('histologic_subtype') or '')
        result['subtype'] = entity(sub) if sub else entity(primary)
        result['broad'] = broad_type(str(structured.get('broad_type') or primary))
        result['malignancy'] = malignancy('Malignancy: '+str(structured.get('malignancy_status') or ''))
        result['grade'] = grade([], explicit=str(structured.get('grade') or ''))
        for key, field in [('invasion', 'invasive_status'), ('necrosis', 'necrosis')]:
            result['findings'][key] = structured_binary(str(structured.get(field) or ''), key)
        result['source'] = 'physician_structured_reference'
    if cohort is not None:
        result['report_diagnosis'] = {k: result[k] for k in DIAGNOSIS_WEIGHTS}
        result['broad'] = fact(cohort['broad_type'], 'TCGA cohort benchmark label')
        result['subtype'] = fact(cohort['subtype'], 'TCGA cohort benchmark label')
        result['malignancy'] = fact(cohort['malignancy'], 'TCGA cohort benchmark label')
        result['cohort_aliases'] = list(cohort['subtype_aliases'])
        result['source'] = 'tcga_cohort_diagnosis_with_case_report_findings'
    return result


def score_field(pred: dict, ref: dict) -> dict:
    applicable = ref['status'] == 'asserted' and ref['value'] is not None
    covered = pred['status'] == 'asserted' and pred['value'] is not None
    return dict(prediction=pred['value'], reference=ref['value'], applicable=applicable,
                covered=covered, score=float(covered and pred['value'] == ref['value']) if applicable else None,
                prediction_status=pred['status'], reference_status=ref['status'])


def cohort_subtype(pred: dict, ref: dict) -> dict:
    result = score_field(pred['subtype'], ref['subtype'])
    aliases = [ref['subtype']['value'], *ref.get('cohort_aliases', [])]
    matched = []
    for c in clauses(pred['diagnosis_text']):
        t = normalize(c)
        if HYPOTHETICAL.search(c):
            continue
        for a in aliases:
            if contains(t, a):
                pos = t.find(normalize(a))
                if polarity(t, pos, pos+len(normalize(a))) == 'present':
                    matched.append(a)
    # Conflicting asserted lineages/entities do not get credit for listing options.
    incompatible_in_situ = ('invasive' in normalize(ref['subtype']['value']) and
                           'in situ' in normalize(pred['diagnosis_text']) and
                           not re.search(r'(?<!non )\binvasive\b', normalize(pred['diagnosis_text'])))
    result.update(score=float(bool(matched) and pred['subtype']['status'] != 'conflict' and not incompatible_in_situ),
                  matched_aliases=matched, target_kind='cohort_compatibility_not_leaf_subtype')
    return result


def rouge(text: str, ref: str, n: int = 1) -> float:
    a, b = normalize(text).split(), normalize(ref).split()
    ac = Counter(tuple(a[i:i+n]) for i in range(len(a)-n+1))
    bc = Counter(tuple(b[i:i+n]) for i in range(len(b)-n+1))
    denominator = sum(ac.values())+sum(bc.values())
    return 2*sum((ac & bc).values())/denominator if denominator else 0.0


def evaluate(report: str, ref: dict, *, reference_text: str = '', diagnosis_only: bool = False,
             narrative_reference: bool = True) -> dict:
    pred = extract(report)
    fields = {key: score_field(pred[key], ref[key]) for key in DIAGNOSIS_WEIGHTS}
    if ref.get('cohort_aliases') is not None:
        fields['subtype'] = cohort_subtype(pred, ref)
    # The primary score is exact agreement, with no sibling/lineage partial credit.
    available = {k: w for k, w in DIAGNOSIS_WEIGHTS.items() if fields[k]['applicable']}
    metrics = {k: fields[k]['score'] for k in fields}
    metrics['diagnosis'] = sum(metrics[k]*w for k, w in available.items())/sum(available.values()) if available else None
    fields.update({k: score_field(pred['findings'][k], ref['findings'][k]) for k in ('invasion', 'necrosis')})
    g = score_field(pred['grade'], ref['grade'])
    same_system = pred['grade']['system'] == ref['grade']['system'] and pred['grade']['system'] is not None
    g.update(system=ref['grade']['system'], prediction_system=pred['grade']['system'],
             ordinal_reference=ref['grade']['ordinal'], ordinal_prediction=pred['grade']['ordinal'],
             compatible_coverage=bool(g['covered'] and same_system), ordinal_score=None)
    if g['applicable']:
        lo, hi = ref['grade']['bounds']
        g['ordinal_score'] = 1-abs(pred['grade']['ordinal']-ref['grade']['ordinal'])/(hi-lo) if same_system and g['covered'] else 0.0
        g['always_high_exact'] = float(ref['grade']['ordinal'] == hi)
        g['always_high_ordinal'] = 1-abs(hi-ref['grade']['ordinal'])/(hi-lo)
    fields['grade'] = g
    metrics.update(invasion=fields['invasion']['score'], necrosis=fields['necrosis']['score'],
                   grade_exact=g['score'], grade_ordinal=g['ordinal_score'],
                   grade_coverage=float(g['compatible_coverage']) if g['applicable'] else None)
    if diagnosis_only:
        for k in ('invasion', 'necrosis', 'grade'):
            fields[k]['capability_assessed'] = False
        for k in ('invasion', 'necrosis', 'grade_exact', 'grade_ordinal', 'grade_coverage'):
            metrics[k] = None
    documented = {k: f for k, f in ref['findings'].items() if f['status'] == 'asserted' and f['value'] is not None}
    comparable = {k: (f['value'], pred['findings'][k]['value']) for k, f in documented.items()}
    tp = sum(a == b for a, b in comparable.values())
    asserted = sum(b is not None for a, b in comparable.values())
    positives = {k for k, f in documented.items() if k in CRITICAL and f['value'] == 'present'}
    assessed = [f for f in fields.values() if f['applicable']]
    metrics.update(
        reference_coverage=sum(f['covered'] for f in assessed)/len(assessed) if assessed else None,
        morphology_fact_f1=2*tp/(len(documented)+asserted) if documented else None,
        critical_positive_recall=sum(pred['findings'][k]['value'] == 'present' for k in positives)/len(positives) if positives else None,
        contradiction_rate=sum(b is not None and a != b for a, b in comparable.values())/len(documented) if documented else None,
        rouge1_f1=rouge(report, reference_text, 1) if narrative_reference and reference_text.strip() else None,
        rouge2_f1=rouge(report, reference_text, 2) if narrative_reference and reference_text.strip() else None,
    )
    if diagnosis_only:
        for k in ('reference_coverage', 'morphology_fact_f1', 'critical_positive_recall', 'contradiction_rate', 'rouge1_f1', 'rouge2_f1'):
            metrics[k] = None
    qa = []
    if re.search(r'<think>', report, re.I) and not re.search(r'</think>', report, re.I):
        qa.append('prediction.unfinished_thinking_not_final_diagnosis')
    for side, obj in [('prediction', pred), ('reference', ref)]:
        for key in ('broad', 'subtype', 'malignancy', 'grade'):
            if obj[key]['status'] != 'asserted':
                qa.append(f"{side}.{key}.{obj[key]['status']}")
    return dict(schema=VERSION, metrics=metrics, fields=fields, prediction=pred,
                reference=ref, diagnosis_only=diagnosis_only,
                report_sha256=hashlib.sha256(report.encode()).hexdigest(),
                quality_denominators=dict(documented_findings=len(documented), positive_critical=len(positives),
                                          eligible_structured_fields=len(assessed)),
                unverified_prediction_findings=[k for k, f in pred['findings'].items() if f['value'] is not None and k not in documented],
                qa_flags=qa)
