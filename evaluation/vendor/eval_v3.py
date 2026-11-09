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
from . import grade_v4
GRADE_RANGES = grade_v4.RANGES

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

# Inflammatory-cell infiltration does not establish tumour invasion. Exclude
# only the matched phrase, retaining separate tumour-invasion statements.
IMMUNE_INFILTRATION = re.compile(
    r"\b(?:lymphocytic|inflammatory)(?:\s+(?:cell|cells))?\s+infiltrat(?:e|es|ing|ion|ive)\b", re.I)



# Keep shared negative lists intact. Only separate coordinated finding statements
# when the left subject has its own overt copula and the right finding precedes
# another copula. Diagnosis and grade extraction retain their original scope.
_FINDING_TERM = re.compile("|".join(FINDINGS.values()), re.I)
_FINDING_COPULA = re.compile(r"\b(?:is|are|was|were)\b", re.I)
_FINDING_EXPLICIT_STATE = re.compile(
    r"\s*(?:present|absent|frequent|rare|prominent|abundant|minimal|predominant|evident|observed|identified|seen|"
    r"not (?:assessable|seen|identified|present|detected|observed|demonstrated|recognized))\b", re.I)
_FINDING_DEPENDENT = re.compile(r"\b(?:although|though|while|whereas|because|as|if|unless|which|whether|respectively)\b", re.I)
_FINDING_COORDINATOR = re.compile(r",\s*(?:and\s+)?|\s+and\s+", re.I)


def finding_clauses(text: str) -> list[str]:
    result = []
    for sentence in clauses(text):
        first = _FINDING_TERM.search(sentence)
        leading = sentence[:first.start()] if first else sentence
        # A leading uncertainty or conditional can govern both predicates.
        # A hypothetical qualifier anywhere already excludes the original clause.
        if HYPOTHETICAL.search(sentence) or UNKNOWN.search(leading) or UNCERTAIN.search(leading) or _FINDING_COPULA.search(leading) or re.search(r"\b(?:if|unless|whether|that|cannot|unclear)\b", leading, re.I):
            result.append(sentence)
            continue
        start = 0
        for cut in _FINDING_COORDINATOR.finditer(sentence):
            prefix = sentence[:cut.start()]
            # Do not split a parenthetical list into independent assertions.
            if prefix.count("(") != prefix.count(")") or prefix.count("[") != prefix.count("]"):
                continue
            left, right = sentence[start:cut.start()], sentence[cut.end():]
            terms = list(_FINDING_TERM.finditer(left))
            left_predicate = _FINDING_COPULA.search(left, terms[-1].end()) if terms else None
            if left_predicate is None or not _FINDING_EXPLICIT_STATE.match(left, left_predicate.end()):
                continue
            if _FINDING_DEPENDENT.search(left):
                continue
            first_term = _FINDING_TERM.search(right)
            first_copula = _FINDING_COPULA.search(right)
            if first_term is None or first_copula is None or first_term.end() > first_copula.start():
                continue
            if _FINDING_DEPENDENT.search(right[:first_term.start()]):
                continue
            right_state = _FINDING_EXPLICIT_STATE.match(right, first_copula.end())
            if right_state is None:
                continue
            remainder = right[right_state.end():]
            if UNKNOWN.search(remainder) or UNCERTAIN.search(remainder) or _FINDING_DEPENDENT.search(remainder) or re.search(r"\b(?:both|neither)\b", remainder, re.I):
                continue
            # A relative/modifying clause is not a separate finding predicate.
            subject_tail = right[first_term.end():first_copula.start()]
            if not re.fullmatch(r"\s*(?:(?:architecture|growth|pattern|patterns|formation|features|activity)\s*)?", subject_tail, re.I):
                continue
            result.append(left.strip())
            start = cut.end()
        result.append(sentence[start:].strip())
    return result


def _independent_later_assessment(clause, end, qualifier_start):
    between = clause[end:qualifier_start]
    state = r"\s*(?:(?:is|are|was|were)\s+)?(?:present|absent|frequent|rare|prominent|abundant|minimal|predominant|evident|observed|identified|seen|not (?:seen|identified|present|detected|observed|demonstrated|recognized))\b"
    subject = r"\s*(?:(?:the|definitive|pathologic|pathological|precise|final)\s+)*(?:staging|stage|grading|grade|diagnosis|margin(?:s| status)?|nodal status|primary site)\s*"
    for cut in _FINDING_COORDINATOR.finditer(between):
        left, right = between[:cut.start()], between[cut.end():]
        if (re.match(state, left, re.I)
                and not _FINDING_DEPENDENT.search(left)
                and not re.search(r"[()\[\]]", left)
                and re.fullmatch(subject, right, re.I)):
            return True
    return False


def _uncertainty_has_other_subject(clause: str, end: int, qualifier_start: int) -> bool:
    """Do not attach a later explanatory diagnosis or artefact to an observation."""
    between = clause[end:qualifier_start]
    # Recognize named diagnostic/technical subjects only. Locatives and anaphora
    # such as '(in these fields, its presence cannot be excluded)' still qualify
    # the finding, even though the parenthetical contains additional words.
    other_subject = r"(?:(?:the|a|an)\s+)?(?:(?:fixation|processing|staining)\s+art[ie]facts?|(?:low-grade|diagnostic)\s+mimics?|(?:presence|absence) of (?:a )?basal cell(?:s| layer))"
    opening = between.rfind('(')
    if opening > between.rfind(')'):
        inner = between[opening + 1:].strip()
        subject = re.split(r",|\b(?:though|although|whereas)\b", inner, flags=re.I)[-1].strip()
        if re.fullmatch(other_subject, subject, re.I):
            return True
        if (re.match(r"\s*(?:is|are|was|were)\s+(?:clearly\s+)?present\b", between, re.I)
                and re.fullmatch(r"(?:its|the)\s+(?:extent|size|distribution)", subject, re.I)):
            return True
    # Described morphology may suggest a diagnosis/grade whose identity is
    # uncertain; that does not make the described morphology uncertain.
    if re.search(r"\b(?:raises?|suggests?|supports?|indicates?)\b.*\b(?:possibility|diagnosis|grade|gleason|component)\b", between, re.I):
        return True
    # A dependent clause with its own named subject keeps its own qualifier.
    # Anaphoric subjects ('this', 'both findings') can refer back to the finding.
    for cut in re.finditer(r"\b(?:though|although|whereas)\b", between, re.I):
        subject = between[cut.end():].strip(' ,')
        if re.fullmatch(other_subject, subject, re.I):
            return True
    return False


def _finding_nominal_list(text: str) -> bool:
    """A bounded noun list, without an independent predicate or relative clause."""
    residual = _FINDING_TERM.sub("", text)
    residual = re.sub(
        r"\b(?:and|or|a|an|the|tumor|tumour|distant|regional|lymph|node|nodes|stromal|"
        r"definite|overt|focal|extensive|geographic|atypical|multipolar|architecture|growth|"
        r"patterns?|formation|features|activity|capsular|venous|sinusoidal|lymphatic|"
        r"vascular|blood|sanguineous|angiovascular|mediastinal|fat|cytologic|atypia)\b|[,/:\s]", "", residual, flags=re.I)
    return not residual


def _negative_lvi_branches(clause: str) -> set[str]:
    """Vessel types explicitly denied by a bounded, unqualified assertion."""
    # Parenthetical vessel sizes describe the same lymphatic/blood branches.
    clause = re.sub(r"\s*\((?:large|small) vessels?\)", "", clause, flags=re.I)
    # A complete negative assertion may follow an unrelated complete finding.
    # Do not strip a leading uncertainty qualifier or a conditional premise.
    joined = re.search(r"\s+and\s+(?=no\s+(?:evidence\s+of\s+)?)", clause, re.I)
    if (joined and re.search(r"\b(?:is|are|was|were)\b", clause[:joined.start()], re.I)
            and not UNCERTAIN.search(clause[:joined.start()])
            and not UNKNOWN.search(clause[:joined.start()])
            and not re.search(r"\b(?:if|whether|unless)\b", clause[:joined.start()], re.I)
            and not HYPOTHETICAL.search(clause[:joined.start()])):
        tail_branches = _negative_lvi_branches(clause[joined.end():])
        if tail_branches:
            return tail_branches
    subject = r"(?P<subject>(?:(?:lymphovascular|angiolymphatic|lymphatic|blood|sanguineous|venous|vascular|angiovascular|invasion|and|or|of)\b[\s,/]*)+)"
    location = r"(?:\s+in\s+(?:the\s+)?(?:(?:examined|sampled|available|provided|submitted|these|this)\s+)+(?:fields?|sections?|material|samples?|tissue|ROIs?|specimens?))?"
    match = re.fullmatch(
        r"\s*(?:(?:there\s+is\s+)?no\s+(?:evidence\s+of\s+)?" + subject
        + r"(?:\s+(?:is|are)\s+(?:identified|seen|present|detected|observed))?|"
        + subject.replace('subject', 'post_subject')
        + r"\s*:?\s*(?:(?:is|are|was|were)\s+)?(?:absent|not\s+(?:identified|seen|present|detected|observed)))" + location
        + r"(?:,\s*(?:although|though)\s+sampling\s+is\s+limited)?\s*", clause, re.I)
    if not match:
        # Independently stated negatives may jointly cover both vessel types.
        parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", clause, flags=re.I)
        if len(parts) > 1:
            branches = [_negative_lvi_branches(part) for part in parts]
            if all(branches):
                return set().union(*branches)
        return set()
    covered = match.group('subject') or match.group('post_subject')
    if re.search(r"\b(?:lymphovascular|angiolymphatic)\b", covered, re.I):
        return {'lymphatic', 'blood'}
    branches = set()
    if re.search(r"\blymphatic\b", covered, re.I):
        branches.add('lymphatic')
    if re.search(r"\b(?:blood|sanguineous|venous|angiovascular)\b", covered, re.I):
        branches.add('blood')
    return branches


def _negative_is_narrower_than_finding(clause: str, start: int, end: int, key: str) -> bool:
    """A negative subtype/extent statement cannot negate the broader concept."""
    prefix = re.split(r"[,;]|\b(?:and|but)\b", clause[:start], flags=re.I)[-1]
    # 'Prominent nucleoli' is itself a qualified vocabulary concept. A degree
    # modifier on the preceding alternative does not qualify this entire term:
    # 'no significant pleomorphism or prominent nucleoli'. Other coordinated
    # heads can share their modifier ('no significant papillary or glandular
    # architecture'), so do not split every 'or'.
    if key == 'nucleoli':
        prefix = re.split(r"\bor\b", prefix, flags=re.I)[-1]
    if re.search(r"\b(?:extensive|geographic|dominant|significant|abundant|conspicuous)\s+(?:[\w-]+\s+){0,2}$", prefix, re.I):
        return True
    if key == 'mitoses' and re.search(r"\b(?:atypical|multipolar)\s+(?:[\w-]+\s+){0,2}$", prefix, re.I):
        return True
    if key == 'necrosis' and re.search(r"\bpseudo-?palisading\s*$", prefix, re.I):
        return True
    if key == 'invasion':
        # Unavailable invasion measurements or interfaces do not deny invasion.
        if re.search(r'\b(?:depth|extent|pattern|interface) of\s*$', prefix, re.I):
            return True
        if re.match(r'\s+(?:interface|depth|extent|pattern)\b', clause[end:], re.I):
            return True
        sites = r"(?:muscularis propria|detrusor|muscle|lamina propria|submucosa[l]?|sinusoidal|venous|renal vein|capsular|mediastinal fat|pleural|serosal|vascular|lymphatic|perineural)"
        if re.search(r"\b" + sites + r"(?:-level)?\s*(?:\([^)]*\)\s*)?$", prefix, re.I):
            return True
        if re.match(r"\s+(?:of|into|through|involving)\s+(?:the\s+)?" + sites + r"\b", clause[end:], re.I):
            return True
    if key == 'lymphovascular_invasion':
        if len(_negative_lvi_branches(clause)) == 2:
            return False
        if re.match(r"lymphatic invasion\b", clause[start:end], re.I):
            return True
        if re.search(r"\b(?:lymphatic|sanguineous|blood|venous)\s*$", prefix, re.I):
            return True
    return False


def finding_polarity(clause: str, start: int, end: int) -> str:
    value = polarity(clause, start, end)
    # Naming items that require assessment does not assert their presence.
    assessment = re.search(r"(?:\b(?:requires?|needs?)\s+(?:further\s+)?|^\s*(?:further\s+)?)assessment of\s+", clause[:start], re.I)
    if assessment and _finding_nominal_list(clause[assessment.end():end]):
        # A subsequent relative clause or repeated subject may explicitly report
        # the observation. 'Requires assessment' must not erase that assertion.
        following = clause[end:]
        relative = re.search(r"\s*,?\s*(?:which|that)\s+(?P<copula>is|are|was|were)\s+(?:(?:clearly|definitely)\s+)?", following, re.I)
        if relative and not _finding_nominal_list(following[:relative.start()]):
            relative = None
        if (relative and relative.group('copula').lower() in {'is', 'was'}
                and not re.fullmatch(r"\s*(?:(?:architecture|growth|pattern|formation|activity)\s*)?", following[:relative.start()], re.I)):
            # A singular predicate on the last head does not assert every item
            # in the assessment list ('necrosis and mitotic activity, which is rare').
            relative = None
        repeated = re.search(re.escape(clause[start:end]), clause[assessment.end():start], re.I)
        direct = re.match(r"\s*(?:is|are|was|were)\s+(?:(?:clearly|definitely)\s+)?", following, re.I)
        explicit = relative or (direct if repeated else None)
        if not explicit or not _FINDING_EXPLICIT_STATE.match(following, explicit.end()):
            return "uncertain"
        observation = clause[start:end] + ' ' + following[explicit.end():]
        return finding_polarity(observation, 0, end - start)
    # A request to confirm presence/absence is not an observation of presence.
    confirmation = re.match(r"\s*,?\s*confirmation of (?:the )?(?:absence|presence) of\s+", clause, re.I)
    if (confirmation and confirmation.end() <= start
            and _finding_nominal_list(clause[confirmation.end():end])
            and re.match(r"\s+(?:requires?|needs?)\s+(?:full\s+slide\s+review|further\s+(?:review|assessment|evaluation))\b", clause[end:], re.I)):
        return "uncertain"
    uncertainty = r"\bnot (?:identified|seen)\s*/\s*(?:assessable|evaluable)\b|\bcannot be (?:confirmed|excluded|assessed|evaluated|determined)\b"
    for qualifier in re.finditer(uncertainty, clause, re.I):
        if qualifier.start() < end:
            continue
        if _uncertainty_has_other_subject(clause, end, qualifier.start()):
            continue
        if not _FINDING_TERM.search(clause[end:qualifier.start()]):
            if _independent_later_assessment(clause, end, qualifier.start()):
                continue
            return "uncertain"
        # An assessment qualifier also governs a shared subject list, but not
        # another earlier complete assertion such as "necrosis is present".
        subject = re.sub(r"\b(?:is|are|was|were)\s*$", "", clause[:qualifier.start()], flags=re.I)
        # Parentheses may repeat a bare finding label without asserting it.
        if _finding_nominal_list(subject.replace('(', '').replace(')', '')):
            return "uncertain"
    if value == "uncertain" or re.search(r"\brespectively\b", clause, re.I):
        return value
    # Negation may govern a long, uninterrupted noun list beyond the legacy
    # character window. A finite predicate breaks this shared scope.
    negative_list = re.match(r"\s*(?:there\s+is\s+)?no\s+(?:evidence\s+of\s+)?", clause, re.I)
    if negative_list and _finding_nominal_list(clause[negative_list.end():end]):
        return "absent"
    absence = r"(?:absent\b|not (?:seen|identified|present|detected|observed|demonstrated|recognized)\b|negative(?=\s*(?:[,;:.)]|$)))"
    tail = clause[end:]
    # Some vocabulary entries are adjectives; retain their nominal head when
    # locating a predicate (for example, "papillary architecture is absent").
    direct = re.match(r"\s*:?[ \t]*(?:(?:architecture|growth|patterns?|formation|features|activity)\s+(?:is|are|was|were)\s+|(?:(?:is|are|was|were)\s+)?)" + absence, tail, re.I)
    # An absence predicate does not negate findings stated elsewhere in the same
    # observation, or a relationship such as being seen together.
    mixed = r"\b(?:present|positive|seen|identified|detected|observed|demonstrated|frequent|rare|prominent|abundant|minimal|predominant|evident|visible|detectable|together|simultaneously|concurrently|concomitantly|cooccur|coexist|associated|association|combination|conjunction|same|except|with)\b"
    nonuniversal = r"\b(?:all|every)\b"
    if (direct and not re.search(mixed, tail[direct.end():], re.I)
            and not (re.search(r"\bnot\b", direct.group(), re.I) and re.search(nonuniversal, tail[direct.end():], re.I))
            and not _FINDING_DEPENDENT.search(tail[direct.end():])):
        return "absent"
    # A shared negative predicate governs each member of a coordinated subject.
    predicate = _FINDING_COPULA.search(tail)
    if predicate:
        between = tail[:predicate.start()]
        between = re.sub(r"^\s*(?:architecture|growth|patterns?|formation|features|activity)\s*(?=,|and\b|or\b)", "", between, flags=re.I)
        nominal_subject = _finding_nominal_list(clause[:end+predicate.start()])
        negative = re.match(r"\s*" + absence, tail[predicate.end():], re.I)
        if (nominal_subject and negative
                and not (re.search(r"\bnot\b", negative.group(), re.I) and re.search(nonuniversal, tail[predicate.end()+negative.end():], re.I))
                and not re.search(mixed, tail[predicate.end()+negative.end():], re.I)
                and not _FINDING_DEPENDENT.search(tail[predicate.end()+negative.end():])
                and re.match(r"\s*(?:,|and\b|or\b)", between, re.I)
                and _FINDING_TERM.search(between)
                and not _FINDING_DEPENDENT.search(between)
                and not re.search(r"[()\[\]]", between)
                and not re.search(r"\b(?:with|of|in|that|which|where|when|associated|due|within|if|whether)\b", between, re.I)
                and re.match(r"\s*" + absence, tail[predicate.end():], re.I)):
            return "absent"
    return value


def finding(text: str, key: str) -> dict:
    hits = []
    negative_branches = {}
    for c in finding_clauses(text):
        if HYPOTHETICAL.search(c):
            continue
        if key == "invasion":
            # General stromal invasion is not the same field as LVI/PNI.
            c = re.sub(r"(?:lymphovascular|angiolymphatic|vascular|lymphatic|perineural|neural|capsular)(?: vessels?)? (?:penetration\s*/\s*)?(?:invasion|infiltrat(?:ion|ive|ing|es?))\b", "", c, flags=re.I)
        if key == 'lymphovascular_invasion':
            for branch in _negative_lvi_branches(c):
                negative_branches[branch] = c
        immune_spans = [m.span() for m in IMMUNE_INFILTRATION.finditer(c)] if key == "invasion" else []
        for m in re.finditer(FINDINGS[key], c, re.I):
            # A bare field label repeated by its body is not another assertion.
            tail = c[m.end():]
            colon = re.match(r"\s*:\s*", tail)
            if not c[:m.start()].strip() and colon:
                body = tail[colon.end():]
                repeated = re.search(FINDINGS[key], body, re.I)
                prefix = re.sub(r"^\s*no\s+", "", body[:repeated.start()], flags=re.I) if repeated else None
                if repeated and _finding_nominal_list(prefix):
                    continue
            if any(start <= m.start() and m.end() <= end for start, end in immune_spans):
                continue
            p = finding_polarity(c, m.start(), m.end())
            if key == 'lymphovascular_invasion' and len(_negative_lvi_branches(c)) == 2:
                p = 'absent'
            if key == 'invasion' and re.search(r"non[ -]?invasive", c, re.I):
                p = 'absent'
            if key == 'lymphovascular_invasion' and p == 'absent':
                for branch in _negative_lvi_branches(c):
                    negative_branches[branch] = c
            if p == 'absent' and _negative_is_narrower_than_finding(c, m.start(), m.end(), key):
                p = 'uncertain'
            hits.append(dict(value=p if p != "uncertain" else None, text=c,
                             status="asserted" if p != "uncertain" else "uncertain"))
    # Separate explicit denials of the two vessel types jointly deny LVI.
    # Keep isolated subtype denials uncertain at the broader finding level.
    if len(negative_branches) == 2 and not any(h['value'] == 'absent' for h in hits):
        hits.append(dict(value='absent', status='asserted',
                         text='; '.join(dict.fromkeys(negative_branches.values()))))
    return resolve(hits, binary=True)


def structured_binary(text: str, key: str) -> dict:
    if not text or UNKNOWN.search(text) or UNCERTAIN.search(text):
        return fact(None, text)
    if re.search(r"\b(?:absent|negative|not identified|not seen|noninvasive|non invasive|non-invasive|in situ)\b", text, re.I):
        return fact("absent", text)
    if re.search(r"\b(?:present|positive|invasive|invasion|focal)\b", text, re.I):
        return fact("present", text)
    return finding(text, key)


def grade(lines, explicit=None):
    return grade_v4.extract("\n".join(line for _, line in lines), explicit=explicit)


def extract(text: str) -> dict:
    lines = sections(text)
    dx = diagnosis_scope(lines)
    evidence = "\n".join(t for s, t in lines if s not in ("site", "excluded"))
    findings = {key: finding(evidence, key) for key in FINDINGS}
    return dict(diagnosis_text=dx, subtype=entity(dx), broad=broad_type(dx),
                malignancy=malignancy(dx), grade=grade_v4.extract(text), findings=findings,
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
    g = grade_v4.score(pred['grade'], ref['grade'])
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
