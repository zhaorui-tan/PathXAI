"""Diagnostic text scope and assertion qualifiers; no case or method identifiers.

Candidate repair of deterministic parsing. This module does not change the
terminology, metric denominators, or extraction of morphology/grade fields.
"""
from __future__ import annotations

import re
from functools import lru_cache
from . import eval_v3 as legacy
from . import subtype_ontology_v4 as ontology

ABBREVIATIONS = re.compile(r'\b(?:e\.g\.|i\.e\.|vs\.|Dr\.|Mr\.|Mrs\.)', re.I)
EXTRA_HEADINGS = {
    'pathologic diagnosis': 'diagnosis', 'pathological diagnosis': 'diagnosis',
    'histopathologic diagnosis': 'diagnosis', 'histological diagnosis': 'diagnosis',
    'histologic diagnosis': 'diagnosis', 'pathology diagnosis': 'diagnosis',
    'final addendum diagnosis': 'diagnosis', 'addendum diagnosis': 'diagnosis',
    'addendum discussion': 'comment', 'microscopic': 'morphology',
    'gross description': 'excluded', 'gross examination': 'excluded',
    'gross findings': 'excluded', 'macroscopic description': 'excluded',
    'macroscopic findings': 'excluded',
    'intraoperative consultation': 'intraoperative', 'intraoperative diagnosis': 'intraoperative',
    'frozen section': 'frozen', 'frozen section diagnosis': 'frozen',
    'intraoperative frozen section diagnosis': 'frozen',
    'clinical history': 'excluded', 'history': 'excluded',
    'clinical information': 'excluded', 'past medical history': 'excluded',
    'clinical findings': 'excluded',
    'previous diagnosis': 'excluded', 'prior diagnosis': 'excluded',
    'patient history': 'excluded', 'synoptic report': 'synoptic',
    'synoptic summary': 'synoptic', 'synoptic diagnosis': 'synoptic',
    'critical findings': 'findings', 'diagnosis comment': 'comment',
    'microscopic exam': 'morphology', 'microscopic examination': 'morphology',
    'ancillary tests ordered': 'excluded', 'ancillary test s ordered': 'excluded',
    'specimen source': 'site', 'specimen submitted': 'site',
    'analyte specific reagent asr disclaimer': 'excluded',
    'malignancy': 'other_field', 'grade': 'grade', 'stage': 'grade',
    'prognostic features': 'grade',
}
SUBTYPE_FIELD = re.compile(
    r'^(?:(?:histologic(?:al)?|histopathologic(?:al)?)\s+)?'
    r'(?:subtype|(?:tumou?r\s+)?type(?:\s*/\s*subtype)?)\s*:\s*(.*)$', re.I)
HEADINGS = {legacy.normalize(k): v for k, v in {**legacy.HEADINGS, **EXTRA_HEADINGS}.items()}
LEAF_FIELDS = ('subtype', 'histologic subtype', 'histological subtype',
               'histologic type', 'histological type', 'tumor type', 'tumour type',
               'tumor histologic type', 'tumour histologic type',
               'tumor histological type', 'tumour histological type')


def _edit_distance(a, b):
    costs = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        next_costs = [i]
        for j, y in enumerate(b, 1):
            next_costs.append(min(next_costs[-1]+1, costs[j]+1, costs[j-1]+(x != y)))
        costs = next_costs
    return costs[-1]


@lru_cache(maxsize=32768)
def heading_kind(title):
    """Limited OCR tolerance for section labels only; never repair disease words."""
    label = legacy.normalize(title)
    if label in HEADINGS:
        return HEADINGS[label]
    if label in LEAF_FIELDS:
        return 'subtype_field'
    tokens = label.split()
    if not 1 <= len(tokens) <= 5:
        return None
    candidates = []
    for key, kind in [*HEADINGS.items(), *((x, 'subtype_field') for x in LEAF_FIELDS)]:
        words = key.split()
        if len(tokens) != len(words):
            continue
        # Do not confuse short distinct labels such as size/site or grade/stage.
        if not any(len(w) >= 7 for w in words):
            continue
        limits = [2 if len(b) >= 8 else 1 if len(b) >= 7 or b in ('final', 'frozen', 'gross') else 0
                  for b in words]
        if any(abs(len(a)-len(b)) > limit for a,b,limit in zip(tokens,words,limits)):
            continue
        distances = [_edit_distance(a, b) for a, b in zip(tokens, words)]
        if sum(distances) <= 2 and all(d <= limit for d,limit in zip(distances,limits)):
            candidates.append((sum(distances), kind))
    if not candidates:
        return None
    best = min(d for d,k in candidates)
    kinds = {k for d,k in candidates if d == best}
    return kinds.pop() if len(kinds) == 1 else None


INLINE_LABELS = sorted({*HEADINGS, *LEAF_FIELDS}, key=len, reverse=True)
INLINE_PATTERN = re.compile(r'(?<=[.;])\s+(?=(?:' + '|'.join(
    re.escape(x).replace(r'\ ', r'\s+') for x in INLINE_LABELS) + r')\s*:)', re.I)
SPECIMEN = re.compile(r'\b(?:bladder|ureter|urethral|prostate|kidney|renal|breast|lung|'
                      r'colon|rectum|brain|nodes?|margin|uterus|ovaries|'
                      r'cervix|adrenal|liver)\b', re.I)

WRAPPED_TERMS = {}
for _, alias, _ in ontology.ALIASES:
    words=legacy.normalize(alias).split()
    for split in range(1,len(words)):
        WRAPPED_TERMS.setdefault(words[split-1],set()).add((tuple(words[:split]),tuple(words[split:])))


def join_wrapped_terms(text):
    """Restore line wrapping inside an existing term; do not repair OCR words."""
    lines=[]
    for line in str(text).splitlines():
        if lines and lines[-1].strip() and line.strip() and not re.search(r'[.!?;:]\s*$',lines[-1]):
            previous=legacy.normalize(lines[-1]).split()
            following=legacy.normalize(line).split()
            if previous and following and not re.match(r'^\s*[-*]\s+',line):
                pairs=WRAPPED_TERMS.get(previous[-1],())
                if any(tuple(previous[-len(left):])==left and tuple(following[:len(right)])==right
                       for left,right in pairs):
                    lines[-1] += ' '+line.strip()
                    continue
        lines.append(line)
    return '\n'.join(lines)


def _specimen_title(line):
    label = line.strip().rstrip(':;+-«').strip()
    if not SPECIMEN.search(label) or len(label) > 120:
        return False
    if re.search(r'\b(?:no|carcinoma|tumou?r|identified|invades|shows|free|benign|negative)\b', label, re.I):
        return False
    return bool(re.search(r'\([^()]{1,3}\)\s*[:;+«-]?', line, re.I) or
                (line.endswith(':') and len(label.split()) <= 12))


def sentences(text):
    """Preserve abbreviation dots, decimal points and semicolon-list context."""
    protected = ABBREVIATIONS.sub(lambda m: m[0].replace('.', '\uff0e'), text)
    protected = re.sub(r'(?<=\d)\.(?=\d)', '\uff0e', protected)
    return [part.replace('\uff0e', '.').strip() for part in
            re.split(r'(?<=[.!?])\s+|\n', protected) if part.strip()]


def sections(text):
    text = re.sub(r'<think>.*?(?:</think>|$)', '', str(text), flags=re.I | re.S)
    text = join_wrapped_terms(text)
    current, result, list_index = 'unsectioned', [], None
    specimen, synoptic_specimen, frozen_resume = '', '', False
    for raw in INLINE_PATTERN.sub('\n', text).splitlines():
        is_item = bool(re.match(r'^\s*(?:\d+[.)]|[-*])\s+', raw))
        line = re.sub(r'^\s*(?:#{1,6}\s*|\d+[.)]\s*|[-*]\s+)', '', raw)
        line = line.replace('**', '').replace('__', '').strip()
        if not line:
            continue
        # Process headings before appending list items. A new section cannot
        # attach to the previous section, even when its heading is a bullet.
        if line.startswith('|'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            if len(cells) == 2 and heading_kind(cells[0].rstrip(':')):
                line = cells[0].rstrip(':') + ': ' + cells[1]
        title, sep, body = line.partition(':')
        anatomical_title=re.sub(r'\bprostatectomy\b','prostate',title,flags=re.I)
        if re.search(r'\bsynoptic (?:report|summary)\s*$',anatomical_title,re.I) and SPECIMEN.search(anatomical_title):
            current, synoptic_specimen, specimen, list_index = 'synoptic', anatomical_title, '', None
            continue
        heading = heading_kind(title)
        leadin = bool(line.endswith(':') and re.search(
            r'\b(?:differential|considerations|possibilities)\b', line, re.I))
        if heading:
            if heading == 'subtype_field':
                if current not in ('excluded', 'frozen', 'intraoperative', 'site'):
                    value=body.strip()
                    source=synoptic_specimen or specimen
                    result.append((current, 'Subtype: ' + (source+': ' if source and value else '') + value))
                    if value:
                        specimen=''
                continue
            if heading == 'frozen':
                frozen_resume = current == 'diagnosis' or (current == 'frozen' and frozen_resume)
            else:
                frozen_resume = False
            current, list_index = heading, None
            if heading in ('diagnosis','excluded','intraoperative','site','summary'):
                synoptic_specimen=''
            if heading not in ('diagnosis', 'frozen'):
                specimen = ''
            if body.strip():
                result.append((current, body.strip()))
                if leadin:
                    list_index = len(result) - 1
            continue
        if (result and result[-1][1] == 'Subtype: ' and not sep and
                current not in ('excluded','frozen','intraoperative','site')):
            source=synoptic_specimen or specimen
            result[-1]=(current,'Subtype: '+(source+': ' if source else '')+line)
            specimen=''
            continue
        # Lettered specimen labels often place the anatomical title after the
        # colon (A: Lymph node ...); retain that title for its diagnostic block.
        lettered_specimen = (sep and re.fullmatch(r'[A-Za-z0-9]{1,3}',title.strip()) and
                             _specimen_title(body.strip()+':'))
        if lettered_specimen:
            specimen, list_index = line, None
            if current == 'frozen' and frozen_resume:
                current = 'diagnosis'
            continue
        specimen_title = title + ':' if sep and body.strip() else line
        if _specimen_title(specimen_title):
            label, list_index = specimen_title.strip().rstrip(':'), None
            if current == 'frozen' and frozen_resume:
                current = 'diagnosis'
            if sep and body.strip():
                # A filled inline field is local; it does not replace a
                # surrounding specimen's scope for later lines.
                result.append((current, label + ': ' + body.strip()))
            else:
                specimen = label
            continue
        if is_item and list_index is not None:
            section, lead = result[list_index]
            result[list_index] = (section, lead + '; ' + line)
            continue
        list_index = None
        if sep and not body.strip() and not SUBTYPE_FIELD.match(line) and not leadin:
            result.append(('template_label', line))
            continue
        # Attach a standalone specimen title to its immediate content only.
        # Filled inline fields never become persistent titles for later lines.
        scoped=specimen + ': ' + line if specimen and current == 'diagnosis' else line
        result.append((current,scoped))
        specimen=''
        if leadin:
            list_index = len(result) - 1
    return result


def diagnosis_text(text, *, include_summary=False):
    values = sections(text)
    primary = [s for section, s in values if section == 'diagnosis']
    chosen = list(primary) if primary else [s for section, s in values
                                            if section in ('unsectioned', 'summary')]
    if primary and include_summary:
        chosen += [s for section, s in values if section == 'summary']
    pending = None
    for section, value in values:
        if section in ('site', 'excluded', 'frozen', 'intraoperative', 'template_label'):
            pending = None
            continue
        field = SUBTYPE_FIELD.match(value)
        if field:
            pending = section if not field[1].strip() else None
            if field[1].strip() and section != 'diagnosis':
                chosen.append(field[1].strip())
        elif pending is not None:
            if section == pending and ':' not in value and section != 'diagnosis':
                chosen.append(value)
            pending = None
    return '\n'.join(chosen)


QUALIFIER = re.compile(
    r'\b(?:differential(?: diagnoses?|s)?|(?:rule|ruling) out|exclud(?:e|ing)|'
    r'exclusion of|mimics?|distinguish(?:es|ing)? (?:this )?from|'
    r'distinguish(?:ing)? between|distinction (?:from|between)|'
    r'recommend(?:ed|ation)?|e g|such as)\b', re.I)
RESET = re.compile(
    r'\b(?:however|but|final diagnosis|the diagnosis (?:is|remains|was)|'
    r'the current (?:evidence|morphology) (?:supports?|favors?))\b', re.I)
UNCERTAINTY = re.compile(
    r'\b(?:uncertain|equivocal|possible|possibly|suspicious|suspected|'
    r'suggest(?:s|ed|ive)?|may|might|could|cannot exclude|cannot rule out|'
    r'cannot be (?:excluded|ruled out)|not (?:excluded|ruled out)|'
    r'not (?:definite|definitive)|not (?:definitively )?confirmed|requires? confirmation|pending|'
    r'favou?r(?:s|ed|ing)?|most likely|likely|probable|presumptive|versus|vs|or|'
    r'not assessable|not specified|unspecified|unknown|indeterminate|'
    r'cannot (?:be )?(?:definitively |reliably |clearly )?(?:assess|determin|evaluat|confirm|distinguish)\w*|'
    r'unable to|insufficient|not evaluable|not provided|not reported|not applicable)\b', re.I)
NEGATIVE_LINK = re.compile(
    r'\b(?:not|no(?: (?:(?:histologic(?:al)?|morphologic(?:al)?|definite|convincing) )?'
    r'evidence (?:of|for|to support))?|without(?: evidence of)?|absence of|negative for|'
    r'lack(?:s|ing)?(?: of)?|free of|not (?:consistent with|diagnostic of)|'
    r'contradict(?:s|ing)?(?: (?:a |the )?(?:initial |prior |proposed |working )?diagnosis of)?|'
    r'does not (?:support|show)(?: a diagnosis of)?)'
    r'(?:\s+(?:residual|definite|recurrent|metastatic|convincing|unequivocal))*\s*$', re.I)
OTHER_SPECIMEN = re.compile(r'\b(?:lymph nodes?|sentinel nodes?|nodes?|margins?|ureters?|urethral margin)\b', re.I)


def _raw_span(text, start, end):
    """Map the ontology's normalized offsets back to punctuation-preserving text."""
    positions = []
    for match in re.finditer(r'[a-z0-9]+', text.lower()):
        if positions:
            positions.append(match.start())
        positions.extend(range(match.start(), match.end()))
    return positions[start], positions[end - 1] + 1


OTHER_FIELD = re.compile(
    r'\b(?:grade|grading|stage|staging|fuhrman|gleason|isup|nottingham|'
    r'necrosis|invasion|invasive pattern|mitotic count)\s*(?:(?:is|was|of)\s+)?'
    r'(?:uncertain|equivocal|cannot (?:be )?(?:determined|assessed|evaluated)|'
    r'not assessable|not specified|unknown|indeterminate|not provided|not reported|'
    r'not applicable|pending|favou?red|suspected|high|low|[0-9]+|iv|iii|ii|i)'
    r'(?:\s*(?:or|versus|vs|[-/])\s*(?:[0-9]+|iv|iii|ii|i))*\b', re.I)
PREPOSED_OTHER_FIELD = re.compile(
    r'\b(?:at most )?(?:focal )?(?:suspicious|possible|suspected|likely)'
    r'(?: for)?(?: focal)? (?:necrosis|invasion)\b', re.I)


def assertion_status(clause, start, end, *, entity_spans=(), broad_entity=False):
    """Resolve a target's assertion without crossing an independent statement.

    Semicolons delimit ordinary negation and uncertainty. A differential/list
    lead-in can govern subsequent list items until an explicit new diagnosis.
    Offsets remain those used by the frozen terminology matcher.
    """
    raw_start, raw_end = _raw_span(clause, start, end)
    before_all, after_all = clause[:raw_start], clause[raw_end:]
    left = before_all.rsplit(';', 1)[-1]
    right = after_all.split(';', 1)[0]
    # Contrasting independent clauses end local qualifiers, just as semicolons do.
    boundary = r'\b(?:but|however)\b|\band\s+(?=(?:no|not|possible|possibly|probable|likely|suspected|suspicious)\b)'
    left = re.split(boundary, left, flags=re.I)[-1]
    right = re.split(boundary, right, flags=re.I)[0]
    # A clause explicitly about the subtype does not qualify the preceding
    # broad diagnosis (e.g. carcinoma, but the subtype is not assessable).
    subtype_boundary = r',\s*(?:the\s+)?(?:(?:specific|histologic(?:al)?)\s+)?subtype\b'
    if broad_entity:
        left = re.split(subtype_boundary, left, flags=re.I)[-1]
        right = re.split(subtype_boundary, right, flags=re.I)[0]
    # A qualifier introducing a different matched entity belongs to that
    # entity, not to the broader diagnosis immediately before it.
    target_qualifier = re.compile(
        r'\b(?:favou?r(?:s|ed|ing)?|possible|possibly|probable|likely|suspected|'
        r'suspicious|suggest(?:s|ed|ive)?|cannot (?:exclude|rule out))\b', re.I)
    bounds = [_raw_span(clause, a, b) for a, b in entity_spans]
    for a, b in sorted(bounds, reverse=True):
        if b <= raw_start and a >= raw_start-len(left):
            between = clause[b:raw_start]
            qualifier = target_qualifier.search(between)
            if qualifier:
                left = between[qualifier.start():]
            break
    for a, b in sorted(bounds):
        if a >= raw_end and b <= raw_end+len(right):
            between = clause[raw_end:a]
            # Separate an additional entity introduced by 'with', including
            # its following qualifier: carcinoma, with adenocarcinoma likely.
            addition = re.search(r'\bwith\b', between, re.I)
            if addition:
                right = between[:addition.start()]
                break
            qualifier = target_qualifier.search(between)
            if qualifier:
                right = between[:qualifier.start()]
            break
    before, after = legacy.normalize(left), legacy.normalize(right)
    inherited = legacy.normalize(before_all)
    resets = list(RESET.finditer(inherited))
    if resets:
        inherited = inherited[resets[-1].start():]
    # A negative inability to exclude is uncertain, not an affirmative denial.
    cannot_exclude = re.search(r'\b(?:cannot (?:exclude|rule out)|'
                              r'(?:unable|not possible) to (?:exclude|rule out))\s*$', before)
    passive_uncertain = re.match(r'^(?:(?:is|was|remains) )?(?:cannot be (?:excluded|ruled out)|'
                                r'not (?:excluded|ruled out))\b', after)
    if cannot_exclude or passive_uncertain:
        return 'uncertain'
    if QUALIFIER.search(inherited):
        return 'hypothetical'
    if re.match(r'^(?:is |are |remains? )?(?:in|among) (?:the )?differential\b', after):
        return 'hypothetical'
    if re.match(r'^(?:markers?|stain(?:ing)?|antibodies)\b', after):
        return 'hypothetical'
    if re.match(r'^(?:differentiation|predominant|features?)\b',after):
        return 'feature_mention'
    if re.search(r'\b(?:history of|previously diagnosed with|following (?:the )?'
                 r'(?:excision|resection) of)\s+(?:\w+\s+){0,4}$', before+' '):
        return 'historical'
    masked_before = left
    left_offset = raw_start - len(left)
    for old_start, old_end in sorted(entity_spans, reverse=True):
        if old_end > start:
            continue
        x, y = _raw_span(clause, old_start, old_end)
        if x >= left_offset:
            masked_before = masked_before[:x-left_offset] + ' entity ' + masked_before[y-left_offset:]
    coordinated_denial = re.search(
        r'\b(?:no(?: evidence of)?|neither|without|negative for)\s+'
        r'(?:entity\s+)*(?:entity\s+(?:or|and|nor)\s*)$', legacy.normalize(masked_before))
    coordinated_denial = coordinated_denial or re.search(
        r'\bno\s+(?:\w+\s+){0,2}(?:dysplasia|atypia|malignancy|tumou?r)\s+(?:or|and)\s*$', before)
    # In a named margin/nodal specimen, an explicit 'no' remains a denial
    # with one intervening adjective even when that adjective has OCR damage.
    specimen_denial = (re.search(r'\bno\s+\w+\s*$', before) and
                       (OTHER_SPECIMEN.search(before) or OTHER_SPECIMEN.search(after)))
    if NEGATIVE_LINK.search(before) or coordinated_denial or specimen_denial or re.match(
        r'^(?:(?:is|was|are|were|has been) )?(?:absent|not (?:seen|identified|present)|'
        r'not (?:supported|substantiated)|negative|contradicted|retracted|ruled out|excluded)\b', after):
        if OTHER_SPECIMEN.search(before) or OTHER_SPECIMEN.search(after):
            return 'other_specimen_absent'
        return 'absent'
    # Remove only a separately named field and its value, not the rest of the
    # diagnostic statement. A later explicit subtype qualification is retained.
    confidence = OTHER_FIELD.sub('', left + ' ' + clause[raw_start:raw_end] + ' ' + right)
    confidence = PREPOSED_OTHER_FIELD.sub('', confidence)
    normalized_confidence = legacy.normalize(confidence)
    # "Or" in a morphology description (e.g. little or no tubule formation)
    # is not a diagnostic alternative. Retain it for coordinated entity choices.
    alternative = any(re.search(r'\b(?:or|versus|vs)\b', clause[min(raw_end,a):max(raw_start,b)], re.I)
                      for a,b in bounds if b <= raw_start or a >= raw_end)
    if not alternative:
        normalized_confidence = re.sub(r'\bor\b', '', normalized_confidence)
    if UNCERTAINTY.search(normalized_confidence) or '?' in confidence:
        return 'uncertain'
    return 'asserted'


def entity_restrictions(clause, start, end):
    a,b = _raw_span(clause,start,end)
    target = legacy.normalize(clause[a:b])
    after = legacy.normalize(clause[b:])
    before = legacy.normalize(clause[:a])
    if 'in situ' in target or re.match(r'^in situ\b', after):
        return ['in_situ']
    if re.search(r'\bnon invasive(?:\s+\w+){0,3}\s*$',before):
        return ['non_invasive']
    if 'invasive' in target or re.search(r'\binvasive\s*$',before):
        return ['invasive']
    return []
