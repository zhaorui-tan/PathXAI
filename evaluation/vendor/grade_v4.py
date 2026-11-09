"""Method-blind, anatomy-aware grade extraction for report agreement.

Named systems are never converted into one another. Ranges are uncertain,
not their lower endpoint. The target organ only excludes unrelated lesions
and disambiguates system names; it never supplies a grade value.
"""
from __future__ import annotations

import re

VERSION = "grade_20261109_r7"
RANGES = {"histologic3": (1, 3), "nuclear3": (1, 3), "nuclear4": (1, 4),
          "who4": (1, 4), "fuhrman4": (1, 4), "renal_isup4": (1, 4),
          "isup5": (1, 5), "gleason10": (2, 10)}
NUMBER = r"(?:10|[1-9]|IV|III|II|I)"
NAMED = re.compile(r"fuhrman|furhman|fukrman|fuhrmann|fuh?rm[ae]n", re.I)
UNCERTAIN = re.compile(r"\b(?:possible|possibly|likely|probable|suspected|suggestive|"
                       r"equivocal|indeterminate|uncertain|cannot exclude|may|might|could)\b", re.I)
UNAVAILABLE = re.compile(r"\b(?:not assessable|not specified|not provided|not reported|"
                         r"not applicable|unknown|cannot (?:be )?(?:assess|determin|evaluat)|"
                         r"unable to|insufficient|deferred|pending|undetermined)\b", re.I)
NONASSERTION = re.compile(r"\b(?:assess(?:ed|ing)? for|evaluate(?:d)? for|look(?:ing)? for|"
                          r"features (?:required|needed)|required for|cannot exclude|"
                          r"rule out|differential|if confirmed|recommend(?:ed)?|"
                          r"should be|would be|could be|such as|e\.g\.)\b", re.I)
NEGATED = re.compile(r"\b(?:no|not|without|lack(?:s|ing)?(?: of)?|absence of)\b", re.I)
MORPH_ONLY = re.compile(r"(?:high|low|intermediate)[ -]grade\s+(?:features|atypia|"
                        r"cytolog|nuclei|nuclear|morpholog)", re.I)
ORGANS = {"renal": r"kidney|renal", "prostate": r"prostat", "urinary": r"bladder|urothelial",
          "breast": r"breast|mammary", "endometrial": r"endometr|uter", "brain": r"brain|glioma|astrocyt|oligodendr"}


def number(s):
    return {"i": 1, "ii": 2, "iii": 3, "iv": 4}.get(s.lower(), int(s) if s.isdigit() else None)


def infer_context(text, context=None):
    if context:
        return {"kidney": "renal", "cns": "brain", "endometrium": "endometrial"}.get(context, context)
    # Use explicit site/diagnosis headings, never a solitary grading keyword.
    lines = [x for x in text.splitlines() if re.match(r"\s*(?:\*\*)?(?:specimen(?:/site)?|site|diagnosis|primary diagnosis)\s*:", x, re.I)]
    joined = " ".join(lines)
    found = [k for k, pat in ORGANS.items() if re.search(pat, joined, re.I)]
    return found[0] if len(found) == 1 else None


def empty(status="missing", evidence=None, reason=None):
    out = dict(value=None, status=status, system=None, ordinal=None, bounds=None, evidence=evidence or [])
    if reason:
        out["reason"] = reason
    return out


def extract(text, *, context=None, explicit=None, primary_diagnosis=None):
    from . import eval_v3 as legacy
    text = str(text or "")
    context = infer_context(text, context)
    if explicit is not None:
        explicit = str(explicit).strip()
        if re.fullmatch(NUMBER, explicit, re.I):
            explicit = "Grade: " + explicit
        elif re.fullmatch(r"low|intermediate|high", explicit, re.I):
            explicit = "Grade: " + explicit + " grade"
    lines = [("grade", explicit)] if explicit is not None else legacy.sections(text)
    primary = str(primary_diagnosis or legacy.diagnosis_scope(lines))
    invasive = bool(re.search(r"(?<!non-)\binvasive\b", primary, re.I))
    hits, excluded = [], []
    active_organ = None
    for section, line in lines:
        if section in {"excluded", "site"}:
            continue
        heading = re.match(r"^\s*(bladder|prostate|kidney|renal|breast)\s+primary\b", line, re.I)
        if heading:
            name = heading.group(1).lower()
            active_organ = {"bladder": "urinary", "kidney": "renal"}.get(name, name)
        if context and active_organ and context != active_organ:
            continue
        # Separate explicit grading statements without splitting numbers or
        # synonyms inside the same assertion.
        split_line = re.sub(r"(?:,|\band\b|\()\s*(?=(?:(?:histologic(?:al)?|nuclear|nucleolar|overall|WHO(?:/ISUP)?|ISUP|Fuhrman)\s+)*grade\b|Gleason\b)", "; ", line, flags=re.I)
        for clause in legacy.clauses(split_line):
            c = clause.strip()
            low = c.lower()
            if re.search(r"dysplasia|intraepithelial lesion|\bincidental\b", c, re.I):
                continue
            if invasive and re.search(r"\bin[ -]situ\b|\bdcis\b|\blcis\b", c, re.I) and not re.search(r"\binvasive\b", c, re.I):
                continue
            if section == "grade" and re.fullmatch(NUMBER, c, re.I):
                c = "Grade: " + c
            c = re.sub(r"^(Grade\s*[:=]\s*)(low|intermediate|high)\b(?![ -]grade)", r"\1\2 grade", c, flags=re.I)
            # An isolated structured grade descriptor still states a grade.
            if section == "grade" and re.fullmatch(r"(?:Grade\s*:\s*)?(?:high|intermediate|low)", c, re.I):
                c += " grade"
            named_fuhrman = bool(NAMED.search(c))
            isup = bool(re.search(r"\bisup\b|\bgrade group\b", c, re.I))
            gleason = bool(re.search(r"\bgleason\b", c, re.I))
            if gleason and context and context != "prostate":
                excluded.append(dict(text=c, reason="grading_of_other_organ"))
                continue
            if isup:
                if re.search(r"\bgrade group\b", c, re.I):
                    if context and context != "prostate":
                        excluded.append(dict(text=c, reason="grading_of_other_organ"))
                        continue
                    system = "isup5"
                elif context == "renal" or re.search(r"\bnucleolar\b|\brenal\b|\bkidney\b", c, re.I):
                    system = "renal_isup4"
                elif context == "prostate" or re.search(r"grade group|prostat", c, re.I):
                    system = "isup5"
                else:
                    system = None
            elif named_fuhrman:
                system = "fuhrman4"
            elif gleason:
                system = "gleason10"
            elif re.search(r"\bwho\b", c, re.I):
                system = "who4" if context in {None, "brain"} else None
            elif re.search(r"\bnuclear grade\b", c, re.I):
                system = "nuclear4" if context == "renal" or re.search(r"(?:of|/)\s*(?:4|iv)\b", c, re.I) else "nuclear3"
            else:
                system = "histologic3"

            priority = 2 if re.match(r"^(?:(?:histologic(?:al)?|nuclear|overall|who(?:/isup)?|isup|fuhrman)\s+)*grade\s*[:=]", c, re.I) else 1 if section == "grade" else 0
            candidates = []
            # Numeric assertions; keep a range as one ambiguous assertion.
            pattern = rf"\b(?:grade(?:\s+group)?|fuhrman(?:n)?|furhman|fukrman|who|gleason(?:\s+score)?)\s*[:=]?\s*({NUMBER})\b"
            for m in re.finditer(pattern, c, re.I):
                if isup and m.group().lower().startswith("gleason"):
                    continue
                value = number(m.group(1))
                tail = c[m.end():]
                if system == "gleason10":
                    pair = re.match(r"\s*\+\s*([1-5])\b", tail)
                    if pair:
                        value += int(pair.group(1))
                        tail = tail[pair.end():]
                interval = re.match(rf"\s*(?:[-–—]|\bto\b|\bor\b|\bvs\b|\band\b)\s*({NUMBER})\b", tail, re.I)
                candidates.append((value, m, bool(interval)))
            if not candidates and system in {"histologic3", "nuclear3"}:
                if section == "grade":
                    candidates.extend((int(m.group(1)), m, False) for m in re.finditer(r"\bG([1-3])\b", c, re.I))
                for value, pattern in [(1, r"\blow[ -]grade\b|\bwell[ -]differentiated\b"),
                                       (2, r"\bintermediate[ -]grade\b|\bmoderately[ -]differentiated\b"),
                                       (3, r"\bhigh[ -]grade\b|\bpoorly[ -]differentiated\b")]:
                    candidates.extend((value, m, False) for m in re.finditer(pattern, c, re.I))
            categorical_range = bool(re.search(r"\b(?:low|intermediate|moderate)(?:ly differentiated)?\s*(?:[-–—]|to|or|and)\s*(?:intermediate|moderate|high|poorly)(?:[ -]grade| differentiated)?\b", c, re.I))
            for value, match, interval in candidates:
                interval = interval or categorical_range
                if NONASSERTION.search(c) or legacy.HYPOTHETICAL.search(c) or (MORPH_ONLY.search(c) and not re.search(r"\bgrade\s*[:=]|\bgrade\s+[1-4iv]+\b", c, re.I)):
                    excluded.append(dict(text=c, reason="grading_discussion_not_assertion"))
                    continue
                if section == "morphology" and not re.search(r"(?:histologic(?:al)?|nottingham|fuhrman|who|gleason|isup|nuclear) grade|grade\s*:", c, re.I):
                    continue
                # Reuse the established scope-aware negation and uncertainty
                # guards, including question marks and "no special type".
                local = c
                # Commas separating an unrelated negative finding do not negate
                # a subsequently explicit grade assertion.
                if "," in c[:match.start()]:
                    start = c.rfind(",", 0, match.start()) + 1
                    local = c[start:]
                    polarity = legacy.polarity(local, match.start()-start, match.end()-start)
                else:
                    polarity = legacy.polarity(c, match.start(), match.end())
                suffix = c[match.end():]
                denied = bool(re.search(r"^(?:\s+(?:carcinoma|tumou?r|sarcoma))?\s*(?:is |was )?not (?:identified|seen|present)\b", suffix, re.I))
                if polarity == "absent" or denied:
                    excluded.append(dict(text=c, reason="negated_grade_assertion"))
                    continue
                uncertain = interval or polarity == "uncertain" or bool(UNCERTAIN.search(local)) or bool(UNAVAILABLE.search(local))
                if system not in RANGES:
                    uncertain = True
                elif not RANGES[system][0] <= value <= RANGES[system][1]:
                    uncertain = True
                hits.append(dict(value=f"{system}:{value}" if system else None,
                                 ordinal=value, system=system,
                                 status="uncertain" if uncertain else "asserted", text=c,
                                 priority=priority, range=interval))
            if re.match(r"^(?:(?:histologic(?:al)?|nuclear|overall)\s+)?grade\s*:", c, re.I) and UNAVAILABLE.search(c):
                hits.append(dict(value=None, status="uncertain", text=c, priority=2))

    def resolved(selected):
        if not selected:
            return empty()
        priority = max(h["priority"] for h in selected)
        selected = [h for h in selected if h["priority"] == priority]
        asserted = {h["value"] for h in selected if h["status"] == "asserted" and h["value"]}
        if len(asserted) > 1:
            return empty("conflict", selected)
        uncertain_hits = [h for h in selected if h["status"] != "asserted"]
        # A definite assertion is not erased by a hedged repetition of the
        # same value. Ranges, unavailable values and differing alternatives
        # remain ambiguous and cannot be rescued by an isolated exact value.
        if uncertain_hits and (not asserted or any(
                h.get("range") or h.get("value") not in asserted
                for h in uncertain_hits)):
            return empty("uncertain", selected, "ambiguous_or_unassessable_grade")
        if not asserted:
            return empty("missing", selected)
        value = next(iter(asserted)); system, ordinal = value.split(":")
        return dict(value=value, status="asserted", evidence=selected, system=system,
                    ordinal=int(ordinal), bounds=list(RANGES[system]))

    by_system = {system: resolved([h for h in hits if h.get("system") == system])
                 for system in RANGES if any(h.get("system") == system for h in hits)}
    overall = [h for h in hits if h.get("system") not in {"nuclear3", "nuclear4"}]
    selected = overall or hits
    # Gleason score and prostate Grade Group can both be explicitly recorded.
    # Prefer the explicitly reported Grade Group as the reference endpoint;
    # retain both systems for scoring against a reference in either system.
    if {h.get("system") for h in selected} == {"isup5", "gleason10"}:
        selected = [h for h in selected if h.get("system") == "isup5"]
    result = resolved(selected)
    result["evidence"] = hits
    result.update(extractor_version=VERSION, context=context, by_system=by_system)
    if excluded:
        result["excluded_evidence"] = excluded
    return result


def score(pred, ref):
    from . import eval_v3 as legacy
    # Match only an explicitly asserted named system, with no cross-system
    # numerical conversion. Ambiguous overall grades are never rescued.
    chosen = pred
    if pred.get("status") == "asserted" and ref.get("system") in pred.get("by_system", {}):
        chosen = pred["by_system"][ref["system"]]
    g = legacy.score_field(chosen, ref)
    same = chosen.get("system") == ref.get("system") and ref.get("system") is not None
    g.update(system=ref.get("system"), prediction_system=chosen.get("system"),
             ordinal_reference=ref.get("ordinal"), ordinal_prediction=chosen.get("ordinal"),
             compatible_coverage=bool(g["covered"] and same), ordinal_score=None)
    if g["applicable"]:
        lo, hi = ref["bounds"]
        g["ordinal_score"] = 1-abs(chosen["ordinal"]-ref["ordinal"])/(hi-lo) if same and g["covered"] else 0.0
        g["always_high_exact"] = float(ref["ordinal"] == hi)
        g["always_high_ordinal"] = 1-abs(hi-ref["ordinal"])/(hi-lo)
    return g
