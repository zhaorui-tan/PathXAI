"""Synonym-aware reference agreement, with separate TCGA compatibility.

V3 is unmodified and supplies every non-subtype metric. V4 independently
normalizes report/reference diagnoses with one method-blind terminology.
Patient subtype is never filled in using its TCGA project label.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re

from . import eval_v3 as legacy
from . import subtype_ontology_v4 as ontology
from .tcga_ground_truth import ground_truth_for

VERSION = "eval_v4_report_only_20261017_r5"
DIAGNOSIS_WEIGHTS = dict(legacy.DIAGNOSIS_WEIGHTS)
EXTRA_EXCLUDED = {"clinical history", "history", "clinical information", "past medical history",
                  "previous diagnosis", "prior diagnosis", "patient history"}
EXTRA_DIAGNOSIS = {"pathologic diagnosis", "pathological diagnosis", "histopathologic diagnosis",
                   "histological diagnosis", "histologic diagnosis", "pathology diagnosis"}
ABSTENTION = re.compile(r"\b(?:favou?rs?|most likely|likely|probable|presumptive|versus|vs|or)\b", re.I)
TRAILING_OTHER_FIELD = re.compile(r"\b(?:grade|grading|stage|staging|fuhrman|gleason|isup|nottingham|"
                                  r"margins?|necrosis|invasion|mitotic count)\b", re.I)
SUBTYPE_FIELD = re.compile(r"^(?:histologic(?:al)?\s+)?subtype\s*:\s*(.*)$", re.I)
LOCAL_TARGET = (r"(?:lymph\s+nodes?|sentinel\s+nodes?|"
                r"(?:(?:surgical|resection|inked|deep|peripheral)\s+)?margins?)")
LOCAL_DENIAL_BEFORE = re.compile(
    LOCAL_TARGET + r"(?:\s+(?:are|is|were|was|remain|all|examined|sampled|regional|reported|found|as)){0,5}"
    r"\s+(?:negative(?:\s+for)?|free\s+of|show\s+no|without)"
    r"(?:\s+(?:metastatic|residual|invasive|tumor|and|any|evidence|of)){0,5}\s*$", re.I)
LOCAL_DENIAL_AFTER = re.compile(
    r"^(?:\s+(?:is|was|are|were|present|identified|seen|found|not|detected)){0,5}"
    r"\s+(?:in|within|at|from)\s+(?:the\s+)?"
    r"(?:(?:sampled|regional|sentinel|axillary|surgical|resection|inked|deep|peripheral)\s+)*"
    + LOCAL_TARGET + r"\b", re.I)
DISTINCT_COMPONENT = re.compile(
    r"\b(?:separate|independent|another primary|not (?:an? )?associated|without association)\b", re.I)


def _local_denial(clause, start, end):
    """A negative margin/node is not a denial of the patient's primary tumor."""
    tail = clause[end:]
    after = LOCAL_DENIAL_AFTER.search(tail)
    if after:
        # A coordinated nonlocal target must not be silently discarded.
        return not tail[after.end():].strip(" .,:;()")
    return bool(LOCAL_DENIAL_BEFORE.search(clause[max(0, start - 140):start])
                and not tail.strip(" .,:;()"))


def _associated_component(hit):
    text = legacy.normalize(hit["text"])
    if DISTINCT_COMPONENT.search(text):
        return False
    start, end = hit["normalized_span"]
    before, after = text[:start], text[end:]
    return bool(re.search(r"\b(?:associated|accompanying|concomitant|coexisting)\s+$", before)
                or re.match(r"\s+(?:(?:was|is)\s+)?also\s+(?:present|identified|seen)\b", after)
                or re.match(r"\s+(?:component\s+)?(?:was|is)\s+(?:associated|concomitant|coexisting)\b", after))


def _diagnosis_text(text):
    text = re.sub(r"<think>.*?(?:</think>|$)", "", str(text), flags=re.I | re.S)
    rewritten = []
    for line in text.splitlines():
        stripped = re.sub(r"^\s*(?:#{1,6}\s*|\d+[.)]\s*|[-*]\s+)", "", line).replace("**", "").strip()
        title, sep, body = stripped.partition(":")
        heading = title.lower().strip()
        if heading in EXTRA_EXCLUDED:
            line = "Differential diagnoses:" + (body if sep else "")
        elif heading in EXTRA_DIAGNOSIS:
            line = "Diagnosis:" + (body if sep else "")
        rewritten.append(line)
    sections = legacy.sections("\n".join(rewritten))
    scoped = legacy.diagnosis_scope(sections)
    # Report templates can put an explicit subtype in the grade/findings section.
    # Retain only that labelled field, never the surrounding narrative/history.
    extra_subtypes = []
    pending_section = None
    for section, value in sections:
        if section in ("site", "excluded"):
            pending_section = None
            continue
        field = SUBTYPE_FIELD.match(value)
        if field:
            pending_section = section if not field[1].strip() else None
            if field[1].strip() and section != "diagnosis":
                extra_subtypes.append(field[1].strip())
        elif pending_section is not None:
            if section == pending_section and ":" not in value and section != "diagnosis":
                extra_subtypes.append(value)
            pending_section = None
    scoped = "\n".join([scoped, *extra_subtypes])
    selected = []
    for clause in legacy.clauses(scoped):
        if re.match(r"(?:broad(?: type| lineage)?|malignancy(?: status)?|grade|confidence|site|specimen)\s*:", clause, re.I):
            continue
        # Associated in-situ components do not replace the invasive primary.
        clause = re.split(r"\bwith\s+(?:(?:focal|associated)\s+)?(?:dcis\b|ductal carcinoma in situ\b)",
                          clause, maxsplit=1, flags=re.I)[0]
        selected.append(clause)
    return "\n".join(selected)


def _organ_conflict(context, entity, text):
    if not context:
        return False
    entity_organs = set(ontology.ENTITIES.get(entity, {}).get("organs", ()))
    allowed = {context}
    if context == "adrenal_paraganglia":
        allowed = {"adrenal", "paraganglia"}
    # A named primary of another organ cannot be relabelled by context.
    if entity_organs and not entity_organs & allowed:
        return True
    explicit = ontology.organs_in(text)
    # Described metastatic destinations are not primary-site contradictions.
    if re.search(r"\bmetasta(?:tic|sis|ses)\b|\bsecondary to\b", text, re.I):
        return False
    # An incompatible organ adjective directly qualifying a carcinoma is
    # stronger than an incidental organ mention elsewhere in the report.
    normalized = legacy.normalize(text)
    for organ in explicit - allowed:
        for alias in ontology.ORGAN_ALIASES[organ]:
            pattern = r"\b" + re.escape(alias) + r"\s+(?:\w+\s+){0,4}(?:carcinoma|adenocarcinoma|cancer)\b"
            if re.search(pattern, normalized):
                return True
    return bool(explicit and not explicit & allowed)


def extract_subtype(report, *, context=None):
    scope = _diagnosis_text(report)
    hits = []
    for clause in legacy.clauses(scope):
        normalized = legacy.normalize(clause)
        matches = []
        for entity, alias, pattern in ontology.ALIASES:
            matches.extend((m.start(), m.end(), entity, alias) for m in pattern.finditer(normalized))
        # Context resolves anatomy, not histology: generic RCC stays generic RCC.
        for alias, entity in ontology.CONTEXTUAL_ALIASES.get(context, {}).items():
            if entity:
                pattern = r"(?<!\w)" + re.escape(legacy.normalize(alias)) + r"(?!\w)"
                matches.extend((m.start(), m.end(), entity, alias) for m in re.finditer(pattern, normalized))
        # Prefer a whole named entity to nested generic tokens, retaining
        # disjoint sibling entities so a multi-diagnosis list cannot cherry-pick.
        selected = []
        for start, end, entity, alias in sorted(matches, key=lambda m: (-(m[1]-m[0]),
                -len(ontology.ancestors(m[2])), m[0], m[2])):
            if any(start < other[1] and end > other[0] for other in selected):
                continue
            selected.append((start, end, entity, alias))
        for start, end, entity, alias in selected:
            # Uncertainty about a trailing grade is not uncertainty about an
            # already named histology (e.g. "clear-cell RCC, grade unknown").
            tail = normalized[end:]
            next_field = TRAILING_OTHER_FIELD.search(tail)
            confidence_text = normalized[:end] + (tail[:next_field.start()] if next_field else tail)
            if legacy.HYPOTHETICAL.search(confidence_text):
                status = "hypothetical"
            else:
                status = legacy.polarity(confidence_text, start, end)
                if status == "present":
                    status = "uncertain" if ABSTENTION.search(confidence_text) else "asserted"
                elif status == "absent" and _local_denial(normalized, start, end):
                    status = "local_absence"
            if "invasive" in entity and re.search(r"\bnon\s*$", normalized[:start]):
                status = "absent"
            if status == "asserted" and _organ_conflict(context, entity, clause):
                status = "organ_conflict"
            hits.append(dict(value=entity, alias=alias, text=clause, status=status,
                             normalized_span=[start, end]))
    invasive_primary = any(hit["status"] == "asserted" and
                           ontology.is_a(hit["value"], "invasive breast carcinoma") for hit in hits)
    if invasive_primary:
        for hit in hits:
            if (hit["status"] == "asserted" and
                    hit["value"] in ("ductal carcinoma in situ", "lobular carcinoma in situ")
                    and _associated_component(hit)):
                hit["status"] = "associated_lesion"
    certain = [hit for hit in hits if hit["status"] == "asserted"]
    for hit in certain:
        if any(hit["value"] != other["value"] and ontology.is_a(other["value"], hit["value"])
               for other in certain):
            hit["status"] = "parent_label"
    values = sorted({hit["value"] for hit in hits if hit["status"] == "asserted"})
    denials = [hit for hit in hits if hit["status"] == "absent"]
    contradictory_polarity = False
    for value in values:
        positive_organs = set(ontology.ENTITIES.get(value, {}).get("organs", ()))
        if not positive_organs:
            positive_organs = set().union(*(ontology.organs_in(hit["text"]) for hit in hits
                                            if hit["value"] == value and hit["status"] == "asserted"))
        for denial in denials:
            negative_organs = ontology.organs_in(denial["text"])
            if positive_organs and negative_organs and not positive_organs & negative_organs:
                continue
            if ontology.is_a(value, denial["value"]):
                contradictory_polarity = True
    if contradictory_polarity:
        value, status = None, "conflict"
    elif len(values) == 1:
        value, status = values[0], "asserted"
    elif len(values) > 1:
        value, status = None, "conflict"
    else:
        value = None
        status = ("organ_conflict" if any(h["status"] == "organ_conflict" for h in hits) else
                  "uncertain" if any(h["status"] == "uncertain" for h in hits) else
                  "absent" if any(h["status"] == "absent" for h in hits) else "missing")
    return dict(value=value, status=status, evidence=hits, diagnosis_text=scope,
                context=context, specific=ontology.is_specific(value))


def reference(text, *, structured=None, cohort=None, cancer_type=None, site=None):
    if isinstance(cohort, str):
        if cancer_type and ontology.project_code(cancer_type) != ontology.project_code(cohort):
            raise ValueError("Conflicting cohort and cancer_type identifiers")
        cancer_type = cohort
        cohort = ground_truth_for(cancer_type)
    if cohort is None and cancer_type:
        cohort = ground_truth_for(cancer_type)
    result = legacy.reference(text, structured=structured, cohort=cohort)
    context = ontology.organ_context(cancer_type, site)
    if structured is not None:
        subtype_text = str(structured.get("histologic_subtype") or "").strip()
        primary_text = str(structured.get("primary_diagnosis") or "").strip()
        # An explicitly unknown structured field is not invented from a label.
        candidate = subtype_text or primary_text
        subtype = extract_subtype(candidate, context=context)
        source = "physician_structured_reference"
    else:
        subtype = extract_subtype(text, context=context)
        source = "case_report_text"
    result["v4_reference"] = dict(subtype=subtype, cancer_type=cancer_type,
                                   context=context, source=source,
                                   text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                   terminology=ontology.VERSION)
    return result


def _strict_field(prediction, reference_subtype):
    applicable = (reference_subtype["status"] == "asserted" and reference_subtype["specific"])
    covered = prediction["status"] == "asserted" and prediction["value"] is not None
    match = covered and prediction["value"] == reference_subtype["value"]
    if not applicable:
        reason = "reference_not_specific" if reference_subtype["status"] == "asserted" else "reference_" + reference_subtype["status"]
    elif not covered:
        reason = "prediction_" + prediction["status"]
    elif match:
        reason = "canonical_synonym_match"
    elif not prediction["specific"]:
        reason = "prediction_not_specific"
    else:
        reason = "different_entity"
    return dict(prediction=prediction["value"], reference=reference_subtype["value"],
                applicable=bool(applicable), covered=bool(covered), score=float(match) if applicable else None,
                prediction_status=prediction["status"], reference_status=reference_subtype["status"],
                reason=reason, target_kind="report_reference_entity_exact", reference_source=None)


def _cohort_field(prediction, meta):
    code = ontology.project_code(meta["cancer_type"])
    members = ontology.COHORT_MEMBERS.get(code)
    covered = prediction["status"] == "asserted" and prediction["value"] is not None
    compatible = bool(covered and members and any(ontology.is_a(prediction["value"], m) for m in members))
    # An in-situ diagnosis cannot establish membership of an invasive cohort.
    if code == "BRCA" and re.search(r"\bin situ\b", prediction["diagnosis_text"], re.I):
        if not re.search(r"(?<!non )\binvasive\b|\binfiltrating\b", prediction["diagnosis_text"], re.I):
            compatible = False
    if code == "LGG" and re.search(r"\b(?:who\s*)?grade\s*(?:iv|4)\b", prediction["diagnosis_text"], re.I):
        compatible = False
    return dict(prediction=prediction["value"], reference="TCGA-"+code if members else None,
                accepted_entities=list(members or ()), applicable=members is not None, covered=covered,
                score=float(compatible) if members is not None else None,
                prediction_status=prediction["status"], target_kind="cohort_compatibility_not_exact_subtype")


def evaluate(report, ref, *, reference_text="", diagnosis_only=False, narrative_reference=True):
    if "v4_reference" not in ref:
        raise ValueError("Build the reference with eval_v4.reference; do not reuse legacy scores/references")
    result = legacy.evaluate(report, deepcopy(ref), reference_text=reference_text,
                             diagnosis_only=diagnosis_only, narrative_reference=narrative_reference)
    previous_metrics = deepcopy(result["metrics"])
    previous_field = deepcopy(result["fields"]["subtype"])
    meta = ref["v4_reference"]
    prediction = extract_subtype(report, context=meta["context"])
    strict = _strict_field(prediction, meta["subtype"])
    strict["reference_source"] = meta["source"]
    compatibility = _cohort_field(prediction, meta)
    result["schema"] = VERSION
    result["terminology_version"] = ontology.VERSION
    result["prediction"]["subtype"] = prediction
    result["reference"]["subtype"] = deepcopy(meta["subtype"])
    result["fields"]["subtype"] = strict
    result["fields"]["cohort_compatibility"] = compatibility
    result["metrics"]["subtype"] = strict["score"]
    result["metrics"]["cohort_compatibility"] = compatibility["score"]
    result["metrics"]["subtype_reference_coverage"] = float(strict["applicable"])
    assessed = [result["fields"][key] for key in (*DIAGNOSIS_WEIGHTS, "invasion", "necrosis", "grade")
                if result["fields"][key]["applicable"]]
    result["metrics"]["reference_coverage_v4"] = (
        sum(field["covered"] for field in assessed)/len(assessed)
        if assessed and not diagnosis_only else None)
    result["metric_scopes"] = dict(reference_coverage="unchanged_v3_applicability",
                                    reference_coverage_v4="v4_reference_applicability",
                                    subtype="anatomy_conditioned_histologic_entity_not_full_WHO_classification")
    available = {key: weight for key, weight in DIAGNOSIS_WEIGHTS.items() if result["fields"][key]["applicable"]}
    result["metrics"]["diagnosis"] = (sum(result["metrics"][key]*weight for key, weight in available.items()) /
                                        sum(available.values()) if available else None)
    cohort_available = dict(available)
    if compatibility["applicable"]:
        cohort_available["subtype"] = DIAGNOSIS_WEIGHTS["subtype"]
        result["metrics"]["diagnosis_cohort"] = sum(
            (compatibility["score"] if key == "subtype" else result["metrics"][key])*weight
            for key, weight in cohort_available.items())/sum(cohort_available.values())
    else:
        result["metrics"]["diagnosis_cohort"] = None
    # This fresh v3 pass uses the exact same text, not stored historical scores.
    result["comparison_to_v3"] = dict(schema=legacy.VERSION, metrics=previous_metrics,
                                      subtype_field=previous_field)
    result["subtype_audit"] = dict(context=meta["context"], cancer_type=meta["cancer_type"],
                                   extraction_policy="diagnosis_plus_explicit_subtype_nonexcluded_sections",
                                   prediction_evidence=prediction["evidence"],
                                   reference_evidence=meta["subtype"]["evidence"],
                                   strict_reason=strict["reason"],
                                   v3_subtype_was_cohort_compatibility=bool(ref.get("cohort_aliases") is not None))
    flags = result.setdefault("qa_flags", [])
    if not strict["applicable"]:
        flags.append("v4_subtype_"+strict["reason"])
    if prediction["status"] in ("conflict", "organ_conflict", "uncertain"):
        flags.append("v4_subtype_prediction_"+prediction["status"])
    if not compatibility["applicable"] and str(meta["cancer_type"] or "").upper().startswith("TCGA-"):
        flags.append("v4_no_supported_tcga_cohort")
    return result
