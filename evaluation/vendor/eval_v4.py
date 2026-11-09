"""Synonym-aware reference agreement, with separate TCGA compatibility.

The shared extraction module supplies the non-subtype metrics and excludes
inflammatory-cell infiltration from tumour invasion. V4 independently
normalizes report/reference diagnoses with one method-blind terminology.
Patient subtype is never filled in using its TCGA project label.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import re

from . import eval_v3 as legacy
from . import diagnostic_text
from . import subtype_ontology_v4 as ontology
from .tcga_ground_truth import ground_truth_for

VERSION = "eval_v4_report_only_20261109_r7"
DIAGNOSIS_WEIGHTS = dict(legacy.DIAGNOSIS_WEIGHTS)
EXTRA_EXCLUDED = {"clinical history", "history", "clinical information", "past medical history",
                  "previous diagnosis", "prior diagnosis", "patient history"}
EXTRA_DIAGNOSIS = {"pathologic diagnosis", "pathological diagnosis", "histopathologic diagnosis",
                   "histological diagnosis", "histologic diagnosis", "pathology diagnosis"}
ABSTENTION = re.compile(r"\b(?:favou?rs?|most likely|likely|probable|presumptive|versus|vs|or)\b", re.I)
TRAILING_OTHER_FIELD = re.compile(r"\b(?:grade|grading|stage|staging|fuhrman|gleason|isup|nottingham|"
                                  r"margins?|necrosis|invasion|mitotic count)\b", re.I)
SUBTYPE_FIELD = re.compile(r"^(?:histologic(?:al)?\s+)?subtype\s*:\s*(.*)$", re.I)


def _diagnosis_text(text, *, include_summary=False):
    return diagnostic_text.diagnosis_text(text, include_summary=include_summary)


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
    # A specimen title ends at its colon; an organ in that title is not an
    # adjective modifying a diagnosis in the following field.
    normalized_parts = [legacy.normalize(part) for part in re.split(r'[:;]', text)]
    for organ in explicit - allowed:
        for alias in ontology.ORGAN_ALIASES[organ]:
            pattern = r"\b" + re.escape(alias) + r"\s+(?:\w+\s+){0,4}(?:carcinoma|adenocarcinoma|cancer)\b"
            if any(re.search(pattern, part) for part in normalized_parts):
                return True
    return bool(explicit and not explicit & allowed)


def extract_subtype(report, *, context=None, include_summary=False):
    scope = _diagnosis_text(report, include_summary=include_summary)
    hits = []
    for clause in diagnostic_text.sentences(scope):
        normalized = legacy.normalize(clause)
        matches = []
        for entity, alias, pattern in ontology.ALIASES:
            matches.extend((m.start(), m.end(), entity, alias) for m in pattern.finditer(normalized)
                           if not (entity == 'small cell neuroendocrine carcinoma' and
                                   re.search(r'\bnon\s*$',normalized[:m.start()])))
        # Context resolves anatomy, not histology: generic RCC stays generic RCC.
        origin=re.search(r'\b(?:from|originating in|primary in)\s+(.+)',clause,re.I)
        explicit_origins=ontology.organs_in(origin[1]) if origin else set()
        contextual_allowed=not explicit_origins or explicit_origins == {context}
        for alias, entity in ontology.CONTEXTUAL_ALIASES.get(context, {}).items():
            if entity and contextual_allowed:
                pattern = r"(?<!\w)" + re.escape(legacy.normalize(alias)) + r"(?!\w)"
                matches.extend((m.start(), m.end(), entity, alias) for m in re.finditer(pattern, normalized)
                               if not (entity == 'small cell neuroendocrine carcinoma' and
                                       re.search(r'\bnon\s*$',normalized[:m.start()])))
        # Prefer a whole named entity to nested generic tokens, retaining
        # disjoint sibling entities so a multi-diagnosis list cannot cherry-pick.
        selected = []
        for start, end, entity, alias in sorted(matches, key=lambda m: (-(m[1]-m[0]),
                -len(ontology.ancestors(m[2])), m[0], m[2])):
            if any(start < other[1] and end > other[0] for other in selected):
                continue
            selected.append((start, end, entity, alias))
        for start, end, entity, alias in selected:
            status = diagnostic_text.assertion_status(clause, start, end,
                                                       entity_spans=[m[:2] for m in selected],
                                                       broad_entity=not ontology.is_specific(entity))
            if "invasive" in entity and re.search(r"\bnon\s*$", normalized[:start]):
                status = "absent"
            if status == "asserted" and _organ_conflict(context, entity, clause):
                status = "organ_conflict"
            hits.append(dict(value=entity, alias=alias, text=clause, status=status,
                             normalized_span=[start, end],
                             restrictions=diagnostic_text.entity_restrictions(clause, start, end)))
    invasive_primary = any(hit["status"] == "asserted" and
                           ontology.is_a(hit["value"], "invasive breast carcinoma")
                           for hit in hits)
    if invasive_primary:
        for hit in hits:
            before = legacy.normalize(hit["text"])[:hit["normalized_span"][0]]
            if hit["value"] == "ductal carcinoma in situ" and re.search(
                r"\b(?:with|associated)\s+(?:(?:associated|focal|extensive|significant|high grade|low grade|intermediate grade)\s+)*$", before):
                hit["status"] = "associated_component"
        # A parenthetical abbreviation inherits the status of the same full name.
        for hit in hits:
            if hit["value"] != "ductal carcinoma in situ":
                continue
            for previous in hits:
                if (previous["value"] == hit["value"] and previous["text"] == hit["text"] and
                        previous["status"] == "associated_component" and
                        previous["normalized_span"][1] < hit["normalized_span"][0] and
                        not legacy.normalize(hit["text"])[previous["normalized_span"][1]:hit["normalized_span"][0]].strip()):
                    hit["status"] = "associated_component"
    certain = [hit for hit in hits if hit["status"] == "asserted"]
    for hit in certain:
        # Invasion is a separate attribute of the generic carcinoma label.
        # An explicit invasive carcinoma description and a named carcinoma
        # subtype can coexist even where the frozen taxonomy does not list
        # invasion as an ancestor. An in-situ assertion remains distinct.
        generic_invasive = hit["value"] == "invasive carcinoma" and any(
            other["value"] not in ("carcinoma", "invasive carcinoma") and
            ontology.is_a(other["value"], "carcinoma") and
            not {'in_situ','non_invasive'}.intersection(other.get('restrictions', []))
            for other in certain)
        if generic_invasive or any(
                hit["value"] != other["value"] and ontology.is_a(other["value"], hit["value"])
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
            # Denial of an in-situ component does not deny a broader/invasive
            # assertion of the same entity; a matching restriction still conflicts.
            if denial.get("restrictions") and not any(
                h["value"] == value and h["status"] == "asserted" and
                set(denial["restrictions"]).issubset(h.get("restrictions", [])) for h in hits):
                continue
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
    result["grade"] = legacy.grade_v4.extract(text, context=context,
        explicit=str(structured.get("grade") or "") if structured is not None else None,
        primary_diagnosis=str(structured.get("primary_diagnosis") or "") if structured is not None else None)
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
    # Apply cohort restrictions to affirmative diagnostic assertions; a
    # differential or negated in-situ/grade-IV mention is not that diagnosis.
    affirmative = [h for h in prediction['evidence']
                   if h['status'] in ('asserted','parent_label')]
    if code == "BRCA":
        in_situ = any(h['value'] == 'ductal carcinoma in situ' or
                      {'in_situ','non_invasive'}.intersection(h.get('restrictions', []))
                      for h in affirmative)
        invasive = any(ontology.is_a(h['value'],'invasive breast carcinoma') or
                       'invasive' in h.get('restrictions', []) for h in affirmative)
        if in_situ and not invasive:
            compatible = False
    if code == "LGG":
        for clause in diagnostic_text.sentences(prediction['diagnosis_text']):
            normalized = legacy.normalize(clause)
            spans = [h['normalized_span'] for h in prediction['evidence']
                     if h['text'] == clause and 'normalized_span' in h]
            for match in re.finditer(r'\b(?:who\s*)?grade\s*(?:iv|4)\b',normalized):
                if diagnostic_text.assertion_status(clause,match.start(),match.end(),
                                                   entity_spans=spans) == 'asserted':
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
    update_grade_result(result, legacy.grade_v4.extract(report, context=meta["context"]), ref["grade"])
    flags = result.setdefault("qa_flags", [])
    if not strict["applicable"]:
        flags.append("v4_subtype_"+strict["reason"])
    if prediction["status"] in ("conflict", "organ_conflict", "uncertain"):
        flags.append("v4_subtype_prediction_"+prediction["status"])
    if not compatibility["applicable"] and str(meta["cancer_type"] or "").upper().startswith("TCGA-"):
        flags.append("v4_no_supported_tcga_cohort")
    return result


def update_grade_result(result, prediction_grade, reference_grade):
    """Replace only grades and dependent completeness in a report evaluation."""
    old = deepcopy(result["fields"]["grade"])
    g = legacy.grade_v4.score(prediction_grade, reference_grade)
    diagnosis_only = result.get("diagnosis_only", result["fields"]["invasion"].get("capability_assessed") is False)
    if diagnosis_only:
        g["capability_assessed"] = False
    result["prediction"]["grade"] = deepcopy(prediction_grade)
    result["reference"]["grade"] = deepcopy(reference_grade)
    result["fields"]["grade"] = g
    result["metrics"].update(grade_exact=g["score"] if not diagnosis_only else None,
        grade_ordinal=g["ordinal_score"] if not diagnosis_only else None,
        grade_coverage=float(g["compatible_coverage"]) if g["applicable"] and not diagnosis_only else None)
    denominators = result.setdefault("quality_denominators", {})
    # Legacy completeness retains its original diagnostic applicability; change
    # only the grade contribution to its numerator and denominator.
    previous = result["metrics"].get("reference_coverage")
    n_old = denominators.get("eligible_structured_fields")
    if n_old is None:
        # Archived compact rows omit the legacy denominator. Reconstruct it
        # from the legacy reference fields, whose original subtype target is
        # recorded independently from the v4 entity target.
        reference = result["reference"]
        legacy_subtype = (reference.get("report_diagnosis", {}).get("subtype")
                          if not reference.get("cohort_aliases") else {"status":"asserted","value":"cohort"})
        if legacy_subtype is None:
            legacy_subtype = reference["subtype"]
        values = [reference[k] for k in ("broad", "malignancy")] + [legacy_subtype]
        values += [reference["findings"][k] for k in ("invasion", "necrosis")]
        n_old = sum(v["status"] == "asserted" and v["value"] is not None for v in values) + int(old["applicable"])
    n_new = n_old - int(old["applicable"]) + int(g["applicable"])
    if previous is not None and not diagnosis_only:
        numerator = round(previous*n_old) - int(old["applicable"] and old["covered"]) + int(g["applicable"] and g["covered"])
        result["metrics"]["reference_coverage"] = numerator/n_new if n_new else None
    denominators["eligible_structured_fields"] = n_new
    assessed = [result["fields"][k] for k in (*DIAGNOSIS_WEIGHTS, "invasion", "necrosis", "grade")
                if result["fields"][k]["applicable"]]
    denominators["eligible_structured_fields_v4"] = len(assessed)
    result["metrics"]["reference_coverage_v4"] = sum(f["covered"] for f in assessed)/len(assessed) if assessed and not diagnosis_only else None
    result["qa_flags"] = [x for x in result.get("qa_flags", []) if not x.startswith(("prediction.grade.","reference.grade."))]
    for role, grade in (("prediction",prediction_grade),("reference",reference_grade)):
        if grade["status"] != "asserted": result["qa_flags"].append(role+".grade."+grade["status"])
    return result
