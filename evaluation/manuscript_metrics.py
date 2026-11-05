"""Per-case manuscript metric mapping; frozen scorer output remains unchanged.

finding_asserted/finding_audit are copied verbatim from the manuscript aggregate
implementation (candidate_aggregate_v4.py, 2026-11-05).
"""
from collections import Counter
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finding_asserted(value: dict[str, Any]) -> bool:
    return value.get("status") == "asserted" and value.get("value") in ("present", "absent")


def finding_audit(prediction: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    pred = prediction.get("findings") or {}
    ref = reference.get("findings") or {}
    require(set(pred) == set(ref), "Prediction/reference finding vocabularies differ")
    documented = {key: value for key, value in ref.items() if finding_asserted(value)}
    counts = Counter({key: 0 for key in (
        "D", "A", "C", "E", "missing", "uncertain", "conflict", "unverified_asserted", "all_asserted"
    )})
    for key, value in pred.items():
        if finding_asserted(value):
            counts["all_asserted"] += 1
            counts["unverified_asserted"] += key not in documented
    for key, reference_fact in documented.items():
        prediction_fact = pred[key]
        counts["D"] += 1
        if finding_asserted(prediction_fact):
            counts["A"] += 1
            if prediction_fact["value"] == reference_fact["value"]:
                counts["C"] += 1
            else:
                counts["E"] += 1
        else:
            status = prediction_fact.get("status")
            require(status in ("missing", "uncertain", "conflict"), "Unexpected non-asserted finding status")
            counts[status] += 1
    require(counts["C"] + counts["E"] == counts["A"], "Finding audit A/C/E partition failed")
    require(counts["C"] + counts["E"] + counts["missing"] + counts["uncertain"] + counts["conflict"] == counts["D"],
            "Finding audit documented-finding partition failed")
    d, a, c, e = (counts[key] for key in ("D", "A", "C", "E"))
    values = {
        "finding_f1": (2 * c / (d + a)) if d + a else None,
        "assertion_coverage": (a / d) if d else None,
        "correct_fact_coverage": (c / d) if d else None,
        "contradiction_asserted": (e / a) if a else None,
    }
    return {**dict(counts), **values}



def manuscript_metrics(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    diagnosis_only = result.get("diagnosis_only", False)
    audit = None if diagnosis_only else finding_audit(result["prediction"], result["reference"])
    return {
        "schema": "pathxai_manuscript_per_case_metrics_v1",
        "diagnostic_mapping": {
            "cohort_compatibility": metrics.get("cohort_compatibility"),
            "diagnosis_cohort": metrics.get("diagnosis_cohort"),
        },
        "reference_specific_diagnostic_mapping": {
            "strict_subtype": metrics.get("subtype"),
            "diagnosis": metrics.get("diagnosis"),
        },
        "findings": audit,
        "finding_metrics_assessed": not diagnosis_only,
        "legacy_contradiction_rate_denominator": "D (documented findings)",
        "manuscript_contradiction_asserted_denominator": "A (asserted predictions on documented findings)",
        "grade": {
            "exact_per_case": metrics.get("grade_exact"),
            "ordinal_per_case": metrics.get("grade_ordinal"),
            "compatible_coverage_per_case": metrics.get("grade_coverage"),
            "class_balanced_grade": None,
            "class_balanced_grade_scope": "Balanced grade accuracy requires a cohort grouped by grading system and represented reference grade; not defined for one case",
        },
    }
