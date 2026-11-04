"""Evaluation module for PathXAI v2."""

from pathxai_agent_v2.eval.metrics import (
    bleu,
    clinical_overclaim_rate,
    confidence_calibration,
    critical_finding_recall,
    cross_case_consistency,
    diagnosis_hit_score,
    evaluate_diagnostic_hierarchy,
    extract_diagnostic_facts_rule,
    morph_term_f1,
    pathology_report_score,
    rouge_n,
    section_similarity,
)
from pathxai_agent_v2.eval.runner import CaseEvalResult, EvalRunner, EvalSummary

__all__ = [
    "EvalRunner",
    "CaseEvalResult",
    "EvalSummary",
    "bleu",
    "rouge_n",
    "section_similarity",
    "morph_term_f1",
    "diagnosis_hit_score",
    "critical_finding_recall",
    "clinical_overclaim_rate",
    "pathology_report_score",
    "extract_diagnostic_facts_rule",
    "evaluate_diagnostic_hierarchy",
    "confidence_calibration",
    "cross_case_consistency",
]
