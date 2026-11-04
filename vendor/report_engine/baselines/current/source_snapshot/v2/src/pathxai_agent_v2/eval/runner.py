"""Evaluation runner — orchestrates evaluation of pipeline outputs.

Usage:
    from pathxai_agent_v2.eval.runner import EvalRunner

    runner = EvalRunner(settings)
    results = runner.evaluate_run(run_dir, reference_dir)
    runner.save_results(results, output_path)
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathxai_agent_v2.config import Settings
from pathxai_agent_v2.eval.metrics import (
    bleu,
    confidence_calibration,
    critical_finding_recall,
    cross_case_consistency,
    diagnosis_hit_score,
    evaluate_diagnosis_tiers,
    evaluate_diagnostic_hierarchy,
    extract_diagnostic_facts_rule,
    morph_term_f1,
    pathology_report_score,
    rouge_n,
    section_similarity,
)


@dataclass
class CaseEvalResult:
    """Evaluation result for a single case."""
    case_id: str
    cancer_type: str = ""
    # Text metrics
    bleu_1: float = 0.0
    bleu_4: float = 0.0
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    # Pathology metrics
    morph_f1: float = 0.0
    diagnosis_hit: float = 0.0
    critical_recall: float = 0.0
    overclaim_rate: float = 0.0
    composite_score: float = 0.0
    # Hierarchy (legacy flat weighted fields)
    hierarchy_score: float = 0.0
    field_scores: dict[str, Any] = field(default_factory=dict)
    # Tiered diagnosis score (independent broad / subtype / malignancy / grade axes)
    diagnosis_score: float = 0.0
    broad_score: float = 0.0
    subtype_score: float = 0.0
    malignancy_score: float = 0.0
    grade_score: float = 0.0
    tier_details: dict[str, Any] = field(default_factory=dict)
    # Section similarities
    section_sims: dict[str, float] = field(default_factory=dict)
    # Metadata
    confidence: str = "low"
    predicted_diagnosis: str = ""
    reference_diagnosis: str = ""
    errors: list[str] = field(default_factory=list)


@dataclass
class EvalSummary:
    """Aggregated evaluation results."""
    total_cases: int = 0
    avg_bleu_1: float = 0.0
    avg_bleu_4: float = 0.0
    avg_rouge_1: float = 0.0
    avg_rouge_2: float = 0.0
    avg_morph_f1: float = 0.0
    avg_diagnosis_hit: float = 0.0
    avg_critical_recall: float = 0.0
    avg_overclaim_rate: float = 0.0
    avg_composite_score: float = 0.0
    avg_hierarchy_score: float = 0.0
    # Tiered diagnosis averages (independent axes)
    avg_diagnosis_score: float = 0.0
    avg_broad_score: float = 0.0
    avg_subtype_score: float = 0.0
    avg_malignancy_score: float = 0.0
    avg_grade_score: float = 0.0
    # By cancer type
    by_cancer_type: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Calibration
    calibration: dict[str, Any] = field(default_factory=dict)
    # Consistency
    consistency: dict[str, Any] = field(default_factory=dict)


class EvalRunner:
    """Evaluates pipeline outputs against reference reports.

    Unlike v1, this runner:
    - Uses rule-based fact extraction by default (no LLM self-preference bias)
    - Computes confidence calibration
    - Checks cross-case consistency
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def evaluate_case(
        self,
        generated_report: str,
        reference_report: str,
        case_id: str = "",
        cancer_type: str = "",
        confidence: str = "low",
        diagnosis_prediction: dict | None = None,
        known_site: bool = False,
        ref_facts_precomputed: dict | None = None,
    ) -> CaseEvalResult:
        """Evaluate a single generated report against its reference."""
        errors = []
        if not generated_report or not reference_report:
            return CaseEvalResult(
                case_id=case_id,
                cancer_type=cancer_type,
                errors=["missing report or reference"],
            )

        # Text metrics
        try:
            b = bleu(generated_report, reference_report)
            r1 = rouge_n(generated_report, reference_report, n=1)
            r2 = rouge_n(generated_report, reference_report, n=2)
        except Exception as e:
            errors.append(f"text_metrics: {e}")
            b = {"bleu_1": 0.0, "bleu_4": 0.0}
            r1 = {"f1": 0.0}
            r2 = {"f1": 0.0}

        # Section similarity
        from pathxai_agent_v2.pipeline.report_generation import _extract_sections
        pred_secs = _extract_sections(generated_report)
        ref_secs = _extract_sections(reference_report)
        sec_sims = section_similarity(pred_secs, ref_secs)

        # Pathology metrics
        try:
            path_score = pathology_report_score(generated_report, reference_report)
            morph = morph_term_f1(generated_report, reference_report)
            crit_recall = critical_finding_recall(generated_report, reference_report)
        except Exception as e:
            errors.append(f"pathology_metrics: {e}")
            path_score = {"composite_score": 0.0, "overclaim_rate": 0.0}
            morph = {"overall_f1": 0.0}
            crit_recall = {"recall": 0.0}

        # Hierarchy scoring (legacy flat weighted fields — kept for continuity).
        # Site is a known input when known_site=True → excluded from scoring.
        try:
            pred_facts = extract_diagnostic_facts_rule(generated_report)
            ref_facts = ref_facts_precomputed or extract_diagnostic_facts_rule(reference_report)
            fw = dict(self.settings.eval.field_weights)
            if known_site:
                fw.pop("site", None)  # site was given, not predicted — don't score it
            hierarchy = evaluate_diagnostic_hierarchy(
                pred_facts, ref_facts,
                field_weights=fw,
            )
            if known_site and "field_scores" in hierarchy:
                hierarchy["field_scores"].pop("site", None)
        except Exception as e:
            errors.append(f"hierarchy: {e}")
            hierarchy = {"overall_score": 0.0, "field_scores": {}}

        # Tiered diagnosis scoring (independent broad / subtype / malignancy).
        # Anchored on the known TCGA cohort ground truth + LLM-extracted ref facts.
        # Uses structured prediction tiers from the pipeline (diagnosis_prediction)
        # so the eval sees exactly what the claim-filtered report committed to.
        try:
            tiers = evaluate_diagnosis_tiers(
                generated_report,
                cancer_type=cancer_type,
                reference=reference_report,
                tier_weights=self.settings.eval.tier_weights,
                pred_facts=diagnosis_prediction,
            )
        except Exception as e:
            errors.append(f"diagnosis_tiers: {e}")
            tiers = {"diagnosis_score": 0.0, "tiers": {}}

        tier_map = tiers.get("tiers", {})
        broad_s = float(tier_map.get("broad", {}).get("score", 0.0))
        subtype_s = float(tier_map.get("subtype", {}).get("score", 0.0))
        malig_s = float(tier_map.get("malignancy", {}).get("score", 0.0))
        grade_s = float(tier_map.get("grade", {}).get("score", 0.0))
        diag_score = float(tiers.get("diagnosis_score", 0.0))

        return CaseEvalResult(
            case_id=case_id,
            cancer_type=cancer_type,
            bleu_1=b.get("bleu_1", 0.0),
            bleu_4=b.get("bleu_4", 0.0),
            rouge_1=r1.get("f1", 0.0),
            rouge_2=r2.get("f1", 0.0),
            morph_f1=morph.get("overall_f1", 0.0),
            diagnosis_hit=subtype_s,   # tier-based subtype match — replaces unreliable text-match
            critical_recall=crit_recall.get("recall", 0.0),
            overclaim_rate=path_score.get("overclaim_rate", 0.0),
            composite_score=path_score.get("composite_score", 0.0),
            hierarchy_score=hierarchy.get("overall_score", 0.0),
            field_scores=hierarchy.get("field_scores", {}),
            diagnosis_score=diag_score,
            broad_score=broad_s,
            subtype_score=subtype_s,
            malignancy_score=malig_s,
            grade_score=grade_s,
            tier_details=tier_map,
            section_sims=sec_sims,
            confidence=confidence,
            predicted_diagnosis=str(tier_map.get("subtype", {}).get("prediction", "")),
            reference_diagnosis=str(tier_map.get("subtype", {}).get("reference", "")),
            errors=errors,
        )

    def evaluate_run(
        self,
        run_dir: str | Path,
        reference_dir: str | Path,
        cancer_type_filter: str | None = None,
    ) -> tuple[list[CaseEvalResult], EvalSummary]:
        """Evaluate all cases in a run directory."""
        run_dir = Path(run_dir)
        ref_dir = Path(reference_dir)
        results: list[CaseEvalResult] = []

        # Find all pipeline_result.json files
        for result_path in sorted(run_dir.glob("**/pipeline_result.json")):
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            case_id = data.get("case_id", result_path.parent.name)
            cancer_type = ""
            generated = data.get("generated_long_report", "")
            confidence = "low"

            # Try to get cancer_type from the directory structure or data
            # (TCGA-XX/case_id/pipeline_result.json)
            if result_path.parent.parent:
                cancer_type = result_path.parent.parent.name

            if cancer_type_filter and cancer_type != cancer_type_filter:
                continue

            # Find reference report
            ref_report = self._find_reference(ref_dir, cancer_type, case_id)
            if not ref_report:
                continue

            # Try to get confidence from diagnosis prediction
            diag_pred = data.get("diagnosis_prediction") or {}
            if isinstance(diag_pred, dict):
                confidence = str(diag_pred.get("confidence", "low"))

            result = self.evaluate_case(
                generated_report=generated,
                reference_report=ref_report,
                case_id=case_id,
                cancer_type=cancer_type,
                confidence=confidence,
                diagnosis_prediction=diag_pred if isinstance(diag_pred, dict) else None,
            )
            results.append(result)

        summary = self._build_summary(results)
        return results, summary

    def _find_reference(
        self, ref_dir: Path, cancer_type: str, case_id: str
    ) -> str | None:
        """Find the reference report for a case."""
        # Try multiple possible locations
        candidates = [
            ref_dir / cancer_type / case_id / "report_image_only.txt",
            ref_dir / cancer_type / case_id / "report_full_structured.txt",
            ref_dir / cancer_type / case_id / "Report.txt",
            ref_dir / cancer_type / case_id / "reference_report.txt",
        ]
        for path in candidates:
            if path.exists():
                text = path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    return text
        return None

    def _build_summary(self, results: list[CaseEvalResult]) -> EvalSummary:
        """Build aggregated summary from individual case results."""
        if not results:
            return EvalSummary()

        def avg(attr: str) -> float:
            vals = [getattr(r, attr) for r in results]
            return sum(vals) / len(vals)

        def avg_grade(rs: list[CaseEvalResult]) -> float:
            # Grade is case-specific (GT from the reference report). The figure
            # grade metric is field_scores.grade (not the coarse tier
            # grade_score); cases whose reference has no grade are marked
            # not-applicable and excluded so they don't drag the average to 0.
            scores = [
                r.field_scores.get("grade", {}).get("score")
                for r in rs
                if r.field_scores.get("grade", {}).get("applicable") is True
            ]
            scores = [s for s in scores if isinstance(s, (int, float))]
            return sum(scores) / len(scores) if scores else 0.0

        # By cancer type
        by_type: dict[str, list[CaseEvalResult]] = defaultdict(list)
        for r in results:
            by_type[r.cancer_type or "unknown"].append(r)

        type_summaries = {}
        for ct, cases in by_type.items():
            type_summaries[ct] = {
                "count": len(cases),
                "avg_composite_score": sum(c.composite_score for c in cases) / len(cases),
                "avg_diagnosis_hit": sum(c.diagnosis_hit for c in cases) / len(cases),
                "avg_hierarchy_score": sum(c.hierarchy_score for c in cases) / len(cases),
                "avg_diagnosis_score": sum(c.diagnosis_score for c in cases) / len(cases),
                "avg_broad_score": sum(c.broad_score for c in cases) / len(cases),
                "avg_subtype_score": sum(c.subtype_score for c in cases) / len(cases),
                "avg_malignancy_score": sum(c.malignancy_score for c in cases) / len(cases),
                "avg_grade_score": avg_grade(cases),
            }

        # Calibration
        calib_data = [
            {"confidence": r.confidence, "accuracy": r.diagnosis_hit}
            for r in results
        ]
        calibration = confidence_calibration(calib_data)

        # Cross-case consistency
        consistency_data = [
            {"cancer_type": r.cancer_type, "primary_diagnosis": r.predicted_diagnosis}
            for r in results
        ]
        consistency = cross_case_consistency(consistency_data)

        return EvalSummary(
            total_cases=len(results),
            avg_bleu_1=avg("bleu_1"),
            avg_bleu_4=avg("bleu_4"),
            avg_rouge_1=avg("rouge_1"),
            avg_rouge_2=avg("rouge_2"),
            avg_morph_f1=avg("morph_f1"),
            avg_diagnosis_hit=avg("diagnosis_hit"),
            avg_critical_recall=avg("critical_recall"),
            avg_overclaim_rate=avg("overclaim_rate"),
            avg_composite_score=avg("composite_score"),
            avg_hierarchy_score=avg("hierarchy_score"),
            avg_diagnosis_score=avg("diagnosis_score"),
            avg_broad_score=avg("broad_score"),
            avg_subtype_score=avg("subtype_score"),
            avg_malignancy_score=avg("malignancy_score"),
            avg_grade_score=avg_grade(results),
            by_cancer_type=type_summaries,
            calibration=calibration,
            consistency=consistency,
        )

    def save_results(
        self,
        results: list[CaseEvalResult],
        summary: EvalSummary,
        output_dir: str | Path,
    ) -> None:
        """Save evaluation results as JSON and CSV."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Per-case results
        rows = []
        for r in results:
            rows.append({
                "case_id": r.case_id,
                "cancer_type": r.cancer_type,
                "bleu_1": r.bleu_1,
                "bleu_4": r.bleu_4,
                "rouge_1": r.rouge_1,
                "rouge_2": r.rouge_2,
                "morph_f1": r.morph_f1,
                "diagnosis_hit": r.diagnosis_hit,
                "critical_recall": r.critical_recall,
                "overclaim_rate": r.overclaim_rate,
                "composite_score": r.composite_score,
                "hierarchy_score": r.hierarchy_score,
                "diagnosis_score": r.diagnosis_score,
                "broad_score": r.broad_score,
                "subtype_score": r.subtype_score,
                "malignancy_score": r.malignancy_score,
                "grade_score": r.grade_score,
                "confidence": r.confidence,
                "predicted_diagnosis": r.predicted_diagnosis,
                "reference_diagnosis": r.reference_diagnosis,
                "errors": r.errors,
            })

        (out / "case_results.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # Summary
        summary_dict = {
            "total_cases": summary.total_cases,
            "avg_bleu_1": summary.avg_bleu_1,
            "avg_bleu_4": summary.avg_bleu_4,
            "avg_rouge_1": summary.avg_rouge_1,
            "avg_rouge_2": summary.avg_rouge_2,
            "avg_morph_f1": summary.avg_morph_f1,
            "avg_diagnosis_hit": summary.avg_diagnosis_hit,
            "avg_critical_recall": summary.avg_critical_recall,
            "avg_overclaim_rate": summary.avg_overclaim_rate,
            "avg_composite_score": summary.avg_composite_score,
            "avg_hierarchy_score": summary.avg_hierarchy_score,
            "avg_diagnosis_score": summary.avg_diagnosis_score,
            "avg_broad_score": summary.avg_broad_score,
            "avg_subtype_score": summary.avg_subtype_score,
            "avg_malignancy_score": summary.avg_malignancy_score,
            "avg_grade_score": summary.avg_grade_score,
            "by_cancer_type": summary.by_cancer_type,
            "calibration": summary.calibration,
            "consistency": summary.consistency,
        }
        (out / "summary.json").write_text(
            json.dumps(summary_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # CSV
        if rows:
            import csv
            with (out / "case_results.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
