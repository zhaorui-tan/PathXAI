"""Synthetic scientific-contract regressions for the report-only v4 evaluator."""
from __future__ import annotations

from copy import deepcopy
import unittest

from ..vendor import eval_v3, eval_v4


def evaluate(report: str, reference: str, **kwargs):
    return eval_v4.evaluate(report, eval_v4.reference(reference, **kwargs))


def subtype_score(report: str, reference: str, **kwargs):
    return evaluate(report, reference, **kwargs)["metrics"]["subtype"]


class EvaluationV4SubtypeTests(unittest.TestCase):
    def test_intraductal_carcinoma_requires_breast_anatomy(self):
        ref = eval_v4.reference("Diagnosis: ductal carcinoma in situ", site="breast")
        self.assertEqual(eval_v4.evaluate("Diagnosis: intraductal carcinoma", ref)["metrics"]["subtype"], 1)
        for text, context in [("intraductal carcinoma", None),
                              ("intraductal carcinoma of prostate", "prostate")]:
            with self.subTest(text=text, context=context):
                pred = eval_v4.extract_subtype("Diagnosis: " + text, context=context)
                self.assertNotEqual(pred["value"], "ductal carcinoma in situ")

    def test_synonym_parity_is_symmetric_for_renal_entities(self):
        pairs = [
            ("clear cell renal cell carcinoma", "clear cell RCC"),
            ("papillary renal cell carcinoma", "papillary RCC"),
            ("chromophobe renal cell carcinoma", "chromophobe RCC"),
            ("clear cell renal cell carcinoma", "conventional clear cell renal cell carcinoma"),
        ]
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.assertEqual(subtype_score(f"Diagnosis: {left}", f"Diagnosis: {right}"), 1)
                self.assertEqual(subtype_score(f"Diagnosis: {right}", f"Diagnosis: {left}"), 1)

    def test_renal_word_order_aliases_are_symmetric(self):
        pairs = [
            ("renal cell carcinoma, clear cell type", "clear cell renal cell carcinoma"),
            ("renal cell carcinoma, papillary type", "papillary renal cell carcinoma"),
            ("renal cell carcinoma, chromophobe type", "chromophobe renal cell carcinoma"),
            ("conventional renal cell carcinoma, clear cell type", "clear cell RCC"),
        ]
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.assertEqual(subtype_score(f"Diagnosis: {left}", f"Diagnosis: {right}"), 1)
                self.assertEqual(subtype_score(f"Diagnosis: {right}", f"Diagnosis: {left}"), 1)

    def test_generic_rcc_does_not_match_clear_cell(self):
        self.assertEqual(subtype_score("Diagnosis: renal cell carcinoma", "Diagnosis: clear cell RCC"), 0)

    def test_clear_cell_does_not_match_generic_rcc(self):
        self.assertIsNone(subtype_score("Diagnosis: clear cell RCC", "Diagnosis: renal cell carcinoma"))

    def test_generic_rcc_is_not_assumed_clear_cell(self):
        result = evaluate("Diagnosis: renal cell carcinoma", "Diagnosis: clear cell renal cell carcinoma")
        self.assertEqual(result["fields"]["subtype"]["prediction"], "renal cell carcinoma")
        self.assertEqual(result["metrics"]["subtype"], 0)

    def test_renal_sibling_entities_remain_distinct(self):
        pairs = [
            ("clear cell RCC", "papillary RCC"),
            ("clear cell RCC", "chromophobe RCC"),
            ("papillary RCC", "chromophobe RCC"),
            ("papillary RCC", "clear cell RCC"),
            ("chromophobe RCC", "papillary RCC"),
        ]
        for predicted, reference in pairs:
            with self.subTest(predicted=predicted, reference=reference):
                self.assertEqual(subtype_score(f"Diagnosis: {predicted}", f"Diagnosis: {reference}"), 0)

    def test_cervical_adenocarcinoma_synonyms_are_symmetric(self):
        pairs = [
            ("cervical adenocarcinoma", "adenocarcinoma of the cervix"),
            ("endocervical adenocarcinoma", "cervical glandular adenocarcinoma"),
        ]
        for left, right in pairs:
            with self.subTest(left=left, right=right):
                self.assertEqual(subtype_score(f"Diagnosis: {left}", f"Diagnosis: {right}"), 1)
                self.assertEqual(subtype_score(f"Diagnosis: {right}", f"Diagnosis: {left}"), 1)

    def test_cervical_adenocarcinoma_is_not_squamous_cell_carcinoma(self):
        self.assertEqual(subtype_score("Diagnosis: cervical adenocarcinoma", "Diagnosis: cervical squamous cell carcinoma"), 0)

    def test_cesc_compatibility_does_not_change_exact_subtype(self):
        result = evaluate("Diagnosis: squamous cell carcinoma", "", cohort="TCGA-CESC")
        self.assertIsNone(result["metrics"]["subtype"])
        self.assertEqual(result["metrics"]["cohort_compatibility"], 1)

    def test_breast_ductal_and_lobular_are_distinct(self):
        self.assertEqual(subtype_score("Diagnosis: invasive ductal carcinoma", "Diagnosis: invasive lobular carcinoma"), 0)

    def test_breast_lobular_and_ductal_are_distinct_symmetrically(self):
        self.assertEqual(subtype_score("Diagnosis: invasive lobular carcinoma", "Diagnosis: invasive ductal carcinoma"), 0)

    def test_physician_structured_subtype_is_exact_target(self):
        result = eval_v4.evaluate(
            "Diagnosis: invasive lobular carcinoma",
            eval_v4.reference("Diagnosis: invasive breast carcinoma",
                              structured={"histologic_subtype": "invasive lobular carcinoma"}),
        )
        self.assertEqual(result["metrics"]["subtype"], 1)

    def test_parent_breast_label_does_not_conflict_with_child(self):
        result = evaluate(
            "Primary diagnosis: invasive ductal carcinoma\nBroad type: carcinoma\nSubtype: invasive breast carcinoma",
            "Diagnosis: invasive ductal carcinoma",
        )
        self.assertEqual(result["metrics"]["subtype"], 1)
        self.assertNotEqual(result["prediction"]["subtype"]["status"], "conflict")

    def test_dcis_does_not_match_invasive_ductal_carcinoma(self):
        self.assertEqual(subtype_score("Diagnosis: ductal carcinoma in situ", "Diagnosis: invasive ductal carcinoma"), 0)

    def test_invasive_ductal_does_not_match_dcis(self):
        self.assertEqual(subtype_score("Diagnosis: invasive ductal carcinoma", "Diagnosis: ductal carcinoma in situ"), 0)

    def test_differential_diagnosis_does_not_become_primary(self):
        result = evaluate(
            "Primary diagnosis: squamous cell carcinoma\nDifferential diagnoses: adenocarcinoma",
            "Diagnosis: squamous cell carcinoma",
        )
        self.assertEqual(result["metrics"]["subtype"], 1)

    def test_morphology_mentions_do_not_replace_primary_diagnosis(self):
        result = evaluate(
            "Diagnosis: invasive ductal carcinoma\nKey Microscopic Findings: papillary architecture and clear cytoplasm",
            "Diagnosis: invasive ductal carcinoma",
        )
        self.assertEqual(result["metrics"]["subtype"], 1)

    def test_clinical_history_does_not_become_primary_diagnosis(self):
        result = evaluate(
            "Clinical history: prior lung adenocarcinoma\nDiagnosis: colorectal adenocarcinoma",
            "Diagnosis: colorectal adenocarcinoma",
        )
        self.assertEqual(result["metrics"]["subtype"], 1)

    def test_negated_entity_is_not_full_credit(self):
        result = evaluate("Diagnosis: no evidence of clear cell RCC", "Diagnosis: clear cell renal cell carcinoma")
        self.assertNotEqual(result["metrics"]["subtype"], 1)

    def test_uncertain_entity_is_not_full_credit(self):
        result = evaluate("Diagnosis: possible clear cell RCC", "Diagnosis: clear cell renal cell carcinoma")
        self.assertNotEqual(result["metrics"]["subtype"], 1)

    def test_trailing_grade_uncertainty_does_not_uncertain_asserted_histology(self):
        for diagnosis in (
            "Clear cell RCC, grade unknown",
            "Clear cell RCC, grade 2 or 3",
            "Clear cell RCC, grade 2 or 3, stage unknown",
        ):
            with self.subTest(diagnosis=diagnosis):
                result = evaluate(
                    f"Diagnosis: {diagnosis}",
                    "Diagnosis: clear cell renal cell carcinoma",
                )
                self.assertEqual(result["prediction"]["subtype"]["status"], "asserted")
                self.assertEqual(result["metrics"]["subtype"], 1)

    def test_trailing_suspected_without_other_field_remains_uncertain(self):
        result = evaluate(
            "Diagnosis: clear cell RCC, suspected",
            "Diagnosis: clear cell renal cell carcinoma",
        )
        self.assertEqual(result["prediction"]["subtype"]["status"], "uncertain")
        self.assertEqual(result["metrics"]["subtype"], 0)

    def test_non_invasive_prefix_does_not_assert_invasive_subtype(self):
        result = evaluate(
            "Diagnosis: non-invasive ductal carcinoma",
            "Diagnosis: invasive ductal carcinoma",
        )
        self.assertNotEqual(result["prediction"]["subtype"]["status"], "asserted")
        self.assertEqual(result["metrics"]["subtype"], 0)

    def test_favor_entity_is_not_full_credit(self):
        result = evaluate("Diagnosis: favor clear cell RCC", "Diagnosis: clear cell renal cell carcinoma")
        self.assertNotEqual(result["metrics"]["subtype"], 1)

    def test_versus_entities_are_not_full_credit(self):
        result = evaluate("Diagnosis: clear cell RCC versus papillary RCC", "Diagnosis: clear cell renal cell carcinoma")
        self.assertNotEqual(result["metrics"]["subtype"], 1)

    def test_multiple_sibling_entities_are_not_full_credit(self):
        result = evaluate("Diagnosis: clear cell RCC and papillary RCC", "Diagnosis: clear cell renal cell carcinoma")
        self.assertNotEqual(result["metrics"]["subtype"], 1)

    def test_distinct_entities_in_same_clause_are_a_prediction_conflict(self):
        result = evaluate("Diagnosis: clear cell RCC and papillary RCC", "Diagnosis: clear cell RCC")
        self.assertEqual(result["prediction"]["subtype"]["status"], "conflict")
        self.assertEqual(result["metrics"]["subtype"], 0)

    def test_reference_uncertainty_makes_subtype_not_applicable(self):
        result = evaluate("Diagnosis: clear cell RCC", "Diagnosis: possible clear cell RCC")
        self.assertFalse(result["fields"]["subtype"]["applicable"])
        self.assertIsNone(result["metrics"]["subtype"])

    def test_asserted_entity_and_same_or_ancestor_denial_conflict(self):
        for diagnosis, context in (
            ("Urothelial carcinoma. No urothelial carcinoma is identified.", None),
            ("Clear cell renal cell carcinoma. No renal cell carcinoma is identified.", "kidney"),
        ):
            with self.subTest(diagnosis=diagnosis):
                prediction = eval_v4.extract_subtype("Diagnosis: " + diagnosis, context=context)
                self.assertEqual(prediction["status"], "conflict")
                self.assertIsNone(prediction["value"])
        ref = eval_v4.reference(
            "Diagnosis: urothelial carcinoma of bladder. No prostate carcinoma is identified.",
            cancer_type="TCGA-BLCA",
        )
        subtype = ref["v4_reference"]["subtype"]
        self.assertEqual(subtype["value"], "urothelial carcinoma")
        self.assertEqual(subtype["status"], "asserted")

    def test_generic_reference_subtype_is_not_applicable(self):
        result = evaluate("Diagnosis: invasive ductal carcinoma", "Diagnosis: carcinoma")
        self.assertFalse(result["fields"]["subtype"]["applicable"])
        self.assertIsNone(result["metrics"]["subtype"])

    def test_ambiguous_reference_subtype_is_not_applicable(self):
        result = evaluate("Diagnosis: invasive ductal carcinoma", "Diagnosis: adenocarcinoma versus squamous cell carcinoma")
        self.assertFalse(result["fields"]["subtype"]["applicable"])
        self.assertIsNone(result["metrics"]["subtype"])

    def test_abstention_on_applicable_subtype_is_zero_not_null(self):
        result = evaluate("Diagnosis: not specified", "Diagnosis: clear cell RCC")
        self.assertTrue(result["fields"]["subtype"]["applicable"])
        self.assertIsNone(result["fields"]["subtype"]["prediction"])
        self.assertEqual(result["metrics"]["subtype"], 0)

    def test_cancer_type_does_not_supply_a_patient_subtype(self):
        result = eval_v4.evaluate(
            "Diagnosis: clear cell RCC",
            eval_v4.reference("", cancer_type="TCGA-KIRC"),
        )
        self.assertFalse(result["fields"]["subtype"]["applicable"])
        self.assertIsNone(result["metrics"]["subtype"])

    def test_cancer_type_provides_organ_context_without_subtype_prediction(self):
        result = eval_v4.evaluate(
            "Diagnosis: papillary renal cell carcinoma",
            eval_v4.reference("Diagnosis: clear cell RCC", cancer_type="TCGA-KIRC"),
        )
        self.assertEqual(result["metrics"]["subtype"], 0)

    def test_plain_adenocarcinoma_gets_cervical_or_colorectal_context_only(self):
        cases = [
            ("TCGA-CESC", "cervical adenocarcinoma"),
            ("TCGA-COAD", "colorectal adenocarcinoma"),
        ]
        for cancer_type, expected in cases:
            with self.subTest(cancer_type=cancer_type):
                ref = eval_v4.reference("Diagnosis: adenocarcinoma", cancer_type=cancer_type)
                self.assertEqual(ref["v4_reference"]["subtype"]["value"], expected)
                result = eval_v4.evaluate(f"Diagnosis: {expected}", ref)
                self.assertEqual(result["metrics"]["subtype"], 1)

        kidney_ref = eval_v4.reference("Diagnosis: adenocarcinoma", cancer_type="TCGA-KIRC")
        self.assertEqual(kidney_ref["v4_reference"]["subtype"]["value"], "adenocarcinoma")
        self.assertFalse(kidney_ref["v4_reference"]["subtype"]["specific"])
        kidney_result = eval_v4.evaluate("Diagnosis: renal cell adenocarcinoma", kidney_ref)
        self.assertIsNone(kidney_result["metrics"]["subtype"])

    def test_organ_contradiction_is_not_a_subtype_match(self):
        result = eval_v4.evaluate(
            "Diagnosis: lung adenocarcinoma",
            eval_v4.reference("Diagnosis: colorectal adenocarcinoma", cancer_type="TCGA-COAD"),
        )
        self.assertEqual(result["prediction"]["subtype"]["status"], "organ_conflict")
        self.assertEqual(result["metrics"]["subtype"], 0)

    def test_same_kidney_cancer_types_share_neutral_organ_context(self):
        for cancer_type in ("TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"):
            with self.subTest(cancer_type=cancer_type):
                result = eval_v4.evaluate(
                    "Diagnosis: papillary renal cell carcinoma",
                    eval_v4.reference("Diagnosis: papillary renal cell carcinoma", cancer_type=cancer_type),
                )
                self.assertEqual(result["metrics"]["subtype"], 1)

    def test_same_lung_cancer_types_share_neutral_organ_context(self):
        for cancer_type in ("TCGA-LUAD", "TCGA-LUSC"):
            with self.subTest(cancer_type=cancer_type):
                result = eval_v4.evaluate(
                    "Diagnosis: lung adenocarcinoma",
                    eval_v4.reference("Diagnosis: lung adenocarcinoma", cancer_type=cancer_type),
                )
                self.assertEqual(result["metrics"]["subtype"], 1)

    def test_lung_entity_conflicts_with_colorectal_context(self):
        result = eval_v4.evaluate(
            "Diagnosis: lung adenocarcinoma",
            eval_v4.reference("Diagnosis: colorectal adenocarcinoma", cancer_type="TCGA-COAD"),
        )
        self.assertEqual(result["metrics"]["subtype"], 0)

    def test_private_project_codes_do_not_force_tcga_cohort_scoring(self):
        for cancer_type in ("P002", "P200"):
            with self.subTest(cancer_type=cancer_type):
                result = eval_v4.evaluate("Diagnosis: clear cell RCC", eval_v4.reference("", cancer_type=cancer_type))
                self.assertIsNone(result["metrics"]["cohort_compatibility"])

    def test_cohort_compatibility_is_separate_from_exact_subtype(self):
        result = eval_v4.evaluate(
            "Diagnosis: squamous cell carcinoma",
            eval_v4.reference("", cohort="TCGA-CESC"),
        )
        self.assertEqual(result["metrics"]["cohort_compatibility"], 1)
        self.assertIsNone(result["metrics"]["subtype"])

    def test_broad_squamous_label_is_cohort_compatible_without_case_leaf(self):
        result = eval_v4.evaluate("Diagnosis: squamous cell carcinoma",
                                  eval_v4.reference("", cohort="TCGA-CESC"))
        self.assertFalse(result["fields"]["subtype"]["applicable"])
        self.assertIsNone(result["metrics"]["subtype"])
        self.assertEqual(result["metrics"]["cohort_compatibility"], 1)

    def test_generic_rcc_does_not_earn_clear_cell_cohort_compatibility(self):
        result = eval_v4.evaluate("Diagnosis: renal cell carcinoma", eval_v4.reference("", cohort="TCGA-KIRC"))
        self.assertEqual(result["metrics"]["cohort_compatibility"], 0)

    def test_wrong_renal_sibling_fails_cohort_compatibility(self):
        for cohort in ("TCGA-KIRC", "TCGA-KIRP", "TCGA-KICH"):
            for subtype, owner in (("clear cell RCC", "TCGA-KIRC"),
                                   ("papillary RCC", "TCGA-KIRP"),
                                   ("chromophobe RCC", "TCGA-KICH")):
                if cohort != owner:
                    with self.subTest(cohort=cohort, subtype=subtype):
                        result = eval_v4.evaluate(f"Diagnosis: {subtype}", eval_v4.reference("", cohort=cohort))
                        self.assertEqual(result["metrics"]["cohort_compatibility"], 0)

    def test_report_hash_and_prediction_are_reference_invariant(self):
        report = "Diagnosis: clear cell RCC\nGrade: 2"
        first = eval_v4.evaluate(report, eval_v4.reference("Diagnosis: clear cell RCC"))
        second = eval_v4.evaluate(report, eval_v4.reference("Diagnosis: papillary RCC"))
        self.assertEqual(first["report_sha256"], second["report_sha256"])
        self.assertEqual(first["prediction"], second["prediction"])

    def test_report_hash_is_sha256_of_exact_report(self):
        import hashlib

        report = "Diagnosis: clear cell RCC\nGrade: 2"
        result = eval_v4.evaluate(report, eval_v4.reference("Diagnosis: clear cell RCC"))
        self.assertEqual(result["report_sha256"], hashlib.sha256(report.encode()).hexdigest())

    def test_evaluation_does_not_mutate_a_reused_reference(self):
        ref = eval_v4.reference(
            "Diagnosis: invasive ductal carcinoma\nGrade: grade 2\nNecrosis: present",
            structured={"histologic_subtype": "invasive ductal carcinoma", "grade": "grade 2", "necrosis": "present"},
        )
        before = deepcopy(ref)
        first = eval_v4.evaluate("Diagnosis: invasive ductal carcinoma\nGrade: grade 2", ref)
        second = eval_v4.evaluate("Diagnosis: invasive lobular carcinoma\nGrade: grade 3", ref)
        self.assertEqual(ref, before)
        self.assertEqual(first["reference"], second["reference"])

    def test_prediction_does_not_depend_on_reference_sibling(self):
        report = "Diagnosis: clear cell RCC"
        references = [
            "Diagnosis: clear cell renal cell carcinoma",
            "Diagnosis: papillary RCC",
            "Diagnosis: chromophobe RCC",
        ]
        predictions = [eval_v4.evaluate(report, eval_v4.reference(text))["prediction"]["subtype"]
                       for text in references]
        self.assertEqual(predictions[0], predictions[1])
        self.assertEqual(predictions[1], predictions[2])

    def test_concomitant_dcis_does_not_displace_invasive_primary(self):
        result = evaluate(
            "Diagnosis: invasive ductal carcinoma with associated DCIS",
            "Diagnosis: invasive ductal carcinoma",
        )
        self.assertEqual(result["metrics"]["subtype"], 1)
        self.assertEqual(result["prediction"]["subtype"]["value"], "invasive ductal carcinoma")

    def test_reference_defined_applicability_mask_includes_abstentions(self):
        ref = eval_v4.reference(
            "Diagnosis: invasive ductal carcinoma\nGrade: grade 2\nNecrosis: present",
            structured={"histologic_subtype": "invasive ductal carcinoma", "grade": "grade 2", "necrosis": "present"},
        )
        result = eval_v4.evaluate("Diagnosis: not specified", ref)
        for key in ("subtype", "grade", "necrosis"):
            self.assertTrue(result["fields"][key]["applicable"], key)
        self.assertEqual(result["metrics"]["subtype"], 0)
        self.assertEqual(result["metrics"]["grade_exact"], 0)
        self.assertEqual(result["metrics"]["necrosis"], 0)

    def test_prism_diagnosis_only_sets_other_capability_metrics_na(self):
        ref = eval_v4.reference("Diagnosis: invasive ductal carcinoma\nGrade: grade 2\nNecrosis: present")
        result = eval_v4.evaluate("Diagnosis: invasive ductal carcinoma", ref, diagnosis_only=True)
        for key in ("invasion", "necrosis", "grade_exact", "grade_ordinal", "grade_coverage",
                    "reference_coverage", "morphology_fact_f1", "critical_positive_recall",
                    "contradiction_rate", "rouge1_f1", "rouge2_f1"):
            self.assertIsNone(result["metrics"][key], key)
        self.assertEqual(result["metrics"]["subtype"], 1)

    def test_v4_reference_coverage_is_separate_and_prism_coverage_is_na(self):
        report = "Diagnosis: carcinoma\nGrade: grade 3\nNecrosis: absent"
        ref = eval_v4.reference("Diagnosis: carcinoma\nGrade: grade 2\nNecrosis: present",
                                cancer_type="TCGA-BRCA")
        result = eval_v4.evaluate(report, ref)
        self.assertEqual(result["metrics"]["reference_coverage"],
                         result["comparison_to_v3"]["metrics"]["reference_coverage"])
        self.assertIn("reference_coverage_v4", result["metrics"])
        self.assertEqual(result["metric_scopes"]["reference_coverage_v4"], "v4_reference_applicability")

        prism = eval_v4.evaluate("Diagnosis: carcinoma", ref, diagnosis_only=True)
        self.assertIsNone(prism["metrics"]["reference_coverage_v4"])
        self.assertEqual(prism["metrics"]["reference_coverage"],
                         prism["comparison_to_v3"]["metrics"]["reference_coverage"])

    def test_lung_context_completes_generic_adenocarcinoma_without_subtype_leak(self):
        ref = eval_v4.reference("Diagnosis: adenocarcinoma", site="lung")
        result = eval_v4.evaluate("Diagnosis: adenocarcinoma", ref)
        self.assertEqual(result["prediction"]["subtype"]["value"], "lung adenocarcinoma")
        self.assertEqual(result["metrics"]["subtype"], 1)
        self.assertIsNone(result["metrics"]["cohort_compatibility"])
        results = {
            code: eval_v4.evaluate(
                "Diagnosis: adenocarcinoma",
                eval_v4.reference("Diagnosis: adenocarcinoma", cancer_type=code),
            )
            for code in ("TCGA-LUAD", "TCGA-LUSC")
        }
        predictions = {result["prediction"]["subtype"]["value"] for result in results.values()}
        self.assertEqual(predictions, {"lung adenocarcinoma"})
        self.assertEqual(results["TCGA-LUAD"]["metrics"]["subtype"], 1)
        self.assertEqual(results["TCGA-LUSC"]["metrics"]["subtype"], 1)
        self.assertEqual(results["TCGA-LUAD"]["metrics"]["cohort_compatibility"], 1)
        self.assertEqual(results["TCGA-LUSC"]["metrics"]["cohort_compatibility"], 0)

    def test_v3_non_subtype_non_diagnosis_metrics_are_preserved(self):
        report = "Diagnosis: invasive ductal carcinoma\nGrade: grade 2\nNecrosis: absent\nInvasion: absent"
        reference = "Diagnosis: invasive ductal carcinoma\nGrade: grade 2\nNecrosis: absent\nInvasion: absent"
        old = eval_v3.evaluate(report, eval_v3.reference(reference), reference_text=reference)
        new = eval_v4.evaluate(report, eval_v4.reference(reference), reference_text=reference)
        excluded = {"subtype", "diagnosis", "cohort_compatibility"}
        for key, value in old["metrics"].items():
            if key not in excluded:
                self.assertEqual(new["metrics"][key], value, key)

    def test_non_subtype_legacy_metrics_stay_v3_equivalent_across_calls(self):
        report = "Diagnosis: invasive ductal carcinoma\nGrade: grade 2\nNecrosis: absent\nInvasion: absent"
        references = [
            "Diagnosis: invasive ductal carcinoma\nGrade: grade 2\nNecrosis: absent\nInvasion: absent",
            "Diagnosis: invasive lobular carcinoma\nGrade: grade 3\nNecrosis: present\nInvasion: present",
            "Diagnosis: carcinoma\nGrade: not assessable",
        ]
        excluded = {"subtype", "diagnosis", "cohort_compatibility"}
        for reference in references:
            with self.subTest(reference=reference):
                expected = eval_v3.evaluate(report, eval_v3.reference(reference))
                actual = eval_v4.evaluate(report, eval_v4.reference(reference))
                for key, value in expected["metrics"].items():
                    if key not in excluded:
                        self.assertEqual(actual["metrics"][key], value, key)


if __name__ == "__main__":
    unittest.main()
