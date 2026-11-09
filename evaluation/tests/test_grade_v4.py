"""Method-blind synthetic grade extraction regression tests."""
import unittest
from ..vendor import grade_v4 as grade, eval_v4 as evaluator

class GradeRevisionContracts(unittest.TestCase):
    def test_definite_and_hedged_repetition(self):
        self.asserted("Grade: 2\nGrade: suggestive of Grade 2", "histologic3:2")
        for extra in ("possible Grade 3", "Grade 2-3", "not assessable", "Grade 3"):
            self.unasserted("Grade: 2\nGrade: " + extra)

    def test_categorical_ranges_and_bare_heading(self):
        self.unasserted("Grade: intermediate to high grade")
        self.asserted("Histologic grade: 2", "histologic3:2")
        self.asserted("Grade: High (G3)", "histologic3:3")

    def asserted(self, text, expected, context=None, **kwargs):
        result = grade.extract(text, context=context, **kwargs)
        self.assertEqual((result["status"], result["value"]), ("asserted", expected), result)
        return result

    def unasserted(self, text, context=None):
        result = grade.extract(text, context=context)
        self.assertNotEqual(result["status"], "asserted", result)
        self.assertIsNone(result["value"], result)
        return result

    def fixture(self, relative):
        path = DATA / relative
        if not path.is_file():
            self.skipTest("Local public fixture unavailable: " + relative)
        return path.read_text()

    def test_ontology_context_aliases(self):
        for context in ("kidney", "renal"):
            with self.subTest(context=context):
                self.asserted("ISUP grade 2", "renal_isup4:2", context)
                self.asserted("Nuclear grade 2", "nuclear4:2", context)
        for context in ("cns", "brain"):
            with self.subTest(context=context):
                self.asserted("WHO grade II", "who4:2", context)

    def test_system_is_never_supplied_by_cohort_alone(self):
        self.asserted("Grade 2", "histologic3:2", "kidney")
        self.asserted("Grade 2", "histologic3:2", "prostate")
        self.unasserted("WHO grade 2", "lung")
        self.unasserted("ISUP grade group 2", "kidney")

    def test_named_and_generic_same_number_do_not_match(self):
        generic = grade.extract("Grade 3", context="kidney")
        named = grade.extract("Fuhrman nuclear grade 3", context="kidney")
        for prediction, reference in ((generic, named), (named, generic)):
            result = grade.score(prediction, reference)
            self.assertEqual(result["score"], 0.0)
            self.assertFalse(result["compatible_coverage"])
            self.assertEqual(result["ordinal_score"], 0.0)

    def test_named_spelling_and_explicit_denominator(self):
        for text in ("Fuhrmann 2", "Furhman nuclear grade 2 of 4", "Fukrman grade II"):
            with self.subTest(text=text):
                self.asserted(text, "fuhrman4:2", "kidney")
        self.asserted("Nuclear grade 2 of 4", "nuclear4:2")

    def test_structured_grade_forms(self):
        for explicit, expected in (("2", "histologic3:2"), ("II", "histologic3:2"),
                                   ("high", "histologic3:3"), ("WHO II", "who4:2")):
            with self.subTest(explicit=explicit):
                self.asserted("", expected, explicit=explicit)

    def test_ranges_and_uncertainty_do_not_select_first_endpoint(self):
        for text in ("Grade 2-3", "Grade II–III", "Grade 2 to 3", "Grade 2 or 3",
                     "Grade 2 vs 3", "Grade 2 vs. 3", "Grade: 2?", "Suspicious for grade 2",
                     "Grade 2 requires confirmation", "Grade: not assessable"):
            with self.subTest(text=text):
                self.unasserted(text)

    def test_negated_and_nonassertive_grades(self):
        for text in ("Assess for high-grade features", "Consider high-grade carcinoma",
                     "High-grade carcinoma not identified", "High-grade carcinoma is not seen",
                     "No high-grade carcinoma", "Recommend assessing for grade 3"):
            with self.subTest(text=text):
                self.unasserted(text)

    def test_unavailable_fuhrman_does_not_cancel_isup_uncertainty(self):
        for text in ("Fuhrman: NA (ISUP nucleolar grade 2?)",
                     "Fuhrman: NA (suspicious for ISUP nucleolar grade 2)"):
            with self.subTest(text=text):
                self.unasserted(text, "kidney")

    def test_local_negation_does_not_remove_explicit_grade(self):
        self.asserted("Invasive carcinoma of no special type, grade 2", "histologic3:2", "breast")
        self.asserted("No necrosis, grade 2", "histologic3:2", "breast")
        self.asserted("No necrosis; histologic grade 2", "histologic3:2", "breast")

    def test_conflicts_and_uncertain_grade_cannot_be_rescued(self):
        for text in ("Grade: 2; Grade: 3", "Grade: 2; Grade: not assessable"):
            with self.subTest(text=text):
                prediction = self.unasserted(text)
                reference = grade.extract("Grade: 2")
                self.assertEqual(grade.score(prediction, reference)["score"], 0.0)

    def test_histologic_and_nuclear_scopes(self):
        for separator in (", ", "; ", " and "):
            with self.subTest(separator=separator):
                result = self.asserted("Histologic grade 2" + separator + "nuclear grade 3",
                                       "histologic3:2", "breast")
                self.assertEqual(result["by_system"]["nuclear3"]["value"], "nuclear3:3")

    def test_explicit_gleason_and_isup_are_punctuation_invariant(self):
        texts = ["Gleason score 3+4 = 7, ISUP grade group 2",
                 "Gleason score 3+4 = 7; ISUP grade group 2",
                 "Gleason score 3+4 = 7 and ISUP grade group 2",
                 "ISUP grade group 2 (Gleason score 3+4 = 7)"]
        for text in texts:
            with self.subTest(text=text):
                prediction = self.asserted(text, "isup5:2", "prostate")
                for reference_text in ("Gleason score 3+4 = 7", "ISUP grade group 2"):
                    reference = grade.extract(reference_text, context="prostate")
                    self.assertEqual(grade.score(prediction, reference)["score"], 1.0)

    def test_reference_and_prediction_use_the_same_contextual_extractor(self):
        for text, cohort in (("Diagnosis: renal cell carcinoma\nGrade: ISUP grade 2", "KIRC"),
                             ("Diagnosis: astrocytoma\nGrade: WHO grade II", "LGG"),
                             ("Diagnosis: carcinoma\nGrade 2-3", "BRCA")):
            with self.subTest(cohort=cohort):
                reference = evaluator.reference(text, cancer_type=cohort)
                result = evaluator.evaluate(text, reference)
                self.assertEqual(result["prediction"]["grade"], reference["grade"])
                expected = 1.0 if reference["grade"]["status"] == "asserted" else None
                self.assertEqual(result["fields"]["grade"]["score"], expected)




