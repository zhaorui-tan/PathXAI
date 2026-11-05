"""Synthetic text-meaning counterexamples from the manuscript parser repair."""
from pathlib import Path
import unittest
from ..vendor import eval_v4 as E

HERE = Path(__file__).resolve().parent / "fixtures"


class TextMeaningTests(unittest.TestCase):
    def extract(self, diagnosis, summary='', context=None):
        return E.extract_subtype('Diagnosis:\n' + diagnosis + '\nFinal Summary:\n' + summary,
                                 context=context, include_summary=True)

    def test_differential_semicolon_list_is_not_multiple_diagnoses(self):
        out = self.extract('Thymoma', 'Thymoma. Differential diagnoses considered: '
                           'Thymic carcinoma; Lymphoma; Metastatic carcinoma.')
        self.assertEqual((out['value'], out['status']), ('thymoma', 'asserted'))

    def test_true_multiple_diagnoses_remain_conflicting(self):
        out = self.extract('Thymoma and thymic carcinoma')
        self.assertEqual(out['status'], 'conflict')

    def test_eg_does_not_remove_differential_qualifier(self):
        out = self.extract('Serous carcinoma', 'Serous carcinoma. Exclusion of mimics '
                           '(e.g., endometrioid carcinoma) requires further studies.')
        self.assertEqual(out['value'], 'serous carcinoma')

    def test_vs_does_not_make_an_unresolved_subtype_asserted(self):
        out = self.extract('Poorly differentiated carcinoma',
                           'The subtype (squamous vs. adenocarcinoma) cannot be '
                           'definitively determined on H&E alone.', 'esophagus')
        self.assertEqual(out['value'], 'carcinoma')
        self.assertFalse(out['specific'])

    def test_morphological_negation_does_not_deny_the_diagnosis(self):
        out = self.extract('Seminoma', 'The lack of glandular formation favors seminoma.')
        self.assertEqual((out['value'], out['status']), ('seminoma', 'asserted'))

    def test_explicit_diagnosis_retraction_still_conflicts(self):
        out = self.extract('Gastric adenocarcinoma', 'The morphology contradicts a diagnosis '
                           'of gastric adenocarcinoma.', 'stomach')
        self.assertEqual(out['status'], 'conflict')

    def test_negative_nodes_do_not_negate_primary_diagnosis(self):
        out = self.extract('Squamous cell carcinoma. Three lymph nodes negative '
                           'for carcinoma (0/3).', context='head_neck')
        self.assertEqual(out['value'], 'squamous cell carcinoma')

    def test_true_negative_diagnosis_is_not_ignored(self):
        out = self.extract('No squamous cell carcinoma is present.', context='head_neck')
        self.assertEqual(out['status'], 'absent')

    def test_unfilled_synoptic_heading_does_not_assert_a_second_disease(self):
        out = self.extract('Invasive ductal carcinoma.\nDUCTAL CARCINOMA IN SITU (DCIS):',
                           context='breast')
        self.assertEqual(out['value'], 'invasive ductal carcinoma')

    def test_gross_and_prior_history_do_not_change_diagnosis(self):
        text = ('Clinical history:\nPapillary renal cell carcinoma\nDiagnosis:\n'
                'Clear cell renal cell carcinoma\nGross description:\n'
                'The old report of papillary renal cell carcinoma was received.')
        self.assertEqual(E.extract_subtype(text, context='kidney')['value'],
                         'clear cell renal cell carcinoma')

    def test_qualified_explicit_subtype_does_not_gain_specificity(self):
        text = ('Diagnosis: Invasive breast carcinoma, not otherwise specified\n'
                'Subtype: Invasive ductal carcinoma (suggested by morphology).')
        out = E.extract_subtype(text, context='breast')
        self.assertEqual(out['value'], 'invasive breast carcinoma')
        self.assertFalse(out['specific'])

    def test_markers_do_not_become_diagnoses(self):
        out = self.extract('Mesothelioma', 'Confirmation requires IHC '
                           '(e.g., positivity for mesothelial markers and '
                           'negativity for adenocarcinoma markers).', 'pleura')
        self.assertEqual(out['value'], 'mesothelioma')

    def test_explicit_diagnosis_after_differential_is_preserved(self):
        out = self.extract('Differential diagnoses include thymoma and carcinoma; '
                           'however, the final diagnosis is thymoma.')
        self.assertEqual(out['value'], 'thymoma')


class AdversarialMeaningTests(unittest.TestCase):
    def test_independent_counterexamples(self):
        import json
        cases = json.loads((HERE / 'adversarial_cases.json').read_text())
        expected = {
            'passive_uncertainty': ('uncertain', None),
            'expanded_negation': ('absent', None),
            'denial_phrase': ('absent', None),
            'backward_negative_scope': ('asserted', 'thymoma'),
            'backward_uncertainty_scope': ('asserted', 'thymoma'),
            'hidden_genuine_conflict': ('conflict', None),
            'multiline_differential': ('missing', None),
            'lost_postgrade_uncertainty': ('uncertain', None),
            'node_scope_overreach': ('absent', None),
            'uncertainty_downgraded_to_missing': ('uncertain', None),
            'comment_explicit_revision': ('conflict', None),
            'true_conflict_control': ('conflict', None),
        }
        for case in cases['cases'] + cases['controls']:
            with self.subTest(case=case['id']):
                out = E.extract_subtype(case['input'])
                self.assertEqual((out['status'], out['value']), expected[case['id']])
                if case['id'] == 'multiline_differential':
                    self.assertTrue(all(h['status'] == 'hypothetical' for h in out['evidence']))
                ref = E.reference(case['input'])['v4_reference']['subtype']
                self.assertEqual(out, ref)

    def test_independent_followup_boundaries(self):
        import json
        cases = json.loads((HERE / 'followup_review.json').read_text())['unseen_boundary_cases']
        for case in cases:
            with self.subTest(case=case['id']):
                expected = case.get('expected_under_confirmed_default_policy', case['expected'])
                out = E.extract_subtype(case['input'])
                self.assertEqual((out['status'], out['value']), (expected['status'], expected['value']))
                out_summary = E.extract_subtype(case['input'], include_summary=True)
                expected_summary = case['expected']
                self.assertEqual((out_summary['status'], out_summary['value']),
                                 (expected_summary['status'], expected_summary['value']))
                self.assertEqual(out, E.reference(case['input'])['v4_reference']['subtype'])

    def test_qualified_specific_diagnosis_preserves_asserted_parent(self):
        for text, value in [
            ('Diagnosis: Diffuse glioma, favoring oligodendroglioma, WHO grade 2', 'diffuse glioma'),
            ('Diagnosis: High-grade diffuse astrocytic glioma with pseudopalisading necrosis, favor glioblastoma.', 'glioma'),
            ('Diagnosis: Carcinoma, possible adenocarcinoma.', 'carcinoma'),
        ]:
            with self.subTest(text=text):
                out = E.extract_subtype(text)
                self.assertEqual((out['status'], out['value']), ('asserted', value))
                self.assertFalse(out['specific'])

    def test_question_mark_qualifies_diagnosis(self):
        self.assertEqual(E.extract_subtype('Diagnosis: thymoma?')['status'], 'uncertain')


