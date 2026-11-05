"""Public synthetic regression checks for the frozen release wrapper."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

from ..manuscript_metrics import finding_audit
from ..run_evaluation import evaluate_paths, visible_report_text
from ..vendor import eval_v4


class ReleaseParityTests(unittest.TestCase):
    def test_wrapper_preserves_frozen_scorer_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, reference = Path(tmp) / 'report.txt', Path(tmp) / 'reference.txt'
            report.write_text('Diagnosis: Clear cell RCC.\nNecrosis: absent.\nInvasion: present.')
            reference.write_text('Diagnosis: Clear cell renal cell carcinoma.\nNecrosis: present.\nInvasion: present.')
            wrapped = evaluate_paths(report, reference, cancer_type='TCGA-KIRC')
            text = reference.read_text()
            original = eval_v4.evaluate(visible_report_text(report),
                eval_v4.reference(text, cancer_type='TCGA-KIRC'), reference_text=text,
                diagnosis_only=False, narrative_reference=True)
        original_part = {key: value for key, value in wrapped.items()
                         if key not in ('run_metadata', 'manuscript_metrics')}
        self.assertEqual(original_part, original)
        self.assertEqual(wrapped['manuscript_metrics']['diagnostic_mapping'],
                         {key: original['metrics'][key] for key in ('cohort_compatibility', 'diagnosis_cohort')})

    def test_synonyms_match_and_distinct_siblings_do_not(self):
        for left, right, site in [
            ('clear cell RCC', 'conventional clear cell renal cell carcinoma', 'kidney'),
            ('infiltrating ductal carcinoma', 'invasive ductal carcinoma', 'breast'),
            ('intraductal carcinoma', 'ductal carcinoma in situ', 'breast'),
        ]:
            for report, ref in ((left, right), (right, left)):
                with self.subTest(report=report, reference=ref):
                    result = eval_v4.evaluate('Diagnosis: ' + report,
                        eval_v4.reference('Diagnosis: ' + ref, site=site))
                    self.assertEqual(result['metrics']['subtype'], 1)
        for report, ref, site in [
            ('papillary renal cell carcinoma', 'clear cell renal cell carcinoma', 'kidney'),
            ('invasive lobular carcinoma', 'invasive ductal carcinoma', 'breast'),
            ('thymic carcinoma', 'thymoma', 'thymus'),
        ]:
            with self.subTest(report=report, reference=ref):
                result = eval_v4.evaluate('Diagnosis: ' + report,
                    eval_v4.reference('Diagnosis: ' + ref, site=site))
                self.assertEqual(result['metrics']['subtype'], 0)

    def test_finding_audit_partitions_and_denominators(self):
        def fact(status, value=None):
            return {'status': status, 'value': value}
        reference = {'findings': {key: fact('asserted', 'present') for key in 'abcde'}}
        reference['findings']['f'] = fact('missing')
        prediction = {'findings': {
            'a': fact('asserted', 'present'), 'b': fact('asserted', 'absent'),
            'c': fact('missing'), 'd': fact('uncertain'), 'e': fact('conflict'),
            'f': fact('asserted', 'present'),
        }}
        result = finding_audit(prediction, reference)
        self.assertEqual(tuple(result[key] for key in ('D', 'A', 'C', 'E')), (5, 2, 1, 1))
        self.assertEqual(tuple(result[key] for key in ('missing', 'uncertain', 'conflict')), (1, 1, 1))
        self.assertEqual(result['finding_f1'], 2 / 7)
        self.assertEqual(result['assertion_coverage'], 2 / 5)
        self.assertEqual(result['correct_fact_coverage'], 1 / 5)
        self.assertEqual(result['contradiction_asserted'], 1 / 2)
        self.assertEqual((result['unverified_asserted'], result['all_asserted']), (1, 3))
        none_asserted = finding_audit({'findings': {'a': fact('missing')}},
                                     {'findings': {'a': fact('asserted', 'present')}})
        self.assertIsNone(none_asserted['contradiction_asserted'])
        self.assertEqual(none_asserted['finding_f1'], 0)
        empty = finding_audit({'findings': {'a': fact('missing')}},
                             {'findings': {'a': fact('missing')}})
        self.assertIsNone(empty['finding_f1'])

    def test_wrapper_retains_legacy_contradiction_and_adds_manuscript_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, reference = Path(tmp) / 'report.txt', Path(tmp) / 'reference.txt'
            report.write_text('Diagnosis: Clear cell RCC.\nNecrosis: absent.')
            reference.write_text('Diagnosis: Clear cell RCC.\nNecrosis: present.\nInvasion: present.')
            result = evaluate_paths(report, reference, site='kidney')
        audit = result['manuscript_metrics']['findings']
        self.assertEqual((audit['D'], audit['A'], audit['E']), (2, 1, 1))
        self.assertEqual(result['metrics']['contradiction_rate'], 0.5)
        self.assertEqual(audit['contradiction_asserted'], 1)
        self.assertIsNone(result['manuscript_metrics']['grade']['class_balanced_grade'])

    def test_cli_plain_text_and_pathxai_html_and_diagnosis_only(self):
        wrapper = Path(__file__).resolve().parents[1] / 'run_evaluation.py'
        with tempfile.TemporaryDirectory() as tmp:
            reference = Path(tmp) / 'ref.txt'
            reference.write_text('Diagnosis: Clear cell renal cell carcinoma.')
            for suffix, text in [
                ('.txt', 'Diagnosis: Clear cell RCC.'),
                ('.html', '<aside>Differential: papillary RCC</aside><section class="report-row">'
                 '<div class="report-row-title">Diagnosis</div>'
                 '<div class="report-row-content">Clear cell RCC.</div></section>'
                 '<section>Reference: chromophobe RCC</section>'),
            ]:
                report, output = Path(tmp) / ('report' + suffix), Path(tmp) / 'out.json'
                report.write_text(text)
                subprocess.run([sys.executable, str(wrapper), '--report', str(report), '--reference',
                    str(reference), '--site', 'kidney', '--diagnosis-only', '--output', str(output)],
                    check=True, capture_output=True, text=True)
                result = json.loads(output.read_text())
                self.assertEqual(result['metrics']['subtype'], 1)
                self.assertIsNone(result['manuscript_metrics']['findings'])
                self.assertIsNone(result['manuscript_metrics']['grade']['exact_per_case'])
                self.assertEqual(result['run_metadata']['site_context'], 'kidney')


if __name__ == '__main__':
    unittest.main()
