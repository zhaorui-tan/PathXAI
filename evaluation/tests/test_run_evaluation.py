from pathlib import Path
import tempfile
import unittest

try:
    from evaluation.run_evaluation import evaluate_paths, visible_report_text
except ModuleNotFoundError:
    from PathXAI_tutorial.evaluation.run_evaluation import evaluate_paths, visible_report_text


class ReportEvaluationTests(unittest.TestCase):
    def test_pathxai_workspace_excludes_differentials_and_reference_panels(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.html"
            report.write_text('<aside>Differential: melanoma</aside><section class="report-row">'
                              '<div class="report-row-title">Diagnosis</div>'
                              '<div class="report-row-content">Clear cell <b>renal cell carcinoma</b>.</div>'
                              '<div class="report-row-tags">Claim C1</div></section>'
                              '<section>Reference: squamous cell carcinoma</section>')
            text = visible_report_text(report)
        self.assertIn("renal cell carcinoma", text)
        self.assertNotIn("melanoma", text)
        self.assertNotIn("squamous", text)
        self.assertNotIn("Claim", text)

    def test_html_text_extraction_skips_nonreport_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.html"
            report.write_text("<html><head><title>hidden</title></head><body><h1>Diagnosis</h1><p>Clear cell renal cell carcinoma.</p><script>secret()</script></body></html>", encoding="utf-8")
            text = visible_report_text(report)
        self.assertIn("Diagnosis", text)
        self.assertIn("Clear cell renal cell carcinoma.", text)
        self.assertNotIn("secret", text)
        self.assertNotIn("hidden", text)

    def test_end_to_end_returns_current_schema_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.html"
            reference = Path(tmp) / "reference.txt"
            report.write_text("<main><h2>Diagnosis</h2><p>Clear cell renal cell carcinoma.</p></main>", encoding="utf-8")
            reference.write_text("Diagnosis: Clear cell renal cell carcinoma.", encoding="utf-8")
            result = evaluate_paths(report, reference, cancer_type="TCGA-KIRC")
        self.assertEqual(result["schema"], "eval_v4_report_only_20261017_r5")
        self.assertEqual(result["fields"]["subtype"]["score"], 1.0)
        self.assertEqual(result["fields"]["cohort_compatibility"]["score"], 1.0)
        self.assertEqual(result["run_metadata"]["cancer_type_context"], "TCGA-KIRC")
        self.assertEqual(len(result["run_metadata"]["reference_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
