import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().parents[1] / "compare_reports" / "app.py"
SINGLE_APP_PATH = Path(__file__).resolve().parents[1] / "single_report" / "app.py"


def load_app_module():
    fake_openslide = types.ModuleType("openslide")
    fake_openslide.OpenSlide = object
    fake_deepzoom = types.ModuleType("openslide.deepzoom")
    fake_deepzoom.DeepZoomGenerator = object
    fake_openslide.deepzoom = fake_deepzoom
    sys.modules["openslide"] = fake_openslide
    sys.modules["openslide.deepzoom"] = fake_deepzoom
    spec = importlib.util.spec_from_file_location("tutorial_compare_app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_single_app_module():
    spec = importlib.util.spec_from_file_location("tutorial_single_app", SINGLE_APP_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CompareIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_module = load_app_module()
        cls.single_app_module = load_single_app_module()

    def test_scores_are_isolated_by_random_session_and_case(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            slide = root / "slide.svs"
            slide.touch()
            reports = {}
            for index in range(5):
                report = root / f"candidate_{index + 1}.txt"
                report.write_text(
                    "SYNTHETIC TRAINING TEXT — NOT A PATIENT REPORT\n"
                    f"Diagnosis: candidate {index + 1}\n",
                    encoding="utf-8",
                )
                reports[f"private_key_{index + 1}"] = str(report)
            manifest = root / "cases.json"
            manifest.write_text(json.dumps([{
                "case_id": "synthetic-001",
                "anon_id": "SYNTHETIC-001",
                "site": "example",
                "slide_path": str(slide),
                "reports": reports,
            }]), encoding="utf-8")

            output = root / "scores"
            app = self.app_module.CompareApp(
                manifest, output, root / "dzi", root / "assignment_secret.txt", None
            )
            case_id = "SYNTHETIC-001"
            first = app.reports_for(case_id, "session_A1234567")
            again = app.reports_for(case_id, "session_A1234567")
            self.assertEqual(first, again)
            self.assertEqual({item["label"] for item in first}, set("ABCDE"))
            self.assertNotIn("private_key_1", json.dumps(first))

            payload = {
                "rank_scale": self.app_module.RANK_SCALE,
                "reader": {"reader_id": "session_A1234567", "name": "must not persist"},
                "base_revision": 0,
                "status": "draft",
                "ratings": {label: {} for label in "ABCDE"},
            }
            app.write_annotation(case_id, "session_A1234567", payload)
            stored = app.read_annotation(case_id, "session_A1234567")
            self.assertIsNotNone(stored)
            self.assertEqual(stored["reader"], {"reader_id": "session_A1234567"})
            self.assertIsNone(app.read_annotation(case_id, "session_B7654321"))
            self.assertNotEqual(
                app.annotation_path(case_id, "session_A1234567"),
                app.annotation_path(case_id, "session_B7654321"),
            )

    def test_single_report_storage_keeps_only_random_session_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            slide = root / "slide.svs"
            slide.touch()
            manifest = root / "cases.json"
            manifest.write_text(json.dumps([{
                "case_id": "synthetic-case-001",
                "anon_id": "SYNTHETIC-001",
                "site": "example",
                "slide_path": str(slide),
            }]), encoding="utf-8")
            app = self.single_app_module.HumanEvalApp(
                data_root=root,
                annotation_root=root / "single_scores",
                manifest_path=manifest,
                dzi_cache_root=root / "single_dzi",
                deployment_mode="local",
                bind_notice="local test",
            )
            payload = {
                "status": "draft",
                "reader": {"reader_id": "session_C1234567", "name": "must not persist"},
                "structured_facts": {},
                "narrative_report": {},
                "diagnostic_rois": [],
                "review_scores": {},
            }
            record = app.write_annotation("SYNTHETIC-001", "session_C1234567", payload)
            self.assertEqual(record["reader"], {"reader_id": "session_C1234567"})
            self.assertIsNone(record["data_root"])


if __name__ == "__main__":
    unittest.main()
