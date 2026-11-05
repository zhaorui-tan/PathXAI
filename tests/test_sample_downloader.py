"""Focused patient-sample manifest and reuse checks; no network/model needed."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathxai_tutorial import sample


def _slide(case_id, file_name, payload, index, role="diagnostic_slide"):
    return {
        "role": role,
        "file_id": f"file-{index}",
        "file_name": file_name,
        "file_size": len(payload),
        "md5sum": hashlib.md5(payload).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "access": "open",
        "data_type": "Slide Image",
        "data_category": "Biospecimen",
        "sample_type": "Primary Tumor",
        "download_url": f"https://example.test/{index}",
    }


def _small_manifest():
    case_id = "TCGA-OR-A5LL"
    slides = [
        _slide(case_id, f"{case_id}-01Z-00-DX{i}.svs", f"slide-{i}".encode(), i)
        for i in range(1, 3)
    ]
    excluded = [_slide(
        case_id, f"{case_id}-01A-01-TS1.svs", b"ts1", 3,
        role="other_tissue_slide_excluded_from_tutorial",
    )]
    return {
        "case_id": case_id,
        "project_id": "TCGA-ACC",
        "known_site": "adrenal gland / adrenocortical tissue",
        "slides": slides,
        "excluded_slides": excluded,
        "expected_download_bytes": sum(x["file_size"] for x in slides),
    }


class PatientSampleTests(unittest.TestCase):
    def test_patient_manifest_pins_all_dx_slides_and_site_without_diagnosis(self):
        manifest = sample.sample_metadata()
        self.assertEqual(manifest["known_site"], "adrenal gland / adrenocortical tissue")
        self.assertNotIn("diagnosis", manifest)
        self.assertEqual(
            [next(part for part in x["file_name"].split(".")[0].split("-") if part.startswith("DX"))
             for x in manifest["slides"]],
            ["DX1", "DX2", "DX3", "DX4", "DX5"],
        )
        self.assertTrue(all(len(x["sha256"]) == 64 and len(x["md5sum"]) == 32
                            for x in manifest["slides"]))
        self.assertTrue(manifest["excluded_slides"][0]["file_name"].endswith(
            "TS1.852B5675-3534-4684-9FA3-2A03A6199B4E.svs"
        ))

    def test_download_patient_recursively_reuses_hash_verified_inventory(self):
        manifest = _small_manifest()
        with patch.object(sample, "sample_metadata", return_value=manifest), \
             patch.object(sample, "_validate_live_inventory", return_value=[]):
            with tempfile.TemporaryDirectory() as tmp:
                existing = Path(tmp) / "source"
                expected = []
                for idx, spec in enumerate(manifest["slides"]):
                    path = existing / f"nested-{idx}" / spec["file_name"]
                    path.parent.mkdir(parents=True)
                    path.write_bytes(f"slide-{idx + 1}".encode())
                    expected.append(path.resolve())
                ts1 = manifest["excluded_slides"][0]
                ts1_path = existing / "another" / ts1["file_name"]
                ts1_path.parent.mkdir(parents=True)
                ts1_path.write_bytes(b"ts1")
                with patch.object(sample, "_download_one", side_effect=AssertionError("unexpected download")):
                    result = sample.download_patient(
                        data_dir=Path(tmp) / "output", existing_dir=existing, progress=None
                    )
                self.assertEqual(result, expected)
                written = json.loads((Path(tmp) / "output" / "sample_source.json").read_text())
                self.assertEqual(written, manifest)

    def test_download_patient_rejects_corrupt_or_unpinned_existing_slide(self):
        manifest = _small_manifest()
        with patch.object(sample, "sample_metadata", return_value=manifest), \
             patch.object(sample, "_validate_live_inventory", return_value=[]):
            with tempfile.TemporaryDirectory() as tmp:
                existing = Path(tmp) / "source"
                existing.mkdir()
                first = manifest["slides"][0]
                (existing / first["file_name"]).write_bytes(b"x" * first["file_size"])
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    sample.download_patient(
                        data_dir=Path(tmp) / "output", existing_dir=existing, progress=None
                    )
                (existing / first["file_name"]).unlink()
                (existing / f"{manifest['case_id']}-01Z-00-DX99.svs").write_bytes(b"unknown")
                with self.assertRaisesRegex(ValueError, "Unexpected slide"):
                    sample.download_patient(
                        data_dir=Path(tmp) / "output", existing_dir=existing, progress=None
                    )


if __name__ == "__main__":
    unittest.main()
