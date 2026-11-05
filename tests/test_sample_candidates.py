"""Strict sample candidate adapter checks; uses tiny fake slides, no WSI/model."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathxai_tutorial import candidates


def _rgb_image(x: int, y: int, width: int, height: int) -> Image.Image:
    return Image.new("RGB", (width, height), ((x * 17) % 256, (y * 31) % 256, (x + y) % 256))


def _png_hash(image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _pixel_hash(image: Image.Image) -> str:
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


class FakeSlide:
    dimensions = (128, 128)
    level_count = 1

    def __init__(self):
        self.closed = False

    def read_region(self, location, level, size):
        x, y = location
        return _rgb_image(x, y, *size)

    def close(self):
        self.closed = True


def _fixture(tmp: Path):
    case_id = "TCGA-OR-A5LL"
    slide_paths = []
    source_slides = []
    for index in range(6):
        name = f"{case_id}-slide-{index}.svs"
        path = tmp / name
        path.write_bytes(f"tiny fake source slide {index}".encode())
        slide_paths.append(path)
        payload = path.read_bytes()
        source_slides.append({
            "basename": name,
            "width": 128,
            "height": 128,
            "mpp": 0.25,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })

    rows = []
    for index in range(20):
        slide_index = index % 6
        x, y, width, height = index, 2, 4, 4
        slide_id = f"{case_id}-slide-{slide_index}"
        roi_id = f"{slide_id}_foundation_x{x}_y{y}_l0_w{width}_h{height}"
        image = _rgb_image(x, y, width, height)
        rows.append({
            "candidate_index": index + 1,
            "slide_id": slide_id,
            "slide_basename": source_slides[slide_index]["basename"],
            "roi_id": roi_id,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "level": 0,
            "source": f"foundation_text_similarity:{slide_id}_tile_{index:05d}",
            "score": 0.5 - index / 100,
            "metadata": {},
            "png_sha256": _png_hash(image),
            "decoded_rgb_sha256": _pixel_hash(image),
        })
    config = {
        "case_id": case_id,
        "source_candidates_sha256": "a" * 64,
        "source_slides": source_slides,
        "candidate_count": 20,
        "candidate_rois": rows,
    }
    config_path = tmp / "sample_candidates.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return slide_paths, config_path, config


class StrictCandidateTests(unittest.TestCase):
    def test_reconstruction_preserves_order_and_original_row_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            slides, config_path, config = _fixture(tmp)
            output = tmp / "crops"
            with patch.object(candidates, "_CONFIG", config_path), patch.object(
                candidates, "_open_slide", side_effect=lambda _path: FakeSlide()
            ):
                rows = candidates.prepare_sample_candidates(slides, output, progress=lambda _msg: None)

            self.assertEqual(len(rows), 20)
            for index, (row, frozen) in enumerate(zip(rows, config["candidate_rois"]), start=1):
                self.assertEqual(row["roi_id"], frozen["roi_id"])
                self.assertEqual(row["score"], frozen["score"])
                self.assertEqual(row["source"], frozen["source"])
                self.assertEqual(row["x"], frozen["x"])
                self.assertEqual(row["y"], frozen["y"])
                self.assertEqual(row["width"], frozen["width"])
                self.assertEqual(row["height"], frozen["height"])
                self.assertEqual(row["level"], frozen["level"])
                self.assertEqual(set(row), {
                    "slide_id", "roi_id", "x", "y", "width", "height", "level",
                    "source", "score", "metadata", "image_path",
                })
                self.assertEqual(row["metadata"]["slide_path"], str(slides[(index - 1) % 6].resolve()))
                with Image.open(row["image_path"]) as image:
                    self.assertEqual(_pixel_hash(image), frozen["decoded_rgb_sha256"])
            receipt = json.loads((output / "reconstruction_receipt.json").read_text())
            self.assertEqual(receipt["verified_count"], 20)
            self.assertEqual(len(receipt["crops"]), 20)

    def test_pixel_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            slides, config_path, config = _fixture(tmp)
            config["candidate_rois"][0]["decoded_rgb_sha256"] = "0" * 64
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(candidates, "_CONFIG", config_path), patch.object(
                candidates, "_open_slide", side_effect=lambda _path: FakeSlide()
            ):
                with self.assertRaisesRegex(ValueError, "Decoded RGB mismatch"):
                    candidates.prepare_sample_candidates(slides, tmp / "crops", progress=lambda _msg: None)
            self.assertFalse((tmp / "crops" / f"{config['candidate_rois'][0]['roi_id']}.png").exists())

    def test_unsafe_roi_id_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            slides, config_path, config = _fixture(tmp)
            config["candidate_rois"][0]["roi_id"] += "../bad"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            with patch.object(candidates, "_CONFIG", config_path), patch.object(
                candidates, "_open_slide", side_effect=lambda _path: FakeSlide()
            ):
                with self.assertRaisesRegex(ValueError, "Unsafe ROI filename"):
                    candidates.prepare_sample_candidates(slides, tmp / "crops", progress=lambda _msg: None)

    def test_partial_reader_open_failure_closes_already_open_reader(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp = Path(temp)
            slides, config_path, _config = _fixture(tmp)
            opened = FakeSlide()

            def open_then_fail(path):
                if Path(path).name == slides[0].name:
                    return opened
                raise RuntimeError("reader open failure")

            with patch.object(candidates, "_CONFIG", config_path), patch.object(
                candidates, "_open_slide", side_effect=open_then_fail
            ):
                with self.assertRaisesRegex(RuntimeError, "reader open failure"):
                    candidates.prepare_sample_candidates(slides, tmp / "crops", progress=lambda _msg: None)
            self.assertTrue(opened.closed)


if __name__ == "__main__":
    unittest.main()
