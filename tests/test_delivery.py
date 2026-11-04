"""Delivery boundary checks, without model/API access or external downloads."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pathxai_tutorial import sample
from pathxai_tutorial import models
from pathxai_tutorial.paths import activate_engine


class Reply(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


class DeliveryTests(unittest.TestCase):
    def spec(self, data):
        return dict(file_id="fixture", file_name="sample.svs", file_size=len(data),
                    md5sum=hashlib.md5(data).hexdigest(), sha256=hashlib.sha256(data).hexdigest(),
                    access="open", metadata_url="https://example.invalid/metadata",
                    download_url="https://example.invalid/data")

    def test_partial_download_resumes_and_verifies(self):
        data = b"fixture-slide-bytes"
        spec = self.spec(data)
        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "sample.svs.part"
            partial.write_bytes(data[:7])
            replies = [Reply(json.dumps({"data": spec}).encode()),
                       Reply(data[7:], 206, {"Content-Range": f"7-{len(data)-1}/{len(data)}"})]
            with patch.object(sample, "sample_metadata", return_value=spec), \
                 patch.object(sample.urllib.request, "urlopen", side_effect=replies) as request:
                result = sample.download_sample(tmp, progress=None)
            self.assertEqual(result.read_bytes(), data)
            self.assertFalse(partial.exists())
            self.assertEqual(request.call_args_list[1].args[0].get_header("Range"), "bytes=7-")

    def test_invalid_range_never_appends(self):
        data = b"fixture-slide-bytes"
        spec = self.spec(data)
        with tempfile.TemporaryDirectory() as tmp:
            partial = Path(tmp) / "sample.svs.part"
            partial.write_bytes(data[:7])
            replies = [Reply(json.dumps({"data": spec}).encode()),
                       Reply(data, 206, {"Content-Range": f"bytes 0-{len(data)-1}/{len(data)}"})]
            with patch.object(sample, "sample_metadata", return_value=spec), \
                 patch.object(sample.urllib.request, "urlopen", side_effect=replies):
                with self.assertRaisesRegex(ValueError, "unexpected range"):
                    sample.download_sample(tmp, progress=None)
            self.assertEqual(partial.read_bytes(), data[:7])

    def test_corrupt_complete_download_is_not_promoted(self):
        data = b"fixture-slide-bytes"
        spec = self.spec(data)
        with tempfile.TemporaryDirectory() as tmp:
            replies = [Reply(json.dumps({"data": spec}).encode()), Reply(b"x"*len(data))]
            with patch.object(sample, "sample_metadata", return_value=spec), \
                 patch.object(sample.urllib.request, "urlopen", side_effect=replies):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    sample.download_sample(tmp, progress=None)
            self.assertFalse((Path(tmp) / "sample.svs").exists())

    def test_release_contract_fetches_adapter_and_upstreams_separately(self):
        calls = []
        def fetch(**kwargs):
            calls.append(kwargs)
            return "/fixture/" + kwargs["filename"]
        fake_hub = types.SimpleNamespace(hf_hub_download=fetch)
        with patch.dict(sys.modules, {"huggingface_hub": fake_hub}), \
             patch.object(models, "verify_asset", side_effect=lambda p, s: Path(p)):
            result = models.download_models(d1_repo_id="fixture/PathXAI-D1", d1_revision="a"*40)
        self.assertEqual([c["repo_id"] for c in calls],
                         ["fixture/PathXAI-D1", "facebook/sam3", "MahmoodLab/conch"])
        self.assertEqual(calls[0]["filename"], "weights/pathxai_d1_adapter.pth")
        self.assertIn("sam3_path", result)
        self.assertNotIn("keep_dir", result)

    def test_notebook_backend_does_not_leak_to_pipeline_worker(self):
        activate_engine()
        from pathxai_demo.engine import _pipeline_child_environment
        with patch.dict("os.environ", {"MPLBACKEND": "module://matplotlib_inline.backend_inline", "QWEN_API_KEY": "test-only"}):
            env = _pipeline_child_environment()
        self.assertEqual(env["MPLBACKEND"], "Agg")
        self.assertNotIn("QWEN_API_KEY", env)

    def test_config_preserves_virtual_environment_interpreter(self):
        from pathxai_tutorial.inference import make_config
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "venv/bin/python"
            exe.parent.mkdir(parents=True)
            exe.symlink_to(sys.executable)
            config = make_config(segmentation_python=str(exe), pipeline_python=str(exe),
                                 qwen_api="http://localhost:1", qwen_model="fixture")
            self.assertEqual(config["pipeline_python"], str(exe))
            self.assertEqual(config["seg_python"], str(exe))


if __name__ == "__main__":
    unittest.main()
