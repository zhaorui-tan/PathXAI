"""Synthetic localhost requests only; no model endpoint or clinical data."""
import base64
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import importlib.util
from pathlib import Path
import socket
import stat
import sys
import threading

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor/report_engine"))
from pathxai_demo.replay import TraceLLMClient, ReplayLLMClient, ReplayError
from pathxai_demo.identifiers import configure_identifiers, IdentifierClient
from pathxai_demo.request_audit import RequestJournal, AuditWriteError


@pytest.fixture
def server():
    state = {"requests": [], "statuses": [], "drop": 0, "before": None}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(raw)
            state["requests"].append(body)
            if state["before"]:
                state["before"]()
            if state["drop"]:
                state["drop"] -= 1
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            status = state["statuses"].pop(0) if state["statuses"] else 200
            content = body["messages"][-1]["content"]
            if isinstance(content, list):
                content = next(item["text"] for item in reversed(content) if item.get("type") == "text")
            answer = {"choices": [{"message": {"content": content}}]}
            if status >= 400:
                answer = {"error": {"type": "synthetic_error"}}
            payload = json.dumps(answer).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state["url"] = f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions"
    yield state
    httpd.shutdown()
    thread.join()
    httpd.server_close()


class Client:
    model = "synthetic"
    temperature = 0
    def __init__(self, url, retries=0):
        self.url, self.retries = url, retries
    def chat(self, prompt, image_paths=None, system_prompt=None, messages=None, **kwargs):
        body = {"model": self.model, "messages": [*(messages or []),
                {"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]}
        session = requests.Session()
        session.trust_env = False
        session.mount("http://", HTTPAdapter(max_retries=Retry(
            total=self.retries, status_forcelist=[503], allowed_methods={"POST"}, backoff_factor=0)))
        with session:
            r = session.post(self.url, json=body, headers={"Authorization": "Bearer synthetic-secret"}, timeout=2)
            r.raise_for_status()
            return {"content": r.json()["choices"][0]["message"]["content"], "raw": r.json()}
    chat_json = chat


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_success_capture_is_durable_before_send_and_replays(tmp_path, server):
    trace = TraceLLMClient(Client(server["url"]), tmp_path / "trace.jsonl")
    server["before"] = lambda: assert_started(trace.events_path)
    answer = trace.chat("synthetic report", system_prompt="system")
    events = rows(trace.events_path)
    assert [r["sequence"] for r in events] == list(range(len(events)))
    assert events[-1]["event"] == "call_succeeded"
    assert ReplayLLMClient(trace.trace_path, image_backend=Client(server["url"])).chat(
        "synthetic report", system_prompt="system") == answer
    assert len(server["requests"]) == 1
    assert stat.S_IMODE(trace.events_path.stat().st_mode) == 0o600
    assert "synthetic-secret" not in trace.events_path.read_text()


def assert_started(path):
    assert rows(path)[0]["event"] == "call_started"
    assert any(r["event"] == "http_attempt_started" for r in rows(path))


@pytest.mark.parametrize("failure", ["status", "connection"])
def test_internal_retry_has_each_attempt(tmp_path, server, failure):
    if failure == "status":
        server["statuses"] = [503, 200]
    else:
        server["drop"] = 1
    trace = TraceLLMClient(Client(server["url"], retries=1), tmp_path / "trace.jsonl")
    assert trace.chat("retry payload")["content"] == "retry payload"
    events = rows(trace.events_path)
    attempts = [e for e in events if e["event"] == "http_attempt_started"]
    assert [e["attempt"] for e in attempts] == [1, 2]
    assert attempts[0]["body_sha256"] == attempts[1]["body_sha256"]
    assert sum(e["event"] == "http_retry_scheduled" for e in events) == 1
    assert events[-1]["event"] == "call_succeeded"


def test_http_failure_records_request_and_returned_error_body(tmp_path, server):
    server["statuses"] = [400]
    trace = TraceLLMClient(Client(server["url"]), tmp_path / "trace.jsonl")
    with pytest.raises(requests.HTTPError):
        trace.chat_json("failed request")
    events = rows(trace.events_path)
    assert events[-1]["event"] == "call_failed"
    assert events[-1]["error_type"] == "HTTPError"
    assert any(e.get("body_json", {}).get("error", {}).get("type") == "synthetic_error" for e in events)
    assert trace.trace_path.read_text() == ""
    assert "synthetic-secret" not in trace.events_path.read_text()


def test_retry_exhaustion_and_later_call_do_not_look_like_complete_replay(tmp_path, server):
    server["statuses"] = [503, 503, 200]
    trace = TraceLLMClient(Client(server["url"], retries=1), tmp_path / "trace.jsonl")
    with pytest.raises(requests.exceptions.RetryError):
        trace.chat("first fails")
    trace.chat("later succeeds")
    events = rows(trace.events_path)
    assert any(e["event"] == "http_retry_exhausted" for e in events)
    assert {e["call_id"] for e in events if e["event"] == "call_failed"} == {0}
    with pytest.raises(ReplayError):
        ReplayLLMClient(trace.trace_path)


def test_parallel_call_events_are_linked(tmp_path, server):
    trace = TraceLLMClient(Client(server["url"]), tmp_path / "trace.jsonl")
    with ThreadPoolExecutor(max_workers=2) as executor:
        answers = list(executor.map(trace.chat, ["alpha", "beta"]))
    assert {a["content"] for a in answers} == {"alpha", "beta"}
    events = rows(trace.events_path)
    for call_id in (0, 1):
        call = [r for r in events if r["call_id"] == call_id]
        assert call[0]["event"] == "call_started" and call[-1]["event"] == "call_succeeded"
        assert len([r for r in call if r["event"] == "http_attempt_started"]) == 1


def test_image_payload_is_hashed_not_dumped(tmp_path, server):
    raw = b"synthetic pixels"
    encoded = base64.b64encode(raw).decode()
    trace = TraceLLMClient(Client(server["url"]), tmp_path / "trace.jsonl")
    trace.chat("image", messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}]}])
    assert encoded not in trace.events_path.read_text()
    assert '"image_sha256"' in trace.events_path.read_text()


def test_cancellation_and_image_preparation_failure_are_recorded(tmp_path):
    class Cancel:
        def chat(self, *args, **kwargs):
            raise KeyboardInterrupt("do not persist this message")
        chat_json = chat
    trace = TraceLLMClient(Cancel(), tmp_path / "trace.jsonl")
    with pytest.raises(KeyboardInterrupt):
        trace.chat("cancel")
    with pytest.raises(FileNotFoundError):
        trace.chat("missing image", image_paths=[tmp_path / "missing.png"])
    assert [e["error_type"] for e in rows(trace.events_path) if e["event"] == "call_failed"] == ["KeyboardInterrupt", "FileNotFoundError"]
    assert "do not persist this message" not in trace.events_path.read_text()


def test_existing_journal_and_replaced_file_are_rejected(tmp_path):
    path = tmp_path / "events.jsonl"
    journal = RequestJournal(path)
    with pytest.raises(FileExistsError):
        RequestJournal(path)
    path.rename(tmp_path / "original")
    path.write_text("replacement")
    with pytest.raises(AuditWriteError):
        journal.append("test")
    assert path.read_text() == "replacement"


def test_pseudonymization_before_trace_and_roundtrip_replay(tmp_path, server):
    case = {"case_id": "TCGA-XX-1234", "slide_paths": ["/private/TCGA-XX-1234.slide.svs"]}
    trace = TraceLLMClient(Client(server["url"]), tmp_path / "trace.jsonl")
    client, meta = configure_identifiers(trace, config={"identifier_mode": "pseudonymize"},
                                        case=case, candidates=[], output_dir=tmp_path)
    prompt = "TCGA-XX-1234: /private/TCGA-XX-1234.slide.svs; grade 2, site adrenal"
    answer = client.chat_json(prompt, system_prompt="case TCGA-XX-1234")
    assert answer["content"] == prompt
    assert "TCGA-XX-1234" not in json.dumps(server["requests"])
    assert "TCGA-XX-1234" not in trace.trace_path.read_text()
    assert "TCGA-XX-1234" not in trace.events_path.read_text()
    assert "grade 2, site adrenal" in json.dumps(server["requests"])
    assert meta["image_pixels_modified"] is False
    assert stat.S_IMODE(Path(meta["mapping_path"]).stat().st_mode) == 0o600
    out = tmp_path / "replay";out.mkdir()
    replay = ReplayLLMClient(trace.trace_path, image_backend=Client(server["url"]))
    wrapped, _ = configure_identifiers(replay, config={"trace_mode": "replay", "identifier_mode": "pseudonymize",
                                      "identifier_map_path": meta["mapping_path"]}, case=case, candidates=[], output_dir=out)
    assert wrapped.chat_json(prompt, system_prompt="case TCGA-XX-1234") == answer
    assert len(server["requests"]) == 1


def test_preserve_default_and_missing_mapping_fail_closed(tmp_path):
    client = object()
    assert configure_identifiers(client, config={}, case={}, candidates=[], output_dir=tmp_path)[0] is client
    with pytest.raises(ValueError, match="original private"):
        configure_identifiers(client, config={"identifier_mode": "pseudonymize", "trace_mode": "replay"},
                              case={"case_id": "C035"}, candidates=[], output_dir=tmp_path)
    with pytest.raises(ValueError, match="record or replay"):
        configure_identifiers(client, config={"identifier_mode": "pseudonymize", "trace_mode": "off"},
                              case={"case_id": "C035"}, candidates=[], output_dir=tmp_path)


def test_identifier_response_keys_and_image_paths(tmp_path):
    alias = "PXID_0123456789abcdef_1"
    class Echo:
        def chat(self, prompt, **kwargs):
            assert kwargs["image_paths"] == ["/local/C035.png"]
            assert kwargs["messages"][0]["content"] == "data:image/jpeg;base64,C035"
            return {prompt: {"content": prompt}}
        chat_json = chat
    client = IdentifierClient(Echo(), {"C035": alias})
    assert client.chat("C035", image_paths=["/local/C035.png"],
                       messages=[{"content": "data:image/jpeg;base64,C035"}]) == {"C035": {"content": "C035"}}


def test_supplied_map_must_cover_slides_too(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"schema": "pathxai-identifier-map-v1",
                               "mapping": {"C035": "PXID_0123456789abcdef_1"}}))
    with pytest.raises(ValueError, match="all declared"):
        configure_identifiers(object(), config={"identifier_mode": "pseudonymize", "identifier_map_path": str(path)},
                              case={"case_id": "C035", "slide_paths": ["slide-123.svs"]},
                              candidates=[], output_dir=tmp_path)
    assert not (tmp_path / "llm_identifier_map.json").exists()


def test_candidate_metadata_identifiers_are_covered(tmp_path):
    path = "/private/PATIENT-OTHER/SLIDE-0042.svs"
    class Echo:
        def chat(self, prompt, **kwargs):
            assert "PATIENT-OTHER" not in prompt and "SLIDE-0042" not in prompt
            return {"content": prompt}
        chat_json = chat
    client, meta = configure_identifiers(Echo(), config={"identifier_mode": "pseudonymize"},
        case={"case_id": "CASE-0042", "slide_paths": []},
        candidates=[{"slide_id": "SLIDE-0042", "metadata": {"slide_path": path}}], output_dir=tmp_path)
    assert client.chat(path)["content"] == path
    mapping = json.loads(Path(meta["mapping_path"]).read_text())["mapping"]
    assert {path, "SLIDE-0042.svs", "SLIDE-0042"}.issubset(mapping)


def test_new_modules_are_preserved_in_run_source_snapshot(tmp_path):
    from pathxai_demo.adapter_provenance import capture_adapter_sources, verify_adapter_sources
    manifest = capture_adapter_sources(tmp_path)
    assert verify_adapter_sources(manifest) == []
    files = json.loads(manifest.read_text())["files"]
    assert {"pathxai_demo/request_audit.py", "pathxai_demo/identifiers.py"}.issubset(files)


def test_failed_tail_and_truncated_journal_are_not_complete_replay(tmp_path, server):
    trace = TraceLLMClient(Client(server["url"]), tmp_path / "trace.jsonl")
    trace.chat("success")
    good = trace.events_path.read_bytes()
    server["statuses"] = [400]
    with pytest.raises(requests.HTTPError):
        trace.chat("failure at end")
    with pytest.raises(ReplayError, match="lifecycle"):
        ReplayLLMClient(trace.trace_path)
    trace.events_path.write_bytes(good[:-5])
    with pytest.raises(ReplayError, match="lifecycle"):
        ReplayLLMClient(trace.trace_path)
    trace.events_path.write_bytes(good)
    assert ReplayLLMClient(trace.trace_path).recording_events_receipt["status"] == "complete_calls_verified"


def test_journal_write_failure_stops_before_network(tmp_path, server, monkeypatch):
    trace = TraceLLMClient(Client(server["url"]), tmp_path / "trace.jsonl")
    append = trace.journal.append
    def fail(event, **fields):
        if event == "http_attempt_started":
            raise AuditWriteError("synthetic disk error")
        append(event, **fields)
    monkeypatch.setattr(trace.journal, "append", fail)
    with pytest.raises(AuditWriteError):
        trace.chat("not sent")
    assert server["requests"] == []


def test_frozen_qwen_payload_images_and_retry_unchanged(tmp_path, server, monkeypatch):
    from PIL import Image
    from urllib.parse import urlsplit
    import hashlib
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    source = ROOT / "vendor/report_engine/baselines/current/source_snapshot/v1/src/pathxai_agent/llm/qwen_client.py"
    spec = importlib.util.spec_from_file_location("frozen_audit_qwen", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    client = module.QwenVisionClient(urlsplit(server["url"]).netloc, model="synthetic-frozen",
             image_max_edge=32, image_jpeg_quality=75, max_images=1, enable_thinking=False,
             max_retries=1, retry_backoff=0, timeout=(2, 2), use_system_role=True)
    image = tmp_path / "synthetic.png"
    Image.new("RGB", (64, 48), (123, 44, 70)).save(image)
    args = dict(image_paths=[image], system_prompt="synthetic system", messages=[{"role": "user", "content": "history"}])
    prompt = '{"grade": 2, "site": "adrenal"}'
    baseline = client.chat_json(prompt, **args)
    baseline_payload = server["requests"][-1]
    trace = TraceLLMClient(client, tmp_path / "trace.jsonl")
    server["drop"] = 1  # The frozen client's own retry loop handles this POST failure.
    recorded = trace.chat_json(prompt, **args)
    assert server["requests"][1:] == [baseline_payload, baseline_payload]
    assert recorded["json"] == baseline["json"]
    assert recorded["raw"] == baseline["raw"]
    events = rows(trace.events_path)
    attempts = [e for e in events if e["event"] == "http_attempt_started"]
    assert len(attempts) == 2
    raw_image = baseline_payload["messages"][-1]["content"][-1]["image_url"]["url"].split(",")[1]
    digest = hashlib.sha256(base64.b64decode(raw_image)).hexdigest()
    assert attempts[0]["body_json"]["messages"][-1]["content"][-1]["image_url"]["url"]["image_sha256"] == digest
    assert rows(trace.trace_path)[0]["request"]["images"][0]["payload_sha256"] == digest
    assert raw_image not in trace.events_path.read_text()
    replay = ReplayLLMClient(trace.trace_path, image_backend=client)
    assert replay.chat_json(prompt, **args) == recorded
    assert len(server["requests"]) == 3


@pytest.mark.parametrize("config,expected", [({}, True), ({"identifier_mode": "bad"}, False),
    ({"identifier_mode": "pseudonymize", "trace_mode": "off"}, False),
    ({"identifier_mode": "pseudonymize", "trace_mode": "replay"}, False)])
def test_preflight_identifier_settings(config, expected):
    from pathxai_demo.profiles import validate_config
    checks = validate_config(config)["checks"]
    assert next(r["ok"] for r in checks if r["name"] == "identifier_mode") is expected
