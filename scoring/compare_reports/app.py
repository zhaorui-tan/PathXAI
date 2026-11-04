#!/usr/bin/env python3
"""Isolated five-report sequential review; original four-report scores stay intact."""

from __future__ import annotations

import argparse
import base64
import csv
import copy
import hashlib
import hmac
import importlib.util
import json
import mimetypes
import os
import re
import ssl
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
LABELS = ("A", "B", "C", "D", "E")
CORE_SECTIONS = {"Diagnosis", "Key Microscopic Findings", "Stage / Grade / Prognostic Features"}
RANK_SCALE = {"min": 1, "max": 5, "direction": "lower_is_better", "ties_allowed": True}


def rank_map(value: dict | None) -> dict[str, str]:
    result = {}
    for label, rank in (value or {}).items():
        if label not in LABELS:
            raise ValueError("Unknown report label")
        if rank is None or rank == "":
            result[label] = ""
        elif not isinstance(rank, bool) and str(rank) in {"1", "2", "3", "4", "5"}:
            result[label] = str(rank)
        else:
            raise ValueError("Ranks must be 1-5; 1 is best. Leave unranked reports blank.")
    return {label: result.get(label, "") for label in LABELS}


def rank_groups(value: dict) -> list[list[str]]:
    ranks = rank_map(value)
    return [[label for label in LABELS if ranks[label] == str(n)]
            for n in range(1, 6) if str(n) in ranks.values()]


def load_human_eval_base():
    spec = importlib.util.spec_from_file_location(
        "pathxai_human_eval_base", APP_ROOT / "vendor" / "human_eval_base.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Human Eval DZI implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_human_eval_base()


class HumanCompareHTTPServer(BASE.HumanEvalHTTPServer):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


SECTION_TITLES = {
    "specimen/site": "Specimen / Site",
    "specimen": "Specimen / Site",
    "site": "Specimen / Site",
    "primary diagnosis": "Diagnosis",
    "diagnosis": "Diagnosis",
    "tumor/lesion description": "Tumor / Lesion Description",
    "key microscopic findings": "Key Microscopic Findings",
    "microscopic findings": "Key Microscopic Findings",
    "key findings": "Key Microscopic Findings",
    "invasion/necrosis/margins": "Invasion / Necrosis / Margins",
    "immunohistochemistry/ancillary studies": "IHC / Ancillary Studies",
    "stage/grade/prognostic features": "Stage / Grade / Prognostic Features",
    "differential diagnoses": "Differential Diagnoses",
    "differential diagnosis": "Differential Diagnoses",
    "limitations": "Limitations",
    "final summary": "Final Summary",
}


def structure_report(text: str) -> list[dict[str, str]]:
    """Split known clinical headings without rewriting report content."""
    sections: list[dict[str, Any]] = []
    current = {"title": "Report", "lines": []}
    heading_re = __import__("re").compile(
        r"^\s*(?:#{1,6}\s*)?(?:\*\*)?([^:*#]+?)(?:\*\*)?\s*:\s*(.*)$"
    )
    for raw_line in text.replace("\r\n", "\n").split("\n"):
        match = heading_re.match(raw_line)
        key = match.group(1).strip().lower() if match else ""
        if match and key in SECTION_TITLES:
            if any(line.strip() for line in current["lines"]):
                sections.append({"title": current["title"], "body": "\n".join(current["lines"]).strip()})
            current = {"title": SECTION_TITLES[key], "lines": []}
            if match.group(2).strip():
                current["lines"].append(match.group(2).strip())
        else:
            current["lines"].append(raw_line)
    if any(line.strip() for line in current["lines"]):
        sections.append({"title": current["title"], "body": "\n".join(current["lines"]).strip()})
    return sections or [{"title": "Report", "body": text.strip()}]


def case_slide_paths(case: dict[str, Any]) -> list[str]:
    # Preserve the manifest's first slide as the stable default viewer image.
    paths = [case["slide_path"]]
    paths.extend(case.get("slide_paths") or [])
    return list(dict.fromkeys(str(Path(p)) for p in paths))


class CompareApp(BASE.HumanEvalApp):
    def __init__(
        self,
        manifest_path: Path,
        annotation_root: Path,
        dzi_cache_root: Path,
        secret_path: Path,
        access_code: str | None,
    ) -> None:
        if not manifest_path.exists():
            raise FileNotFoundError(f"comparison manifest does not exist: {manifest_path}")
        source_rows = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not source_rows:
            raise ValueError("comparison manifest is empty")
        manifest_dir = manifest_path.resolve().parent
        for row in source_rows:
            def resolve_input(value: str) -> str:
                candidate = Path(value).expanduser()
                return str((candidate if candidate.is_absolute() else manifest_dir / candidate).resolve())

            row["slide_path"] = resolve_input(row["slide_path"])
            row["slide_paths"] = [resolve_input(value) for value in row.get("slide_paths", [])]
            row["reports"] = {key: resolve_input(value) for key, value in (row.get("reports") or {}).items()}
            for path in case_slide_paths(row):
                if not Path(path).is_file():
                    raise FileNotFoundError(f"missing WSI: {path}")
            reports = row.get("reports") or {}
            if len(reports) != 5:
                raise ValueError(f"{row.get('case_id')}: five reports are required")
            for method, report_path in reports.items():
                if not Path(report_path).exists():
                    raise FileNotFoundError(f"missing report for {method}: {report_path}")

        data_root = Path(os.path.commonpath([row["slide_path"] for row in source_rows])).parent
        super().__init__(
            data_root=data_root,
            annotation_root=annotation_root,
            manifest_path=manifest_path,
            dzi_cache_root=dzi_cache_root,
            deployment_mode="local_comparison",
            bind_notice=(
                "Blinded local comparison. Raw WSI download and method identity disclosure "
                "are disabled. Scores remain on this machine."
            ),
            access_code=access_code,
        )
        self.secret_path = secret_path.resolve()
        self.secret_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.secret_path.exists():
            self.secret_path.write_text(os.urandom(32).hex(), encoding="ascii")
            self.secret_path.chmod(0o600)
        self.assignment_secret = self.secret_path.read_text(encoding="ascii").strip().encode()
        self.assignment_root = self.secret_path.parent / "assignment_audit"
        self.assignment_root.mkdir(parents=True, exist_ok=True)
        self._case_slides: dict[str, dict[str, dict[str, str]]] = {}
        slide_cases = {}
        for case in self.cases_list:
            case_id = case["case_id"]
            entries = {}
            for index, path in enumerate(case_slide_paths(case), 1):
                slide_id = f"S{index:03d}"
                key = f"{case_id}__{slide_id}"
                cache_id = self.dzi_cache_id(case_id)
                if index > 1:
                    digest = hashlib.sha256(path.encode()).hexdigest()[:16]
                    cache_id = f"{cache_id}__{slide_id}__{digest}"
                entries[slide_id] = {"key": key, "cache_id": cache_id, "slide_path": path}
                slide_cases[key] = {"slide_path": path}
            self._case_slides[case_id] = entries
        self._patient_slide_cache = BASE.SlideCache(slide_cases)
        self._annotation_lock = threading.RLock()

    def slide_set_revision(self, case_id: str) -> str:
        entries = self._case_slides[case_id]
        identity = [(sid, row["slide_path"]) for sid, row in entries.items()]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]

    def public_slides(self, case_id: str) -> list[dict[str, str]]:
        return [
            {"slide_id": sid, "label": f"Slide {index}",
             "dzi_url": f"/api/dzi/{case_id}/{sid}.dzi"}
            for index, sid in enumerate(self._case_slides[case_id], 1)
        ]

    def slide_review_required(self, case_id: str, annotation: dict | None) -> bool:
        if case_id not in self._case_slides:
            return False
        if not annotation or (annotation.get("status") != "submitted" and not annotation.get("submitted_at")):
            return False
        confirmed = (annotation.get("slide_review") or {}).get("confirmed_revision")
        return len(self._case_slides[case_id]) > 1 and confirmed != self.slide_set_revision(case_id)

    def _slide_entry(self, case_id: str, slide_id: str) -> dict[str, str]:
        # Resolve opaque IDs only. Never accept a client-supplied filename.
        return self._case_slides[case_id][slide_id]

    def get_dzi_descriptor(self, case_id: str, slide_id: str = "S001") -> bytes:
        entry = self._slide_entry(case_id, slide_id)
        path = self.dzi_cache_root / f"{entry['cache_id']}.dzi"
        if path.exists():
            return path.read_bytes()
        with self._patient_slide_cache.lock_for(entry["key"]):
            if not path.exists():
                descriptor = self._patient_slide_cache.get_deepzoom(entry["key"]).get_dzi(BASE.DZI_FORMAT)
                BASE.atomic_write_bytes(path, descriptor.encode("utf-8"))
            return path.read_bytes()

    def get_dzi_tile(self, case_id: str, level: int, col: int, row: int, slide_id: str = "S001") -> bytes:
        entry = self._slide_entry(case_id, slide_id)
        path = self.dzi_cache_root / f"{entry['cache_id']}_files" / str(level) / f"{col}_{row}.jpeg"
        if path.exists():
            return path.read_bytes()
        with self._patient_slide_cache.lock_for(entry["key"]):
            if not path.exists():
                tile = self._patient_slide_cache.get_deepzoom(entry["key"]).get_tile(level, (col, row))
                BASE.atomic_write_bytes(path, BASE.encode_dzi_tile(tile))
            return path.read_bytes()

    def reports_for(self, case_id: str, reader_id: str) -> list[dict[str, str]]:
        reader_id = BASE.normalize_reader_id(reader_id)
        case = self.cases[case_id]
        all_reports = case.get("reports") or {}
        methods = list(all_reports)
        if len(methods) != 5:
            raise ValueError(f"{case_id}: exactly five reports are required")
        methods.sort(
            key=lambda method: hmac.new(
                self.assignment_secret,
                f"{reader_id}|{case_id}|{method}".encode(),
                hashlib.sha256,
            ).digest()
        )
        labels = list(LABELS)
        assignment = dict(zip(labels, methods))
        audit_path = (
            self.assignment_root
            / BASE.safe_id(reader_id)
            / f"{BASE.safe_id(case_id)}.json"
        )
        if not audit_path.exists():
            atomic_write_json(
                audit_path,
                {
                    "schema_version": "pathxai_compare_assignment_v1",
                    "reader_id": reader_id,
                    "case_id": case_id,
                    "assignment": assignment,
                    "created_at": utc_now(),
                },
            )
        rendered = []
        for label, method in assignment.items():
            text = Path(all_reports[method]).read_text(encoding="utf-8", errors="replace")
            rendered.append({"label": label, "sections": structure_report(text)})
        return rendered

    def prepare_annotation(self, case_id: str, reader_id: str, record: dict) -> dict:
        return record

    def annotation_path(self, case_id: str, reader_id: str) -> Path:
        reader_id = BASE.normalize_reader_id(reader_id)
        return self.annotation_root / BASE.safe_id(reader_id) / f"{BASE.safe_id(case_id)}.json"

    def read_annotation(self, case_id: str, reader_id: str) -> dict[str, Any] | None:
        reader_id = BASE.normalize_reader_id(reader_id)
        path = self.annotation_path(case_id, reader_id)
        if path.exists():
            return self.prepare_annotation(case_id, reader_id, json.loads(path.read_text(encoding="utf-8")))
        return None

    def write_annotation(
        self, case_id: str, reader_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        with self._annotation_lock:
            return self._write_annotation(case_id, reader_id, payload)

    def annotation_metadata(self, case_id: str, reader_id: str, existing: dict | None) -> dict:
        return {}

    def _write_annotation(
        self, case_id: str, reader_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if case_id not in self.cases:
            raise KeyError(case_id)
        if payload.get("rank_scale") != RANK_SCALE:
            raise ValueError("Ranking changed to 1-5. Refresh this page; your local draft is retained.")
        reader_id = BASE.normalize_reader_id(reader_id)
        reader = payload.get("reader") or {}
        if reader.get("reader_id") != reader_id:
            raise ValueError("reader_id does not match the active reader session")
        existing = self.read_annotation(case_id, reader_id)
        if payload.get("base_revision") != (existing or {}).get("revision", 0):
            raise ValueError("This review changed in another tab. Your local draft is safe; reload before saving.")
        if (existing or {}).get("status") == "submitted" and payload.get("confirm_edit_submitted") is not True:
            raise ValueError("submitted comparison requires confirmation before editing")
        ratings = payload.get("ratings") or {}
        for label in LABELS:
            if label not in ratings:
                raise ValueError(f"missing ratings for Report {label}")
        if payload.get("status", "draft") not in {"draft", "submitted"}:
            raise ValueError("Invalid review status")
        sections = {key: rank_map(value) for key, value in (payload.get("section_rankings") or {}).items()}
        overall_values = rank_map(payload.get("overall_rank_values"))
        groups = rank_groups(overall_values)
        complete = lambda values: all(values.get(label, "") != "" for label in LABELS)
        if payload.get("status") == "submitted":
            if not complete(overall_values):
                raise ValueError("Rank all five reports before submitting")
            available = {s["title"] for report in self.reports_for(case_id, reader_id) for s in report["sections"]}
            for section in CORE_SECTIONS & available:
                if not complete(sections.get(section, {})):
                    raise ValueError(f"Complete all five ranks for {section}")
            for section, values in sections.items():
                if any(values.values()) and not complete(values):
                    raise ValueError(f"Finish or clear the optional ranks for {section}")
            for label in LABELS:
                if ratings[label].get("acceptability") not in {"no_correction", "minor_correction", "major_correction", "not_usable"}:
                    raise ValueError(f"Select clinical acceptability for Report {label}")
            if len(self._case_slides[case_id]) > 1 and payload.get("confirm_slide_review") is not True:
                raise ValueError("Confirm review of all available slides before submitting")
        now = utc_now()
        slide_review = dict((existing or {}).get("slide_review") or {})
        revision = self.slide_set_revision(case_id)
        confirm_slide_review = payload.get("confirm_slide_review") is True
        if confirm_slide_review:
            if payload.get("slide_set_revision") != revision:
                raise ValueError("The slide set changed. Reload this case before confirming review; your draft is preserved.")
        if payload.get("status") == "submitted" and confirm_slide_review:
            slide_review = {"confirmed_revision": revision, "confirmed_at": now,
                            "slide_ids": list(self._case_slides[case_id])}
        # Persist draft form state separately from the last submitted review audit.
        record = {
            "schema_version": "pathxai_human_compare_five_report_v2",
            "evaluation_phase": "five_report_comparison",
            "revision": (existing or {}).get("revision", 0) + 1,
            "rank_scale": dict(RANK_SCALE),
            "case_id": case_id,
            "site": self.cases[case_id].get("site"),
            "status": payload.get("status", "draft"),
            "reader": {"reader_id": reader_id},
            "ratings": ratings,
            "section_rankings": sections,
            "overall_rank_values": overall_values,
            "overall_ranking": [label for group in groups for label in group],
            "overall_rank_groups": groups,
            "best_report": groups[0][0] if groups and len(groups[0]) == 1 else None,
            "best_report_tie": bool(groups and len(groups[0]) > 1),
            "comparison_notes": str(payload.get("comparison_notes") or "")[:4000],
            "created_at": (existing or {}).get("created_at") or now,
            "updated_at": now,
            "submitted_at": now if payload.get("status") == "submitted" else (existing or {}).get("submitted_at"),
            "confirm_slide_review": confirm_slide_review,
            "slide_set_revision": revision,
            "slide_review": slide_review,
        }
        record.update(self.annotation_metadata(case_id, reader_id, existing))
        if existing:
            history = self.annotation_root / "_history" / BASE.safe_id(reader_id) / BASE.safe_id(case_id)
            old_path = self.annotation_path(case_id, reader_id)
            original_record = json.loads(old_path.read_text()) if old_path.exists() else existing
            atomic_write_json(history / f"{time.time_ns()}.json", original_record)
        atomic_write_json(self.annotation_path(case_id, reader_id), record)
        return {**record, "slide_review_required": self.slide_review_required(case_id, record)}

    def iter_annotation_paths(self) -> list[Path]:
        return sorted(
            path
            for path in self.annotation_root.rglob("*.json")
            if "_history" not in path.relative_to(self.annotation_root).parts
        )


def make_handler(app: CompareApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PathXAIHumanCompare/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

        def send_json(self, obj: Any, status: int = 200) -> None:
            data = json.dumps(obj, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def send_bytes(self, data: bytes, content_type: str, cache: bool = True) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
            self.end_headers()
            self.wfile.write(data)

        def require_auth(self) -> bool:
            if not app.access_code:
                return True
            header = self.headers.get("Authorization", "")
            try:
                scheme, encoded = header.split(" ", 1)
                username, password = base64.b64decode(encoded, validate=True).decode().split(":", 1)
            except (ValueError, UnicodeDecodeError):
                scheme, username, password = "", "", ""
            if (
                scheme.lower() == "basic"
                and hmac.compare_digest(username, "pathologist")
                and hmac.compare_digest(password, app.access_code)
            ):
                return True
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("WWW-Authenticate", 'Basic realm="PathXAI Human Compare"')
            self.send_header("Connection", "close")
            self.send_header("Content-Length", "0")
            self.end_headers()
            self.close_connection = True
            return False

        def reader_id(self, query: dict[str, list[str]]) -> str:
            return BASE.normalize_reader_id((query.get("reader_id") or [""])[0])

        def do_GET(self) -> None:  # noqa: N802
            if not self.require_auth():
                return
            try:
                self._get()
            except (BrokenPipeError, ConnectionResetError):
                return
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
            except KeyError:
                self.send_json({"error": "unknown case or slide"}, 404)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)

        def _get(self) -> None:
            parsed = urlparse(self.path)
            path, query = unquote(parsed.path), parse_qs(parsed.query)
            if path == "/":
                return self.serve_static("index.html")
            if path.startswith("/static/"):
                return self.serve_static(path[len("/static/"):])
            if path == "/api/status":
                return self.send_json({"ok": True, "case_count": len(app.cases_list),
                                       "multislide": True, "report_count": 5, "rank_min": 1, "rank_max": 5,
                                       "ui_version": "tutorial-local-five-report-v1"})
            if path == "/api/cases":
                reader = self.reader_id(query)
                return self.send_json([
                    {
                        "case_id": case["case_id"],
                        "site": case.get("site"),
                        "case_label": case.get("case_label"),
                        "status": (app.read_annotation(case["case_id"], reader) or {}).get("status", "not_started"),
                        "slide_count": len(app._case_slides[case["case_id"]]),
                        "slide_review_required": app.slide_review_required(case["case_id"], app.read_annotation(case["case_id"], reader)),
                    }
                    for case in app.cases_list
                ])
            if path.startswith("/api/case/"):
                reader = self.reader_id(query)
                case_id = path.rsplit("/", 1)[-1]
                case = app.cases.get(case_id)
                if not case:
                    return self.send_json({"error": "unknown case"}, 404)
                return self.send_json({
                    "case_id": case_id,
                    "site": case.get("site"),
                    "case_label": case.get("case_label"),
                    "reports": app.reports_for(case_id, reader),
                    "annotation": app.read_annotation(case_id, reader),
                    "slides": app.public_slides(case_id),
                    "slide_set_revision": app.slide_set_revision(case_id),
                    "slide_review_required": app.slide_review_required(case_id, app.read_annotation(case_id, reader)),
                })
            if path.startswith("/api/dzi/"):
                rel = path[len("/api/dzi/"):]
                match = re.fullmatch(r"([^/]+)/(S\d{3})_files/(\d+)/(\d+)_(\d+)\.(?:jpg|jpeg)", rel, re.IGNORECASE)
                if match:
                    case_id, slide_id, level, col, row = match.groups()
                    data = app.get_dzi_tile(case_id, int(level), int(col), int(row), slide_id)
                    return self.send_bytes(data, "image/jpeg")
                descriptor = re.fullmatch(r"([^/]+)/(S\d{3})\.dzi", rel)
                if descriptor:
                    return self.send_bytes(app.get_dzi_descriptor(*descriptor.groups()), "application/xml; charset=utf-8")
                match = re.fullmatch(r"([^/]+)_files/(\d+)/(\d+)_(\d+)\.(?:jpg|jpeg)", rel, re.IGNORECASE)
                if match:
                    case_id, level, col, row = match.groups()
                    data = app.get_dzi_tile(case_id, int(level), int(col), int(row))
                    return self.send_bytes(data, "image/jpeg")
                if re.fullmatch(r"[^/]+\.dzi", rel):
                    return self.send_bytes(app.get_dzi_descriptor(rel[:-4]), "application/xml; charset=utf-8")
                return self.send_json({"error": "invalid DZI path"}, 404)
            if path == "/api/export/jsonl":
                rows = []
                for item in app.iter_annotation_paths():
                    row = app.read_annotation(item.stem, item.parent.name)
                    row["slide_review_required"] = app.slide_review_required(row["case_id"], row)
                    rows.append(json.dumps(row))
                return self.send_bytes(("\n".join(rows) + "\n").encode(), "application/jsonl", False)
            if path == "/api/export/csv":
                return self.export_csv()
            return self.send_json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            if not self.require_auth():
                return
            try:
                parsed = urlparse(self.path)
                path, query = unquote(parsed.path), parse_qs(parsed.query)
                if not path.startswith("/api/annotation/"):
                    return self.send_json({"error": "not found"}, 404)
                reader = self.reader_id(query)
                case_id = path.rsplit("/", 1)[-1]
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode() or "{}")
                return self.send_json(app.write_annotation(case_id, reader, payload))
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 500)

        def serve_static(self, rel: str) -> None:
            target = (STATIC_ROOT / (rel or "index.html")).resolve()
            if not str(target).startswith(str(STATIC_ROOT.resolve())) or not target.exists():
                return self.send_json({"error": "not found"}, 404)
            is_html = target.suffix.lower() == ".html"
            self.send_bytes(
                target.read_bytes(),
                mimetypes.guess_type(str(target))[0] or "application/octet-stream",
                cache=not is_html,
            )

        def export_csv(self) -> None:
            fields = ["case_id", "reader_id", "status", "rank_scale", "overall_ranking", "overall_rank_values", "section_rankings", "best_report", "best_report_tie", "updated_at", "slide_count", "slide_review_required"]
            out = __import__("io").StringIO()
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for item in app.iter_annotation_paths():
                row = app.read_annotation(item.stem, item.parent.name)
                writer.writerow({
                    "case_id": row.get("case_id"),
                    "reader_id": (row.get("reader") or {}).get("reader_id"),
                    "status": row.get("status"),
                    "rank_scale": json.dumps(row.get("rank_scale")),
                    "overall_ranking": ">".join("=".join(g) for g in (row.get("overall_rank_groups") or [])),
                    "overall_rank_values": json.dumps(row.get("overall_rank_values") or {}),
                    "section_rankings": json.dumps(row.get("section_rankings") or {}),
                    "best_report": row.get("best_report"),
                    "best_report_tie": row.get("best_report_tie"),
                    "updated_at": row.get("updated_at"),
                    "slide_count": len(app._case_slides.get(row["case_id"], {})) or None,
                    "slide_review_required": app.slide_review_required(row["case_id"], row),
                })
            self.send_bytes(out.getvalue().encode(), "text/csv", False)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="your private JSON case/report manifest")
    parser.add_argument("--annotation-root", required=True, help="local score output directory")
    parser.add_argument("--dzi-cache-root", required=True, help="local DZI cache directory")
    parser.add_argument("--secret-file", required=True, help="private path for the report assignment secret")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--access-code-file")
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    args = parser.parse_args()
    access_code = Path(args.access_code_file).read_text().strip() if args.access_code_file else None
    tls = bool(args.tls_cert and args.tls_key)
    if args.host not in {"127.0.0.1", "localhost", "::1"} and (not access_code or not tls):
        raise SystemExit("LAN binding requires the shared access-code file and TLS certificate/key")
    common = [Path(args.manifest), Path(args.annotation_root), Path(args.dzi_cache_root), Path(args.secret_file), access_code]
    app = CompareApp(*common)
    server = HumanCompareHTTPServer((args.host, args.port), make_handler(app))
    if tls:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(args.tls_cert, args.tls_key)
        server.enable_tls(context)
    print(f"PathXAI Human Compare: {'https' if tls else 'http'}://{args.host}:{args.port}")
    print(f"Cases: {len(app.cases_list)}; annotations: {app.annotation_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
