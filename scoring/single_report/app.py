#!/usr/bin/env python3
"""Local-only PathXAI human evaluation demo.

This server is intentionally dependency-light and privacy-conservative:
- reads WSI files in-place from a configured local data root;
- never exposes raw slide download endpoints;
- serves OpenSlide-backed Deep Zoom Image (DZI) descriptors and JPEG tiles;
- writes physician annotations as JSON on the deployment machine.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import hmac
import io
import json
import mimetypes
import os
import re
import socket
import ssl
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

try:
    import openslide
    from openslide.deepzoom import DeepZoomGenerator
except Exception as exc:  # pragma: no cover - runtime environment check
    print(f"ERROR: openslide-python is required: {exc}", file=sys.stderr)
    raise

from PIL import Image


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"


class HumanEvalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128
    tls_context = None
    handshake_timeout = 10
    connection_timeout = 30

    def enable_tls(self, context: ssl.SSLContext) -> None:
        # Leave accept() on a plain socket; TLS belongs in the client worker.
        self.tls_context = context

    def get_request(self):
        request, address = super().get_request()
        request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        request.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        request.settimeout(self.connection_timeout)
        return request, address

    def process_request_thread(self, request, client_address):
        if self.tls_context is not None:
            try:
                request = self.tls_context.wrap_socket(
                    request, server_side=True, do_handshake_on_connect=False
                )
                request.settimeout(self.handshake_timeout)
                request.do_handshake()
                request.settimeout(self.connection_timeout)
            except OSError:
                self.shutdown_request(request)
                return
        super().process_request_thread(request, client_address)


SLIDE_EXTS = {".svs", ".tif", ".tiff", ".ndpi", ".mrxs"}
PUBLIC_CASE_FIELDS = {
    "case_id",
    "site",
    "case_label",
}
DZI_TILE_SIZE = int(os.environ.get("PATHXAI_DZI_TILE_SIZE", "254"))
DZI_OVERLAP = int(os.environ.get("PATHXAI_DZI_OVERLAP", "1"))
DZI_FORMAT = "jpeg"
DZI_JPEG_QUALITY = int(os.environ.get("PATHXAI_DZI_JPEG_QUALITY", "90"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "case"


def normalize_reader_id(text: str) -> str:
    reader_id = text.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", reader_id):
        raise ValueError(
            "reader_id must be 1-64 letters, numbers, dots, underscores, or hyphens"
        )
    return reader_id


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def encode_dzi_tile(tile: Image.Image) -> bytes:
    if tile.mode != "RGB":
        rgb = Image.new("RGB", tile.size, "white")
        alpha = tile.getchannel("A") if "A" in tile.getbands() else None
        rgb.paste(tile, mask=alpha)
        tile = rgb
    buf = io.BytesIO()
    tile.save(
        buf,
        format="JPEG",
        quality=DZI_JPEG_QUALITY,
        optimize=True,
    )
    return buf.getvalue()


def build_manifest(data_root: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for slide_path in sorted(data_root.rglob("*")):
        if not slide_path.is_file() or slide_path.suffix.lower() not in SLIDE_EXTS:
            continue
        rel = slide_path.relative_to(data_root)
        diagnosis_hint = rel.parts[0] if len(rel.parts) > 1 else ""
        stem = slide_path.stem
        case_id = safe_id(stem)
        anon_hash = hashlib.sha256(str(rel).encode("utf-8")).hexdigest()[:10]
        cases.append(
            {
                "case_id": case_id,
                "display_id": case_id,
                "anon_id": f"CASE-{anon_hash}",
                "dataset": "local_dataset",
                "site": "breast",
                "cohort_label": diagnosis_hint,
                "slide_path": str(slide_path),
                "relative_slide_path": str(rel),
            }
        )
    return cases


class SlideCache:
    def __init__(self, cases: dict[str, dict[str, Any]], max_open: int = 8) -> None:
        self.cases = cases
        self.max_open = max_open
        self._cache: dict[str, tuple[float, openslide.OpenSlide]] = {}
        self._deepzoom: dict[str, tuple[openslide.OpenSlide, DeepZoomGenerator]] = {}
        self._cache_lock = threading.RLock()
        self._case_locks: dict[str, threading.RLock] = {}

    def lock_for(self, case_id: str) -> threading.RLock:
        with self._cache_lock:
            return self._case_locks.setdefault(case_id, threading.RLock())

    def get(self, case_id: str) -> openslide.OpenSlide:
        with self._cache_lock:
            if case_id not in self.cases:
                raise KeyError(case_id)
            now = time.time()
            if case_id in self._cache:
                _, slide = self._cache[case_id]
                self._cache[case_id] = (now, slide)
                return slide

            # Do not close a handle while another tile request is using it.
            # If every cached case is busy, temporarily exceed the LRU target.
            while len(self._cache) >= self.max_open:
                evicted = False
                for old_case in sorted(self._cache, key=lambda cid: self._cache[cid][0]):
                    old_lock = self.lock_for(old_case)
                    if not old_lock.acquire(blocking=False):
                        continue
                    try:
                        _, old_slide = self._cache.pop(old_case)
                        self._deepzoom.pop(old_case, None)
                        old_slide.close()
                        evicted = True
                        break
                    finally:
                        old_lock.release()
                if not evicted:
                    break

            slide = openslide.OpenSlide(self.cases[case_id]["slide_path"])
            self._cache[case_id] = (now, slide)
            return slide

    def get_deepzoom(self, case_id: str) -> DeepZoomGenerator:
        with self._cache_lock:
            slide = self.get(case_id)
            cached = self._deepzoom.get(case_id)
            if cached and cached[0] is slide:
                return cached[1]
            generator = DeepZoomGenerator(
                slide,
                tile_size=DZI_TILE_SIZE,
                overlap=DZI_OVERLAP,
                limit_bounds=False,
            )
            self._deepzoom[case_id] = (slide, generator)
            return generator


class HumanEvalApp:
    def __init__(
        self,
        data_root: Path,
        annotation_root: Path,
        manifest_path: Path,
        dzi_cache_root: Path,
        deployment_mode: str,
        bind_notice: str,
        access_code: str | None = None,
    ) -> None:
        self.data_root = data_root.resolve()
        self.annotation_root = annotation_root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.dzi_cache_root = dzi_cache_root.resolve()
        self.deployment_mode = deployment_mode
        self.bind_notice = bind_notice
        self.access_code = access_code
        self.annotation_root.mkdir(parents=True, exist_ok=True)
        self.dzi_cache_root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            manifest_cases = json.loads(self.manifest_path.read_text())
        else:
            manifest_cases = build_manifest(self.data_root)
            self.manifest_path.write_text(
                json.dumps(manifest_cases, indent=2), encoding="utf-8"
            )
        self.cases_list = self.blind_cases_for_frontend(manifest_cases)
        self.cases = {c["case_id"]: c for c in self.cases_list}
        self.slides = SlideCache(self.cases)

    @staticmethod
    def blind_cases_for_frontend(
        manifest_cases: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        blinded: list[dict[str, Any]] = []
        site_counts: dict[str, int] = {}
        seen: set[str] = set()
        for source in manifest_cases:
            row = dict(source)
            source_case_id = str(row["case_id"])
            public_case_id = safe_id(
                str(row.get("anon_id") or "")
                or "CASE-"
                + hashlib.sha256(source_case_id.encode("utf-8")).hexdigest()[:12]
            )
            if public_case_id in seen:
                raise ValueError(f"duplicate blinded case ID: {public_case_id}")
            seen.add(public_case_id)
            site = str(row.get("site") or "case").strip().lower()
            site_counts[site] = site_counts.get(site, 0) + 1
            row["source_case_id"] = source_case_id
            row["dzi_cache_id"] = source_case_id
            row["case_id"] = public_case_id
            row["case_label"] = f"{site.title()} case {site_counts[site]:03d}"
            blinded.append(row)
        return blinded

    def dzi_cache_id(self, case_id: str) -> str:
        if case_id not in self.cases:
            raise KeyError(case_id)
        return safe_id(str(self.cases[case_id].get("dzi_cache_id") or case_id))

    def dzi_descriptor_path(self, case_id: str) -> Path:
        return self.dzi_cache_root / f"{self.dzi_cache_id(case_id)}.dzi"

    def dzi_tile_path(self, case_id: str, level: int, col: int, row: int) -> Path:
        return (
            self.dzi_cache_root
            / f"{self.dzi_cache_id(case_id)}_files"
            / str(level)
            / f"{col}_{row}.jpeg"
        )

    def get_dzi_descriptor(self, case_id: str) -> bytes:
        if case_id not in self.cases:
            raise KeyError(case_id)
        path = self.dzi_descriptor_path(case_id)
        if path.exists():
            return path.read_bytes()
        with self.slides.lock_for(case_id):
            if path.exists():
                return path.read_bytes()
            descriptor = self.slides.get_deepzoom(case_id).get_dzi(DZI_FORMAT)
            data = descriptor.encode("utf-8")
            atomic_write_bytes(path, data)
            return data

    def get_dzi_tile(self, case_id: str, level: int, col: int, row: int) -> bytes:
        if case_id not in self.cases:
            raise KeyError(case_id)
        path = self.dzi_tile_path(case_id, level, col, row)
        if path.exists():
            return path.read_bytes()
        with self.slides.lock_for(case_id):
            if path.exists():
                return path.read_bytes()
            tile = self.slides.get_deepzoom(case_id).get_tile(level, (col, row))
            data = encode_dzi_tile(tile)
            atomic_write_bytes(path, data)
            return data

    def annotation_path(self, case_id: str, reader_id: str) -> Path:
        reader_id = normalize_reader_id(reader_id)
        return (
            self.annotation_root
            / safe_id(reader_id)
            / f"{safe_id(case_id)}.json"
        )

    def read_annotation(
        self, case_id: str, reader_id: str
    ) -> dict[str, Any] | None:
        reader_id = normalize_reader_id(reader_id)
        path = self.annotation_path(case_id, reader_id)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def write_annotation(
        self, case_id: str, reader_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if case_id not in self.cases:
            raise KeyError(case_id)
        reader_id = normalize_reader_id(reader_id)
        payload_reader = payload.get("reader") or {}
        if payload_reader.get("reader_id") != reader_id:
            raise ValueError("reader_id does not match the active reader session")
        existing = self.read_annotation(case_id, reader_id)
        allow_submitted_edit = payload.get("confirm_edit_submitted") is True
        if (existing or {}).get("status") == "submitted" and not allow_submitted_edit:
            raise ValueError("submitted report is locked and cannot be overwritten")
        now = utc_now()
        record = {
            "schema_version": "pathxai_human_eval_annotation_v1",
            "case_id": case_id,
            "anon_id": self.cases[case_id].get("anon_id"),
            "dataset": self.cases[case_id].get("dataset"),
            "site": self.cases[case_id].get("site"),
            "deployment_mode": self.deployment_mode,
            "data_root": None,
            "created_at": (existing or {}).get("created_at") or now,
            "updated_at": now,
            "submitted_at": payload.get("submitted_at") or ((existing or {}).get("submitted_at")),
            "status": payload.get("status", "draft"),
            "reader": {"reader_id": reader_id},
            "structured_facts": payload.get("structured_facts", {}),
            "narrative_report": payload.get("narrative_report", {}),
            "diagnostic_rois": payload.get("diagnostic_rois", []),
            "review_scores": payload.get("review_scores", {}),
            "notes": payload.get("notes", ""),
        }
        path = self.annotation_path(case_id, reader_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        if existing:
            hist_dir = (
                self.annotation_root
                / "_history"
                / safe_id(reader_id)
                / safe_id(case_id)
            )
            hist_dir.mkdir(parents=True, exist_ok=True)
            hist_path = hist_dir / f"{time.time_ns()}.json"
            atomic_write_bytes(hist_path, json.dumps(existing, indent=2).encode("utf-8"))
        atomic_write_bytes(path, json.dumps(record, indent=2).encode("utf-8"))
        return record

    def iter_annotation_paths(self) -> list[Path]:
        paths = sorted(
            path
            for path in self.annotation_root.rglob("*.json")
            if "_history" not in path.relative_to(self.annotation_root).parts
        )
        return paths


def make_handler(app: HumanEvalApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "PathXAIHumanEval/0.1"
        protocol_version = "HTTP/1.1"

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
            if cache:
                self.send_header("Cache-Control", "private, max-age=3600")
            else:
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        def reader_id_from_query(self, qs: dict[str, list[str]]) -> str:
            return normalize_reader_id((qs.get("reader_id") or [""])[0])

        def require_auth(self) -> bool:
            if not app.access_code:
                return True
            header = self.headers.get("Authorization", "")
            try:
                scheme, encoded = header.split(" ", 1)
                decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
                username, password = decoded.split(":", 1)
            except (ValueError, UnicodeDecodeError):
                username, password, scheme = "", "", ""
            authorized = (
                scheme.lower() == "basic"
                and hmac.compare_digest(username, "pathologist")
                and hmac.compare_digest(password, app.access_code)
            )
            if authorized:
                return True
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("WWW-Authenticate", 'Basic realm="PathXAI Human Eval"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", "0")
            self.end_headers()
            self.close_connection = True
            return False

        def do_GET(self) -> None:  # noqa: N802
            if not self.require_auth():
                return
            try:
                self._do_GET()
            except (BrokenPipeError, ConnectionResetError):
                # OpenSeadragon may cancel speculative tile requests after a
                # pan/zoom. The response is no longer needed and is not a
                # server-side tile failure.
                return
            except ValueError as exc:
                self.send_json({"error": str(exc)}, status=400)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)

        def _do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            qs = parse_qs(parsed.query)
            if path == "/":
                return self.serve_static("index.html")
            if path.startswith("/static/"):
                return self.serve_static(path[len("/static/") :])
            if path == "/api/status":
                restricted = app.deployment_mode == "remote_in_region"
                return self.send_json(
                    {
                        "ok": True,
                        "case_count": len(app.cases_list),
                        "data_root": (
                            "<in-region restricted storage>"
                            if restricted
                            else str(app.data_root)
                        ),
                        "annotation_root": (
                            "<in-region annotation storage>"
                            if restricted
                            else str(app.annotation_root)
                        ),
                        "dzi_cache_root": (
                            "<in-region derived-image cache>"
                            if restricted
                            else str(app.dzi_cache_root)
                        ),
                        "deployment_mode": app.deployment_mode,
                        "policy": {
                            "raw_slide_download": False,
                            "bulk_tile_export": False,
                            "cross_region_sync": False,
                        },
                        "notice": app.bind_notice,
                    }
                )
            if path == "/api/cases":
                reader_id = self.reader_id_from_query(qs)
                rows = []
                for c in app.cases_list:
                    ann = app.read_annotation(c["case_id"], reader_id)
                    rows.append(
                        {
                            "case_id": c["case_id"],
                            "site": c.get("site"),
                            "case_label": c.get("case_label"),
                            "annotation_status": (ann or {}).get("status", "not_started"),
                            "updated_at": (ann or {}).get("updated_at"),
                        }
                    )
                return self.send_json(rows)
            if path.startswith("/api/case/"):
                reader_id = self.reader_id_from_query(qs)
                case_id = path.split("/")[-1]
                case = app.cases.get(case_id)
                if not case:
                    return self.send_json({"error": "unknown case"}, status=404)
                with app.slides.lock_for(case_id):
                    slide = app.slides.get(case_id)
                    dimensions = slide.dimensions
                    level_count = slide.level_count
                    level_dimensions = slide.level_dimensions
                    level_downsamples = [float(x) for x in slide.level_downsamples]
                return self.send_json(
                    {
                        **{k: v for k, v in case.items() if k in PUBLIC_CASE_FIELDS},
                        "dimensions": dimensions,
                        "level_count": level_count,
                        "level_dimensions": level_dimensions,
                        "level_downsamples": level_downsamples,
                        "annotation": app.read_annotation(case_id, reader_id),
                    }
                )
            if path.startswith("/api/annotation/"):
                reader_id = self.reader_id_from_query(qs)
                case_id = path.split("/")[-1]
                ann = app.read_annotation(case_id, reader_id)
                return self.send_json(ann or {})
            if path.startswith("/api/thumbnail/"):
                case_id = Path(path).stem
                return self.serve_thumbnail(case_id)
            if path.startswith("/api/view/"):
                case_id = Path(path).stem
                return self.serve_view(case_id, qs)
            if path.startswith("/api/dzi/"):
                dzi_path = path[len("/api/dzi/") :]
                tile_match = re.fullmatch(
                    r"(.+)_files/(\d+)/(\d+)_(\d+)\.(?:jpg|jpeg)",
                    dzi_path,
                    flags=re.IGNORECASE,
                )
                if tile_match:
                    case_id, level, col, row = tile_match.groups()
                    return self.serve_dzi_tile(case_id, int(level), int(col), int(row))
                if dzi_path.endswith(".dzi"):
                    return self.serve_dzi(dzi_path[:-4])
                return self.send_json({"error": "invalid DZI path"}, status=404)
            if path == "/api/export/jsonl":
                return self.export_jsonl()
            if path == "/api/export/csv":
                return self.export_csv()
            self.send_json({"error": "not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            if not self.require_auth():
                return
            try:
                parsed = urlparse(self.path)
                path = unquote(parsed.path)
                qs = parse_qs(parsed.query)
                if path.startswith("/api/annotation/"):
                    reader_id = self.reader_id_from_query(qs)
                    case_id = path.split("/")[-1]
                    payload = self.read_json_body()
                    if payload.get("status") == "submitted" and not payload.get("submitted_at"):
                        payload["submitted_at"] = utc_now()
                    return self.send_json(
                        app.write_annotation(case_id, reader_id, payload)
                    )
                self.send_json({"error": "not found"}, status=404)
            except (BrokenPipeError, ConnectionResetError):
                return
            except ValueError as exc:
                self.send_json({"error": str(exc)}, status=400)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=500)

        def serve_static(self, rel: str) -> None:
            rel = rel or "index.html"
            target = (STATIC_ROOT / rel).resolve()
            if not str(target).startswith(str(STATIC_ROOT.resolve())) or not target.exists():
                return self.send_json({"error": "static file not found"}, status=404)
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_bytes(target.read_bytes(), content_type)

        def serve_thumbnail(self, case_id: str) -> None:
            with app.slides.lock_for(case_id):
                slide = app.slides.get(case_id)
                thumb = slide.get_thumbnail((640, 420)).convert("RGB")
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=85)
            self.send_bytes(buf.getvalue(), "image/jpeg")

        def serve_view(self, case_id: str, qs: dict[str, list[str]]) -> None:
            with app.slides.lock_for(case_id):
                slide = app.slides.get(case_id)
                level = int(qs.get("level", [max(0, slide.level_count - 2)])[0])
                level = max(0, min(level, slide.level_count - 1))
                w = max(256, min(int(qs.get("w", [1000])[0]), 1800))
                h = max(256, min(int(qs.get("h", [720])[0]), 1400))
                dim0 = slide.dimensions
                cx = float(qs.get("cx", [dim0[0] / 2])[0])
                cy = float(qs.get("cy", [dim0[1] / 2])[0])
                down = float(slide.level_downsamples[level])
                region_w0 = w * down
                region_h0 = h * down
                x0 = int(max(0, min(cx - region_w0 / 2, max(0, dim0[0] - region_w0))))
                y0 = int(max(0, min(cy - region_h0 / 2, max(0, dim0[1] - region_h0))))
                img = slide.read_region((x0, y0), level, (w, h)).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            self.send_bytes(buf.getvalue(), "image/jpeg", cache=False)

        def serve_dzi(self, case_id: str) -> None:
            if case_id not in app.cases:
                return self.send_json({"error": "unknown case"}, status=404)
            self.send_bytes(
                app.get_dzi_descriptor(case_id),
                "application/xml; charset=utf-8",
                cache=True,
            )

        def serve_dzi_tile(self, case_id: str, level: int, col: int, row: int) -> None:
            if case_id not in app.cases:
                return self.send_json({"error": "unknown case"}, status=404)
            try:
                data = app.get_dzi_tile(case_id, level, col, row)
            except (IndexError, ValueError):
                return self.send_json({"error": "tile outside pyramid"}, status=404)
            # Restricted remote pixels may be viewed through a tunnel but must
            # not be persisted in the physician workstation's browser cache.
            self.send_bytes(
                data,
                "image/jpeg",
                cache=app.deployment_mode != "remote_in_region",
            )

        def export_jsonl(self) -> None:
            lines: list[str] = []
            for path in app.iter_annotation_paths():
                try:
                    lines.append(json.dumps(json.loads(path.read_text(encoding="utf-8"))))
                except Exception:
                    continue
            self.send_bytes(("\n".join(lines) + ("\n" if lines else "")).encode("utf-8"), "application/jsonl", cache=False)

        def export_csv(self) -> None:
            fields = [
                "case_id",
                "status",
                "reader_id",
                "reader_session",
                "primary_diagnosis",
                "broad_type",
                "histologic_subtype",
                "malignancy_status",
                "invasive_status",
                "grade",
                "necrosis",
                "confidence",
                "diagnostic_roi_count",
                "updated_at",
                "submitted_at",
            ]
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fields)
            writer.writeheader()
            for path in app.iter_annotation_paths():
                try:
                    ann = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                sf = ann.get("structured_facts") or {}
                reader = ann.get("reader") or {}
                writer.writerow(
                    {
                        "case_id": ann.get("case_id"),
                        "status": ann.get("status"),
                        "reader_id": reader.get("reader_id"),
                        "reader_session": reader.get("reader_id"),
                        "primary_diagnosis": sf.get("primary_diagnosis"),
                        "broad_type": sf.get("broad_type"),
                        "histologic_subtype": sf.get("histologic_subtype"),
                        "malignancy_status": sf.get("malignancy_status"),
                        "invasive_status": sf.get("invasive_status"),
                        "grade": sf.get("grade"),
                        "necrosis": sf.get("necrosis"),
                        "confidence": sf.get("confidence"),
                        "diagnostic_roi_count": len(ann.get("diagnostic_rois") or []),
                        "updated_at": ann.get("updated_at"),
                        "submitted_at": ann.get("submitted_at"),
                    }
                )
            self.send_bytes(buf.getvalue().encode("utf-8"), "text/csv", cache=False)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="directory containing your local WSI files")
    parser.add_argument("--annotation-root", required=True, help="local score output directory")
    parser.add_argument("--manifest", required=True, help="your JSON case manifest")
    parser.add_argument(
        "--dzi-cache-root",
        required=True,
        help="Persistent local DZI descriptor/tile cache directory.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--access-code-file",
        default=None,
        help="File containing the shared LAN access code for HTTP Basic Auth.",
    )
    parser.add_argument("--tls-cert", default=None, help="HTTPS certificate PEM file.")
    parser.add_argument("--tls-key", default=None, help="HTTPS private-key PEM file.")
    parser.add_argument("--deployment-mode", choices=["local", "remote_in_region"], default="local",
                        help="Local deployment mode; no external host is required.")
    args = parser.parse_args()
    access_code = None
    if args.access_code_file:
        access_code = Path(args.access_code_file).read_text(encoding="utf-8").strip()
        if len(access_code) < 12:
            raise SystemExit("access code must contain at least 12 characters")
    tls_enabled = bool(args.tls_cert or args.tls_key)
    if tls_enabled and not (args.tls_cert and args.tls_key):
        raise SystemExit("both --tls-cert and --tls-key are required for HTTPS")
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        if not access_code:
            raise SystemExit("LAN binding requires --access-code-file")
        if not tls_enabled:
            raise SystemExit("LAN binding requires --tls-cert and --tls-key")
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"data root does not exist: {data_root}")
    dzi_cache_root = (
        Path(args.dzi_cache_root)
        if args.dzi_cache_root
        else data_root.parent / f"{data_root.name}_dzi_cache"
    )
    bind_notice = (
        "Local-only service. Raw WSI downloads and cross-region sync are disabled."
    )
    app = HumanEvalApp(
        data_root=data_root,
        annotation_root=Path(args.annotation_root),
        manifest_path=Path(args.manifest),
        dzi_cache_root=dzi_cache_root,
        deployment_mode=args.deployment_mode,
        bind_notice=bind_notice,
        access_code=access_code,
    )
    server = HumanEvalHTTPServer((args.host, args.port), make_handler(app))
    if tls_enabled:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(certfile=args.tls_cert, keyfile=args.tls_key)
        server.enable_tls(context)
    scheme = "https" if tls_enabled else "http"
    print(f"PathXAI human-eval server: {scheme}://{args.host}:{args.port}")
    print(f"Cases: {len(app.cases_list)}")
    print(f"Data root: {app.data_root}")
    print(f"Annotations: {app.annotation_root}")
    print(f"DZI cache: {app.dzi_cache_root}")
    print(bind_notice)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
