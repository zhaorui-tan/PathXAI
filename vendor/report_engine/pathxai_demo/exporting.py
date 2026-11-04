"""Portable HTML output through the study's existing, verified exporter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any
import zipfile

from .config import atomic_json, redact


def _artifact_hashes(directory: Path) -> dict[str, str]:
    hashes = {}
    for name in ("pipeline_result.json", "generated_long_report.txt", "report.html"):
        path = directory / name
        if not path.is_file():
            raise ValueError("Scientific report is missing a required artifact: " + name)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _assert_original_artifacts(directory: Path, expected: dict[str, str]) -> None:
    if _artifact_hashes(directory) != expected:
        raise RuntimeError("A scientific artifact changed during display export.")


def _check_display_text(original: Path, view: Path) -> dict:
    """Verify existing clinical and evidence panels before adding display assets."""
    from lxml import html

    before = html.fromstring(original.read_bytes())
    after = html.fromstring(view.read_bytes())
    classes = ("report-row-content", "roi-card", "decision-node", "related-card", "tier-item")
    counts = {}
    for name in classes:
        selector = '//*[contains(concat(" ",normalize-space(@class)," ")," ' + name + ' ")]'
        old = [node.text_content() for node in before.xpath(selector)]
        new = [node.text_content() for node in after.xpath(selector)]
        if name == "report-row-content" and len(old) < 7:
            raise RuntimeError("The scientific report lacks the expected clinical sections.")
        if old != new:
            raise RuntimeError("Report display adaptation changed existing text in " + name + ".")
        counts[name] = len(old)
    return {"clinical_and_existing_evidence_panels_unchanged": True, "checked_panel_counts": counts}


def _communicate(process, cancel_event, *, payload=None, seconds=600):
    deadline = time.monotonic() + seconds
    try:
        while True:
            try:
                result = process.communicate(input=payload, timeout=0.5)
                return result
            except subprocess.TimeoutExpired:
                payload = None
                if cancel_event.is_set():
                    raise InterruptedError("Report export cancelled.")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Report display/export exceeded its time budget.")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()


def prepare_report(config: dict, case: dict, run_dir: Path, outcome: dict,
                   event, cancel_event) -> dict:
    """Keep the scientific output intact and build a relocatable viewing copy."""
    source = Path(outcome["report_path"]).resolve()
    if source.name != "report.html" or source.parent.name != case["case_id"]:
        raise ValueError("Unexpected scientific report layout; refusing to export another case.")
    script = Path(config["engine_root"]) / "scripts" / "export_report_samples_offline.py"
    if not script.is_file():
        raise FileNotFoundError("The original portable HTML exporter is missing: " + str(script))
    original_hashes = _artifact_hashes(source.parent)
    result = json.loads((source.parent / "pipeline_result.json").read_text(encoding="utf-8"))
    if result.get("case_id") != case["case_id"]:
        raise ValueError("Scientific result belongs to a different case.")
    python = str(config.get("pipeline_python") or "")
    if not python or not Path(python).is_file():
        raise ValueError("Report display requires the configured pipeline interpreter.")
    # The original batch command adds WSI thumbnails after scientific inference.
    # The single-case adapter supplies these in a separate, uniquely named view
    # tree and renders through the same frozen payload builder/renderer.
    view_root = Path(tempfile.mkdtemp(prefix="view_source_", dir=run_dir))
    view_dir = view_root / case["case_id"]
    request = {
        "config": {key: config[key] for key in ("engine_root", "v1_root")},
        "case": {key: case[key] for key in ("case_id", "mode", "slide_paths") if key in case},
        "scientific_case_dir": str(source.parent), "view_case_dir": str(view_dir),
    }
    event({"stage": "export", "message": "Building real WSI overviews for the report display."})
    process = subprocess.Popen([python, "-B", "-m", "pathxai_demo.reporting_worker"],
                               cwd=str(Path(__file__).resolve().parents[1]), stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace")
    try:
        output, error = _communicate(process, cancel_event, payload=json.dumps(request))
    except InterruptedError:
        return dict(outcome, status="cancelled")
    finally:
        _assert_original_artifacts(source.parent, original_hashes)
    if process.returncode:
        raise RuntimeError("Report display failed: " + "\n".join((output + error).splitlines()[-12:]))
    display = json.loads(output)
    view_report = view_dir / "report.html"
    if not display.get("ok") or Path(display.get("report_path", "")).resolve() != view_report.resolve():
        raise RuntimeError("Display worker did not produce the expected report copy.")
    for name in ("pipeline_result.json", "generated_long_report.txt"):
        if hashlib.sha256((view_dir / name).read_bytes()).hexdigest() != original_hashes[name]:
            raise RuntimeError("Report display copy changed a scientific artifact: " + name)
    audit = _check_display_text(source, view_report)
    audit.update(scientific_artifact_sha256=original_hashes, display=display)
    atomic_json(run_dir / "display_audit.json", audit)
    destination = run_dir / "portable"
    event({"stage": "export", "message": "Preparing the portable report and checking original report wording."})
    # The app runtime carries lxml/Pillow; the original CONCH environment need
    # not contain display/export-only dependencies.
    command = [sys.executable, str(script), "--case", case["case_id"] + "=report",
               "--source-root", str(view_root), "--destination", str(destination),
               "--audit", str(run_dir / "export_audit.json")]
    # No model/API calls happen here. The original exporter verifies unchanged text.
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace")
    try:
        output, _ = _communicate(process, cancel_event)
        if process.returncode:
            # Last lines are sufficient for diagnostics; the original case is preserved.
            raise RuntimeError("Report export failed: " + "\n".join(output.splitlines()[-12:]))
    except InterruptedError:
        return dict(outcome, status="cancelled")
    finally:
        _assert_original_artifacts(source.parent, original_hashes)
    portable = destination / "report" / "report.html"
    if not portable.is_file():
        raise RuntimeError("The exporter exited without creating a portable report.")
    return dict(outcome, report_path=str(portable), scientific_report_path=str(source))


def make_archive(run_dir: Path, job: dict) -> Path:
    portable = run_dir / "portable"
    report = portable / "report" / "report.html"
    if not report.is_file():
        raise ValueError("This task has no completed portable report to export.")
    archive = run_dir / "report.zip"
    with tempfile.TemporaryDirectory(prefix="pathxai-export-", dir=run_dir) as directory:
        stage = Path(directory)
        shutil.copytree(portable / "report", stage / "report")
        case_dir = run_dir / "cases" / job["case_id"]
        text = case_dir / "generated_long_report.txt"
        if text.is_file():
            shutil.copyfile(text, stage / "generated_long_report.txt")
        # Share only a minimal provenance index, not original case manifests or absolute paths.
        manifest: dict[str, Any] = {
            "case_id": job["case_id"], "mode": job["mode"], "created_at": job["created_at"],
            "report": "report/report.html", "report_text_unchanged": True,
            "original_wsi_included": False,
            "files": [],
        }
        provenance = run_dir / "worker_outcome.json"
        if provenance.exists():
            original = json.loads(provenance.read_text(encoding="utf-8"))
            for key in ("ablation", "profile", "baseline_id", "source_hashes", "asset_hashes"):
                if key in original:
                    manifest[key] = redact(original[key])
        resolved_path = case_dir / "resolved_profile.json"
        if resolved_path.is_file():
            resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
            settings = resolved.get("settings") or {}
            share_settings = {section: value for section, value in settings.items() if section != "paths"}
            if isinstance(share_settings.get("llm"), dict):
                share_settings["llm"] = {key: value for key, value in share_settings["llm"].items()
                                         if key not in {"qwen_api", "medgemma_api", "report_api", "seg_reader_api"}}
            manifest["scientific_configuration"] = share_settings
            identity = (resolved.get("segmentation_cache") or {}).get("identity") or {}
            manifest["source_sha256"] = identity.get("source_sha256")
            assets = resolved.get("segmentation_checkpoint") or {}
            manifest["model_assets"] = {
                name: {key: value for key, value in info.items()
                       if key in {"sha256", "size_bytes", "known_baseline_match"}}
                for name, info in assets.items() if isinstance(info, dict)
            }
        trace = case_dir / "llm_trace_metadata.json"
        if trace.is_file():
            trace_record = json.loads(trace.read_text(encoding="utf-8"))
            manifest["llm_trace"] = {key: value for key, value in trace_record.items() if key != "path"}
            manifest["llm_trace"]["included"] = False
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                manifest["files"].append({"path": path.relative_to(stage).as_posix(),
                                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                          "bytes": path.stat().st_size})
        atomic_json(stage / "manifest.json", manifest)
        temporary = run_dir / "report.zip.tmp"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as stream:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    stream.write(path, path.relative_to(stage).as_posix())
        temporary.replace(archive)
    return archive
