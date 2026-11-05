"""Download the pinned TCGA-FD-A5BS source WSI set from NCI GDC."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import urllib.parse
import urllib.request

from .paths import ROOT


_GDC_API = "https://api.gdc.cancer.gov"
_BLOCK_SIZE = 8 * 1024 * 1024


def sample_metadata():
    """Return the pinned patient manifest, including the anatomy-only prior."""
    return json.loads((ROOT / "configs" / "sample_tcga.json").read_text())


def _hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(_BLOCK_SIZE), b""):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def verify_slide(path, slide=None):
    """Validate a slide against its pinned GDC size, MD5 and SHA-256."""
    manifest = sample_metadata()
    path = Path(path).expanduser()
    if slide is None:
        slide = next(
            (item for item in manifest["slides"] if item["file_name"] == path.name),
            None,
        )
    elif isinstance(slide, str):
        slide = next(
            (item for item in manifest["slides"] if item["file_id"] == slide
             or item["file_name"] == slide),
            None,
        )
    if slide is None:
        raise ValueError(f"Unpinned patient slide: {path.name}")
    if not path.is_file() or path.stat().st_size != slide["file_size"]:
        raise ValueError(f"Slide missing or size mismatch: {slide['file_name']}")
    md5, sha = _hashes(path)
    if md5 != slide["md5sum"] or sha != slide["sha256"]:
        raise ValueError(f"Slide checksum mismatch: {slide['file_name']}")
    return path.resolve()


def verify_sample(path):
    """Backward-compatible verifier for the first pinned sample file."""
    return verify_slide(path, sample_metadata()["slides"][0])


def _api_json(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "PathXAI-tutorial/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _fetch_live_inventory(manifest):
    """Return all live GDC Slide Image records for the pinned case."""
    filters = {
        "op": "in",
        "content": {
            "field": "cases.submitter_id",
            "value": [manifest["case_id"]],
        },
    }
    fields = [
        "file_id", "file_name", "file_size", "md5sum", "access",
        "data_type", "data_category", "cases.submitter_id",
        "cases.case_id", "cases.project.project_id", "cases.samples.sample_type",
    ]
    query = urllib.parse.urlencode({
        "filters": json.dumps(filters, separators=(",", ":")),
        "fields": ",".join(fields),
        "format": "json",
        "size": "1000",
    })
    data = _api_json(f"{_GDC_API}/files?{query}")["data"]
    if data.get("pagination", {}).get("total") != len(data.get("hits", [])):
        raise ValueError("GDC returned a truncated case inventory; refusing to proceed.")
    return [row for row in data["hits"] if row.get("data_type") == "Slide Image"]


def _validate_live_inventory(manifest):
    """Fail closed if GDC's slide inventory or pinned metadata has changed."""
    expected = {row["file_id"]: row for row in manifest["slides"]}
    excluded = {row["file_id"]: row for row in manifest.get("excluded_slides", [])}
    pinned = expected | excluded
    rows = _fetch_live_inventory(manifest)
    live = {row.get("file_id"): row for row in rows}
    if len(live) != len(rows) or set(live) != set(pinned):
        raise ValueError(
            "GDC case slide inventory changed; review the patient manifest before use."
        )
    keys = ("file_name", "file_size", "md5sum", "access", "data_type", "data_category")
    for file_id, spec in pinned.items():
        row = live[file_id]
        for key in keys:
            if row.get(key) != spec.get(key):
                raise ValueError(
                    f"GDC metadata changed for {spec['file_name']} ({key}); "
                    "review the pinned manifest before use."
                )
        cases = row.get("cases", [])
        if not any(c.get("submitter_id") == manifest["case_id"]
                   and c.get("project", {}).get("project_id") == manifest["project_id"]
                   for c in cases):
            raise ValueError(f"GDC case/project mismatch for {spec['file_name']}.")
        sample_types = {
            s.get("sample_type")
            for case in cases
            for s in case.get("samples", [])
        }
        if spec.get("sample_type") not in sample_types:
            raise ValueError(f"GDC sample type changed for {spec['file_name']}.")
    return rows


def _index_existing_slides(existing_dir, manifest):
    """Find exact pinned names recursively and reject ambiguous/unknown files."""
    root = Path(existing_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    known = {
        row["file_name"]: row
        for row in manifest["slides"] + manifest.get("excluded_slides", [])
    }
    matches = {name: [] for name in known}
    prefix = f"{manifest['case_id']}-"
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".svs" or not path.name.startswith(prefix):
            continue
        if path.name not in known:
            raise ValueError(f"Unexpected slide for {manifest['case_id']}: {path.name}")
        if path.name in matches:
            matches[path.name].append(path)
    found = {}
    for row in manifest["slides"]:
        paths = matches[row["file_name"]]
        if len(paths) > 1:
            raise ValueError(f"Multiple existing copies found for {row['file_name']}.")
        if paths:
            found[row["file_id"]] = verify_slide(paths[0], row)
    return found


def _download_one(slide, destination, progress=print):
    destination = Path(destination)
    if destination.exists():
        return verify_slide(destination, slide)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > slide["file_size"]:
        raise ValueError(f"Partial file is too large: {part.name}")
    if offset < slide["file_size"]:
        if shutil.disk_usage(destination.parent).free < (
            slide["file_size"] - offset + 256 * 1024**2
        ):
            raise OSError(f"Not enough free disk space for {slide['file_name']}.")
        headers = {"User-Agent": "PathXAI-tutorial/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(
            f"{_GDC_API}/data/{slide['file_id']}", headers=headers
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            if offset and response.status == 206:
                value = response.headers.get("Content-Range", "").strip()
                match = re.fullmatch(r"(?:bytes\s+)?(\d+)-(\d+)/(\d+)", value)
                if (not match or int(match[1]) != offset
                        or not offset <= int(match[2]) < slide["file_size"]
                        or int(match[3]) != slide["file_size"]):
                    raise ValueError("GDC returned an unexpected range; refusing to append bytes.")
                mode, received = "ab", offset
            elif response.status == 200:
                mode, received = "wb", 0
            else:
                raise ValueError(f"Unexpected GDC download status: {response.status}")
            last = received
            with part.open(mode) as stream:
                while True:
                    block = response.read(_BLOCK_SIZE)
                    if not block:
                        break
                    received += len(block)
                    if received > slide["file_size"]:
                        raise ValueError(f"GDC transfer exceeded expected size: {slide['file_name']}")
                    stream.write(block)
                    if progress and received - last >= 128 * 1024**2:
                        progress(f"{slide['file_name']}: {received / slide['file_size']:.0%}")
                        last = received
                stream.flush()
                os.fsync(stream.fileno())
    verify_slide(part, slide)
    os.replace(part, destination)
    return destination.resolve()


def download_patient(data_dir="data", *, existing_dir=None, progress=print):
    """Return all pinned patient slide paths in their recorded manifest order.

    ``existing_dir`` is searched recursively. Matching files are reused only
    after exact size, GDC MD5 and pinned SHA-256 checks. Missing slides are
    downloaded into ``data_dir``. The sample includes the five DX slides and TS1.
    """
    manifest = sample_metadata()
    _validate_live_inventory(manifest)
    existing = _index_existing_slides(existing_dir, manifest) if existing_dir else {}
    output_dir = Path(data_dir).expanduser().resolve()
    results = []
    for slide in manifest["slides"]:
        if slide["file_id"] in existing:
            results.append(existing[slide["file_id"]])
        else:
            results.append(_download_one(slide, output_dir / slide["file_name"], progress))
    (output_dir / "sample_source.json").parent.mkdir(parents=True, exist_ok=True)
    (output_dir / "sample_source.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return results


def download_sample(data_dir="data", *, existing_file=None, existing_dir=None, progress=print):
    """Backward-compatible single-slide download helper.

    New callers should use :func:`download_patient` to obtain the complete patient slide set.
    """
    manifest = sample_metadata()
    if "slides" not in manifest:
        # Compatibility with callers and archived tests that pin a single file.
        slide = manifest
        def verify_legacy(path):
            path = Path(path).expanduser()
            if not path.is_file() or path.stat().st_size != slide["file_size"]:
                raise ValueError("Sample file missing or size does not match the pinned GDC record.")
            md5, sha = _hashes(path)
            if md5 != slide["md5sum"] or sha != slide["sha256"]:
                raise ValueError("Sample checksum mismatch; do not use this file as the tutorial sample.")
            return path.resolve()

        if existing_file is not None:
            return verify_legacy(existing_file)
        metadata_url = slide.get("metadata_url")
        if metadata_url:
            query = urllib.parse.urlencode({
                "fields": "file_id,file_name,file_size,md5sum,access"
            })
            current = _api_json(metadata_url + "?" + query)["data"]
            for key in ("file_id", "file_name", "file_size", "md5sum", "access"):
                if current.get(key) != slide.get(key):
                    raise ValueError(f"GDC metadata changed for {key}; inspect the pinned sample before downloading.")
        output_dir = Path(data_dir).expanduser().resolve()
        return _download_one(slide, output_dir / slide["file_name"], progress)

    slide = manifest["slides"][0]
    if existing_file is not None:
        return verify_slide(existing_file, slide)
    if existing_dir is not None:
        found = _index_existing_slides(existing_dir, manifest)
        if slide["file_id"] in found:
            return found[slide["file_id"]]
    _validate_live_inventory(manifest)
    output_dir = Path(data_dir).expanduser().resolve()
    path = _download_one(slide, output_dir / slide["file_name"], progress)
    (output_dir / "sample_source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return path
