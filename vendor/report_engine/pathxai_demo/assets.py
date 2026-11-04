"""Pinned model asset verification, import, and user-authorized download.

No token is persisted. Downloads are restricted to catalogued Hugging Face
repositories and immutable revisions; weights are installed only after the
catalogue's byte count and SHA-256 both match.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import uuid


_APP_ROOT = Path(__file__).resolve().parents[1]
_CHUNK_SIZE = 1024 * 1024
_DEFAULT_TIMEOUT = 30.0
_MAX_TIMEOUT = 600.0
_PORTABLE_SEG_CHECKPOINT_ALIAS = "PathXAI-Seg.pth"
_KEEP_REFERENCE_FILENAME = "keep_references.json"
_KEEP_REPOSITORY = "Loie/KEEP_dataset"
_KEEP_REVISION = "dbf19e71e7d9b6516b890cfd60e6e94baf881e23"
_KEEP_FILES = {
    "disease_knowledge_graph.json": "keep_disease_graph",
    "pathology_vl_semantic_groups.json": "keep_semantic_groups",
}


class AssetError(RuntimeError):
    """Safe, user-facing asset operation error."""


class AssetAccessDenied(AssetError):
    """The current user's Hugging Face account cannot access an asset."""


class AssetCancelled(AssetError):
    """An asset operation was cancelled by the caller."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssetError(f"Asset catalogue is unavailable: {path.name}") from exc
    if not isinstance(value, dict):
        raise AssetError(f"Asset catalogue has an invalid structure: {path.name}")
    return value


def _safe_filename(value: Any) -> str:
    name = str(value or "")
    if not name or name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        raise AssetError("Asset catalogue contains an unsafe filename")
    return name


def _catalog_map() -> dict[str, dict[str, Any]]:
    models = _read_json(_APP_ROOT / "assets" / "model_references.json")
    bundled = _read_json(_APP_ROOT / "assets" / "manifest.json")
    entries: dict[str, dict[str, Any]] = {}
    model_rows = models.get("assets")
    bundled_rows = bundled.get("assets")
    if not isinstance(model_rows, list) or not isinstance(bundled_rows, list):
        raise AssetError("Asset catalogue has an invalid structure")
    for item in model_rows:
        if not isinstance(item, dict):
            raise AssetError("Model reference catalogue contains an invalid asset entry")
        asset_id = str(item.get("id") or "")
        filename = _safe_filename(item.get("filename"))
        size = item.get("bytes")
        digest = item.get("sha256")
        if (not asset_id or asset_id in entries or not isinstance(size, int) or isinstance(size, bool)
                or size <= 0 or not _valid_sha256(digest)):
            raise AssetError("Model reference catalogue contains an invalid asset entry")
        if asset_id == "seg_d1":
            entries[asset_id] = {
                "id": asset_id, "filename": filename, "bytes": size, "sha256": digest,
                "kind": "PathXAI-Seg checkpoint", "source": "local-only",
                "label": "PathXAI-Seg", "repo_id": None, "repository": None,
                "revision": None, "downloadable": False, "can_download": False,
            }
        else:
            repo_id = str(item.get("repository") or "")
            revision = str(item.get("revision") or "")
            if not repo_id or not revision or revision.lower() in {"main", "master", "latest"}:
                raise AssetError(f"Asset {asset_id!r} does not have a pinned Hugging Face revision")
            entries[asset_id] = {
                "id": asset_id, "filename": filename, "bytes": size, "sha256": digest,
                "kind": asset_id.upper(), "source": "huggingface",
                "label": asset_id.upper(), "repo_id": repo_id, "repository": repo_id,
                "revision": revision, "downloadable": True, "can_download": True,
            }
    for item in bundled_rows:
        if not isinstance(item, dict):
            raise AssetError("Bundled asset manifest contains an invalid entry")
        asset_id = str(item.get("id") or "")
        filename = _safe_filename(item.get("filename"))
        size = item.get("bytes")
        digest = item.get("sha256")
        if (not asset_id or asset_id in entries or not isinstance(size, int) or isinstance(size, bool)
                or size <= 0 or not _valid_sha256(digest)):
            raise AssetError("Bundled asset manifest contains an invalid entry")
        entries[asset_id] = {
            "id": asset_id, "filename": filename, "bytes": size, "sha256": digest,
            "kind": "SAM3 tokenizer", "source": "bundled",
            "label": "SAM3 tokenizer", "repo_id": None, "repository": None,
            "revision": None, "downloadable": False, "can_download": False,
        }
    return entries


def _valid_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(char in "0123456789abcdefABCDEF" for char in text)


def asset_catalog() -> list[dict[str, Any]]:
    """Return the pinned asset index for the GUI, with no local token data."""
    entries = [dict(entry) for entry in _catalog_map().values()]
    definition = _keep_dataset_definition(required=False)
    if definition:
        dataset = definition["dataset"]
        files = definition["assets"]
        entries.append({
            "id": "keep_dataset", "filename": "KEEP_dataset", "bytes": sum(row["bytes"] for row in files),
            "sha256": None, "kind": "KEEP knowledge resources", "source": "huggingface_dataset",
            "label": "KEEP knowledge resources", "repo_id": dataset["repository"],
            "repository": dataset["repository"], "revision": dataset["revision"],
            "repo_type": dataset["repo_type"], "downloadable": True, "can_download": True,
            "files": [{"filename": row["filename"], "bytes": row["bytes"], "sha256": row["sha256"]}
                      for row in files],
            "access_note": dataset["access_note"],
            "source_identity_status": dataset["source_identity_status"],
        })
    return entries


def _keep_dataset_definition(*, required: bool = True,
                             app_root: str | Path | None = None) -> dict[str, Any] | None:
    root = Path(app_root) if app_root is not None else _APP_ROOT
    path = root / "assets" / _KEEP_REFERENCE_FILENAME
    if not path.is_file():
        if required:
            raise AssetError("KEEP reference catalogue is unavailable")
        return None
    payload = _read_json(path)
    dataset = payload.get("dataset")
    rows = payload.get("assets")
    if (payload.get("schema") != "pathxai-keep-references-v1" or not isinstance(dataset, dict)
            or dataset.get("id") != "keep_dataset" or dataset.get("repo_type") != "dataset"
            or dataset.get("repository") != _KEEP_REPOSITORY
            or dataset.get("revision") != _KEEP_REVISION
            or str(dataset.get("revision")).lower() in {"main", "master", "latest"}
            or not isinstance(rows, list) or len(rows) != 2):
        raise AssetError("KEEP reference catalogue has an invalid structure")
    seen: set[str] = set()
    clean_rows: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            raise AssetError("KEEP reference catalogue contains an invalid file entry")
        filename = _safe_filename(item.get("filename"))
        asset_id = str(item.get("id") or "")
        size = item.get("bytes")
        digest = item.get("sha256")
        if (filename not in _KEEP_FILES or asset_id != _KEEP_FILES[filename] or filename in seen
                or not isinstance(size, int) or isinstance(size, bool) or size <= 0
                or not _valid_sha256(digest)):
            raise AssetError("KEEP reference catalogue contains an invalid file identity")
        seen.add(filename)
        clean_rows.append({**item, "filename": filename, "id": asset_id,
                           "bytes": size, "sha256": str(digest).lower(),
                           "repository": str(dataset["repository"]),
                           "repo_id": str(dataset["repository"]),
                           "revision": str(dataset["revision"]), "repo_type": "dataset",
                           "kind": "KEEP knowledge resource", "label": "KEEP knowledge resource",
                           "source": "huggingface_dataset", "downloadable": True, "can_download": True})
    if seen != set(_KEEP_FILES):
        raise AssetError("KEEP reference catalogue must contain both required files")
    return {"dataset": dataset, "assets": sorted(clean_rows, key=lambda row: row["filename"])}


def _regular_file(path: Path) -> bool:
    """Cheap discovery predicate only; byte identity is checked before use."""
    try:
        return path.is_file()
    except OSError:
        return False


def _model_candidates(asset: dict[str, Any], app_root: Path, home: Path,
                      environ: dict[str, str]) -> list[Path]:
    """Return deterministic package/workspace/HF-cache locations for one asset.

    No directory walk or content hashing is done here. The scientific worker
    performs the catalogued size and SHA check before inference.
    """
    asset_id = str(asset["id"])
    filename = str(asset["filename"])
    workspace_models = app_root / "workspace" / "models"
    packaged_models = app_root / "models"
    candidates: list[Path] = []

    if asset_id == "seg_d1":
        # The shipped D1 checkpoint uses the portable alias; imported copies
        # use the fixed catalog filename under workspace/models.
        candidates.extend((packaged_models / "PathXAI-Seg" / _PORTABLE_SEG_CHECKPOINT_ALIAS,
                          packaged_models / filename,
                          workspace_models / filename,
                          workspace_models / "PathXAI-Seg" / _PORTABLE_SEG_CHECKPOINT_ALIAS))
        return candidates

    candidates.extend((workspace_models / filename,
                       packaged_models / filename,
                       packaged_models / asset_id / filename))
    repo_id = str(asset.get("repo_id") or "")
    revision = str(asset.get("revision") or "")
    if not repo_id or not revision:
        return candidates

    repo_dir = "models--" + repo_id.replace("/", "--")
    cache_roots: list[Path] = []
    for value in (environ.get("HF_HUB_CACHE"), environ.get("HUGGINGFACE_HUB_CACHE")):
        if value:
            cache_roots.append(Path(value).expanduser())
    hf_home = environ.get("HF_HOME")
    if hf_home:
        cache_roots.append(Path(hf_home).expanduser() / "hub")
    cache_roots.append(home / ".cache" / "huggingface" / "hub")
    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in cache_roots:
        try:
            key = str(root.resolve())
        except (OSError, RuntimeError):
            key = str(root.absolute())
        if key not in seen:
            seen.add(key)
            unique_roots.append(root)
    for root in unique_roots:
        repository = root / repo_dir
        candidates.extend((repository / "snapshots" / revision / filename,
                           repository / "blobs" / str(asset["sha256"]).lower()))
    return candidates


def discover_asset_path(asset_id: str, *, app_root: str | Path | None = None,
                        environ: dict[str, str] | None = None,
                        home: str | Path | None = None) -> str:
    """Find an existing package/import/HF-cache path for a pinned asset.

    This is path discovery, not verification. It returns the first exact
    filename/revision location whose expected file exists, or ``""``. The
    engine later verifies the full catalogued byte count and SHA-256 before
    starting scientific work.
    """
    try:
        asset = _asset(asset_id)
    except AssetError:
        # Configuration defaults must remain constructible so normal
        # preflight can report a malformed catalogue without spawning a
        # worker. No path is selected in that case.
        return ""
    root = Path(app_root) if app_root is not None else _APP_ROOT
    env = dict(os.environ if environ is None else environ)
    home_path = Path(home).expanduser() if home is not None else Path.home()
    for candidate in _model_candidates(asset, root, home_path, env):
        # CONCH's preparation adapter accepts either a snapshot directory
        # containing the pinned weight or the weight itself.
        if asset_id == "conch" and candidate.name != str(asset["filename"]):
            weight = candidate / str(asset["filename"])
            if _regular_file(weight):
                return str(candidate)
        if _regular_file(candidate):
            return str(candidate)
    return ""


def _keep_entry(asset_id: str) -> dict[str, Any]:
    definition = _keep_dataset_definition()
    for entry in definition["assets"]:
        if entry["id"] == asset_id:
            return entry
    raise AssetError("Unknown KEEP asset")


def _keep_candidate_roots(app_root: Path, home: Path, environ: dict[str, str]) -> list[Path]:
    definition = _keep_dataset_definition(required=False, app_root=app_root)
    if not definition:
        return []
    dataset = definition["dataset"]
    repo_dir = "datasets--" + str(dataset["repository"]).replace("/", "--")
    cache_roots: list[Path] = []
    for value in (environ.get("HF_HUB_CACHE"), environ.get("HUGGINGFACE_HUB_CACHE")):
        if value:
            cache_roots.append(Path(value).expanduser())
    if environ.get("HF_HOME"):
        cache_roots.append(Path(environ["HF_HOME"]).expanduser() / "hub")
    cache_roots.append(home / ".cache" / "huggingface" / "hub")
    roots = [app_root / "workspace" / "keep"]
    roots.extend(root / repo_dir / "snapshots" / str(dataset["revision"]) for root in cache_roots)
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except (OSError, RuntimeError):
            key = str(root.absolute())
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def discover_keep_root(*, app_root: str | Path | None = None,
                       environ: dict[str, str] | None = None,
                       home: str | Path | None = None) -> str:
    """Return a KEEP directory only if both fixed files pass full SHA checks."""
    root = Path(app_root) if app_root is not None else _APP_ROOT
    env = dict(os.environ if environ is None else environ)
    home_path = Path(home).expanduser() if home is not None else Path.home()
    try:
        candidates = _keep_candidate_roots(root, home_path, env)
    except AssetError:
        return ""
    for candidate in candidates:
        try:
            result = verify_keep_dataset(candidate, app_root=root)
        except (AssetError, OSError):
            continue
        if result["ok"]:
            return result["path"]
    return ""


def verify_keep_dataset(root: str | Path, *, app_root: str | Path | None = None,
                        progress: Callable[[dict[str, Any]], None] | None = None,
                        cancel_event: Any = None) -> dict[str, Any]:
    """Verify both required KEEP files and return no content or credentials."""
    definition = _keep_dataset_definition(app_root=app_root)
    supplied = Path(root).expanduser()
    try:
        resolved_root = supplied.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AssetError("KEEP folder does not exist or cannot be resolved") from exc
    if not resolved_root.is_dir():
        raise AssetError("Select the folder containing both KEEP JSON files")
    verified: list[dict[str, Any]] = []
    total = sum(row["bytes"] for row in definition["assets"])
    done = 0
    for entry in definition["assets"]:
        file_path = resolved_root / entry["filename"]
        if not file_path.is_file():
            raise AssetError("KEEP folder must contain both fixed JSON files")
        digest, size = _sha256_file(file_path, cancel_event=cancel_event)
        if size != entry["bytes"] or digest.lower() != entry["sha256"]:
            raise AssetError(f"KEEP file failed pinned size or SHA-256 verification: {entry['filename']}")
        done += size
        if progress:
            progress({"stage": "asset_verify", "asset_id": "keep_dataset",
                      "message": "Verified KEEP file " + entry["filename"],
                      "downloaded": done, "total": total,
                      "bytes_downloaded": done, "bytes_total": total,
                      "percent": round(done * 100.0 / total, 2)})
        verified.append({"filename": entry["filename"], "bytes": size, "sha256": digest})
    return {"ok": True, "asset_id": "keep_dataset", "path": str(resolved_root),
            "files": verified, "message": "Both KEEP files match the pinned identities"}


def _replace_verified_directory(stage: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    parent = target.parent.resolve()
    if target.parent.resolve() != parent or target.is_symlink():
        raise AssetError("Unsafe KEEP destination")
    backup = target.with_name(".keep.previous." + uuid.uuid4().hex)
    had_target = target.exists()
    if had_target:
        if not target.is_dir():
            raise AssetError("KEEP destination is not a directory")
        os.replace(target, backup)
    try:
        os.replace(stage, target)
    except Exception:
        if had_target and backup.exists():
            os.replace(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _install_keep_pair(source: Path, destination_root: str | Path, *,
                       token: str | None, download: bool,
                       progress: Callable[[dict[str, Any]], None] | None,
                       cancel_event: Any) -> dict[str, Any]:
    definition = _keep_dataset_definition()
    target = Path(destination_root).expanduser().absolute()
    parent = target.parent.resolve()
    if target.is_symlink() or target.parent.resolve() != parent:
        raise AssetError("Unsafe KEEP destination")
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.parent / (".keep-stage." + uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    total = sum(row["bytes"] for row in definition["assets"])
    offset = 0
    try:
        for entry in definition["assets"]:
            _check_cancel(cancel_event)

            def file_progress(event, *, base=offset, current=entry):
                if progress:
                    row = dict(event)
                    count = min(int(event.get("downloaded", 0)), int(current["bytes"]))
                    row.update(asset_id="keep_dataset", downloaded=base + count,
                               total=total, bytes_downloaded=base + count,
                               bytes_total=total,
                               percent=round((base + count) * 100.0 / total, 2),
                               message=("Downloading " if download else "Importing ") + current["filename"])
                    progress(row)

            if download:
                download_asset(entry["id"], stage, token=str(token or ""),
                               progress=file_progress, cancel_event=cancel_event)
            else:
                import_asset(entry["id"], source, stage,
                             progress=file_progress, cancel_event=cancel_event)
            offset += entry["bytes"]
        _check_cancel(cancel_event)
        verification = verify_keep_dataset(stage, progress=progress, cancel_event=cancel_event)
        _check_cancel(cancel_event)
        _replace_verified_directory(stage, target)
        return {**verification, "path": str(target), "downloaded": bool(download), "imported": not download,
                "repository": definition["dataset"]["repository"],
                "revision": definition["dataset"]["revision"],
                "source_identity_status": definition["dataset"]["source_identity_status"]}
    except Exception:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
        raise


def import_keep_dataset(source_path: str | Path, destination_root: str | Path, *,
                        progress: Callable[[dict[str, Any]], None] | None = None,
                        cancel_event: Any = None) -> dict[str, Any]:
    """Import both locally supplied KEEP files as one verified operation."""
    source = Path(source_path).expanduser()
    if not source.is_dir():
        raise AssetError("Select the folder containing both KEEP JSON files")
    return _install_keep_pair(source, destination_root, token=None, download=False,
                              progress=progress, cancel_event=cancel_event)


def download_keep_dataset(destination_root: str | Path, token: str = "", *,
                          progress: Callable[[dict[str, Any]], None] | None = None,
                          cancel_event: Any = None) -> dict[str, Any]:
    """Download both gated KEEP files from the fixed dataset revision."""
    return _install_keep_pair(Path(), destination_root, token=token, download=True,
                              progress=progress, cancel_event=cancel_event)


def _asset(asset_id: str) -> dict[str, Any]:
    try:
        return _catalog_map()[str(asset_id)]
    except KeyError as exc:
        try:
            return _keep_entry(str(asset_id))
        except AssetError:
            raise AssetError("Unknown model asset") from exc


def _resolve_asset_file(asset: dict[str, Any], path: str | Path) -> Path:
    supplied = Path(path).expanduser()
    accepted_names = [str(asset["filename"])]
    if asset.get("id") == "seg_d1" and _PORTABLE_SEG_CHECKPOINT_ALIAS not in accepted_names:
        accepted_names.append(_PORTABLE_SEG_CHECKPOINT_ALIAS)
    if supplied.is_dir():
        selected = next((supplied / name for name in accepted_names if (supplied / name).is_file()), None)
        if selected is None:
            expected = " or ".join(repr(name) for name in accepted_names)
            raise AssetError(f"Select a directory containing the catalogued file {expected}")
        supplied = selected
    elif supplied.name not in accepted_names:
        expected = " or ".join(repr(name) for name in accepted_names)
        raise AssetError(f"Select the catalogued file {expected}")
    try:
        resolved = supplied.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise AssetError("Selected asset file does not exist or cannot be resolved") from exc
    if not resolved.is_file():
        raise AssetError("Selected asset is not a regular file")
    return resolved


def _sha256_file(path: Path, *, cancel_event: Any = None,
                 progress: Callable[[dict[str, Any]], None] | None = None,
                 asset: dict[str, Any] | None = None) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        while True:
            _check_cancel(cancel_event)
            chunk = stream.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            count += len(chunk)
            if progress and asset:
                progress(_progress_event(asset, count, asset["bytes"], "verifying"))
    return digest.hexdigest(), count


def _check_cancel(cancel_event: Any) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AssetCancelled("Asset operation cancelled")


def _progress_event(asset: dict[str, Any], current: int, total: int, stage: str) -> dict[str, Any]:
    return {
        "stage": "asset_download" if stage == "downloading" else "asset_verify",
        "asset_id": asset["id"], "message": "Downloading asset" if stage == "downloading" else "Verifying asset",
        "downloaded": current, "total": total,
        "bytes_downloaded": current, "bytes_total": total,
        "percent": round(min(100.0, current * 100.0 / total), 2) if total else 0.0,
    }


def verify_asset(asset_id: str, path: str | Path,
                 progress: Callable[[dict[str, Any]], None] | None = None,
                 cancel_event: Any = None) -> dict[str, Any]:
    """Verify a local file (or a directory containing its catalogued filename)."""
    asset = _asset(asset_id)
    resolved = _resolve_asset_file(asset, path)
    digest, size = _sha256_file(resolved, cancel_event=cancel_event, progress=progress, asset=asset)
    valid = size == asset["bytes"] and digest.lower() == asset["sha256"].lower()
    return {
        "ok": valid, "asset_id": asset["id"], "path": str(resolved),
        "expected_bytes": asset["bytes"], "actual_bytes": size,
        "expected_sha256": asset["sha256"], "actual_sha256": digest,
        "message": "Asset verified" if valid else "Asset size or SHA-256 does not match the pinned reference",
    }


def _model_destination(asset: dict[str, Any], model_dir: str | Path) -> tuple[Path, Path]:
    root = Path(model_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(asset["filename"])
    target = root / filename
    if target.is_symlink():
        raise AssetError("Refusing to replace a symbolic-link asset destination")
    if target.parent.resolve() != root:
        raise AssetError("Asset destination escapes the selected model directory")
    return root, target


def _copy_verified(source: Path, target: Path, asset: dict[str, Any], *,
                   cancel_event: Any, progress: Callable[[dict[str, Any]], None] | None) -> dict[str, Any]:
    fd, temp_name = tempfile.mkstemp(prefix=f".{asset['id']}.", suffix=".part", dir=str(target.parent))
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    copied = 0
    try:
        with os.fdopen(fd, "wb") as out, source.open("rb") as inp:
            while True:
                _check_cancel(cancel_event)
                chunk = inp.read(_CHUNK_SIZE)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > asset["bytes"]:
                    raise AssetError("Selected file exceeds the pinned asset size")
                out.write(chunk)
                digest.update(chunk)
                if progress:
                    progress(_progress_event(asset, copied, asset["bytes"], "downloading"))
            out.flush()
            os.fsync(out.fileno())
        if copied != asset["bytes"] or digest.hexdigest().lower() != asset["sha256"].lower():
            raise AssetError("Selected file failed pinned size or SHA-256 verification")
        _check_cancel(cancel_event)
        os.replace(temp_path, target)
        return verify_asset(asset["id"], target)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def import_asset(asset_id: str, source_path: str | Path, model_dir: str | Path, *,
                 progress: Callable[[dict[str, Any]], None] | None = None,
                 cancel_event: Any = None) -> dict[str, Any]:
    """Import a locally supplied exact asset under its fixed catalog filename."""
    asset = _asset(asset_id)
    try:
        source = _resolve_asset_file(asset, source_path)
    except AssetError:
        raise
    root, target = _model_destination(asset, model_dir)
    if source == target:
        result = verify_asset(asset_id, target, progress=progress, cancel_event=cancel_event)
        if not result["ok"]:
            raise AssetError(result["message"])
        return result
    # A source symlink is acceptable only after resolving it to a regular file
    # with the exact pinned bytes. The destination is always a normal file.
    _check_cancel(cancel_event)
    result = _copy_verified(source, target, asset, cancel_event=cancel_event, progress=progress)
    return {**result, "imported": True, "model_dir": str(root)}


class _HuggingFaceRedirectHandler(HTTPRedirectHandler):
    """Permit HTTPS redirects but never forward auth to non-HF hosts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = urlsplit(newurl)
        if new.scheme != "https" or not new.hostname or "@" in new.netloc:
            return None
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None
        if new.hostname.lower() != "huggingface.co":
            sensitive = {"authorization", "cookie", "proxy-authorization"}
            for header_map in (redirected.headers, redirected.unredirected_hdrs):
                for key in tuple(header_map):
                    if key.lower() in sensitive:
                        header_map.pop(key, None)
        return redirected


def _build_hf_opener():
    return build_opener(_HuggingFaceRedirectHandler())


def _hf_url(asset: dict[str, Any]) -> str:
    if not asset.get("downloadable") or not asset.get("repo_id") or not asset.get("revision"):
        raise AssetError("This asset has no approved download repository; import a local verified copy")
    repo = str(asset["repo_id"])
    revision = str(asset["revision"])
    filename = _safe_filename(asset["filename"])
    if ".." in repo.split("/") or repo.startswith("/") or "?" in repo or "#" in repo:
        raise AssetError("Pinned asset repository is invalid")
    prefix = "datasets/" if asset.get("repo_type") == "dataset" else ""
    return ("https://huggingface.co/" + prefix + quote(repo, safe="/") + "/resolve/"
            + quote(revision, safe="") + "/" + quote(filename, safe=""))


def download_asset(asset_id: str, model_dir: str | Path, token: str = "", *,
                   progress: Callable[[dict[str, Any]], None] | None = None,
                   cancel_event: Any = None, timeout: float = _DEFAULT_TIMEOUT,
                   opener: Any = None) -> dict[str, Any]:
    """Download a pinned HF asset with caller-owned credentials and atomic install.

    ``opener`` is injectable for tests; production uses urllib with a redirect
    handler that strips auth before any request leaves huggingface.co.
    """
    asset = _asset(asset_id)
    url = _hf_url(asset)
    try:
        bounded_timeout = float(timeout)
    except (TypeError, ValueError) as exc:
        raise AssetError("Download timeout must be a positive number") from exc
    if bounded_timeout <= 0 or bounded_timeout > _MAX_TIMEOUT:
        raise AssetError(f"Download timeout must be between 0 and {_MAX_TIMEOUT:g} seconds")
    _check_cancel(cancel_event)
    root, target = _model_destination(asset, model_dir)
    if target.exists():
        existing = verify_asset(asset_id, target, progress=progress, cancel_event=cancel_event)
        if existing["ok"]:
            return {**existing, "downloaded": False, "model_dir": str(root)}

    request_headers = {"User-Agent": "PathXAI-Demo/asset-manager"}
    if token:
        request_headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=request_headers, method="GET")
    client = opener or _build_hf_opener()
    fd, temp_name = tempfile.mkstemp(prefix=f".{asset_id}.", suffix=".part", dir=str(root))
    temp_path = Path(temp_name)
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with os.fdopen(fd, "wb") as out:
            try:
                response = client.open(request, timeout=bounded_timeout)
            except HTTPError as exc:
                if exc.code in {401, 403}:
                    raise AssetAccessDenied(
                        "Hugging Face denied access. Request access to this gated model, accept its terms, "
                        "then retry with your own read token."
                    ) from None
                if exc.code == 404:
                    raise AssetAccessDenied(
                        "The pinned asset is unavailable to this account or repository revision. "
                        "Check your Hugging Face access approval and retry."
                    ) from None
                raise AssetError(f"Hugging Face download failed with HTTP status {exc.code}") from None
            except URLError as exc:
                reason = getattr(exc, "reason", None)
                if isinstance(reason, TimeoutError):
                    raise AssetError("Hugging Face download timed out") from None
                raise AssetError("Could not connect to Hugging Face") from None
            except TimeoutError:
                raise AssetError("Hugging Face download timed out") from None
            with response:
                status = getattr(response, "status", 200)
                if status != 200:
                    if status in {401, 403}:
                        raise AssetAccessDenied(
                            "Hugging Face denied access. Request access to this gated model, accept its terms, "
                            "then retry with your own read token."
                        )
                    raise AssetError(f"Hugging Face download failed with HTTP status {status}")
                while True:
                    _check_cancel(cancel_event)
                    try:
                        chunk = response.read(_CHUNK_SIZE)
                    except TimeoutError:
                        raise AssetError("Hugging Face download timed out") from None
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > asset["bytes"]:
                        raise AssetError("Downloaded asset exceeds the pinned size")
                    out.write(chunk)
                    digest.update(chunk)
                    if progress:
                        progress(_progress_event(asset, downloaded, asset["bytes"], "downloading"))
            out.flush()
            os.fsync(out.fileno())
        if downloaded != asset["bytes"] or digest.hexdigest().lower() != asset["sha256"].lower():
            raise AssetError("Downloaded file failed pinned size or SHA-256 verification")
        _check_cancel(cancel_event)
        if target.is_symlink():
            raise AssetError("Refusing to replace a symbolic-link asset destination")
        os.replace(temp_path, target)
        result = verify_asset(asset_id, target)
        if not result["ok"]:
            raise AssetError(result["message"])
        return {**result, "downloaded": True, "model_dir": str(root), "repository": asset["repo_id"],
                "revision": asset["revision"]}
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
