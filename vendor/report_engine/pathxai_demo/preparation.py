"""WSI to candidate-ROI adapter using the original v1 CONCH implementation."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from typing import Any, Callable

from .assets import _asset
from .profiles import get_profile


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_identity_with_capture(path_value: str, capture_file: Path) -> tuple[str, bool, str | None]:
    """Return existing model identity plus a file SHA captured in the same walk."""
    if not path_value:
        return "", False, None
    path = Path(path_value).expanduser()
    if path.is_file():
        file_digest = _file_sha256(path)
        captured = file_digest if path.resolve() == capture_file.resolve() else None
        return file_digest, True, captured
    if path.is_dir():
        digest = hashlib.sha256()
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            return "", False, None
        captured = None
        for item in files:
            relative = item.relative_to(path).as_posix()
            file_digest = _file_sha256(item)
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(file_digest.encode("ascii"))
            digest.update(b"\n")
            if item.resolve() == capture_file.resolve():
                captured = file_digest
        return digest.hexdigest(), True, captured
    return hashlib.sha256(path_value.encode("utf-8")).hexdigest(), False, None


def _path_identity(path_value: str) -> tuple[str, bool]:
    """Preserve the cache identity API used by existing callers."""
    identity, cacheable, _captured = _path_identity_with_capture(path_value, Path("."))
    return identity, cacheable


def _conch_model_identity(config: dict[str, Any]) -> tuple[str, bool]:
    """Pin the consumed CONCH weight while preserving the existing cache digest."""
    model_path = str(config.get("foundation_model_path") or "")
    if not model_path:
        raise ValueError("The selected CONCH model path is empty")
    pin = _asset("conch")
    path = Path(model_path).expanduser()
    weight = path if path.is_file() else path / str(pin["filename"])
    if not weight.is_file():
        raise ValueError("The selected CONCH model must be a local file or snapshot containing the pinned weight file")
    identity, cacheable, weight_sha = _path_identity_with_capture(model_path, weight)
    if (not cacheable or weight_sha != str(pin["sha256"]).lower()
            or weight.stat().st_size != pin["bytes"]):
        raise ValueError("The selected CONCH weight does not match its pinned identity")
    return identity, cacheable


def _source_identity(source_root: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(source_root.rglob("*.py")):
        if "__pycache__" in item.parts:
            continue
        digest.update(item.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_ranking(path: Path, expected: dict[str, Any]) -> list[dict] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        settings = payload.get("settings") or {}
        if payload.get("model_name") != expected.get("model_name"):
            return None
        if str(payload.get("model_path") or "") != str(expected.get("model_path") or ""):
            return None
        if str(Path(str(payload.get("slide_path") or "")).expanduser().resolve()) != expected.get("slide_path"):
            return None
        expected_settings = expected.get("settings", expected)
        if any(settings.get(key) != value for key, value in expected_settings.items()):
            return None
        rows = list(payload.get("cached_rois") or [])
        if not rows or any(not Path(str(row.get("image_path") or "")).is_file() for row in rows):
            return None
        return rows
    except (OSError, ValueError, TypeError):
        return None


def _source_roots(config: dict[str, Any]) -> tuple[Path, Path]:
    v1_root = Path(str(config.get("v1_root") or config_defaults()["v1_root"])).expanduser()
    pathxai_root = Path(str(config.get("pathxai_root") or config_defaults()["pathxai_root"])).expanduser()
    candidates = [v1_root, v1_root / "src", pathxai_root / "src", pathxai_root]
    source = next((path for path in candidates if (path / "pathxai_agent").is_dir()), None)
    if source is not None:
        sys.path.insert(0, str(source))
    else:
        raise FileNotFoundError("Could not locate original v1 pathxai_agent package")
    return source, pathxai_root


def prepare_candidates(
    config: dict[str, Any],
    case: dict[str, Any],
    case_output: Path,
    emit: Callable[[dict], None],
) -> list[dict]:
    """Generate the original diagnostic candidate pool, or load supplied rows.

    ``candidate_rois`` and ``candidate_json`` support parity/replay fixtures.
    ``manifest_path`` accepts the original JSONL case manifest schema.
    """
    case_id = str(case.get("case_id") or case.get("id") or "").strip()
    if not case_id:
        raise ValueError("case.case_id is required")
    if case.get("candidate_rois") is not None:
        rows = case["candidate_rois"]
        if not isinstance(rows, list):
            raise ValueError("case.candidate_rois must be a list")
        return [dict(row) for row in rows]
    candidate_json = case.get("candidate_json") or case.get("roi_json")
    manifest_path = case.get("manifest_path")
    if candidate_json or manifest_path:
        path = Path(str(candidate_json or manifest_path)).expanduser()
        if candidate_json:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                rows = data.get("candidate_rois", data.get("rois"))
            else:
                rows = data
        else:
            rows = None
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                item = json.loads(raw)
                if str(item.get("case_id") or item.get("item_id") or "") == case_id:
                    rows = item.get("candidate_rois")
                    break
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"No candidate_rois found for case {case_id}")
        return [dict(row) for row in rows]

    slide_paths = case.get("slide_paths") or case.get("slides") or []
    if not isinstance(slide_paths, list) or not slide_paths:
        raise ValueError("case.slide_paths must contain at least one WSI path")
    paths = [Path(str(p)).expanduser() for p in slide_paths]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("WSI file(s) not found: " + ", ".join(missing))
    resolved_paths = [path.resolve() for path in paths]
    stem_sources: dict[str, Path] = {}
    for path in resolved_paths:
        previous = stem_sources.get(path.stem)
        if previous is not None and previous != path:
            raise ValueError(
                f"Multiple slides share the basename {path.stem!r}; original slide_id semantics would collide"
            )
        stem_sources[path.stem] = path

    # ROI selection behavior is fixed by the frozen profile. Users may select
    # the compute device and model asset location only.
    fixed = get_profile(str(case.get("mode") or "core"))["preparation"]
    model = str(fixed["foundation_model"])
    tile_size = int(fixed["tile_size"])
    final_size = int(fixed["final_roi_size"])
    mpp = float(fixed["mpp"])
    batch_size = int(fixed["batch_size"])
    max_rois = int(fixed["max_candidate_rois_per_case"])
    foundation_device = str(config.get("device", "cuda:0"))
    model_path = str(config.get("foundation_model_path") or "")
    model_identity, model_cacheable = _conch_model_identity(config)

    source_root, _runtime_root = _source_roots(config)
    from pathxai_agent.wsi.foundation_roi import FOUNDATION_RANKING_VERSION, foundation_text_similarity_rois
    source_identity = _source_identity(source_root)
    cache_root = Path(str(config.get("roi_cache_dir") or "")).expanduser()
    expected_settings = {
        "tile_size": tile_size,
        "final_roi_size": final_size,
        "max_final_rois": max_rois,
        "mpp": mpp,
        "batch_size": batch_size,
        "num_workers": 0,
        "device": foundation_device,
        "amp": bool(fixed["amp"]),
        "max_tiles": None,
        "diversity_factor": 0.75,
        "site_top_k_tiles": 8,
        "ranking_mode": "diagnostic",
        "site_roi_fraction": 0.4,
        "max_site_rois": 3,
        "max_tumor_rois": 5,
        "ranking_version": FOUNDATION_RANKING_VERSION,
    }
    all_rows: list[dict] = []
    run_roi_root = case_output / "candidate_rois"
    for index, slide in enumerate(resolved_paths, start=1):
        if emit:
            emit({"stage": "wsi_preparation", "message": f"Preparing slide {index}/{len(paths)} with original {model.upper()} ranking"})
        slide_id = slide.stem
        # Keep original slide_id values in ROI payloads, while namespacing the
        # physical cache by input identity, model path, and ranking settings.
        stat = slide.stat()
        identity = {
            "slide_path": str(slide), "slide_sha256": _file_sha256(slide),
            "model_name": model, "model_path": model_path, "model_sha256": model_identity,
            "source_sha256": source_identity, "settings": expected_settings,
        }
        namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        cacheable = model_cacheable and bool(str(cache_root))
        roi_dir = cache_root / model / namespace / slide_id if cacheable else run_roi_root / namespace / slide_id
        roi_dir.mkdir(parents=True, exist_ok=True)
        ranking_path = roi_dir / "foundation_roi_ranking.json"
        expected = {
            "settings": expected_settings,
            "model_name": model,
            "model_path": model_path,
            "slide_path": str(slide),
        }
        rows = _load_ranking(ranking_path, expected) if cacheable else None
        if rows is None:
            rows, _metadata = foundation_text_similarity_rois(
                slide_path=slide,
                slide_id=slide_id,
                out_dir=roi_dir,
                model_name=model,
                model_path=model_path or None,
                token=config.get("hf_token") or None,
                tile_size=tile_size,
                final_roi_size=final_size,
                max_final_rois=max_rois,
                mpp=mpp,
                batch_size=batch_size,
                num_workers=0,
                device=foundation_device,
                amp=bool(fixed["amp"]),
                max_tiles=None,
                diversity_factor=0.75,
                ranking_mode="diagnostic",
                pbar=True,
            )
            # The frozen implementation writes this file. It lives under the
            # content-addressed cache namespace, never inside the source tree.
            if cacheable and not ranking_path.is_file():
                metadata_path = roi_dir / "foundation_roi_ranking.json"
                if metadata_path != ranking_path:
                    metadata_path.write_text(json.dumps({
                        "slide_path": str(slide), "model_name": model, "model_path": model_path,
                        "settings": expected_settings, "cached_rois": rows,
                    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        for row in rows:
            enriched = dict(row)
            metadata = dict(enriched.get("metadata") or {})
            # Match the original patient-manifest adapter: the case and cache
            # index live outside the ROI payload, whose added field was only
            # the real source WSI. Extra metadata can alter serialized evidence.
            metadata["slide_path"] = str(slide)
            enriched["metadata"] = metadata
            all_rows.append(enriched)
    all_rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    return all_rows[:max_rois]

