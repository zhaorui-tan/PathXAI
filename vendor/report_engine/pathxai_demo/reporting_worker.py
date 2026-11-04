"""Build an isolated display copy of a completed PathXAI report.

This adapter uses the frozen v2 payload builder and frozen v1 renderer. It
copies scientific inputs without editing them and writes WSI display images
only into the requested view directory.
"""

from __future__ import annotations

import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from .profiles import CORE_ABLATION, FULL_ABLATION


PALETTE = (
    (59, 130, 246),
    (239, 68, 68),
    (34, 197, 94),
    (245, 158, 11),
    (168, 85, 247),
    (20, 184, 166),
    (236, 72, 153),
    (251, 146, 60),
)
MAX_THUMBNAIL_SIDE = 1600
MAX_THUMBNAIL_PIXELS = 32_000_000


class ReportingError(RuntimeError):
    """A required report display input could not be built safely."""


def _slide_id_from_roi(roi: dict[str, Any]) -> str:
    sid = str(roi.get("slide_id") or "").strip()
    if sid:
        return sid
    rid = str(roi.get("roi_id") or "")
    for marker in ("_foundation_", "_zoom_", "_roi_"):
        if marker in rid:
            return rid.split(marker, 1)[0]
    return rid


def _safe_name(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"wsi_thumbnail_{digest}_annotated.jpg"


def product_name_for_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    if normalized == "core":
        return "PathXAI-Report (core)"
    if normalized == "full":
        return "PathXAI-Report"
    raise ReportingError("Report mode must be core or full.")


def _looks_like_slide_id(path: Path, slide_id: str) -> bool:
    stem = path.name
    for suffix in (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".bif"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = stem.casefold()
    wanted = slide_id.casefold()
    return stem == wanted or stem.endswith("." + wanted)


def _candidate_rows(result: dict[str, Any]):
    """Yield explicitly named ROI candidate collections, skipping truth data."""
    blocked = {"private_map", "diagnosis_truth", "ground_truth", "eval", "evaluation", "reference"}
    candidate_keys = {"candidates", "candidate_rois", "roi_candidates"}

    def walk(value: Any):
        if isinstance(value, dict):
            for key, nested in value.items():
                if str(key).lower() in blocked:
                    continue
                if str(key).lower() in candidate_keys and isinstance(nested, list):
                    for row in nested:
                        if isinstance(row, dict):
                            yield row
                yield from walk(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from walk(nested)

    yield from walk(result)


def _path_from_roi_metadata(roi: dict[str, Any]) -> str:
    metadata = roi.get("metadata")
    if isinstance(metadata, dict) and metadata.get("slide_path"):
        return str(metadata["slide_path"])
    return str(roi.get("slide_path") or "")


def resolve_slide_path(
    roi: dict[str, Any], result: dict[str, Any], case: dict[str, Any]
) -> Path:
    """Resolve a WSI only from explicit ROI/candidate/case mappings."""
    sid = _slide_id_from_roi(roi)
    explicit = _path_from_roi_metadata(roi)
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path.resolve()
        raise ReportingError(f"Selected ROI maps to a missing WSI file: {path.name}")

    matching: set[Path] = set()
    for candidate in _candidate_rows(result):
        if _slide_id_from_roi(candidate).casefold() != sid.casefold():
            continue
        candidate_path = _path_from_roi_metadata(candidate)
        if candidate_path:
            candidate_file = Path(candidate_path).expanduser()
            if candidate_file.is_file():
                matching.add(candidate_file.resolve())
    if len(matching) == 1:
        return next(iter(matching))
    if len(matching) > 1:
        raise ReportingError(f"Multiple candidate WSIs map to slide {sid}; refusing to guess.")

    raw_slides = case.get("slide_paths") or []
    if not isinstance(raw_slides, list):
        raise ReportingError("case.slide_paths must be a list of WSI paths.")
    slide_paths = [Path(str(value)).expanduser() for value in raw_slides if str(value).strip()]
    if len(slide_paths) == 1 and slide_paths[0].is_file():
        return slide_paths[0].resolve()
    exact = [path for path in slide_paths if _looks_like_slide_id(path, sid) and path.is_file()]
    unique = {path.resolve() for path in exact}
    if len(unique) == 1:
        return next(iter(unique))
    if len(unique) > 1:
        raise ReportingError(f"Multiple case slide paths match slide {sid}; refusing to guess.")
    raise ReportingError(f"Cannot map selected ROI slide {sid!r} to an original WSI path.")


def _number(value: Any, field: str, *, minimum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportingError(f"Selected ROI has invalid {field}.") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ReportingError(f"Selected ROI has invalid {field}.")
    return number


def _roi_box(roi: dict[str, Any], figure: int, downsample: float) -> dict[str, Any]:
    x = _number(roi.get("x"), "x", minimum=0)
    y = _number(roi.get("y"), "y", minimum=0)
    width = _number(roi.get("width"), "width", minimum=1)
    height = _number(roi.get("height"), "height", minimum=1)
    if downsample <= 0 or not math.isfinite(downsample):
        raise ReportingError("OpenSlide returned an invalid level downsample.")
    # OpenSlide read_region uses level-0 x/y locations and level-local size.
    return {
        "figure": figure,
        "roi_id": str(roi.get("roi_id") or ""),
        "x": x,
        "y": y,
        "width_level0": width * downsample,
        "height_level0": height * downsample,
        "color": "#%02x%02x%02x" % PALETTE[(figure - 1) % len(PALETTE)],
    }


def _generate_annotated_overview(
    slide_path: Path,
    roi_rows: list[tuple[int, dict[str, Any]]],
    out_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        import openslide  # type: ignore
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ReportingError(f"Required WSI display library is unavailable: {exc.name}") from exc

    slide = None
    try:
        slide = openslide.OpenSlide(str(slide_path))
        slide_width, slide_height = (int(v) for v in slide.dimensions)
        downsamples = list(slide.level_downsamples)
        if slide_width <= 0 or slide_height <= 0 or not downsamples:
            raise ReportingError("OpenSlide returned invalid slide dimensions or pyramid levels.")
        target_downsample = max(slide_width / MAX_THUMBNAIL_SIDE,
                                slide_height / MAX_THUMBNAIL_SIDE)
        best_level = int(slide.get_best_level_for_downsample(target_downsample))
        level_dimensions = list(slide.level_dimensions)
        if best_level < 0 or best_level >= len(level_dimensions):
            raise ReportingError("OpenSlide selected an invalid pyramid level for the thumbnail.")
        selected_width, selected_height = (int(v) for v in level_dimensions[best_level])
        if (selected_width <= 0 or selected_height <= 0
                or selected_width * selected_height > MAX_THUMBNAIL_PIXELS):
            raise ReportingError("OpenSlide thumbnail source level exceeds the 32 MP decode guard.")
        thumbnail = slide.get_thumbnail((MAX_THUMBNAIL_SIDE, MAX_THUMBNAIL_SIDE)).convert("RGB")
        tw, th = thumbnail.size
        if (tw <= 0 or th <= 0 or max(tw, th) > MAX_THUMBNAIL_SIDE
                or tw * th > MAX_THUMBNAIL_PIXELS):
            raise ReportingError("OpenSlide thumbnail exceeds the 32 MP display guard.")
        draw = ImageDraw.Draw(thumbnail)
        scale_x, scale_y = tw / slide_width, th / slide_height
        line_width = max(2, tw // 500)
        boxes = []
        figure_numbers = []
        for figure, roi in roi_rows:
            try:
                level = int(roi.get("level", 0) or 0)
            except (TypeError, ValueError) as exc:
                raise ReportingError("Selected ROI has invalid pyramid level.") from exc
            if level < 0 or level >= len(downsamples):
                raise ReportingError("Selected ROI level is outside the OpenSlide pyramid.")
            box = _roi_box(roi, figure, float(downsamples[level]))
            x0 = max(0, min(tw - 1, int(box["x"] * scale_x)))
            y0 = max(0, min(th - 1, int(box["y"] * scale_y)))
            x1 = max(x0 + 1, min(tw - 1, int((box["x"] + box["width_level0"]) * scale_x)))
            y1 = max(y0 + 1, min(th - 1, int((box["y"] + box["height_level0"]) * scale_y)))
            color_rgb = PALETTE[(figure - 1) % len(PALETTE)]
            for inset in range(line_width):
                left, top = x0 + inset, y0 + inset
                right, bottom = max(left, x1 - inset), max(top, y1 - inset)
                draw.rectangle((left, top, right, bottom), outline=color_rgb)
            badge = max(14, line_width * 5)
            draw.rectangle((x0, y0, min(tw - 1, x0 + badge), min(th - 1, y0 + badge)), fill=color_rgb)
            draw.text((x0 + 3, y0 + 2), str(figure), fill=(255, 255, 255))
            boxes.append(box)
            figure_numbers.append(figure)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        thumbnail.save(out_path, format="JPEG", quality=88, optimize=True)
        with Image.open(out_path) as written:
            written.verify()
        return {
            "image": str(out_path),
            "annotated_path": str(out_path),
            "slide_id": _slide_id_from_roi(roi_rows[0][1]),
            "figure_numbers": figure_numbers,
            "roi_boxes": boxes,
        }, boxes
    finally:
        if slide is not None:
            slide.close()


def build_wsi_overviews(
    result: dict[str, Any], case: dict[str, Any], view_case_dir: Path
) -> list[dict[str, Any]]:
    selected = result.get("final_selected_rois") or (result.get("roi_selection") or {}).get("selected_rois") or []
    if not isinstance(selected, list) or not selected:
        raise ReportingError("No selected ROIs are available for WSI overview generation.")

    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for figure, roi in enumerate(selected, start=1):
        if not isinstance(roi, dict):
            continue
        slide_id = _slide_id_from_roi(roi)
        if not slide_id:
            raise ReportingError(f"Selected ROI {figure} has no slide identifier.")
        grouped.setdefault(slide_id, []).append((figure, roi))
    if not grouped:
        raise ReportingError("No selected ROI has a usable slide identifier.")

    overviews = []
    for slide_id, rows in grouped.items():
        slide_path = resolve_slide_path(rows[0][1], result, case)
        if any(resolve_slide_path(roi, result, case) != slide_path for _, roi in rows[1:]):
            raise ReportingError(f"Selected ROIs for slide {slide_id} resolve to different WSI files.")
        out_path = view_case_dir / _safe_name(slide_id)
        overview, _ = _generate_annotated_overview(slide_path, rows, out_path)
        overview["slide_id"] = slide_id
        overviews.append(overview)
    return overviews


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _explicit_slide_path(roi: dict[str, Any], label: str) -> tuple[str, Path]:
    metadata = roi.get("metadata")
    raw = metadata.get("slide_path") if isinstance(metadata, dict) else None
    if not isinstance(raw, str) or not raw.strip():
        raise ReportingError(f"{label} has no explicit source slide mapping.")
    path = Path(raw).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ReportingError(f"{label} source slide is missing or unsafe.")
    return _slide_id_from_roi(roi), path.resolve()


def _validate_auxiliary_grade_rows(
    scientific_dir: Path, case_id: str, result: dict[str, Any], case: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify grade-only rows against their actual manifest and diagnostic output."""
    checkpoint_path = scientific_dir / "ckpt_aux_grade_rois.json"
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise ReportingError("Full result is missing ckpt_aux_grade_rois.json.")
    manifest_path = scientific_dir / "candidate_rois_input.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ReportingError("Full result is missing candidate_rois_input.json.")
    try:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        candidates = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportingError("Full auxiliary checkpoint or candidate manifest is unreadable.") from exc
    meta = checkpoint.get("meta") if isinstance(checkpoint, dict) else None
    cached_rows = meta.get("selected_rois") if isinstance(meta, dict) else None
    if (not isinstance(meta, dict) or meta.get("case_id") != case_id
            or meta.get("purpose") != "grade" or not isinstance(cached_rows, list)):
        raise ReportingError("Full auxiliary grade checkpoint has invalid case, purpose, or selected_rois.")
    # The frozen selector's cache-hit path filters stale cached rows against the
    # current diagnostic selection, but preserves the original cached list in
    # meta.selected_rois. The checkpoint's top-level ids are written from the
    # actual returned RoiInfo list and therefore define what this branch used.
    aux_ids = checkpoint.get("aux_roi_ids")
    if (not isinstance(aux_ids, list)
            or any(not isinstance(value, str) or not value for value in aux_ids)
            or len(set(aux_ids)) != len(aux_ids)):
        raise ReportingError("Auxiliary grade checkpoint has invalid returned ROI IDs.")
    if "aux_roi_count" not in checkpoint:
        raise ReportingError("Auxiliary grade checkpoint is missing its returned ROI count.")
    count_value = checkpoint["aux_roi_count"]
    if isinstance(count_value, bool) or not isinstance(count_value, int) or count_value != len(aux_ids):
        raise ReportingError("Auxiliary grade checkpoint count differs from returned ROI IDs.")
    if not isinstance(candidates, list) or not all(isinstance(row, dict) for row in candidates):
        raise ReportingError("Candidate manifest must be an ordered list of ROI objects.")
    candidate_by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        roi_id = str(candidate.get("roi_id") or "")
        if not roi_id or roi_id in candidate_by_id:
            raise ReportingError("Candidate manifest contains a missing or duplicate ROI ID.")
        candidate_by_id[roi_id] = candidate

    max_aux = meta.get("max_aux_rois")
    if max_aux is not None:
        if isinstance(max_aux, bool) or not isinstance(max_aux, int):
            raise ReportingError("Auxiliary grade checkpoint has an invalid selection limit.")
        if max_aux < len(aux_ids):
            raise ReportingError("Auxiliary grade checkpoint exceeds its recorded selection limit.")

    cached_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in cached_rows:
        if not isinstance(row, dict):
            continue
        nested = row.get("roi")
        nested_id = str(nested.get("roi_id") or "") if isinstance(nested, dict) else ""
        row_id = str(row.get("roi_id") or nested_id or row.get("roi_key") or "")
        if row_id:
            cached_by_id.setdefault(row_id, []).append(row)

    roi_selection = result.get("roi_selection") or {}
    if not isinstance(roi_selection, dict):
        raise ReportingError("Pipeline diagnostic ROI selection metadata is malformed.")
    diagnostic_ids: set[str] = set()
    for collection in (result.get("final_selected_rois") or [],
                       roi_selection.get("selected_rois") or []):
        if not isinstance(collection, list):
            raise ReportingError("Pipeline diagnostic ROI selections are malformed.")
        for item in collection:
            if isinstance(item, dict) and item.get("roi_id"):
                diagnostic_ids.add(str(item["roi_id"]))
    overlap_ids = diagnostic_ids.intersection(aux_ids)

    case_slides = case.get("slide_paths") or []
    if not isinstance(case_slides, list):
        raise ReportingError("case.slide_paths must be a list of WSI paths.")
    case_slide_paths = set()
    for raw_path in case_slides:
        slide = Path(str(raw_path)).expanduser()
        if slide.is_file() and not slide.is_symlink():
            case_slide_paths.add(slide.resolve())

    validated: list[dict[str, Any]] = []
    coordinate_fields = ("x", "y", "width", "height", "level")
    source_modes: list[str] = []
    for index, roi_id in enumerate(aux_ids, start=1):
        candidate = candidate_by_id.get(roi_id)
        cached_matches = [row for row in cached_by_id.get(roi_id, [])
                          if isinstance(row.get("roi"), dict)
                          and str(row["roi"].get("roi_id") or "") == roi_id]
        if candidate is not None:
            # The current candidate manifest is authoritative when available:
            # the frozen selector resolves by key into this pool before using
            # an embedded cached RoiInfo. Cached rows can be stale on a hit.
            roi = candidate
            source_mode = "candidate_manifest"
        elif cached_matches:
            # The frozen selector supports a legitimate embedded-cache fallback
            # if an old candidate key is no longer in the present pool. Use the
            # embedded source row only when its nested ID is exactly returned by
            # the checkpoint; never use a stale row merely because it is cached.
            matching_rows = [row for row in cached_matches
                             if str(row.get("roi_id") or roi_id) == roi_id]
            if not matching_rows:
                raise ReportingError("Embedded auxiliary cache row does not match returned ROI ID.")
            roi = matching_rows[0]["roi"]
            source_mode = "embedded_cache_fallback"
        else:
            raise ReportingError("Returned auxiliary ROI is absent from candidate manifest and valid embedded cache.")
        if str(roi.get("roi_id") or "") != roi_id:
            raise ReportingError("Resolved auxiliary ROI source row differs from returned ROI ID.")
        image_value = str(roi.get("image_path") or "")
        if not image_value:
            raise ReportingError("Auxiliary ROI source row is missing its image path.")
        candidate_image_value = str(candidate.get("image_path") or "") if candidate is not None else ""
        image_path = Path(image_value).expanduser()
        if image_path.is_symlink() or not image_path.is_file():
            raise ReportingError("Auxiliary ROI source image is missing or unsafe.")
        if candidate is not None:
            candidate_image = Path(candidate_image_value).expanduser()
            if (not candidate_image_value or candidate_image.is_symlink()
                    or not candidate_image.is_file()
                    or image_path.resolve() != candidate_image.resolve()):
                raise ReportingError("Auxiliary ROI image does not map to its candidate manifest image.")
            for field in coordinate_fields:
                if field not in roi or field not in candidate or roi[field] != candidate[field]:
                    raise ReportingError(f"Auxiliary ROI {field} differs from its candidate manifest row.")
        else:
            if any(field not in roi or roi[field] is None for field in coordinate_fields):
                raise ReportingError("Embedded auxiliary ROI is missing source coordinates.")
        slide_id, slide_path = _explicit_slide_path(roi, "Auxiliary ROI")
        if candidate is not None:
            candidate_slide_id, candidate_slide = _explicit_slide_path(candidate, "Candidate ROI")
            if slide_path != candidate_slide or slide_id != candidate_slide_id:
                raise ReportingError("Auxiliary ROI slide mapping differs from its candidate manifest row.")
        if case_slides and slide_path not in case_slide_paths:
            raise ReportingError("Auxiliary ROI source slide is absent from case.slide_paths.")
        if _slide_id_from_roi(roi) != slide_id:
            raise ReportingError("Auxiliary ROI slide ID differs from source row identity.")
        source_sha = _sha256_file(image_path)
        validated.append({
            "rank": index, "roi": roi, "roi_id": roi_id,
            "image_path": image_path.resolve(), "image_sha256": source_sha,
            "image_size_bytes": image_path.stat().st_size,
            "slide_id": slide_id, "slide_path": slide_path,
            "coordinates": {field: roi[field] for field in coordinate_fields},
            "source": roi.get("source"), "source_resolution": source_mode,
            "also_diagnostic": roi_id in overlap_ids,
        })
        source_modes.append(source_mode)
    receipt = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "candidate_manifest_path": str(manifest_path),
        "candidate_manifest_sha256": _sha256_file(manifest_path),
        "selected_count": len(validated),
        "selected_roi_ids": [row["roi_id"] for row in validated],
        "source_mapping": "top-level returned aux_roi_ids resolved in order against candidate_rois_input.json; embedded cache fallback only for absent IDs",
        "source_resolution": source_modes,
        "also_diagnostic_roi_ids": [roi_id for roi_id in aux_ids if roi_id in overlap_ids],
    }
    return validated, receipt


def _copy_auxiliary_grade_images(rows: list[dict[str, Any]], view_dir: Path) -> dict[str, Any]:
    if not rows:
        return {"rows": [], "checkpoint": None}
    aux_dir = view_dir / "auxiliary_grade"
    aux_dir.mkdir(parents=True, exist_ok=False)
    display_rows = []
    for row in rows:
        suffix = row["image_path"].suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            raise ReportingError("Auxiliary ROI image format is not supported by the portable exporter.")
        name = hashlib.sha256(row["roi_id"].encode("utf-8")).hexdigest()[:16] + suffix
        target = aux_dir / name
        shutil.copyfile(row["image_path"], target)
        target_sha = _sha256_file(target)
        if target_sha != row["image_sha256"]:
            raise ReportingError("Copied auxiliary ROI image failed SHA-256 verification.")
        display_rows.append({**row, "display_path": f"auxiliary_grade/{name}",
                             "display_sha256": target_sha})
    return {"rows": display_rows}


def _auxiliary_grade_section(display: dict[str, Any]) -> str:
    cards = []
    for row in display["rows"]:
        roi = row["roi"]
        rid = escape(row["roi_id"], quote=True)
        slide = escape(row["slide_id"], quote=True)
        image = escape(row["display_path"], quote=True)
        coordinates = row["coordinates"]
        coord_text = (
            f"x={coordinates['x']}, y={coordinates['y']} (level-0 origin); "
            f"width={coordinates['width']}, height={coordinates['height']} "
            f"(pixels at ROI level {coordinates['level']})"
        )
        coords = escape(coord_text, quote=True)
        cards.append(
            '<article class="auxiliary-grade-card" data-role="grade-only-evidence">'
            f'<h3>Auxiliary ROI {row["rank"]}</h3>'
            f'<img src="{image}" alt="Auxiliary grade evidence ROI {row["rank"]}">'
            f'<p>ROI: <code>{rid}</code></p><p>Source slide: <code>{slide}</code></p>'
            f'<p>ROI coordinates: {coords}</p>'
            + ('<p>This ROI also appears in the pipeline diagnostic selection.</p>'
               if row.get("also_diagnostic") else
               '<p>This ROI is not listed in the final diagnostic selection.</p>')
            +
            '<p>Input evidence reserved for the grade-only branch. This display adds no diagnosis or grade.</p>'
            '</article>')
    return (
        '<style>.auxiliary-grade-evidence{margin:2rem 0;padding:1.25rem;border:1px solid #d7deeb;'
        'border-radius:12px}.auxiliary-grade-grid{display:grid;grid-template-columns:'
        'repeat(auto-fit,minmax(260px,1fr));gap:1rem}.auxiliary-grade-card{padding:1rem;'
        'border:1px solid #d7deeb;border-radius:10px}.auxiliary-grade-card img{display:block;'
        'width:100%;height:auto;object-fit:contain}</style>'
        '<section id="auxiliary-grade-evidence" class="auxiliary-grade-evidence" data-role="view-only">'
        '<h2>Full · Auxiliary grade-only evidence</h2>'
        '<p>These are the ROIs returned by the grade-support branch. A later diagnostic selection may also reuse a listed ROI.</p>'
        '<div class="auxiliary-grade-grid">' + "".join(cards) + '</div></section>')


def _frozen_renderers(config: dict[str, Any]):
    engine_root = Path(str(config.get("engine_root") or "")).expanduser().resolve()
    v1_root = Path(str(config.get("v1_root") or "")).expanduser().resolve()
    v2_src = engine_root / "src"
    v1_scripts = v1_root / "scripts"
    if not (v2_src / "pathxai_agent_v2" / "html_viz.py").is_file():
        raise ReportingError("Configured frozen v2 html_viz.py is unavailable.")
    if not (v1_scripts / "render_review_workspace_html.py").is_file():
        raise ReportingError("Configured frozen v1 report renderer is unavailable.")
    sys.path.insert(0, str(v2_src))
    sys.path.insert(0, str(v1_scripts))
    from pathxai_agent_v2 import html_viz
    from render_review_workspace_html import render_html as render_frozen_v1
    return html_viz, render_frozen_v1, v1_root


def _copy_original_text(scientific_dir: Path, view_dir: Path) -> None:
    names = ("pipeline_result.json", "generated_long_report.txt", "generated_long_report_rich.txt")
    for name in names:
        source = scientific_dir / name
        target = view_dir / name
        if source.is_file():
            shutil.copyfile(source, target)
        elif name == "pipeline_result.json":
            raise ReportingError("Scientific case is missing pipeline_result.json.")
        elif name == "generated_long_report.txt":
            raise ReportingError("Scientific case is missing the original generated_long_report.txt.")


def prepare_display_report(request: dict[str, Any]) -> dict[str, Any]:
    config = request.get("config") or {}
    case = request.get("case") or {}
    scientific_dir = Path(str(request.get("scientific_case_dir") or "")).expanduser().resolve()
    view_dir = Path(str(request.get("view_case_dir") or "")).expanduser().resolve()
    if not isinstance(config, dict) or not isinstance(case, dict):
        raise ReportingError("config and case must be JSON objects.")
    if not scientific_dir.is_dir():
        raise ReportingError("Scientific case directory is missing.")
    if view_dir.exists():
        raise ReportingError("Display directory already exists; refusing to overwrite an existing output.")
    if view_dir == scientific_dir or scientific_dir in view_dir.parents:
        raise ReportingError("Display directory must be separate from the scientific case directory.")
    result_path = scientific_dir / "pipeline_result.json"
    if not result_path.is_file():
        raise ReportingError("Scientific case is missing pipeline_result.json.")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReportingError("Scientific pipeline_result.json is unreadable.") from exc
    if not isinstance(result, dict):
        raise ReportingError("Scientific pipeline result must be a JSON object.")
    case_id = str(case.get("case_id") or "").strip()
    result_case_id = str(result.get("case_id") or "").strip()
    if not case_id or not result_case_id or case_id != result_case_id:
        raise ReportingError("Case ID does not match the scientific pipeline result.")
    mode = str(case.get("mode") or "").strip().lower()
    if mode not in {"core", "full"}:
        raise ReportingError("case.mode must be core or full.")
    expected_ablation = CORE_ABLATION if mode == "core" else FULL_ABLATION
    result_ablation = result.get("ablation")
    if result_ablation and str(result_ablation) != expected_ablation:
        raise ReportingError("Case mode does not match the scientific result ablation.")
    legacy_result_mode = str(result.get("mode") or "").strip().lower()
    if legacy_result_mode in {"core", "full"} and legacy_result_mode != mode:
        raise ReportingError("Case mode does not match the legacy scientific result mode.")

    auxiliary_rows: list[dict[str, Any]] = []
    auxiliary_receipt: dict[str, Any] | None = None
    if mode == "full":
        auxiliary_rows, auxiliary_receipt = _validate_auxiliary_grade_rows(
            scientific_dir, case_id, result, case)

    view_dir.mkdir(parents=True, exist_ok=True)
    _copy_original_text(scientific_dir, view_dir)
    overviews = build_wsi_overviews(result, case, view_dir)
    auxiliary_display = _copy_auxiliary_grade_images(auxiliary_rows, view_dir)
    if auxiliary_receipt is not None:
        auxiliary_display["checkpoint"] = auxiliary_receipt
        (view_dir / "auxiliary_grade_display.json").write_text(
            json.dumps(auxiliary_display, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8")

    html_viz, frozen_v1_render, v1_root = _frozen_renderers(config)
    old_v1_src = os.environ.get("V1_SRC_ROOT")
    os.environ["V1_SRC_ROOT"] = str(v1_root / "src")
    try:
        payload = html_viz._build_payload(scientific_dir, result)
    finally:
        if old_v1_src is None:
            os.environ.pop("V1_SRC_ROOT", None)
        else:
            os.environ["V1_SRC_ROOT"] = old_v1_src
    payload["_case_dir"] = str(view_dir)
    payload["cancer_type"] = ""
    payload["_files"]["artifact_index"]["cancer_type"] = ""
    payload["wsi_overviews"] = overviews

    product_name = product_name_for_mode(mode)
    escaped_case_id = escape(case_id, quote=True)
    mode_badge = "Full" if mode == "full" else ""
    report_path = view_dir / "report.html"
    rendered = frozen_v1_render(payload, report_path)
    rendered = rendered.replace(
        f"<title>PathXAI Case Review - {escaped_case_id}</title>",
        f"<title>{product_name} - {escaped_case_id}</title>",
        1,
    )
    rendered = rendered.replace(
        '<div class="case-title">Pathology Case Review</div>',
        (f'<div class="case-title">{product_name} '
         f'<span class="report-mode-badge">{mode_badge}</span></div>' if mode_badge else
         f'<div class="case-title">{product_name}</div>'),
        1,
    )
    if mode_badge:
        rendered = rendered.replace(
            "</style>",
            ".report-mode-badge{display:inline-block;vertical-align:middle;margin-left:8px;"
            "padding:4px 9px;border-radius:999px;background:#edf3ff;color:#3154a6;"
            "font-size:.68em;font-weight:800;letter-spacing:.03em}</style>",
            1,
        )
    if auxiliary_display["rows"]:
        closing = rendered.lower().rfind("</body>")
        if closing < 0:
            raise ReportingError("Frozen report renderer did not produce a body element.")
        section = _auxiliary_grade_section(auxiliary_display)
        rendered = rendered[:closing] + section + rendered[closing:]
    report_path.write_text(rendered, encoding="utf-8")
    return {
        "report_path": str(report_path),
        "mode": mode,
        "product_name": product_name,
        "mode_badge": mode_badge,
        "wsi_count": len(overviews),
        "figure_count": len(payload.get("figures") or []),
        "auxiliary_grade_card_count": len(auxiliary_display["rows"]),
        "auxiliary_grade_checkpoint": auxiliary_receipt,
    }


def main() -> int:
    try:
        request = json.load(sys.stdin)
        result = prepare_display_report(request)
        print(json.dumps({"ok": True, **result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
