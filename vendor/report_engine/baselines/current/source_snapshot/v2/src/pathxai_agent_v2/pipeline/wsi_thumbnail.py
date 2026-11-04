"""WSI thumbnail generation + ROI bounding-box annotation for PathXAI v2.

Generates two images per case and writes them to the case output dir:
  wsi_thumbnail_<slide_id>.jpg          — clean WSI overview (shared cache)
  wsi_thumbnail_<slide_id>_annotated.jpg — same + colored ROI boxes + badges

The annotated image is what the HTML report displays in the Evidence tab.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# ── Colour palette (matches v1) ───────────────────────────────────────────────
_PALETTE_RGB = [
    (59, 130, 246),   # blue
    (239, 68, 68),    # red
    (34, 197, 94),    # green
    (245, 158, 11),   # amber
    (168, 85, 247),   # purple
    (20, 184, 166),   # teal
    (236, 72, 153),   # pink
    (251, 146, 60),   # orange
]

_THUMB_CACHE_ROOT = Path("/local/PathXAI_Agent/shared_artifacts/wsi_thumbnails")


def _cache_key(slide_path: str | Path) -> str:
    p = Path(str(slide_path))
    stat_text = ""
    try:
        st = p.stat()
        stat_text = f"|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        pass
    return hashlib.sha1(
        (str(p.resolve() if p.exists() else p) + stat_text).encode()
    ).hexdigest()[:20]


def _get_or_generate_clean_thumb(
    slide_path: str | Path,
    max_size: int = 1024,
) -> tuple[Path | None, int, int]:
    """Return (thumb_path, slide_w, slide_h).  thumb_path is None on failure."""
    slide_path = Path(str(slide_path))
    if not slide_path.is_file():
        return None, 0, 0

    cache_dir = _THUMB_CACHE_ROOT / _cache_key(slide_path)
    thumb_path = cache_dir / f"thumbnail_max{max_size}.jpg"
    dims_path = cache_dir / "slide_dims.json"

    # Read cached dims
    slide_w = slide_h = 0
    if dims_path.exists():
        try:
            dims = json.loads(dims_path.read_text())
            slide_w, slide_h = int(dims.get("width", 0)), int(dims.get("height", 0))
        except Exception:
            pass

    if thumb_path.exists() and slide_w and slide_h:
        return thumb_path, slide_w, slide_h

    try:
        import openslide  # type: ignore
        slide = openslide.OpenSlide(str(slide_path))
        slide_w, slide_h = slide.dimensions
        cache_dir.mkdir(parents=True, exist_ok=True)
        dims_path.write_text(json.dumps({"width": slide_w, "height": slide_h}))
        thumb = slide.get_thumbnail((max_size, max_size))
        thumb.save(str(thumb_path), format="JPEG", quality=88, optimize=True)
        slide.close()
        return thumb_path, slide_w, slide_h
    except Exception:
        return None, slide_w, slide_h


def _annotate_thumbnail(
    thumb_path: Path,
    rois: list[dict[str, Any]],
    slide_w: int,
    slide_h: int,
    out_path: Path,
) -> Path | None:
    """Draw ROI bounding boxes + numbered badges on the WSI thumbnail.

    Each roi dict may carry a ``_fig_num`` key with the global figure number
    (1-based, matching the HTML report's "Figure N" labelling).  Falls back to
    the per-slide enumerate index if absent.
    """
    if not thumb_path.is_file() or not rois or not slide_w or not slide_h:
        return None
    try:
        from PIL import Image, ImageDraw  # type: ignore

        thumb = Image.open(str(thumb_path)).convert("RGB")
        tw, th = thumb.size
        sx, sy = tw / slide_w, th / slide_h
        draw = ImageDraw.Draw(thumb)
        lw = max(2, tw // 300)

        for i, roi in enumerate(rois):
            x0 = int((roi.get("x") or 0) * sx)
            y0 = int((roi.get("y") or 0) * sy)
            bw = max(lw + 2, int((roi.get("width") or 0) * sx))
            bh = max(lw + 2, int((roi.get("height") or 0) * sy))
            x1, y1 = x0 + bw, y0 + bh
            # Use global figure number (colour palette index 0-based from fig_num)
            fig_num = roi.get("_fig_num") or (i + 1)
            color_idx = (int(fig_num) - 1) % len(_PALETTE_RGB)
            color = _PALETTE_RGB[color_idx]
            for t in range(lw):
                draw.rectangle([x0 + t, y0 + t, x1 - t, y1 - t], outline=color)
            badge = max(10, lw * 5)
            bx0, by0 = x0 + lw, y0 + lw
            draw.rectangle([bx0, by0, bx0 + badge, by0 + badge], fill=color)
            label = str(fig_num)
            tx = bx0 + max(1, (badge - len(label) * 6) // 2)
            ty = by0 + max(1, (badge - 8) // 2)
            draw.text((tx, ty), label, fill=(255, 255, 255))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        thumb.save(str(out_path), format="JPEG", quality=88, optimize=True)
        return out_path
    except Exception:
        return None


def _slide_id_from_roi(roi: dict[str, Any]) -> str:
    """Extract slide_id from roi dict, falling back to parsing roi_id."""
    sid = str(roi.get("slide_id") or "")
    if sid:
        return sid
    roi_id = str(roi.get("roi_id") or "")
    # roi_id format: <slide_id>_foundation_x..._y..._l..._w..._h...
    for sep in ("_foundation_", "_zoom_", "_roi_"):
        if sep in roi_id:
            return roi_id.split(sep)[0]
    return roi_id


def generate_wsi_thumbnails(
    case_dir: Path,
    candidate_rois: list[dict[str, Any]],
    selected_rois: list[dict[str, Any]],
) -> dict[str, str]:
    """Generate clean + annotated WSI thumbnails for all slides referenced by the case.

    Groups ROIs by slide_id, generates one clean thumbnail per slide (shared
    cache), and one annotated thumbnail per slide (written to case_dir).

    Returns a dict mapping slide_id -> annotated_path (or clean_path if annotation failed).
    Only processes slides whose .svs file is accessible.
    """
    roi_cache_root = Path("/local/PathXAI_Agent_v2/shared_artifacts/roi_cache/conch")

    # Collect slide_id → slide_path from foundation_roi_ranking.json
    slide_paths: dict[str, str] = {}
    seen_slide_ids: set[str] = set()
    for roi in list(candidate_rois or []) + list(selected_rois or []):
        sid = _slide_id_from_roi(roi)
        if not sid or sid in seen_slide_ids:
            continue
        seen_slide_ids.add(sid)
        ranking_path = roi_cache_root / sid / "foundation_roi_ranking.json"
        if ranking_path.exists():
            try:
                sp = json.loads(ranking_path.read_text()).get("slide_path", "")
                if sp and Path(sp).is_file():
                    slide_paths[sid] = sp
            except Exception:
                pass

    if not slide_paths:
        return {}

    # Group selected ROIs by slide_id, attaching global figure number to each
    from collections import defaultdict
    selected_by_slide: dict[str, list[dict]] = defaultdict(list)
    for fig_num, roi in enumerate(selected_rois, start=1):
        sid = _slide_id_from_roi(roi)
        if sid:
            selected_by_slide[sid].append({**roi, "_fig_num": fig_num})

    results: dict[str, str] = {}
    for sid, sp in slide_paths.items():
        thumb, w, h = _get_or_generate_clean_thumb(sp)
        if not thumb:
            continue

        sel = selected_by_slide.get(sid, [])
        # Use slide UUID portion only for filename (last UUID-looking segment)
        safe_sid = sid.split(".")[-1] if "." in sid else sid[-40:]

        if sel and w and h:
            ann_path = case_dir / f"wsi_thumbnail_{safe_sid}_annotated.jpg"
            ann = _annotate_thumbnail(thumb, sel, w, h, ann_path)
            results[sid] = str(ann if ann and ann.exists() else thumb)
        # Skip slides with no selected ROIs (don't clutter case_dir)

    return results
