"""Build ROI + segmentation composite images and global legend for LLM visual input.

Cache: shared_artifacts/seg_roi_pairs/<cache_key>/
  composite.jpg    – left H&E | right overlay (no embedded legend)
  legend.jpg       – global tissue class color legend (one per checkpoint)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_CACHE_ROOT: Path = Path("/local/PathXAI_Agent/shared_artifacts/seg_roi_pairs")

# ── Tissue colors MUST match _save_overlay() palette in medsam_roi.py ──
TISSUE_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),        # Background
    1: (220, 38, 38),    # Neoplastic
    2: (37, 99, 235),    # Inflammatory
    3: (22, 163, 74),    # Connective
    4: (234, 179, 8),    # Epithelial
    5: (147, 51, 234),   # Dead
    6: (14, 165, 233),   # Gland
}

TISSUE_NAMES: dict[int, str] = {
    1: "Neoplastic", 2: "Inflammatory", 3: "Connective",
    4: "Epithelial", 5: "Dead", 6: "Gland",
}


def _composite_cache_key(h_and_e_path: str | Path, overlay_path: str | Path) -> str:
    payload = {
        "h_and_e_path": str(Path(h_and_e_path).resolve()),
        "h_and_e_mtime": Path(h_and_e_path).stat().st_mtime,
        "overlay_path": str(Path(overlay_path).resolve()),
        "overlay_mtime": Path(overlay_path).stat().st_mtime,
        "version": "seg_composite_v4",
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_roi_seg_composite(
    h_and_e_path: str | Path,
    overlay_path: str | Path,
    max_edge: int = 1024,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
) -> dict[str, Any]:
    """Left H&E | right overlay with labels. Cached.

    Returns:
        {"composite_path": str, "cache_key": str, "cache_hit": bool}
    """
    cache_root = Path(cache_root)
    cache_key = _composite_cache_key(h_and_e_path, overlay_path)
    cache_dir = cache_root / cache_key
    composite_path = cache_dir / "composite.jpg"

    if composite_path.exists():
        return {"composite_path": str(composite_path), "cache_key": cache_key, "cache_hit": True}

    cache_dir.mkdir(parents=True, exist_ok=True)

    h_and_e = Image.open(h_and_e_path).convert("RGB")
    overlay = Image.open(overlay_path).convert("RGB")

    max_edge_val = max_edge
    target_h = min(max(h_and_e.height, overlay.height), max_edge_val)

    def _resize(img: Image.Image) -> Image.Image:
        w, h = img.size
        new_h = target_h
        new_w = max(1, int(w * target_h / h))
        if new_w > max_edge_val:
            new_w = max_edge_val
            new_h = max(1, int(h * max_edge_val / w))
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    h_and_e = _resize(h_and_e)
    overlay = _resize(overlay)

    label_h = 28
    total_w = h_and_e.width + overlay.width
    total_h = max(h_and_e.height, overlay.height) + label_h
    composite = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    # Labels
    draw = ImageDraw.Draw(composite)
    draw.rectangle([0, 0, h_and_e.width, label_h], fill=(248, 250, 252))
    draw.rectangle([h_and_e.width, 0, total_w, label_h], fill=(239, 246, 255))
    draw.text((8, 7), "H&E ROI", fill=(15, 23, 42))
    draw.text((h_and_e.width + 8, 7), "Segmentation", fill=(15, 23, 42))
    # Images
    composite.paste(h_and_e, (0, label_h))
    composite.paste(overlay, (h_and_e.width, label_h))
    draw.line([(h_and_e.width, label_h), (h_and_e.width, total_h)], fill=(220, 220, 220), width=2)

    composite.save(composite_path, format="JPEG", quality=85, optimize=True)
    return {"composite_path": str(composite_path), "cache_key": cache_key, "cache_hit": False}


def build_global_seg_legend(
    class_names: dict[int, str] | None = None,
    ckpt_path: str | Path = "",
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
) -> str:
    """Generate one global tissue class color legend image per checkpoint. Cached."""
    cnames = class_names or {cid: TISSUE_NAMES.get(cid, f"Class{cid}") for cid in sorted(TISSUE_COLORS_RGB)}
    ckpt = str(Path(ckpt_path).name) if ckpt_path else "default"
    key = hashlib.sha1(f"global_legend_v2:{ckpt}:{sorted(cnames.items())}".encode()).hexdigest()
    cache_dir = Path(cache_root) / "_legend"
    legend_path = cache_dir / f"{key}.jpg"

    if legend_path.exists():
        return str(legend_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    box_w, box_h = 30, 30
    gap = 12
    text_w_per = 110
    n_items = len([c for c in sorted(cnames) if int(c) != 0])
    total_w = n_items * (box_w + gap + text_w_per) + gap
    total_h = box_h + 16

    legend = Image.new("RGB", (total_w, total_h), (248, 250, 252))
    draw = ImageDraw.Draw(legend)
    x = gap
    for cid in sorted(cnames):
        if int(cid) == 0:
            continue
        color = TISSUE_COLORS_RGB.get(int(cid), (128, 128, 128))
        name = str(cnames.get(cid, f"C{cid}"))
        draw.rectangle([x, 8, x + box_w, 8 + box_h], fill=color, outline=(180, 180, 180))
        draw.text((x + box_w + 6, 10), name, fill=(15, 23, 42))
        x += box_w + gap + text_w_per
    legend.save(legend_path, format="JPEG", quality=95, optimize=True)
    return str(legend_path)


def build_roi_seg_composite_from_arrays(
    h_and_e_arr: np.ndarray,
    overlay_arr: np.ndarray,
    max_edge: int = 1024,
    cache_dir: Path | None = None,
) -> tuple[Path, str]:
    """In-memory version: build composite from numpy arrays.

    Format matches build_roi_seg_composite: left H&E | right overlay with labels.
    """
    h_and_e = Image.fromarray(h_and_e_arr.astype(np.uint8)).convert("RGB")
    overlay = Image.fromarray(overlay_arr.astype(np.uint8)).convert("RGB")
    key_data = h_and_e_arr.tobytes()[:4096] + overlay_arr.tobytes()[:4096]
    cache_key = hashlib.sha1(key_data).hexdigest()
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_ROOT / cache_key
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    composite_path = cache_dir / "composite.jpg"
    if composite_path.exists():
        return composite_path, cache_key

    max_edge_val = max_edge
    target_h = min(max(h_and_e.height, overlay.height), max_edge_val)

    def _resize(img: Image.Image) -> Image.Image:
        w, h = img.size
        new_h = target_h
        new_w = max(1, int(w * target_h / h))
        if new_w > max_edge_val:
            new_w = max_edge_val
            new_h = max(1, int(h * max_edge_val / w))
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    h_and_e = _resize(h_and_e)
    overlay = _resize(overlay)

    label_h = 28
    total_w = h_and_e.width + overlay.width
    total_h = max(h_and_e.height, overlay.height) + label_h
    composite = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    # Labels (same as build_roi_seg_composite)
    draw = ImageDraw.Draw(composite)
    draw.rectangle([0, 0, h_and_e.width, label_h], fill=(248, 250, 252))
    draw.rectangle([h_and_e.width, 0, total_w, label_h], fill=(239, 246, 255))
    draw.text((8, 7), "H&E ROI", fill=(15, 23, 42))
    draw.text((h_and_e.width + 8, 7), "Segmentation", fill=(15, 23, 42))
    # Images
    composite.paste(h_and_e, (0, label_h))
    composite.paste(overlay, (h_and_e.width, label_h))
    draw.line([(h_and_e.width, label_h), (h_and_e.width, total_h)], fill=(220, 220, 220), width=2)
    composite.save(composite_path, format="JPEG", quality=85, optimize=True)
    return composite_path, cache_key


def crop_seg_for_zoom(
    parent_h_and_e_path: str | Path,
    parent_mask_path: str | Path,
    zoom_x: int, zoom_y: int, zoom_w: int, zoom_h: int,
    cache_root: str | Path,
) -> dict[str, Any]:
    """Crop parent seg mask to match a zoom ROI. No re-segmentation.

    Cache is per-run (cache_root should be the run's out_dir/zoom_seg_cache).
    Zoom seg is fast (crop only, no model inference), so cache is kept within
    the run for resume support, NOT in shared_artifacts.
    """
    cache_root = Path(cache_root)
    mask_path = Path(parent_mask_path)
    h_and_e_path = Path(parent_h_and_e_path)
    payload = {
        "parent_h_and_e": str(h_and_e_path.resolve()),
        "parent_h_and_e_mtime": h_and_e_path.stat().st_mtime,
        "parent_mask": str(mask_path.resolve()),
        "parent_mtime": mask_path.stat().st_mtime,
        "zoom_x": int(zoom_x), "zoom_y": int(zoom_y),
        "zoom_w": int(zoom_w), "zoom_h": int(zoom_h),
        "version": "seg_zoom_crop_v4",
    }
    cache_key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    cache_dir = cache_root / cache_key
    overlay_path = cache_dir / "overlay.jpg"
    composite_path = cache_dir / "composite.jpg"
    if composite_path.exists():
        return {"overlay_path": str(overlay_path), "composite_path": str(composite_path),
                "cache_key": cache_key, "cache_hit": True}
    cache_dir.mkdir(parents=True, exist_ok=True)

    mask_img = Image.open(mask_path)
    if mask_img.mode != "L":
        mask_img = mask_img.convert("L")
    mask_np = np.array(mask_img)
    mask_crop = mask_np[zoom_y:zoom_y + zoom_h, zoom_x:zoom_x + zoom_w]
    palette = np.array([[c[0], c[1], c[2]] for _, c in sorted(TISSUE_COLORS_RGB.items())], dtype=np.float32)
    # Alpha-blend overlay: H&E background + transparent mask colors (matching MedSAM _save_overlay)
    try:
        he = Image.open(h_and_e_path).convert("RGB")
        he_crop = he.crop((zoom_x, zoom_y, zoom_x + zoom_w, zoom_y + zoom_h))
    except Exception:
        he_crop = Image.fromarray(np.zeros((zoom_h, zoom_w, 3), dtype=np.uint8) + 128)
    base = np.array(he_crop).astype(np.float32)
    colors = np.zeros_like(base)
    for cid_idx, (cid, _) in enumerate(sorted(TISSUE_COLORS_RGB.items())):
        if cid_idx < len(palette):
            colors[mask_crop == int(cid)] = palette[cid_idx]
    alpha = (mask_crop > 0).astype(np.float32)[..., None] * 0.45
    overlay_arr = (base * (1 - alpha) + colors * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay_arr).save(overlay_path, format="JPEG", quality=85, optimize=True)
    cp, ck = build_roi_seg_composite_from_arrays(np.array(he_crop), overlay_arr, cache_dir=cache_dir)
    return {"overlay_path": str(overlay_path), "composite_path": str(cp), "cache_key": ck, "cache_hit": False}


def attach_composites_to_seg_context(
    segmentation_context: dict,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
) -> dict:
    """Add composite_path to each roi_result + generate global legend."""
    if not segmentation_context.get("available"):
        return segmentation_context
    roi_results = segmentation_context.get("roi_results") or []
    for r in roi_results:
        if r.get("composite_path"):
            continue
        h_and_e = r.get("image_path")
        overlay = r.get("overlay_path_file") or r.get("overlay_path")
        if h_and_e and overlay:
            try:
                result = build_roi_seg_composite(h_and_e, overlay, cache_root=cache_root)
                r["composite_path"] = result["composite_path"]
            except Exception:
                pass
    if not segmentation_context.get("legend_path"):
        class_names = segmentation_context.get("class_names") or {}
        ckpt = segmentation_context.get("checkpoint") or ""
        try:
            segmentation_context["legend_path"] = build_global_seg_legend(
                class_names=class_names, ckpt_path=ckpt, cache_root=cache_root,
            )
        except Exception:
            pass
    return segmentation_context


def get_overlay_path_for_roi(row: dict) -> str | None:
    """Get the overlay path for an ROI result dict."""
    return row.get("overlay_path_file") or row.get("overlay_path")


def build_image_paths_for_rois(
    selected_rois: list[dict],
    segmentation_context: dict,
    seg_visual_mode: str,
) -> list[str]:
    """Build image_paths for LLM calls based on seg_visual_mode.

    composite: [legend, pair_1, pair_2, ...] where pair = H&E|overlay side-by-side
    text_only / none: [H&E_1, H&E_2, ...]
    """
    if seg_visual_mode in ("none", "text_only"):
        return [row["image_path"] for row in selected_rois]

    # composite mode: global legend first, then composite pair per ROI
    paths: list[str] = []
    try:
        legend = build_global_seg_legend()
        if legend and Path(legend).exists():
            paths.append(legend)
    except Exception:
        pass

    roi_results = segmentation_context.get("roi_results") or []
    overlay_by_id: dict[str, str] = {}
    for r in roi_results:
        rid = r.get("roi_id")
        ov = get_overlay_path_for_roi(r)
        if rid and ov and Path(ov).exists():
            overlay_by_id[str(rid)] = ov

    for row in selected_rois:
        he = row.get("image_path")
        rid = row.get("roi_id")
        ov = overlay_by_id.get(str(rid))
        if he and ov and Path(he).exists():
            # Build composite pair: H&E | overlay, cached
            result = build_roi_seg_composite(he, ov)
            paths.append(result["composite_path"])
        elif he and Path(he).exists():
            # Fallback: just H&E if no overlay
            paths.append(he)

    return paths
