"""Reinspection stage — dynamically adds/zooms ROIs based on evidence gaps.

When evidence is insufficient, plans and executes additional ROI acquisition:
- Zoom: higher-magnification crops of existing ROIs (for cytology/mitotic detail)
- Add: independent new ROIs from candidate pool (for missing features like necrosis/interface)
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pathxai_agent_v2.config import Settings
from pathxai_agent_v2.llm import LLMClient
from pathxai_agent_v2.pipeline.base import (
    DiagnosisPrediction,
    EvidenceSufficiency,
    PipelineStage,
    RoiInfo,
    SitePrediction,
    StageResult,
    ok,
)

# ── Term sets for action classification ───────────────────────────────────────

ZOOM_ACTIONABLE = {
    "high power", "high-power", "cytology", "nuclear", "atypia",
    "mitotic", "mitoses", "cellular", "detail", "small focus", "subtle",
}

ADD_ROI_ACTIONABLE = {
    "necrosis", "interface", "tumor-normal", "tumour-normal",
    "tumor stroma", "tumour stroma", "invasive front",
    "alternative morphology", "heterogeneity", "representative",
    "hemorrhage", "haemorrhage",
}

NONIMAGE_ONLY = {
    "ihc", "immunohistochemistry", "molecular", "genetic", "mutation",
    "ki-67", "gross", "size", "measurement", "margin", "tnm", "stage",
    "clinical", "history", "procedure",
}


@dataclass
class ReinspectionPlan:
    """Output of the reinspection planning stage."""
    action: str = "none"  # zoom_current_roi | add_independent_roi | require_nonimage | none
    new_rois: list[RoiInfo] = field(default_factory=list)
    zoom_rois: list[dict] = field(default_factory=list)
    missing_image: list[str] = field(default_factory=list)
    missing_nonimage: list[str] = field(default_factory=list)
    target_roi_ids: list[str] = field(default_factory=list)
    zoom_error: str = ""
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "new_rois": [r.to_dict() for r in self.new_rois],
            "zoom_rois": self.zoom_rois,
            "missing_image": self.missing_image,
            "missing_nonimage": self.missing_nonimage,
            "target_roi_ids": self.target_roi_ids,
            "zoom_error": self.zoom_error,
            "stop_reason": self.stop_reason,
        }


class ReinspectionStage(PipelineStage[dict, ReinspectionPlan]):
    """Stage: Plan additional ROI acquisition based on evidence gaps."""

    stage_name = "reinspection"

    def run(
        self,
        stage_input: dict,
        dry_run: bool = False,
    ) -> StageResult[ReinspectionPlan]:
        """Plan reinspection.

        stage_input expects:
            candidates: list[RoiInfo]  — remaining unused candidates
            selected: list[RoiInfo]    — currently selected ROIs
            sufficiency: EvidenceSufficiency
            diagnosis: DiagnosisPrediction
            max_extra_rois: int
            max_zoom_depth: int
        """
        candidates: list[RoiInfo] = stage_input.get("candidates", [])
        selected: list[RoiInfo] = stage_input.get("selected", [])
        sufficiency: EvidenceSufficiency | None = stage_input.get("sufficiency")
        diagnosis: DiagnosisPrediction | None = stage_input.get("diagnosis")
        segmentation_context: dict = stage_input.get("segmentation_context") or {}
        max_extra = stage_input.get("max_extra_rois", 2)
        max_zoom = stage_input.get("max_zoom_depth", 1)

        if dry_run or not sufficiency:
            return ok(ReinspectionPlan(action="none"), self.stage_name)

        # Classify gaps
        img_gaps = [g for g in sufficiency.missing_image_evidence if g]
        nonimg_gaps = [g for g in sufficiency.missing_nonimage_metadata if g]

        # Determine action
        zoom_gaps = [g for g in img_gaps if any(t in g.lower() for t in ZOOM_ACTIONABLE)]
        add_gaps = [g for g in img_gaps if any(t in g.lower() for t in ADD_ROI_ACTIONABLE)]

        if zoom_gaps and max_zoom > 0:
            action = "zoom_current_roi"
            zoomed_parent_ids = {
                str((r.metadata or {}).get("zoom_parent_roi_id"))
                for r in selected
                if (r.metadata or {}).get("zoom_parent_roi_id")
            }
            # Prefer the original high-value ROIs, but never zoom the same
            # parent twice across reinspection rounds. Repeated zoom crops add
            # latency and make the evidence graph look larger without adding
            # new visual evidence.
            target_ids = [
                r.roi_id
                for r in selected
                if r.roi_id not in zoomed_parent_ids and r.source != "zoom"
            ][:3]
            if not target_ids:
                add_gaps = img_gaps
            else:
                zoom_rois = self._create_zoom_rois(selected, target_ids, max_zoom, max_extra, segmentation_context)
                if zoom_rois:
                    return ok(
                        ReinspectionPlan(
                            action=action,
                            new_rois=[RoiInfo.from_candidate(z) for z in zoom_rois],
                            zoom_rois=zoom_rois,
                            missing_image=img_gaps,
                            missing_nonimage=nonimg_gaps,
                            target_roi_ids=target_ids,
                        ),
                        self.stage_name,
                    )
                # Cached foundation ROIs may not carry slide_path metadata, making
                # true zoom crops impossible. In that case, keep gathering evidence
                # by adding independent unused ROIs rather than stopping early.
                add_gaps = img_gaps

        if add_gaps or img_gaps:
            action = "add_independent_roi"
            # Select spatially diverse unused candidates
            used_ids = {r.roi_id for r in selected}
            available = [c for c in candidates if c.roi_id not in used_ids]
            if not available:
                return ok(
                    ReinspectionPlan(
                        action="none",
                        stop_reason="candidate pool exhausted",
                        missing_image=img_gaps,
                        missing_nonimage=nonimg_gaps,
                    ),
                    self.stage_name,
                )
            new = _select_diverse(available, max_extra)
            return ok(
                ReinspectionPlan(
                    action=action,
                    new_rois=new,
                    missing_image=img_gaps,
                    missing_nonimage=nonimg_gaps,
                    zoom_error="zoom requested but unavailable; fell back to independent ROI selection" if zoom_gaps else "",
                ),
                self.stage_name,
            )

        if nonimg_gaps and not img_gaps:
            return ok(
                ReinspectionPlan(
                    action="require_nonimage_metadata",
                    missing_image=img_gaps,
                    missing_nonimage=nonimg_gaps,
                    stop_reason="gaps are non-image metadata only",
                ),
                self.stage_name,
            )

        return ok(ReinspectionPlan(action="none"), self.stage_name)

    def _create_zoom_rois(
        self,
        selected: list[RoiInfo],
        target_ids: list[str],
        max_zoom_depth: int,
        max_count: int,
        segmentation_context: dict | None = None,
    ) -> list[dict]:
        """Create zoomed-in crops from parent ROIs."""
        zoom_rois = []
        try:
            import sys
            v1_src = str(Path(self.settings.paths.v1_src_root))
            if v1_src not in sys.path:
                sys.path.insert(0, v1_src)
            from pathxai_agent.wsi.slide_reader import SlideReader
        except ImportError:
            return zoom_rois

        zoom_dir = self.out_dir / "zoom_rois"
        zoom_dir.mkdir(parents=True, exist_ok=True)
        seg_by_roi = {
            str(r.get("roi_id")): r
            for r in ((segmentation_context or {}).get("roi_results") or [])
            if r.get("roi_id")
        }

        for roi in selected:
            if len(zoom_rois) >= max_count:
                break
            if roi.roi_id not in target_ids:
                continue

            slide_path = roi.metadata.get("slide_path", "")

            try:
                # Center 50% crop
                zw = max(512, (roi.width or 4096) // 2)
                zh = max(512, (roi.height or 4096) // 2)
                dx = max(0, ((roi.width or 4096) - zw) // 2)
                dy = max(0, ((roi.height or 4096) - zh) // 2)
                cx = (roi.x or 0) + dx
                cy = (roi.y or 0) + dy

                zoom_id = f"{roi.roi_id}_zoom{len(zoom_rois)+1}_center_x{cx}_y{cy}_w{zw}_h{zh}"
                zoom_path = zoom_dir / f"{zoom_id}.png"

                if not zoom_path.exists():
                    cropped = False
                    if slide_path and Path(slide_path).exists():
                        try:
                            reader = SlideReader(slide_path)
                            # SlideReader.read_region signature is
                            # (x, y, width, height, level=0).
                            region = reader.read_region(cx, cy, zw, zh, roi.level or 0)
                            region.save(str(zoom_path))
                            cropped = True
                        except Exception:
                            cropped = False
                    if not cropped:
                        _crop_zoom_from_roi_image(roi, dx, dy, zw, zh, zoom_path)

                parent_seg = seg_by_roi.get(roi.roi_id)
                inherited_seg = None
                if parent_seg:
                    inherited_seg = _crop_inherited_segmentation(
                        parent_seg=parent_seg,
                        zoom_roi_id=zoom_id,
                        zoom_image_path=zoom_path,
                        crop_box=(dx, dy, dx + zw, dy + zh),
                        out_dir=zoom_dir / f"{zoom_id}_seg",
                    )

                zoom_rois.append({
                    "roi_id": zoom_id,
                    "image_path": str(zoom_path),
                    "x": cx, "y": cy,
                    "width": zw, "height": zh,
                    "level": roi.level or 0,
                    "source": "zoom",
                    "score": roi.score * 0.9,
                    "metadata": {
                        "zoom_depth": 1,
                        "zoom_parent_roi_id": roi.roi_id,
                        "zoom_region": "center",
                        "slide_path": slide_path,
                        "crop_source": "wsi" if slide_path and Path(slide_path).exists() else "roi_image",
                        "roi_crop_box": [dx, dy, dx + zw, dy + zh],
                        "inherited_segmentation": inherited_seg or {},
                    },
                })
            except Exception:
                continue

        return zoom_rois


def _crop_zoom_from_roi_image(
    roi: RoiInfo,
    dx: int,
    dy: int,
    zw: int,
    zh: int,
    zoom_path: Path,
) -> None:
    """Create a zoom crop directly from the cached ROI image.

    This is the robust fallback for shared ROI caches that do not carry the
    source WSI path.  The crop keeps absolute x/y in the ROI metadata for WSI
    visualization, while the saved image is simply the local ROI subcrop.
    """
    from PIL import Image  # type: ignore

    src = Path(roi.image_path)
    if not src.exists():
        raise FileNotFoundError(f"parent ROI image not found: {src}")
    zoom_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(str(src)).convert("RGB")
    iw, ih = img.size
    # Map WSI-pixel crop to image-pixel crop.  Usually both are 4096, but keep
    # this scale-safe for resized ROI caches.
    rw = max(1, roi.width or iw)
    rh = max(1, roi.height or ih)
    sx, sy = iw / rw, ih / rh
    box = (
        max(0, int(dx * sx)),
        max(0, int(dy * sy)),
        min(iw, int((dx + zw) * sx)),
        min(ih, int((dy + zh) * sy)),
    )
    crop = img.crop(box)
    crop.save(str(zoom_path))


def _crop_inherited_segmentation(
    parent_seg: dict,
    zoom_roi_id: str,
    zoom_image_path: Path,
    crop_box: tuple[int, int, int, int],
    out_dir: Path,
) -> dict | None:
    """Crop parent ROI mask/overlay to create zoom segmentation evidence.

    This avoids rerunning MedSAM for zooms.  The inherited mask is appropriate
    because the zoom is a strict subcrop of a ROI that has already been
    segmented.
    """
    try:
        from PIL import Image  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    mask_path = Path(str(parent_seg.get("mask_path") or ""))
    overlay_path = Path(str(parent_seg.get("overlay_path") or ""))
    if not mask_path.exists() or not overlay_path.exists():
        return None

    x0, y0, x1, y1 = crop_box
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mask = out_dir / "mask.png"
    out_overlay = out_dir / "overlay.jpg"
    out_composite = out_dir / "composite.jpg"

    mask_img = Image.open(str(mask_path))
    overlay_img = Image.open(str(overlay_path)).convert("RGB")
    he_img = Image.open(str(zoom_image_path)).convert("RGB")

    # Parent mask/overlay are in parent ROI image pixel coordinates.
    crop_box_px = (
        max(0, x0),
        max(0, y0),
        min(mask_img.size[0], x1),
        min(mask_img.size[1], y1),
    )
    mask_crop = mask_img.crop(crop_box_px)
    overlay_crop = overlay_img.crop(crop_box_px)
    mask_crop.save(str(out_mask))
    overlay_crop.save(str(out_overlay), quality=90)

    _save_pair_composite(he_img, overlay_crop, out_composite)

    arr = np.array(mask_crop)
    total = int(arr.size) if arr.size else 0
    counts = Counter(int(x) for x in arr.reshape(-1)) if total else Counter()
    named_parent = ((parent_seg.get("stats") or {}).get("named_class_fractions") or {})
    class_names = {
        int(info.get("class_id")): name
        for name, info in named_parent.items()
        if isinstance(info, dict) and info.get("class_id") is not None
    }
    class_pixel_counts = {str(k): int(v) for k, v in sorted(counts.items())}
    class_fractions = {str(k): (float(v) / total if total else 0.0) for k, v in sorted(counts.items())}
    named_class_fractions = {}
    for class_id, pixels in sorted(counts.items()):
        name = class_names.get(class_id, f"Class {class_id}")
        named_class_fractions[name] = {
            "class_id": class_id,
            "pixels": int(pixels),
            "fraction": float(pixels) / total if total else 0.0,
        }

    return {
        "roi_index": parent_seg.get("roi_index"),
        "roi_id": zoom_roi_id,
        "image_path": str(zoom_image_path),
        "mask_path": str(out_mask),
        "overlay_path": str(out_overlay),
        "composite_path": str(out_composite),
        "width": he_img.size[0],
        "height": he_img.size[1],
        "inherited_from_roi_id": parent_seg.get("roi_id"),
        "inherited_from_parent_mask": True,
        "stats": {
            "total_pixels": total,
            "foreground_fraction": 1.0 - class_fractions.get("0", 0.0),
            "class_pixel_counts": class_pixel_counts,
            "class_fractions": class_fractions,
            "named_class_fractions": named_class_fractions,
        },
    }


def _save_pair_composite(he_img, overlay_img, out_path: Path) -> None:
    from PIL import Image  # type: ignore

    def resize_panel(img, max_side: int = 1024):
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        return img

    he = resize_panel(he_img)
    ov = resize_panel(overlay_img)
    h = max(he.size[1], ov.size[1])
    w = he.size[0] + ov.size[0]
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(he, (0, 0))
    canvas.paste(ov, (he.size[0], 0))
    canvas.save(str(out_path), quality=88)


def _select_diverse(candidates: list[RoiInfo], n: int) -> list[RoiInfo]:
    """Select spatially diverse candidates from the pool."""
    if len(candidates) <= n:
        return list(candidates)

    # Prefer high-score, then spatially diverse
    sorted_candidates = sorted(candidates, key=lambda r: r.score, reverse=True)
    selected = [sorted_candidates[0]]
    used = {selected[0].roi_id}

    for _ in range(n - 1):
        best = None
        best_dist = -1
        for c in sorted_candidates:
            if c.roi_id in used:
                continue
            if c.x is None or c.y is None:
                if best is None:
                    best = c
                continue
            # Min distance to any selected ROI
            min_dist = min(
                ((c.x - s.x) ** 2 + (c.y - s.y) ** 2) ** 0.5
                for s in selected
                if s.x is not None and s.y is not None
            )
            if min_dist > best_dist:
                best_dist = min_dist
                best = c
        if best:
            selected.append(best)
            used.add(best.roi_id)
        else:
            break

    return selected
