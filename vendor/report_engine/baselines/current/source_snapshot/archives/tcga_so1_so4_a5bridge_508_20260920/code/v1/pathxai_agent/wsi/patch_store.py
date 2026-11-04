from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .roi_candidates import RoiCandidate
from .slide_reader import SlideReader


def cache_roi_crops(slide_path: str, slide_id: str, rois: Iterable[RoiCandidate], out_dir: Path) -> List[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = SlideReader(slide_path)
    saved = []
    try:
        for roi in rois:
            image = reader.read_region(roi.x, roi.y, roi.width, roi.height, roi.level)
            image_path = out_dir / f"{roi.roi_id}.png"
            image.save(image_path)
            item = {
                "slide_id": slide_id,
                "roi_id": roi.roi_id,
                "image_path": str(image_path),
                "x": roi.x,
                "y": roi.y,
                "width": roi.width,
                "height": roi.height,
                "level": roi.level,
                "source": roi.source,
                "score": roi.score,
            }
            saved.append(item)
    finally:
        reader.close()

    with (out_dir / "rois.json").open("w", encoding="utf-8") as f:
        json.dump(saved, f, indent=2, ensure_ascii=False)
    return saved
