from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, List


@dataclass
class RoiCandidate:
    slide_id: str
    roi_id: str
    x: int
    y: int
    width: int
    height: int
    level: int = 0
    source: str = "grid"
    score: float = 0.0


def grid_candidates(
    slide_id: str,
    width: int,
    height: int,
    patch_size: int = 4096,
    stride: int | None = None,
    max_rois: int = 32,
) -> List[RoiCandidate]:
    stride = stride or patch_size
    candidates: List[RoiCandidate] = []
    for y in range(0, max(1, height), stride):
        for x in range(0, max(1, width), stride):
            x0 = min(x, max(0, width - patch_size))
            y0 = min(y, max(0, height - patch_size))
            roi_id = f"{slide_id}_x{x0}_y{y0}_l0_w{patch_size}_h{patch_size}"
            candidates.append(RoiCandidate(slide_id, roi_id, x0, y0, patch_size, patch_size))
            if len(candidates) >= max_rois:
                return _dedupe(candidates)
    return _dedupe(candidates)


def center_candidates(
    slide_id: str,
    width: int,
    height: int,
    patch_size: int = 4096,
    max_rois: int = 5,
) -> List[RoiCandidate]:
    centers = [
        (0.5, 0.5),
        (0.33, 0.33),
        (0.67, 0.33),
        (0.33, 0.67),
        (0.67, 0.67),
    ][:max_rois]
    candidates = []
    for cx, cy in centers:
        x0 = int(max(0, min(width - patch_size, width * cx - patch_size / 2)))
        y0 = int(max(0, min(height - patch_size, height * cy - patch_size / 2)))
        roi_id = f"{slide_id}_x{x0}_y{y0}_l0_w{patch_size}_h{patch_size}"
        candidates.append(RoiCandidate(slide_id, roi_id, x0, y0, patch_size, patch_size, source="center"))
    return _dedupe(candidates)


def _dedupe(candidates: Iterable[RoiCandidate]) -> List[RoiCandidate]:
    seen = set()
    out = []
    for cand in candidates:
        key = (cand.x, cand.y, cand.width, cand.height, cand.level)
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
    return out


def roi_dicts(candidates: Iterable[RoiCandidate]) -> List[dict]:
    return [asdict(c) for c in candidates]
