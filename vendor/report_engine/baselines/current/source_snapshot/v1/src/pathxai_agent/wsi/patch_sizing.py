from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

from .slide_reader import SlideInfo


@dataclass
class PatchSizeDecision:
    patch_size: int
    method: str
    native_crop_px: int
    native_mpp: float
    slide_mpp: Optional[float]
    slide_magnification: Optional[float]
    fallback_patch_size: int


def resolve_magnification_aware_patch_size(
    slide_info: SlideInfo,
    native_crop_px: int = 4096,
    native_mpp: float = 0.25,
    fallback_patch_size: int = 2048,
    min_patch_size: int = 512,
    max_patch_size: int = 4096,
    round_to: int = 128,
) -> PatchSizeDecision:
    """Match a 40x native crop to the slide's physical resolution.

    A 4096px crop at 40x is treated as 4096 * 0.25um ~= 1024um wide.
    For a slide scanned at 20x / 0.5um-per-pixel, the matched pixel size is
    1024um / 0.5um-per-pixel = 2048px.
    """
    slide_mpp = _mean_mpp(slide_info.mpp_x, slide_info.mpp_y)
    if slide_mpp and slide_mpp > 0:
        physical_um = native_crop_px * native_mpp
        raw_size = physical_um / slide_mpp
        method = "mpp"
    elif slide_info.magnification and slide_info.magnification > 0:
        raw_size = native_crop_px * (slide_info.magnification / 40.0)
        method = "magnification"
    else:
        raw_size = fallback_patch_size
        method = "fallback"
    patch_size = _round_clip(raw_size, min_patch_size, max_patch_size, round_to)
    return PatchSizeDecision(
        patch_size=patch_size,
        method=method,
        native_crop_px=native_crop_px,
        native_mpp=native_mpp,
        slide_mpp=slide_mpp,
        slide_magnification=slide_info.magnification,
        fallback_patch_size=fallback_patch_size,
    )


def patch_size_decision_dict(decision: PatchSizeDecision) -> dict:
    return asdict(decision)


def _mean_mpp(mpp_x: Optional[float], mpp_y: Optional[float]) -> Optional[float]:
    vals = [v for v in (mpp_x, mpp_y) if v is not None and v > 0]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _round_clip(value: float, min_value: int, max_value: int, round_to: int) -> int:
    clipped = max(min_value, min(max_value, int(round(value))))
    if round_to > 1:
        clipped = int(round(clipped / round_to) * round_to)
    return max(min_value, min(max_value, clipped))
