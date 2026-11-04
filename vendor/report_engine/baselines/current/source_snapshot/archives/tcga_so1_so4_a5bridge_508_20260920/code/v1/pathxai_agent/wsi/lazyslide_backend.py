from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .roi_candidates import RoiCandidate


class LazySlideUnavailable(RuntimeError):
    pass


def check_lazyslide_available() -> Dict[str, Any]:
    """Return LazySlide availability details without importing at module load."""
    details: Dict[str, Any] = {
        "available": False,
        "lazyslide_version": None,
        "wsidata_available": False,
        "error": None,
    }
    try:
        import lazyslide as zs  # type: ignore

        details["available"] = True
        details["lazyslide_version"] = getattr(zs, "__version__", "unknown")
    except Exception as exc:
        details["error"] = f"lazyslide import failed: {exc}"
        return details
    try:
        import wsidata  # type: ignore  # noqa: F401

        details["wsidata_available"] = True
    except Exception as exc:
        details["error"] = f"wsidata import failed: {exc}"
    return details


def _import_lazyslide():
    details = check_lazyslide_available()
    if not details["available"] or not details["wsidata_available"]:
        raise LazySlideUnavailable(
            "LazySlide backend is not available in this Python environment. "
            f"Details: {details}"
        )
    import lazyslide as zs  # type: ignore
    from wsidata import open_wsi  # type: ignore

    return zs, open_wsi


def _shape_bounds(geometry: Any) -> Tuple[float, float, float, float]:
    bounds = getattr(geometry, "bounds", None)
    if bounds is None:
        raise ValueError(f"Geometry has no bounds: {geometry!r}")
    return tuple(float(v) for v in bounds)


def _candidate_from_bounds(
    slide_id: str,
    bounds: Tuple[float, float, float, float],
    patch_size: int,
    source: str,
    idx: int,
    score: float = 0.0,
) -> RoiCandidate:
    minx, miny, maxx, maxy = bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    x = max(0, int(round(cx - patch_size / 2)))
    y = max(0, int(round(cy - patch_size / 2)))
    roi_id = f"{slide_id}_lazy{idx:04d}_x{x}_y{y}_l0_w{patch_size}_h{patch_size}"
    return RoiCandidate(
        slide_id=slide_id,
        roi_id=roi_id,
        x=x,
        y=y,
        width=patch_size,
        height=patch_size,
        level=0,
        source=source,
        score=score,
    )


def lazyslide_tissue_tile_candidates(
    slide_path: str | Path,
    slide_id: str,
    patch_size: int = 2048,
    mpp: Optional[float] = 0.5,
    max_rois: int = 32,
    overlap: Optional[float] = None,
    stride_px: Optional[int] = None,
    tissue_key: str = "tissues",
    tiles_key: str = "tiles",
    write_store: bool = False,
) -> Tuple[List[RoiCandidate], Dict[str, Any]]:
    """Create ROI candidates from LazySlide tissue segmentation and tiling.

    LazySlide owns the tissue finding and tile geometry. This function converts
    those geometries into the project's plain ROI contract.
    """
    zs, open_wsi = _import_lazyslide()
    slide_path = Path(slide_path)
    wsi = open_wsi(str(slide_path))

    metadata: Dict[str, Any] = {
        "backend": "lazyslide",
        "slide_path": str(slide_path),
        "tissue_key": tissue_key,
        "tiles_key": tiles_key,
        "patch_size": patch_size,
        "mpp": mpp,
        "overlap": overlap,
        "stride_px": stride_px,
    }
    props = getattr(wsi, "properties", None)
    if props is not None:
        metadata["properties"] = _jsonable_properties(props)

    if tissue_key not in _wsi_keys(wsi):
        zs.pp.find_tissues(wsi, key_added=tissue_key)

    tile_kwargs: Dict[str, Any] = {"key_added": tiles_key}
    if mpp is not None:
        tile_kwargs["mpp"] = mpp
    if overlap is not None:
        tile_kwargs["overlap"] = overlap
    if stride_px is not None:
        tile_kwargs["stride_px"] = stride_px
    if tissue_key != "tissues":
        tile_kwargs["tissue_key"] = tissue_key

    try:
        zs.pp.tile_tissues(wsi, patch_size, **tile_kwargs)
    except TypeError:
        # Older/newer LazySlide versions may not expose tissue_key in the same
        # way. Retry with default tissue key before failing.
        tile_kwargs.pop("tissue_key", None)
        zs.pp.tile_tissues(wsi, patch_size, **tile_kwargs)

    tiles = wsi[tiles_key]
    metadata["tile_count"] = int(len(tiles))
    candidates: List[RoiCandidate] = []
    for idx, row in enumerate(_iter_rows(tiles)):
        if len(candidates) >= max_rois:
            break
        geometry = row.get("geometry")
        if geometry is None:
            continue
        bounds = _shape_bounds(geometry)
        area = float(getattr(geometry, "area", 0.0) or 0.0)
        candidates.append(
            _candidate_from_bounds(
                slide_id=slide_id,
                bounds=bounds,
                patch_size=patch_size,
                source="lazyslide_tissue_tile",
                idx=idx,
                score=area,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[:max_rois]
    metadata["roi_count"] = len(candidates)
    metadata["rois"] = [asdict(c) for c in candidates]
    if write_store and hasattr(wsi, "write"):
        wsi.write()
        metadata["wrote_wsidata_store"] = True
    return candidates, metadata


def _wsi_keys(wsi: Any) -> set[str]:
    keys = set()
    for attr in ("keys",):
        fn = getattr(wsi, attr, None)
        if callable(fn):
            try:
                keys.update(str(k) for k in fn())
            except Exception:
                pass
    for attr in ("shapes", "tables"):
        value = getattr(wsi, attr, None)
        if isinstance(value, dict):
            keys.update(str(k) for k in value.keys())
    return keys


def _iter_rows(table: Any):
    if hasattr(table, "iterrows"):
        for _, row in table.iterrows():
            if hasattr(row, "to_dict"):
                yield row.to_dict()
            else:
                yield dict(row)
        return
    if isinstance(table, list):
        for row in table:
            yield row
        return
    raise TypeError(f"Unsupported LazySlide tile table type: {type(table)!r}")


def _jsonable_properties(props: Any) -> Dict[str, Any]:
    try:
        items = props.items()
    except Exception:
        return {}
    out: Dict[str, Any] = {}
    for key, value in items:
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = value
        else:
            out[str(key)] = str(value)
    return out
