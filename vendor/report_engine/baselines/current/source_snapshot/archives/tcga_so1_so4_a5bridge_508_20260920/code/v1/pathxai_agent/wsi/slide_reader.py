from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image


@dataclass
class SlideInfo:
    path: str
    width: int
    height: int
    levels: int
    backend: str
    mpp_x: Optional[float] = None
    mpp_y: Optional[float] = None
    magnification: Optional[float] = None


class SlideReader:
    """Small wrapper around WSI readers with a PIL fallback for ordinary images."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._backend = "pil"
        self._slide = None
        self._image = None
        try:
            import openslide  # type: ignore

            self._slide = openslide.OpenSlide(str(self.path))
            self._backend = "openslide"
        except Exception:
            try:
                import tiffslide  # type: ignore

                self._slide = tiffslide.TiffSlide(str(self.path))
                self._backend = "tiffslide"
            except Exception:
                self._image = Image.open(self.path).convert("RGB")

    @property
    def size(self) -> Tuple[int, int]:
        if self._backend in {"openslide", "tiffslide"}:
            return tuple(int(v) for v in self._slide.dimensions)
        return self._image.size

    def info(self) -> SlideInfo:
        width, height = self.size
        if self._backend in {"openslide", "tiffslide"}:
            props = dict(self._slide.properties)
            mpp_x = props.get("openslide.mpp-x")
            mpp_y = props.get("openslide.mpp-y")
            if not mpp_x:
                mpp_x = props.get("tiffslide.mpp-x") or props.get("aperio.MPP")
            if not mpp_y:
                mpp_y = props.get("tiffslide.mpp-y") or props.get("aperio.MPP")
            magnification = (
                props.get("openslide.objective-power")
                or props.get("aperio.AppMag")
                or props.get("tiffslide.objective-power")
            )
            return SlideInfo(
                path=str(self.path),
                width=width,
                height=height,
                levels=int(getattr(self._slide, "level_count", len(getattr(self._slide, "level_dimensions", [self.size])))),
                backend=self._backend,
                mpp_x=float(mpp_x) if mpp_x else None,
                mpp_y=float(mpp_y) if mpp_y else None,
                magnification=float(magnification) if magnification else None,
            )
        return SlideInfo(path=str(self.path), width=width, height=height, levels=1, backend=self._backend)

    def read_region(self, x: int, y: int, width: int, height: int, level: int = 0) -> Image.Image:
        if self._backend in {"openslide", "tiffslide"}:
            return self._slide.read_region((int(x), int(y)), int(level), (int(width), int(height))).convert("RGB")
        crop = self._image.crop((int(x), int(y), int(x + width), int(y + height)))
        if crop.size != (width, height):
            padded = Image.new("RGB", (width, height), "white")
            padded.paste(crop, (0, 0))
            crop = padded
        return crop.convert("RGB")

    def thumbnail(self, max_size: int = 2048) -> Image.Image:
        if self._backend in {"openslide", "tiffslide"}:
            return self._slide.get_thumbnail((max_size, max_size)).convert("RGB")
        im = self._image.copy()
        im.thumbnail((max_size, max_size))
        return im.convert("RGB")

    def close(self) -> None:
        if self._slide is not None:
            self._slide.close()
        if self._image is not None:
            self._image.close()
