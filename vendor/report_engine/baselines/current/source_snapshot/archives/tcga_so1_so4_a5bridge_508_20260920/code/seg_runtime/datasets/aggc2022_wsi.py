"""AGGC2022 lazy WSI discovery, window sampling, and mask reading."""

from __future__ import annotations

import csv
import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

import openslide
import tifffile


EXPECTED_TEST_IMAGES = {1: 45, 2: 16, 3: 67}
MASK_NAMES = ("G3_Mask.tif", "G4_Mask.tif", "G5_Mask.tif")
ANNOTATION_NAMES = MASK_NAMES + ("Normal_Mask.tif", "Stroma_Mask.tif")


@dataclass(frozen=True)
class AGGCCase:
    subset: int
    specimen_id: str
    scanner: str
    slide_id: str
    image_path: Path
    annotation_dir: Path


def _specimen_id(stem: str) -> str:
    match = re.match(r"(Subset\d+_(?:Train|Test)_\d+)", stem)
    return match.group(1) if match else stem


def collect_test_cases(root: Path, require_complete: bool = True) -> list[AGGCCase]:
    root = Path(root)
    cases: list[AGGCCase] = []
    for subset in (1, 2):
        image_dir = root / f"subset_{subset}_image_train_test_rename_tiff" / "Test"
        ann_root = root / f"subset_{subset}_annotation_train_test_rename" / "Test"
        for image_path in sorted(image_dir.glob("*.tiff")):
            ann_dir = ann_root / image_path.stem
            if ann_dir.is_dir():
                cases.append(AGGCCase(
                    subset, _specimen_id(image_path.stem), "Akoya",
                    image_path.stem, image_path, ann_dir))

    image_root = root / "subset_3_scanner_train_test_rename_tiff" / "Test"
    ann_root = root / "subset_3_scanner_annotation_train_test_rename" / "Test"
    for image_path in sorted(image_root.glob("*/*.tiff")):
        scanner = image_path.parent.name
        ann_dir = ann_root / scanner / image_path.stem
        if ann_dir.is_dir():
            cases.append(AGGCCase(
                3, _specimen_id(image_path.stem), scanner,
                image_path.stem, image_path, ann_dir))

    counts = {subset: sum(c.subset == subset for c in cases) for subset in (1, 2, 3)}
    if require_complete and counts != EXPECTED_TEST_IMAGES:
        raise RuntimeError(
            f"AGGC2022 Test is incomplete: found {counts}, expected "
            f"{EXPECTED_TEST_IMAGES}. Finish downloading before evaluation.")
    return cases


class LazyMask:
    """Window reader for either OpenSlide-compatible or stripped TIFF masks."""

    def __init__(self, path: Path, cache_strips: int = 0):
        self.path = Path(path)
        self._slide = None
        self._tiff = None
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_strips = cache_strips
        try:
            self._slide = openslide.OpenSlide(str(path))
            self.dimensions = self._slide.dimensions
        except openslide.OpenSlideError:
            self._tiff = tifffile.TiffFile(path)
            self._page = self._tiff.pages[0]
            self.dimensions = (int(self._page.shape[1]), int(self._page.shape[0]))

    def _strip(self, index: int) -> np.ndarray:
        cached = self._cache.pop(index, None)
        if cached is not None:
            self._cache[index] = cached
            return cached
        page = self._page
        handle = self._tiff.filehandle
        handle.seek(page.dataoffsets[index])
        data = handle.read(page.databytecounts[index])
        # AGGC binary masks use WhiteIsZero: black is background and white is
        # the annotated region. tifffile exposes the packed sample bits here,
        # so invert them to return a conventional foreground=True mask.
        decoded = ~page.decode(data, index)[0][0, :, :, 0].astype(bool, copy=False)
        self._cache[index] = decoded
        while len(self._cache) > self._cache_strips:
            self._cache.popitem(last=False)
        return decoded

    def read(self, x: int, y: int, w: int, h: int,
             out_size: int) -> np.ndarray:
        if self._slide is not None:
            image = self._slide.read_region((x, y), 0, (w, h)).convert("L")
            if image.size != (out_size, out_size):
                image = image.resize(
                    (out_size, out_size), Image.Resampling.NEAREST)
            return np.asarray(image, dtype=np.uint8) > 0

        rows_per_strip = int(self._page.rowsperstrip)
        out = np.zeros((h, w), dtype=bool)
        first = y // rows_per_strip
        last = (y + h - 1) // rows_per_strip
        for strip_index in range(first, last + 1):
            strip = self._strip(strip_index)
            strip_y = strip_index * rows_per_strip
            src_y0 = max(y, strip_y) - strip_y
            src_y1 = min(y + h, strip_y + strip.shape[0]) - strip_y
            dst_y0 = max(y, strip_y) - y
            dst_y1 = dst_y0 + (src_y1 - src_y0)
            out[dst_y0:dst_y1] = strip[src_y0:src_y1, x:x + w]
        if out.shape != (out_size, out_size):
            out = np.asarray(
                Image.fromarray(out).resize(
                    (out_size, out_size), Image.Resampling.NEAREST),
                dtype=bool)
        return out

    def sample_foreground_centers(self, rng: random.Random,
                                  count: int) -> list[tuple[int, int]]:
        if self._tiff is None:
            return []
        page = self._page
        # Blank CCITT strips have the smallest compressed byte counts. Decode
        # likely non-empty strips only, then inspect a sparse 32-pixel grid.
        minimum = min(page.databytecounts)
        candidates = [
            i for i, size in enumerate(page.databytecounts)
            if size > minimum * 1.05
        ]
        rng.shuffle(candidates)
        centers: list[tuple[int, int]] = []
        for strip_index in candidates:
            strip = self._strip(strip_index)
            points = np.argwhere(strip[::32, ::32])
            if points.size == 0:
                continue
            order = list(range(len(points)))
            rng.shuffle(order)
            for point_index in order[:max(1, count - len(centers))]:
                py, px = points[point_index]
                centers.append((int(px) * 32, strip_index * int(page.rowsperstrip) + int(py) * 32))
                if len(centers) >= count:
                    return centers
        return centers

    def close(self) -> None:
        self._cache.clear()
        if self._slide is not None:
            self._slide.close()
        if self._tiff is not None:
            self._tiff.close()


def _open_masks(case: AGGCCase) -> dict[str, LazyMask]:
    return {
        name: LazyMask(case.annotation_dir / name)
        for name in ANNOTATION_NAMES if (case.annotation_dir / name).is_file()
    }


def _read_binary(slide: LazyMask, x: int, y: int,
                 w: int, h: int, out_size: int) -> np.ndarray:
    return slide.read(x, y, w, h, out_size)


def read_tumor_mask(case: AGGCCase, x: int, y: int, w: int, h: int,
                    out_size: int) -> np.ndarray:
    masks = _open_masks(case)
    try:
        tumor = np.zeros((out_size, out_size), dtype=bool)
        for name in MASK_NAMES:
            if name in masks:
                tumor |= _read_binary(masks[name], x, y, w, h, out_size)
        return tumor.astype(np.int64)
    finally:
        for slide in masks.values():
            slide.close()


def _window_label(masks: dict[str, LazyMask], x: int, y: int,
                  size: int, probe_size: int = 64) -> tuple[bool, float]:
    annotated = np.zeros((probe_size, probe_size), dtype=bool)
    tumor = np.zeros_like(annotated)
    for name, slide in masks.items():
        hit = _read_binary(slide, x, y, size, size, probe_size)
        annotated |= hit
        if name in MASK_NAMES:
            tumor |= hit
    ann_fraction = float(annotated.mean())
    return bool(tumor.any()), ann_fraction


def _rgb_tissue_fraction(slide: openslide.OpenSlide, x: int, y: int,
                         size: int, probe_size: int = 64) -> float:
    probe = slide.read_region((x, y), 0, (size, size)).convert("RGB")
    probe = probe.resize((probe_size, probe_size), Image.Resampling.BILINEAR)
    rgb = np.asarray(probe, dtype=np.int16)
    brightness = rgb.mean(axis=2)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return float(((brightness < 235) & (chroma > 10)).mean())


def build_manifest(root: Path, output_csv: Path, native_crop_at_40x: int = 4096,
                   stride_at_40x: int = 2048, max_windows_per_slide: int = 20,
                   seed: int = 42, require_complete: bool = True,
                   max_cases: int = 0,
                   max_cases_per_subset: int = 0) -> list[dict]:
    cases = collect_test_cases(root, require_complete=require_complete)
    if max_cases_per_subset > 0:
        subset_counts: dict[int, int] = {}
        filtered = []
        for case in cases:
            if subset_counts.get(case.subset, 0) >= max_cases_per_subset:
                continue
            filtered.append(case)
            subset_counts[case.subset] = subset_counts.get(case.subset, 0) + 1
        cases = filtered
    if max_cases > 0:
        cases = cases[:max_cases]
    rows: list[dict] = []
    native_mag = 20.0
    read_size = int(round(native_crop_at_40x * native_mag / 40.0))
    stride = int(round(stride_at_40x * native_mag / 40.0))

    for case_index, case in enumerate(cases):
        print(f"[AGGC manifest] {case_index + 1}/{len(cases)} {case.slide_id}", flush=True)
        image = openslide.OpenSlide(str(case.image_path))
        masks = {
            name: LazyMask(case.annotation_dir / name)
            for name in MASK_NAMES if (case.annotation_dir / name).is_file()
        }
        try:
            width, height = image.dimensions
            for name, mask in masks.items():
                if mask.dimensions != (width, height):
                    raise RuntimeError(
                        f"Dimension mismatch for {case.slide_id}/{name}: "
                        f"image={(width, height)} mask={mask.dimensions}")
            xs = list(range(0, max(1, width - read_size + 1), stride))
            ys = list(range(0, max(1, height - read_size + 1), stride))
            if not xs or xs[-1] != max(0, width - read_size):
                xs.append(max(0, width - read_size))
            if not ys or ys[-1] != max(0, height - read_size):
                ys.append(max(0, height - read_size))
            rng = random.Random(seed + case_index)
            target_pos = (max_windows_per_slide + 1) // 2
            target_neg = max_windows_per_slide // 2
            positive: list[tuple[int, int, float, bool]] = []
            seen = set()
            for mask in masks.values():
                for center_x, center_y in mask.sample_foreground_centers(rng, target_pos * 2):
                    x = min(max(0, center_x - read_size // 2), max(0, width - read_size))
                    y = min(max(0, center_y - read_size // 2), max(0, height - read_size))
                    key = (x, y)
                    if key in seen:
                        continue
                    seen.add(key)
                    tissue_fraction = _rgb_tissue_fraction(image, x, y, read_size)
                    positive.append((x, y, tissue_fraction, True))
                    if len(positive) >= target_pos:
                        break
                if len(positive) >= target_pos:
                    break

            candidates = [(x, y) for y in ys for x in xs]
            rng.shuffle(candidates)
            negative: list[tuple[int, int, float, bool]] = []
            for x, y in candidates:
                if (x, y) in seen:
                    continue
                tissue_fraction = _rgb_tissue_fraction(image, x, y, read_size)
                if tissue_fraction < 0.05:
                    continue
                is_positive, _ = _window_label(masks, x, y, read_size)
                if is_positive and len(positive) < target_pos:
                    positive.append((x, y, tissue_fraction, True))
                elif not is_positive:
                    negative.append((x, y, tissue_fraction, False))
                if len(positive) >= target_pos and len(negative) >= target_neg:
                    break
            selected = (positive[:target_pos] + negative[:target_neg])
            selected.sort(key=lambda item: (item[1], item[0]))
            for window_index, (x, y, tissue_fraction, is_positive) in enumerate(selected):
                rows.append({
                    "dataset": "AGGC2022_Tumor",
                    "subset": case.subset,
                    "specimen_id": case.specimen_id,
                    "scanner": case.scanner,
                    "slide_id": case.slide_id,
                    "image_path": str(case.image_path),
                    "annotation_dir": str(case.annotation_dir),
                    "window_idx": window_index,
                    "x": x, "y": y, "w": read_size, "h": read_size,
                    "native_mag": native_mag,
                    "native_crop_at_40x": native_crop_at_40x,
                    "positive": int(is_positive),
                    "annotated_fraction_probe": f"{tissue_fraction:.6f}",
                })
        finally:
            image.close()
            for mask in masks.values():
                mask.close()

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def load_manifest(path: Path) -> list[dict]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("subset", "window_idx", "x", "y", "w", "h",
                    "native_crop_at_40x", "positive"):
            row[key] = int(row[key])
        row["native_mag"] = float(row["native_mag"])
        row["annotated_fraction_probe"] = float(row["annotated_fraction_probe"])
    return rows
