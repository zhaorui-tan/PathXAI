"""
External evaluation datasets for train_medsam_FINAL.py.

These loaders support the public dataset folders used for external validation
and also keep a simple prepared-folder fallback.  For fallback data, put each
external dataset under:

    external_eval_datasets/<DatasetName>/
        images/
        masks/
        mapping.json        # optional

Image and mask filenames are matched by stem.  ``mapping.json`` may override
the defaults below, for example:

{
  "magnification": 20.0,
  "tissue_type": 3,
  "binary_foreground_class": 6,
  "mask_label_map": {"0": 0, "1": 6, "255": 6}
}
"""

from __future__ import annotations

import json
import os
import random
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw

from all_datasets import BaseDataset


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
_MASK_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


EXTERNAL_DATASET_DEFAULTS: Dict[str, dict] = {
    # Lung cancer/tumor region segmentation.
    "ACDC_LungHP": {
        "aliases": ("ACDCLungHP", "ACDC@LungHP", "LungHP"),
        "magnification": 20.0,
        "tissue_type": 2,               # Lung
        "binary_foreground_class": 1,   # Neoplastic/tumor
    },
    # PAIP 2019 liver viable tumor / tumor-region segmentation.
    "PAIP2019": {
        "aliases": ("PAIP_2019", "PAIP", "PILP"),
        "magnification": 20.0,
        "tissue_type": 6,               # Liver
        "binary_foreground_class": 1,   # Neoplastic/tumor
    },
    # RINGS colorectal patches, local TEST split, manual tumor masks.
    "RINGS_Tumor": {
        "aliases": ("RINGS", "RINGS_TUMOR", "RINGS-Tumor"),
        "magnification": 20.0,
        "tissue_type": 3,               # Colon
        "binary_foreground_class": 1,   # Neoplastic/tumor
    },
    # RINGS colorectal patches, local TEST split, manual gland masks.
    "RINGS_Gland": {
        "aliases": ("RINGS_GLAND", "RINGS-Gland"),
        "magnification": 20.0,
        "tissue_type": 3,               # Colon
        "binary_foreground_class": 6,   # Gland
    },
    # Prepared-folder addon datasets under /local/datasets/public/external_eval_addons.
    "METU_CCTGS": {
        "aliases": ("METU", "CCTGS"),
        "magnification": 20.0,
        "tissue_type": 3,               # Colon
        "binary_foreground_class": 1,   # Default tumor/grade foreground
    },
    "CAMELYON16": {
        "aliases": ("Camelyon16", "CAMELYON"),
        "magnification": 20.0,
        "tissue_type": 1,               # Breast / lymph node metastasis context
        "binary_foreground_class": 1,   # Metastasis/tumor
    },
    "CRAG": {
        "aliases": ("CRAG_Gland",),
        "magnification": 20.0,
        "tissue_type": 3,               # Colon
        "binary_foreground_class": 6,   # Gland
    },
    "TIGER": {
        "aliases": ("TiGER", "TIGER_Breast"),
        "magnification": 20.0,
        "tissue_type": 1,               # Breast
        "binary_foreground_class": 1,   # Default tumor foreground
    },
    "TIGER_Tile": {
        "aliases": ("TIGER-Tile", "TIGER_tile", "TIGERTile"),
        "magnification": 20.0,
        "tissue_type": 1,               # Breast
        "binary_foreground_class": 1,   # Tumor foreground
    },
}


def _iter_files(folder: Path, exts: Iterable[str], recursive: bool = False) -> list[Path]:
    files: list[Path] = []
    pattern_prefix = "**/" if recursive else ""
    for ext in exts:
        files.extend(folder.glob(f"{pattern_prefix}*{ext}"))
        files.extend(folder.glob(f"{pattern_prefix}*{ext.upper()}"))
    return sorted(set(files))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _normalise_dataset_name(name: str) -> Optional[str]:
    if name in EXTERNAL_DATASET_DEFAULTS:
        return name
    low = name.lower()
    for canonical, cfg in EXTERNAL_DATASET_DEFAULTS.items():
        if low == canonical.lower():
            return canonical
        if any(low == a.lower() for a in cfg.get("aliases", ())):
            return canonical
    return None


def _scaled_size(width: int, height: int, max_long_side: int) -> tuple[int, int]:
    if max_long_side <= 0 or max(width, height) <= max_long_side:
        return width, height
    scale = float(max_long_side) / float(max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def _load_resized_image(path: str | os.PathLike, mode: str,
                        size: tuple[int, int],
                        resample=Image.BILINEAR) -> Image.Image:
    if mode == "RGB":
        try:
            import openslide
            slide = openslide.OpenSlide(str(path))
            try:
                im = slide.get_thumbnail(size).convert("RGB")
            finally:
                slide.close()
            if im.size != size:
                im = im.resize(size, resample)
            return im
        except Exception:
            pass
    im = Image.open(path).convert(mode)
    if im.size != size:
        im = im.resize(size, resample)
    return im


def _rasterize_asap_xml(xml_path: str | os.PathLike,
                        size: tuple[int, int],
                        orig_size: tuple[int, int],
                        include_terms: tuple[str, ...]) -> np.ndarray:
    sx = float(size[0]) / float(orig_size[0])
    sy = float(size[1]) / float(orig_size[1])
    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    try:
        tree = ET.parse(xml_path)
        for ann in tree.getroot().findall(".//Annotation"):
            group = (ann.get("PartOfGroup") or ann.get("Name") or "").lower()
            if include_terms and not any(term in group for term in include_terms):
                continue
            pts = []
            for c in ann.findall(".//Coordinate"):
                x = c.get("X")
                y = c.get("Y")
                if x is not None and y is not None:
                    pts.append((float(x) * sx, float(y) * sy))
            if len(pts) >= 3:
                draw.polygon(pts, fill=1)
    except Exception:
        pass
    return np.array(canvas, dtype=np.int64)


class ExternalFolderSegDataset(BaseDataset):
    """Generic external pixel-segmentation dataset with unified tissue labels."""

    def __init__(self, root_dir: str | os.PathLike, dataset_name: str):
        canonical = _normalise_dataset_name(dataset_name) or dataset_name
        defaults = dict(EXTERNAL_DATASET_DEFAULTS.get(canonical, {}))
        mapping = _load_json(Path(root_dir) / "mapping.json")
        cfg = {**defaults, **mapping}

        super().__init__(
            str(root_dir),
            magnification=float(cfg.get("magnification", 20.0)),
            tissue_type=int(cfg.get("tissue_type", 0)),
        )
        self.dataset_name = canonical
        self.binary_foreground_class = int(cfg.get("binary_foreground_class", 1))
        self.ignore_values = {int(v) for v in cfg.get("ignore_values", [])}
        self.mask_label_map = {
            int(k): int(v) for k, v in (cfg.get("mask_label_map") or {}).items()
        }
        self.color_label_map = {
            tuple(int(x) for x in str(k).replace(";", ",").split(",")): int(v)
            for k, v in (cfg.get("color_label_map") or {}).items()
        }

        img_dir = Path(root_dir) / "images"
        mask_dir = Path(root_dir) / "masks"
        images = _iter_files(img_dir, _IMAGE_EXTS)
        masks = _iter_files(mask_dir, _MASK_EXTS)
        mask_by_stem = {p.stem: p for p in masks}

        for img_path in images:
            mask_path = mask_by_stem.get(img_path.stem)
            if mask_path is not None:
                self.image_paths.append(str(img_path))
                self.mask_paths.append(str(mask_path))

    def _load_tissue_mask(self, idx):
        mask_img = Image.open(self.mask_paths[idx])
        if mask_img.mode in ("RGB", "RGBA") and self.color_label_map:
            rgb = np.array(mask_img.convert("RGB"), dtype=np.uint8)
            out = np.zeros(rgb.shape[:2], dtype=np.int64)
            for color, target in self.color_label_map.items():
                if len(color) != 3:
                    continue
                hit = np.all(rgb == np.array(color, dtype=np.uint8), axis=2)
                out[hit] = target
            return out

        m = np.array(mask_img.convert("L"), dtype=np.int64)
        out = np.zeros_like(m, dtype=np.int64)
        if self.mask_label_map:
            for raw, target in self.mask_label_map.items():
                out[m == raw] = target
        else:
            out[m > 0] = self.binary_foreground_class
        for raw in self.ignore_values:
            out[m == raw] = -1
        return out


class ACDCLungHPPublicDataset(BaseDataset):
    """ACDC@LungHP public extracted patch dataset.

    Local layout:
      ACDC_LungHP/{wsi,mask}/{wsi,mask}/*.jpg
      ACDC_LungHP/val_data/{wsi,mask}/{wsi,mask}/*.jpg

    Filenames match after CP/MP replacement, e.g.
      8_CP_60256.jpg ↔ 8_MP_60256.jpg.
    """

    def __init__(self, root_dir: str | os.PathLike, split: str = "all",
                 magnification: float = 20.0):
        super().__init__(str(root_dir), magnification=magnification,
                         tissue_type=2)  # Lung
        self.dataset_name = "ACDC_LungHP"
        root = Path(root_dir)
        parts = []
        if split in ("train", "all"):
            parts.append(root)
        if split in ("val", "all"):
            parts.append(root / "val_data")
        for base in parts:
            img_dir = base / "wsi" / "wsi"
            mask_dir = base / "mask" / "mask"
            images = _iter_files(img_dir, (".jpg", ".jpeg"))
            mask_by_name = {
                p.name.replace("_MP_", "_CP_"): p
                for p in _iter_files(mask_dir, (".jpg", ".jpeg"))
            }
            for img_path in images:
                mask_path = mask_by_name.get(img_path.name)
                if mask_path is not None:
                    self.image_paths.append(str(img_path))
                    self.mask_paths.append(str(mask_path))

    def _load_tissue_mask(self, idx):
        m = np.array(Image.open(self.mask_paths[idx]).convert("L"), dtype=np.uint8)
        return (m > 127).astype(np.int64) * 1  # cancer tissue -> Neoplastic


class PAIP2019PublicDataset(BaseDataset):
    """PAIP 2019 public extracted tile dataset.

    Local layout:
      PAIP_2019/<slide>/tiles_files/0/*.jpeg
      PAIP_2019/<slide>/seg_files/0/*.jpeg

    Segmentation values >0 are treated as neoplastic/viable tumor.  The local
    masks contain values such as 1 and 2; both are target tumor-region labels
    for the current broad tissue taxonomy.
    """

    def __init__(self, root_dir: str | os.PathLike,
                 split: str = "all",
                 magnification: float = 20.0):
        super().__init__(str(root_dir), magnification=magnification,
                         tissue_type=6)  # Liver
        self.dataset_name = "PAIP2019"
        root = Path(root_dir)
        slide_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        if split == "official_test":
            split_file = root / "official_test_slides.txt"
            if split_file.exists():
                wanted = {
                    line.strip() for line in split_file.read_text().splitlines()
                    if line.strip() and not line.strip().startswith("#")
                }
                slide_dirs = [p for p in slide_dirs if p.name in wanted]
            else:
                print("[external_eval] PAIP2019: no official_test_slides.txt found; "
                      "skipping PAIP official-test eval to avoid evaluating all tiles.")
                slide_dirs = []
        elif split != "all":
            raise ValueError(f"Unknown PAIP2019 split: {split!r}")

        for slide_dir in slide_dirs:
            img_dir = slide_dir / "tiles_files" / "0"
            mask_dir = slide_dir / "seg_files" / "0"
            if not img_dir.is_dir() or not mask_dir.is_dir():
                continue
            mask_by_name = {p.name: p for p in _iter_files(mask_dir, (".jpeg", ".jpg"))}
            for img_path in _iter_files(img_dir, (".jpeg", ".jpg")):
                mask_path = mask_by_name.get(img_path.name)
                if mask_path is not None:
                    self.image_paths.append(str(img_path))
                    self.mask_paths.append(str(mask_path))

    def _load_tissue_mask(self, idx):
        m = np.array(Image.open(self.mask_paths[idx]).convert("L"), dtype=np.uint8)
        return (m > 0).astype(np.int64) * 1  # viable/tumor region -> Neoplastic


class RINGSPublicDataset(BaseDataset):
    """RINGS colorectal patch dataset with manual tumor or gland masks.

    Local layout:
      RINGS/{TRAIN,TEST}/IMAGES/*.png
      RINGS/{TRAIN,TEST}/MANUAL TUMOR/*.png
      RINGS/{TRAIN,TEST}/MANUAL GLANDS/*.png

    RINGS is kept external-only here; the default split is TEST.
    """

    def __init__(self, root_dir: str | os.PathLike,
                 target: str = "tumor",
                 split: str = "test",
                 magnification: float = 20.0):
        target = str(target).lower()
        if target not in ("tumor", "gland"):
            raise ValueError(f"Unknown RINGS target: {target!r}")
        if split.lower() not in ("train", "test", "all"):
            raise ValueError(f"Unknown RINGS split: {split!r}")

        foreground_class = 1 if target == "tumor" else 6
        super().__init__(str(root_dir), magnification=magnification,
                         tissue_type=3)  # Colon
        self.dataset_name = "RINGS_Tumor" if target == "tumor" else "RINGS_Gland"
        self.foreground_class = foreground_class

        root = Path(root_dir)
        split_names = []
        if split.lower() in ("train", "all"):
            split_names.append("TRAIN")
        if split.lower() in ("test", "all"):
            split_names.append("TEST")

        mask_folder = "MANUAL TUMOR" if target == "tumor" else "MANUAL GLANDS"
        for split_name in split_names:
            img_dir = root / split_name / "IMAGES"
            mask_dir = root / split_name / mask_folder
            if not img_dir.is_dir() or not mask_dir.is_dir():
                continue
            mask_by_name = {p.name: p for p in _iter_files(mask_dir, _MASK_EXTS)}
            for img_path in _iter_files(img_dir, _IMAGE_EXTS):
                mask_path = mask_by_name.get(img_path.name)
                if mask_path is not None:
                    self.image_paths.append(str(img_path))
                    self.mask_paths.append(str(mask_path))

    def _load_tissue_mask(self, idx):
        m = np.array(Image.open(self.mask_paths[idx]).convert("L"), dtype=np.uint8)
        return (m > 0).astype(np.int64) * self.foreground_class


class METUCCTGSPublicDataset(BaseDataset):
    """METU_CCTGS official split folders."""

    def __init__(self, root_dir: str | os.PathLike,
                 split: str = "validation",
                 magnification: float = 20.0):
        super().__init__(str(root_dir), magnification=magnification,
                         tissue_type=3)  # Colon
        self.dataset_name = "METU_CCTGS"
        root = Path(root_dir)
        if (root / "metucctgs_ds").is_dir():
            root = root / "metucctgs_ds"

        img_dir = root / "images" / split
        mask_dir = root / "annotations" / split
        if not img_dir.is_dir() or not mask_dir.is_dir():
            return
        mask_by_name = {p.name: p for p in _iter_files(mask_dir, _MASK_EXTS)}
        for img_path in _iter_files(img_dir, _IMAGE_EXTS):
            mask_path = mask_by_name.get(img_path.name)
            if mask_path is not None:
                self.image_paths.append(str(img_path))
                self.mask_paths.append(str(mask_path))

    def _load_tissue_mask(self, idx):
        m = np.array(Image.open(self.mask_paths[idx]).convert("L"), dtype=np.int64)
        return (m > 0).astype(np.int64) * 1


class CAMELYON16XMLDataset(BaseDataset):
    """CAMELYON16 test WSIs with ASAP XML tumor annotations."""

    def __init__(self, root_dir: str | os.PathLike,
                 split_prefix: str = "test",
                 max_long_side: int = 4096):
        super().__init__(str(root_dir), magnification=40.0,
                         tissue_type=1)  # Breast / lymph node context
        self.dataset_name = "CAMELYON16"
        self.max_long_side = int(max_long_side)
        root = Path(root_dir)
        ds_root = root / "CAMELYON16" if (root / "CAMELYON16").is_dir() else root
        img_dir = ds_root / "images"
        ann_dir = ds_root / "annotations"
        if not img_dir.is_dir() or not ann_dir.is_dir():
            return

        ann_by_stem = {p.stem: p for p in _iter_files(ann_dir, (".xml",))}
        for img_path in _iter_files(img_dir, (".tif", ".tiff")):
            if split_prefix and not img_path.stem.startswith(f"{split_prefix}_"):
                continue
            ann_path = ann_by_stem.get(img_path.stem)
            if ann_path is not None:
                self.image_paths.append(str(img_path))
                self.mask_paths.append(str(ann_path))

    def _target_size(self, idx) -> tuple[int, int]:
        with Image.open(self.image_paths[idx]) as im:
            return _scaled_size(im.size[0], im.size[1], self.max_long_side)

    def _load_image(self, idx) -> np.ndarray:
        size = self._target_size(idx)
        return np.array(
            _load_resized_image(self.image_paths[idx], "RGB", size, Image.BILINEAR),
            dtype=np.uint8)

    def _load_tissue_mask(self, idx):
        size = self._target_size(idx)
        with Image.open(self.image_paths[idx]) as im:
            orig_size = im.size
        return _rasterize_asap_xml(
            self.mask_paths[idx], size, orig_size, ("tumor", "metast"))


class TIGERTilePublicDataset(BaseDataset):
    """TIGER wsibulk images paired with tumor-region masks."""

    def __init__(self, root_dir: str | os.PathLike,
                 magnification: float = 40.0,
                 max_long_side: int = 4096):
        super().__init__(str(root_dir), magnification=magnification,
                         tissue_type=1)  # Breast
        self.dataset_name = "TIGER_Tile"
        self.max_long_side = int(max_long_side)
        root = Path(root_dir)

        img_dir = root / "wsibulk" / "images"
        ann_dir = root / "wsibulk" / "annotations-tumor-bulk" / "xmls"
        if not img_dir.is_dir() or not ann_dir.is_dir():
            return
        ann_by_stem = {p.stem: p for p in _iter_files(ann_dir, (".xml",))}
        for img_path in _iter_files(img_dir, _IMAGE_EXTS):
            ann_path = ann_by_stem.get(img_path.stem)
            if ann_path is not None:
                self.image_paths.append(str(img_path))
                self.mask_paths.append(str(ann_path))

    def _target_size(self, idx) -> tuple[int, int]:
        with Image.open(self.image_paths[idx]) as im:
            return _scaled_size(im.size[0], im.size[1], self.max_long_side)

    def _load_image(self, idx) -> np.ndarray:
        size = self._target_size(idx)
        return np.array(
            _load_resized_image(self.image_paths[idx], "RGB", size, Image.BILINEAR),
            dtype=np.uint8)

    def _load_tissue_mask(self, idx):
        size = self._target_size(idx)
        with Image.open(self.image_paths[idx]) as im:
            orig_size = im.size
        return _rasterize_asap_xml(
            self.mask_paths[idx], size, orig_size, ("tumor", "bulk"))


class _SubsetDataset(BaseDataset):
    """Lightweight deterministic subset wrapper preserving BaseDataset API."""

    def __init__(self, base: BaseDataset, indices: list[int]):
        super().__init__(base.root_dir, base.magnification, base.tissue_type)
        self.base = base
        self.indices = list(indices)
        self.image_paths = [base.image_paths[i] for i in self.indices]
        self.mask_paths = [base.mask_paths[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base[self.indices[idx]]

    def _load_tissue_mask(self, idx):
        return self.base._load_tissue_mask(self.indices[idx])


def _maybe_subset(ds: BaseDataset, max_samples: Optional[int],
                  seed: int = 42,
                  subset_strategy: str = "random",
                  cache_dir: Optional[str | os.PathLike] = None,
                  dataset_name: Optional[str] = None,
                  stratify_pool_multiplier: int = 10) -> BaseDataset:
    if max_samples is None or max_samples <= 0 or len(ds) <= max_samples:
        return ds

    if subset_strategy == "random":
        rng = random.Random(seed)
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        idx = sorted(idx[:max_samples])
        return _SubsetDataset(ds, idx)

    if subset_strategy != "label_area":
        raise ValueError(
            f"Unknown external subset_strategy={subset_strategy!r}. "
            "Expected 'random' or 'label_area'."
        )

    idx = _label_area_stratified_indices(
        ds, max_samples=max_samples, seed=seed,
        cache_dir=cache_dir, dataset_name=dataset_name,
        stratify_pool_multiplier=stratify_pool_multiplier)
    return _SubsetDataset(ds, idx)


def _label_area_bin(fg_fraction: float) -> str:
    if fg_fraction <= 0.0:
        return "fg_000"
    if fg_fraction <= 0.01:
        return "fg_000_001"
    if fg_fraction <= 0.05:
        return "fg_001_005"
    if fg_fraction <= 0.25:
        return "fg_005_025"
    return "fg_025_100"


def _subset_cache_path(ds: BaseDataset, max_samples: int, seed: int,
                       cache_dir: Optional[str | os.PathLike],
                       dataset_name: Optional[str],
                       stratify_pool_multiplier: int) -> Optional[Path]:
    if not cache_dir:
        return None
    name = dataset_name or getattr(ds, "dataset_name", ds.__class__.__name__)
    first = ds.mask_paths[0] if getattr(ds, "mask_paths", None) else ""
    last = ds.mask_paths[-1] if getattr(ds, "mask_paths", None) else ""
    key = "|".join([
        str(name), str(len(ds)), str(max_samples), str(seed),
        str(stratify_pool_multiplier), first, last,
    ])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))
    return Path(cache_dir) / (
        f"{safe_name}_label_area_n{max_samples}_pool{stratify_pool_multiplier}"
        f"_seed{seed}_{digest}.json"
    )


def _foreground_fraction_from_mask(mask: np.ndarray) -> float:
    valid = mask >= 0
    denom = int(valid.sum())
    if denom <= 0:
        return 0.0
    return float(((mask > 0) & valid).sum()) / float(denom)


def _summarise_bins(indices_by_bin: Dict[str, list[int]]) -> Dict[str, int]:
    return {k: len(v) for k, v in sorted(indices_by_bin.items())}


def _allocate_stratified_counts(indices_by_bin: Dict[str, list[int]],
                                max_samples: int) -> Dict[str, int]:
    total = sum(len(v) for v in indices_by_bin.values())
    if total <= 0:
        return {}

    raw = {
        b: (len(v) * float(max_samples) / float(total))
        for b, v in indices_by_bin.items()
    }
    counts = {
        b: min(len(indices_by_bin[b]), int(raw[b]))
        for b in indices_by_bin
    }

    # Keep every non-empty stratum represented when the budget allows it.
    if max_samples >= len(indices_by_bin):
        for b, vals in indices_by_bin.items():
            if vals and counts[b] == 0:
                counts[b] = 1

    while sum(counts.values()) > max_samples:
        candidates = [b for b, c in counts.items() if c > 1]
        if not candidates:
            break
        b = min(candidates, key=lambda x: raw[x] - counts[x])
        counts[b] -= 1

    while sum(counts.values()) < max_samples:
        candidates = [b for b, vals in indices_by_bin.items() if counts[b] < len(vals)]
        if not candidates:
            break
        b = max(candidates, key=lambda x: raw[x] - counts[x])
        counts[b] += 1

    return counts


def _label_area_stratified_indices(ds: BaseDataset, max_samples: int,
                                   seed: int = 42,
                                   cache_dir: Optional[str | os.PathLike] = None,
                                   dataset_name: Optional[str] = None,
                                   stratify_pool_multiplier: int = 10) -> list[int]:
    cache_path = _subset_cache_path(
        ds, max_samples, seed, cache_dir, dataset_name, stratify_pool_multiplier)
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            payload = json.load(f)
        idx = [int(i) for i in payload["indices"]]
        print(
            f"  [external_eval] {dataset_name or getattr(ds, 'dataset_name', 'dataset')}: "
            f"loaded label-area subset cache {cache_path}"
        )
        print(f"  [external_eval] candidate label-area bins: {payload.get('candidate_bins', payload.get('full_bins', {}))}")
        print(f"  [external_eval] sampled label-area bins: {payload.get('sampled_bins', {})}")
        return idx

    pool_size = min(len(ds), max_samples * max(1, int(stratify_pool_multiplier)))
    scan_indices = list(range(len(ds)))
    if pool_size < len(ds):
        rng = random.Random(seed)
        rng.shuffle(scan_indices)
        scan_indices = sorted(scan_indices[:pool_size])

    indices_by_bin: Dict[str, list[int]] = {}
    for i in scan_indices:
        mask = ds._load_tissue_mask(i)
        bin_name = _label_area_bin(_foreground_fraction_from_mask(mask))
        indices_by_bin.setdefault(bin_name, []).append(i)

    rng = random.Random(seed)
    counts = _allocate_stratified_counts(indices_by_bin, max_samples)
    selected: list[int] = []
    sampled_bins: Dict[str, list[int]] = {}
    for bin_name in sorted(indices_by_bin):
        candidates = list(indices_by_bin[bin_name])
        rng.shuffle(candidates)
        chosen = sorted(candidates[:counts.get(bin_name, 0)])
        sampled_bins[bin_name] = chosen
        selected.extend(chosen)

    selected = sorted(selected)
    candidate_summary = _summarise_bins(indices_by_bin)
    sampled_summary = {k: len(v) for k, v in sorted(sampled_bins.items()) if v}
    print(
        f"  [external_eval] {dataset_name or getattr(ds, 'dataset_name', 'dataset')}: "
        f"label-area stratified subset {len(selected)}/{len(ds)} "
        f"(scanned {len(scan_indices)} candidates)"
    )
    print(f"  [external_eval] candidate label-area bins: {candidate_summary}")
    print(f"  [external_eval] sampled label-area bins: {sampled_summary}")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({
                "indices": selected,
                "candidate_bins": candidate_summary,
                "sampled_bins": sampled_summary,
                "seed": seed,
                "max_samples": max_samples,
                "strategy": "label_area",
                "candidate_count": len(scan_indices),
                "stratify_pool_multiplier": stratify_pool_multiplier,
            }, f, indent=2)

    return selected


def build_external_eval_datasets(
    root_dir: str | os.PathLike = "/local/datasets/public/datasets",
    names: Optional[Iterable[str]] = None,
    max_samples_per_dataset: Optional[int] = None,
    max_samples_by_dataset: Optional[Dict[str, int]] = None,
    seed: int = 42,
    subset_strategy: str = "random",
    subset_cache_dir: Optional[str | os.PathLike] = None,
    stratify_pool_multiplier: int = 10,
    acdc_split: str = "val",
    paip_split: str = "all",
    acdc_magnification: float = 20.0,
    paip_magnification: float = 20.0,
) -> Dict[str, ExternalFolderSegDataset]:
    """Build external validation datasets from public roots or fallback folders."""
    root = Path(root_dir)
    wanted = None
    if names:
        wanted = {_normalise_dataset_name(n.strip()) or n.strip()
                  for n in names if str(n).strip()}
    sample_overrides: Dict[str, int] = {}
    if max_samples_by_dataset:
        for raw_name, raw_limit in max_samples_by_dataset.items():
            key = _normalise_dataset_name(str(raw_name).strip()) or str(raw_name).strip()
            sample_overrides[key] = int(raw_limit)

    datasets: Dict[str, ExternalFolderSegDataset] = {}
    candidates = list(EXTERNAL_DATASET_DEFAULTS.keys())
    if wanted is not None:
        candidates = [n for n in candidates if n in wanted]
        candidates.extend(n for n in sorted(wanted) if n not in candidates)

    for name in candidates:
        if name == "ACDC_LungHP":
            ds_root = root / "ACDC_LungHP"
            ds = ACDCLungHPPublicDataset(
                ds_root, split=acdc_split, magnification=acdc_magnification)
        elif name == "PAIP2019":
            ds_root = root / "PAIP_2019"
            ds = PAIP2019PublicDataset(
                ds_root, split=paip_split, magnification=paip_magnification)
        elif name == "RINGS_Tumor":
            ds_root = root / "RINGS"
            ds = RINGSPublicDataset(ds_root, target="tumor", split="test")
        elif name == "RINGS_Gland":
            ds_root = root / "RINGS"
            ds = RINGSPublicDataset(ds_root, target="gland", split="test")
        elif name == "METU_CCTGS":
            ds_root = root / "METU_CCTGS"
            if not ds_root.is_dir():
                ds_root = root
            ds = METUCCTGSPublicDataset(ds_root, split="validation")
        elif name == "CAMELYON16":
            ds_root = root / "CAMELYON16"
            if not ds_root.is_dir():
                ds_root = root
            ds = CAMELYON16XMLDataset(ds_root, split_prefix="test")
        elif name == "TIGER_Tile":
            # TIGER wsibulk images paired with annotations-tumor-bulk masks.
            ds = TIGERTilePublicDataset(root, magnification=40.0)
        else:
            ds_root = root / name
            ds = ExternalFolderSegDataset(ds_root, name)
        this_max_samples = sample_overrides.get(name, max_samples_per_dataset)
        ds = _maybe_subset(
            ds, this_max_samples, seed=seed,
            subset_strategy=subset_strategy,
            cache_dir=subset_cache_dir,
            dataset_name=name,
            stratify_pool_multiplier=stratify_pool_multiplier)
        if len(ds) > 0:
            datasets[name] = ds
        elif wanted is not None and name in wanted:
            print(f"  [external_eval] {name}: no matched images/masks under {ds_root}")
    return datasets
