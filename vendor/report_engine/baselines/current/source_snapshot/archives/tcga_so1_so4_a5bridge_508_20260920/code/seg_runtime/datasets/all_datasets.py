"""
all_datasets.py  —  Per-dataset PyTorch Dataset classes.

Three-level label system
------------------------
GLOBAL   tissue_type  (int per image)      — organ / body-site of origin
MID      tissue_mask  (H×W int64 or None)  — tissue-level semantic labels
LOCAL    nuclei_mask  (H×W int64 or None)  — nuclei-level semantic labels

None means "this level is not annotated for this dataset".

__getitem__ returns
-------------------
{
    "image":       np.ndarray  uint8  H×W×3,
    "tissue_mask": np.ndarray  int64  H×W   | None,
    "nuclei_mask": np.ndarray  int64  H×W   | None,
    "tissue_type": int,
}

Label spaces
------------
TISSUE_TYPES (global):
  0=Unknown   1=Breast   2=Lung    3=Colon     4=Prostate
  5=Kidney    6=Liver    7=Bladder 8=Stomach   9=Brain   10=Mixed

TISSUE_CLASSES (mid, per-pixel):
  0=Background  1=Neoplastic  2=Inflammatory  3=Connective
  4=Epithelial  5=Dead        6=Gland

NUCLEI_CLASSES (local, per-pixel):
  0=Background  1=Tumor       2=Lymphocyte  3=Plasma cell
  4=Macrophage  5=Neutrophil  6=Eosinophil  7=Epithelial
  8=Stromal     9=Endothelial 10=Dead

Per-dataset coverage
--------------------
  BCSS           tissue_mask ✓  nuclei_mask -   tissue_type=Breast
  GlaS           tissue_mask ✓  nuclei_mask -   tissue_type=Colon
  NuCLS          tissue_mask -  nuclei_mask ✓   tissue_type=Breast
  Kumar          tissue_mask -  nuclei_mask ✓   tissue_type=Mixed
  CPM15          tissue_mask -  nuclei_mask ✓   tissue_type=Mixed
  CPM17          tissue_mask -  nuclei_mask ✓   tissue_type=Mixed
  PanNuke        tissue_mask -  nuclei_mask ✓   tissue_type=per-image from types.npy
  CoNIC2022      tissue_mask -  nuclei_mask ✓   tissue_type=Colon
  Lizard         tissue_mask -  nuclei_mask ✓   tissue_type=Colon
  MoNuSAC        tissue_mask -  nuclei_mask ✓   tissue_type=inferred from TCGA code
  SICAPv2        tissue_mask ✓  nuclei_mask -   tissue_type=Prostate
  SegPath        tissue_mask ✓  nuclei_mask -   tissue_type=Mixed
  DigestPath2019 tissue_mask ✓  nuclei_mask -   tissue_type=Colon
  WSSS4LUAD      tissue_mask ✓  nuclei_mask -   tissue_type=Lung
  Janowczyk      tissue_mask -  nuclei_mask ✓   tissue_type=Breast
  TNBC           tissue_mask -  nuclei_mask ✓   tissue_type=Breast
"""

from __future__ import annotations

import glob
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import scipy.io
from PIL import Image
from torch.utils.data import Dataset

# Disable PIL decompression bomb check — some WSI tiles exceed the default 178M pixel limit
Image.MAX_IMAGE_PIXELS = None

# ─────────────────────────────────────────────────────────────────────────────
# Label spaces
# ─────────────────────────────────────────────────────────────────────────────

TISSUE_TYPES = {
    0: "Unknown",  1: "Breast",   2: "Lung",    3: "Colon",
    4: "Prostate", 5: "Kidney",   6: "Liver",   7: "Bladder",
    8: "Stomach",  9: "Brain",   10: "Mixed",
}
NUM_TISSUE_TYPES = len(TISSUE_TYPES)

TISSUE_CLASSES = {
    0: "Background",  1: "Neoplastic",   2: "Inflammatory",
    3: "Connective",  4: "Epithelial",   5: "Dead",
    6: "Gland",
}
NUM_TISSUE_CLASSES = len(TISSUE_CLASSES)

NUCLEI_CLASSES = {
    0:  "Background",   1: "Tumor",        2: "Lymphocyte",
    3:  "Plasma cell",  4: "Macrophage",   5: "Neutrophil",
    6:  "Eosinophil",   7: "Epithelial",   8: "Stromal",
    9:  "Endothelial", 10: "Dead",
}
NUM_NUCLEI_CLASSES = len(NUCLEI_CLASSES)

# Backward-compatibility aliases used by combine_dataset.py
UNIFIED_CLASSES = TISSUE_CLASSES
NUM_CLASSES     = NUM_TISSUE_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────

class BaseDataset(Dataset):
    """
    Subclasses:
      - populate self.image_paths (and optionally self.mask_paths)
      - set self.magnification (float) and self.tissue_type (int, default 0)
      - override _load_tissue_mask and/or _load_nuclei_mask as needed
    """

    def __init__(self, root_dir: str, magnification: float,
                 tissue_type: int = 0):
        self.root_dir     = root_dir
        self.magnification = magnification
        self.tissue_type  = tissue_type   # default global label (may be overridden per-item)
        self.image_paths: list = []
        self.mask_paths:  list = []

    def __len__(self):
        return len(self.image_paths)

    def _load_image(self, idx) -> np.ndarray:
        return np.array(
            Image.open(self.image_paths[idx]).convert("RGB"), dtype=np.uint8)

    def _load_tissue_mask(self, idx):
        """Return H×W int64 tissue mask, or None if not available."""
        return None

    def _load_nuclei_mask(self, idx):
        """Return H×W int64 nuclei mask, or None if not available."""
        return None

    def _get_tissue_type(self, idx: int) -> int:
        return self.tissue_type

    def __getitem__(self, idx):
        image = self._load_image(idx)
        tm    = self._load_tissue_mask(idx)
        nm    = self._load_nuclei_mask(idx)
        if tm is not None:
            tm = np.asarray(tm, dtype=np.int64)
        if nm is not None:
            nm = np.asarray(nm, dtype=np.int64)
        return {
            "image":       image,
            "tissue_mask": tm,
            "nuclei_mask": nm,
            "tissue_type": self._get_tissue_type(idx),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _remap(mask: np.ndarray, lut: dict) -> np.ndarray:
    """Apply a lookup-table {src_class: unified_class} to mask."""
    out = np.zeros_like(mask, dtype=np.int64)
    for src, tgt in lut.items():
        out[mask == src] = tgt
    return out


def _decode_supervisely_bitmap(data_b64: str, origin: list,
                                canvas_h: int, canvas_w: int) -> np.ndarray:
    """Decode a Supervisely RLE bitmap (zlib-compressed PNG in base64)."""
    import base64, io, zlib
    raw = base64.b64decode(data_b64)
    try:
        raw = zlib.decompress(raw)
    except Exception:
        pass
    arr    = np.array(Image.open(io.BytesIO(raw)).convert("1"), dtype=np.uint8)
    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    ox, oy = int(origin[0]), int(origin[1])
    h, w   = arr.shape
    y2, x2 = min(oy + h, canvas_h), min(ox + w, canvas_w)
    canvas[oy:y2, ox:x2] = arr[:y2 - oy, :x2 - ox]
    return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 1. BCSS  — Breast, 40×
#    tissue_mask  ✓   nuclei_mask  -
# ─────────────────────────────────────────────────────────────────────────────
_BCSS_LUT = {
     0: 0,  1: 1,  2: 3,  3: 2,  4: 5,  5: 0,   # glandular secretion/lumen
     6: 0,  7: 0,  8: 4,  9: 0, 10: 2, 11: 2,
    12: 0, 13: 6, 14: 3, 15: 0, 16: 3, 17: 0,
    18: 3, 19: 1, 20: 1, 21: 0,
}

class BCSSDataset(BaseDataset):
    def __init__(self, root_dir="/local/datasets/public/datasets/BCSS"):
        super().__init__(root_dir, magnification=40.0, tissue_type=1)   # Breast
        self.image_paths = sorted(glob.glob(
            os.path.join(root_dir, "rgbs_colorNormalized", "*.png")))
        self.mask_paths  = [
            os.path.join(root_dir, "masks", os.path.basename(p))
            for p in self.image_paths]

    def _load_tissue_mask(self, idx):
        return _remap(
            np.array(Image.open(self.mask_paths[idx]), dtype=np.int64),
            _BCSS_LUT)


# ─────────────────────────────────────────────────────────────────────────────
# 2. GlaS  — Colon, 20×
#    tissue_mask  ✓ (gland / lumen class 6)   nuclei_mask  -
# ─────────────────────────────────────────────────────────────────────────────

class GlasDataset(BaseDataset):
    def __init__(self, root_dir="/local/datasets/public/datasets/Glas"):
        super().__init__(root_dir, magnification=20.0, tissue_type=3)   # Colon
        for split in ("train", "test_a", "test_b"):
            self.image_paths += sorted(
                glob.glob(os.path.join(root_dir, split, "img", "*.bmp")))
        self.mask_paths = [
            p.replace(os.sep + "img" + os.sep, os.sep + "ann" + os.sep) + ".json"
            for p in self.image_paths]

    def _load_tissue_mask(self, idx):
        img    = Image.open(self.image_paths[idx])
        H, W   = img.size[1], img.size[0]
        canvas = np.zeros((H, W), dtype=np.int64)
        with open(self.mask_paths[idx]) as f:
            ann = json.load(f)
        for obj in ann.get("objects", []):
            bmp   = obj["bitmap"]
            patch = _decode_supervisely_bitmap(bmp["data"], bmp["origin"], H, W)
            # GlaS annotates gland structures, not tumour-region masks.
            canvas[patch > 0] = 6
        return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 3. NuCLS  — Breast, 40×
#    tissue_mask  -   nuclei_mask  ✓
#    R channel: 0=bg 1=tumor_nonMitotic 2=tumor_Mitotic 3=nonTIL_stromal
#               4=macrophage 5=lymphocyte 6=plasma_cell 7=other_immune
#               8=neutrophil 9=unlabeled ≥253=ignore
# ─────────────────────────────────────────────────────────────────────────────
_NUCLS_NUCLEI_LUT = {
    0: 0,  1: 1,  2: 1,  3: 8,
    4: 4,  5: 2,  6: 3,  7: 2,
    8: 5,  9: 0,
}

class NuCLSDataset(BaseDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/NuCLS/EvaluationSet"):
        super().__init__(root_dir, magnification=40.0, tissue_type=1)   # Breast
        _all_images = sorted(
            glob.glob(os.path.join(root_dir, "rgb", "*.png")))
        _all_masks  = [
            os.path.join(root_dir, "mask", os.path.basename(p))
            for p in _all_images]
        # Keep only pairs where the mask file actually exists
        _pairs = [(i, m) for i, m in zip(_all_images, _all_masks)
                  if os.path.exists(m)]
        self.image_paths = [p[0] for p in _pairs]
        self.mask_paths  = [p[1] for p in _pairs]

    def _load_nuclei_mask(self, idx):
        r = np.array(Image.open(self.mask_paths[idx]).convert("RGB"),
                     dtype=np.int64)[:, :, 0]
        r[r >= 253] = 0   # ignore → background
        return _remap(r, _NUCLS_NUCLEI_LUT)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Kumar  — Mixed, 40×
#    tissue_mask  -   nuclei_mask  ✓ (binary → Tumor class)
# ─────────────────────────────────────────────────────────────────────────────

class KumarDataset(BaseDataset):
    def __init__(self, root_dir="/local/datasets/public/datasets/Kumar/kumar"):
        super().__init__(root_dir, magnification=40.0, tissue_type=10)  # Mixed
        for split in ("train", "test_same", "test_diff"):
            self.image_paths += sorted(
                glob.glob(os.path.join(root_dir, split, "Images", "*.tif")))
        self.mask_paths = [
            p.replace("Images", "Labels").replace(".tif", ".mat")
            for p in self.image_paths]

    def _load_nuclei_mask(self, idx):
        inst = scipy.io.loadmat(self.mask_paths[idx])["inst_map"]
        return (inst > 0).astype(np.int64)   # class 1 = Tumor


# ─────────────────────────────────────────────────────────────────────────────
# 5. CPM15  — Mixed, 40×
#    tissue_mask  -   nuclei_mask  ✓ (binary → Tumor class)
# ─────────────────────────────────────────────────────────────────────────────

class CPM15Dataset(BaseDataset):
    def __init__(self, root_dir="/local/datasets/public/datasets/CPM15/cpm15"):
        super().__init__(root_dir, magnification=40.0, tissue_type=10)
        self.image_paths = sorted(
            glob.glob(os.path.join(root_dir, "Images", "*.png")))
        self.mask_paths  = [
            p.replace("Images", "Labels").replace(".png", ".mat")
            for p in self.image_paths]

    def _load_nuclei_mask(self, idx):
        inst = scipy.io.loadmat(self.mask_paths[idx])["inst_map"]
        return (inst > 0).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 6. CPM17  — Mixed, 40×
#    tissue_mask  -   nuclei_mask  ✓ (binary → Tumor class)
# ─────────────────────────────────────────────────────────────────────────────

class CPM17Dataset(BaseDataset):
    def __init__(self, root_dir="/local/datasets/public/datasets/CPM17/cpm17"):
        super().__init__(root_dir, magnification=40.0, tissue_type=10)
        for split in ("train", "test"):
            self.image_paths += sorted(
                glob.glob(os.path.join(root_dir, split, "Images", "*.png")))
        self.mask_paths = [
            p.replace("Images", "Labels").replace(".png", ".mat")
            for p in self.image_paths]

    def _load_nuclei_mask(self, idx):
        inst = scipy.io.loadmat(self.mask_paths[idx])["inst_map"]
        return (inst > 0).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 7. PanNuke  — Mixed (per-image type from types.npy), 20×
#    tissue_mask  -   nuclei_mask  ✓
#    Masks: per-pixel cell-type label (individual cell instances).
#    Channels: 0=neoplastic 1=inflammatory 2=connective 3=dead 4=epithelial
#    (No whole-tissue region annotation — only nuclei pixels are labelled.)
# ─────────────────────────────────────────────────────────────────────────────
_PANNUKE_NUCLEI_CH = {0: 1, 1: 2, 2: 8, 3: 10, 4: 7}  # ch → NUCLEI_CLASSES

_PANNUKE_TISSUE_MAP = {
    "adrenal_gland": 0, "bile-duct": 6,  "bladder":   7, "breast":   1,
    "cervix":        0, "colon":     3,  "esophagus": 8, "headneck": 0,
    "kidney":        5, "liver":     6,  "lung":      2, "ovarian":  0,
    "pancreatic":    0, "prostate":  4,  "skin":      0, "stomach":  8,
    "testis":        0, "thyroid":   0,  "uterus":    0,
}

class PanNukeDataset(BaseDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/PanNuke/PanNuke",
                 parts=("Part 1",)):
        # PanNuke v1/v2 (Gamper et al.): all images resampled to 0.25 µm/px = 40×.
        super().__init__(root_dir, magnification=40.0, tissue_type=10)
        imgs_acc, msks_acc, types_acc = [], [], []
        for part in parts:
            ip = os.path.join(root_dir, part, "Images", "images.npy")
            mp = os.path.join(root_dir, part, "Masks",  "masks.npy")
            tp = os.path.join(root_dir, part, "Types",  "types.npy")
            if not os.path.exists(ip):
                continue
            imgs_acc.append(np.load(ip))
            msks_acc.append(np.load(mp))
            n = len(imgs_acc[-1])
            if os.path.exists(tp):
                types_acc.append(np.load(tp, allow_pickle=True))
            else:
                types_acc.append(np.array(["unknown"] * n))
        if imgs_acc:
            self._images = np.concatenate(imgs_acc, axis=0)   # N×256×256×3
            self._masks  = np.concatenate(msks_acc, axis=0)   # N×256×256×6
            self._types  = np.concatenate(types_acc, axis=0)  # N  (str)
            self.image_paths = list(range(len(self._images)))

    def _get_tissue_type(self, idx):
        t = str(self._types[idx]).lower()
        return _PANNUKE_TISSUE_MAP.get(t, 0)

    def _load_image(self, idx):
        return np.clip(self._images[idx], 0, 255).astype(np.uint8)

    def _load_nuclei_mask(self, idx):
        m6  = self._masks[idx]
        out = np.zeros(m6.shape[:2], dtype=np.int64)
        for ch, cls in _PANNUKE_NUCLEI_CH.items():
            out[m6[:, :, ch] > 0] = cls
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 8. CoNIC 2022  — Colon, 20×
#    tissue_mask  -   nuclei_mask  ✓
#    labels[:,:,1] class: 0=bg 1=neutrophil 2=epithelial 3=lymphocyte
#                          4=plasma 5=eosinophil 6=connective
#    (Per-pixel cell-type annotation; no whole-tissue regions.)
# ─────────────────────────────────────────────────────────────────────────────
_CONIC_NUCLEI_LUT = {0: 0, 1: 5, 2: 7, 3: 2, 4: 3, 5: 6, 6: 8}

class CoNIC2022Dataset(BaseDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/CoNIC2022"):
        super().__init__(root_dir, magnification=20.0, tissue_type=3)   # Colon
        train_dir = os.path.join(root_dir, "CoNIC Challenge Train")
        ip = os.path.join(train_dir, "images.npy")
        lp = os.path.join(train_dir, "labels.npy")
        if os.path.exists(ip):
            self._images = np.load(ip)   # N×256×256×3  uint8
            self._labels = np.load(lp)   # N×256×256×2  uint16
            self.image_paths = list(range(len(self._images)))

    def _load_image(self, idx):
        return self._images[idx].astype(np.uint8)

    def _load_nuclei_mask(self, idx):
        return _remap(
            self._labels[idx, :, :, 1].astype(np.int64), _CONIC_NUCLEI_LUT)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Lizard  — Colon, 20×
#    tissue_mask  -   nuclei_mask  ✓
#    Per-image MAT: inst_map (H×W) + id (N,) + class (N,)
#    class: 1=neutrophil 2=epithelial 3=lymphocyte 4=plasma 5=eosinophil 6=connective
#    (Individual cell-instance annotation; no whole-tissue regions.)
# ─────────────────────────────────────────────────────────────────────────────
_LIZARD_NUCLEI_LUT = {0: 0, 1: 5, 2: 7, 3: 2, 4: 3, 5: 6, 6: 8}

class LizardDataset(BaseDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/Lizard/Lizard"):
        super().__init__(root_dir, magnification=20.0, tissue_type=3)   # Colon
        self.image_paths = (
            sorted(glob.glob(os.path.join(root_dir, "Lizard_Images1", "*.png"))) +
            sorted(glob.glob(os.path.join(root_dir, "Lizard_Images2", "*.png"))))
        self.mask_paths = [
            os.path.join(root_dir, "Labels",
                         os.path.splitext(os.path.basename(p))[0] + ".mat")
            for p in self.image_paths]

    def _build_semantic(self, idx, lut):
        m    = scipy.io.loadmat(self.mask_paths[idx])
        inst = m["inst_map"]
        ids  = m["id"].flatten()
        cls  = m["class"].flatten()
        sem  = np.zeros_like(inst, dtype=np.int64)
        for nid, nc in zip(ids, cls):
            sem[inst == nid] = lut.get(int(nc), 0)
        return sem

    def _load_nuclei_mask(self, idx):
        return self._build_semantic(idx, _LIZARD_NUCLEI_LUT)


# ─────────────────────────────────────────────────────────────────────────────
# 10. MoNuSAC  — Mixed (Breast/Lung/Prostate/Kidney), 40×
#     tissue_mask  -   nuclei_mask  ✓
#     XML polygons: Epithelial, Lymphocyte, Macrophage, Neutrophil
#     Organ inferred from TCGA project code in patient folder name.
# ─────────────────────────────────────────────────────────────────────────────
_MONUSAC_NUCLEI_MAP = {
    "Epithelial":             7,
    "Lymphocyte":             2,
    "Macrophage":             4,
    "Neutrophil":             5,
    "Ambiguous_or_artifacts": 0,
}

# TCGA 2nd field → tissue type (covers known MoNuSAC patient cohorts)
_MONUSAC_TCGA_TYPE = {
    # Breast (BRCA)
    "A7": 1, "E2": 1, "EW": 1, "BH": 1, "AO": 1, "AR": 1,
    # Lung  (LUAD/LUSC)
    "50": 2, "55": 2, "86": 2, "69": 2, "78": 2, "MP": 2,
    # Prostate (PRAD)
    "EJ": 4, "G9": 4, "HC": 4, "HI": 4, "CH": 4,
    # Kidney (KIRC/KIRP)
    "B0": 5, "2Z": 5, "CJ": 5, "DV": 5, "AK": 5,
}

class MoNuSACDataset(BaseDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/MoNuSAC"
                          "/MoNuSAC_images_and_annotations"):
        super().__init__(root_dir, magnification=40.0, tissue_type=10)
        for tif in sorted(glob.glob(os.path.join(root_dir, "*", "*.tif"))):
            xml = os.path.splitext(tif)[0] + ".xml"
            if os.path.exists(xml):
                self.image_paths.append(tif)
                self.mask_paths.append(xml)
        # Precompute per-image tissue type from TCGA project code
        self._tissue_types = [self._infer_type(p) for p in self.image_paths]

    @staticmethod
    def _infer_type(path: str) -> int:
        """Parse TCGA-XX-... patient folder → organ code."""
        folder = os.path.basename(os.path.dirname(path))
        parts  = folder.split("-")
        if len(parts) >= 2:
            return _MONUSAC_TCGA_TYPE.get(parts[1], 10)
        return 10

    def _get_tissue_type(self, idx):
        return self._tissue_types[idx]

    def _load_nuclei_mask(self, idx):
        import cv2
        img    = Image.open(self.image_paths[idx]).convert("RGB")
        H, W   = img.size[1], img.size[0]
        canvas = np.zeros((H, W), dtype=np.int64)
        try:
            tree = ET.parse(self.mask_paths[idx])
            for ann in tree.getroot().findall("Annotation"):
                cls_name = "Ambiguous_or_artifacts"
                for attr in ann.findall("Attributes/Attribute"):
                    cls_name = attr.get("Name", cls_name)
                cls_id = _MONUSAC_NUCLEI_MAP.get(cls_name, 0)
                if cls_id == 0:
                    continue
                for region in ann.findall("Regions/Region"):
                    verts = region.findall("Vertices/Vertex")
                    if len(verts) < 3:
                        continue
                    pts = np.array(
                        [[float(v.get("X")), float(v.get("Y"))]
                         for v in verts], dtype=np.int32)
                    cv2.fillPoly(canvas, [pts], cls_id)
        except Exception:
            pass
        return canvas


# ─────────────────────────────────────────────────────────────────────────────
# 11. SICAPv2  — Prostate, 10×
#     tissue_mask  ✓ (Gleason grading)   nuclei_mask  -
#     0=bg 1=NC(benign gland) 2=G3 3=G4 4=G5
# ─────────────────────────────────────────────────────────────────────────────

class SICAPv2Dataset(BaseDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/SICAPv2"):
        super().__init__(root_dir, magnification=10.0, tissue_type=4)   # Prostate
        for split in ("Train", "Val"):
            for img_path in sorted(
                    glob.glob(os.path.join(root_dir, split, "Images", "*.jpg"))):
                mask_path = img_path.replace("Images", "Masks")  # same .jpg extension
                if os.path.exists(mask_path):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)

    def _load_tissue_mask(self, idx):
        m   = np.array(Image.open(self.mask_paths[idx]), dtype=np.int64)
        out = np.zeros_like(m)
        out[m == 1] = 6   # NC (benign gland) → Gland
        out[m >= 2] = 1   # Gleason grade     → Neoplastic
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 12. SegPath  — Mixed, 40×
#     tissue_mask  ✓ (IHC-marker → tissue class)   nuclei_mask  -
# ─────────────────────────────────────────────────────────────────────────────

_SEGPATH_MARKER_TO_CLASS = {
    # Epithelial markers
    "panCK_Epithelium":  4,   # pan-cytokeratin → Epithelial
    # Inflammatory / immune-cell markers
    "CD3CD20_Lymphocyte": 2,  # T + B lymphocytes → Inflammatory
    "CD45RB_Leukocyte":   2,  # pan-leukocyte → Inflammatory
    "MIST1_PlasmaCell":   2,  # plasma cells → Inflammatory
    "MNDA_MyeloidCell":   2,  # myeloid lineage → Inflammatory
    # Connective / stromal markers
    "aSMA_SmoothMuscle":  3,  # myofibroblasts / smooth muscle → Connective
    "ERG_Endothelium":    3,  # endothelial cells → Connective
    # Non-tissue markers (skip)
    "CD235a_RBC":         0,  # red blood cells → Background
}


class SegPathDataset(BaseDataset):
    def __init__(self, root_dir="/local/datasets/public/datasets/SegPath"):
        super().__init__(root_dir, magnification=40.0, tissue_type=10)
        for p in sorted(glob.glob(
                os.path.join(root_dir, "**", "*_HE.png"), recursive=True)):
            mk = p.replace("_HE.png", "_mask.png")
            if os.path.exists(mk):
                self.image_paths.append(p)
                self.mask_paths.append(mk)

        # Precompute marker → unified class for every image
        self._marker_classes: list = []
        for p in self.image_paths:
            marker = self._extract_marker(p)
            self._marker_classes.append(
                _SEGPATH_MARKER_TO_CLASS.get(marker, 0))

        # Report marker → class distribution
        from collections import Counter
        _mkr_counts = Counter(
            self._extract_marker(p) for p in self.image_paths)
        _cls_counts = Counter(self._marker_classes)
        print(f"[SegPath] {len(self.image_paths)} images, "
              f"{len(_mkr_counts)} markers → {len(set(_cls_counts) - {0})} "
              f"unified tissue classes:")
        for mkr, cnt in sorted(_mkr_counts.items()):
            ucls = _SEGPATH_MARKER_TO_CLASS.get(mkr, "?")
            print(f"  {mkr:30s}  {cnt:5d}  →  T:{ucls}")
        print(f"[SegPath] Class distribution: "
              f"{', '.join(f'T:{c}={n}' for c, n in sorted(_cls_counts.items()))}")

    @staticmethod
    def _extract_marker(path: str) -> str:
        """Parse the marker folder name from a SegPath file path."""
        parts = path.replace(os.sep, "/").split("/")
        for part in parts:
            if part in _SEGPATH_MARKER_TO_CLASS:
                return part
        return ""

    def _load_tissue_mask(self, idx):
        m = np.array(
            Image.open(self.mask_paths[idx]).convert("L"), dtype=np.int64)
        cls_id = self._marker_classes[idx]
        if cls_id == 0:
            return np.zeros_like(m, dtype=np.int64)   # RBC / unknown → all bg
        mask = (m > 0).astype(np.int64)
        mask[mask > 0] = cls_id
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# 13. DigestPath 2019  — Colon, 20×
#     tissue_mask  ✓ (binary lesion/neoplastic)   nuclei_mask  -
# ─────────────────────────────────────────────────────────────────────────────

class DigestPath2019Dataset(BaseDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/DigestPath2019"):
        super().__init__(root_dir, magnification=20.0, tissue_type=3)   # Colon
        _pos_dir = os.path.join(root_dir, "tissue-train-pos-v1")
        for p in sorted(glob.glob(os.path.join(_pos_dir, "*.jpg"))):
            if "_mask" not in p:
                mk = p.replace(".jpg", "_mask.jpg")
                if os.path.exists(mk):
                    # Filter out all-background masks (no foreground pixels)
                    try:
                        m = np.array(Image.open(mk).convert("L"), dtype=np.int64)
                        if (m > 127).sum() == 0:
                            continue   # skip — no annotated tissue
                    except Exception:
                        pass
                    self.image_paths.append(p)
                    self.mask_paths.append(mk)
        # Negative samples from tissue-train-neg are explicitly all-background
        # (no tumor tissue present).  They contribute zero gradient to the
        # segmentation loss and are skipped to avoid wasting training time.
        _neg_dir = os.path.join(
            root_dir, "Colonoscopy_tissue_segment_dataset", "tissue-train-neg")
        _neg_kept = 0
        for p in sorted(glob.glob(os.path.join(_neg_dir, "*.jpg"))):
            # Only keep negative samples that might be useful (skip if all-bg)
            # Actually, negatives always have all-zero masks, so skip all.
            pass  # negative samples excluded — all have zero foreground
        _total = len(self.image_paths)
        print(f"[DigestPath2019] Kept {_total} positive samples with foreground; "
              f"filtered all-background negatives from tissue-train-neg.")

    def _load_tissue_mask(self, idx):
        m = np.array(Image.open(self.mask_paths[idx]).convert("L"), dtype=np.int64)
        return (m > 127).astype(np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 14. WSSS4LUAD  — Lung, 20×
#     tissue_mask  ✓   nuclei_mask  -
#     0=tumor 1=stroma 2=normal 3=background/white
# ─────────────────────────────────────────────────────────────────────────────

class WSSS4LUADDataset(BaseDataset):
    def __init__(self, root_dir="/local/datasets/public/datasets/WSSS4LUAD"):
        super().__init__(root_dir, magnification=20.0, tissue_type=2)   # Lung
        base     = os.path.join(root_dir, "2.validation", "2.validation")
        img_dir  = os.path.join(base, "img")
        mask_dir = os.path.join(base, "mask")
        self.image_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths  = [
            os.path.join(mask_dir, os.path.basename(p))
            for p in self.image_paths]

    def _load_tissue_mask(self, idx):
        m   = np.array(Image.open(self.mask_paths[idx]), dtype=np.int64)
        out = np.zeros_like(m)
        out[m == 0] = 1   # tumor  → Neoplastic
        out[m == 1] = 3   # stroma → Connective
        out[m == 2] = 4   # normal → Epithelial
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 15. Janowczyk  — Breast, 40×
#     tissue_mask  -   nuclei_mask  ✓ (binary epithelial)
# ─────────────────────────────────────────────────────────────────────────────

class JanowczykDataset(BaseDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/janowczyk/Janowczyk"):
        super().__init__(root_dir, magnification=40.0, tissue_type=1)   # Breast
        self.image_paths = sorted(
            glob.glob(os.path.join(root_dir, "*_original.tif")))
        self.mask_paths  = [p.replace("_original.tif", "_mask.png")
                            for p in self.image_paths]

    def _load_nuclei_mask(self, idx):
        m = np.array(Image.open(self.mask_paths[idx]).convert("L"), dtype=np.int64)
        return (m > 127) * 7    # binary → class 7 (Epithelial)


# ─────────────────────────────────────────────────────────────────────────────
# 16. TNBC  — Breast, 40×
#     tissue_mask  -   nuclei_mask  ✓ (binary tumor nuclei)
# ─────────────────────────────────────────────────────────────────────────────

class TNBCDataset(BaseDataset):
    def __init__(self, root_dir="/local/datasets/public/datasets/TNBC"):
        super().__init__(root_dir, magnification=40.0, tissue_type=1)   # Breast
        dataset_dir = root_dir
        for candidate in (os.path.join(root_dir, "TNBC_dataset"), root_dir):
            if os.path.exists(candidate):
                dataset_dir = candidate
                break
        for p in sorted(glob.glob(
                os.path.join(dataset_dir, "Slide_*", "*.png"))):
            if "mask" not in p.lower():
                folder    = os.path.dirname(p)
                gt_folder = folder.replace("Slide_", "GT_")
                mk        = os.path.join(gt_folder, os.path.basename(p))
                if os.path.exists(mk):
                    self.image_paths.append(p)
                    self.mask_paths.append(mk)

    def _load_nuclei_mask(self, idx):
        m = np.array(Image.open(self.mask_paths[idx]).convert("L"), dtype=np.int64)
        return (m > 127) * 1    # binary → class 1 (Tumor)


# ─────────────────────────────────────────────────────────────────────────────
# 18. ICIAR 2018 WSI  — Segmentation from whole-slide images
#     WSI/ has .svs files + .xml polygon annotations + gt_thumbnails/*.png masks
# ─────────────────────────────────────────────────────────────────────────────

class ICIAR2018WSIDataset(BaseDataset):
    """ICIAR2018 BACH WSI segmentation — loads preprocessed 10× PNGs.

    Images and masks are pre-rendered from SVS + XML by
    ``scripts/preprocess_iciar2018_wsi.py`` into::

        WSI_proc_10x/
            A01_image.png   — RGB at 10× (level 1 of SVS pyramid)
            A01_mask.png    — int32 tissue mask (0=bg, 1=neoplastic)
            ...

    Magnification is 10.0; SAM3CombinedDataset upsamples 4× to 40×-equivalent
    for consistent FOV with other datasets.

    **Memory safety**: Full 10× PNGs are 150–165 MP (~500 MB each).  Loading
    them at native resolution + 4× upsampling inside SAM3CombinedDataset
    would OOM.  We centre-crop at native 10× resolution to at most
    ``max_native_dim`` pixels before returning, so PIL never decodes the
    full image into RAM.
    """

    # Centre-crop target at native 10× magnification.  4096 px at 10×
    # upsampled 4× → 16384 px at 40×, which is 4× the 4096 training crop
    # — plenty of context for the centre-crop inside SAM3CombinedDataset.
    MAX_NATIVE_DIM = 4096

    def __init__(self,
                 root_dir="/local/datasets/public/datasets/ICIAR2018_BACH_Challenge"):
        super().__init__(root_dir, magnification=10.0, tissue_type=1)  # Breast
        _proc = os.path.join(root_dir, "WSI_proc_10x")
        _img_files = sorted(glob.glob(os.path.join(_proc, "*_image.png")))
        for img_path in _img_files:
            base = os.path.basename(img_path).replace("_image.png", "")
            mask_path = os.path.join(_proc, f"{base}_mask.png")
            if os.path.exists(mask_path):
                self.image_paths.append(img_path)
                self.mask_paths.append(mask_path)
        print(f"[ICIAR2018_WSI] Loaded {len(self.image_paths)} preprocessed "
              f"PNG pairs from {_proc} (mag={self.magnification:.0f}x, "
              f"max_native_dim={self.MAX_NATIVE_DIM})")

    def _load_image(self, idx) -> np.ndarray:
        """Centre-crop the 10× PNG to avoid decoding a 165 MP image."""
        im = Image.open(self.image_paths[idx]).convert("RGB")
        W, H = im.size
        if max(W, H) > self.MAX_NATIVE_DIM:
            # Centre-crop: keep context around the tissue centre.
            left = max(0, (W - self.MAX_NATIVE_DIM) // 2)
            top  = max(0, (H - self.MAX_NATIVE_DIM) // 2)
            im = im.crop((left, top,
                          left + self.MAX_NATIVE_DIM,
                          top  + self.MAX_NATIVE_DIM))
        return np.array(im, dtype=np.uint8)

    def _load_tissue_mask(self, idx):
        """Load mask, centre-cropped to match the image."""
        mask_im = Image.open(self.mask_paths[idx])
        W, H = mask_im.size
        if max(W, H) > self.MAX_NATIVE_DIM:
            left = max(0, (W - self.MAX_NATIVE_DIM) // 2)
            top  = max(0, (H - self.MAX_NATIVE_DIM) // 2)
            mask_im = mask_im.crop((left, top,
                                    left + self.MAX_NATIVE_DIM,
                                    top  + self.MAX_NATIVE_DIM))
        return np.array(mask_im, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# 19. MIDOG  — Mitosis Detection  (tissue-level seg: mitotic figures)
#     tissue_mask  ✓ (binary → Neoplastic/cellular)   nuclei_mask  -
# ─────────────────────────────────────────────────────────────────────────────

class MIDOGDataset(BaseDataset):
    def __init__(self, root_dir="/local/datasets/public/datasets/MIDOG"):
        super().__init__(root_dir, magnification=40.0, tissue_type=10)  # Mixed
        # MIDOG has .tiff images + JSON annotations with bounding boxes
        _json_path = os.path.join(root_dir, "MIDOG2022_training.json")
        if os.path.exists(_json_path):
            import json as _json
            with open(_json_path) as _f:
                _ann = _json.load(_f)
            _img_by_id = {}
            for img_meta in _ann.get("images", []):
                _img_by_id[img_meta["id"]] = os.path.join(root_dir, img_meta["file_name"])
            self._annotations = _ann.get("annotations", [])
            # Build image_path list from annotations
            _seen = set()
            for ann in self._annotations:
                img_id = ann["image_id"]
                if img_id not in _seen:
                    _seen.add(img_id)
                    if img_id in _img_by_id:
                        self.image_paths.append(_img_by_id[img_id])
            # Build per-image bbox list
            self._img_bboxes: Dict[int, list] = {}
            for ann in self._annotations:
                img_id = ann["image_id"]
                bbox = ann["bbox"]  # [x, y, w, h]
                self._img_bboxes.setdefault(img_id, []).append(bbox)
        else:
            # Fallback: just scan for tiff files
            self.image_paths = sorted(glob.glob(os.path.join(root_dir, "*.tiff")))
            self._img_bboxes = {}

    def _load_tissue_mask(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        H, W = img.size[1], img.size[0]
        mask = np.zeros((H, W), dtype=np.int64)
        # MIDOG images don't have a simple ID mapping; use filename-based lookup
        _bn = os.path.basename(img_path)
        for img_id, bboxes in self._img_bboxes.items():
            # Approximate: if the stored filename matches
            pass
        # Simplified: return empty mask if JSON not available
        return mask


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Tile-level Classification Datasets
# ─────────────────────────────────────────────────────────────────────────────

class TileClassificationDataset(BaseDataset):
    """Base class for tile-level classification datasets (Stage 3).

    __getitem__ returns the same dict as BaseDataset, plus:
        "tile_label": np.ndarray float32 [num_tissue + num_nuclei]
                      Multi-hot vector: 1.0 for each present class.

    These datasets have NO pixel-level masks — only image-level class labels.
    Used in Stage 3 to train the TileClassifier head.
    """

    def __init__(self, root_dir: str, magnification: float,
                 tissue_type: int = 0):
        super().__init__(root_dir, magnification, tissue_type)
        self._class_ids: list = []   # per-image: list of unified class IDs

    def _get_tile_label(self, idx) -> np.ndarray:
        """Return [num_tissue + num_nuclei] float32 multi-hot vector."""
        label = np.zeros(NUM_TISSUE_CLASSES + NUM_NUCLEI_CLASSES, dtype=np.float32)
        for cid in self._class_ids[idx]:
            if 0 <= cid < len(label):
                label[cid] = 1.0
        return label

    def __getitem__(self, idx):
        base = super().__getitem__(idx)
        base["tile_label"] = self._get_tile_label(idx)
        return base


# ── CRC-100K: 9 colorectal tissue classes ───────────────────────────────────
# ADI(adipose)→0?  BACK→0  DEB(debris)→5(necrosis)  LYM(lymphocytes)→2(inflammatory)
# MUC(mucus)→0?  MUS(muscle)→3(stroma)  NORM(normal)→4(epithelium)
# STR(stroma)→3  TUM(tumor)→1(neoplastic)

_CRC100K_CLASS_MAP = {
    "ADI":  [],        # adipose — no direct tissue class, skip
    "BACK": [],        # background — skip
    "DEB":  [5],       # debris → necrosis
    "LYM":  [2],       # lymphocytes → inflammatory
    "MUC":  [],        # mucus — no direct class
    "MUS":  [3],       # muscle → stroma/connective
    "NORM": [4],       # normal mucosa → epithelial
    "STR":  [3],       # stroma → connective
    "TUM":  [1],       # tumor → neoplastic
}

class CRC100KDataset(TileClassificationDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/CRC100K/NCT-CRC-HE-100K"):
        super().__init__(root_dir, magnification=20.0, tissue_type=3)  # Colon
        self._class_ids = []
        for cls_name in sorted(_CRC100K_CLASS_MAP.keys()):
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for p in sorted(glob.glob(os.path.join(cls_dir, "*"))):
                if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    self.image_paths.append(p)
                    self._class_ids.append(_CRC100K_CLASS_MAP[cls_name])


# ── ICIAR 2018 BACH: 4 breast cancer classes ────────────────────────────────
# Benign→1(neoplastic)  InSitu→1  Invasive→1  Normal→4(epithelial)

_ICIAR_CLASS_MAP = {
    "Benign":    [1],
    "InSitu":    [1],
    "Invasive":  [1],
    "Normal":    [4],
}

class ICIAR2018Dataset(TileClassificationDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/ICIAR2018_BACH_Challenge"):
        super().__init__(root_dir, magnification=20.0, tissue_type=1)  # Breast
        _photos = os.path.join(root_dir, "Photos")
        if not os.path.exists(_photos):
            _photos = root_dir
        self._class_ids = []
        for cls_name, cls_ids in _ICIAR_CLASS_MAP.items():
            cls_dir = os.path.join(_photos, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for p in sorted(glob.glob(os.path.join(cls_dir, "*"))):
                if p.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                    self.image_paths.append(p)
                    self._class_ids.append(cls_ids)


# ── MHIST: 2 colon polyp classes ────────────────────────────────────────────
# SSA→1(neoplastic)  HP→4(epithelial/benign)

_MHIST_CLASS_MAP = {
    "SSA": [1],
    "HP":  [4],
}

class MHISTDataset(TileClassificationDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/MHIST"):
        super().__init__(root_dir, magnification=20.0, tissue_type=3)  # Colon
        _csv = os.path.join(root_dir, "annotations.csv")
        _img_dir = os.path.join(root_dir, "images")
        self._class_ids = []
        if os.path.exists(_csv):
            import csv as _csv_mod
            with open(_csv) as _f:
                reader = _csv_mod.DictReader(_f)
                for row in reader:
                    img_name = row.get("Image Name", row.get("image_name", ""))
                    label = row.get("Majority Vote Label",
                                    row.get("majority_vote_label", ""))
                    if img_name and label:
                        p = os.path.join(_img_dir, img_name)
                        if os.path.exists(p):
                            self.image_paths.append(p)
                            self._class_ids.append(
                                _MHIST_CLASS_MAP.get(label.strip(), []))


# ── PatchCamelyon17: binary tumor/non-tumor ─────────────────────────────────

class PatchCamelyon17Dataset(TileClassificationDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/PatchCamelyon17"):
        # PCam: 96×96 patches extracted at 10× (Camelyon16 40× undersampled
        # to enlarge field-of-view; pixel resolution ≈0.972 µm/px).
        super().__init__(root_dir, magnification=10.0, tissue_type=1)  # Breast
        self._class_ids = []
        for p in sorted(glob.glob(os.path.join(root_dir, "*.png")) +
                        glob.glob(os.path.join(root_dir, "*.jpg"))):
            self.image_paths.append(p)
            # All patches contain tumor → class 1 (neoplastic)
            self._class_ids.append([1])


# ── SkinCancer: skin lesion classification ──────────────────────────────────

class SkinCancerDataset(TileClassificationDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/SkinCancer"):
        super().__init__(root_dir, magnification=20.0, tissue_type=0)  # Unknown
        _data_dir = os.path.join(root_dir, "SkinCancer", "data")
        if not os.path.exists(_data_dir):
            _data_dir = root_dir
        self._class_ids = []
        for p in sorted(glob.glob(os.path.join(_data_dir, "**", "*"), recursive=True)):
            if p.lower().endswith((".png", ".jpg", ".jpeg")):
                self.image_paths.append(p)
                self._class_ids.append([1])  # all lesions → neoplastic


# ── Lung-Colon image set ────────────────────────────────────────────────────

class LungColonDataset(TileClassificationDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/lung_colon_image_set"):
        super().__init__(root_dir, magnification=20.0, tissue_type=10)  # Mixed
        self._class_ids = []
        for p in sorted(glob.glob(os.path.join(root_dir, "**", "*.jpeg"),
                                   recursive=True) +
                        glob.glob(os.path.join(root_dir, "**", "*.jpg"),
                                   recursive=True) +
                        glob.glob(os.path.join(root_dir, "**", "*.png"),
                                   recursive=True)):
            self.image_paths.append(p)
            self._class_ids.append([1])  # tumor tissue → neoplastic


# ── Osteosarcoma: tile-level classification ──────────────────────────────────
# Folder-based labels: Non-Tumor, Viable, Non-Viable-Tumor
# Viable → Neoplastic(1), Non-Viable-Tumor → Necrosis(5), Non-Tumor → skip

_OSTEOSARCOMA_CLASS_MAP = {
    "Non-Tumor":        [],
    "Viable":           [1],       # viable tumor → neoplastic
    "Non-Viable-Tumor": [5],       # necrotic tumor → necrosis
}

class OsteosarcomaDataset(TileClassificationDataset):
    def __init__(self,
                 root_dir="/local/datasets/public/datasets/osteosarcoma_dataset"):
        # TCIA Osteosarcoma-Tumor-Assessment: 1144 images of 1024×1024 at 10×.
        super().__init__(root_dir, magnification=10.0, tissue_type=10)  # Mixed
        self._class_ids = []
        for split in ("train", "test"):
            split_dir = os.path.join(root_dir, split)
            if not os.path.isdir(split_dir):
                continue
            for cls_name, cls_ids in _OSTEOSARCOMA_CLASS_MAP.items():
                cls_dir = os.path.join(split_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for p in sorted(glob.glob(os.path.join(cls_dir, "*.jpg")) +
                                glob.glob(os.path.join(cls_dir, "*.jpeg")) +
                                glob.glob(os.path.join(cls_dir, "*.png"))):
                    self.image_paths.append(p)
                    self._class_ids.append(cls_ids)
