"""
combine_dataset.py
==================
Combines all pathology datasets into a single three-level training pipeline.

Each sample carries three label levels:
  GLOBAL   tissue_type  (LongTensor scalar) — organ/body-site
  MID      tissue_mask  (LongTensor H×W)    — tissue-level segmentation
  LOCAL    nuclei_mask  (LongTensor H×W)    — nuclei-level segmentation

Masks not annotated for a dataset are filled with -1 (ignore index).

Usage
-----
    from combine_dataset import build_dataloaders
    train_loader, test_loader = build_dataloaders(batch_size=8, num_workers=4)

    for batch in train_loader:
        imgs        = batch["image"]         # FloatTensor B×C×H×W
        tissue_mask = batch["tissue_mask"]   # LongTensor  B×H×W  (-1 = ignore)
        nuclei_mask = batch["nuclei_mask"]   # LongTensor  B×H×W  (-1 = ignore)
        tissue_type = batch["tissue_type"]   # LongTensor  B
        names       = batch["dataset_name"]  # list[str]   B

CLI
---
    python combine_dataset.py --generate-splits
    python combine_dataset.py --sanity-batch
"""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from all_datasets import (
    BCSSDataset,
    CoNIC2022Dataset,
    CPM15Dataset,
    CPM17Dataset,
    CRC100KDataset,
    DigestPath2019Dataset,
    GlasDataset,
    ICIAR2018Dataset,
    ICIAR2018WSIDataset,
    JanowczykDataset,
    KumarDataset,
    LizardDataset,
    LungColonDataset,
    MHISTDataset,
    MoNuSACDataset,
    NuCLSDataset,
    OsteosarcomaDataset,
    PanNukeDataset,
    PatchCamelyon17Dataset,
    SICAPv2Dataset,
    SegPathDataset,
    SkinCancerDataset,
    TNBCDataset,
    WSSS4LUADDataset,
    NUM_TISSUE_CLASSES,
    NUM_NUCLEI_CLASSES,
    NUM_TISSUE_TYPES,
    TISSUE_CLASSES,
    NUCLEI_CLASSES,
    TISSUE_TYPES,
    # backward-compat aliases
    UNIFIED_CLASSES,
    NUM_CLASSES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SEED               = 42
TARGET_MAG         = 20.0
TARGET_SIZE        = (256, 256)   # H × W  (legacy; kept for CombinedPathologyDataset)
NATIVE_CROP_AT_40X = 1024         # SAM3CombinedDataset crop size at 40×;
                                  # other mags scale proportionally so every
                                  # sample covers the same physical FOV.
SPLIT_DIR          = os.path.join(os.path.dirname(__file__), "splits")


# ─────────────────────────────────────────────────────────────────────────────
# Magnification normalisation  (handles list of masks)
# ─────────────────────────────────────────────────────────────────────────────

def normalise_magnification(image: np.ndarray,
                             masks: list,        # list[np.ndarray | None]
                             native_mag: float,
                             target_mag: float = TARGET_MAG,
                             ) -> Tuple[np.ndarray, list]:
    """
    Rescale image + every mask in `masks` so that native_mag → target_mag.
    None entries in `masks` are passed through unchanged.
    """
    factor = target_mag / native_mag
    if abs(factor - 1.0) < 1e-6:
        return image, masks
    H, W = image.shape[:2]
    nH   = max(1, round(H * factor))
    nW   = max(1, round(W * factor))
    img_out = np.array(
        Image.fromarray(image).resize((nW, nH), Image.BILINEAR),
        dtype=np.uint8)
    mask_out = []
    for m in masks:
        if m is None:
            mask_out.append(None)
        else:
            mp = Image.fromarray(m.astype(np.int32), mode="I").resize(
                     (nW, nH), Image.NEAREST)
            mask_out.append(np.array(mp, dtype=np.int64))
    return img_out, mask_out


# ─────────────────────────────────────────────────────────────────────────────
# PIL helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_pil_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8))


def _to_pil_mask(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.int32), mode="I")


def _arr_to_pil_masks(masks: list) -> list:
    return [_to_pil_mask(m) if m is not None else None for m in masks]


def _pil_masks_to_arr(masks: list) -> list:
    return [np.array(m, dtype=np.int64) if m is not None else None
            for m in masks]


# ─────────────────────────────────────────────────────────────────────────────
# Colour jitter  (image only)
# ─────────────────────────────────────────────────────────────────────────────

def random_colour_jitter(img: Image.Image) -> Image.Image:
    img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
    img = TF.adjust_contrast(img,   random.uniform(0.7, 1.3))
    img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
    img = TF.adjust_hue(img,        random.uniform(-0.1, 0.1))
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Joint augmentation  (applies the SAME spatial transforms to all masks)
# ─────────────────────────────────────────────────────────────────────────────

def augment_sample(img: Image.Image,
                   masks: list,          # list[PIL Image | None]
                   target_size: Tuple[int, int],
                   ) -> Tuple[Image.Image, list]:
    """Random flip, rotate, colour jitter, crop/resize.  Masks follow exactly."""
    # Horizontal flip
    if random.random() > 0.5:
        img   = TF.hflip(img)
        masks = [TF.hflip(m) if m is not None else None for m in masks]
    # Vertical flip
    if random.random() > 0.5:
        img   = TF.vflip(img)
        masks = [TF.vflip(m) if m is not None else None for m in masks]
    # Rotation (0 / 90 / 180 / 270)
    k = random.choice([0, 1, 2, 3])
    if k:
        img   = img.rotate(k * 90, expand=True)
        masks = [m.rotate(k * 90, expand=True) if m is not None else None
                 for m in masks]
    # Colour jitter (image only)
    img = random_colour_jitter(img)
    # Random crop → resize
    W, H   = img.size
    th, tw = target_size
    scale  = random.uniform(0.7, 1.0)
    ch, cw = max(int(H * scale), th), max(int(W * scale), tw)
    i      = random.randint(0, max(0, H - ch))
    j      = random.randint(0, max(0, W - cw))
    img    = TF.crop(img, i, j, ch, cw)
    img    = img.resize((tw, th), Image.BILINEAR)
    new_masks = []
    for m in masks:
        if m is not None:
            m = TF.crop(m, i, j, ch, cw)
            m = m.resize((tw, th), Image.NEAREST)
        new_masks.append(m)
    return img, new_masks


def eval_sample(img: Image.Image,
                masks: list,
                target_size: Tuple[int, int],
                ) -> Tuple[Image.Image, list]:
    """Centre-crop (with padding if needed), no augmentation."""
    W, H   = img.size
    th, tw = target_size
    if H < th or W < tw:
        pad_h, pad_w = max(0, th - H), max(0, tw - W)
        img   = TF.pad(img,   (0, 0, pad_w, pad_h))
        masks = [TF.pad(m, (0, 0, pad_w, pad_h)) if m is not None else None
                 for m in masks]
        W, H = img.size
    i = (H - th) // 2
    j = (W - tw) // 2
    img   = TF.crop(img, i, j, th, tw)
    masks = [TF.crop(m, i, j, th, tw) if m is not None else None
             for m in masks]
    return img, masks


# ─────────────────────────────────────────────────────────────────────────────
# Item wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _Item:
    __slots__ = ("dataset", "local_idx", "dataset_name")

    def __init__(self, dataset, local_idx: int, dataset_name: str):
        self.dataset      = dataset
        self.local_idx    = local_idx
        self.dataset_name = dataset_name


# ─────────────────────────────────────────────────────────────────────────────
# Combined dataset
# ─────────────────────────────────────────────────────────────────────────────

class CombinedPathologyDataset(Dataset):
    """
    Parameters
    ----------
    items       : list[_Item]
    augment     : bool   — apply random augmentations (True for training)
    target_size : (H, W) — output spatial size
    target_mag  : float  — magnification to normalise all images to
    """

    def __init__(self,
                 items: List[_Item],
                 augment: bool = True,
                 target_size: Tuple[int, int] = TARGET_SIZE,
                 target_mag: float = TARGET_MAG):
        self.items       = items
        self.augment     = augment
        self.target_size = target_size
        self.target_mag  = target_mag

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item   = self.items[idx]
        sample = item.dataset[item.local_idx]    # dict from BaseDataset

        image_np  = sample["image"]              # H×W×3  uint8
        t_mask_np = sample["tissue_mask"]        # H×W int64 | None
        n_mask_np = sample["nuclei_mask"]        # H×W int64 | None
        t_type    = int(sample["tissue_type"])

        # 1. Magnification normalisation
        image_np, (t_mask_np, n_mask_np) = normalise_magnification(
            image_np, [t_mask_np, n_mask_np],
            native_mag=item.dataset.magnification,
            target_mag=self.target_mag)

        # 2. Convert to PIL
        img_pil   = _to_pil_image(image_np)
        t_pil     = _to_pil_mask(t_mask_np) if t_mask_np is not None else None
        n_pil     = _to_pil_mask(n_mask_np) if n_mask_np is not None else None

        # 3. Spatial augmentations (identical for all masks)
        if self.augment:
            img_pil, (t_pil, n_pil) = augment_sample(
                img_pil, [t_pil, n_pil], self.target_size)
        else:
            img_pil, (t_pil, n_pil) = eval_sample(
                img_pil, [t_pil, n_pil], self.target_size)

        # 4. Convert to tensors
        img_t = TF.to_tensor(img_pil)
        img_t = TF.normalize(img_t,
                             mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])

        def _mask_tensor(pil_mask):
            if pil_mask is None:
                return torch.full(self.target_size, -1, dtype=torch.long)
            return torch.from_numpy(np.array(pil_mask, dtype=np.int64))

        return {
            "image":        img_t,                                # C×H×W float32
            "tissue_mask":  _mask_tensor(t_pil),                 # H×W   long
            "nuclei_mask":  _mask_tensor(n_pil),                 # H×W   long
            "tissue_type":  torch.tensor(t_type, dtype=torch.long),
            "dataset_name": item.dataset_name,                   # str
        }


# ─────────────────────────────────────────────────────────────────────────────
# Split helpers
# ─────────────────────────────────────────────────────────────────────────────

def _split_indices(n: int, train_ratio: float = 0.8,
                   seed: int = SEED) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    cut = int(n * train_ratio)
    return idx[:cut], idx[cut:]


def generate_splits(datasets: dict,
                    split_dir: str = SPLIT_DIR,
                    train_ratio: float = 0.8,
                    seed: int = SEED) -> Tuple[List[_Item], List[_Item]]:
    """
    Perform per-dataset 80:20 splits, write <name>_{train,test}.txt,
    return combined (train_items, test_items).
    """
    os.makedirs(split_dir, exist_ok=True)
    train_items: List[_Item] = []
    test_items:  List[_Item] = []

    for name, ds in datasets.items():
        n = len(ds)
        if n == 0:
            print(f"  [SKIP] {name}: 0 samples")
            continue

        tr_idx, te_idx = _split_indices(n, train_ratio, seed)

        def _path_of(local_idx: int) -> str:
            try:
                p = ds.image_paths[local_idx]
                return str(p)
            except Exception:
                return str(local_idx)

        with open(os.path.join(split_dir, f"{name}_train.txt"), "w") as f:
            for i in tr_idx:
                f.write(_path_of(i) + "\n")
        with open(os.path.join(split_dir, f"{name}_test.txt"), "w") as f:
            for i in te_idx:
                f.write(_path_of(i) + "\n")

        for i in tr_idx:
            train_items.append(_Item(ds, i, name))
        for i in te_idx:
            test_items.append(_Item(ds, i, name))

        print(f"  {name:20s}: total={n:5d}  train={len(tr_idx):5d}  "
              f"test={len(te_idx):5d}")

    return train_items, test_items


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def _build_default_datasets() -> dict:
    return {
        # ── Segmentation datasets ──────────────────────────────────────
        "BCSS":           BCSSDataset(),
        "GlaS":           GlasDataset(),
        "NuCLS":          NuCLSDataset(),
        "Kumar":          KumarDataset(),
        "CPM15":          CPM15Dataset(),
        "CPM17":          CPM17Dataset(),
        "PanNuke":        PanNukeDataset(),
        "CoNIC2022":      CoNIC2022Dataset(),
        "Lizard":         LizardDataset(),
        "MoNuSAC":        MoNuSACDataset(),
        "SICAPv2":        SICAPv2Dataset(),
        "SegPath":        SegPathDataset(),
        "DigestPath2019": DigestPath2019Dataset(),
        "WSSS4LUAD":      WSSS4LUADDataset(),
        "Janowczyk":      JanowczykDataset(),
        "TNBC":           TNBCDataset(),
        "ICIAR2018_WSI":  ICIAR2018WSIDataset(),
        # ── Tile-level classification datasets (Stage 3) ───────────────
        "CRC100K":        CRC100KDataset(),
        "ICIAR2018":      ICIAR2018Dataset(),
        "MHIST":          MHISTDataset(),
        "PatchCamelyon17": PatchCamelyon17Dataset(),
        "SkinCancer":     SkinCancerDataset(),
        "LungColon":      LungColonDataset(),
        "Osteosarcoma":   OsteosarcomaDataset(),
    }


def build_dataloaders(
    batch_size:  int                    = 8,
    num_workers: int                    = 4,
    target_size: Tuple[int, int]        = TARGET_SIZE,
    target_mag:  float                  = TARGET_MAG,
    train_ratio: float                  = 0.8,
    seed:        int                    = SEED,
    split_dir:   str                    = SPLIT_DIR,
    datasets:    dict                   = None,
    pin_memory:  bool                   = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader).
    Each batch is a dict:
        image        FloatTensor B×C×H×W
        tissue_mask  LongTensor  B×H×W   (-1 = not annotated)
        nuclei_mask  LongTensor  B×H×W   (-1 = not annotated)
        tissue_type  LongTensor  B
        dataset_name list[str]   B
    """
    if datasets is None:
        print("Initialising datasets…")
        datasets = _build_default_datasets()

    print("Generating train / test splits…")
    train_items, test_items = generate_splits(
        datasets, split_dir=split_dir, train_ratio=train_ratio, seed=seed)

    rng = random.Random(seed)
    rng.shuffle(train_items)
    rng.shuffle(test_items)

    train_ds = CombinedPathologyDataset(
        train_items, augment=True,
        target_size=target_size, target_mag=target_mag)
    test_ds  = CombinedPathologyDataset(
        test_items,  augment=False,
        target_size=target_size, target_mag=target_mag)

    print(f"\nCombined train: {len(train_ds):,} samples")
    print(f"Combined test : {len(test_ds):,} samples")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
        worker_init_fn=lambda wid: random.seed(seed + wid))
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _main():
    parser = argparse.ArgumentParser(
        description="Combined pathology dataset utilities")
    parser.add_argument("--generate-splits", action="store_true")
    parser.add_argument("--sanity-batch",    action="store_true")
    parser.add_argument("--batch-size",  type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    print("Building datasets…")
    datasets = _build_default_datasets()

    if args.generate_splits:
        print(f"\nSplit files → {SPLIT_DIR}")
        generate_splits(datasets)
        print("\nDone.")
        return

    if args.sanity_batch:
        train_loader, test_loader = build_dataloaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            datasets=datasets)
        for split, loader in (("TRAIN", train_loader), ("TEST", test_loader)):
            print(f"\n--- {split} ---")
            for batch in loader:
                imgs  = batch["image"]
                tmask = batch["tissue_mask"]
                nmask = batch["nuclei_mask"]
                ttype = batch["tissue_type"]
                print(f"  image        {tuple(imgs.shape)}  "
                      f"min={imgs.min():.2f}  max={imgs.max():.2f}")
                print(f"  tissue_mask  {tuple(tmask.shape)}  "
                      f"classes={sorted(tmask.unique().tolist())}")
                print(f"  nuclei_mask  {tuple(nmask.shape)}  "
                      f"classes={sorted(nmask.unique().tolist())}")
                print(f"  tissue_type  {ttype.tolist()}")
                print(f"  datasets     {batch['dataset_name']}")
                break
        return

    parser.print_help()


if __name__ == "__main__":
    _main()
