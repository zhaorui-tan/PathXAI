"""
check_sanity.py
===============
Sanity-check all sub-datasets AND the combined dataset / loaders.

Run:
    python check_sanity.py                  # individual datasets only
    python check_sanity.py --combined       # also test combined loader
    python check_sanity.py --visualise N    # save N random patches as PNG
"""

from __future__ import annotations

import argparse
import os
import traceback

import numpy as np
from PIL import Image

from all_datasets import (
    BCSSDataset,
    CoNIC2022Dataset,
    CPM15Dataset,
    CPM17Dataset,
    DigestPath2019Dataset,
    GlasDataset,
    JanowczykDataset,
    KumarDataset,
    LizardDataset,
    MoNuSACDataset,
    NuCLSDataset,
    PanNukeDataset,
    SICAPv2Dataset,
    SegPathDataset,
    TNBCDataset,
    WSSS4LUADDataset,
    TISSUE_TYPES,
    TISSUE_CLASSES,
    NUCLEI_CLASSES,
    NUM_TISSUE_CLASSES,
    NUM_NUCLEI_CLASSES,
    # backward-compat aliases
    UNIFIED_CLASSES,
    NUM_CLASSES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palettes
# ─────────────────────────────────────────────────────────────────────────────
TISSUE_PALETTE = np.array([
    [  0,   0,   0],   # 0 Background   — black
    [255,   0,   0],   # 1 Neoplastic   — red
    [  0,   0, 255],   # 2 Inflammatory — blue
    [255, 165,   0],   # 3 Connective   — orange
    [  0, 200,   0],   # 4 Epithelial   — green
    [128,   0, 128],   # 5 Dead         — purple
    [255, 255,   0],   # 6 Gland        — yellow
], dtype=np.uint8)

NUCLEI_PALETTE = np.array([
    [  0,   0,   0],   #  0 Background
    [220,  20,  60],   #  1 Tumor       — crimson
    [ 30, 144, 255],   #  2 Lymphocyte  — dodger blue
    [148,   0, 211],   #  3 Plasma cell — violet
    [255, 140,   0],   #  4 Macrophage  — dark orange
    [  0, 128,   0],   #  5 Neutrophil  — green
    [255,  20, 147],   #  6 Eosinophil  — deep pink
    [ 64, 224, 208],   #  7 Epithelial  — turquoise
    [139,  69,  19],   #  8 Stromal     — saddle brown
    [173, 216, 230],   #  9 Endothelial — light blue
    [ 50,  50,  50],   # 10 Dead        — dark grey
], dtype=np.uint8)

# backward compat
PALETTE = TISSUE_PALETTE

SEP = "─" * 60


def colourise_mask(mask_np: np.ndarray,
                   palette: np.ndarray = None) -> np.ndarray:
    """Convert a H×W int64 mask to H×W×3 RGB using palette."""
    if palette is None:
        palette = TISSUE_PALETTE
    h, w = mask_np.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(palette)):
        rgb[mask_np == cls_id] = palette[cls_id]
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset checks
# ─────────────────────────────────────────────────────────────────────────────

def _check_mask(label: str, mask, n_classes: int) -> tuple:
    """Check a single mask array. Returns (classes, bad_classes, status_str)."""
    if mask is None:
        return None, [], "(not annotated)"
    assert isinstance(mask, np.ndarray), f"{label} is not ndarray"
    assert mask.ndim == 2,               f"{label} expected H×W, got {mask.shape}"
    assert mask.dtype == np.int64,       f"{label} dtype {mask.dtype} ≠ int64"
    classes  = sorted(np.unique(mask).tolist())
    bad_cls  = [c for c in classes if c < 0 or c >= n_classes]
    status   = "OK" if not bad_cls else f"WARN(invalid {bad_cls})"
    return classes, bad_cls, status


def check_single_dataset(name: str, ds, n_check: int = 3,
                          visualise_dir: str = None) -> bool:
    print(f"\n{SEP}\nDataset : {name}")
    print(f"Class   : {type(ds).__name__}")
    print(f"Magnification: {ds.magnification}×   "
          f"tissue_type={ds.tissue_type} ({TISSUE_TYPES.get(ds.tissue_type,'?')})")
    n = len(ds)
    print(f"Samples : {n}")

    if n == 0:
        print("  [WARNING] Empty dataset — check paths.")
        return False

    ok = True
    for i in range(min(n_check, n)):
        try:
            sample = ds[i]
            img    = sample["image"]
            t_mask = sample["tissue_mask"]
            n_mask = sample["nuclei_mask"]
            t_type = sample["tissue_type"]

            assert isinstance(img, np.ndarray), "image is not ndarray"
            assert img.ndim == 3 and img.shape[2] == 3, \
                f"image expected H×W×3, got {img.shape}"
            assert img.dtype == np.uint8, f"image dtype {img.dtype} ≠ uint8"

            h, w = img.shape[:2]
            t_cls, t_bad, t_status = _check_mask(
                "tissue_mask", t_mask, NUM_TISSUE_CLASSES)
            n_cls, n_bad, n_status = _check_mask(
                "nuclei_mask", n_mask, NUM_NUCLEI_CLASSES)

            print(f"  [{i:4d}] img {h:4d}×{w:4d}  "
                  f"tissue={t_cls} {t_status}  "
                  f"nuclei={n_cls} {n_status}  "
                  f"type={t_type}")

            if t_bad or n_bad:
                ok = False

            # Visualise: image | tissue_mask | nuclei_mask
            if visualise_dir:
                os.makedirs(visualise_dir, exist_ok=True)
                h_vis = min(h, 512)
                w_vis = min(w, 512)
                img_s = np.array(
                    Image.fromarray(img).resize((w_vis, h_vis), Image.BILINEAR))
                panels = [img_s]
                for mask_arr, pal in ((t_mask, TISSUE_PALETTE),
                                      (n_mask, NUCLEI_PALETTE)):
                    if mask_arr is not None:
                        m_s = np.array(
                            Image.fromarray(mask_arr.astype(np.int32),
                                            mode="I").resize(
                                (w_vis, h_vis), Image.NEAREST))
                        panels.append(colourise_mask(m_s, pal))
                    else:
                        panels.append(np.zeros((h_vis, w_vis, 3), dtype=np.uint8))
                side = np.concatenate(panels, axis=1)
                Image.fromarray(side).save(
                    os.path.join(visualise_dir, f"{name}_{i:04d}.png"))

        except Exception as e:
            print(f"  [{i:4d}] ERROR: {e}")
            traceback.print_exc()
            ok = False

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Combined dataset checks
# ─────────────────────────────────────────────────────────────────────────────

def check_combined_loaders(n_batches: int = 3, batch_size: int = 4,
                            visualise_dir: str = None):
    from combine_dataset import build_dataloaders

    print(f"\n{SEP}")
    print("Checking CombinedPathologyDataset loaders …")
    train_loader, test_loader = build_dataloaders(
        batch_size=batch_size, num_workers=0)

    for split, loader in (("TRAIN", train_loader), ("TEST", test_loader)):
        print(f"\n  [{split}]  {len(loader.dataset):,} samples  "
              f"/ {len(loader):,} batches")
        for bi, batch in enumerate(loader):
            if bi >= n_batches:
                break
            imgs   = batch["image"]
            tmasks = batch["tissue_mask"]
            nmasks = batch["nuclei_mask"]
            ttypes = batch["tissue_type"]
            names  = batch["dataset_name"]
            print(f"    batch {bi}: imgs {tuple(imgs.shape)}  "
                  f"min={imgs.min():.2f}  max={imgs.max():.2f}")
            print(f"      tissue_mask {tuple(tmasks.shape)}  "
                  f"classes={sorted(tmasks.unique().tolist())}")
            print(f"      nuclei_mask {tuple(nmasks.shape)}  "
                  f"classes={sorted(nmasks.unique().tolist())}")
            print(f"      tissue_type {ttypes.tolist()}")
            print(f"      datasets    {names}")

            if visualise_dir and bi == 0:
                os.makedirs(visualise_dir, exist_ok=True)
                mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std  = imgs.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                for j in range(min(2, imgs.shape[0])):
                    img_np = ((imgs[j] * std + mean).clamp(0, 1) * 255
                              ).byte().permute(1, 2, 0).cpu().numpy()
                    tm_np  = tmasks[j].cpu().numpy()
                    nm_np  = nmasks[j].cpu().numpy()
                    panels = [
                        img_np,
                        colourise_mask(tm_np, TISSUE_PALETTE),
                        colourise_mask(nm_np, NUCLEI_PALETTE),
                    ]
                    side = np.concatenate(panels, axis=1)
                    path = os.path.join(
                        visualise_dir,
                        f"combined_{split.lower()}_b{bi}_s{j}.png")
                    Image.fromarray(side).save(path)


# ─────────────────────────────────────────────────────────────────────────────
# Class-distribution summary
# ─────────────────────────────────────────────────────────────────────────────

def class_distribution(ds, n_samples: int = 50):
    """Estimate class-pixel distribution for tissue and nuclei masks."""
    t_total = np.zeros(NUM_TISSUE_CLASSES, dtype=np.int64)
    n_total = np.zeros(NUM_NUCLEI_CLASSES, dtype=np.int64)
    n       = min(n_samples, len(ds))
    if n == 0:
        return t_total, n_total
    indices = np.random.RandomState(42).choice(len(ds), n, replace=False)
    for i in indices:
        try:
            sample = ds[i]
            if sample["tissue_mask"] is not None:
                for c in range(NUM_TISSUE_CLASSES):
                    t_total[c] += (sample["tissue_mask"] == c).sum()
            if sample["nuclei_mask"] is not None:
                for c in range(NUM_NUCLEI_CLASSES):
                    n_total[c] += (sample["nuclei_mask"] == c).sum()
        except Exception:
            pass
    return t_total, n_total


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ALL_DATASETS = {
    "BCSS":           BCSSDataset,
    "GlaS":           GlasDataset,
    "NuCLS":          NuCLSDataset,
    "Kumar":          KumarDataset,
    "CPM15":          CPM15Dataset,
    "CPM17":          CPM17Dataset,
    "PanNuke":        PanNukeDataset,
    "CoNIC2022":      CoNIC2022Dataset,
    "Lizard":         LizardDataset,
    "MoNuSAC":        MoNuSACDataset,
    "SICAPv2":        SICAPv2Dataset,
    "SegPath":        SegPathDataset,
    "DigestPath2019": DigestPath2019Dataset,
    "WSSS4LUAD":      WSSS4LUADDataset,
    "Janowczyk":      JanowczykDataset,
    "TNBC":           TNBCDataset,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined",   action="store_true",
                        help="Also test the combined DataLoader.")
    parser.add_argument("--visualise",  type=int, default=0, metavar="N",
                        help="Save N visual samples per dataset to ./sanity_vis/")
    parser.add_argument("--dist",       action="store_true",
                        help="Print class-pixel distribution per dataset.")
    parser.add_argument("--only",       type=str, default=None,
                        help="Only check this dataset name.")
    args = parser.parse_args()

    vis_dir = "./sanity_vis" if args.visualise else None
    results = {}

    print(f"\nTissue types ({len(TISSUE_TYPES)} global classes):")
    for cid, cname in TISSUE_TYPES.items():
        print(f"  {cid:2d}: {cname}")
    print(f"\nTissue classes ({NUM_TISSUE_CLASSES} mid-level):")
    for cid, cname in TISSUE_CLASSES.items():
        print(f"  {cid}: {cname}")
    print(f"\nNuclei classes ({NUM_NUCLEI_CLASSES} local):")
    for cid, cname in NUCLEI_CLASSES.items():
        print(f"  {cid:2d}: {cname}")

    to_check = {k: v for k, v in ALL_DATASETS.items()
                if args.only is None or k == args.only}

    for name, cls in to_check.items():
        try:
            ds   = cls()
            ok   = check_single_dataset(
                       name, ds, n_check=args.visualise if args.visualise else 3,
                       visualise_dir=vis_dir)
            results[name] = ok

            if args.dist and len(ds) > 0:
                t_dist, n_dist = class_distribution(ds)
                t_px = t_dist.sum()
                n_px = n_dist.sum()
                if t_px > 0:
                    print("  Tissue-mask distribution (% of sampled pixels):")
                    for cid, cnt in enumerate(t_dist):
                        print(f"    {cid} {TISSUE_CLASSES[cid]:15s}: "
                              f"{cnt / t_px * 100:.1f}%")
                if n_px > 0:
                    print("  Nuclei-mask distribution (% of sampled pixels):")
                    for cid, cnt in enumerate(n_dist):
                        print(f"    {cid:2d} {NUCLEI_CLASSES[cid]:15s}: "
                              f"{cnt / n_px * 100:.1f}%")

        except Exception as e:
            print(f"\n[FAIL] {name}: could not initialise — {e}")
            traceback.print_exc()
            results[name] = False

    # Summary
    print(f"\n{SEP}")
    print("SUMMARY")
    print(f"{SEP}")
    passed = sum(v for v in results.values())
    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        n_str  = ""
        try:
            n_str = f"  ({len(ALL_DATASETS[name]())} samples)"
        except Exception:
            pass
        print(f"  {'[' + status + ']':8s} {name}{n_str}")
    print(f"\n{passed}/{len(results)} datasets passed.")

    if args.combined:
        check_combined_loaders(
            n_batches=2, batch_size=4, visualise_dir=vis_dir)

    if vis_dir:
        print(f"\nVisualisation images saved to: {os.path.abspath(vis_dir)}")


if __name__ == "__main__":
    main()
