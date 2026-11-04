"""
train_medsam_FINAL.py
====================
Three-stage MedSAM3 fine-tuning on the combined pathology corpus.

Stage 1  (--stage 1)
    Fresh MedSAM3 with rank-8 LoRA in the vision encoder and a
    PromptTunedTextEncoder wrapper.  Known class prompts use the learnable
    class-specific embeddings directly, so effective gradients update those
    embeddings and vision LoRA.  The wrapped language backbone is accidentally
    left requires_grad=True but runs under no_grad for string prompts and is
    therefore not updated by the final tissue-training lineage.
    organ_bias is constructed for checkpoint compatibility but is frozen when
    organ conditioning is disabled (as in the final A5 lineage).

Stage 2  (--stage 2, 5 epochs, Adapter=2e-4 / LoRA=1e-6 / prompts=1e-6)
    Load Stage-1 checkpoint.  Add cell-density conditioning from InstanSeg:
    (1) SlideMorphAdapter adds a low-rank prompt bias from slide-level density.
    (2) DensityFusion injects the per-pixel density map into ViT spatial features.
    (3) Density-consistency loss encourages same-density areas to share tissue masks.
    Optimizer groups can include SlideMorphAdapter + DensityFusion + vision
    LoRA + PromptTunedTextEncoder.  For known prompts, only class embeddings
    (and organ_bias when active) receive prompt-path gradients; the wrapped
    language backbone is evaluated under no_grad.

Stage 3  (--stage 3, 5 epochs)
    Former stage 4: add tile-level cls / MIL / intra-image contrastive heads.
    The final A5 run starts directly from a Stage-1 checkpoint, does not use
    the Stage-2 density adapter, and freezes the complete prompt/text pathway.

Outputs per stage
-----------------
    run_<timestamp>_stage<N>/
        best_model.pth               — full checkpoint (prompt, LoRA, adapter, metrics)
        lora_weights.pt              — LoRA-only weights  (stages 1 & 2)
        val_metrics_{cond,nocond}.csv     — per-val-step summary (conditioned / unconditioned)
        per_class_metrics_{cond,nocond}.csv — per-class per-dataset Dice / IoU / Prec / Rec / F1
        test_metrics_{cond,nocond}.csv     — final test summary
        test_per_class_metrics_{cond,nocond}.csv
        vis/                         — visualisation images (image | tissue pred | nuclei pred)

Usage
-----
    # Stage 1 (vision LoRA + learnable class-prompt embeddings):
    python train_medsam_FINAL.py --stage 1 --epochs 5

    # Stage 2 (cell-density conditioning, from Stage-1 ckpt):
    python train_medsam_FINAL.py --stage 2 --epochs 5 --prev_ckpt <stage1/best_model.pth>
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter
from skimage.transform import resize as sk_resize
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import math
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import torchvision.transforms.functional as TVF
from torchvision.transforms import Normalize as TVNormalize

warnings.filterwarnings("ignore")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
MEDSAM3_ROOT = '/local/PathXAI_Agent_v2/PathXAI_Seg2Report/results/tcga_so1_so4_a5bridge_508_20260920/provenance/code/medsam3'
DATASETS_ROOT = "/local/datasets/public/datasets"
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR  = os.path.join(SCRIPT_DIR, "datasets")

for p in (MEDSAM3_ROOT, DATASETS_ROOT, SCRIPT_DIR, DATASETS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from sam3.model_builder import build_sam3_image_model
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    FindQueryLoaded,
    Image as SAMImage,
    InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)
from lora_layers import (
    LoRAConfig,
    LoRALinear,
    apply_lora_to_model,
    count_parameters,
    save_lora_weights,
)

from all_datasets import (
    NUCLEI_CLASSES,
    NUM_NUCLEI_CLASSES,
    NUM_TISSUE_CLASSES,
    NUM_TISSUE_TYPES,
    TISSUE_CLASSES,
    TISSUE_TYPES,
)
from combine_dataset import (
    _Item,
    _build_default_datasets,
    _split_indices,
    generate_splits,
    normalise_magnification,
    NATIVE_CROP_AT_40X,
    SEED,
    SPLIT_DIR,
    TARGET_MAG,
    TARGET_SIZE,
)
try:
    from external_eval_datasets import build_external_eval_datasets
    _EXTERNAL_EVAL_IMPORT_ERROR = None
except Exception as _e:
    build_external_eval_datasets = None
    _EXTERNAL_EVAL_IMPORT_ERROR = _e

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
SAM3_RESOLUTION = 1008      # must match the resolution MedSAM3 was built with (RoPE precomputed at 1008)
EPOCHS_PER_STAGE = 5
# LR_PROMPT = 1e-3
# LR_LORA   = 2e-4
# LR_ADAPTER = 2e-4
# WEIGHT_DECAY = 0.05
LR_PROMPT    = 1e-3   # was 1e-3
LR_LORA      = 2e-4   # keep
LR_ADAPTER   = 2e-4   # keep
WEIGHT_DECAY = 0.01   # was 0.05 — apply 0.05 only to LoRA, 0.01 to prompts
AMP_DTYPE = torch.bfloat16
THRESHOLD = 0.4             # argmax threshold for background class
VIS_NUM_TRAIN = 4           # images to visualise per epoch (train)
VIS_NUM_VAL   = 8           # images to visualise per epoch (val)
ORTHO_WEIGHT  = 0.05        # orthogonality regularisation on prompt embeddings
AUX_PRIOR_WEIGHT = 0.05     # InstanSeg density-consistency loss weight (stage 2+)
UNLABELED_REPULSION_WEIGHT = 1.0   # mild penalty in unlabeled regions (gt==0)
# Kept low (1.0) because symmetric BCE (no pos_weight) + Dice already prevent
# over-prediction naturally.  Higher values risk under-prediction.
NEGATIVE_BOOST = 1.0          # keep 1.0 for symmetric BCE (no pos_weight)
AREA_LOSS_WEIGHT = 0.0       # disabled — symmetric BCE+Dice is sufficient; area loss
# explodes when resuming from pos_weight-trained checkpoints that over-predict.
CONTRASTIVE_WEIGHT = 0.0001     # contrastive loss: separate target vs non-target
CONTRASTIVE_MARGIN = 0.1     # desired gap between mean(target) and mean(non-target)
TILE_CLS_WEIGHT = 0.1       # tile-level classification loss weight (stage 3)
# SSL_WEIGHT is used throughout the code, but only conchSSL_WEIGHT is defined. Alias for compatibility.
conchSSL_WEIGHT      = 0.0       # self-supervised (BYOL) loss weight — DISABLED
SSL_WEIGHT = conchSSL_WEIGHT
# BYOL is inter-image contrastive (view1 vs view2 → identical features).
# This competes with segmentation which needs INTRA-image discrimination
# (class A vs class B within same image → different features).
# Replaced by intra-image contrastive loss below.
INTRA_IMAGE_CONTRAST_WEIGHT = 0.01   # intra-image pixel-level contrastive loss (stage 3)
# Pulls same-class trunk features together, pushes different-class features apart
DENSITY_CONSISTENCY_SIGMA = 0.15    # bandwidth for bilateral TV density-consistency loss (stage 2+)
# Density is normalised to [0,1]; sigma=0.15 means only pixels within ~1/6 of
# the full density range are treated as iso-density and pulled together.
# within each image.  Directly aligned with segmentation objective — encourages
# spatially discriminative ViT features that the mask decoder can separate.
INTER_CLS_CONTRAST_WEIGHT = 0.03  # inter-class prototype contrastive (stage 3)
STRUCTURE_CLASS_LOSS_WEIGHT = 0.2

# ── Morphology-Relation Self-Supervision (Stage 3) ──────────────────────
# Builds cell graphs on-the-fly via InstanSeg (no offline precomputation).
# The projector lives alongside contrast_proj and is checkpointed with it.
MORPH_GRAPH_WEIGHT = 0.0          # local morphology stats regression weight
MORPH_TRANSITION_WEIGHT = 0.0     # morphology-boundary transition loss weight
MORPH_GRAPH_GRID_SIZE = 16        # spatial grid for local graph statistics
MORPH_KEEP_N_CELLS = 512          # max cells per graph (subsampled on-the-fly)
MORPH_RELATION_MODE = "none"      # "none" | "stats_only" | "transition_only" | "full"

# Mild suppression of foreground predictions in obvious WSI background
# (near-white / glass-slide regions).  Stage 2+ only; small weight so it
# doesn't compete with real tissue annotations.
BG_SUPPRESSION_WEIGHT = 0.005  # p² penalty on white-region predictions
# Non-contrastive cls-only supervision (Stage 3):
#  - MIL presence loss from class-presence labels
#  - absent-class suppression on pooled segmentation probabilities
#  - pixel entropy minimisation on tissue prob maps (sharpens decisions)
#  - organ-head warmup: skip organ-classification gradient for first N steps
#    of Stage 3 so cls task does not destabilise segmentation geometry.
CLS_ONLY_MIL_WEIGHT = 0.15
CLS_ONLY_ABSENT_WEIGHT = 0.05
CLS_ONLY_TOPK_RATIO = 0.02
CLS_ONLY_ENTROPY_WEIGHT = 0.02
STAGE4_ORGAN_WARMUP_STEPS = 500

BPE_PATH = os.path.join(MEDSAM3_ROOT, "sam3/assets/bpe_simple_vocab_16e6.txt.gz")

SLIDE_PATCH_SIZE = 256   # fallback default — actual sliding-window patch size
SLIDE_STRIDE     = 128   # is now derived from native_crop_at_40x at call time

# ─────────────────────────────────────────────────────────────────────────────
# Prompt specifications
# 18 prompts = 7 tissue classes + 11 nuclei classes
# class_id encodes both spaces: tissue 0-6, nuclei 100-110
# ─────────────────────────────────────────────────────────────────────────────
# TISSUE_PROMPT_TEXTS = {
#     0: "background tissue",
#     1: "neoplastic tumor tissue",
#     2: "inflammatory infiltrate lymphocyte",
#     3: "connective stromal tissue",
#     4: "normal non-glandular epithelial tissue",
#     5: "necrotic dead tissue debris",
#     6: "gland lumen structural tissue",
# }

# NUCLEI_PROMPT_TEXTS = {
#     0:  "background cell",
#     1:  "tumor cancer cell nucleus",
#     2:  "lymphocyte immune cell nucleus",
#     3:  "plasma cell nucleus",
#     4:  "macrophage nucleus",
#     5:  "neutrophil nucleus",
#     6:  "eosinophil nucleus",
#     7:  "epithelial cell nucleus",
#     8:  "stromal fibroblast cell nucleus",
#     9:  "endothelial cell nucleus",
#     10: "dead apoptotic cell nucleus",
# }

TISSUE_CLASS_SPECS = {
    0: {"prompt": "background", "axis": "type", "ortho_group": "tissue_type"},
    1: {"prompt": "tumor", "axis": "type", "ortho_group": "tissue_type"},
    2: {"prompt": "inflammatory", "axis": "type", "ortho_group": "tissue_type"},
    3: {"prompt": "stroma", "axis": "type", "ortho_group": "tissue_type"},
    4: {"prompt": "epithelium", "axis": "type", "ortho_group": "tissue_type"},
    5: {"prompt": "necrosis", "axis": "type", "ortho_group": "tissue_type"},
    6: {"prompt": "gland", "axis": "structure", "ortho_group": "tissue_structure"},
}


def _tissue_class_spec(cid: int) -> dict:
    name = TISSUE_CLASSES.get(cid, f"tissue {cid}")
    return {
        "prompt": str(name).lower(),
        "axis": "type",
        "ortho_group": "tissue_type",
        **TISSUE_CLASS_SPECS.get(cid, {}),
    }


TISSUE_PROMPT_TEXTS = {
    cid: _tissue_class_spec(cid)["prompt"]
    for cid in sorted(TISSUE_CLASSES)
}

TISSUE_PROMPT_GROUPS = {
    cid: _tissue_class_spec(cid)["ortho_group"]
    for cid in sorted(TISSUE_CLASSES)
}

TISSUE_PROMPT_AXES = {
    cid: _tissue_class_spec(cid)["axis"]
    for cid in sorted(TISSUE_CLASSES)
}

NUCLEI_PROMPT_TEXTS = {
    0:  "background",
    1:  "tumor cell",
    2:  "lymphocyte",
    3:  "plasma cell",
    4:  "macrophage",
    5:  "neutrophil",
    6:  "eosinophil",
    7:  "epithelial cell",
    8:  "stromal cell",
    9:  "endothelial cell",
    10: "apoptotic cell",
}


def _known_prompt_keys() -> frozenset:
    keys = {_normalise_key(t) for t in TISSUE_PROMPT_TEXTS.values()}
    keys.update(_normalise_key(t) for t in NUCLEI_PROMPT_TEXTS.values())
    keys.update(f"tissue_{cid}" for cid in TISSUE_PROMPT_TEXTS)
    keys.update(f"nuclei_{cid}" for cid in NUCLEI_PROMPT_TEXTS)
    return frozenset(keys)


def _is_known_prompt_text(prompt_text: str) -> bool:
    return _normalise_key(prompt_text) in _known_prompt_keys()

# Dataset → set of class keys present in its masks / derivable tile labels.
# "T:c" = tissue class c, "N:c" = nuclei class c.
# Used for class-balanced sampling, one-per-class val/vis selection, AND for
# building the per-sample tile_label supervision mask (only declared classes
# carry a known target — undeclared classes are "unknown" and ignored by BCE).
#
# Nuclei-only datasets (PanNuke/CoNIC/...) also declare DERIVED tissue classes
# (via _NUCLEI_TO_TISSUE).  They have no tissue mask but their nuclei mask
# implies which tissue classes are present → used as cls-only tile supervision.
_DS_CLASSES: Dict[str, frozenset] = {
    "BCSS":           frozenset({"T:1","T:2","T:3","T:4","T:5","T:6"}),
    "GlaS":           frozenset({"T:6"}),  # gland structure masks
    # ── Nuclei seg datasets, used as cls-only in tissue training ──────────
    # T:* keys are DERIVED from nuclei via _NUCLEI_TO_TISSUE (cls supervision only).
    # N:* keys are the dataset's actual annotated nuclei classes (for nuclei seg).
    "NuCLS":          frozenset({"T:1","T:2","T:3","N:1","N:2","N:3","N:4","N:5","N:8"}),
    "Kumar":          frozenset({"T:1","N:1"}),
    "CPM15":          frozenset({"T:1","N:1"}),
    "CPM17":          frozenset({"T:1","N:1"}),
    "PanNuke":        frozenset({"T:1","T:2","T:3","T:4","T:5","N:1","N:2","N:7","N:8","N:10"}),
    "CoNIC2022":      frozenset({"T:2","T:3","T:4","N:2","N:3","N:5","N:6","N:7","N:8"}),
    "Lizard":         frozenset({"T:2","T:3","T:4","N:2","N:3","N:5","N:6","N:7","N:8"}),
    "MoNuSAC":        frozenset({"T:2","T:4","N:2","N:4","N:5","N:7"}),
    "Janowczyk":      frozenset({"T:4","N:7"}),
    "TNBC":           frozenset({"T:1","N:1"}),
    # "SICAPv2":        frozenset({"T:1","T:6"}), # bad quality, escape
    "SegPath":        frozenset({"T:2","T:3","T:4"}),  # IHC-marker → tissue class
    "DigestPath2019": frozenset({"T:1"}),
    "WSSS4LUAD":      frozenset({"T:1","T:3","T:4"}),
    "ICIAR2018_WSI":  frozenset({"T:1","T:4"}),  # preprocessed 10× PNGs from WSI_proc_10x/
    # ── Tile-level classification datasets (Stage 3+) ──────────────────
    "CRC100K":        frozenset({"T:1","T:2","T:3","T:4","T:5"}),  # 9→5 tissue classes
    "MHIST":          frozenset({"T:1","T:4"}),  # colon polyp: SSA/HP
    "ICIAR2018":      frozenset({"T:1","T:4"}),  # breast: Benign/InSitu/Invasive→1, Normal→4
    "PatchCamelyon17": frozenset({"T:1"}),        # binary tumor
    "SkinCancer":     frozenset({"T:1"}),         # skin lesions
    "LungColon":      frozenset({"T:1"}),         # lung/colon tumors
    "Osteosarcoma":   frozenset({"T:1","T:5"}),   # Viable→neoplastic, Non-Viable→necrosis
}

# WSI datasets currently backed by low-resolution thumbnail PNGs rather than
# native high-magnification patches.  Do not treat these as pixel-supervised
# segmentation datasets under native_crop_at_40x; that would violate the common
# physical-FOV assumption and make quick-val/final metrics misleading.
_THUMBNAIL_WSI_SEG_DS: frozenset = frozenset()  # ICIAR2018_WSI loads preprocessed PNGs from WSI_proc_10x/

# Nuclei class → derived tissue class.  Used to give nuclei-only datasets
# (PanNuke / CoNIC / Lizard / MoNuSAC / NuCLS / Kumar / CPM / Janowczyk / TNBC)
# tile-level tissue cls supervision when no tissue mask is available.
# Cell-type presence is a strong cue for tissue class:
#   tumor cells     → Neoplastic   (T:1)
#   immune cells    → Inflammatory (T:2)
#   stromal/endo    → Connective   (T:3)
#   epithelial      → Epithelial   (T:4)
#   apoptotic       → Necrosis     (T:5)
_NUCLEI_TO_TISSUE: Dict[int, int] = {
    1: 1,   # tumor cell        → T:1 Neoplastic
    2: 2,   # lymphocyte        → T:2 Inflammatory
    3: 2,   # plasma cell       → T:2 Inflammatory
    4: 2,   # macrophage        → T:2 Inflammatory
    5: 2,   # neutrophil        → T:2 Inflammatory
    6: 2,   # eosinophil        → T:2 Inflammatory
    7: 4,   # epithelial cell   → T:4 Epithelial
    8: 3,   # stromal cell      → T:3 Connective
    9: 3,   # endothelial cell  → T:3 Connective
    10: 5,  # apoptotic cell    → T:5 Necrosis
}

# Datasets with ≥ 2 annotated classes — used as the "rich" warmup subset in curriculum learning.
# Excludes single-class datasets (SegPath T:1 ≈94% of data, GlaS, Kumar/CPM N:1, …)
# that alone cannot teach the model to discriminate between classes.
# Separate warmup sets per modality so --seg_mode tissue/nuclei/both all get meaningful warmup.
_MULTICLS_TISSUE_DATASETS: frozenset = frozenset(
    ds for ds, cks in _DS_CLASSES.items()
    if sum(1 for c in cks if c.startswith("T:")) >= 2
)  # BCSS, SICAPv2, WSSS4LUAD

_MULTICLS_NUCLEI_DATASETS: frozenset = frozenset(
    ds for ds, cks in _DS_CLASSES.items()
    if sum(1 for c in cks if c.startswith("N:")) >= 2
)  # NuCLS, PanNuke, CoNIC2022, Lizard, MoNuSAC

# Union kept for reference / backward compat
_MULTICLS_DATASETS: frozenset = _MULTICLS_TISSUE_DATASETS | _MULTICLS_NUCLEI_DATASETS

# ── Per-dataset annotation completeness ──────────────────────────────────────
# Datasets where mask value 0 means "background" (known negative) rather
# than "unlabeled" (unknown).  For these, every pixel carries a valid label,
# so the model SHOULD be penalised for predicting foreground in bg regions.
_TISSUE_FULLY_ANNOTATED: frozenset = frozenset({
    # BCSS excluded — despite _BCSS_LUT, many gt==0 pixels are unlabeled, not true bg.
    # ONLY BCSS has unlabeled pixels; all other tissue seg datasets have complete masks.
    "DigestPath2019",  # Binary tumor/background — every pixel labeled
    "GlaS",            # Gland segmentation: non-gland pixels are true negatives for T:6
    "ICIAR2018_WSI",   # Binary tissue/background — every pixel labeled
    "SegPath",         # IHC masks: marker absent is true negative
    "SICAPv2",         # Full prostate tissue annotation — all pixels labeled
    "WSSS4LUAD",       # CAM pseudo-labels — every pixel assigned a class
})

_NUCLEI_FULLY_ANNOTATED: frozenset = frozenset({
    "PanNuke",
    "CoNIC2022",
    "Lizard",
    "MoNuSAC",
    "Kumar",
    "CPM15",
    "CPM17",
    "Janowczyk",
    "TNBC",
    # NuCLS excluded — has explicit unlabeled (9) and ignore (≥253)
    # pixels mapped to 0, so 0 is a mix of true bg and unlabeled.
})

# ── Slide-adaptive cell morphology adapter feature count ────────────────────
NUM_MORPH_FEATURES = 16  # 4×4 spatial pool of density map — preserves intra-image structure

# ─────────────────────────────────────────────────────────────────────────────
# Colour palettes for visualisation
# ─────────────────────────────────────────────────────────────────────────────
_TISSUE_BASE_COLORS = [
    "#000000", "#E41A1C", "#377EB8", "#FF7F00", "#4DAF4A", "#984EA3", "#A65628",
]


def _extend_palette(base_colors: List[str], n: int) -> List[str]:
    colors = list(base_colors[:n])
    if len(colors) >= n:
        return colors
    cmap = plt.get_cmap("tab20")
    i = 0
    while len(colors) < n:
        rgba = cmap(i % cmap.N)
        colors.append("#%02X%02X%02X" % tuple(int(255 * c) for c in rgba[:3]))
        i += 1
    return colors


TISSUE_PAL = ListedColormap(_extend_palette(_TISSUE_BASE_COLORS, NUM_TISSUE_CLASSES))
NUCLEI_PAL = ListedColormap([
    "#000000","#E41A1C","#377EB8","#984EA3","#FF7F00","#4DAF4A","#F781BF",
    "#00CED1","#A0522D","#ADD8E6","#696969",
])


# ─────────────────────────────────────────────────────────────────────────────
# Tile-level multi-hot label from masks  (used by Stage 3 TileClassifier)
# ─────────────────────────────────────────────────────────────────────────────

def _make_tile_label(t_mask: Optional[np.ndarray],
                     n_mask: Optional[np.ndarray],
                     nuclei_to_tissue: bool = True) -> np.ndarray:
    """Return soft ratio vector [NUM_TISSUE_CLASSES + NUM_NUCLEI_CLASSES] float32.

    Label[c] = fraction of pixels belonging to class c in the mask.
    E.g., if 80% of pixels are tumor (class 1) and 20% are stroma (class 4),
    returns [0, 0.8, 0, 0, 0.2, 0, ...].
    Tissue classes occupy indices [0, NUM_TISSUE_CLASSES).
    Nuclei classes occupy  [NUM_TISSUE_CLASSES, NUM_TISSUE_CLASSES + NUM_NUCLEI_CLASSES).

    When ``nuclei_to_tissue=True`` and a nuclei mask is present but no tissue
    mask, tissue indices are additionally filled via ``_NUCLEI_TO_TISSUE`` —
    e.g. presence of N:1 tumor cells implies T:1 Neoplastic tissue.  The
    derived tissue label is binary (1.0 if any pixels of a contributing
    nuclei class are present), not ratio, since cells underestimate tissue
    extent.
    """
    label = np.zeros(NUM_TISSUE_CLASSES + NUM_NUCLEI_CLASSES, dtype=np.float32)
    if t_mask is not None:
        _total = t_mask.size
        for c in np.unique(t_mask):
            if 1 <= c < NUM_TISSUE_CLASSES:
                label[c] = (t_mask == c).sum() / _total  # soft ratio
    if n_mask is not None:
        _total = n_mask.size
        for c in np.unique(n_mask):
            if 1 <= c < NUM_NUCLEI_CLASSES:
                label[NUM_TISSUE_CLASSES + c] = (n_mask == c).sum() / _total
        # Derive tissue presence from nuclei classes when there is no
        # explicit tissue mask (nuclei-only datasets used as cls-only tissue).
        if nuclei_to_tissue and t_mask is None:
            for n_cls, t_cls in _NUCLEI_TO_TISSUE.items():
                if 0 < t_cls < NUM_TISSUE_CLASSES:
                    if (n_mask == n_cls).any():
                        label[t_cls] = 1.0
    return label


def _make_label_mask(dataset_name: str) -> np.ndarray:
    """Return [NUM_TISSUE_CLASSES + NUM_NUCLEI_CLASSES] float32 binary mask.

    1.0 = this index is a KNOWN target for ``dataset_name``  (BCE supervises)
    0.0 = this index is UNKNOWN / unlabeled for ``dataset_name`` (BCE skips)

    Built from ``_DS_CLASSES``: only declared T:c / N:c entries are supervised.
    Index 0 (background) of both modalities is never supervised — the model
    does not need to learn "background presence" from a multi-label cls head.

    This avoids the bug where a DigestPath sample (only T:1 labeled) would
    push T:2/T:3/T:4/T:5 logits toward absent — those classes might really be
    present in the image but simply unlabeled by this dataset's schema.
    """
    mask = np.zeros(NUM_TISSUE_CLASSES + NUM_NUCLEI_CLASSES, dtype=np.float32)
    declared = _DS_CLASSES.get(dataset_name, frozenset())
    for ck in declared:
        try:
            mod, cid_s = ck.split(":")
            cid = int(cid_s)
        except Exception:
            continue
        if mod == "T" and 0 < cid < NUM_TISSUE_CLASSES:
            mask[cid] = 1.0
        elif mod == "N" and 0 < cid < NUM_NUCLEI_CLASSES:
            mask[NUM_TISSUE_CLASSES + cid] = 1.0
    return mask


def _masked_tile_cls_bce(logits: torch.Tensor,
                         tile_label: torch.Tensor,
                         tile_label_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Multi-label BCE on tile-cls logits, restricted to KNOWN class indices.

    ``tile_label_mask`` is a 0/1 float vector of the same shape as ``logits``.
    BCE is averaged only over indices where mask>0.  Returns 0.0 if no index
    is supervised (caller can skip).
    """
    logits = logits.reshape(-1)
    tile_label = tile_label.reshape(-1)
    if tile_label_mask is None:
        return F.binary_cross_entropy_with_logits(logits, tile_label)
    m = tile_label_mask.reshape(-1).bool()
    if not m.any():
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits[m], tile_label[m])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset adapter — returns PIL images + numpy masks (NO tensor normalisation)
# ─────────────────────────────────────────────────────────────────────────────

class SAM3CombinedDataset(Dataset):
    """
    Wraps _Item list.  __getitem__ returns:
        {
            "image":        PIL.Image   H×W RGB  (native-mag crop, no rescaling),
            "tissue_mask":  np.int64    H×W  or None,
            "nuclei_mask":  np.int64    H×W  or None,
            "tissue_type":  int,
            "dataset_name": str,
            "sample_idx":   int,
            "content_hw":   (int, int)  real content size within the crop
                                        (< crop size only when the source image
                                         was smaller than the crop and was padded;
                                         always equals crop size for normal images),
        }

    Crop size is determined by the dataset's native magnification so that every
    sample covers the same physical tissue area as ``native_crop_at_40x`` pixels
    at 40×:
        40× dataset → native_crop = native_crop_at_40x          (1024)
        20× dataset → native_crop = native_crop_at_40x / 2      ( 512)
        10× dataset → native_crop = native_crop_at_40x / 4      ( 256)

    Images smaller than native_crop are zero-padded on the right/bottom.
    The padded region is reflected in ``content_hw`` so visualisation code can
    trim it before display.  NO ImageNet normalisation is applied here; that
    is handled inside SAM3's own transform pipeline.
    """

    def __init__(self, items: List[_Item],
                 augment: bool = False,
                 native_crop_at_40x: int = NATIVE_CROP_AT_40X,
                 crop_to_target: bool    = True,
                 max_eval_size:  int     = 1024,
                 fg_crop_prob:   float   = 0.0,
                 cache_samples:  bool    = False,
                 normalize_magnification: bool = True):
        self.items               = items
        self.augment             = augment
        self.native_crop_at_40x  = native_crop_at_40x
        self.crop_to_target      = crop_to_target
        self.max_eval_size       = max_eval_size
        self.fg_crop_prob        = float(max(0.0, min(1.0, fg_crop_prob)))
        # In-memory cache of fully-decoded sample dicts (post-resize, post-crop).
        # Only safe when augment=False (deterministic output per idx).  Used by
        # quick-val datasets so repeated validation passes don't re-decode JPEGs
        # or re-resample masks.  ~3MB per 1008x1008 RGB sample.
        self.cache_samples       = bool(cache_samples) and (not augment)
        self.normalize_magnification = bool(normalize_magnification)
        self._sample_cache: Dict[int, dict] = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.cache_samples and idx in self._sample_cache:
            return self._sample_cache[idx]
        item   = self.items[idx]
        sample = item.dataset[item.local_idx]

        img_np  = sample["image"]             # uint8 H×W×3
        t_mask  = sample["tissue_mask"]       # int64 H×W | None
        n_mask  = sample["nuclei_mask"]       # int64 H×W | None
        t_type  = int(sample["tissue_type"])

        # Normalise every dataset to a common 40× physical pixel size.
        # 20× source → upsample 2×; 10× → upsample 4×; 40× → no change.
        # After this, native_crop is always native_crop_at_40x.
        native_mag  = item.dataset.magnification
        mag_scale   = (
            40.0 / float(native_mag)
            if self.normalize_magnification else 1.0
        )
        native_crop = int(self.native_crop_at_40x)

        img_pil = PILImage.fromarray(img_np)
        W, H    = img_pil.size
        if abs(mag_scale - 1.0) > 1e-3:
            nH, nW = max(1, int(round(H * mag_scale))), max(1, int(round(W * mag_scale)))
            img_pil = img_pil.resize((nW, nH), PILImage.BILINEAR)
            if t_mask is not None:
                t_mask = np.array(
                    PILImage.fromarray(t_mask.astype(np.int32), mode="I")
                    .resize((nW, nH), PILImage.NEAREST), dtype=np.int64)
            if n_mask is not None:
                n_mask = np.array(
                    PILImage.fromarray(n_mask.astype(np.int32), mode="I")
                    .resize((nW, nH), PILImage.NEAREST), dtype=np.int64)
            H, W = nH, nW

        # ── Full-image mode: no crop (used with sliding-window eval / reference vis) ──
        if not self.crop_to_target:
            orig_h, orig_w = H, W
            max_s = max(H, W)
            if max_s > self.max_eval_size:
                scale = self.max_eval_size / max_s
                nH, nW = max(1, int(H * scale)), max(1, int(W * scale))
                img_pil = img_pil.resize((nW, nH), PILImage.BILINEAR)
                if t_mask is not None:
                    t_mask = np.array(
                        PILImage.fromarray(t_mask.astype(np.int32), mode="I")
                        .resize((nW, nH), PILImage.NEAREST), dtype=np.int64)
                if n_mask is not None:
                    n_mask = np.array(
                        PILImage.fromarray(n_mask.astype(np.int32), mode="I")
                        .resize((nW, nH), PILImage.NEAREST), dtype=np.int64)
                # Effective magnification after resize — used by sliding-window
                # to compute a patch size that matches the training FOV.
                # Image has already been rescaled to 40× equivalent above.
                eff_mag = (
                    40.0 * nH / orig_h
                    if self.normalize_magnification else 40.0
                )
            else:
                eff_mag = 40.0
            W, H = img_pil.size
            if self.augment:
                if random.random() > 0.5:
                    img_pil = TVF.hflip(img_pil)
                    if t_mask is not None:
                        t_mask = t_mask[:, ::-1].copy()
                    if n_mask is not None:
                        n_mask = n_mask[:, ::-1].copy()
                img_pil = TVF.adjust_brightness(
                    img_pil, random.uniform(0.8, 1.2))
                img_pil = TVF.adjust_contrast(
                    img_pil, random.uniform(0.8, 1.2))
                img_pil = TVF.adjust_saturation(
                    img_pil, random.uniform(0.8, 1.2))
                img_pil = TVF.adjust_hue(
                    img_pil, random.uniform(-0.05, 0.05))
            return {
                "image":        img_pil,
                "tissue_mask":  t_mask,
                "nuclei_mask":  n_mask,
                "tissue_type":  t_type,
                "dataset_name": item.dataset_name,
                "local_idx":    item.local_idx,
                "sample_idx":   idx,
                "content_hw":   (H, W),
                "magnification": eff_mag,
                "tile_label":   _make_tile_label(t_mask, n_mask),
                "tile_label_mask": _make_label_mask(item.dataset_name),
            }

        # ── Crop-to-target mode (training and standard eval) ─────────────────
        # Source has already been rescaled to 40× equivalent above.  Images that
        # remain smaller than native_crop are zero-padded to the right/bottom
        # so all samples share a common ``native_crop_at_40x × native_crop_at_40x``
        # canvas at a single physical pixel size.
        orig_h, orig_w = H, W

        # Pad to at least native_crop × native_crop (right/bottom) so small
        # images survive the crop.  Content always starts at pixel (0, 0).
        pad_h = max(0, native_crop - H)
        pad_w = max(0, native_crop - W)
        if pad_h > 0 or pad_w > 0:
            img_pil = TVF.pad(img_pil, (0, 0, pad_w, pad_h))
            W, H    = img_pil.size

        if self.augment:
            # Foreground-aware crop: sample crop center from any
            # annotated (gt > 0) tissue/nuclei pixel.
            # NEVER centre on background/unlabeled (gt==0) areas —
            # those contain no supervision signal.
            i = 0
            j = 0
            fg = None
            if t_mask is not None:
                fg = (t_mask > 0)
            if n_mask is not None:
                fg = (n_mask > 0) if fg is None else (fg | (n_mask > 0))
            if fg is not None and fg.any():
                ys, xs = np.where(fg)
                k = random.randint(0, len(ys) - 1)
                cy, cx = int(ys[k]), int(xs[k])
                i = max(0, min(cy - native_crop // 2, max(0, H - native_crop)))
                j = max(0, min(cx - native_crop // 2, max(0, W - native_crop)))
            else:
                # No annotated pixels at all — fall back to random
                # (should only happen for cls-only datasets with no masks).
                i = random.randint(0, max(0, H - native_crop))
                j = random.randint(0, max(0, W - native_crop))
            # fg_crop_prob is now unused — all crops are fg-centered.
            # Kept as a no-op for backward compat with saved configs.
        else:
            i = (H - native_crop) // 2
            j = (W - native_crop) // 2

        img_pil = TVF.crop(img_pil, i, j, native_crop, native_crop)

        def _crop_mask(m):
            if m is None:
                return None
            m_pil = PILImage.fromarray(m.astype(np.int32), mode="I")
            if m_pil.size[0] < native_crop or m_pil.size[1] < native_crop:
                m_pil = TVF.pad(m_pil, (0, 0,
                                         max(0, native_crop - m_pil.size[0]),
                                         max(0, native_crop - m_pil.size[1])))
            return np.array(TVF.crop(m_pil, i, j, native_crop, native_crop),
                            dtype=np.int64)

        t_mask = _crop_mask(t_mask)
        n_mask = _crop_mask(n_mask)

        # Real content extent inside the crop.
        # For augment=False (eval): content occupies [:content_h, :content_w].
        # Padded rows/cols beyond these bounds are zeros (background class 0).
        content_h = max(0, min(orig_h - i, native_crop))
        content_w = max(0, min(orig_w - j, native_crop))

        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                img_pil = TVF.hflip(img_pil)
                if t_mask is not None:
                    t_mask = t_mask[:, ::-1].copy()
                if n_mask is not None:
                    n_mask = n_mask[:, ::-1].copy()
            # Color jitter (image only — masks unchanged)
            img_pil = TVF.adjust_brightness(img_pil, random.uniform(0.8, 1.2))
            img_pil = TVF.adjust_contrast(img_pil,   random.uniform(0.8, 1.2))
            img_pil = TVF.adjust_saturation(img_pil, random.uniform(0.8, 1.2))
            img_pil = TVF.adjust_hue(img_pil,        random.uniform(-0.05, 0.05))

        return {
            "image":        img_pil,
            "tissue_mask":  t_mask,
            "nuclei_mask":  n_mask,
            "tissue_type":  t_type,
            "dataset_name": item.dataset_name,
            "local_idx":    item.local_idx,
            "sample_idx":   idx,
            "content_hw":   (content_h, content_w),
            # Use underlying dataset's tile_label when available (cls-only
            # datasets have real class labels); fall back to mask-derived.
            "tile_label":   (sample.get("tile_label")
                             if t_mask is None and n_mask is None
                             else _make_tile_label(t_mask, n_mask)),
            "tile_label_mask": _make_label_mask(item.dataset_name),
        }


def list_collate(batch):
    return batch


class ClassConditionedDataset(Dataset):
    """Training dataset that guarantees every crop contains the target class.

    Each entry is ``(item, class_key)`` where ``class_key`` ∈ ``{"T:1", …, "N:10"}``.
    Uniform sampling over this flat list gives exact class balance at the *tile*
    level — not just at the image level.  Memory overhead is O(n_pairs) lightweight
    Python tuples (item references + short strings), with no extra image data.
    """

    def __init__(self,
                 class_items: List[Tuple["_Item", str]],
                 native_crop_at_40x: int = NATIVE_CROP_AT_40X):
        self.class_items        = class_items
        self.native_crop_at_40x = native_crop_at_40x

    def __len__(self):
        return len(self.class_items)

    def __getitem__(self, idx):
        item, class_key = self.class_items[idx]
        mod = "tissue" if class_key.startswith("T:") else "nuclei"
        cid = int(class_key[2:])

        sample  = item.dataset[item.local_idx]
        img_np  = sample["image"]
        t_mask  = sample["tissue_mask"]
        n_mask  = sample["nuclei_mask"]
        t_type  = int(sample["tissue_type"])

        native_mag  = item.dataset.magnification
        mag_scale   = 40.0 / float(native_mag)
        native_crop = int(self.native_crop_at_40x)

        img_pil = PILImage.fromarray(img_np)
        W, H    = img_pil.size

        # Normalise to 40× equivalent so all datasets share the same physical
        # pixel size (20× ×2, 10× ×4, 40× unchanged).
        if abs(mag_scale - 1.0) > 1e-3:
            nH, nW = max(1, int(round(H * mag_scale))), max(1, int(round(W * mag_scale)))
            img_pil = img_pil.resize((nW, nH), PILImage.BILINEAR)
            if t_mask is not None:
                t_mask = np.array(
                    PILImage.fromarray(t_mask.astype(np.int32), mode="I")
                    .resize((nW, nH), PILImage.NEAREST), dtype=np.int64)
            if n_mask is not None:
                n_mask = np.array(
                    PILImage.fromarray(n_mask.astype(np.int32), mode="I")
                    .resize((nW, nH), PILImage.NEAREST), dtype=np.int64)
            W, H = nW, nH

        # Pad to at least native_crop × native_crop
        pad_h = max(0, native_crop - H)
        pad_w = max(0, native_crop - W)
        if pad_h > 0 or pad_w > 0:
            img_pil = TVF.pad(img_pil, (0, 0, pad_w, pad_h))
            if t_mask is not None:
                t_mask = np.pad(t_mask, ((0, pad_h), (0, pad_w)), constant_values=0)
            if n_mask is not None:
                n_mask = np.pad(n_mask, ((0, pad_h), (0, pad_w)), constant_values=0)
            W, H = W + pad_w, H + pad_h

        # Class-conditioned crop: centre on a randomly sampled pixel of the
        # target class.  Falls back to a random crop only if the class is absent
        # (should not happen if the flat index was built correctly).
        target_mask = t_mask if mod == "tissue" else n_mask
        i = random.randint(0, max(0, H - native_crop))
        j = random.randint(0, max(0, W - native_crop))
        if target_mask is not None:
            ys, xs = np.where(target_mask == cid)
            if len(ys):
                k  = random.randint(0, len(ys) - 1)
                cy, cx = int(ys[k]), int(xs[k])
                i  = max(0, min(cy - native_crop // 2, H - native_crop))
                j  = max(0, min(cx - native_crop // 2, W - native_crop))

        img_pil = TVF.crop(img_pil, i, j, native_crop, native_crop)
        if t_mask is not None:
            t_mask = t_mask[i:i + native_crop, j:j + native_crop].copy()
        if n_mask is not None:
            n_mask = n_mask[i:i + native_crop, j:j + native_crop].copy()

        # Augmentation (always applied — this dataset is training-only)
        if random.random() > 0.5:
            img_pil = TVF.hflip(img_pil)
            if t_mask is not None:
                t_mask = t_mask[:, ::-1].copy()
            if n_mask is not None:
                n_mask = n_mask[:, ::-1].copy()
        img_pil = TVF.adjust_brightness(img_pil, random.uniform(0.8, 1.2))
        img_pil = TVF.adjust_contrast(img_pil,   random.uniform(0.8, 1.2))
        img_pil = TVF.adjust_saturation(img_pil, random.uniform(0.8, 1.2))
        img_pil = TVF.adjust_hue(img_pil,        random.uniform(-0.05, 0.05))

        return {
            "image":        img_pil,
            "tissue_mask":  t_mask,
            "nuclei_mask":  n_mask,
            "tissue_type":  t_type,
            "dataset_name": item.dataset_name,
            "local_idx":    item.local_idx,
            "sample_idx":   idx,
            "content_hw":   (native_crop, native_crop),
            "tile_label":   _make_tile_label(t_mask, n_mask),
            "tile_label_mask": _make_label_mask(item.dataset_name),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SAM3 input preparation
# ─────────────────────────────────────────────────────────────────────────────

def _build_transform(resolution: int):
    return ComposeAPI(transforms=[
        RandomResizeAPI(sizes=resolution, max_size=resolution,
                        square=True, consistent_transform=True),
        ToTensorAPI(),
        NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def _create_datapoint(pil_image: PILImage.Image, text_prompt: str) -> Datapoint:
    W, H = pil_image.size
    sam_image = SAMImage(data=pil_image, objects=[], size=[H, W])
    query = FindQueryLoaded(
        query_text=text_prompt,
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=0,
        inference_metadata=InferenceMetadata(
            coco_image_id=0, original_image_id=0,
            original_category_id=1, original_size=[H, W],
            object_id=0, frame_index=0,
        ),
    )
    return Datapoint(find_queries=[query], images=[sam_image])


def build_sam_batch(pil_image: PILImage.Image, prompt_text: str,
                    transform, device: torch.device):
    dp = _create_datapoint(pil_image, prompt_text)
    dp = transform(dp)
    batch = collate_fn_api([dp], dict_key="input")["input"]
    batch = copy_data_to_device(batch, device, non_blocking=True)
    return batch  # keep batch in float32; autocast handles casting during forward


def build_sam_batch_multi(pil_image: PILImage.Image, prompt_texts: List[str],
                          transform, device: torch.device):
    """Build a single batched SAM3 input for N prompts on the same image.
    Returns a batch of size N — vision encoder runs N times (once per prompt),
    but reduces Python/kernel-launch overhead from N calls to 1.

    The image is pre-resized to SAM3 resolution ONCE before the prompt loop so
    that RandomResizeAPI inside the transform becomes a trivial same-size copy
    instead of a full upscale repeated N times.
    """
    # Pre-resize once: avoids N × (TARGET_SIZE → SAM3_RESOLUTION) PIL resizes.
    pil_rs = pil_image.resize((SAM3_RESOLUTION, SAM3_RESOLUTION), PILImage.BILINEAR)
    dps = [transform(_create_datapoint(pil_rs, pt)) for pt in prompt_texts]
    batch = collate_fn_api(dps, dict_key="input")["input"]
    batch = copy_data_to_device(batch, device, non_blocking=True)
    return batch


def _batch_to_dtype(obj, dtype):
    if isinstance(obj, torch.Tensor) and obj.is_floating_point():
        return obj.to(dtype=dtype)
    if isinstance(obj, dict):
        return {k: _batch_to_dtype(v, dtype) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        result = [_batch_to_dtype(x, dtype) for x in obj]
        return type(obj)(result)
    return obj


def get_fused_prob(last_output: dict,
                   target_hw: Tuple[int, int],
                   mask_logit_bias: Optional[torch.Tensor] = None,
                   ) -> torch.Tensor:
    """Return [1, H, W] probability map from SAM3's last output dict.

    Args:
        last_output: SAM3 output dict.
        target_hw:   (H, W) for the returned probability map.
        mask_logit_bias: Optional additive bias on the *raw* decoder mask
                         logits, shape [1, H', W'] or [1, 1, H', W'] (any
                         spatial resolution; bilinearly resampled).  Used
                         for the gated density prior so inference matches
                         training.
    """
    pred_logits = last_output["pred_logits"]   # [1, Q, C]
    pred_masks  = last_output["pred_masks"]    # [1, Q, h, w]

    if mask_logit_bias is not None:
        b = mask_logit_bias
        if b.dim() == 3:
            b = b.unsqueeze(1)
        if b.shape[-2:] != pred_masks.shape[-2:]:
            b = F.interpolate(b.float(), size=pred_masks.shape[-2:],
                              mode="bilinear", align_corners=False)
        pred_masks = pred_masks + b.to(pred_masks.dtype)

    query_scores = pred_logits.sigmoid().max(dim=-1)[0]  # [1, Q]
    query_probs  = pred_masks.sigmoid()                  # [1, Q, h, w]
    weighted     = query_scores[:, :, None, None] * query_probs
    fused_small  = weighted.amax(dim=1, keepdim=True)    # [1, 1, h, w]
    fused = F.interpolate(fused_small, size=target_hw,
                          mode="bilinear", align_corners=False).squeeze(0)
    return fused   # [1, H, W]


def get_fused_prob_batch(last_output: dict,
                         target_hw: Tuple[int, int],
                         mask_logit_bias: Optional[torch.Tensor] = None,
                         ) -> torch.Tensor:
    """Return [N, H, W] probability maps from a batched SAM3 output dict.

    Args:
        last_output:     SAM3 output dict with 'pred_logits' [N,Q,C] and
                         'pred_masks' [N,Q,h,w].
        target_hw:       (H, W) for the returned probability map.
        mask_logit_bias: Optional per-sample additive bias on the *raw*
                         decoder mask logits (true-logit space).  Shape
                         [N, H', W'] or [N, 1, H', W'] in any spatial
                         resolution — it is bilinearly resampled to the
                         decoder's native (h, w) and broadcast across the
                         Q query dimension.  Use this for density priors,
                         attention bias, etc.
    """
    pred_logits = last_output["pred_logits"]   # [N, Q, C]
    pred_masks  = last_output["pred_masks"]    # [N, Q, h, w]

    if mask_logit_bias is not None:
        b = mask_logit_bias
        if b.dim() == 3:
            b = b.unsqueeze(1)            # [N, 1, H', W']
        if b.shape[-2:] != pred_masks.shape[-2:]:
            b = F.interpolate(b.float(), size=pred_masks.shape[-2:],
                              mode="bilinear", align_corners=False)
        # Broadcast across queries: [N, 1, h, w] + [N, Q, h, w]
        pred_masks = pred_masks + b.to(pred_masks.dtype)

    query_scores = pred_logits.sigmoid().max(dim=-1)[0]          # [N, Q]
    query_probs  = pred_masks.sigmoid()                           # [N, Q, h, w]
    weighted     = query_scores[:, :, None, None] * query_probs  # [N, Q, h, w]
    fused_small  = weighted.amax(dim=1, keepdim=True)             # [N, 1, h, w]
    fused = F.interpolate(fused_small, size=target_hw,
                          mode="bilinear", align_corners=False).squeeze(1)
    return fused   # [N, H, W]


@contextmanager
def _base_sam_for_unknown_prompt(model: nn.Module, prompt_text: str):
    """Temporarily route unknown free-text prompts through base SAM3.

    Known project prompts use prompt tuning, LoRA, density fusion, and adapters.
    Unknown prompts keep SAM3's frozen text embedding and disable pathology
    LoRA/density adaptations so free-text inference behaves like original SAM3.
    """
    if _is_known_prompt_text(prompt_text):
        yield False
        return

    raw_model = model.module if hasattr(model, "module") else model
    saved_scales = []
    for module in raw_model.modules():
        lora = getattr(module, "lora", None)
        if lora is not None and hasattr(lora, "scaling"):
            saved_scales.append((lora, lora.scaling))
            lora.scaling = 0.0

    old_density_mod = getattr(raw_model, "_density_fusion_mod", None)
    old_pending_density = getattr(raw_model, "_pending_density", None)
    raw_model._density_fusion_mod = None
    raw_model._pending_density = None

    prompt_enc = getattr(getattr(raw_model, "backbone", None),
                         "language_backbone", None)
    old_adapter_bias = getattr(prompt_enc, "_adapter_bias_per_idx", None)
    if hasattr(prompt_enc, "_adapter_bias_per_idx"):
        prompt_enc._adapter_bias_per_idx = None

    try:
        yield True
    finally:
        for lora, scale in saved_scales:
            lora.scaling = scale
        raw_model._density_fusion_mod = old_density_mod
        raw_model._pending_density = old_pending_density
        if hasattr(prompt_enc, "_adapter_bias_per_idx"):
            prompt_enc._adapter_bias_per_idx = old_adapter_bias


def _build_nuclei_density_logit_bias(
    prompt_entries: list,
    density_map: Optional[torch.Tensor],
    scale: float,
    device: torch.device,
    eps: float = 1e-3,
    pred_masks: Optional[torch.Tensor] = None,
    density_gate_net: Optional[nn.Module] = None,
) -> Optional[torch.Tensor]:
    """Construct per-prompt additive logit bias for Option-A density prior.

    Returns a [N, h_pm, w_pm] tensor (at pred_masks resolution when gated,
    else at density-map resolution) where nuclei entries get
        bias = gate(pred_masks) * scale * log(density)   if gate provided
        bias = scale * log(density)                       otherwise
    and tissue entries get 0.  Returns None if the prior is disabled or
    no nuclei prompts are present.
    """
    if scale <= 0.0 or density_map is None or not prompt_entries:
        return None
    if not any(pt == "nuclei" for (pt, _, _) in prompt_entries):
        return None
    d = density_map
    if not torch.is_tensor(d):
        d = torch.as_tensor(d)
    d = d.to(device=device, dtype=torch.float32)
    if d.dim() != 2:
        d = d.view(*d.shape[-2:])

    use_gate = (density_gate_net is not None) and (pred_masks is not None)
    if use_gate:
        h, w = pred_masks.shape[-2:]
        d_low = F.interpolate(
            d.view(1, 1, *d.shape), size=(h, w),
            mode="bilinear", align_corners=False).view(h, w)
        log_d = torch.log(d_low.clamp(min=eps, max=1.0)) * float(scale)  # [h, w]
        gate  = density_gate_net(pred_masks)  # [N, 1, h, w] in (0,1)
        N = len(prompt_entries)
        bias = torch.zeros((N, h, w), device=device, dtype=torch.float32)
        for i, (pt, _cid, _txt) in enumerate(prompt_entries):
            if pt == "nuclei":
                bias[i] = gate[i, 0].float() * log_d
        return bias
    else:
        log_d = torch.log(d.clamp(min=eps, max=1.0)) * float(scale)
        N = len(prompt_entries)
        bias = torch.zeros((N,) + tuple(log_d.shape), device=device, dtype=torch.float32)
        for i, (pt, _cid, _txt) in enumerate(prompt_entries):
            if pt == "nuclei":
                bias[i] = log_d
        return bias


@torch.no_grad()
def sliding_window_predict_class(
    model,
    transform,
    pil_image: PILImage.Image,
    prompt_text: str,
    device: torch.device,
    patch_size: int = SLIDE_PATCH_SIZE,
    stride:     int = SLIDE_STRIDE,
    density_map: Optional[torch.Tensor] = None,
    density_gate_net: Optional[nn.Module] = None,
    density_logit_prior_scale: float = 0.0,
    is_nuclei_prompt: bool = False,
) -> np.ndarray:
    """
    Run one class/prompt over the full image using a sliding window.
    Returns [H, W] float32 probability map.
    Falls through to a single forward pass when image fits in one patch.

    Inference-time density prior (matches training):
        If is_nuclei_prompt and density_map + scale + gate_net are all
        provided, applies bias = gate(pred_masks) * scale * log(density)
        to the raw decoder mask logits before sigmoid — identical to the
        gated path used in evaluate().
    """
    W, H = pil_image.size
    _base_unknown_prompt = not _is_known_prompt_text(prompt_text)
    _use_gate = (
        not _base_unknown_prompt
        and
        is_nuclei_prompt
        and density_map is not None
        and density_gate_net is not None
        and density_logit_prior_scale > 0.0
    )
    _raw_model = model.module if hasattr(model, "module") else model
    _use_density_fusion = (
        not _base_unknown_prompt
        and density_map is not None
        and getattr(_raw_model, "_density_fusion_mod", None) is not None
    )

    def _set_patch_density(y0: int, x0: int, ph: int, pw: int) -> None:
        if not _use_density_fusion:
            return
        d = density_map
        if not torch.is_tensor(d):
            d = torch.as_tensor(d)
        d = d.to(device=device, dtype=torch.float32)
        if d.dim() != 2:
            d = d.view(*d.shape[-2:])
        d_crop = d[y0:y0 + ph, x0:x0 + pw]
        if d_crop.shape[0] != patch_size or d_crop.shape[1] != patch_size:
            d_pad = torch.zeros((patch_size, patch_size),
                                device=device, dtype=torch.float32)
            d_pad[:d_crop.shape[0], :d_crop.shape[1]] = d_crop
            d_crop = d_pad
        dmax = float(d_crop.max()) if d_crop.numel() > 0 else 0.0
        _raw_model._pending_density = (d_crop / dmax if dmax > 0 else d_crop)

    def _patch_bias(y0: int, x0: int, ph: int, pw: int,
                    out_dict: dict) -> Optional[torch.Tensor]:
        """Build per-patch logit bias (gate(pred_masks) * scale * log(d)).
        Returns [1, h_pm, w_pm] tensor at decoder resolution, or None."""
        if not _use_gate:
            return None
        d = density_map
        if not torch.is_tensor(d):
            d = torch.as_tensor(d)
        d = d.to(device=device, dtype=torch.float32)
        if d.dim() != 2:
            d = d.view(*d.shape[-2:])
        # Crop density to this patch's image region; pad if patch overflows.
        d_crop = d[y0:y0 + ph, x0:x0 + pw]
        if d_crop.shape[0] != patch_size or d_crop.shape[1] != patch_size:
            d_pad = torch.zeros((patch_size, patch_size),
                                device=device, dtype=torch.float32)
            d_pad[:d_crop.shape[0], :d_crop.shape[1]] = d_crop
            d_crop = d_pad
        pm = out_dict["pred_masks"]  # [1, Q, h, w]
        h, w = pm.shape[-2:]
        d_low = F.interpolate(
            d_crop.view(1, 1, *d_crop.shape), size=(h, w),
            mode="bilinear", align_corners=False).view(h, w)
        log_d = torch.log(d_low.clamp(min=1e-3, max=1.0)) \
            * float(density_logit_prior_scale)
        gate = density_gate_net(pm)  # [1, 1, h, w]
        return (gate[:, 0].float() * log_d).to(device)  # [1, h, w]

    if H <= patch_size and W <= patch_size:
        pad_w = max(0, patch_size - W)
        pad_h = max(0, patch_size - H)
        img_p = TVF.pad(pil_image, (0, 0, pad_w, pad_h)) if (pad_w or pad_h) else pil_image
        batch = build_sam_batch(img_p, prompt_text, transform, device)
        _set_patch_density(0, 0, H, W)
        with _base_sam_for_unknown_prompt(model, prompt_text), \
                torch.no_grad(), autocast(dtype=AMP_DTYPE):
            out = model(batch)
        _raw_model._pending_density = None
        _bias = _patch_bias(0, 0, H, W, out[-1])
        # Upsample to patch_size×patch_size (matching the padded input),
        # then crop to the original content region — same as multi-pass loop.
        # Passing (H, W) directly would stretch the padding into the output.
        prob_full = get_fused_prob(out[-1], (patch_size, patch_size),
                                   mask_logit_bias=_bias).squeeze(0).detach().cpu().float().numpy()
        return prob_full[:H, :W]

    def _anchors(length, patch, step):
        pts = list(range(0, length - patch, step))
        if not pts or pts[-1] + patch < length:
            pts.append(max(0, length - patch))
        return pts

    prob_sum = np.zeros((H, W), dtype=np.float32)
    count    = np.zeros((H, W), dtype=np.float32)
    for y0 in _anchors(H, patch_size, stride):
        for x0 in _anchors(W, patch_size, stride):
            x1, y1 = min(x0 + patch_size, W), min(y0 + patch_size, H)
            ph, pw  = y1 - y0, x1 - x0
            patch   = pil_image.crop((x0, y0, x1, y1))
            if ph < patch_size or pw < patch_size:
                patch = TVF.pad(patch, (0, 0, patch_size - pw, patch_size - ph))
            batch = build_sam_batch(patch, prompt_text, transform, device)
            _set_patch_density(y0, x0, ph, pw)
            with _base_sam_for_unknown_prompt(model, prompt_text), \
                    torch.no_grad(), autocast(dtype=AMP_DTYPE):
                out = model(batch)
            _raw_model._pending_density = None
            _bias = _patch_bias(y0, x0, ph, pw, out[-1])
            prob_p = get_fused_prob(
                out[-1], (patch_size, patch_size),
                mask_logit_bias=_bias,
            ).squeeze(0).detach().cpu().float().numpy()
            prob_sum[y0:y1, x0:x1] += prob_p[:ph, :pw]
            count[y0:y1, x0:x1]    += 1.0
    return prob_sum / np.maximum(count, 1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Learnable prompt-tuned text encoder  (Stage 1+)
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_key(text: str) -> str:
    return text.lower().strip().replace("_", " ")


class PromptTunedTextEncoder(nn.Module):
    """
    Wraps SAM3's language backbone.  Per-class text embeddings are learnable.
    Supports tissue prompts (keys: tissue_0 … tissue_6) and
    nuclei prompts (keys: nuclei_0 … nuclei_10).
    """

    def __init__(self, base_text_encoder: nn.Module,
                 all_prompt_texts: Dict[str, str],   # key → prompt string
                 device: torch.device):
        super().__init__()
        self.base_text_encoder = base_text_encoder
        for p in self.base_text_encoder.parameters():
            p.requires_grad = False

        self.key_to_idx: Dict[str, int] = {}
        prompts_list: List[str] = []
        for i, (key, text) in enumerate(all_prompt_texts.items()):
            prompts_list.append(text)
            self.key_to_idx[key] = i
            # Also index by normalised prompt text for lookup during forward
            self.key_to_idx[_normalise_key(text)] = i

        # Initialise from frozen text encoder
        with torch.no_grad():
            attn_mask, text_mem, _ = self.base_text_encoder(
                prompts_list, device=device)
            init_emb  = text_mem.transpose(0, 1).contiguous()   # [N, L, D]
            init_mask = attn_mask.contiguous()                   # [N, L]

        self.prompt_embeddings = nn.Parameter(init_emb.clone().to(torch.float32))
        self.register_buffer("prompt_masks", init_mask.clone())

        # Organ-conditioned prompt bias: [NUM_TISSUE_TYPES, N_prompts, D]
        # Applied per-class to the *first token* of each prompt embedding.
        # Per-class means "breast tumor" and "breast stroma" get different
        # organ-conditioned shifts, not the same uniform bias.
        # This is essential when new tissue classes and organ types are added
        # via future stages (tile-classification, SSL on unlabelled data).
        # tissue_type=0 (Unknown) serves as the unconditioned/fallback path.
        _D = init_emb.shape[2]
        self.organ_bias = nn.Parameter(
            torch.zeros(NUM_TISSUE_TYPES, len(prompts_list), _D,
                        dtype=torch.float32))
        self.organ_dropout_p: float = 0.3   # prob of dropping to unconditioned during training
        self.current_tissue_type:  int = 0  # caller sets before each forward
        self.use_organ_bias: bool = False   # kept for ckpt compatibility; off by default
        # Per-forward adapter bias {prompt_idx → [D]}; set by caller, cleared after forward
        self._adapter_bias_per_idx: Optional[Dict[int, torch.Tensor]] = None

    def regularisation_loss(self, active_indices: Optional[List[int]] = None,
                            active_groups: Optional[List[str]] = None) -> torch.Tensor:
        """Encourage inter-class prompt orthogonality.

        Args:
            active_indices: if given, compute ortho only on these prompt indices
                (e.g. the prompts that were active in the current forward pass).
                This prevents unused prompts (e.g. nuclei prompts in tissue mode)
                from dominating the loss via their mutually-correlated embeddings.
            active_groups: optional group label for each active index.  When
                provided, orthogonality is applied only within a group, never
                across groups (e.g. tissue type vs tissue structure).
        """
        p = self.prompt_embeddings.mean(dim=1)           # [N, D]
        if active_indices is not None and len(active_indices) > 1:
            p = p[active_indices]                        # [K, D] — active only
            if active_groups is not None and len(active_groups) != len(active_indices):
                active_groups = None
        elif active_indices is not None and len(active_indices) <= 1:
            return torch.tensor(0.0, device=p.device)
        p_n   = F.normalize(p, dim=1)
        gram  = p_n @ p_n.t()
        diag  = torch.eye(gram.size(0), device=gram.device)
        off_diag = ~diag.bool()
        if active_groups is not None:
            group_eq = torch.tensor(
                [[g1 == g2 for g2 in active_groups] for g1 in active_groups],
                device=gram.device, dtype=torch.bool)
            pair_mask = off_diag & group_eq
            if not pair_mask.any():
                return torch.tensor(0.0, device=p.device)
            return gram.abs()[pair_mask].sum()
        return gram.abs()[off_diag].sum()

    def forward(self, text, input_boxes=None, device=None):
        if not isinstance(text, (list, tuple)) or not isinstance(text[0], str):
            return self.base_text_encoder(text, input_boxes=input_boxes,
                                          device=device)
        was_train = self.base_text_encoder.training
        self.base_text_encoder.eval()
        with torch.no_grad():
            base_mask, base_mem, base_emb = self.base_text_encoder(
                text, input_boxes=input_boxes, device=device)
            base_mem = base_mem.transpose(0, 1).contiguous()   # [B, L, D]
        if was_train:
            self.base_text_encoder.train()

        out_mem, out_mask = [], []
        for b, caption in enumerate(text):
            key = _normalise_key(caption)
            idx = self.key_to_idx.get(key)
            if idx is not None:
                out_mem.append(self.prompt_embeddings[idx])
                out_mask.append(self.prompt_masks[idx])
            else:
                out_mem.append(base_mem[b])
                out_mask.append(base_mask[b])

        out_mem  = torch.stack(out_mem,  dim=0)   # [B, L, D]
        out_mask = torch.stack(out_mask, dim=0)   # [B, L]

        # Apply adapter bias if set (Stage 2)
        if self._adapter_bias_per_idx is not None:
            out_mem_parts = list(out_mem.unbind(0))
            for b, caption in enumerate(text):
                key = _normalise_key(caption)
                idx = self.key_to_idx.get(key)
                if idx is not None and idx in self._adapter_bias_per_idx:
                    out_mem_parts[b] = out_mem_parts[b] + \
                        self._adapter_bias_per_idx[idx].to(out_mem_parts[b].dtype).unsqueeze(0)
            out_mem = torch.stack(out_mem_parts, dim=0)

        # Apply organ-conditioned bias (ttype=0 → nothing added = true unconditioned).
        # Per-class: organ_bias[organ, class_idx, :] → added to first token of prompt.
        # This allows "breast tumor" and "breast stroma" to learn distinct organ-specific
        # shifts, unlocking organ conditioning when more diverse data arrives in Stages 3/4.
        ttype = self.current_tissue_type if self.use_organ_bias else 0
        if self.use_organ_bias and self.training and random.random() < self.organ_dropout_p:
            ttype = 0   # drop to unconditioned path
        if self.use_organ_bias and ttype != 0:
            for b, caption in enumerate(text):
                idx = self.key_to_idx.get(_normalise_key(caption))
                if idx is not None:
                    out_mem[b, 0, :] = out_mem[b, 0, :] + self.organ_bias[ttype, idx, :]

        return out_mask, out_mem.transpose(0, 1).contiguous(), base_emb


# ─────────────────────────────────────────────────────────────────────────────
# Slide-adaptive cell morphology adapter  (Stage 2 — replaces NucleiPriorAdapter)
# ─────────────────────────────────────────────────────────────────────────────

class SlideMorphAdapter(nn.Module):
    """Density-based adapter: spatial cell density → prompt embedding bias.

    Uses a 4×4 spatial pool of the InstanSeg density map as input (16 values),
    preserving intra-image spatial variation (tumor vs stroma pockets) rather
    than losing everything in a global mean.

    Only outputs biases for tissue classes — nuclei prompts excluded.
    """

    NUM_MORPH: int = NUM_MORPH_FEATURES  # 16 (4×4 pool)
    POOL_SIZE: int = 4

    def __init__(self, num_tissue_classes: int, num_nuclei_classes: int, emb_dim: int):
        super().__init__()
        self.num_tissue = num_tissue_classes
        self.num_nuclei = num_nuclei_classes
        self.C_all      = num_tissue_classes   # tissue only — nuclei seg removed
        self.emb_dim    = emb_dim

        # Spatial density features → prompt embedding bias MLP
        # (zero-init last layer → starts as identity at step 0)
        _hidden = max(64, self.NUM_MORPH * 8)  # 128
        self.bias_mlp = nn.Sequential(
            nn.Linear(self.NUM_MORPH, _hidden),
            nn.GELU(),
            nn.Linear(_hidden, self.C_all * emb_dim),
        )
        nn.init.zeros_(self.bias_mlp[-1].weight)
        nn.init.zeros_(self.bias_mlp[-1].bias)

    def forward(self, density_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            density_feats: [16] — 4×4 spatial pool of the density map, on device
        Returns:
            biases: [C_tissue, D] — add to tissue class prompt embeddings
        """
        return self.bias_mlp(density_feats).view(self.C_all, self.emb_dim)


# ── Morphology-Relation Self-Supervision Projector ───────────────────────────

class SpatialRelationProjector(nn.Module):
    """Two lightweight heads on ViT block-31 spatial features.

    Operates on the same `_trunk_features_spatial` as `intra_image_contrast`
    but uses *precomputed cell-graph statistics* rather than class labels
    as the supervisory signal.  No class assumption — only morphology.

    1. ``local_morph_head`` — per-grid-cell morphology descriptor regression
    2. ``transition_head`` — boundary-awareness via pairwise grid differences
    """
    def __init__(self, in_dim: int = 1024, hidden_dim: int = 128,
                 morph_dim: int = 12):
        super().__init__()
        self.morph_dim = morph_dim
        self.local_morph_head = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, morph_dim, 1),
        )
        self.transition_head = nn.Sequential(
            nn.Conv2d(in_dim * 2, hidden_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

    def forward_local(self, trunk_feats: torch.Tensor) -> torch.Tensor:
        """trunk_feats [1, C, Hf, Wf] → [1, morph_dim, Hf, Wf]."""
        return self.local_morph_head(trunk_feats.float())

    def forward_transition(
        self, trunk_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict horizontal and vertical morphology-transition edges.

        Returns `[B, 1, H, W-1]` horizontal and `[B, 1, H-1, W]`
        vertical scores. Keeping directions separate prevents the old padded
        representation from mixing two different edge targets at each cell.
        """
        feats = trunk_feats.float()
        diff_h = torch.cat([feats[:, :, :, :-1], feats[:, :, :, 1:]], dim=1)
        diff_v = torch.cat([feats[:, :, :-1, :], feats[:, :, 1:, :]], dim=1)
        return self.transition_head(diff_h), self.transition_head(diff_v)


# ── Density-conditioned feature fusion (Stage 2/3) ──────────────────────────

class DensityFusion(nn.Module):
    """Cell-cue feature fusion for Stage 2/3.

    A feature-sized density map is expanded into seven biologically meaningful
    cue channels, encoded by a small CNN, and injected into the deep ViT spatial
    feature with a zero-initialised FiLM residual:

        fused = features * (1 + gamma(cell_cues)) + beta(cell_cues)

    Fresh fusion starts as an exact identity, so Stage 2 begins identical to the
    previous checkpoint and learns when cell morphology should modify features.
    """

    NUM_CELL_FEATURES: int = 7

    def __init__(self, feature_dim: int = 1024, hidden: int = 96,
                 max_gamma: float = 0.25, max_beta: float = 0.25):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.max_gamma = float(max_gamma)
        self.max_beta = float(max_beta)
        hidden = int(hidden)
        groups = 8 if hidden >= 8 and hidden % 8 == 0 else (
            4 if hidden >= 4 and hidden % 4 == 0 else 1)
        self.encoder = nn.Sequential(
            nn.Conv2d(self.NUM_CELL_FEATURES, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
        )
        self.gamma_head = nn.Conv2d(hidden, self.feature_dim, kernel_size=1)
        self.beta_head = nn.Conv2d(hidden, self.feature_dim, kernel_size=1)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    @staticmethod
    def _avg_pool_edge_safe(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        pad = int(kernel_size) // 2
        if pad <= 0:
            return F.avg_pool2d(x, kernel_size=kernel_size, stride=1)
        return F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"),
                            kernel_size=kernel_size, stride=1)

    @staticmethod
    def _grad_mag(x: torch.Tensor) -> torch.Tensor:
        dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
        g = torch.sqrt(dx.square() + dy.square() + 1e-6)
        return g / g.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)

    @classmethod
    def cue_stack(cls, density: torch.Tensor) -> torch.Tensor:
        d = density.float()
        if d.dim() == 2:
            d = d.unsqueeze(0).unsqueeze(0)
        elif d.dim() == 3:
            d = d.unsqueeze(1)
        elif d.dim() != 4:
            d = d.view(1, 1, *d.shape[-2:])
        if d.shape[1] != 1:
            d = d[:, :1]
        d = d.clamp(0.0, 1.0)
        local = cls._avg_pool_edge_safe(d, kernel_size=5)
        broader = cls._avg_pool_edge_safe(d, kernel_size=15)
        contrast = (local - broader).abs().clamp(0.0, 1.0)
        grad = cls._grad_mag(local).clamp(0.0, 1.0)
        high_cluster = (torch.sigmoid((local - 0.45) * 10.0) * d).clamp(0.0, 1.0)
        low_density = torch.sigmoid((0.22 - local) * 12.0).clamp(0.0, 1.0)
        lumen_boundary = cls._grad_mag(low_density).clamp(0.0, 1.0)
        return torch.cat([
            d, local, contrast, grad, high_cluster, low_density, lumen_boundary
        ], dim=1)

    def forward(self, features: torch.Tensor,
                density: torch.Tensor) -> torch.Tensor:
        cues = self.cue_stack(density).to(device=features.device,
                                          dtype=features.dtype)
        h = self.encoder(cues)
        gamma = self.max_gamma * torch.tanh(self.gamma_head(h))
        beta = self.max_beta * torch.tanh(self.beta_head(h))
        return features * (1.0 + gamma) + beta


class DensityGateNet(nn.Module):
    """Predicts a per-pixel gate (in (0, 1)) conditioned on SAM's mask
    logits, used to modulate the cell-density logit prior:

        bias = gate(pred_masks) * scale * log(density)

    Init: gate output ~= 0.01 everywhere.  Small but strictly nonzero so
    gradients flow through both the gate and the density signal; the model
    learns where and how strongly to trust density without paying a
    cold-start regression.

    Input: SAM decoder mask logits [N, Q, h, w] (max-pooled over queries).
    Output: gate map [N, 1, h, w] in (0, 1).
    """

    def __init__(self, hidden: int = 16, init_gate: float = 1e-2):
        super().__init__()
        # sigmoid^{-1}(init_gate) = ln(init_gate / (1 - init_gate))
        _init_bias = math.log(init_gate / (1.0 - init_gate))
        self.conv1 = nn.Conv2d(1, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv2.bias)
        # Final layer: small (nonzero) weights, bias such that sigmoid ~= init_gate.
        nn.init.normal_(self.conv3.weight, std=1e-2)
        nn.init.constant_(self.conv3.bias, _init_bias)

    def forward(self, mask_logits: torch.Tensor) -> torch.Tensor:
        """mask_logits: [N, Q, h, w] → gate: [N, 1, h, w] in (0, 1)."""
        x = mask_logits.amax(dim=1, keepdim=True).float()  # [N, 1, h, w]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return torch.sigmoid(self.conv3(x))


# ─────────────────────────────────────────────────────────────────────────────
# InstanSeg nuclei prior extractor  (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

_INSTANSEG_NUCLEI_SIGMA = 8.0   # uniform Gaussian smoothing for cell density

class InstanSegNucleiExtractor:
    """
    Runs InstanSeg on a PIL image and returns class-agnostic nuclei-density
    maps broadcast to all unified nuclei classes:
    [NUM_NUCLEI_CLASSES, H, W] float32, CPU.

    InstanSeg provides only instance segmentation (no cell-type labels).
    All channels receive the same Gaussian-smoothed binary cell-presence
    density, giving NucleiPriorAdapter a spatial nuclei signal without
    relying on unreliable cell-type classification.

    Caching design: only raw InstanSeg output (840×840 instance map, ~2.8 MB)
    is cached in an LRU buffer.  Density maps and morphological features are
    recomputed from the cached instance map on every hit — both are cheap
    CPU operations (Gaussian blur, regionprops, centroid splatting).  The
    expensive GPU inference is skipped entirely on cache hit.

    Memory budget with max_cache_size=256: ~716 MB host RAM (controlled,
    bounded).  Without this LRU cap the cache would grow to len(dataset)
    entries → multi-TB OOM.
    """

    _SIZE = 840
    _DEFAULT_MAX_CACHE = 256   # ~716 MB RAM at 840×840 int32 per entry

    def __init__(self, device: torch.device, max_cache_size: int = _DEFAULT_MAX_CACHE):
        from lazyslide.models.segmentation import Instanseg
        self.device    = device
        self.model     = Instanseg()
        self.model.to(device)
        self.model.model.eval()
        self.transform = self.model.get_transform()
        self.max_cache_size = max_cache_size
        # OrderedDict LRU: most-recently-used at end, least-recently-used at front.
        # Value: np.int32 [840, 840] instance map → ~2.82 MB per entry.
        from collections import OrderedDict as _OD
        self._inst_cache: _OD = _OD()

    def _cache_put(self, key: str, inst_small: np.ndarray):
        """LRU insert: evict oldest if at capacity, then put at end (most recent)."""
        if key in self._inst_cache:
            del self._inst_cache[key]   # move-to-end below
        elif len(self._inst_cache) >= self.max_cache_size:
            self._inst_cache.popitem(last=False)  # evict LRU (oldest)
        self._inst_cache[key] = inst_small        # insert at end (most recent)

    def _cache_get(self, key: str) -> Optional[np.ndarray]:
        """LRU fetch: move key to end (most recent) on hit."""
        v = self._inst_cache.get(key)
        if v is not None:
            self._inst_cache.move_to_end(key)
        return v

    @torch.no_grad()
    def extract(self, pil_image: PILImage.Image,
                cache_key: Optional[str] = None) -> torch.Tensor:
        """Returns [NUM_NUCLEI_CLASSES, H, W] float32 CPU tensor (density maps).

        If cache_key is provided, the raw 840×840 InstanSeg instance map is
        stored/retrieved in a bounded LRU cache.  Density maps are always
        recomputed from the cached instance map (cheap CPU ops).
        """
        H, W = pil_image.height, pil_image.width

        if cache_key is not None:
            _cached = self._cache_get(cache_key)
            if _cached is not None:
                inst_small = _cached
            else:
                inst_small = self._run_instanseg(pil_image)
                self._cache_put(cache_key, inst_small)
        else:
            inst_small = self._run_instanseg(pil_image)

        # --- Density maps (always recomputed — cheap CPU ops) ---
        inst = sk_resize(inst_small, (H, W), order=0,
                         preserve_range=True, anti_aliasing=False).astype(np.int32)

        cell_mask = (inst > 0).astype(np.float32)
        # Downsample large masks before gaussian_filter to avoid scipy OOM/crash.
        _max_dim = max(H, W)
        if _max_dim > 2048:
            _scale = 2048.0 / _max_dim
            _sh, _sw = max(1, int(H * _scale)), max(1, int(W * _scale))
            _small = sk_resize(cell_mask, (_sh, _sw), order=1,
                               preserve_range=True, anti_aliasing=True)
            _sigma_small = _INSTANSEG_NUCLEI_SIGMA * _scale
            _density_small = gaussian_filter(_small, sigma=_sigma_small)
            cell_density = sk_resize(_density_small, (H, W), order=1,
                                     preserve_range=True, anti_aliasing=True)
        else:
            cell_density = gaussian_filter(cell_mask, sigma=_INSTANSEG_NUCLEI_SIGMA)
        mx = cell_density.max()
        if mx > 0:
            cell_density /= mx

        prior = np.stack([cell_density] * NUM_NUCLEI_CLASSES, axis=0)  # [C, H, W]
        return torch.from_numpy(prior)   # CPU float32

    @torch.no_grad()
    def get_instance_map(self, pil_image: PILImage.Image,
                         cache_key: Optional[str] = None) -> np.ndarray:
        """Return full-resolution instance map [H, W] int32 (no Gaussian blur).

        Reuses the same InstanSeg forward + LRU cache as ``extract()``,
        so calling both in sequence costs only the (cheap) resize + regionprops.
        """
        H, W = pil_image.height, pil_image.width
        if cache_key is not None:
            _cached = self._cache_get(cache_key)
            inst_small = _cached if _cached is not None else self._run_instanseg(pil_image)
            if _cached is None:
                self._cache_put(cache_key, inst_small)
        else:
            inst_small = self._run_instanseg(pil_image)
        from skimage.transform import resize as sk_resize
        inst = sk_resize(inst_small, (H, W), order=0,
                         preserve_range=True, anti_aliasing=False)
        return inst.astype(np.int32)

    def _run_instanseg(self, pil_image: PILImage.Image) -> np.ndarray:
        """Run InstanSeg on the 840×840 resized image, return int32 instance map."""
        img_t = TVF.to_tensor(pil_image)   # [3, H, W] float32 in [0, 1]
        img_r = TVF.resize(img_t, [self._SIZE, self._SIZE], antialias=True)
        img_r = self.transform(img_r).unsqueeze(0).to(self.device)
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            result = self.model.segment(img_r)

        inst_small = result["instance_map"][0]
        if isinstance(inst_small, torch.Tensor):
            inst_small = inst_small.cpu().numpy()
        return inst_small.astype(np.int32)


class HematoxylinDensityExtractor:
    """Fast stain-derived cell-density proxy.

    This avoids InstanSeg during Stage 2/3 by projecting RGB optical density
    onto a hematoxylin-like direction.  It returns the same shape as the
    InstanSeg extractor: [NUM_NUCLEI_CLASSES, H, W] on CPU, where all channels
    share the same normalized density map.
    """

    def __init__(self, max_side: int = 2048, sigma: float = 8.0,
                 normalize: bool = True):
        self.max_side = int(max_side)
        self.sigma = float(sigma)
        self.normalize = bool(normalize)

    @staticmethod
    def _robust_rgb_normalize(rgb: np.ndarray) -> np.ndarray:
        x = rgb.astype(np.float32) / 255.0
        y = np.empty_like(x)
        for c in range(3):
            ch = x[..., c]
            lo, hi = np.percentile(ch, (1.0, 99.0))
            if hi <= lo + 1e-6:
                y[..., c] = ch
            else:
                y[..., c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        return y

    @staticmethod
    def _replace_outer_border(x: np.ndarray, border: int = 3) -> np.ndarray:
        b = int(border)
        if b <= 0 or x.shape[0] <= 2 * b or x.shape[1] <= 2 * b:
            return x
        y = x.copy()
        y[:b, :] = y[b:b + 1, :]
        y[-b:, :] = y[-b - 1:-b, :]
        y[:, :b] = y[:, b:b + 1]
        y[:, -b:] = y[:, -b - 1:-b]
        return y

    @torch.no_grad()
    def extract(self, pil_image: PILImage.Image,
                cache_key: Optional[str] = None,
                tiled: bool = False,
                coarse: bool = False) -> torch.Tensor:
        H, W = pil_image.height, pil_image.width
        rgb = np.asarray(pil_image.convert("RGB"))
        if self.max_side > 0 and max(H, W) > self.max_side:
            scale = float(self.max_side) / float(max(H, W))
            work_hw = (
                max(1, int(round(H * scale))),
                max(1, int(round(W * scale))),
            )
            rgb_work = sk_resize(rgb, (*work_hw, 3), order=1,
                                 preserve_range=True,
                                 anti_aliasing=True).astype(np.uint8)
        else:
            scale = 1.0
            rgb_work = rgb

        x = (self._robust_rgb_normalize(rgb_work)
             if self.normalize else rgb_work.astype(np.float32) / 255.0)
        od = -np.log(np.clip(x, 1e-6, 1.0))
        # Ruifrok-like hematoxylin vector.
        h_vec = np.asarray([0.650, 0.704, 0.286], dtype=np.float32)
        h_vec /= np.linalg.norm(h_vec) + 1e-8
        h = (od * h_vec.reshape(1, 1, 3)).sum(axis=-1)
        lo, hi = np.percentile(h, (2.0, 98.0))
        if hi > lo + 1e-6:
            h = np.clip((h - lo) / (hi - lo), 0.0, 1.0)
        else:
            h = np.zeros_like(h, dtype=np.float32)

        sigma = self.sigma * scale if scale != 1.0 else self.sigma
        if sigma > 0:
            h = gaussian_filter(h.astype(np.float32), sigma=max(0.5, sigma))
        mx = float(h.max()) if h.size else 0.0
        if mx > 0:
            h = h / mx
        h = self._replace_outer_border(h.astype(np.float32), border=3)
        if h.shape != (H, W):
            h = sk_resize(h, (H, W), order=1,
                          preserve_range=True,
                          anti_aliasing=True).astype(np.float32)
            h = self._replace_outer_border(h, border=3)
        prior = np.stack([h] * NUM_NUCLEI_CLASSES, axis=0).astype(np.float32)
        return torch.from_numpy(prior)


# ─────────────────────────────────────────────────────────────────────────────
# Tile-level classification head  (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

class TileClassifier(nn.Module):
    """Predict tile-level class presence from ViT encoder pooled features.

    Operates on the vision encoder's patch tokens (global average pool → MLP).
    Used in Stage 3 to leverage tile-level classification datasets (SkinCancer,
    MHIST, CRC-100K, BACH, LC25000, Osteo) as weak supervision for the encoder.

    The head is deliberately kept separate from the prompt embeddings so that
    classification and segmentation gradients don't interfere (avoids the
    well-known task-interference problem in multi-task learning).
    """

    def __init__(self, encoder_dim: int, num_tissue_classes: int,
                 num_nuclei_classes: int):
        super().__init__()
        self.num_tissue = num_tissue_classes
        self.num_nuclei = num_nuclei_classes
        self.num_classes = num_tissue_classes + num_nuclei_classes
        self.head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, self.num_classes),
        )

    def forward(self, vit_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vit_features: [B, N_patches, D] — ViT encoder output (patch tokens),
                          or [B, D] — already pooled features (from trunk hook).
        Returns:
            logits: [B, num_tissue + num_nuclei] — class-presence logits
        """
        if vit_features.ndim == 3:
            pooled = vit_features.mean(dim=1)   # global average pool over patches
        else:
            pooled = vit_features               # already pooled [B, D]
        return self.head(pooled)


class TissueTypeClassifier(nn.Module):
    """Predict organ/tissue type (breast, lung, colon, …) from ViT features.

    Lightweight MLP: LayerNorm → Linear → GELU → Linear → NUM_TISSUE_TYPES.
    Trained with cross-entropy on sample["tissue_type"].
    """

    def __init__(self, encoder_dim: int):
        super().__init__()
        _hidden = max(64, encoder_dim // 4)
        self.num_types = NUM_TISSUE_TYPES + 1  # +1 for "Other" catch-all
        self.head = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, _hidden),
            nn.GELU(),
            nn.Linear(_hidden, self.num_types),
        )

    def forward(self, vit_features: torch.Tensor) -> torch.Tensor:
        """vit_features: [1, N, D] or [1, D] → logits [1, NUM_TISSUE_TYPES]"""
        if vit_features.ndim == 3:
            pooled = vit_features.mean(dim=1)
        else:
            pooled = vit_features
        return self.head(pooled)


def _build_tile_classifier_heads(device: torch.device,
                                 ckpt: Optional[dict] = None,
                                 tag: str = "tilecls") -> nn.ModuleDict:
    """Create organ + tissue tile-classification heads and optionally load state."""
    encoder_dim = 1024
    organ_head = TissueTypeClassifier(encoder_dim=encoder_dim).to(device)
    tissue_head = TileClassifier(
        encoder_dim=encoder_dim,
        num_tissue_classes=NUM_TISSUE_CLASSES,
        num_nuclei_classes=NUM_NUCLEI_CLASSES,
    ).to(device)
    tile_classifier = nn.ModuleDict({
        "organ": organ_head,
        "tissue": tissue_head,
    })
    if ckpt is not None and ckpt.get("tile_cls_state"):
        _load_compat(tile_classifier, ckpt["tile_cls_state"],
                     tag=tag, device=device)
    _cls_params = sum(p.numel() for p in tile_classifier.parameters())
    print(f"[TileCls] Heads enabled ({_cls_params:,} trainable params)")
    return tile_classifier


# ── Gradient Reversal Layer (DANN-style adversarial training) ───────────────

class _GradientReversal(torch.autograd.Function):
    """Reverses gradient sign during backward — used to remove organ signal
    from late ViT features so segmentation is organ-invariant."""

    @staticmethod
    def forward(ctx, x, alpha: float = 1.0):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # Negate → gradient ascent on the adversarial head's loss
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, alpha)


# ─────────────────────────────────────────────────────────────────────────────
# BYOL self-supervised projection head  (Stage 3)
# ─────────────────────────────────────────────────────────────────────────────

class BYOLProjection(nn.Module):
    """BYOL-style projection + prediction head for self-supervised learning.

    Used in Stage 3 on unlabeled data.  Two augmented views of the same image
    are passed through the ViT encoder; the projector maps pooled features to
    a latent space, and the predictor tries to match the other view's projection
    (with stop-gradient).  Works with batch_size=1 — no negative pairs needed.

    Architecture:
        encoder(x) → projector → z  (target, stop-grad)
        encoder(x') → projector → z' → predictor → p'
        Loss: 2 - 2·cosine_sim(p', sg(z))
    """

    def __init__(self, encoder_dim: int, hidden_dim: int = 256,
                 out_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),          # LayerNorm works with batch_size=1
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),          # LayerNorm works with batch_size=1
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, D] — pooled ViT features
        Returns:
            z: [B, out_dim] — projection (target branch, stop-grad)
            p: [B, out_dim] — prediction (online branch)
        """
        z = self.projector(x)
        p = self.predictor(z)
        return z, p


class ContrastProj(nn.Module):
    """Tiny projection head used to stabilise intra-image contrastive loss.

    Maps the 1024-channel ViT block-31 feature map to a low-dim space
    (default 128) via two 1x1 convolutions.  Cosine similarity in this
    projected space is more well-behaved than directly on raw 1024-d
    ViT features.  Only ~290K params.  Stage 3 only.
    """

    def __init__(self, in_dim: int = 1024, hidden: int = 256, out_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, out_dim, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, in_dim, Hf, Wf]  →  [B, out_dim, Hf, Wf]
        return self.proj(x)


class BYOLPixelProjection(nn.Module):
    """Pixel-level BYOL head using 1×1 convolutions.

    Instead of pooling ViT features to [B, D] and comparing globally,
    this head operates on the full patch feature map [B, C, H, W],
    computing per-patch cosine similarity.  This teaches the ViT
    dense (pixel-level) representations — more transferable to
    segmentation than global-pooled BYOL.
    """

    def __init__(self, encoder_dim: int, hidden_dim: int = 256,
                 out_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Conv2d(encoder_dim, hidden_dim, 1),
            nn.GroupNorm(1, hidden_dim),        # GN(1,·) ≈ LayerNorm for conv
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, 1),
        )
        self.predictor = nn.Sequential(
            nn.Conv2d(out_dim, hidden_dim, 1),
            nn.GroupNorm(1, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, out_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, C, H, W] — ViT patch feature map (block-31 output)
        Returns:
            z: [B, out_dim, H, W] — projection (target branch, stop-grad)
            p: [B, out_dim, H, W] — prediction (online branch)
        """
        z = self.projector(x)
        p = self.predictor(z)
        return z, p


# ─────────────────────────────────────────────────────────────────────────────
# Loss and metrics
# ─────────────────────────────────────────────────────────────────────────────

def binary_loss(prob: torch.Tensor, gt_mask: torch.Tensor,
                class_id: int, focal_gamma: float = 0.0,
                unlabeled_repulsion_weight: float = 0.0,
                contrastive_weight: float = 0.0,
                contrastive_margin: float = 0.5,
                negative_boost: float = 1.0,
                area_loss_weight: float = 0.0,
                dice_weight: float = 1.0,
                bce_weight: float = 1.0,
                dataset_name: str = "") -> torch.Tensor:
    """Binary Dice + BCE + Area + contrastive loss for one class query.

    For class X:
      - gt == X           → positive
      - gt == Y (Y≠X,Y>0) → negative (annotated as another class)
      - gt == 0           → depends on dataset:
          * BCSS: ambiguous unlabeled/background — excluded from loss
          * Other tissue seg datasets: true background — negative

    BCSS is the only current tissue seg dataset where gt==0 is treated as
    unlabeled/ambiguous.  For all other tissue segmentation datasets, gt==0
    is true background and participates in Dice/BCE/repulsion as negative.

    ``area_loss_weight`` adds (sum(p)/sum(g) - 1)² — directly penalises
    the total predicted probability mass deviating from the annotation
    mass.  This is the strongest single constraint against over-prediction.

    ``unlabeled_repulsion_weight`` adds an additional p² penalty on
    gt==0 pixels, pushing predictions toward zero in unlabeled regions.
    Only applied for fully-annotated datasets.
    """
    g   = (gt_mask == class_id).float()
    p   = prob.squeeze(0).float()

    # ── Determine if gt==0 is true background or unlabeled ──────────
    is_fully_annotated = (dataset_name in _TISSUE_FULLY_ANNOTATED)

    if is_fully_annotated:
        # All pixels are labeled — standard binary loss over the full image.
        labeled = torch.ones_like(g)
        n_labeled = int(g.numel())
    else:
        # BCSS only: pixels with explicit nonzero annotation are labeled.
        # gt==0 is unlabeled/ambiguous, so do NOT penalise predictions there.
        labeled = (gt_mask > 0).float()
        n_labeled = int(labeled.sum().item())
        if n_labeled == 0:
            return torch.tensor(0.0, device=p.device, requires_grad=True)

    # ── Dice — computed only over labeled pixels ────────────────────
    if g.sum() > 0:
        inter = (p * g).sum()
        # p_labeled: predictions restricted to labeled region only.
        # Predictions in unlabeled areas (gt==0) do NOT inflate the
        # denominator for partially-annotated datasets.
        p_labeled_sum = (p * labeled).sum()
        dice  = 1.0 - (2.0 * inter + 1.0) / (p_labeled_sum + g.sum() + 1.0)
    else:
        dice = torch.tensor(0.0, device=p.device, requires_grad=True)

    # ── BCE — only on labeled pixels ────────────────────────────────
    # Symmetric BCE (no pos_weight).  Mean is taken over labeled pixels
    # only, so unlabeled regions (gt==0) contribute zero gradient.
    p_clamped = p.clamp(1e-6, 1.0 - 1e-6)
    bce_per_pixel = -(
        g * p_clamped.log()
        + negative_boost * (1.0 - g) * (1.0 - p_clamped).log()
    )
    if focal_gamma > 0.0:
        pt = g * p_clamped + (1.0 - g) * (1.0 - p_clamped)
        focal_weight = (1.0 - pt) ** focal_gamma
        bce_per_pixel = focal_weight * bce_per_pixel
    bce = (bce_per_pixel * labeled).sum() / max(n_labeled, 1)

    loss = dice_weight * dice + bce_weight * bce

    # ── Area Matching Loss ──────────────────────────────────────────
    # (sum(p) / sum(g) - 1)² — directly penalises the predicted
    # probability mass deviating from the annotation mass.  If the model
    # predicts 10× more area than GT this term contributes 81, providing
    # a hard global constraint against over-prediction.
    if area_loss_weight > 0 and g.sum() > 0:
        area_ratio = p.sum() / (g.sum() + 1e-8)
        area_loss = (area_ratio - 1.0) ** 2
        loss = loss + area_loss

    # ── Repulsion on true background pixels (fully-annotated ONLY) ──
    # Only applied when gt==0 is KNOWN to be background.
    # For partially-annotated datasets, gt==0 is unlabeled — skip.
    if unlabeled_repulsion_weight > 0 and is_fully_annotated:
        bg_pixels = (gt_mask == 0)
        if bg_pixels.sum() > 0:
            p_bg = p[bg_pixels]
            repulsion = (p_bg ** 2).mean()
            loss = loss + unlabeled_repulsion_weight * repulsion

    # ── Contrastive loss: target vs labeled non-target ──────────────
    if contrastive_weight > 0:
        p_target = p[g == 1]
        if is_fully_annotated:
            p_nontarget = p[g == 0]
        else:
            nontarget_mask = (gt_mask > 0) & (gt_mask != class_id)
            p_nontarget = p[nontarget_mask]
        if p_target.numel() > 0 and p_nontarget.numel() > 0:
            mean_pos = p_target.mean()
            mean_neg = p_nontarget.mean()
            contrastive = F.relu(contrastive_margin - (mean_pos - mean_neg))
            loss = loss + contrastive_weight * contrastive

    return loss


def _intra_image_contrastive_loss(
    feat_spatial: torch.Tensor,     # [1, C, Hf, Wf] trunk feature map (from hook, has grad)
    gt_mask: torch.Tensor,          # [H, W] int64 ground truth at original resolution
    class_id: int,
    temperature: float = 0.07,
    num_samples: int = 256,
    pixel_weight: Optional[torch.Tensor] = None,   # [H, W] or [Hf, Wf] float in [0, 1]
) -> torch.Tensor:
    """Intra-image pixel-level contrastive loss.

    Pulls same-class trunk features together, pushes different-class features
    apart WITHIN a single image.  Unlike BYOL (inter-image, view invariance),
    this directly supports segmentation: the ViT learns to produce spatially
    discriminative features that the mask decoder can separate.

    Only labeled pixels participate (gt>0).  Unlabeled pixels (gt==0) are
    excluded for partially-annotated datasets.

    Gradient flow: loss → trunk spatial features (_f_deep) → trunk weights.
    This is the SAME path the seg loss gradient takes (through mask decoder
    → trunk features → trunk weights), so the two gradients are aligned, not competing.
    """
    if feat_spatial.ndim != 4:
        return torch.tensor(0.0, device=feat_spatial.device)

    _, C, Hf, Wf = feat_spatial.shape
    gt_t = gt_mask.to(feat_spatial.device)

    # Downsample GT mask to feature resolution
    if gt_t.shape != (Hf, Wf):
        gt_t = gt_t.unsqueeze(0).unsqueeze(0).float()           # [1,1,H,W]
        gt_t = F.interpolate(gt_t, (Hf, Wf), mode='nearest')   # [1,1,Hf,Wf]
        gt_t = gt_t.squeeze(0).squeeze(0).long()                # [Hf, Wf]

    # Optional per-pixel confidence weight (used for cls-only pseudo-labels).
    if pixel_weight is not None:
        pw = pixel_weight.to(feat_spatial.device).float()
        if pw.shape != (Hf, Wf):
            pw = pw.unsqueeze(0).unsqueeze(0)
            pw = F.interpolate(pw, (Hf, Wf), mode='bilinear', align_corners=False)
            pw = pw.squeeze(0).squeeze(0)
        pw = pw.clamp(0.0, 1.0).reshape(-1)              # [Hf*Wf]
    else:
        pw = None

    # Masks for target class and other labeled classes
    target_mask = (gt_t == class_id)       # [Hf, Wf]
    labeled_mask = (gt_t > 0)              # [Hf, Wf]
    other_mask = labeled_mask & ~target_mask  # [Hf, Wf]

    n_target = target_mask.sum().item()
    n_other  = other_mask.sum().item()

    if n_target < 4 or n_other < 4:
        return torch.tensor(0.0, device=feat_spatial.device)

    # Sample pixel features (avoid OOM from all-pixel computation)
    feat_flat = feat_spatial[0].reshape(C, -1)  # [C, Hf*Wf]

    # Sample target and other pixel indices
    t_idx = torch.nonzero(target_mask.reshape(-1), as_tuple=True)[0]
    o_idx = torch.nonzero(other_mask.reshape(-1), as_tuple=True)[0]

    n_sample_t = min(num_samples // 2, n_target)
    n_sample_o = min(num_samples // 2, n_other)

    t_perm = torch.randperm(n_target, device=feat_spatial.device)[:n_sample_t]
    o_perm = torch.randperm(n_other, device=feat_spatial.device)[:n_sample_o]

    t_sel = t_idx[t_perm]
    o_sel = o_idx[o_perm]

    t_feats = feat_flat[:, t_sel]  # [C, n_sample_t]
    o_feats = feat_flat[:, o_sel]  # [C, n_sample_o]

    # Per-sample weights (1.0 when no pixel_weight provided).
    if pw is not None:
        t_w = pw[t_sel]                       # [n_sample_t]
        o_w = pw[o_sel]                       # [n_sample_o]
    else:
        t_w = torch.ones(n_sample_t, device=feat_spatial.device)
        o_w = torch.ones(n_sample_o, device=feat_spatial.device)

    # Normalise features
    t_feats = F.normalize(t_feats, dim=0, eps=1e-8)  # [C, n_sample_t]
    o_feats = F.normalize(o_feats, dim=0, eps=1e-8)  # [C, n_sample_o]

    # ── Pull same-class together: cos distance within target class ──
    if n_sample_t >= 2:
        t_sim = (t_feats.T @ t_feats) / temperature   # [n_t, n_t]
        # Mask out self-similarity
        diag = torch.eye(n_sample_t, dtype=torch.bool, device=t_sim.device)
        # Pairwise weights (both endpoints must be confident).
        pair_w = (t_w.unsqueeze(1) * t_w.unsqueeze(0))   # [n_t, n_t]
        pair_w = pair_w.masked_fill(diag, 0.0)
        # Loss = 1 - cos_sim (bounded [0, 2]).
        per_pair = (1.0 - t_sim * temperature)            # undo /temperature for loss scale
        denom = pair_w.sum().clamp_min(1e-6)
        t_loss = (per_pair * pair_w).sum() / denom
    else:
        t_loss = torch.tensor(0.0, device=feat_spatial.device)

    # ── Push different-class apart: target vs other ──
    cross_sim = (t_feats.T @ o_feats) / temperature   # [n_t, n_o]
    cross_pair_w = (t_w.unsqueeze(1) * o_w.unsqueeze(0))  # [n_t, n_o]
    push_per_pair = F.relu(cross_sim - 0.0)              # cos_sim > 0 → too similar
    push_denom = cross_pair_w.sum().clamp_min(1e-6)
    push_loss = (push_per_pair * cross_pair_w).sum() / push_denom

    return t_loss + push_loss


# ── Morphology-Relation Cell-Graph Builder ──────────────────────────────────

def _build_cell_graph_stats(
    inst_map: np.ndarray,          # [H, W] int32  instance ids (0=bg)
    grid_size: int = 16,
    max_cells: int = 512,
    intensity_image: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Extract per-grid-cell morphology statistics from an instance map.

    Returns a dict with:
      - ``local_stats_grid``  [G, G, S] float32  per-cell morphology stats
      - ``valid_mask``        [G, G] bool         cells-present mask

    The S=12 statistics are (per grid cell, pooled over contained cells):
      0. cell count
      1. mean area
      2. std area
      3. mean eccentricity
      4. std eccentricity
      5. mean solidity
      6. mean nearest-neighbour distance (px, normalised by grid cell side)
      7. clustering coefficient proxy (log(1+adjacent_pairs/cells))
      8. density (cells / grid_cell_area)
      9. local density contrast (vs 3×3 neighbourhood)
     10. mean hematoxylin intensity
     11. direction entropy (angular spread of cell pairs)

    Completely label-free — no class mapping, no pseudo-labelling.
    """
    import skimage.measure as skm
    H, W = inst_map.shape
    # --- Extract cell centroids and morphology ---
    properties = ['label', 'centroid', 'area', 'eccentricity', 'solidity']
    if intensity_image is not None:
        if intensity_image.shape[:2] != inst_map.shape:
            raise ValueError(
                f"intensity/instance shape mismatch: {intensity_image.shape} "
                f"vs {inst_map.shape}")
        properties.append('intensity_mean')
    props = skm.regionprops_table(
        inst_map, properties=tuple(properties), intensity_image=intensity_image)
    if len(props['label']) == 0:
        # No cells detected → return all-zeros grid
        return {
            'local_stats_grid': np.zeros((grid_size, grid_size, 12), dtype=np.float32),
            'valid_mask':       np.zeros((grid_size, grid_size), dtype=bool),
        }

    labels = props['label'].astype(np.int32)
    centroids = np.column_stack([props['centroid-1'], props['centroid-0']])  # [N, 2] (y, x)
    areas     = props['area'].astype(np.float32)
    eccs      = props['eccentricity'].astype(np.float32)
    solids    = props['solidity'].astype(np.float32)
    intensities = (
        props['intensity_mean'].astype(np.float32) / 255.0
        if 'intensity_mean' in props
        else np.zeros_like(areas, dtype=np.float32)
    )

    N = len(labels)
    if N > max_cells:
        keep = np.random.RandomState(42).choice(N, max_cells, replace=False)
        centroids = centroids[keep]
        areas     = areas[keep]
        eccs      = eccs[keep]
        solids    = solids[keep]
        intensities = intensities[keep]
        N = max_cells

    # --- Nearest-neighbour distance (approx via kd-tree) ---
    from scipy.spatial import cKDTree
    tree = cKDTree(centroids)
    nn_dists, _ = tree.query(centroids, k=min(3, N))
    if nn_dists.ndim == 1:
        nn_mean = nn_dists
    else:
        nn_mean = nn_dists[:, 1] if nn_dists.shape[1] >= 2 else nn_dists[:, 0]
    nn_mean = nn_mean.astype(np.float32)

    # --- Grid assignment ---
    cy = centroids[:, 0]               # row
    cx = centroids[:, 1]               # col
    gy = np.clip((cy / H * grid_size).astype(np.int32), 0, grid_size - 1)
    gx = np.clip((cx / W * grid_size).astype(np.int32), 0, grid_size - 1)
    gid = gy * grid_size + gx          # flat grid index [0, G*G-1]

    # --- Per-grid-cell statistics ---
    S = 12
    stats = np.zeros((grid_size * grid_size, S), dtype=np.float32)
    cell_side = (H / grid_size + W / grid_size) / 2.0  # avg grid cell side in px

    for i in range(grid_size * grid_size):
        mask = (gid == i)
        nc = mask.sum()
        if nc < 3:
            continue

        a  = areas[mask]
        e  = eccs[mask]
        s  = solids[mask]
        nd = nn_mean[mask]
        it = intensities[mask]

        stats[i, 0] = nc                                                    # count
        stats[i, 1] = a.mean()                                              # mean area
        stats[i, 2] = a.std()                                               # std area
        stats[i, 3] = e.mean()                                              # mean ecc
        stats[i, 4] = e.std()                                               # std ecc
        stats[i, 5] = s.mean()                                              # mean solidity
        stats[i, 6] = nd.mean() / max(cell_side, 1.0)                       # norm NN dist
        stats[i, 7] = np.log1p(float(nc)) if nc > 1 else 0.0                # clustering
        stats[i, 8] = nc / (cell_side * cell_side)                          # density
        stats[i, 10] = it.mean()
        # Direction entropy: angular spread of cell pairs within grid cell
        if nc >= 4:
            c_xy = centroids[mask]
            dx = c_xy[:, 0, None] - c_xy[None, :, 0]
            dy = c_xy[:, 1, None] - c_xy[None, :, 1]
            ang = np.arctan2(dy, dx)
            ang_flat = ang[np.triu_indices(nc, k=1)]
            hist, _ = np.histogram(ang_flat, bins=8, range=(-np.pi, np.pi))
            hist = hist.astype(np.float32) / max(hist.sum(), 1.0)
            stats[i, 11] = -(hist * np.log(hist + 1e-8)).sum() / np.log(8.0)  # norm entropy
        else:
            stats[i, 11] = 0.0

    # --- Local density contrast (3×3 neighbourhood) ---
    density_grid = stats[:, 8].reshape(grid_size, grid_size).copy()
    from scipy.ndimage import uniform_filter
    local_avg = uniform_filter(density_grid, size=3, mode='reflect')
    stats[:, 9] = np.abs(density_grid.ravel() - local_avg.ravel())

    # --- Valid mask ---
    valid = (stats[:, 0] >= 3).astype(bool)
    stats_grid = stats.reshape(grid_size, grid_size, S)

    return {
        'local_stats_grid': stats_grid.astype(np.float32),
        'valid_mask':       valid.reshape(grid_size, grid_size),
    }


# ── Morphology-Relation Self-Supervision losses (Stage 3) ────────────────────

def _local_morphology_regression_loss(
    feat_spatial: torch.Tensor,
    local_stats: torch.Tensor,     # [G, G, S] float  precomputed graph stats
    valid_mask: torch.Tensor,      # [G, G] bool       cells-present mask
    projector: nn.Module,
) -> torch.Tensor:
    """Predict per-grid-cell morphology statistics from ViT spatial features.

    No class label needed — the task is pure morphology-descriptor regression.
    Only grid cells with sufficient cells (valid_mask) contribute to the loss.
    """
    _, _, Hf, Wf = feat_spatial.shape
    G = local_stats.shape[0]
    if Hf != G or Wf != G:
        return torch.tensor(0.0, device=feat_spatial.device)

    pred = projector.forward_local(feat_spatial)                 # [1, S, G, G]
    stats = local_stats.to(pred.device).float()
    valid = valid_mask.to(pred.device).bool()
    if valid.sum() < 2:
        return pred.sum() * 0.0
    valid_stats = stats[valid]
    # Stabilize heterogeneous units (pixel area, density, eccentricity, etc.).
    valid_stats = valid_stats.clone()
    valid_stats[:, 0:3] = torch.log1p(valid_stats[:, 0:3].clamp_min(0))
    valid_stats[:, 6:10] = torch.log1p(valid_stats[:, 6:10].clamp_min(0))
    mean = valid_stats.mean(dim=0, keepdim=True)
    std = valid_stats.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    normalized = (valid_stats - mean) / std
    tgt_grid = torch.zeros_like(stats)
    tgt_grid[valid] = normalized
    tgt = tgt_grid.permute(2, 0, 1).unsqueeze(0)
    mask = valid_mask.unsqueeze(0).unsqueeze(0).float().to(pred.device)  # [1, 1, G, G]

    diff = F.smooth_l1_loss(pred * mask, tgt * mask, reduction='none')
    n_valid = (mask.sum() * pred.shape[1]).clamp_min(1.0)
    return diff.sum() / n_valid


def _morphology_transition_loss(
    feat_spatial: torch.Tensor,
    local_stats: torch.Tensor,     # [G, G, S] float
    valid_mask: torch.Tensor,      # [G, G]  bool
    projector: nn.Module,
) -> torch.Tensor:
    """Predict whether adjacent grid cells undergo a morphology transition.

    Only high-confidence pairs (very similar or very different cell-graph
    structure) are supervised; ambiguous middle zone is ignored.
    """
    G = local_stats.shape[0]
    _, _, Hf, Wf = feat_spatial.shape
    if Hf != G or Wf != G:
        return torch.tensor(0.0, device=feat_spatial.device)

    stats = local_stats.to(feat_spatial.device).float()
    valid = valid_mask.to(feat_spatial.device).bool()
    if valid.sum() < 4:
        return feat_spatial.sum() * 0.0

    valid_stats = stats[valid].clone()
    valid_stats[:, 0:3] = torch.log1p(valid_stats[:, 0:3].clamp_min(0))
    valid_stats[:, 6:10] = torch.log1p(valid_stats[:, 6:10].clamp_min(0))
    mean = valid_stats.mean(dim=0, keepdim=True)
    std = valid_stats.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-4)
    normalized = torch.zeros_like(stats)
    normalized[valid] = (valid_stats - mean) / std

    dist_h = (normalized[:, 1:] - normalized[:, :-1]).square().mean(dim=-1).sqrt()
    dist_v = (normalized[1:] - normalized[:-1]).square().mean(dim=-1).sqrt()
    valid_h = valid[:, 1:] & valid[:, :-1]
    valid_v = valid[1:] & valid[:-1]
    valid_dist = torch.cat([dist_h[valid_h], dist_v[valid_v]])
    if valid_dist.numel() < 8:
        return feat_spatial.sum() * 0.0

    # Per-image quantiles keep both easy-same and clear-transition edges even
    # when absolute morphology scales differ substantially between organs.
    q_low, q_high = torch.quantile(
        valid_dist.detach(), torch.tensor([0.35, 0.65], device=valid_dist.device))
    selected_h = valid_h & ((dist_h <= q_low) | (dist_h >= q_high))
    selected_v = valid_v & ((dist_v <= q_low) | (dist_v >= q_high))
    target_h = (dist_h >= q_high).float()
    target_v = (dist_v >= q_high).float()

    pred_h, pred_v = projector.forward_transition(feat_spatial)
    pred_h = pred_h[0, 0]
    pred_v = pred_v[0, 0]
    losses = []
    if selected_h.any():
        losses.append(F.binary_cross_entropy(pred_h[selected_h], target_h[selected_h]))
    if selected_v.any():
        losses.append(F.binary_cross_entropy(pred_v[selected_v], target_v[selected_v]))
    return torch.stack(losses).mean() if losses else feat_spatial.sum() * 0.0


def _cls_only_mil_losses(
    tissue_prob_maps: torch.Tensor,
    tile_label: torch.Tensor,
    topk_ratio: float = CLS_ONLY_TOPK_RATIO,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute non-contrastive cls-only weak losses from tile-level labels.

    Args:
        tissue_prob_maps: [C_t, H, W] probability maps for tissue classes 1..C_t.
        tile_label:       [NUM_TISSUE_CLASSES + NUM_NUCLEI_CLASSES] soft class ratios.
        topk_ratio:       ratio used for top-k MIL pooling.

    Returns:
        mil_bce:     BCE on top-k pooled logits vs tissue presence labels.
        absent_push: mean pooled probability on absent tissue classes.
    """
    dev = tissue_prob_maps.device

    if tissue_prob_maps.ndim != 3:
        z = torch.tensor(0.0, device=dev)
        return z, z
    if tile_label.ndim != 1 or tile_label.numel() < NUM_TISSUE_CLASSES:
        z = torch.tensor(0.0, device=dev)
        return z, z

    # Tissue labels exclude background: classes 1..NUM_TISSUE_CLASSES-1
    tgt = (tile_label[1:NUM_TISSUE_CLASSES] > 0).float()
    C_t, H, W = tissue_prob_maps.shape
    if tgt.numel() != C_t:
        z = torch.tensor(0.0, device=dev)
        return z, z

    flat = tissue_prob_maps.reshape(C_t, -1).clamp(1e-6, 1.0 - 1e-6)
    k = max(1, int(flat.shape[1] * float(topk_ratio)))
    topk = torch.topk(flat, k=k, dim=1, largest=True).values
    pooled = topk.mean(dim=1)
    logits = torch.logit(pooled.clamp(1e-4, 1.0 - 1e-4))
    mil_bce = F.binary_cross_entropy_with_logits(logits, tgt)

    absent_mask = (1.0 - tgt)
    if absent_mask.sum() > 0:
        absent_push = (pooled * absent_mask).sum() / absent_mask.sum().clamp_min(1.0)
    else:
        absent_push = torch.tensor(0.0, device=dev)

    return mil_bce, absent_push


def _cls_only_entropy_loss(
    tissue_prob_maps: torch.Tensor,
) -> torch.Tensor:
    """Pixel-wise binary entropy averaged over tissue classes and pixels.

    Encourages confident predictions on cls-only images (low-cost regulariser
    that, combined with MIL+absent-suppression, sharpens decision boundaries
    without needing pixel GT).

    Args:
        tissue_prob_maps: [C_t, H, W] sigmoid-style probabilities per class.
    Returns:
        Scalar entropy loss (always >= 0).
    """
    if tissue_prob_maps.ndim != 3:
        return torch.tensor(0.0, device=tissue_prob_maps.device)
    p = tissue_prob_maps.clamp(1e-6, 1.0 - 1e-6)
    h = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
    return h.mean()


def _bin_metrics(pred_bin: np.ndarray, gt_bin: np.ndarray) -> dict:
    """Compute binary metrics.

    All pixels are labeled — non-target pixels are negatives (background).
    - TP: predicted X where GT says X.
    - FN: missed labeled pixels (GT says X, pred says 0 or other).
    - FP: predicted X where GT says anything else (0, other class, unlabeled).

    Returns dice, iou, prec, rec.
    """
    tp = float((pred_bin & gt_bin).sum())
    fp = float((pred_bin & ~gt_bin).sum())
    fn = float((~pred_bin & gt_bin).sum())
    denom_dice = 2 * tp + fp + fn
    denom_iou  = tp + fp + fn
    eps = 1e-8
    return {
        "dice":  (2 * tp + eps) / (denom_dice + eps),
        "iou":   (tp + eps) / (denom_iou + eps),
        "prec":  (tp + eps) / (tp + fp + eps),
        "rec":   (tp + eps) / (tp + fn + eps),
    }


# Datasets where label 0 conflates true background with unlabelled regions
# (e.g. BCSS: unannotated tissue and pure background are both stored as 0).
# For these datasets we skip class 0 in dice/iou/prec/rec so unlabelled
# pixels don't get scored as if they were genuine background.
_DS_SKIP_BG_IN_METRICS: frozenset = frozenset({"BCSS"})


def compute_per_class_metrics(
    pred_multi: np.ndarray,        # [H, W] argmax prediction
    gt_mask: np.ndarray,           # [H, W] ground truth
    class_dict: dict,              # {id: name}
    prefix: str,                   # "tissue" or "nuclei"
    dataset_name: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class Dice/IoU/Prec/Rec.

    For class X, ALL other pixels (gt==0, gt==Y≠X) are negatives.
    This directly measures whether prediction matches the labeled area.

    Class 0 (background) is included by default — except for datasets in
    ``_DS_SKIP_BG_IN_METRICS`` where label 0 is ambiguous (e.g. BCSS, where
    0 means "unlabelled OR background").
    """
    skip_bg = dataset_name in _DS_SKIP_BG_IN_METRICS
    result = {}
    for cid, cname in class_dict.items():
        if skip_bg and cid == 0:
            continue   # ambiguous background — don't score it
        if gt_mask is None:
            continue
        valid = gt_mask >= 0
        p = (pred_multi == cid) & valid
        g = (gt_mask == cid) & valid
        if g.sum() == 0:
            continue   # class absent from gt for this image — skip
        m = _bin_metrics(p, g)
        m["f1"] = (2 * m["prec"] * m["rec"] + 1e-8) / \
                    (m["prec"] + m["rec"] + 1e-8)
        result[f"{prefix}_{cid}_{cname}"] = m
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model assembly helpers
# ─────────────────────────────────────────────────────────────────────────────

def _all_prompt_texts_dict() -> Dict[str, str]:
    d = {}
    for cid, txt in TISSUE_PROMPT_TEXTS.items():
        d[f"tissue_{cid}"] = txt
    for cid, txt in NUCLEI_PROMPT_TEXTS.items():
        d[f"nuclei_{cid}"] = txt
    return d


def build_stage1_model(device: torch.device):
    """
    Build the Stage-1 model used by the final A5 checkpoint lineage.

    Effective gradient updates: rank-8 vision-encoder LoRA and class-specific
    prompt embeddings.  The base language encoder is included in the prompt
    optimizer group but known string prompts evaluate it under no_grad and then
    replace its output with the learned embedding lookup, so it is not updated.
    The geometry
    encoder, DETR encoder/decoder, mask decoder, and original vision weights
    remain frozen.  organ_bias is explicitly frozen below.
    """
    model = build_sam3_image_model(
        device=device.type, compile=False,
        load_from_HF=True, bpe_path=BPE_PATH, eval_mode=True,
    )
    model.to(device).to(torch.float32)  # cast to float32; autocast handles mixed precision
    for p in model.parameters():
        p.requires_grad = False

    prompt_enc = PromptTunedTextEncoder(
        base_text_encoder=model.backbone.language_backbone,
        all_prompt_texts=_all_prompt_texts_dict(),
        device=device,
    ).to(device)
    model.backbone.language_backbone = prompt_enc

    # Inject LoRA immediately — without adaptive image-encoder weights the model
    # cannot localise small structures (tissue patches / nuclei) and all-positive
    # collapse is guaranteed from the first epoch.
    # lora_cfg = LoRAConfig(
    #     rank=4, alpha=8, dropout=0.05,
    #     apply_to_vision_encoder=True,
    #     apply_to_text_encoder=False,
    #     apply_to_geometry_encoder=False,
    #     apply_to_detr_encoder=False,
    #     apply_to_detr_decoder=False,
    #     apply_to_mask_decoder=False,
    # )
    lora_cfg = LoRAConfig(
        rank=8, alpha=16, dropout=0.05,
        apply_to_vision_encoder=True,
        apply_to_text_encoder=False,
        apply_to_geometry_encoder=False,
        apply_to_detr_encoder=False,
        apply_to_detr_decoder=False,
        apply_to_mask_decoder=False,
    )
    model = apply_lora_to_model(model, lora_cfg)
    model = model.to(device)  # LoRA params default to CPU — move back to device
    # apply_lora freezes everything and then enables only LoRA parameters.
    # Re-enable the complete prompt wrapper to match existing checkpoints.
    # Important: known string prompts run base_text_encoder under no_grad in
    # PromptTunedTextEncoder.forward and use prompt_embeddings directly.  Thus
    # the language-backbone tensors enter the optimizer but receive no gradient;
    # effective Stage-1 prompt adaptation is the class embedding table only.
    for p in prompt_enc.parameters():
        p.requires_grad = True
    # Stage 1 has no organ-type supervision — freeze organ_bias so it
    # stays at zero.  Later stages (2+) re-enable it when tile-cls data
    # provides meaningful organ labels.
    prompt_enc.organ_bias.requires_grad = False

    _cp = count_parameters(model)
    print(f"[Stage 1] Total params: {_cp['total_parameters']:,}  Trainable: {_cp['trainable_parameters']:,}")
    return model, prompt_enc, lora_cfg


def build_bare_medsam3_model(device: torch.device):
    """Original MedSAM3 checkpoint, without PathXAI LoRA or prompt tuning."""
    model = build_sam3_image_model(
        device=device.type, compile=False,
        load_from_HF=True, bpe_path=BPE_PATH, eval_mode=True,
    )
    model.to(device).to(torch.float32)
    for p in model.parameters():
        p.requires_grad = False
    prompt_enc = model.backbone.language_backbone
    _cp = count_parameters(model)
    print(f"[Bare MedSAM3] Total params: {_cp['total_parameters']:,}  Trainable: {_cp['trainable_parameters']:,}")
    return model, prompt_enc, None


# ── Helper: load state_dict skipping shape-mismatched keys ───────────────────
def _load_compat(target: nn.Module, src_dict: dict, tag: str = "",
                 device: Optional[torch.device] = None) -> int:
    """Load ``src_dict`` into *target*, skipping keys whose shape differs.

    Returns the number of keys successfully loaded.
    """
    _cur = target.state_dict()
    _compat = {}
    _skipped = []
    for k, v in src_dict.items():
        if k in _cur and _cur[k].shape == v.shape:
            _compat[k] = v.to(device) if device else v
        else:
            _skipped.append(
                f"    {k}: ckpt.shape={tuple(v.shape)}, "
                f"cur.shape={tuple(_cur[k].shape) if k in _cur else 'N/A'}")
    if _compat:
        target.load_state_dict(_compat, strict=False)
    if _skipped:
        print(f"  [_load_compat/{tag}] Skipped {len(_skipped)} shape-mismatched key(s):")
        for _s in _skipped:
            print(_s)
    print(f"  [_load_compat/{tag}] Loaded {len(_compat)}/{len(src_dict)} keys")
    return len(_compat)


def _apply_decoder_lora(model: nn.Module, rank: int = 8, alpha: int = 16,
                        dropout: float = 0.05) -> int:
    """Apply LoRA to transformer.decoder + segmentation_head ONLY.
    Preserves existing ViT LoRA and other trainable params.
    Returns number of LoRA modules added.
    """
    # Save existing requires_grad state for ALL currently-trainable params
    _saved_grad = {}
    for n, p in model.named_parameters():
        _saved_grad[n] = p.requires_grad

    # Apply LoRA only to decoder components
    _decoder_cfg = LoRAConfig(
        rank=rank, alpha=alpha, dropout=dropout,
        apply_to_vision_encoder=False,
        apply_to_text_encoder=False,
        apply_to_geometry_encoder=False,
        apply_to_detr_encoder=False,
        apply_to_detr_decoder=True,     # transformer.decoder layers
        apply_to_mask_decoder=False,     # handled manually below
    )
    model = apply_lora_to_model(model, _decoder_cfg)
    model = model.to(device)  # LoRA params default to CPU — move back to device

    # Also apply LoRA to segmentation_head Linear/Conv layers manually.
    # segmentation_head contains pixel_decoder (Conv2d) and mask_predictor (MLP).
    # We apply LoRA to their Linear layers only (Conv2d not supported by LoRA).
    _n_seg = 0
    for _nm, _mod in model.named_modules():
        if 'segmentation_head' not in _nm:
            continue
        if isinstance(_mod, nn.Linear):
            _parent = model
            _parts = _nm.split('.')
            for _p in _parts[:-1]:
                _parent = getattr(_parent, _p)
            _lora = LoRALinear(_mod, rank=rank, alpha=alpha, dropout=dropout)
            setattr(_parent, _parts[-1], _lora)
            _n_seg += 1
    print(f"  [DecoderLoRA] segmentation_head LinearLoRA: {_n_seg} modules")

    # Restore ALL previously-trainable params (ViT LoRA, prompt_enc, adapter, etc.)
    for n, p in model.named_parameters():
        if n in _saved_grad and _saved_grad[n]:
            p.requires_grad = True

    # Count new decoder LoRA params
    _dec_lora = sum(1 for n, _ in model.named_parameters()
                    if ('transformer.decoder' in n or 'segmentation_head' in n)
                    and 'lora' in n.lower())
    print(f"  [DecoderLoRA] Decoder LoRA params: {_dec_lora}")
    return _dec_lora


def build_stage2_model(prev_ckpt_path: str, device: torch.device):
    """
    Load Stage-1 checkpoint, add SlideMorphAdapter.
    Optimizer groups include SlideMorphAdapter + vision LoRA + the complete
    PromptTunedTextEncoder.  For known prompts, only prompt embeddings and any
    enabled prompt bias can receive gradients.  This stage is not part of the
    final A5 lineage.
    """
    model, prompt_enc, lora_cfg = build_stage1_model(device)

    # Load Stage-1 lora + prompt state
    ckpt = torch.load(prev_ckpt_path, map_location=device, weights_only=False)
    if ckpt.get("lora_state"):
        sd = model.state_dict()
        sd.update({k: v.to(device) for k, v in ckpt["lora_state"].items()})
        model.load_state_dict(sd)
    _load_compat(prompt_enc, ckpt["prompt_enc"], tag="stage2_init", device=device)
    print(f"[Stage 2] Loaded Stage-1 weights from {prev_ckpt_path}")

    # Restore the checkpoint's prompt-wrapper optimizer state.  Known prompts
    # still bypass language-backbone gradients in PromptTunedTextEncoder.forward.
    for p in prompt_enc.parameters():
        p.requires_grad = True

    # Infer embedding dim from prompt_enc
    emb_dim = prompt_enc.prompt_embeddings.shape[-1]
    adapter = SlideMorphAdapter(
        num_tissue_classes=NUM_TISSUE_CLASSES,
        num_nuclei_classes=NUM_NUCLEI_CLASSES,
        emb_dim=emb_dim,
    ).to(device)

    # Load adapter state from checkpoint if available.
    # This is critical when resuming Stage 3 from a Stage 2 checkpoint —
    # without it the adapter is zero-initialised and the model starts worse
    # than the checkpoint it was loaded from.
    if ckpt.get("adapter_state"):
        _load_compat(adapter,
                     ckpt["adapter_state"], tag="stage2_adapter", device=device)

    _cp = count_parameters(model)
    _adapter_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"[Stage 2] Total params: {_cp['total_parameters']:,}  "
          f"Trainable (with SlideMorphAdapter): {_cp['trainable_parameters'] + _adapter_params:,}")
    return model, prompt_enc, adapter, lora_cfg


# ── Former Stage 3: tile classifier helper ─────────────────────────────────

def build_tile_classifier_model(prev_ckpt_path: str, device: torch.device):
    """Load Stage-1 or Stage-2 checkpoint, add TileClassifier.

    Optimizer groups before CLI freeze handling: TileClassifier + vision LoRA +
    PromptTunedTextEncoder, plus SlideMorphAdapter when present.  The effective
    known-prompt gradient reaches prompt embeddings, not the base text encoder.
    """
    ckpt = torch.load(prev_ckpt_path, map_location=device, weights_only=False)
    prev_stage = int(ckpt.get("stage", 1))
    print(f"[TileCls init] Previous stage={prev_stage}, ckpt={prev_ckpt_path}")

    if prev_stage >= 2:
        model, prompt_enc, adapter, lora_cfg = build_stage2_model(
            prev_ckpt_path, device)
    else:
        model, prompt_enc, lora_cfg = build_stage1_model(device)
        adapter = None
        _m = model.module if hasattr(model, "module") else model
        if ckpt.get("lora_state"):
            sd = _m.state_dict()
            sd.update({k: v.to(device) for k, v in ckpt["lora_state"].items()})
            _m.load_state_dict(sd)
        _load_compat(prompt_enc, ckpt["prompt_enc"], tag="stage3_init", device=device)
        for p in prompt_enc.parameters():
            p.requires_grad = True

    tile_classifier = _build_tile_classifier_heads(
        device, ckpt=ckpt, tag="stage3_tilecls")

    _cp = count_parameters(model)
    _cls_params = sum(p.numel() for p in tile_classifier.parameters())
    print(f"[TileCls init] Total params: {_cp['total_parameters']:,}  "
          f"Trainable (with dual classifier): {_cp['trainable_parameters'] + _cls_params:,}")
    return model, prompt_enc, adapter, lora_cfg, tile_classifier


# ── Stage 3: former Stage 4 (cls/MIL + intra-image contrastive) ─────────────

def build_stage3_model(prev_ckpt_path: str, device: torch.device):
    """Add tile-level cls/MIL and optional SSL/contrastive heads.

    The starting modules depend on the checkpoint stage.  Before CLI freeze
    handling this builder enables vision LoRA, the prompt wrapper,
    classification heads, and projection heads; an adapter exists only when the
    input checkpoint already contains Stage 2.  Final A5 starts from Stage 1,
    has no adapter, freezes the whole prompt/text wrapper, and trains vision
    LoRA plus the enabled classification/contrastive heads.
    """
    ckpt = torch.load(prev_ckpt_path, map_location=device, weights_only=False)
    prev_stage = int(ckpt.get("stage", 1))
    print(f"[Stage 3] Previous stage={prev_stage}, ckpt={prev_ckpt_path}")

    if prev_stage >= 3:
        model, prompt_enc, adapter, lora_cfg, tile_classifier = build_tile_classifier_model(prev_ckpt_path, device)
    elif prev_stage >= 2:
        model, prompt_enc, adapter, lora_cfg = build_stage2_model(prev_ckpt_path, device)
        tile_classifier = _build_tile_classifier_heads(
            device, ckpt=ckpt, tag="stage3_tilecls")
    else:
        model, prompt_enc, lora_cfg = build_stage1_model(device)
        adapter = None
        _m = model.module if hasattr(model, "module") else model
        if ckpt.get("lora_state"):
            sd = _m.state_dict()
            sd.update({k: v.to(device) for k, v in ckpt["lora_state"].items()})
            _m.load_state_dict(sd)
        _load_compat(prompt_enc, ckpt["prompt_enc"], tag="stage3_init", device=device)
        for p in prompt_enc.parameters():
            p.requires_grad = True
        # Stage-1 checkpoints usually do not carry cls heads.  Stage 3 should
        # still create fresh heads so A1/A2/A3... can train tile classification.
        tile_classifier = _build_tile_classifier_heads(
            device, ckpt=ckpt, tag="stage3_tilecls_st1")

    encoder_dim = 1024
    ssl_head = BYOLProjection(encoder_dim=encoder_dim).to(device)
    pixel_ssl_head = BYOLPixelProjection(encoder_dim=encoder_dim).to(device)
    if ckpt.get("ssl_head_state"):
        _load_compat(ssl_head, ckpt["ssl_head_state"], tag="stage3_ssl", device=device)
    if ckpt.get("pixel_ssl_state"):
        _load_compat(pixel_ssl_head, ckpt["pixel_ssl_state"], tag="stage3_pixel_ssl", device=device)

    _cp = count_parameters(model)
    _ssl_params = sum(p.numel() for p in ssl_head.parameters())
    _pssl_params = sum(p.numel() for p in pixel_ssl_head.parameters())
    print(f"[Stage 3] Total params: {_cp['total_parameters']:,}  "
          f"Trainable (with BYOLProjection): {_cp['trainable_parameters'] + _ssl_params + _pssl_params:,}")
    return model, prompt_enc, adapter, lora_cfg, tile_classifier, ssl_head, pixel_ssl_head


# ─────────────────────────────────────────────────────────────────────────────
# Multi-task training step  (shared across stages)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _vit_pooled_features(model, pil_image: PILImage.Image,
                         transform, device: torch.device,
                         early: bool = False) -> torch.Tensor:
    """Extract global-average-pooled features [1, D] via forward_image().
    Decorated with @no_grad — features are detached, safe for readout heads.
    early=True → shallow ViT features (organ cls).
    early=False → deep ViT features (tissue cls)."""
    return _vit_features_impl(model, pil_image, transform, device,
                              detach=True, early=early)


def _vit_features_grad(model, pil_image: PILImage.Image,
                       transform, device: torch.device,
                       early: bool = False) -> torch.Tensor:
    """Same as _vit_pooled_features but WITH gradient tracking.
    Used by BYOL online branch so contrastive loss flows back into ViT.
    early=True → shallow ViT features. early=False → deep ViT features."""
    return _vit_features_impl(model, pil_image, transform, device,
                              detach=False, early=early)


def _vit_features_impl(model, pil_image: PILImage.Image,
                       transform, device: torch.device,
                       detach: bool, pool: bool = True,
                       early: bool = False) -> torch.Tensor:
    """Shared implementation: resize → transform → forward_image → [pool].

    Args:
        pool: if True (default), returns global-average-pooled [B, D].
              if False, returns raw patch feature map [B, C, H, W].
        early: if True, returns shallow (first) ViT layer features.
               if False (default), returns deep (last) ViT layer features.
    """
    _raw = model.module if hasattr(model, "module") else model
    pil_rs = pil_image.resize((SAM3_RESOLUTION, SAM3_RESOLUTION), PILImage.BILINEAR)
    dp = _create_datapoint(pil_rs, "background")
    dp = transform(dp)
    # dp.images are SAMImage objects after transform — .data gives the tensor.
    tensors = [img.data for img in dp.images]
    img_t = torch.stack(tensors).to(device)
    if detach:
        img_t = img_t.detach()
    try:
        out = _raw.backbone.forward_image(img_t)
        if isinstance(out, dict) and "vision_features" in out:
            fl = out["vision_features"]
            if isinstance(fl, (list, tuple)) and len(fl) > 0:
                feats = fl[0] if early else fl[-1]
                if feats.ndim == 4:
                    if pool:
                        return feats.mean(dim=(2, 3)).float()   # [B, D]
                    else:
                        return feats.float()                    # [B, C, H, W]
    except Exception:
        pass
    if pool:
        return torch.zeros(1, 1024, device=device)
    else:
        # Return zero patch map matching expected spatial size: 1008 → 63 patches
        return torch.zeros(1, 1024, 63, 63, device=device)


def _vit_patch_features_grad(model, pil_image: PILImage.Image,
                             transform, device: torch.device) -> torch.Tensor:
    """Extract ViT patch feature map [B, C, H, W] WITH gradient tracking.
    Used by pixel-level BYOL online branch."""
    return _vit_features_impl(model, pil_image, transform, device,
                              detach=False, pool=False)


def _make_byol_views(pil_image: PILImage.Image) -> Tuple[PILImage.Image, PILImage.Image]:
    """Create two augmented views for BYOL contrastive learning.
    Standard SimCLR-style augmentations applied independently to each view."""
    def _augment(img: PILImage.Image) -> PILImage.Image:
        w, h = img.size
        # RandomResizedCrop: scale 0.7–1.0
        scale = random.uniform(0.7, 1.0)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        i = random.randint(0, max(0, h - new_h))
        j = random.randint(0, max(0, w - new_w))
        img = TVF.crop(img, i, j, new_h, new_w)
        # Horizontal flip
        if random.random() > 0.5:
            img = TVF.hflip(img)
        # Color jitter
        img = TVF.adjust_brightness(img, random.uniform(0.6, 1.4))
        img = TVF.adjust_contrast(img,   random.uniform(0.6, 1.4))
        img = TVF.adjust_saturation(img, random.uniform(0.6, 1.4))
        img = TVF.adjust_hue(img,        random.uniform(-0.07, 0.07))
        return img
    return _augment(pil_image), _augment(pil_image)


def _density_consistency_loss(
    prob: torch.Tensor,
    density_map: torch.Tensor,
    sigma: float = DENSITY_CONSISTENCY_SIGMA,
) -> torch.Tensor:
    """Bilateral total-variation tissue consistency loss.

    Penalises prediction discontinuities across pixels that share similar cell
    density.  The biological rationale: regions with the same cellularity are
    likely the same tissue type, so the class probability should be locally
    smooth within iso-density zones.

    Unlike density-weighted Dice (which incorrectly equates high density with
    tumour), this loss is class-agnostic: it only asks that the *gradient* of
    the prediction map aligns with the *gradient* of the density map.  Where
    density is flat the prediction should also be flat; where density changes
    sharply (tissue boundary) the model is free to change its prediction.

    Args:
        prob:        [1, H, W] or [H, W] float — tissue class probability.
        density_map: [H, W] float on CPU — normalised cell density ∈ [0, 1].
        sigma:       Gaussian bandwidth; only pixels within ~sigma density
                     units are considered iso-density.
    Returns:
        Scalar loss tensor (already in the same ~[0,1] range as Dice).
    """
    import torch.nn.functional as _F
    p = prob.squeeze(0).float()              # [H, W]
    d = density_map.to(p.device).float()     # [H, W]
    # Align spatial resolution if density was computed at a different scale.
    if d.shape != p.shape:
        d = _F.interpolate(
            d.unsqueeze(0).unsqueeze(0), size=p.shape,
            mode='bilinear', align_corners=False).squeeze()

    # Horizontal finite differences
    dp_h = p[:, 1:] - p[:, :-1]   # [H, W-1]
    dd_h = d[:, 1:] - d[:, :-1]   # [H, W-1]
    # Vertical finite differences
    dp_v = p[1:, :] - p[:-1, :]   # [H-1, W]
    dd_v = d[1:, :] - d[:-1, :]   # [H-1, W]

    # Density-similarity weights: w → 1 when neighbouring densities are equal,
    # w → 0 when they differ by more than ~2σ (tissue boundary).
    w_h = torch.exp(-(dd_h ** 2) / (2.0 * sigma ** 2))
    w_v = torch.exp(-(dd_v ** 2) / (2.0 * sigma ** 2))

    return (w_h * dp_h ** 2).mean() + (w_v * dp_v ** 2).mean()


def _soft_mask_boundary(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Differentiable soft boundary from dilation - erosion."""
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    pad = kernel_size // 2
    dil = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - x, kernel_size=kernel_size, stride=1, padding=pad)
    return (dil - ero).clamp(0.0, 1.0).squeeze()


def _is_structure_tissue_class(class_id: int) -> bool:
    return TISSUE_PROMPT_AXES.get(int(class_id), "type") == "structure"


def _structure_class_loss(prob: torch.Tensor,
                          gt_mask: torch.Tensor,
                          class_id: int) -> torch.Tensor:
    """Structure-aware refinement for tissue structure masks.

    Structure classes such as gland are often low-recall and contour-sensitive.
    This combines a foreground recall term with a soft boundary Dice term, and
    is intended for Stage 2/3 only after the broad tissue taxonomy is learned.
    """
    p = prob.squeeze(0).float().clamp(1e-6, 1.0 - 1e-6)
    g = (gt_mask == class_id).float()
    if g.sum() < 4:
        return torch.tensor(0.0, device=p.device)

    recall_loss = 1.0 - (p * g).sum() / g.sum().clamp_min(1.0)

    pred_b = _soft_mask_boundary(p)
    gt_b = _soft_mask_boundary(g)
    boundary_dice = 1.0 - (2.0 * (pred_b * gt_b).sum() + 1.0) / (
        pred_b.sum() + gt_b.sum() + 1.0)

    return 0.5 * recall_loss + 0.5 * boundary_dice


def _scale_regularization_loss(prob: torch.Tensor,
                               gt_mask: torch.Tensor,
                               class_id: int) -> torch.Tensor:
    """Match prediction scale to GT scale using a stable log-area penalty."""
    p = prob.squeeze(0).float().clamp(1e-6, 1.0 - 1e-6)
    g = (gt_mask == class_id).float()
    if g.sum() < 4:
        return torch.tensor(0.0, device=p.device)
    pred_frac = p.mean().clamp_min(1e-6)
    gt_frac = g.mean().clamp_min(1e-6)
    return F.smooth_l1_loss(pred_frac.log(), gt_frac.log())


def _shape_regularization_loss(prob: torch.Tensor,
                               gt_mask: torch.Tensor,
                               class_id: int) -> torch.Tensor:
    """Weakly align predicted and GT object shape through soft boundaries."""
    p = prob.squeeze(0).float().clamp(0.0, 1.0)
    g = (gt_mask == class_id).float()
    if g.sum() < 4:
        return torch.tensor(0.0, device=p.device)
    pred_b = _soft_mask_boundary(p)
    gt_b = _soft_mask_boundary(g)
    boundary_dice = 1.0 - (2.0 * (pred_b * gt_b).sum() + 1.0) / (
        pred_b.sum() + gt_b.sum() + 1.0)
    # Mild total-variation smoothness keeps the term shape-like without
    # forcing a universal topology across all tissue classes.
    tv = (p[:, 1:] - p[:, :-1]).abs().mean() + (p[1:, :] - p[:-1, :]).abs().mean()
    return boundary_dice + 0.05 * tv


def _density_grad_for_bio(density_map: torch.Tensor,
                          out_hw: Tuple[int, int]) -> torch.Tensor:
    d = density_map.float()
    if d.dim() == 2:
        d = d.unsqueeze(0).unsqueeze(0)
    elif d.dim() == 3:
        d = d.unsqueeze(1)
    elif d.dim() != 4:
        d = d.view(1, 1, *d.shape[-2:])
    d = F.interpolate(d, size=out_hw, mode="bilinear", align_corners=False)
    d = d.clamp(0.0, 1.0)
    dx = F.pad(d[:, :, :, 1:] - d[:, :, :, :-1], (0, 1, 0, 0))
    dy = F.pad(d[:, :, 1:, :] - d[:, :, :-1, :], (0, 0, 0, 1))
    g = torch.sqrt(dx.square() + dy.square() + 1e-6)
    return (g / g.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)).squeeze()


def _boundary_alignment_loss(prob: torch.Tensor,
                             density_map: torch.Tensor) -> torch.Tensor:
    """Weakly prefer predicted boundaries to coincide with density transitions."""
    p = prob.squeeze(0).float()
    pred_b = _soft_mask_boundary(p)
    dgrad = _density_grad_for_bio(density_map.to(p.device), tuple(p.shape))
    return (pred_b * (1.0 - dgrad.detach())).mean()


def _class_density_compatibility_loss(prob: torch.Tensor,
                                      density_map: torch.Tensor,
                                      class_id: int) -> torch.Tensor:
    """Soft class/cell-cue compatibility; intentionally weak and non-exclusive."""
    p = prob.squeeze(0).float()
    d = density_map.to(p.device).float()
    if d.shape != p.shape:
        d = F.interpolate(d.unsqueeze(0).unsqueeze(0), size=p.shape,
                          mode="bilinear", align_corners=False).squeeze()
    d = d.clamp(0.0, 1.0)
    local = F.avg_pool2d(
        F.pad(d.unsqueeze(0).unsqueeze(0), (15, 15, 15, 15), mode="replicate"),
        kernel_size=31, stride=1).squeeze()
    grad = _density_grad_for_bio(d, tuple(p.shape))
    high_cluster = torch.sigmoid((local - 0.45) * 10.0)
    low_density = torch.sigmoid((0.25 - local) * 12.0)
    lumen_boundary = _density_grad_for_bio(low_density, tuple(p.shape))
    if int(class_id) in (1, 2):      # tumor / inflammatory
        compat = torch.maximum(local, high_cluster)
    elif int(class_id) == 3:         # stroma/connective
        compat = (1.0 - grad) * (1.0 - high_cluster)
    elif int(class_id) == 5:         # necrosis
        compat = low_density * (1.0 - lumen_boundary)
    elif int(class_id) == 6:         # gland/structure, annotated as filled region
        # GlaS-style gland masks are filled objects, not epithelial rims.
        # Use boundary/organization cues only for localization; do not reward
        # a low-density hollow center as the gland class itself.
        compat = torch.maximum(lumen_boundary, 0.5 * grad)
    else:
        return torch.tensor(0.0, device=p.device)
    return -(p * compat.detach()).mean()


def _gland_lumen_ring_loss(prob: torch.Tensor,
                           density_map: torch.Tensor,
                           gt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Weak gland localization loss for filled gland annotations.

    Earlier versions added ``p * lumen`` as an exclusion term, which encouraged
    ring-like predictions with hollow centers.  GlaS-style GT usually stores a
    filled gland object, so lumen/low-density cues should only help locate
    gland boundaries, never suppress the predicted gland interior.
    """
    p = prob.squeeze(0).float()
    d = density_map.to(p.device).float()
    if d.shape != p.shape:
        d = F.interpolate(d.unsqueeze(0).unsqueeze(0), size=p.shape,
                          mode="bilinear", align_corners=False).squeeze()
    local = F.avg_pool2d(
        F.pad(d.unsqueeze(0).unsqueeze(0), (15, 15, 15, 15), mode="replicate"),
        kernel_size=31, stride=1).squeeze()
    lumen = torch.sigmoid((0.22 - local) * 12.0)
    lumen_b = _density_grad_for_bio(lumen, tuple(p.shape))
    pred_b = _soft_mask_boundary(p)
    if gt_mask is not None and (gt_mask == 6).sum() < 4:
        return torch.tensor(0.0, device=p.device)
    # Boundary alignment only: no lumen-fill exclusion.
    return (pred_b * (1.0 - lumen_b.detach())).mean()


def _bg_suppression_loss(prob: torch.Tensor,
                         pil_image: PILImage.Image,
                         white_thresh: int = 235) -> torch.Tensor:
    """Mildly suppress foreground predictions in obvious WSI background regions.

    Detects near-white / glass-slide areas (all RGB channels ≥ white_thresh)
    and applies a per-class p² penalty, encouraging the model to stay quiet
    where there is clearly no tissue.

    Args:
        prob:        [1, H, W] tissue class probability map (after sigmoid).
        pil_image:   Original PIL image at the same spatial size as prob.
        white_thresh: RGB threshold for background detection (default 235).
    Returns:
        Scalar loss (mean p² over detected background pixels).
    """
    p = prob.squeeze(0).float()          # [H, W]
    img_np = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
    # Detect white/blank background: all three channels near 255.
    bg_mask_np = (img_np[..., 0] >= white_thresh) & \
                 (img_np[..., 1] >= white_thresh) & \
                 (img_np[..., 2] >= white_thresh)
    bg_mask = torch.from_numpy(bg_mask_np).to(p.device).float()  # [H, W]
    if bg_mask.sum() < 4:
        return torch.tensor(0.0, device=p.device)
    # Mean p² over detected white background pixels — mild, but pushes toward 0.
    return (p.square() * bg_mask).sum() / bg_mask.sum().clamp_min(1.0)


def train_one_sample(
    model, transform, sample: dict, device: torch.device,
    prompt_enc: PromptTunedTextEncoder,
    adapter: Optional[nn.Module] = None,
    histoplus: Optional[InstanSegNucleiExtractor] = None,
    scaler: Optional[GradScaler] = None,
    ortho_weight: float = ORTHO_WEIGHT,
    aux_weight: float   = AUX_PRIOR_WEIGHT,
    focal_gamma: float  = 0.0,
    seg_mode: str       = "both",   # "tissue" | "nuclei" | "both"
    tile_classifier:    Optional[nn.Module] = None,  # Stage 3
    ssl_head:           Optional[nn.Module] = None,  # Stage 3 (pooled BYOL — disabled, kept for cls-only)
    pixel_ssl_head:     Optional[nn.Module] = None,  # Stage 3 (pixel-level BYOL — disabled)
    intra_image_contrast_weight: float = INTRA_IMAGE_CONTRAST_WEIGHT,
    cls_weight:         float = TILE_CLS_WEIGHT,
    ssl_weight:         float = SSL_WEIGHT,
    repulsion_weight:   float = UNLABELED_REPULSION_WEIGHT,
    contrastive_weight: float = CONTRASTIVE_WEIGHT,
    contrastive_margin: float = CONTRASTIVE_MARGIN,
    negative_boost:    float = NEGATIVE_BOOST,
    area_loss_weight:  float = AREA_LOSS_WEIGHT,
    structure_class_weight: float = 0.0,
    dice_weight:       float = 1.0,
    bce_weight:        float = 1.0,
    density_fusion:    Optional[nn.Module] = None,  # Stage 2/3: FiLM fusion of density into ViT spatial features
    contrast_proj:     Optional[nn.Module] = None,  # Stage 3: 1024→128 projection for intra-image contrastive
    morph_projector:   Optional[nn.Module] = None,  # Stage 3: morphology-relation SSL projector
    morph_graph_weight: float = MORPH_GRAPH_WEIGHT,
    morph_transition_weight: float = MORPH_TRANSITION_WEIGHT,
    morph_relation_mode: str = MORPH_RELATION_MODE,
    morph_relation_grid_size: int = MORPH_GRAPH_GRID_SIZE,
    morph_keep_n_cells: int = MORPH_KEEP_N_CELLS,
    morph_warmup_steps: int = 0,
    morph_ramp_steps: int = 0,
    bio_boundary_weight: float = 0.0,
    bio_compat_weight: float = 0.0,
    bio_gland_lumen_weight: float = 0.0,
    bg_suppression_weight: float = 0.0,
    scale_reg_weight: float = 0.0,
    shape_reg_weight: float = 0.0,
    cls_only_mil_weight: float = CLS_ONLY_MIL_WEIGHT,
    cls_only_absent_weight: float = CLS_ONLY_ABSENT_WEIGHT,
    cls_only_topk_ratio: float = CLS_ONLY_TOPK_RATIO,
    cls_only_entropy_weight: float = CLS_ONLY_ENTROPY_WEIGHT,
    organ_warmup_steps: int = STAGE4_ORGAN_WARMUP_STEPS,
    global_step: int = 0,
    vis_dir: str = "",
) -> Optional[torch.Tensor]:
    """
    For one PIL image, iterate over present tissue and nuclei classes,
    run SAM3 once per class, accumulate loss.
    Returns scalar loss tensor or None if no valid annotations.
    """
    t_mask = sample["tissue_mask"]   # np.int64 H×W or None
    n_mask = sample["nuclei_mask"]   # np.int64 H×W or None
    # Honour seg_mode: discard the mask we're not training on so the model
    # never sees gradient from the excluded modality.
    if seg_mode == "tissue":
        n_mask = None
    elif seg_mode == "nuclei":
        t_mask = None
    pil_im = sample["image"]
    H, W   = pil_im.height, pil_im.width

    # Set organ conditioning for this sample
    prompt_enc.current_tissue_type = int(sample.get("tissue_type", 0))

    # ── Stash per-sample density on the raw model so the trunk hook can
    # apply F1 density-fusion (post block-31, pre-neck) during every forward.
    # Density is computed once here, reused across all forwards for this image.
    _raw_m_top = model.module if hasattr(model, "module") else model
    _raw_m_top._pending_density = None
    if density_fusion is not None and histoplus is not None:
        try:
            _ds = str(sample.get("dataset_name", ""))
            _li = str(sample.get("local_idx", ""))
            _k  = f"{_ds}:{_li}" if _ds and _li else ""
            _nuc_pri = histoplus.extract(pil_im, cache_key=_k)  # [C, H, W] cpu
            _dens = _nuc_pri[0]  # [H, W]
            # Normalise to ~[0,1] for stable FiLM input.
            _dmax = float(_dens.max()) if _dens.numel() > 0 else 0.0
            if _dmax > 0:
                _dens = _dens / _dmax
            _raw_m_top._pending_density = _dens.to(device).float()  # [H, W]
        except Exception as _de:
            print(f"[Density] WARNING: failed to compute density ({_de}); F1 fusion is a no-op for this sample")
            _raw_m_top._pending_density = None

    # ── Stage 3+: Classification-only samples (no masks) ──────────────────
    if t_mask is None and n_mask is None:
        if tile_classifier is not None:
            _tissue_type = int(sample.get("tissue_type", 0))
            _organ_head = tile_classifier["organ"]
            _tissue_head = tile_classifier["tissue"]
            adapter_bias_all: Optional[torch.Tensor] = None

            _total = torch.tensor(0.0, device=device)

            if ssl_head is not None:
                # ── Stage 3: BYOL for cls-only samples ──────────────────────
                # Online branch = online view (v2), live gradient through ViT.
                # Target branch = target view (v1), stop-grad.
                # Cls heads use the online view features (no separate background
                # forward needed — saves 1 ViT forward per cls-only sample).
                _v1, _v2 = _make_byol_views(pil_im)

                # ── Target branch: stop-grad ──
                with torch.no_grad():
                    _f1_pool = _vit_features_impl(model, _v1, transform, device,
                                                  detach=True, pool=True)   # [1, D]
                    _z1, _ = ssl_head(_f1_pool)

                # ── Online branch: live gradient → ViT ──
                # _vit_features_impl with detach=False keeps grad on output.
                # The trunk hook will fire and update _trunk_features_for_cls
                # with v2 features — use those for cls heads.
                with autocast(dtype=AMP_DTYPE):
                    _f2_pool = _vit_features_impl(model, _v2, transform, device,
                                                  detach=False, pool=True)  # [1, D]
                    _, _p2 = ssl_head(_f2_pool)
                    _ssl_loss = 2.0 - 2.0 * F.cosine_similarity(
                        _p2, _z1.detach(), dim=-1).mean()
                    _total = _total + ssl_weight * _ssl_loss

                # ── Cls heads: use online view (v2) trunk features ──────────
                # Trunk hook was just updated by the v2 forward above.
                _raw_m_cls = model.module if hasattr(model, "module") else model
                _v2_feats = getattr(_raw_m_cls, "_trunk_features_for_cls", None)
                if _v2_feats is not None and _v2_feats.numel() > 0:
                    _v2_feats = _v2_feats[0:1]  # [1, D]
                _cls_feats = _f2_pool
                _cls_feats_used = _v2_feats if _v2_feats is not None else _cls_feats

                with autocast(dtype=AMP_DTYPE):
                    # Organ cls — use early ViT features (block 7, texture-level)
                    _organ_feats_v2 = getattr(_raw_m_cls, "_trunk_features_for_organ", None)
                    if _organ_feats_v2 is not None and _organ_feats_v2.numel() > 0:
                        _organ_feats_v2 = _organ_feats_v2[0:1]  # [1, D]
                    _organ_feats_used = _organ_feats_v2 if _organ_feats_v2 is not None else _cls_feats_used
                    if _organ_feats_used is not None and _organ_feats_used.shape[-1] >= 64:
                        _type_logits = _organ_head(_organ_feats_used.detach())
                        _organ_active = (global_step >= int(organ_warmup_steps))
                        if _organ_active and 1 <= _tissue_type <= 9:
                            _total = _total + cls_weight * F.cross_entropy(
                                _type_logits, torch.tensor([_tissue_type], device=device))
                        elif _organ_active and _tissue_type in (0, 10):
                            _, _pred = _type_logits.softmax(dim=-1).max(dim=-1)
                            _total = _total + 0.5 * cls_weight * F.cross_entropy(
                                _type_logits, _pred)

                    # Tissue cls
                    if _cls_feats_used is not None and _cls_feats_used.shape[-1] >= 64:
                        _tissue_logits = _tissue_head(_cls_feats_used)
                        _tile_label = sample.get("tile_label")
                        if _tile_label is not None:
                            _tl = torch.from_numpy(_tile_label).to(device)
                            _tlm_np = sample.get("tile_label_mask")
                            _tlm = (torch.from_numpy(_tlm_np).to(device)
                                    if _tlm_np is not None else None)
                            _total = _total + 0.5 * cls_weight * _masked_tile_cls_bce(
                                _tissue_logits, _tl, _tlm)

                            # ── Inter-class prototype contrastive ──────
                            _W = _tissue_head.head[1].weight.detach()  # [C, D]
                            _sim = F.cosine_similarity(
                                _cls_feats_used.unsqueeze(1), _W.unsqueeze(0), dim=-1)  # [1, C]
                            _tau = 0.2
                            _logits = _sim / _tau
                            _pos_mask = _tl > 0  # multi-hot positives
                            if _pos_mask.any():
                                _pos_sum = (_logits * _pos_mask.float()).sum()
                                _neg_sum = torch.logsumexp(_logits, dim=-1)
                                _contrast_loss = -(_pos_sum / _pos_mask.sum().clamp_min(1)) + _neg_sum
                                _total = _total + INTER_CLS_CONTRAST_WEIGHT * _contrast_loss
            else:
                # ── Stage 3: no BYOL — run one forward to get cls features ──
                with torch.no_grad():
                    _cls_feats = _vit_features_impl(model, pil_im, transform, device,
                                                    detach=True, pool=True)  # [1, D]
                _raw_m_cls = model.module if hasattr(model, "module") else model
                with autocast(dtype=AMP_DTYPE):
                    # Organ cls — use early ViT features from trunk hook
                    _organ_feats_st3 = getattr(_raw_m_cls, "_trunk_features_for_organ", None)
                    if _organ_feats_st3 is not None and _organ_feats_st3.numel() > 0:
                        _organ_feats_st3 = _organ_feats_st3[0:1]
                    _organ_feats_used_st3 = _organ_feats_st3 if _organ_feats_st3 is not None else _cls_feats
                    if _organ_feats_used_st3 is not None and _organ_feats_used_st3.shape[-1] >= 64:
                        _type_logits = _organ_head(_organ_feats_used_st3.detach())
                        _organ_active = (global_step >= int(organ_warmup_steps))
                        if _organ_active and 1 <= _tissue_type <= 9:
                            _total = _total + cls_weight * F.cross_entropy(
                                _type_logits, torch.tensor([_tissue_type], device=device))
                        elif _organ_active and _tissue_type in (0, 10):
                            _, _pred = _type_logits.softmax(dim=-1).max(dim=-1)
                            _total = _total + 0.5 * cls_weight * F.cross_entropy(
                                _type_logits, _pred)

                    # Tissue cls
                    if _cls_feats is not None and _cls_feats.shape[-1] >= 64:
                        _tissue_logits = _tissue_head(_cls_feats)
                        _tile_label = sample.get("tile_label")
                        if _tile_label is not None:
                            _tl = torch.from_numpy(_tile_label).to(device)
                            _tlm_np = sample.get("tile_label_mask")
                            _tlm = (torch.from_numpy(_tlm_np).to(device)
                                    if _tlm_np is not None else None)
                            _total = _total + 0.5 * cls_weight * _masked_tile_cls_bce(
                                _tissue_logits, _tl, _tlm)

            # ── Intra-image contrastive on cls-only data ─────────────────
            # Stage 3 only — contrast_proj is None before Stage 3.
            if ssl_head is not None and intra_image_contrast_weight > 0:
                _raw_m_cls = model.module if hasattr(model, "module") else model
                # Adapter bias (matching seg path)
                if adapter is not None and histoplus is not None and adapter_bias_all is None:
                    ds = str(sample.get("dataset_name", ""))
                    li = str(sample.get("local_idx", ""))
                    key = f"{ds}:{li}" if ds and li else ""
                    _nuc_prior = histoplus.extract(pil_im, cache_key=key)
                    _dens_map = _nuc_prior[0]
                    _d = torch.as_tensor(_dens_map, device=device).float()
                    _density_feats = torch.nn.functional.adaptive_avg_pool2d(
                        _d.unsqueeze(0).unsqueeze(0),
                        (adapter.POOL_SIZE, adapter.POOL_SIZE),
                    ).flatten()  # [16]
                    adapter_bias_all = adapter(_density_feats)
                # Apply adapter bias to tissue prompts
                _all_tissue_texts = [TISSUE_PROMPT_TEXTS[c] for c in range(1, NUM_TISSUE_CLASSES)]
                if adapter_bias_all is not None:
                    _bias_d = {}
                    for _cid2 in range(1, NUM_TISSUE_CLASSES):
                        _pidx = prompt_enc.key_to_idx.get(_normalise_key(TISSUE_PROMPT_TEXTS[_cid2]))
                        if _pidx is not None:
                            _bias_d[_pidx] = adapter_bias_all[_cid2]
                    prompt_enc._adapter_bias_per_idx = _bias_d or None
                _cls_batch = build_sam_batch_multi(pil_im, _all_tissue_texts, transform, device)
                with autocast(dtype=AMP_DTYPE):
                    _cls_out = model(_cls_batch)
                prompt_enc._adapter_bias_per_idx = None
                _cls_probs = get_fused_prob_batch(_cls_out[-1], (H, W))  # [N_cls, H, W]
                _pseudo_labels = _cls_probs.argmax(dim=0).long() + 1      # [H, W], class ids 1..C
                _confidence    = _cls_probs.max(dim=0).values             # [H, W]

                # Non-contrastive cls-only supervision from tile-level labels.
                _tile_label = sample.get("tile_label")
                if _tile_label is not None:
                    _tl = torch.from_numpy(_tile_label).to(device).float()
                    _mil_bce, _abs_push = _cls_only_mil_losses(
                        _cls_probs, _tl, topk_ratio=cls_only_topk_ratio)
                    _total = (_total +
                              cls_only_mil_weight * _mil_bce +
                              cls_only_absent_weight * _abs_push)
                # Pixel-entropy minimisation on cls-only tissue probabilities.
                # Sharpens decision boundaries — only effective once MIL has
                # pushed probs above floor, so weight is small by default.
                if cls_only_entropy_weight > 0:
                    _ent = _cls_only_entropy_loss(_cls_probs)
                    _total = _total + cls_only_entropy_weight * _ent

                _trunk_feats_cls = getattr(_raw_m_cls, "_trunk_features_spatial", None)
                if _trunk_feats_cls is not None and _trunk_feats_cls.numel() > 0:
                    _trunk_feats_cls = _trunk_feats_cls[0:1]  # [1, 1024, Hf, Wf]
                    if contrast_proj is not None:
                        _proj_feats = contrast_proj(_trunk_feats_cls.float())
                    else:
                        _proj_feats = _trunk_feats_cls

                    # Build pseudo-label restricted to classes present in the
                    # tile label (tile-level supervision constrains which
                    # pseudo-classes are plausible).  Pixels assigned to an
                    # absent class are dropped (set to 0/unlabeled).
                    # _cls_probs: [C, H, W] from sigmoid (independent per class).
                    _C = _cls_probs.shape[0]
                    _tl_present = None
                    if _tile_label is not None:
                        _tl_tensor = torch.from_numpy(_tile_label).to(device).float()
                        # tile_label layout: [tissue 0..NUM_TISSUE_CLASSES, nuclei 0..NUM_NUCLEI_CLASSES]
                        _tl_present = (_tl_tensor[1:1 + _C] > 0.0)  # [C] bool
                    # Renormalise across tissue heads via softmax for a
                    # principled per-pixel confidence (entropy-based).
                    _logits = _cls_probs.clamp(1e-6, 1.0 - 1e-6).logit()  # [C,H,W]
                    _soft = _logits.softmax(dim=0)                        # [C,H,W]
                    # Argmax pseudo-label (class ids 1..C; 0 = unlabeled).
                    _max_p, _argm = _soft.max(dim=0)                      # [H,W]
                    _pseudo_lbl = (_argm + 1).long()                      # 1..C
                    if _tl_present is not None and _tl_present.any():
                        # Zero-out pixels assigned to absent classes.
                        _valid_pix = _tl_present[_argm]                   # [H,W]
                        _pseudo_lbl = torch.where(
                            _valid_pix, _pseudo_lbl,
                            torch.zeros_like(_pseudo_lbl))
                    # Confidence = 1 - H/log(C) over the softmax distribution,
                    # clamped at 0.  Gated at 0.2 so near-uniform pixels
                    # contribute zero weight.
                    _ent = -(_soft * (_soft.clamp_min(1e-8)).log()).sum(dim=0)  # [H,W]
                    _conf = (1.0 - _ent / math.log(max(_C, 2))).clamp(0.0, 1.0)
                    _conf = _conf * (_conf > 0.2).float()

                    for _pcid in range(1, NUM_TISSUE_CLASSES):
                        if (_pseudo_lbl == _pcid).sum() < 4:
                            continue
                        _ic_loss = _intra_image_contrastive_loss(
                            _proj_feats, _pseudo_lbl, class_id=_pcid,
                            temperature=0.07, num_samples=256,
                            pixel_weight=_conf)
                        # Down-weight cls-only contrast vs labeled-seg contrast
                        # to limit confirmation bias from pseudo-labels.
                        _total = _total + 0.5 * intra_image_contrast_weight * _ic_loss

            if _total.requires_grad and torch.isfinite(_total).all():
                return _total
        return None

    # InstanSeg density + adapter bias (Stage 2); computed once per image
    nuclei_prior_cpu  = None
    adapter_bias_all: Optional[torch.Tensor] = None   # [C_all, D]
    density_map:      Optional[torch.Tensor] = None   # [H, W] cpu, for spatial weighting
    if histoplus is not None:
        # Stable cache key: unique per (dataset, image), not per (image, class) pair.
        ds_name = str(sample.get("dataset_name", ""))
        loc_idx = str(sample.get("local_idx", ""))
        key = f"{ds_name}:{loc_idx}" if ds_name and loc_idx else ""
        nuclei_prior_cpu = histoplus.extract(pil_im, cache_key=key)  # [C, H, W] cpu
        density_map = nuclei_prior_cpu[0]  # [H, W] — all channels are identical
        if adapter is not None:
            # 4×4 spatial pool → [16] preserves intra-image density variation
            _d = density_map.to(device).float()
            _density_feats = torch.nn.functional.adaptive_avg_pool2d(
                _d.unsqueeze(0).unsqueeze(0), (adapter.POOL_SIZE, adapter.POOL_SIZE)
            ).flatten()  # [16]
            adapter_bias_all = adapter(_density_feats)          # [C_tissue, D]

        # ── On-the-fly cell-graph building (morph SSL) ──────────────
        if morph_projector is not None:
            try:
                _inst_map = histoplus.get_instance_map(pil_im, cache_key=key)
                _intensity = np.asarray(pil_im.convert("L"), dtype=np.uint8)
                sample["cell_graph"] = _build_cell_graph_stats(
                    _inst_map,
                    grid_size=morph_relation_grid_size,
                    max_cells=morph_keep_n_cells,
                    intensity_image=_intensity)
            except Exception as exc:
                raise RuntimeError(
                    f"MorphGraph target construction failed for {key or ds_name}"
                ) from exc
    else:
        sample["cell_graph"] = None

    ds_name = sample.get("dataset_name", "")

    losses = []
    ctx = autocast(dtype=AMP_DTYPE) if scaler else nullcontext()

    present_t = [c for c in np.unique(t_mask) if 1 <= c < NUM_TISSUE_CLASSES] \
        if t_mask is not None else []

    if present_t:
        gt_t = torch.from_numpy(t_mask).to(device) if t_mask is not None else None

        prompt_entries = [
            ("tissue", cid, TISSUE_PROMPT_TEXTS[cid]) for cid in present_t
        ]
        prompt_texts = [text for _, _, text in prompt_entries]

        if adapter_bias_all is not None:
            bias_dict = {}
            for prompt_type, cid, prompt_text in prompt_entries:
                prompt_idx = prompt_enc.key_to_idx.get(_normalise_key(prompt_text))
                if prompt_idx is None:
                    continue
                if isinstance(adapter, SlideMorphAdapter):
                    bias_dict[prompt_idx] = adapter_bias_all[cid]
            prompt_enc._adapter_bias_per_idx = bias_dict or None

        batch = build_sam_batch_multi(pil_im, prompt_texts, transform, device)
        with ctx:
            out = model(batch)
            # Option-A: true-logit density prior on nuclei prompts.
            _raw_m_ndp = model.module if hasattr(model, "module") else model
            _ndp_scale = float(getattr(
                _raw_m_ndp, "_nuclei_density_logit_prior_scale", 0.0))
            _gate_net  = getattr(_raw_m_ndp, "_density_gate_net", None)
            _logit_bias = _build_nuclei_density_logit_bias(
                prompt_entries, density_map, _ndp_scale, device,
                pred_masks=out[-1]["pred_masks"] if _gate_net is not None else None,
                density_gate_net=_gate_net)
            probs = get_fused_prob_batch(
                out[-1], (H, W), mask_logit_bias=_logit_bias)
            for i, (_, cid, _prompt_text) in enumerate(prompt_entries):
                prob = probs[i].unsqueeze(0)                 # [1, H, W]
                loss = binary_loss(prob, gt_t, cid, focal_gamma,
                                   unlabeled_repulsion_weight=repulsion_weight,
                                   contrastive_weight=contrastive_weight,
                                   contrastive_margin=contrastive_margin,
                                   negative_boost=negative_boost,
                                   area_loss_weight=area_loss_weight,
                                   dice_weight=dice_weight,
                                   bce_weight=bce_weight,
                                   dataset_name=ds_name)
                # Density-consistency regulariser (stages 2+, tissue only):
                # penalise prediction changes where cell density is smooth
                # (iso-density zones are likely the same tissue type).
                # At density boundaries the weight → 0, so the model is
                # free to change its prediction at tissue interfaces.
                if density_map is not None:
                    dc = _density_consistency_loss(prob, density_map)
                    loss = loss + aux_weight * dc
                    if bio_boundary_weight > 0:
                        loss = loss + bio_boundary_weight * _boundary_alignment_loss(
                            prob, density_map)
                    if bio_compat_weight > 0:
                        loss = loss + bio_compat_weight * _class_density_compatibility_loss(
                            prob, density_map, cid)
                    if bio_gland_lumen_weight > 0 and int(cid) == 6:
                        loss = loss + bio_gland_lumen_weight * _gland_lumen_ring_loss(
                            prob, density_map, gt_t)
                # Mild background suppression: penalise foreground predictions
                # in obvious WSI white/blank regions (stage 2+).
                if bg_suppression_weight > 0:
                    loss = loss + bg_suppression_weight * _bg_suppression_loss(
                        prob, pil_im)
                if scale_reg_weight > 0:
                    loss = loss + scale_reg_weight * _scale_regularization_loss(
                        prob, gt_t, class_id=cid)
                if shape_reg_weight > 0:
                    loss = loss + shape_reg_weight * _shape_regularization_loss(
                        prob, gt_t, class_id=cid)
                if structure_class_weight > 0 and _is_structure_tissue_class(cid):
                    loss = loss + structure_class_weight * _structure_class_loss(
                        prob, gt_t, class_id=cid)
                losses.append(loss)
        prompt_enc._adapter_bias_per_idx = None

    if not losses:
        return None

    total_loss = torch.stack(losses).mean()

    # ── Intra-image contrastive on raw pixel_embed (same features decoder uses) ──
    # Using raw pixel_embed ensures contrastive and seg operate on the SAME
    # feature space — no divergence from density fusion before the decoder.
    _raw_m = model.module if hasattr(model, "module") else model

    # ── Dual-head classification loss ───────────────────────────────────
    # Both organ and tissue heads use deepest ViT features (_o[-1], block 31).
    if tile_classifier is not None:
        _cls_feats = getattr(_raw_m, "_trunk_features_for_cls", None)
        if _cls_feats is not None and _cls_feats.ndim == 2:
            _cls_feats = _cls_feats[0:1]  # [1, 1024]
        _organ_head = tile_classifier["organ"]
        _tissue_head = tile_classifier["tissue"]
        _tissue_type = int(sample.get("tissue_type", 0))

        # ── Organ type loss (deep ViT features, DETACHED) ──
        _organ_feats = getattr(_raw_m, "_trunk_features_for_organ", None)
        if _organ_feats is not None and _organ_feats.ndim == 2:
            _organ_feats = _organ_feats[0:1]  # [1, 1024]
        if _organ_feats is not None and _organ_feats.shape[-1] >= 64:
            with autocast(dtype=AMP_DTYPE):
                _type_logits = _organ_head(_organ_feats.detach())
            if 1 <= _tissue_type <= 9:
                total_loss = total_loss + cls_weight * F.cross_entropy(
                    _type_logits, torch.tensor([_tissue_type], device=device))
            elif _tissue_type in (0, 10):
                _, _pred = _type_logits.softmax(dim=-1).max(dim=-1)
                total_loss = total_loss + 0.5 * cls_weight * F.cross_entropy(
                    _type_logits, _pred)

        # ── Tissue class presence loss (detached from ViT — test which cls task conflicts with seg) ──
        if _cls_feats is not None and _cls_feats.shape[-1] >= 64:
            with autocast(dtype=AMP_DTYPE):
                _tissue_logits = _tissue_head(_cls_feats.detach())
            _tile_label = sample.get("tile_label")
            if _tissue_logits is not None and _tile_label is not None:
                _tl = torch.from_numpy(_tile_label).to(device)
                _tlm_np = sample.get("tile_label_mask")
                _tlm = (torch.from_numpy(_tlm_np).to(device)
                        if _tlm_np is not None else None)
                if _tissue_type in (0, 10):
                    # Unknown/Mixed organ → pseudo-label (only on supervised indices)
                    _tprobs = _tissue_logits.sigmoid().reshape(-1)
                    _conf_mask = _tprobs > 0.7
                    if _tlm is not None:
                        _conf_mask = _conf_mask & _tlm.reshape(-1).bool()
                    if _conf_mask.any():
                        _tloss = F.binary_cross_entropy_with_logits(
                            _tissue_logits.reshape(-1)[_conf_mask],
                            _tprobs.detach()[_conf_mask])
                        total_loss = total_loss + 0.3 * cls_weight * _tloss
                else:
                    _tloss = _masked_tile_cls_bce(_tissue_logits, _tl, _tlm)
                    total_loss = total_loss + 0.5 * cls_weight * _tloss

    # ── Stage 3: Intra-image contrastive on labeled segmentation samples ──
    # Pulls same-class trunk features together, pushes different-class features
    # apart within each image.  Only applied when GT mask has labeled pixels.
    # NOTE: applied on ViT block-31 spatial features (after F1 density fusion
    # if active) projected through contrast_proj to 128-d for stable cosine
    # geometry.  Gradient flows directly into the ViT backbone.
    if intra_image_contrast_weight > 0:
        _trunk_feats = getattr(_raw_m, "_trunk_features_spatial", None)
        if _trunk_feats is not None and _trunk_feats.numel() > 0:
            _trunk_feats = _trunk_feats[0:1]  # [1, 1024, Hf, Wf]
            if contrast_proj is not None:
                _contrast_feats = contrast_proj(_trunk_feats.float())  # [1, 128, Hf, Wf]
            else:
                _contrast_feats = _trunk_feats
            has_labeled = ((t_mask is not None and (np.array(t_mask) > 0).any()) or
                           (n_mask is not None and (np.array(n_mask) > 0).any()))
            if has_labeled:
                for _ptype, _cid, _ in prompt_entries:
                    _gt = gt_t if _ptype == "tissue" else gt_n
                    if _gt is None:
                        continue
                    _ic_loss = _intra_image_contrastive_loss(
                        _contrast_feats, _gt, _cid,
                        temperature=0.07, num_samples=256)
                    total_loss = total_loss + intra_image_contrast_weight * _ic_loss

    # ── Stage 3: Morphology-Relation Self-Supervision ────────────────────
    # Uses on-the-fly cell-graph statistics to constrain ViT spatial features
    # without any class-label assumption.  Two complementary signals:
    #   1. ℒ_morph  — local morphology descriptor regression
    #   2. ℒ_trans  — morphology-boundary transition prediction
    # Both operate on the same _trunk_features_spatial as intra_image_contrast
    # (block-31, after density fusion if active).
    if (morph_projector is not None
            and (morph_graph_weight > 0 or morph_transition_weight > 0)):
        _trunk_feats = getattr(_raw_m, "_trunk_features_spatial", None)
        if _trunk_feats is not None and _trunk_feats.numel() > 0:
            _graph_data = sample.get("cell_graph")
            if _graph_data is not None:
                _trunk_feats = _trunk_feats[0:1]  # [1, 1024, Hf, Wf]
                _local_stats = torch.from_numpy(
                    _graph_data["local_stats_grid"]).float()
                _valid_mask = torch.from_numpy(
                    _graph_data["valid_mask"]).bool()

                # Interpolate ViT features to match the morphology grid
                # (ViT spatial map is 64×64 for 4096 crops; morph grid is
                # 16×16 by default — bilinear resize aligns them).
                G = _local_stats.shape[0]
                _, _, Hf, Wf = _trunk_feats.shape
                if Hf != G or Wf != G:
                    _trunk_feats = F.interpolate(
                        _trunk_feats.float(), size=(G, G),
                        mode='bilinear', align_corners=False)

                if global_step < morph_warmup_steps:
                    _morph_scale = 0.0
                elif morph_ramp_steps > 0:
                    _morph_scale = min(
                        1.0,
                        (global_step - morph_warmup_steps + 1) / morph_ramp_steps)
                else:
                    _morph_scale = 1.0

                if morph_relation_mode in ("stats_only", "full"):
                    _ml = _local_morphology_regression_loss(
                        _trunk_feats, _local_stats, _valid_mask,
                        morph_projector)
                    total_loss = total_loss + _morph_scale * morph_graph_weight * _ml
                else:
                    _ml = _trunk_feats.sum() * 0.0

                if morph_relation_mode in ("transition_only", "full"):
                    _tl = _morphology_transition_loss(
                        _trunk_feats, _local_stats, _valid_mask,
                        morph_projector)
                    total_loss = total_loss + _morph_scale * morph_transition_weight * _tl
                else:
                    _tl = _trunk_feats.sum() * 0.0

                _debug_calls = int(getattr(morph_projector, "_debug_calls", 0)) + 1
                morph_projector._debug_calls = _debug_calls
                if _debug_calls <= 3 or _debug_calls % 100 == 0:
                    _valid_cells = int(_valid_mask.sum().item())
                    _valid_h = int((_valid_mask[:, 1:] & _valid_mask[:, :-1]).sum().item())
                    _valid_v = int((_valid_mask[1:] & _valid_mask[:-1]).sum().item())
                    print(
                        f"[MorphGraphActive] call={_debug_calls} "
                        f"valid_cells={_valid_cells} valid_edges={_valid_h + _valid_v} "
                        f"feature_grad={int(_trunk_feats.requires_grad)} "
                        f"weight_scale={_morph_scale:.3f} "
                        f"local_loss={float(_ml.detach()):.6f} "
                        f"transition_loss={float(_tl.detach()):.6f}")

    # Orthogonality regularisation — only on prompts active in this forward pass
    # (avoids nuclei prompts in tissue mode inflating the loss by ~10× due to their
    # mutually-correlated unused embeddings after stage-1 tissue-only training).
    _active_prompt_indices: Optional[List[int]] = None
    _active_prompt_groups: Optional[List[str]] = None
    if prompt_entries:
        _active_prompt_indices = [
            prompt_enc.key_to_idx[_normalise_key(t)]
            for _, _, t in prompt_entries
            if _normalise_key(t) in prompt_enc.key_to_idx
        ]
        _active_prompt_groups = [
            TISSUE_PROMPT_GROUPS.get(cid, "tissue_type")
            for _, cid, t in prompt_entries
            if _normalise_key(t) in prompt_enc.key_to_idx
        ]
    ortho = prompt_enc.regularisation_loss(
        active_indices=_active_prompt_indices,
        active_groups=_active_prompt_groups)
    total_loss = total_loss + ortho_weight * ortho

    return total_loss


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation  (per-class, per-dataset)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model, transform, val_dataset: SAM3CombinedDataset,
    device: torch.device,
    prompt_enc: PromptTunedTextEncoder,
    vis_dir: Optional[str] = None,
    epoch: int = 0,
    vis_num: int = VIS_NUM_VAL,
    tag: str = "",
    organ_cond: bool = True,
    seg_mode: str = "both",   # "tissue" | "nuclei" | "both"
    adapter:           Optional[nn.Module] = None,  # Stage 2+: SlideMorphAdapter
    histoplus:         Optional[InstanSegNucleiExtractor] = None,  # InstanSeg for density
    tile_classifier:   Optional[nn.Module] = None,  # Stage 3: classification eval
    density_fusion:    Optional[nn.Module] = None,  # Stage 2/3: density fusion
    vis_mode:          str = "quota",                # "quota" (split vis_num across ds) | "best_per_dataset" (1 vis per ds, picked by class diversity)
    save_density_vis:  bool = False,
    cls_only_seg_vis:  bool = True,   # run seg forward on cls-only data for vis; set False to save time in train quick-val
) -> Tuple[dict, list, list]:
    """
    Returns:
        summary   — {mean_tissue_dice, mean_nuclei_dice, ...}
        per_class_rows — list of dicts for CSV (one row per class per dataset)
        cls_per_class_rows — list of dicts for classification metrics (Stage 3+)

    organ_cond: if True, sets prompt_enc.current_tissue_type from sample["tissue_type"].
                if False, always uses tissue_type=0 (unconditioned/Unknown).
    seg_mode:   which modality to evaluate ("tissue", "nuclei", or "both").
    tile_classifier: if provided, also evaluates tile-level classification accuracy.
    """
    _eval_with_checkpointing(model)

    # Clear any stale density stashed by the last training sample so the
    # trunk-hook F1 fusion does not apply wrong density during eval.
    _raw_m_eval = model.module if hasattr(model, "module") else model
    if hasattr(_raw_m_eval, "_pending_density"):
        _raw_m_eval._pending_density = None

    # Accumulators: dataset_name → class_key → list of metric dicts
    per_ds_cls: Dict[str, Dict[str, list]] = {}
    # Classification accumulators (Stage 3) — per-dataset
    _type_correct: Dict[str, int] = {}
    _type_total:   Dict[str, int] = {}
    # Tissue class presence: dataset → list of (pred, gt) tuples
    _tissue_cls_preds: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    _vis_sub = tag if tag else f"ep{epoch:03d}"
    if vis_dir:
        os.makedirs(os.path.join(vis_dir, _vis_sub), exist_ok=True)
    # Equal vis budget: split vis_num evenly across all datasets in the val set.
    # Each dataset gets at least 1 image.  With vis_num=15 and 8 datasets → 1 each
    # + 7 extras drawn from the first datasets encountered.
    _all_ds_names = sorted({it.dataset_name for it in val_dataset.items})
    _n_ds         = max(1, len(_all_ds_names))
    _per_ds_quota = max(1, vis_num // _n_ds)
    _vis_per_ds: Dict[str, int] = {}
    # best_per_dataset mode saves a fixed per-dataset quota with stable names.

    for vis_i, sample in enumerate(
            tqdm(val_dataset, desc=f"Eval {_vis_sub}", leave=False)):

        pil_im  = sample["image"]
        t_mask  = sample["tissue_mask"]
        n_mask  = sample["nuclei_mask"]
        ds_name = sample["dataset_name"]
        H, W    = pil_im.height, pil_im.width

        # Set organ conditioning
        _raw_model = model.module if hasattr(model, "module") else model
        _penc = _raw_model.backbone.language_backbone
        if hasattr(_penc, "current_tissue_type"):
            _penc.current_tissue_type = (
                int(sample["tissue_type"]) if organ_cond else 0)

        if ds_name not in per_ds_cls:
            per_ds_cls[ds_name] = {}

        # ── Only predict classes ACTUALLY present in this image's GT ────
        # Uses np.unique() on the mask — NOT _get_present_tissue_classes —
        # because the latter returns ALL declared classes for multi-class
        # datasets, but individual images may only have a subset (e.g.
        # ICIAR2018_WSI declares {T:1,T:4} but each image has only one).
        # Always include class 0 (background) in inference so argmax has
        # a real probability to compete with foreground — prevents the model
        # from predicting foreground everywhere on single-class images.
        _t_present: frozenset = frozenset()
        if t_mask is not None and seg_mode in ("tissue", "both"):
            _t_present = frozenset(int(c) for c in np.unique(t_mask)
                                   if 0 <= c < NUM_TISSUE_CLASSES)

        t_scores = np.zeros((NUM_TISSUE_CLASSES, H, W), dtype=np.float32)
        prompt_entries = []
        for cid in sorted(_t_present):
            prompt_entries.append(("tissue", cid, TISSUE_PROMPT_TEXTS[cid]))

        # Run seg forward.  For cls-only samples (no masks), run a single
        # background-prompt forward to capture ViT trunk features so the
        # classification eval below works on ALL datasets (seg + cls).
        _has_masks = (t_mask is not None) or (n_mask is not None)
        _need_trunk_feats = (tile_classifier is not None and not _has_masks)
        _run_seg_forward = bool(prompt_entries) or _need_trunk_feats

        _raw_model._pending_density = None

        _density_map = None
        # ── Adapter bias + density fusion (matches training: Stages 2/3) ─
        if (_run_seg_forward and histoplus is not None
                and (adapter is not None or density_fusion is not None)):
            ds = str(sample.get("dataset_name", ""))
            li = str(sample.get("local_idx", ""))
            key = f"{ds}:{li}" if ds and li else ""
            _nuclei_prior = histoplus.extract(pil_im, cache_key=key)
            _density_map = _nuclei_prior[0]  # [H, W]
            if density_fusion is not None:
                _dmax = float(_density_map.max()) if _density_map.numel() > 0 else 0.0
                _dens = _density_map / _dmax if _dmax > 0 else _density_map
                _raw_model._pending_density = _dens.to(device).float()
            if adapter is not None:
                _d = torch.as_tensor(_density_map, device=device).float()
                _density_feats = torch.nn.functional.adaptive_avg_pool2d(
                    _d.unsqueeze(0).unsqueeze(0), (adapter.POOL_SIZE, adapter.POOL_SIZE)
                ).flatten()  # [16]
                _adapter_bias = adapter(_density_feats)  # [C_tissue, D]
                # Apply bias to prompt entries
                _bias_dict = {}
                for _pt, _cid, _ptext in prompt_entries:
                    _pidx = prompt_enc.key_to_idx.get(_normalise_key(_ptext))
                    if _pidx is None:
                        continue
                    _bias_dict[_pidx] = _adapter_bias[_cid]
                prompt_enc._adapter_bias_per_idx = _bias_dict or None

        if _run_seg_forward:
            if prompt_entries:
                batch = build_sam_batch_multi(
                    pil_im, [text for _, _, text in prompt_entries],
                    transform, device)
            else:
                # Cls-only: minimal forward to populate trunk hook features
                batch = build_sam_batch(pil_im, "background", transform, device)
            with autocast(dtype=AMP_DTYPE):
                out = model(batch)
            # Clear adapter bias after forward (matches training)
            prompt_enc._adapter_bias_per_idx = None
            _raw_model._pending_density = None
            if prompt_entries:
                # Option-A: true-logit density prior on nuclei prompts.
                _ndp_scale = float(getattr(
                    _raw_model, "_nuclei_density_logit_prior_scale", 0.0))
                _gate_net  = getattr(_raw_model, "_density_gate_net", None)
                _eval_density = None
                if _ndp_scale > 0.0 and adapter is not None and histoplus is not None:
                    try:
                        _eval_density = _nuclei_prior[0]  # [H, W] cpu tensor
                    except Exception:
                        _eval_density = None
                _logit_bias = _build_nuclei_density_logit_bias(
                    prompt_entries, _eval_density, _ndp_scale, device,
                    pred_masks=out[-1]["pred_masks"] if _gate_net is not None else None,
                    density_gate_net=_gate_net)
                all_scores = get_fused_prob_batch(
                    out[-1], (H, W),
                    mask_logit_bias=_logit_bias).cpu().float().numpy()
                for score_map, (_, cid, _prompt_text) in zip(all_scores, prompt_entries):
                    t_scores[cid] = score_map

        # For absent classes, t_scores remain zero → argmax picks background (0).
        pred_t = t_scores.argmax(0).astype(np.int64)
        pred_n = np.zeros((H, W), dtype=np.int64)

        # ── Dual-head classification eval ──────────────────────────────
        # Must run AFTER seg forward so trunk hook has updated features.
        # Both organ and tissue heads use deepest ViT features (_o[-1]).
        if tile_classifier is not None:
            _gt_type = int(sample.get("tissue_type", 0))
            _cls_feats = getattr(_raw_model, "_trunk_features_for_cls", None)
            # One-time diagnostic: are trunk features captured during eval?
            if not hasattr(_raw_model, '_eval_trunk_diag'):
                _raw_model._eval_trunk_diag = True
                _raw_model._eval_gt_types = {}   # track gt_type distribution
            # Track gt_type distribution for diagnostic
            if hasattr(_raw_model, '_eval_gt_types') and len(_raw_model._eval_gt_types) < 200:
                _raw_model._eval_gt_types[_gt_type] = _raw_model._eval_gt_types.get(_gt_type, 0) + 1
                if len(_raw_model._eval_gt_types) == 200:  # print once
                    _cls_ok = _cls_feats is not None
                    _counts = dict(sorted(_raw_model._eval_gt_types.items()))
                    _passing = sum(v for k, v in _counts.items() if k not in (0, 10))
                    print(f"[EvalDiag] trunk features: {'✓' if _cls_ok else '✗'} | "
                          f"gt_type counts: {_counts} | "
                          f"non-filtered (1-9): {_passing} samples")
            if _cls_feats is not None and _cls_feats.ndim == 2:
                _cls_feats = _cls_feats[0:1]  # [B, 1024] → [1, 1024]
            _organ_head = tile_classifier["organ"]
            _tissue_head = tile_classifier["tissue"]
            # Organ type accuracy (deepest ViT features) — per-dataset
            if _cls_feats is not None and _cls_feats.shape[-1] >= 64:
                with autocast(dtype=AMP_DTYPE):
                    _type_logits = _organ_head(_cls_feats.detach())
                _type_pred = int(_type_logits.argmax(dim=-1).item())
                if _gt_type not in (0, 10):
                    _type_correct[ds_name] = _type_correct.get(ds_name, 0) + int(_type_pred == _gt_type)
                    _type_total[ds_name]   = _type_total.get(ds_name, 0) + 1
            # Tissue class presence (deepest ViT features)
            # Evaluate on ALL samples including tissue_type 0/10 — the tissue
            # classifier output is still meaningful even when organ is unknown.
            if _cls_feats is not None and _cls_feats.shape[-1] >= 64:
                with autocast(dtype=AMP_DTYPE):
                    _tissue_logits = _tissue_head(_cls_feats)
                _tile_label = sample.get("tile_label")
                if _tile_label is not None:
                    _tpred = (_tissue_logits.sigmoid().squeeze(0).cpu().float().numpy() > 0.5).astype(np.float32)
                    # Binarize soft ground truth: any class with ratio > 0 is "present"
                    _tgt_bin = (_tile_label > 0.0).astype(np.float32)
                    _tissue_cls_preds.setdefault(ds_name, []).append((_tpred, _tgt_bin))

        # Metrics: all non-target pixels are negatives
        if t_mask is not None and seg_mode in ("tissue", "both"):
            ms = compute_per_class_metrics(pred_t, t_mask, TISSUE_CLASSES, "tissue",
                                           dataset_name=ds_name)
            for k, v in ms.items():
                per_ds_cls[ds_name].setdefault(k, []).append(v)

        # Visualisation
        if vis_mode == "best_per_dataset":
            _want_vis = bool(vis_dir) and _vis_per_ds.get(ds_name, 0) < _per_ds_quota
        else:
            # "quota" (default): split vis_num evenly across datasets.
            _want_vis = vis_dir and _vis_per_ds.get(ds_name, 0) < _per_ds_quota
        if _want_vis:
            # For cls-only samples, run all class prompts in the active
            # seg_mode so the vis shows the model's full per-class guess
            # even though there is no GT mask to compare against.
            # Skip when cls_only_seg_vis=False (train quick-val: saves
            # time; cls eval still runs from the background forward).
            if not _has_masks and not cls_only_seg_vis:
                pass  # skip seg vis for cls-only — cls eval already done
            elif not _has_masks:
                _all_prompts = []
                if seg_mode in ("tissue", "both"):
                    for cid in range(1, NUM_TISSUE_CLASSES):
                        _all_prompts.append(("tissue", cid, TISSUE_PROMPT_TEXTS[cid]))
                pass  # nuclei seg removed; only tissue prompts in vis
                if _all_prompts:
                    _batch = build_sam_batch_multi(
                        pil_im, [text for _, _, text in _all_prompts],
                        transform, device)
                    with autocast(dtype=AMP_DTYPE):
                        _out = model(_batch)
                    # Align with training: apply gated density prior to nuclei prompts.
                    _ndp_scale_v = float(getattr(
                        _raw_model, "_nuclei_density_logit_prior_scale", 0.0))
                    _gate_net_v  = getattr(_raw_model, "_density_gate_net", None)
                    _density_v   = None
                    if _ndp_scale_v > 0.0 and adapter is not None and histoplus is not None:
                        try:
                            _density_v = _density_map  # extracted earlier in this iteration
                        except NameError:
                            _density_v = None
                    _lbias_v = _build_nuclei_density_logit_bias(
                        _all_prompts, _density_v, _ndp_scale_v, device,
                        pred_masks=_out[-1]["pred_masks"] if _gate_net_v is not None else None,
                        density_gate_net=_gate_net_v)
                    _all_scores = get_fused_prob_batch(
                        _out[-1], (H, W),
                        mask_logit_bias=_lbias_v).cpu().float().numpy()
                    _t_full = np.zeros((NUM_TISSUE_CLASSES, H, W), dtype=np.float32)
                    _n_full = np.zeros((NUM_NUCLEI_CLASSES, H, W), dtype=np.float32)
                    for _sm, (_pt, _cid, _) in zip(_all_scores, _all_prompts):
                        if _pt == "tissue":
                            _t_full[_cid] = _sm
                        else:
                            _n_full[_cid] = _sm
                    if seg_mode in ("tissue", "both"):
                        pred_t = _t_full.argmax(0).astype(np.int64)
                    if seg_mode in ("nuclei", "both"):
                        pred_n = _n_full.argmax(0).astype(np.int64)
            if vis_mode == "best_per_dataset":
                _vis_idx = _vis_per_ds.get(ds_name, 0)
                _vis_name = f"{ds_name}_best_{_vis_idx:04d}.png"
                _vis_per_ds[ds_name] = _vis_idx + 1
            else:
                _vis_name = f"{ds_name}_{_vis_per_ds.get(ds_name, 0):04d}.png"
                _vis_per_ds[ds_name] = _vis_per_ds.get(ds_name, 0) + 1
            _save_vis(pil_im, t_mask, n_mask, pred_t, pred_n,
                      os.path.join(vis_dir, _vis_sub, _vis_name),
                      content_hw=sample.get("content_hw"),
                      density_map=(_density_map if save_density_vis else None),
                      density_title=(
                          "Hematoxylin density"
                          if histoplus is not None and
                          histoplus.__class__.__name__ == "HematoxylinDensityExtractor"
                          else "Cell density"))

    # Aggregate
    all_tissue_dice, all_nuclei_dice = [], []
    per_class_rows = []
    for ds_name, cls_dict in per_ds_cls.items():
        for cls_key, metric_list in cls_dict.items():
            avg = {k: float(np.mean([m[k] for m in metric_list]))
                   for k in ("dice", "iou", "prec", "rec")}
            avg["f1"] = (2 * avg["prec"] * avg["rec"] + 1e-8) / \
                        (avg["prec"] + avg["rec"] + 1e-8)
            per_class_rows.append({
                "epoch": epoch, "dataset": ds_name, "class_key": cls_key,
                **avg,
            })
            if cls_key.startswith("tissue_"):
                all_tissue_dice.append(avg["dice"])
            else:
                all_nuclei_dice.append(avg["dice"])

    summary = {
        "epoch":              epoch,
        "mean_tissue_dice":   float(np.mean(all_tissue_dice)) if all_tissue_dice else 0.0,
        "mean_nuclei_dice":   float(np.mean(all_nuclei_dice)) if all_nuclei_dice else 0.0,
    }
    summary["mean_dice"] = float(np.mean(
        all_tissue_dice + all_nuclei_dice)) if (all_tissue_dice + all_nuclei_dice) else 0.0

    # ── Tissue-type classification accuracy (Stage 3) — per-dataset ─────
    _all_type_correct = sum(_type_correct.values())
    _all_type_total   = sum(_type_total.values())
    if _all_type_total > 0:
        summary["tissue_type_acc"] = _all_type_correct / _all_type_total
    elif tile_classifier is not None:
        summary["tissue_type_acc"] = 0.0   # always show when classifier exists
    # Tissue class presence metrics (Stage 3)
    _per_ds_hamming: Dict[str, float] = {}
    if _tissue_cls_preds:
        _all_hamming = []
        for _ds, _preds in _tissue_cls_preds.items():
            _ap = np.stack([p for p, _ in _preds])
            _ag = np.stack([g for _, g in _preds])
            _h = float((_ap == _ag).mean())
            _per_ds_hamming[_ds] = _h
            _all_hamming.append(_h)
        summary["tissue_cls_hamming"] = float(np.mean(_all_hamming)) if _all_hamming else 0.0
    elif tile_classifier is not None:
        summary["tissue_cls_hamming"] = 0.0  # always show when classifier exists
    # ── Per-dataset Dice aggregation ────────────────────────────────────
    _per_ds_tissue_dice: Dict[str, list] = {}
    _per_ds_nuclei_dice: Dict[str, list] = {}
    for _row in per_class_rows:
        _ds = _row["dataset"]
        if _row["class_key"].startswith("tissue_"):
            _per_ds_tissue_dice.setdefault(_ds, []).append(_row["dice"])
        else:
            _per_ds_nuclei_dice.setdefault(_ds, []).append(_row["dice"])

    # ── Build per-dataset classification rows ───────────────────────────
    cls_per_class_rows: List[dict] = []
    _all_cls_ds = (set(_type_total.keys()) | set(_per_ds_hamming.keys())
                   | set(_per_ds_tissue_dice.keys()) | set(_per_ds_nuclei_dice.keys()))
    for _ds in sorted(_all_cls_ds):
        _row = {"epoch": epoch, "dataset": _ds}
        if _ds in _type_total and _type_total[_ds] > 0:
            _row["organ_acc"] = _type_correct.get(_ds, 0) / _type_total[_ds]
        if _ds in _per_ds_hamming:
            _row["tissue_hamming"] = _per_ds_hamming[_ds]
        if _ds in _per_ds_tissue_dice:
            _row["tissue_dice"] = float(np.mean(_per_ds_tissue_dice[_ds]))
        if _ds in _per_ds_nuclei_dice:
            _row["nuclei_dice"] = float(np.mean(_per_ds_nuclei_dice[_ds]))
        if len(_row) > 2:   # has at least one metric beyond epoch/dataset
            cls_per_class_rows.append(_row)

    return summary, per_class_rows, cls_per_class_rows


def _eval_with_checkpointing(model: nn.Module) -> None:
    """Set model to eval mode but keep the ViT trunk in train mode.

    vitdet.py gates activation checkpointing on `self.training`::

        if self.use_act_checkpoint and self.training:
            x = checkpoint.checkpoint(blk, x, ...)

    When the whole model is in eval mode (needed to suppress _compute_matching),
    activation checkpointing is silently disabled, causing the full 32-block
    activation graph to stay in GPU memory and producing OOM at 1008×1008.

    Re-enabling train mode on just the ViT trunk restores checkpointing while
    keeping eval semantics everywhere else (no dropout, no batch-norm updates,
    matcher not triggered).
    """
    model.eval()
    _raw = model.module if hasattr(model, "module") else model
    _trunk = getattr(
        getattr(_raw, "backbone", None),
        "vision_backbone", None,
    )
    if _trunk is not None:
        _trunk = getattr(_trunk, "trunk", _trunk)
    if _trunk is not None and getattr(_trunk, "use_act_checkpoint", False):
        _trunk.train()   # re-enable checkpointing; does NOT affect matcher


def _register_trunk_feature_hook(model: nn.Module, stage_label: str = "eval"):
    """Capture deep ViT features and optionally inject cell-density fusion.

    The same hook is needed during training and checkpoint evaluation.  Without
    registering it at eval/test time, a saved ``DensityFusion`` module would be
    loaded but never used.
    """
    _raw_m = model.module if hasattr(model, "module") else model
    _neck = getattr(_raw_m.backbone, "vision_backbone", None)
    _trunk = getattr(_neck, "trunk", None) if _neck is not None else None
    if _trunk is None:
        return None

    _raw_m._trunk_features_for_cls = None
    _raw_m._trunk_features_for_organ = None
    _raw_m._trunk_features_spatial = None

    def _capture_trunk(_m, _i, _o):
        if isinstance(_o, (list, tuple)) and len(_o) > 0:
            _outs = list(_o)
            _f_deep = _outs[-1]
            _fusion_active = False
            if getattr(_f_deep, "ndim", 0) == 4:
                _fusion_mod = getattr(_raw_m, "_density_fusion_mod", None)
                _dens = getattr(_raw_m, "_pending_density", None)
                if _fusion_mod is not None and _dens is not None:
                    _d = _dens.to(_f_deep.device, dtype=_f_deep.dtype)
                    if _d.ndim == 2:
                        _d = _d.unsqueeze(0).unsqueeze(0)
                    elif _d.ndim == 3:
                        _d = _d.unsqueeze(1)
                    _d = F.interpolate(
                        _d, size=_f_deep.shape[-2:],
                        mode="bilinear", align_corners=False)
                    _f_deep = _fusion_mod(_f_deep, _d)
                    _outs[-1] = _f_deep
                    _fusion_active = True
                _pooled = _f_deep.mean(dim=(2, 3))
                _raw_m._trunk_features_for_organ = _pooled
                _raw_m._trunk_features_for_cls = _pooled
                _raw_m._trunk_features_spatial = _f_deep
            if _raw_m._trunk_features_for_cls is not None and \
               not hasattr(_raw_m, "_trunk_diag_done"):
                _raw_m._trunk_diag_done = True
                print(f"[TrunkHook/{stage_label}] organ=deep {_raw_m._trunk_features_for_organ.shape}, "
                      f"cls=deep {_raw_m._trunk_features_for_cls.shape}, "
                      f"spatial={_raw_m._trunk_features_spatial.shape}, "
                      f"fusion={'on' if _fusion_active else 'off'}")
            return type(_o)(_outs) if isinstance(_o, tuple) else _outs
        return None

    return _trunk.register_forward_hook(_capture_trunk)


def _save_vis(pil_im, t_gt, n_gt, pred_t, pred_n, save_path,
              content_hw: Optional[Tuple[int, int]] = None,
              density_map: Optional[Union[np.ndarray, torch.Tensor]] = None,
              density_title: str = "Cell density"):
    # Trim padded rows/cols so the visualisation shows only real tissue content.
    if content_hw is not None:
        ch, cw = content_hw
        H_im, W_im = pil_im.height, pil_im.width
        if ch < H_im or cw < W_im:
            pil_im = pil_im.crop((0, 0, cw, ch))
            if t_gt is not None:
                t_gt   = t_gt[:ch, :cw]
            if n_gt is not None:
                n_gt   = n_gt[:ch, :cw]
            pred_t = pred_t[:ch, :cw]
            pred_n = pred_n[:ch, :cw]
            if density_map is not None:
                if torch.is_tensor(density_map):
                    density_map = density_map.detach().cpu().numpy()
                density_map = np.asarray(density_map)[:ch, :cw]
    img_np = np.array(pil_im)
    has_t  = t_gt is not None
    has_density = density_map is not None
    # Tissue-only vis: Image | Tissue GT (if present) | Tissue Pred
    # Nuclei segmentation is no longer trained — removed from vis.
    ncols  = 1 + (1 if has_t else 0) + 1 + (1 if has_density else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    ax = iter(axes)

    a = next(ax)
    a.imshow(img_np); a.set_title("Image"); a.axis("off")
    if has_t:
        a2 = next(ax)
        a2.imshow(img_np)
        a2.imshow(t_gt, alpha=0.5, cmap=TISSUE_PAL, vmin=0, vmax=NUM_TISSUE_CLASSES - 1)
        a2.set_title("Tissue GT"); a2.axis("off")
    # Tissue prediction — always shown (meaningful even without GT)
    a3 = next(ax)
    a3.imshow(img_np)
    a3.imshow(pred_t, alpha=0.5, cmap=TISSUE_PAL, vmin=0, vmax=NUM_TISSUE_CLASSES - 1)
    a3.set_title("Tissue Pred"); a3.axis("off")
    if has_density:
        den = density_map.detach().cpu().numpy() if torch.is_tensor(density_map) else np.asarray(density_map)
        den = den.astype(np.float32)
        if den.ndim > 2:
            den = den.reshape(*den.shape[-2:])
        finite = np.isfinite(den)
        if den.size and finite.any():
            vals = den[finite]
            den_min, den_max = np.percentile(vals, (2.0, 98.0))
            if den_max <= den_min + 1e-6:
                den_min = float(vals.min())
                den_max = float(vals.max())
            if den_max > den_min + 1e-6:
                den = np.clip((den - den_min) / (den_max - den_min), 0.0, 1.0)
        a4 = next(ax)
        a4.imshow(den, cmap="magma", vmin=0.0, vmax=1.0)
        a4.set_title(density_title); a4.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _select_reference_samples(
    val_items: List[_Item],
    n_per_ds: int = 2,
    seed: int = 42,
) -> List[int]:
    """
    Pick a fixed set of val indices: up to n_per_ds per dataset.
    Chosen once before training so the same images are compared across all checkpoints.
    """
    from collections import defaultdict
    ds_idx: Dict[str, List[int]] = defaultdict(list)
    for i, item in enumerate(val_items):
        ds_idx[item.dataset_name].append(i)
    rng = random.Random(seed)
    ref = []
    for ds_name in sorted(ds_idx):
        chosen = rng.sample(ds_idx[ds_name], min(n_per_ds, len(ds_idx[ds_name])))
        ref.extend(chosen)
    return ref


def _select_n_per_class_samples(
    items: List[_Item],
    seed: int = 42,
    n_per_class: int = 2,
    native_crop_at_40x: int = NATIVE_CROP_AT_40X,
    max_candidates_to_verify: int = 50,
    per_dataset:   bool = False,
) -> List[_Item]:
    """
    Pick ``n_per_class`` samples per semantic class.

    If ``per_dataset=False`` (legacy): pools all items across ALL datasets for
    each class, shuffles, and picks ``n_per_class`` total.  A class present in
    5 datasets may end up with all picks from BCSS.

    If ``per_dataset=True``: for each class, picks ``n_per_class`` samples
    from **each** dataset that contains that class (guaranteeing 1 per dataset,
    or n_per_class if n_per_class > 1).  Global dedup ensures a single sample
    is not used for multiple (class, dataset) slots.

    Both modes verify that the chosen centre-crop actually contains at least
    one pixel of the target class.
    """
    from collections import defaultdict as _dd2

    # Parse class-key → (modality, class_id)  e.g. "T:1" → ("tissue", 1)
    def _parse_ck(ck: str):
        mod, cid = ck.split(":")
        return ("tissue" if mod == "T" else "nuclei"), int(cid)

    rng = random.Random(seed)
    all_chosen_ids: set = set()   # global dedup
    result: List[_Item] = []

    if not per_dataset:
        # ── Legacy: flat pool per class ──────────────────────────────────
        class_to_indices: Dict[str, List[int]] = _dd2(list)
        for i, item in enumerate(items):
            for ck in _DS_CLASSES.get(item.dataset_name, frozenset()):
                class_to_indices[ck].append(i)

        chosen_for_class: Dict[str, List[int]] = {}
        for ck in sorted(class_to_indices):
            modality, cid = _parse_ck(ck)
            pool = list(class_to_indices[ck])
            rng.shuffle(pool)

            picked: List[int] = []
            for idx in pool[:max_candidates_to_verify]:
                if len(picked) >= n_per_class:
                    break
                item = items[idx]
                try:
                    raw = item.dataset[item.local_idx]
                except Exception:
                    continue
                mask = raw["tissue_mask"] if modality == "tissue" else raw["nuclei_mask"]
                if mask is None:
                    continue
                native_mag  = getattr(item.dataset, "magnification", 40.0)
                native_crop = max(32, int(native_crop_at_40x * native_mag / 40.0))
                H_ds, W_ds = mask.shape[:2]
                i_c  = max(0, (H_ds - native_crop) // 2)
                j_c  = max(0, (W_ds - native_crop) // 2)
                crop = mask[i_c : i_c + native_crop, j_c : j_c + native_crop]
                if (crop == cid).any():
                    if idx not in [p for cls_picked in chosen_for_class.values()
                                   for p in cls_picked]:
                        picked.append(idx)
            # Fallback
            if len(picked) < n_per_class:
                for idx in pool:
                    if len(picked) >= n_per_class:
                        break
                    if idx not in picked:
                        picked.append(idx)
            chosen_for_class[ck] = picked
            for idx in picked:
                if idx not in all_chosen_ids:
                    all_chosen_ids.add(idx)
                    result.append(items[idx])
    else:
        # ── Per-dataset: guarantee at least 1 sample from every dataset ───
        #  class → dataset → list of item indices
        cls_ds_to_indices: Dict[str, Dict[str, List[int]]] = _dd2(lambda: _dd2(list))
        for i, item in enumerate(items):
            for ck in _DS_CLASSES.get(item.dataset_name, frozenset()):
                cls_ds_to_indices[ck][item.dataset_name].append(i)

        for ck in sorted(cls_ds_to_indices):
            modality, cid = _parse_ck(ck)
            for ds_name in sorted(cls_ds_to_indices[ck]):
                pool = list(cls_ds_to_indices[ck][ds_name])
                rng.shuffle(pool)

                picked: List[int] = []
                # Verification pass
                for idx in pool[:max_candidates_to_verify]:
                    if len(picked) >= n_per_class:
                        break
                    item = items[idx]
                    try:
                        raw = item.dataset[item.local_idx]
                    except Exception:
                        continue
                    mask = raw["tissue_mask"] if modality == "tissue" else raw["nuclei_mask"]
                    if mask is None:
                        continue
                    native_mag  = getattr(item.dataset, "magnification", 40.0)
                    native_crop = max(32, int(native_crop_at_40x * native_mag / 40.0))
                    H_ds, W_ds = mask.shape[:2]
                    i_c  = max(0, (H_ds - native_crop) // 2)
                    j_c  = max(0, (W_ds - native_crop) // 2)
                    crop = mask[i_c : i_c + native_crop, j_c : j_c + native_crop]
                    if (crop == cid).any():
                        if idx not in picked:
                            picked.append(idx)
                # Fallback
                if len(picked) < n_per_class:
                    for idx in pool:
                        if len(picked) >= n_per_class:
                            break
                        if idx not in picked:
                            picked.append(idx)
                for idx in picked:
                    if idx not in all_chosen_ids:
                        all_chosen_ids.add(idx)
                        result.append(items[idx])

    return result


def _item_ref(it: _Item) -> Dict[str, int | str]:
    return {"dataset_name": str(it.dataset_name), "local_idx": int(it.local_idx)}


def _items_from_refs(refs: List[dict], pool: List[_Item]) -> Optional[List[_Item]]:
    """Resolve cached (dataset_name, local_idx) refs back to live _Item objects."""
    lookup = {(str(it.dataset_name), int(it.local_idx)): it for it in pool}
    out: List[_Item] = []
    for ref in refs:
        key = (str(ref.get("dataset_name", "")), int(ref.get("local_idx", -1)))
        it = lookup.get(key)
        if it is None:
            return None
        out.append(it)
    return out


def _dataset_count_fingerprint(items: List[_Item]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for it in items:
        counts[str(it.dataset_name)] = counts.get(str(it.dataset_name), 0) + 1
    return dict(sorted(counts.items()))


# ── Per-image present-class lookup (for inference) ──────────────────────────

def _get_present_tissue_classes(item: _Item) -> frozenset:
    """Return the set of tissue class IDs actually present in this image.

    - SegPath: uses precomputed ``_marker_classes`` to return the single
      IHC-marker-derived class (e.g. {2}, {3}, or {4}).
    - Other single-class tissue datasets (GlaS, DigestPath2019): returns the
      single declared class from ``_DS_CLASSES``.
    - Multi-class tissue datasets (BCSS, WSSS4LUAD, …): returns all declared
      tissue classes from ``_DS_CLASSES``.
    """
    ds   = item.dataset_name
    decl = _DS_CLASSES.get(ds, frozenset())
    tids = {int(ck[2:]) for ck in decl if ck.startswith("T:")}
    if not tids:
        return frozenset()
    # SegPath: only the IHC-marker class is present
    if ds == "SegPath":
        try:
            cid = int(item.dataset._marker_classes[item.local_idx])
            if 1 <= cid < NUM_TISSUE_CLASSES:
                return frozenset({cid})
        except Exception:
            pass
        return frozenset(tids)  # fallback: all declared (should not happen)
    # Single-class datasets: the one declared class IS the present class
    if len(tids) == 1:
        return frozenset(tids)
    # Multi-class: all declared are potentially present
    return frozenset(tids)


def _get_present_nuclei_classes(item: _Item) -> frozenset:
    """Return the set of nuclei class IDs actually present in this image."""
    decl = _DS_CLASSES.get(item.dataset_name, frozenset())
    nids = {int(ck[2:]) for ck in decl if ck.startswith("N:")}
    if not nids:
        return frozenset()
    if len(nids) == 1:
        return frozenset(nids)
    return frozenset(nids)


# Backward-compatible alias used by run_final_test
def _select_one_per_class_samples(
    items: List[_Item],
    seed: int = 42,
) -> List[_Item]:
    return _select_n_per_class_samples(items, seed=seed, n_per_class=1, per_dataset=True)


@torch.no_grad()
def save_reference_vis(
    model,
    transform,
    val_ds: SAM3CombinedDataset,
    ref_indices: List[int],
    global_step: int,
    vis_dir: str,
    device: torch.device,
    native_crop_at_40x: int = NATIVE_CROP_AT_40X,
) -> None:
    """
    Run inference on the fixed reference set and write one PNG per image.
    Saved under  vis/ref/step_{global_step:07d}/{dataset}_{idx:05d}.png
    so you can visually compare the same images across all checkpoints.
    """
    step_dir = os.path.join(vis_dir, "ref", f"step_{global_step:07d}")
    os.makedirs(step_dir, exist_ok=True)

    # The reference set uses centre-cropped images (crop_to_target=True), so
    # each image is already at the training crop size.  Use a single-patch
    # forward pass per class — no sliding window needed.
    model.eval()   # ref vis is @no_grad inference, full eval OK
    for idx in tqdm(ref_indices, desc=f"RefVis step {global_step}", leave=False):
        sample  = val_ds[idx]
        pil_im  = sample["image"]
        t_mask  = sample["tissue_mask"]
        n_mask  = sample["nuclei_mask"]
        ds_name = sample["dataset_name"]
        H, W    = pil_im.height, pil_im.width

        # ── Only tissue classes (nuclei seg removed) ──────────────────
        _t_present: frozenset = frozenset()
        if t_mask is not None:
            _t_present = frozenset(int(c) for c in np.unique(t_mask)
                                   if 0 <= c < NUM_TISSUE_CLASSES)

        t_scores = np.zeros((NUM_TISSUE_CLASSES, H, W), dtype=np.float32)
        for cid in sorted(_t_present):
            t_scores[cid] = sliding_window_predict_class(
                model, transform, pil_im, TISSUE_PROMPT_TEXTS[cid], device,
                patch_size=max(H, W), stride=max(H, W) // 2)

        pred_t = t_scores.argmax(0).astype(np.int64)
        pred_n = np.zeros((H, W), dtype=np.int64)

        _save_vis(pil_im, t_mask, n_mask, pred_t, pred_n,
                  os.path.join(step_dir, f"{ds_name}_{idx:05d}.png"),
                  content_hw=sample.get("content_hw"))

# ─────────────────────────────────────────────────────────────────────────────

def run_training(
    stage:        int,
    model:        nn.Module,
    prompt_enc:   PromptTunedTextEncoder,
    train_items:  List[_Item],
    save_dir:     Path,
    device:       torch.device,
    epochs:       int            = EPOCHS_PER_STAGE,
    batch_size:   int            = 1,           # images per optimiser step
    lr:           float          = LR_PROMPT,
    val_items:    Optional[List[_Item]]          = None,
    adapter:      Optional[nn.Module]                     = None,
    histoplus:    Optional[InstanSegNucleiExtractor]   = None,
    lora_cfg:     Optional[LoRAConfig]                 = None,
    val_steps:    int                                  = 5000,
    num_workers:  int                                  = 0,
    fg_crop_prob: float                                = 0.0,
    steps_per_epoch: Optional[int]                    = None,
    total_steps:     Optional[int]                    = None,
    seed:         int                                  = SEED,
    resume_ckpt:  Optional[str]                        = None,
    resume_load:  str                                  = "all",
    prompt_lr:    Optional[float]                      = None,
    adapter_lr:   Optional[float]                      = None,
    reset_best_dice: bool                              = False,
    run_initial_eval: bool                             = True,
    save_best_vis: bool                                = False,
    vis_num:      int                                  = 8,
    quick_val_max_samples: Optional[int]              = None,
    curriculum:               bool                     = False,
    curriculum_warmup_epochs: int                      = 2,
    focal_gamma:              float                    = 0.0,
    val_samples_per_class:    int                      = 2,
    val_samples_per_dataset:  Optional[int]            = None,
    repulsion_weight:         float                    = UNLABELED_REPULSION_WEIGHT,
    negative_boost:           float                    = NEGATIVE_BOOST,
    area_loss_weight:         float                    = AREA_LOSS_WEIGHT,
    density_consistency_weight: float                  = AUX_PRIOR_WEIGHT,
    structure_class_weight:   float                    = STRUCTURE_CLASS_LOSS_WEIGHT,
    freeze_prompts_after_stage1: bool                  = False,
    dice_weight:              float                    = 1.0,
    bce_weight:               float                    = 1.0,
    seg_mode:                 str                      = "tissue",   # tissue-only (nuclei no longer trained)
    native_crop_at_40x:       int                      = NATIVE_CROP_AT_40X,
    train_cfg:                Optional[dict]           = None,
    tile_classifier:          Optional[nn.Module]      = None,    # Stage 3
    ssl_head:                 Optional[nn.Module]      = None,    # Stage 3 (pooled)
    pixel_ssl_head:           Optional[nn.Module]      = None,    # Stage 3 (pixel)
    density_fusion:           Optional[nn.Module]      = None,    # Stage 2/3 (density→ViT spatial fusion)
    density_gate_net:         Optional[nn.Module]      = None,    # Stage 2+ (SAM-conditioned density-prior gate)
    contrast_proj:            Optional[nn.Module]      = None,    # Stage 3 (1024→128 proj for intra-image contrastive)
    morph_projector:          Optional[nn.Module]      = None,    # Stage 3 (morphology-relation SSL)
    morph_graph_weight:       float                    = MORPH_GRAPH_WEIGHT,
    morph_transition_weight:  float                    = MORPH_TRANSITION_WEIGHT,
    morph_relation_mode:      str                      = MORPH_RELATION_MODE,
    morph_relation_grid_size: int                      = MORPH_GRAPH_GRID_SIZE,
    morph_keep_n_cells:       int                      = MORPH_KEEP_N_CELLS,
    morph_warmup_steps:       int                      = 0,
    morph_ramp_steps:         int                      = 0,
    hard_mining_start_epoch:  int                      = -1,      # -1 = disabled
    ssl_weight:               float                    = SSL_WEIGHT,
    intra_image_contrast_weight: float                = INTRA_IMAGE_CONTRAST_WEIGHT,
    cls_weight:               float                    = TILE_CLS_WEIGHT,
    cls_only_mil_weight:      float                    = CLS_ONLY_MIL_WEIGHT,
    cls_only_absent_weight:   float                    = CLS_ONLY_ABSENT_WEIGHT,
    cls_only_topk_ratio:      float                    = CLS_ONLY_TOPK_RATIO,
    cls_only_entropy_weight:  float                    = CLS_ONLY_ENTROPY_WEIGHT,
    organ_warmup_steps:       int                      = STAGE4_ORGAN_WARMUP_STEPS,
    cls_only_stride:          int                      = 3,
    nuclei_density_logit_prior: float                  = 0.0,
    freeze_lora_in_stage2:    bool                     = False,
    bio_boundary_weight:      float                    = 0.0,
    bio_compat_weight:        float                    = 0.0,
    bio_gland_lumen_weight:   float                    = 0.0,
    bg_suppression_weight:    float                    = 0.0,
    scale_reg_weight:         float                    = 0.0,
    shape_reg_weight:         float                    = 0.0,
    use_organ_bias:           bool                     = False,
    setup_cache_dir:          Optional[str]             = None,
    magnification_alignment:  bool                     = True,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir   = str(save_dir / "vis")
    tb_writer = SummaryWriter(log_dir=str(save_dir / "tb")) \
        if TENSORBOARD_AVAILABLE else None

    # Stash Option-A density-prior scale on the raw model so both the
    # training seg loop and evaluate() can read it without threading a
    # kwarg through every call site.
    _raw_m_top0 = model.module if hasattr(model, "module") else model
    _prompt_mod = _raw_m_top0.backbone.language_backbone
    if hasattr(_prompt_mod, "use_organ_bias"):
        _prompt_mod.use_organ_bias = bool(use_organ_bias)
    if hasattr(_prompt_mod, "organ_bias") and not use_organ_bias:
        _prompt_mod.organ_bias.requires_grad_(False)
        print("[OrganBias] Disabled: using unconditioned prompts only.")
    _raw_m_top0._nuclei_density_logit_prior_scale = float(nuclei_density_logit_prior)
    # Stash learnable SAM-output-conditioned gate on the model so both the
    # training seg loop and evaluate() can pick it up.  Constructed (and
    # optionally pre-loaded) by main() so its weights survive checkpoint
    # save/load cycles across stages.
    _raw_m_top0._density_gate_net = density_gate_net
    if density_gate_net is not None:
        print(f"[Option-A] DensityGateNet attached "
              f"(params={sum(p.numel() for p in density_gate_net.parameters())})")
    if nuclei_density_logit_prior > 0.0:
        print(f"[Option-A] Nuclei density logit prior ENABLED (scale={nuclei_density_logit_prior:.3f})")

    _eval_modes = ([(False, "nocond"), (True, "cond")]
                   if use_organ_bias else [(False, "nocond")])

    # Filter training items to the active segmentation modality.
    # This keeps the DataLoader lean and prevents wasted GPU time on samples
    # whose only annotation is the modality we're NOT training.
    train_items_all = list(train_items)  # save before filtering (for cls-only items)
    # Classification-only datasets: have tile_label from _class_ids, no masks.
    # Nuclei seg datasets are also routed here for tissue-mode training: they
    # have no tissue mask, but their nuclei mask is used to derive a tissue
    # tile_label via _NUCLEI_TO_TISSUE (and the mask supervision is restricted
    # to the declared (T, N) classes via tile_label_mask).
    _CLS_ONLY_DS: frozenset = frozenset({
        "CRC100K", "MHIST", "ICIAR2018", "PatchCamelyon17",
        "SkinCancer", "LungColon", "Osteosarcoma",
        # Nuclei datasets used as tissue cls-only in tissue mode:
        "PanNuke", "CoNIC2022", "Lizard", "MoNuSAC", "NuCLS",
        "Kumar", "CPM15", "CPM17", "Janowczyk", "TNBC",
        # SegPath: IHC cell-level masks, not tissue region masks — cls-only
        "SegPath",
        # SICAPv2: bad quality, escape — excluded from seg training/val
        "SICAPv2",
    })
    def _has_tissue(it: _Item) -> bool:
        return bool(_DS_CLASSES.get(it.dataset_name, frozenset()) & {
            c for c in _DS_CLASSES.get(it.dataset_name, frozenset()) if c.startswith("T:")})
    def _has_nuclei(it: _Item) -> bool:
        return bool(_DS_CLASSES.get(it.dataset_name, frozenset()) & {
            c for c in _DS_CLASSES.get(it.dataset_name, frozenset()) if c.startswith("N:")})
    def _is_cls_only(it: _Item) -> bool:
        return it.dataset_name in _CLS_ONLY_DS

    _cls_only_items: List[_Item] = []
    if seg_mode == "tissue":
        train_items = [it for it in train_items
                       if _has_tissue(it) and not _is_cls_only(it)
                       and it.dataset_name not in _THUMBNAIL_WSI_SEG_DS]
        if stage >= 3:
            _cls_only_items = [it for it in train_items_all
                               if _is_cls_only(it) and _has_tissue(it)]
    elif seg_mode == "nuclei":
        train_items = [it for it in train_items if _has_nuclei(it)]
    else:  # "both"
        train_items = [it for it in train_items
                       if not _is_cls_only(it)
                       and it.dataset_name not in _THUMBNAIL_WSI_SEG_DS]
        if stage >= 3:
            _cls_only_items = [it for it in train_items_all if _is_cls_only(it)]
    n_cls = len(_cls_only_items)
    if n_cls:
        print(f"[seg_mode={seg_mode}] Stage {stage}: including {n_cls:,} classification-only items "
              f"({', '.join(sorted(set(it.dataset_name for it in _cls_only_items)))})")
    print(f"[seg_mode={seg_mode}] Active train items: {len(train_items):,}")

    transform = _build_transform(SAM3_RESOLUTION)

    def _item_has_declared_foreground(it: _Item) -> bool:
        """True when the center crop contains a declared foreground seg class."""
        declared = _DS_CLASSES.get(it.dataset_name, frozenset())
        tissue_ids = [int(ck[2:]) for ck in declared if ck.startswith("T:") and int(ck[2:]) > 0]
        nuclei_ids = [int(ck[2:]) for ck in declared if ck.startswith("N:") and int(ck[2:]) > 0]
        try:
            raw = it.dataset[it.local_idx]
        except Exception:
            return False

        def _crop_has(mask, ids: List[int]) -> bool:
            if mask is None or not ids:
                return False
            native_mag = getattr(it.dataset, "magnification", 40.0)
            native_crop = max(32, int(native_crop_at_40x * native_mag / 40.0))
            H_ds, W_ds = mask.shape[:2]
            i_c = max(0, (H_ds - native_crop) // 2)
            j_c = max(0, (W_ds - native_crop) // 2)
            crop = mask[i_c:i_c + native_crop, j_c:j_c + native_crop]
            if crop.size and np.isin(crop, ids).any():
                return True
            return np.isin(mask, ids).any()

        if seg_mode in ("tissue", "both") and _crop_has(raw.get("tissue_mask"), tissue_ids):
            return True
        if seg_mode in ("nuclei", "both") and _crop_has(raw.get("nuclei_mask"), nuclei_ids):
            return True
        return False

    _setup_cache_path: Optional[Path] = None
    _cached_quick_val_items: Optional[List[_Item]] = None
    _cached_holdout_val_items: Optional[List[_Item]] = None
    _setup_cache_loaded = False
    if setup_cache_dir and stage >= 3:
        _setup_cache_root = Path(setup_cache_dir)
        _setup_cache_root.mkdir(parents=True, exist_ok=True)
        _setup_fingerprint = {
            "version": 1,
            "stage": int(stage),
            "seg_mode": str(seg_mode),
            "seed": int(seed),
            "native_crop_at_40x": int(native_crop_at_40x),
            "val_samples_per_class": int(val_samples_per_class),
            "val_samples_per_dataset": (
                None if val_samples_per_dataset is None else int(val_samples_per_dataset)
            ),
            "quick_val_max_samples": quick_val_max_samples,
            "train_counts": _dataset_count_fingerprint(train_items),
            "cls_only_counts": _dataset_count_fingerprint(_cls_only_items),
            "val_counts": _dataset_count_fingerprint(val_items or []),
        }
        _setup_cache_key = hashlib.sha1(
            json.dumps(_setup_fingerprint, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        _setup_cache_path = _setup_cache_root / f"stage3_setup_{_setup_cache_key}.json"
        if _setup_cache_path.exists():
            try:
                with open(_setup_cache_path) as _f:
                    _cached = json.load(_f)
                _resolve_pool = list(train_items_all) + list(val_items or [])
                _qv = _items_from_refs(_cached.get("quick_val_items", []), _resolve_pool)
                _hv = _items_from_refs(_cached.get("holdout_val_items", []), _resolve_pool)
                if _qv is not None and _hv is not None:
                    _cached_quick_val_items = _qv
                    _cached_holdout_val_items = _hv
                    _setup_cache_loaded = True
                    print(f"[setup-cache] Loaded fixed Stage-3 val selections from "
                          f"{_setup_cache_path}")
                else:
                    print(f"[setup-cache] Ignoring stale cache with missing item refs: "
                          f"{_setup_cache_path}")
            except Exception as _e:
                print(f"[setup-cache] WARNING: failed to read {_setup_cache_path}: {_e}")

    # Fixed train quick-val: class-balanced N samples per semantic class per dataset.
    # Sort before selection so the same seed always picks the same items
    # regardless of dataset load order — enables meaningful cross-stage comparison.
    if _cached_quick_val_items is not None:
        quick_val_items = list(_cached_quick_val_items)
    else:
        _sorted_train = sorted(train_items, key=lambda it: (it.dataset_name, it.local_idx))
        quick_val_items = _select_n_per_class_samples(
            _sorted_train, seed=seed, n_per_class=val_samples_per_class,
            native_crop_at_40x=native_crop_at_40x, per_dataset=True)
        # Add cls-only items for classification eval (no seg masks, but organ/tissue labels)
        if _cls_only_items:
            _n_cls_val = min(len(_cls_only_items), max(20, val_samples_per_class * 2))
            _sorted_cls = sorted(_cls_only_items, key=lambda it: (it.dataset_name, it.local_idx))
            _cls_val = random.Random(seed).sample(_sorted_cls, _n_cls_val)
            quick_val_items.extend(_cls_val)
        if quick_val_max_samples is not None and quick_val_max_samples > 0 and \
                len(quick_val_items) > quick_val_max_samples:
            rng = random.Random(seed)
            quick_val_items = rng.sample(quick_val_items, quick_val_max_samples)
    quick_val_ds    = SAM3CombinedDataset(quick_val_items, augment=False,
                                          crop_to_target=magnification_alignment,
                                          native_crop_at_40x=native_crop_at_40x,
                                          cache_samples=True,
                                          normalize_magnification=magnification_alignment)
    n_cls_val = sum(1 for it in quick_val_items if _is_cls_only(it))
    print(f"Quick-val (train) set: {len(quick_val_items)} samples "
          f"({val_samples_per_class} per class per dataset, "
          f"{n_cls_val} cls-only, pixel-verified, fixed from train)")

    # ── Hold-out quick-val: dataset-balanced N samples per dataset ───────
    # Evaluated alongside the train quick-val during training but NEVER used
    # for model-selection (best checkpoint is decided by train quick-val only).
    quick_val_val_items: List[_Item] = []
    quick_val_val_ds    = None
    _n_hold_per_ds = max(
        1, int(val_samples_per_dataset if val_samples_per_dataset is not None
               else val_samples_per_class)
    )
    if _cached_holdout_val_items is not None:
        quick_val_val_items = list(_cached_holdout_val_items)
    elif val_items:
        # Filter to datasets that have annotations for the active seg_mode
        _val_pool = list(val_items)
        if seg_mode == "tissue":
            _val_pool = [it for it in _val_pool
                         if any(ck.startswith("T:") for ck in _DS_CLASSES.get(it.dataset_name, frozenset()))
                         and not _is_cls_only(it)
                         and it.dataset_name not in _THUMBNAIL_WSI_SEG_DS]
        elif seg_mode == "nuclei":
            _val_pool = [it for it in _val_pool
                         if any(ck.startswith("N:") for ck in _DS_CLASSES.get(it.dataset_name, frozenset()))]
        else:  # "both" — exclude cls-only datasets from seg validation
            _val_pool = [it for it in _val_pool if not _is_cls_only(it)]
        if _val_pool:
            # Training-time holdout: N foreground-verified samples per dataset.
            # The final test-only eval uses ALL holdout samples.
            _ds_groups: Dict[str, List[_Item]] = {}
            for _it in sorted(_val_pool, key=lambda it: (it.dataset_name, it.local_idx)):
                _ds_groups.setdefault(_it.dataset_name, []).append(_it)
            _rng_hold = random.Random(seed + 1)
            for _ds, _items in sorted(_ds_groups.items()):
                _items = list(_items)
                _rng_hold.shuffle(_items)
                _fg_items = [it for it in _items if _item_has_declared_foreground(it)]
                _pool = _fg_items if _fg_items else _items
                if not _fg_items:
                    print(f"  [holdout_val] WARNING: {_ds} has no foreground-verified "
                          f"candidate; falling back to random samples.")
                quick_val_val_items.extend(_pool[:min(_n_hold_per_ds, len(_pool))])
        # Add cls-only items from held-out val for classification eval
        if stage >= 3:
            _val_cls_only = [it for it in val_items if it.dataset_name in _CLS_ONLY_DS]
            if _val_cls_only:
                _n_val_cls = min(len(_val_cls_only), max(20, val_samples_per_class * 2))
                _cls_val_items = random.Random(seed + 2).sample(_val_cls_only, _n_val_cls)
                quick_val_val_items.extend(_cls_val_items)
        if quick_val_val_items:
            if quick_val_max_samples is not None and quick_val_max_samples > 0 and \
                    len(quick_val_val_items) > quick_val_max_samples:
                rng2 = random.Random(seed + 1)
                quick_val_val_items = rng2.sample(quick_val_val_items, quick_val_max_samples)
            quick_val_val_ds = SAM3CombinedDataset(quick_val_val_items, augment=False,
                                                    crop_to_target=magnification_alignment,
                                                    native_crop_at_40x=native_crop_at_40x,
                                                    cache_samples=True,
                                                    normalize_magnification=magnification_alignment)
            _n_val_cls_print = sum(1 for it in quick_val_val_items
                                   if it.dataset_name in _CLS_ONLY_DS)
            print(f"Quick-val (hold-out) set: {len(quick_val_val_items)} samples "
                  f"({_n_hold_per_ds} per dataset target from --val_samples_per_dataset, "
                  f"foreground-verified when possible, "
                  f"{_n_val_cls_print} cls-only, fixed from val, "
                  f"seg_mode={seg_mode})")
        else:
            print(f"Quick-val (hold-out) set: empty — no val datasets with '{seg_mode}' annotations")

    if _cached_holdout_val_items is not None:
        if quick_val_val_items:
            quick_val_val_ds = SAM3CombinedDataset(quick_val_val_items, augment=False,
                                                    crop_to_target=magnification_alignment,
                                                    native_crop_at_40x=native_crop_at_40x,
                                                    cache_samples=True,
                                                    normalize_magnification=magnification_alignment)
            _n_val_cls_print = sum(1 for it in quick_val_val_items
                                   if it.dataset_name in _CLS_ONLY_DS)
            print(f"Quick-val (hold-out) set: {len(quick_val_val_items)} samples "
                  f"({_n_hold_per_ds} per dataset target from --val_samples_per_dataset, "
                  f"foreground-verified when possible, "
                  f"{_n_val_cls_print} cls-only, fixed from cached val, "
                  f"seg_mode={seg_mode})")
        else:
            print(f"Quick-val (hold-out) set: empty — cached selection has no items")

    if _setup_cache_path is not None and not _setup_cache_loaded:
        try:
            with open(_setup_cache_path, "w") as _f:
                json.dump({
                    "quick_val_items": [_item_ref(it) for it in quick_val_items],
                    "holdout_val_items": [_item_ref(it) for it in quick_val_val_items],
                }, _f, indent=2)
            print(f"[setup-cache] Saved fixed Stage-3 val selections to "
                  f"{_setup_cache_path}")
        except Exception as _e:
            print(f"[setup-cache] WARNING: failed to write {_setup_cache_path}: {_e}")

    _num_workers = max(0, int(num_workers))

    # ── Precompute which tissue classes are actually present per image ──
    # Only needed for single-class-per-image datasets (SegPath).
    # Uses SegPath's precomputed _marker_classes — zero disk I/O.
    _item_present_classes: Dict[int, set] = {}
    if stage >= 2 and seg_mode == "tissue":
        _n_checked = 0
        for _idx, _it in enumerate(train_items):
            if _it.dataset_name != "SegPath":
                continue  # multi-class — all declared present, skip
            _n_checked += 1
            try:
                _cid = int(_it.dataset._marker_classes[_it.local_idx])
                _present = {_cid} if 1 <= _cid < NUM_TISSUE_CLASSES else set()
            except Exception:
                _present = set()
            _item_present_classes[_idx] = _present
        if _n_checked:
            _n_before = _n_checked * len(_DS_CLASSES.get("SegPath", frozenset()))  # 3
            _n_after = sum(len(s) for s in _item_present_classes.values())
            print(f"[SegPath fix] Checked {_n_checked:,} SegPath images (zero disk I/O).  "
                  f"Pairs: {_n_after:,} / {_n_before:,} "
                  f"({_n_before - _n_after:,} absent-class pairs filtered).")

    # ── DataLoader factory: build a class-balanced DataLoader from any item subset ──
    from collections import defaultdict as _dd
    def _make_train_loader(items: List[_Item], label: str = "",
                           hm_dice: Optional[Dict[str, Dict[str, float]]] = None) -> DataLoader:
        """Build a class-balanced DataLoader from `items`.

        Stage 1: WeightedRandomSampler (image-level class balance).
        Stage 2: ClassConditionedDataset (tile-level class balance).

        ``hm_dice``: if provided, nested dict ``class_key → dataset → dice``.
        Built into per-class-per-dataset inverse-Dice weights so poorly-performing
        (class, dataset) combos are sampled more often.
        """
        _hm_w: Dict[str, Dict[str, float]] = {}
        if hm_dice:
            # Inverse Dice per (class, dataset): low Dice → high weight.
            # Clip to [0.05, 1.0] range, then normalize by global max.
            _all_vals = []
            for _ck, _ds_dict in hm_dice.items():
                _hm_w[_ck] = {}
                for _ds, _d in _ds_dict.items():
                    _w = 1.0 / max(_d + 0.1, 0.05)
                    _hm_w[_ck][_ds] = _w
                    _all_vals.append(_w)
            _max_w = max(_all_vals) if _all_vals else 1.0
            if _max_w > 0:
                for _ck in _hm_w:
                    for _ds in _hm_w[_ck]:
                        _hm_w[_ck][_ds] /= _max_w  # norm to [0,1]

        _actual_class_cache: Dict[Tuple[str, int], frozenset] = {}
        def _actual_class_keys(_it: _Item) -> frozenset:
            """Return declared classes that are truly present in this image mask."""
            _cache_key = (_it.dataset_name, int(_it.local_idx))
            if _cache_key in _actual_class_cache:
                return _actual_class_cache[_cache_key]
            _decl = _DS_CLASSES.get(_it.dataset_name, frozenset())
            _present = set()
            try:
                _raw = _it.dataset[_it.local_idx]
                _tm = _raw.get("tissue_mask")
                if _tm is not None:
                    for _cid in np.unique(_tm):
                        _ck = f"T:{int(_cid)}"
                        if int(_cid) > 0 and _ck in _decl:
                            _present.add(_ck)
                _nm = _raw.get("nuclei_mask")
                if _nm is not None:
                    for _cid in np.unique(_nm):
                        _ck = f"N:{int(_cid)}"
                        if int(_cid) > 0 and _ck in _decl:
                            _present.add(_ck)
            except Exception:
                _present = set()
            if not _present:
                # Fallback keeps unusual datasets trainable, but the normal path
                # avoids absent-class random crops for multi-class masks.
                _present = set(_decl)
            _actual_class_cache[_cache_key] = frozenset(_present)
            return _actual_class_cache[_cache_key]

        if stage >= 2:
            # Tile-level class balance via ClassConditionedDataset
            # Dataset-balanced: after a class is chosen, each dataset containing
            # that class has equal total weight (not proportional to dataset size).
            _class_items: List[Tuple[_Item, str]] = []
            _cc: Dict[str, int] = _dd(int)
            _cdc: Dict[str, Dict[str, int]] = _dd(lambda: _dd(int))  # class→dataset→count
            _n_items = len(items)
            for _idx, _it in enumerate(items):
                if _idx > 0 and _idx % 1000 == 0:
                    print(f"  [class-scan] {_idx:,}/{_n_items:,} items …", flush=True)
                _present = _item_present_classes.get(_idx, set())
                for _ck in _actual_class_keys(_it):
                    _cid = int(_ck[2:])
                    if _present and _cid not in _present:
                        continue
                    _class_items.append((_it, _ck))
                    _cc[_ck] += 1
                    _cdc[_ck][_it.dataset_name] += 1
            # Compute dataset-balanced weights per (item, ck) pair.
            # weight = 1/(num_datasets_for_class * count_in_this_dataset_for_class)
            # This gives each dataset equal total weight for the class.
            _pair_ws = []
            for _it, _ck in _class_items:
                _n_ds = len(_cdc[_ck])
                _cnt  = _cdc[_ck][_it.dataset_name]
                _db_w = 1.0 / max(_n_ds * _cnt, 1)
                if _hm_w:
                    _db_w *= _hm_w.get(_ck, {}).get(_it.dataset_name, 0.5)  # blend with per-dataset inverse-Dice weight
                _pair_ws.append(_db_w)
            if label:
                print(f"  [{label}] Class-conditioned tile sampler (dataset-balanced): "
                      f"{len(_class_items)} (item, class) pairs from {len(items)} images")
                for _ck, _cnt in sorted(_cc.items()):
                    _n_ds = len(_cdc[_ck])
                    _hint = ""
                    if _hm_w and _ck in _hm_w:
                        _ds_ws = ", ".join(f"{ds}={_hm_w[_ck].get(ds, 0):.2f}"
                                           for ds in sorted(_hm_w[_ck]))
                        _hint = f"  hm_w=[{_ds_ws}]"
                    print(f"    {_ck}: {_cnt:6d} image-pairs from {_n_ds} datasets{_hint}")
            _ds  = ClassConditionedDataset(_class_items,
                                            native_crop_at_40x=native_crop_at_40x)
            _samp = WeightedRandomSampler(
                _pair_ws, num_samples=len(_class_items), replacement=True)
            _lkw = dict(batch_size=1, sampler=_samp, collate_fn=list_collate,
                        num_workers=_num_workers, pin_memory=True)
            if _num_workers > 0:
                _lkw.update(persistent_workers=True, prefetch_factor=4)
            return DataLoader(_ds, **_lkw)
        else:
            # Stage 1: image-level class balance via WeightedRandomSampler
            # Dataset-balanced: after a class is chosen, each dataset containing
            # that class has equal total weight.
            _ds = SAM3CombinedDataset(items, augment=True, fg_crop_prob=fg_crop_prob,
                                       native_crop_at_40x=native_crop_at_40x,
                                       crop_to_target=magnification_alignment,
                                       normalize_magnification=magnification_alignment)
            _cc: Dict[str, int] = _dd(int)
            _cdc: Dict[str, Dict[str, int]] = _dd(lambda: _dd(int))  # class→dataset→count
            for _it in items:
                for _ck in _DS_CLASSES.get(_it.dataset_name, frozenset()):
                    _cc[_ck] += 1
                    _cdc[_ck][_it.dataset_name] += 1
            _ws = []
            for _it in items:
                _cks = _DS_CLASSES.get(_it.dataset_name, frozenset())
                # Dataset-balanced weight: 1/(num_datasets * count_in_this_dataset)
                # per class, summed across all classes this image carries.
                _cb_w = 0.0
                for _ck in _cks:
                    _n_ds = len(_cdc[_ck])
                    _cnt  = _cdc[_ck][_it.dataset_name]
                    _cb_w += 1.0 / max(_n_ds * _cnt, 1)
                if _cb_w == 0.0:
                    _cb_w = 1e-8
                if _hm_w:
                    _hm_blend = sum(_hm_w.get(_ck, {}).get(_it.dataset_name, 0.5)
                                    for _ck in _cks) if _cks else 0.5
                    # Blend: 70% class-balance, 30% hard-mining
                    _ws.append(0.7 * _cb_w + 0.3 * _hm_blend)
                else:
                    _ws.append(_cb_w)
            if label:
                mode = "hard-mining" if _hm_w else "dataset-balanced"
                print(f"  [{label}] {mode} sampler: {len(items)} items")
                for _ck, _cnt in sorted(_cc.items()):
                    _n_ds = len(_cdc[_ck])
                    _hint = ""
                    if _hm_w and _ck in _hm_w:
                        _ds_ws = ", ".join(f"{ds}={_hm_w[_ck].get(ds, 0):.2f}"
                                           for ds in sorted(_hm_w[_ck]))
                        _hint = f"  hm_w=[{_ds_ws}]"
                    print(f"    {_ck}: {_cnt:6d} samples from {_n_ds} datasets{_hint}")
            _samp = WeightedRandomSampler(_ws, num_samples=len(items), replacement=True)
            _lkw  = dict(batch_size=1, sampler=_samp, collate_fn=list_collate,
                         num_workers=_num_workers, pin_memory=True)
            if _num_workers > 0:
                _lkw.update(persistent_workers=True, prefetch_factor=4)
            return DataLoader(_ds, **_lkw)

    # ── Curriculum: warmup on multi-class datasets per modality, then full mix ──
    # Warmup set is modality-specific:
    #   tissue  → BCSS, SICAPv2, WSSS4LUAD  (≥2 tissue classes each)
    #   nuclei  → NuCLS, PanNuke, CoNIC2022, Lizard, MoNuSAC  (≥2 nuclei classes each)
    #   both    → union of the above
    # This prevents the SegPath (T:1 only, 94% of tissue data) and single-class
    # nuclei datasets (Kumar/CPM/TNBC: only N:1) from dominating early training.
    if seg_mode == "tissue":
        _warmup_ds = _MULTICLS_TISSUE_DATASETS
    elif seg_mode == "nuclei":
        _warmup_ds = _MULTICLS_NUCLEI_DATASETS
    else:
        _warmup_ds = _MULTICLS_DATASETS
    _warmup_items = (
        [it for it in train_items if it.dataset_name in _warmup_ds]
        if curriculum else train_items
    )
    if curriculum:
        print(f"\n[Curriculum] seg_mode={seg_mode}  warmup_ds: {', '.join(sorted(_warmup_ds))}")
        print(f"  Warmup for {curriculum_warmup_epochs} epoch(s) on "
              f"{len(_warmup_items)} items")
        print(f"  Phase 2: full {len(train_items)}-item dataset from epoch "
              f"{curriculum_warmup_epochs} onwards.")
        if focal_gamma > 0:
            print(f"  Focal loss: gamma={focal_gamma}")
        print()
    # start_epoch is 0 for fresh runs; overridden below if resuming from a checkpoint.
    # Must be defined before the curriculum phase decision which depends on it.
    start_epoch  = 0

    # Choose starting phase (handles resume correctly)
    _curr_phase  = (
        "warmup" if (curriculum and curriculum_warmup_epochs > 0
                     and start_epoch < curriculum_warmup_epochs)
        else "full"
    )
    train_loader = _make_train_loader(
        _warmup_items if _curr_phase == "warmup" else train_items,
        label="warmup" if _curr_phase == "warmup" else "full")

    # ── Classification-only DataLoader (Stage 3+) ──────────────────────
    _cls_loader = None
    if _cls_only_items:
        _cls_ds = SAM3CombinedDataset(_cls_only_items, augment=True,
                                       fg_crop_prob=fg_crop_prob,
                                       native_crop_at_40x=native_crop_at_40x,
                                       crop_to_target=magnification_alignment,
                                       normalize_magnification=magnification_alignment)
        _cls_loader = DataLoader(_cls_ds, batch_size=1, shuffle=True,
                                  collate_fn=list_collate,
                                  num_workers=_num_workers, pin_memory=True,
                                  persistent_workers=(_num_workers > 0))
        print(f"  [cls-only] Separate loader: {len(_cls_only_items):,} items "
              f"(tissue-type + tile classification only)")

    # Optimizer: separate param groups
    # LR priority (highest to lowest):
    #   prompt_enc → --prompt_lr (CLI), then --lr (CLI), then LR_PROMPT (hardcoded)
    #   LoRA       → --lr (CLI), then LR_LORA (hardcoded)
    #   adapter    → --adapter_lr (CLI), then --lr (CLI), then LR_ADAPTER (hardcoded)
    _lr_prompt  = prompt_lr if prompt_lr is not None else (
                     lr if lr is not None else LR_PROMPT)
    _lr_lora    = lr if lr is not None else LR_LORA
    _lr_adapter = adapter_lr if adapter_lr is not None else (
                     LR_ADAPTER if stage == 2 else (
                         lr if lr is not None else LR_ADAPTER))
    if stage >= 2 and freeze_prompts_after_stage1:
        _n_prompt_frozen = 0
        for p in prompt_enc.parameters():
            if p.requires_grad:
                p.requires_grad_(False)
                _n_prompt_frozen += 1
        print(f"[Stage {stage}] Prompt encoder frozen ({_n_prompt_frozen} tensors).")
    prompt_params = [p for p in prompt_enc.parameters() if p.requires_grad]
    param_groups = []
    if prompt_params:
        param_groups.append({"params": prompt_params, "lr": _lr_prompt})
    if lora_cfg is not None:
        if freeze_lora_in_stage2 and stage == 2:
            # Stage 2: freeze LoRA so only adapter+prompt train.
            # LoRA representations from stage 1 are already well-optimised;
            # mutating them under the new density-consistency + ortho losses
            # causes the degradation we observed.  Gradients are disabled
            # entirely (not just lr=0) to save memory and compute.
            lora_params = []
            _n_frozen = 0
            for n, p in model.named_parameters():
                if "lora" in n.lower() and p.requires_grad:
                    p.requires_grad_(False)
                    _n_frozen += 1
            print(f"[Stage 2] LoRA frozen ({_n_frozen} params) — "
                  f"only adapter + prompt_enc will train.")
        else:
            lora_params = [p for n, p in model.named_parameters()
                           if "lora" in n.lower() and p.requires_grad]
            param_groups.append({"params": lora_params, "lr": _lr_lora})
    if adapter is not None:
        param_groups.append({"params": list(adapter.parameters()), "lr": _lr_adapter})
    if tile_classifier is not None:
        param_groups.append({"params": list(tile_classifier.parameters()), "lr": _lr_lora})
    if ssl_head is not None:
        param_groups.append({"params": list(ssl_head.parameters()), "lr": _lr_lora})
    if pixel_ssl_head is not None:
        param_groups.append({"params": list(pixel_ssl_head.parameters()), "lr": _lr_lora})
    if density_fusion is not None:
        param_groups.append({"params": list(density_fusion.parameters()), "lr": _lr_adapter})
    if density_gate_net is not None:
        param_groups.append({"params": list(density_gate_net.parameters()), "lr": _lr_adapter})
    if contrast_proj is not None:
        param_groups.append({"params": list(contrast_proj.parameters()), "lr": _lr_lora})

    # ── Separate param lists for independent gradient clipping ──────────
    # Classification heads (fresh MLPs with CE loss) produce much larger
    # gradients than LoRA/prompt params.  If clipped together, the global
    # norm is dominated by cls heads, starving segmentation of effective LR.
    _seg_params = []   # prompt_enc + LoRA + adapter + density modules
    _cls_params = []   # tile_classifier + ssl_head + pixel_ssl_head
    # Collect explicitly by source to be robust to group ordering
    _seg_params.extend(list(prompt_enc.parameters()))
    if lora_cfg is not None and lora_params:
        _seg_params.extend(lora_params)
    if adapter is not None:
        _seg_params.extend(list(adapter.parameters()))
    if density_fusion is not None:
        _seg_params.extend(list(density_fusion.parameters()))
    if density_gate_net is not None:
        _seg_params.extend(list(density_gate_net.parameters()))
    if contrast_proj is not None:
        _seg_params.extend(list(contrast_proj.parameters()))
    if tile_classifier is not None:
        _cls_params.extend(list(tile_classifier.parameters()))
    if ssl_head is not None:
        _cls_params.extend(list(ssl_head.parameters()))
    if pixel_ssl_head is not None:
        _cls_params.extend(list(pixel_ssl_head.parameters()))
    # Deduplicate (some params may appear in multiple groups via DataParallel wrappers)
    _seg_params = list(set(_seg_params))
    _cls_params = list(set(_cls_params))

    opt    = AdamW(param_groups, weight_decay=WEIGHT_DECAY)

    # Step-based warmup + cosine decay.
    # Warmup for first 5% of optimizer steps (min 200, max 1000), then cosine to 1% of peak LR.
    # NOTE: the train loader's effective length (= number of batches it yields)
    # is the only correct unit of "steps per epoch", because:
    #   - Stage 2+ uses a (item, class)-pair sampler whose length exceeds
    #     len(train_items) (multiple pairs per image).
    #   - The internal DataLoader batch_size is fixed to 1 (the `batch_size`
    #     arg is used for grad-accum, not loader batching).
    _loader_len = len(train_loader)
    _eff_epoch_size = _loader_len if steps_per_epoch is None \
        else min(steps_per_epoch, _loader_len)
    _steps_per_epoch = max(1, _eff_epoch_size)
    # When --total_steps is given, it overrides epoch-based scheduling.
    # The training loop runs for as many epochs as needed to accumulate
    # `total_steps` optimizer steps; LR cosine is computed against total_steps.
    if total_steps is not None and total_steps > 0:
        _total_opt_steps = int(total_steps)
        # Inflate epoch budget so the outer loop never terminates early; the
        # inner loop's `n_steps >= total_steps` check is the real terminator.
        _epoch_budget = max(epochs, int(math.ceil(total_steps / max(1, _steps_per_epoch))) + 1)
        print(f"  LR schedule: cosine decay / {_total_opt_steps} total optimizer steps "
              f"(--total_steps overrides --epochs; ~{_steps_per_epoch} steps/epoch, "
              f"will run up to {_epoch_budget} epochs)")
        epochs = _epoch_budget
    else:
        _total_opt_steps    = _steps_per_epoch * epochs
        print(f"  LR schedule: cosine decay (no warmup) / {_total_opt_steps} total optimizer steps "
              f"({_steps_per_epoch} steps/epoch × {epochs} epochs)")
    # Adjust displayed epoch size for the warmup phase
    if _curr_phase == "warmup":
        _eff_epoch_size = len(_warmup_items) if steps_per_epoch is None \
            else min(steps_per_epoch, len(_warmup_items))

    if stage > 2:
        # Stages 3/4: use PyTorch's native CosineAnnealingLR.
        # Each param group decays proportionally from its base LR → 0.
        scheduler = CosineAnnealingLR(opt, T_max=_total_opt_steps, eta_min=0.0)
        print(f"  [Stage {stage}] Using CosineAnnealingLR (T_max={_total_opt_steps}, eta_min=0)")
    else:
        # Stages 1/2: proportional cosine decay with a 1% floor via LambdaLR.
        def _lr_lambda(current_opt_step: int) -> float:
            progress = float(current_opt_step) / float(max(1, _total_opt_steps))
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = LambdaLR(opt, lr_lambda=_lr_lambda)
    scaler    = GradScaler()

    best_dice         = 0.0
    best_holdout_dice = 0.0
    global_step  = 0
    opt_step     = 0
    recent_losses: List[float] = []

    # ── Resume from checkpoint ───────────────────────────────────────────────
    if resume_ckpt:
        print(f"\n[Resume] Loading training state from: {resume_ckpt}")
        print(f"[Resume] resume_load='{resume_load}' "
              f"(choices: all | lora | prompt | none)")
        _rc = torch.load(resume_ckpt, map_location=device, weights_only=False)
        _raw_m = model.module if hasattr(model, "module") else model

        # ── Reload prompt_enc ─────────────────────────────────────────
        if resume_load in ("all", "prompt"):
            if "prompt_enc" in _rc:
                _load_compat(
                    _raw_m.backbone.language_backbone,
                    _rc["prompt_enc"], tag="resume_prompt_enc",
                    device=device)
        else:
            print(f"  [Resume] Skipping prompt_enc (resume_load='{resume_load}')")

        # ── Reload LoRA weights ───────────────────────────────────────
        if resume_load in ("all", "lora"):
            if lora_cfg is not None and _rc.get("lora_state"):
                _sd = _raw_m.state_dict()
                _src_lora = {k: v.to(device) for k, v in _rc["lora_state"].items()}
                _sd.update(_src_lora)
                _raw_m.load_state_dict(_sd)
                print(f"  [Resume/lora] Loaded {len(_src_lora)} keys")
            elif lora_cfg is None:
                print(f"  [Resume] Skipping lora (not configured in this stage)")
        else:
            print(f"  [Resume] Skipping LoRA (resume_load='{resume_load}')")

        # ── Reload Adapter weights ────────────────────────────────────
        if resume_load in ("all",):
            if adapter is not None and _rc.get("adapter_state"):
                _load_compat(
                    adapter,
                    _rc["adapter_state"], tag="resume_adapter",
                    device=device)
            if tile_classifier is not None and _rc.get("tile_cls_state"):
                _load_compat(tile_classifier, _rc["tile_cls_state"],
                             tag="resume_tilecls", device=device)
            if ssl_head is not None and _rc.get("ssl_head_state"):
                _load_compat(ssl_head, _rc["ssl_head_state"],
                             tag="resume_ssl", device=device)
            if pixel_ssl_head is not None and _rc.get("pixel_ssl_state"):
                _load_compat(pixel_ssl_head, _rc["pixel_ssl_state"],
                             tag="resume_pixel_ssl", device=device)
            if density_fusion is not None and _rc.get("density_fusion_state"):
                _load_compat(density_fusion, _rc["density_fusion_state"],
                             tag="resume_density_fusion", device=device)
            if density_gate_net is not None and _rc.get("density_gate_state"):
                _load_compat(density_gate_net, _rc["density_gate_state"],
                             tag="resume_density_gate", device=device)
            if contrast_proj is not None and _rc.get("contrast_proj_state"):
                _load_compat(contrast_proj, _rc["contrast_proj_state"],
                             tag="resume_contrast_proj", device=device)
        else:
            print(f"  [Resume] Skipping adapter (resume_load='{resume_load}')")

        # ── Restore optimiser / scheduler / scaler ────────────────────
        # WARNING: opt.load_state_dict does NOT validate tensor shapes — it
        # silently loads wrong-shaped momentum buffers, and the crash only
        # surfaces later at Adam.step().  Detect mismatch preemptively by
        # comparing the total element count of state tensors vs current params.
        _restore_opt = True
        if _rc.get("opt_state"):
            _ckpt_state = _rc["opt_state"].get("state", {})
            _ckpt_nel = sum(
                _state[k].numel()
                for _state in _ckpt_state.values()
                for k in ("exp_avg", "exp_avg_sq") if k in _state
            )
            _cur_nel = sum(p.numel() for pg in opt.param_groups for p in pg["params"])
            if _ckpt_nel != _cur_nel:
                _restore_opt = False
                print(f"  [Resume] WARNING: Optimizer state shape mismatch — "
                      f"skipping optimizer/scheduler/scaler restoration. "
                      f"(checkpoint has {_ckpt_nel:,} state elements, "
                      f"current model has {_cur_nel:,}; "
                      f"common cause: prompt count changed between runs)")
            else:
                try:
                    opt.load_state_dict(_rc["opt_state"])
                    print(f"  [Resume] Optimizer state restored")
                except Exception as _opt_e:
                    _restore_opt = False
                    print(f"  [Resume] WARNING: Optimizer state load failed "
                          f"({_opt_e}). Skipping restore.")
        if _restore_opt:
            if _rc.get("scheduler_state"):
                scheduler.load_state_dict(_rc["scheduler_state"])
            if _rc.get("scaler_state"):
                scaler.load_state_dict(_rc["scaler_state"])
        # Restore counters
        global_step   = _rc.get("global_step", 0)
        opt_step      = _rc.get("opt_step", 0)
        best_dice         = _rc.get("best_dice", 0.0)
        best_holdout_dice = _rc.get("best_holdout_dice", 0.0)
        if reset_best_dice:
            print(f"[Resume] --reset_best_dice: ignoring previous best_dice={best_dice:.4f} "
                  f"and best_holdout_dice={best_holdout_dice:.4f}, "
                  f"starting from 0.0 (validation set changed)")
            best_dice = 0.0
            best_holdout_dice = 0.0
        recent_losses = list(_rc.get("recent_losses", []))
        start_epoch   = _rc.get("epoch", -1) + 1
        print(f"[Resume] Restored: epoch={_rc.get('epoch')}, global_step={global_step}, "
              f"opt_step={opt_step}, best_dice={best_dice:.4f}, "
              f"resuming from epoch {start_epoch}")
        del _rc
    # ─────────────────────────────────────────────────────────────────────────

    # CSV helpers — written once per file, then appended
    val_csv_nocond_path = save_dir / "val_metrics_nocond.csv"
    val_csv_cond_path   = save_dir / "val_metrics_cond.csv"
    cls_csv_nocond_path = save_dir / "per_class_metrics_nocond.csv"
    cls_csv_cond_path   = save_dir / "per_class_metrics_cond.csv"
    val_hdrs_written:  Dict[str, bool] = {"cond": False, "nocond": False}
    cls_hdrs_written:  Dict[str, bool] = {"cond": False, "nocond": False}

    # Hold-out val CSV paths (separate from train quick-val)
    hold_val_csv_nocond_path = save_dir / "holdout_val_metrics_nocond.csv"
    hold_val_csv_cond_path   = save_dir / "holdout_val_metrics_cond.csv"
    hold_cls_csv_nocond_path = save_dir / "holdout_per_class_metrics_nocond.csv"
    hold_cls_csv_cond_path   = save_dir / "holdout_per_class_metrics_cond.csv"
    hold_hdrs_written: Dict[str, bool] = {"cond": False, "nocond": False}
    hold_cls_hdrs_written: Dict[str, bool] = {"cond": False, "nocond": False}
    hold_cls_ds_hdrs_written: Dict[str, bool] = {"cond": False, "nocond": False}

    # ── Hard example mining: track per-class Dice from validation ────────
    # Updated after each periodic validation.  Used to up-weight poorly
    # performing classes in the training sampler.
    _per_class_dice: Dict[str, Dict[str, float]] = {}  # class_key → dataset → dice (cond mode)
    _use_hard_mining = (hard_mining_start_epoch >= 0)

    # Register before initial eval so step-0 metrics match training/test behavior.
    _raw_m = model.module if hasattr(model, "module") else model
    _raw_m._density_fusion_mod = density_fusion
    _raw_m._pending_density = None
    _trunk_hook_handle = (
        _register_trunk_feature_hook(model, stage_label=f"stage{stage}")
        if (tile_classifier is not None or density_fusion is not None or contrast_proj is not None)
        else None
    )
    _decoder_feat_hook_handle = None

    def _append_val_rows(
        summary: dict, pcls: list, cond_key: str, gstep: int, ep: int,
        loss: float, cls_rows: Optional[list] = None,
    ):
        v_path = val_csv_cond_path   if cond_key == "cond" else val_csv_nocond_path
        c_path = cls_csv_cond_path   if cond_key == "cond" else cls_csv_nocond_path
        row = {"global_step": gstep, "epoch": ep,
               "train_loss_recent": loss, **summary}
        with open(v_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            # Check file is empty (or doesn't exist) — handles resume correctly
            if not val_hdrs_written[cond_key] and (not v_path.exists() or v_path.stat().st_size == 0):
                w.writeheader()
            val_hdrs_written[cond_key] = True
            w.writerow(row)
        # Write per-dataset classification metrics to the cls CSV
        if cls_rows:
            with open(c_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(cls_rows[0].keys()))
                if not cls_hdrs_written[cond_key] and (not c_path.exists() or c_path.stat().st_size == 0):
                    w.writeheader()
                cls_hdrs_written[cond_key] = True
                w.writerows(cls_rows)

    def _eval_and_log_holdout(
        ds, log_prefix: str, gstep: int, ep: int, tag: str = "",
        save_vis: bool = True,
    ) -> Optional[float]:
        """Evaluate on the hold-out val set, save CSV + vis.

        Returns the mean of mean_dice across all evaluated modes (cond/nocond),
        or None if `ds` is None. The returned value is used by the caller for
        an *additional* best-on-holdout checkpoint, but never influences the
        primary `best_model.pth` selection (still driven by train quick-val).
        When save_vis=False, only CSV is written (no vis images)."""
        if ds is None:
            return None
        _hold_means: List[float] = []
        prompt_enc.eval()
        for _oc, _oc_tag in _eval_modes:
            _vs, _pcls, _cls_rows = evaluate(
                model, transform, ds, device,
                vis_dir=os.path.join(vis_dir, "holdout_val") if save_vis else None,
                epoch=global_step, vis_num=min(vis_num, len(ds)),
                tag=f"{tag}_{_oc_tag}" if tag else f"step_{gstep:07d}_{_oc_tag}",
                organ_cond=_oc, seg_mode=seg_mode,
                tile_classifier=tile_classifier,
                              prompt_enc=prompt_enc, adapter=adapter, histoplus=histoplus, density_fusion=density_fusion,
                vis_mode="best_per_dataset")
            _hold_means.append(float(_vs.get("mean_dice", 0.0)))
            print(f"  [{_oc_tag}] TissueDice {_vs['mean_tissue_dice']:.4f} | "
                  f"NucleiDice {_vs['mean_nuclei_dice']:.4f}"
                  + (f" | OrganAcc {_vs['tissue_type_acc']:.4f}"
                     if "tissue_type_acc" in _vs else "")
                  + (f" | TClsAcc {_vs['tissue_cls_hamming']:.4f}"
                     if "tissue_cls_hamming" in _vs else ""))
            # ── Per-class Dice per dataset ──
            if _pcls:
                for _r in _pcls:
                    print(f"   {log_prefix} [{_oc_tag}] {_r['dataset']:20s}  "
                          f"{_r['class_key']:35s}  "
                          f"Dice={_r['dice']:.4f}  IoU={_r['iou']:.4f}")
            # ── Per-dataset metrics ──
            if _cls_rows:
                for _r in _cls_rows:
                    _parts = [f"  [{_oc_tag}] {_r['dataset']:20s}"]
                    if "tissue_dice" in _r:
                        _parts.append(f"TisDice={_r['tissue_dice']:.4f}")
                    if "nuclei_dice" in _r:
                        _parts.append(f"NucDice={_r['nuclei_dice']:.4f}")
                    if "organ_acc" in _r:
                        _parts.append(f"OrganAcc={_r['organ_acc']:.4f}")
                    if "tissue_hamming" in _r:
                        _parts.append(f"TClsAcc={_r['tissue_hamming']:.4f}")
                    print("   " + "  ".join(_parts))
            # Write hold-out CSV
            v_path = hold_val_csv_cond_path if _oc_tag == "cond" else hold_val_csv_nocond_path
            c_path = hold_cls_csv_cond_path if _oc_tag == "cond" else hold_cls_csv_nocond_path
            row = {"global_step": gstep, "epoch": ep, **{k: _vs.get(k, 0) for k in
                   ("mean_tissue_dice", "mean_nuclei_dice", "mean_dice",
                    "tissue_type_acc", "tissue_cls_hamming")}}
            with open(v_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not hold_hdrs_written[_oc_tag] and (not v_path.exists() or v_path.stat().st_size == 0):
                    w.writeheader()
                hold_hdrs_written[_oc_tag] = True
                w.writerow(row)
            if _pcls:
                with open(c_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(_pcls[0].keys()))
                    if not hold_cls_hdrs_written[_oc_tag] and (not c_path.exists() or c_path.stat().st_size == 0):
                        w.writeheader()
                    hold_cls_hdrs_written[_oc_tag] = True
                    w.writerows(_pcls)
            # ── Per-dataset classification CSV ──
            _cls_ds_path = hold_cls_csv_cond_path if _oc_tag == "cond" else hold_cls_csv_nocond_path
            # Use a separate file: hold_cls_per_ds_{cond/nocond}.csv
            _cls_ds_path = Path(str(c_path).replace("hold_cls_", "hold_cls_per_ds_"))
            if _cls_rows:
                with open(_cls_ds_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(_cls_rows[0].keys()))
                    if not hold_cls_ds_hdrs_written.get(_oc_tag, False) and (not _cls_ds_path.exists() or _cls_ds_path.stat().st_size == 0):
                        w.writeheader()
                    hold_cls_ds_hdrs_written[_oc_tag] = True
                    w.writerows(_cls_rows)
            if tb_writer:
                tb_writer.add_scalar(f"holdout_{_oc_tag}/mean_dice",
                                     _vs["mean_dice"],        global_step)
                if "tissue_type_acc" in _vs:
                    tb_writer.add_scalar(f"holdout_{_oc_tag}/tissue_type_acc",
                                         _vs["tissue_type_acc"], global_step)
                if "tissue_cls_hamming" in _vs:
                    tb_writer.add_scalar(f"holdout_{_oc_tag}/tissue_cls_hamming",
                                         _vs["tissue_cls_hamming"], global_step)
        prompt_enc.train()
        return float(np.max(_hold_means)) if _hold_means else None

    # ── Initial evaluation before any training ───────────────────────────────
    if run_initial_eval:
        print("\n─── Initial eval (step 0) ───")
        _eval_with_checkpointing(model)
        prompt_enc.eval()
        _init_summaries = {}
        _init_pcls = None  # per-class rows (cond mode, for hard-mining seed)
        for _oc, _oc_tag in _eval_modes:
            _s, _p, _cls_rows = evaluate(model, transform, quick_val_ds, device,
                              vis_dir=os.path.join(vis_dir, "val"), epoch=0,
                              vis_num=max(1, len(quick_val_ds)),
                              tag=f"step_0000000_{_oc_tag}",
                              organ_cond=_oc, seg_mode=seg_mode,
                              tile_classifier=tile_classifier,
                              prompt_enc=prompt_enc, adapter=adapter, histoplus=histoplus, density_fusion=density_fusion,
                              vis_mode="best_per_dataset",
                              save_density_vis=(histoplus is not None),
                              cls_only_seg_vis=False)
            _init_summaries[_oc_tag] = _s
            if _oc_tag == "cond":
                _init_pcls = _p   # capture for hard-mining seed below
            print(f"  [{_oc_tag}] TissueDice {_s['mean_tissue_dice']:.4f} | ",
                  f"NucleiDice {_s['mean_nuclei_dice']:.4f}",
                  end="")
            if "tissue_type_acc" in _s:
                print(f" | OrganAcc {_s['tissue_type_acc']:.4f}", end="")
            if "tissue_cls_hamming" in _s:
                print(f" | TClsAcc {_s['tissue_cls_hamming']:.4f}", end="")
            print()
            if _cls_rows:
                for _r in _cls_rows:
                    _parts = [f"[{_oc_tag}] {_r['dataset']:20s}"]
                    if "tissue_dice" in _r:
                        _parts.append(f"TisDice={_r['tissue_dice']:.4f}")
                    if "nuclei_dice" in _r:
                        _parts.append(f"NucDice={_r['nuclei_dice']:.4f}")
                    if "organ_acc" in _r:
                        _parts.append(f"OrganAcc={_r['organ_acc']:.4f}")
                    if "tissue_hamming" in _r:
                        _parts.append(f"TClsAcc={_r['tissue_hamming']:.4f}")
                    print("   " + "  ".join(_parts))
            _append_val_rows(_s, _p, _oc_tag, gstep=0, ep=-1, loss=float("nan"), cls_rows=_cls_rows)
            if tb_writer:
                tb_writer.add_scalar(f"val_{_oc_tag}/mean_dice",   _s["mean_dice"],        0)
                tb_writer.add_scalar(f"val_{_oc_tag}/tissue_dice", _s["mean_tissue_dice"], 0)
                tb_writer.add_scalar(f"val_{_oc_tag}/nuclei_dice", _s["mean_nuclei_dice"], 0)
        prompt_enc.train()
        # ── Use initial eval Dice as best_dice floor ─────────────────
        # When loading from a previous-stage checkpoint, the loaded model
        # already has a certain Dice on the current val set.  Setting
        # best_dice to this value prevents saving checkpoints that are
        # worse than the loaded model (e.g. before the adapter/tile_cls
        # heads have learned anything useful).  Only applies when the
        # initial Dice exceeds the current best_dice (which is 0.0 for
        # fresh stages and the resumed value for resumed stages).
        if _init_summaries:
            _init_dice = float(np.max([
                _init_summaries[k]["mean_dice"] for k in _init_summaries]))
            if _init_dice > best_dice:
                print(f"  [Init] Updated best_dice: {best_dice:.4f} → {_init_dice:.4f} "
                      f"(from initial eval on current val set)")
                best_dice = _init_dice
        # ── Populate per-class Dice for hard example mining from step 0 ─
        # Without this, hard mining is a no-op until the first periodic
        # validation — meaning the first epoch gets no per-class weighting.
        if _use_hard_mining and _init_pcls:
            for _r in _init_pcls:
                _per_class_dice.setdefault(_r["class_key"], {})[_r["dataset"]] = _r["dice"]
            _total_cb = sum(len(ds_dict) for ds_dict in _per_class_dice.values())
            print(f"  [Init] Hard-mining seeded with {_total_cb} class×dataset Dice values")
    else:
        print("\n─── Initial eval skipped for speed ───")
    # ─────────────────────────────────────────────────────────────────────────

    # ── Register persistent trunk hook to capture ViT features for
    # dual-head classifier (Stage 3+).
    _raw_m = model.module if hasattr(model, "module") else model
    # Expose density_fusion module and per-sample pending density to the
    # trunk hook (F1 fusion: applied to block-31 output before the neck).
    _raw_m._density_fusion_mod = density_fusion
    _raw_m._pending_density = None
    # The trunk hook is normally installed before initial eval.  Keep this
    # fallback for unusual call paths where no early hook was needed.
    _neck = getattr(_raw_m.backbone, "vision_backbone", None)
    if _trunk_hook_handle is None and _neck is not None and (
            tile_classifier is not None or density_fusion is not None
            or contrast_proj is not None or morph_projector is not None):
        _trunk = getattr(_neck, "trunk", None)
        if _trunk is not None:
            _raw_m._trunk_features_for_cls = None
            _raw_m._trunk_features_for_organ = None
            _raw_m._trunk_features_spatial = None
            def _capture_trunk(_m, _i, _o):
                # _o is List[Tensor] from ViT stages (4 outputs at global attn blocks 7,15,23,31).
                # Organ cls uses deep features (_o[-1] = block 31) — LIVE gradient.
                # Tissue cls also uses deep features but detached at head.
                if isinstance(_o, (list, tuple)) and len(_o) > 0:
                    _outs = list(_o)
                    _f_deep = _outs[-1]
                    if _f_deep.ndim == 4:
                        _fusion_mod = getattr(_raw_m, "_density_fusion_mod", None)
                        _dens = getattr(_raw_m, "_pending_density", None)
                        _fusion_active = False
                        if _fusion_mod is not None and _dens is not None:
                            _d = _dens.to(_f_deep.device, dtype=_f_deep.dtype)
                            if _d.ndim == 2:
                                _d = _d.unsqueeze(0).unsqueeze(0)
                            elif _d.ndim == 3:
                                _d = _d.unsqueeze(1)
                            _d = F.interpolate(
                                _d, size=_f_deep.shape[-2:],
                                mode="bilinear", align_corners=False)
                            _f_deep = _fusion_mod(_f_deep, _d)
                            _outs[-1] = _f_deep
                            _fusion_active = True
                        # Capture deep ViT features after optional density fusion.
                        _pooled = _f_deep.mean(dim=(2, 3))
                        _raw_m._trunk_features_for_organ = _pooled  # LIVE grad → deep ViT
                        _raw_m._trunk_features_for_cls = _pooled  # tissue cls (detached downstream)
                        _raw_m._trunk_features_spatial = _f_deep
                    if _raw_m._trunk_features_for_cls is not None and \
                       not hasattr(_raw_m, '_trunk_diag_done'):
                        _raw_m._trunk_diag_done = True
                        print(f"[TrunkHook] organ=deep {_raw_m._trunk_features_for_organ.shape}, "
                              f"cls=deep {_raw_m._trunk_features_for_cls.shape}, "
                              f"spatial={_raw_m._trunk_features_spatial.shape}, "
                              f"fusion={'on' if _fusion_active else 'off'}")
                    return type(_o)(_outs) if isinstance(_o, tuple) else _outs
                return None
            _trunk_hook_handle = _trunk.register_forward_hook(_capture_trunk)
            print(f"[Stage {stage}] Trunk hook registered on {type(_trunk).__name__}")

    # ── Register hook on decoder's semantic_seg_head to capture pixel_embed
    # for intra-image contrastive (Stage 3).  pixel_embed is the decoder's
    # spatial feature map just before mask prediction — already in a
    # segmentation-aware space, so contrastive gradients are aligned with seg.
    _seg_head = getattr(_raw_m, "segmentation_head", None)
    if _seg_head is not None:
        _sem_head = getattr(_seg_head, "semantic_seg_head", None)
        if _sem_head is not None:
            _raw_m._decoder_features_spatial = None
            def _capture_pixel_embed(_m, _i):
                # _i[0] = pixel_embed [B, 256, Hf, Wf]
                if isinstance(_i, tuple) and len(_i) > 0:
                    _raw_m._decoder_features_spatial = _i[0]
            _decoder_feat_hook_handle = _sem_head.register_forward_pre_hook(
                _capture_pixel_embed)
            print(f"[Stage 3] Decoder feature hook registered on semantic_seg_head")

    for ep in range(start_epoch, epochs):
        if total_steps is not None and global_step >= total_steps:
            print(f"[total_steps] reached {global_step}/{total_steps}; stopping.")
            break
        # Keep the frozen model body in eval mode so self.training=False inside
        # SAM3, which prevents _compute_matching() from being called (matcher=None
        # because we used eval_mode=True). We compute our own binary loss externally.
        # _eval_with_checkpointing also re-enables the ViT trunk's train flag so
        # vitdet.py's activation checkpointing fires, halving activation memory.
        _eval_with_checkpointing(model)
        # Explicitly set the tunable submodules to train mode so dropout/BN work.
        prompt_enc.train()
        if lora_cfg is not None:
            for name, mod in model.named_modules():
                if "lora" in name.lower():
                    mod.train()
        if adapter:
            adapter.train()
        if tile_classifier:
            tile_classifier.train()
        if ssl_head:
            ssl_head.train()
        if pixel_ssl_head:
            pixel_ssl_head.train()
        if density_fusion:
            density_fusion.train()
        if contrast_proj:
            contrast_proj.train()

        total_loss_ep  = 0.0
        n_steps        = 0
        opt.zero_grad()
        _grads_in_window = 0   # backward calls since last optimizer step

        # Curriculum phase switch: graduate from warmup to full dataset.
        if curriculum and ep >= curriculum_warmup_epochs and _curr_phase == "warmup":
            print(f"\n[Curriculum] Epoch {ep}: graduating to full dataset "
                  f"({len(_warmup_items)} -> {len(train_items)} items)")
            _curr_phase   = "full"
            train_loader  = _make_train_loader(train_items, label="full")
            # Reload the warmup-phase snapshot so the full-dataset phase
            # starts from the best weights seen during curriculum warmup.
            _warmup_best_path = save_dir / "warmup_best.pth"
            if _warmup_best_path.exists():
                print(f"[Curriculum] Reloading warmup_best.pth → starting full phase from best warmup weights")
                _wrc = torch.load(_warmup_best_path, map_location=device, weights_only=False)
                _raw_m = model.module if hasattr(model, "module") else model
                if "prompt_enc" in _wrc:
                    prompt_enc.load_state_dict(
                        {k: v.to(device) for k, v in _wrc["prompt_enc"].items()})
                if lora_cfg is not None and _wrc.get("lora_state"):
                    _sd = _raw_m.state_dict()
                    _sd.update({k: v.to(device) for k, v in _wrc["lora_state"].items()})
                    _raw_m.load_state_dict(_sd)
                if adapter is not None and _wrc.get("adapter_state"):
                    adapter.load_state_dict(
                        {k: v.to(device) for k, v in _wrc["adapter_state"].items()})
                best_dice = _wrc.get("best_dice", best_dice)
                del _wrc

        # Recompute effective epoch size each epoch based on current phase.
        _active_items   = _warmup_items if _curr_phase == "warmup" else train_items
        _eff_epoch_size = (len(_active_items) if steps_per_epoch is None
                           else min(steps_per_epoch, len(_active_items)))

        # ── Hard example mining: rebuild loader with updated per-class weights ──
        if _use_hard_mining and ep >= hard_mining_start_epoch and _per_class_dice:
            _total_combos = sum(len(ds_dict) for ds_dict in _per_class_dice.values())
            print(f"\n[HardMining] Epoch {ep}: rebuilding sampler with per-class-per-dataset Dice weights "
                  f"({_total_combos} class×dataset combos tracked)")
            for _ck, _ds_dict in sorted(_per_class_dice.items()):
                for _ds, _d in sorted(_ds_dict.items(), key=lambda x: x[1]):
                    _w = 1.0 / max(_d + 0.1, 0.05)
                    print(f"    {_ck:8s} [{_ds:20s}] Dice={_d:.4f} → hm_weight={_w:.2f}")
            train_loader = _make_train_loader(
                _warmup_items if _curr_phase == "warmup" else train_items,
                label="hard-mining", hm_dice=_per_class_dice)

        pbar = tqdm(train_loader,
                   desc=f"Stage {stage} Ep {ep}/{epochs-1}",
                   total=len(train_loader))
        _cls_iter = iter(_cls_loader) if _cls_loader else None
        for step, batch_list in enumerate(pbar):
            if steps_per_epoch is not None and n_steps >= steps_per_epoch:
                break
            if total_steps is not None and global_step >= total_steps:
                break
            # Interleave cls-only samples every `cls_only_stride` steps.
            _stride = max(1, int(cls_only_stride))
            if _cls_iter is not None and step % _stride == 0:
                try:
                    sample = next(_cls_iter)[0]
                except StopIteration:
                    _cls_iter = iter(_cls_loader)
                    try:
                        sample = next(_cls_iter)[0]
                    except StopIteration:
                        sample = batch_list[0]
            else:
                sample = batch_list[0]

            loss = train_one_sample(
                model, transform, sample, device,
                prompt_enc=prompt_enc,
                adapter=adapter,
                histoplus=histoplus,
                scaler=scaler,
                ortho_weight=ORTHO_WEIGHT if stage >= 2 else 0.0,
                contrastive_weight=CONTRASTIVE_WEIGHT if stage >= 2 else 0.0,
                contrastive_margin=CONTRASTIVE_MARGIN,
                aux_weight=(density_consistency_weight if stage >= 2 else 0.0),
                focal_gamma=focal_gamma,
                seg_mode=seg_mode,
                tile_classifier=tile_classifier,
                ssl_head=ssl_head,
                pixel_ssl_head=pixel_ssl_head,
                intra_image_contrast_weight=intra_image_contrast_weight,
                cls_weight=cls_weight,
                ssl_weight=ssl_weight,
                repulsion_weight=repulsion_weight,
                negative_boost=negative_boost,
                area_loss_weight=area_loss_weight,
                structure_class_weight=(
                    structure_class_weight if stage >= 2 else 0.0),
                dice_weight=dice_weight,
                bce_weight=bce_weight,
                density_fusion=density_fusion,
                contrast_proj=contrast_proj,
                bio_boundary_weight=bio_boundary_weight,
                bio_compat_weight=bio_compat_weight,
                bio_gland_lumen_weight=bio_gland_lumen_weight,
                bg_suppression_weight=bg_suppression_weight,
                scale_reg_weight=scale_reg_weight,
                shape_reg_weight=shape_reg_weight,
                cls_only_mil_weight=cls_only_mil_weight,
                cls_only_absent_weight=cls_only_absent_weight,
                cls_only_topk_ratio=cls_only_topk_ratio,
                cls_only_entropy_weight=cls_only_entropy_weight,
                organ_warmup_steps=organ_warmup_steps,
                morph_projector=morph_projector,
                morph_graph_weight=morph_graph_weight,
                morph_transition_weight=morph_transition_weight,
                morph_relation_mode=morph_relation_mode,
                morph_relation_grid_size=morph_relation_grid_size,
                morph_keep_n_cells=morph_keep_n_cells,
                morph_warmup_steps=morph_warmup_steps,
                morph_ramp_steps=morph_ramp_steps,
                global_step=global_step,
                vis_dir=vis_dir,
            )
            if loss is None:
                continue

            scaler.scale(loss).backward()
            _grads_in_window += 1
            if (morph_projector is not None
                    and not getattr(morph_projector, "_grad_verified", False)
                    and getattr(morph_projector, "_debug_calls", 0) > 0):
                _morph_grad = sum(
                    float(param.grad.detach().abs().sum())
                    for param in morph_projector.parameters()
                    if param.grad is not None)
                if _morph_grad > 0:
                    morph_projector._grad_verified = True
                    print(f"[MorphGraphGradient] projector_grad_sum={_morph_grad:.6e}")

            if (step + 1) % batch_size == 0:
                if _grads_in_window > 0:
                    scaler.unscale_(opt)
                    # Clip seg and cls params independently — classification
                    # heads have much larger gradients and would otherwise
                    # throttle segmentation through global norm clipping.
                    if _seg_params:
                        torch.nn.utils.clip_grad_norm_(_seg_params, 1.0)
                    if _cls_params:
                        torch.nn.utils.clip_grad_norm_(_cls_params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                    scheduler.step()
                    opt_step += 1
                opt.zero_grad()
                _grads_in_window = 0

            total_loss_ep += loss.detach().item()
            n_steps += 1
            global_step += 1
            recent_losses.append(loss.detach().item())
            if len(recent_losses) > 200:
                recent_losses.pop(0)
            pbar.set_postfix(loss=f"{np.mean(recent_losses):.4f}",
                             gs=(f"{global_step}/{total_steps}"
                                 if total_steps is not None else global_step),
                             lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # ── Periodic validation ──────────────────────────────────────────
            if val_steps > 0 and global_step % val_steps == 0:
                recent_loss = float(np.mean(recent_losses))
                _cond_summaries = {}
                print(f"\nStep {global_step:>7d} | Loss {recent_loss:.4f}")
                for _oc, _oc_tag in _eval_modes:
                    _vs, _pcls, _cls_rows = evaluate(
                        model, transform, quick_val_ds, device,
                        vis_dir=os.path.join(vis_dir, "val"), epoch=global_step,
                        vis_num=max(1, len(quick_val_ds)),
                        tag=f"step_{global_step:07d}_{_oc_tag}",
                        organ_cond=_oc, seg_mode=seg_mode,
                        tile_classifier=tile_classifier,
                              prompt_enc=prompt_enc, adapter=adapter, histoplus=histoplus, density_fusion=density_fusion,
                        vis_mode="best_per_dataset",
                        cls_only_seg_vis=False)
                    _cond_summaries[_oc_tag] = _vs
                    print(f"  [{_oc_tag}] TissueDice {_vs['mean_tissue_dice']:.4f} | "
                          f"NucleiDice {_vs['mean_nuclei_dice']:.4f}"
                          + (f" | OrganAcc {_vs['tissue_type_acc']:.4f}"
                             if "tissue_type_acc" in _vs else "")
                          + (f" | TClsAcc {_vs['tissue_cls_hamming']:.4f}"
                             if "tissue_cls_hamming" in _vs else ""))
                    if _pcls:
                        for _r in _pcls:
                            print(f"   [{_oc_tag}] {_r['dataset']:20s}  "
                                  f"{_r['class_key']:35s}  "
                                  f"Dice={_r['dice']:.4f}  IoU={_r['iou']:.4f}")
                        # ── Update per-class-per-dataset Dice for hard example mining ─
                        if _oc_tag == "cond" and _use_hard_mining:
                            for _r in _pcls:
                                _per_class_dice.setdefault(_r["class_key"], {})[_r["dataset"]] = _r["dice"]
                    # ── Per-dataset metrics ──
                    if _cls_rows:
                        for _r in _cls_rows:
                            _parts = [f"  [{_oc_tag}] {_r['dataset']:20s}"]
                            if "tissue_dice" in _r:
                                _parts.append(f"TisDice={_r['tissue_dice']:.4f}")
                            if "nuclei_dice" in _r:
                                _parts.append(f"NucDice={_r['nuclei_dice']:.4f}")
                            if "organ_acc" in _r:
                                _parts.append(f"OrganAcc={_r['organ_acc']:.4f}")
                            if "tissue_hamming" in _r:
                                _parts.append(f"TClsAcc={_r['tissue_hamming']:.4f}")
                            print("   " + "  ".join(_parts))
                    _append_val_rows(_vs, _pcls, _oc_tag,
                                     gstep=global_step, ep=ep, loss=recent_loss,
                                     cls_rows=_cls_rows)
                    if tb_writer:
                        tb_writer.add_scalar(f"val_{_oc_tag}/mean_dice",
                                             _vs["mean_dice"],        global_step)
                        tb_writer.add_scalar(f"val_{_oc_tag}/tissue_dice",
                                             _vs["mean_tissue_dice"], global_step)
                        tb_writer.add_scalar(f"val_{_oc_tag}/nuclei_dice",
                                             _vs["mean_nuclei_dice"], global_step)
                        if "tissue_type_acc" in _vs:
                            tb_writer.add_scalar(f"val_{_oc_tag}/tissue_type_acc",
                                                 _vs["tissue_type_acc"], global_step)
                        if "tissue_cls_hamming" in _vs:
                            tb_writer.add_scalar(f"val_{_oc_tag}/tissue_cls_hamming",
                                                 _vs["tissue_cls_hamming"], global_step)
                if tb_writer:
                    tb_writer.add_scalar("train/loss", recent_loss, global_step)
                    tb_writer.add_scalar("train/lr",   scheduler.get_last_lr()[0], global_step)
                # Use the BEST of cond / nocond modes for best-ckpt decision
                # (rather than the mean) so improvement in either head triggers a save.
                mean_dice = float(np.max([
                    _cond_summaries[k]["mean_dice"] for k in _cond_summaries]))
                _is_best = (mean_dice >= best_dice)
                if _is_best:
                    best_dice = mean_dice
                    _save_ckpt(save_dir / "best_model.pth",
                               model, prompt_enc, adapter,
                               epoch=ep, best_dice=best_dice,
                               best_holdout_dice=best_holdout_dice,
                               stage=stage, lora_cfg=lora_cfg,
                               opt=opt, scheduler=scheduler, scaler=scaler,
                               global_step=global_step, opt_step=opt_step,
                               recent_losses=recent_losses,
                               train_cfg=train_cfg,
                               tile_classifier=tile_classifier,
                               ssl_head=ssl_head,
                               pixel_ssl_head=pixel_ssl_head,
                               density_fusion=density_fusion,
                               density_gate_net=density_gate_net,
                               contrast_proj=contrast_proj)
                    print(f"  [Best] Saved: {save_dir / 'best_model.pth'}  "
                          f"(mean_dice={best_dice:.4f}, step={global_step}, ep={ep})")
                    if lora_cfg is not None:
                        save_lora_weights(model, str(save_dir / "lora_weights_best.pt"))
                    _save_ckpt(save_dir / "last_checkpoint.pth",
                               model, prompt_enc, adapter,
                               epoch=ep, best_dice=best_dice,
                               best_holdout_dice=best_holdout_dice,
                               stage=stage, lora_cfg=lora_cfg,
                               opt=opt, scheduler=scheduler, scaler=scaler,
                               global_step=global_step, opt_step=opt_step,
                               recent_losses=recent_losses,
                               train_cfg=train_cfg,
                               tile_classifier=tile_classifier,
                               ssl_head=ssl_head,
                               pixel_ssl_head=pixel_ssl_head,
                               density_fusion=density_fusion,
                               density_gate_net=density_gate_net,
                               contrast_proj=contrast_proj)
                    if lora_cfg is not None:
                        save_lora_weights(model, str(save_dir / "lora_weights_last.pt"))
                    print(f"  Best ckpt saved (dice={best_dice:.4f})  → also wrote last_checkpoint.pth")
                    if save_best_vis:
                        print("  Saving best-checkpoint visualisations")
                        for _oc, _oc_tag in _eval_modes:
                            evaluate(model, transform, quick_val_ds, device,
                                     prompt_enc=prompt_enc,
                                     vis_dir=os.path.join(vis_dir, "val"),
                                     epoch=global_step,
                                     vis_num=min(vis_num, len(quick_val_items)),
                                     tag=f"step_{global_step:07d}_{_oc_tag}",
                                     organ_cond=_oc, seg_mode=seg_mode,
                                     adapter=adapter, histoplus=histoplus,
                                     tile_classifier=tile_classifier,
                                     density_fusion=density_fusion,
                                     cls_only_seg_vis=False)
                # Restore train mode after validation
                _eval_with_checkpointing(model)
                prompt_enc.train()
                if lora_cfg is not None:
                    for _name, _mod in model.named_modules():
                        if "lora" in _name.lower():
                            _mod.train()
                if adapter:
                    adapter.train()
                if tile_classifier:
                    tile_classifier.train()
                if ssl_head:
                    ssl_head.train()
                if pixel_ssl_head:
                    pixel_ssl_head.train()

                # ── Hold-out val (separate from model-selection) ──────────
                _hold_mean = _eval_and_log_holdout(
                    quick_val_val_ds,
                    log_prefix=f"[HoldVal] Step {global_step:>7d}",
                    gstep=global_step, ep=ep,
                    tag=f"step_{global_step:07d}",
                    save_vis=True)
                if _hold_mean is not None and _hold_mean >= best_holdout_dice:
                    best_holdout_dice = _hold_mean
                    _save_ckpt(save_dir / "best_holdout_model.pth",
                               model, prompt_enc, adapter,
                               epoch=ep, best_dice=best_dice,
                               best_holdout_dice=best_holdout_dice,
                               stage=stage, lora_cfg=lora_cfg,
                               opt=opt, scheduler=scheduler, scaler=scaler,
                               global_step=global_step, opt_step=opt_step,
                               recent_losses=recent_losses,
                               train_cfg=train_cfg,
                               tile_classifier=tile_classifier,
                               ssl_head=ssl_head,
                               pixel_ssl_head=pixel_ssl_head,
                               density_fusion=density_fusion,
                               density_gate_net=density_gate_net,
                               contrast_proj=contrast_proj)
                    print(f"  [BestHoldout] Saved: {save_dir / 'best_holdout_model.pth'}  "
                          f"(holdout_mean_dice={best_holdout_dice:.4f}, step={global_step}, ep={ep})")
                    if lora_cfg is not None:
                        save_lora_weights(model, str(save_dir / "lora_weights_best_holdout.pt"))

        # Flush remaining grads
        if n_steps % batch_size != 0 and _grads_in_window > 0:
            scaler.unscale_(opt)
            # Clip seg and cls params independently (see comment above).
            if _seg_params:
                torch.nn.utils.clip_grad_norm_(_seg_params, 1.0)
            if _cls_params:
                torch.nn.utils.clip_grad_norm_(_cls_params, 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            opt_step += 1
            opt.zero_grad()

        mean_train_loss = total_loss_ep / max(n_steps, 1)
        print(f"Stage {stage} | Ep {ep:3d} complete | GlobalStep {global_step:,} | "
              f"AvgLoss {mean_train_loss:.4f} | BestDice {best_dice:.4f}")

        # End-of-epoch validation (only when val_steps == 0, i.e. no intra-epoch val)
        if val_steps == 0:
            _cond_summaries = {}
            print(f"\nEpoch {ep} end-of-epoch eval (val_steps=0)")
            for _oc, _oc_tag in _eval_modes:
                _vs, _pcls, _cls_rows = evaluate(
                    model, transform, quick_val_ds, device,
                    vis_dir=os.path.join(vis_dir, "val"), epoch=global_step,
                    vis_num=max(1, len(quick_val_ds)),
                    tag=f"ep{ep:03d}_{_oc_tag}",
                    organ_cond=_oc,
                    tile_classifier=tile_classifier,
                              prompt_enc=prompt_enc, adapter=adapter, histoplus=histoplus, density_fusion=density_fusion,
                    vis_mode="best_per_dataset",
                    cls_only_seg_vis=False)
                _cond_summaries[_oc_tag] = _vs
                print(f"  [{_oc_tag}] TissueDice {_vs['mean_tissue_dice']:.4f} | "
                      f"NucleiDice {_vs['mean_nuclei_dice']:.4f}"
                      + (f" | OrganAcc {_vs['tissue_type_acc']:.4f}"
                         if "tissue_type_acc" in _vs else "")
                      + (f" | TClsAcc {_vs['tissue_cls_hamming']:.4f}"
                         if "tissue_cls_hamming" in _vs else ""))
                if _pcls:
                    for _r in _pcls:
                        print(f"   [{_oc_tag}] {_r['dataset']:20s}  "
                              f"{_r['class_key']:35s}  "
                              f"Dice={_r['dice']:.4f}  IoU={_r['iou']:.4f}")
                    # ── Update per-class-per-dataset Dice for hard example mining ─
                    if _oc_tag == "cond" and _use_hard_mining:
                        for _r in _pcls:
                            _per_class_dice.setdefault(_r["class_key"], {})[_r["dataset"]] = _r["dice"]
                # ── Per-dataset metrics ──
                if _cls_rows:
                    for _r in _cls_rows:
                        _parts = [f"  [{_oc_tag}] {_r['dataset']:20s}"]
                        if "tissue_dice" in _r:
                            _parts.append(f"TisDice={_r['tissue_dice']:.4f}")
                        if "nuclei_dice" in _r:
                            _parts.append(f"NucDice={_r['nuclei_dice']:.4f}")
                        if "organ_acc" in _r:
                            _parts.append(f"OrganAcc={_r['organ_acc']:.4f}")
                        if "tissue_hamming" in _r:
                            _parts.append(f"TClsAcc={_r['tissue_hamming']:.4f}")
                        print("   " + "  ".join(_parts))
                _append_val_rows(_vs, _pcls, _oc_tag,
                                 gstep=global_step, ep=ep, loss=mean_train_loss,
                                 cls_rows=_cls_rows)
                if tb_writer:
                    tb_writer.add_scalar(f"val_{_oc_tag}/mean_dice",
                                         _vs["mean_dice"],        global_step)
                    tb_writer.add_scalar(f"val_{_oc_tag}/tissue_dice",
                                         _vs["mean_tissue_dice"], global_step)
                    tb_writer.add_scalar(f"val_{_oc_tag}/nuclei_dice",
                                         _vs["mean_nuclei_dice"], global_step)
                    if "tissue_type_acc" in _vs:
                        tb_writer.add_scalar(f"val_{_oc_tag}/tissue_type_acc",
                                             _vs["tissue_type_acc"], global_step)
                    if "tissue_cls_hamming" in _vs:
                        tb_writer.add_scalar(f"val_{_oc_tag}/tissue_cls_hamming",
                                             _vs["tissue_cls_hamming"], global_step)
            if tb_writer:
                tb_writer.add_scalar("train/loss", mean_train_loss, global_step)
            # Use the BEST of cond / nocond modes for best-ckpt decision
            # (rather than the mean) so improvement in either head triggers a save.
            mean_dice = float(np.max([
                _cond_summaries[k]["mean_dice"] for k in _cond_summaries]))
            _is_best = (mean_dice >= best_dice)
            if _is_best:
                best_dice = mean_dice
                _save_ckpt(save_dir / "best_model.pth",
                           model, prompt_enc, adapter,
                           epoch=ep, best_dice=best_dice,
                           best_holdout_dice=best_holdout_dice,
                           stage=stage, lora_cfg=lora_cfg,
                           opt=opt, scheduler=scheduler, scaler=scaler,
                           global_step=global_step, opt_step=opt_step,
                           recent_losses=recent_losses,
                           train_cfg=train_cfg,
                           tile_classifier=tile_classifier,
                           ssl_head=ssl_head,
                           pixel_ssl_head=pixel_ssl_head,
                           density_fusion=density_fusion,
                           density_gate_net=density_gate_net,
                           contrast_proj=contrast_proj)
                print(f"  [Best] Saved: {save_dir / 'best_model.pth'}  "
                      f"(mean_dice={best_dice:.4f}, step={global_step}, ep={ep})")
                if lora_cfg is not None:
                    save_lora_weights(model, str(save_dir / "lora_weights_best.pt"))
                _save_ckpt(save_dir / "last_checkpoint.pth",
                           model, prompt_enc, adapter,
                           epoch=ep, best_dice=best_dice,
                           best_holdout_dice=best_holdout_dice,
                           stage=stage, lora_cfg=lora_cfg,
                           opt=opt, scheduler=scheduler, scaler=scaler,
                           global_step=global_step, opt_step=opt_step,
                           recent_losses=recent_losses,
                           train_cfg=train_cfg,
                           tile_classifier=tile_classifier,
                           ssl_head=ssl_head,
                           pixel_ssl_head=pixel_ssl_head,
                           density_fusion=density_fusion,
                           density_gate_net=density_gate_net,
                           contrast_proj=contrast_proj)
                if lora_cfg is not None:
                    save_lora_weights(model, str(save_dir / "lora_weights_last.pt"))
                print(f"  Best ckpt saved (dice={best_dice:.4f})  → also wrote last_checkpoint.pth")
                if save_best_vis:
                    print("  Saving best-checkpoint visualisations")
                    for _oc, _oc_tag in _eval_modes:
                        evaluate(model, transform, quick_val_ds, device,
                                 prompt_enc=prompt_enc,
                                 vis_dir=os.path.join(vis_dir, "val"),
                                 epoch=global_step,
                                 vis_num=min(vis_num, len(quick_val_items)),
                                 tag=f"ep{ep:03d}_{_oc_tag}",
                                 organ_cond=_oc, seg_mode=seg_mode,
                                 adapter=adapter, histoplus=histoplus,
                                 tile_classifier=tile_classifier,
                                 density_fusion=density_fusion,
                                 cls_only_seg_vis=False)
            # Restore train mode after validation
            _eval_with_checkpointing(model)
            prompt_enc.train()
            if lora_cfg is not None:
                for _name, _mod in model.named_modules():
                    if "lora" in _name.lower():
                        _mod.train()
            if adapter:
                adapter.train()
            if tile_classifier:
                tile_classifier.train()
            if ssl_head:
                ssl_head.train()
            if pixel_ssl_head:
                pixel_ssl_head.train()

            # ── Hold-out val (separate from model-selection) ──────────────
            _hold_mean = _eval_and_log_holdout(
                quick_val_val_ds,
                log_prefix=f"[HoldVal] Epoch {ep}",
                gstep=global_step, ep=ep,
                tag=f"ep{ep:03d}",
                save_vis=True)
            if _hold_mean is not None and _hold_mean >= best_holdout_dice:
                best_holdout_dice = _hold_mean
                _save_ckpt(save_dir / "best_holdout_model.pth",
                           model, prompt_enc, adapter,
                           epoch=ep, best_dice=best_dice,
                           best_holdout_dice=best_holdout_dice,
                           stage=stage, lora_cfg=lora_cfg,
                           opt=opt, scheduler=scheduler, scaler=scaler,
                           global_step=global_step, opt_step=opt_step,
                           recent_losses=recent_losses,
                           train_cfg=train_cfg,
                           tile_classifier=tile_classifier,
                           ssl_head=ssl_head,
                           pixel_ssl_head=pixel_ssl_head,
                           density_fusion=density_fusion,
                           density_gate_net=density_gate_net,
                           contrast_proj=contrast_proj)
                print(f"  [BestHoldout] Saved: {save_dir / 'best_holdout_model.pth'}  "
                      f"(holdout_mean_dice={best_holdout_dice:.4f}, step={global_step}, ep={ep})")
                if lora_cfg is not None:
                    save_lora_weights(model, str(save_dir / "lora_weights_best_holdout.pt"))

        # Always save a resume checkpoint at the end of every epoch so training
        # can be restarted cleanly from an epoch boundary after a server failure.
        _save_ckpt(save_dir / "last_checkpoint.pth",
                   model, prompt_enc, adapter,
                   epoch=ep, best_dice=best_dice,
                   best_holdout_dice=best_holdout_dice,
                   stage=stage, lora_cfg=lora_cfg,
                   opt=opt, scheduler=scheduler, scaler=scaler,
                   global_step=global_step, opt_step=opt_step,
                   recent_losses=recent_losses,
                   train_cfg=train_cfg,
                   tile_classifier=tile_classifier,
                   ssl_head=ssl_head,
                   pixel_ssl_head=pixel_ssl_head,
                   density_fusion=density_fusion,
                   density_gate_net=density_gate_net,
                   contrast_proj=contrast_proj)
        if lora_cfg is not None:
            save_lora_weights(model, str(save_dir / "lora_weights_last.pt"))
        print(f"  Resume ckpt saved → last_checkpoint.pth  (epoch {ep})")

        # At the end of the last warmup epoch: snapshot current weights so the
        # full-dataset phase reloads from the best warmup state.
        if curriculum and ep == curriculum_warmup_epochs - 1 and _curr_phase == "warmup":
            _warmup_best_path = save_dir / "warmup_best.pth"
            _save_ckpt(_warmup_best_path,
                       model, prompt_enc, adapter,
                       epoch=ep, best_dice=best_dice,
                       best_holdout_dice=best_holdout_dice,
                       stage=stage, lora_cfg=lora_cfg,
                       opt=opt, scheduler=scheduler, scaler=scaler,
                       global_step=global_step, opt_step=opt_step,
                       recent_losses=recent_losses,
                       train_cfg=train_cfg,
                       tile_classifier=tile_classifier,
                       ssl_head=ssl_head,
                       pixel_ssl_head=pixel_ssl_head,
                       density_fusion=density_fusion,
                       density_gate_net=density_gate_net,
                       contrast_proj=contrast_proj)
            print(f"  Warmup-phase snapshot saved → warmup_best.pth  (epoch {ep})")

    if tb_writer:
        tb_writer.close()
    if _trunk_hook_handle is not None:
        _trunk_hook_handle.remove()


def _save_ckpt(path: Path, model, prompt_enc, adapter,
               epoch, best_dice, stage, lora_cfg,
               opt=None, scheduler=None, scaler=None,
               global_step: int = 0, opt_step: int = 0,
               recent_losses=None,
               train_cfg: Optional[dict] = None,
               tile_classifier: Optional[nn.Module] = None,
               ssl_head:        Optional[nn.Module] = None,
               pixel_ssl_head:  Optional[nn.Module] = None,
               density_fusion:  Optional[nn.Module] = None,
               density_gate_net: Optional[nn.Module] = None,
               contrast_proj:   Optional[nn.Module] = None,
               morph_projector: Optional[nn.Module] = None,
               best_holdout_dice: float = 0.0):
    """Save a full checkpoint including all tunable state and training metadata for resumption."""
    lora_state = None
    if lora_cfg is not None:
        _raw_m = model.module if hasattr(model, "module") else model
        lora_state = {k: v.cpu() for k, v in _raw_m.state_dict().items()
                      if "lora" in k.lower() and k in _raw_m.state_dict()}
    adapter_state = adapter.state_dict() if adapter is not None else None
    cls_state = tile_classifier.state_dict() if tile_classifier is not None else None
    ssl_state = ssl_head.state_dict() if ssl_head is not None else None
    pixel_ssl_state = pixel_ssl_head.state_dict() if pixel_ssl_head is not None else None
    density_fusion_state = density_fusion.state_dict() if density_fusion is not None else None
    density_gate_state = density_gate_net.state_dict() if density_gate_net is not None else None
    contrast_proj_state = contrast_proj.state_dict() if contrast_proj is not None else None
    morph_proj_state = morph_projector.state_dict() if morph_projector is not None else None

    torch.save({
        "stage":           stage,
        "epoch":           epoch,
        "best_dice":       best_dice,
        "best_holdout_dice": best_holdout_dice,
        "prompt_enc":      {k: v.cpu() for k, v in prompt_enc.state_dict().items()},
        "lora_state":      lora_state,
        "adapter_state":   adapter_state,
        "tile_cls_state":  cls_state,
        "ssl_head_state":  ssl_state,
        "pixel_ssl_state": pixel_ssl_state,
        "density_fusion_state": density_fusion_state,
        "density_gate_state":   density_gate_state,
        "contrast_proj_state":  contrast_proj_state,
        "morph_proj_state":     morph_proj_state,
        # Training state for resumption
        "global_step":     global_step,
        "opt_step":        opt_step,
        "recent_losses":   list(recent_losses) if recent_losses else [],
        "opt_state":       opt.state_dict() if opt is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state":    scaler.state_dict() if scaler is not None else None,
        "train_cfg":       train_cfg,
    }, path)


def _load_ckpt_weights(
    ckpt_path: Path,
    model: nn.Module,
    prompt_enc: "PromptTunedTextEncoder",
    adapter: Optional[nn.Module],
    lora_cfg: Optional["LoRAConfig"],
    tile_classifier: Optional[nn.Module],
    ssl_head: Optional[nn.Module],
    pixel_ssl_head: Optional[nn.Module],
    density_fusion: Optional[nn.Module],
    density_gate_net: Optional[nn.Module],
    contrast_proj: Optional[nn.Module],
    device: torch.device,
    morph_projector: Optional[nn.Module] = None,
) -> dict:
    """Load all model weights from *ckpt_path*.  Returns the raw checkpoint dict."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    _m = model.module if hasattr(model, "module") else model
    _load_compat(_m.backbone.language_backbone,
                 ckpt["prompt_enc"], tag="ckpt_load", device=device)
    if lora_cfg is not None and ckpt.get("lora_state"):
        sd = _m.state_dict()
        sd.update({k: v.to(device) for k, v in ckpt["lora_state"].items()})
        _m.load_state_dict(sd)
    if adapter is not None and ckpt.get("adapter_state"):
        _load_compat(adapter, ckpt["adapter_state"], tag="ckpt_adapter", device=device)
    if tile_classifier is not None and ckpt.get("tile_cls_state"):
        _load_compat(tile_classifier, ckpt["tile_cls_state"], tag="ckpt_tilecls", device=device)
    if ssl_head is not None and ckpt.get("ssl_head_state"):
        _load_compat(ssl_head, ckpt["ssl_head_state"], tag="ckpt_ssl", device=device)
    if pixel_ssl_head is not None and ckpt.get("pixel_ssl_state"):
        _load_compat(pixel_ssl_head, ckpt["pixel_ssl_state"], tag="ckpt_pixel_ssl", device=device)
    if density_fusion is not None and ckpt.get("density_fusion_state"):
        _load_compat(density_fusion, ckpt["density_fusion_state"],
                     tag="ckpt_density_fusion", device=device)
    if contrast_proj is not None and ckpt.get("contrast_proj_state"):
        _load_compat(contrast_proj, ckpt["contrast_proj_state"],
                     tag="ckpt_contrast_proj", device=device)
    if morph_projector is not None and ckpt.get("morph_proj_state"):
        _load_compat(morph_projector, ckpt["morph_proj_state"],
                     tag="ckpt_morph_proj", device=device)
    if density_gate_net is not None and ckpt.get("density_gate_state"):
        try:
            density_gate_net.load_state_dict(ckpt["density_gate_state"])
        except Exception as _e:
            print(f"  [density_gate] WARNING: {_e}")
    _m._density_gate_net = density_gate_net
    _m._density_fusion_mod = density_fusion
    _m._pending_density = None
    _m._nuclei_density_logit_prior_scale = float(
        (ckpt.get("train_cfg") or {}).get("nuclei_density_logit_prior", 0.0))
    return ckpt


# ── Datasets that have pixel-level seg masks and are used in seg training ────
# SegPath excluded (cell-level IHC, not region-level tissue).
_SEG_TISSUE_DS: frozenset = frozenset({
    "BCSS", "GlaS", "DigestPath2019", "WSSS4LUAD", "ICIAR2018_WSI",
})
_SEG_NUCLEI_DS: frozenset = frozenset({
    "NuCLS", "Kumar", "CPM15", "CPM17", "PanNuke", "CoNIC2022",
    "Lizard", "MoNuSAC", "Janowczyk", "TNBC",
})
# All datasets with any declared classes — used for cls eval
_ALL_TRAINED_DS: frozenset = frozenset(_DS_CLASSES.keys())


def _csv_arg_to_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    out = [x.strip() for x in str(value).split(",") if x.strip()]
    return out or None


def _csv_arg_to_int_map(value: Optional[str]) -> Optional[Dict[str, int]]:
    if value is None or not str(value).strip():
        return None
    out: Dict[str, int] = {}
    for chunk in str(value).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(
                "--external_eval_max_samples_by_dataset expects Dataset=N pairs, "
                f"got {chunk!r}"
            )
        name, raw_count = chunk.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty dataset name in {chunk!r}")
        count = int(raw_count.strip())
        if count < 0:
            raise ValueError(f"Negative sample count for {name}: {count}")
        out[name] = count
    return out or None


def _build_external_eval_items_from_args(args) -> Tuple[List[_Item], set]:
    """Build external validation items in the same wrapper format as train/val."""
    if build_external_eval_datasets is None:
        raise RuntimeError(
            "External eval loader could not be imported: "
            f"{_EXTERNAL_EVAL_IMPORT_ERROR!r}"
        )

    ds_map = build_external_eval_datasets(
        root_dir=args.external_eval_root,
        names=_csv_arg_to_list(args.external_eval_datasets),
        max_samples_per_dataset=args.external_eval_max_samples,
        max_samples_by_dataset=_csv_arg_to_int_map(
            args.external_eval_max_samples_by_dataset),
        seed=args.seed,
        subset_strategy=args.external_eval_subset_strategy,
        subset_cache_dir=args.external_eval_subset_cache_dir,
        stratify_pool_multiplier=args.external_eval_stratify_pool_multiplier,
        acdc_split=args.external_acdc_split,
        paip_split=args.external_paip_split,
        acdc_magnification=args.external_acdc_magnification,
        paip_magnification=args.external_paip_magnification,
    )

    items: List[_Item] = []
    seg_dataset_names = set()
    for fallback_name, ds in ds_map.items():
        ds_name = str(getattr(ds, "dataset_name", fallback_name))
        seg_dataset_names.add(ds_name)
        items.extend(_Item(ds, i, ds_name) for i in range(len(ds)))
        print(
            f"  [external_eval] {ds_name}: {len(ds)} items, "
            f"mag={float(getattr(ds, 'magnification', 0.0)):.1f}x, "
            f"tissue_type={int(getattr(ds, 'tissue_type', 0))}"
        )

    if not items:
        print("WARNING: no external eval items were found.")
    return items, seg_dataset_names


def run_final_test(
    best_ckpt_path: Optional[Path],
    model:        nn.Module,
    prompt_enc:   "PromptTunedTextEncoder",
    test_items:   List[_Item],
    save_dir:     Path,
    device:       torch.device,
    adapter:      Optional[nn.Module] = None,
    lora_cfg:     Optional["LoRAConfig"]        = None,
    seed:         int                           = SEED,
    stage:        int                           = 1,
    seg_mode:     str                           = "both",
    native_crop_at_40x: int                     = NATIVE_CROP_AT_40X,
    tile_classifier: Optional[nn.Module]       = None,
    ssl_head:        Optional[nn.Module]       = None,
    histoplus:       Optional["InstanSegNucleiExtractor"] = None,
    density_fusion:  Optional[nn.Module]       = None,
    density_gate_net: Optional[nn.Module]      = None,
    contrast_proj:   Optional[nn.Module]       = None,
    sw_threshold:    int                        = 2048,
    test_eval:       str                        = "both",  # "seg" | "cls" | "both"
    extra_seg_dataset_names: Optional[set]      = None,
    vis_num:         int                        = 8,
    magnification_alignment: bool               = True,
) -> None:
    """Final holdout evaluation with sliding-window inference for large images.

    Seg metrics  — only on pixel-supervised datasets
                   (``_SEG_TISSUE_DS`` / ``_SEG_NUCLEI_DS``).
    Cls metrics  — organ accuracy + tissue-class Hamming on ALL trained
                   datasets (including tile-cls-only sets).

    Outputs per cond-mode (cond / nocond)
    ─────────────────────────────────────
    test_seg_per_image_{tag}.csv
        One row per (dataset, sample, class_key) — raw per-image seg metrics.
    test_seg_per_class_{tag}.csv
        Avg Dice/IoU/Prec/Rec per class_key (averaged across all images & datasets).
    test_seg_per_dataset_{tag}.csv
        Avg Dice per dataset (averaged across classes).
    test_seg_summary_{tag}.csv
        Global mean Dice / tissue Dice / nuclei Dice.
    test_cls_per_dataset_{tag}.csv
        Organ accuracy + tissue-cls Hamming per dataset.
    test_cls_summary_{tag}.csv
        Mean organ accuracy + mean Hamming across all cls-capable datasets.
    vis/test/{tag}/
        One PNG per sample (best-per-dataset).
    """
    print("\n" + "=" * 60)
    print(f"  Final Test  ({len(test_items)} holdout items)")
    ckpt_name = best_ckpt_path.name if best_ckpt_path is not None else "bare_medsam3_no_checkpoint"
    print(f"  Ckpt: {ckpt_name}")
    print("=" * 60)

    if best_ckpt_path is not None and not best_ckpt_path.exists():
        print(f"  WARNING: checkpoint not found at {best_ckpt_path}. Skipping test.")
        return

    if best_ckpt_path is None:
        ckpt = {"stage": stage, "epoch": 0, "best_dice": 0.0}
        print("  Using original MedSAM3 weights without PathXAI checkpoint/LoRA.")
    else:
        ckpt = _load_ckpt_weights(best_ckpt_path, model, prompt_enc, adapter,
                                   lora_cfg, tile_classifier, ssl_head,
                                   None, density_fusion,
                                   density_gate_net, contrast_proj, device)
    print(f"  Stage={ckpt.get('stage')}  epoch={ckpt.get('epoch')}  "
          f"best_val_dice={ckpt.get('best_dice', 0.0):.4f}")

    transform = _build_transform(SAM3_RESOLUTION)
    _eval_with_checkpointing(model)
    _test_trunk_hook = _register_trunk_feature_hook(model, stage_label=f"test_stage{stage}")
    prompt_enc.eval()
    if adapter:     adapter.eval()
    if tile_classifier: tile_classifier.eval()
    _raw_m = model.module if hasattr(model, "module") else model
    _gate_net  = getattr(_raw_m, "_density_gate_net", None)
    _ndp_scale = float(getattr(_raw_m, "_nuclei_density_logit_prior_scale", 0.0))

    # ── Determine which items to use for seg vs cls ──────────────────────
    _seg_ds_set = set()
    if seg_mode in ("tissue", "both"):
        _seg_ds_set |= _SEG_TISSUE_DS
    if seg_mode in ("nuclei", "both"):
        _seg_ds_set |= _SEG_NUCLEI_DS
    if extra_seg_dataset_names:
        _seg_ds_set |= set(extra_seg_dataset_names)

    seg_items = [it for it in test_items if it.dataset_name in _seg_ds_set]
    cls_items = [it for it in test_items if it.dataset_name in _ALL_TRAINED_DS]
    print(f"  Seg items : {len(seg_items)}  from {sorted({it.dataset_name for it in seg_items})}")
    print(f"  Cls items : {len(cls_items)}  from {sorted({it.dataset_name for it in cls_items})}")

    save_dir.mkdir(parents=True, exist_ok=True)
    vis_base = save_dir / "vis" / "test"

    _do_seg = test_eval in ("seg", "both")
    _do_cls = test_eval in ("cls", "both") and tile_classifier is not None

    _penc_for_modes = _raw_m.backbone.language_backbone
    _use_organ_bias_eval = bool(getattr(_penc_for_modes, "use_organ_bias", False))
    _eval_modes = ([(False, "nocond"), (True, "cond")]
                   if _use_organ_bias_eval else [(False, "nocond")])

    for _oc, _oc_tag in _eval_modes:
        print(f"\n{'─'*50}\n  [{_oc_tag.upper()}]\n{'─'*50}")
        _penc = _raw_m.backbone.language_backbone
        if hasattr(_penc, "current_tissue_type"):
            _penc.current_tissue_type = 0

        # ══════════════════════════════════════════════════════════════════
        # 1.  SEG EVALUATION  (pixel-supervised datasets, sliding window)
        #     Uses SAM3CombinedDataset with crop_to_target=False so the
        #     image is at full native resolution (capped at max_sw_size).
        # ══════════════════════════════════════════════════════════════════
        seg_rows: List[dict] = []   # one row per (dataset, sample, class_key)

        _vis_dir = str(vis_base / _oc_tag)
        os.makedirs(_vis_dir, exist_ok=True)
        _saved_vis = 0
        _saved_vis_by_ds: Dict[str, int] = {}
        # vis_num <= 0 → save ALL vis (unlimited).
        # vis_num > 0  → save only the first N samples (keeps large external tests compact).
        _vis_limit = None if int(vis_num) <= 0 else int(vis_num)
        _vis_limit_by_ds: Optional[int] = None
        if _vis_limit is not None and extra_seg_dataset_names and seg_items:
            _vis_ds_count = max(1, len({it.dataset_name for it in seg_items}))
            _vis_limit_by_ds = max(1, int(math.ceil(_vis_limit / _vis_ds_count)))

        if _do_seg and seg_items:
            # Large max_eval_size → nearly native resolution for SW inference.
            # SAM3CombinedDataset handles magnification normalisation.
            _seg_ds = SAM3CombinedDataset(
                seg_items, augment=False, crop_to_target=False,
                max_eval_size=sw_threshold * 4,
                native_crop_at_40x=native_crop_at_40x,
                normalize_magnification=magnification_alignment)

            for _sidx in tqdm(range(len(_seg_ds)), desc=f"Seg [{_oc_tag}]", leave=False):
                sample  = _seg_ds[_sidx]
                _source_item = _seg_ds.items[_sidx]
                _source_dataset = _source_item.dataset
                _source_local_idx = int(_source_item.local_idx)
                _source_image_paths = getattr(_source_dataset, "image_paths", [])
                _source_mask_paths = getattr(_source_dataset, "mask_paths", [])
                _source_image_path = (
                    str(_source_image_paths[_source_local_idx])
                    if _source_local_idx < len(_source_image_paths) else ""
                )
                _source_mask_path = (
                    str(_source_mask_paths[_source_local_idx])
                    if _source_local_idx < len(_source_mask_paths) else ""
                )
                pil_im  = sample["image"]
                t_mask  = sample["tissue_mask"]
                n_mask  = sample["nuclei_mask"]
                ds_name = sample["dataset_name"]
                mag     = float(sample.get("magnification", 40.0))
                H, W    = pil_im.height, pil_im.width

                if seg_mode == "tissue":   n_mask = None
                elif seg_mode == "nuclei": t_mask = None

                if hasattr(_penc, "current_tissue_type"):
                    _penc.current_tissue_type = (
                        int(sample.get("tissue_type", 0)) if _oc else 0)

                # Density map for Stage 2/3 density fusion and optional nuclei prior.
                _density_map_cpu = None
                if histoplus is not None and (density_fusion is not None or (_ndp_scale > 0.0 and _gate_net is not None)):
                    try:
                        _key = f"{ds_name}:{sample.get('local_idx', _sidx)}"
                        _density_map_cpu = histoplus.extract(pil_im, cache_key=_key)[0]
                    except Exception:
                        pass

                # Patch size matched to training FOV scaled by effective mag.
                sw_patch = max(64, int(native_crop_at_40x * mag / 40.0))
                sw_patch = max(sw_patch, SAM3_RESOLUTION)
                sw_patch = min(sw_patch, max(H, W))
                sw_stride = max(32, sw_patch // 2)
                # Present classes — include background (0) so argmax has a real
                # probability to compete with foreground, matching training val.
                _t_present = (frozenset(int(c) for c in np.unique(t_mask)
                                        if 0 <= c < NUM_TISSUE_CLASSES)
                              if t_mask is not None else frozenset())

                t_scores = np.zeros((NUM_TISSUE_CLASSES, H, W), dtype=np.float32)

                # Always use sliding_window_predict_class for ALL image sizes.
                for cid in sorted(_t_present):
                    t_scores[cid] = sliding_window_predict_class(
                        model, transform, pil_im, TISSUE_PROMPT_TEXTS[cid], device,
                        patch_size=sw_patch, stride=sw_stride,
                        density_map=_density_map_cpu,
                        density_gate_net=_gate_net,
                        density_logit_prior_scale=_ndp_scale,
                        is_nuclei_prompt=False)

                pred_t = t_scores.argmax(0).astype(np.int64)
                pred_n = np.zeros((H, W), dtype=np.int64)

                # Per-image seg metrics
                if t_mask is not None and seg_mode in ("tissue", "both"):
                    for cls_key, m in compute_per_class_metrics(
                            pred_t, t_mask, TISSUE_CLASSES, "tissue",
                            dataset_name=ds_name).items():
                        seg_rows.append({"dataset": ds_name, "class_key": cls_key, **m})
                        seg_rows[-1].update({
                            "sample_idx": _source_local_idx,
                            "eval_index": _sidx,
                            "image_path": _source_image_path,
                            "mask_path": _source_mask_path,
                        })

                # Save vis.  External eval concatenates datasets, so a global
                # first-N quota can accidentally save only the first dataset.
                # Use a small per-dataset quota there; keep legacy global quota
                # for internal tests.
                if _vis_limit is None:
                    _want_vis = True
                elif _vis_limit_by_ds is not None:
                    _want_vis = _saved_vis_by_ds.get(ds_name, 0) < _vis_limit_by_ds
                else:
                    _want_vis = _saved_vis < _vis_limit

                if _want_vis:
                    _save_vis(pil_im, t_mask, n_mask, pred_t, pred_n,
                              os.path.join(_vis_dir,
                                           f"{_saved_vis:05d}_{ds_name}_{sample.get('local_idx', _sidx):05d}.png"),
                              content_hw=sample.get("content_hw"))
                    _saved_vis_by_ds[ds_name] = _saved_vis_by_ds.get(ds_name, 0) + 1
                    _saved_vis += 1

        # ── Aggregate seg ────────────────────────────────────────────────
        # Per-image rows CSV
        if seg_rows:
            _metric_keys = ("dice", "iou", "prec", "rec", "f1")
            with open(save_dir / f"test_seg_per_image_{_oc_tag}.csv", "w", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=["dataset", "sample_idx", "class_key"]
                    + list(_metric_keys)
                    + ["eval_index", "image_path", "mask_path"],
                )
                w.writeheader(); w.writerows(seg_rows)

            # Per-class (avg across all images & datasets)
            from collections import defaultdict as _dd2
            _by_cls: Dict[str, List[dict]] = _dd2(list)
            for r in seg_rows:
                _by_cls[r["class_key"]].append(r)
            per_cls_rows = []
            for ck in sorted(_by_cls):
                vals = _by_cls[ck]
                per_cls_rows.append({
                    "class_key": ck,
                    "n_images": len(vals),
                    **{k: float(np.mean([v[k] for v in vals])) for k in _metric_keys},
                })
            with open(save_dir / f"test_seg_per_class_{_oc_tag}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(per_cls_rows[0].keys()))
                w.writeheader(); w.writerows(per_cls_rows)

            # Per-dataset (avg across classes present in that dataset)
            _by_ds: Dict[str, List[dict]] = _dd2(list)
            for r in seg_rows:
                _by_ds[r["dataset"]].append(r)
            per_ds_rows = []
            for ds in sorted(_by_ds):
                vals = _by_ds[ds]
                _t_dice = [v["dice"] for v in vals if v["class_key"].startswith("tissue_")]
                _n_dice = [v["dice"] for v in vals if v["class_key"].startswith("nuclei_")]
                per_ds_rows.append({
                    "dataset": ds,
                    "n_images": len(vals),
                    "mean_dice": float(np.mean([v["dice"] for v in vals])),
                    "mean_tissue_dice": float(np.mean(_t_dice)) if _t_dice else float("nan"),
                    "mean_nuclei_dice": float(np.mean(_n_dice)) if _n_dice else float("nan"),
                    **{k: float(np.mean([v[k] for v in vals])) for k in ("iou", "prec", "rec", "f1")},
                })
            with open(save_dir / f"test_seg_per_dataset_{_oc_tag}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(per_ds_rows[0].keys()))
                w.writeheader(); w.writerows(per_ds_rows)

            # Global summary
            all_dice = [r["dice"] for r in seg_rows]
            t_dice   = [r["dice"] for r in seg_rows if r["class_key"].startswith("tissue_")]
            n_dice   = [r["dice"] for r in seg_rows if r["class_key"].startswith("nuclei_")]
            seg_summary = {
                "split": f"test_{_oc_tag}",
                "n_seg_images": len(seg_rows),
                "mean_dice": float(np.mean(all_dice)),
                "mean_tissue_dice": float(np.mean(t_dice)) if t_dice else float("nan"),
                "mean_nuclei_dice": float(np.mean(n_dice)) if n_dice else float("nan"),
            }
            with open(save_dir / f"test_seg_summary_{_oc_tag}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(seg_summary.keys()))
                w.writeheader(); w.writerow(seg_summary)

            print(f"  [{_oc_tag}] Seg — mean Dice {seg_summary['mean_dice']:.4f}  "
                  f"tissue {seg_summary['mean_tissue_dice']:.4f}  "
                  f"nuclei {seg_summary['mean_nuclei_dice']:.4f}")
            print(f"  [{_oc_tag}] Per-dataset:")
            for r in per_ds_rows:
                print(f"    {r['dataset']:22s}  Dice={r['mean_dice']:.4f}")
            print(f"  [{_oc_tag}] Per-class:")
            for r in per_cls_rows:
                print(f"    {r['class_key']:40s}  Dice={r['dice']:.4f}  n={r['n_images']}")
        elif _do_seg and not seg_items:
            print(f"  [{_oc_tag}] No seg items evaluated.")

        # ══════════════════════════════════════════════════════════════════
        # 2.  CLS EVALUATION (all trained datasets, use existing evaluate())
        # ══════════════════════════════════════════════════════════════════
        if _do_cls and cls_items:
            cls_ds = SAM3CombinedDataset(
                cls_items, augment=False, crop_to_target=True,
                native_crop_at_40x=native_crop_at_40x, cache_samples=True)
            _, _, cls_per_ds_rows = evaluate(
                model, transform, cls_ds, device,
                vis_dir=None, epoch=0, vis_num=0,
                organ_cond=_oc, seg_mode=seg_mode,
                tile_classifier=tile_classifier,
                prompt_enc=prompt_enc, adapter=adapter,
                histoplus=histoplus, density_fusion=density_fusion)

            # Per-dataset cls CSV
            if cls_per_ds_rows:
                with open(save_dir / f"test_cls_per_dataset_{_oc_tag}.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(cls_per_ds_rows[0].keys()))
                    w.writeheader(); w.writerows(cls_per_ds_rows)

                # Summary
                _organ_accs = [r["organ_acc"] for r in cls_per_ds_rows if "organ_acc" in r]
                _ham_accs   = [r["tissue_hamming"] for r in cls_per_ds_rows if "tissue_hamming" in r]
                cls_summary = {
                    "split": f"test_{_oc_tag}",
                    "n_datasets": len(cls_per_ds_rows),
                    "mean_organ_acc": float(np.mean(_organ_accs)) if _organ_accs else float("nan"),
                    "mean_tissue_hamming": float(np.mean(_ham_accs)) if _ham_accs else float("nan"),
                }
                with open(save_dir / f"test_cls_summary_{_oc_tag}.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(cls_summary.keys()))
                    w.writeheader(); w.writerow(cls_summary)

                print(f"  [{_oc_tag}] Cls — organ_acc {cls_summary['mean_organ_acc']:.4f}  "
                      f"tissue_hamming {cls_summary['mean_tissue_hamming']:.4f}")
                for r in cls_per_ds_rows:
                    _parts = [f"    {r['dataset']:22s}"]
                    if "organ_acc"       in r: _parts.append(f"OrganAcc={r['organ_acc']:.4f}")
                    if "tissue_hamming"  in r: _parts.append(f"TisHam={r['tissue_hamming']:.4f}")
                    if "tissue_dice"     in r: _parts.append(f"TisDice={r['tissue_dice']:.4f}")
                    print("  ".join(_parts))

        print(f"  [{_oc_tag}] Outputs → {save_dir}")

    if _test_trunk_hook is not None:
        _test_trunk_hook.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Full-image evaluation  (standalone --test_only mode)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _save_fullimg_vis_sliding(
    model,
    transform,
    eval_ds,
    vis_dir: str,
    tag: str,
    device: torch.device,
    organ_cond: bool,
    seg_mode: str = "both",
    native_crop_at_40x: int = NATIVE_CROP_AT_40X,
    tile_classifier: Optional[nn.Module] = None,
    histoplus:       Optional[InstanSegNucleiExtractor] = None,
):
    """Run sliding-window inference on **all** eval samples and save
    full-resolution PNGs + per-image Dice metrics.

    Patch size is derived from ``native_crop_at_40x`` scaled by each image's
    effective magnification so the model sees tissue at the same scale as training.

    Inference is gate-aligned with training: if the model carries a
    ``_density_gate_net`` and a positive ``_nuclei_density_logit_prior_scale``,
    nuclei prompts get the same per-patch gated density bias as training.
    """
    os.makedirs(vis_dir, exist_ok=True)
    _raw_model = model.module if hasattr(model, "module") else model
    _penc = _raw_model.backbone.language_backbone
    _gate_net  = getattr(_raw_model, "_density_gate_net", None)
    _ndp_scale = float(getattr(_raw_model, "_nuclei_density_logit_prior_scale", 0.0))

    per_image_rows: List[dict] = []

    for idx in tqdm(range(len(eval_ds)), desc=f"Vis SW {tag}", leave=False):
        sample  = eval_ds[idx]
        pil_im  = sample["image"]
        t_mask  = sample["tissue_mask"]
        n_mask  = sample["nuclei_mask"]
        ds_name = sample["dataset_name"]
        H, W    = pil_im.height, pil_im.width
        mag     = float(sample.get("magnification", 40.0))

        # Match training FOV: same physical area per patch as a training crop.
        # Lower-bound by SAM3_RESOLUTION so patches are never upsampled.
        sw_patch = max(64, int(native_crop_at_40x * mag / 40.0))
        sw_patch = max(sw_patch, SAM3_RESOLUTION)   # never upsample patches
        sw_patch = min(sw_patch, max(W, H))          # don't exceed image size
        sw_stride = max(32, sw_patch // 2)

        if hasattr(_penc, "current_tissue_type"):
            _penc.current_tissue_type = int(sample["tissue_type"]) if organ_cond else 0

        # ── Only predict classes ACTUALLY present in this image's GT ────
        _do_tissue = t_mask is not None and seg_mode in ("tissue", "both")
        # Always include class 0 (background) in inference.
        _t_present: frozenset = frozenset()
        if _do_tissue:
            _t_present = frozenset(int(c) for c in np.unique(t_mask)
                                   if 0 <= c < NUM_TISSUE_CLASSES)

        t_scores = np.zeros((NUM_TISSUE_CLASSES, H, W), dtype=np.float32)
        _density_map_cpu = None
        if histoplus is not None and getattr(_raw_model, "_density_fusion_mod", None) is not None:
            try:
                _key = f"{ds_name}:{sample.get('local_idx', idx)}"
                _density_map_cpu = histoplus.extract(pil_im, cache_key=_key)[0]
            except Exception:
                _density_map_cpu = None

        for cid in sorted(_t_present):
            t_scores[cid] = sliding_window_predict_class(
                model, transform, pil_im, TISSUE_PROMPT_TEXTS[cid], device,
                patch_size=sw_patch, stride=sw_stride,
                density_map=_density_map_cpu,
                density_gate_net=_gate_net,
                density_logit_prior_scale=_ndp_scale,
                is_nuclei_prompt=False)

        pred_t = t_scores.argmax(0).astype(np.int64) if _do_tissue else np.zeros((H, W), dtype=np.int64)
        pred_n = np.zeros((H, W), dtype=np.int64)

        # Per-image metrics
        if _do_tissue and t_mask is not None:
            img_metrics = compute_per_class_metrics(pred_t, t_mask, TISSUE_CLASSES,
                                                     "tissue", dataset_name=ds_name)
            for cls_key, m in img_metrics.items():
                per_image_rows.append({
                    "sample_idx": idx, "dataset": ds_name,
                    "class_key": cls_key, **m,
                    "pred_tissue_type": -1,
                    "gt_tissue_type": int(sample.get("tissue_type", 0)),
                })
        if False and n_mask is not None:  # nuclei seg removed
            img_metrics = compute_per_class_metrics(pred_n, n_mask, NUCLEI_CLASSES,
                                                     "nuclei", dataset_name=ds_name)
            for cls_key, m in img_metrics.items():
                per_image_rows.append({
                    "sample_idx": idx, "dataset": ds_name,
                    "class_key": cls_key, **m,
                    "pred_tissue_type": -1,
                    "gt_tissue_type": int(sample.get("tissue_type", 0)),
                })

        _save_vis(pil_im, t_mask if _do_tissue else None, n_mask if _do_nuclei else None,
                  pred_t, pred_n,
                  os.path.join(vis_dir, f"{ds_name}_{idx:05d}.png"),
                  content_hw=sample.get("content_hw"))

    # Write per-image CSV
    if per_image_rows:
        csv_path = os.path.join(vis_dir, f"per_image_metrics_{tag}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            w.writeheader()
            w.writerows(per_image_rows)
        all_dice = [r["dice"] for r in per_image_rows]
        print(f"  [{tag}] Sliding-window — {len(per_image_rows)} class-imgs, "
              f"mean Dice={np.mean(all_dice):.4f}  → {csv_path}")


@torch.no_grad()
def run_fullimg_eval(
    ckpt_path:     Path,
    model:         nn.Module,
    prompt_enc:    PromptTunedTextEncoder,
    items:         List[_Item],
    save_dir:      Path,
    device:        torch.device,
    adapter:       Optional[nn.Module] = None,
    lora_cfg:      Optional[LoRAConfig]         = None,
    tile_classifier: Optional[nn.Module] = None,
    histoplus:       Optional[InstanSegNucleiExtractor] = None,
    density_fusion:  Optional[nn.Module] = None,
    density_gate_net: Optional[nn.Module] = None,
    max_eval_size: int  = 2048,
    sw_max_eval_size: Optional[int] = None,
    vis_num:       int  = 8,
    seed:          int  = SEED,
    seg_mode:      str  = "both",
    native_crop_at_40x: int = NATIVE_CROP_AT_40X,
) -> None:
    """
    Full-image evaluation: loads ``ckpt_path``, runs inference on each item.

    Two evaluation passes:
    1. Quick single-pass: images capped at ``max_eval_size``, resized to 1008²
    2. Sliding-window: images capped at ``sw_max_eval_size`` (defaults to
       ``max_eval_size * 4``, i.e. closer to native), processed in tiles whose
       size matches the training FOV (``native_crop_at_40x`` scaled by mag).

    Writes:
        <save_dir>/fullimg_eval_metrics_{cond,nocond}.csv
        <save_dir>/fullimg_eval_per_class_metrics_{cond,nocond}.csv
        <save_dir>/vis/fullimg_eval/...
    """
    if sw_max_eval_size is None:
        sw_max_eval_size = max_eval_size * 4

    print("\n" + "=" * 60)
    print(f"  Full-Image Eval  ({len(items)} samples)")
    print(f"  Quick-pass max_eval_size  : {max_eval_size}")
    print(f"  Sliding-window max size   : {sw_max_eval_size}")
    print(f"  native_crop_at_40x        : {native_crop_at_40x}")
    print("=" * 60)
    if not ckpt_path.exists():
        print(f"  WARNING: checkpoint not found at {ckpt_path}.  Skipping.")
        return

    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    _m    = model.module if hasattr(model, "module") else model
    _load_compat(_m.backbone.language_backbone,
                 ckpt["prompt_enc"], tag="fullimg_eval", device=device)
    if lora_cfg is not None and ckpt.get("lora_state"):
        sd = _m.state_dict()
        sd.update({k: v.to(device) for k, v in ckpt["lora_state"].items()})
        _m.load_state_dict(sd)
    if adapter is not None and ckpt.get("adapter_state"):
        _load_compat(adapter,
                     ckpt["adapter_state"], tag="fullimg_eval_adapter", device=device)
    if density_fusion is not None and ckpt.get("density_fusion_state"):
        _load_compat(density_fusion,
                     ckpt["density_fusion_state"], tag="fullimg_eval_density_fusion", device=device)
    if density_gate_net is not None and ckpt.get("density_gate_state"):
        try:
            density_gate_net.load_state_dict(ckpt["density_gate_state"])
            print("  [density_gate] Loaded gate weights from checkpoint")
        except Exception as _e:
            print(f"  [density_gate] WARNING: failed to load gate weights: {_e}")
    _m._density_gate_net = density_gate_net
    _m._density_fusion_mod = density_fusion
    _m._pending_density = None
    _m._nuclei_density_logit_prior_scale = float(
        (ckpt.get("train_cfg") or {}).get("nuclei_density_logit_prior", 0.0)
    )
    print(f"  Loaded: {ckpt_path}  "
          f"(stage={ckpt.get('stage')}  epoch={ckpt.get('epoch')}  "
          f"best_val_dice={ckpt.get('best_dice', 0.0):.4f})")

    transform = _build_transform(SAM3_RESOLUTION)
    model.eval(); prompt_enc.eval()
    if adapter is not None:
        adapter.eval()
    _fullimg_trunk_hook = _register_trunk_feature_hook(
        model, stage_label=f"fullimg_stage{ckpt.get('stage', '?')}")

    # Quick-pass dataset: images capped at max_eval_size for fast evaluation
    eval_ds = SAM3CombinedDataset(
        items, augment=False, crop_to_target=False,
        max_eval_size=max_eval_size, native_crop_at_40x=native_crop_at_40x)

    # Sliding-window dataset: images capped at sw_max_eval_size (larger, closer
    # to native) for high-quality tiled inference.
    sw_eval_ds = SAM3CombinedDataset(
        items, augment=False, crop_to_target=False,
        max_eval_size=sw_max_eval_size, native_crop_at_40x=native_crop_at_40x)

    save_dir.mkdir(parents=True, exist_ok=True)
    _use_organ_bias_eval = bool(getattr(prompt_enc, "use_organ_bias", False))
    _eval_modes = ([(False, "nocond"), (True, "cond")]
                   if _use_organ_bias_eval else [(False, "nocond")])
    for _oc, _oc_tag in _eval_modes:
        summary, pcls_rows, _ = evaluate(
            model, transform, eval_ds, device,
            vis_dir=None, epoch=0, vis_num=0,
            organ_cond=_oc, seg_mode=seg_mode,
            tile_classifier=tile_classifier,
                              prompt_enc=prompt_enc, adapter=adapter, histoplus=histoplus, density_fusion=density_fusion)
        print(f"  [{_oc_tag}] TissueDice {summary['mean_tissue_dice']:.4f}  "
              f"NucleiDice {summary['mean_nuclei_dice']:.4f}  "
              f"MeanDice {summary['mean_dice']:.4f}")

        row = {"split": f"fullimg_{_oc_tag}", "ckpt": ckpt_path.name, **summary}
        with open(save_dir / f"fullimg_eval_metrics_{_oc_tag}.csv",
                  "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            w.writeheader(); w.writerow(row)
        if pcls_rows:
            with open(save_dir / f"fullimg_eval_per_class_metrics_{_oc_tag}.csv",
                      "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(pcls_rows[0].keys()))
                w.writeheader(); w.writerows(pcls_rows)

        # ── Sliding-window inference on ALL eval images ──────────────────
        vis_out = str(save_dir / "vis" / "fullimg_eval")
        _save_fullimg_vis_sliding(
            model, transform, sw_eval_ds,
            vis_dir=os.path.join(vis_out, f"fullimg_{_oc_tag}"),
            tag=f"fullimg_{_oc_tag}",
            device=device, organ_cond=_oc, seg_mode=seg_mode,
            native_crop_at_40x=native_crop_at_40x,
            tile_classifier=tile_classifier,
            histoplus=histoplus)
        print(f"  Vis saved (sliding-window): {vis_out}/fullimg_{_oc_tag}/")

    print("  Full-image eval done.\n")
    if _fullimg_trunk_hook is not None:
        _fullimg_trunk_hook.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Three-stage MedSAM3 fine-tuning on combined pathology data")
    # ── Resume shortcut ───────────────────────────────────────────────────
    p.add_argument("--resume_run_dir", type=str, default=None,
                   help="Path to an existing run directory. Loads train_config.json "
                        "from that directory (restoring all original hyperparameters) "
                        "and sets --resume_ckpt automatically. "
                        "All other arguments are ignored when this is set.")
    p.add_argument("--resume_from", type=str, default="best",
                   choices=["last", "best"],
                   help="Which checkpoint to resume from when using --resume_run_dir. "
                        "'last' (default) = last_checkpoint.pth (resumes training state). "
                        "'best' = best_model.pth (resumes from best val weights).")
    p.add_argument("--resume_load", type=str, default="all",
                   choices=["all", "lora", "prompt", "none"],
                   help="Which checkpoint weights to load when resuming. "
                        "'all' (default) = prompt_enc + lora + adapter + optimizer state. "
                        "'lora' = LoRA weights only (ignore prompt_enc/adapter size mismatches). "
                        "'prompt' = prompt_enc only (ignore LoRA/adapter size mismatches). "
                        "'none' = skip model weights entirely, only restore step counters.")
    # ── Normal arguments ─────────────────────────────────────────────────
    p.add_argument("--stage",      type=int,  default=None, choices=[1, 2, 3],
                   help="Training stage: 1=LoRA+prompts+organ_bias from scratch, "
                        "2=+cell-density adapter/fusion, "
                        "3=+tile cls/MIL + intra-image contrastive/SSL heads")
    p.add_argument("--prev_ckpt",  type=str,  default=None,
                   help="Path to previous-stage best_model.pth (required for stages 2/3)")
    p.add_argument("--prev_lora",  type=str,  default=None,
                   help="Path to a checkpoint .pth to load ONLY LoRA weights from. "
                        "Useful for fine-tuning with different prompt configs. "
                        "Ignores prompt_enc/adapter/optimizer — just LoRA.")
    p.add_argument("--stage2_variant", type=str, default="base",
                   choices=["base", "2-1", "bio"],
                   help="Stage-2 subvariant. 'base' uses feature fusion only. "
                        "'2-1'/'bio' additionally enables weak biological "
                        "regularizers on boundaries, density compatibility, "
                        "and gland lumen/ring cues.")
    p.add_argument("--resume_ckpt", type=str,  default=None,
                   help="Path to last_checkpoint.pth from the SAME stage to resume interrupted "
                        "training (restores model weights, optimizer, scheduler, scaler, "
                        "global_step, and epoch counter).")
    p.add_argument("--epochs",     type=int,  default=EPOCHS_PER_STAGE)
    p.add_argument("--batch_acc",  type=int,  default=8,
                   help="Gradient accumulation steps (effective batch size). "
                        "Default=8 tuned for RTX 5090 32 GB.")
    p.add_argument("--val_steps",  type=int,  default=1000,
                   help="Validate every N training samples (0 = end-of-epoch only).")
    p.add_argument("--num_workers",type=int,  default=8,
                   help="DataLoader workers for training. Increase to improve input pipeline throughput.")
    p.add_argument("--fg_crop_prob", type=float, default=0.5,
                   help="Probability of sampling training crops centered on "
                        "foreground (non-background) mask pixels. Range [0,1].")
    p.add_argument("--steps_per_epoch", type=int, default=None,
                   help="Cap training samples per epoch. Use e.g. 2000 to limit "
                        "a 15-hour epoch to ~1 hour without changing batch or LR.")
    p.add_argument("--total_steps", type=int, default=None,
                   help="Total optimizer-step budget. When set, training runs "
                        "until this many global steps are accumulated and "
                        "--epochs becomes an upper bound. Cosine LR decay is "
                        "computed against this value. Use this when you want "
                        "step-based budgeting instead of epoch-based.")
    p.add_argument("--nuclei_density_logit_prior", type=float, default=0.0,
                   help="Option-A density prior strength. >0 adds "
                        "scale*log(density) to nuclei prompt logits before "
                        "loss/argmax. Suppresses nuclei predictions in "
                        "low-density regions. Try 0.5-1.0. 0 = disabled.")
    p.add_argument("--instanseg_batch_size", type=int, default=0,
                   help="Batch size for InstanSeg density extraction. Kept for "
                        "compatibility; this older extractor path runs one image "
                        "at a time.")
    p.add_argument("--cell_density_source", type=str, default="instanseg",
                   choices=["instanseg", "hematoxylin", "none"],
                   help="Source for Stage 2/3 cell-density maps. "
                        "'hematoxylin' uses a fast optical-density proxy; "
                        "'none' disables cell-density feature fusion/prior.")
    p.add_argument("--no_cell_info", action="store_true",
                   help="Hard-disable all cell/density information: no InstanSeg/"
                        "hematoxylin extractor, no cell-cue feature fusion, no "
                        "density logit prior, and no density/bio auxiliary losses. "
                        "Useful for Stage-2 ablations where only LoRA/prompts/"
                        "adapter/cls-head choices should remain.")
    p.add_argument("--hematoxylin_density_max_side", type=int, default=2048,
                   help="Max side used internally by the hematoxylin-density proxy.")
    p.add_argument("--hematoxylin_density_sigma", type=float, default=8.0,
                   help="Gaussian smoothing sigma for hematoxylin-density proxy.")
    p.add_argument("--hematoxylin_density_norm", dest="hematoxylin_density_norm",
                   action="store_true", default=True,
                   help="Apply robust RGB normalization before hematoxylin-density calculation.")
    p.add_argument("--no_hematoxylin_density_norm", dest="hematoxylin_density_norm",
                   action="store_false",
                   help="Disable hematoxylin-density-specific RGB normalization.")
    p.add_argument("--cell_density_train_mode", type=str, default="coarse",
                   choices=["fast", "coarse", "tiled"],
                   help="Compatibility flag for newer commands. The current "
                        "InstanSeg extractor ignores tiled/coarse mode; "
                        "hematoxylin is always fast.")
    p.add_argument("--cell_cue_feature_fusion", dest="cell_cue_feature_fusion",
                   action="store_true", default=True,
                   help="Fuse cell cue maps into deep SAM image features in stages 2/3.")
    p.add_argument("--no_cell_cue_feature_fusion", dest="cell_cue_feature_fusion",
                   action="store_false",
                   help="Disable Stage 2/3 cell-cue feature fusion.")
    p.add_argument("--cell_cue_fusion_hidden", type=int, default=96,
                   help="Hidden channels for the Stage 2/3 cell-cue fusion CNN.")
    p.add_argument("--cell_cue_fusion_max_gamma", type=float, default=0.25,
                   help="Maximum absolute multiplicative FiLM strength.")
    p.add_argument("--cell_cue_fusion_max_beta", type=float, default=0.25,
                   help="Maximum absolute additive FiLM strength.")
    p.add_argument("--run_initial_eval", dest="run_initial_eval",
                   action="store_true", default=True,
                   help="Run the step-0 validation pass before training starts (default: True).")
    p.add_argument("--skip_initial_eval", dest="run_initial_eval",
                   action="store_false",
                   help="Skip the step-0 validation pass before training starts.")
    p.add_argument("--save_best_vis", action="store_true",
                   help="Save visualisations whenever a new best checkpoint is found.")
    p.add_argument("--skip_auto_test", action="store_true",
                   help="Skip the auto-run segmentation test after training completes.")
    p.add_argument("--reset_best_dice", action="store_true",
                   help="Ignore the previous best_dice from the checkpoint and start "
                        "from 0.0.  Use when the validation set has changed (e.g. "
                        "different sampling, new datasets) so the old best_dice is "
                        "no longer meaningful.")
    p.add_argument("--quick_val_max_samples", type=int, default=None,
                   help="Optionally subsample the fixed quick-val set for faster validation.")
    p.add_argument("--setup_cache_dir", type=str, default=None,
                   help="Optional directory for caching deterministic train/holdout "
                        "quick-val item selections. Useful for Stage-3 ablation "
                        "groups that reuse the same split and validation sampler.")
    p.add_argument("--val_samples_per_class", type=int, default=3,
                   help="Train quick-val metric sampler: number of pixel-verified "
                        "samples per class per dataset. This is the primary "
                        "best-checkpoint validation set. Default: 3.")
    p.add_argument("--val_samples_per_dataset", type=int, default=None,
                   help="Holdout-val metric and visualization sampler: number of "
                        "foreground-verified samples per dataset during training. "
                        "Defaults to --val_samples_per_class. Use 2 or 3 to save "
                        "multiple holdout-val vis images per dataset.")
    p.add_argument("--curriculum", action="store_true",
                   help="Enable curriculum learning: warm up for --curriculum_warmup_epochs "
                        "epochs on multi-class datasets (BCSS, NuCLS, PanNuke, CoNIC2022, "
                        "Lizard, MoNuSAC, SICAPv2, WSSS4LUAD) before introducing the full "
                        "SegPath-dominated dataset.  Prevents the single-class majority "
                        "from dominating early feature learning.")
    p.add_argument("--curriculum_warmup_epochs", type=int, default=2,
                   help="Warmup duration in epochs (default: 2). Only used with --curriculum.")
    p.add_argument("--focal_gamma", type=float, default=0.0,
                   help="Focal loss gamma (default: 0 = standard BCE). "
                        "Recommended value: 2.0.  Applies a (1-p_t)^gamma weight "
                        "that down-weights easy negatives and suppresses all-positive "
                        "collapse without changing the Dice component.")
    p.add_argument("--dice_weight", type=float, default=1.0,
                   help="Weight for the Dice loss component (default: 1.0). "
                        "Higher values emphasise overlap quality.")
    p.add_argument("--bce_weight", type=float, default=1.0,
                   help="Weight for the BCE (focal if --focal_gamma>0) loss component "
                        "(default: 1.0). Higher values emphasise per-pixel accuracy.")
    p.add_argument("--repulsion_weight", type=float, default=UNLABELED_REPULSION_WEIGHT,
                   help=f"Repulsion weight for p² penalty on gt==0 pixels "
                        f"(default: {UNLABELED_REPULSION_WEIGHT}). "
                        f"Higher = stronger push toward zero in unlabeled regions.")
    p.add_argument("--negative_boost", type=float, default=NEGATIVE_BOOST,
                   help=f"Multiplier for the (1-g) BCE term to strengthen "
                        f"false-positive penalty on unlabeled pixels "
                        f"(default: {NEGATIVE_BOOST}). Higher = stronger push "
                        f"toward zero in non-target regions.")
    p.add_argument("--area_loss_weight", type=float, default=AREA_LOSS_WEIGHT,
                   help=f"Weight for (sum(p)/sum(g) - 1)² area-matching loss "
                        f"(default: {AREA_LOSS_WEIGHT}). Directly penalises "
                        f"the predicted probability mass deviating from the "
                        f"annotation mass. If pred area is 10× GT area, "
                        f"loss contribution = 81 * weight.")
    p.add_argument("--density_consistency_weight", type=float,
                   default=AUX_PRIOR_WEIGHT,
                   help=f"Stage 2/3 cell-density consistency regularizer weight. "
                        f"Default: {AUX_PRIOR_WEIGHT}. Use 0 for ablation.")
    p.add_argument("--bio_boundary_weight", type=float, default=0.0,
                   help="Weak Stage 2-1 boundary alignment loss weight.")
    p.add_argument("--bio_compat_weight", type=float, default=0.0,
                   help="Weak Stage 2-1 class-specific density compatibility loss weight.")
    p.add_argument("--bio_gland_lumen_weight", type=float, default=0.0,
                   help="Weak Stage 2-1 gland lumen/ring regularization weight.")
    p.add_argument("--bg_suppression_weight", type=float, default=0.0,
                   help="Stage 2+ mild suppression of foreground predictions in "
                        "WSI white/blank background regions. p² penalty on near-white "
                        "pixels (RGB>=235). Default: 0 (disabled). Try 0.005.")
    p.add_argument("--scale_reg_weight", type=float, default=0.0,
                   help="Stage 3 scale regularization weight. Matches predicted "
                        "foreground area fraction to GT area fraction with a "
                        "stable log-area penalty. Default: 0.")
    p.add_argument("--shape_reg_weight", type=float, default=0.0,
                   help="Stage 3 shape regularization weight. Weakly aligns "
                        "predicted and GT soft boundaries plus mild smoothness. "
                        "Default: 0.")
    p.add_argument("--structure_class_weight", type=float, 
                   default=STRUCTURE_CLASS_LOSS_WEIGHT,
                   help=f"Stage 2/3-only loss weight for tissue classes whose "
                        f"prompt spec axis is 'structure' (soft recall + "
                        f"boundary Dice). Default: {STRUCTURE_CLASS_LOSS_WEIGHT}. "
                        f"Use 0 to disable.")
    p.add_argument("--gland_structure_weight", dest="structure_class_weight",
                   type=float, default=argparse.SUPPRESS,
                   help=argparse.SUPPRESS)
    p.add_argument("--hard_mining_start_epoch", type=int, default=-1,
                   help="Epoch at which to start up-weighting poorly-performing "
                        "classes in the training sampler (-1 = disabled). "
                        "Set to 0 to enable from the very first epoch. "
                        "Classes with lower per-class Dice get higher sampling "
                        "weight to focus training on weak classes.")
    p.add_argument("--ssl_weight", type=float, default=SSL_WEIGHT,
                   help="Stage-3 pooled BYOL weight (default from SSL_WEIGHT).")
    p.add_argument("--intra_image_contrast_weight", type=float,
                   default=INTRA_IMAGE_CONTRAST_WEIGHT,
                   help="Stage-3 intra-image contrastive loss weight.")
    p.add_argument("--tile_cls_weight", type=float, default=TILE_CLS_WEIGHT,
                   help="Tile-level classification loss weight used in Stage 3.")
    p.add_argument("--cls_only_mil_weight", type=float, default=CLS_ONLY_MIL_WEIGHT,
                   help="Weight for cls-only MIL presence loss from segmentation maps.")
    p.add_argument("--cls_only_absent_weight", type=float, default=CLS_ONLY_ABSENT_WEIGHT,
                   help="Weight for cls-only absent-class suppression loss.")
    p.add_argument("--cls_only_topk_ratio", type=float, default=CLS_ONLY_TOPK_RATIO,
                   help="Top-k pooling ratio for cls-only MIL loss.")
    p.add_argument("--cls_only_entropy_weight", type=float, default=CLS_ONLY_ENTROPY_WEIGHT,
                   help="Pixel-entropy minimisation weight on cls-only tissue probs.")
    p.add_argument("--organ_warmup_steps", type=int, default=STAGE4_ORGAN_WARMUP_STEPS,
                   help="Defer organ-classification gradient in cls-only branch for N steps.")
    p.add_argument("--cls_only_stride", type=int, default=3,
                   help="Interleave one cls-only sample every N train steps.")
    # ── Morphology-Relation Self-Supervision ────────────────────────────
    p.add_argument("--morph_graph_weight", type=float,
                   default=MORPH_GRAPH_WEIGHT,
                   help="Stage-3 local morphology stats regression weight.")
    p.add_argument("--morph_transition_weight", type=float,
                   default=MORPH_TRANSITION_WEIGHT,
                   help="Stage-3 morphology-boundary transition loss weight.")
    p.add_argument("--morph_grid_size", type=int, default=MORPH_GRAPH_GRID_SIZE,
                   help="Spatial grid resolution for local graph statistics.")
    p.add_argument("--morph_keep_n_cells", type=int, default=MORPH_KEEP_N_CELLS,
                   help="Max cells per graph (subsampled on-the-fly).")
    p.add_argument("--morph_warmup_steps", type=int, default=0,
                   help="Optimizer steps with morphology losses held at zero.")
    p.add_argument("--morph_ramp_steps", type=int, default=0,
                   help="Steps used to linearly ramp morphology loss weights.")
    p.add_argument("--morph_relation_mode", type=str,
                   default=MORPH_RELATION_MODE,
                   choices=["none", "stats_only", "transition_only", "full"],
                   help="Morphology-relation SSL mode: none / stats_only / "
                        "transition_only / full.")
    p.add_argument("--seg_mode", type=str, default="both",
                   choices=["tissue", "nuclei", "both"],
                   help="Which segmentation modality to train and evaluate. "
                        "'tissue': tissue masks only (faster, no nuclei prompts). "
                        "'nuclei': nuclei masks only. "
                        "'both': all 18 classes simultaneously (default).")
    p.add_argument("--seed",       type=int,  default=SEED)
    p.add_argument("--out_dir",    type=str,  default="./medsam3_combined_runs")
    p.add_argument("--bpe_path",   type=str,  default=BPE_PATH)
    p.add_argument("--train_ratio",type=float,default=0.8)
    p.add_argument("--native_crop_at_40x", type=int, default=NATIVE_CROP_AT_40X,
                   help="Crop size in pixels at 40× magnification. Scaled by "
                        "native_mag/40 for other magnifications. Default: 1024.")
    p.add_argument(
        "--no_magnification_alignment", action="store_true",
        help="Standard-baseline mode: skip magnification rescaling and physical-FOV "
             "cropping; resize each dataset's native full-image view for the model.")
    # ── Full-image eval (--test_only mode) ───────────────────────────────
    p.add_argument("--test_only", action="store_true",
                   help="Skip training and run full-image evaluation only. "
                        "Use --test_ckpt to specify the checkpoint, or provide "
                        "--resume_run_dir to auto-select best_model.pth from that run.")
    p.add_argument("--test_ckpt", type=str, default=None,
                   help="Explicit path to a .pth checkpoint for --test_only mode. "
                        "Defaults to best_model.pth inside the run directory.")
    p.add_argument("--bare_medsam3_test_only", action="store_true",
                   help="In --test_only mode, evaluate the original MedSAM3 model "
                        "without PathXAI LoRA/prompt-tuning checkpoint weights.")
    p.add_argument("--bare_medsam3_out_dir", type=str,
                   default="./medsam3_combined_runs/bare_medsam3_baseline",
                   help="Output base directory for --bare_medsam3_test_only results.")
    p.add_argument("--test_datasets", type=str, default=None,
                   help="Comma-separated dataset names to restrict --test_only "
                        "evaluation to (e.g. 'BCSS,GlaS,NuCLS'). "
                        "If omitted, all items in the selected split are evaluated.")
    p.add_argument("--test_split", type=str, default="val",
                   choices=["train", "val", "all"],
                   help="Which data split to evaluate in --test_only mode (default: 'val').")
    p.add_argument("--test_eval", type=str, default="both",
                   choices=["seg", "cls", "both"],
                   help="What to evaluate in --test_only mode: 'seg' (pixel masks only), "
                        "'cls' (tile-level organ+tissue classification only), "
                        "or 'both' (default).")
    p.add_argument("--test_target", type=str, default="internal",
                   choices=["internal", "external", "both"],
                   help="Which test set to evaluate in --test_only mode. "
                        "'internal' uses the trained-dataset split, 'external' "
                        "uses external_eval_datasets.py, and 'both' runs both.")
    p.add_argument("--test_max_eval_size", type=int, default=2048,
                   help="Maximum image dimension (px) for the quick single-pass eval. "
                        "Images larger than this are downsampled on the longest side. "
                        "Default: 2048.")
    p.add_argument("--test_sw_max_size", type=int, default=None,
                   help="Maximum image dimension (px) for sliding-window eval. "
                        "Defaults to test_max_eval_size * 4.  Set to a large value "
                        "(e.g. 16384) to keep images close to native resolution.")
    p.add_argument("--test_vis_num", type=int, default=8,
                   help="Number of visualisation images to save during full-image eval.")
    p.add_argument("--external_eval_root", type=str,
                   default="/local/datasets/public/datasets",
                   help="Root directory containing external eval datasets.")
    p.add_argument("--external_eval_datasets", type=str,
                   default="ACDC_LungHP,PAIP2019",
                   help="Comma-separated external eval datasets to use.")
    p.add_argument("--external_eval_max_samples", type=int, default=0,
                   help="Optional max samples per external dataset. 0 means full dataset.")
    p.add_argument("--external_eval_max_samples_by_dataset", type=str, default=None,
                   help="Optional comma-separated Dataset=N overrides, e.g. "
                        "ACDC_LungHP=5000,PAIP2019=5000. These override "
                        "--external_eval_max_samples for matching datasets.")
    p.add_argument("--external_eval_subset_strategy", type=str, default="random",
                   choices=["random", "label_area"],
                   help="How to subsample large external datasets. 'label_area' stratifies "
                        "by foreground/tumor mask area bins.")
    p.add_argument("--external_eval_subset_cache_dir", type=str,
                   default="/local/PathXAI/.cache/external_eval_subsets",
                   help="Cache directory for deterministic external subset indices.")
    p.add_argument("--external_eval_stratify_pool_multiplier", type=int, default=10,
                   help="For label_area external subsampling, scan this many times "
                        "--external_eval_max_samples as a deterministic candidate pool "
                        "before stratified sampling.")
    p.add_argument("--external_acdc_split", type=str, default="all",
                   choices=["train", "val", "all"],
                   help="ACDC_LungHP split for external eval.")
    p.add_argument("--external_paip_split", type=str, default="all",
                   choices=["official_test", "all"],
                   help="PAIP2019 split for external eval.")
    p.add_argument("--external_acdc_magnification", type=float, default=20.0,
                   help="Native magnification used for ACDC_LungHP external eval.")
    p.add_argument("--external_paip_magnification", type=float, default=20.0,
                   help="Native magnification used for PAIP2019 external eval.")
    # ── Training vis ────────────────────────────────────────────────────
    p.add_argument("--vis_num", type=int, default=8,
                   help="Number of vis images during training validation / best-ckpt. "
                        "0 disables training vis; >0 caps per-snapshot vis count.")
    # ── Learning rate override ────────────────────────────────────────────
    p.add_argument("--lr", type=float, default=None,
                   help="Override learning rate for all trainable param groups "
                        "(prompt_embeddings, LoRA, adapter). Stage 2/3 default "
                        "is conservative for LoRA/prompts (1e-6) while adapter "
                        "uses LR_ADAPTER unless --adapter_lr is set.")
    p.add_argument("--prompt_lr", type=float, default=None,
                   help="Override prompt_embeddings + organ_bias LR independently. "
                        "When set, ``--lr`` (if provided) only controls LoRA/adapter. "
                        "Example: ``--prompt_lr 5e-4 --lr 2e-4`` gives prompts 5e-4 "
                        "and LoRA/adapter 2e-4.")
    p.add_argument("--adapter_lr", type=float, default=None,
                   help="Override cell-adapter (SlideMorphAdapter) LR independently. "
                        "When set, takes priority over --lr for the adapter param group. "
                        "Example: ``--adapter_lr 5e-5 --lr 1e-5`` gives adapter 5e-5 "
                        "and LoRA 1e-5.")
    p.add_argument("--freeze_lora_stage2", action="store_true", default=False,
                   help="Freeze LoRA in stage 2. Default is False: stage 2 "
                        "keeps LoRA trainable at a small LR so density fusion "
                        "can refine spatial segmentation.")
    p.add_argument("--no_freeze_lora_stage2", action="store_true", default=False,
                   help=argparse.SUPPRESS)
    p.add_argument("--freeze_prompts_after_stage1", dest="freeze_prompts_after_stage1",
                   action="store_true", default=False,
                   help="Freeze prompt embeddings/organ bias in stages 2/3.")
    p.add_argument("--no_freeze_prompts_after_stage1", dest="freeze_prompts_after_stage1",
                   action="store_false",
                   help="Allow prompt encoder updates in stages 2/3.")
    p.add_argument("--use_organ_bias", dest="use_organ_bias",
                   action="store_true", default=False,
                   help="Enable organ-conditioned prompt bias and cond/nocond eval. "
                        "Default is disabled because it has not improved current runs.")
    p.add_argument("--no_organ_bias", dest="use_organ_bias",
                   action="store_false",
                   help="Disable organ-conditioned prompt bias (default).")
    # ── --resume_run_dir: load saved config as defaults so CLI args override ─
    # Pre-parse just to detect --resume_run_dir before the main parse.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--resume_run_dir", type=str, default=None)
    _pre_args, _ = _pre.parse_known_args()

    _saved_cfg: dict = {}
    if _pre_args.resume_run_dir:
        _run_dir = Path(_pre_args.resume_run_dir)
        _cfg_path = _run_dir / "train_config.json"
        if _cfg_path.exists():
            with open(_cfg_path) as _f:
                _saved_cfg = json.load(_f)
        else:
            # Try to pull it from last_checkpoint.pth
            _lc = _run_dir / "last_checkpoint.pth"
            if _lc.exists():
                _rc = torch.load(_lc, map_location="cpu", weights_only=False)
                _saved_cfg = _rc.get("train_cfg") or {}
        if not _saved_cfg:
            raise FileNotFoundError(
                f"No train_config.json or 'train_cfg' key found in {_run_dir}")
        # Apply saved values as parser defaults — CLI args passed on the
        # command line will naturally take precedence over these defaults.
        _cfg_for_defaults = {k: v for k, v in _saved_cfg.items()
                             if k not in ("resume_ckpt", "resume_run_dir",
                                          "_resume_save_dir", "prev_ckpt",
                                          "resume_load")}
        p.set_defaults(**_cfg_for_defaults)
        print(f"[resume_run_dir] Loaded base config from {_cfg_path}")

    args = p.parse_args()

    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir)
        last_ckpt = run_dir / "last_checkpoint.pth"
        best_ckpt = run_dir / "best_model.pth"
        _prefer_last = (args.resume_from == "last")
        if _prefer_last and last_ckpt.exists():
            args.resume_ckpt = str(last_ckpt)
        elif best_ckpt.exists():
            args.resume_ckpt = str(best_ckpt)
            if _prefer_last:
                print(f"[resume_run_dir] last_checkpoint.pth not found; "
                      f"falling back to best_model.pth")
        elif last_ckpt.exists():
            args.resume_ckpt = str(last_ckpt)
            print(f"[resume_run_dir] best_model.pth not found; "
                  f"falling back to last_checkpoint.pth")
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {run_dir} "
                "(neither last_checkpoint.pth nor best_model.pth).")
        # Keep the run in its original directory
        args._resume_save_dir = str(run_dir)
        print(f"[resume_run_dir] Resuming from {args.resume_ckpt} "
              f"(--resume_from={args.resume_from})")
        # Print which args were overridden by CLI vs loaded from config
        import sys as _sys
        _cli_keys = {a.lstrip("-").replace("-", "_")
                     for a in _sys.argv[1:] if a.startswith("--")}
        _overridden = _cli_keys & set(_saved_cfg.keys())
        if _overridden:
            print(f"[resume_run_dir] CLI overrides: "
                  + ", ".join(f"{k}={getattr(args, k)}" for k in sorted(_overridden)
                               if hasattr(args, k)))
    else:
        args._resume_save_dir = None

    if args.stage is None:
        if args.test_only and args.test_ckpt:
            _tmp = torch.load(args.test_ckpt, map_location="cpu", weights_only=False)
            args.stage = int(_tmp.get("stage", 1))
            del _tmp
            print(f"[test_only] Inferred --stage={args.stage} from checkpoint.")
        else:
            p.error("--stage is required unless --resume_run_dir is used")

    # Auto-infer seg_mode and native_crop_at_40x from checkpoint's train_cfg
    # when --test_only is used.  Saves retyping params that are already stored.
    if args.test_only and args.test_ckpt:
        _tcfg = None
        _ckpt_path = Path(args.test_ckpt)
        if _ckpt_path.exists():
            _tc = torch.load(_ckpt_path, map_location="cpu", weights_only=False)
            _tcfg = _tc.get("train_cfg") or {}
            del _tc
        if _tcfg:
            # Only override if the user didn't explicitly pass them on the CLI.
            import sys as _sys
            _cli_flags = {a.lstrip("-").replace("-", "_")
                          for a in _sys.argv[1:] if a.startswith("--")}
            if "seg_mode" not in _cli_flags and "seg_mode" in _tcfg:
                args.seg_mode = _tcfg["seg_mode"]
                print(f"[test_only] Inferred --seg_mode={args.seg_mode} from checkpoint train_cfg.")
            if "native_crop_at_40x" not in _cli_flags and "native_crop_at_40x" in _tcfg:
                args.native_crop_at_40x = _tcfg["native_crop_at_40x"]
                print(f"[test_only] Inferred --native_crop_at_40x={args.native_crop_at_40x} from checkpoint train_cfg.")

    if args.no_cell_info:
        args.cell_density_source = "none"
        args.cell_cue_feature_fusion = False
        args.nuclei_density_logit_prior = 0.0
        args.density_consistency_weight = 0.0
        args.bio_boundary_weight = 0.0
        args.bio_compat_weight = 0.0
        args.bio_gland_lumen_weight = 0.0
        print("[CellInfo] Disabled by --no_cell_info: no density extractor, "
              "fusion, density prior, or density/bio losses.")

    if args.stage2_variant in ("2-1", "bio") and not args.no_cell_info:
        if args.bio_boundary_weight == 0.0:
            args.bio_boundary_weight = 0.01
        if args.bio_compat_weight == 0.0:
            args.bio_compat_weight = 0.005
        if args.bio_gland_lumen_weight == 0.0:
            args.bio_gland_lumen_weight = 0.01
        if args.density_consistency_weight == AUX_PRIOR_WEIGHT:
            args.density_consistency_weight = 0.005

    return args


def main():
    args   = _parse_args()

    # Reduce CUDA memory fragmentation.  Must be set before any CUDA allocation.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if not args.test_only:
        # Timestamp-based run directory (new folder even when resuming)
        if args._resume_save_dir:
            _original_run = Path(args._resume_save_dir)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(args.out_dir) / f"run_{ts}_stage{args.stage}_{args.seg_mode}"
            save_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Resume] Original run: {_original_run}")
            print(f"[Resume] New run folder: {save_dir}")
            # Point resume_ckpt to the original checkpoint so weights are loaded
            if not args.resume_ckpt:
                _best = _original_run / "best_model.pth"
                _last = _original_run / "last_checkpoint.pth"
                if _last.exists():
                    args.resume_ckpt = str(_last)
                elif _best.exists():
                    args.resume_ckpt = str(_best)
                print(f"[Resume] Loading weights from: {args.resume_ckpt}")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = Path(args.out_dir) / f"run_{ts}_stage{args.stage}_{args.seg_mode}"
            save_dir.mkdir(parents=True, exist_ok=True)

        # ── Save training config (skip transient resume fields) ───────────────
        # Written once at the start of every NEW run so that --resume_run_dir
        # can reconstruct the exact original hyperparameters.
        cfg_path = save_dir / "train_config.json"
        if not cfg_path.exists():
            _cfg = {k: v for k, v in vars(args).items()
                    if k not in ("resume_ckpt", "resume_run_dir", "_resume_save_dir",
                                 "prev_lora")}
            with open(cfg_path, "w") as _f:
                json.dump(_cfg, _f, indent=2)
            print(f"Config saved → {cfg_path}")

        # ── Save the exact command line used to launch this run ──────────────
        # A reproducible bash script alongside the config; useful for re-running
        # the experiment or sharing the exact invocation.
        _cmd_path = save_dir / "run_command.sh"
        if not _cmd_path.exists():
            import shlex
            _quoted = " ".join(shlex.quote(a) for a in sys.argv)
            with open(_cmd_path, "w") as _f:
                _f.write("#!/usr/bin/env bash\n")
                _f.write(f"# Auto-saved at run start: {datetime.now().isoformat()}\n")
                _f.write(f"# CWD: {os.getcwd()}\n")
                _f.write(f"# Python: {sys.executable}\n")
                _f.write(f"cd {shlex.quote(os.getcwd())}\n")
                _f.write(f"{shlex.quote(sys.executable)} {_quoted}\n")
            try:
                os.chmod(_cmd_path, 0o755)
            except OSError:
                pass
            print(f"Run command saved → {_cmd_path}")
    else:
        # --test_only: no training run dir needed
        save_dir = Path(".")  # unused; everything goes via _test_out

    # ── Build combined dataset splits ──────────────────────────────────────
    print("Loading sub-datasets …")
    sub_datasets = _build_default_datasets()
    # Drop datasets that are unused for stage 1/2 tissue/both-mode training —
    # they are filtered out of train_items anyway and only add overhead.
    if args.stage <= 2 and args.seg_mode in ("tissue", "both"):
        _skip_ds = {
            # Nuclei-only datasets (no tissue region masks)
            "NuCLS", "Kumar", "CPM15", "CPM17", "PanNuke", "CoNIC2022",
            "Lizard", "MoNuSAC", "Janowczyk", "TNBC",
            # Tile-cls-only datasets (no pixel masks — used only at stage 3+)
            "CRC100K", "MHIST", "ICIAR2018", "PatchCamelyon17",
            "SkinCancer", "LungColon", "Osteosarcoma",
            # SegPath: IHC cell-blob masks, excluded from tissue seg training
            "SegPath",
            # SICAPv2: bad quality, escape — excluded from all training
            "SICAPv2",
        }
        sub_datasets = {k: v for k, v in sub_datasets.items()
                        if k not in _skip_ds}
        print(f"  [stage {args.stage}/{args.seg_mode}] Skipped {len(_skip_ds)} unused datasets.")
    else:
        print(f"  [stage {args.stage}/{args.seg_mode}] Loading all datasets (no skip).")

    print("Building splits …")
    train_items, val_items = generate_splits(
        sub_datasets, split_dir=SPLIT_DIR,
        train_ratio=args.train_ratio, seed=args.seed)
    rng = random.Random(args.seed)
    rng.shuffle(train_items)
    rng.shuffle(val_items)
    print(f"  Train items: {len(train_items):,}  Val items: {len(val_items):,}")

    # ── Stage-specific model setup ─────────────────────────────────────────
    # If --lr was provided on the CLI, use it for ALL param groups (prompt, LoRA,
    # adapter).  Otherwise `run_training` falls back to the script's hardcoded
    # per-group defaults (LR_PROMPT / LR_LORA / LR_ADAPTER).
    _lr = args.lr   # None → run_training uses hardcoded defaults
    if args.bare_medsam3_test_only:
        if not args.test_only:
            raise ValueError("--bare_medsam3_test_only is only valid with --test_only")
        if args.stage != 1:
            print("[Bare MedSAM3] Forcing --stage=1 for no-LoRA baseline evaluation.")
            args.stage = 1
        model, prompt_enc, lora_cfg = build_bare_medsam3_model(device)
        adapter   = None
        histoplus = None
        ssl_head = None
        pixel_ssl_head = None
        lr        = _lr if _lr is not None else LR_LORA
        tile_classifier = None

    elif args.stage == 1:
        model, prompt_enc, lora_cfg = build_stage1_model(device)
        adapter   = None
        histoplus = None
        ssl_head = None
        pixel_ssl_head = None
        lr        = _lr if _lr is not None else LR_LORA
        tile_classifier = None

        # ── Initialise Stage-1 weights from a previous full checkpoint ─────
        if args.prev_ckpt:
            _pc_path = Path(args.prev_ckpt)
            if not _pc_path.exists():
                print(f"WARNING: --prev_ckpt path not found: {_pc_path}")
            else:
                print(f"[prev_ckpt] Loading Stage-1 weights from {_pc_path}")
                _pc = torch.load(_pc_path, map_location=device, weights_only=False)
                _raw_m = model.module if hasattr(model, "module") else model
                if _pc.get("lora_state"):
                    _sd = _raw_m.state_dict()
                    _sd.update({k: v.to(device) for k, v in _pc["lora_state"].items()})
                    _raw_m.load_state_dict(_sd)
                    print(f"  [prev_ckpt] Loaded {len(_pc['lora_state'])} LoRA keys")
                if _pc.get("prompt_enc"):
                    _load_compat(
                        _raw_m.backbone.language_backbone,
                        _pc["prompt_enc"], tag="prev_ckpt_prompt", device=device)
                del _pc

    else:  # stage 2 or 3
        _init_ckpt = args.prev_ckpt or args.resume_ckpt or (
            args.test_ckpt if args.test_only else None)
        if not _init_ckpt:
            raise ValueError("--prev_ckpt (or --resume_ckpt) is required for stages 2/3")
        if args.stage == 2:
            model, prompt_enc, adapter, lora_cfg = build_stage2_model(_init_ckpt, device)
            ssl_head = None
            pixel_ssl_head = None
            tile_classifier = None
            if args.tile_cls_weight > 0.0:
                _tck = torch.load(_init_ckpt, map_location=device, weights_only=False)
                tile_classifier = _build_tile_classifier_heads(
                    device, ckpt=_tck, tag="stage2_tilecls")
                del _tck
                print("[Stage 2] Tile classification heads will train on seg datasets only "
                      "(features detached; no cls-only datasets are added).")
        elif args.stage == 3:
            model, prompt_enc, adapter, lora_cfg, tile_classifier, ssl_head, pixel_ssl_head = \
                build_stage3_model(_init_ckpt, device)
        if args.stage >= 2 and args.cell_density_source != "none":
            if args.cell_density_source == "instanseg":
                print("Initialising InstanSeg extractor …")
                histoplus = InstanSegNucleiExtractor(device)
            elif args.cell_density_source == "hematoxylin":
                print("Initialising hematoxylin-density extractor …")
                histoplus = HematoxylinDensityExtractor(
                    max_side=args.hematoxylin_density_max_side,
                    sigma=args.hematoxylin_density_sigma,
                    normalize=args.hematoxylin_density_norm)
            else:
                histoplus = None
        else:
            histoplus = None
        # ── Morphology-Relation SSL override ─────────────────────────────
        # Even when --no_cell_info is set, we need InstanSeg for on-the-fly
        # cell-graph building.  Density fusion / bio losses remain disabled
        # (they were already zeroed by --no_cell_info).
        if histoplus is None and (args.morph_graph_weight > 0
                                  or args.morph_transition_weight > 0):
            print("[MorphGraph] Enabling InstanSeg extractor for on-the-fly cell graphs …")
            histoplus = InstanSegNucleiExtractor(device)
        lr        = _lr if _lr is not None else 1e-6

    # ── Density fusion (Stage 2+) + contrast projection (Stage 3) ───────────
    # density_fusion: cue-map modulation injected via the trunk forward hook so
    # the neck/decoder see cell-aware features.
    density_fusion = None
    if (args.stage >= 2
            and args.cell_density_source != "none"
            and args.cell_cue_feature_fusion):
        density_fusion = DensityFusion(
            feature_dim=1024,
            hidden=args.cell_cue_fusion_hidden,
            max_gamma=args.cell_cue_fusion_max_gamma,
            max_beta=args.cell_cue_fusion_max_beta,
        ).to(device)
        _n_fuse = sum(p.numel() for p in density_fusion.parameters())
        print(f"[CellCueFusion] Enabled feature-level fusion "
              f"(params={_n_fuse:,}, hidden={args.cell_cue_fusion_hidden}, "
              f"max_gamma={args.cell_cue_fusion_max_gamma}, "
              f"max_beta={args.cell_cue_fusion_max_beta})")
    contrast_proj = ContrastProj(in_dim=1024).to(device) \
        if args.stage >= 3 else None
    morph_projector = SpatialRelationProjector(
        in_dim=1024, hidden_dim=128, morph_dim=12).to(device) \
        if ((args.morph_graph_weight > 0 or args.morph_transition_weight > 0)
            and args.stage >= 3) else None
    if morph_projector is not None:
        _n_mp = sum(p.numel() for p in morph_projector.parameters())
        print(f"[MorphGraph] SpatialRelationProjector enabled "
              f"(params={_n_mp:,}, mode={args.morph_relation_mode})")
    _module_init_ckpt = args.resume_ckpt or args.prev_ckpt or (
        args.test_ckpt if args.test_only else None)
    if _module_init_ckpt and Path(_module_init_ckpt).exists():
        try:
            _mck = torch.load(_module_init_ckpt, map_location=device, weights_only=False)
            if density_fusion is not None and _mck.get("density_fusion_state"):
                _load_compat(density_fusion, _mck["density_fusion_state"],
                             tag="density_fusion_init", device=device)
            if contrast_proj is not None and _mck.get("contrast_proj_state"):
                _load_compat(contrast_proj, _mck["contrast_proj_state"],
                             tag="contrast_proj_init", device=device)
            if morph_projector is not None and _mck.get("morph_proj_state"):
                _load_compat(morph_projector, _mck["morph_proj_state"],
                             tag="morph_proj_init", device=device)
            del _mck
        except Exception as _e:
            print(f"[module_init] WARNING: failed to load density/contrast modules: {_e}")

    # ── Learnable SAM-conditioned density-prior gate (Stage 2+) ───────────
    # Constructed here (not inside train()) so its weights can be loaded
    # from prev_ckpt/resume_ckpt and persisted into checkpoints saved by
    # this run.  Initial output ~= 0.01 (small but strictly nonzero so
    # gradients flow on day 1) — see DensityGateNet for details.
    density_gate_net = None
    if args.nuclei_density_logit_prior > 0.0 and args.stage >= 2:
        density_gate_net = DensityGateNet().to(device)
        # Try to restore gate weights from prev_ckpt/resume_ckpt.
        _gate_init_ckpt = args.resume_ckpt or args.prev_ckpt
        if _gate_init_ckpt:
            _gp = Path(_gate_init_ckpt)
            if _gp.exists():
                try:
                    _gck = torch.load(_gp, map_location=device, weights_only=False)
                    _gst = _gck.get("density_gate_state")
                    if _gst:
                        density_gate_net.load_state_dict(_gst)
                        print(f"[density_gate] Loaded gate weights from {_gp.name}")
                    else:
                        print(f"[density_gate] No 'density_gate_state' in {_gp.name} — using fresh init (~0.01)")
                    del _gck
                except Exception as _e:
                    print(f"[density_gate] WARNING: failed to load gate weights from {_gp}: {_e}")

    model.to(device)
    _raw_prompt_model = model.module if hasattr(model, "module") else model
    _prompt_mod_main = _raw_prompt_model.backbone.language_backbone
    if hasattr(_prompt_mod_main, "use_organ_bias"):
        _prompt_mod_main.use_organ_bias = bool(args.use_organ_bias)
    if hasattr(_prompt_mod_main, "organ_bias") and not args.use_organ_bias:
        _prompt_mod_main.organ_bias.requires_grad_(False)

    # ── Load LoRA-only weights from a previous checkpoint ─────────────────
    # This overrides whatever LoRA weights the stage builder may have loaded.
    # Useful for fine-tuning with a different prompt config while keeping
    # the image encoder's LoRA adaptation from a previous run.
    if args.prev_lora:
        _pl_path = Path(args.prev_lora)
        if not _pl_path.exists():
            print(f"WARNING: --prev_lora path not found: {_pl_path}")
        else:
            _pl_ckpt = torch.load(_pl_path, map_location=device, weights_only=False)
            _pl_raw = model.module if hasattr(model, "module") else model
            _pl_sd = _pl_raw.state_dict()
            _pl_lo = {k: v.to(device) for k, v in _pl_ckpt.get("lora_state", _pl_ckpt).items()
                      if "lora" in k.lower() and k in _pl_sd}
            if not _pl_lo:
                # Try loading the whole thing as a LoRA-only state dict
                _pl_lo = {k: v.to(device) for k, v in _pl_ckpt.items()
                          if "lora" in k.lower() and k in _pl_sd}
            if _pl_lo:
                _pl_sd.update(_pl_lo)
                _pl_raw.load_state_dict(_pl_sd)
                print(f"[prev_lora] Loaded {len(_pl_lo)} LoRA keys from {_pl_path}")
            else:
                print(f"WARNING: --prev_lora: no LoRA keys found in {_pl_path}")
            del _pl_ckpt

    # ── Multi-GPU ─────────────────────────────────────────────────────────
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # ── Training ──────────────────────────────────────────────────────────
    if args.test_only:
        # ── Resolve checkpoint ─────────────────────────────────────────
        # When --resume_run_dir is set, look for the ckpt in the *original*
        # run directory (the one with train_config.json), not the new one.
        _orig_run = Path(args._resume_save_dir) if args._resume_save_dir else None
        if args.bare_medsam3_test_only:
            _ckpt = None
        elif args.test_ckpt:
            _ckpt = Path(args.test_ckpt)
        elif _orig_run is not None:
            _ckpt = _orig_run / "best_holdout_model.pth"
            if not _ckpt.exists():
                _ckpt = _orig_run / "best_model.pth"
        else:
            print("ERROR: --test_only requires either --test_ckpt or --resume_run_dir")
            return

        if _ckpt is not None and not _ckpt.exists():
            print(f"ERROR: checkpoint not found at {_ckpt}")
            return

        # ── Resolve output directory ───────────────────────────────────
        # Timestamped subfolder under test_results/ so re-running never
        # overwrites a previous test run.  Nothing in the original run dir
        # is touched.
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.bare_medsam3_test_only:
            _test_out_base = Path(args.bare_medsam3_out_dir)
            _ckpt_stem = "bare_medsam3_no_checkpoint"
        else:
            _test_out_base = _orig_run if _orig_run is not None else _ckpt.parent
            _ckpt_stem = _ckpt.stem
        _test_out = _test_out_base / "test_results" / f"{_ckpt_stem}_{_ts}"
        _test_out.mkdir(parents=True, exist_ok=True)

        # ── Filter eval items ──────────────────────────────────────────
        _runs: List[Tuple[str, List[_Item], set]] = []

        if args.test_target in ("internal", "both"):
            if args.test_split == "val":
                _eval_items = val_items
            elif args.test_split == "train":
                _eval_items = train_items
            else:   # "all"
                _eval_items = train_items + val_items
            if args.test_datasets:
                _ds_filter = {d.strip() for d in args.test_datasets.split(",") if d.strip()}
                _eval_items = [it for it in _eval_items if it.dataset_name in _ds_filter]
                if not _eval_items:
                    print(f"WARNING: no items match --test_datasets={args.test_datasets!r}")
                    if args.test_target == "internal":
                        return
                else:
                    print(f"  Filtered internal eval to {len(_eval_items)} items "
                          f"from: {sorted(_ds_filter)}")
            if _eval_items:
                _runs.append(("internal", _eval_items, set()))

        if args.test_target in ("external", "both"):
            _external_items, _external_seg_names = _build_external_eval_items_from_args(args)
            if _external_items:
                _runs.append(("external", _external_items, _external_seg_names))
            elif args.test_target == "external":
                return

        if not _runs:
            print("WARNING: no test items selected.")
            return

        for _target_label, _eval_items, _extra_seg_names in _runs:
            _this_out = _test_out / _target_label if args.test_target == "both" else _test_out
            _ckpt_print = _ckpt.name if _ckpt is not None else "bare_medsam3_no_checkpoint"
            print(f"\n{'='*60}")
            print(f"  Final Test  |  target={_target_label}  split={args.test_split}  "
                  f"({len(_eval_items)} items)  |  ckpt={_ckpt_print}")
            print(f"  Datasets: {sorted({it.dataset_name for it in _eval_items})}")
            print(f"  Output -> {_this_out}")
            print(f"{'='*60}\n")

            # ── Seg + Cls evaluation ───────────────────────────────────
            # run_final_test handles:
            #   - Seg metrics on pixel-supervised datasets, including
            #     external datasets supplied through extra_seg_dataset_names.
            #   - Cls metrics on all trained datasets (skipped for stages 1-2
            #     where tile_classifier is None).
            #   - Per-dataset and per-class aggregation.
            # Nothing in the original run dir is modified.
            run_final_test(
                best_ckpt_path = _ckpt,
                model          = model,
                prompt_enc     = prompt_enc,
                test_items     = _eval_items,
                save_dir       = _this_out,
                device         = device,
                adapter        = adapter,
                lora_cfg       = lora_cfg,
                seed           = args.seed,
                stage          = args.stage,
                seg_mode       = args.seg_mode,
                native_crop_at_40x = args.native_crop_at_40x,
                tile_classifier = tile_classifier,
                ssl_head        = ssl_head,
                histoplus       = histoplus,
                density_fusion  = density_fusion,
                density_gate_net = density_gate_net,
                contrast_proj   = contrast_proj,
                sw_threshold    = args.test_max_eval_size,
                test_eval       = args.test_eval,
                extra_seg_dataset_names = _extra_seg_names,
                vis_num         = args.test_vis_num,
                magnification_alignment = not args.no_magnification_alignment,
            )

        print(f"\nDone.  Outputs in: {_test_out}")
        return

    print(f"\n{'='*60}")
    print(f"  Stage {args.stage} Training  |  {args.epochs} epochs  |  seg_mode={args.seg_mode}")
    print(f"  Save dir: {save_dir}")
    print(f"{'='*60}\n")

    # Build train_cfg dict once so every checkpoint carries it
    _train_cfg = {k: v for k, v in vars(args).items()
                  if k not in ("resume_ckpt", "resume_run_dir",
                               "_resume_save_dir", "resume_load")}

    run_training(
        stage      = args.stage,
        model      = model,
        prompt_enc = prompt_enc,
        train_items= train_items,
        val_items  = val_items,
        save_dir   = save_dir,
        device     = device,
        epochs     = args.epochs,
        batch_size = args.batch_acc,
        lr         = lr,
        adapter    = adapter,
        histoplus  = histoplus,
        lora_cfg   = lora_cfg,
        val_steps  = args.val_steps,
        num_workers = args.num_workers,
        fg_crop_prob = args.fg_crop_prob,
        steps_per_epoch = args.steps_per_epoch,
        total_steps     = args.total_steps,
        seed       = args.seed,
        resume_ckpt = args.resume_ckpt,
        resume_load = args.resume_load,
        prompt_lr   = args.prompt_lr,
        adapter_lr  = args.adapter_lr,
        reset_best_dice = args.reset_best_dice,
        run_initial_eval = args.run_initial_eval,
        save_best_vis = args.save_best_vis,
        vis_num      = args.vis_num,
        quick_val_max_samples    = args.quick_val_max_samples,
        curriculum               = args.curriculum,
        curriculum_warmup_epochs = args.curriculum_warmup_epochs,
        focal_gamma              = args.focal_gamma,
        val_samples_per_class    = args.val_samples_per_class,
        val_samples_per_dataset  = args.val_samples_per_dataset,
        repulsion_weight         = args.repulsion_weight,
        negative_boost           = args.negative_boost,
        area_loss_weight         = args.area_loss_weight,
        density_consistency_weight = args.density_consistency_weight,
        structure_class_weight   = args.structure_class_weight,
        freeze_prompts_after_stage1 = args.freeze_prompts_after_stage1,
        dice_weight              = args.dice_weight,
        bce_weight               = args.bce_weight,
        seg_mode                 = args.seg_mode,
        native_crop_at_40x       = args.native_crop_at_40x,
        magnification_alignment  = not args.no_magnification_alignment,
        train_cfg                = _train_cfg,
        tile_classifier          = tile_classifier,
        ssl_head                 = ssl_head,
        pixel_ssl_head           = pixel_ssl_head,
        density_fusion           = density_fusion,
        density_gate_net         = density_gate_net,
        contrast_proj            = contrast_proj,
        morph_projector          = morph_projector,
        morph_graph_weight       = (args.morph_graph_weight
                                     if args.stage >= 3 else 0.0),
        morph_transition_weight  = (args.morph_transition_weight
                                     if args.stage >= 3 else 0.0),
        morph_relation_mode      = args.morph_relation_mode,
        morph_relation_grid_size = args.morph_grid_size,
        morph_keep_n_cells       = args.morph_keep_n_cells,
        morph_warmup_steps       = args.morph_warmup_steps,
        morph_ramp_steps         = args.morph_ramp_steps,
        hard_mining_start_epoch  = args.hard_mining_start_epoch,
        ssl_weight               = args.ssl_weight,
        intra_image_contrast_weight = (
            args.intra_image_contrast_weight if args.stage >= 3 else 0.0),
        cls_weight               = args.tile_cls_weight,
        cls_only_mil_weight      = args.cls_only_mil_weight,
        cls_only_absent_weight   = args.cls_only_absent_weight,
        cls_only_topk_ratio      = args.cls_only_topk_ratio,
        cls_only_entropy_weight  = args.cls_only_entropy_weight,
        organ_warmup_steps       = args.organ_warmup_steps,
        cls_only_stride          = args.cls_only_stride,
        nuclei_density_logit_prior = args.nuclei_density_logit_prior,
        freeze_lora_in_stage2      = args.freeze_lora_stage2,
        bio_boundary_weight        = args.bio_boundary_weight,
        bio_compat_weight          = args.bio_compat_weight,
        bio_gland_lumen_weight     = args.bio_gland_lumen_weight,
        bg_suppression_weight       = args.bg_suppression_weight,
        scale_reg_weight            = (args.scale_reg_weight if args.stage >= 3 else 0.0),
        shape_reg_weight            = (args.shape_reg_weight if args.stage >= 3 else 0.0),
        use_organ_bias             = args.use_organ_bias,
        setup_cache_dir            = args.setup_cache_dir,
    )

    # ── Auto-run seg test after training ──────────────────────────────
    if not args.skip_auto_test:
        for _ckpt_name in ("best_model.pth", "best_holdout_model.pth"):
            _ckpt = save_dir / _ckpt_name
            if not _ckpt.exists():
                continue
            _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            _test_out = save_dir / "test_results" / f"{_ckpt.stem}_{_ts}"
            _test_out.mkdir(parents=True, exist_ok=True)
            print(f"\nAuto-test seg: {_ckpt_name}  →  {_test_out}")
            run_final_test(
                best_ckpt_path = _ckpt,
                model          = model,
                prompt_enc     = prompt_enc,
                test_items     = val_items,
                save_dir       = _test_out,
                device         = device,
                adapter        = adapter,
                lora_cfg       = lora_cfg,
                seed           = args.seed,
                stage          = args.stage,
                seg_mode       = args.seg_mode,
                native_crop_at_40x = args.native_crop_at_40x,
                tile_classifier = tile_classifier,
                ssl_head        = ssl_head,
                histoplus       = histoplus,
                density_fusion  = density_fusion,
                density_gate_net = density_gate_net,
                contrast_proj   = contrast_proj,
                test_eval       = "seg",
                vis_num         = 0,   # 0 = save ALL vis for internal test
            )

            print(f"\nDone.  Outputs in: {_test_out}")
            return

        return  # auto-test complete, exit

    # ── Training already ran above; nothing more to do ─────────────────
    return


if __name__ == "__main__":
    main()
