from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .public_dataset_scan import IMAGE_EXTS, REPORT_HINTS, TABLE_EXTS, WSI_EXTS


@dataclass
class ManifestItem:
    item_id: str
    dataset_name: str
    case_id: str
    item_type: str
    path: str
    split: str = "unsplit"
    diagnosis_label: Optional[str] = None
    report_text: Optional[str] = None
    report_path: Optional[str] = None
    annotation_paths: List[str] = field(default_factory=list)
    mask_paths: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    usable_for: List[str] = field(default_factory=list)


def stable_id(*parts: object) -> str:
    text = "::".join(str(p) for p in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(items: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def _dataset_name(root: Path, path: Path) -> str:
    return path.relative_to(root).parts[0]


def _case_id(path: Path) -> str:
    return path.stem


def _looks_report_like(path: Path) -> bool:
    text = str(path).lower()
    return any(h in text for h in REPORT_HINTS)


def _nearby_annotation_paths(slide_path: Path, dataset_dir: Path, limit: int = 20) -> List[str]:
    stem = slide_path.stem.lower()
    hits = []
    for p in dataset_dir.rglob("*"):
        if not p.is_file():
            continue
        low = str(p).lower()
        name_stem = p.stem.lower()
        exactish = name_stem == stem or name_stem.startswith(f"{stem}_") or name_stem.endswith(f"_{stem}")
        if exactish and any(h in low for h in ("mask", "label", "annotation", "ann", "gtruth", "ground")):
            hits.append(str(p))
            if len(hits) >= limit:
                break
    return hits


def build_wsi_manifest(root: Path) -> List[ManifestItem]:
    root = root.resolve()
    items: List[ManifestItem] = []
    for slide_path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in WSI_EXTS):
        dataset_name = _dataset_name(root, slide_path)
        dataset_dir = root / dataset_name
        annotations = _nearby_annotation_paths(slide_path, dataset_dir)
        usable_for = ["wsi_roi", "report_baseline"]
        if annotations:
            usable_for.append("roi_or_segmentation_eval")
        case_id = _case_id(slide_path)
        items.append(
            ManifestItem(
                item_id=stable_id(dataset_name, case_id, slide_path),
                dataset_name=dataset_name,
                case_id=case_id,
                item_type="wsi",
                path=str(slide_path),
                annotation_paths=annotations,
                metadata={"source_root": str(root)},
                usable_for=usable_for,
            )
        )
    return items


def build_image_manifest(
    root: Path,
    include_datasets: Optional[set[str]] = None,
    max_per_dataset: int = 1000,
    include_mask_like: bool = False,
) -> List[ManifestItem]:
    """Build a bounded manifest for ROI/patch images.

    The public tree contains more than a million images, so this function caps
    each dataset by default. It is meant for smoke tests and report-baseline
    development, not exhaustive training ingestion.
    """
    root = root.resolve()
    items: List[ManifestItem] = []
    counts: Dict[str, int] = {}
    skip_names = {"thumbs.db"}
    for image_path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        if image_path.name.lower() in skip_names:
            continue
        dataset_name = _dataset_name(root, image_path)
        if include_datasets and dataset_name not in include_datasets:
            continue
        lower = str(image_path).lower()
        is_mask_like = any(h in lower for h in ("mask", "label", "annotation", "ann", "gtruth", "ground"))
        if is_mask_like and not include_mask_like:
            continue
        counts[dataset_name] = counts.get(dataset_name, 0)
        if max_per_dataset > 0 and counts[dataset_name] >= max_per_dataset:
            continue
        counts[dataset_name] += 1
        case_id = _case_id(image_path)
        mask_paths = _matched_mask_paths(root / dataset_name, image_path)
        usable_for = ["patch_or_roi", "report_baseline"]
        usable_for.append("image_report_smoke")
        if mask_paths:
            usable_for.append("segmentation_eval")
        items.append(
            ManifestItem(
                item_id=stable_id(dataset_name, case_id, image_path),
                dataset_name=dataset_name,
                case_id=case_id,
                item_type="image",
                path=str(image_path),
                mask_paths=mask_paths,
                metadata={
                    "source_root": str(root),
                    "mask_like_filename": is_mask_like,
                },
                usable_for=usable_for,
            )
        )
    return items


def _matched_mask_paths(dataset_dir: Path, image_path: Path, limit: int = 10) -> List[str]:
    stem = image_path.stem
    candidates = [
        dataset_dir / "masks" / f"{stem}.png",
        dataset_dir / "masks" / f"{stem}.tif",
        dataset_dir / "mask" / f"{stem}.png",
        dataset_dir / "mask" / f"{stem}.tif",
        dataset_dir / "ann" / f"{stem}.png",
        dataset_dir / "ann" / f"{stem}.bmp",
    ]
    hits = [str(p) for p in candidates if p.exists()]
    if hits:
        return hits[:limit]
    return []


def build_report_candidates(root: Path, max_text_chars: int = 12000) -> List[ManifestItem]:
    root = root.resolve()
    items: List[ManifestItem] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        ext = path.suffix.lower()
        if ext not in TABLE_EXTS or not _looks_report_like(path):
            continue
        report_text = None
        if ext in {".txt", ".json", ".jsonl", ".csv", ".tsv", ".xml"}:
            try:
                report_text = path.read_text(encoding="utf-8", errors="ignore")[:max_text_chars]
            except OSError:
                report_text = None
        dataset_name = _dataset_name(root, path)
        items.append(
            ManifestItem(
                item_id=stable_id(dataset_name, path),
                dataset_name=dataset_name,
                case_id=_case_id(path),
                item_type="report_candidate",
                path=str(path),
                report_path=str(path),
                report_text=report_text,
                usable_for=["report_pair_candidate"],
            )
        )
    return items


def infer_labels_from_common_tables(root: Path) -> List[ManifestItem]:
    """Build lightweight label items from common public dataset tables.

    These are not full reports; they are useful for classification/report-field
    evaluation while true report-paired data is missing.
    """
    root = root.resolve()
    items: List[ManifestItem] = []

    bach_csv = root / "ICIAR2018_BACH_Challenge" / "Photos" / "microscopy_ground_truth.csv"
    if bach_csv.exists():
        with bach_csv.open("r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                image_id, label = row[0].strip(), row[1].strip()
                if not image_id or image_id.lower() in {"image", "filename"}:
                    continue
                items.append(
                    ManifestItem(
                        item_id=stable_id("ICIAR2018_BACH_Challenge", image_id, label),
                        dataset_name="ICIAR2018_BACH_Challenge",
                        case_id=Path(image_id).stem,
                        item_type="diagnosis_label",
                        path=str(bach_csv),
                        diagnosis_label=label,
                        metadata={"label_source": str(bach_csv), "image_id": image_id},
                        usable_for=["diagnosis_eval", "weak_report_eval"],
                    )
                )

    mhist_csv = root / "MHIST" / "annotations.csv"
    if mhist_csv.exists():
        with mhist_csv.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row.get("Image Name") or row.get("image") or row.get("filename")
                label = row.get("Majority Vote Label") or row.get("label") or row.get("Diagnosis")
                if not image_id or not label:
                    continue
                items.append(
                    ManifestItem(
                        item_id=stable_id("MHIST", image_id, label),
                        dataset_name="MHIST",
                        case_id=Path(image_id).stem,
                        item_type="diagnosis_label",
                        path=str(mhist_csv),
                        diagnosis_label=label,
                        metadata={"label_source": str(mhist_csv), "image_id": image_id},
                        usable_for=["diagnosis_eval", "weak_report_eval"],
                    )
                )

    return items


def items_to_dicts(items: Iterable[ManifestItem]) -> List[dict]:
    return [asdict(item) for item in items]
