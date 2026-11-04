from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List


WSI_EXTS = {".svs", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MASK_HINTS = ("mask", "masks", "label", "labels", "annotation", "annotations", "ann", "gtruth", "ground")
REPORT_HINTS = (
    "report",
    "reports",
    "clinical",
    "diagnosis",
    "diagnostic",
    "caption",
    "captions",
    "finding",
    "findings",
    "impression",
    "summary",
)
TABLE_EXTS = {".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".xls", ".txt", ".xml", ".mat", ".npy"}


@dataclass
class DatasetSummary:
    dataset_name: str
    dataset_path: str
    total_files: int = 0
    ext_counts: Dict[str, int] = field(default_factory=dict)
    wsi_count: int = 0
    image_count: int = 0
    table_count: int = 0
    mask_or_annotation_count: int = 0
    report_hint_count: int = 0
    sample_wsi: List[str] = field(default_factory=list)
    sample_images: List[str] = field(default_factory=list)
    sample_tables: List[str] = field(default_factory=list)
    sample_report_hints: List[str] = field(default_factory=list)
    usable_for: List[str] = field(default_factory=list)


def _has_any_hint(path: Path, hints: Iterable[str]) -> bool:
    text = str(path).lower()
    return any(h in text for h in hints)


def _append_sample(samples: List[str], value: str, max_samples: int) -> None:
    if len(samples) < max_samples:
        samples.append(value)


def scan_public_datasets(root: Path, max_samples: int = 5) -> Dict[str, object]:
    root = root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    summaries: Dict[str, DatasetSummary] = {}
    for dataset_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        summaries[dataset_dir.name] = DatasetSummary(
            dataset_name=dataset_dir.name,
            dataset_path=str(dataset_dir),
        )

    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        try:
            dataset_name = file_path.relative_to(root).parts[0]
        except IndexError:
            continue
        summary = summaries.setdefault(
            dataset_name,
            DatasetSummary(dataset_name=dataset_name, dataset_path=str(root / dataset_name)),
        )
        rel = str(file_path.relative_to(root))
        ext = file_path.suffix.lower()
        summary.total_files += 1
        summary.ext_counts[ext or "<none>"] = summary.ext_counts.get(ext or "<none>", 0) + 1

        if ext in WSI_EXTS:
            summary.wsi_count += 1
            _append_sample(summary.sample_wsi, rel, max_samples)
        if ext in IMAGE_EXTS:
            summary.image_count += 1
            _append_sample(summary.sample_images, rel, max_samples)
        if ext in TABLE_EXTS:
            summary.table_count += 1
            _append_sample(summary.sample_tables, rel, max_samples)
        if _has_any_hint(file_path, MASK_HINTS):
            summary.mask_or_annotation_count += 1
        if _has_any_hint(file_path, REPORT_HINTS):
            summary.report_hint_count += 1
            _append_sample(summary.sample_report_hints, rel, max_samples)

    for summary in summaries.values():
        uses = []
        if summary.wsi_count:
            uses.append("wsi_roi")
        if summary.mask_or_annotation_count:
            uses.append("segmentation_or_roi_eval")
        if summary.report_hint_count:
            uses.append("report_pair_candidate")
        if summary.image_count and not summary.wsi_count:
            uses.append("patch_or_roi")
        summary.usable_for = uses
        summary.ext_counts = dict(sorted(summary.ext_counts.items(), key=lambda x: (-x[1], x[0])))

    by_use = defaultdict(list)
    for summary in summaries.values():
        for use in summary.usable_for:
            by_use[use].append(summary.dataset_name)

    totals = Counter()
    for summary in summaries.values():
        totals["datasets"] += 1
        totals["files"] += summary.total_files
        totals["wsi"] += summary.wsi_count
        totals["images"] += summary.image_count
        totals["tables"] += summary.table_count
        totals["mask_or_annotation"] += summary.mask_or_annotation_count
        totals["report_hints"] += summary.report_hint_count

    return {
        "root": str(root),
        "totals": dict(totals),
        "by_use": {k: sorted(v) for k, v in sorted(by_use.items())},
        "datasets": [asdict(s) for s in sorted(summaries.values(), key=lambda s: s.dataset_name)],
    }


def write_scan(scan: Dict[str, object], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(scan, f, indent=2, ensure_ascii=False)
