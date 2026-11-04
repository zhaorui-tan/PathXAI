from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Literal, Optional


TargetKind = Literal["summary", "full", "both"]


@dataclass
class WsiCaptionSample:
    item_id: str
    case_id: str
    cancer_type: str
    split: str
    summary_report: Optional[str]
    full_report: Optional[str]
    target_text: str | Dict[str, str]
    image_paths: List[str]
    slide_paths: List[str]
    pdf_paths: List[str]
    annotation_paths: List[str]
    report_path: Optional[str]


class WsiCaptionDataset:
    """Manifest-backed loader for Wsi-Caption report supervision.

    The case directory has two report targets:
    - `annotation`: short summary/caption target
    - `Report.txt`: full OCR-extracted pathology report target
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: Optional[str] = None,
        target: TargetKind = "summary",
        require_images: bool = True,
        require_slides: bool = False,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.target = target
        self.require_images = require_images
        self.require_slides = require_slides
        self.items = self._load_items()

    def _load_items(self) -> List[dict]:
        items = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if self.split and item.get("split") != self.split:
                    continue
                metadata = item.get("metadata") or {}
                if self.require_images and not metadata.get("image_paths"):
                    continue
                if self.require_slides and not metadata.get("slide_paths"):
                    continue
                if self.target == "summary" and not metadata.get("summary_report_text"):
                    continue
                if self.target == "full" and not metadata.get("full_report_text"):
                    continue
                if self.target == "both" and not (
                    metadata.get("summary_report_text") and metadata.get("full_report_text")
                ):
                    continue
                items.append(item)
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> WsiCaptionSample:
        item = self.items[idx]
        metadata = item.get("metadata") or {}
        summary = metadata.get("summary_report_text")
        full = metadata.get("full_report_text") or item.get("report_text")
        if self.target == "summary":
            target_text: str | Dict[str, str] = summary or ""
        elif self.target == "full":
            target_text = full or ""
        else:
            target_text = {"summary": summary or "", "full": full or ""}
        return WsiCaptionSample(
            item_id=item["item_id"],
            case_id=item["case_id"],
            cancer_type=metadata.get("cancer_type") or "",
            split=item.get("split", "unsplit"),
            summary_report=summary,
            full_report=full,
            target_text=target_text,
            image_paths=list(metadata.get("image_paths") or []),
            slide_paths=list(metadata.get("slide_paths") or []),
            pdf_paths=list(metadata.get("pdf_paths") or []),
            annotation_paths=list(item.get("annotation_paths") or []),
            report_path=item.get("report_path"),
        )

    def __iter__(self) -> Iterator[WsiCaptionSample]:
        for idx in range(len(self)):
            yield self[idx]


def build_report_prompt_input(sample: WsiCaptionSample, max_images: int = 4) -> Dict[str, object]:
    """Return a model-ready, evidence-oriented input object."""
    return {
        "case_id": sample.case_id,
        "cancer_type": sample.cancer_type,
        "images": sample.image_paths[:max_images],
        "slide_paths": sample.slide_paths,
        "task": (
            "Generate a high-quality pathology report from the available WSI/ROI image evidence. "
            "Use the short summary as the concise target when training summary generation, and "
            "use the full OCR report as the detailed target when training full-report generation."
        ),
        "target": sample.target_text,
    }
